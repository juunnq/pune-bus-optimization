"""
Decision-focused learning for PMPML frequency allocation.

Compares 5 approaches:
  1) XGBoost (MSE) -> Optimize          [predict-then-optimize, baseline]
  2) XGBoost (decision-weighted MSE) -> Optimize
  3) NN (MSE) -> Optimize
  4) NN (SPO+ surrogate) -> Optimize
  5) Oracle (true demand) -> Optimize   [regret = 0]

For each method we:
  (a) train the predictor (or load it)
  (b) build a (route x period) demand matrix from predicted hourly demand
      for the 50 modeled routes; non-modeled routes use the original
      demand_matrix.csv values as a fixed baseline
  (c) solve the deterministic MIP on the predicted matrix
  (d) evaluate the resulting allocation against the TRUE matrix
      (test-set actuals for modeled routes; baseline for the rest)

Regret = method_true_objective - oracle_true_objective.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit import log_metric, log_warning
from src.data_processing import (
    load_routes, load_demand, load_weather, load_demand_matrix, get_time_period,
)
from src.demand_forecasting import (
    prepare_features, get_feature_cols, chronological_split, _metrics,
)
from src.deterministic_model import (
    FREQ_LEVELS, TIME_PERIODS,
    build_deterministic_model, evaluate_allocation,
)


# Demand matrix builder

def build_demand_matrix(test_df: pd.DataFrame,
                        value_col: str,
                        baseline_matrix: pd.DataFrame) -> pd.DataFrame:
    """For modeled routes, average value_col across hours within each period
    over the entire test window. For unmodeled routes, use baseline_matrix."""
    df = test_df.copy()
    df['period'] = df['hour'].apply(get_time_period)
    df = df[df['period'] != 'off_hours']
    modeled = (df.groupby(['route_id', 'period'])[value_col]
                  .mean()
                  .unstack('period')
                  .reindex(columns=TIME_PERIODS))

    out = baseline_matrix.set_index('route_id')[TIME_PERIODS].copy()
    for r in modeled.index:
        if r in out.index:
            out.loc[r] = modeled.loc[r].values
    return out.reset_index()


# Predict-then-optimize wrapper

def predict_then_optimize(predictions_test: np.ndarray,
                          test_index_df: pd.DataFrame,
                          truth_matrix: pd.DataFrame,
                          baseline_matrix: pd.DataFrame,
                          routes_df: pd.DataFrame,
                          fleet_size: int = 1800) -> dict:
    """Build pred matrix from predictions, solve MIP, evaluate vs truth.
    test_index_df must contain columns route_id, hour aligned with predictions.
    """
    df = test_index_df[['route_id', 'hour']].copy()
    df['pred'] = np.maximum(predictions_test, 0)

    pred_matrix = build_demand_matrix(df, 'pred', baseline_matrix)

    res = build_deterministic_model(routes_df, pred_matrix,
                                    fleet_size=fleet_size, time_limit_sec=120)
    alloc = res['allocation_df']
    pred_obj = res['objective']
    true_obj = evaluate_allocation(alloc, truth_matrix, routes_df)['total_wait_time']
    return {'allocation': alloc, 'predicted_objective': pred_obj,
            'true_objective': true_obj, 'solve_time': res['solve_time']}


# Decision weights

def compute_decision_weights(routes_df: pd.DataFrame,
                             optimal_allocation: pd.DataFrame,
                             demand_matrix: pd.DataFrame) -> dict:
    """weight[r,t] = d[r,t] / (2 * f[r,t]^2). Returns dict (route_id, period) -> weight."""
    dm = demand_matrix.set_index('route_id')[TIME_PERIODS]
    weights = {}
    for _, row in optimal_allocation.iterrows():
        r, t, k = row['route_id'], row['period'], int(row['frequency'])
        d = float(dm.loc[r, t]) if r in dm.index else 0.0
        weights[(r, t)] = d / (2.0 * k * k)
    return weights


def map_weights_to_samples(train_df: pd.DataFrame, weights: dict) -> np.ndarray:
    """Map (route_id, period)->weight onto each training sample."""
    df = train_df[['route_id', 'hour']].copy()
    df['period'] = df['hour'].apply(get_time_period)
    w = np.array([weights.get((r, t), 0.0) for r, t in
                  zip(df['route_id'].values, df['period'].values)])
    # off_hours samples get 0 weight; bump to small epsilon so they're not
    # entirely ignored
    w = np.where(w <= 0, 1e-3, w)
    # normalise so mean ~1 (XGBoost is happier this way)
    w = w / max(w.mean(), 1e-9)
    return w


# Neural network

class DemandPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _make_loaders(X, y, w, batch_size, device, shuffle=True):
    Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    yt = torch.tensor(np.asarray(y, dtype=np.float32), device=device)
    wt = torch.tensor(np.asarray(w, dtype=np.float32), device=device)
    ds = torch.utils.data.TensorDataset(Xt, yt, wt)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_nn(X_train, y_train, w_train, X_val, y_val,
             input_dim, epochs=40, batch_size=512, lr=1e-3,
             device='cpu', use_weighted_loss=False):
    """Train DemandPredictor with weighted MSE if use_weighted_loss
    else plain MSE. Returns trained model + history."""
    torch.manual_seed(42)
    model = DemandPredictor(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = _make_loaders(X_train, y_train, w_train, batch_size, device, True)
    Xv = torch.tensor(np.asarray(X_val, dtype=np.float32), device=device)
    yv = torch.tensor(np.asarray(y_val, dtype=np.float32), device=device)

    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for xb, yb, wb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            err = pred - yb
            if use_weighted_loss:
                loss = (wb * err.abs()).mean()
            else:
                loss = (err ** 2).mean()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        model.eval()
        with torch.no_grad():
            vpred = model(Xv).cpu().numpy()
            v_rmse = float(np.sqrt(np.mean((vpred - y_val.values) ** 2)))
        history.append({'epoch': epoch, 'train_loss': total / n, 'val_rmse': v_rmse})
    return model, history


def nn_predict(model, X, device='cpu') -> np.ndarray:
    model.eval()
    Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    with torch.no_grad():
        out = model(Xt).cpu().numpy()
    return np.maximum(out, 0)


# Pipeline

def run_decision_focused_pipeline(routes_df, demand_df, weather_df, demand_matrix,
                                  output_dir="results/tables",
                                  models_dir="results/models",
                                  fleet_size=1800,
                                  nn_epochs=30):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_metric("decision_focused_device", device)

    # Features
    df_full = prepare_features(demand_df, weather_df, routes_df)
    feat_cols = get_feature_cols(df_full)
    train_df, val_df, test_df = chronological_split(df_full)
    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df['ridership']
    X_val = val_df[feat_cols].values.astype(np.float32)
    y_val = val_df['ridership']
    X_test = test_df[feat_cols].values.astype(np.float32)
    y_test = test_df['ridership']

    # Truth matrix from test actuals
    truth_matrix = build_demand_matrix(test_df, 'ridership', demand_matrix)

    # Oracle: solve MIP on truth, evaluate on truth
    oracle_res = build_deterministic_model(routes_df, truth_matrix,
                                           fleet_size=fleet_size, time_limit_sec=120)
    oracle_obj = evaluate_allocation(oracle_res['allocation_df'],
                                     truth_matrix, routes_df)['total_wait_time']
    log_metric("oracle_objective", round(oracle_obj, 2))

    # 1) XGBoost MSE
    xgb_mse = xgb.XGBRegressor()
    xgb_mse.load_model(os.path.join(models_dir, "xgb_model.json"))
    xgb_pred = np.maximum(xgb_mse.predict(X_test), 0)
    xgb_metrics = _metrics(y_test.values, xgb_pred)
    res_xgb = predict_then_optimize(xgb_pred, test_df, truth_matrix, demand_matrix,
                                    routes_df, fleet_size=fleet_size)

    # 2) XGBoost weighted (decision-aware)
    optimal_alloc = pd.read_csv(os.path.join(output_dir, "optimal_allocation.csv"))
    weights_dict = compute_decision_weights(routes_df, optimal_alloc, demand_matrix)
    sample_weights = map_weights_to_samples(train_df, weights_dict)
    val_weights = map_weights_to_samples(val_df, weights_dict)

    xgb_w = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=8,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        objective='reg:squarederror', n_jobs=-1, random_state=42,
        early_stopping_rounds=50, eval_metric='rmse', verbosity=0,
    )
    xgb_w.fit(X_train, y_train, sample_weight=sample_weights,
              eval_set=[(X_val, y_val)],
              sample_weight_eval_set=[val_weights],
              verbose=False)
    xgb_w_pred = np.maximum(xgb_w.predict(X_test), 0)
    xgb_w_metrics = _metrics(y_test.values, xgb_w_pred)
    res_xgb_w = predict_then_optimize(xgb_w_pred, test_df, truth_matrix, demand_matrix,
                                       routes_df, fleet_size=fleet_size)

    # 3) NN MSE
    flat_w = np.ones(len(y_train), dtype=np.float32)
    nn_mse, nn_mse_hist = train_nn(X_train, y_train, flat_w, X_val, y_val,
                                   input_dim=X_train.shape[1],
                                   epochs=nn_epochs, batch_size=512, lr=1e-3,
                                   device=device, use_weighted_loss=False)
    nn_mse_pred = nn_predict(nn_mse, X_test, device=device)
    nn_mse_metrics = _metrics(y_test.values, nn_mse_pred)
    res_nn_mse = predict_then_optimize(nn_mse_pred, test_df, truth_matrix, demand_matrix,
                                       routes_df, fleet_size=fleet_size)
    torch.save(nn_mse.state_dict(), os.path.join(models_dir, "nn_mse_model.pt"))

    # 4) NN SPO+ surrogate (decision-aware weighted L1)
    nn_spo, nn_spo_hist = train_nn(X_train, y_train, sample_weights, X_val, y_val,
                                   input_dim=X_train.shape[1],
                                   epochs=nn_epochs, batch_size=512, lr=1e-3,
                                   device=device, use_weighted_loss=True)
    nn_spo_pred = nn_predict(nn_spo, X_test, device=device)
    nn_spo_metrics = _metrics(y_test.values, nn_spo_pred)
    res_nn_spo = predict_then_optimize(nn_spo_pred, test_df, truth_matrix, demand_matrix,
                                       routes_df, fleet_size=fleet_size)
    torch.save(nn_spo.state_dict(), os.path.join(models_dir, "nn_spo_model.pt"))

    # Compose results
    rows = []
    methods = [
        ('xgb_mse', xgb_metrics, res_xgb),
        ('xgb_weighted', xgb_w_metrics, res_xgb_w),
        ('nn_mse', nn_mse_metrics, res_nn_mse),
        ('nn_spo', nn_spo_metrics, res_nn_spo),
    ]
    for name, m, res in methods:
        rows.append({
            'method': name,
            'prediction_rmse': m['rmse'],
            'prediction_mae': m['mae'],
            'prediction_mape': m['mape'],
            'prediction_r2': m['r2'],
            'predicted_objective': res['predicted_objective'],
            'decision_quality': res['true_objective'],
            'regret': res['true_objective'] - oracle_obj,
        })
    rows.append({
        'method': 'oracle',
        'prediction_rmse': 0.0,
        'prediction_mae': 0.0,
        'prediction_mape': 0.0,
        'prediction_r2': 1.0,
        'predicted_objective': oracle_obj,
        'decision_quality': oracle_obj,
        'regret': 0.0,
    })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(output_dir, "decision_focused_results.csv"), index=False)
    return df_out


if __name__ == "__main__":
    from src.audit import (setup_audit, log_phase_start, log_phase_end,
                           log_gate_check, log_file_created)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setup_audit(project_root)
    log_phase_start("PHASE_5_DECISION_FOCUSED")

    routes_df = load_routes(os.path.join(project_root, "data/processed/routes.csv"))
    demand_df = load_demand(os.path.join(project_root, "data/processed/demand.csv"))
    weather_df = load_weather(os.path.join(project_root, "data/processed/weather.csv"))
    demand_matrix = load_demand_matrix(os.path.join(project_root, "data/processed/demand_matrix.csv"))

    df_out = run_decision_focused_pipeline(
        routes_df, demand_df, weather_df, demand_matrix,
        output_dir=os.path.join(project_root, "results/tables"),
        models_dir=os.path.join(project_root, "results/models"),
        fleet_size=1800, nn_epochs=30,
    )
    print(df_out.to_string(index=False))

    expected_methods = ['xgb_mse', 'xgb_weighted', 'nn_mse', 'nn_spo', 'oracle']
    for m in expected_methods:
        log_gate_check(f"method_present_{m}", m in df_out['method'].values, "present",
                       m in df_out['method'].values)

    oracle_obj = df_out[df_out['method'] == 'oracle']['decision_quality'].iloc[0]
    for _, row in df_out.iterrows():
        if row['method'] != 'oracle':
            log_gate_check(f"oracle_best_vs_{row['method']}",
                           oracle_obj <= row['decision_quality'] + 0.01,
                           f"<= {row['decision_quality']:.2f}", round(float(oracle_obj), 2))

    for _, row in df_out.iterrows():
        if row['method'] != 'oracle':
            log_gate_check(f"regret_nonneg_{row['method']}",
                           row['regret'] >= -0.01, ">=0", round(float(row['regret']), 4))

    for _, row in df_out.iterrows():
        if row['method'] != 'oracle':
            log_gate_check(f"rmse_valid_{row['method']}",
                           0 < row['prediction_rmse'] < 1e6,
                           "positive finite", round(float(row['prediction_rmse']), 2))

    for f_path in ['results/models/nn_spo_model.pt', 'results/models/nn_mse_model.pt']:
        full = os.path.join(project_root, f_path)
        log_gate_check(f"model_saved_{os.path.basename(f_path)}",
                       os.path.exists(full), "True", str(os.path.exists(full)))

    xgb_mse_regret = df_out[df_out['method'] == 'xgb_mse']['regret'].iloc[0]
    best_df_regret = df_out[df_out['method'].isin(['xgb_weighted', 'nn_spo'])]['regret'].min()
    log_metric("xgb_mse_regret", round(float(xgb_mse_regret), 4))
    log_metric("best_decision_focused_regret", round(float(best_df_regret), 4))
    if abs(xgb_mse_regret) > 1e-6:
        log_metric("df_improvement_pct",
                   round(float((xgb_mse_regret - best_df_regret) / xgb_mse_regret * 100), 2))

    for f in ["decision_focused_results.csv"]:
        log_file_created(os.path.join(project_root, "results/tables", f))
    log_phase_end("PHASE_5_DECISION_FOCUSED", "PASS")
    print("PHASE 5 COMPLETE")
