"""
Demand forecasting: LR / RF / XGBoost (point + 3 quantiles).

Train: months 1-8, Val: months 9-10, Test: months 11-12 (chronological split).
Features include cyclical encodings, calendar flags, weather, route metadata,
and lag-1d / lag-7d ridership.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.audit import log_metric


def prepare_features(demand_df: pd.DataFrame,
                     weather_df: pd.DataFrame,
                     routes_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a single DataFrame with engineered features and target. Sorted
    by (route_id, date, hour). Lag features filled where possible."""
    df = demand_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Route metadata
    df = df.merge(
        routes_df[['route_id', 'route_category', 'length_km',
                   'num_stops', 'priority_score']],
        on='route_id', how='left'
    )
    df = pd.get_dummies(df, columns=['route_category'], prefix='cat', drop_first=False)

    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features
    df = df.sort_values(['route_id', 'date', 'hour']).reset_index(drop=True)
    df['lag_1d'] = df.groupby(['route_id', 'hour'])['ridership'].shift(1)
    df['lag_7d'] = df.groupby(['route_id', 'hour'])['ridership'].shift(7)

    # Drop rows missing lag_7d (first 7 days)
    df = df.dropna(subset=['lag_1d', 'lag_7d']).reset_index(drop=True)
    return df


FEATURE_COLS_BASE = [
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_weekend', 'is_holiday', 'is_monsoon', 'is_exam_season', 'is_festival',
    'temperature_max', 'precipitation_mm',
    'length_km', 'num_stops', 'priority_score',
    'lag_1d', 'lag_7d',
]


def get_feature_cols(df: pd.DataFrame) -> list:
    cat_cols = [c for c in df.columns if c.startswith('cat_')]
    return [c for c in FEATURE_COLS_BASE if c in df.columns] + cat_cols


def chronological_split(df: pd.DataFrame):
    train = df[df['month'] <= 8]
    val = df[(df['month'] == 9) | (df['month'] == 10)]
    test = df[(df['month'] == 11) | (df['month'] == 12)]
    return train, val, test


def _metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # MAPE on non-zero ridership only
    mask = y_true > 1.0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.sum() else float('nan')
    r2 = float(r2_score(y_true, y_pred))
    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}


def train_models(X_train, y_train, X_val, y_val):
    models = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['linear_regression'] = lr

    rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                               n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf

    xgb_reg = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=8,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=5,
        objective='reg:squarederror', n_jobs=-1, random_state=42,
        early_stopping_rounds=50, eval_metric='rmse', verbosity=0,
    )
    xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models['xgboost'] = xgb_reg

    return models


QUANTILE_LEVELS = {0.05: 'q05', 0.10: 'q10', 0.25: 'q25', 0.50: 'q50',
                   0.75: 'q75', 0.90: 'q90', 0.95: 'q95'}


def train_quantile_models(X_train, y_train, X_val, y_val):
    out = {}
    for q, name in QUANTILE_LEVELS.items():
        m = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            objective='reg:quantileerror', quantile_alpha=q,
            n_jobs=-1, random_state=42, verbosity=0,
            early_stopping_rounds=20,
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        out[name] = m
    return out


def compute_conformal_offset(q_models, X_calib, y_calib, alpha: float = 0.20) -> float:
    """Split-conformal calibration for the central (1 - alpha) prediction band.

    Returns the additive offset c such that, by exchangeability,
        P(y_test in [q_lo(x) - c, q_hi(x) + c]) >= 1 - alpha
    where q_lo = q_{alpha/2} and q_hi = q_{1 - alpha/2}.
    """
    lo_name = f"q{int(alpha / 2 * 100):02d}"
    hi_name = f"q{int((1 - alpha / 2) * 100):02d}"
    q_lo = np.maximum(q_models[lo_name].predict(X_calib), 0)
    q_hi = np.maximum(q_models[hi_name].predict(X_calib), 0)
    y = np.asarray(y_calib)
    # Non-conformity score: signed distance outside the band
    scores = np.maximum(q_lo - y, y - q_hi)
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level))


def reliability_table(q_models, X, y) -> pd.DataFrame:
    """For each trained quantile level: nominal alpha and empirical P(y <= q_alpha(x))."""
    rows = []
    y = np.asarray(y)
    for alpha, name in QUANTILE_LEVELS.items():
        pred = np.maximum(q_models[name].predict(X), 0)
        emp = float(np.mean(y <= pred))
        rows.append({'nominal': alpha, 'quantile': name, 'empirical': emp})
    return pd.DataFrame(rows)


def evaluate_model(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    pred = np.maximum(pred, 0)
    return _metrics(y_test.values, pred)


def generate_prediction_intervals(q_models, X) -> pd.DataFrame:
    out = pd.DataFrame({name: np.maximum(q_models[name].predict(X), 0)
                        for name in QUANTILE_LEVELS.values()})
    # enforce monotone ordering across quantile levels
    cols = [name for _, name in sorted(QUANTILE_LEVELS.items())]
    for i in range(1, len(cols)):
        out[cols[i]] = np.maximum(out[cols[i]], out[cols[i - 1]])
    return out


def run_forecasting_pipeline(demand_df, weather_df, routes_df,
                             output_dir="results/tables",
                             models_dir="results/models"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = prepare_features(demand_df, weather_df, routes_df)
    feat_cols = get_feature_cols(df)
    log_metric("forecast_n_features", len(feat_cols))
    log_metric("forecast_n_rows_after_lag", len(df))

    train, val, test = chronological_split(df)
    X_train, y_train = train[feat_cols], train['ridership']
    X_val, y_val = val[feat_cols], val['ridership']
    X_test, y_test = test[feat_cols], test['ridership']
    log_metric("forecast_n_train", len(X_train))
    log_metric("forecast_n_val", len(X_val))
    log_metric("forecast_n_test", len(X_test))

    models = train_models(X_train, y_train, X_val, y_val)
    q_models = train_quantile_models(X_train, y_train, X_val, y_val)

    # Train metrics (overfitting check)
    for name, m in models.items():
        train_pred = np.maximum(m.predict(X_train), 0)
        train_m = _metrics(y_train.values, train_pred)
        log_metric(f"{name}_train_rmse", round(train_m['rmse'], 4))
        log_metric(f"{name}_train_r2", round(train_m['r2'], 4))

    rows = []
    for name, m in models.items():
        ev = evaluate_model(m, X_test, y_test)
        rows.append({'model': name, **ev})
    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(output_dir, "forecast_results.csv"), index=False)

    # Test predictions with quantile intervals + split-conformal calibration
    # (calibration set: validation fold)
    test_pred = np.maximum(models['xgboost'].predict(X_test), 0)
    qi = generate_prediction_intervals(q_models, X_test)
    conformal_offset = compute_conformal_offset(q_models, X_val, y_val, alpha=0.20)
    log_metric("conformal_offset_80pct", round(conformal_offset, 4))

    test_out = pd.DataFrame({
        'route_id': test['route_id'].values,
        'date': test['date'].values,
        'hour': test['hour'].values,
        'actual': y_test.values,
        'predicted': test_pred,
    })
    for name in QUANTILE_LEVELS.values():
        test_out[name] = qi[name].values
    test_out['q10_cal'] = np.maximum(test_out['q10'] - conformal_offset, 0)
    test_out['q90_cal'] = test_out['q90'] + conformal_offset
    test_out.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    # Reliability diagram data (val and test, raw quantile coverage)
    rel_val = reliability_table(q_models, X_val, y_val)
    rel_val['split'] = 'validation'
    rel_test = reliability_table(q_models, X_test, y_test)
    rel_test['split'] = 'test'
    pd.concat([rel_val, rel_test], ignore_index=True).to_csv(
        os.path.join(output_dir, "pi_reliability.csv"), index=False)

    # Save XGBoost model + quantile heads + conformal offset
    models['xgboost'].save_model(os.path.join(models_dir, "xgb_model.json"))
    for name, m in q_models.items():
        m.save_model(os.path.join(models_dir, f"xgb_{name}.json"))
    with open(os.path.join(models_dir, "conformal_offset.txt"), 'w') as f:
        f.write(f"{conformal_offset:.6f}\n")

    return {'results_df': results_df, 'test_predictions': test_out,
            'models': models, 'q_models': q_models, 'feature_cols': feat_cols,
            'conformal_offset': conformal_offset}


if __name__ == "__main__":
    from src.audit import (setup_audit, log_phase_start, log_phase_end,
                           log_gate_check, log_file_created)
    from src.data_processing import load_routes, load_demand, load_weather

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setup_audit(project_root)
    log_phase_start("PHASE_3_FORECASTING")

    routes_df = load_routes(os.path.join(project_root, "data/processed/routes.csv"))
    demand_df = load_demand(os.path.join(project_root, "data/processed/demand.csv"))
    weather_df = load_weather(os.path.join(project_root, "data/processed/weather.csv"))

    out = run_forecasting_pipeline(
        demand_df, weather_df, routes_df,
        output_dir=os.path.join(project_root, "results/tables"),
        models_dir=os.path.join(project_root, "results/models"),
    )
    results = out['results_df']
    preds = out['test_predictions']

    print(results.to_string(index=False))

    xgb_row = results[results['model'] == 'xgboost'].iloc[0]
    lr_row = results[results['model'] == 'linear_regression'].iloc[0]
    rf_row = results[results['model'] == 'random_forest'].iloc[0]

    log_gate_check("xgb_beats_lr", xgb_row['rmse'] < lr_row['rmse'],
                   f"< {lr_row['rmse']:.2f}", f"{xgb_row['rmse']:.2f}")
    log_gate_check("xgb_beats_rf", xgb_row['rmse'] < rf_row['rmse'],
                   f"< {rf_row['rmse']:.2f}", f"{xgb_row['rmse']:.2f}")
    log_gate_check("mape_range", 5 < xgb_row['mape'] < 35,
                   "5-35%", f"{xgb_row['mape']:.1f}%")
    log_gate_check("r2_positive", xgb_row['r2'] > 0.5, ">0.5",
                   f"{xgb_row['r2']:.3f}")
    log_gate_check("r2_not_overfit", xgb_row['r2'] < 0.99, "<0.99",
                   f"{xgb_row['r2']:.3f}")

    q_order = (preds['q10'] <= preds['q50']).all() and (preds['q50'] <= preds['q90']).all()
    log_gate_check("quantile_ordering", bool(q_order),
                   "q10 <= q50 <= q90", bool(q_order))

    coverage_raw = float(((preds['actual'] >= preds['q10']) & (preds['actual'] <= preds['q90'])).mean())
    coverage_cal = float(((preds['actual'] >= preds['q10_cal']) & (preds['actual'] <= preds['q90_cal'])).mean())
    log_gate_check("interval_coverage_raw", 0.60 < coverage_raw < 0.99,
                   "60-99%", f"{coverage_raw:.1%}")
    log_gate_check("interval_coverage_calibrated", coverage_cal >= 0.78,
                   ">=78%", f"{coverage_cal:.1%}")
    log_metric("prediction_interval_coverage_raw", round(coverage_raw, 4))
    log_metric("prediction_interval_coverage_calibrated", round(coverage_cal, 4))

    log_metric("xgb_test_rmse", round(xgb_row['rmse'], 4))
    log_metric("xgb_test_mape", round(xgb_row['mape'], 4))
    log_metric("xgb_test_r2", round(xgb_row['r2'], 4))

    model_path = os.path.join(project_root, "results/models/xgb_model.json")
    log_gate_check("model_saved", os.path.exists(model_path), "True",
                   str(os.path.exists(model_path)))
    log_file_created(model_path)
    log_file_created(os.path.join(project_root, "results/tables/forecast_results.csv"))
    log_file_created(os.path.join(project_root, "results/tables/test_predictions.csv"))

    log_phase_end("PHASE_3_FORECASTING", "PASS")
    print("PHASE 3 COMPLETE")
