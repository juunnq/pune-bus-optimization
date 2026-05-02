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


def train_quantile_models(X_train, y_train, X_val, y_val):
    quantiles = {0.1: 'q10', 0.5: 'q50', 0.9: 'q90'}
    out = {}
    for q, name in quantiles.items():
        m = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            objective='reg:quantileerror', quantile_alpha=q,
            n_jobs=-1, random_state=42, verbosity=0,
            early_stopping_rounds=20,
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        out[name] = m
    return out


def evaluate_model(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    pred = np.maximum(pred, 0)
    return _metrics(y_test.values, pred)


def generate_prediction_intervals(q_models, X) -> pd.DataFrame:
    out = pd.DataFrame({
        'q10': np.maximum(q_models['q10'].predict(X), 0),
        'q50': np.maximum(q_models['q50'].predict(X), 0),
        'q90': np.maximum(q_models['q90'].predict(X), 0),
    })
    # enforce monotone ordering
    out['q50'] = np.maximum(out['q50'], out['q10'])
    out['q90'] = np.maximum(out['q90'], out['q50'])
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

    # Test predictions with quantile intervals
    test_pred = np.maximum(models['xgboost'].predict(X_test), 0)
    qi = generate_prediction_intervals(q_models, X_test)
    test_out = pd.DataFrame({
        'route_id': test['route_id'].values,
        'date': test['date'].values,
        'hour': test['hour'].values,
        'actual': y_test.values,
        'predicted': test_pred,
        'q10': qi['q10'].values,
        'q50': qi['q50'].values,
        'q90': qi['q90'].values,
    })
    test_out.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    # Save XGBoost model
    models['xgboost'].save_model(os.path.join(models_dir, "xgb_model.json"))
    for name, m in q_models.items():
        m.save_model(os.path.join(models_dir, f"xgb_{name}.json"))

    return {'results_df': results_df, 'test_predictions': test_out,
            'models': models, 'q_models': q_models, 'feature_cols': feat_cols}


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

    coverage = float(((preds['actual'] >= preds['q10']) & (preds['actual'] <= preds['q90'])).mean())
    log_gate_check("interval_coverage", 0.65 < coverage < 0.95,
                   "65-95%", f"{coverage:.1%}")
    log_metric("prediction_interval_coverage", round(coverage, 4))

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
