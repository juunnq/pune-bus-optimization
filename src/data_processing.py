"""
Data loading and feature preparation for pune-bus-optimization.
"""
import os
import pandas as pd
import numpy as np

PERIOD_BOUNDS = [
    ("early_morning", 6, 8),
    ("AM_peak", 8, 10),
    ("midday", 10, 17),
    ("PM_peak", 17, 20),
    ("evening", 20, 23),
]


def get_time_period(hour: int) -> str:
    for name, lo, hi in PERIOD_BOUNDS:
        if lo <= hour < hi:
            return name
    return "off_hours"


def load_routes(path="data/processed/routes.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_weather(path="data/processed/weather.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_demand(path="data/processed/demand.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_demand_matrix(path="data/processed/demand_matrix.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_ml_features(demand_df: pd.DataFrame,
                        weather_df: pd.DataFrame,
                        routes_df: pd.DataFrame):
    """Returns (X, y) ready for training. Joins demand with route metadata,
    one-hot encodes route_category, returns numeric features + ridership target."""
    df = demand_df.merge(
        routes_df[['route_id', 'route_category', 'length_km', 'priority_score']],
        on='route_id', how='left'
    )
    df = pd.get_dummies(df, columns=['route_category'], drop_first=True)

    feature_cols = [
        'hour', 'day_of_week', 'is_weekend', 'is_holiday', 'month',
        'temperature_max', 'precipitation_mm', 'is_monsoon',
        'is_exam_season', 'is_festival', 'length_km', 'priority_score',
    ]
    feature_cols += [c for c in df.columns if c.startswith('route_category_')]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df['ridership'].copy()
    return X, y
