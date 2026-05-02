"""Phase 1 gate checks. Run from project root."""
import os
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.audit import (
    setup_audit, log_phase_start, log_phase_end,
    log_gate_check, log_metric, log_file_created,
)
from src.data_processing import (
    load_routes, load_demand, load_weather, load_demand_matrix,
)


def main():
    setup_audit(".")
    log_phase_start("PHASE_1_DATA")

    routes_df = load_routes()
    demand_df = load_demand()
    weather_df = load_weather()
    demand_matrix = load_demand_matrix()

    # 1.1 routes
    log_gate_check("routes_count", 300 <= len(routes_df) <= 380, "300-380", len(routes_df))
    log_gate_check("routes_no_nulls",
                   routes_df.isnull().sum().sum() == 0,
                   "0 nulls", int(routes_df.isnull().sum().sum()))
    log_gate_check("routes_positive_length",
                   bool((routes_df['length_km'] > 0).all()),
                   "all positive", int((routes_df['length_km'] > 0).sum()))

    # 1.2 fleet feasibility
    buses_needed = int(np.ceil(routes_df['current_peak_freq']
                               * routes_df['avg_trip_time_min'] / 60).sum())
    log_gate_check("fleet_current_feasible",
                   1500 < buses_needed < 2200,
                   "1500-2200", buses_needed)
    log_metric("current_fleet_needed", buses_needed)

    # 1.3 demand
    log_gate_check("demand_rows", len(demand_df) > 100000, ">100000 rows", len(demand_df))
    log_gate_check("demand_no_negative",
                   bool((demand_df['ridership'] >= 0).all()),
                   "all non-negative", int((demand_df['ridership'] >= 0).sum()))
    log_gate_check("demand_reasonable_max",
                   demand_df['ridership'].max() < 5000,
                   "max < 5000", round(float(demand_df['ridership'].max()), 1))
    log_metric("demand_mean_ridership", round(float(demand_df['ridership'].mean()), 2))

    # 1.4 weather
    log_gate_check("weather_days", len(weather_df) >= 365, ">=365 days", len(weather_df))
    log_gate_check("temp_range",
                   weather_df['temperature_max'].max() < 50,
                   "max < 50C", round(float(weather_df['temperature_max'].max()), 1))

    # 1.5 demand matrix
    log_gate_check("demand_matrix_routes",
                   len(demand_matrix) >= 300,
                   ">=300 routes", len(demand_matrix))
    log_gate_check("demand_matrix_periods",
                   demand_matrix.shape[1] >= 5,
                   ">=5 periods", demand_matrix.shape[1])

    # 1.6 files on disk
    for f in ['routes.csv', 'demand.csv', 'weather.csv', 'demand_matrix.csv']:
        path = f"data/processed/{f}"
        exists = os.path.exists(path)
        log_gate_check(f"file_exists_{f}", exists, "True", exists)
        if exists:
            log_file_created(path, os.path.getsize(path))

    log_phase_end("PHASE_1_DATA", "PASS", {
        "routes": len(routes_df),
        "demand_rows": len(demand_df),
        "weather_days": len(weather_df),
        "fleet_needed": buses_needed,
    })

    print("\nPHASE 1 COMPLETE - all gates passed")
    print(f"  Routes: {len(routes_df)}")
    print(f"  Demand rows: {len(demand_df)}")
    print(f"  Weather days: {len(weather_df)}")
    print(f"  Current fleet needed: {buses_needed}")


if __name__ == "__main__":
    main()
