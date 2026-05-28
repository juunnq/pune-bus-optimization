"""Phase 1 gate checks (real-data PMPML instance).

Run from project root:
    python scripts/gate_check_phase1.py

Used as a regression guard by scripts/run_all.py before downstream phases.
Exits with code 1 if any gate FAILs so an outer pipeline halts early.
"""
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
from src.deterministic_model import DWELL_MIN


# Aggregate FAIL count from log_gate_check via a wrapper. The bare audit
# function does not return a status, so we count locally and report.
_FAIL_COUNT = 0


def _gate(name, condition, expected, actual):
    """log_gate_check + local FAIL counter."""
    global _FAIL_COUNT
    log_gate_check(name, condition, expected, actual)
    if not condition:
        _FAIL_COUNT += 1


def main():
    setup_audit(".")
    log_phase_start("PHASE_1_DATA")

    routes_df = load_routes()
    demand_df = load_demand()
    weather_df = load_weather()
    demand_matrix = load_demand_matrix()

    # 1.1 routes -- thresholds anchored to the real PMPML network size
    # (544 routes from OpenCity after collapsing direction pairs).
    _gate("routes_count",
          400 <= len(routes_df) <= 700,
          "400-700 (real PMPML ~544)",
          len(routes_df))
    _gate("routes_no_nulls",
          routes_df.isnull().sum().sum() == 0,
          "0 nulls", int(routes_df.isnull().sum().sum()))
    _gate("routes_positive_length",
          bool((routes_df['length_km'] > 0).all()),
          "all positive", int((routes_df['length_km'] > 0).sum()))

    # 1.2 fleet feasibility -- corrected tau formula
    # (tau = 2 * one_way_trip + DWELL_MIN), matching deterministic_model.
    tau = 2.0 * routes_df['avg_trip_time_min'] + DWELL_MIN
    buses_needed = int(np.ceil(routes_df['current_peak_freq'] * tau / 60).sum())
    _gate("fleet_current_peak_in_range",
          1500 < buses_needed < 2100,
          "1500-2100 (peak utilization < B=2000 cap)",
          buses_needed)
    log_metric("current_fleet_needed", buses_needed)

    # 1.3 demand
    _gate("demand_rows",
          len(demand_df) > 100000,
          ">100000 rows", len(demand_df))
    _gate("demand_no_negative",
          bool((demand_df['ridership'] >= 0).all()),
          "all non-negative",
          int((demand_df['ridership'] >= 0).sum()))
    _gate("demand_reasonable_max",
          demand_df['ridership'].max() < 5000,
          "max < 5000",
          round(float(demand_df['ridership'].max()), 1))
    log_metric("demand_mean_ridership",
               round(float(demand_df['ridership'].mean()), 2))

    # 1.4 weather
    _gate("weather_days",
          len(weather_df) >= 365,
          ">=365 days", len(weather_df))
    _gate("temp_range",
          weather_df['temperature_max'].max() < 50,
          "max < 50C",
          round(float(weather_df['temperature_max'].max()), 1))

    # 1.5 demand matrix -- threshold matches the real network size.
    _gate("demand_matrix_routes",
          len(demand_matrix) >= 400,
          ">=400 routes",
          len(demand_matrix))
    _gate("demand_matrix_periods",
          demand_matrix.shape[1] >= 5,
          ">=5 periods", demand_matrix.shape[1])

    # 1.6 files on disk -- log_file_created takes only `path`.
    for f in ['routes.csv', 'demand.csv', 'weather.csv', 'demand_matrix.csv']:
        path = f"data/processed/{f}"
        exists = os.path.exists(path)
        _gate(f"file_exists_{f}", exists, "True", exists)
        if exists:
            log_file_created(path)

    status = "PASS" if _FAIL_COUNT == 0 else "FAIL"
    log_phase_end("PHASE_1_DATA", status, {
        "routes": len(routes_df),
        "demand_rows": len(demand_df),
        "weather_days": len(weather_df),
        "fleet_needed": buses_needed,
        "failed_gates": _FAIL_COUNT,
    })

    if _FAIL_COUNT > 0:
        print(f"\nPHASE 1 FAILED - {_FAIL_COUNT} gate(s) did not pass.",
              file=sys.stderr)
        print("  See audit.log for the offending gates.", file=sys.stderr)
        sys.exit(1)

    print("\nPHASE 1 COMPLETE - all gates passed")
    print(f"  Routes:                {len(routes_df)}")
    print(f"  Demand rows:           {len(demand_df)}")
    print(f"  Weather days:          {len(weather_df)}")
    print(f"  Current peak fleet:    {buses_needed}")


if __name__ == "__main__":
    main()
