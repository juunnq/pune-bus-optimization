"""
Stochastic and robust (minimax) frequency allocation MIPs.

Builds 6 demand scenarios from quantile predictions, solves:
  - Stochastic MIP: minimize probability-weighted expected wait
  - Robust MIP: minimize worst-case wait across scenarios
And computes:
  - Value of Stochastic Solution (VSS) = EEV - RP
  - Expected Value of Perfect Information (EVPI) = RP - WS
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit import log_metric
from src.deterministic_model import (
    FREQ_LEVELS, TIME_PERIODS, PEAK_PERIODS,
    PRIORITY_THRESHOLD, PRIORITY_MIN_FREQ_PEAK,
    _cycle_time_min, _wait_time, _buses_needed, _depot_capacities,
    build_deterministic_model, evaluate_allocation,
    compute_baseline_status_quo,
)


SCENARIO_DEFS = [
    ('S1_expected', 'q50 baseline', 0.35),
    ('S2_high_demand', 'q90 high', 0.15),
    ('S3_low_demand', 'q10 low', 0.15),
    ('S4_monsoon', 'q10 with rain dampening', 0.15),
    ('S5_peak_surge', 'q90 +15% on trunk during peak', 0.10),
    ('S6_festival_dip', 'q50 x 0.75 all routes', 0.10),
]


# Scenario generation

def _aggregate_quantile_to_matrix(test_predictions: pd.DataFrame,
                                  routes_df: pd.DataFrame,
                                  demand_matrix: pd.DataFrame,
                                  quantile_col: str) -> pd.DataFrame:
    """Aggregate hourly quantile predictions to a (route x period) matrix.

    For routes IN test_predictions: average the quantile across hours
    in each period across the test set.
    For routes NOT in test_predictions: scale demand_matrix by the
    average ratio quantile/expected observed on the modeled routes.
    """
    from src.data_processing import get_time_period

    df = test_predictions.copy()
    df['period'] = df['hour'].apply(get_time_period)
    df = df[df['period'] != 'off_hours']

    modeled = (df.groupby(['route_id', 'period'])[quantile_col]
                  .mean()
                  .unstack('period')
                  .reindex(columns=TIME_PERIODS))

    matrix = demand_matrix.set_index('route_id')[TIME_PERIODS]
    overlap = modeled.index.intersection(matrix.index)
    if len(overlap) > 0:
        ratio = (modeled.loc[overlap] / matrix.loc[overlap]).replace(
            [np.inf, -np.inf], np.nan).fillna(1.0)
        avg_ratio = ratio.mean()
    else:
        avg_ratio = pd.Series(1.0, index=TIME_PERIODS)

    out = matrix.copy()
    out = out.multiply(avg_ratio, axis=1)
    out.loc[overlap] = modeled.loc[overlap].values
    out = out.fillna(matrix)
    return out.reset_index()


def generate_scenarios(demand_matrix: pd.DataFrame,
                       test_predictions: pd.DataFrame,
                       routes_df: pd.DataFrame) -> tuple[dict, dict]:
    """Returns ({scenario_name: demand_matrix_df}, {scenario_name: probability})."""
    q10_mat = _aggregate_quantile_to_matrix(test_predictions, routes_df, demand_matrix, 'q10')
    q50_mat = _aggregate_quantile_to_matrix(test_predictions, routes_df, demand_matrix, 'q50')
    q90_mat = _aggregate_quantile_to_matrix(test_predictions, routes_df, demand_matrix, 'q90')

    scenarios = {}
    probabilities = {}

    scenarios['S1_expected'] = q50_mat.copy()
    scenarios['S2_high_demand'] = q90_mat.copy()
    scenarios['S3_low_demand'] = q10_mat.copy()

    s4 = q10_mat.copy()
    for t in TIME_PERIODS:
        s4[t] = s4[t] * 0.8
    scenarios['S4_monsoon'] = s4

    s5 = q50_mat.copy()
    trunk_ids = set(routes_df[routes_df['route_category'] == 'trunk']['route_id'])
    is_trunk = s5['route_id'].isin(trunk_ids)
    for t in PEAK_PERIODS:
        boosted = q90_mat[t] * 1.15
        s5.loc[is_trunk, t] = boosted[is_trunk].values
    scenarios['S5_peak_surge'] = s5

    s6 = q50_mat.copy()
    for t in TIME_PERIODS:
        s6[t] = s6[t] * 0.75
    scenarios['S6_festival_dip'] = s6

    for name, _, p in SCENARIO_DEFS:
        probabilities[name] = p
    return scenarios, probabilities


# Stochastic MIP

def build_stochastic_model(routes_df: pd.DataFrame,
                           scenarios: dict,
                           probabilities: dict,
                           fleet_size: int = 2000,
                           depot_caps: dict = None,
                           time_limit_sec: int = 300) -> dict:
    routes = list(routes_df['route_id'])
    tau = _cycle_time_min(routes_df)
    priority = routes_df.set_index('route_id')['priority_score'].astype(float)
    depot_of = routes_df.set_index('route_id')['depot_id']
    if depot_caps is None:
        depot_caps = _depot_capacities(routes_df, fleet_size)

    combined = None
    for name, dm in scenarios.items():
        p = probabilities[name]
        m = dm.set_index('route_id')[TIME_PERIODS] * p
        combined = m if combined is None else combined.add(m, fill_value=0)

    b = {(r, k): _buses_needed(tau[r], k) for r in routes for k in FREQ_LEVELS}
    w = {k: _wait_time(k) for k in FREQ_LEVELS}

    model = pulp.LpProblem("PMPML_Stochastic", pulp.LpMinimize)
    x = {(r, t, k): pulp.LpVariable(f"x_{r}_{t}_{k}", cat='Binary')
         for r in routes for t in TIME_PERIODS for k in FREQ_LEVELS}

    model += pulp.lpSum(
        combined.loc[r, t] * w[k] * x[(r, t, k)]
        for r in routes for t in TIME_PERIODS for k in FREQ_LEVELS
    )
    for r in routes:
        for t in TIME_PERIODS:
            model += pulp.lpSum(x[(r, t, k)] for k in FREQ_LEVELS) == 1
    for t in TIME_PERIODS:
        model += pulp.lpSum(b[(r, k)] * x[(r, t, k)]
                            for r in routes for k in FREQ_LEVELS) <= fleet_size
    for r in routes:
        if priority[r] > PRIORITY_THRESHOLD:
            for t in PEAK_PERIODS:
                model += pulp.lpSum(x[(r, t, k)] for k in FREQ_LEVELS
                                    if k >= PRIORITY_MIN_FREQ_PEAK) == 1
    for d_id, cap in depot_caps.items():
        depot_routes = [r for r in routes if depot_of[r] == d_id]
        for t in TIME_PERIODS:
            model += pulp.lpSum(b[(r, k)] * x[(r, t, k)]
                                for r in depot_routes for k in FREQ_LEVELS) <= cap

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_sec)
    t0 = time.time()
    status_code = model.solve(solver)
    solve_time = time.time() - t0

    rows = []
    for r in routes:
        for t in TIME_PERIODS:
            chosen_k = FREQ_LEVELS[0]
            for k in FREQ_LEVELS:
                v = pulp.value(x[(r, t, k)])
                if v is not None and v > 0.5:
                    chosen_k = k
                    break
            rows.append({'route_id': r, 'period': t,
                         'frequency': chosen_k,
                         'buses_assigned': b[(r, chosen_k)]})
    return {
        'status': pulp.LpStatus[status_code],
        'objective': float(pulp.value(model.objective)) if pulp.value(model.objective) is not None else float('nan'),
        'solve_time': solve_time,
        'allocation_df': pd.DataFrame(rows),
    }


# Robust (minimax) MIP

def build_robust_model(routes_df: pd.DataFrame,
                       scenarios: dict,
                       fleet_size: int = 2000,
                       depot_caps: dict = None,
                       time_limit_sec: int = 300) -> dict:
    routes = list(routes_df['route_id'])
    tau = _cycle_time_min(routes_df)
    priority = routes_df.set_index('route_id')['priority_score'].astype(float)
    depot_of = routes_df.set_index('route_id')['depot_id']
    if depot_caps is None:
        depot_caps = _depot_capacities(routes_df, fleet_size)

    b = {(r, k): _buses_needed(tau[r], k) for r in routes for k in FREQ_LEVELS}
    w = {k: _wait_time(k) for k in FREQ_LEVELS}

    model = pulp.LpProblem("PMPML_Robust", pulp.LpMinimize)
    x = {(r, t, k): pulp.LpVariable(f"x_{r}_{t}_{k}", cat='Binary')
         for r in routes for t in TIME_PERIODS for k in FREQ_LEVELS}
    z = pulp.LpVariable("z", lowBound=0)
    model += z

    for name, dm in scenarios.items():
        d = dm.set_index('route_id')[TIME_PERIODS]
        model += pulp.lpSum(
            d.loc[r, t] * w[k] * x[(r, t, k)]
            for r in routes for t in TIME_PERIODS for k in FREQ_LEVELS
        ) <= z, f"worst_{name}"

    for r in routes:
        for t in TIME_PERIODS:
            model += pulp.lpSum(x[(r, t, k)] for k in FREQ_LEVELS) == 1
    for t in TIME_PERIODS:
        model += pulp.lpSum(b[(r, k)] * x[(r, t, k)]
                            for r in routes for k in FREQ_LEVELS) <= fleet_size
    for r in routes:
        if priority[r] > PRIORITY_THRESHOLD:
            for t in PEAK_PERIODS:
                model += pulp.lpSum(x[(r, t, k)] for k in FREQ_LEVELS
                                    if k >= PRIORITY_MIN_FREQ_PEAK) == 1
    for d_id, cap in depot_caps.items():
        depot_routes = [r for r in routes if depot_of[r] == d_id]
        for t in TIME_PERIODS:
            model += pulp.lpSum(b[(r, k)] * x[(r, t, k)]
                                for r in depot_routes for k in FREQ_LEVELS) <= cap

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_sec)
    t0 = time.time()
    status_code = model.solve(solver)
    solve_time = time.time() - t0

    rows = []
    for r in routes:
        for t in TIME_PERIODS:
            chosen_k = FREQ_LEVELS[0]
            for k in FREQ_LEVELS:
                v = pulp.value(x[(r, t, k)])
                if v is not None and v > 0.5:
                    chosen_k = k
                    break
            rows.append({'route_id': r, 'period': t,
                         'frequency': chosen_k,
                         'buses_assigned': b[(r, chosen_k)]})
    return {
        'status': pulp.LpStatus[status_code],
        'objective': float(pulp.value(z)) if pulp.value(z) is not None else float('nan'),
        'solve_time': solve_time,
        'allocation_df': pd.DataFrame(rows),
    }


# Evaluation across scenarios

def evaluate_solution_across_scenarios(allocation_df: pd.DataFrame,
                                       scenarios: dict,
                                       routes_df: pd.DataFrame) -> dict:
    return {name: evaluate_allocation(allocation_df, dm, routes_df)['total_wait_time']
            for name, dm in scenarios.items()}


# VSS / EVPI / WS

def compute_ws(scenarios: dict,
               probabilities: dict,
               routes_df: pd.DataFrame,
               fleet_size: int = 2000,
               depot_caps: dict = None) -> dict:
    """Solve each scenario independently. WS = sum_s p_s * z_s*."""
    if depot_caps is None:
        depot_caps = _depot_capacities(routes_df, fleet_size)
    per_scenario = {}
    ws_total = 0.0
    for name, dm in scenarios.items():
        res = build_deterministic_model(routes_df, dm, fleet_size=fleet_size,
                                        time_limit_sec=120)
        per_scenario[name] = res['objective']
        ws_total += probabilities[name] * res['objective']
    return {'ws': ws_total, 'per_scenario': per_scenario}


def compute_vss_evpi(det_alloc: pd.DataFrame,
                     stoch_alloc: pd.DataFrame,
                     scenarios: dict,
                     probabilities: dict,
                     routes_df: pd.DataFrame,
                     fleet_size: int = 2000,
                     depot_caps: dict = None) -> dict:
    eev_per = evaluate_solution_across_scenarios(det_alloc, scenarios, routes_df)
    rp_per = evaluate_solution_across_scenarios(stoch_alloc, scenarios, routes_df)
    eev = sum(probabilities[n] * eev_per[n] for n in scenarios)
    rp = sum(probabilities[n] * rp_per[n] for n in scenarios)
    ws_res = compute_ws(scenarios, probabilities, routes_df, fleet_size, depot_caps)
    return {
        'EEV': eev,
        'RP': rp,
        'WS': ws_res['ws'],
        'VSS': eev - rp,
        'EVPI': rp - ws_res['ws'],
        'ws_per_scenario': ws_res['per_scenario'],
    }


# Driver

def run_stochastic_pipeline(routes_df, demand_matrix, test_predictions,
                            fleet_size=2000,
                            output_dir="results/tables"):
    os.makedirs(output_dir, exist_ok=True)

    scenarios, probabilities = generate_scenarios(
        demand_matrix, test_predictions, routes_df
    )

    sc_def_rows = [{'scenario': n, 'description': desc, 'probability': p}
                   for (n, desc, p) in SCENARIO_DEFS]
    pd.DataFrame(sc_def_rows).to_csv(
        os.path.join(output_dir, "scenario_definitions.csv"), index=False)

    depot_caps = _depot_capacities(routes_df, fleet_size)

    # Deterministic on expected demand (q50 aggregated)
    det_res = build_deterministic_model(routes_df, scenarios['S1_expected'],
                                        fleet_size=fleet_size, time_limit_sec=300)
    log_metric("stoch_det_solve_sec", round(det_res['solve_time'], 2))

    stoch_res = build_stochastic_model(routes_df, scenarios, probabilities,
                                       fleet_size=fleet_size,
                                       depot_caps=depot_caps,
                                       time_limit_sec=300)
    log_metric("stoch_stochastic_solve_sec", round(stoch_res['solve_time'], 2))

    rob_res = build_robust_model(routes_df, scenarios, fleet_size=fleet_size,
                                 depot_caps=depot_caps, time_limit_sec=300)
    log_metric("stoch_robust_solve_sec", round(rob_res['solve_time'], 2))

    sq_alloc = compute_baseline_status_quo(routes_df, demand_matrix)

    method_alloc = {
        'deterministic': det_res['allocation_df'],
        'stochastic': stoch_res['allocation_df'],
        'robust': rob_res['allocation_df'],
        'status_quo': sq_alloc,
    }
    rows = []
    for method, alloc in method_alloc.items():
        per = evaluate_solution_across_scenarios(alloc, scenarios, routes_df)
        for sc, val in per.items():
            rows.append({'method': method, 'scenario': sc, 'objective_value': val})
    stoch_results_df = pd.DataFrame(rows)
    stoch_results_df.to_csv(os.path.join(output_dir, "stochastic_results.csv"), index=False)

    metrics = compute_vss_evpi(
        det_res['allocation_df'], stoch_res['allocation_df'],
        scenarios, probabilities, routes_df, fleet_size=fleet_size,
        depot_caps=depot_caps,
    )
    vss_evpi_df = pd.DataFrame([
        {'metric': 'EEV', 'value': metrics['EEV']},
        {'metric': 'RP', 'value': metrics['RP']},
        {'metric': 'WS', 'value': metrics['WS']},
        {'metric': 'VSS', 'value': metrics['VSS']},
        {'metric': 'EVPI', 'value': metrics['EVPI']},
    ])
    vss_evpi_df.to_csv(os.path.join(output_dir, "vss_evpi.csv"), index=False)

    return {
        'stochastic_results': stoch_results_df,
        'vss_evpi': vss_evpi_df,
        'allocations': method_alloc,
        'metrics': metrics,
    }


if __name__ == "__main__":
    from src.audit import (setup_audit, log_phase_start, log_phase_end,
                           log_gate_check, log_file_created)
    from src.data_processing import load_routes, load_demand_matrix

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setup_audit(project_root)
    log_phase_start("PHASE_4_STOCHASTIC")

    routes_df = load_routes(os.path.join(project_root, "data/processed/routes.csv"))
    demand_matrix = load_demand_matrix(os.path.join(project_root, "data/processed/demand_matrix.csv"))
    test_predictions = pd.read_csv(os.path.join(project_root, "results/tables/test_predictions.csv"))

    out = run_stochastic_pipeline(
        routes_df, demand_matrix, test_predictions,
        fleet_size=2000,
        output_dir=os.path.join(project_root, "results/tables"),
    )
    stoch_results = out['stochastic_results']
    vss_evpi = out['vss_evpi']
    print(stoch_results.to_string(index=False))
    print(vss_evpi.to_string(index=False))

    vss = vss_evpi[vss_evpi['metric'] == 'VSS']['value'].iloc[0]
    evpi = vss_evpi[vss_evpi['metric'] == 'EVPI']['value'].iloc[0]

    for method in ['deterministic', 'stochastic', 'robust', 'status_quo']:
        n = len(stoch_results[stoch_results['method'] == method])
        log_gate_check(f"{method}_all_scenarios", n == 6, "6", n)
    log_gate_check("vss_non_negative", vss >= -0.01, ">=0", round(vss, 4))
    log_gate_check("evpi_non_negative", evpi >= -0.01, ">=0", round(evpi, 4))
    log_gate_check("evpi_geq_vss", evpi >= vss - 0.01, f">= {vss:.4f}", round(evpi, 4))

    methods_worst = stoch_results.groupby('method')['objective_value'].max()
    robust_worst = methods_worst['robust']
    for method in ['deterministic', 'stochastic', 'status_quo']:
        log_gate_check(f"robust_worst_vs_{method}",
                       robust_worst <= methods_worst[method] + 0.01,
                       f"<= {methods_worst[method]:.2f}",
                       round(robust_worst, 2))
    log_metric("vss", round(float(vss), 4))
    log_metric("evpi", round(float(evpi), 4))

    stoch_avg = (stoch_results[stoch_results['method'] == 'stochastic']['objective_value'].mean())
    det_avg = (stoch_results[stoch_results['method'] == 'deterministic']['objective_value'].mean())
    log_gate_check("stochastic_beats_deterministic_avg",
                   stoch_avg <= det_avg + 0.01,
                   f"<= {det_avg:.2f}", round(stoch_avg, 2))

    log_file_created(os.path.join(project_root, "results/tables/stochastic_results.csv"))
    log_file_created(os.path.join(project_root, "results/tables/vss_evpi.csv"))
    log_file_created(os.path.join(project_root, "results/tables/scenario_definitions.csv"))
    log_phase_end("PHASE_4_STOCHASTIC", "PASS")
    print("PHASE 4 COMPLETE")
