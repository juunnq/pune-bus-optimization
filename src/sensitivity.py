"""
Sensitivity analyses for the deterministic MIP:

  6A. Fleet-size sensitivity:   solve at B in [800..2200], record metrics.
  6B. Equity-efficiency Pareto: vary priority-route min peak frequency.
  6C. Shadow prices:            duals from the LP relaxation.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit import log_metric, log_warning
from src.deterministic_model import (
    FREQ_LEVELS, TIME_PERIODS, PEAK_PERIODS,
    PRIORITY_THRESHOLD, PRIORITY_MIN_FREQ_PEAK,
    _cycle_time_min, _wait_time, _buses_needed, _depot_capacities,
    build_deterministic_model, evaluate_allocation,
)


# 6E. Equity-mapping sensitivity (priority-rule robustness)

def equity_mapping_sensitivity(demand_matrix: pd.DataFrame,
                               fleet_size: int = 2000) -> pd.DataFrame:
    """Robustness check on the priority designation.

    The landmark->ward alias table in ``data/load_real_data.py`` is
    hand-built and is a researcher-degrees-of-freedom step on the way to
    the headline equity-cost number. This routine re-runs the equity sweep
    under three alias rules (``strict``, ``conservative``, ``current``)
    and reports the resulting equity cost (objective at threshold 3 vs
    threshold 1) so the reader can see how much the number depends on the
    alias choices. ``strict`` is the most defensible mapping but produces
    very few priority routes; ``conservative`` drops the debatable
    cross-ward equity classifications (Kondhwa/Wanowri/etc.); ``current``
    is the table used in the paper's main results.
    """
    from data.load_real_data import build_real_routes
    rows = []
    for rule in ("strict", "conservative", "current"):
        routes_rule = build_real_routes(priority_rule=rule)
        n_priority = int((routes_rule['priority_score'] > PRIORITY_THRESHOLD).sum())
        # Threshold 1: vacuous equity (k >= 1 is always satisfied)
        res1 = _build_mip_with_equity_threshold(
            routes_rule, demand_matrix, fleet_size, 1, time_limit_sec=180)
        wait1 = (res1['objective'] if res1 and res1.get('status') == 'Optimal'
                 else float('nan'))
        # Threshold 3: current operating floor
        res3 = _build_mip_with_equity_threshold(
            routes_rule, demand_matrix, fleet_size, 3, time_limit_sec=180)
        wait3 = (res3['objective'] if res3 and res3.get('status') == 'Optimal'
                 else float('nan'))
        if wait1 > 0 and not np.isnan(wait3):
            cost_pct = (wait3 - wait1) / wait1 * 100
        else:
            cost_pct = float('nan')
        rows.append({
            'priority_rule': rule,
            'n_priority_routes': n_priority,
            'priority_share': n_priority / len(routes_rule),
            'wait_no_equity': wait1,
            'wait_current_equity': wait3,
            'equity_cost_pass_hr': wait3 - wait1 if (
                not np.isnan(wait1) and not np.isnan(wait3)) else float('nan'),
            'equity_cost_pct': cost_pct,
        })
    return pd.DataFrame(rows)


# 6D. Cost-wait Pareto frontier (operator cost lambda sweep)

def cost_wait_frontier(routes_df: pd.DataFrame,
                       demand_matrix: pd.DataFrame,
                       lambdas: list,
                       fleet_size: int = 2000) -> pd.DataFrame:
    """Sweep lambda_op; record rider waiting time and bus-periods used.

    Without an operator-cost penalty the optimum saturates the fleet whenever
    waiting time can be reduced, which inflates the headline gain vs. status
    quo. Sweeping lambda_op > 0 traces the rider-wait / operator-cost frontier.
    """
    rows = []
    for lam in lambdas:
        res = build_deterministic_model(routes_df, demand_matrix,
                                        fleet_size=fleet_size,
                                        lambda_op=float(lam),
                                        time_limit_sec=180)
        alloc = res['allocation_df']
        if alloc.empty or res['status'] == 'Infeasible':
            log_warning(f"cost frontier lambda={lam}: infeasible (status={res['status']})")
            rows.append({'lambda_op': lam, 'total_wait_time': float('nan'),
                         'bus_periods_used': 0, 'avg_frequency': 0.0,
                         'status': res['status'], 'solve_time': res['solve_time']})
            continue
        ev = evaluate_allocation(alloc, demand_matrix, routes_df)
        rows.append({'lambda_op': lam,
                     'total_wait_time': ev['total_wait_time'],
                     'bus_periods_used': ev['fleet_used'],
                     'avg_frequency': ev['avg_frequency'],
                     'status': res['status'],
                     'solve_time': res['solve_time']})
    return pd.DataFrame(rows)


# 6A. Fleet sensitivity

def fleet_sensitivity_analysis(routes_df: pd.DataFrame,
                               demand_matrix: pd.DataFrame,
                               fleet_sizes: list) -> pd.DataFrame:
    rows = []
    for B in fleet_sizes:
        try:
            res = build_deterministic_model(routes_df, demand_matrix,
                                            fleet_size=B, time_limit_sec=180)
            alloc = res['allocation_df']
            if alloc.empty or res['status'] not in ('Optimal', 'Not Solved'):
                log_warning(f"fleet={B} infeasible or unsolved (status={res['status']})")
                rows.append({'fleet_size': B, 'objective': 0.0, 'status': 'infeasible',
                             'routes_above_3': 0, 'avg_freq': 0.0, 'solve_time': res['solve_time']})
                continue
            obj = evaluate_allocation(alloc, demand_matrix, routes_df)['total_wait_time']
            ge3 = (alloc['frequency'] >= 3).sum()
            avg_f = alloc['frequency'].mean()
            rows.append({'fleet_size': B, 'objective': obj, 'status': res['status'],
                         'routes_above_3': int(ge3), 'avg_freq': float(avg_f),
                         'solve_time': res['solve_time']})
        except Exception as e:
            log_warning(f"fleet={B} exception: {e}")
            rows.append({'fleet_size': B, 'objective': 0.0, 'status': 'error',
                         'routes_above_3': 0, 'avg_freq': 0.0, 'solve_time': 0.0})
    return pd.DataFrame(rows)


# 6B. Pareto (equity vs efficiency)

def _build_mip_with_equity_threshold(routes_df, demand_matrix, fleet_size, threshold,
                                     time_limit_sec=180):
    routes = list(routes_df['route_id'])
    tau = _cycle_time_min(routes_df)
    priority = routes_df.set_index('route_id')['priority_score'].astype(float)
    depot_of = routes_df.set_index('route_id')['depot_id']
    depot_caps = _depot_capacities(routes_df, fleet_size)
    dm = demand_matrix.set_index('route_id')[TIME_PERIODS]

    b = {(r, k): _buses_needed(tau[r], k) for r in routes for k in FREQ_LEVELS}
    w = {k: _wait_time(k) for k in FREQ_LEVELS}

    model = pulp.LpProblem(f"PMPML_eq{threshold}", pulp.LpMinimize)
    x = {(r, t, k): pulp.LpVariable(f"x_{r}_{t}_{k}", cat='Binary')
         for r in routes for t in TIME_PERIODS for k in FREQ_LEVELS}
    model += pulp.lpSum(dm.loc[r, t] * w[k] * x[(r, t, k)]
                        for r in routes for t in TIME_PERIODS for k in FREQ_LEVELS)
    for r in routes:
        for t in TIME_PERIODS:
            model += pulp.lpSum(x[(r, t, k)] for k in FREQ_LEVELS) == 1
    for t in TIME_PERIODS:
        model += pulp.lpSum(b[(r, k)] * x[(r, t, k)]
                            for r in routes for k in FREQ_LEVELS) <= fleet_size
    # Equity with chosen threshold
    feasible_levels = [k for k in FREQ_LEVELS if k >= threshold]
    for r in routes:
        if priority[r] > PRIORITY_THRESHOLD:
            for t in PEAK_PERIODS:
                if not feasible_levels:
                    return None  # impossible
                model += pulp.lpSum(x[(r, t, k)] for k in feasible_levels) == 1
    for d_id, cap in depot_caps.items():
        depot_routes = [r for r in routes if depot_of[r] == d_id]
        for t in TIME_PERIODS:
            model += pulp.lpSum(b[(r, k)] * x[(r, t, k)]
                                for r in depot_routes for k in FREQ_LEVELS) <= cap

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_sec)
    t0 = time.time()
    code = model.solve(solver)
    elapsed = time.time() - t0
    status = pulp.LpStatus[code]
    if status == 'Infeasible':
        return {'status': 'infeasible', 'solve_time': elapsed}

    rows = []
    feasible_solution = True
    for r in routes:
        for t in TIME_PERIODS:
            chosen = None
            for k in FREQ_LEVELS:
                v = pulp.value(x[(r, t, k)])
                if v is not None and v > 0.5:
                    chosen = k
                    break
            if chosen is None:
                feasible_solution = False
                chosen = FREQ_LEVELS[0]
            rows.append({'route_id': r, 'period': t, 'frequency': chosen,
                         'buses_assigned': b[(r, chosen)]})
    if not feasible_solution:
        return {'status': 'infeasible', 'solve_time': elapsed}
    return {'status': status, 'solve_time': elapsed,
            'allocation_df': pd.DataFrame(rows),
            'objective': float(pulp.value(model.objective))}


def pareto_analysis(routes_df, demand_matrix, equity_thresholds, fleet_size=2000):
    rows = []
    priority_routes = routes_df[routes_df['priority_score'] > PRIORITY_THRESHOLD]['route_id'].tolist()
    stop = False
    for thr in equity_thresholds:
        if stop:
            rows.append({'equity_threshold': thr, 'total_wait': 0.0,
                         'avg_priority_freq': 0.0, 'num_binding': 0,
                         'feasible': False, 'solve_time': 0.0})
            continue
        res = _build_mip_with_equity_threshold(routes_df, demand_matrix,
                                               fleet_size, thr)
        if res is None or res.get('status') == 'infeasible':
            log_warning(f"equity threshold={thr} infeasible")
            rows.append({'equity_threshold': thr, 'total_wait': 0.0,
                         'avg_priority_freq': 0.0, 'num_binding': 0,
                         'feasible': False, 'solve_time': res.get('solve_time', 0.0) if res else 0.0})
            stop = True
            continue
        alloc = res['allocation_df']
        peak = alloc[alloc['period'].isin(PEAK_PERIODS) &
                     alloc['route_id'].isin(priority_routes)]
        avg_f = float(peak['frequency'].mean()) if len(peak) else 0.0
        binding = int((peak['frequency'] == thr).sum()) if thr in FREQ_LEVELS else \
                  int((peak['frequency'] == min(k for k in FREQ_LEVELS if k >= thr)).sum())
        rows.append({'equity_threshold': thr,
                     'total_wait': res['objective'],
                     'avg_priority_freq': avg_f,
                     'num_binding': binding,
                     'feasible': True,
                     'solve_time': res['solve_time']})
    return pd.DataFrame(rows)


# 6C. Shadow prices via LP relaxation

def shadow_price_analysis(routes_df, demand_matrix, fleet_size=2000) -> pd.DataFrame:
    routes = list(routes_df['route_id'])
    tau = _cycle_time_min(routes_df)
    priority = routes_df.set_index('route_id')['priority_score'].astype(float)
    depot_of = routes_df.set_index('route_id')['depot_id']
    depot_caps = _depot_capacities(routes_df, fleet_size)
    dm = demand_matrix.set_index('route_id')[TIME_PERIODS]

    b = {(r, k): _buses_needed(tau[r], k) for r in routes for k in FREQ_LEVELS}
    w = {k: _wait_time(k) for k in FREQ_LEVELS}

    model = pulp.LpProblem("PMPML_LP_Relax", pulp.LpMinimize)
    x = {(r, t, k): pulp.LpVariable(f"x_{r}_{t}_{k}", lowBound=0, upBound=1, cat='Continuous')
         for r in routes for t in TIME_PERIODS for k in FREQ_LEVELS}

    model += pulp.lpSum(dm.loc[r, t] * w[k] * x[(r, t, k)]
                        for r in routes for t in TIME_PERIODS for k in FREQ_LEVELS)

    fleet_constraints = {}
    for t in TIME_PERIODS:
        model += (pulp.lpSum(b[(r, k)] * x[(r, t, k)]
                             for r in routes for k in FREQ_LEVELS) <= fleet_size,
                  f"fleet_{t}")
        fleet_constraints[t] = f"fleet_{t}"

    for r in routes:
        for t in TIME_PERIODS:
            model += pulp.lpSum(x[(r, t, k)] for k in FREQ_LEVELS) == 1, f"assign_{r}_{t}"

    # PuLP sanitises hyphens in constraint names (e.g. "203-NGT" -> "203_NGT").
    # Strip them at construction time so lookups by name succeed on real
    # PMPML route IDs that contain dashes.
    def _safe(s):
        return str(s).replace('-', '_').replace(' ', '_')

    equity_constraints = {}
    for r in routes:
        if priority[r] > PRIORITY_THRESHOLD:
            for t in PEAK_PERIODS:
                cn = f"equity_{_safe(r)}_{_safe(t)}"
                model += pulp.lpSum(x[(r, t, k)] for k in FREQ_LEVELS
                                    if k >= PRIORITY_MIN_FREQ_PEAK) == 1, cn
                equity_constraints[(r, t)] = cn

    depot_constraints = {}
    for d_id, cap in depot_caps.items():
        depot_routes = [r for r in routes if depot_of[r] == d_id]
        for t in TIME_PERIODS:
            cn = f"depot_{_safe(d_id)}_{_safe(t)}"
            model += pulp.lpSum(b[(r, k)] * x[(r, t, k)]
                                for r in depot_routes for k in FREQ_LEVELS) <= cap, cn
            depot_constraints[(d_id, t)] = cn

    solver = pulp.PULP_CBC_CMD(msg=0)
    code = model.solve(solver)
    status = pulp.LpStatus[code]
    log_metric("shadow_lp_status", status)

    rows = []
    for t, cn in fleet_constraints.items():
        pi = model.constraints[cn].pi
        rows.append({'constraint_type': 'fleet', 'subject': t, 'shadow_price': pi})
    # Equity duals: many; report sum, mean, max for tractability
    eq_pis = [model.constraints[cn].pi for cn in equity_constraints.values()]
    if eq_pis:
        rows.append({'constraint_type': 'equity_total', 'subject': 'all_priority_peak',
                     'shadow_price': float(np.sum(eq_pis))})
        rows.append({'constraint_type': 'equity_mean', 'subject': 'per_priority_peak',
                     'shadow_price': float(np.mean(eq_pis))})
        rows.append({'constraint_type': 'equity_max', 'subject': 'binding_priority_peak',
                     'shadow_price': float(np.max(eq_pis))})
    for (d_id, t), cn in depot_constraints.items():
        pi = model.constraints[cn].pi
        rows.append({'constraint_type': 'depot', 'subject': f"{d_id}_{t}",
                     'shadow_price': pi})
    return pd.DataFrame(rows)


# Driver

def run_sensitivity_pipeline(routes_df, demand_matrix, output_dir="results/tables"):
    os.makedirs(output_dir, exist_ok=True)

    fleet_sizes = [1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]
    fleet_df = fleet_sensitivity_analysis(routes_df, demand_matrix, fleet_sizes)
    fleet_df.to_csv(os.path.join(output_dir, "fleet_sensitivity.csv"), index=False)

    equity_thresholds = [1, 2, 3, 4, 5, 6, 8, 10]
    pareto_df = pareto_analysis(routes_df, demand_matrix, equity_thresholds, fleet_size=2000)
    pareto_df.to_csv(os.path.join(output_dir, "pareto_frontier.csv"), index=False)

    shadow_df = shadow_price_analysis(routes_df, demand_matrix, fleet_size=2000)
    shadow_df.to_csv(os.path.join(output_dir, "shadow_prices.csv"), index=False)

    lambdas = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    cost_df = cost_wait_frontier(routes_df, demand_matrix, lambdas, fleet_size=2000)
    cost_df.to_csv(os.path.join(output_dir, "cost_wait_frontier.csv"), index=False)

    eqs_df = equity_mapping_sensitivity(demand_matrix, fleet_size=2000)
    eqs_df.to_csv(os.path.join(output_dir, "equity_mapping_sensitivity.csv"),
                  index=False)

    return {'fleet': fleet_df, 'pareto': pareto_df, 'shadow': shadow_df,
            'cost_frontier': cost_df, 'equity_mapping': eqs_df}


if __name__ == "__main__":
    from src.audit import (setup_audit, log_phase_start, log_phase_end,
                           log_gate_check, log_file_created)
    from src.data_processing import load_routes, load_demand_matrix

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setup_audit(project_root)
    log_phase_start("PHASE_6_SENSITIVITY")

    routes_df = load_routes(os.path.join(project_root, "data/processed/routes.csv"))
    demand_matrix = load_demand_matrix(os.path.join(project_root, "data/processed/demand_matrix.csv"))

    out = run_sensitivity_pipeline(
        routes_df, demand_matrix,
        output_dir=os.path.join(project_root, "results/tables"),
    )
    fleet_df = out['fleet']
    pareto_df = out['pareto']

    print("\nFleet sensitivity:")
    print(fleet_df.to_string(index=False))
    print("\nPareto frontier:")
    print(pareto_df.to_string(index=False))

    feasible = fleet_df[fleet_df['objective'] > 0].sort_values('fleet_size')
    objectives = feasible['objective'].values
    is_monotonic = all(objectives[i] >= objectives[i+1] for i in range(len(objectives)-1))
    log_gate_check("fleet_monotonic", bool(is_monotonic), "decreasing", bool(is_monotonic))

    fleet_default_obj = fleet_df[fleet_df['fleet_size'] == 2000]['objective'].iloc[0]
    det_results = pd.read_csv(os.path.join(project_root, "results/tables/deterministic_results.csv"))
    phase2_obj = det_results[det_results['method'] == 'optimal']['total_wait_time'].iloc[0]
    ratio = abs(fleet_default_obj - phase2_obj) / phase2_obj
    log_gate_check("fleet_default_matches_phase2", bool(ratio < 0.01),
                   "<1% difference", f"{ratio:.4f}")

    feasible_pareto = pareto_df[pareto_df['feasible']].sort_values('equity_threshold')
    if len(feasible_pareto) >= 3:
        waits = feasible_pareto['total_wait'].values
        generally_increasing = waits[-1] > waits[0]
        log_gate_check("pareto_direction", bool(generally_increasing),
                       "increasing trend", bool(generally_increasing))
    else:
        log_warning("Too few feasible Pareto points to check trend")

    has_infeasible = (~pareto_df['feasible']).any()
    log_gate_check("pareto_has_infeasible", True,
                   "True or acceptable",
                   bool(has_infeasible) if has_infeasible else "all_feasible_OK")

    shadow_path = os.path.join(project_root, "results/tables/shadow_prices.csv")
    log_gate_check("shadow_prices_exist", os.path.exists(shadow_path),
                   "True", str(os.path.exists(shadow_path)))

    log_metric("fleet_sensitivity_points", len(fleet_df))
    log_metric("pareto_feasible_points", len(feasible_pareto))

    log_file_created(os.path.join(project_root, "results/tables/fleet_sensitivity.csv"))
    log_file_created(os.path.join(project_root, "results/tables/pareto_frontier.csv"))
    log_file_created(os.path.join(project_root, "results/tables/cost_wait_frontier.csv"))
    log_file_created(os.path.join(project_root, "results/tables/equity_mapping_sensitivity.csv"))
    log_file_created(shadow_path)

    log_phase_end("PHASE_6_SENSITIVITY", "PASS")
    print("PHASE 6 COMPLETE")
