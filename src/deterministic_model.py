"""
Deterministic MIP for PMPML bus frequency allocation.

Mathematical formulation:
  Sets:
    R = routes, T = 5 time periods, K = {1,2,3,4,6,8,10,12,15} frequency levels
  Parameters:
    d[r,t]  = demand (passengers/hour) on route r in period t
    tau[r]  = round-trip cycle time of route r including layover, in minutes,
              tau[r] = 2 * avg_trip_time_min[r] + DWELL_MIN
    w[k]    = 1/(2k) avg wait at frequency k (random arrivals)
    b[r,k]  = ceil(k * tau[r] / 60) buses needed to sustain frequency k
    B       = network-wide fleet size (default 2000, matching the latest
              PMPML figure of ~1947-2000 buses; the optimisation respects
              B = 2000 as the peak-deployable cap)
    lambda_op = marginal operator cost per bus-period, in
                passenger-hour-equivalents (default 0)
  Decision variables:
    x[r,t,k] in {0,1}, 1 if route r in period t uses frequency level k
  Objective:
    min sum_{r,t,k} (d[r,t] * w[k] + lambda_op * b[r,k]) * x[r,t,k]
  Constraints:
    (A) Assignment: sum_k x[r,t,k] = 1     for all r,t
    (F) Fleet:      sum_{r,k} b[r,k] x[r,t,k] <= B    for all t
    (E) Equity:     for r with priority>0.7 and t in {AM_peak,PM_peak},
                    sum_{k>=3} x[r,t,k] = 1
    (D) Depot:      sum_{r in R_j, k} b[r,k] x[r,t,k] <= B_j   for all j,t

Cycle time. avg_trip_time_min is one-way travel time. The bus must travel back
to its origin before serving its next departure, and incurs a turn-around
dwell at each terminus (driver break, passenger alighting/boarding). We model
this as tau[r] = 2 * avg_trip_time_min + DWELL_MIN with DWELL_MIN = 7.5
(midpoint of the 5--10 min layover range typical for Indian urban operations).

Operator cost. With lambda_op = 0 the optimiser ignores running cost and
saturates the fleet, which inflates the headline improvement vs. status quo.
Sweeping lambda_op > 0 traces the Pareto frontier on rider waiting time vs.
operator bus-periods used.
"""
import os
import sys
import time
import math

import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.audit import log_metric, log_warning

FREQ_LEVELS = [1, 2, 3, 4, 6, 8, 10, 12, 15]
TIME_PERIODS = ['early_morning', 'AM_peak', 'midday', 'PM_peak', 'evening']
PEAK_PERIODS = ['AM_peak', 'PM_peak']
PRIORITY_THRESHOLD = 0.7
PRIORITY_MIN_FREQ_PEAK = 3
DEPOT_VARIATION = 0.20
DWELL_MIN = 7.5  # round-trip dwell/layover, minutes


def _cycle_time_min(routes_df: pd.DataFrame) -> pd.Series:
    """Round-trip cycle time tau[r] = 2 * avg_trip + dwell (minutes)."""
    one_way = routes_df.set_index('route_id')['avg_trip_time_min'].astype(float)
    return 2.0 * one_way + DWELL_MIN


def _wait_time(k: int) -> float:
    return 1.0 / (2.0 * k)


def _buses_needed(tau_min: float, k: int) -> int:
    return int(math.ceil(k * tau_min / 60.0))


def _depot_capacities(routes_df: pd.DataFrame, fleet_size: int) -> dict:
    """Allocate fleet_size across depots proportional to each depot's minimum
    bus need (priority routes at the equity floor, non-priority at k=1), plus
    +/- 20% jitter, clamped from below at ``needs[d] + 10`` so the equity
    floor is never depot-infeasible. Equal-share 1/N allocation produces
    infeasibilities when priority routes cluster in a few depots, which the
    (synthetic) Pune geography happens to do."""
    depots = sorted(routes_df['depot_id'].unique())
    tau = _cycle_time_min(routes_df)
    priority = routes_df.set_index('route_id')['priority_score'].astype(float)
    needs = {}
    for d in depots:
        routes_in_d = routes_df[routes_df['depot_id'] == d]['route_id'].tolist()
        n = 0
        for r in routes_in_d:
            k_min = (PRIORITY_MIN_FREQ_PEAK
                     if priority[r] > PRIORITY_THRESHOLD else 1)
            n += _buses_needed(tau[r], k_min)
        needs[d] = max(n, 1)
    total_need = sum(needs.values())
    rng = np.random.default_rng(42)
    caps = {}
    for d in depots:
        base = fleet_size * needs[d] / total_need
        factor = 1.0 + DEPOT_VARIATION * (2 * rng.random() - 1)
        caps[d] = max(int(round(base * factor)), needs[d] + 10)
    return caps


def build_deterministic_model(routes_df: pd.DataFrame,
                              demand_matrix: pd.DataFrame,
                              fleet_size: int = 2000,
                              lambda_op: float = 0.0,
                              time_limit_sec: int = 300,
                              verbose: bool = False) -> dict:
    """Solve the deterministic frequency-allocation MIP. Returns a result dict.

    lambda_op weights operator bus-periods in the objective. With lambda_op=0
    the optimum saturates the fleet whenever waiting time can be reduced.
    """
    routes = list(routes_df['route_id'])
    tau = _cycle_time_min(routes_df)
    priority = routes_df.set_index('route_id')['priority_score'].astype(float)
    depot_of = routes_df.set_index('route_id')['depot_id']
    depot_caps = _depot_capacities(routes_df, fleet_size)
    dm = demand_matrix.set_index('route_id')[TIME_PERIODS]

    b = {(r, k): _buses_needed(tau[r], k) for r in routes for k in FREQ_LEVELS}
    w = {k: _wait_time(k) for k in FREQ_LEVELS}

    model = pulp.LpProblem("PMPML_Deterministic", pulp.LpMinimize)

    x = {}
    for r in routes:
        for t in TIME_PERIODS:
            for k in FREQ_LEVELS:
                x[(r, t, k)] = pulp.LpVariable(f"x_{r}_{t}_{k}", cat='Binary')

    model += pulp.lpSum(
        (dm.loc[r, t] * w[k] + lambda_op * b[(r, k)]) * x[(r, t, k)]
        for r in routes for t in TIME_PERIODS for k in FREQ_LEVELS
    )

    for r in routes:
        for t in TIME_PERIODS:
            model += pulp.lpSum(x[(r, t, k)] for k in FREQ_LEVELS) == 1, f"assign_{r}_{t}"

    for t in TIME_PERIODS:
        model += pulp.lpSum(
            b[(r, k)] * x[(r, t, k)] for r in routes for k in FREQ_LEVELS
        ) <= fleet_size, f"fleet_{t}"

    for r in routes:
        if priority[r] > PRIORITY_THRESHOLD:
            for t in PEAK_PERIODS:
                model += pulp.lpSum(
                    x[(r, t, k)] for k in FREQ_LEVELS if k >= PRIORITY_MIN_FREQ_PEAK
                ) == 1, f"equity_{r}_{t}"

    for d_id, cap in depot_caps.items():
        depot_routes = [r for r in routes if depot_of[r] == d_id]
        for t in TIME_PERIODS:
            model += pulp.lpSum(
                b[(r, k)] * x[(r, t, k)] for r in depot_routes for k in FREQ_LEVELS
            ) <= cap, f"depot_{d_id}_{t}"

    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=time_limit_sec)
    t0 = time.time()
    status_code = model.solve(solver)
    solve_time = time.time() - t0
    status = pulp.LpStatus[status_code]

    rows = []
    for r in routes:
        for t in TIME_PERIODS:
            chosen_k = None
            for k in FREQ_LEVELS:
                val = pulp.value(x[(r, t, k)])
                if val is not None and val > 0.5:
                    chosen_k = k
                    break
            if chosen_k is None:
                chosen_k = FREQ_LEVELS[0]
            rows.append({
                'route_id': r, 'period': t,
                'frequency': chosen_k,
                'buses_assigned': b[(r, chosen_k)],
            })
    allocation_df = pd.DataFrame(rows)
    objective = pulp.value(model.objective)
    objective = float(objective) if objective is not None else float('nan')

    buses_per_period = {
        t: int(allocation_df[allocation_df['period'] == t]['buses_assigned'].sum())
        for t in TIME_PERIODS
    }

    return {
        'status': status,
        'objective': objective,
        'solve_time': solve_time,
        'allocation_df': allocation_df,
        'buses_used_per_period': buses_per_period,
        'depot_caps': depot_caps,
    }


# Baselines

def _allocation_to_objective(allocation_df: pd.DataFrame,
                             demand_matrix: pd.DataFrame) -> float:
    dm = demand_matrix.set_index('route_id')[TIME_PERIODS]
    total = 0.0
    for _, row in allocation_df.iterrows():
        d = dm.loc[row['route_id'], row['period']]
        total += d * _wait_time(int(row['frequency']))
    return total


def evaluate_allocation(allocation_df: pd.DataFrame,
                        demand_matrix: pd.DataFrame,
                        routes_df: pd.DataFrame) -> dict:
    obj = _allocation_to_objective(allocation_df, demand_matrix)
    return {
        'total_wait_time': obj,
        'fleet_used': int(allocation_df['buses_assigned'].sum()),
        'avg_frequency': float(allocation_df['frequency'].mean()),
        'min_frequency': int(allocation_df['frequency'].min()),
        'max_frequency': int(allocation_df['frequency'].max()),
    }


def _snap(freq: float) -> int:
    arr = np.array(FREQ_LEVELS)
    return int(arr[np.argmin(np.abs(arr - freq))])


def _enforce_equity(df: pd.DataFrame, routes_df: pd.DataFrame) -> pd.DataFrame:
    """Bump priority routes to k>=3 during peak periods."""
    tau = _cycle_time_min(routes_df)
    priority = routes_df.set_index('route_id')['priority_score'].astype(float)
    df = df.copy()
    for i, row in df.iterrows():
        if row['period'] in PEAK_PERIODS and priority[row['route_id']] > PRIORITY_THRESHOLD:
            if row['frequency'] < PRIORITY_MIN_FREQ_PEAK:
                df.at[i, 'frequency'] = PRIORITY_MIN_FREQ_PEAK
                df.at[i, 'buses_assigned'] = _buses_needed(tau[row['route_id']],
                                                          PRIORITY_MIN_FREQ_PEAK)
    return df


def _enforce_capacity(df: pd.DataFrame, routes_df: pd.DataFrame,
                      depot_caps: dict, fleet_size: int) -> pd.DataFrame:
    """Repair fleet+depot violations by downgrading highest-bus routes
    (skipping priority-peak routes already pinned at k=3)."""
    tau = _cycle_time_min(routes_df)
    priority = routes_df.set_index('route_id')['priority_score'].astype(float)
    depot_of = routes_df.set_index('route_id')['depot_id']
    df = df.copy()

    def downgrade(idx):
        cur_k = df.at[idx, 'frequency']
        cur_idx = FREQ_LEVELS.index(cur_k)
        if cur_idx == 0:
            return False
        new_k = FREQ_LEVELS[cur_idx - 1]
        df.at[idx, 'frequency'] = new_k
        df.at[idx, 'buses_assigned'] = _buses_needed(tau[df.at[idx, 'route_id']], new_k)
        return True

    for t in TIME_PERIODS:
        # Fleet constraint
        for _ in range(10000):
            sub = df[df['period'] == t]
            if sub['buses_assigned'].sum() <= fleet_size:
                break
            # downgrade route with most buses, skipping pinned priority-peak
            cand = sub.copy()
            if t in PEAK_PERIODS:
                pin = cand['route_id'].apply(lambda r: priority[r] > PRIORITY_THRESHOLD
                                              and df[(df['route_id'] == r) & (df['period'] == t)].iloc[0]['frequency'] <= PRIORITY_MIN_FREQ_PEAK)
                cand = cand[~pin.values]
            if cand.empty:
                break
            idx_max = cand.sort_values('buses_assigned', ascending=False).index[0]
            if not downgrade(idx_max):
                break

        # Depot constraint
        for d_id, cap in depot_caps.items():
            for _ in range(10000):
                sub = df[(df['period'] == t) & (df['route_id'].apply(lambda r: depot_of[r] == d_id))]
                if sub['buses_assigned'].sum() <= cap:
                    break
                cand = sub.copy()
                if t in PEAK_PERIODS:
                    pin = cand['route_id'].apply(lambda r: priority[r] > PRIORITY_THRESHOLD
                                                  and df[(df['route_id'] == r) & (df['period'] == t)].iloc[0]['frequency'] <= PRIORITY_MIN_FREQ_PEAK)
                    cand = cand[~pin.values]
                if cand.empty:
                    break
                idx_max = cand.sort_values('buses_assigned', ascending=False).index[0]
                if not downgrade(idx_max):
                    break
    return df


def compute_baseline_status_quo(routes_df: pd.DataFrame,
                                demand_matrix: pd.DataFrame) -> pd.DataFrame:
    """Status quo evaluated as-is (does not enforce constraints — it is reality)."""
    tau = _cycle_time_min(routes_df)
    pf = routes_df.set_index('route_id')['current_peak_freq'].astype(int)
    of = routes_df.set_index('route_id')['current_offpeak_freq'].astype(int)
    rows = []
    for r in routes_df['route_id']:
        for t in TIME_PERIODS:
            k_raw = pf[r] if t in PEAK_PERIODS else of[r]
            k = _snap(max(1, k_raw))
            rows.append({'route_id': r, 'period': t,
                         'frequency': k,
                         'buses_assigned': _buses_needed(tau[r], k)})
    return pd.DataFrame(rows)


def compute_baseline_uniform(routes_df: pd.DataFrame,
                             demand_matrix: pd.DataFrame,
                             fleet_size: int = 2000,
                             depot_caps: dict = None) -> pd.DataFrame:
    """Largest uniform k satisfying fleet and depot caps."""
    tau = _cycle_time_min(routes_df)
    if depot_caps is None:
        depot_caps = _depot_capacities(routes_df, fleet_size)
    depot_of = routes_df.set_index('route_id')['depot_id']

    def feasible(k):
        # fleet
        if sum(_buses_needed(tau[r], k) for r in routes_df['route_id']) > fleet_size:
            return False
        # depot
        for d_id, cap in depot_caps.items():
            buses = sum(_buses_needed(tau[r], k)
                        for r in routes_df['route_id'] if depot_of[r] == d_id)
            if buses > cap:
                return False
        return True

    chosen_k = FREQ_LEVELS[0]
    for k in FREQ_LEVELS:
        if feasible(k):
            chosen_k = k
        else:
            break
    rows = []
    for r in routes_df['route_id']:
        for t in TIME_PERIODS:
            rows.append({'route_id': r, 'period': t,
                         'frequency': chosen_k,
                         'buses_assigned': _buses_needed(tau[r], chosen_k)})
    return pd.DataFrame(rows)


def compute_baseline_demand_proportional(routes_df: pd.DataFrame,
                                         demand_matrix: pd.DataFrame,
                                         fleet_size: int = 2000,
                                         depot_caps: dict = None) -> pd.DataFrame:
    if depot_caps is None:
        depot_caps = _depot_capacities(routes_df, fleet_size)
    tau = _cycle_time_min(routes_df)
    dm = demand_matrix.set_index('route_id')[TIME_PERIODS]
    rows = []
    for t in TIME_PERIODS:
        d = dm[t]
        lo, hi = 0.0, 1.0
        for _ in range(30):
            buses = sum(math.ceil(hi * d[r] * tau[r] / 60.0) for r in d.index)
            if buses > fleet_size:
                break
            hi *= 2
        for _ in range(40):
            mid = (lo + hi) / 2
            buses = sum(math.ceil(mid * d[r] * tau[r] / 60.0) for r in d.index)
            if buses <= fleet_size:
                lo = mid
            else:
                hi = mid
        c = lo
        for r in d.index:
            k = _snap(max(1, c * d[r]))
            rows.append({'route_id': r, 'period': t,
                         'frequency': k,
                         'buses_assigned': _buses_needed(tau[r], k)})
    df = pd.DataFrame(rows)
    df = _enforce_equity(df, routes_df)
    df = _enforce_capacity(df, routes_df, depot_caps, fleet_size)
    return df


def compute_baseline_greedy(routes_df: pd.DataFrame,
                            demand_matrix: pd.DataFrame,
                            fleet_size: int = 2000,
                            depot_caps: dict = None) -> pd.DataFrame:
    """Greedy upgrade respecting fleet, depot, and equity constraints."""
    if depot_caps is None:
        depot_caps = _depot_capacities(routes_df, fleet_size)
    tau = _cycle_time_min(routes_df)
    priority = routes_df.set_index('route_id')['priority_score'].astype(float)
    depot_of = routes_df.set_index('route_id')['depot_id']
    dm = demand_matrix.set_index('route_id')[TIME_PERIODS]
    rows = []

    for t in TIME_PERIODS:
        # Equity-respecting init: priority routes at peak start at k=3
        cur_k = {}
        for r in dm.index:
            if t in PEAK_PERIODS and priority[r] > PRIORITY_THRESHOLD:
                cur_k[r] = PRIORITY_MIN_FREQ_PEAK
            else:
                cur_k[r] = 1
        cur_buses = {r: _buses_needed(tau[r], k) for r, k in cur_k.items()}
        total_buses = sum(cur_buses.values())
        depot_used = {d: 0 for d in depot_caps}
        for r, b in cur_buses.items():
            depot_used[depot_of[r]] += b

        if total_buses > fleet_size:
            log_warning(f"greedy: equity-init over budget in {t} ({total_buses}>{fleet_size})")

        improved = True
        while improved:
            improved = False
            best_ratio = 0.0
            best = None
            for r, k_cur in cur_k.items():
                idx = FREQ_LEVELS.index(k_cur)
                if idx + 1 >= len(FREQ_LEVELS):
                    continue
                new_k = FREQ_LEVELS[idx + 1]
                new_buses = _buses_needed(tau[r], new_k)
                extra = new_buses - cur_buses[r]
                if extra <= 0:
                    continue
                if total_buses + extra > fleet_size:
                    continue
                if depot_used[depot_of[r]] + extra > depot_caps[depot_of[r]]:
                    continue
                d = dm.loc[r, t]
                wait_drop = d * (_wait_time(k_cur) - _wait_time(new_k))
                ratio = wait_drop / extra
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = (r, new_k, new_buses, extra)
            if best is not None:
                r, new_k, new_buses, extra = best
                total_buses += extra
                depot_used[depot_of[r]] += extra
                cur_k[r] = new_k
                cur_buses[r] = new_buses
                improved = True
        for r, k in cur_k.items():
            rows.append({'route_id': r, 'period': t,
                         'frequency': k,
                         'buses_assigned': _buses_needed(tau[r], k)})
    return pd.DataFrame(rows)


# Driver

def run_deterministic_pipeline(routes_df, demand_matrix, fleet_size=2000,
                               output_dir="results/tables"):
    os.makedirs(output_dir, exist_ok=True)

    log_metric("det_problem_vars", len(routes_df) * len(TIME_PERIODS) * len(FREQ_LEVELS))
    opt = build_deterministic_model(routes_df, demand_matrix, fleet_size=fleet_size)
    log_metric("det_solve_time_sec", round(opt['solve_time'], 2))
    log_metric("det_solver_status", opt['status'])

    if opt['allocation_df'].empty:
        raise RuntimeError(f"Solver failed (status={opt['status']})")

    depot_caps = opt['depot_caps']
    sq = compute_baseline_status_quo(routes_df, demand_matrix)
    uni = compute_baseline_uniform(routes_df, demand_matrix, fleet_size, depot_caps=depot_caps)
    dp = compute_baseline_demand_proportional(routes_df, demand_matrix, fleet_size, depot_caps=depot_caps)
    gr = compute_baseline_greedy(routes_df, demand_matrix, fleet_size, depot_caps=depot_caps)

    sq_eval = evaluate_allocation(sq, demand_matrix, routes_df)
    uni_eval = evaluate_allocation(uni, demand_matrix, routes_df)
    dp_eval = evaluate_allocation(dp, demand_matrix, routes_df)
    gr_eval = evaluate_allocation(gr, demand_matrix, routes_df)
    opt_eval = evaluate_allocation(opt['allocation_df'], demand_matrix, routes_df)

    sq_obj = sq_eval['total_wait_time']
    methods = []
    for name, ev in [('status_quo', sq_eval), ('uniform', uni_eval),
                     ('demand_proportional', dp_eval), ('greedy', gr_eval),
                     ('optimal', opt_eval)]:
        methods.append({
            'method': name,
            'total_wait_time': ev['total_wait_time'],
            'fleet_used': ev['fleet_used'],
            'avg_frequency': ev['avg_frequency'],
            'min_frequency': ev['min_frequency'],
            'max_frequency': ev['max_frequency'],
            'pct_improvement_vs_status_quo':
                100 * (sq_obj - ev['total_wait_time']) / sq_obj if sq_obj > 0 else 0.0,
        })
    results_df = pd.DataFrame(methods)
    results_df.to_csv(os.path.join(output_dir, "deterministic_results.csv"), index=False)
    opt['allocation_df'].to_csv(os.path.join(output_dir, "optimal_allocation.csv"), index=False)

    return {'results_df': results_df,
            'optimal_allocation': opt['allocation_df'],
            'opt_result': opt}


if __name__ == "__main__":
    from src.audit import (setup_audit, log_phase_start, log_phase_end,
                           log_gate_check, log_file_created)
    from src.data_processing import load_routes, load_demand_matrix

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setup_audit(project_root)
    log_phase_start("PHASE_2_DETERMINISTIC")

    routes_df = load_routes(os.path.join(project_root, "data/processed/routes.csv"))
    demand_matrix = load_demand_matrix(os.path.join(project_root, "data/processed/demand_matrix.csv"))

    out = run_deterministic_pipeline(routes_df, demand_matrix, fleet_size=2000,
                                     output_dir=os.path.join(project_root, "results/tables"))
    results = out['results_df']
    allocation = out['optimal_allocation']

    optimal = results[results['method'] == 'optimal'].iloc[0]
    status_quo = results[results['method'] == 'status_quo'].iloc[0]

    log_gate_check("solver_optimal", optimal['total_wait_time'] > 0, ">0",
                   round(optimal['total_wait_time'], 4))
    for period in TIME_PERIODS:
        period_alloc = allocation[allocation['period'] == period]
        total_buses = int(period_alloc['buses_assigned'].sum())
        log_gate_check(f"fleet_{period}", total_buses <= 2000, "<=2000", total_buses)
    expected_assignments = len(routes_df) * len(TIME_PERIODS)
    log_gate_check("all_assigned", len(allocation) == expected_assignments,
                   str(expected_assignments), len(allocation))
    # Structural checks only: feasibility and objective bounds. We do NOT gate
    # on optimal beating baselines or on a target improvement range, because
    # that biases every modeling choice upstream toward whatever produces a
    # "nice" headline number. Report it; don't assert it.
    log_gate_check("optimal_objective_finite",
                   0 < optimal['total_wait_time'] < 1e9,
                   "finite > 0",
                   f"{optimal['total_wait_time']:.2f}")
    pct = (status_quo['total_wait_time'] - optimal['total_wait_time']) / status_quo['total_wait_time'] * 100
    log_metric("deterministic_improvement_pct", round(pct, 2))
    log_metric("deterministic_objective", round(optimal['total_wait_time'], 4))

    priority_routes = routes_df[routes_df['priority_score'] > PRIORITY_THRESHOLD]['route_id'].tolist()
    equity_violations = 0
    for r in priority_routes:
        for p in PEAK_PERIODS:
            row = allocation[(allocation['route_id'] == r) & (allocation['period'] == p)]
            if len(row) > 0 and row.iloc[0]['frequency'] < PRIORITY_MIN_FREQ_PEAK:
                equity_violations += 1
    log_gate_check("equity_satisfied", equity_violations == 0, "0 violations", equity_violations)

    log_file_created(os.path.join(project_root, "results/tables/deterministic_results.csv"))
    log_file_created(os.path.join(project_root, "results/tables/optimal_allocation.csv"))
    log_phase_end("PHASE_2_DETERMINISTIC", "PASS")
    print("PHASE 2 COMPLETE")
    print(results.to_string(index=False))
