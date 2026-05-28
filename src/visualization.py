"""
All paper-quality figures for pune-bus-optimization.

Saves PNG to results/figures/ and PDF to paper/figures/.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})
PALETTE = sns.color_palette("colorblind")
TIME_PERIODS = ['early_morning', 'AM_peak', 'midday', 'PM_peak', 'evening']


def save_fig(fig, name, fig_dir="results/figures", paper_dir="paper/figures"):
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(paper_dir, f"{name}.pdf"), bbox_inches='tight')
    plt.close(fig)


# 1. Allocation heatmap

def fig_allocation_heatmap(routes_df, allocation_df, demand_matrix):
    # Top 30 routes by total demand
    dm = demand_matrix.set_index('route_id')[TIME_PERIODS]
    total_demand = dm.sum(axis=1).sort_values(ascending=False)
    top30 = total_demand.head(30).index.tolist()

    pf = routes_df.set_index('route_id')['current_peak_freq']
    of = routes_df.set_index('route_id')['current_offpeak_freq']
    PEAK = ['AM_peak', 'PM_peak']
    cur = pd.DataFrame(index=top30, columns=TIME_PERIODS, dtype=int)
    for r in top30:
        for t in TIME_PERIODS:
            cur.at[r, t] = int(pf[r]) if t in PEAK else int(of[r])

    opt_pivot = (allocation_df.pivot(index='route_id', columns='period',
                                     values='frequency')
                 .reindex(index=top30, columns=TIME_PERIODS))

    name_map = routes_df.set_index('route_id')['route_name'].to_dict()
    short_names = [name_map.get(r, r)[:22] for r in top30]
    cur.index = short_names
    opt_pivot.index = short_names

    vmax = max(int(cur.max().max()), int(opt_pivot.max().max()))
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    sns.heatmap(cur.astype(int), cmap='YlOrRd', annot=True, fmt='d',
                vmin=1, vmax=vmax, ax=axes[0], cbar=False)
    axes[0].set_title('Current PMPML allocation')
    axes[0].set_xlabel('Period'); axes[0].set_ylabel('Route')
    sns.heatmap(opt_pivot.astype(int), cmap='YlOrRd', annot=True, fmt='d',
                vmin=1, vmax=vmax, ax=axes[1],
                cbar_kws={'label': 'Frequency (buses/hour)'})
    axes[1].set_title('Optimal allocation')
    axes[1].set_xlabel('Period'); axes[1].set_ylabel('')
    fig.tight_layout()
    save_fig(fig, 'fig_allocation_heatmap')


# 2. Forecast scatter

def fig_demand_forecast_scatter(test_predictions, routes_df):
    df = test_predictions.merge(routes_df[['route_id', 'route_category']],
                                on='route_id', how='left')
    fig, ax = plt.subplots(figsize=(8, 6))
    cats = ['trunk', 'feeder', 'suburban']
    colors = [PALETTE[i] for i in range(3)]
    for c, col in zip(cats, colors):
        sub = df[df['route_category'] == c]
        ax.scatter(sub['actual'], sub['predicted'], s=4, alpha=0.3,
                   color=col, label=c)
    lim = max(df['actual'].max(), df['predicted'].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='y = x')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('Actual ridership (passengers/hour)')
    ax.set_ylabel('Predicted ridership (passengers/hour)')
    ax.set_title('XGBoost test-set predictions')
    rmse = float(np.sqrt(np.mean((df['actual'] - df['predicted']) ** 2)))
    ss_res = np.sum((df['actual'] - df['predicted']) ** 2)
    ss_tot = np.sum((df['actual'] - df['actual'].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    ax.text(0.04, 0.96, f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}",
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.legend(loc='lower right')
    save_fig(fig, 'fig_demand_forecast_scatter')


# 3 & 4. SHAP figures

def _compute_shap(xgb_model, X_sample, feat_cols):
    import shap
    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(X_sample)
    return sv


def fig_shap_importance(xgb_model, X_sample, feat_cols):
    sv = _compute_shap(xgb_model, X_sample, feat_cols)
    importance = np.abs(sv).mean(axis=0)
    order = np.argsort(importance)[::-1][:12]
    names = [feat_cols[i] for i in order]
    vals = importance[order]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names))[::-1], vals[::-1], color=PALETTE[0])
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel('mean |SHAP value|')
    ax.set_title('Top 12 features by SHAP importance')
    save_fig(fig, 'fig_shap_importance')


def fig_shap_beeswarm(xgb_model, X_sample, feat_cols):
    sv = _compute_shap(xgb_model, X_sample, feat_cols)
    importance = np.abs(sv).mean(axis=0)
    order = np.argsort(importance)[::-1][:12]

    fig, ax = plt.subplots(figsize=(10, 7))
    rng = np.random.default_rng(42)
    for plot_i, j in enumerate(order):
        y = len(order) - 1 - plot_i
        x = sv[:, j]
        feat = X_sample[:, j]
        # normalize feature for color
        rng_min, rng_max = np.percentile(feat, 5), np.percentile(feat, 95)
        denom = max(rng_max - rng_min, 1e-9)
        col = np.clip((feat - rng_min) / denom, 0, 1)
        jitter = rng.normal(0, 0.12, size=len(x))
        sc = ax.scatter(x, np.full_like(x, y) + jitter, c=col, cmap='coolwarm',
                        s=4, alpha=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_cols[order[len(order) - 1 - i]] for i in range(len(order))])
    ax.axvline(0, color='gray', lw=0.8)
    ax.set_xlabel('SHAP value (impact on prediction)')
    ax.set_title('SHAP beeswarm — top 12 features')
    cb = plt.colorbar(sc, ax=ax, ticks=[0, 1])
    cb.ax.set_yticklabels(['low', 'high'])
    cb.set_label('feature value')
    save_fig(fig, 'fig_shap_beeswarm')


# 5. Forecast time series

def fig_forecast_timeseries(test_predictions, routes_df):
    df = test_predictions.merge(routes_df[['route_id', 'route_category', 'route_name']],
                                on='route_id', how='left')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['route_id', 'date', 'hour'])
    df['ts'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')

    chosen = {}
    for cat in ['trunk', 'feeder', 'suburban']:
        cand = df[df['route_category'] == cat]
        if cand.empty:
            continue
        best_route = (cand.groupby('route_id')['actual'].sum()
                      .sort_values(ascending=False).index[0])
        chosen[cat] = best_route

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    for ax, (cat, rid) in zip(axes, chosen.items()):
        sub = df[df['route_id'] == rid].sort_values('ts').head(14 * 24)
        rname = sub['route_name'].iloc[0] if len(sub) else rid
        ax.plot(sub['ts'], sub['actual'], color='black', lw=1, label='actual')
        ax.plot(sub['ts'], sub['predicted'], color=PALETTE[0], lw=1, alpha=0.9, label='predicted')
        ax.fill_between(sub['ts'], sub['q10'], sub['q90'], color=PALETTE[0], alpha=0.2,
                        label='80% PI')
        ax.set_title(f"{cat.title()}: {rname} ({rid})")
        ax.set_ylabel('ridership')
        ax.legend(loc='upper right', ncol=3)
    axes[-1].set_xlabel('time')
    fig.tight_layout()
    save_fig(fig, 'fig_forecast_timeseries')


# 6. Scenario comparison

def fig_scenario_comparison(stoch_results):
    df = stoch_results.copy()
    methods_order = ['status_quo', 'deterministic', 'stochastic', 'robust']
    scenarios_order = sorted(df['scenario'].unique())
    pivot = df.pivot(index='scenario', columns='method', values='objective_value')
    pivot = pivot.reindex(index=scenarios_order, columns=methods_order)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(scenarios_order))
    width = 0.2
    for i, m in enumerate(methods_order):
        ax.bar(x + (i - 1.5) * width, pivot[m].values, width=width,
               label=m, color=PALETTE[i])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_order, rotation=20, ha='right')
    ax.set_ylabel('Total waiting time (passenger-hours)')
    ax.set_title('Method performance across scenarios')
    sq_mean = pivot['status_quo'].mean()
    ax.axhline(sq_mean, ls='--', color='gray', alpha=0.7,
               label=f'status_quo mean ({sq_mean:.0f})')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    save_fig(fig, 'fig_scenario_comparison')


# 7. VSS / EVPI

def fig_vss_evpi(vss_evpi):
    df = vss_evpi.set_index('metric')
    vals = [df.loc['VSS', 'value'], df.loc['EVPI', 'value']]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['VSS', 'EVPI'], vals, color=[PALETTE[0], PALETTE[1]])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha='center',
                va='bottom')
    ax.set_ylabel('Value (passenger-hours)')
    ax.set_title('Value of Information')
    ax.set_ylim(0, max(vals) * 1.25)
    save_fig(fig, 'fig_vss_evpi')


# 8. Decision-focused

def fig_decision_focused(df_results, multiseed_df=None):
    """Two panels:
      (left)  test RMSE per method (mean over seeds, with SE error bars when
              multiseed_df is provided);
      (right) regret distribution per method as boxplots over seeds (or bars
              if only single-seed data is provided).
    """
    pred_methods = ['xgb_mse', 'xgb_weighted', 'nn_mse', 'nn_spo']
    colors = [PALETTE[i] for i in range(len(pred_methods))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if multiseed_df is not None and not multiseed_df.empty:
        # RMSE bars with SE
        rmse_means = [multiseed_df[multiseed_df['method'] == m]['prediction_rmse'].mean()
                      for m in pred_methods]
        rmse_ses = [multiseed_df[multiseed_df['method'] == m]['prediction_rmse'].std(ddof=1)
                    / max(1, multiseed_df[multiseed_df['method'] == m].shape[0]) ** 0.5
                    for m in pred_methods]
        axes[0].bar(pred_methods, rmse_means, yerr=rmse_ses, color=colors, capsize=4)
        axes[0].set_ylabel('Test RMSE (mean over seeds $\\pm$ SE)')
        axes[0].set_title('Prediction error (multi-seed)')

        # Regret boxplot across seeds
        regrets_by_method = [multiseed_df[multiseed_df['method'] == m]['regret'].values
                             for m in pred_methods]
        bp = axes[1].boxplot(regrets_by_method, labels=pred_methods, patch_artist=True,
                             showmeans=True, meanline=True)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        axes[1].axhline(0, ls='--', color='gray', label='oracle')
        axes[1].set_ylabel('Regret across seeds (passenger-hours)')
        axes[1].set_title('Decision quality (boxplot over seeds)')
        axes[1].legend(loc='upper right')
    else:
        df = df_results.copy()
        rmse_vals = [df[df['method'] == m]['prediction_rmse'].iloc[0] for m in pred_methods]
        axes[0].bar(pred_methods, rmse_vals, color=colors)
        axes[0].set_ylabel('Test RMSE')
        axes[0].set_title('Prediction error')
        regrets = [df[df['method'] == m]['regret'].iloc[0] for m in pred_methods]
        axes[1].bar(pred_methods, regrets, color=colors)
        axes[1].axhline(0, ls='--', color='gray', label='oracle')
        axes[1].set_ylabel('Regret (vs oracle)')
        axes[1].set_title('Decision quality')
        axes[1].legend()

    axes[0].tick_params(axis='x', rotation=20)
    axes[1].tick_params(axis='x', rotation=20)
    fig.tight_layout()
    save_fig(fig, 'fig_decision_focused')


# 9. Fleet sensitivity

def fig_fleet_sensitivity(fleet_df):
    df = fleet_df[fleet_df['objective'] > 0].sort_values('fleet_size')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df['fleet_size'], df['objective'], 'o-', color=PALETTE[0],
             lw=2, label='Total wait time')
    ax1.set_xlabel('Fleet size (# buses)')
    ax1.set_ylabel('Total waiting time (passenger-hours)', color=PALETTE[0])
    ax1.tick_params(axis='y', labelcolor=PALETTE[0])
    ax2 = ax1.twinx()
    ax2.plot(df['fleet_size'], df['routes_above_3'], 's-', color=PALETTE[1],
             lw=2, label='Routes with freq >= 3')
    ax2.set_ylabel('Routes with frequency >= 3', color=PALETTE[1])
    ax2.tick_params(axis='y', labelcolor=PALETTE[1])
    ax2.spines['top'].set_visible(False)
    ax1.axvline(2000, ls='--', color='gray', alpha=0.7)
    ax1.text(2000, ax1.get_ylim()[1] * 0.97, 'fleet cap (B=2000)',
             ha='center', va='top', color='gray')
    ax1.set_title('Fleet-size sensitivity')
    save_fig(fig, 'fig_fleet_sensitivity')


# 10. Pareto frontier

def fig_pareto_frontier(pareto_df):
    feasible = pareto_df[pareto_df['feasible']].sort_values('equity_threshold')
    infeasible = pareto_df[~pareto_df['feasible']]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(feasible['avg_priority_freq'], feasible['total_wait'], 'o-',
            color=PALETTE[0], lw=2, ms=8, label='feasible')
    if len(infeasible):
        # plot infeasible with a placeholder x at their threshold (no avg_priority_freq)
        # use threshold value as x for visualization; offset
        x_marker = feasible['avg_priority_freq'].max() + 0.5
        ax.scatter([x_marker] * len(infeasible),
                   [feasible['total_wait'].max() * 1.05] * len(infeasible),
                   marker='x', color='red', s=80,
                   label=f'{len(infeasible)} infeasible thresholds')

    annotations = {1: 'no equity', 3: 'current policy'}
    if len(feasible):
        max_thr = int(feasible['equity_threshold'].max())
        annotations[max_thr] = 'max feasible'
    for _, row in feasible.iterrows():
        thr = int(row['equity_threshold'])
        if thr in annotations:
            ax.annotate(f"{annotations[thr]} (thr={thr})",
                        xy=(row['avg_priority_freq'], row['total_wait']),
                        xytext=(8, 8), textcoords='offset points', fontsize=9,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax.set_xlabel('Avg priority-route frequency at peak (equity)')
    ax.set_ylabel('Total waiting time (efficiency)')
    ax.set_title('Equity-efficiency Pareto frontier')
    ax.legend()
    save_fig(fig, 'fig_pareto_frontier')


# 11c. Cost-wait frontier

def fig_cost_wait_frontier(cost_df):
    """Cost-wait Pareto frontier with collision-avoiding annotations.

    Multiple low-lambda points saturate the fleet at the same (x, y); we
    collapse coincident points to a single annotation listing all lambdas
    that land there. Saturation at the top-right and floor at the bottom-
    left both get this treatment.
    """
    df = (cost_df.dropna(subset=['total_wait_time'])
                  .sort_values('bus_periods_used')
                  .reset_index(drop=True))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['bus_periods_used'], df['total_wait_time'], 'o-',
            color=PALETTE[0], lw=2, ms=7)

    # Group lambdas that share the same optimum (within 1 bus-period
    # and 0.5 pass-hr) into a single annotation.
    groups = []
    for _, row in df.iterrows():
        placed = False
        for g in groups:
            if (abs(row['bus_periods_used'] - g['x']) < 1
                    and abs(row['total_wait_time'] - g['y']) < 0.5):
                g['lambdas'].append(row['lambda_op'])
                placed = True
                break
        if not placed:
            groups.append({'x': row['bus_periods_used'],
                           'y': row['total_wait_time'],
                           'lambdas': [row['lambda_op']]})

    # Annotate each group once; alternate offset direction so labels do
    # not run off the chart at extreme points.
    for g in groups:
        lambdas = sorted(g['lambdas'])
        if len(lambdas) == 1:
            label = f"$\\lambda$={lambdas[0]:g}"
        else:
            label = "$\\lambda \\in \\{" + ", ".join(f"{lam:g}" for lam in lambdas) + "\\}$"
        # Push labels away from chart edges
        x_frac = (g['x'] - df['bus_periods_used'].min()) / (
            df['bus_periods_used'].max() - df['bus_periods_used'].min() + 1e-9)
        if x_frac > 0.7:
            xytext = (-10, 12)
            ha = 'right'
        elif x_frac < 0.15:
            xytext = (10, -14)
            ha = 'left'
        else:
            xytext = (6, 8)
            ha = 'left'
        ax.annotate(label,
                    xy=(g['x'], g['y']),
                    xytext=xytext, textcoords='offset points',
                    ha=ha, fontsize=8, color='gray')

    ax.set_xlabel('Operator bus-periods used')
    ax.set_ylabel('Total rider waiting time (passenger-hours)')
    ax.set_title('Rider wait vs operator cost: Pareto frontier')
    ax.grid(alpha=0.3)
    save_fig(fig, 'fig_cost_wait_frontier')


# 11b. PI reliability diagram

def fig_pi_reliability(reliability_df, test_predictions):
    """Two-panel: (left) reliability of trained quantile heads on test set;
    (right) 80% PI coverage before vs after split-conformal calibration."""
    rel = reliability_df[reliability_df['split'] == 'test'].sort_values('nominal')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='ideal')
    axes[0].plot(rel['nominal'], rel['empirical'], 'o-',
                 color=PALETTE[0], lw=2, ms=7, label='XGBoost quantile heads')
    axes[0].set_xlabel('Nominal level $\\alpha$')
    axes[0].set_ylabel('Empirical $P(y \\leq \\hat q_\\alpha)$')
    axes[0].set_title('Reliability diagram (test fold)')
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)

    p = test_predictions
    raw = float(((p['actual'] >= p['q10']) & (p['actual'] <= p['q90'])).mean())
    if 'q10_cal' in p.columns and 'q90_cal' in p.columns:
        cal = float(((p['actual'] >= p['q10_cal']) & (p['actual'] <= p['q90_cal'])).mean())
    else:
        cal = raw
    bars = axes[1].bar(['raw\n80% PI', 'split-conformal\n80% PI'],
                       [raw, cal], color=[PALETTE[3], PALETTE[2]])
    axes[1].axhline(0.80, ls='--', color='gray', label='nominal 80%')
    for b, v in zip(bars, [raw, cal]):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.1%}",
                     ha='center', va='bottom')
    axes[1].set_ylabel('Empirical coverage')
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title('80% PI: raw vs conformally calibrated')
    axes[1].legend(loc='lower right')

    fig.tight_layout()
    save_fig(fig, 'fig_pi_reliability')


# 11. Equity boxplot

def fig_equity_boxplot(routes_df, allocation_df):
    PEAK = ['AM_peak', 'PM_peak']
    pri_ids = set(routes_df[routes_df['priority_score'] > 0.7]['route_id'])
    pf = routes_df.set_index('route_id')['current_peak_freq'].astype(int)
    rows = []
    for _, row in allocation_df.iterrows():
        if row['period'] not in PEAK:
            continue
        kind = 'priority' if row['route_id'] in pri_ids else 'non-priority'
        rows.append({'group': f'{kind} (after)', 'frequency': row['frequency']})
        rows.append({'group': f'{kind} (before)', 'frequency': pf[row['route_id']]})
    df = pd.DataFrame(rows)
    order = ['priority (before)', 'priority (after)',
             'non-priority (before)', 'non-priority (after)']
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='group', y='frequency', order=order, ax=ax,
                palette=[PALETTE[3], PALETTE[2], PALETTE[3], PALETTE[2]])
    sns.stripplot(data=df, x='group', y='frequency', order=order, color='black',
                  alpha=0.25, size=2, ax=ax, jitter=0.25)
    ax.set_xlabel('')
    ax.set_ylabel('Frequency at peak (buses/hour)')
    ax.set_title('Frequency distribution: before vs after, by priority')
    ax.tick_params(axis='x', rotation=10)
    save_fig(fig, 'fig_equity_boxplot')


# Driver

def generate_all_figures(project_root="."):
    routes_df = pd.read_csv(os.path.join(project_root, 'data/processed/routes.csv'))
    demand_matrix = pd.read_csv(os.path.join(project_root, 'data/processed/demand_matrix.csv'))
    allocation = pd.read_csv(os.path.join(project_root, 'results/tables/optimal_allocation.csv'))
    test_predictions = pd.read_csv(os.path.join(project_root, 'results/tables/test_predictions.csv'))
    stoch_results = pd.read_csv(os.path.join(project_root, 'results/tables/stochastic_results.csv'))
    vss_evpi = pd.read_csv(os.path.join(project_root, 'results/tables/vss_evpi.csv'))
    df_results = pd.read_csv(os.path.join(project_root, 'results/tables/decision_focused_results.csv'))
    fleet_df = pd.read_csv(os.path.join(project_root, 'results/tables/fleet_sensitivity.csv'))
    pareto_df = pd.read_csv(os.path.join(project_root, 'results/tables/pareto_frontier.csv'))

    print("Figure 1/11: allocation heatmap"); fig_allocation_heatmap(routes_df, allocation, demand_matrix)
    print("Figure 2/11: forecast scatter"); fig_demand_forecast_scatter(test_predictions, routes_df)

    # SHAP figures need the XGBoost model + features
    import xgboost as xgb
    from src.demand_forecasting import prepare_features, get_feature_cols, chronological_split
    from src.data_processing import load_demand, load_weather
    demand_df = load_demand(os.path.join(project_root, 'data/processed/demand.csv'))
    weather_df = load_weather(os.path.join(project_root, 'data/processed/weather.csv'))
    df_full = prepare_features(demand_df, weather_df, routes_df)
    feat_cols = get_feature_cols(df_full)
    _, _, test_df = chronological_split(df_full)
    X_test = test_df[feat_cols].values.astype(np.float32)
    sample = X_test[np.random.default_rng(42).choice(len(X_test), size=min(2000, len(X_test)), replace=False)]
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(project_root, 'results/models/xgb_model.json'))

    print("Figure 3/11: SHAP importance"); fig_shap_importance(xgb_model, sample, feat_cols)
    print("Figure 4/11: SHAP beeswarm"); fig_shap_beeswarm(xgb_model, sample, feat_cols)
    print("Figure 5/11: forecast timeseries"); fig_forecast_timeseries(test_predictions, routes_df)
    print("Figure 6/11: scenario comparison"); fig_scenario_comparison(stoch_results)
    print("Figure 7/11: VSS/EVPI"); fig_vss_evpi(vss_evpi)
    multiseed_path = os.path.join(project_root, 'results/tables/decision_focused_multiseed.csv')
    multiseed_df = pd.read_csv(multiseed_path) if os.path.exists(multiseed_path) else None
    print("Figure 8/13: decision-focused"); fig_decision_focused(df_results, multiseed_df)
    print("Figure 9/13: fleet sensitivity"); fig_fleet_sensitivity(fleet_df)
    print("Figure 10/13: Pareto frontier"); fig_pareto_frontier(pareto_df)
    print("Figure 11/13: equity boxplot"); fig_equity_boxplot(routes_df, allocation)

    rel_path = os.path.join(project_root, 'results/tables/pi_reliability.csv')
    if os.path.exists(rel_path):
        reliability_df = pd.read_csv(rel_path)
        print("Figure 12/13: PI reliability"); fig_pi_reliability(reliability_df, test_predictions)

    cost_path = os.path.join(project_root, 'results/tables/cost_wait_frontier.csv')
    if os.path.exists(cost_path):
        cost_df = pd.read_csv(cost_path)
        print("Figure 13/13: cost-wait frontier"); fig_cost_wait_frontier(cost_df)


if __name__ == "__main__":
    from src.audit import (setup_audit, log_phase_start, log_phase_end,
                           log_gate_check, log_metric)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setup_audit(project_root)
    log_phase_start("PHASE_7_VISUALIZATIONS")

    generate_all_figures(project_root)

    expected_figs = [
        'fig_allocation_heatmap', 'fig_demand_forecast_scatter',
        'fig_shap_importance', 'fig_shap_beeswarm', 'fig_forecast_timeseries',
        'fig_scenario_comparison', 'fig_vss_evpi', 'fig_decision_focused',
        'fig_fleet_sensitivity', 'fig_pareto_frontier', 'fig_equity_boxplot',
        'fig_pi_reliability', 'fig_cost_wait_frontier',
    ]
    fig_dir = os.path.join(project_root, 'results/figures')
    paper_fig_dir = os.path.join(project_root, 'paper/figures')
    for fig_name in expected_figs:
        png = os.path.join(fig_dir, f"{fig_name}.png")
        pdf = os.path.join(paper_fig_dir, f"{fig_name}.pdf")
        log_gate_check(f"png_exists_{fig_name}", os.path.exists(png), "True", str(os.path.exists(png)))
        log_gate_check(f"pdf_exists_{fig_name}", os.path.exists(pdf), "True", str(os.path.exists(pdf)))
        if os.path.exists(png):
            size = os.path.getsize(png)
            log_gate_check(f"fig_size_{fig_name}", size > 5000, ">5KB", f"{size}")
    log_metric("total_figures", len([f for f in os.listdir(fig_dir) if f.endswith('.png')]))
    log_phase_end("PHASE_7_VISUALIZATIONS", "PASS")
    print("PHASE 7 COMPLETE")
