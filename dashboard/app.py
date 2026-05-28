"""
Interactive dashboard for pune-bus-optimization.
Run with: streamlit run dashboard/app.py
"""
import os
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

st.set_page_config(page_title="Pune Bus Optimization", layout="wide")


@st.cache_data
def load_data():
    return {
        'det_results': pd.read_csv('results/tables/deterministic_results.csv'),
        'allocation': pd.read_csv('results/tables/optimal_allocation.csv'),
        'forecast': pd.read_csv('results/tables/forecast_results.csv'),
        'stoch_results': pd.read_csv('results/tables/stochastic_results.csv'),
        'vss_evpi': pd.read_csv('results/tables/vss_evpi.csv'),
        'scenario_defs': pd.read_csv('results/tables/scenario_definitions.csv'),
        'df_results': pd.read_csv('results/tables/decision_focused_results.csv'),
        'fleet_sens': pd.read_csv('results/tables/fleet_sensitivity.csv'),
        'pareto': pd.read_csv('results/tables/pareto_frontier.csv'),
        'shadow': pd.read_csv('results/tables/shadow_prices.csv'),
        'routes': pd.read_csv('data/processed/routes.csv'),
    }


data = load_data()

st.sidebar.title("Pune Bus Optimization")
st.sidebar.markdown("PMPML frequency allocation under demand uncertainty.")
view = st.sidebar.radio("View", [
    "Overview", "Optimization Results", "Demand Forecasting",
    "Stochastic Analysis", "Decision-Focused Learning",
    "Sensitivity Analysis", "About",
])

# Overview
if view == "Overview":
    st.title("Overview")
    det = data['det_results']
    sq = det[det['method'] == 'status_quo']['total_wait_time'].iloc[0]
    opt = det[det['method'] == 'optimal']['total_wait_time'].iloc[0]
    pct = (sq - opt) / sq * 100
    vss = data['vss_evpi'].set_index('metric').loc['VSS', 'value']
    evpi = data['vss_evpi'].set_index('metric').loc['EVPI', 'value']
    fleet_used = int(data['allocation']['buses_assigned'].sum() / 5)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wait-time reduction", f"{pct:.1f}%", delta="vs current PMPML")
    col2.metric("VSS", f"{vss:.2f}")
    col3.metric("EVPI", f"{evpi:.2f}")
    col4.metric("Avg fleet/period", f"{fleet_used}")

    st.markdown("### Method comparison across scenarios")
    st.image("results/figures/fig_scenario_comparison.png", use_column_width=True)

    st.markdown("### Equity-efficiency tradeoff")
    st.image("results/figures/fig_pareto_frontier.png", use_column_width=True)

# Optimization Results
elif view == "Optimization Results":
    st.title("Deterministic Optimization Results")
    st.markdown("Compare the MIP optimum against four baselines on the same fleet "
                "/ equity / depot constraints.")
    st.dataframe(data['det_results'].round(2))

    st.markdown("### Top-30 routes: current vs optimal allocation")
    st.image("results/figures/fig_allocation_heatmap.png", use_column_width=True)

    st.markdown("### Equity: priority vs non-priority routes at peak")
    st.image("results/figures/fig_equity_boxplot.png", use_column_width=True)

# Demand Forecasting
elif view == "Demand Forecasting":
    st.title("Demand Forecasting")
    st.markdown("Three models compared on a chronological train / val / test split.")
    st.dataframe(data['forecast'].round(3))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Predictions vs actuals")
        st.image("results/figures/fig_demand_forecast_scatter.png", use_column_width=True)
    with col2:
        st.markdown("### SHAP feature importance")
        st.image("results/figures/fig_shap_importance.png", use_column_width=True)

    st.markdown("### Test-set time series with 80% prediction intervals")
    st.image("results/figures/fig_forecast_timeseries.png", use_column_width=True)

    st.markdown("### SHAP beeswarm")
    st.image("results/figures/fig_shap_beeswarm.png", use_column_width=True)

# Stochastic Analysis
elif view == "Stochastic Analysis":
    st.title("Stochastic and Robust Optimization")
    st.markdown("### Scenario definitions")
    st.dataframe(data['scenario_defs'])

    st.markdown("### Method performance per scenario")
    st.image("results/figures/fig_scenario_comparison.png", use_column_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### VSS and EVPI")
        st.dataframe(data['vss_evpi'].round(4))
    with col2:
        st.image("results/figures/fig_vss_evpi.png", use_column_width=True)

    st.markdown("### Per-method per-scenario objectives")
    pivot = (data['stoch_results']
             .pivot(index='scenario', columns='method', values='objective_value')
             .round(2))
    st.dataframe(pivot)

# Decision-Focused Learning
elif view == "Decision-Focused Learning":
    st.title("Decision-Focused Learning")
    st.markdown("""
    Standard predict-then-optimize trains the predictor with MSE.  Decision-focused
    methods weight or directly minimise the *downstream* decision regret.  The result
    below shows that **lower RMSE does not imply lower regret**.
    """)
    st.dataframe(data['df_results'].round(4))
    st.image("results/figures/fig_decision_focused.png", use_column_width=True)

    df = data['df_results']
    xgb_mse_regret = df[df['method'] == 'xgb_mse']['regret'].iloc[0]
    nn_spo_regret = df[df['method'] == 'nn_spo']['regret'].iloc[0]
    if xgb_mse_regret > 0:
        improvement = (xgb_mse_regret - nn_spo_regret) / xgb_mse_regret * 100
        st.info(f"NN with SPO+ surrogate cuts regret by **{improvement:.1f}%** "
                f"vs XGBoost (MSE), despite larger RMSE.")

# Sensitivity
elif view == "Sensitivity Analysis":
    st.title("Sensitivity Analysis")

    st.markdown("### Fleet size")
    fs = data['fleet_sens']
    fleet_size = st.slider("Fleet size to highlight", min_value=int(fs['fleet_size'].min()),
                           max_value=int(fs['fleet_size'].max()),
                           value=2000, step=200)
    closest = fs.iloc[(fs['fleet_size'] - fleet_size).abs().argmin()]
    if closest['objective'] > 0:
        st.write(f"At **B = {int(closest['fleet_size'])}**: "
                 f"obj = {closest['objective']:.2f}, "
                 f"routes with freq>=3: {int(closest['routes_above_3'])}, "
                 f"avg freq: {closest['avg_freq']:.2f}")
    else:
        st.warning(f"B = {int(closest['fleet_size'])} is infeasible.")
    st.image("results/figures/fig_fleet_sensitivity.png", use_column_width=True)

    st.markdown("### Equity-efficiency Pareto frontier")
    st.dataframe(data['pareto'].round(3))
    st.image("results/figures/fig_pareto_frontier.png", use_column_width=True)

    st.markdown("### LP relaxation shadow prices")
    st.dataframe(data['shadow'].round(4))

# About
else:
    st.title("About")
    st.markdown("""
    This project optimizes bus frequency allocation for Pune's PMPML network.

    **Pipeline.** A deterministic MIP allocates ~2000 buses across 544 routes and 5
    time periods to minimise total passenger waiting time, subject to fleet, equity,
    and depot capacity constraints.  An XGBoost demand model with quantile heads
    drives stochastic and robust formulations; a decision-focused XGBoost variant
    trains with sample weights proportional to downstream sensitivity, and is
    evaluated against MSE-XGBoost across 20 seeds with a paired Wilcoxon test.
    Sensitivity sweeps over fleet size, equity threshold, operator-cost penalty
    λ, and priority-mapping rule; LP-relaxation duals reveal binding
    constraints.

    **Reproduce.**  `python scripts/run_all.py`.
    """)
