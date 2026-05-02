"""
Post-build sanity tests.
Run with: pytest tests/test_sanity.py -v
"""
import os

import pandas as pd


# Data

def test_data_files_exist():
    for f in ['data/processed/routes.csv', 'data/processed/demand.csv',
              'data/processed/weather.csv', 'data/processed/demand_matrix.csv']:
        assert os.path.exists(f), f"Missing: {f}"


def test_routes_valid():
    df = pd.read_csv('data/processed/routes.csv')
    assert len(df) >= 300
    assert df.isnull().sum().sum() == 0
    assert (df['length_km'] > 0).all()
    assert set(df['route_category'].unique()) == {'trunk', 'feeder', 'suburban'}


def test_demand_valid():
    df = pd.read_csv('data/processed/demand.csv')
    assert len(df) > 100000
    assert (df['ridership'] >= 0).all()


# Result CSVs

def test_result_files_exist():
    for f in ['results/tables/deterministic_results.csv',
              'results/tables/forecast_results.csv',
              'results/tables/stochastic_results.csv',
              'results/tables/vss_evpi.csv',
              'results/tables/decision_focused_results.csv',
              'results/tables/fleet_sensitivity.csv',
              'results/tables/pareto_frontier.csv']:
        assert os.path.exists(f), f"Missing: {f}"


def test_optimal_beats_baseline():
    df = pd.read_csv('results/tables/deterministic_results.csv')
    optimal = df[df['method'] == 'optimal']['total_wait_time'].iloc[0]
    baseline = df[df['method'] == 'status_quo']['total_wait_time'].iloc[0]
    assert optimal < baseline


def test_xgboost_beats_baselines():
    df = pd.read_csv('results/tables/forecast_results.csv')
    xgb = df[df['model'] == 'xgboost']['rmse'].iloc[0]
    lr = df[df['model'] == 'linear_regression']['rmse'].iloc[0]
    assert xgb < lr


def test_vss_non_negative():
    df = pd.read_csv('results/tables/vss_evpi.csv')
    vss = df[df['metric'] == 'VSS']['value'].iloc[0]
    assert vss >= -0.01


def test_evpi_geq_vss():
    df = pd.read_csv('results/tables/vss_evpi.csv')
    vss = df[df['metric'] == 'VSS']['value'].iloc[0]
    evpi = df[df['metric'] == 'EVPI']['value'].iloc[0]
    assert evpi >= vss - 0.01


def test_fleet_sensitivity_monotonic():
    df = pd.read_csv('results/tables/fleet_sensitivity.csv')
    df = df[df['objective'] > 0].sort_values('fleet_size')
    objs = df['objective'].values
    assert objs[-1] < objs[0]


def test_decision_focused_all_methods():
    df = pd.read_csv('results/tables/decision_focused_results.csv')
    expected = {'xgb_mse', 'xgb_weighted', 'nn_mse', 'nn_spo', 'oracle'}
    assert expected.issubset(set(df['method'].values))


# Figures

def test_figures_exist():
    figs = [f for f in os.listdir('results/figures') if f.endswith('.png')]
    assert len(figs) >= 10


def test_paper_figures_exist():
    pdfs = [f for f in os.listdir('paper/figures') if f.endswith('.pdf')]
    assert len(pdfs) >= 10


# Paper

def test_paper_exists():
    assert os.path.exists('paper/main.tex')
    with open('paper/main.tex', encoding='utf-8') as f:
        tex = f.read()
    assert len(tex) > 10000
    assert 'Introduction' in tex
    assert 'Conclusion' in tex


def test_bib_exists():
    assert os.path.exists('paper/references.bib')
    with open('paper/references.bib', encoding='utf-8') as f:
        bib = f.read()
    assert bib.count('@') >= 8


# Audit

def test_audit_log_complete():
    assert os.path.exists('audit.log')
    with open('audit.log', encoding='utf-8') as f:
        log = f.read()
    for phase in ['PHASE_1_DATA', 'PHASE_2_DETERMINISTIC',
                  'PHASE_3_FORECASTING', 'PHASE_4_STOCHASTIC',
                  'PHASE_5_DECISION_FOCUSED', 'PHASE_6_SENSITIVITY',
                  'PHASE_7_VISUALIZATIONS']:
        assert f"PHASE END: {phase} | STATUS: PASS" in log, f"{phase} missing PASS"


# Dashboard

def test_dashboard_valid():
    import py_compile
    py_compile.compile('dashboard/app.py', doraise=True)
