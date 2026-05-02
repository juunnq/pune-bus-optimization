# Pune Bus Frequency Optimization

Optimal bus frequency allocation for Pune's PMPML network using mixed-integer
programming, machine learning demand forecasting, and decision-focused
learning.

## Overview

PMPML serves over 800,000 daily riders across 340 routes with a fleet of about
1,800 buses. Frequency decisions today are largely manual. This project
replaces that with a transparent optimisation pipeline: a deterministic MIP
that minimises total passenger waiting time subject to fleet, depot, and
equity constraints; an XGBoost demand model with quantile heads that drives
stochastic and robust extensions; and a decision-focused neural network that
beats the standard MSE-trained pipeline on regret.

The full pipeline reproduces in roughly 75 minutes end-to-end and delivers a
26.0% reduction in total passenger waiting time at the same fleet size.

## Methodology

### Stage 1: Deterministic optimisation

A mixed-integer program with 15,300 binary variables (340 routes &times; 5
periods &times; 9 frequency levels) chooses one frequency per (route, period).
Constraints enforce a global fleet cap, per-depot capacity, and an equity
floor of 3 buses/hour at peak for priority routes that serve lower-income
neighbourhoods. CBC solves it to optimality in under five seconds.

### Stage 2: Predict-then-optimize

XGBoost (depth 8, learning rate 0.03, 2,000 trees with early stopping) is
trained on hourly ridership over months 1-8, validated on 9-10, tested on
11-12. Quantile regression heads at 0.1 / 0.5 / 0.9 supply prediction
intervals; six demand scenarios derived from those quantiles drive a stochastic
MIP (probability-weighted expected wait) and a robust minimax MIP (worst-case
wait).

### Stage 3: Decision-focused learning

Standard MSE training treats all prediction errors equally. We compare two
decision-aware variants: (a) XGBoost with sample weights proportional to
&part;wait/&part;f, the downstream optimisation sensitivity; (b) a small
fully-connected neural network trained with an SPO+-style surrogate loss
that weights L1 errors by the same sensitivity. Both are evaluated by their
*decision regret* against the oracle that sees true demand.

## Key Results

- **26.0% wait-time reduction** vs current PMPML at the same fleet size
  (16,826 vs 22,748 passenger-hours of waiting).
- **XGBoost** achieves test RMSE 12.27, MAPE 8.8%, R&sup2; 0.973, beating
  random forest (RMSE 13.04) and linear regression (RMSE 16.38).
- **80% prediction intervals** cover 72.9% of test observations.
- **VSS = 0.26** and **EVPI = 2.47** passenger-hours; both small but positive,
  with the robust solution achieving the best worst-case (18,387 vs 24,943
  for status quo).
- **Decision-focused NN** has *worse* prediction (RMSE 13.94) yet *better*
  decision (regret 0.072 vs 0.130 for XGBoost-MSE) - a 44.7% reduction in
  regret despite a 13.5% increase in RMSE. Lower prediction error does not
  imply lower decision quality.
- **Fleet sensitivity**: feasible from B=1,600 (obj 19,521) to B=2,200
  (obj 13,435); below 1,600 the depot caps render the equity floor
  infeasible. The next 400 buses are worth roughly 3,400 passenger-hours
  of waiting.
- **Pareto frontier**: equity floors 1, 2, 3 are feasible at B=1,800 and
  cost only 0.9% of total waiting; floor 4 is the cliff (infeasible).
  D5 and D6 are the binding depots during peaks (LP shadow prices -5.56
  and -9.40 passenger-hours per bus of additional capacity at AM peak).

### Scenario Comparison
![Scenario Comparison](results/figures/fig_scenario_comparison.png)

### Equity-Efficiency Tradeoff
![Pareto Frontier](results/figures/fig_pareto_frontier.png)

## Repository Structure

```
pune-bus-optimization/
&#9500;&#9472; data/                 raw/processed CSVs and fetcher
&#9500;&#9472; src/
&#9474;  &#9500;&#9472; audit.py           audit-log helpers
&#9474;  &#9500;&#9472; data_processing.py loaders + feature prep
&#9474;  &#9500;&#9472; deterministic_model.py   MIP + 4 baselines
&#9474;  &#9500;&#9472; demand_forecasting.py    LR / RF / XGBoost (incl. quantile)
&#9474;  &#9500;&#9472; stochastic_model.py      stochastic + robust + VSS/EVPI
&#9474;  &#9500;&#9472; decision_focused.py      decision-focused learning
&#9474;  &#9500;&#9472; sensitivity.py           fleet / Pareto / shadow prices
&#9474;  &#9492;&#9472; visualization.py         all 11 figures
&#9500;&#9472; dashboard/app.py        Streamlit dashboard
&#9500;&#9472; paper/main.tex          LaTeX paper + references.bib
&#9500;&#9472; results/tables/         CSV outputs
&#9500;&#9472; results/figures/        PNG figures
&#9500;&#9472; results/models/         saved XGBoost / NN models
&#9500;&#9472; scripts/run_all.py      master pipeline
&#9500;&#9472; tests/test_sanity.py    pytest sanity suite
&#9492;&#9472; audit.log               full run trace
```

## Installation & Usage

```bash
git clone https://github.com/juunnq/pune-bus-optimization.git
cd pune-bus-optimization
pip install -r requirements.txt
python scripts/run_all.py
```

### Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

### Run Tests

```bash
pytest tests/test_sanity.py -v
```

## Paper

The accompanying paper is in `paper/main.tex`. To compile:

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Citation

```bibtex
@misc{veluri2026pune,
  title={Decision-Focused Demand Learning for Equitable Bus Frequency
         Optimization},
  author={Veluri, Arjun},
  year={2026},
  howpublished={\url{https://github.com/juunnq/pune-bus-optimization}},
  note={Working paper}
}
```

## License

MIT
