# Pune Bus Frequency Optimization (PMPML)

A predict-then-optimize benchmark for bus frequency allocation, evaluated on
the 544-route PMPML (Pune Mahanagar Parivahan Mahamandal) network anchored
to public Indian transit data. The pipeline combines mixed-integer
programming, demand forecasting, and decision-focused learning to study how
prediction quality and decision quality relate in predict-then-optimize
systems.

## Overview

The instance is built from publicly available data:

- **Route topology, distances**: 544 PMPML routes from the OpenCity
  Pune Bus Stops and Routes dataset
  ([data.opencity.in](https://data.opencity.in/dataset/pune-bus-stops-and-routes)),
  reduced from the 1,030 direction-pair entries in the source CSV.
- **Equity priorities**: per-route designation derived from Census 2011
  ward-level Scheduled Caste + Scheduled Tribe population share
  ([data.opencity.in](https://data.opencity.in/dataset/pune-census-2011-data)),
  mapped to route endpoints via a landmark→ward alias table.
- **Depot list**: 13 depots (9 CNG + 4 electric) per published PMPML
  statistics — Aundh, Bhosari, Dhankawadi, Hadapsar, Katraj, Kothrud,
  Nigdi, Pimpri, Swargate (CNG); Balewadi, Maan-Hinjewadi, Shewalewadi,
  Charholi (electric).
- **Fleet scale**: B = 2,000 peak-deployable buses, matching the latest
  PMPML stated figure.

Per-route hourly ridership remains simulated (PMPML does not release
AFC / smart-card data); the diurnal profile is calibrated to PMPML's
~1.1 million daily ridership and the 8–11 AM / 5–9 PM peaks reported in
published Pune traffic studies. The contribution is methodological:

- a deterministic MIP with fleet, depot, equity, and operator-cost
  constraints, plus a $\lambda$-sweep that traces the rider-wait /
  operator-cost Pareto frontier (the headline "% improvement" depends
  strongly on $\lambda$);
- a forecasting pipeline with seven quantile heads and split-conformal
  calibration of the 80% prediction interval on the validation fold;
- a 20-seed sweep comparing four decision-focused training variants with a
  paired Wilcoxon signed-rank test for inference, rather than a single-seed
  point comparison.

## Methodology

### Stage 1: Deterministic optimisation

A mixed-integer program with 24,480 binary variables (544 routes × 5
periods × 9 frequency levels) chooses one frequency per (route, period).
Constraints enforce a global fleet cap of B = 2,000 buses, per-depot
capacity (allocated proportional to each depot's minimum bus need plus
jitter, since PMPML does not publish per-depot fleet counts), and an
equity floor of 3 buses/hour at peak for priority routes serving
high-SC/ST wards. CBC solves it to optimality in ~15 seconds.

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

## Key Results (real PMPML instance)

- **19.5% wait-time reduction** at λ=0 (no operator-cost penalty) — MIP
  optimum 26,226 vs status quo 32,583 passenger-hours. The optimum at
  λ=0 saturates the fleet (10,000 bus-periods across the five operating
  periods, i.e., B=2,000 per period). At realistic λ the gap shrinks
  rapidly: λ=2 yields 26,911 pass-hr at 9,608 bus-periods; λ=5 yields
  30,503 pass-hr at 8,463 bus-periods (worse than status quo on rider
  waiting). See `results/tables/cost_wait_frontier.csv` for the full sweep.
- **XGBoost** achieves test RMSE 7.17, MAPE 15.2%, R² 0.889 on the 50
  modelled routes, beating random forest (RMSE 7.59) and linear
  regression (RMSE 8.88). The DGP includes a latent per-day factor,
  hidden surge events, and heavy-tail noise on a subset of routes,
  capping achievable R² near 0.9.
- **80% prediction intervals** cover 70.0% of test observations from the
  raw quantile heads; split-conformal calibration on the validation
  fold raises empirical coverage to 80.4% (matches nominal).
- **VSS = 0.53** and **EVPI = 1.91** passenger-hours — both positive but
  extremely small (≈2 parts in 10⁴ of the objective).
- **Decision-focused NN vs XGBoost-MSE**: regret comparison over 20 seeds
  with a paired Wilcoxon signed-rank test; see
  `results/tables/decision_focused_summary.csv` and
  `decision_focused_wilcoxon.csv` for the inferential outcome.
- **Fleet sensitivity**: infeasible at B<2,000 under the equity floor;
  B=2,000→2,200 recovers 3,144 pass-hr, B=2,000→2,400 recovers
  5,532 pass-hr. The CIRT benchmark of ~50 buses per lakh population
  predicts ~2,000 for Pune's ~4M service area — the feasibility cliff
  matches the operational reality.
- **Pareto frontier**: equity floors 1, 2, 3 are feasible at B=2,000;
  floor 4 is the cliff (infeasible). The current k=3 floor costs ~8% of
  total waiting time relative to no equity constraint. The binding
  constraint at peak is fleet, not depot — depot caps are allocated
  proportional to each depot's minimum bus need (priority floor at k=3 +
  remaining routes at k=1) plus jitter.

### Scenario Comparison
![Scenario Comparison](results/figures/fig_scenario_comparison.png)

### Equity-Efficiency Tradeoff
![Pareto Frontier](results/figures/fig_pareto_frontier.png)

## Repository Structure

```
pune-bus-optimization/
&#9500;&#9472; data/
&#9474;  &#9500;&#9472; raw/                 real PMPML routes + Census 2011 ward CSVs
&#9474;  &#9500;&#9472; processed/           pipeline-generated CSVs
&#9474;  &#9500;&#9472; fetch_data.py        synthetic ridership generator on real topology
&#9474;  &#9492;&#9472; load_real_data.py    real route + ward + depot loader
&#9500;&#9472; src/
&#9474;  &#9500;&#9472; audit.py           audit-log helpers
&#9474;  &#9500;&#9472; data_processing.py loaders + feature prep
&#9474;  &#9500;&#9472; deterministic_model.py   MIP + 4 baselines
&#9474;  &#9500;&#9472; demand_forecasting.py    LR / RF / XGBoost (incl. 7 quantiles + conformal)
&#9474;  &#9500;&#9472; stochastic_model.py      stochastic + robust + VSS/EVPI
&#9474;  &#9500;&#9472; decision_focused.py      decision-focused 20-seed sweep + Wilcoxon
&#9474;  &#9500;&#9472; sensitivity.py           fleet / Pareto / shadow / cost-wait frontier
&#9474;  &#9492;&#9472; visualization.py         all 13 figures
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
