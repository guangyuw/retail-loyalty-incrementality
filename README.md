# Retail Loyalty Product — Analytics + Incrementality Experiment (X5 RetailHero)

One end-to-end **product data science** project on a single product: first
**understand the product** (metrics, retention, segmentation — in SQL over
~45.8M raw transactions), then **experiment on it** (measure the causal
incremental effect of a loyalty campaign and decide *who* to target). Built on a
real randomized experiment (X5 RetailHero), from raw logs rather than a
pre-aggregated table.

## Overview

**Understand the product (product analytics)**
- **SQL aggregation layer (DuckDB)**: ~45.8M transaction rows → one feature row per customer
- **Core product metrics** (frequency, monetary, recency, purchase rate as a North-Star candidate)
- **Acquisition-cohort retention** analysis (SQL)
- **RFM segmentation** (SQL NTILE → champions / at-risk / mid)

**Experiment on the product (incrementality experiment)**
- Experiment hygiene: **SRM** (sample-ratio chi-square) + covariate balance (**SMD**)
- Average treatment effect with **regression-adjusted variance reduction** (CUPED-style)
- **Heterogeneous treatment effects (uplift / CATE)** via S- and T-learners
- Model evaluation with **Qini / AUUC** curves vs random targeting
- A **targeting policy** with **guardrails** (do-no-harm + no loyalty cannibalization)
- **Monitoring + feedback-loop plan**: PSI drift, epsilon randomized control (selective labels)

## Data
- Programmatic (downloads + caches): `from sklift.datasets import fetch_x5`
- **No manual download needed** — the first run pulls ~640MB of `*.csv.gz` and
  caches them in `./data/` (next to `pipeline.py`); later runs reuse the cache.
- Three raw tables joined on `client_id`:
  - `train` — ~200K experiment customers with `treatment_flg` + `target` (made a purchase in the post-communication window)
  - `clients` — ~400K rows of client info (age, gender, signup dates)
  - `purchases` — ~45.8M transaction rows, **all prior to communication**
- Leakage-free by construction: every engineered feature comes from
  pre-communication history.

## Run
```bash
pip install -r requirements.txt
python pipeline.py          # runs on a 30% customer sample (fast, ~40s on a laptop)
```
`fetch_x5` loads all ~45.8M transaction rows into memory; the **full-data**
DuckDB aggregation exceeds an 8GB laptop, so the pipeline defaults to a **30%
customer sample** for fast iteration (50% also fits, ~2.7GB peak, but is slower).
On a larger box, run the full dataset with `main(sample_frac=None)`.

## Walkthrough notebook
`analysis.ipynb` walks through all 9 steps interactively (one markdown
explanation + code + visualization per step), calling the functions in
`pipeline.py` — the cleanest way to read the project end-to-end.
```bash
jupyter lab analysis.ipynb   # or open in VS Code / Cursor and pick the venv
```
A `SAMPLE_FRAC` toggle in the Setup cell trades speed vs. realism:
`0.3` = the committed numbers (fast default), `0.1` = a ~30s pass while
iterating (structure identical, uplift signal noisier).

## Structure
`pipeline.py` runs `step1_load` → `step9_monitoring`: understand the product
(steps 1–3, product analytics) then experiment on it (steps 4–9). Run
end-to-end (`python pipeline.py`) or step by step. `analysis.ipynb` walks the
same steps interactively; `development.ipynb` is the exploratory scratch
notebook the functions were distilled from.
