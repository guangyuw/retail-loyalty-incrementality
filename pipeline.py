"""
Retail Loyalty Product: Analytics + Incrementality Experiment (X5 RetailHero).

One end-to-end *product data science* project on a single product that first
understands the product, then experiments on it:

  1. Understand the product - define core metrics, measure acquisition-cohort
     retention, and segment customers (RFM), with the heavy aggregation done in
     SQL (DuckDB) over the raw ~45.8M transaction logs.

  2. Experiment on the product - analyze the randomized loyalty campaign:
     experiment hygiene, average effect with CUPED-style variance reduction,
     heterogeneous effects (uplift/CATE), a targeting policy with guardrails,
     and a monitoring + feedback-loop plan.

Why this scope: it covers both sides product DS interviews probe - product
analytics (metrics / cohorts / segmentation / SQL) AND experimentation /
causal inference - inside one coherent narrative ("understand the product,
then experiment on it"), on real raw logs rather than a pre-aggregated table.

Run:  python pipeline.py        # runs on a 30% customer sample (fast, 8GB-safe)
Data: X5 RetailHero uplift dataset (real retail loyalty experiment).
      ~200K experiment customers (treatment/target) on top of ~45.8M raw
      purchase transactions (prior to communication) + client info.

Memory note: fetch_x5 loads all ~45.8M transaction rows into memory; the
full-data DuckDB aggregation exceeds an 8GB laptop. The default run uses a 30%
customer sample for fast iteration (50% also fits, ~2.7GB peak, but is slower).
Pass sample_frac=None on a bigger box to run the full dataset.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

TREATMENT = "treatment_flg"   # 1 = received communication, 0 = control
OUTCOME = "target"            # binary: made a purchase in the post-communication window
ID = "client_id"             # primary key joining clients / train / purchases

# Cache the raw X5 download next to this file (a `data/` folder) instead of the
# default ~/scikit-uplift-data, so the project is self-contained and portable.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _duck(sql: str, **frames: pd.DataFrame) -> pd.DataFrame:
    """Run a SQL query (DuckDB) over in-memory pandas frames.

    Lets us do the heavy per-customer aggregation and the cohort/RFM analytics
    in real SQL - the bread-and-butter skill product DS interviews expect -
    while staying in one Python script. Each kwarg name becomes a SQL table.
    """
    import tempfile
    import duckdb

    con = duckdb.connect()
    try:
        # Memory hygiene for the full 45.8M-row table: let DuckDB spill to disk
        # and drop insertion-order tracking (cheaper, and we don't rely on it).
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET temp_directory='{tempfile.gettempdir()}'")
        # Single-threaded so the parallel float-summation order is fixed: the
        # aggregated features become bit-identical across runs, which (together
        # with the client_id sort below) makes the whole uplift pipeline
        # reproducible. The small uplift signal is sensitive to ~1e-10 feature
        # jitter, which would otherwise move the split and the Qini metric.
        con.execute("SET threads TO 1")
        for name, frame in frames.items():
            con.register(name, frame)
        return con.execute(sql).df()
    finally:
        con.close()


# ======================================================================
# Understand the product (product analytics)
# ======================================================================

# ----------------------------------------------------------------------
# Step 1. Load raw tables
# ----------------------------------------------------------------------
def step1_load(data_home: str = DATA_DIR,
               sample_frac: float | None = None,
               seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the X5 RetailHero dataset via scikit-uplift (downloads + caches).

    Unlike pre-aggregated datasets, X5 hands you THREE raw tables:
      - train     : the ~200K experiment customers (treatment_flg, target)
      - clients   : general client info (age, gender, signup dates, ...)
      - purchases : ~45.8M transaction rows, all BEFORE the communication

    We attach treatment/target onto `train` and return the three tables.
    The feature matrix + analytics are BUILT downstream (that's the point of X5).

    Args:
        data_home: where the ~640MB of raw `*.csv.gz` are cached (default: the
            project's `data/` folder). The first call downloads; later calls reuse.
        sample_frac: if set (e.g. 0.1), keep a random fraction of *customers* and
            only their purchases. Makes the whole pipeline run in seconds on a
            laptop while exercising every step - ideal for the notebook walkthrough.
            Leave as None for the full ~200K customers / ~45.8M transactions.

    Memory note: at full size `purchases` is ~45.8M rows (a few GB in pandas). On
    a 16GB laptop it loads; if it's tight, use `sample_frac` or aggregate in chunks.
    """
    from sklift.datasets import fetch_x5

    bunch = fetch_x5(data_home=data_home)
    clients = bunch.data["clients"].copy()
    train = bunch.data["train"].copy()

    # fetch_x5 returns target/treatment as separate Series aligned to `train`.
    train[OUTCOME] = bunch.target.values
    train[TREATMENT] = bunch.treatment.values

    if sample_frac is not None and sample_frac < 1.0:
        # Subsample at the customer level so every per-customer aggregate stays
        # internally consistent, then keep only those customers' transactions.
        train = (train.sample(frac=sample_frac, random_state=seed)
                       .reset_index(drop=True))
        keep = set(train[ID])
        purchases = bunch.data["purchases"]
        purchases = purchases[purchases[ID].isin(keep)].reset_index(drop=True)
    else:
        purchases = bunch.data["purchases"]

    print(f"[load] train={len(train):,}  clients={len(clients):,}  "
          f"purchases={len(purchases):,}"
          + (f"  (sample_frac={sample_frac})" if sample_frac else ""))
    return train, clients, purchases


# ----------------------------------------------------------------------
# Step 2. SQL aggregation layer: 45.8M rows -> one feature row per customer
# ----------------------------------------------------------------------
def step2_features_sql(train: pd.DataFrame,
                       clients: pd.DataFrame,
                       purchases: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate raw transactions into per-customer RFM + behavioral features,
    in SQL (DuckDB).

    Leakage guard: every purchase here is *prior to communication*, so any
    aggregate is a legitimate pre-treatment covariate. We use NO post-
    communication signal.

    Doing this in SQL (vs pandas groupby) is deliberate: it mirrors how this
    work happens against a warehouse and shows the SQL muscle product DS roles
    test for. The CTEs roll product-line rows -> transactions -> customers.
    """
    purchases["transaction_datetime"] = pd.to_datetime(
        purchases["transaction_datetime"], errors="coerce")
    ref = purchases["transaction_datetime"].max()  # proxy "campaign start"

    feats = _duck(
        """
        WITH tx AS (                       -- product lines -> one row per transaction
            SELECT client_id, transaction_id,
                   MAX(purchase_sum)            AS tx_sum,
                   COUNT(DISTINCT product_id)   AS tx_items,
                   MAX(transaction_datetime)    AS tx_time,
                   MAX(regular_points_received) AS tx_pts_received,
                   MAX(regular_points_spent)    AS tx_pts_spent,
                   MAX(express_points_received) AS tx_exp_received,
                   MAX(express_points_spent)    AS tx_exp_spent,
                   MAX(trn_sum_from_iss)        AS tx_sum_from_iss,
                   MAX(trn_sum_from_red)        AS tx_sum_from_red
            FROM purchases
            GROUP BY client_id, transaction_id
        ),
        cust AS (                          -- transactions -> one row per customer
            SELECT client_id,
                   COUNT(*)            AS n_tx,
                   SUM(tx_sum)         AS spend_total,
                   AVG(tx_sum)         AS spend_mean,
                   STDDEV_SAMP(tx_sum) AS spend_std,
                   AVG(tx_items)       AS basket_mean,
                   MAX(tx_time)        AS last_tx,
                   MIN(tx_time)        AS first_tx,
                   SUM(tx_pts_received) AS pts_received,
                   SUM(tx_pts_spent)    AS pts_spent,
                   SUM(tx_exp_received) AS exp_pts_received,
                   SUM(tx_exp_spent)    AS exp_pts_spent,
                   AVG(tx_sum_from_iss) AS sum_from_iss_mean,
                   AVG(tx_sum_from_red) AS sum_from_red_mean
            FROM tx GROUP BY client_id
        ),
        prod AS (                          -- breadth: distinct products/stores
            SELECT client_id,
                   COUNT(DISTINCT product_id)   AS n_products,
                   COUNT(DISTINCT store_id)     AS n_stores
            FROM purchases GROUP BY client_id
        )
        SELECT c.*, p.n_products, p.n_stores
        FROM cust c LEFT JOIN prod p USING (client_id)
        """,
        purchases=purchases,
    )

    feats["last_tx"] = pd.to_datetime(feats["last_tx"])
    feats["first_tx"] = pd.to_datetime(feats["first_tx"])
    feats["recency_days"] = (ref - feats["last_tx"]).dt.days
    feats["tenure_days"] = (feats["last_tx"] - feats["first_tx"]).dt.days
    feats = feats.drop(columns=["last_tx", "first_tx"])

    # Join demographics + loyalty-card dates from clients. The first_redeem_date
    # signals (did they ever redeem points, how soon, how recently) turned out to
    # be the single biggest lever on uplift quality -- redeeming reveals a
    # promo-sensitive customer, exactly the kind a campaign can move. We derive
    # them here rather than copying any external feature list.
    demo_cols = [c for c in ("age", "gender") if c in clients.columns]
    date_cols = [c for c in ("first_issue_date", "first_redeem_date")
                 if c in clients.columns]
    cl = clients[[ID] + demo_cols + date_cols].copy()
    for c in date_cols:
        cl[c] = pd.to_datetime(cl[c], errors="coerce")
    if "first_issue_date" in cl.columns:
        cl["issue_tenure_days"] = (ref - cl["first_issue_date"]).dt.days
    if "first_redeem_date" in cl.columns:
        cl["has_redeemed"] = cl["first_redeem_date"].notna().astype(float)
        cl["redeem_recency_days"] = (ref - cl["first_redeem_date"]).dt.days
        if "first_issue_date" in cl.columns:
            cl["issue_to_redeem_days"] = (
                cl["first_redeem_date"] - cl["first_issue_date"]).dt.days
    cl = cl.drop(columns=date_cols)

    df = (train.merge(feats, on=ID, how="left")
               .merge(cl, on=ID, how="left"))
    if "gender" in df.columns:
        df = pd.get_dummies(df, columns=["gender"], dummy_na=True, drop_first=True)

    # Deterministic row order: the DuckDB aggregation above runs with
    # preserve_insertion_order=false (memory hygiene for the 45.8M-row table),
    # so its output order is not stable. Sort by client_id so the downstream
    # train/test split (and therefore every uplift metric) is reproducible.
    df = df.sort_values(ID, kind="mergesort").reset_index(drop=True)

    feature_names = [c for c in df.columns if c not in (ID, OUTCOME, TREATMENT)]
    print(f"[features] customers={len(df):,}  n_features={len(feature_names)} (built in SQL)")
    return df, feature_names


# ----------------------------------------------------------------------
# Step 3. Product analytics: core metrics + cohort retention + RFM segments
# ----------------------------------------------------------------------
def step3_product_analytics(df: pd.DataFrame, purchases: pd.DataFrame) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Understand the product: what's its state before we touch it? Three classic
    product-analytics views.

    Returns (df, retention) where df has an `rfm_segment` column (used in later
    experiment steps to check whether the targeting policy just re-targets
    already-loyal customers) and retention is the month-offset pivot table
    (for charting via step3_charts).
    """
    # 1) Core product metrics (a one-glance health readout) -- all built from
    #    pre-communication history, so this stays a clean pre-experiment view.
    #    The purchase rate is reported later (step5_ate), measured on the control
    #    arm, so it isn't contaminated by the campaign's effect.
    metrics = {
        "customers": int(len(df)),
        "avg_purchase_frequency": round(float(df["n_tx"].mean()), 2),
        "avg_monetary_per_customer": round(float(df["spend_total"].mean()), 2),
        "median_recency_days": float(df["recency_days"].median()),
    }
    print(f"[analytics] core product metrics: {metrics}")

    # 2) Acquisition-cohort monthly retention (SQL): group customers by their
    #    first-purchase month, then count how many are active m months later.
    cohort = _duck(
        """
        WITH first_m AS (
            SELECT client_id,
                   date_trunc('month', MIN(transaction_datetime)) AS cohort
            FROM purchases GROUP BY client_id
        ),
        act AS (
            SELECT DISTINCT client_id,
                   date_trunc('month', transaction_datetime) AS active_month
            FROM purchases
        )
        SELECT f.cohort,
               date_diff('month', f.cohort, a.active_month) AS month_offset,
               COUNT(*)                                     AS n_active
        FROM first_m f JOIN act a USING (client_id)
        GROUP BY 1, 2 ORDER BY 1, 2
        """,
        purchases=purchases,
    )
    pivot = cohort.pivot(index="cohort", columns="month_offset", values="n_active")
    retention = pivot.div(pivot[0], axis=0)  # month_offset 0 = cohort base size
    print("[analytics] cohort retention (rows=cohort, cols=months since first buy):")
    print(retention.iloc[-6:, :6].round(2))

    # 3) RFM segmentation (SQL NTILE): score Recency / Frequency / Monetary into
    #    quintiles, then label simple segments. High R = most recent.
    rfm_in = df.dropna(subset=["recency_days", "n_tx", "spend_total"])[
        [ID, "recency_days", "n_tx", "spend_total"]]
    rfm = _duck(
        """
        SELECT client_id,
               NTILE(5) OVER (ORDER BY recency_days DESC, client_id) AS R,
               NTILE(5) OVER (ORDER BY n_tx, client_id)              AS F,
               NTILE(5) OVER (ORDER BY spend_total, client_id)       AS M
        FROM rfm_in
        """,
        rfm_in=rfm_in,
    )
    rfm["rfm_segment"] = np.where((rfm.R >= 4) & (rfm.F >= 4), "champions",
                         np.where((rfm.R <= 2) & (rfm.F <= 2), "at_risk", "mid"))
    print(f"[analytics] RFM segments: {rfm['rfm_segment'].value_counts().to_dict()}")

    df = df.merge(rfm[[ID, "rfm_segment"]], on=ID, how="left")
    df["rfm_segment"] = df["rfm_segment"].fillna("unknown")
    return df, retention


def step3_charts(df: pd.DataFrame, retention: "pd.DataFrame") -> None:
    """Draw the two product-analytics charts produced by step3_product_analytics.

    Separated so pipeline.py (script mode) can skip plotting while the
    notebook can call this after step3_product_analytics.
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2, 1]})

    im = ax[0].imshow(retention.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax[0].set_title("Acquisition-cohort retention")
    ax[0].set_xlabel("months since first purchase")
    ax[0].set_ylabel("cohort (first-purchase month)")
    ax[0].set_yticks(range(len(retention)))
    ax[0].set_yticklabels([str(c)[:7] for c in retention.index], fontsize=7)
    fig.colorbar(im, ax=ax[0], label="retained share")

    seg_counts = df["rfm_segment"].value_counts()
    colors = ["#2ca02c", "#7f7f7f", "#d62728", "#bbbbbb"][:len(seg_counts)]
    ax[1].bar(seg_counts.index, seg_counts.values, color=colors)
    ax[1].set_title("RFM segments")
    ax[1].set_ylabel("customers")
    for i, v in enumerate(seg_counts.values):
        ax[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()


# ======================================================================
# Experiment on the product (incrementality experiment)
# ======================================================================

# ----------------------------------------------------------------------
# Step 4. Experiment hygiene: split, balance, base rates
# ----------------------------------------------------------------------
def step4_validate(df: pd.DataFrame, feature_names: list[str]) -> None:
    """Experiment design (retrospective) + validity checks.

    Design block makes the standard A/B-test scaffolding explicit -- the part a
    PM/DS owns *before* launch -- even though X5's experiment is already run:
      - Hypotheses: H0 = treat & control purchase rates equal; Ha = they differ
        (two-sided).
      - Primary metric: purchase rate (`target`, measured in the post-
        communication window). Secondary (ATE by segment, descriptive) plus
        population-level all-send guardrails (no segment harmed, no loyalty
        cannibalization) gate the launch decision in Step 5. The separate
        top-k% *targeting* guardrail (is the model picking the right people?)
        lives in Step 8 and is a different question.
      - Power / sample size: estimate the baseline rate from PRE-experiment
        history (never the experiment's own outcome -- that would be circular),
        then ask how many users/arm a two-proportion test needs for a target
        MDE at alpha=0.05, power=0.80, and compare to X5's actual n.

    Validity block is the post-data hygiene gate (run this on real data before
    trusting any effect): SRM (sample-ratio), covariate balance (SMD), and base
    rates by arm. (An AA test -- comparing two never-treated slices -- is the
    other classic randomization check when a pre-period is available.)
    """
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    n = len(df)
    n_t = int(df[TREATMENT].sum())
    n_c = n - n_t

    # --- Experiment design: sample size from a PRE-experiment baseline ---
    # Sizing must rely only on data available *before* launch -- never the
    # experiment's own outcome, which would be circular (you can't read the
    # result to decide whether to run the study). We estimate the baseline
    # purchase rate from pre-communication history: the share of customers
    # who bought within the last `window` days. recency_days is a pre-treatment
    # feature built purely from pre-communication purchases, and `window` matches the
    # post-communication window over which `target` is measured.
    window = 7   # days; matches how `target` is defined
    base_rate = float((df["recency_days"] <= window).mean())
    mde = 0.01   # minimum detectable effect we'd act on: +1 percentage point
    h = proportion_effectsize(base_rate + mde, base_rate)    # Cohen's h
    req_n = NormalIndPower().solve_power(
        effect_size=h, alpha=0.05, power=0.80, alternative="two-sided")
    actual_n = min(n_t, n_c)
    # Compute achievable MDE at the actual n (what effect can we reliably detect?)
    achievable_h = NormalIndPower().solve_power(
        nobs1=actual_n, alpha=0.05, power=0.80, alternative="two-sided")
    # Convert Cohen's h back to percentage-point difference at this base rate.
    phi0 = 2 * np.arcsin(np.sqrt(base_rate))
    achievable_mde_pp = np.sin((phi0 + achievable_h) / 2) ** 2 - base_rate
    print("[design] H0: rate_treat == rate_control | Ha: != (two-sided, "
          "alpha=0.05, power=0.80)")
    print(f"[design] baseline(hist {window}d pre-period)={base_rate:.4f}  "
          f"n/arm={actual_n:,}  achievable MDE={achievable_mde_pp:+.4f}pp "
          f"(designed for +{mde:.2f}pp)")

    # --- Validity: split, SRM, balance, base rates ---
    print(f"[split] treat={n_t:,} ({n_t/n:.3f})  control={n_c:,} ({n_c/n:.3f})")

    # SRM (sample-ratio mismatch): chi-square of the realized counts vs the
    # design ratio. SRM catches pipeline/instrumentation bugs and is the
    # standard first gate. X5 doesn't publish its design ratio, so we test
    # against 50/50 as the most likely intent -- a "failure" here could just
    # mean the design wasn't 50/50, so we lean on covariate balance (SMD) below
    # as the deeper randomization check. (If you own the experiment and know the
    # intended ratio, SRM is the first thing to run.)
    chi2, p = stats.chisquare([n_t, n_c], f_exp=[n/2, n/2])
    print(f"[srm] chi-square vs 50/50: chi2={chi2:.2f}  p={p:.3f} "
          f"({'OK' if p > 0.001 else 'MISMATCH - investigate'})")

    # Covariate balance: standardized mean difference (SMD) per engineered
    # feature. In a clean randomized experiment most should have |SMD| < 0.1.
    num_feats = [f for f in feature_names if pd.api.types.is_numeric_dtype(df[f])]
    t = df[df[TREATMENT] == 1]
    c = df[df[TREATMENT] == 0]
    smd = {}
    for f in num_feats:
        pooled_sd = np.sqrt((t[f].var() + c[f].var()) / 2)
        if pooled_sd and pooled_sd > 0:
            smd[f] = (t[f].mean() - c[f].mean()) / pooled_sd
    worst = sorted(smd.items(), key=lambda kv: -abs(kv[1]))[:5]
    print(f"[balance] largest |SMD|: {[(f, round(v, 3)) for f, v in worst]}")

    # Base rates, split by arm. The control rate is the baseline purchase rate;
    # the treatment rate is the treated purchase rate; their difference is a
    # preview of the naive ATE that Step 5 estimates properly. A
    # pooled rate over both arms blends baseline + effect + split ratio, so it
    # has no clean interpretation -- we report the arms separately.
    rate_t = df.loc[df[TREATMENT] == 1, OUTCOME].mean()
    rate_c = df.loc[df[TREATMENT] == 0, OUTCOME].mean()
    print(f"[rates] {OUTCOME}: control={rate_c:.4f}  treat={rate_t:.4f}  "
          f"diff={rate_t - rate_c:.4f}")


# ----------------------------------------------------------------------
# Step 5. Average treatment effect (ATE) + variance reduction
# ----------------------------------------------------------------------
def step5_ate(df: pd.DataFrame, feature_names: list[str]) -> dict:
    """Estimate + test the average lift, then turn it into a launch decision.

    Four readouts, each with its own significance test: (1) naive difference in
    rates with a 95% CI, (2) the textbook two-proportion z-test (pooled-variance
    null), (3) CUPED variance reduction using a single pre-experiment covariate
    (longitudinal), and (4) a covariate-adjusted ANCOVA estimate (multiple
    covariates; CUPED is its single-covariate special case) that shrinks the SE
    further. The primary check is ONE pre-specified estimator (ANCOVA) weighing
    statistical AND practical significance -- but that is only NECESSARY: the
    launch decision also requires the population-level guardrails (secondary ATE
    by segment: G1 no segment significantly hurt, G2 champions aren't the main
    source of lift). LAUNCH iff primary passes AND every guardrail passes.
    """
    from statsmodels.stats.proportion import proportions_ztest

    t = df[df[TREATMENT] == 1][OUTCOME]
    c = df[df[TREATMENT] == 0][OUTCOME]

    # Control-arm purchase rate = the baseline the lift is measured against (the
    # North-Star metric, reported here because only the control arm measures it
    # without the campaign's effect mixed in).
    print(f"[baseline] purchase rate (control) = {c.mean():.4f}")

    ate = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    ci = (ate - 1.96 * se, ate + 1.96 * se)
    print(f"[ate] naive  = {ate:.5f}  95% CI [{ci[0]:.5f}, {ci[1]:.5f}]  SE={se:.5f}")

    # Explicit two-proportion z-test -- the standard A/B-test significance call.
    # (Pooled-variance null; for a binary metric this is the canonical readout
    # that the CI above already implies, stated as a hypothesis test.)
    z, pval = proportions_ztest([int(t.sum()), int(c.sum())], [len(t), len(c)])
    print(f"[test] two-proportion z={z:.3f}  p={pval:.4g}  "
          f"({'reject H0' if pval < 0.05 else 'fail to reject H0'} @ alpha=0.05)")

    # Variance reduction #1 -- CUPED (single pre-experiment covariate, longitudinal).
    # X = bought in the 7 days BEFORE the experiment (same window/definition as
    # `target` and as the Step 4 baseline), the cleanest pre-period proxy of the
    # outcome. Y_cuped = Y - theta*(X - E[X]), theta = Cov(Y,X)/Var(X). Under
    # randomization the point estimate is unchanged and the variance drops by
    # ~corr(Y,X)^2. (CUPED is the single-covariate special case of the ANCOVA below.)
    x_pre = (df["recency_days"] <= 7).astype(float)   # missing (no history) -> not bought
    theta = np.cov(df[OUTCOME].astype(float), x_pre, ddof=1)[0, 1] / x_pre.var(ddof=1)
    y_cuped = df[OUTCOME].astype(float) - theta * (x_pre - x_pre.mean())
    tcup = y_cuped[df[TREATMENT] == 1]
    ccup = y_cuped[df[TREATMENT] == 0]
    cuped_ate = tcup.mean() - ccup.mean()
    cuped_se = np.sqrt(tcup.var(ddof=1) / len(tcup) + ccup.var(ddof=1) / len(ccup))
    # Same SE drives both the CI and the test here, so they agree exactly (unlike
    # the pooled-variance two-proportion z above). z = estimate / its own SE.
    cuped_z = cuped_ate / cuped_se
    cuped_p = 2 * stats.norm.sf(abs(cuped_z))
    print(f"[ate] cuped  = {cuped_ate:.5f}  SE={cuped_se:.5f}  z={cuped_z:.3f}  p={cuped_p:.4g}  "
          f"(SE reduced {100*(1-cuped_se/se):.1f}%, theta={theta:.3f})")

    # Variance reduction #2 -- regression adjustment (ANCOVA, multiple covariates;
    # CUPED above is its single-covariate special case). Treatment coefficient =
    # adjusted ATE, usually with an even smaller SE. OLS can't take NaN, so
    # median-fill here. statsmodels gives the t/z-test and CI off the SAME HC1 SE,
    # so for this estimator the p-value and the CI are algebraically equivalent.
    num_feats = [f for f in feature_names if pd.api.types.is_numeric_dtype(df[f])]
    Xdf = df[[TREATMENT] + num_feats].astype(float)
    Xdf = Xdf.fillna(Xdf.median())
    X = sm.add_constant(Xdf)
    model = sm.OLS(df[OUTCOME].astype(float), X).fit(cov_type="HC1")
    adj_ate = model.params[TREATMENT]
    adj_se = model.bse[TREATMENT]
    adj_p = model.pvalues[TREATMENT]
    adj_ci = model.conf_int().loc[TREATMENT].tolist()
    print(f"[ate] adjusted = {adj_ate:.5f}  SE={adj_se:.5f}  p={adj_p:.4g}  "
          f"95% CI [{adj_ci[0]:.5f}, {adj_ci[1]:.5f}]  (SE reduced {100*(1-adj_se/se):.1f}%)")

    # Translate the SE reduction into "how much less sample buys the same
    # precision". Required n for a fixed MDE/power scales with the variance, so
    # the sample the adjusted estimator needs is (adj_se/se)^2 of the naive one.
    sample_ratio = (adj_se / se) ** 2
    print(f"[efficiency] regression adjustment reaches the naive precision with "
          f"~{100*(1-sample_ratio):.0f}% less sample (same MDE & power)")

    # Primary check (NECESSARY, not yet sufficient). Pre-specify ONE primary
    # estimator and judge on it -- here the ANCOVA-adjusted estimate (most
    # covariates -> smallest SE). Deciding on the variance-reduced estimate is
    # the whole point of the adjustment; the naive z-test and CUPED above are
    # reported only as sensitivity checks, NOT cherry-picked for whichever
    # happens to be significant (that would be p-hacking). Statistical
    # significance is necessary but not sufficient -- the effect must also clear
    # a practical bar (the MDE we'd act on), AND pass the guardrails below.
    practical_mde = 0.01   # same +1pp bar used to size the experiment in Step 4
    stat_sig = (adj_p < 0.05) and (adj_ci[0] > 0)
    practical = adj_ate >= practical_mde
    primary_pass = stat_sig and practical
    print(f"[primary] ANCOVA  stat_sig={stat_sig}  "
          f"practical(>=+{practical_mde:.0%})={practical}  ->  "
          f"{'OK (necessary, pending guardrails)' if primary_pass else 'FAIL -> HOLD'}")

    # Secondary (descriptive; does NOT gate launch on its own) + all-send
    # guardrails (population-level do-no-harm; any failure -> HOLD). X5's window
    # only exposes the binary `target`, so the secondary we can compute here is
    # where the lift lands: ATE by RFM segment. (Continuous secondaries -- basket
    # size, frequency, retention -- would come from production logs.)
    seg_rows = []
    for s in ("champions", "at_risk", "mid"):
        m = df["rfm_segment"] == s
        ts = df.loc[m & (df[TREATMENT] == 1), OUTCOME].astype(float)
        cs = df.loc[m & (df[TREATMENT] == 0), OUTCOME].astype(float)
        if len(ts) < 2 or len(cs) < 2:
            continue
        ate_s = ts.mean() - cs.mean()
        se_s = np.sqrt(ts.var(ddof=1) / len(ts) + cs.var(ddof=1) / len(cs))
        seg_rows.append({"segment": s, "n": len(ts) + len(cs), "ate": ate_s,
                         "ci_lo": ate_s - 1.96 * se_s, "ci_hi": ate_s + 1.96 * se_s})
    print("[secondary] ATE by RFM segment (descriptive, does not gate launch):")
    for r in seg_rows:
        print(f"[secondary]   {r['segment']:<10} n={r['n']:>6,}  ate={r['ate']:+.4f}  "
              f"95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")

    # Guardrail G1 (do no harm): no segment is significantly hurt (its whole 95%
    # CI sits below 0).
    harmed = [r["segment"] for r in seg_rows if r["ci_hi"] < 0]
    g1_pass = len(harmed) == 0
    print(f"[guardrail G1] do-no-harm: harmed segments = {harmed or 'none'} -> "
          f"{'PASS' if g1_pass else 'FAIL'}")

    # Guardrail G2 (no loyalty cannibalization): champions (~25% of customers)
    # must not be the main SOURCE of incremental purchases, else we're mostly
    # paying people who'd have bought anyway.
    champ_incr_max = 0.35
    total_incr = sum(r["ate"] * r["n"] for r in seg_rows)
    champ_incr = sum(r["ate"] * r["n"] for r in seg_rows if r["segment"] == "champions")
    champ_share = champ_incr / total_incr if total_incr > 0 else float("nan")
    g2_pass = bool(np.isfinite(champ_share) and champ_share <= champ_incr_max)
    print(f"[guardrail G2] champions incremental share = {champ_share:.2f} "
          f"(> {champ_incr_max:.0%} = cannibalizing loyalty) -> "
          f"{'PASS' if g2_pass else 'FAIL'}")

    # Launch decision: primary is only NECESSARY; LAUNCH also needs every
    # guardrail to pass.
    guardrails_pass = g1_pass and g2_pass
    decision = "LAUNCH" if (primary_pass and guardrails_pass) else "HOLD / iterate"
    print(f"[decision] primary={primary_pass}  guardrails(G1&G2)={guardrails_pass}  "
          f"->  {decision}")
    return {"ate": ate, "se": se, "pval": pval,
            "cuped_ate": cuped_ate, "cuped_se": cuped_se, "cuped_p": cuped_p,
            "adj_ate": adj_ate, "adj_se": adj_se, "adj_p": adj_p,
            "se_reduction": 1 - adj_se / se, "sample_saving": 1 - sample_ratio,
            "stat_sig": stat_sig, "practical": practical, "primary_pass": primary_pass,
            "g1_pass": g1_pass, "g2_pass": g2_pass,
            "champions_incr_share": champ_share, "launch": primary_pass and guardrails_pass}


# ----------------------------------------------------------------------
# Step 6. Heterogeneous treatment effects (uplift / CATE)
# ----------------------------------------------------------------------
def step6_uplift(df: pd.DataFrame, feature_names: list[str]):
    """Train uplift models that score each customer's incremental effect.

    The average effect (Step 5) hides *who* responds. Uplift is inherently
    counterfactual: for any one customer we observe only the treated OR the
    control outcome, never both, so the per-person effect has no label -- it is
    modeled from the treated/control groups and checked with Qini/AUUC (Step 7).

    We fit TWO meta-learners on a shared base estimator so the only variable is
    the uplift approach: S-learner (treatment as a feature) and T-learner
    (separate treated/control models). A controlled experiment (see the notebook)
    showed the meta-learner choice matters less than feature engineering, which
    was the real lever. The base estimator is shallow (depth 3): with a ~62% base
    rate the uplift signal is small, so deeper trees overfit noise.

    Uses LightGBM (LGBMClassifier): fast on the sampled rows and handles the
    NaNs from feature engineering natively (no imputation needed).
    """
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMClassifier
    from sklift.models import SoloModel, TwoModels

    def base():
        return LGBMClassifier(
            max_depth=3, n_estimators=500, learning_rate=0.05,
            num_leaves=15, min_child_samples=30,
            colsample_bytree=0.8,   # column subsampling per tree
            reg_lambda=1.0, random_state=42, verbose=-1,
            # fully reproducible: the uplift signal is small, so multi-threaded
            # float-summation order would otherwise jitter the metrics run-to-run.
            # n_jobs=1 + deterministic makes every run/notebook return identical
            # numbers (fast enough on the 30% sample).
            n_jobs=1, deterministic=True, force_row_wise=True)

    num_feats = [f for f in feature_names if pd.api.types.is_numeric_dtype(df[f])]
    X = df[num_feats]
    y = df[OUTCOME]
    treat = df[TREATMENT]
    seg = df["rfm_segment"]
    (X_tr, X_te, y_tr, y_te,
     t_tr, t_te, _seg_tr, seg_te) = train_test_split(
        X, y, treat, seg, test_size=0.3, random_state=42, stratify=treat)

    # S-learner: one model with treatment as a feature
    s = SoloModel(base())
    s.fit(X_tr, y_tr, t_tr)
    s_scores = s.predict(X_te)

    # T-learner: separate models for treated vs control
    t2 = TwoModels(estimator_trmnt=base(), estimator_ctrl=base(), method="vanilla")
    t2.fit(X_tr, y_tr, t_tr)
    t_scores = t2.predict(X_te)

    return {"X_te": X_te, "y_te": y_te, "t_te": t_te, "seg_te": seg_te,
            "s_scores": s_scores, "t_scores": t_scores}


# ----------------------------------------------------------------------
# Step 7. Evaluate uplift models (Qini / AUUC)
# ----------------------------------------------------------------------
def step7_evaluate(u: dict) -> None:
    """Report standard uplift ranking metrics.

    Qini and AUUC numbers are reported as reference. For this dataset the values
    are small (≈ 0.007–0.008) — consistent with weak heterogeneous effects — so
    the Qini curve is nearly indistinguishable from the random baseline and adds
    little visual information. The primary rank-validity check is step7b_decile.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning,
                            message=".*stable_cumsum.*")
    from sklift.metrics import qini_auc_score, uplift_auc_score

    y, t = u["y_te"], u["t_te"]
    learners = [("S-learner", u["s_scores"]), ("T-learner", u["t_scores"])]
    for name, scores in learners:
        q = qini_auc_score(y, scores, t)
        a = uplift_auc_score(y, scores, t)
        print(f"[eval] {name:<15} qini_auc={q:+.4f}  uplift_auc={a:+.4f}   (small -> weak HTE)")


# ----------------------------------------------------------------------
# Step 7b. Holdout decile validation (rank-validity check)
# ----------------------------------------------------------------------
def step7b_decile(u: dict) -> "pd.DataFrame":
    """Cut holdout by predicted uplift score into 10 equal bands and compare
    treated vs. control conversion rates per band.

    This is the primary rank-validity check: does the model actually rank
    high-lift customers near the top? Lift concentrated in D8-D9 (top 20%)
    means the model is useful for budget-constrained targeting even when global
    HTE is weak.
    """
    val = pd.DataFrame({
        "score": np.asarray(u["t_scores"]),
        "y":     np.asarray(u["y_te"]),
        "t":     np.asarray(u["t_te"]),
    })
    # rank(method="first") breaks ties so pd.qcut produces equal-sized deciles.
    val["decile"] = pd.qcut(val["score"].rank(method="first"),
                            10, labels=[f"D{i}" for i in range(10)])

    def _incr(g: pd.DataFrame) -> pd.Series:
        tr = g.loc[g["t"] == 1, "y"].mean()
        ct = g.loc[g["t"] == 0, "y"].mean()
        return pd.Series({"n": len(g), "treat": tr, "control": ct,
                          "uplift_pp": (tr - ct) * 100})

    dec = val.groupby("decile", observed=True)[["y", "t"]].apply(_incr)
    print(dec.iloc[::-1].round({"treat": 4, "control": 4, "uplift_pp": 2}).to_string())

    top20   = dec.loc[["D9", "D8"], "uplift_pp"].mean()
    overall = (dec["uplift_pp"] * dec["n"]).sum() / dec["n"].sum()
    print(f"\n[ranking check] top-20% (D8-D9) mean uplift {top20:+.2f}pp "
          f"vs overall {overall:+.2f}pp  ->  {top20 / overall:.1f}x")
    print("note: lift is concentrated in the top ~20%; middle deciles are flat/noisy "
          "(consistent with the small Qini) - an honest weak-HTE result")

    order = [f"D{i}" for i in range(10)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(order, dec.loc[order, "uplift_pp"], color="#1f77b4")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("uplift decile (D0 = lowest predicted, D9 = highest)")
    ax.set_ylabel("observed incremental response (pp)")
    ax.set_title("Holdout decile validation: does the score rank real lift?")
    fig.savefig("decile_validation.png", dpi=150, bbox_inches="tight")
    print("[eval] saved decile_validation.png")
    return dec


# ----------------------------------------------------------------------
# Step 8. Targeting policy + business value + guardrail
# ----------------------------------------------------------------------
def step8_policy(u: dict, k_percent: float = 0.30) -> dict:
    """Treat only the top-k% by predicted uplift; quantify value and guardrail it."""
    scores = np.asarray(u["t_scores"])
    y, t = u["y_te"].values, u["t_te"].values
    cutoff = np.quantile(scores, 1 - k_percent)
    top = scores >= cutoff

    def uplift(mask: np.ndarray) -> float:
        yt = y[mask & (t == 1)]
        yc = y[mask & (t == 0)]
        if not len(yt) or not len(yc):
            return float("nan")
        return float(yt.mean() - yc.mean())

    up_top, up_rest = uplift(top), uplift(~top)
    print(f"[policy] targeting top {k_percent:.0%} (cutoff uplift={cutoff:.5f}), "
          f"{int(top.sum())} customers")
    print(f"[policy] measured uplift  targeted={up_top:.5f}  non-targeted={up_rest:.5f}  "
          f"(separation {up_top/up_rest:.1f}x)" if up_rest else "")

    # Business value: incremental responses captured. Scale each group's measured
    # per-customer uplift by its size to get incremental purchases; compare
    # targeting the top-k% against treating EVERYONE (the naive spray policy).
    n = len(scores)
    incr_top = up_top * top.sum()
    incr_all = up_top * top.sum() + up_rest * (~top).sum()
    captured = incr_top / incr_all if incr_all else float("nan")
    print(f"[policy][value] treating top {k_percent:.0%} captures ~{captured:.0%} of the "
          f"incremental responses of treating all, at {k_percent:.0%} of the send cost "
          f"(~{1-k_percent:.0%} spend saved)")

    # Guardrail 1 (do no harm / actually separates): targeted uplift should be
    # positive and clearly above the non-targeted group, else the policy isn't
    # separating responders.
    # Guardrail 2 (no cannibalization of loyalty): check we're not ONLY hitting
    # already-loyal "champions" (who'd likely buy anyway -> low incrementality).
    seg = u["seg_te"].values
    share_champ = float((seg[top] == "champions").mean())
    base_champ = float((seg == "champions").mean())
    ok = share_champ <= base_champ    # not over-represented vs the population
    print(f"[policy][guardrail] champions share in targeted set = {share_champ:.2f} "
          f"vs {base_champ:.2f} population -> "
          f"{'OK (not over-targeting already-loyal customers)' if ok else 'WATCH (may be paying people who would buy anyway)'}")
    return {"k_percent": k_percent, "uplift_top": up_top, "uplift_rest": up_rest,
            "captured": captured, "champions_share": share_champ}


# ----------------------------------------------------------------------
# Step 9. Productionization: drift monitoring + feedback loop
# ----------------------------------------------------------------------
def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and a new distribution.

    Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 significant
    drift (retrain / investigate). Bins are fixed on the reference quantiles so
    the same edges score the new data.
    """
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)  # avoid log(0)
    return float(np.sum((a - e) * np.log(a / e)))


def step9_monitoring(u: dict) -> None:
    """What it takes to RUN this targeting model, not just fit it once.

    Two production realities a one-shot experiment hides:

    1. Drift - the customer mix and the uplift-score distribution move over
       time. We monitor PSI on the score (and key features); a sustained PSI
       above ~0.25 is the signal to re-estimate, because the policy was tuned
       on a stale distribution.

    2. Selective labels / feedback loop - once we ONLY treat the top-k%, future
       logs contain outcomes for the treated tail only. The model then trains
       on data its own policy shaped, which can entrench whoever it already
       favors. Mitigation to state in the memo: hold out a small randomized
       control (epsilon exploration) every cycle so unbiased treated/control
       outcomes keep flowing, and re-validate uplift on that slice.
    """
    # Illustrative PSI: split the test scores in half as "reference" vs
    # "recent" and watch for score drift. Wire to real periods in production.
    scores = np.asarray(u["t_scores"])
    half = len(scores) // 2
    psi = _psi(scores[:half], scores[half:])
    flag = "OK" if psi < 0.1 else ("WATCH" if psi < 0.25 else "DRIFT")
    print(f"[monitor] uplift-score PSI (ref vs recent) = {psi:.3f} -> {flag}")
    print("[monitor] feedback-loop guard: keep an epsilon randomized control "
          "each cycle so unbiased outcomes keep flowing (selective-labels fix).")

    # Epsilon-greedy targeting sketch: how that feedback-loop guard actually
    # runs. Treat this test set as one cycle's candidate pool. A small epsilon
    # slice is randomized (its own mini treated/control A/B) so unbiased outcomes
    # keep flowing across the whole score range; everyone else gets the Step-8
    # top-k% policy. Next cycle re-validates uplift on the explore slice.
    def epsilon_greedy(scores: np.ndarray, epsilon: float = 0.05,
                       k_percent: float = 0.30, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = len(scores)
        action = np.empty(n, dtype=object)
        explore = rng.random(n) < epsilon                 # random slice (ignores score)
        coin = rng.random(n) < 0.5                         # mini A/B inside explore
        action[explore & coin] = "explore_treat"
        action[explore & ~coin] = "explore_control"
        exploit = ~explore                                 # rest: model top-k% only
        cutoff = np.quantile(scores[exploit], 1 - k_percent)
        action[exploit & (scores >= cutoff)] = "exploit_treat"
        action[exploit & (scores < cutoff)] = "exploit_skip"
        return action

    act = epsilon_greedy(scores)
    uniq, cnt = np.unique(act, return_counts=True)
    n_explore = int(np.char.startswith(act.astype(str), "explore").sum())
    print(f"[monitor] epsilon-greedy buckets = {dict(zip(uniq.tolist(), cnt.tolist()))}")
    print(f"[monitor] explore (unbiased randomized control) = {n_explore} "
          f"({n_explore/len(act):.1%}); rest served by the model's top-30% policy.")


def main(sample_frac: float | None = 0.3) -> None:
    # Understand the product
    train, clients, purchases = step1_load(sample_frac=sample_frac)
    df, feats = step2_features_sql(train, clients, purchases)
    df, _retention = step3_product_analytics(df, purchases)
    # Experiment on the product
    step4_validate(df, feats)
    step5_ate(df, feats)
    u = step6_uplift(df, feats)
    step7b_decile(u)
    step8_policy(u)
    step9_monitoring(u)


if __name__ == "__main__":
    main()
