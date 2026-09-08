# FinSight

### Financial Offer Experimentation & Risk-Aware Customer Intelligence

FinSight is an end-to-end data science portfolio project that answers a question
every financial products team eventually asks: **"We ran a personalized offer
experiment — did it work, for whom, and should we roll it out?"**

It walks the full path from raw data to a deployed decision-support tool:

**SQL → EDA → Statistics → A/B Testing → Segmentation → ML → Explainability → Business Impact → Deployment**

---

## Problem

A financial institution wants to know whether replacing its standard offer with a
**personalized financial offer** improves customer conversion — and, if so, whether
rolling it out is a good idea once financial risk and cost are taken into account.

## Why this problem matters

Most ML portfolio projects stop at "here's my model's accuracy." Real business
decisions require more:

- Was the observed lift **real**, or noise? (statistical inference)
- Is the lift **big enough to matter**? (practical significance)
- **Who** should get the offer? (segmentation + predictive targeting)
- What if a likely-to-convert customer is also **high risk**? (risk-aware targeting)
- What's the actual **dollar impact** of a rollout decision? (business simulation)

FinSight is built around this full chain, not just the model in the middle.

## Business Questions

1. Did the personalized offer actually improve conversion?
2. Is the observed improvement statistically significant?
3. Which customer segments benefited most?
4. Can we predict which customers are likely to convert?
5. Can we identify likely converters while maintaining acceptable financial risk?
6. What is the expected financial impact of targeting different customer groups?
7. Should the company roll out the treatment, and to whom?

---

## Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(Moro, S., Cortez, P., & Rita, P., 2014, *Decision Support Systems*). 41,188 rows,
20 real customer/campaign attributes (age, job, education, loan history, prior
campaign outcomes, macroeconomic indicators, etc.) from a Portuguese bank's phone
marketing campaigns.

**⚠️ Data honesty disclosure — please read:**

> The public UCI dataset does **not** contain a real randomized treatment/control
> experiment, and it does not contain income, credit score, or risk fields (real
> financial institutions do not publish these). FinSight therefore:
>
> 1. Uses the real UCI attributes for realistic demographic/behavioral customer data.
> 2. **Derives** synthetic-but-plausible financial attributes (`income`, `credit_score`,
>    `tenure_years`, `credit_utilization`, `financial_exposure`, `risk_score`) from the
>    real fields via documented, reproducible formulas (see
>    `src/feature_engineering.py::add_synthetic_financial_attributes`).
> 3. **Randomly assigns** customers to `control` / `treatment` with a fixed seed
>    (`src/feature_engineering.py::assign_experiment_groups`).
> 4. **Simulates** the `conversion` outcome via a transparent logistic mechanism —
>    a documented baseline propensity formula plus a documented treatment uplift with
>    a small number of named heterogeneous-effect segments — all specified in
>    `config.yaml` under `experiment.base_propensity` / `experiment.treatment_effect`.
>
> **The experimentation layer is simulated for portfolio/educational purposes
> because public financial datasets generally do not contain proprietary
> randomized campaign assignments.** No result in this project should be read as
> a real-world finding about any actual bank, product, or customer population.
>
> The dataset's own real historical outcome column (`historical_campaign_outcome`,
> originally `y`) is retained only for provenance/EDA context and is **never**
> used to generate or model the synthetic `conversion` outcome.

The raw CSV (~5.8MB) is committed to the repo for zero-friction setup. To
re-download it from source: `python scripts/download_data.py`.

**Data dictionary** (key columns): see `src/data_ingestion.py` (raw schema),
`src/feature_engineering.py` (derived/synthetic columns), and the printed
`df.info()` in `notebooks/01_eda.ipynb`.

---

## Experiment Design

| | |
|---|---|
| **Population** | All 41,188 customers |
| **Eligibility rule** | Customers currently in credit default (`default == 'yes'`) are excluded (`experiment_group = 'ineligible'`) — a personalized offer would not be extended to them in practice |
| **Analysis population** | 41,185 eligible customers → 20,753 control / 20,432 treatment |
| **Assignment** | Uniform random, `numpy.random.default_rng(seed=42)`, ~50/50 split |
| **Primary metric** | `conversion` (binary) |
| **Control** | Standard offer (baseline propensity model only) |
| **Treatment** | Personalized offer (baseline propensity + documented uplift) |

**Balance check** (pre-treatment covariates only — income, credit score, tenure,
credit utilization, age): all five variables show standardized mean differences
well under the common `|SMD| < 0.10` threshold, i.e., randomization produced
comparable groups. Full table in `src/statistics.check_group_balance` / dashboard Page 2.

---

## Statistical Methodology

Two-proportion z-test (`statsmodels.stats.proportion.proportions_ztest`) comparing
treatment vs. control conversion rates, with:

- Wald 95% confidence interval on the absolute lift
- Cohen's h effect size
- A **separate** practical/business-significance check (minimum relative lift the
  business considers meaningful — configurable, default 5%) — statistical
  significance and practical significance are reported and interpreted independently
- Power analysis (`statsmodels.stats.power.NormalIndPower`): required sample size
  for a range of minimum detectable effects, plus achieved power at the realized
  sample size and effect
- Explicit exploratory-vs-confirmatory framing for segment-level results (see below)

## Analytical Findings

On the full analysis population (41,185 customers):

| Metric | Value |
|---|---|
| Control conversion rate | **10.50%** (2,180 / 20,753) |
| Treatment conversion rate | **14.96%** (3,056 / 20,432) |
| Absolute lift | **+4.45 pp** |
| Relative lift | **+42.4%** |
| 95% CI on absolute lift | **[+3.81 pp, +5.10 pp]** |
| z-statistic | 13.56 |
| p-value | **6.7 × 10⁻⁴²** |
| Statistically significant? | ✅ Yes (α = 0.05) |
| Practically significant? | ✅ Yes (clears the 5% relative-lift bar by a wide margin) |

**Segment-level (exploratory — not individually powered):** lift is largest for
**high-engagement** customers (prior campaign success/contact) and **mid-income**
customers, and smallest for low-engagement, low-income customers — see
`src/segmentation.py`, `sql/portfolio_analysis.sql` (`segment_lift` query), and
dashboard Page 1/3 for the full breakdown. The README, dashboard, and notebook all
explicitly flag that segment cuts increase false-positive risk and should be
treated as hypothesis-generating, not confirmatory.

---

## ML Approach

**Leakage safeguards:**
- `duration` (call length, only known after a call completes) is dropped immediately
  — a well-documented leakage trap in this dataset.
- The dataset's real historical outcome (`historical_campaign_outcome`) is excluded
  from all modeling features.
- Train/validation/test split (60/20/20, stratified, seed=42) happens **before**
  any preprocessing is fit — no information from val/test leaks into training.
- Class imbalance (~13% base conversion rate) handled via `class_weight="balanced"`
  (Logistic Regression) and `scale_pos_weight` (XGBoost), not by oversampling before
  the split.

**Two models**, both inside an `sklearn.Pipeline` (shared `ColumnTransformer`:
standard-scaled numerics + one-hot categoricals):

1. **Logistic Regression** — interpretable baseline/benchmark.
2. **XGBoost** — nonlinear model, tuned for realistic (not maximal) depth/estimators
   to avoid overfitting a ~41K-row dataset.

## Model Results

Held-out test set (8,237 customers), threshold = 0.5:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | **0.686** | **0.269** | 0.206 | 0.623 | 0.310 |
| XGBoost | 0.674 | 0.265 | 0.204 | 0.589 | 0.303 |

**Selected final model: Logistic Regression**, chosen by highest **PR-AUC** on the
test set (the business objective — ranking customers by conversion likelihood under
class imbalance — not raw accuracy).

**Honest finding, not a tuning failure:** Logistic Regression matches or slightly
outperforms XGBoost here because FinSight's synthetic conversion outcome is
generated by a **logistic** mechanism (see `config.yaml`) — so a linear model is
close to the true functional form by construction. This is flagged explicitly
rather than hidden: it's a good illustration of *why* model selection should be
driven by held-out metrics and the business objective, not by defaulting to the
more complex algorithm. (On real-world data with genuine nonlinear/interaction
effects, XGBoost would be expected to pull ahead — the pipeline supports swapping
which model is "final" purely by changing the PR-AUC comparison, no code changes.)

ROC / PR / calibration curves for both models: `reports/figures/*.html` (generated
by `python -m src.evaluate`) and interactively on dashboard Page 3.

## Explainability

SHAP (`TreeExplainer`) on the XGBoost model provides:

- **Global explanation** — mean |SHAP value| ranking across a sample of test
  customers (top drivers: `credit_score`, `tenure_years`, `income`, `risk_score`,
  `experiment_group`, matching the documented simulation mechanism — a good sanity
  check that the model recovered the true generative signal).
- **Individual explanation** — for any selected customer: predicted probability,
  top positive and negative SHAP contributors, always phrased as *association with
  the model's prediction*, never as proof of causation.

## Business Simulator

A transparent, config-driven expected-value framework — the business should not
simply target every high-probability customer without weighing risk and cost:

```
Expected Value = P(conversion) × financial_value
                 − intervention_cost
                 − P(conversion) × risk_cost(risk_level)
```

Risk cost is only incurred *if* the customer converts (e.g., takes on a product
that can later default), so it's weighted by predicted probability. All dollar
assumptions (financial value per conversion, intervention cost, risk cost by risk
band) are configurable in `config.yaml` and live-adjustable in the dashboard —
nothing is hardcoded, and every simulator output recalculates on the fly.

**Example run** (default assumptions, held-out test set, Logistic Regression
predictions, min P(conversion) ≥ 15%, max risk = Medium): 5,482 customers eligible,
100% targeted, ~2,829 expected conversions, ~$65.8K intervention cost, ~$410.2K
expected gross value, ~$57.7K expected risk cost → **~$286.7K expected net impact**.
Adjust any assumption on dashboard Page 4 to see this recompute live.

**Rollout recommendation logic** (`src/business_simulator.recommend_rollout`):
combines statistical significance, the practical-lift bar, and a positive expected
net impact into a transparent **ROLLOUT / HOLD-REVIEW / DO NOT ROLLOUT** verdict.
This is explicitly framed in the app as **decision support, not an automated
financial decision** — real rollout calls should involve human review.

---

## Architecture

```
Raw CSV (UCI Bank Marketing)
        │
        ▼
 data_ingestion.py  ──►  data_validation.py (raw)
        │
        ▼
 feature_engineering.py
   ├─ synthetic financial attributes
   ├─ experiment assignment (control/treatment)
   └─ simulated conversion outcome
        │
        ▼
 data_validation.py (processed)  ──►  data/processed/*.parquet, *.csv
        │
        ├──► sql_analytics.py (DuckDB)         ──► sql/portfolio_analysis.sql
        ├──► statistics.py (A/B test, power)
        ├──► segmentation.py
        ├──► train.py (LogReg + XGBoost, MLflow) ──► models/*.joblib
        ├──► evaluate.py (ROC/PR/calibration)     ──► reports/
        ├──► explain.py (SHAP)
        └──► business_simulator.py (targeting, EV, recommendation)
                     │
                     ▼
            Streamlit dashboard (app/)
```

## Tech Stack

Python 3.12 · pandas/NumPy · DuckDB (SQL) · SciPy/statsmodels (statistics) ·
scikit-learn + XGBoost (ML) · SHAP (explainability) · Plotly (viz) · MLflow
(experiment tracking, local SQLite backend — no external server needed) ·
Streamlit (app) · Docker (deployment)

## Project Structure

```
finSight/
├── app/                     # Streamlit application
│   ├── streamlit_app.py     # entry point + navigation
│   ├── views/                 page_1..4_*.py — one module per dashboard page
│   ├── components/             charts.py, metrics.py, tables.py
│   └── utils/                  app_utils.py — cached data/model loading
├── src/                     # Core analytical pipeline (importable, tested)
│   ├── config.py, pipeline.py
│   ├── data_ingestion.py, data_validation.py, feature_engineering.py
│   ├── sql_analytics.py, statistics.py, segmentation.py
│   ├── train.py, evaluate.py, explain.py, business_simulator.py
├── sql/portfolio_analysis.sql
├── notebooks/01_eda.ipynb
├── data/{raw,processed}/
├── models/                  # trained pipelines (.joblib) — gitignored, regenerated
├── reports/{figures,metrics}/
├── tests/                   # pytest — 38+ tests across data/stats/model/business logic
├── scripts/                 # download_data.py, build_eda_notebook.py
├── config.yaml              # every assumption/parameter, single source of truth
├── requirements.txt, Dockerfile, .dockerignore, Makefile
```

---

## Local Setup

```bash
git clone <your-repo-url>
cd finSight

# 1. Install dependencies
make setup                     # or: pip install -r requirements.txt

# 2. (Optional) re-download the raw dataset from source
python scripts/download_data.py   # the CSV is already committed, so this is optional

# 3. Run the full pipeline: data → train → evaluate
make pipeline

# 4. Run the tests
make test

# 5. Launch the dashboard
make run                        # or: streamlit run app/streamlit_app.py
```

Then open **http://localhost:8501**.

Individual steps, if you want to run them one at a time:
```bash
python -m src.pipeline    # data ingestion -> validation -> feature engineering
python -m src.train       # trains + logs both models to local MLflow (SQLite)
python -m src.evaluate    # test-set metrics + ROC/PR/calibration plots
python -m src.sql_analytics   # print all named SQL query results
python -m src.explain     # print global SHAP feature importance
```

## Docker Setup

The Docker image builds the **entire pipeline at build time** (data → train →
evaluate), so the container starts instantly with everything the dashboard needs
already on disk — no training happens at runtime.

```bash
make docker-build     # or: docker build -t finsight:latest .
make docker-run        # or: docker run --rm -p 8501:8501 finsight:latest
```

Then open **http://localhost:8501**.

## Streamlit Deployment

Deployable directly from GitHub via **Streamlit Community Cloud**:

1. Push this repository to GitHub (the committed raw CSV means no separate data
   step is needed).
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at your repo,
   branch, and set the main file path to `app/streamlit_app.py`.
3. Streamlit Cloud installs `requirements.txt` automatically. Because `models/` and
   `data/processed/` are gitignored, add a one-time build step or a `packages.txt`/
   startup hook that runs `python -m src.pipeline && python -m src.train && python -m src.evaluate`
   before the app boots — or simply commit the generated `data/processed/` and
   `models/` artifacts for the simplest possible Cloud deploy (they're small: a
   few MB of parquet/joblib files).

## Optional AWS Deployment

To run the same Docker image on an EC2 instance:

```bash
# On the EC2 instance (Amazon Linux 2023 / Ubuntu, with Docker installed):
git clone <your-repo-url> && cd finSight
docker build -t finsight:latest .
docker run -d --restart unless-stopped -p 80:8501 finsight:latest
```

Open a security group inbound rule for port 80 (or 8501, and adjust the `-p` flag
accordingly) from your IP or `0.0.0.0/0`. For S3-based data storage instead of the
committed CSV, swap `src/data_ingestion.py`'s `load_raw_data` to read via `boto3`
or `pandas.read_csv("s3://...")` (requires `s3fs`) — not required for the default
local-file setup. AWS is **not** required for local development or the primary
Streamlit Cloud deployment path.

---

## Limitations

- **Simulated experiment layer.** As disclosed above, `experiment_group` and
  `conversion` are synthetically generated, not a real randomized trial. All
  quantitative findings are illustrative of the *methodology*, not claims about
  any real bank or customer population.
- **Synthetic financial attributes.** `income`, `credit_score`, `tenure_years`,
  `credit_utilization`, `financial_exposure`, and `risk_score` are derived, not
  observed — they are internally consistent but not real financial records.
- **Segment-level results are exploratory.** They were not each independently
  powered and should not be treated as confirmatory without a dedicated follow-up
  test.
- **Business dollar assumptions are illustrative**, not fitted from real cost/value
  data — they're fully configurable in `config.yaml` precisely so they can be
  swapped for real assumptions in a production setting.
- **No causal claims from SHAP.** SHAP values describe association with model
  predictions, not causal effects.

## Responsible Use

This project uses public/simulated data for educational and portfolio purposes and
is **not intended for real-world credit or financial decision-making**. The
rollout recommendation logic is explicitly framed as decision *support*; it is not
an automated financial decision system, and no output here should be used to make
actual lending, credit, or marketing decisions about real people.

## Future Improvements

- Replace the hand-specified simulation mechanism with a fitted uplift model
  (e.g., two-model or causal-forest approach) if/when a real experiment dataset
  becomes available.
- Add a proper train/serve feature store if this were to move beyond a portfolio
  project.
- Expand the business simulator to support multi-touch/multi-period customer
  value instead of a single-conversion expected-value framework.
- Add model monitoring / drift detection for a production deployment.

---

## Screenshots

Screenshots aren't checked into the repo to keep it lightweight, but every view
below is one click away after `make run`:

- **Executive Overview** (Page 1) — headline lift, p-value, CI, top segments, rollout recommendation
- **Experiment & Statistics** (Page 2) — hypothesis, full A/B test output, balance diagnostics, power analysis
- **Customer & ML Insights** (Page 3) — per-customer prediction + SHAP waterfall, global feature importance, model comparison
- **Targeting & Business Simulator** (Page 4) — live-adjustable targeting assumptions and dollar-impact recommendation

---

**Author's note / interview story:** I wanted to build a project that went beyond
model accuracy. I designed FinSight around a financial product experimentation
problem: SQL and EDA to understand customer behavior, rigorous A/B testing to
determine whether an observed lift was real and meaningful, segmentation to find
where the treatment worked best, two calibrated ML models to predict individual
outcomes, SHAP for both global and individual explainability, and a risk-aware
targeting simulator that turns all of the above into an actual dollar-impact
rollout recommendation — deployed via Streamlit and Docker.
