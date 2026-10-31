# Bank Marketing Segmentation: Data Pipeline and Predictive Model

Project: Customer Segmentation and Ethical AI for Term-Deposit Outreach (BAN6800, Nexford University).

A reproducible analytics project on the UCI Bank Marketing data for term-deposit outreach: a data pipeline (Module 3), a ranking model with explainability, fairness checks and an API (Module 4), a stakeholder dashboard (Module 5), and CI, containers and monitoring added for the final project. Customer segmentation was planned in Module 1 but was not built; the final system ranks customers with one model plus age-band cutoffs. The pipeline loads, cleans, checks, anonymises and audits the data, and writes the tables the later modelling modules use.

## Final project: live system and quick links

This repository holds the full BAN6800 analytics project: the data pipeline (Module 3), the model, explainability and fairness work (Module 4), the stakeholder dashboard (Module 5), and the monitoring, CI and container setup added for the final project.

| Component | Where |
|---|---|
| Live dashboard (results, what-if scoring, fairness, limitations, Monitoring tab) | https://bank-marketing-segmentation-xmqjcsbytc8lkcrabwkzaz.streamlit.app/ |
| Live API (demonstration only): `/health`, `/predict`, `/docs` | https://term-deposit-ranking-api.onrender.com/docs |
| Monitoring dashboard | Monitoring tab of the live dashboard; script `src/monitoring.py`; output `reports/monitoring_summary.json` |
| CI (GitHub Actions) | `.github/workflows/ci.yml` runs the 23 tests on every push |
| Docker Compose | `docker-compose.yml` (API container built from `Dockerfile.api`) |
| Model card | `MODEL_CARD.md` |

The API runs on a free hosting plan that sleeps when idle, so the first request can take about a minute.

### Run the final-project components

```
docker compose up -d --build api                 # API at http://localhost:8000/docs
python api_demo.py --url http://127.0.0.1:8000   # sends example customers to the API
python -m src.monitoring                         # rebuilds reports/monitoring_summary.json
streamlit run dashboard/app.py                   # dashboard at http://localhost:8501
```

### Status and limits

- Monitoring is a baseline built and run on the Module 4 hold-out data, with alert rules for drift (subscription rate shift above 5 points) and age-band recall gaps (above 10 points). No live outcome data exists, so this is not production monitoring.
- Automated retraining is designed but not built.
- The API has no authentication and loads a pickled model. It is a public demonstration endpoint and is not fit for real customer data.
- The live API and dashboard serve the age-band cutoff configuration. The Model Card recommends the refit without age, marital status and education for ethics committee review (see the Module 4 section below).


## What the pipeline does

Orchestrated by Prefect (`src/pipeline.py`), in this order:

1. **Load**: reads the semicolon-separated CSV and checks the schema and row count (`src/preprocessing.py`).
2. **Transform**: splits `pdays` into `previously_contacted` (from `previous`) and `days_since_previous_contact`.
3. **Clean**: removes exact duplicate rows.
4. **Privacy**: scans for direct identifiers and measures re-identification risk with k-anonymity (`src/anonymization.py`).
5. **Validate**: runs 37 Great Expectations checks (`src/validation.py`). The flow stops if validation fails.
6. **Bias**: writes a representation report by age band, marital status and education using Fairlearn (`src/bias.py`).
7. **Output**: writes the cleaned table and a model-inputs table without `duration` and `campaign`.

Every step writes an entry to `logs/audit.log` (`src/audit.py`): user, UTC time, step, rows in and out, file path and SHA-256 checksum. Data values are never logged.

## Setup

Download `bank-additional.zip` from the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing) (CC BY 4.0) and place `bank-additional-full.csv` in `data/raw/`. Raw and processed data are not committed to this repository.

```
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; use .venv/bin/activate on macOS or Linux
pip install -r requirements.txt
```

## Run

```
python -m pytest tests -q          # unit tests
python -m src.pipeline             # run the pipeline
```

Outputs:

- `data/processed/bank_marketing_clean.csv`: full cleaned table, kept for retrospective evaluation
- `data/processed/bank_marketing_model_inputs.csv`: table for model training, without `duration` and `campaign`
- `reports/bias_representation.csv`: group sizes and subscription rates
- `logs/audit.log`: audit log
- `gx_project/`: Great Expectations Data Docs (open `gx_project/gx/uncommitted/data_docs/local_site/index.html`)

## Run in Docker

```
docker build -t bank-marketing-pipeline .
docker run --rm bank-marketing-pipeline python -m pytest tests -q
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W)/data:/app/data" -v "$(pwd -W)/logs:/app/logs" bank-marketing-pipeline
```

On macOS or Linux, use `-v "$(pwd)/data:/app/data"` and `-v "$(pwd)/logs:/app/logs"` and omit `MSYS_NO_PATHCONV=1`. The data is mounted at run time and is never copied into the image. The cleaned output has the same checksum on a Windows PC and inside the container.

## Repository layout

```
src/            pipeline code (preprocessing, validation, audit, bias, anonymization, pipeline) and monitoring.py
app/            FastAPI scoring service (/health, /predict)
m4/             Module 4 modelling scripts, run in numeric order
notebooks/      analysis notebooks (SHAP analysis in 05_shap_analysis.ipynb)
dashboard/      Streamlit dashboard (app.py) and the Monitoring tab (monitoring_tab.py)
models/         trained models (.joblib) and age_band_cutoffs.json
tests/          pytest tests (23)
reports/        bias report, data dictionary, figures, fairness metrics, monitoring_summary.json
.github/workflows/ci.yml   GitHub Actions: installs dependencies and runs the tests
Dockerfile      pipeline container
Dockerfile.api  API container
docker-compose.yml   starts the API container
requirements*.txt    pinned dependencies (pipeline, model, API, all)
MODEL_CARD.md, DATA_GOVERNANCE.md, ETHICAL_AI_CHARTER.md   documentation
```

## Key findings

- 12 exact duplicate rows removed (41,188 to 41,176 rows).
- 4,110 rows have `pdays` = 999 but a recorded earlier contact, so `previously_contacted` is built from `previous`.
- 789 rows (1.92%) sit in groups smaller than 5 on age band, job, marital status and education.
- Two groups have fewer than 100 rows: `education = illiterate` and `marital = unknown`.

## Limitations

- The data covers one Portuguese bank's retail telemarketing campaigns from 2008 to 2010. It is not validated for Nigerian banking.
- The bias report checks group sizes and rates, not model fairness. Model fairness is evaluated in Module 4.
- There is no gender field, so gender cannot be assessed.
- Role-based access control and the retention policy are designs for deployment, not implemented in this prototype.

## AI disclosure

The pipeline (Module 3) and the modelling, explainability, fairness and API code (Module 4) were built with substantial help from Claude (Anthropic). The final-project additions (the CI workflow, the API Dockerfile and docker-compose file, the monitoring script and dashboard Monitoring tab, the README additions and the integrated report) were also drafted with Claude, then run, tested and checked by the author. The code was generated and reviewed in collaboration, and every output was run and checked by the author. See each module's AI Disclosure Form, and the final project's form, for details.

## Module 4: predictive model, explainability and fairness

Goal: rank customer records by their likelihood of subscribing to a term deposit, so that a fixed outreach budget (the top 30% of the ranked list) goes to the most promising customers, and check that the ranking is explainable and treats age, marital status and education groups fairly.

### Headline results (stratified hold-out test set, 8,236 records, 928 subscribers)

| Configuration | Capture in top 30% | ROC-AUC |
|---|---|---|
| Random selection | 29.9% | 0.50 |
| RM-style rule (previous-campaign outcome) | 46.1% | 0.623 |
| Gradient Boosting without economic indicators (proposed scorer) | 67.2% | 0.777 |
| Same model with age-band cutoffs (fairness mitigation) | 62.4% | n/a |
| Gradient Boosting with economic indicators (upper bound only) | 73.5% | 0.810 |

- The 70% capture target from Modules 1 and 2 was **not met**. The 73.5% model depends on economic indicators that mostly identify the campaign period, so it is not used.
- Age-band cutoffs reduced the age bands beyond the 10-point recall gap from 5 to 0, at a cost of 4.8 points of capture. Education gaps were not mitigated (3 groups beyond the gap before and after).
- Refitting without age, marital status and education cost 0.5 points of capture and reduced the breaches from 5 to 2. This is the option recommended in the Model Card, for review by an ethics committee.
- Under a time-ordered split, subscription rates rose from 6.4% in training to 30.8% in testing and performance fell (AUC 0.65 to 0.70), so drift monitoring is essential.

### Where to find each deliverable

| Deliverable | Location |
|---|---|
| Model Card | `MODEL_CARD.md` |
| Fairness Metrics Report | `reports/FAIRNESS_REPORT.md` (tables in `reports/fairness_metrics.csv` and `reports/fairness_mitigation.csv`) |
| SHAP analysis notebook | `notebooks/05_shap_analysis.ipynb` |
| Figures (ROC, gain, confusion matrix, SHAP, API, MLflow) | `reports/figures/` |
| MLflow run log (25 runs with timestamps) | `reports/mlflow_run_log.csv`; screenshots `fig9` to `fig11` |
| Serialized models | `models/*.joblib`; registered in MLflow as `term_deposit_ranker` v1 (alias `production`) |
| FastAPI app with `/predict` | `app/main.py`; tests in `tests/test_predict_api.py` |
| Modelling scripts | `m4/` (run in numeric order) |

### Reproduce

Needs `data/processed/bank_marketing_clean.csv` from the Module 3 pipeline. Tested on Windows with Python 3.11.

```
python -m venv .venv-m4
source .venv-m4/Scripts/activate
pip install -r requirements-model.txt
python m4/00_declare_targets.py      # logs the targets before any model is trained
# then run the other scripts in m4/ in numeric order
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn app.main:app --port 8000     # API docs at http://127.0.0.1:8000/docs
python tests/test_predict_api.py     # API tests
```

The MLflow database (`mlflow.db`) and row-level scores (`m4/outputs/`) are not committed because they hold record-level data. Running the scripts recreates them.

### Limitations

- One Portuguese bank, retail telemarketing, 2008 to 2010. Not validated for Nigerian banking or for corporate relationship banking.
- The data has no customer identifier and no gender field.
- Fairness estimates for small groups are imprecise (50 to 346 subscribers per age band in the test set).
- The API has no authentication, applies a per-customer cutoff and loads a pickled model, so it is a demonstration and not a production service.
- Using age to set cutoffs is a protected attribute used at decision time and needs compliance review.

