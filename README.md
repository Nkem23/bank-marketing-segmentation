# Bank Marketing Segmentation: Data Pipeline

Project: Customer Segmentation and Ethical AI for Term-Deposit Outreach (BAN6800, Nexford University).

A reproducible data pipeline that prepares the UCI Bank Marketing data for customer segmentation and term-deposit outreach scoring. It loads, cleans, checks, anonymises and audits the data, and writes the tables the later modelling modules use.

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
src/            pipeline code (preprocessing, validation, audit, bias, anonymization, pipeline)
tests/          pytest unit tests
reports/        bias report and data dictionary
Dockerfile      container definition
requirements.txt, requirements-pipeline.txt   pinned dependencies
DATA_GOVERNANCE.md, ETHICAL_AI_CHARTER.md     governance and ethics documents
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

This pipeline was built with substantial help from Claude (Anthropic). The code was generated and reviewed in collaboration, and every output was run and checked by the author. See the module's AI Disclosure Form for details.