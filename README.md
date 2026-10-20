# Bank Marketing Segmentation

Customer segmentation and ethical AI for term-deposit outreach. BAN6800 capstone project, Nexford University.

## Project summary

A prototype that segments bank customers, estimates each customer's probability of subscribing to a term deposit, and allocates a limited number of outreach slots from a ranked list. It is designed to be fair, explainable and capacity-aware, and to support relationship-manager decisions, not replace them.

Baseline from the dataset: 22.79 campaign contacts per subscriber (105,754 contacts, 4,640 subscribers, 41,188 records).

## Status

- Module 1: vision document (complete)
- Module 2: project plan, architecture, RAID log, data dictionary, repository and board (in reports/)
- Modules 3 to 5 and the final project: data pipeline, modelling, fairness, dashboard and integration (planned; see the project board)

## Repository structure

    data/          raw and processed data (not committed; see data/README.md)
    notebooks/     01_data_quality, 02_segmentation, 03_prediction, 04_fairness_evaluation
    src/           preprocessing.py, segmentation.py, model_training.py, evaluation.py
    dashboard/     app.py (Streamlit)
    tests/         unit tests
    reports/       architecture and fishbone diagrams, RAID log, data dictionary, board screenshot
    DATA_GOVERNANCE.md, ETHICAL_AI_CHARTER.md, requirements.txt

## Setup

    git clone https://github.com/Nkem23/bank-marketing-segmentation.git
    cd bank-marketing-segmentation
    python -m venv .venv
    source .venv/Scripts/activate
    pip install -r requirements.txt

On Mac or Linux, activate with source .venv/bin/activate. The pinned versions were tested with Python 3.11.9. Then obtain the data as described in data/README.md.

## Governance, ethics and fairness

See DATA_GOVERNANCE.md and ETHICAL_AI_CHARTER.md. Fairness is evaluated by age band, marital status and education. Gender cannot be assessed because the dataset has no gender column.

## Limitations

The data covers one Portuguese bank's retail telemarketing campaigns from 2008 to 2010. Results may not transfer to Nigerian banking, and the analysis is observational, so findings are hypotheses to test, not proven causes.

## Data source and licence

Moro, S., Rita, P., and Cortez, P. (2012). Bank Marketing [Data set]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing. Licensed under CC BY 4.0.
