# Data Governance

Project: Customer Segmentation and Ethical AI for Term-Deposit Outreach (BAN6800)

## Data provenance

* Source: UCI Machine Learning Repository, Bank Marketing, dataset 222 (Moro, Rita and Cortez, 2012)
* File used: bank-additional-full.csv, from bank-additional.zip
* Size: 41,188 records and 21 columns
* Access date: 10 October 2026
* Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
* SHA-256 checksum: 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8

## Handling and access

* Raw and processed data are not committed to the public repository (see .gitignore).
* Access is restricted to authorised users, and confidential data stays out of public repositories.
* Data use is minimised to the fields the analysis needs.
* Retention: temporary processed datasets and model outputs are deleted within 30 days of project completion, unless institutional requirements or an approved retention policy require otherwise. The source file and documentation needed for reproducibility are retained where permitted.
* The dataset has no direct customer identifier or gender field. This does not mean every privacy risk is eliminated. A real deployment with identifiable data would need privacy, security and governance controls under the Nigeria Data Protection Act 2023.

## Data quality and leakage controls

* A data dictionary in the reports folder records each column's meaning, missing values (coded unknown) and whether it is available at decision time.
* The duration variable is excluded because it is known only after a call ends.
* The campaign variable is excluded from model inputs and used only for retrospective evaluation.
* Only information available before the outreach decision is eligible as a model input.

## Versioning and change management

* Git records code, model parameters, feature sets, random seeds and evaluation results.
* Performance, fairness and operational impact are reassessed whenever features, segments, models or allocation rules change.

