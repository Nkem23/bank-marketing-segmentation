# Data

Raw and processed data are not committed to this repository (see DATA_GOVERNANCE.md and .gitignore).

## How to obtain the data

1. Download the Bank Marketing dataset (UCI dataset 222): https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing
2. Extract bank-additional-full.csv from bank-additional.zip.
3. Place the file in data/raw/.

## What to expect

- Semicolon-separated file with 41,188 rows and 21 columns (20 inputs and the target y)
- Target y: 4,640 yes and 36,548 no
- SHA-256 checksum: 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8
- Licence: CC BY 4.0. Cite Moro, Rita and Cortez (2012).

data/processed/ holds generated files and is also excluded from Git. Column definitions are in reports/Data Dictionary.xlsx.
