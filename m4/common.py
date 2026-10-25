import os, sys
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bias import add_age_band

DATA = "data/processed/bank_marketing_clean.csv"
CAT = ["job", "marital", "education", "default", "housing", "loan",
       "contact", "month", "day_of_week", "poutcome"]
NUM = ["age", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
       "euribor3m", "nr.employed", "previously_contacted",
       "days_since_previous_contact"]
FEATURES = CAT + NUM
GROUPS = ["age_band", "marital", "education"]


def load():
    df = pd.read_csv(DATA)
    df["y"] = df["y"].astype(str).str.lower().isin(["yes", "1"]).astype(int)
    df["previously_contacted"] = df["previously_contacted"].astype(int)
    return add_age_band(df)


def split(df, train_frac=0.80):
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def make_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                    ("sc", StandardScaler())])
    return ColumnTransformer([("num", num, NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def capture_at(y, score, frac=0.30):
    y = np.asarray(y)
    n = int(round(len(y) * frac))
    top = np.argsort(-np.asarray(score), kind="stable")[:n]
    return y[top].sum() / y.sum()


from sklearn.model_selection import train_test_split


def split_stratified(df, test_size=0.20, seed=42):
    return train_test_split(df, test_size=test_size, stratify=df["y"], random_state=seed)
