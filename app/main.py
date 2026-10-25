"""Term-deposit outreach ranking API (BAN6800 Module 4). Decision support only."""
import json, os
from typing import Literal, Optional

import joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = joblib.load(os.path.join(ROOT, "models", "stratified_no_macro_gboost.joblib"))
CUTOFFS = json.load(open(os.path.join(ROOT, "models", "age_band_cutoffs.json")))["age_band_cutoffs"]

# Mirrors AGE_BINS / AGE_LABELS in src/bias.py
AGE_BINS = [17, 25, 35, 45, 55, 65, 200]
AGE_LABELS = ["17-24", "25-34", "35-44", "45-54", "55-64", "65+"]

FEATURES = ["job", "marital", "education", "default", "housing", "loan", "contact", "month",
            "day_of_week", "poutcome", "age", "previous", "previously_contacted",
            "days_since_previous_contact"]

Yn = Literal["yes", "no", "unknown"]


class Customer(BaseModel):
    age: int = Field(ge=17, le=100)
    job: Literal["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
                 "self-employed", "services", "student", "technician", "unemployed", "unknown"]
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Literal["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                       "professional.course", "university.degree", "unknown"]
    default: Yn
    housing: Yn
    loan: Yn
    contact: Literal["cellular", "telephone"]
    month: Literal["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    day_of_week: Literal["mon", "tue", "wed", "thu", "fri"]
    poutcome: Literal["failure", "nonexistent", "success"]
    previous: int = Field(ge=0, le=50)
    previously_contacted: Literal[0, 1]
    days_since_previous_contact: Optional[float] = Field(default=None, ge=0, le=60)

    @model_validator(mode="after")
    def history_is_consistent(self):
        if self.previously_contacted != int(self.previous > 0):
            raise ValueError("previously_contacted must be 1 when previous > 0, otherwise 0")
        return self


class Prediction(BaseModel):
    probability: float
    age_band: str
    age_band_cutoff: float
    selected_for_outreach: bool
    model: str
    note: str


app = FastAPI(title="Term-deposit outreach ranking API", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(c: Customer):
    row = pd.DataFrame([c.model_dump()])[FEATURES]
    row["days_since_previous_contact"] = row["days_since_previous_contact"].astype(float)
    p = float(MODEL.predict_proba(row)[:, 1][0])
    band = str(pd.cut([c.age], bins=AGE_BINS, labels=AGE_LABELS, right=False)[0])
    cutoff = float(CUTOFFS[band])
    return Prediction(
        probability=round(p, 4), age_band=band, age_band_cutoff=round(cutoff, 4),
        selected_for_outreach=p >= cutoff, model="term_deposit_ranker v1 (no economic indicators)",
        note="Decision support only. The cutoff approximates the 30% outreach budget; "
             "a relationship manager makes the final decision. Not for credit decisions.")
