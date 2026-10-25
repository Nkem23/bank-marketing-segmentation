import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

HIGH = {"age": 59, "job": "admin.", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular", "month": "sep",
        "day_of_week": "tue", "poutcome": "success", "previous": 1, "previously_contacted": 1,
        "days_since_previous_contact": 6}
LOW = {"age": 37, "job": "blue-collar", "marital": "divorced", "education": "basic.4y",
       "default": "no", "housing": "yes", "loan": "no", "contact": "telephone", "month": "may",
       "day_of_week": "mon", "poutcome": "nonexistent", "previous": 0, "previously_contacted": 0,
       "days_since_previous_contact": None}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_valid_prediction():
    r = client.post("/predict", json=HIGH)
    assert r.status_code == 200
    b = r.json()
    assert 0 <= b["probability"] <= 1 and b["age_band"] == "55-64"


def test_rejects_bad_month():
    assert client.post("/predict", json={**HIGH, "month": "xyz"}).status_code == 422


def test_rejects_age_out_of_range():
    assert client.post("/predict", json={**HIGH, "age": 5}).status_code == 422


def test_rejects_inconsistent_history():
    assert client.post("/predict", json={**HIGH, "previous": 0}).status_code == 422


if __name__ == "__main__":
    for t in [test_health, test_valid_prediction, test_rejects_bad_month,
              test_rejects_age_out_of_range, test_rejects_inconsistent_history]:
        t(); print("PASS", t.__name__)
    for name, c in [("HIGH", HIGH), ("LOW", LOW)]:
        r = client.post("/predict", json=c)
        print(name, r.status_code, {k: v for k, v in r.json().items() if k != "note"})
