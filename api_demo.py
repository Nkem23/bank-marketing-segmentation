"""API demo for BAN6800 Module 5: sends example customers to the Module 4 scoring API.

Run from the repository root (the folder that contains app/ and models/):

    python api_demo.py                      # runs the API in-process, no server needed
    python api_demo.py --url http://127.0.0.1:8000   # or call an API you started with uvicorn

Illustrative customers only. No real customer data. Decision support only.
"""
import argparse
import json

import httpx

EXAMPLES = {
    "1. Typical record (cellular, June, no earlier contact)": dict(
        age=38, job="admin.", marital="married", education="university.degree", default="no",
        housing="yes", loan="no", contact="cellular", month="jun", day_of_week="tue",
        poutcome="nonexistent", previous=0, previously_contacted=0),
    "2. Earlier successful contact (cellular, September)": dict(
        age=52, job="retired", marital="married", education="high.school", default="no",
        housing="no", loan="no", contact="cellular", month="sep", day_of_week="thu",
        poutcome="success", previous=1, previously_contacted=1, days_since_previous_contact=6),
    "3. First contact by landline (May)": dict(
        age=44, job="blue-collar", marital="married", education="basic.9y", default="unknown",
        housing="yes", loan="no", contact="telephone", month="may", day_of_week="mon",
        poutcome="nonexistent", previous=0, previously_contacted=0),
}
# Same as example 2 but the earlier-contact flag contradicts the contact count
INVALID = {**EXAMPLES["2. Earlier successful contact (cellular, September)"],
           "previously_contacted": 0}


def get_client(url):
    if url:
        return httpx.Client(base_url=url, timeout=30)
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="base URL of a running API, e.g. http://127.0.0.1:8000")
    client = get_client(ap.parse_args().url)

    print("GET /health ->", client.get("/health").json(), "\n")
    for name, payload in EXAMPLES.items():
        r = client.post("/predict", json=payload)
        print(name, "| HTTP", r.status_code)
        print(json.dumps(r.json(), indent=2), "\n")

    r = client.post("/predict", json=INVALID)
    print("4. Inconsistent input (previous = 1 but previously_contacted = 0) | HTTP", r.status_code)
    print("   The API rejects it:", r.json()["detail"][0]["msg"])


if __name__ == "__main__":
    main()
