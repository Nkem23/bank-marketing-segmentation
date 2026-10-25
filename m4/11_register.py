import joblib, mlflow, mlflow.sklearn
from mlflow import MlflowClient
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-m4")
NAME = "term_deposit_ranker"
pipe = joblib.load("models/stratified_no_macro_gboost.joblib")
with mlflow.start_run(run_name="90_register_final_model"):
    mlflow.set_tags({"model": "gradient boosting without economic indicators",
                     "fairness_config": "age-band cutoffs (models/age_band_cutoffs.json)"})
    mlflow.log_artifact("models/age_band_cutoffs.json")
    mlflow.sklearn.log_model(pipe, name="model", registered_model_name=NAME,
                             serialization_format="cloudpickle")
c = MlflowClient()
v = str(max(int(m.version) for m in c.search_model_versions(f"name='{NAME}'")))
c.set_registered_model_alias(NAME, "production", v)
c.set_model_version_tag(NAME, v, "capture30_unmitigated", "0.672")
c.set_model_version_tag(NAME, v, "capture30_with_age_band_cutoffs", "0.624")
c.set_model_version_tag(NAME, v, "target_70pct_met", "no")
print("registered", NAME, "version", v, "with alias production")
