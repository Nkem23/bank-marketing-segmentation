import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-m4")

with mlflow.start_run(run_name="00_targets_declared"):
    mlflow.log_params({
        "target_capture_at_top30pct": 0.70,
        "target_age_band_recall_gap_pp": 10,
        "baseline_contacts_per_subscriber": 22.79,
        "min_subscribers_per_judged_group": 30,
        "disparate_impact_screen": 0.80,
        "split": "time-ordered 80/20",
        "tuning_cv": "TimeSeriesSplit(4), inside train only",
        "excluded_features": "duration, campaign, pdays",
        "rows": 41176,
        "subscribers": 4639,
    })
    mlflow.set_tag("note", "Targets from Modules 1 and 2, set before any model was trained")

runs = mlflow.search_runs(experiment_names=["bank-marketing-m4"])
print(runs[["tags.mlflow.runName", "start_time", "status"]])
