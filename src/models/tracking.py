"""Optional MLflow tracking; the core scoring runtime needs no tracking server."""
from pathlib import Path


def log_experiment(name, pipeline, metrics, artifact_path: Path, params, tracking_uri):
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("FinShield / chronological fraud baselines")
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        mlflow.log_params({f"model_{key}": value for key, value in pipeline.named_steps["model"].get_params().items()
                           if value is not None and isinstance(value, (str, bool, int, float))})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))})
        mlflow.set_tags({"dataset": "PaySim", "use_case": "post-transaction", "calibrated": "false"})
        mlflow.log_artifact(str(artifact_path), artifact_path="model")
        return run.info.run_id
