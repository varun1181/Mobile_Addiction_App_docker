"""Stage 5 - evaluate every model on the test set, log everything to MLflow, write reports."""
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("model_evaluation")


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def evaluate(model, X_train, y_train, X_test, y_test) -> dict:
    pred_train = model.predict(X_train)
    t0 = time.time()
    pred_test = model.predict(X_test)
    test_time = time.time() - t0
    train_r2, test_r2 = r2_score(y_train, pred_train), r2_score(y_test, pred_test)
    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "overfit_gap": train_r2 - test_r2,
        "mae": mean_absolute_error(y_test, pred_test),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "test_time_s": test_time,
    }


def make_chart(comparison: pd.DataFrame, path: Path) -> None:
    df = comparison.set_index("Model")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    df[["Training R2", "Testing R2"]].plot(kind="bar", ax=axes[0], title="R2 (train vs test)")
    df[["MAE", "RMSE"]].plot(kind="bar", ax=axes[1], title="Error (lower is better)")
    df["Overfit gap"].plot(kind="bar", ax=axes[2], color="tab:red", title="Overfit gap")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    params = load_params()
    target = params["base"]["target"]
    mcfg = params["model_evaluation"]

    data_dir = ROOT / "data" / "processed"
    train = pd.read_csv(data_dir / "train_features.csv")
    test = pd.read_csv(data_dir / "test_features.csv")
    X_train, y_train = train.drop(columns=target), train[target]
    X_test, y_test = test.drop(columns=target), test[target]

    models_dir = ROOT / "models"
    with open(models_dir / "model_info.json") as f:
        meta = json.load(f)
    best_name, info = meta["best_model"], meta["models"]

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(mcfg["mlflow_tracking_uri"])
    mlflow.set_experiment(mcfg["experiment_name"])
    session = time.strftime("%Y%m%d_%H%M%S")      # groups the runs of one execution

    rows, run_ids = [], {}
    # One TOP-LEVEL run per model so every chart in the MLflow UI can plot all of them
    for name, details in info.items():
        model = joblib.load(models_dir / "candidates" / f"{details['key']}.pkl")
        metrics = evaluate(model, X_train, y_train, X_test, y_test)
        metrics.update({
            "cv_r2": details["cv_r2"],
            "search_time_s": details["search_time_s"],
            "model_size_kb": details["model_size_kb"],
            "model_id": details["model_id"],
        })
        rows.append({
            "Model": name,
            "Training R2": metrics["train_r2"],
            "Testing R2": metrics["test_r2"],
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "Overfit gap": metrics["overfit_gap"],
            "CV R2": metrics["cv_r2"],
        })

        with mlflow.start_run(run_name=name) as run:
            run_ids[name] = run.info.run_id
            mlflow.set_tags({
                "model_type": name, "task": "regression", "tuner": "optuna",
                "run_type": "final_model", "session": session,
                "is_best": str(name == best_name),
            })
            mlflow.log_param("model_name", name)
            mlflow.log_params(details["best_params"])
            mlflow.log_param("n_trials", details["n_trials"])
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

            # Optuna optimisation history (chart "trial_cv_r2" in the MLflow UI)
            best_so_far = -np.inf
            for step, value in enumerate(details["trial_values"]):
                if value is None:
                    continue
                best_so_far = max(best_so_far, value)
                mlflow.log_metric("trial_cv_r2", value, step=step)
                mlflow.log_metric("best_cv_r2_so_far", best_so_far, step=step)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=infer_signature(X_train, model.predict(X_train)),
                input_example=X_train.head(3),
                serialization_format="cloudpickle",
            )
        log.info("%s: test R2 %.4f | RMSE %.4f | gap %.4f", name, metrics["test_r2"], metrics["rmse"], metrics["overfit_gap"])

    comparison = pd.DataFrame(rows)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    make_chart(comparison, reports_dir / "model_comparison_chart.png")

    best_row = comparison.set_index("Model").loc[best_name]
    best_by_test = comparison.loc[comparison["Testing R2"].idxmax(), "Model"]
    if best_by_test != best_name:
        log.warning("Best by CV R2 is %s but best by test R2 is %s", best_name, best_by_test)

    with open(reports_dir / "metrics.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "test_r2": float(best_row["Testing R2"]),
            "mae": float(best_row["MAE"]),
            "rmse": float(best_row["RMSE"]),
            "cv_r2": float(best_row["CV R2"]),
            "overfit_gap": float(best_row["Overfit gap"]),
        }, f, indent=2)

    # separate summary run: comparison table, chart, schema and model info
    with mlflow.start_run(run_name=f"Summary_{session}"):
        mlflow.set_tags({"run_type": "summary", "session": session,
                         "best_model": best_name, "best_run_id": run_ids[best_name]})
        mlflow.log_artifact(str(reports_dir / "model_comparison.csv"))
        mlflow.log_artifact(str(reports_dir / "model_comparison_chart.png"))
        mlflow.log_artifact(str(models_dir / "feature_schema.json"))
        mlflow.log_artifact(str(models_dir / "model_info.json"))

    log.info("Best model: %s | run id %s | session %s", best_name, run_ids[best_name], session)
    log.info('MLflow UI filter: tags.session = "%s" and tags.run_type = "final_model"', session)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Model evaluation failed")
        sys.exit(1)
