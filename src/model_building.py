"""Stage 4 - tune five regressors with Optuna, save every model and pick the best by CV R2."""
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import optuna
import pandas as pd
import yaml
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("model_building")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------
# One builder per algorithm: takes an Optuna trial and returns an estimator
# ----------------------------------------------------------------------------
def build_ridge(trial):
    return Ridge(
        alpha=trial.suggest_float("alpha", 1e-3, 1e3, log=True),
        solver=trial.suggest_categorical("solver", ["auto", "svd", "cholesky"]),
        random_state=42,
    )


def build_decision_tree(trial):
    return DecisionTreeRegressor(
        max_depth=trial.suggest_int("max_depth", 2, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 50),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        criterion=trial.suggest_categorical("criterion", ["squared_error", "absolute_error"]),
        random_state=42,
    )


def build_random_forest(trial):
    return RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
        random_state=42,
    )


def build_gradient_boosting(trial):
    return GradientBoostingRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 5),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        random_state=42,
    )


def build_svr(trial):
    return SVR(
        kernel="rbf",
        C=trial.suggest_float("C", 0.5, 10, log=True),
        gamma=trial.suggest_float("gamma", 1e-3, 1.0, log=True),
        epsilon=trial.suggest_float("epsilon", 0.01, 0.3),
    )


MODELS = {
    "ridge": ("Linear Regression (Ridge)", build_ridge),
    "decision_tree": ("Decision Tree", build_decision_tree),
    "random_forest": ("Random Forest", build_random_forest),
    "gradient_boosting": ("Gradient Boosting", build_gradient_boosting),
    "svr": ("SVR", build_svr),
}


def make_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num_pipeline, num_cols), ("ohe", onehot_pipeline, cat_cols)],
        remainder="drop",
    )


def make_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])


def tune(name, build_fn, preprocessor, X, y, params) -> tuple:
    """Run one Optuna study, then fit the best pipeline on all training data."""
    cfg = params["model_building"]
    seed = params["base"]["random_state"]
    cv = KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)

    def objective(trial):
        pipe = make_pipeline(preprocessor, build_fn(trial))
        return cross_val_score(pipe, X, y, scoring="r2", cv=cv, n_jobs=cfg["n_jobs"]).mean()

    log.info("Optimizing %s (%s trials)", name, cfg["n_trials"])
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed), study_name=name)
    t0 = time.time()
    study.optimize(objective, n_trials=cfg["n_trials"])
    search_time = time.time() - t0

    best_pipe = make_pipeline(preprocessor, build_fn(optuna.trial.FixedTrial(study.best_params)))
    best_pipe.fit(X, y)
    log.info("%s: best CV R2 = %.4f (%.1fs)", name, study.best_value, search_time)
    return best_pipe, study, search_time


def _py(v):
    return v.item() if hasattr(v, "item") else v


def build_schema(X: pd.DataFrame, y: pd.Series, cat_cols: list, params: dict, best_name: str) -> dict:
    """Input description used by the Streamlit app to build its form."""
    fe = params["feature_engineering"]
    engineered = list(fe["ratio_features"])
    num_inputs = [c for c in X.select_dtypes("number").columns if c not in engineered]
    return {
        "target": params["base"]["target"],
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "best_model": best_name,
        "all_columns": X.columns.tolist(),
        "engineered": engineered,
        "ratio_features": fe["ratio_features"],
        "epsilon": float(fe["epsilon"]),
        "numeric": {
            c: {
                "min": float(X[c].min()),
                "max": float(X[c].max()),
                "median": float(X[c].median()),
                "is_int": bool(pd.api.types.is_integer_dtype(X[c])),
            }
            for c in num_inputs
        },
        "categorical": {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in cat_cols},
        # values for columns the model ignores but still expects in the input frame
        "defaults": {
            c: (_py(X[c].mode().iloc[0]) if not X[c].mode().empty else None) for c in X.columns
        },
    }


def main() -> None:
    params = load_params()
    target = params["base"]["target"]

    train = pd.read_csv(ROOT / "data" / "processed" / "train_features.csv")
    X, y = train.drop(columns=target), train[target]

    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in params["model_building"]["onehot_columns"] if c in X.columns]
    preprocessor = make_preprocessor(num_cols, cat_cols)
    log.info("Training on %s rows: %s numeric and %s categorical features", len(X), len(num_cols), len(cat_cols))

    models_dir = ROOT / "models"
    cand_dir = models_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    info, fitted = {}, {}
    for model_id, (key, (name, build_fn)) in enumerate(MODELS.items()):
        pipe, study, search_time = tune(name, build_fn, preprocessor, X, y, params)
        path = cand_dir / f"{key}.pkl"
        joblib.dump(pipe, path)
        fitted[name] = pipe
        info[name] = {
            "key": key,
            "model_id": model_id,
            "best_params": study.best_params,
            "cv_r2": study.best_value,
            "n_trials": len(study.trials),
            "search_time_s": search_time,
            "model_size_kb": path.stat().st_size / 1024,
            "trial_values": [t.value for t in study.trials],
        }

    # best model = highest cross-validated R2 (the test set is only used in the next stage)
    best_name = max(info, key=lambda n: info[n]["cv_r2"])
    joblib.dump(fitted[best_name], models_dir / "best_model.pkl")
    log.info("Best model by CV R2: %s (%.4f)", best_name, info[best_name]["cv_r2"])

    with open(models_dir / "model_info.json", "w") as f:
        json.dump({"best_model": best_name, "models": info}, f, indent=2, default=str)
    with open(models_dir / "feature_schema.json", "w") as f:
        json.dump(build_schema(X, y, cat_cols, params, best_name), f, indent=2, default=str)
    log.info("Saved models and feature_schema.json to %s", models_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Model building failed")
        sys.exit(1)
