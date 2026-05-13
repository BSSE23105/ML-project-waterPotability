# Model definitions, hyperparameter tuning with Optuna, and best model selection

import warnings
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAMES = ["logistic_regression", "svm", "decision_tree", "random_forest"]

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM (RBF Kernel)",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}


def _create_objective(model_name, X, y, cv_folds, search_spaces, seed):
    """Creates the function that Optuna will call on each trial to score a hyperparameter combo."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial):
        if model_name == "logistic_regression":
            space = search_spaces["logistic_regression"]
            params = {
                "C": trial.suggest_float("C", space["C"][0], space["C"][1], log=True),
                "solver": trial.suggest_categorical("solver", space["solver"]),
                "penalty": "l2",
                "max_iter": 1000,
                "random_state": seed,
                "class_weight": "balanced",
            }
            estimator = LogisticRegression(**params)

        elif model_name == "svm":
            space = search_spaces["svm"]
            gamma_choices = [g for g in space["gamma"] if isinstance(g, str)]
            gamma_float = [g for g in space["gamma"] if isinstance(g, (int, float))]
            use_str = trial.suggest_categorical("gamma_type", ["string", "float"])
            if use_str == "string":
                gamma = trial.suggest_categorical("gamma_str", gamma_choices)
            else:
                gamma = trial.suggest_float("gamma_float", min(gamma_float), max(gamma_float), log=True)
            params = {
                "C": trial.suggest_float("C", space["C"][0], space["C"][1], log=True),
                "gamma": gamma,
                "kernel": "rbf",
                "probability": True,
                "random_state": seed,
                "class_weight": "balanced",
            }
            estimator = SVC(**params)

        elif model_name == "decision_tree":
            space = search_spaces["decision_tree"]
            params = {
                "max_depth": trial.suggest_int("max_depth", space["max_depth"][0], space["max_depth"][1]),
                "min_samples_split": trial.suggest_int("min_samples_split", space["min_samples_split"][0], space["min_samples_split"][1]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", space["min_samples_leaf"][0], space["min_samples_leaf"][1]),
                "ccp_alpha": trial.suggest_float("ccp_alpha", space["ccp_alpha"][0], space["ccp_alpha"][1]),
                "criterion": trial.suggest_categorical("criterion", space["criterion"]),
                "random_state": seed,
                "class_weight": "balanced",
            }
            estimator = DecisionTreeClassifier(**params)

        elif model_name == "random_forest":
            space = search_spaces["random_forest"]
            params = {
                "n_estimators": trial.suggest_int("n_estimators", space["n_estimators"][0], space["n_estimators"][1]),
                "max_depth": trial.suggest_int("max_depth", space["max_depth"][0], space["max_depth"][1]),
                "min_samples_split": trial.suggest_int("min_samples_split", space["min_samples_split"][0], space["min_samples_split"][1]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", space["min_samples_leaf"][0], space["min_samples_leaf"][1]),
                "max_features": trial.suggest_categorical("max_features", space["max_features"]),
                "class_weight": trial.suggest_categorical("class_weight", space["class_weight"]),
                "random_state": seed,
                "n_jobs": -1,
            }
            estimator = RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Pipeline: SMOTEENN (balance classes) -> Scale -> Train
        pipeline = ImbPipeline([
            ("resampler", SMOTEENN(random_state=seed)),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

        scores = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc", n_jobs=-1, error_score="raise")
        return scores.mean()

    return objective


def run_optuna_study(model_name, X, y, config):
    """Runs Bayesian hyperparameter search and returns the best params and CV score."""
    seed = config["seed"]
    n_trials = config["optuna_trials"]
    cv_folds = config["cv_folds"]

    print(f"\n  Tuning {model_name} with {n_trials} trials ({cv_folds}-fold CV)...")

    objective = _create_objective(model_name, X, y, cv_folds, config["search_spaces"], seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best CV AUC: {study.best_value:.4f}")
    return study.best_params, study.best_value


def build_model(model_name, best_params, seed):
    """Builds the final pipeline using the best hyperparameters from Optuna."""
    params = {k: v for k, v in best_params.items()
              if k not in ["gamma_type", "gamma_str", "gamma_float"]}

    if model_name == "logistic_regression":
        params.setdefault("penalty", "l2")
        params.setdefault("max_iter", 1000)
        params.setdefault("random_state", seed)
        params.setdefault("class_weight", "balanced")
        estimator = LogisticRegression(**params)

    elif model_name == "svm":
        if "gamma_str" in best_params:
            params["gamma"] = best_params["gamma_str"]
        elif "gamma_float" in best_params:
            params["gamma"] = best_params["gamma_float"]
        params.setdefault("kernel", "rbf")
        params.setdefault("probability", True)
        params.setdefault("random_state", seed)
        params.setdefault("class_weight", "balanced")
        estimator = SVC(**params)

    elif model_name == "decision_tree":
        params.setdefault("random_state", seed)
        params.setdefault("class_weight", "balanced")
        estimator = DecisionTreeClassifier(**params)

    elif model_name == "random_forest":
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
        estimator = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = ImbPipeline([
        ("resampler", SMOTEENN(random_state=seed)),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
    return pipeline


def select_best_model(results):
    """Picks the best model by filtering out overfit ones, then choosing highest F1."""
    valid = {name: r for name, r in results.items() if r["overfit_gap"] <= 0.05}

    if not valid:
        print("WARNING: All models overfit! Picking the one with smallest gap.")
        valid = results
        best = min(valid.items(), key=lambda x: x[1]["overfit_gap"])
        return best[0], best[1]

    best = max(valid.items(), key=lambda x: (x[1]["metrics"]["f1"], x[1]["metrics"]["recall"]))
    return best[0], best[1]
