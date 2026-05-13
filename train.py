# Main training script for the Water Potability Classifier
# Trains 4 models, picks the best one, and saves everything to artifacts/

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, set_seed, save_model, ensure_dirs
from src.data import (
    load_dataset, fit_imputer, apply_imputer,
    fit_outlier_clipper, apply_outlier_clipper,
)
from src.models import (
    run_optuna_study, build_model, select_best_model,
    MODEL_NAMES, MODEL_DISPLAY_NAMES,
)
from src.evaluate import (
    compute_all_metrics, print_classification_report,
    plot_confusion_matrix, plot_roc_curves, plot_pr_curves,
    plot_feature_importance, generate_comparison_table,
)

SEP = "#" * 55


def main():
    start_time = time.time()
    print(f"\n{SEP}")
    print("  Water Potability Classifier - Training Pipeline")
    print(SEP)


    # Load settings
    config = load_config("config.yaml")
    set_seed(config["seed"])
    ensure_dirs(config)

    plot_dir = config["paths"]["plot_dir"]
    model_dir = config["paths"]["model_dir"]


    print(f"\n\n{SEP}")
    print("  Step 1: Loading Dataset")
    print(SEP)

    df = load_dataset(config["paths"]["data"])


    print(f"\n\n{SEP}")
    print("  Step 2: Splitting Data (Train/Test)")
    print(SEP)

    feature_cols = [c for c in df.columns if c != "Potability"]
    X = df[feature_cols]
    y = df["Potability"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["seed"],
        stratify=y,
    )

    # Rejoin target column temporarily for class-conditional imputation
    df_train = X_train_raw.copy()
    df_train["Potability"] = y_train.values
    df_test = X_test_raw.copy()
    df_test["Potability"] = y_test.values

    print(f"  Train: {X_train_raw.shape[0]} samples | Test: {X_test_raw.shape[0]} samples")
    print(f"  Train classes: {dict(y_train.value_counts())}")
    print(f"  Test classes: {dict(y_test.value_counts())}")


    print(f"\n\n{SEP}")
    print("  Step 3: Preprocessing (fitted on training data only)")
    print(SEP)

    medians = fit_imputer(df_train)
    df_train = apply_imputer(df_train, medians)
    df_test = apply_imputer(df_test, medians)

    bounds = fit_outlier_clipper(df_train)
    df_train = apply_outlier_clipper(df_train, bounds)
    df_test = apply_outlier_clipper(df_test, bounds)

    feature_cols = [c for c in df_train.columns if c != "Potability"]
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]

    print(f"  Features ({len(feature_cols)}): {feature_cols}")


    print(f"\n\n{SEP}")
    print("  Step 4: Hyperparameter Optimization (Optuna)")
    print(SEP)

    all_results = {}

    for model_name in MODEL_NAMES:
        display_name = MODEL_DISPLAY_NAMES[model_name]

        # Centered model name with underline
        title = f"  {display_name}  "
        underline = "  " + "_" * len(display_name)
        print(f"\n{title}")
        print(underline)

        best_params, cv_auc = run_optuna_study(model_name, X_train, y_train, config)
        print(f"  Best params: {best_params}")

        pipeline = build_model(model_name, best_params, config["seed"])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        train_prob = pipeline.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_prob)
        test_auc = roc_auc_score(y_test, y_prob)
        gap = train_auc - test_auc

        metrics = compute_all_metrics(y_test, y_pred, y_prob)

        print_classification_report(y_test, y_pred, display_name)
        print(f"  Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | Gap: {gap:.4f}")
        if gap > 0.05:
            print(f"  WARNING: This model is overfitting (gap > 0.05)")

        all_results[model_name] = {
            "display_name": display_name,
            "best_params": best_params,
            "cv_auc": cv_auc,
            "train_auc": train_auc,
            "metrics": metrics,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "overfit_gap": gap,
            "pipeline": pipeline,
        }

        plot_confusion_matrix(y_test, y_pred, display_name, plot_dir)

        if model_name in ["decision_tree", "random_forest"]:
            plot_feature_importance(pipeline, feature_cols, display_name, plot_dir)

        print("\n\n")  # 3 blank lines between models


    print(f"\n\n{SEP}")
    print("  Step 5: Model Comparison")
    print(SEP)

    generate_comparison_table(all_results)
    plot_roc_curves(all_results, y_test, plot_dir)
    plot_pr_curves(all_results, y_test, plot_dir)
    print(f"  Saved all comparison plots to {plot_dir}/")


    print(f"\n\n{SEP}")
    print("  Step 6: Selecting Best Model")
    print(SEP)

    best_name, best_result = select_best_model(all_results)
    m = best_result["metrics"]

    print(f"  Best Model: {MODEL_DISPLAY_NAMES[best_name]}")
    print(f"  Test AUC:      {m['roc_auc']:.4f}")
    print(f"  Accuracy:      {m['accuracy']:.4f}")
    print(f"  Precision(0):  {m['precision']:.4f}  (unsafe detection accuracy)")
    print(f"  Recall(0):     {m['recall']:.4f}  (unsafe water caught)")
    print(f"  F1(0):         {m['f1']:.4f}")
    print(f"  Recall(1):     {m['recall_class1']:.4f}  (safe water identified)")
    print(f"  Overfit Gap:   {best_result['overfit_gap']:.4f}")


    print(f"\n\n{SEP}")
    print("  Step 7: Most Influential Chemical Factors")
    print(SEP)

    best_pipeline = best_result["pipeline"]
    estimator = best_pipeline.named_steps.get("model", best_pipeline)
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        ranked = importances.argsort()[::-1]
        print(f"  {'Rank':<6} {'Feature':<20} {'Importance':<12}")
        for rank, idx in enumerate(ranked, 1):
            print(f"  {rank:<6} {feature_cols[idx]:<20} {importances[idx]:.4f}")
    else:
        print("  Best model does not support feature importances")


    print(f"\n\n{SEP}")
    print("  Step 8: Saving Artifacts")
    print(SEP)

    model_path = os.path.join(model_dir, "best_model.joblib")
    save_model(best_result["pipeline"], model_path)

    model_info = {
        "model_name": best_name,
        "display_name": MODEL_DISPLAY_NAMES[best_name],
        "best_params": best_result["best_params"],
        "cv_auc": best_result["cv_auc"],
        "metrics": best_result["metrics"],
        "overfit_gap": best_result["overfit_gap"],
        "feature_columns": feature_cols,
    }

    info_path = os.path.join(model_dir, "best_model_info.json")
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    features_path = os.path.join(model_dir, "feature_columns.json")
    with open(features_path, "w") as f:
        json.dump(feature_cols, f)

    print(f"  Saved model, metadata, and feature columns to {model_dir}/")
    print(f"  Saved all plots to {plot_dir}/")


    print(f"\n\n{SEP}")
    elapsed = time.time() - start_time
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Best model: {MODEL_DISPLAY_NAMES[best_name]}")
    print(f"  All artifacts saved to artifacts/")
    print(SEP)


if __name__ == "__main__":
    main()
