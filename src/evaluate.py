# Evaluation functions for computing metrics, saving plots, and comparing models

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
)


def compute_all_metrics(y_true, y_pred, y_prob):
    """Calculates all the performance metrics we need. Positive class = Unsafe (0)."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, pos_label=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=0)),
        "precision_class0": float(precision_score(y_true, y_pred, pos_label=0)),
        "recall_class0": float(recall_score(y_true, y_pred, pos_label=0)),
        "f1_class0": float(f1_score(y_true, y_pred, pos_label=0)),
        "precision_class1": float(precision_score(y_true, y_pred, pos_label=1)),
        "recall_class1": float(recall_score(y_true, y_pred, pos_label=1)),
        "f1_class1": float(f1_score(y_true, y_pred, pos_label=1)),
    }
    return metrics


def print_classification_report(y_true, y_pred, model_name="Model"):
    """Prints a custom performance summary for each class with data percentages."""
    total = len(y_true)
    count_0 = int((y_true == 0).sum())
    count_1 = int((y_true == 1).sum())
    pct_0 = count_0 / total * 100
    pct_1 = count_1 / total * 100

    p0 = precision_score(y_true, y_pred, pos_label=0)
    r0 = recall_score(y_true, y_pred, pos_label=0)
    f0 = f1_score(y_true, y_pred, pos_label=0)
    p1 = precision_score(y_true, y_pred, pos_label=1)
    r1 = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    acc = accuracy_score(y_true, y_pred)

    print(f"\n  {model_name} Results\n")
    print(f"  Not Potable ({pct_0:.0f}% of data)  ->  Precision: {p0:.2f}  |  Recall: {r0:.2f}  |  F1: {f0:.2f}")
    print(f"  Potable ({pct_1:.0f}% of data)      ->  Precision: {p1:.2f}  |  Recall: {r1:.2f}  |  F1: {f1:.2f}")
    print(f"  Overall Accuracy: {acc:.2f}\n")


def plot_confusion_matrix(y_true, y_pred, model_name, save_dir):
    """Creates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Potable", "Potable"],
                yticklabels=["Not Potable", "Potable"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(save_dir, f"confusion_matrix_{filename}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_roc_curves(all_results, y_test, save_dir):
    """Plots ROC curves for all models on a single chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, result in all_results.items():
        fpr, tpr, _ = roc_curve(y_test, result["y_prob"])
        auc_val = result["metrics"]["roc_auc"]
        label = result.get("display_name", name)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.4f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Model Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, "roc_curves_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_pr_curves(all_results, y_test, save_dir):
    """Plots Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, result in all_results.items():
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, result["y_prob"])
        ap = average_precision_score(y_test, result["y_prob"])
        label = result.get("display_name", name)
        ax.plot(rec_vals, prec_vals, label=f"{label} (AP={ap:.4f})", linewidth=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves - Model Comparison", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, "pr_curves_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_feature_importance(model, feature_names, model_name, save_dir):
    """Creates a bar chart showing which features matter most."""
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)
    else:
        estimator = model

    if not hasattr(estimator, "feature_importances_"):
        print(f"  {model_name} doesnt support feature importance, skipping.")
        return

    importances = estimator.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(20, len(feature_names))
    top_order = order[:top_n]
    ax.barh(range(top_n), importances[top_order][::-1],
            color=sns.color_palette("viridis", top_n))
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_order][::-1])
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances - {model_name}", fontsize=14)
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(save_dir, f"feature_importance_{filename}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)





def generate_comparison_table(all_results):
    """Prints a side-by-side comparison of all trained models."""
    print(f"\n{'Model':<25} {'CV AUC':>8} {'Test AUC':>9} {'Prec(0)':>9} {'Rec(0)':>8} {'F1(0)':>8} {'Rec(1)':>8} {'Gap':>8} {'Status':>10}")
    print("-" * 100)

    for name, result in all_results.items():
        m = result["metrics"]
        gap = result["overfit_gap"]
        label = result.get("display_name", name)
        status = "OK" if gap <= 0.05 else "OVERFIT"
        print(f"{label:<25} {result['cv_auc']:>8.4f} {m['roc_auc']:>9.4f} "
              f"{m['precision']:>9.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
              f"{m['recall_class1']:>8.4f} {gap:>8.4f} {status:>10}")
    print()
