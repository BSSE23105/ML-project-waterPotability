# SHAP explanations for the potability model
# Answers "which readings pushed this sample towards safe or unsafe?"

import numpy as np
import pandas as pd
import shap


class PotabilityExplainer:
    """Wraps a fitted pipeline and explains its predictions with SHAP values."""

    def __init__(self, pipeline, feature_names):
        self.pipeline = pipeline
        self.feature_names = list(feature_names)
        if hasattr(pipeline, "named_steps"):
            self.estimator = pipeline.named_steps.get("model", pipeline)
            self.scaler = pipeline.named_steps.get("scaler")
        else:
            self.estimator = pipeline
            self.scaler = None
        self._explainer = shap.TreeExplainer(self.estimator)

    def _prepare(self, X):
        """Puts the input in the model's column order and applies the scaler."""
        frame = pd.DataFrame(X, columns=self.feature_names) if not isinstance(X, pd.DataFrame) else X[self.feature_names]
        values = self.scaler.transform(frame) if self.scaler is not None else frame.to_numpy(dtype=float)
        return frame, values

    def shap_values(self, X):
        """Returns one SHAP value per feature per row, for the potable class."""
        _, values = self._prepare(X)
        raw = self._explainer.shap_values(values)
        if isinstance(raw, list):
            contributions = np.asarray(raw[1]) if len(raw) > 1 else np.asarray(raw[0])
        else:
            raw = np.asarray(raw)
            contributions = raw[:, :, 1] if raw.ndim == 3 else raw
        return contributions

    def explain(self, X, top_n=3):
        """Lists the strongest contributing readings for every row, largest first."""
        frame, _ = self._prepare(X)
        contributions = self.shap_values(frame)
        explanations = []
        for row_idx in range(contributions.shape[0]):
            row = contributions[row_idx]
            order = np.argsort(np.abs(row))[::-1][:top_n]
            reasons = []
            for idx in order:
                reasons.append({
                    "feature": self.feature_names[idx],
                    "value": float(frame.iloc[row_idx, idx]),
                    "contribution": round(float(row[idx]), 4),
                    "direction": "towards potable" if row[idx] > 0 else "towards not potable",
                })
            explanations.append(reasons)
        return explanations

    def full_breakdown(self, X):
        """Returns every feature's contribution for a single row, sorted by strength."""
        frame, _ = self._prepare(X)
        row = self.shap_values(frame)[0]
        breakdown = []
        for idx in np.argsort(np.abs(row))[::-1]:
            breakdown.append({
                "feature": self.feature_names[idx],
                "value": float(frame.iloc[0, idx]),
                "contribution": round(float(row[idx]), 4),
            })
        base = np.ravel(self._explainer.expected_value)
        return {"base_value": round(float(base[1] if base.size > 1 else base[0]), 4), "contributions": breakdown}

    def summary(self, X):
        """Global importance: mean absolute SHAP value per feature, ranked."""
        contributions = self.shap_values(X)
        mean_abs = np.abs(contributions).mean(axis=0)
        ranked = np.argsort(mean_abs)[::-1]
        return [{"feature": self.feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 4)} for i in ranked]
