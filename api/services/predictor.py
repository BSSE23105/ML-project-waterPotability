# Loads the champion model once and scores single samples, batches and CSV uploads
# Every result carries guideline warnings and the SHAP reasons behind the verdict

import csv
import io
import json
import os
import sys

import joblib
import pandas as pd
from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.explain import PotabilityExplainer  # noqa: E402
from api.schemas import WaterSample  # noqa: E402

# Drinking-water guideline ranges (WHO guidelines, US EPA for chloramines and trihalomethanes).
# Values outside the range do not decide the verdict; they explain it.
GUIDELINE_LIMITS = {
    "ph": (6.5, 8.5, "6.5 to 8.5", "pH outside the WHO range of 6.5 to 8.5"),
    "Hardness": (None, 500.0, "up to 500 mg/L", "Hardness above 500 mg/L (very hard water)"),
    "Solids": (None, 1000.0, "up to 1000 mg/L", "Dissolved solids above 1000 mg/L (WHO: unpalatable)"),
    "Chloramines": (None, 4.0, "up to 4 mg/L", "Chloramines above the 4 mg/L disinfectant limit"),
    "Sulfate": (None, 250.0, "up to 250 mg/L", "Sulfate above the 250 mg/L taste threshold"),
    "Conductivity": (None, 400.0, "up to 400 uS/cm", "Conductivity above 400 uS/cm"),
    "Organic_carbon": (None, 4.0, "up to 4 mg/L", "Organic carbon above 4 mg/L"),
    "Trihalomethanes": (None, 80.0, "up to 80 ppb", "Trihalomethanes above the 80 ppb limit"),
    "Turbidity": (None, 5.0, "up to 5 NTU", "Turbidity above 5 NTU"),
}
INFO_KEYS = ("model_name", "display_name", "best_params", "cv_auc", "overfit_gap")
METRIC_KEYS = {"test_auc": "roc_auc", "accuracy": "accuracy", "f1_unsafe": "f1", "recall_unsafe": "recall"}


def check_guidelines(sample):
    """Compares each reading with its guideline range and returns the readings that fall outside."""
    warnings = []
    for feature, (low, high, limit, message) in GUIDELINE_LIMITS.items():
        value = sample.get(feature)
        if value is None:
            continue
        if (low is not None and value < low) or (high is not None and value > high):
            warnings.append({"feature": feature, "value": float(value), "limit": limit, "message": message})
    return warnings


def parse_csv(text):
    """Turns uploaded CSV text into validated sample dictionaries; reports the bad rows."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("The CSV file is empty")
    missing = [c for c in WaterSample.model_fields if c not in reader.fieldnames]
    if missing:
        raise ValueError(f"The CSV file is missing columns: {', '.join(missing)}")
    samples, errors = [], []
    for number, row in enumerate(reader, start=2):
        try:
            samples.append(WaterSample(**{k: row[k] for k in WaterSample.model_fields}).model_dump())
        except (ValidationError, ValueError) as exc:
            errors.append({"row": number, "error": str(exc).splitlines()[0]})
    return samples, errors


class Predictor:
    """Holds the model, its metadata and the explainer, and turns readings into verdicts."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        model_file = "best_model.joblib" if os.path.exists(os.path.join(model_dir, "best_model.joblib")) else "champion_model.joblib"
        info_file = "best_model_info.json" if os.path.exists(os.path.join(model_dir, "best_model_info.json")) else "champion_info.json"
        self.model = joblib.load(os.path.join(model_dir, model_file))
        with open(os.path.join(model_dir, "feature_columns.json"), "r") as f:
            self.feature_columns = json.load(f)
        with open(os.path.join(model_dir, info_file), "r") as f:
            self.info = json.load(f)
        self._explainer = None

    @property
    def explainer(self):
        """Builds the SHAP explainer on first use, so start-up stays fast."""
        if self._explainer is None:
            self._explainer = PotabilityExplainer(self.model, self.feature_columns)
        return self._explainer

    def to_frame(self, samples):
        """Puts a list of sample dictionaries into the model's column order."""
        return pd.DataFrame(samples)[self.feature_columns]

    def predict_many(self, samples, explain=True):
        """Scores a list of samples and attaches warnings and reasons to each."""
        frame = self.to_frame(samples)
        predictions = self.model.predict(frame)
        probabilities = self.model.predict_proba(frame)[:, 1]
        reasons = self.explainer.explain(frame) if explain else [[] for _ in samples]
        results = []
        for sample, prediction, probability, why in zip(samples, predictions, probabilities, reasons):
            prediction, probability = int(prediction), float(probability)
            confidence = probability if prediction == 1 else 1 - probability
            results.append({
                "prediction": prediction,
                "result": "Potable" if prediction == 1 else "Not Potable",
                "confidence": round(confidence * 100, 2),
                "probability_potable": round(probability * 100, 2),
                "input": sample,
                "warnings": check_guidelines(sample),
                "reasons": why,
            })
        return results

    def predict_one(self, sample, explain=True):
        return self.predict_many([sample], explain=explain)[0]

    def breakdown(self, sample):
        """Full SHAP contribution of every reading for one sample."""
        return self.explainer.full_breakdown(self.to_frame([sample]))

    @staticmethod
    def summarise(results):
        """Counts verdicts and warnings across a batch."""
        total = len(results)
        potable = sum(1 for r in results if r["prediction"] == 1)
        return {"total": total, "potable": potable, "not_potable": total - potable,
                "unsafe_share": round((total - potable) / total * 100, 2) if total else 0.0,
                "samples_with_warnings": sum(1 for r in results if r["warnings"])}

    def model_info(self):
        """What is serving predictions: model type, tuned parameters, held-out metrics."""
        info = {key: self.info.get(key) for key in INFO_KEYS}
        info.update({key: self.info.get("metrics", {}).get(source) for key, source in METRIC_KEYS.items()})
        info["display_name"] = info["display_name"] or "Unknown"
        info["feature_columns"] = self.feature_columns
        info["guideline_limits"] = {k: v[2] for k, v in GUIDELINE_LIMITS.items()}
        return info
