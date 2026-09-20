# Unit tests for the explainer, the predictor service and the history store

import os

import numpy as np
import pytest

from api.services.history import PredictionHistory
from api.services.predictor import GUIDELINE_LIMITS, Predictor, check_guidelines, parse_csv
from src.explain import PotabilityExplainer

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "models")


@pytest.fixture(scope="module")
def predictor():
    return Predictor(MODEL_DIR)


def test_guideline_check_flags_every_bad_reading(extreme_sample):
    warnings = check_guidelines(extreme_sample)
    assert {w["feature"] for w in warnings} == set(GUIDELINE_LIMITS)
    assert all(w["limit"] and w["message"] for w in warnings)


def test_guideline_check_passes_a_normal_sample(safe_sample):
    assert check_guidelines(safe_sample) == []


def test_guideline_check_catches_low_ph():
    warnings = check_guidelines({"ph": 5.0})
    assert len(warnings) == 1
    assert warnings[0]["feature"] == "ph"


def test_predict_one_returns_verdict_warnings_and_reasons(predictor, extreme_sample):
    result = predictor.predict_one(extreme_sample)
    assert result["prediction"] in (0, 1)
    assert result["result"] in ("Potable", "Not Potable")
    assert 0 <= result["confidence"] <= 100
    assert abs(result["confidence"] + result["probability_potable"] - 100) < 0.01 or result["prediction"] == 1
    assert len(result["reasons"]) == 3
    assert len(result["warnings"]) == len(GUIDELINE_LIMITS)


def test_predict_without_explanation_skips_reasons(predictor, safe_sample):
    result = predictor.predict_one(safe_sample, explain=False)
    assert result["reasons"] == []


def test_predict_many_matches_predict_one(predictor, safe_sample, extreme_sample):
    many = predictor.predict_many([safe_sample, extreme_sample])
    assert [r["prediction"] for r in many] == [predictor.predict_one(safe_sample)["prediction"],
                                              predictor.predict_one(extreme_sample)["prediction"]]
    summary = Predictor.summarise(many)
    assert summary["total"] == 2
    assert summary["potable"] + summary["not_potable"] == 2
    assert summary["samples_with_warnings"] == 1


def test_summarise_empty_batch():
    assert Predictor.summarise([])["unsafe_share"] == 0.0


def test_model_info_lists_features_and_metrics(predictor):
    info = predictor.model_info()
    assert info["feature_columns"] == list(GUIDELINE_LIMITS)
    assert 0.5 < info["test_auc"] <= 1.0
    assert info["display_name"]


def test_explainer_contributions_sum_to_prediction(predictor, safe_sample):
    explainer = PotabilityExplainer(predictor.model, predictor.feature_columns)
    frame = predictor.to_frame([safe_sample])
    breakdown = explainer.full_breakdown(frame)
    total = breakdown["base_value"] + sum(c["contribution"] for c in breakdown["contributions"])
    probability = predictor.model.predict_proba(frame)[0, 1]
    assert abs(total - probability) < 0.01
    assert len(breakdown["contributions"]) == 9


def test_explainer_summary_is_ranked(predictor, safe_sample, extreme_sample):
    explainer = PotabilityExplainer(predictor.model, predictor.feature_columns)
    ranking = explainer.summary(predictor.to_frame([safe_sample, extreme_sample]))
    values = [r["mean_abs_shap"] for r in ranking]
    assert values == sorted(values, reverse=True)
    assert np.all(np.array(values) >= 0)


def test_parse_csv_reads_valid_rows_and_reports_bad_ones(safe_sample):
    header = ",".join(safe_sample)
    good = ",".join(str(v) for v in safe_sample.values())
    bad = good.replace("7.2", "not-a-number", 1)
    samples, errors = parse_csv(f"{header}\n{good}\n{bad}\n")
    assert len(samples) == 1
    assert samples[0]["ph"] == 7.2
    assert errors == [{"row": 3, "error": errors[0]["error"]}]


def test_parse_csv_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        parse_csv("ph,Hardness\n7,100\n")


def test_history_add_get_list_stats_and_clear(tmp_path, predictor, safe_sample, extreme_sample):
    store = PredictionHistory(str(tmp_path / "h.db"))
    results = predictor.predict_many([safe_sample, extreme_sample])
    ids = store.add_many(results)
    assert ids == [1, 2]
    record = store.get(2)
    assert record["inputs"] == extreme_sample
    assert len(record["warnings"]) == 9
    assert store.get(99) is None
    assert [r["id"] for r in store.list()] == [2, 1]
    assert len(store.list(result=results[0]["result"])) >= 1
    stats = store.stats()
    assert stats["total"] == 2
    assert stats["most_common_warnings"][0]["count"] == 1
    assert stats["predictions_per_day"][0]["count"] == 2
    assert store.clear() == 2
    assert store.count() == 0
    assert store.stats()["unsafe_share"] == 0.0
    store.close()
