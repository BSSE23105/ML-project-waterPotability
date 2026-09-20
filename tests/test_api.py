# End-to-end tests of the FastAPI endpoints through the test client


def test_home_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Water Potability" in response.text


def test_health_reports_model(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model"]


def test_predict_returns_reasons_and_stores_history(client, safe_sample):
    response = client.post("/predict", json=safe_sample)
    assert response.status_code == 200
    body = response.json()
    assert body["result"] in ("Potable", "Not Potable")
    assert len(body["reasons"]) == 3
    assert body["warnings"] == []
    stored = client.get(f"/history/{body['id']}").json()
    assert stored["inputs"] == safe_sample


def test_predict_can_skip_explanation(client, safe_sample):
    body = client.post("/predict?explain=false", json=safe_sample).json()
    assert body["reasons"] == []


def test_predict_rejects_out_of_range_ph(client, safe_sample):
    safe_sample["ph"] = 15
    assert client.post("/predict", json=safe_sample).status_code == 422


def test_predict_batch_json(client, safe_sample, extreme_sample):
    response = client.post("/predict/batch", json={"samples": [safe_sample, extreme_sample]})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total"] == 2
    assert body["summary"]["samples_with_warnings"] == 1
    assert all("id" in r for r in body["results"])


def test_predict_batch_rejects_empty_list(client):
    assert client.post("/predict/batch", json={"samples": []}).status_code == 422


def test_predict_batch_csv_upload(client, safe_sample, extreme_sample):
    header = ",".join(safe_sample)
    rows = [",".join(str(v) for v in s.values()) for s in (safe_sample, extreme_sample)]
    csv_text = "\n".join([header] + rows + ["bad,row"])
    response = client.post("/predict/batch/csv", files={"file": ("samples.csv", csv_text, "text/csv")})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total"] == 2
    assert body["skipped_rows"][0]["row"] == 4


def test_predict_batch_csv_rejects_missing_columns(client):
    response = client.post("/predict/batch/csv", files={"file": ("bad.csv", "ph,Hardness\n7,100\n", "text/csv")})
    assert response.status_code == 400
    assert "missing columns" in response.json()["detail"]


def test_explain_breaks_down_every_reading(client, extreme_sample):
    body = client.post("/explain", json=extreme_sample).json()
    assert len(body["contributions"]) == 9
    assert "base_value" in body


def test_history_listing_and_stats(client, safe_sample):
    client.post("/predict", json=safe_sample)
    listing = client.get("/history?limit=5").json()
    assert listing["total"] >= 1
    assert len(listing["items"]) <= 5
    filtered = client.get("/history?result=Not%20Potable").json()
    assert all(item["result"] == "Not Potable" for item in filtered["items"])
    assert client.get("/history?result=Maybe").status_code == 422
    stats = client.get("/history/stats").json()
    assert stats["total"] == listing["total"]
    assert stats["potable"] + stats["not_potable"] == stats["total"]


def test_history_missing_id_is_404(client):
    assert client.get("/history/999999").status_code == 404


def test_model_info_and_clear_history(client):
    info = client.get("/model/info").json()
    assert len(info["feature_columns"]) == 9
    assert "guideline_limits" in info
    deleted = client.delete("/history").json()["deleted"]
    assert deleted >= 1
    assert client.get("/history/stats").json()["total"] == 0
