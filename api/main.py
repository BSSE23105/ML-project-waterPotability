# Water Potability Prediction API
# Accepts chemical readings and predicts if water is safe to drink, explains why,
# scores whole batches, and keeps a history of every verdict

import os
import sys

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from api.schemas import BatchRequest, BatchResponse, HistoryStats, PredictionResponse, WaterSample  # noqa: E402
from api.services.history import PredictionHistory  # noqa: E402
from api.services.predictor import Predictor, parse_csv  # noqa: E402

app = FastAPI(
    title="Water Potability Classifier",
    description="Predict whether a water sample is safe to drink, with guideline warnings, SHAP reasons, batch scoring and history.",
    version="2.0",
)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
MODEL_DIR = os.path.join(ROOT_DIR, "artifacts", "models")
HISTORY_DB = os.environ.get("WATER_HISTORY_DB", os.path.join(ROOT_DIR, "artifacts", "history.db"))

predictor = Predictor(MODEL_DIR)
history = PredictionHistory(HISTORY_DB)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Shows the web form where users can enter water readings."""
    info = predictor.model_info()
    context = {"model_name": info["display_name"], "model_auc": info["test_auc"] or 0}
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.post("/predict", response_model=PredictionResponse)
async def predict(sample: WaterSample, explain: bool = Query(True)):
    """Scores one sample, stores it, and returns the verdict with warnings and reasons."""
    result = predictor.predict_one(sample.model_dump(), explain=explain)
    result["id"] = history.add(result, source="api")
    return result


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(body: BatchRequest):
    """Scores up to 1000 samples sent as JSON in one call."""
    samples = [s.model_dump() for s in body.samples]
    results = predictor.predict_many(samples, explain=body.explain)
    for result, new_id in zip(results, history.add_many(results, source="batch")):
        result["id"] = new_id
    return {"summary": predictor.summarise(results), "results": results}


@app.post("/predict/batch/csv", responses={400: {"description": "The file is not a readable CSV with the nine reading columns"}})
async def predict_batch_csv(file: UploadFile = File(...), explain: bool = Query(True)):
    """Scores every row of an uploaded CSV that has the nine reading columns."""
    try:
        samples, skipped = parse_csv((await file.read()).decode("utf-8-sig"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not samples:
        raise HTTPException(status_code=400, detail="No valid rows found in the CSV file")
    results = predictor.predict_many(samples, explain=explain)
    for result, new_id in zip(results, history.add_many(results, source="csv")):
        result["id"] = new_id
    return {"summary": predictor.summarise(results), "results": results, "skipped_rows": skipped}


@app.post("/explain")
async def explain(sample: WaterSample):
    """Returns the SHAP contribution of every reading for one sample."""
    return predictor.breakdown(sample.model_dump())


@app.get("/history")
async def list_history(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                       result: str = Query(None, pattern="^(Potable|Not Potable)$")):
    """Past predictions, newest first."""
    return {"total": history.count(), "items": history.list(limit=limit, offset=offset, result=result)}


@app.get("/history/stats", response_model=HistoryStats)
async def history_stats():
    """How many samples were checked, how many were unsafe, and which readings were most often out of range."""
    return history.stats()


@app.get("/history/{prediction_id}", responses={404: {"description": "No prediction with that id"}})
async def get_prediction(prediction_id: int):
    record = history.get(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")
    return record


@app.delete("/history")
async def clear_history():
    return {"deleted": history.clear()}


@app.get("/model/info")
async def model_info():
    """Which model is serving, its tuned parameters and held-out metrics."""
    return predictor.model_info()


@app.get("/health")
async def health():
    """Simple health check to verify the API is running."""
    return {"status": "healthy", "model": predictor.model_info()["display_name"], "predictions_stored": history.count()}
