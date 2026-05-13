# Water Potability Prediction API
# Accepts chemical readings and predicts if water is safe to drink

import os
import json
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import joblib

app = FastAPI(
    title="Water Potability Classifier",
    description="Predict whether a water sample is safe to drink based on chemical readings.",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load the trained model and its metadata
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "artifacts", "models")

# Try new naming first, fall back to old naming for compatibility
model_file = "best_model.joblib" if os.path.exists(os.path.join(MODEL_DIR, "best_model.joblib")) else "champion_model.joblib"
info_file = "best_model_info.json" if os.path.exists(os.path.join(MODEL_DIR, "best_model_info.json")) else "champion_info.json"

model = joblib.load(os.path.join(MODEL_DIR, model_file))

with open(os.path.join(MODEL_DIR, "feature_columns.json"), "r") as f:
    FEATURE_COLUMNS = json.load(f)

with open(os.path.join(MODEL_DIR, info_file), "r") as f:
    MODEL_INFO = json.load(f)


class WaterSample(BaseModel):
    """The 9 chemical readings we need from the user."""
    ph: float = Field(..., ge=0, le=14, description="pH value (0-14)")
    Hardness: float = Field(..., ge=0, description="Hardness (mg/L)")
    Solids: float = Field(..., ge=0, description="Total Dissolved Solids (mg/L)")
    Chloramines: float = Field(..., ge=0, description="Chloramines (mg/L)")
    Sulfate: float = Field(..., ge=0, description="Sulfate (mg/L)")
    Conductivity: float = Field(..., ge=0, description="Conductivity (uS/cm)")
    Organic_carbon: float = Field(..., ge=0, description="Organic Carbon (mg/L)")
    Trihalomethanes: float = Field(..., ge=0, description="Trihalomethanes (ppm)")
    Turbidity: float = Field(..., ge=0, description="Turbidity (NTU)")


def preprocess_input(sample: dict) -> pd.DataFrame:
    """Puts the input into the correct column order for the model."""
    df = pd.DataFrame([sample])
    df = df[FEATURE_COLUMNS]
    return df


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Shows the web form where users can enter water readings."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_name": MODEL_INFO.get("display_name", "Unknown"),
            "model_auc": MODEL_INFO.get("metrics", {}).get("roc_auc", 0),
        },
    )


@app.post("/predict")
async def predict(sample: WaterSample):
    """Takes in chemical readings and returns whether the water is safe."""
    try:
        sample_dict = sample.model_dump()
        X = preprocess_input(sample_dict)

        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])

        result = "Potable" if prediction == 1 else "Not Potable"
        confidence = probability if prediction == 1 else (1 - probability)

        return JSONResponse({
            "prediction": prediction,
            "result": result,
            "confidence": round(confidence * 100, 2),
            "probability_potable": round(probability * 100, 2),
            "input": sample_dict,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    """Simple health check to verify the API is running."""
    return {
        "status": "healthy",
        "model": MODEL_INFO.get("display_name", "Unknown"),
    }
