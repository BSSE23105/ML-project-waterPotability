# Water Potability Classifier

Machine Learning pipeline to classify whether a water sample is safe to drink based on chemical sensor readings.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train all models
python train.py

# Run API locally
uvicorn api.main:app --reload --port 8000
```

## Project Structure

- `src/` — Modular pipeline code (data, features, models, evaluation)
- `api/` — FastAPI application with web UI
- `artifacts/` — Saved models, plots, and metrics
- `train.py` — Main training orchestrator
- `config.yaml` — All configuration and hyperparameters

## Models

- Logistic Regression
- SVM (RBF Kernel)
- Decision Tree
- Random Forest

## Deployment

```bash
docker build -t water-potability .
docker run -p 8000:8000 water-potability
```
