# Water Potability Classifier - Project Documentation

## 1. Problem Statement

We are given a dataset of water samples with 9 chemical sensor readings. The task is to:
1. Classify whether a water sample is safe to drink (Potable) or not
2. Identify the most dangerous chemical factors
3. Deploy the model as a public API

This is a **binary classification problem** where:
- **Class 0** = Not Potable (unsafe to drink)
- **Class 1** = Potable (safe to drink)

---

## 2. Dataset

**Source:** Kaggle - Water Potability Dataset  
**File:** `water_potability.csv`  
**Size:** 3,276 water samples, 10 columns

### 2.1 Features (9 chemical readings)

| Feature | Unit | Description |
|---------|------|-------------|
| `ph` | 0-14 | Acidity/alkalinity of water |
| `Hardness` | mg/L | Calcium and magnesium content |
| `Solids` | mg/L | Total dissolved solids |
| `Chloramines` | mg/L | Disinfectant chemical level |
| `Sulfate` | mg/L | Sulfate mineral concentration |
| `Conductivity` | uS/cm | Electrical conductivity |
| `Organic_carbon` | mg/L | Organic carbon content |
| `Trihalomethanes` | ppm | Chemical byproducts |
| `Turbidity` | NTU | Water cloudiness |

### 2.2 Target Variable
- `Potability`: 0 = Not Potable, 1 = Potable

### 2.3 Class Distribution (Imbalanced)
- Not Potable (0): 1,998 samples (61%)
- Potable (1): 1,278 samples (39%)

This imbalance is important — without handling it, the model would tend to predict "Not Potable" for everything and still get 61% accuracy.

### 2.4 Missing Values
- `ph`: 491 missing (15%)
- `Sulfate`: 781 missing (24%)
- `Trihalomethanes`: 162 missing (5%)

### 2.5 Dataset Selection Justification
This dataset was chosen because:
1. It directly matches the project requirement (water potability classification)
2. It contains real-world chemical sensor readings
3. It has enough samples (3,276) for meaningful training
4. It presents challenges (missing values, class imbalance) that demonstrate preprocessing skills

---

## 3. Project Structure

```
ML-Project/
├── water_potability.csv          # Raw dataset
├── config.yaml                   # All configuration (seeds, paths, hyperparameter spaces)
├── train.py                      # Main training orchestrator
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container for deployment
├── docker-compose.yml            # Local Docker orchestration
├── README.md                     # Quick start guide
├── documentation.md              # This file
├── viva.md                       # Viva preparation Q&A
│
├── src/                          # Source code modules
│   ├── __init__.py               # Package marker
│   ├── data.py                   # Data loading and preprocessing
│   ├── models.py                 # Model definitions, HPO, champion selection
│   ├── evaluate.py               # Metrics, plots, SHAP analysis
│   └── utils.py                  # Config loading, seed setting, model saving
│
├── api/                          # FastAPI web service
│   ├── main.py                   # API endpoints (/predict, /health, /)
│   └── templates/
│       └── index.html            # Web UI form
│
└── artifacts/                    # Generated outputs (created by train.py)
    ├── models/
    │   ├── champion_model.joblib     # Serialized trained model
    │   ├── champion_info.json        # Model metadata and metrics
    │   └── feature_columns.json      # Feature names in correct order
    ├── plots/
    │   ├── confusion_matrix_*.png    # Per-model confusion matrices
    │   ├── roc_curves_comparison.png # ROC curves for all 4 models
    │   ├── pr_curves_comparison.png  # Precision-Recall curves
    │   ├── feature_importance_*.png  # Tree-based feature importances
    │   ├── shap_summary.png          # SHAP feature impact plot
    │   └── shap_bar.png              # SHAP bar chart
    └── metrics/
        └── metrics_*.json            # Per-model metric files
```

---

## 4. File-by-File Explanation

### 4.1 `config.yaml` — Configuration

All settings are externalized here (no hardcoded values in code):
- `seed: 42` — Random seed for reproducibility (same results every run)
- `test_size: 0.2` — 20% of data reserved for testing
- `cv_folds: 5` — 5-fold cross-validation during hyperparameter tuning
- `optuna_trials: 60` — Number of hyperparameter combinations to try
- `search_spaces` — Min/max ranges for each model's hyperparameters

### 4.2 `src/data.py` — Data Loading & Preprocessing

**Functions:**

| Function | What it does |
|----------|-------------|
| `load_dataset(path)` | Reads the CSV file and prints basic info (shape, missing values, class distribution) |
| `fit_imputer(df_train)` | Computes median values from TRAINING data only, separately for each class. Returns a dictionary of medians |
| `apply_imputer(df, impute_stats)` | Uses the pre-computed medians to fill missing values in any dataframe |
| `fit_outlier_clipper(df_train)` | Computes the 1st and 99th percentile boundaries from TRAINING data only |
| `apply_outlier_clipper(df, clip_bounds)` | Clips extreme values using pre-computed boundaries |

**Why class-conditional imputation?**  
If unsafe water typically has pH around 6.8 and safe water has pH around 7.4, filling all missing pH values with a single global median (7.0) blurs the difference between the two classes. By computing medians per class, we preserve the within-class patterns.

**Why outlier clipping instead of removal?**  
Removing outlier rows loses data. Clipping caps extreme values (e.g., if a reading is impossibly high, cap it at the 99th percentile) while keeping the sample.

**Why fit on train only?**  
If we computed medians or percentiles on the full dataset (including test data), the model would indirectly "see" test data during training. This is called data leakage. By fitting only on training data, the test set remains truly unseen.

### 4.3 `src/models.py` — Model Definitions & Hyperparameter Optimization

**Functions:**

| Function | What it does |
|----------|-------------|
| `_create_objective(model_name, X, y, ...)` | Creates an Optuna objective function that builds a pipeline (SMOTEENN + StandardScaler + Model), runs 5-fold CV, and returns the mean ROC-AUC score |
| `run_optuna_study(model_name, X, y, config)` | Runs Bayesian optimization for 60 trials to find the best hyperparameters |
| `build_model(model_name, best_params, seed)` | Builds the final pipeline with the best hyperparameters found by Optuna |
| `select_champion(results)` | Selects the best model: filters out overfit models (gap > 0.05), then picks the one with highest F1 score for the unsafe class (class 0) |

**The 4 Models and Why They Were Chosen:**

| Model | Why included | Strengths | Weaknesses |
|-------|-------------|-----------|------------|
| **Logistic Regression** | Simplest linear baseline | Fast, interpretable | Cannot capture non-linear patterns |
| **SVM (RBF Kernel)** | Non-linear classifier | Good with clear margins | Slow, tends to overfit here |
| **Decision Tree** | Non-linear, interpretable | Easy to explain splits | Can overfit without pruning |
| **Random Forest** | Ensemble of many trees | Robust, handles noise well | Less interpretable than single tree |

We compare 4 models because the project rubric requires showing multiple models and selecting the best one.

**What is SMOTEENN and why is it used?**

The dataset is imbalanced (61% Not Potable, 39% Potable). Without handling this:
- The model would just predict "Not Potable" for everything
- It would get 61% accuracy but miss all safe water

SMOTEENN is a two-step process:
1. **SMOTE** (Synthetic Minority Over-sampling): Creates synthetic Potable samples by interpolating between existing Potable samples. This balances the dataset.
2. **ENN** (Edited Nearest Neighbors): After SMOTE, removes noisy/ambiguous samples that are surrounded by the opposite class. This cleans the boundary.

SMOTEENN is applied **inside each cross-validation fold** (not before the split) to prevent data leakage.

**What is Optuna and how does it work?**

Optuna is a hyperparameter optimization framework. Instead of testing every combination (GridSearch) or random guessing (RandomSearch), Optuna uses **Bayesian optimization**:
1. Trial 1: Try random hyperparameters, get AUC score
2. Trial 2: Based on Trial 1's result, try hyperparameters that are likely better
3. ... repeat for 60 trials
4. Each trial learns from all previous trials

This is more efficient than GridSearch because it focuses on promising regions of the search space.

**What is the ImbPipeline?**

```python
ImbPipeline([
    ("resampler", SMOTEENN(random_state=42)),    # Step 1: Balance classes
    ("scaler", StandardScaler()),                 # Step 2: Normalize features
    ("model", RandomForestClassifier(...)),       # Step 3: Train model
])
```

This pipeline ensures:
- SMOTEENN only runs during `.fit()` (training), never during `.predict()` (inference)
- StandardScaler fits on training data and transforms both train and test
- Everything happens in the correct order automatically

### 4.4 `src/evaluate.py` — Evaluation & Visualization

**Functions:**

| Function | What it does |
|----------|-------------|
| `compute_all_metrics(y_true, y_pred, y_prob)` | Computes accuracy, precision, recall, F1, ROC-AUC for both classes |
| `print_classification_report(y_true, y_pred, model_name)` | Prints sklearn's classification report |
| `plot_confusion_matrix(y_true, y_pred, model_name, save_dir)` | Saves a heatmap of the confusion matrix |
| `plot_roc_curves(all_results, y_test, save_dir)` | Plots ROC curves for all 4 models on one chart |
| `plot_pr_curves(all_results, y_test, save_dir)` | Plots Precision-Recall curves for all 4 models |
| `plot_feature_importance(model, feature_names, model_name, save_dir)` | Bar chart of feature importances (tree-based models only) |
| `run_shap_analysis(model, X_test, feature_names, save_dir)` | Generates SHAP summary and bar plots |
| `save_metrics(metrics, model_name, save_dir)` | Saves metrics as JSON |
| `generate_comparison_table(all_results)` | Prints a formatted table comparing all models |

**Metrics Explained (Positive class = Not Potable / Unsafe):**

| Metric | Formula | What it means in this project |
|--------|---------|-------------------------------|
| **Accuracy** | Correct / Total | Overall percentage of correct predictions |
| **Precision(0)** | TP / (TP + FP) | When we say water is unsafe, how often are we right? |
| **Recall(0)** | TP / (TP + FN) | Out of all actually unsafe water, how much did we catch? |
| **F1(0)** | 2 * Prec * Rec / (Prec + Rec) | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under ROC curve | Overall ability to distinguish safe from unsafe (0.5 = random, 1.0 = perfect) |

**Why Positive = Unsafe (Class 0)?**
In water safety, the dangerous case is our "positive" — we want to DETECT unsafe water. This is similar to medical diagnosis where Positive = Disease. Recall then answers the most critical question: "Did we catch all the dangerous water?"

### 4.5 `src/utils.py` — Utility Functions

| Function | What it does |
|----------|-------------|
| `load_config(path)` | Loads `config.yaml` and returns a dictionary |
| `set_seed(seed)` | Sets random seed for numpy, random, and sklearn for reproducibility |
| `save_model(model, path)` | Saves a trained model to disk using joblib |
| `ensure_dirs(config)` | Creates the artifacts directories if they don't exist |

### 4.6 `train.py` — Main Orchestrator

This is the script you run to train everything. It follows this sequence:

```
Step 1: Load config.yaml and set random seed
Step 2: Load CSV and split into 80% train / 20% test (stratified)
Step 3: Fit preprocessing on TRAIN ONLY:
        - Compute class-conditional medians → fill missing values
        - Compute outlier bounds → clip extreme values
        Apply to BOTH train and test
Step 4: For each of 4 models:
        - Run 60 trials of Optuna HPO (5-fold CV on train)
        - Build final model with best hyperparameters
        - Train on full training set
        - Evaluate on test set
        - Save confusion matrix, metrics
Step 5: Print model comparison table
Step 6: Select champion (best F1 among non-overfit models)
Step 7: Run SHAP explainability analysis on champion
Step 8: Save champion model to artifacts/
```

**Command to run:** `python train.py`

### 4.7 `api/main.py` — FastAPI Web Service

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/` | GET | Serves the web UI form (HTML page) |
| `/predict` | POST | Accepts JSON with 9 chemical readings, returns prediction |
| `/health` | GET | Returns model status (name, version) |
| `/docs` | GET | Auto-generated Swagger API documentation |

**Command to run:** `uvicorn api.main:app --reload --port 8000`

### 4.8 `api/templates/index.html` — Web UI

A dark-themed web form where users enter 9 chemical readings and click "Analyze Water Sample". The JavaScript sends a POST request to `/predict` and displays the result (Potable/Not Potable with confidence percentage).

---

## 5. Training Pipeline — How It Works

### 5.1 Data Flow (No Data Leakage)

```
Raw CSV (3276 samples)
       |
       v
train_test_split (stratified by Potability)
       |
       ├── Train set (2620 samples, 80%)
       |       |
       |       v
       |   fit_imputer() ──── compute medians from TRAIN ONLY
       |   fit_clipper()  ──── compute bounds from TRAIN ONLY
       |
       ├── Apply imputer to train ──> imputed train
       ├── Apply imputer to test  ──> imputed test (using TRAIN medians)
       ├── Apply clipper to train ──> clipped train
       ├── Apply clipper to test  ──> clipped test (using TRAIN bounds)
       |
       v
   9 clean features (no missing values, no extreme outliers)
```

### 5.2 Model Training (Inside ImbPipeline)

For each model, the ImbPipeline does this during `.fit(X_train, y_train)`:
1. **SMOTEENN**: Balance the classes (only on training data)
2. **StandardScaler**: Normalize all features to mean=0, std=1
3. **Model**: Train the classifier on balanced, scaled data

During `.predict(X_test)`:
1. SMOTEENN is **skipped** (never applied to test data)
2. StandardScaler transforms test data using train's mean/std
3. Model makes predictions

### 5.3 Overfitting Check

For each model, we compute:
- **Train AUC**: How well the model fits the training data
- **Test AUC**: How well the model performs on unseen data
- **Gap**: Train AUC - Test AUC

If Gap > 0.05, the model is **overfitting** (memorizing training data instead of learning patterns). Overfit models are rejected during champion selection.

---

## 6. Results

### 6.1 Model Comparison

| Model | CV AUC | Test AUC | Prec(0) | Rec(0) | F1(0) | Rec(1) | Gap | Status |
|-------|--------|----------|---------|--------|-------|--------|-----|--------|
| Logistic Regression | 0.4791 | 0.5395 | 0.6347 | 0.5125 | 0.5671 | 0.5391 | -0.036 | OK |
| SVM (RBF Kernel) | 0.6129 | 0.6190 | 0.6794 | 0.5775 | 0.6243 | 0.5742 | 0.077 | OVERFIT |
| Decision Tree | 0.7431 | 0.7433 | 0.7515 | 0.6350 | 0.6883 | 0.6719 | 0.024 | OK |
| **Random Forest** | **0.7613** | **0.8163** | **0.8078** | **0.7250** | **0.7642** | **0.7305** | **0.034** | **OK** |

### 6.2 Champion: Random Forest

- **Test AUC: 0.8163** — Strong ability to distinguish safe from unsafe water
- **Precision(0): 80.78%** — When we say water is unsafe, we are right 81% of the time
- **Recall(0): 72.50%** — We catch 72.5% of all actually unsafe water
- **Recall(1): 73.05%** — We correctly identify 73% of safe water
- **Overfit Gap: 0.034** — Healthy generalization (well below 0.05 threshold)

### 6.3 Best Hyperparameters (found by Optuna)

| Parameter | Value | What it controls |
|-----------|-------|-----------------|
| `n_estimators` | 364 | Number of trees in the forest |
| `max_depth` | 10 | Maximum depth of each tree (limits complexity) |
| `min_samples_split` | 15 | Minimum samples needed to split a node |
| `min_samples_leaf` | 6 | Minimum samples in each leaf node |
| `max_features` | sqrt | Number of features considered at each split |
| `class_weight` | balanced_subsample | Cost-sensitive learning per bootstrap sample |

---

## 7. Understanding the Plots

### 7.1 Confusion Matrix

A 2x2 grid showing:
```
                    Predicted
                Not Potable    Potable
Actual  Not Potable  [  TN  ]    [  FP  ]
        Potable      [  FN  ]    [  TP  ]
```

- **TN (True Negative)**: Correctly identified unsafe water as unsafe
- **FP (False Positive)**: Safe water incorrectly flagged as unsafe (false alarm — annoying but not dangerous)
- **FN (False Negative)**: Unsafe water incorrectly labeled safe (DANGEROUS — we missed a threat)
- **TP (True Positive)**: Correctly identified safe water as safe

Note: With Positive = Unsafe (Class 0):
- TN and TP swap meaning
- The top-left cell becomes TP (correctly caught unsafe water)

### 7.2 ROC Curve (Receiver Operating Characteristic)

- **X-axis**: False Positive Rate (how many safe samples we wrongly flagged)
- **Y-axis**: True Positive Rate (how many unsafe samples we correctly caught)
- **Diagonal line**: Random guessing (AUC = 0.5)
- **Closer to top-left corner**: Better model
- **AUC (Area Under Curve)**: Single number summarizing performance (0.5 = random, 1.0 = perfect)

Our Random Forest curve is well above the diagonal with AUC = 0.8163.

### 7.3 Precision-Recall Curve

- **X-axis**: Recall (how many unsafe samples we caught)
- **Y-axis**: Precision (when we say unsafe, how often we are right)
- Shows the trade-off: catching more unsafe water (higher recall) usually means more false alarms (lower precision)

### 7.4 Feature Importance Plot

Bar chart showing which chemical features the model uses most for its decisions. Higher bar = more important feature. This directly answers the project requirement "identify the most dangerous chemical factors."

### 7.5 SHAP Plots

SHAP (SHapley Additive exPlanations) shows how each feature contributes to individual predictions:
- **SHAP Summary Plot**: Each dot is a sample. Red = high feature value, Blue = low. Shows which direction each feature pushes the prediction.
- **SHAP Bar Plot**: Average absolute SHAP value per feature — ranks features by overall importance.

---

## 8. Deployment

### 8.1 Local API
```bash
uvicorn api.main:app --reload --port 8000
```
Then open `http://127.0.0.1:8000` in a browser.

### 8.2 Docker
```bash
docker build -t water-potability-api .
docker run -p 8000:8000 water-potability-api
```

### 8.3 AWS Elastic Beanstalk (Learner Lab)
1. Build and push Docker image to AWS ECR
2. Create Elastic Beanstalk environment (Docker platform)
3. Deploy using `Dockerrun.aws.json` pointing to ECR image

---

## 9. Key Design Decisions

| Decision | Reason |
|----------|--------|
| Split BEFORE preprocessing | Prevents data leakage — test set stays truly unseen |
| SMOTEENN inside ImbPipeline | Applied only during training, never during prediction |
| 60 Optuna trials | Enough to explore the search space without taking too long |
| Overfit threshold = 0.05 | Strict enough to reject memorizing models |
| Positive = Unsafe (Class 0) | Safety-critical framing — Recall answers "did we catch all dangerous water?" |
| 4 models compared | Shows the best model was chosen systematically, not assumed |
| Random seed = 42 | Standard convention for reproducibility |
