"""
train_model.py — Phase 3, Step 2
AI-Powered Satellite Drag Prediction and Orbital Decay Simulator

Machine Learning Model Trainer
================================
This module loads the training dataset built by dataset_builder.py,
trains a Random Forest regression model to predict atmospheric drag force,
evaluates it, and saves the trained model to disk.

WHY RANDOM FOREST FOR DRAG PREDICTION?
---------------------------------------
Drag force depends on atmospheric density, which varies non-linearly with
altitude, solar flux (F10.7), and geomagnetic activity (Kp, Ap). A Random
Forest handles these non-linear relationships well without requiring feature
scaling or explicit interaction terms. It also provides feature importance
scores that let us validate the physics (altitude and velocity should
dominate, as they do in F = 0.5 * Cd * rho * A * v²).

PIPELINE
--------
    data/training_dataset.csv   (from dataset_builder.py)
        ↓
    train_model.py  ← YOU ARE HERE
        load → split → train → evaluate → save
        ↓
    models/drag_predictor.pkl   (consumed by predictor.py in Phase 3 Step 3)
"""

import logging
import math
import os
import pickle
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Path setup — allows running with: python -m app.ML.train_model
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "training_dataset_large.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH   = os.path.join(MODELS_DIR, "drag_predictor.pkl")

# ---------------------------------------------------------------------------
# Feature and target column names
# These must match the columns written by dataset_builder.py exactly.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "altitude_km",
    "orbital_velocity_kms",
    "f107",
    "kp",
    "ap",
]
TARGET_COLUMN = "target_drag_force_N"

# ---------------------------------------------------------------------------
# Model hyperparameters
# n_estimators : number of decision trees in the forest
#                More trees = more stable predictions, slower training.
#                100 is a solid default for small datasets.
# random_state : seed for reproducibility — same split and tree structure
#                every run, so results are comparable across experiments.
# n_jobs       : use all available CPU cores for parallel tree training.
# ---------------------------------------------------------------------------
N_ESTIMATORS = 100
RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 80% train / 20% test split


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_model(
    dataset_path: str = DATASET_PATH,
    model_path:   str = MODEL_PATH,
    n_estimators: int = N_ESTIMATORS,
    test_size:  float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Load the training dataset, train a RandomForestRegressor, evaluate it,
    and save the model to disk.

    Parameters
    ----------
    dataset_path : Path to the CSV produced by dataset_builder.py
    model_path   : Output path for the pickled model bundle
    n_estimators : Number of trees in the Random Forest
    test_size    : Fraction of data held out for evaluation (0–1)
    random_state : Random seed for reproducibility

    Returns
    -------
    dict with keys:
        "mae"            : float — Mean Absolute Error [N]
        "rmse"           : float — Root Mean Squared Error [N]
        "r2"             : float — R² score (1.0 = perfect)
        "n_train"        : int   — training sample count
        "n_test"         : int   — test sample count
        "model_path"     : str   — path where model was saved
        "feature_names"  : list  — feature columns used

    Raises
    ------
    FileNotFoundError : If the dataset CSV does not exist
    ValueError        : If required columns are missing or dataset is too small
    """

    SEP = "=" * 62

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}\n"
            "Run dataset_builder.py first to generate the training data."
        )

    logger.info("Loading dataset from: %s", dataset_path)
    df = pd.read_csv(dataset_path)
    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))

    # ------------------------------------------------------------------
    # 2. Validate required columns are present
    # ------------------------------------------------------------------
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # ------------------------------------------------------------------
    # 3. Check minimum sample count
    #    With fewer than 10 samples a train/test split is unreliable.
    #    Warn loudly but don't crash — useful for debugging pipelines.
    # ------------------------------------------------------------------
    if len(df) < 10:
        logger.warning(
            "Only %d samples found. Model performance metrics will be "
            "unreliable. Consider adding more data via ingestion pipelines "
            "or simulator trajectory export.", len(df)
        )
    if len(df) < 2:
        raise ValueError(
            f"Dataset has only {len(df)} row(s). Cannot train a model. "
            "Run dataset_builder.py after ingesting more satellite data."
        )

    # ------------------------------------------------------------------
    # 4. Extract features (X) and target (y)
    #
    #    X : 2-D array of shape (n_samples, n_features)
    #    y : 1-D array of shape (n_samples,) — drag force in Newtons
    # ------------------------------------------------------------------
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Drop rows where any feature or target is NaN
    combined = pd.concat([X, y], axis=1)
    before   = len(combined)
    combined.dropna(inplace=True)
    if len(combined) < before:
        logger.warning(
            "Dropped %d row(s) containing NaN values.", before - len(combined)
        )
    X = combined[FEATURE_COLUMNS]
    y = combined[TARGET_COLUMN]

    # ------------------------------------------------------------------
    # 5. Train / test split
    #
    #    WHY SPLIT THE DATA?
    #    We train the model on one portion and evaluate it on a held-out
    #    portion the model has never seen. This tells us whether the model
    #    has genuinely learned the relationship or merely memorised the
    #    training data (overfitting). An overfit model scores perfectly on
    #    training data but fails on new inputs — useless for prediction.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info(
        "Dataset split: %d training samples, %d test samples.",
        len(X_train), len(X_test),
    )

    # ------------------------------------------------------------------
    # 6. Train the Random Forest
    #
    #    A Random Forest builds N decision trees, each on a random
    #    bootstrap sample of the training data and a random subset of
    #    features at each split. Predictions are averaged across all
    #    trees, which reduces variance (overfitting) compared to a
    #    single deep tree.
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  Phase 3 Step 2 — Training Drag Force Predictor")
    print(SEP)
    print(f"  Dataset       : {dataset_path}")
    print(f"  Samples       : {len(X)} total  ({len(X_train)} train / {len(X_test)} test)")
    print(f"  Features      : {FEATURE_COLUMNS}")
    print(f"  Target        : {TARGET_COLUMN}")
    print(f"  Model         : RandomForestRegressor (n_estimators={n_estimators})")
    print(f"  Random seed   : {random_state}")
    print(SEP)
    print("\n  Training... ", end="", flush=True)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,          # use all CPU cores
    )
    model.fit(X_train, y_train)
    print("done.")

    # ------------------------------------------------------------------
    # 7. Evaluate on the held-out test set
    #
    #    MAE  (Mean Absolute Error)
    #         Average absolute difference between predicted and actual
    #         drag force. Easy to interpret: "on average, predictions
    #         are off by X Newtons."
    #
    #    RMSE (Root Mean Squared Error)
    #         Like MAE but penalises large errors more heavily (squaring
    #         magnifies outliers). Lower is better.
    #
    #    R²   (Coefficient of Determination)
    #         Fraction of target variance explained by the model.
    #         1.0 = perfect, 0.0 = no better than predicting the mean,
    #         < 0 = worse than the mean (model is broken).
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # ------------------------------------------------------------------
    # 8. Feature importances
    #    Random Forest computes how much each feature reduces prediction
    #    error across all trees. Higher = more important.
    #    We expect altitude_km and orbital_velocity_kms to dominate
    #    because drag scales as rho(altitude) × v².
    # ------------------------------------------------------------------
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    sorted_imp  = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # 9. Print evaluation report
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  Evaluation Metrics (test set)")
    print(SEP)
    print(f"  MAE   : {mae:.6e} N")
    print(f"  RMSE  : {rmse:.6e} N")
    print(f"  R²    : {r2:.6f}  {'✓ good fit' if r2 >= 0.8 else '⚠ needs more data'}")
    print(f"\n  Feature Importances:")
    for feat, imp in sorted_imp:
        bar = "█" * int(imp * 40)
        print(f"    {feat:<26} {imp:.4f}  {bar}")
    print(SEP)

    # ------------------------------------------------------------------
    # 10. Save model bundle (model + feature names) to disk
    #
    #     We pickle a dict rather than the bare model so predictor.py
    #     can verify it is loading the right feature set and model version
    #     without needing a separate metadata file.
    # ------------------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_bundle = {
        "model":          model,
        "feature_names":  FEATURE_COLUMNS,
        "target_name":    TARGET_COLUMN,
        "n_estimators":   n_estimators,
        "random_state":   random_state,
        "train_samples":  len(X_train),
        "test_samples":   len(X_test),
        "metrics": {
            "mae":  mae,
            "rmse": rmse,
            "r2":   r2,
        },
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    logger.info("Model bundle saved to: %s", model_path)

    print(f"\n  Model saved to  : {model_path}")
    print(f"  Bundle contains : model, feature_names, metrics, metadata")
    print(f"{SEP}\n")

    return {
        "mae":           mae,
        "rmse":          rmse,
        "r2":            r2,
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "model_path":    model_path,
        "feature_names": FEATURE_COLUMNS,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        results = train_model()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)