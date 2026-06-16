"""
predictor.py — Phase 3, Step 3
AI-Powered Satellite Drag Prediction and Orbital Decay Simulator

Drag Force Predictor
====================
This module loads the trained Random Forest model from disk and exposes
a single clean function — predict_drag_force() — that any other module
can call to get an ML-based drag force estimate.

HOW THIS FITS INTO THE PIPELINE
--------------------------------
    train_model.py
        saved → models/drag_predictor.pkl
                    ↓
            predictor.py  ← YOU ARE HERE
                loads model → validates inputs → returns prediction
                    ↓
            (future) simulator.py / Flask API route / comparison tool

WHY A SEPARATE PREDICTOR MODULE?
---------------------------------
Keeping prediction logic isolated from training logic means:
  - The Flask API, simulator, and any notebook can import one function.
  - The model can be swapped (e.g. XGBoost replacing Random Forest)
    without touching any caller.
  - Input validation lives in one place — not scattered across routes.
"""

import logging
import os
import pickle
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allows: python -m app.ML.predictor
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
# Model path
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "drag_predictor.pkl")


# ---------------------------------------------------------------------------
# Module-level model cache
#
# The model bundle is loaded once when first needed, then stored here.
# This avoids re-reading the pickle file on every prediction call —
# important when the predictor is used inside a Flask request handler
# that may be called thousands of times.
# ---------------------------------------------------------------------------
_model_bundle: dict | None = None


def _load_model(model_path: str = MODEL_PATH) -> dict:
    """
    Load the model bundle from disk and cache it in _model_bundle.

    The bundle is a dict saved by train_model.py containing:
        "model"         : trained RandomForestRegressor
        "feature_names" : list of feature column names (order matters)
        "target_name"   : name of the target column
        "metrics"       : MAE, RMSE, R² from training evaluation
        + training metadata

    Parameters
    ----------
    model_path : path to the .pkl file

    Returns
    -------
    dict — the full model bundle

    Raises
    ------
    FileNotFoundError : if the .pkl file does not exist
    ValueError        : if the bundle is missing required keys
    """
    global _model_bundle

    # Return cached version if already loaded
    if _model_bundle is not None:
        return _model_bundle

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Run train_model.py first to generate the model bundle."
        )

    logger.info("Loading model bundle from: %s", model_path)

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    # Validate that the bundle has the keys we depend on
    required_keys = ["model", "feature_names"]
    missing = [k for k in required_keys if k not in bundle]
    if missing:
        raise ValueError(
            f"Model bundle is missing required keys: {missing}. "
            "Re-run train_model.py to regenerate the bundle."
        )

    logger.info(
        "Model loaded successfully. Features: %s | "
        "Training metrics — MAE: %.4e, R²: %.4f",
        bundle["feature_names"],
        bundle.get("metrics", {}).get("mae", float("nan")),
        bundle.get("metrics", {}).get("r2",  float("nan")),
    )

    _model_bundle = bundle
    return _model_bundle


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    altitude_km: float,
    orbital_velocity_kms: float,
    f107: float,
    kp: float,
    ap: float,
) -> None:
    """
    Raise ValueError / TypeError if any input is outside physical bounds.

    Parameters are validated against the same ranges used during training
    (dataset_builder.py) so predictions are never made in extrapolation
    territory without an explicit warning.
    """
    params = {
        "altitude_km":           altitude_km,
        "orbital_velocity_kms":  orbital_velocity_kms,
        "f107":                  f107,
        "kp":                    kp,
        "ap":                    ap,
    }
    for name, val in params.items():
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"'{name}' must be numeric, got {type(val).__name__}"
            )

    if altitude_km < 0:
        raise ValueError(
            f"altitude_km must be >= 0, got {altitude_km}. "
            "Use 0 for ground level (re-entry)."
        )
    if orbital_velocity_kms <= 0:
        raise ValueError(
            f"orbital_velocity_kms must be > 0, got {orbital_velocity_kms}."
        )
    if not (60.0 <= f107 <= 300.0):
        raise ValueError(
            f"f107 should be between 60 and 300 sfu, got {f107}. "
            "Typical range: 70 (solar min) to 230 (solar max)."
        )
    if not (0.0 <= kp <= 9.0):
        raise ValueError(
            f"kp must be between 0 and 9, got {kp}."
        )
    if ap < 0:
        raise ValueError(
            f"ap must be >= 0, got {ap}."
        )


# ---------------------------------------------------------------------------
# Public prediction function
# ---------------------------------------------------------------------------

def predict_drag_force(
    altitude_km: float,
    orbital_velocity_kms: float,
    f107: float,
    kp: float,
    ap: float,
    model_path: str = MODEL_PATH,
) -> dict:
    """
    Predict the atmospheric drag force on a satellite using the trained
    Random Forest model.

    This is the single public entry point for ML-based drag prediction.
    It can be called from the Flask API, simulator, or any notebook.

    Parameters
    ----------
    altitude_km           : Satellite altitude above Earth's surface [km]
    orbital_velocity_kms  : Orbital speed [km/s]
    f107                  : F10.7 solar flux index [sfu], range 60–300
    kp                    : Planetary geomagnetic index [0–9]
    ap                    : Equivalent linear geomagnetic index [nT], >= 0
    model_path            : Override path to the .pkl bundle (optional)

    Returns
    -------
    dict with keys:
        "predicted_drag_force_N"  : float — predicted drag force [Newtons]
        "inputs"                  : dict  — echo of validated inputs
        "model_features"          : list  — feature order used by the model

    Raises
    ------
    TypeError         : if any input is non-numeric
    ValueError        : if any input is outside physical bounds
    FileNotFoundError : if the model bundle does not exist

    Example
    -------
    >>> result = predict_drag_force(
    ...     altitude_km=408.0,
    ...     orbital_velocity_kms=7.67,
    ...     f107=150.0,
    ...     kp=2.0,
    ...     ap=15.0,
    ... )
    >>> print(result["predicted_drag_force_N"])
    """

    # ------------------------------------------------------------------
    # Step 1: Validate inputs before touching the model
    # Fail fast with a clear message rather than letting sklearn raise
    # a cryptic array-shape error later.
    # ------------------------------------------------------------------
    _validate_inputs(altitude_km, orbital_velocity_kms, f107, kp, ap)

    # ------------------------------------------------------------------
    # Step 2: Load the model bundle (cached after first call)
    # ------------------------------------------------------------------
    bundle        = _load_model(model_path)
    model         = bundle["model"]
    feature_names = bundle["feature_names"]   # e.g. ["altitude_km", "orbital_velocity_kms", ...]

    # ------------------------------------------------------------------
    # Step 3: Build a single-row DataFrame using the stored feature_names.
    #
    # WHY A DATAFRAME AND NOT A PLAIN ARRAY?
    # sklearn's RandomForestRegressor was trained on a DataFrame, so it
    # expects column names in exactly the same order. Passing a raw numpy
    # array would work numerically but bypasses this safety check.
    # Building a DataFrame with the stored feature_names guarantees the
    # column order is always correct, even if we add features later.
    # ------------------------------------------------------------------
    input_data = {
        "altitude_km":           [altitude_km],
        "orbital_velocity_kms":  [orbital_velocity_kms],
        "f107":                  [f107],
        "kp":                    [kp],
        "ap":                    [ap],
    }

    # Select and order columns exactly as the model was trained on
    input_df = pd.DataFrame(input_data)[feature_names]

    # ------------------------------------------------------------------
    # Step 4: Run model.predict()
    #
    # predict() returns a numpy array of shape (1,).
    # We extract the single scalar value with [0].
    # ------------------------------------------------------------------
    prediction_array = model.predict(input_df)
    predicted_force  = float(prediction_array[0])

    logger.debug(
        "predict_drag_force | alt=%.1f km, v=%.3f km/s, "
        "F10.7=%.1f, Kp=%.1f, Ap=%.1f → F_pred=%.4e N",
        altitude_km, orbital_velocity_kms, f107, kp, ap, predicted_force,
    )

    # ------------------------------------------------------------------
    # Step 5: Return a structured result dict
    #
    # Returning a dict (not just the float) lets callers log inputs and
    # outputs together, and makes it easy to extend with confidence
    # intervals or physics-baseline comparisons later.
    # ------------------------------------------------------------------
    return {
        "predicted_drag_force_N": predicted_force,
        "inputs": {
            "altitude_km":          altitude_km,
            "orbital_velocity_kms": orbital_velocity_kms,
            "f107":                 f107,
            "kp":                   kp,
            "ap":                   ap,
        },
        "model_features": feature_names,
    }


# ---------------------------------------------------------------------------
# Entry point — demo prediction
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    SEP = "=" * 58

    print(SEP)
    print("  Phase 3 Step 3 — Drag Force Predictor Demo")
    print(SEP)

    # Demo inputs: ISS-like orbit, moderate space weather
    DEMO_INPUTS = {
        "altitude_km":          408.0,
        "orbital_velocity_kms":   7.67,
        "f107":                 150.0,
        "kp":                     2.0,
        "ap":                    15.0,
    }

    print("\n  Input parameters:")
    for key, val in DEMO_INPUTS.items():
        print(f"    {key:<26} : {val}")

    try:
        result = predict_drag_force(**DEMO_INPUTS)

        print(f"\n  Predicted drag force : {result['predicted_drag_force_N']:.6e} N")
        print(f"  Feature order used   : {result['model_features']}")

        # ------------------------------------------------------------------
        # Physics sanity check: compare ML prediction to the analytical
        # drag equation using the atmosphere module.
        # F = 0.5 * Cd * rho * A * v²
        # ------------------------------------------------------------------
        try:
            from app.physics.atmosphere import estimate_density_with_space_weather
            from app.physics.drag_calculator import compute_drag

            weather = estimate_density_with_space_weather(
                altitude_km=DEMO_INPUTS["altitude_km"],
                f107=DEMO_INPUTS["f107"],
                kp=DEMO_INPUTS["kp"],
            )
            physics_result = compute_drag(
                cd=2.2,
                rho=weather["density_kg_m3"],
                area=10.0,
                velocity=DEMO_INPUTS["orbital_velocity_kms"] * 1000.0,
                mass=500.0,
            )
            physics_force = physics_result["drag_force_N"]

            print(f"\n  Physics baseline     : {physics_force:.6e} N")
            ratio = result["predicted_drag_force_N"] / physics_force if physics_force else float("nan")
            print(f"  ML / Physics ratio   : {ratio:.4f}  "
                  f"({'close' if 0.5 <= ratio <= 2.0 else 'diverged — may need more training data'})")
        except Exception as physics_err:
            print(f"\n  (Physics comparison skipped: {physics_err})")

    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    print(f"\n{SEP}")
    print("  predictor.py is ready for Flask API and simulator integration.")
    print(SEP)