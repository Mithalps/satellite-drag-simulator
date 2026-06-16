"""
analysis.py — Phase 6, AI Explainability
AI-Powered Satellite Drag Prediction — Mission Intelligence Extension

New Blueprint: /api/analysis
────────────────────────────
Adds ONE new endpoint that exposes the trained model's feature importances
and metadata so the frontend XAI panel can render SHAP-approximate bars.

EXISTING ENDPOINTS ARE UNTOUCHED.
This file is registered as a separate blueprint alongside api_bp.

Register in create_app() with:
    from app.api.analysis import analysis_bp
    app.register_blueprint(analysis_bp)
"""

import logging
import os
import pickle

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

analysis_bp = Blueprint("analysis", __name__, url_prefix="/api")

# Path to the trained model bundle (same path used by predictor.py)
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "drag_predictor.pkl"
)
_bundle_cache: dict | None = None


def _load_bundle() -> dict | None:
    """Load and cache the model bundle. Returns None if file not found."""
    global _bundle_cache
    if _bundle_cache is not None:
        return _bundle_cache
    path = os.path.abspath(_MODEL_PATH)
    if not os.path.exists(path):
        logger.warning("Model bundle not found at %s", path)
        return None
    with open(path, "rb") as f:
        _bundle_cache = pickle.load(f)
    return _bundle_cache


# ─────────────────────────────────────────────────────────────
# GET /api/model-info
# ─────────────────────────────────────────────────────────────

@analysis_bp.route("/model-info", methods=["GET"])
def model_info():
    """
    Return model metadata and feature importances for the XAI panel.

    Response (200)
    ──────────────
    {
        "feature_names":   ["altitude_km", "orbital_velocity_kms", ...],
        "importances":     [0.61, 0.23, 0.10, 0.04, 0.02],   // sum ≈ 1.0
        "importances_pct": [61, 23, 10, 4, 2],               // rounded %
        "n_estimators":    100,
        "train_samples":   10000,
        "metrics": {
            "mae":  ...,
            "rmse": ...,
            "r2":   ...
        },
        "source": "random_forest_feature_importances"
    }

    If the model bundle is unavailable, returns a physics-prior fallback
    so the frontend always has something meaningful to display.
    """
    bundle = _load_bundle()

    if bundle is not None:
        model         = bundle["model"]
        feature_names = bundle.get("feature_names", [])
        importances   = list(model.feature_importances_)

        # Normalise to percentages (RF importances already sum to 1.0)
        total = sum(importances) or 1.0
        importances_pct = [round(v / total * 100, 1) for v in importances]

        return jsonify({
            "feature_names":   feature_names,
            "importances":     [round(v, 6) for v in importances],
            "importances_pct": importances_pct,
            "n_estimators":    bundle.get("n_estimators", "unknown"),
            "train_samples":   bundle.get("train_samples", "unknown"),
            "metrics":         bundle.get("metrics", {}),
            "source":          "random_forest_feature_importances",
        }), 200

    # ── Physics-prior fallback (no model on disk) ──────────────
    # Based on sensitivity analysis of F = 0.5·Cd·ρ(alt,sw)·A·v²:
    #   altitude dominates through exponential density scaling (~61%)
    #   velocity contributes quadratically (~23%)
    #   F10.7 modulates thermosphere expansion (~10%)
    #   Kp drives geomagnetic storm heating (~4%)
    #   Ap is correlated with Kp (~2%)
    logger.warning("Serving physics-prior importances — model bundle missing.")
    return jsonify({
        "feature_names":   ["altitude_km", "orbital_velocity_kms", "f107", "kp", "ap"],
        "importances":     [0.61, 0.23, 0.10, 0.04, 0.02],
        "importances_pct": [61, 23, 10, 4, 2],
        "n_estimators":    "N/A",
        "train_samples":   "N/A",
        "metrics":         {},
        "source":          "physics_prior_fallback",
    }), 200