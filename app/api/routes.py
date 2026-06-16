"""
routes.py — Flask REST API
AI-Powered Satellite Drag Prediction and Orbital Decay Simulator

This Blueprint wires HTTP endpoints to the existing physics and ML modules.
It contains ZERO physics equations and ZERO ML code — all logic is
delegated to the modules that were built in earlier phases.

ENDPOINT SUMMARY
----------------
  GET  /             — project info
  GET  /health       — liveness check
  POST /predict      — ML drag force prediction  (predictor.py)
  POST /simulate     — orbital decay simulation  (orbit_propagator.py)

HOW TO REGISTER THIS BLUEPRINT
-------------------------------
In your create_app() factory (app/__init__.py), add:

    from app.api.routes import api_bp
    app.register_blueprint(api_bp)

That is the only change needed in the existing codebase.

CURL EXAMPLES (run from any terminal after `flask run`)
-------------------------------------------------------
# 1. Project info
curl http://127.0.0.1:5000/

# 2. Health check
curl http://127.0.0.1:5000/health

# 3. Drag force prediction
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"altitude_km":408,"orbital_velocity_kms":7.67,"f107":150,"kp":2,"ap":15}'

# 4. Orbital decay simulation
curl -X POST http://127.0.0.1:5000/simulate \
     -H "Content-Type: application/json" \
     -d '{"altitude_km":408,"steps":100,"dt":10,"mass":420000,"cd":2.2,"area":2500,"f107":150,"kp":2}'
"""

import logging

from flask import Blueprint, jsonify, request

# ---------------------------------------------------------------------------
# Import existing modules — no physics or ML logic lives in this file
# ---------------------------------------------------------------------------
# ML predictor (Phase 3 Step 3)
from app.ML.predictor import predict_drag_force

# Orbit propagator (Phase 2 Step 3) — propagate_step does all the physics
from app.physics.orbit_propagator import circular_orbit_state, propagate_step

# ---------------------------------------------------------------------------
# Blueprint definition
# All routes in this file are prefixed by whatever the app registers this
# blueprint under. Here we use no url_prefix so routes are at root (/).
# ---------------------------------------------------------------------------
api_bp = Blueprint("api", __name__)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_json():
    """
    Return the parsed JSON body, or raise a 400 error if the request
    body is missing or not valid JSON.
    """
    data = request.get_json(silent=True)
    if data is None:
        return None, (
            jsonify({"error": "Request body must be JSON with "
                              "Content-Type: application/json"}),
            400,
        )
    return data, None


def _get_float(data: dict, key: str, min_val=None, max_val=None,
               required=True, default=None):
    """
    Extract a float from the request dict with optional range validation.

    Returns (value, error_response) where error_response is None on success.
    """
    if key not in data:
        if required:
            return None, (
                jsonify({"error": f"Missing required field: '{key}'"}), 400
            )
        return default, None

    try:
        value = float(data[key])
    except (TypeError, ValueError):
        return None, (
            jsonify({"error": f"Field '{key}' must be a number, "
                              f"got: {data[key]!r}"}), 400
        )

    if min_val is not None and value < min_val:
        return None, (
            jsonify({"error": f"'{key}' must be >= {min_val}, got {value}"}),
            400,
        )
    if max_val is not None and value > max_val:
        return None, (
            jsonify({"error": f"'{key}' must be <= {max_val}, got {value}"}),
            400,
        )

    return value, None


def _get_int(data: dict, key: str, min_val=None, max_val=None,
             required=True, default=None):
    """
    Extract an integer from the request dict with optional range validation.
    """
    if key not in data:
        if required:
            return None, (
                jsonify({"error": f"Missing required field: '{key}'"}), 400
            )
        return default, None

    try:
        value = int(data[key])
    except (TypeError, ValueError):
        return None, (
            jsonify({"error": f"Field '{key}' must be an integer, "
                              f"got: {data[key]!r}"}), 400
        )

    if min_val is not None and value < min_val:
        return None, (
            jsonify({"error": f"'{key}' must be >= {min_val}, got {value}"}),
            400,
        )
    if max_val is not None and value > max_val:
        return None, (
            jsonify({"error": f"'{key}' must be <= {max_val}, got {value}"}),
            400,
        )

    return value, None


# ---------------------------------------------------------------------------
# Endpoint 1 — GET /
# ---------------------------------------------------------------------------

@api_bp.route("/", methods=["GET"])
def index():
    """
    Project information endpoint.
    Always returns 200 — useful as a quick "is the server up?" check.
    """
    return jsonify({
        "project": "AI Satellite Drag Prediction",
        "status":  "running",
        "version": "1.0",
        "endpoints": {
            "GET  /":         "Project info",
            "GET  /health":   "Health check",
            "POST /predict":  "ML drag force prediction",
            "POST /simulate": "Orbital decay simulation",
        },
    }), 200


# ---------------------------------------------------------------------------
# Endpoint 2 — GET /health
# ---------------------------------------------------------------------------

@api_bp.route("/health", methods=["GET"])
def health():
    """
    Liveness check endpoint.
    Load balancers and container orchestrators (Docker, K8s) call this to
    decide whether the container is healthy.
    """
    return jsonify({"status": "healthy"}), 200


# ---------------------------------------------------------------------------
# Endpoint 3 — POST /predict
# ---------------------------------------------------------------------------

@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    ML drag force prediction.

    Accepts orbital and space weather parameters, delegates to
    predict_drag_force() in predictor.py, and returns the result.

    Request body (JSON)
    -------------------
    {
        "altitude_km":          408,    // km,  >= 100
        "orbital_velocity_kms": 7.67,   // km/s, > 0
        "f107":                 150,    // sfu,  60–300
        "kp":                   2,      // 0–9
        "ap":                   15      // nT,   >= 0
    }

    Success response (200)
    ----------------------
    {
        "predicted_drag_force_N": 3.45e-4,
        "model_info": {
            "features_used": [...],
            "inputs": {...}
        }
    }
    """
    # --- parse body ---
    data, err = _require_json()
    if err:
        return err

    # --- validate each field ---
    altitude_km, err = _get_float(data, "altitude_km",          min_val=100.0)
    if err: return err

    velocity_kms, err = _get_float(data, "orbital_velocity_kms", min_val=0.01)
    if err: return err

    f107, err = _get_float(data, "f107", min_val=60.0, max_val=300.0)
    if err: return err

    kp, err = _get_float(data, "kp", min_val=0.0, max_val=9.0)
    if err: return err

    ap, err = _get_float(data, "ap", min_val=0.0)
    if err: return err

    # --- delegate to ML predictor — no equations here ---
    try:
        result = predict_drag_force(
            altitude_km=altitude_km,
            orbital_velocity_kms=velocity_kms,
            f107=f107,
            kp=kp,
            ap=ap,
        )
    except (ValueError, TypeError) as exc:
        logger.warning("Prediction input error: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
        return jsonify({"error": "Model not loaded. Run train_model.py first.",
                        "detail": str(exc)}), 503
    except Exception as exc:
        logger.exception("Unexpected prediction error")
        return jsonify({"error": "Internal prediction error",
                        "detail": str(exc)}), 500

    return jsonify({
        "predicted_drag_force_N": result["predicted_drag_force_N"],
        "model_info": {
            "features_used": result["model_features"],
            "inputs":        result["inputs"],
        },
    }), 200


# ---------------------------------------------------------------------------
# Endpoint 4 — POST /simulate
# ---------------------------------------------------------------------------

@api_bp.route("/simulate", methods=["POST"])
def simulate():
    """
    Orbital decay simulation.

    Initialises a circular orbit and calls propagate_step() in a loop for
    the requested number of steps. All physics live in orbit_propagator.py.

    Request body (JSON)
    -------------------
    {
        "altitude_km": 408,      // km,  >= 100
        "steps":       100,      // 1–10000
        "dt":          10,       // seconds per step, 1–300
        "mass":        420000,   // kg,  > 0
        "cd":          2.2,      // > 0
        "area":        2500,     // m²,  > 0
        "f107":        150,      // sfu, 60–300
        "kp":          2         // 0–9
    }

    Success response (200)
    ----------------------
    {
        "final_altitude_km":  407.98,
        "final_velocity_ms":  7660.1,
        "total_time_s":       1000,
        "reentry":            false,
        "steps_completed":    100,
        "trajectory_points":  [
            {"step": 1, "time_s": 0, "altitude_km": 408.0,
             "velocity_ms": 7660.2, "drag_force_N": 1.2e-2},
            ...
        ]
    }
    """
    # --- parse body ---
    data, err = _require_json()
    if err:
        return err

    # --- validate fields ---
    altitude_km, err = _get_float(data, "altitude_km", min_val=100.0)
    if err: return err

    steps, err = _get_int(data, "steps", min_val=1, max_val=10_000)
    if err: return err

    dt, err = _get_float(data, "dt", min_val=1.0, max_val=300.0)
    if err: return err

    mass, err = _get_float(data, "mass", min_val=0.001)
    if err: return err

    cd, err = _get_float(data, "cd", min_val=0.001)
    if err: return err

    area, err = _get_float(data, "area", min_val=0.001)
    if err: return err

    f107, err = _get_float(data, "f107", min_val=60.0, max_val=300.0)
    if err: return err

    kp, err = _get_float(data, "kp", min_val=0.0, max_val=9.0)
    if err: return err

    # --- initialise circular orbit ---
    try:
        state = circular_orbit_state(altitude_km)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    x,  y,  z  = state["x"],  state["y"],  state["z"]
    vx, vy, vz = state["vx"], state["vy"], state["vz"]

    # --- run propagation loop — physics stay in orbit_propagator ---
    trajectory = []
    result      = {}
    completed   = 0

    try:
        for step in range(1, steps + 1):
            result = propagate_step(
                x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz,
                mass=mass,
                cd=cd,
                area=area,
                dt=dt,
                f107=f107,
                kp=kp,
            )

            # Record a lightweight trajectory point (not the full dict,
            # to keep the response payload a reasonable size)
            trajectory.append({
                "step":         step,
                "time_s":       round((step - 1) * dt, 2),
                "altitude_km":  round(result["altitude_km"], 4),
                "velocity_ms":  round(result["velocity_mag_ms"], 4),
                "drag_force_N": result["drag_force_N"],
            })

            # Unpack new state for next iteration
            x,  y,  z  = result["x"],  result["y"],  result["z"]
            vx, vy, vz = result["vx"], result["vy"], result["vz"]
            completed   = step

            if result["reentry"]:
                break

    except (ValueError, TypeError) as exc:
        logger.warning("Simulation error at step %d: %s", completed, exc)
        return jsonify({"error": str(exc), "steps_completed": completed}), 400
    except Exception as exc:
        logger.exception("Unexpected simulation error at step %d", completed)
        return jsonify({"error": "Internal simulation error",
                        "detail": str(exc),
                        "steps_completed": completed}), 500

    return jsonify({

    "final_altitude_km": result["altitude_km"],

    "final_velocity_ms": result["velocity_mag_ms"],

    "reentry": result["reentry"],

    "steps_completed": completed,

    "total_time_s": completed * dt,

    "trajectory_points": trajectory

}), 200