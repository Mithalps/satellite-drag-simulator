"""
dataset_builder.py — Phase 3, Step 1
AI-Powered Satellite Drag Prediction and Orbital Decay Simulator

ML Dataset Builder
==================
This module reads FeatureRecord rows from the SQLite database (built by
Phase 1's feature engineering pipeline) and converts them into a clean
pandas DataFrame suitable for training a drag-prediction ML model.

WHAT THIS MODULE DOES
---------------------
    Phase 1 SQLite DB (FeatureRecord table)
        ↓
    dataset_builder.py  ← YOU ARE HERE
        reads rows → extracts features → computes target via drag_calculator
        ↓
    data/training_dataset.csv
        ↓
    train_model.py (Phase 3 Step 2)

WHY COMPUTE THE TARGET HERE (NOT STORE IT IN THE DB)?
------------------------------------------------------
The drag force is a deterministic function of features already in the DB
(velocity, density, area, Cd). Storing it separately would duplicate data
and create a maintenance risk if the drag equation is updated. Instead we
compute it on-the-fly using compute_drag() from drag_calculator.py — the
single source of truth for all drag physics in this project.

ASSUMPTIONS ABOUT FeatureRecord SCHEMA
----------------------------------------
The following columns are expected (set by Phase 1 feature_engineer.py):
    timestamp           — ISO datetime string or Unix float
    altitude_km         — satellite altitude [km]
    orbital_velocity_kms— orbital speed [km/s]
    f107                — F10.7 solar flux index [sfu]
    kp                  — geomagnetic Kp index [0–9]
    ap                  — geomagnetic Ap index [nT]

Satellite physical parameters (Cd, area, mass) are not stored per-row in
FeatureRecord; representative ISS-like defaults are used for the target
computation. These can be overridden via build_training_dataset() kwargs.
"""

import logging
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Adjust sys.path so this module can be run with:
#   python -m app.ml.dataset_builder
# from the project root without install.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import create_app          # Flask application factory
from app.models import FeatureRecord  # SQLAlchemy model from Phase 1
from app.physics.drag_calculator import compute_drag  # Phase 2 — no duplicate equations

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "training_dataset.csv")

# ---------------------------------------------------------------------------
# Feature columns to extract from FeatureRecord
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "timestamp",
    "altitude_km",
    "orbital_velocity_kms",
    "f107",
    "kp",
    "ap",
]

# ---------------------------------------------------------------------------
# Default satellite parameters for target computation
# These represent a generic LEO satellite; override via function kwargs.
# ---------------------------------------------------------------------------
DEFAULT_CD     = 2.2       # drag coefficient (dimensionless)
DEFAULT_AREA   = 10.0      # cross-sectional area [m²]
DEFAULT_MASS   = 500.0     # satellite mass [kg]


# ---------------------------------------------------------------------------
# Helper: safely extract a numeric value from a FeatureRecord attribute
# ---------------------------------------------------------------------------

def _safe_float(value, field_name: str, row_index: int) -> float:
    """
    Convert a value to float, raising a clear error if it is None or
    non-numeric.

    Parameters
    ----------
    value      : raw value from the database row
    field_name : column name (for error messages)
    row_index  : row number (for error messages)

    Returns
    -------
    float
    """
    if value is None:
        raise ValueError(
            f"Row {row_index}: '{field_name}' is NULL in the database. "
            "Check the feature engineering pipeline."
        )
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Row {row_index}: '{field_name}' = {value!r} cannot be "
            "converted to float."
        )


# ---------------------------------------------------------------------------
# Core dataset builder
# ---------------------------------------------------------------------------

def build_training_dataset(
    cd: float   = DEFAULT_CD,
    area: float = DEFAULT_AREA,
    mass: float = DEFAULT_MASS,
    output_path: str = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Read all FeatureRecord rows from the SQLite database, compute the
    drag force target for each row, and save the result as a CSV.

    Parameters
    ----------
    cd          : Drag coefficient used for target computation (default 2.2)
    area        : Cross-sectional area [m²] (default 10.0)
    mass        : Satellite mass [kg] (default 500.0)
    output_path : Full path for the output CSV file

    Returns
    -------
    pd.DataFrame
        Dataset with FEATURE_COLUMNS + "target_drag_force_N" column.

    Raises
    ------
    ValueError  : If the database contains no FeatureRecord rows, or if
                  required columns contain NULL / non-numeric values.
    RuntimeError: If the database query fails unexpectedly.
    """

    # ------------------------------------------------------------------
    # 1. Validate satellite parameters before touching the database
    # ------------------------------------------------------------------
    if cd <= 0:
        raise ValueError(f"Drag coefficient cd must be > 0, got {cd}")
    if area <= 0:
        raise ValueError(f"Cross-sectional area must be > 0, got {area}")
    if mass <= 0:
        raise ValueError(f"Satellite mass must be > 0, got {mass}")

    # ------------------------------------------------------------------
    # 2. Query FeatureRecord table inside a Flask application context.
    #
    #    WHY application context?
    #    SQLAlchemy in Flask binds the database session to the app context.
    #    Without it, db.session.query() raises a RuntimeError. We push a
    #    context here so this script can be run standalone (python -m ...)
    #    without a running Flask server.
    # ------------------------------------------------------------------
    app = create_app()

    with app.app_context():
        logger.info("Querying FeatureRecord table from database...")

        try:
            records = FeatureRecord.query.all()
        except Exception as exc:
            raise RuntimeError(
                f"Database query failed: {exc}. "
                "Ensure Phase 1 ingestion and feature engineering have been run."
            ) from exc

        if not records:
            raise ValueError(
                "No FeatureRecord rows found in the database. "
                "Run the satellite and space weather ingestion pipelines first."
            )

        logger.info("Found %d FeatureRecord rows. Building dataset...", len(records))

        # ------------------------------------------------------------------
        # 3. Extract features and compute drag target for each row
        #
        #    For each record we:
        #      a) Pull the raw feature values from the ORM object
        #      b) Convert orbital velocity from km/s to m/s (SI units)
        #      c) Use atmosphere.py indirectly: rho is approximated from
        #         altitude via a density lookup embedded in the feature record
        #         OR we use the exponential model directly.
        #      d) Call compute_drag() — single source of truth for F_drag
        # ------------------------------------------------------------------
        rows = []
        skipped = 0

        for i, rec in enumerate(records):
            try:
                # --- extract feature values ---
                timestamp    = rec.timestamp   # kept as-is (string or datetime)
                altitude_km  = _safe_float(rec.altitude_km,           "altitude_km",           i)
                vel_kms      = _safe_float(rec.orbital_velocity_kms,  "orbital_velocity_kms",  i)
                f107_val     = _safe_float(rec.f107,                  "f107",                  i)
                kp_val       = _safe_float(rec.kp,                    "kp",                    i)
                ap_val       = _safe_float(rec.ap,                    "ap",                    i)

                # --- unit conversion: km/s → m/s for drag equation ---
                velocity_ms = vel_kms * 1000.0

                # --- atmospheric density from altitude
                #     Import here (inside loop guard) keeps the import lazy
                #     and avoids circular import issues at module load time.
                from app.physics.atmosphere import estimate_density_with_space_weather

                # Clamp f107 and kp to valid ranges before passing to atmosphere model.
                # Out-of-range space weather values can occur in noisy CSV data;
                # we clamp rather than skip so no training samples are wasted.
                f107_clamped = max(60.0, min(300.0, f107_val))
                kp_clamped   = max(0.0,  min(9.0,   kp_val))

                weather = estimate_density_with_space_weather(
                    altitude_km=altitude_km,
                    f107=f107_clamped,
                    kp=kp_clamped,
                )
                rho = weather["density_kg_m3"]

                # --- compute target drag force via drag_calculator ---
                # compute_drag() encapsulates F = 0.5 * Cd * rho * A * v²
                # We never write that equation here — it lives in drag_calculator.py.
                drag_result = compute_drag(
                    cd=cd,
                    rho=rho,
                    area=area,
                    velocity=velocity_ms,
                    mass=mass,
                )
                target_drag_force = drag_result["drag_force_N"]

                rows.append({
                    "timestamp":            timestamp,
                    "altitude_km":          altitude_km,
                    "orbital_velocity_kms": vel_kms,
                    "f107":                 f107_val,
                    "kp":                   kp_val,
                    "ap":                   ap_val,
                    "target_drag_force_N":  target_drag_force,
                })

            except (ValueError, TypeError) as exc:
                # Log and skip bad rows rather than crashing the entire build
                logger.warning("Skipping row %d: %s", i, exc)
                skipped += 1
                continue

        if not rows:
            raise ValueError(
                "All rows were skipped due to validation errors. "
                "Check the database content and ingestion pipeline."
            )

        if skipped > 0:
            logger.warning(
                "%d row(s) skipped due to missing or invalid values.", skipped
            )

    # ------------------------------------------------------------------
    # 4. Build DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)

    # Ensure numeric columns are stored as float64 (not object dtype)
    numeric_cols = [
        "altitude_km", "orbital_velocity_kms",
        "f107", "kp", "ap", "target_drag_force_N",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any rows where numeric conversion produced NaN
    before = len(df)
    df.dropna(subset=numeric_cols, inplace=True)
    if len(df) < before:
        logger.warning(
            "Dropped %d row(s) with NaN after numeric conversion.",
            before - len(df),
        )

    df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 5. Save to CSV
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Dataset saved to: %s", output_path)

    # ------------------------------------------------------------------
    # 6. Print summary
    # ------------------------------------------------------------------
    feature_cols = [c for c in df.columns if c != "target_drag_force_N"]

    print("\n" + "=" * 60)
    print("  ML Dataset Builder — Complete")
    print("=" * 60)
    print(f"  Samples built      : {len(df):,}")
    print(f"  Rows skipped       : {skipped:,}")
    print(f"  Feature columns    : {feature_cols}")
    print(f"  Target column      : target_drag_force_N")
    print(f"  Target range       : "
          f"{df['target_drag_force_N'].min():.4e} — "
          f"{df['target_drag_force_N'].max():.4e} N")
    print(f"  Output file        : {output_path}")
    print("=" * 60 + "\n")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        dataset = build_training_dataset()
    except (ValueError, RuntimeError) as exc:
        logger.error("Dataset build failed: %s", exc)
        sys.exit(1)