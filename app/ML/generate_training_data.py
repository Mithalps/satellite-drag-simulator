"""
generate_training_data.py
AI-Powered Satellite Drag Prediction and Orbital Decay Simulator

Synthetic Training Data Generator
===================================
Generates 10,000 physically grounded training samples by randomly sampling
orbital and space weather parameters, then computing the true drag force
using the existing physics modules (atmosphere.py + drag_calculator.py).

WHY SYNTHETIC DATA?
-------------------
The SQLite database currently holds only ~12 real ingested records — far
too few to train a reliable ML model. Rather than collecting months of
real TLE data, we generate synthetic samples that:
    - Span the full physically relevant parameter space (200–800 km LEO)
    - Use the same physics equations the real system relies on
    - Provide a consistent, reproducible training baseline

The generated targets are NOT random — they are the deterministic output
of F = 0.5 * Cd * rho(altitude, F10.7, Kp) * A * v², so the model learns
the true physics relationship, not noise.

OUTPUT
------
    data/training_dataset_large.csv
    Columns: altitude_km, orbital_velocity_kms, f107, kp, ap,
             target_drag_force_N
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allows: python -m app.ML.generate_training_data
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.physics.atmosphere import estimate_density_with_space_weather
from app.physics.drag_calculator import compute_drag

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "training_dataset_large.csv")

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------
N_SAMPLES    = 10_000
RANDOM_SEED  = 42

# Satellite defaults used for drag computation
# (representative mid-sized LEO spacecraft)
DEFAULT_CD   = 2.2
DEFAULT_AREA = 10.0   # m²
DEFAULT_MASS = 500.0  # kg

# Parameter ranges
ALT_MIN_KM,  ALT_MAX_KM  = 200.0, 800.0
VEL_MIN_KMS, VEL_MAX_KMS = 7.2,   8.2
F107_MIN,    F107_MAX    = 70.0,   250.0
KP_MIN,      KP_MAX      = 0.0,   8.0
AP_MIN,      AP_MAX      = 0.0,   100.0


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_dataset(
    n_samples:   int = N_SAMPLES,
    output_path: str = OUTPUT_FILE,
    cd:   float = DEFAULT_CD,
    area: float = DEFAULT_AREA,
    mass: float = DEFAULT_MASS,
    seed: int   = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic training samples and save them as a CSV.

    Parameters
    ----------
    n_samples   : Number of samples to generate (default 10,000)
    output_path : Destination CSV path
    cd          : Drag coefficient for target computation
    area        : Cross-sectional area [m²]
    mass        : Satellite mass [kg]
    seed        : Random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns:
        altitude_km, orbital_velocity_kms, f107, kp, ap,
        target_drag_force_N
    """

    SEP = "=" * 58
    print(SEP)
    print("  Synthetic Training Data Generator")
    print(SEP)
    print(f"  Samples to generate : {n_samples:,}")
    print(f"  Random seed         : {seed}")
    print(f"  Cd={cd}, Area={area} m², Mass={mass} kg")
    print(SEP)

    # ------------------------------------------------------------------
    # 1. Draw random samples across the full parameter space
    #
    #    np.random.uniform(low, high, size) draws from a uniform
    #    distribution — every value in [low, high] is equally likely.
    #    This ensures the model sees the full operating envelope of a
    #    LEO satellite, not just typical conditions.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)   # modern numpy RNG (reproducible)

    altitudes  = rng.uniform(ALT_MIN_KM,  ALT_MAX_KM,  n_samples)  # km
    velocities = rng.uniform(VEL_MIN_KMS, VEL_MAX_KMS, n_samples)  # km/s
    f107s      = rng.uniform(F107_MIN,    F107_MAX,     n_samples)  # sfu
    kps        = rng.uniform(KP_MIN,      KP_MAX,       n_samples)  # 0–8
    aps        = rng.uniform(AP_MIN,      AP_MAX,       n_samples)  # nT

    # ------------------------------------------------------------------
    # 2. Compute drag force for each sample
    #
    #    For each row:
    #      a) Get atmospheric density from altitude + space weather
    #      b) Compute drag force via drag_calculator (no duplicate math)
    # ------------------------------------------------------------------
    drag_forces = []
    skipped     = 0

    logger.info("Computing drag force for %d samples...", n_samples)

    for i in range(n_samples):
        try:
            # atmospheric density at this altitude and space weather state
            weather = estimate_density_with_space_weather(
                altitude_km=float(altitudes[i]),
                f107=float(f107s[i]),
                kp=float(kps[i]),
            )
            rho = weather["density_kg_m3"]

            # velocity: km/s → m/s for SI drag equation
            velocity_ms = float(velocities[i]) * 1000.0

            # drag force via physics module (F = 0.5 * Cd * rho * A * v²)
            drag_result = compute_drag(
                cd=cd,
                rho=rho,
                area=area,
                velocity=velocity_ms,
                mass=mass,
            )
            drag_forces.append(drag_result["drag_force_N"])

        except (ValueError, TypeError) as exc:
            logger.warning("Sample %d skipped: %s", i, exc)
            drag_forces.append(None)
            skipped += 1

        # Progress indicator every 1,000 rows
        if (i + 1) % 1_000 == 0:
            logger.info("  %d / %d samples processed...", i + 1, n_samples)

    # ------------------------------------------------------------------
    # 3. Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "altitude_km":          altitudes,
        "orbital_velocity_kms": velocities,
        "f107":                 f107s,
        "kp":                   kps,
        "ap":                   aps,
        "target_drag_force_N":  drag_forces,
    })

    # Drop any rows where drag computation failed
    before = len(df)
    df.dropna(subset=["target_drag_force_N"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d failed rows.", dropped)

    # ------------------------------------------------------------------
    # 4. Save to CSV
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Dataset saved to: %s", output_path)

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print(f"\n  Samples generated   : {len(df):,}")
    print(f"  Samples skipped     : {skipped:,}")
    print(f"\n  Target statistics (target_drag_force_N):")
    print(f"    Min   : {df['target_drag_force_N'].min():.4e} N")
    print(f"    Max   : {df['target_drag_force_N'].max():.4e} N")
    print(f"    Mean  : {df['target_drag_force_N'].mean():.4e} N")
    print(f"    Std   : {df['target_drag_force_N'].std():.4e} N")
    print(f"\n  Output file         : {output_path}")
    print(SEP + "\n")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        generate_dataset()
    except Exception as exc:
        logger.error("Data generation failed: %s", exc)
        sys.exit(1)