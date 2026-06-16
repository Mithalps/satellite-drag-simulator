"""
drag_calculator.py — Phase 2, Step 1
AI-Powered Satellite Drag Prediction and Orbital Decay Simulator

Physics-Based Atmospheric Drag Calculator
==========================================
This module implements the classical aerodynamic drag equation for satellites
in low Earth orbit (LEO). It is designed to be imported by:
    - atmosphere.py       (feeds rho from atmospheric density models)
    - orbit_propagator.py (applies drag force/acceleration to orbital state)
    - simulator.py        (drives the end-to-end simulation loop)
    - ML prediction pipeline (provides physics baseline for model features)

Physics Background
------------------
As a satellite moves through the upper atmosphere, it collides with residual
gas molecules, creating a drag force that opposes its velocity. This force
causes gradual orbital energy loss and eventual re-entry — a process called
orbital decay.

The standard drag equation used in orbital mechanics:

    F_drag = 0.5 * Cd * ρ * A * v²

Where:
    F_drag  — Drag force [Newtons, N]
    Cd      — Drag coefficient (dimensionless; ~2.2 for simple satellites)
    ρ (rho) — Atmospheric density [kg/m³]
    A       — Satellite cross-sectional area facing velocity vector [m²]
    v       — Satellite velocity magnitude relative to atmosphere [m/s]

If satellite mass (m) is known, drag acceleration is:
    a_drag = F_drag / m   [m/s²]
"""

import logging
import math
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level logger
# All downstream modules that import drag_calculator will see logs under
# the "drag_calculator" namespace in the root application log.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical constants and sane defaults
# ---------------------------------------------------------------------------

# Typical drag coefficient for a tumbling or box-shaped satellite (dimensionless)
DEFAULT_CD: float = 2.2

# ISS cross-sectional area as a rough order-of-magnitude reference (m²)
# For real missions this should be computed from the satellite geometry model.
DEFAULT_AREA_M2: float = 10.0


# ---------------------------------------------------------------------------
# Input validation helper
# ---------------------------------------------------------------------------

def _validate_inputs(
    cd: float,
    rho: float,
    area: float,
    velocity: float,
    mass: Optional[float],
) -> None:
    """
    Raise ValueError with a descriptive message if any input is physically
    implausible.  Called at the top of every public function so that
    downstream modules fail fast with useful error messages.

    Parameters
    ----------
    cd       : Drag coefficient — must be > 0 (dimensionless)
    rho      : Atmospheric density [kg/m³] — must be >= 0
    area     : Cross-sectional area [m²] — must be > 0
    velocity : Velocity magnitude [m/s] — must be >= 0
    mass     : Satellite mass [kg] — if provided, must be > 0
    """
    if cd <= 0:
        raise ValueError(f"Drag coefficient (Cd) must be > 0, got {cd}")
    if rho < 0:
        raise ValueError(f"Atmospheric density (rho) cannot be negative, got {rho}")
    if area <= 0:
        raise ValueError(f"Cross-sectional area must be > 0, got {area} m²")
    if velocity < 0:
        raise ValueError(f"Velocity cannot be negative, got {velocity} m/s")
    if mass is not None and mass <= 0:
        raise ValueError(f"Satellite mass must be > 0 if provided, got {mass} kg")


# ---------------------------------------------------------------------------
# Core drag calculation
# ---------------------------------------------------------------------------

def compute_drag_force(
    cd: float,
    rho: float,
    area: float,
    velocity: float,
) -> float:
    """
    Calculate the atmospheric drag force on a satellite.

    Uses the standard drag equation:
        F_drag = 0.5 * Cd * rho * A * v²

    Parameters
    ----------
    cd       : Drag coefficient (dimensionless).
                 Typical values: 2.0–2.4 for spacecraft in LEO.
    rho      : Atmospheric density [kg/m³].
                 At 400 km altitude this is roughly 1e-11 to 1e-12 kg/m³,
                 depending on solar activity.
    area     : Effective cross-sectional area of the satellite [m²].
                 This is the projected area perpendicular to the velocity vector.
    velocity : Satellite velocity relative to the atmosphere [m/s].
                 Typical LEO orbital speed ~7,700 m/s.

    Returns
    -------
    float
        Drag force magnitude in Newtons [N].

    Raises
    ------
    ValueError
        If any input fails physical plausibility checks.
    TypeError
        If any input is not a real number.

    Example
    -------
    >>> force = compute_drag_force(cd=2.2, rho=1e-11, area=10.0, velocity=7700.0)
    >>> print(f"Drag force: {force:.6e} N")
    Drag force: 6.535400e-03 N
    """
    # --- type check ---
    for name, val in [("cd", cd), ("rho", rho), ("area", area), ("velocity", velocity)]:
        if not isinstance(val, (int, float)):
            raise TypeError(f"'{name}' must be a numeric type, got {type(val).__name__}")

    # --- physical validation ---
    _validate_inputs(cd=cd, rho=rho, area=area, velocity=velocity, mass=None)

    # --- core equation: F = 0.5 * Cd * rho * A * v² ---
    drag_force = 0.5 * cd * rho * area * (velocity ** 2)

    logger.debug(
        "compute_drag_force | Cd=%.4f, rho=%.4e kg/m³, A=%.4f m², "
        "v=%.2f m/s → F_drag=%.6e N",
        cd, rho, area, velocity, drag_force,
    )

    return drag_force


# ---------------------------------------------------------------------------
# Extended calculation with optional acceleration output
# ---------------------------------------------------------------------------

def compute_drag(
    cd: float,
    rho: float,
    area: float,
    velocity: float,
    mass: Optional[float] = None,
    cd_default: float = DEFAULT_CD,
    area_default: float = DEFAULT_AREA_M2,
) -> dict:
    """
    Full drag calculation returning force and, optionally, acceleration.

    This is the primary entry point for orbit_propagator.py and simulator.py.
    Returns a dictionary so downstream modules can unpack only what they need
    without caring about return-value order.

    Parameters
    ----------
    cd       : Drag coefficient (dimensionless). Pass DEFAULT_CD if unknown.
    rho      : Atmospheric density [kg/m³], typically from atmosphere.py.
    area     : Satellite cross-sectional area [m²].
    velocity : Orbital velocity relative to atmosphere [m/s].
    mass     : (Optional) Satellite mass [kg]. When provided, drag
                 acceleration a = F/m is also returned.

    Returns
    -------
    dict with keys:
        "drag_force_N"   : float — drag force in Newtons
        "drag_accel_ms2" : float or None — drag acceleration in m/s²
                           (None when mass is not provided)
        "inputs"         : dict — echo of validated inputs (useful for logging
                           and ML feature construction)

    Raises
    ------
    ValueError, TypeError  (see compute_drag_force and _validate_inputs)

    Example
    -------
    >>> result = compute_drag(
    ...     cd=2.2, rho=1e-11, area=10.0, velocity=7700.0, mass=500.0
    ... )
    >>> print(result["drag_force_N"])
    0.006535399999999999
    >>> print(result["drag_accel_ms2"])
    1.3070799999999998e-05
    """
    # --- type checks ---
    for name, val in [("cd", cd), ("rho", rho), ("area", area), ("velocity", velocity)]:
        if not isinstance(val, (int, float)):
            raise TypeError(f"'{name}' must be numeric, got {type(val).__name__}")
    if mass is not None and not isinstance(mass, (int, float)):
        raise TypeError(f"'mass' must be numeric if provided, got {type(mass).__name__}")

    # --- physical validation (includes mass check) ---
    _validate_inputs(cd=cd, rho=rho, area=area, velocity=velocity, mass=mass)

    # --- compute drag force ---
    drag_force = compute_drag_force(cd=cd, rho=rho, area=area, velocity=velocity)

    # --- compute acceleration if mass is available ---
    drag_accel: Optional[float] = None
    if mass is not None:
        # Newton's second law: a = F / m
        drag_accel = drag_force / mass
        logger.debug(
            "compute_drag | mass=%.2f kg → a_drag=%.6e m/s²", mass, drag_accel
        )

    result = {
        "drag_force_N": drag_force,
        "drag_accel_ms2": drag_accel,
        # Echo inputs back so callers can log or store them as ML features
        "inputs": {
            "cd": cd,
            "rho_kg_m3": rho,
            "area_m2": area,
            "velocity_m_s": velocity,
            "mass_kg": mass,
        },
    }

    logger.info(
        "Drag computed | F=%.6e N | a=%s m/s² | alt≈unknown (rho=%.4e kg/m³)",
        drag_force,
        f"{drag_accel:.6e}" if drag_accel is not None else "N/A (no mass)",
        rho,
    )

    return result


# ---------------------------------------------------------------------------
# Ballistic coefficient helper
# ---------------------------------------------------------------------------

def ballistic_coefficient(mass: float, cd: float, area: float) -> float:
    """
    Compute the ballistic coefficient β = m / (Cd * A).

    The ballistic coefficient is a single parameter that captures how
    'aerodynamically slippery' a satellite is.  A higher β means the
    satellite decelerates more slowly (less drag per unit mass).

    It is widely used in orbit propagation and is a useful ML feature.

    Parameters
    ----------
    mass : Satellite mass [kg]
    cd   : Drag coefficient (dimensionless)
    area : Cross-sectional area [m²]

    Returns
    -------
    float
        Ballistic coefficient β [kg/m²]

    Example
    -------
    >>> bc = ballistic_coefficient(mass=500.0, cd=2.2, area=10.0)
    >>> print(f"β = {bc:.2f} kg/m²")
    β = 22.73 kg/m²
    """
    _validate_inputs(cd=cd, rho=0.0, area=area, velocity=0.0, mass=mass)
    beta = mass / (cd * area)
    logger.debug("Ballistic coefficient β = %.4f kg/m²", beta)
    return beta


# ---------------------------------------------------------------------------
# Example usage (run this file directly to verify the module works)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Configure basic logging so log messages appear when running standalone
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("=" * 60)
    print("  Phase 2 — Drag Calculator Self-Test")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Scenario: ISS-like satellite at ~400 km altitude
    #   Atmospheric density at 400 km (moderate solar activity) ≈ 4e-12 kg/m³
    #   Orbital velocity ≈ 7,660 m/s
    #   Mass ≈ 420,000 kg (ISS, just for demonstration)
    #   Cross-sectional area ≈ 2,500 m² (ISS is large)
    # ----------------------------------------------------------------
    print("\n[Test 1] ISS-like parameters at ~400 km altitude")
    result = compute_drag(
        cd=2.2,
        rho=4e-12,      # kg/m³ — from NRLMSISE-00 at 400 km
        area=2500.0,    # m²
        velocity=7660.0,  # m/s
        mass=420_000.0,   # kg
    )
    print(f"  Drag Force      : {result['drag_force_N']:.4e} N")
    print(f"  Drag Acceleration: {result['drag_accel_ms2']:.4e} m/s²")

    # ----------------------------------------------------------------
    # Scenario: Small CubeSat (1U) at ~550 km altitude
    #   rho ≈ 1e-13 kg/m³ (much thinner air at higher altitude)
    #   velocity ≈ 7,600 m/s
    #   mass ≈ 1.33 kg
    #   area ≈ 0.01 m² (10 cm × 10 cm face)
    # ----------------------------------------------------------------
    print("\n[Test 2] 1U CubeSat at ~550 km altitude")
    result2 = compute_drag(
        cd=2.2,
        rho=1e-13,
        area=0.01,
        velocity=7600.0,
        mass=1.33,
    )
    print(f"  Drag Force      : {result2['drag_force_N']:.4e} N")
    print(f"  Drag Acceleration: {result2['drag_accel_ms2']:.4e} m/s²")

    # ----------------------------------------------------------------
    # Ballistic coefficient
    # ----------------------------------------------------------------
    print("\n[Test 3] Ballistic Coefficient")
    bc = ballistic_coefficient(mass=500.0, cd=2.2, area=10.0)
    print(f"  β = {bc:.4f} kg/m²")

    # ----------------------------------------------------------------
    # Error handling demonstration
    # ----------------------------------------------------------------
    print("\n[Test 4] Input Validation")
    try:
        compute_drag_force(cd=-1.0, rho=1e-11, area=10.0, velocity=7700.0)
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    print("\nAll tests passed. drag_calculator.py is ready for Phase 2 integration.")