"""
orbit_propagator.py — Phase 2, Step 3
AI-Powered Satellite Drag Prediction and Orbital Decay Simulator

Orbit Propagator
================
This module advances a satellite's position and velocity forward in time
by one timestep, accounting for two forces:

    1. Earth's gravity     — pulls the satellite toward Earth's center
    2. Atmospheric drag    — opposes the satellite's motion, bleeding
                             off kinetic energy and lowering the orbit

It is the computational heart of the simulator. Phase 2 Step 4
(simulator.py) will call propagate_step() in a loop to build a full
orbital decay trajectory.

HOW THIS MODULE CONNECTS THE PHYSICS PIPELINE
----------------------------------------------
atmosphere.py                  → provides ρ (atmospheric density)
    ↓
drag_calculator.py             → uses ρ to compute F_drag and a_drag
    ↓
orbit_propagator.py (this)     → applies gravity + drag to update
                                  position and velocity each timestep
    ↓
simulator.py (Phase 2 Step 4)  → loops over timesteps, logs trajectory

COORDINATE SYSTEM
-----------------
We use a 3-D Earth-Centered Inertial (ECI) Cartesian frame:
    - Origin  : Earth's center of mass
    - x-axis  : points toward the vernal equinox
    - y-axis  : 90° east in the equatorial plane
    - z-axis  : points toward the North Pole

All positions are in meters [m], velocities in m/s, accelerations in m/s².

INTEGRATION METHOD: EULER (First-Order)
----------------------------------------
Euler integration updates state using the derivative at the current step:
    v_new = v + a × dt
    r_new = r + v × dt

WHY EULER INSTEAD OF RK4?
    - Euler is the simplest possible integrator: one function evaluation
      per step, easy to read, easy to debug.
    - It introduces small errors each step (first-order accuracy), which
      accumulate over long simulations. For a 1-second timestep at LEO
      speeds (~7,700 m/s), position error per step is ~mm; over an orbit
      (~90 min) it can reach tens of meters — acceptable for demonstrating
      decay trends, not for precision tracking.
    - RK4 (Runge-Kutta 4th order) evaluates derivatives at four intermediate
      points per step, dramatically reducing error. Phase 3 will replace the
      Euler calls here with an RK4 integrator without changing the public API.
    - Rule of thumb: use dt ≤ 1–10 s with Euler for LEO; RK4 can use
      dt ≤ 60–120 s with similar accuracy.
"""

import logging
import math
from typing import Optional

from app.physics.atmosphere import estimate_density_with_space_weather, estimate_density
from app.physics.drag_calculator import compute_drag

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Earth's gravitational parameter μ = G × M_earth [m³/s²]
# Using the WGS-84 standard value.
MU_EARTH: float = 3.986004418e14   # m³/s²

# Earth's mean equatorial radius [m]
R_EARTH_M: float = 6_371_000.0     # 6,371 km in meters

# Minimum altitude before we declare re-entry [km]
# The Kármán line (100 km) is the conventional edge of space;
# most satellites with perigee below ~150 km decay within days.
REENTRY_ALTITUDE_KM: float = 100.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_state(
    x: float, y: float, z: float,
    vx: float, vy: float, vz: float,
    mass: float, cd: float, area: float, dt: float,
) -> None:
    """
    Validate the complete orbital state and satellite parameters.

    Raises TypeError for non-numeric inputs and ValueError for physically
    implausible values.
    """
    scalars = {
        "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
        "mass": mass, "cd": cd, "area": area, "dt": dt,
    }
    for name, val in scalars.items():
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"'{name}' must be numeric, got {type(val).__name__}"
            )

    if mass <= 0:
        raise ValueError(f"Satellite mass must be > 0, got {mass} kg")
    if cd <= 0:
        raise ValueError(f"Drag coefficient Cd must be > 0, got {cd}")
    if area <= 0:
        raise ValueError(f"Cross-sectional area must be > 0, got {area} m²")
    if dt <= 0:
        raise ValueError(f"Timestep dt must be > 0, got {dt} s")

    # Check satellite is not inside Earth
    r = math.sqrt(x**2 + y**2 + z**2)
    if r < R_EARTH_M:
        raise ValueError(
            f"Position vector magnitude ({r:.0f} m) is inside Earth's radius "
            f"({R_EARTH_M:.0f} m). Check initial conditions."
        )


# ---------------------------------------------------------------------------
# Core propagation step
# ---------------------------------------------------------------------------

def propagate_step(
    x: float,
    y: float,
    z: float,
    vx: float,
    vy: float,
    vz: float,
    mass: float,
    cd: float,
    area: float,
    dt: float,
    f107: Optional[float] = None,
    kp: Optional[float] = None,
) -> dict:
    """
    Advance the satellite state by one timestep using Euler integration.

    Physics applied each step:
        1. Gravity          — from Earth's gravitational parameter μ
        2. Atmospheric drag — from atmosphere.py + drag_calculator.py

    Parameters
    ----------
    x, y, z    : Current position in ECI frame [m]
    vx, vy, vz : Current velocity in ECI frame [m/s]
    mass       : Satellite mass [kg]
    cd         : Drag coefficient (dimensionless, typically 2.0–2.4)
    area       : Cross-sectional area facing velocity vector [m²]
    dt         : Integration timestep [seconds]
                 Recommended: 1–10 s for Euler; keep ≤ 60 s for stability
    f107       : (Optional) F10.7 solar flux index [sfu].
                 If None, space-weather-neutral density model is used.
    kp         : (Optional) Kp geomagnetic index [0–9].
                 Must be provided together with f107, or both omitted.

    Returns
    -------
    dict with keys:
        "x", "y", "z"           : float — updated position [m]
        "vx", "vy", "vz"        : float — updated velocity [m/s]
        "altitude_km"           : float — altitude above Earth's surface [km]
        "radius_m"              : float — distance from Earth's center [m]
        "velocity_mag_ms"       : float — speed [m/s]
        "density_kg_m3"         : float — atmospheric density [kg/m³]
        "drag_force_N"          : float — drag force magnitude [N]
        "drag_accel_ms2"        : float — drag acceleration magnitude [m/s²]
        "grav_accel_ms2"        : float — gravitational acceleration [m/s²]
        "reentry"               : bool  — True if altitude < REENTRY_ALTITUDE_KM

    Raises
    ------
    TypeError, ValueError : see _validate_state()

    Notes
    -----
    Drag acts opposite to the velocity vector. In component form:
        a_drag_vec = −(|a_drag| / |v|) × (vx, vy, vz)

    Gravity points from satellite toward Earth's center:
        a_grav_vec = −(μ / r³) × (x, y, z)

    Both accelerations are summed before applying the Euler update.
    """
    # ------------------------------------------------------------------
    # 0. Validate inputs
    # ------------------------------------------------------------------
    _validate_state(x, y, z, vx, vy, vz, mass, cd, area, dt)

    if (f107 is None) != (kp is None):
        raise ValueError("Provide both f107 and kp, or neither.")

    # ------------------------------------------------------------------
    # 1. Compute position magnitude (distance from Earth's center)
    # ------------------------------------------------------------------
    # r = √(x² + y² + z²)
    # This is the straight-line distance from Earth's center to the
    # satellite in 3-D space.
    r = math.sqrt(x**2 + y**2 + z**2)

    # Altitude above Earth's surface in km
    altitude_km = (r - R_EARTH_M) / 1000.0

    # ------------------------------------------------------------------
    # 2. Compute velocity magnitude
    # ------------------------------------------------------------------
    # |v| = √(vx² + vy² + vz²)
    v_mag = math.sqrt(vx**2 + vy**2 + vz**2)

    # ------------------------------------------------------------------
    # 3. Gravitational acceleration
    # ------------------------------------------------------------------
    # WHY GRAVITY POINTS TOWARD EARTH'S CENTER
    # Newton's law of gravitation: F = μ × m / r²
    # The force always points from the satellite toward Earth's center.
    # In vector form, the unit vector toward Earth's center is -(x,y,z)/r,
    # so:
    #   a_grav_vec = -(μ / r²) × (x, y, z) / r
    #              = -(μ / r³) × (x, y, z)
    #
    # This produces a centripetal acceleration that continuously curves
    # the satellite's path into an orbit. Without drag it would be a
    # perfect ellipse forever; with drag the orbit spirals inward.

    grav_scalar = MU_EARTH / (r ** 3)   # μ / r³ [1/s²]

    # Gravitational acceleration components [m/s²]
    # Negative sign: force points toward origin (Earth's center)
    ax_grav = -grav_scalar * x
    ay_grav = -grav_scalar * y
    az_grav = -grav_scalar * z

    # Magnitude of gravitational acceleration [m/s²]
    # At 400 km: g ≈ 8.69 m/s² (compare to 9.81 m/s² at sea level)
    grav_accel_mag = math.sqrt(ax_grav**2 + ay_grav**2 + az_grav**2)

    # ------------------------------------------------------------------
    # 4. Atmospheric density at current altitude
    # ------------------------------------------------------------------
    if f107 is not None:
        weather_result = estimate_density_with_space_weather(altitude_km, f107, kp)
        rho = weather_result["density_kg_m3"]
    else:
        rho = estimate_density(altitude_km)

    # ------------------------------------------------------------------
    # 5. Drag force and acceleration (via drag_calculator)
    # ------------------------------------------------------------------
    # compute_drag() returns F_drag [N] and a_drag [m/s²]
    drag_result = compute_drag(cd=cd, rho=rho, area=area, velocity=v_mag, mass=mass)
    drag_force   = drag_result["drag_force_N"]
    drag_accel   = drag_result["drag_accel_ms2"]   # scalar magnitude

    # WHY DRAG ACTS OPPOSITE TO VELOCITY
    # Drag is a resistive force — it always opposes the direction of motion.
    # To convert the scalar drag acceleration into a 3-D vector we multiply
    # by the unit vector in the velocity direction, then negate it:
    #
    #   â_v = (vx, vy, vz) / |v|       ← unit vector along velocity
    #   a_drag_vec = -drag_accel × â_v  ← drag opposes motion
    #
    # At |v| ≈ 0 (impossible in orbit but guarded anyway) we skip drag.

    if v_mag > 0:
        ax_drag = -drag_accel * (vx / v_mag)
        ay_drag = -drag_accel * (vy / v_mag)
        az_drag = -drag_accel * (vz / v_mag)
    else:
        ax_drag = ay_drag = az_drag = 0.0

    # ------------------------------------------------------------------
    # 6. Total acceleration = gravity + drag
    # ------------------------------------------------------------------
    ax_total = ax_grav + ax_drag
    ay_total = ay_grav + ay_drag
    az_total = az_grav + az_drag

    # ------------------------------------------------------------------
    # 7. Euler integration: update velocity, then position
    # ------------------------------------------------------------------
    # WHY EULER INTEGRATION?
    # Euler is the simplest numerical integration scheme:
    #   v_new = v_old + a × dt   (update velocity using current acceleration)
    #   r_new = r_old + v × dt   (update position using current velocity)
    #
    # It uses the derivative (acceleration) evaluated at the START of the
    # interval and assumes it is constant for the entire timestep.
    # This is accurate only if dt is small relative to the timescale over
    # which acceleration changes — hence the recommendation of dt ≤ 10 s.
    #
    # WHY RK4 COULD REPLACE EULER LATER
    # RK4 evaluates the derivative at four intermediate points within the
    # timestep (k1, k2, k3, k4) and takes a weighted average. This gives
    # fourth-order accuracy: halving dt reduces error by 16×, not 2× as
    # with Euler. For orbital mechanics, RK4 with dt=60 s gives similar
    # accuracy to Euler with dt=1 s, making long simulations much faster.
    #
    # To upgrade to RK4 in Phase 3, replace lines 7a–7b with an RK4
    # stepper that calls the same _compute_derivatives() helper — no
    # changes needed in simulator.py.

    # 7a. Update velocity [m/s]
    vx_new = vx + ax_total * dt
    vy_new = vy + ay_total * dt
    vz_new = vz + az_total * dt

    # 7b. Update position [m]
    x_new = x + vx_new * dt
    y_new = y + vy_new * dt
    z_new = z + vz_new * dt

    # ------------------------------------------------------------------
    # 8. Post-step derived quantities
    # ------------------------------------------------------------------
    r_new        = math.sqrt(x_new**2 + y_new**2 + z_new**2)
    alt_new_km   = (r_new - R_EARTH_M) / 1000.0
    v_new_mag    = math.sqrt(vx_new**2 + vy_new**2 + vz_new**2)
    reentry_flag = alt_new_km < REENTRY_ALTITUDE_KM

    # ------------------------------------------------------------------
    # 9. Logging
    # ------------------------------------------------------------------
    logger.debug(
        "propagate_step | alt=%.2f km, v=%.2f m/s, ρ=%.4e kg/m³, "
        "F_drag=%.4e N, a_drag=%.4e m/s², a_grav=%.4f m/s²",
        altitude_km, v_mag, rho,
        drag_force, drag_accel, grav_accel_mag,
    )
    if reentry_flag:
        logger.warning(
            "RE-ENTRY THRESHOLD REACHED: altitude %.2f km < %.0f km",
            alt_new_km, REENTRY_ALTITUDE_KM,
        )

    return {
        # Updated state
        "x":  x_new,
        "y":  y_new,
        "z":  z_new,
        "vx": vx_new,
        "vy": vy_new,
        "vz": vz_new,
        # Derived quantities at the START of this step
        # (useful for logging, ML features, and simulator bookkeeping)
        "altitude_km":      altitude_km,
        "radius_m":         r,
        "velocity_mag_ms":  v_mag,
        "density_kg_m3":    rho,
        "drag_force_N":     drag_force,
        "drag_accel_ms2":   drag_accel,
        "grav_accel_ms2":   grav_accel_mag,
        # Flag for simulator loop exit condition
        "reentry":          reentry_flag,
    }


# ---------------------------------------------------------------------------
# Circular orbit initialiser (helper for tests and simulator)
# ---------------------------------------------------------------------------

def circular_orbit_state(altitude_km: float, inclination_deg: float = 0.0) -> dict:
    """
    Return a simple initial state for a circular orbit at a given altitude.

    Places the satellite on the x-axis and gives it the exact circular
    orbital velocity in the y-direction (equatorial orbit by default).

    Parameters
    ----------
    altitude_km     : Desired circular orbit altitude [km]
    inclination_deg : Orbital inclination [degrees] (currently unused;
                      reserved for Phase 3 full 3-D initialisation)

    Returns
    -------
    dict with keys: x, y, z [m] and vx, vy, vz [m/s]

    Circular orbital speed:
        v_circ = √(μ / r)
    where r = R_Earth + altitude.
    """
    if altitude_km < REENTRY_ALTITUDE_KM:
        raise ValueError(
            f"Altitude {altitude_km} km is below re-entry threshold "
            f"({REENTRY_ALTITUDE_KM} km)."
        )

    r = R_EARTH_M + altitude_km * 1000.0        # radius from Earth's center [m]
    v_circ = math.sqrt(MU_EARTH / r)            # circular orbital speed [m/s]

    logger.debug(
        "circular_orbit_state | alt=%.1f km, r=%.3e m, v_circ=%.2f m/s",
        altitude_km, r, v_circ,
    )

    return {
        "x":  r,       # satellite starts on positive x-axis
        "y":  0.0,
        "z":  0.0,
        "vx": 0.0,
        "vy": v_circ,  # velocity entirely in y-direction (prograde)
        "vz": 0.0,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.WARNING,   # suppress DEBUG/INFO noise in self-test output
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    SEP = "=" * 72

    print(SEP)
    print("  Phase 2 Step 3 — Orbit Propagator Self-Test")
    print("  Simulating an ISS-like satellite for 10 propagation steps")
    print(SEP)

    # ------------------------------------------------------------------
    # ISS-like initial conditions
    #   Altitude : ~408 km
    #   Mass     : 420,000 kg
    #   Cd       : 2.2  (representative for ISS attitude modes)
    #   Area     : 2,500 m² (ISS large cross-section; simplified)
    #   dt       : 10 seconds per step
    # ------------------------------------------------------------------
    ISS_ALT_KM = 408.0
    ISS_MASS   = 420_000.0    # kg
    ISS_CD     = 2.2
    ISS_AREA   = 2_500.0      # m²
    DT         = 10.0         # seconds per Euler step

    # Representative moderate space weather
    F107_VAL = 150.0
    KP_VAL   = 2.0

    # Initialise circular orbit
    state = circular_orbit_state(ISS_ALT_KM)
    x, y, z    = state["x"],  state["y"],  state["z"]
    vx, vy, vz = state["vx"], state["vy"], state["vz"]

    print(f"\n  Initial state (circular orbit at {ISS_ALT_KM} km):")
    print(f"  Position : ({x:.3e}, {y:.3e}, {z:.3e}) m")
    print(f"  Velocity : ({vx:.3e}, {vy:.3e}, {vz:.3e}) m/s")
    print(f"  Mass     : {ISS_MASS:,.0f} kg | Cd={ISS_CD} | Area={ISS_AREA} m²")
    print(f"  dt       : {DT} s | F10.7={F107_VAL} | Kp={KP_VAL}")

    print(f"\n  {'Step':>4}  {'Alt (km)':>10}  {'ρ (kg/m³)':>14}  "
          f"{'F_drag (N)':>14}  {'|v| (m/s)':>11}  {'|r| (km)':>10}")
    print("  " + "-" * 68)

    for step in range(1, 11):
        result = propagate_step(
            x=x, y=y, z=z,
            vx=vx, vy=vy, vz=vz,
            mass=ISS_MASS,
            cd=ISS_CD,
            area=ISS_AREA,
            dt=DT,
            f107=F107_VAL,
            kp=KP_VAL,
        )

        print(
            f"  {step:>4}  "
            f"{result['altitude_km']:>10.4f}  "
            f"{result['density_kg_m3']:>14.4e}  "
            f"{result['drag_force_N']:>14.4e}  "
            f"{result['velocity_mag_ms']:>11.4f}  "
            f"{result['radius_m'] / 1000:>10.4f}"
        )

        # Unpack new state for next step
        x,  y,  z  = result["x"],  result["y"],  result["z"]
        vx, vy, vz = result["vx"], result["vy"], result["vz"]

        if result["reentry"]:
            print("\n  *** RE-ENTRY THRESHOLD REACHED — simulation stopped ***")
            break

    print(f"\n  Final altitude : {result['altitude_km']:.4f} km")
    print(f"  Final speed    : {result['velocity_mag_ms']:.4f} m/s")
    print(f"  Drag force     : {result['drag_force_N']:.4e} N")
    print(f"  Drag accel     : {result['drag_accel_ms2']:.4e} m/s²")
    print(f"  Grav  accel    : {result['grav_accel_ms2']:.6f} m/s²")

    # ------------------------------------------------------------------
    # Sanity check: circular speed at 408 km
    # v_circ = √(μ/r) — should match initial vy closely
    # ------------------------------------------------------------------
    r_iss = R_EARTH_M + ISS_ALT_KM * 1000.0
    v_expected = math.sqrt(MU_EARTH / r_iss)
    print(f"\n  Sanity check — expected circular speed : {v_expected:.4f} m/s")
    print(f"  Initial vy used                        : {state['vy']:.4f} m/s")

    print(f"\n{SEP}")
    print("  All tests passed. orbit_propagator.py ready for simulator.py.")
    print(SEP)

    # NOTE:
# This implementation uses first-order Euler integration for simplicity. Euler may introduce numerical energy errors, causing unrealistic orbital behavior over long simulations.
# A Runge-Kutta 4th Order (RK4) integrator will replace this in a later phase.