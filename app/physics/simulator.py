"""
simulator.py — Phase 2, Step 4
AI-Powered Satellite Drag Prediction and Orbital Decay Simulator

Orbital Decay Simulator
========================
This module is the top-level orchestrator of the simulation. It does
NO physics of its own — all gravity and drag calculations live in
orbit_propagator.py. The simulator's only job is to:

    1. Set up initial conditions
    2. Call propagate_step() in a loop
    3. Collect and display the trajectory history
    4. Detect re-entry and stop cleanly

HOW simulator.py FITS INTO THE PIPELINE
----------------------------------------
    atmosphere.py       — density model
        ↓
    drag_calculator.py  — drag force/acceleration
        ↓
    orbit_propagator.py — one Euler step (gravity + drag)
        ↓
    simulator.py        ← YOU ARE HERE
        loops over propagate_step() until re-entry or max steps
        stores trajectory history
        prints summary
"""

import logging

from app.physics.orbit_propagator import propagate_step, circular_orbit_state

# ---------------------------------------------------------------------------
# Logging — WARNING level to suppress DEBUG noise from physics modules
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

# Satellite physical properties (ISS-like)
MASS_KG     = 420_000.0   # kg
CD          = 2.2         # drag coefficient (dimensionless)
AREA_M2     = 2_500.0     # cross-sectional area [m²]

# Integration settings
DT_S        = 1.0        # timestep [seconds] — small enough for Euler stability
MAX_STEPS   = 10_000      # hard cap on iterations (~27.8 hours at dt=10 s)

# Space weather (moderate, representative conditions)
F107        = 150.0       # solar flux index [sfu]
KP          = 2.0         # geomagnetic activity index [0–9]

# Starting orbit
INITIAL_ALTITUDE_KM = 408.0   # ISS nominal altitude

# Print every N steps to keep output readable (1 = every step)
PRINT_EVERY = 100


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_simulation() -> list:
    """
    Run the orbital decay simulation loop.

    Initialises a circular orbit at INITIAL_ALTITUDE_KM, then advances
    the satellite state by calling propagate_step() repeatedly until
    either re-entry is detected or MAX_STEPS is reached.

    Returns
    -------
    list of dict
        Full trajectory history. Each entry is the dict returned by
        propagate_step() for that step, with an added "step" and
        "time_s" key for convenience.
    """

    # ------------------------------------------------------------------
    # 1. Initialise state from circular orbit helper
    #    circular_orbit_state() computes the exact orbital speed needed
    #    to maintain a circular orbit at the given altitude.
    #    v_circ = sqrt(mu / r)
    # ------------------------------------------------------------------
    initial = circular_orbit_state(INITIAL_ALTITUDE_KM)
    x,  y,  z  = initial["x"],  initial["y"],  initial["z"]
    vx, vy, vz = initial["vx"], initial["vy"], initial["vz"]

    SEP = "=" * 72
    print(SEP)
    print("  Phase 2 Step 4 — Orbital Decay Simulator")
    print(SEP)
    print(f"  Satellite  : ISS-like | Mass={MASS_KG:,.0f} kg | "
          f"Cd={CD} | Area={AREA_M2:,.0f} m^2")
    print(f"  Orbit      : Circular at {INITIAL_ALTITUDE_KM} km")
    print(f"  Timestep   : {DT_S} s | Max steps: {MAX_STEPS:,} "
          f"(~{MAX_STEPS * DT_S / 3600:.1f} hours)")
    print(f"  Space wx   : F10.7={F107} sfu | Kp={KP}")
    print(SEP)
    print(f"\n  {'Step':>7}  {'Time (s)':>10}  {'Alt (km)':>10}  "
          f"{'rho (kg/m3)':>14}  {'F_drag (N)':>14}  {'|v| (m/s)':>11}")
    print("  " + "-" * 72)

    # ------------------------------------------------------------------
    # 2. History list — each entry is one propagate_step() result dict
    #    augmented with step index and elapsed time.
    # ------------------------------------------------------------------
    history = []
    result  = {}
    step    = 0

    # ------------------------------------------------------------------
    # 3. Main simulation loop
    #
    #    At each iteration:
    #      a. Call propagate_step() with the current state.
    #         All physics (gravity, drag, Euler integration) happen
    #         inside that function — simulator.py never touches equations.
    #      b. Unpack the new position and velocity from the returned dict.
    #      c. Append the full result to history for later analysis.
    #      d. Print a progress row every PRINT_EVERY steps.
    #      e. Check the "reentry" flag — if True, stop the loop.
    # ------------------------------------------------------------------
    for step in range(1, MAX_STEPS + 1):
        elapsed_s = (step - 1) * DT_S

        # --- core physics call (one Euler step) ---
        result = propagate_step(
            x=x, y=y, z=z,
            vx=vx, vy=vy, vz=vz,
            mass=MASS_KG,
            cd=CD,
            area=AREA_M2,
            dt=DT_S,
            f107=F107,
            kp=KP,
        )

        # --- tag the result with simulation metadata ---
        result["step"]   = step
        result["time_s"] = elapsed_s

        # --- store in history ---
        history.append(result)

        # --- print progress row ---
        if step == 1 or step % PRINT_EVERY == 0 or result["reentry"]:
            print(
                f"  {step:>7,}  "
                f"{elapsed_s:>10,.0f}  "
                f"{result['altitude_km']:>10.4f}  "
                f"{result['density_kg_m3']:>14.4e}  "
                f"{result['drag_force_N']:>14.4e}  "
                f"{result['velocity_mag_ms']:>11.4f}"
            )

        # --- update state for next iteration ---
        x,  y,  z  = result["x"],  result["y"],  result["z"]
        vx, vy, vz = result["vx"], result["vy"], result["vz"]

        # --- re-entry exit condition ---
        if result["reentry"]:
            print("\n  *** RE-ENTRY THRESHOLD REACHED — simulation stopped ***")
            break

    # ------------------------------------------------------------------
    # 4. Simulation summary
    # ------------------------------------------------------------------
    total_time_s = step * DT_S
    hours   = int(total_time_s // 3600)
    minutes = int((total_time_s % 3600) // 60)
    seconds = total_time_s % 60

    print(f"\n{SEP}")
    print("  SIMULATION COMPLETE — Summary")
    print(SEP)
    print(f"  Total steps simulated : {step:,}")
    print(f"  Total simulation time : {total_time_s:,.0f} s  "
          f"({hours}h {minutes}m {seconds:.0f}s)")
    print(f"  Final altitude        : {result.get('altitude_km', 0):.4f} km")
    print(f"  Final velocity        : {result.get('velocity_mag_ms', 0):.4f} m/s")
    print(f"  Final drag force      : {result.get('drag_force_N', 0):.4e} N")
    print(f"  Final density         : {result.get('density_kg_m3', 0):.4e} kg/m^3")
    print(f"  Re-entry detected     : {result.get('reentry', False)}")
    print(f"  History records stored: {len(history):,}")
    print(SEP)

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trajectory = run_simulation()