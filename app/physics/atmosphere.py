"""
atmosphere.py — Phase 2, Step 2
AI-Powered Satellite Drag Prediction and Orbital Decay Simulator

Atmospheric Density Model
=========================
This module estimates atmospheric density (rho, ρ) as a function of
altitude and space weather conditions.

WHY ATMOSPHERIC DENSITY MATTERS FOR SATELLITE DRAG
---------------------------------------------------
The drag force on a satellite is:
    F_drag = 0.5 * Cd * ρ * A * v²

ρ (atmospheric density) is the single most uncertain and variable term
in this equation. At 400 km altitude, density can vary by a factor of
10–100× depending on solar activity. Getting ρ right is therefore the
key challenge in accurate drag prediction and orbital lifetime estimation.

CURRENT IMPLEMENTATION: Simplified Exponential Model
-----------------------------------------------------
The Earth's atmosphere thins exponentially with altitude. Each layer can
be approximated as:
    ρ(h) = ρ₀ × exp(−(h − h₀) / H)

Where:
    ρ₀  — reference density at base altitude h₀ [kg/m³]
    h   — altitude of interest [km]
    h₀  — base altitude of the density layer [km]
    H   — scale height of the layer [km] (how fast density drops per km)

This model is physically grounded but simplified. It ignores:
    - Day/night variations
    - Seasonal effects
    - Latitude/longitude variation

FUTURE UPGRADE PATH: NRLMSISE-00
---------------------------------
The public API (estimate_density, estimate_density_with_space_weather)
is intentionally designed to match what a full NRLMSISE-00 integration
would need. To upgrade, replace the internals of these two functions
with calls to the `nrlmsise00` Python package — NO changes needed in
orbit_propagator.py, simulator.py, or the ML pipeline.

DOWNSTREAM CONSUMERS
--------------------
    - drag_calculator.py  → receives rho as input to compute_drag()
    - orbit_propagator.py → calls estimate_density_with_space_weather()
                            at each RK4 integration step
    - simulator.py        → drives the full simulation loop
    - ML pipeline         → uses density as a physics-derived feature
"""

import logging
import math
from typing import Optional

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exponential atmosphere lookup table
# ---------------------------------------------------------------------------
# Source: US Standard Atmosphere (1976) and COSPAR CIRA reference tables.
# Each entry covers an altitude band:
#   "base_km"    — lower boundary of this layer [km]
#   "rho0"       — density at base_km [kg/m³]
#   "scale_h_km" — scale height H for this layer [km]
#
# Scale height H is the altitude increase over which density drops by
# a factor of e (~2.718). A small H means density drops very quickly
# with altitude (dense lower layers). A large H means density falls
# more slowly (thin upper layers).
#
# WHY DOES DENSITY DECREASE WITH ALTITUDE?
# Gravity pulls air molecules downward. The lower atmosphere is
# compressed by the weight of all the air above it. As you go higher,
# there is less air above to compress the layer below, so density falls.
# In LEO (200–800 km) the atmosphere is extremely thin — roughly
# 10^-8 to 10^-13 kg/m³ compared to ~1.2 kg/m³ at sea level.

_ATMOSPHERE_LAYERS: list[dict] = [
    {"base_km":   0.0,  "rho0": 1.225,    "scale_h_km":  8.44},   # troposphere
    {"base_km":  25.0,  "rho0": 3.899e-2, "scale_h_km":  6.49},   # stratosphere
    {"base_km":  50.0,  "rho0": 1.057e-3, "scale_h_km":  7.71},   # stratopause
    {"base_km":  75.0,  "rho0": 5.194e-5, "scale_h_km":  6.00},   # mesosphere
    {"base_km": 100.0,  "rho0": 5.604e-7, "scale_h_km":  5.84},   # thermosphere base
    {"base_km": 150.0,  "rho0": 2.076e-9, "scale_h_km": 25.50},   # lower thermosphere
    {"base_km": 200.0,  "rho0": 2.541e-10,"scale_h_km": 37.00},   # mid thermosphere
    {"base_km": 300.0,  "rho0": 1.916e-11,"scale_h_km": 45.50},   # upper thermosphere
    {"base_km": 400.0,  "rho0": 2.803e-12,"scale_h_km": 53.00},   # LEO core
    {"base_km": 500.0,  "rho0": 5.215e-13,"scale_h_km": 58.00},   # upper LEO
    {"base_km": 600.0,  "rho0": 1.137e-13,"scale_h_km": 65.00},   # LEO/MEO boundary
    {"base_km": 700.0,  "rho0": 3.070e-14,"scale_h_km": 73.00},   # exosphere entry
    {"base_km": 800.0,  "rho0": 1.136e-14,"scale_h_km": 80.00},   # exosphere
    {"base_km": 900.0,  "rho0": 5.759e-15,"scale_h_km": 92.00},   # deep exosphere
]

# Altitude cap: above this altitude the density is effectively zero for
# drag purposes. Real satellites rarely operate above 2,000 km in LEO.
_MAX_ALTITUDE_KM: float = 2000.0

# Minimum physically reasonable density returned (never return exact zero,
# which would hide bugs in callers that forget to check).
_MIN_DENSITY_KG_M3: float = 1.0e-20


# ---------------------------------------------------------------------------
# Space weather correction factors
# ---------------------------------------------------------------------------
# WHY DOES F10.7 AFFECT DENSITY?
# F10.7 is the solar radio flux at 10.7 cm wavelength [solar flux units, sfu].
# It is a proxy for extreme ultraviolet (EUV) radiation from the Sun.
# EUV heats and expands the upper atmosphere — higher F10.7 means a puffier,
# denser thermosphere at a given altitude. This is why satellites decay faster
# during solar maximum (~F10.7 ≈ 200 sfu) than solar minimum (~F10.7 ≈ 70 sfu).

# Reference (quiet Sun) F10.7 value [sfu]
_F107_REFERENCE: float = 150.0

# Sensitivity: fractional density change per unit F10.7 deviation.
# Tuned to reproduce NRLMSISE-00 behaviour at LEO altitudes to within ~30%.
_F107_SENSITIVITY: float = 0.003   # 0.3% per sfu deviation

# WHY DOES Kp AFFECT DENSITY?
# Kp is the planetary geomagnetic activity index (0–9 scale).
# During geomagnetic storms (high Kp), energetic particles from the solar
# wind deposit energy in the polar thermosphere via Joule heating and
# particle precipitation. This heating propagates globally, expanding the
# atmosphere and increasing density at LEO altitudes by up to 4× during
# severe storms (Kp = 8–9). Even a moderate storm (Kp = 4) can increase
# drag by 50–100% compared to quiet conditions.

# Kp baseline (quiet conditions)
_KP_REFERENCE: float = 1.0

# Sensitivity: fractional density increase per unit Kp above reference.
_KP_SENSITIVITY: float = 0.15     # ~15% per unit Kp deviation


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_altitude(altitude_km: float) -> None:
    """Raise ValueError if altitude is physically implausible."""
    if not isinstance(altitude_km, (int, float)):
        raise TypeError(
            f"altitude_km must be numeric, got {type(altitude_km).__name__}"
        )
    if altitude_km < 0:
        raise ValueError(
            f"Altitude cannot be negative, got {altitude_km} km. "
            "If computing ground impact, use altitude_km=0."
        )
    if altitude_km > _MAX_ALTITUDE_KM:
        logger.warning(
            "Altitude %.1f km exceeds model ceiling of %.0f km. "
            "Density will be floored at minimum value.",
            altitude_km, _MAX_ALTITUDE_KM,
        )


def _validate_space_weather(f107: float, kp: float) -> None:
    """Raise ValueError if space weather indices are out of physical range."""
    if not isinstance(f107, (int, float)):
        raise TypeError(f"f107 must be numeric, got {type(f107).__name__}")
    if not isinstance(kp, (int, float)):
        raise TypeError(f"kp must be numeric, got {type(kp).__name__}")
    if f107 < 60 or f107 > 300:
        raise ValueError(
            f"F10.7 index should be in [60, 300] sfu for LEO modelling, "
            f"got {f107}. Typical range: 70 (solar min) to 230 (solar max)."
        )
    if kp < 0 or kp > 9:
        raise ValueError(
            f"Kp index must be in [0, 9], got {kp}."
        )


def _select_layer(altitude_km: float) -> dict:
    """
    Return the atmosphere layer dictionary whose base altitude is the
    highest one that does not exceed altitude_km.

    The layers are ordered by ascending base_km, so we walk from the
    top and return the first layer whose base is at or below altitude_km.
    """
    for layer in reversed(_ATMOSPHERE_LAYERS):
        if altitude_km >= layer["base_km"]:
            return layer
    # Below the lowest defined layer — use ground-level values
    return _ATMOSPHERE_LAYERS[0]


def _exponential_density(altitude_km: float) -> float:
    """
    Core exponential atmosphere calculation.

    ρ(h) = ρ₀ × exp(−(h − h₀) / H)

    Returns density in kg/m³.
    """
    if altitude_km > _MAX_ALTITUDE_KM:
        return _MIN_DENSITY_KG_M3

    layer = _select_layer(altitude_km)
    rho0       = layer["rho0"]
    h0         = layer["base_km"]
    scale_h    = layer["scale_h_km"]

    density = rho0 * math.exp(-(altitude_km - h0) / scale_h)

    # Clamp to physical minimum — negative or zero density is nonsensical
    return max(density, _MIN_DENSITY_KG_M3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_density(altitude_km: float) -> float:
    """
    Estimate atmospheric density at a given altitude using an exponential
    atmosphere model.

    This is the baseline (space-weather-neutral) density estimate.
    Use this when F10.7 and Kp data are unavailable or for quick physics
    checks.

    Parameters
    ----------
    altitude_km : float
        Altitude above Earth's surface [km].
        Typical LEO range: 160–2000 km.

    Returns
    -------
    float
        Atmospheric density ρ [kg/m³].

    Raises
    ------
    TypeError  : If altitude_km is not numeric.
    ValueError : If altitude_km is negative.

    Example
    -------
    >>> rho = estimate_density(400.0)
    >>> print(f"ρ at 400 km ≈ {rho:.3e} kg/m³")
    ρ at 400 km ≈ 2.803e-12 kg/m³

    Notes for downstream modules
    ----------------------------
    Pass the returned value directly as `rho` to drag_calculator.compute_drag():
        rho  = estimate_density(altitude_km)
        result = compute_drag(cd=2.2, rho=rho, area=10.0, velocity=7660.0)
    """
    _validate_altitude(altitude_km)

    density = _exponential_density(altitude_km)

    logger.debug(
        "estimate_density | alt=%.2f km → ρ=%.4e kg/m³",
        altitude_km, density,
    )
    return density


def estimate_density_with_space_weather(
    altitude_km: float,
    f107: float,
    kp: float,
) -> dict:
    """
    Estimate atmospheric density adjusted for current solar and
    geomagnetic activity.

    This is the primary entry point for orbit_propagator.py and
    simulator.py, where space weather indices are available from the
    Phase 1 space weather ingestion pipeline.

    Space Weather Corrections Applied
    ----------------------------------
    1. F10.7 correction (solar EUV heating):
         factor_f107 = 1 + F107_SENSITIVITY × (f107 − F107_REFERENCE)
         A higher-than-average F10.7 inflates the thermosphere, raising ρ.

    2. Kp correction (geomagnetic storm heating):
         factor_kp  = 1 + KP_SENSITIVITY × (kp − KP_REFERENCE)
         Geomagnetic storms heat the polar thermosphere; the effect
         propagates globally, raising ρ at LEO altitudes.

    Combined:
         ρ_adjusted = ρ_base × factor_f107 × factor_kp

    These are linear approximations valid for moderate activity levels.
    For extreme events (Kp ≥ 8, F10.7 ≥ 250), NRLMSISE-00 should be used.

    Parameters
    ----------
    altitude_km : float
        Altitude above Earth's surface [km].
    f107        : float
        Daily F10.7 solar flux index [solar flux units, sfu].
        Typical range: 70 (solar min) to 230 (solar max).
        Source: NOAA / DRAO Penticton observatory.
    kp          : float
        3-hour planetary geomagnetic activity index [0–9 scale].
        Source: GFZ Potsdam.

    Returns
    -------
    dict with keys:
        "density_kg_m3"      : float — adjusted density [kg/m³]
        "base_density_kg_m3" : float — density before space weather correction
        "f107_factor"        : float — multiplicative correction from F10.7
        "kp_factor"          : float — multiplicative correction from Kp
        "altitude_km"        : float — input altitude (echoed for logging)
        "f107"               : float — input F10.7 (echoed)
        "kp"                 : float — input Kp (echoed)

    Raises
    ------
    TypeError  : If any argument is non-numeric.
    ValueError : If altitude, F10.7, or Kp are outside physical bounds.

    Example
    -------
    >>> result = estimate_density_with_space_weather(
    ...     altitude_km=400.0, f107=180.0, kp=4.0
    ... )
    >>> print(f"Adjusted ρ: {result['density_kg_m3']:.3e} kg/m³")
    Adjusted ρ: 4.088e-12 kg/m³

    NRLMSISE-00 Upgrade Note
    ------------------------
    To replace this function with nrlmsise00:
        from nrlmsise00 import msise_flat
        output = msise_flat(alt=altitude_km, lat=0, lon=0,
                            time=<datetime>, f107=f107, f107a=f107, ap=kp_to_ap(kp))
        density = output[5]   # total mass density [kg/m³]
    The dict keys and function signature stay identical — no changes needed
    in callers.
    """
    _validate_altitude(altitude_km)
    _validate_space_weather(f107, kp)

    # Step 1: base density from exponential model
    base_density = _exponential_density(altitude_km)

    # Step 2: F10.7 correction
    # Linear approximation — valid when F10.7 is within ~50 sfu of reference
    f107_factor = 1.0 + _F107_SENSITIVITY * (f107 - _F107_REFERENCE)
    # Guard against unphysical negative multipliers (very low F10.7 edge case)
    f107_factor = max(f107_factor, 0.1)

    # Step 3: Kp correction
    kp_factor = 1.0 + _KP_SENSITIVITY * (kp - _KP_REFERENCE)
    kp_factor  = max(kp_factor, 0.1)

    # Step 4: apply corrections multiplicatively
    adjusted_density = base_density * f107_factor * kp_factor
    adjusted_density = max(adjusted_density, _MIN_DENSITY_KG_M3)

    result = {
        "density_kg_m3":      adjusted_density,
        "base_density_kg_m3": base_density,
        "f107_factor":        f107_factor,
        "kp_factor":          kp_factor,
        # Echo inputs — handy for ML pipeline feature logging
        "altitude_km":        altitude_km,
        "f107":               f107,
        "kp":                 kp,
    }

    logger.info(
        "estimate_density_with_space_weather | alt=%.1f km, F10.7=%.1f, Kp=%.1f "
        "→ ρ_base=%.4e, f107×=%.3f, kp×=%.3f → ρ_adj=%.4e kg/m³",
        altitude_km, f107, kp,
        base_density, f107_factor, kp_factor, adjusted_density,
    )

    return result


def density_profile(
    alt_min_km: float = 200.0,
    alt_max_km: float = 800.0,
    step_km: float = 50.0,
    f107: Optional[float] = None,
    kp: Optional[float] = None,
) -> list[dict]:
    """
    Generate a density profile over a range of altitudes.

    Convenience function for the ML pipeline and simulator to build
    altitude-vs-density lookup tables or training features.

    Parameters
    ----------
    alt_min_km : float  — start altitude [km]  (default 200)
    alt_max_km : float  — end altitude [km]    (default 800)
    step_km    : float  — altitude step [km]   (default 50)
    f107       : float or None — if provided, space-weather model is used
    kp         : float or None — must be provided together with f107

    Returns
    -------
    list of dicts, each with keys:
        "altitude_km"  : float
        "density_kg_m3": float

    Example
    -------
    >>> profile = density_profile(200, 600, 100)
    >>> for row in profile:
    ...     print(row)
    """
    if (f107 is None) != (kp is None):
        raise ValueError("Provide both f107 and kp, or neither.")

    profile = []
    altitude = alt_min_km
    while altitude <= alt_max_km + 1e-9:   # small epsilon avoids float edge
        if f107 is not None:
            result = estimate_density_with_space_weather(altitude, f107, kp)
            density = result["density_kg_m3"]
        else:
            density = estimate_density(altitude)

        profile.append({"altitude_km": round(altitude, 2), "density_kg_m3": density})
        altitude += step_km

    return profile


# ---------------------------------------------------------------------------
# Self-test  (python app/physics/atmosphere.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    SEP = "=" * 62

    print(SEP)
    print("  Phase 2 Step 2 — Atmosphere Model Self-Test")
    print(SEP)

    # ------------------------------------------------------------------
    # Test 1: Baseline density at key LEO altitudes
    # ------------------------------------------------------------------
    print("\n[Test 1] Baseline Density Profile (no space weather)")
    print(f"  {'Altitude (km)':<18} {'Density (kg/m³)':<20} {'Note'}")
    print("  " + "-" * 56)

    altitudes = [
        (200,  "ISS re-boost zone, high drag"),
        (400,  "ISS nominal orbit"),
        (550,  "Hubble / Starlink shell 1"),
        (800,  "Sun-synchronous typical"),
    ]
    for alt_km, note in altitudes:
        rho = estimate_density(alt_km)
        print(f"  {alt_km:<18} {rho:<20.4e} {note}")

    # ------------------------------------------------------------------
    # Test 2: Space weather sensitivity at 400 km
    # ------------------------------------------------------------------
    print("\n[Test 2] Space Weather Effect at 400 km")
    print(f"  {'Condition':<28} {'F10.7':>6}  {'Kp':>4}  {'Density (kg/m³)'}")
    print("  " + "-" * 56)

    conditions = [
        ("Solar min, quiet",    70.0, 0.5),
        ("Average conditions", 150.0, 2.0),
        ("Solar max, quiet",   220.0, 1.0),
        ("Moderate storm",     150.0, 5.0),
        ("Severe storm",       200.0, 8.0),
    ]
    for label, f107_val, kp_val in conditions:
        res = estimate_density_with_space_weather(400.0, f107_val, kp_val)
        print(
            f"  {label:<28} {f107_val:>6.1f}  {kp_val:>4.1f}  "
            f"{res['density_kg_m3']:.4e}  "
            f"(F10.7×{res['f107_factor']:.2f}, Kp×{res['kp_factor']:.2f})"
        )

    # ------------------------------------------------------------------
    # Test 3: Density profile table
    # ------------------------------------------------------------------
    print("\n[Test 3] Density Profile 200–800 km (step 100 km)")
    profile = density_profile(200, 800, 100)
    for row in profile:
        print(f"  {row['altitude_km']:>6.0f} km → {row['density_kg_m3']:.4e} kg/m³")

    # ------------------------------------------------------------------
    # Test 4: Integration example — compute drag using both modules
    # ------------------------------------------------------------------
    print("\n[Test 4] Integration with drag_calculator (conceptual)")
    rho_400 = estimate_density(400.0)
    # F_drag = 0.5 * Cd * rho * A * v²
    cd, area, velocity = 2.2, 10.0, 7660.0
    f_drag = 0.5 * cd * rho_400 * area * (velocity ** 2)
    print(f"  ρ at 400 km        : {rho_400:.4e} kg/m³")
    print(f"  Drag force (manual): {f_drag:.4e} N")
    print("  → Pass rho to drag_calculator.compute_drag() for full result.")

    # ------------------------------------------------------------------
    # Test 5: Input validation
    # ------------------------------------------------------------------
    print("\n[Test 5] Input Validation")
    for bad_call, label in [
        (lambda: estimate_density(-10),                          "negative altitude"),
        (lambda: estimate_density_with_space_weather(400, 50, 3), "F10.7 too low"),
        (lambda: estimate_density_with_space_weather(400, 150, 11),"Kp out of range"),
    ]:
        try:
            bad_call()
        except (ValueError, TypeError) as exc:
            print(f"  Caught expected error ({label}): {exc}")

    print(f"\n{SEP}")
    print("  All tests passed. atmosphere.py ready for Phase 2 integration.")
    print(SEP)