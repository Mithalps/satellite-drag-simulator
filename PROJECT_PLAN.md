# AI-Powered Satellite Drag Prediction and Orbital Decay Simulator

## Overview

End-to-end system to ingest satellite and space weather data, engineer predictive features, simulate orbital decay with a physics engine, and forecast drag-driven decay using machine learning. Results are exposed through a web dashboard with Three.js orbital visualization.

## Current Phase: Phase 1 — Data Foundation

Build the backend infrastructure and data pipeline that all later phases depend on.

| # | Goal | Description | Status |
|---|------|-------------|--------|
| 1 | Flask backend | REST API for ingestion, queries, and future simulation/ML endpoints | Not started |
| 2 | SQLite database | Persistent storage for satellites, orbits, space weather, and engineered features | Not started |
| 3 | Satellite data ingestion | Import TLE/orbital parameters, physical properties (mass, area, \( C_d \)) | Not started |
| 4 | Space weather ingestion | Import solar/geomagnetic indices (e.g., F10.7, Kp, Ap) for density modeling | Not started |
| 5 | Feature engineering | Derive ML-ready features: altitude, area-to-mass ratio, density proxies, decay rates | Not started |

### Phase 1 Deliverables

- [ ] Flask app with modular blueprints (`/api/satellites`, `/api/space-weather`, `/api/features`)
- [ ] SQLite schema and migrations for core tables
- [ ] Ingestion scripts/services for satellite TLE and space weather sources
- [ ] Feature pipeline that joins satellite + space weather data into training-ready rows
- [ ] Basic API tests and sample seed data

---

## Future Phases

| Phase | Goal | Description | Status |
|-------|------|-------------|--------|
| 2 | Physics engine | Numerical orbit propagation with atmospheric drag perturbation | Planned |
| 3 | ML model | Train and serve decay/drag prediction models on engineered features | Planned |
| 4 | Dashboard | Web UI for simulations, predictions, and historical analysis | Planned |
| 5 | Three.js visualization | Interactive 3D Earth orbit and decay trajectory rendering | Planned |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Sources   │────▶│  Flask Backend   │────▶│  SQLite DB      │
│  TLE, SW indices│     │  Ingestion + API │     │  Satellites,    │
└─────────────────┘     └──────────────────┘     │  Space Weather, │
                              │                  │  Features       │
                              ▼                  └─────────────────┘
                    ┌──────────────────┐
                    │ Feature Pipeline │
                    └──────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │  Physics    │    │  ML Model   │    │  Dashboard  │
   │  Engine     │    │  (Phase 3)  │    │  + Three.js │
   └─────────────┘    └─────────────┘    └─────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask |
| Database | SQLite |
| Data ingestion | Requests, scheduled jobs (APScheduler or cron) |
| Feature engineering | Pandas, NumPy |
| Physics (future) | Custom propagator, optional SGP4 for TLE parsing |
| ML (future) | scikit-learn / XGBoost or PyTorch |
| Frontend (future) | HTML/CSS/JS, Three.js |
| API format | JSON REST |

## Project Structure (proposed)

```
SATELLITE DRAG/
├── PROJECT_PLAN.md
├── README.md
├── requirements.txt
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── config.py
│   ├── models/              # SQLAlchemy or raw SQLite models
│   ├── routes/              # API blueprints
│   ├── services/
│   │   ├── ingestion/       # Satellite + space weather fetchers
│   │   └── features/        # Feature engineering pipeline
│   └── db/
│       └── schema.sql
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   └── seed_db.py
└── tests/
```

## Data Model (Phase 1)

### Satellites

| Field | Notes |
|-------|-------|
| `norad_id` | Primary identifier |
| `name` | Satellite name |
| `tle_line1`, `tle_line2` | Two-line element set |
| `mass_kg`, `area_m2`, `cd` | Drag-relevant physical properties |
| `inclination`, `eccentricity`, `mean_motion` | Parsed from TLE |
| `updated_at` | Last ingestion timestamp |

### Space Weather

| Field | Notes |
|-------|-------|
| `timestamp` | Observation time (UTC) |
| `f107` | Solar radio flux (10.7 cm) |
| `kp`, `ap` | Geomagnetic activity indices |
| `source` | Data provider |

### Engineered Features

| Feature | Source |
|---------|--------|
| Altitude / semi-major axis | TLE |
| Area-to-mass ratio | Satellite properties |
| Orbital velocity | Derived from mean motion |
| Solar activity lag features | Space weather time series |
| Density proxy | f(F10.7, Kp, altitude) — simplified until physics engine |

## Phase 1 Next Steps

1. Initialize Flask project and SQLite schema
2. Implement satellite TLE ingestion (CelesTrak / Space-Track)
3. Implement space weather ingestion (NOAA SWPC or similar)
4. Build feature engineering pipeline and persist to DB
5. Expose read endpoints and document API

## Risks & Assumptions

- **Assumption:** TLE updates are sufficient for Phase 1; high-fidelity ephemerides come later
- **Assumption:** Public space weather APIs are available without restricted credentials initially
- **Risk:** Missing physical properties (mass, area) for many satellites — may require defaults or catalog lookup
- **Risk:** Space weather and TLE timestamps must be aligned for valid feature joins

## References

- Vallado, *Fundamentals of Astrodynamics and Applications*
- CelesTrak / Space-Track — TLE data
- NOAA SWPC — space weather indices
- NRLMSISE-00 — atmospheric density (Phase 2)
