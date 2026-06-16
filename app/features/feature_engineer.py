"""Feature engineering: join satellite state data with space weather by timestamp."""

import logging
from typing import Any

import pandas as pd
from flask import Flask

from app.database import db
from app.models.feature import FeatureRecord
from app.models.satellite import Satellite
from app.models.satellite_state import SatelliteStateRecord
from app.models.space_weather import SpaceWeather

logger = logging.getLogger(__name__)

DEFAULT_SATELLITE_NORAD_ID = 99999
DEFAULT_SATELLITE_NAME = "CSV State Import"


class FeatureEngineeringError(Exception):
    """Base error for feature engineering failures."""


class FeatureEngineer:
    """Build ML-ready features from satellite state and space weather records."""

    def _ensure_feature_schema(self) -> None:
        """Add new columns to feature_records if the database was created earlier."""
        with db.engine.connect() as conn:
            columns = {
                row[1]
                for row in conn.exec_driver_sql(
                    "PRAGMA table_info(feature_records)"
                ).fetchall()
            }
            if "ap" not in columns:
                logger.info("Adding missing ap column to feature_records table")
                conn.exec_driver_sql("ALTER TABLE feature_records ADD COLUMN ap FLOAT")
                conn.commit()

    def _get_default_satellite_id(self) -> int:
        """Return a satellite id for feature rows sourced from state vector CSV data."""
        satellite = Satellite.query.filter_by(
            norad_id=DEFAULT_SATELLITE_NORAD_ID
        ).first()
        if satellite is None:
            satellite = Satellite(
                norad_id=DEFAULT_SATELLITE_NORAD_ID,
                name=DEFAULT_SATELLITE_NAME,
            )
            db.session.add(satellite)
            db.session.commit()
            logger.info("Created default satellite record for feature engineering")

        return satellite.id

    def load_satellite_states(self) -> pd.DataFrame:
        """Load satellite state records from SQLite into a DataFrame."""
        logger.info("Loading satellite state records from database")

        records = SatelliteStateRecord.query.order_by(SatelliteStateRecord.epoch).all()
        if not records:
            raise FeatureEngineeringError(
                "No satellite state records found. Run satellite ingestion first."
            )

        df = pd.DataFrame(
            [
                {
                    "state_id": r.id,
                    "epoch": r.epoch,
                    "altitude_km": r.altitude_km,
                    "velocity_magnitude_kms": r.velocity_magnitude_kms,
                }
                for r in records
            ]
        )
        df["epoch"] = pd.to_datetime(df["epoch"], utc=True)
        logger.info("Loaded %d satellite state records", len(df))
        return df

    def load_space_weather(self) -> pd.DataFrame:
        """Load space weather records from SQLite into a DataFrame."""
        logger.info("Loading space weather records from database")

        records = SpaceWeather.query.order_by(SpaceWeather.timestamp).all()
        if not records:
            raise FeatureEngineeringError(
                "No space weather records found. Run space weather ingestion first."
            )

        df = pd.DataFrame(
            [
                {
                    "weather_id": r.id,
                    "timestamp": r.timestamp,
                    "f107": r.f107,
                    "kp": r.kp,
                    "ap": r.ap,
                }
                for r in records
            ]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        logger.info("Loaded %d space weather records", len(df))
        return df

    def match_by_nearest_timestamp(
        self, states_df: pd.DataFrame, weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Match each satellite state row to the nearest space weather timestamp."""
        states = states_df.sort_values("epoch").reset_index(drop=True)
        weather = weather_df.sort_values("timestamp").reset_index(drop=True)

        merged = pd.merge_asof(
            states,
            weather,
            left_on="epoch",
            right_on="timestamp",
            direction="nearest",
        )

        missing = merged["f107"].isna().sum()
        if missing:
            logger.warning(
                "%d satellite rows could not be matched to space weather", missing
            )

        matched = merged.dropna(subset=["f107", "kp", "ap"]).reset_index(drop=True)
        if matched.empty:
            raise FeatureEngineeringError(
                "No records could be matched between satellite states and space weather."
            )

        logger.info("Matched %d feature rows", len(matched))
        return matched

    def _rows_to_feature_records(
        self, dataset: pd.DataFrame, satellite_id: int
    ) -> list[FeatureRecord]:
        """Convert a merged DataFrame into FeatureRecord ORM objects."""
        records: list[FeatureRecord] = []

        for _, row in dataset.iterrows():
            records.append(
                FeatureRecord(
                    satellite_id=satellite_id,
                    timestamp=row["epoch"].to_pydatetime(),
                    altitude_km=float(row["altitude_km"]),
                    orbital_velocity_kms=float(row["velocity_magnitude_kms"]),
                    f107=float(row["f107"]),
                    kp=float(row["kp"]),
                    ap=float(row["ap"]),
                )
            )

        return records

    def build_and_save(self) -> dict[str, Any]:
        """
        Build the feature dataset and save rows to feature_records.

        Must be called inside a Flask application context.
        """
        self._ensure_feature_schema()
        satellite_id = self._get_default_satellite_id()

        states_df = self.load_satellite_states()
        weather_df = self.load_space_weather()
        dataset = self.match_by_nearest_timestamp(states_df, weather_df)

        feature_records = self._rows_to_feature_records(dataset, satellite_id)
        if not feature_records:
            raise FeatureEngineeringError("No feature records were generated.")

        try:
            db.session.add_all(feature_records)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception("Failed to save feature records")
            raise FeatureEngineeringError(
                f"Failed to store feature records in database: {exc}"
            ) from exc

        summary = {
            "satellite_states_loaded": len(states_df),
            "space_weather_loaded": len(weather_df),
            "features_created": len(feature_records),
        }
        logger.info("Feature engineering complete: %s", summary)
        return summary


def build_feature_dataset(app: Flask | None = None) -> dict[str, Any]:
    """
    Convenience wrapper that opens a Flask app context and runs feature engineering.

    Example:
        from app.features.feature_engineer import build_feature_dataset
        result = build_feature_dataset()
    """
    if app is None:
        from app import create_app

        app = create_app()

    with app.app_context():
        engineer = FeatureEngineer()
        return engineer.build_and_save()
