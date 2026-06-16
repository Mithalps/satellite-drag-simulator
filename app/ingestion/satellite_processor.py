"""Satellite CSV ingestion: validate, compute derived fields, persist to SQLite."""

import logging
import math
import os
from typing import Any

import pandas as pd
from flask import Flask

from app.database import db
from app.models.satellite_state import SatelliteStateRecord

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["epoch", "x", "y", "z", "vx", "vy", "vz"]
EARTH_RADIUS_KM = 6371.0


class SatelliteIngestionError(Exception):
    """Base error for satellite ingestion failures."""


class ColumnValidationError(SatelliteIngestionError):
    """Raised when the CSV is missing required columns."""


class SatelliteProcessor:
    """Read satellite state CSV files and store processed records in the database."""

    def __init__(self) -> None:
        self.required_columns = REQUIRED_COLUMNS

    @staticmethod
    def calculate_altitude(x: float, y: float, z: float) -> float:
        """Altitude in km from ECI position (x, y, z) in km."""
        radius_km = math.sqrt(x * x + y * y + z * z)
        return radius_km - EARTH_RADIUS_KM

    @staticmethod
    def calculate_velocity_magnitude(vx: float, vy: float, vz: float) -> float:
        """Speed in km/s from velocity components (vx, vy, vz) in km/s."""
        return math.sqrt(vx * vx + vy * vy + vz * vz)

    def validate_columns(self, df: pd.DataFrame) -> None:
        """Ensure all required columns exist."""
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ColumnValidationError(
                f"CSV is missing required columns: {', '.join(missing)}"
            )

    def read_csv(self, csv_path: str) -> pd.DataFrame:
        """Load a CSV file with basic existence and emptiness checks."""
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info("Reading satellite CSV: %s", csv_path)

        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError as exc:
            raise SatelliteIngestionError(f"CSV file is empty: {csv_path}") from exc
        except Exception as exc:
            raise SatelliteIngestionError(
                f"Failed to read CSV file: {csv_path}"
            ) from exc

        if df.empty:
            raise SatelliteIngestionError(f"CSV file contains no data rows: {csv_path}")

        self.validate_columns(df)
        logger.info("Validated columns. Rows to process: %d", len(df))
        return df

    def _row_to_record(
        self, row: pd.Series, source_file: str
    ) -> SatelliteStateRecord | None:
        """Convert one CSV row into a database record, or None if the row is invalid."""
        try:
            epoch = pd.to_datetime(row["epoch"], utc=True).to_pydatetime()
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            vx = float(row["vx"])
            vy = float(row["vy"])
            vz = float(row["vz"])
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping invalid row (epoch=%s): %s", row.get("epoch"), exc)
            return None

        altitude_km = self.calculate_altitude(x, y, z)
        velocity_magnitude_kms = self.calculate_velocity_magnitude(vx, vy, vz)

        return SatelliteStateRecord(
            epoch=epoch,
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            altitude_km=altitude_km,
            velocity_magnitude_kms=velocity_magnitude_kms,
            source_file=source_file,
        )

    def process_csv(self, csv_path: str) -> dict[str, Any]:
        """
        Ingest a satellite state CSV into SQLite.

        Returns a summary dict with inserted/skipped/failed counts.
        Must be called inside a Flask application context.
        """
        csv_path = os.path.abspath(csv_path)
        source_name = os.path.basename(csv_path)
        df = self.read_csv(csv_path)

        records: list[SatelliteStateRecord] = []
        skipped = 0

        for _, row in df.iterrows():
            record = self._row_to_record(row, source_name)
            if record is None:
                skipped += 1
                continue
            records.append(record)

        if not records:
            raise SatelliteIngestionError(
                f"No valid rows to insert from CSV: {csv_path}"
            )

        try:
            db.session.add_all(records)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception("Database commit failed for %s", csv_path)
            raise SatelliteIngestionError(
                f"Failed to store records in database: {exc}"
            ) from exc

        summary = {
            "source_file": source_name,
            "rows_read": len(df),
            "rows_inserted": len(records),
            "rows_skipped": skipped,
        }
        logger.info("Ingestion complete: %s", summary)
        return summary


def ingest_satellite_csv(csv_path: str, app: Flask | None = None) -> dict[str, Any]:
    """
    Convenience wrapper that opens a Flask app context and runs ingestion.

    Example:
        from app.ingestion.satellite_processor import ingest_satellite_csv
        result = ingest_satellite_csv("data/satellite_states.csv")
    """
    if app is None:
        from app import create_app

        app = create_app()

    with app.app_context():
        processor = SatelliteProcessor()
        return processor.process_csv(csv_path)
