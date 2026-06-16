"""Space weather CSV ingestion: validate and persist to SQLite."""

import logging
import os
from typing import Any

import pandas as pd
from flask import Flask

from app.database import db
from app.models.space_weather import SpaceWeather

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["date", "f107", "kp", "ap"]


class SpaceWeatherIngestionError(Exception):
    """Base error for space weather ingestion failures."""


class ColumnValidationError(SpaceWeatherIngestionError):
    """Raised when the CSV is missing required columns."""


class WeatherProcessor:
    """Read space weather CSV files and store records in the database."""

    def __init__(self) -> None:
        self.required_columns = REQUIRED_COLUMNS

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

        logger.info("Reading space weather CSV: %s", csv_path)

        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError as exc:
            raise SpaceWeatherIngestionError(
                f"CSV file is empty: {csv_path}"
            ) from exc
        except Exception as exc:
            raise SpaceWeatherIngestionError(
                f"Failed to read CSV file: {csv_path}"
            ) from exc

        if df.empty:
            raise SpaceWeatherIngestionError(
                f"CSV file contains no data rows: {csv_path}"
            )

        self.validate_columns(df)
        logger.info("Validated columns. Rows to process: %d", len(df))
        return df

    def _row_to_record(
        self, row: pd.Series, source_file: str
    ) -> SpaceWeather | None:
        """Convert one CSV row into a database record, or None if the row is invalid."""
        try:
            timestamp = pd.to_datetime(row["date"], utc=True).to_pydatetime()
            f107 = float(row["f107"])
            kp = float(row["kp"])
            ap = float(row["ap"])
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping invalid row (date=%s): %s", row.get("date"), exc)
            return None

        return SpaceWeather(
            timestamp=timestamp,
            f107=f107,
            kp=kp,
            ap=ap,
            source=source_file,
        )

    def process_csv(self, csv_path: str) -> dict[str, Any]:
        """
        Ingest a space weather CSV into SQLite.

        Returns a summary dict with inserted/skipped counts.
        Must be called inside a Flask application context.
        """
        csv_path = os.path.abspath(csv_path)
        source_name = os.path.basename(csv_path)
        df = self.read_csv(csv_path)

        records: list[SpaceWeather] = []
        skipped = 0

        for _, row in df.iterrows():
            record = self._row_to_record(row, source_name)
            if record is None:
                skipped += 1
                continue
            records.append(record)

        if not records:
            raise SpaceWeatherIngestionError(
                f"No valid rows to insert from CSV: {csv_path}"
            )

        try:
            db.session.add_all(records)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception("Database commit failed for %s", csv_path)
            raise SpaceWeatherIngestionError(
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


def ingest_space_weather_csv(
    csv_path: str, app: Flask | None = None
) -> dict[str, Any]:
    """
    Convenience wrapper that opens a Flask app context and runs ingestion.

    Example:
        from app.space_weather.weather_processor import ingest_space_weather_csv
        result = ingest_space_weather_csv("data/sample_space_weather.csv")
    """
    if app is None:
        from app import create_app

        app = create_app()

    with app.app_context():
        processor = WeatherProcessor()
        return processor.process_csv(csv_path)
