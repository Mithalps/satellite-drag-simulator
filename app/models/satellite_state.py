from datetime import datetime, timezone

from app.database import db


class SatelliteStateRecord(db.Model):
    """State vector snapshot ingested from CSV (position + velocity)."""

    __tablename__ = "satellite_state_records"

    id = db.Column(db.Integer, primary_key=True)
    epoch = db.Column(db.DateTime, nullable=False, index=True)

    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)
    z = db.Column(db.Float, nullable=False)
    vx = db.Column(db.Float, nullable=False)
    vy = db.Column(db.Float, nullable=False)
    vz = db.Column(db.Float, nullable=False)

    altitude_km = db.Column(db.Float, nullable=False)
    velocity_magnitude_kms = db.Column(db.Float, nullable=False)

    source_file = db.Column(db.String(512))
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    def to_dict(self):
        return {
            "id": self.id,
            "epoch": self.epoch.isoformat() if self.epoch else None,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "altitude_km": self.altitude_km,
            "velocity_magnitude_kms": self.velocity_magnitude_kms,
            "source_file": self.source_file,
        }
