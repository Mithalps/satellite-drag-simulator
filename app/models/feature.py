from datetime import datetime, timezone

from app.database import db


class FeatureRecord(db.Model):
    """Engineered features used for drag and decay analysis."""

    __tablename__ = "feature_records"

    id = db.Column(db.Integer, primary_key=True)
    satellite_id = db.Column(
        db.Integer, db.ForeignKey("satellites.id"), nullable=False, index=True
    )
    timestamp = db.Column(db.DateTime, nullable=False, index=True)

    altitude_km = db.Column(db.Float)
    area_to_mass_ratio = db.Column(db.Float)
    orbital_velocity_kms = db.Column(db.Float)
    f107 = db.Column(db.Float)
    kp = db.Column(db.Float)
    ap = db.Column(db.Float)
    density_proxy = db.Column(db.Float)

    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    satellite = db.relationship("Satellite", back_populates="features")

    def to_dict(self):
        return {
            "id": self.id,
            "satellite_id": self.satellite_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "altitude_km": self.altitude_km,
            "area_to_mass_ratio": self.area_to_mass_ratio,
            "orbital_velocity_kms": self.orbital_velocity_kms,
            "f107": self.f107,
            "kp": self.kp,
            "ap": self.ap,
            "density_proxy": self.density_proxy,
        }
