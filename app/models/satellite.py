from datetime import datetime, timezone

from app.database import db


class Satellite(db.Model):
    """Orbital and physical properties for a tracked satellite."""

    __tablename__ = "satellites"

    id = db.Column(db.Integer, primary_key=True)
    norad_id = db.Column(db.Integer, unique=True, nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)

    tle_line1 = db.Column(db.String(255))
    tle_line2 = db.Column(db.String(255))

    mass_kg = db.Column(db.Float)
    area_m2 = db.Column(db.Float)
    drag_coefficient = db.Column(db.Float)

    inclination = db.Column(db.Float)
    eccentricity = db.Column(db.Float)
    mean_motion = db.Column(db.Float)

    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    features = db.relationship("FeatureRecord", back_populates="satellite")

    def to_dict(self):
        return {
            "id": self.id,
            "norad_id": self.norad_id,
            "name": self.name,
            "mass_kg": self.mass_kg,
            "area_m2": self.area_m2,
            "drag_coefficient": self.drag_coefficient,
            "inclination": self.inclination,
            "eccentricity": self.eccentricity,
            "mean_motion": self.mean_motion,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
