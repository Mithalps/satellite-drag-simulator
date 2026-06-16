from datetime import datetime, timezone

from app.database import db


class SpaceWeather(db.Model):
    """Solar and geomagnetic activity measurements."""

    __tablename__ = "space_weather"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    f107 = db.Column(db.Float)
    kp = db.Column(db.Float)
    ap = db.Column(db.Float)
    source = db.Column(db.String(100), default="unknown")

    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "f107": self.f107,
            "kp": self.kp,
            "ap": self.ap,
            "source": self.source,
        }
