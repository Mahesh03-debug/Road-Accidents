import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Location(Base):
    """
    MySQL entity: locations
    Stores standardized geographic locations and road segments.
    """
    __tablename__ = "locations"

    location_id = Column(Integer, primary_key=True, autoincrement=True)
    region = Column(String(100), nullable=False, index=True)
    road_name_optional = Column(String(150), nullable=True, index=True)
    area_type = Column(String(50), nullable=True, default="Urban")  # Urban, Suburban, Highway, Rural
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    accidents = relationship("Accident", back_populates="location", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "location_id": self.location_id,
            "region": self.region,
            "road_name_optional": self.road_name_optional or "Not Specified",
            "area_type": self.area_type or "Unknown",
            "display_name": f"{self.region} ({self.road_name_optional})" if self.road_name_optional else self.region,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self):
        return f"<Location id={self.location_id} region='{self.region}' road='{self.road_name_optional}'>"


class Accident(Base):
    """
    MySQL entity: accidents
    Stores validated traffic accident records.
    Strictly anonymized: contains NO personally identifying details.
    """
    __tablename__ = "accidents"

    accident_id = Column(Integer, primary_key=True, autoincrement=True)
    accident_datetime = Column(DateTime, nullable=False, index=True)
    location_id = Column(Integer, ForeignKey("locations.location_id", ondelete="CASCADE"), nullable=False, index=True)
    vehicle_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(30), nullable=False, index=True)  # Minor, Moderate, Severe, Fatal
    weather_optional = Column(String(50), nullable=True)      # Clear, Rain, Fog / Mist, etc.
    light_condition_optional = Column(String(50), nullable=True) # Daylight, Dusk / Dawn, Dark, etc.
    casualty_count_optional = Column(Integer, nullable=True, default=0)
    notes_optional = Column(Text, nullable=True)  # Strictly non-PII context notes

    # Relationships
    location = relationship("Location", back_populates="accidents")

    def to_dict(self):
        return {
            "accident_id": self.accident_id,
            "accident_datetime": self.accident_datetime.strftime("%Y-%m-%d %H:%M:%S") if self.accident_datetime else None,
            "date": self.accident_datetime.strftime("%Y-%m-%d") if self.accident_datetime else None,
            "time": self.accident_datetime.strftime("%H:%M") if self.accident_datetime else None,
            "location_id": self.location_id,
            "region": self.location.region if self.location else "Unknown",
            "road_name_optional": self.location.road_name_optional if self.location else None,
            "area_type": self.location.area_type if self.location else None,
            "vehicle_type": self.vehicle_type,
            "severity": self.severity,
            "weather_optional": self.weather_optional or "Not Reported",
            "light_condition_optional": self.light_condition_optional or "Not Reported",
            "casualty_count_optional": self.casualty_count_optional or 0,
            "notes_optional": self.notes_optional or ""
        }

    def __repr__(self):
        return f"<Accident id={self.accident_id} dt='{self.accident_datetime}' severity='{self.severity}'>"
