import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load .env file if available
load_dotenv(BASE_DIR / ".env")

class Config:
    """Configuration settings for Road Accident Data Analysis System."""
    
    SECRET_KEY = os.environ.get("SECRET_KEY", "road-accident-analysis-secret-key-2026")
    
    # MySQL Database Credentials
    DB_USER = os.environ.get("DB_USER", "root")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "admin")
    DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
    DB_PORT = os.environ.get("DB_PORT", "3306")
    DB_NAME = os.environ.get("DB_NAME", "road_accident_db")
    
    # Engine type preference: 'mysql' or 'sqlite'
    # Will automatically attempt MySQL first, with fallback to SQLite
    DB_ENGINE_TYPE = os.environ.get("DB_ENGINE_TYPE", "mysql")
    
    # SQLite fallback path
    SQLITE_PATH = BASE_DIR / "accident_data.db"
    
    # Matplotlib chart output directory
    STATIC_DIR = BASE_DIR / "static"
    CHARTS_DIR = STATIC_DIR / "charts"
    REPORTS_DIR = BASE_DIR / "reports"
    SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
    
    # Standard classification categories for validation
    ALLOWED_SEVERITIES = ["Minor", "Moderate", "Severe", "Fatal"]
    ALLOWED_VEHICLE_TYPES = [
        "Two-Wheeler",
        "Passenger Car",
        "Commercial Truck",
        "Bus",
        "Bicycle / Pedestrian",
        "Van / SUV",
        "Auto Rickshaw",
        "Other"
    ]
    ALLOWED_WEATHER = [
        "Clear",
        "Rain",
        "Fog / Mist",
        "Heavy Wind",
        "Snow / Sleet",
        "Overcast / Cloudy",
        "Unknown"
    ]
    ALLOWED_LIGHT_CONDITIONS = [
        "Daylight",
        "Dusk / Dawn",
        "Dark - Streetlights On",
        "Dark - No Streetlights",
        "Unknown"
    ]
    ALLOWED_AREA_TYPES = [
        "Urban",
        "Suburban",
        "Highway",
        "Rural"
    ]
    
    # Mandatory analytical caveat
    DISCLAIMER_TEXT = (
        "ANALYTICAL NOTICE: Findings describe statistical associations and patterns in recorded data. "
        "They should not be treated as proof of causation. Environmental, behavioral, and infrastructure "
        "factors interact complexly in traffic incidents."
    )
