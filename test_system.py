import datetime
from pathlib import Path
import pytest
import pandas as pd
from config import Config
from models import Location, Accident
from database import get_session, init_db
from validator import RecordValidator, DataQualityAuditor
from analyzer import AccidentAnalyzer
from visualizer import AccidentVisualizer
from reporter import AccidentReporter
from app import app

@pytest.fixture(scope="module")
def setup_test_db():
    """Initializes tables for test runs."""
    init_db()
    yield

def test_validation_valid_record():
    valid_data = {
        "accident_datetime": "2025-06-15 14:30:00",
        "region": "Downtown Commercial District",
        "road_name_optional": "Main Street Arterial",
        "area_type": "Urban",
        "vehicle_type": "car",
        "severity": "severe",
        "weather_optional": "Rain",
        "light_condition_optional": "Daylight",
        "casualty_count_optional": 2,
        "notes_optional": "Wet asphalt skid marks observed."
    }
    is_valid, cleaned, errors = RecordValidator.validate_accident_record(valid_data)
    assert is_valid is True
    assert len(errors) == 0
    assert cleaned["vehicle_type"] == "Passenger Car"
    assert cleaned["severity"] == "Severe"
    assert cleaned["casualty_count_optional"] == 2

def test_validation_future_datetime_rejected():
    future_data = {
        "accident_datetime": "2099-01-01 12:00:00",
        "region": "Downtown Commercial District",
        "vehicle_type": "car",
        "severity": "minor"
    }
    is_valid, cleaned, errors = RecordValidator.validate_accident_record(future_data)
    assert is_valid is False
    assert any("future" in err.lower() for err in errors)

def test_validation_invalid_severity_rejected():
    bad_sev_data = {
        "accident_datetime": "2025-05-10 10:00:00",
        "region": "Downtown Commercial District",
        "vehicle_type": "car",
        "severity": "super_catastrophic"
    }
    is_valid, cleaned, errors = RecordValidator.validate_accident_record(bad_sev_data)
    assert is_valid is False
    assert any("invalid severity" in err.lower() for err in errors)

def test_pii_detection_blocks_personal_data():
    # Phone number
    has_pii, types = RecordValidator.check_pii("Driver phone was +1-555-839-2019 at scene.")
    assert has_pii is True
    assert any("phone" in t.lower() for t in types)

    # Email
    has_pii, types = RecordValidator.check_pii("Contact officer at john.doe@police.gov.in")
    assert has_pii is True
    assert any("email" in t.lower() for t in types)

    # License plate
    has_pii, types = RecordValidator.check_pii("Vehicle plate DL 01 AB 1234 hit divider")
    assert has_pii is True
    assert any("plate" in t.lower() for t in types)

    # Clean text
    has_pii, types = RecordValidator.check_pii("Rear-end impact due to sudden braking in slow queue.")
    assert has_pii is False
    assert len(types) == 0

def test_data_quality_auditor():
    sample_df = pd.DataFrame([
        {"accident_id": 1, "accident_datetime": "2025-01-01 10:00:00", "region": "North Zone", "road_name_optional": None, "vehicle_type": "Passenger Car", "severity": "Minor"},
        {"accident_id": 2, "accident_datetime": "2025-01-02 11:00:00", "region": "South Zone", "road_name_optional": "High St", "vehicle_type": "Two-Wheeler", "severity": "Severe"}
    ])
    audit = DataQualityAuditor.audit_dataset(sample_df)
    assert audit["total_records"] == 2
    assert "road_name_optional" in audit["missing_fields"]
    assert audit["missing_fields"]["road_name_optional"]["missing_count"] == 1
    assert audit["pii_compliance_status"] == "CERTIFIED_NO_PII"

def test_analyzer_load_and_numpy_stats(setup_test_db):
    df = AccidentAnalyzer.load_data()
    assert not df.empty
    assert len(df) >= 100

    summary = AccidentAnalyzer.get_full_analysis_summary(df)
    assert summary["status"] == "success"
    assert "hourly" in summary["time_analysis"]
    assert "by_region" in summary["location_analysis"]
    assert "distribution" in summary["severity_analysis"]

    # Verify NumPy statistics
    nstats = summary["numpy_descriptive_statistics"]
    assert nstats["hourly_statistics"]["mean"] > 0
    assert nstats["daily_statistics"]["mean_accidents_per_day"] > 0
    assert "ci_95_percent" in nstats["daily_statistics"]

def test_visualizer_outputs(setup_test_db):
    df = AccidentAnalyzer.load_data()
    charts = AccidentVisualizer.generate_all_charts(df)
    assert len(charts) == 8
    for chart_key, chart_path in charts.items():
        p = Config.CHARTS_DIR / f"{Path(chart_path).name}"
        assert p.exists()
        assert p.stat().st_size > 1000  # At least 1KB

def test_reporter_generation(setup_test_db):
    md_path = AccidentReporter.generate_markdown_report()
    html_path = AccidentReporter.generate_html_report()
    assert md_path.exists()
    assert html_path.exists()
    assert "Road Accident Data Analysis" in md_path.read_text(encoding="utf-8")
    assert "Road Accident Data Analysis" in html_path.read_text(encoding="utf-8")

def test_flask_web_routes():
    app.config["TESTING"] = True
    client = app.test_client()

    routes = ["/", "/records", "/analytics", "/data-quality", "/reports", "/api/summary", "/api/quality-audit"]
    for r in routes:
        resp = client.get(r)
        assert resp.status_code == 200, f"Route {r} failed with status {resp.status_code}"
