import io
import time
import datetime
from pathlib import Path
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, jsonify, Response
)
import pandas as pd
from sqlalchemy import or_, desc

from config import Config
from database import get_session, check_db_health, init_db
from models import Location, Accident
from validator import RecordValidator, DataQualityAuditor
from analyzer import AccidentAnalyzer
from visualizer import AccidentVisualizer
from reporter import AccidentReporter
from data_generator import seed_database

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Custom Jinja template filter: zip
@app.template_filter("zip")
def jinja_zip(*args):
    return zip(*args)

@app.before_request
def ensure_db():
    # Pre-flight check handled transparently by database.py
    pass

# ==========================================
# DASHBOARD ROUTE
# ==========================================
@app.route("/")
def dashboard():
    """Main executive intelligence dashboard with KPIs and dynamic charts."""
    df = AccidentAnalyzer.load_data()
    if df.empty:
        # Seed if totally empty on first launch
        seed_database(count=1200)
        df = AccidentAnalyzer.load_data()

    summary = AccidentAnalyzer.get_full_analysis_summary(df)

    session = get_session()
    try:
        # Get 5 most recent records
        recent_objs = session.query(Accident).order_by(desc(Accident.accident_datetime)).limit(5).all()
        recent_records = [r.to_dict() for r in recent_objs]
    finally:
        session.close()

    return render_template(
        "dashboard.html",
        active_page="dashboard",
        summary=summary,
        recent_records=recent_records
    )

# ==========================================
# ACCIDENT RECORDS REGISTRY & FILTERING
# ==========================================
@app.route("/records")
def records_list():
    """Browse and filter individual anonymized accident records."""
    session = get_session()
    try:
        page = request.args.get("page", 1, type=int)
        per_page = 20
        search = request.args.get("search", "").strip()
        selected_severity = request.args.get("severity", "").strip()
        selected_region = request.args.get("region", "").strip()
        selected_vehicle = request.args.get("vehicle_type", "").strip()

        query = session.query(Accident).join(Location)

        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    Location.region.ilike(search_term),
                    Location.road_name_optional.ilike(search_term),
                    Accident.vehicle_type.ilike(search_term),
                    Accident.weather_optional.ilike(search_term),
                    Accident.notes_optional.ilike(search_term)
                )
            )

        if selected_severity:
            query = query.filter(Accident.severity == selected_severity)

        if selected_region:
            query = query.filter(Location.region == selected_region)

        if selected_vehicle:
            query = query.filter(Accident.vehicle_type == selected_vehicle)

        total_count = query.count()
        total_pages = max(1, (total_count + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))

        records_objs = query.order_by(desc(Accident.accident_datetime)).offset((page - 1) * per_page).limit(per_page).all()
        records_data = [r.to_dict() for r in records_objs]

        # Filter dropdown lists
        all_regions = [r[0] for r in session.query(Location.region).distinct().order_by(Location.region).all()]
        severities = Config.ALLOWED_SEVERITIES
        vehicle_types = Config.ALLOWED_VEHICLE_TYPES

        # Pagination range
        start_p = max(1, page - 2)
        end_p = min(total_pages, page + 2)
        page_range = list(range(start_p, end_p + 1))

        return render_template(
            "records.html",
            active_page="records",
            records=records_data,
            total_count=total_count,
            current_page=page,
            total_pages=total_pages,
            page_range=page_range,
            search_query=search,
            selected_severity=selected_severity,
            selected_region=selected_region,
            selected_vehicle=selected_vehicle,
            regions=all_regions,
            severities=severities,
            vehicle_types=vehicle_types
        )
    finally:
        session.close()

# ==========================================
# ADD NEW RECORD (WITH VALIDATION & NON-PII)
# ==========================================
@app.route("/records/add", methods=["GET", "POST"])
def add_record():
    """Manual form ingestion for new accident records with validation engine."""
    session = get_session()
    try:
        if request.method == "POST":
            form_dict = request.form.to_dict()
            is_valid, cleaned, errors = RecordValidator.validate_accident_record(form_dict)

            if not is_valid:
                for err in errors:
                    flash(f"Validation Error: {err}", "danger")
                return redirect(url_for("add_record"))

            # Check or create location
            reg_clean = cleaned["region"]
            road_clean = cleaned.get("road_name_optional")
            area_clean = cleaned.get("area_type", "Urban")

            loc = session.query(Location).filter(
                Location.region == reg_clean,
                Location.road_name_optional == road_clean
            ).first()

            if not loc:
                loc = Location(
                    region=reg_clean,
                    road_name_optional=road_clean,
                    area_type=area_clean
                )
                session.add(loc)
                session.flush()

            # Create accident record
            acc = Accident(
                accident_datetime=cleaned["accident_datetime"],
                location_id=loc.location_id,
                vehicle_type=cleaned["vehicle_type"],
                severity=cleaned["severity"],
                weather_optional=cleaned.get("weather_optional"),
                light_condition_optional=cleaned.get("light_condition_optional"),
                casualty_count_optional=cleaned.get("casualty_count_optional", 0),
                notes_optional=cleaned.get("notes_optional")
            )
            session.add(acc)
            session.commit()

            flash(f"Accident record #{acc.accident_id} successfully validated and ingested!", "success")
            return redirect(url_for("records_list"))

        # GET: render form
        existing_regions = [r[0] for r in session.query(Location.region).distinct().order_by(Location.region).all()]
        current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")

        return render_template(
            "add_record.html",
            active_page="add_record",
            existing_regions=existing_regions,
            vehicle_types=Config.ALLOWED_VEHICLE_TYPES,
            current_time=current_time
        )
    finally:
        session.close()

# ==========================================
# DELETE RECORD
# ==========================================
@app.route("/records/delete/<int:accident_id>", methods=["POST"])
def delete_record(accident_id):
    """Deletes an accident record."""
    session = get_session()
    try:
        acc = session.query(Accident).get(accident_id)
        if acc:
            session.delete(acc)
            session.commit()
            flash(f"Record #{accident_id} removed.", "info")
        else:
            flash(f"Record #{accident_id} not found.", "warning")
        return redirect(url_for("records_list"))
    finally:
        session.close()

# ==========================================
# PATTERN ANALYTICS & NUMPY VIEW
# ==========================================
@app.route("/analytics")
def analytics_view():
    """In-depth breakdown of Pandas groupings and NumPy descriptive metrics."""
    df = AccidentAnalyzer.load_data()
    summary = AccidentAnalyzer.get_full_analysis_summary(df)
    return render_template(
        "analytics.html",
        active_page="analytics",
        summary=summary
    )

# ==========================================
# DATA QUALITY & ACCEPTANCE CRITERIA AUDIT
# ==========================================
@app.route("/data-quality")
def data_quality_view():
    """Explicit audit report satisfying project acceptance criteria."""
    df = AccidentAnalyzer.load_data()
    audit = DataQualityAuditor.audit_dataset(df)
    return render_template(
        "data_quality.html",
        active_page="quality",
        audit=audit
    )

# ==========================================
# EXECUTIVE REPORTS & CHARTS GALLERY
# ==========================================
@app.route("/reports")
def reports_view():
    """View generated Matplotlib charts and analytical trend report."""
    md_path = Config.REPORTS_DIR / "accident_trend_report.md"
    if not md_path.exists():
        AccidentReporter.generate_markdown_report()
        AccidentReporter.generate_html_report()

    md_content = md_path.read_text(encoding="utf-8") if md_path.exists() else "No report generated yet."

    return render_template(
        "reports_view.html",
        active_page="reports",
        markdown_content=md_content,
        timestamp=int(time.time())
    )

@app.route("/reports/regenerate-charts")
def regenerate_charts():
    """Re-executes Matplotlib plotting pipeline."""
    df = AccidentAnalyzer.load_data()
    AccidentVisualizer.generate_all_charts(df)
    AccidentReporter.generate_markdown_report()
    AccidentReporter.generate_html_report()
    flash("Successfully re-generated all Matplotlib charts and analytical reports!", "success")
    return redirect(url_for("reports_view"))

# ==========================================
# CSV IMPORT & EXPORT TOOLS
# ==========================================
@app.route("/import-csv", methods=["GET", "POST"])
def import_csv():
    """Upload external CSV accident records with automated validation."""
    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file or file.filename == "":
            flash("Please choose a valid CSV file.", "warning")
            return redirect(url_for("import_csv"))

        try:
            df_uploaded = pd.read_csv(file)
        except Exception as e:
            flash(f"Error parsing CSV: {e}", "danger")
            return redirect(url_for("import_csv"))

        session = get_session()
        inserted_count = 0
        rejected_count = 0
        rejections = []

        try:
            existing_locs = session.query(Location).all()
            loc_cache = {(l.region.strip().lower(), (l.road_name_optional or "").strip().lower()): l.location_id for l in existing_locs}

            for idx, row in df_uploaded.iterrows():
                row_dict = row.to_dict()
                is_valid, cleaned, errors = RecordValidator.validate_accident_record(row_dict)

                if not is_valid:
                    rejected_count += 1
                    if len(rejections) < 8:
                        rejections.append(f"Row {idx + 2}: {'; '.join(errors)}")
                    continue

                reg_key = cleaned["region"].strip().lower()
                road_key = (cleaned.get("road_name_optional") or "").strip().lower()
                cache_key = (reg_key, road_key)

                if cache_key in loc_cache:
                    loc_id = loc_cache[cache_key]
                else:
                    new_loc = Location(
                        region=cleaned["region"],
                        road_name_optional=cleaned.get("road_name_optional"),
                        area_type=cleaned.get("area_type", "Urban")
                    )
                    session.add(new_loc)
                    session.flush()
                    loc_id = new_loc.location_id
                    loc_cache[cache_key] = loc_id

                acc = Accident(
                    accident_datetime=cleaned["accident_datetime"],
                    location_id=loc_id,
                    vehicle_type=cleaned["vehicle_type"],
                    severity=cleaned["severity"],
                    weather_optional=cleaned.get("weather_optional"),
                    light_condition_optional=cleaned.get("light_condition_optional"),
                    casualty_count_optional=cleaned.get("casualty_count_optional", 0),
                    notes_optional=cleaned.get("notes_optional")
                )
                session.add(acc)
                inserted_count += 1

            session.commit()
            flash(f"Batch import completed: {inserted_count} records inserted, {rejected_count} rejected.", "success" if inserted_count > 0 else "warning")

            return render_template(
                "import_csv.html",
                active_page="records",
                import_results={
                    "inserted": inserted_count,
                    "rejected": rejected_count,
                    "rejection_details": rejections
                }
            )

        except Exception as e:
            session.rollback()
            flash(f"Database ingestion error: {e}", "danger")
            return redirect(url_for("import_csv"))
        finally:
            session.close()

    return render_template("import_csv.html", active_page="records", import_results=None)

@app.route("/export-csv")
def export_csv():
    """Exports all current validated accident records to CSV."""
    df = AccidentAnalyzer.load_data()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=road_accidents_export.csv"}
    )

@app.route("/download/sample-csv")
def download_sample_csv():
    """Downloads sample benchmark dataset."""
    sample_path = Config.SAMPLE_DATA_DIR / "accident_records_sample.csv"
    if not sample_path.exists():
        seed_database(count=1200)
    return send_file(sample_path, as_attachment=True, download_name="accident_records_sample.csv")

@app.route("/download/html-report")
def download_html_report():
    """Downloads the standalone HTML executive trend report."""
    html_path = Config.REPORTS_DIR / "accident_trend_report.html"
    if not html_path.exists():
        AccidentReporter.generate_html_report()
    return send_file(html_path, as_attachment=True, download_name="accident_trend_report.html")

# ==========================================
# REST API FOR PROGRAMMATIC INTEGRATION
# ==========================================
@app.route("/api/summary")
def api_summary():
    """REST endpoint returning full analytical summary JSON."""
    df = AccidentAnalyzer.load_data()
    summary = AccidentAnalyzer.get_full_analysis_summary(df)
    return jsonify(summary)

@app.route("/api/quality-audit")
def api_quality_audit():
    """REST endpoint returning data completeness & PII audit JSON."""
    df = AccidentAnalyzer.load_data()
    audit = DataQualityAuditor.audit_dataset(df)
    return jsonify(audit)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
