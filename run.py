import sys
import webbrowser
import logging
from config import Config
from database import init_db, check_db_health, get_session
from models import Accident
from data_generator import seed_database
from visualizer import AccidentVisualizer
from reporter import AccidentReporter
from app import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def setup_and_run():
    print("""
    ===================================================================
      PROJECT 13: ROAD ACCIDENT DATA ANALYSIS SYSTEM
      Empirical Spatio-Temporal & Severity Pattern Analytics
    ===================================================================
    """)
    print(f"NOTICE: {Config.DISCLAIMER_TEXT}\n")

    # 1. Initialize DB tables
    logger.info("Verifying database schema...")
    init_db()

    # 2. Check existing records
    session = get_session()
    try:
        count = session.query(Accident).count()
        if count == 0:
            logger.info("Database is empty. Populating with realistic benchmark accident data (1,200 records)...")
            seed_database(count=1200)
        else:
            logger.info(f"Database contains {count:,} accident records.")
    finally:
        session.close()

    # 3. Generate initial charts & reports if missing
    dashboard_chart = Config.CHARTS_DIR / "accident_analysis_dashboard.png"
    if not dashboard_chart.exists():
        logger.info("Generating publication-quality Matplotlib figures...")
        AccidentVisualizer.generate_all_charts()
        AccidentReporter.generate_markdown_report()
        AccidentReporter.generate_html_report()

    # 4. Start web application
    host = "127.0.0.1"
    port = 5000
    url = f"http://{host}:{port}"
    print(f"\n[+] Web Application Server starting at: {url}")
    print("    • Dashboard:            http://127.0.0.1:5000/")
    print("    • Accident Records:     http://127.0.0.1:5000/records")
    print("    • Ingest New Record:    http://127.0.0.1:5000/records/add")
    print("    • Pattern Analytics:    http://127.0.0.1:5000/analytics")
    print("    • Data Quality Audit:   http://127.0.0.1:5000/data-quality")
    print("    • Reports & Charts:     http://127.0.0.1:5000/reports")
    print("\nPress Ctrl+C to terminate the web server.\n")

    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    setup_and_run()
