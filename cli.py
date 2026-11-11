import argparse
import sys
from pathlib import Path
import pandas as pd
from config import Config
from database import init_db, check_db_health, get_session
from models import Location, Accident
from data_generator import seed_database
from analyzer import AccidentAnalyzer
from visualizer import AccidentVisualizer
from reporter import AccidentReporter
from validator import RecordValidator, DataQualityAuditor

def print_banner():
    banner = r"""
========================================================================
   PROJECT 13: ROAD ACCIDENT DATA ANALYSIS SYSTEM
   Empirical Pattern Profiling & Safety Intelligence
========================================================================
    """
    print(banner)
    print(f"NOTICE: {Config.DISCLAIMER_TEXT}\n")

def run_analysis_cli():
    print("\n--- RUNNING PANDAS & NUMPY DATA ANALYSIS ---")
    df = AccidentAnalyzer.load_data()
    if df.empty:
        print("[!] No records found in the database. Run --seed first.")
        return

    summary = AccidentAnalyzer.get_full_analysis_summary(df)
    t = summary["time_analysis"]
    l = summary["location_analysis"]
    s = summary["severity_analysis"]
    st = summary["numpy_descriptive_statistics"]
    assoc = summary["association_analysis"]

    print(f"\n[+] Total Validated Records Analyzed: {summary['total_records']:,}")
    print(f"[+] Date Range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}")

    print("\n[TEMPORAL PATTERNS]")
    print(f"  • Peak Hour of Day: {t['hourly']['peak_hour_formatted']} ({t['hourly']['peak_hour_count']} accidents)")
    print(f"  • Lowest Hour: {t['hourly']['lowest_hour_formatted']} ({t['hourly']['lowest_hour_count']} accidents)")
    print(f"  • Peak Day of Week: {t['day_of_week']['peak_day']} ({t['day_of_week']['peak_day_count']} accidents)")

    print("\n[SPATIAL HOTSPOTS (TOP 5 REGIONS)]")
    for idx, r in enumerate(l["by_region"][:5], 1):
        print(f"  {idx}. {r['region']:<32} | Total: {r['total_accidents']:<4} | Fatal: {r['fatal_count']:<2} | Risk Index: {r['risk_index']:.1f}")

    print("\n[MICRO-CORRIDOR BLACKSPOTS (TOP 5 ROADS)]")
    for idx, b in enumerate(l["top_blackspots"][:5], 1):
        print(f"  {idx}. {b['region']} -> {b['road_name_clean']} ({b['total_accidents']} incidents, {b['fatalities']} fatalities)")

    print("\n[SEVERITY CATEGORIES SUMMARY]")
    for cat, cnt, pct in zip(s["distribution"]["categories"], s["distribution"]["counts"], s["distribution"]["percentages"]):
        print(f"  • {cat:<10}: {cnt:<5} ({pct:.1f}%)")
    print(f"  • Total Casualties Recorded: {s['casualties']['total_casualties']}")

    print("\n[NUMPY DESCRIPTIVE STATISTICS]")
    hs = st["hourly_statistics"]
    ds = st["daily_statistics"]
    print(f"  • Hourly Accidents: Mean = {hs['mean']}, Median = {hs['median']}, StdDev = {hs['std_dev']}, IQR = {hs['iqr']}, Skewness = {hs['skewness']}")
    print(f"  • Daily Accidents:  Mean = {ds['mean_accidents_per_day']}, Median = {ds['median_accidents_per_day']}, StdDev = {ds['std_dev']}")
    print(f"  • 95% Confidence Interval (Daily Mean): [{ds['ci_95_percent']['lower']} - {ds['ci_95_percent']['upper']}]")

    print("\n[STATISTICAL ASSOCIATION INSIGHTS]")
    print(f"  • Chi-Square: {assoc['chi_square_stat']}, p-value: {assoc['p_value']:.4f}")
    print(f"  • Cramér's V: {assoc['cramers_v']} ({assoc['association_strength']} association)")
    print("  • Most Vulnerable Vehicle Types (% Fatal/Severe):")
    for v in assoc["vulnerability_ranking"][:3]:
        print(f"    - {v['vehicle_type']:<22}: {v['fatal_or_severe_pct']:.1f}%")

    print(f"\n[!] Reminder: {assoc['causation_caveat']}\n")

def run_quality_audit_cli():
    print("\n--- DATA QUALITY & ACCEPTANCE CRITERIA AUDIT ---")
    df = AccidentAnalyzer.load_data()
    if df.empty:
        print("[!] No records to audit.")
        return

    audit = DataQualityAuditor.audit_dataset(df)
    print(f"\n[+] Total Audited Records: {audit['total_records']:,}")
    print(f"[+] Overall Completeness Score: {audit['completeness_score_pct']}%")
    print(f"[+] Mandatory Completeness: {audit['mandatory_completeness_pct']}%")
    print(f"[+] PII Screening Certification: {audit['pii_compliance_status']}")

    print("\n[MISSING DATA REPORT]")
    for col, info in audit["missing_fields"].items():
        status = "MANDATORY" if info["is_mandatory"] else "OPTIONAL"
        print(f"  • {col:<26} [{status}]: {info['missing_count']} missing ({info['missing_percentage']}%)")

    print("\n[LOCATION GRANULARITY & SOURCE DOCUMENTATION]")
    loc = audit["location_granularity"]
    print(f"  • Hierarchy Level: {loc['level']}")
    print(f"  • Unique Regions: {loc['unique_regions_count']}")
    print(f"  • Unique Road Corridors: {loc['unique_road_corridors_count']}")
    print(f"  • Road Name Reporting Rate: {loc['road_name_reporting_rate_pct']}%")
    print(f"  • Notes: {loc['notes']}")
    print(f"  • Source Provenance: {audit['data_source_documentation']['provenance']}")
    print(f"  • Privacy Standard: {audit['data_source_documentation']['anonymization_standard']}\n")

def import_csv_cli(file_path_str: str):
    file_path = Path(file_path_str)
    if not file_path.exists():
        print(f"[!] File not found: {file_path}")
        return

    print(f"\n--- IMPORTING & VALIDATING ACCIDENT RECORDS FROM: {file_path.name} ---")
    try:
        df_raw = pd.read_csv(file_path)
    except Exception as e:
        print(f"[!] Failed to read CSV: {e}")
        return

    total_rows = len(df_raw)
    print(f"Read {total_rows} rows from file. Validating records...")

    session = get_session()
    inserted_count = 0
    rejected_count = 0
    rejections = []

    try:
        from models import Location, Accident
        # Pre-cache existing locations
        existing_locs = session.query(Location).all()
        loc_cache = {(l.region.strip().lower(), (l.road_name_optional or "").strip().lower()): l.location_id for l in existing_locs}

        for idx, row in df_raw.iterrows():
            row_dict = row.to_dict()
            is_valid, cleaned, errors = RecordValidator.validate_accident_record(row_dict)

            if not is_valid:
                rejected_count += 1
                if len(rejections) < 10:
                    rejections.append(f"Row {idx + 2}: {'; '.join(errors)}")
                continue

            # Lookup or insert location
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
        print(f"\n[+] Import Completed:")
        print(f"    • Successfully Ingested: {inserted_count} records")
        print(f"    • Rejected Due to Validation / PII: {rejected_count} records")

        if rejections:
            print("\nSample Validation Rejections:")
            for rej in rejections:
                print(f"    - {rej}")

    except Exception as e:
        session.rollback()
        print(f"[!] Import failed: {e}")
    finally:
        session.close()

def interactive_menu():
    print_banner()
    while True:
        print("\n--- MAIN MENU ---")
        print("1. View Statistical & Pattern Analysis (Pandas & NumPy)")
        print("2. Run Data Quality & Acceptance Criteria Audit (Missing Data, PII)")
        print("3. Generate All Matplotlib Charts (PNG outputs)")
        print("4. Generate Executive Analytical Reports (Markdown & HTML)")
        print("5. Import Accident Records from CSV")
        print("6. Reseed Database with Synthetic Benchmark Data")
        print("7. Check Database Status")
        print("8. Exit")

        choice = input("\nEnter choice [1-8]: ").strip()
        if choice == "1":
            run_analysis_cli()
        elif choice == "2":
            run_quality_audit_cli()
        elif choice == "3":
            print("\nGenerating Matplotlib charts...")
            charts = AccidentVisualizer.generate_all_charts()
            print(f"[+] Generated {len(charts)} publication-ready charts in '{Config.CHARTS_DIR}'.")
        elif choice == "4":
            print("\nGenerating Markdown and HTML reports...")
            md = AccidentReporter.generate_markdown_report()
            html = AccidentReporter.generate_html_report()
            print(f"[+] Markdown report: {md}")
            print(f"[+] HTML report: {html}")
        elif choice == "5":
            p = input("Enter path to CSV file: ").strip()
            import_csv_cli(p)
        elif choice == "6":
            confirm = input("Reset and reseed database? (y/N): ").strip().lower()
            if confirm == "y":
                cnt = input("Number of records to generate [default 1200]: ").strip()
                count = int(cnt) if cnt.isdigit() else 1200
                seed_database(count=count, force=True)
        elif choice == "7":
            status = check_db_health()
            print(f"\nDatabase Status: {status}")
        elif choice == "8":
            print("Exiting. Drive safely!")
            break
        else:
            print("Invalid choice. Please enter a number 1 to 8.")

def main():
    parser = argparse.ArgumentParser(description="Road Accident Data Analysis System")
    parser.add_argument("--init-db", action="store_true", help="Initialize database tables")
    parser.add_argument("--seed", type=int, nargs="?", const=1200, help="Seed database with synthetic records")
    parser.add_argument("--analyze", action="store_true", help="Run analytical routines and print summary")
    parser.add_argument("--charts", action="store_true", help="Generate all Matplotlib charts")
    parser.add_argument("--report", action="store_true", help="Generate Markdown and HTML reports")
    parser.add_argument("--audit-quality", action="store_true", help="Audit data completeness and missing fields")
    parser.add_argument("--import-csv", type=str, help="Import records from CSV file with validation")

    args = parser.parse_args()

    # If no flags provided, run interactive menu
    if len(sys.argv) == 1:
        interactive_menu()
        return

    if args.init_db:
        init_db()
    if args.seed is not None:
        seed_database(count=args.seed, force=True)
    if args.analyze:
        run_analysis_cli()
    if args.charts:
        AccidentVisualizer.generate_all_charts()
        print(f"[+] Generated charts in {Config.CHARTS_DIR}")
    if args.report:
        md = AccidentReporter.generate_markdown_report()
        html = AccidentReporter.generate_html_report()
        print(f"[+] Reports generated at:\n    {md}\n    {html}")
    if args.audit_quality:
        run_quality_audit_cli()
    if args.import_csv:
        import_csv_cli(args.import_csv)

if __name__ == "__main__":
    main()
