# Project 13: Road Accident Data Analysis System

> **Difficulty Level:** Intermediate  
> **Primary Technology Stack:** Python 3.13, MySQL 8.0 / SQLAlchemy, Pandas, NumPy, Matplotlib, Flask, Bootstrap 5  
> **Key Focus:** Empirical spatio-temporal profiling, vehicle vulnerability estimation, and statistical association modeling.

---

## 1. Objective, Scope & Boundaries

### Objective
Analyze structured traffic accident records to discover empirical patterns across:
1. **Time Dimensions:** 24-hour diurnal cycle, morning and evening commute rush periods, and weekday vs. weekend dynamics.
2. **Geographic Locations:** Macro administrative regions and micro road corridor blackspots.
3. **Vehicle Classifications:** Vulnerability profiles across two-wheelers, cars, commercial trucks, buses, vans, and pedestrians.
4. **Recorded Severity Levels:** Distribution across Minor, Moderate, Severe, and Fatal outcomes.

### Scope & Boundaries
- **Synthetic Public Benchmark Data:** Utilizes public, synthetic, and strictly anonymized data modeled on standard highway safety telemetry protocols.
- **Association vs. Causation Warning:** **Findings describe statistical associations and patterns in recorded data and should not be treated as proof of causation.** Traffic incidents arise from complex multi-variate interactions (e.g. driver reaction times, velocity, road pavement friction, lighting) that cannot be reduced to single-variable causal claims.
- **Privacy & Non-PII Standard:** Zero personally identifiable information (PII) is captured. No individual names, phone numbers, email addresses, national IDs, or vehicle license plates are stored.

---

## 2. MySQL Database Schema & Architecture

The database schema strictly conforms to the requested entities with normalized relationships and indexing for high-performance aggregations:

```sql
-- 1. locations table
CREATE TABLE locations (
    location_id INT AUTO_INCREMENT PRIMARY KEY,
    region VARCHAR(100) NOT NULL,
    road_name_optional VARCHAR(150) NULL,
    area_type VARCHAR(50) DEFAULT 'Urban',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_region (region),
    INDEX idx_road (road_name_optional)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. accidents table
CREATE TABLE accidents (
    accident_id INT AUTO_INCREMENT PRIMARY KEY,
    accident_datetime DATETIME NOT NULL,
    location_id INT NOT NULL,
    vehicle_type VARCHAR(50) NOT NULL,
    severity VARCHAR(30) NOT NULL,
    weather_optional VARCHAR(50) NULL,
    light_condition_optional VARCHAR(50) NULL,
    casualty_count_optional INT DEFAULT 0,
    notes_optional TEXT NULL,
    FOREIGN KEY (location_id) REFERENCES locations(location_id) ON DELETE CASCADE,
    INDEX idx_datetime (accident_datetime),
    INDEX idx_severity (severity),
    INDEX idx_vehicle (vehicle_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

*Note: The application connects to MySQL 8.0 by default (`root:admin@127.0.0.1:3306/road_accident_db`), and provides automatic seamless fallback to local SQLite (`accident_data.db`) if MySQL is ever inaccessible.*

---

## 3. Features Implemented

| Suggested Feature | Implementation Details |
| :--- | :--- |
| **Import & Manage Records** | Web UI interactive table with pagination, search, filters, CRUD operations, and CSV batch upload tool with instant validation feedback. |
| **Validation Engine** | Validates date/time (format, bounds, not in future), standardized region, vehicle type normalization, 4-tier severity enforcement, and automated regex PII scanner. |
| **Time & Location Comparison** | Pandas grouping across 24 hourly bins, commute windows (morning/evening rush), day-of-week comparisons, regional rankings, and corridor blackspots. |
| **Summarize Severity Categories** | Quantitative distributions of Minor, Moderate, Severe, and Fatal accidents with casualty aggregates and cross-tabulations. |
| **Identify Peak Periods & Hotspots** | Mathematical peak hour detection, statistical outlier thresholding (> 1.5 IQR above Q75), and severity-weighted risk indexing. |

---

## 4. Data Analysis & Visualization Architecture

### A. Pandas Multi-Dimensional Grouping (`analyzer.py`)
- **Temporal Grouping:** `df.groupby('hour')`, `df.groupby('time_bin')`, `df.groupby('day_name')`, `df.groupby('year_month')`.
- **Spatial Aggregation:** Multi-index grouping `df.groupby(['region', 'road_name_clean'])` to compute total volume, casualty sum, and risk indices.
- **Cross-Tabulation:** `pd.crosstab(df['vehicle_type'], df['severity'], normalize='index')` to derive vehicle vulnerability profiles.

### B. NumPy Descriptive Statistics (`analyzer.py`)
- **Hourly Statistics:** Mean ($\mu$), Median, Standard Deviation ($\sigma$), Variance ($\sigma^2$), Interquartile Range ($IQR = Q_{75} - Q_{25}$), and Distribution Skewness.
- **Daily Volume Statistics:** Mean accidents per day, sample standard deviation, and 95% Confidence Interval:
  $$\text{CI}_{95\%} = \bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{n}}$$
- **Spatial Outlier Detection:** Identification of severe regional clusters exceeding the $Q_{75} + 1.5 \cdot IQR$ statistical barrier.

### C. Matplotlib Publication Figures (`visualizer.py`)
1. `accidents_by_hour.png` — 24-hour accident cycle highlighting morning (08:00–10:00) and evening (17:00–20:00) rush-hour zones.
2. `accidents_by_day_of_week.png` — Weekday vs. weekend distribution with percentage markers.
3. `severity_distribution.png` — Donut chart and bar frequency of Minor, Moderate, Severe, and Fatal outcomes.
4. `location_hotspots.png` — Horizontal bar chart ranking top incident-prone regions.
5. `vehicle_type_vs_severity.png` — Stacked horizontal vulnerability proportions comparing vehicle classes.
6. `monthly_trend_analysis.png` — Time-series line chart tracking monthly incident frequency.
7. `weather_severity_matrix.png` — Stacked distribution assessing wet, foggy, and clear conditions.
8. `accident_analysis_dashboard.png` — High-resolution composite master dashboard figure combining all core dimensions.

---

## 5. Acceptance Criteria Verification

### 1. Missing Data is Reported
- The system includes a dedicated **Data Quality & Governance Audit** page (`/data-quality`) and table reporting exact null counts, completeness percentages, and imputation strategies for every attribute (`road_name_optional`, `weather_optional`, `light_condition_optional`, `notes_optional`).
- Mandatory fields maintain **100.0%** completeness.

### 2. Location Granularity and Source are Documented
- **Macro Granularity:** 12 standardized administrative sectors/regions.
- **Micro Granularity:** Over 50 individual road and arterial corridors.
- **Provenance Documentation:** Derived from synthetic benchmark traffic safety telemetry modeling real-world diurnal rhythms.

### 3. No Personally Identifying Details (PII) are Included
- Automated regex audit verifies the absence of telephone numbers, email addresses, credit cards, national ID numbers (SSN/Aadhaar), and motor vehicle license plates.
- Status: **`CERTIFIED_NO_PII`**.

---

## 6. How to Run the System

### Option A: Web Application (Interactive Dashboard & CRUD)
To launch the complete web portal:
```powershell
py -3.13 run.py
```
Open your browser to: **`http://127.0.0.1:5000`**

Available Web Portals:
- **Executive Dashboard:** `http://127.0.0.1:5000/`
- **Accident Records Registry:** `http://127.0.0.1:5000/records`
- **Ingest New Record:** `http://127.0.0.1:5000/records/add`
- **Pattern Analytics & NumPy:** `http://127.0.0.1:5000/analytics`
- **Data Quality & PII Audit:** `http://127.0.0.1:5000/data-quality`
- **Reports & Matplotlib Visuals:** `http://127.0.0.1:5000/reports`
- **Import Dataset (CSV):** `http://127.0.0.1:5000/import-csv`

### Option B: Command-Line Interface (CLI)
You can also run analysis or batch jobs directly in the console:

```powershell
# Interactive console menu
py -3.13 cli.py

# Print analytical summaries (Pandas & NumPy)
py -3.13 cli.py --analyze

# Audit missing data & PII compliance
py -3.13 cli.py --audit-quality

# Generate all Matplotlib PNG charts
py -3.13 cli.py --charts

# Compile Markdown & HTML executive reports
py -3.13 cli.py --report

# Import records from a CSV file
py -3.13 cli.py --import-csv sample_data/accident_records_sample.csv
```

### Option C: Automated Test Suite (Pytest)
Run the 9-point unit and integration test suite:
```powershell
py -3.13 -m pytest test_system.py -v
```
All 9 tests verify validation logic, future timestamp rejections, PII blocker regexes, Pandas/NumPy calculations, chart file generation, and Flask web routes.

---

## 7. Project File Structure

```
d:\Mahesh project\
├── config.py                  # Database credentials, paths, allowed categories, caveats
├── database.py                # MySQL connection, table initialization, SQLite fallback
├── models.py                  # SQLAlchemy ORM models (Location, Accident)
├── validator.py               # Record validation, PII scanner, DataQualityAuditor
├── data_generator.py          # Benchmark dataset generator with empirical rush-hour modeling
├── analyzer.py                # Pandas groupings, NumPy descriptive estimators, Chi-Square
├── visualizer.py              # Matplotlib publication chart suite & master dashboard
├── reporter.py                # Markdown and standalone HTML executive report generator
├── app.py                     # Flask web application and REST API endpoints
├── cli.py                     # Command-line interface with interactive and argument modes
├── run.py                     # Automated entrypoint runner
├── test_system.py             # Pytest automated test suite (9 test cases)
├── requirements.txt           # Python dependencies
├── sample_data/
│   └── accident_records_sample.csv
├── reports/
│   ├── accident_trend_report.md
│   └── accident_trend_report.html
├── static/
│   └── charts/
│       ├── accidents_by_hour.png
│       ├── accidents_by_day_of_week.png
│       ├── severity_distribution.png
│       ├── location_hotspots.png
│       ├── vehicle_type_vs_severity.png
│       ├── monthly_trend_analysis.png
│       ├── weather_severity_matrix.png
│       └── accident_analysis_dashboard.png
└── templates/
    ├── base.html
    ├── dashboard.html
    ├── records.html
    ├── add_record.html
    ├── analytics.html
    ├── data_quality.html
    ├── import_csv.html
    └── reports_view.html
```
