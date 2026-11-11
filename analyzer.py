import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from sqlalchemy import text
from config import Config
from database import get_engine, get_session

class AccidentAnalyzer:
    """
    Core data analysis engine for the Road Accident Data Analysis System.
    Leverages Pandas for time/location multi-dimensional grouping and aggregations,
    and NumPy for descriptive statistical computations.
    """

    SEVERITY_WEIGHTS = {
        "Minor": 1,
        "Moderate": 2,
        "Severe": 3,
        "Fatal": 4
    }

    @classmethod
    def load_data(cls, engine=None) -> pd.DataFrame:
        """Loads all accident records joined with location data into a pandas DataFrame."""
        if engine is None:
            engine = get_engine()

        query = """
            SELECT 
                a.accident_id,
                a.accident_datetime,
                a.location_id,
                l.region,
                l.road_name_optional,
                l.area_type,
                a.vehicle_type,
                a.severity,
                a.weather_optional,
                a.light_condition_optional,
                a.casualty_count_optional,
                a.notes_optional
            FROM accidents a
            INNER JOIN locations l ON a.location_id = l.location_id
            ORDER BY a.accident_datetime ASC
        """
        df = pd.read_sql(query, engine)
        if df.empty:
            return df

        return cls.enrich_dataframe(df)

    @classmethod
    def enrich_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Enriches the raw DataFrame with time-based features and numeric severity weights."""
        df = df.copy()

        # Parse datetime if string
        if not pd.api.types.is_datetime64_any_dtype(df["accident_datetime"]):
            df["accident_datetime"] = pd.to_datetime(df["accident_datetime"], errors="coerce")

        # Drop any records with unparseable datetime
        df = df.dropna(subset=["accident_datetime"]).copy()

        # Temporal feature extraction
        df["date"] = df["accident_datetime"].dt.date
        df["year"] = df["accident_datetime"].dt.year
        df["month"] = df["accident_datetime"].dt.month
        df["month_name"] = df["accident_datetime"].dt.strftime("%B")
        df["year_month"] = df["accident_datetime"].dt.strftime("%Y-%m")
        df["day_of_week"] = df["accident_datetime"].dt.dayofweek  # 0=Monday, 6=Sunday
        df["day_name"] = df["accident_datetime"].dt.strftime("%A")
        df["hour"] = df["accident_datetime"].dt.hour
        df["is_weekend"] = df["day_of_week"].isin([5, 6])
        df["weekend_label"] = np.where(df["is_weekend"], "Weekend", "Weekday")

        # Time of day binning
        def categorize_time_bin(hour):
            if 8 <= hour < 11:
                return "Morning Rush (08:00 - 11:00)"
            elif 11 <= hour < 16:
                return "Midday Off-Peak (11:00 - 16:00)"
            elif 16 <= hour < 20:
                return "Evening Rush (16:00 - 20:00)"
            elif 20 <= hour <= 23:
                return "Late Evening (20:00 - 23:59)"
            else:
                return "Late Night / Early Morning (00:00 - 08:00)"

        df["time_bin"] = df["hour"].apply(categorize_time_bin)

        # Severity numeric weighting for index computation
        df["severity_weight"] = df["severity"].map(cls.SEVERITY_WEIGHTS).fillna(1)
        df["casualty_count_optional"] = df["casualty_count_optional"].fillna(0).astype(int)

        # Clean optional strings
        df["road_name_clean"] = df["road_name_optional"].fillna("Unspecified Road")
        df["weather_clean"] = df["weather_optional"].fillna("Unrecorded Weather")
        df["light_clean"] = df["light_condition_optional"].fillna("Unrecorded Light")

        return df

    # ==========================================
    # TIME PERIOD COMPARISONS & AGGREGATIONS
    # ==========================================
    @classmethod
    def get_time_analysis(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes accidents across temporal dimensions using Pandas grouping."""
        if df.empty:
            return {}

        # 1. Hourly distribution (0-23)
        hourly_counts = df["hour"].value_counts().reindex(range(24), fill_value=0).sort_index()
        hourly_pct = (hourly_counts / len(df)) * 100

        # Peak hours identification
        peak_hour = int(hourly_counts.idxmax())
        peak_hour_count = int(hourly_counts.max())
        lowest_hour = int(hourly_counts.idxmin())
        lowest_hour_count = int(hourly_counts.min())

        # 2. Day of Week distribution
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = df["day_name"].value_counts().reindex(days_order, fill_value=0)
        dow_pct = (dow_counts / len(df)) * 100

        # 3. Time bin summary
        time_bin_summary = df.groupby("time_bin", observed=False).agg(
            total_accidents=("accident_id", "count"),
            avg_severity_weight=("severity_weight", "mean"),
            total_fatalities=("severity", lambda s: (s == "Fatal").sum()),
            total_severe=("severity", lambda s: (s == "Severe").sum())
        ).reset_index()
        time_bin_summary["pct_of_total"] = (time_bin_summary["total_accidents"] / len(df)) * 100

        # 4. Weekday vs Weekend comparison
        weekend_comp = df.groupby("weekend_label").agg(
            total_accidents=("accident_id", "count"),
            avg_severity_weight=("severity_weight", "mean"),
            fatal_count=("severity", lambda s: (s == "Fatal").sum()),
            fatal_pct=("severity", lambda s: ((s == "Fatal").sum() / len(s)) * 100 if len(s) > 0 else 0)
        ).reset_index()

        # 5. Monthly trend
        monthly_trend = df.groupby("year_month").agg(
            total_accidents=("accident_id", "count"),
            fatal_count=("severity", lambda s: (s == "Fatal").sum()),
            severe_count=("severity", lambda s: (s == "Severe").sum()),
            moderate_count=("severity", lambda s: (s == "Moderate").sum()),
            minor_count=("severity", lambda s: (s == "Minor").sum()),
            total_casualties=("casualty_count_optional", "sum")
        ).reset_index().sort_values("year_month")

        return {
            "hourly": {
                "hours": list(range(24)),
                "counts": [int(c) for c in hourly_counts.values],
                "percentages": [round(float(p), 2) for p in hourly_pct.values],
                "peak_hour": peak_hour,
                "peak_hour_formatted": f"{peak_hour:02d}:00 - {(peak_hour+1)%24:02d}:00",
                "peak_hour_count": peak_hour_count,
                "lowest_hour": lowest_hour,
                "lowest_hour_formatted": f"{lowest_hour:02d}:00 - {(lowest_hour+1)%24:02d}:00",
                "lowest_hour_count": lowest_hour_count
            },
            "day_of_week": {
                "days": days_order,
                "counts": [int(c) for c in dow_counts.values],
                "percentages": [round(float(p), 2) for p in dow_pct.values],
                "peak_day": str(dow_counts.idxmax()),
                "peak_day_count": int(dow_counts.max())
            },
            "time_bins": time_bin_summary.to_dict(orient="records"),
            "weekday_vs_weekend": weekend_comp.to_dict(orient="records"),
            "monthly_trend": monthly_trend.to_dict(orient="records")
        }

    # ==========================================
    # LOCATION PATTERNS & HOTSPOT IDENTIFICATION
    # ==========================================
    @classmethod
    def get_location_analysis(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes spatial accident distribution and identifies high-risk hotspots."""
        if df.empty:
            return {}

        total_accidents = len(df)

        # 1. Regional distribution
        region_grp = df.groupby("region").agg(
            total_accidents=("accident_id", "count"),
            avg_severity_score=("severity_weight", "mean"),
            fatal_count=("severity", lambda s: (s == "Fatal").sum()),
            severe_count=("severity", lambda s: (s == "Severe").sum()),
            moderate_count=("severity", lambda s: (s == "Moderate").sum()),
            minor_count=("severity", lambda s: (s == "Minor").sum()),
            casualties=("casualty_count_optional", "sum")
        ).reset_index()

        region_grp["percentage"] = (region_grp["total_accidents"] / total_accidents) * 100
        # Calculate Risk Index = total_accidents * avg_severity_score
        region_grp["risk_index"] = region_grp["total_accidents"] * region_grp["avg_severity_score"]
        region_sorted = region_grp.sort_values("total_accidents", ascending=False)

        # 2. Road Corridor / Blackspot analysis (Micro-granularity)
        corridor_grp = df.groupby(["region", "road_name_clean"]).agg(
            total_accidents=("accident_id", "count"),
            avg_severity_score=("severity_weight", "mean"),
            fatalities=("severity", lambda s: (s == "Fatal").sum()),
            casualties=("casualty_count_optional", "sum")
        ).reset_index()
        corridor_grp["risk_index"] = corridor_grp["total_accidents"] * corridor_grp["avg_severity_score"]
        top_blackspots = corridor_grp.sort_values("risk_index", ascending=False).head(10)

        # 3. Area Type breakdown (Urban, Highway, Suburban, Rural)
        area_grp = df.groupby("area_type").agg(
            total_accidents=("accident_id", "count"),
            avg_severity_score=("severity_weight", "mean"),
            fatal_pct=("severity", lambda s: ((s == "Fatal").sum() / len(s)) * 100 if len(s) > 0 else 0)
        ).reset_index()
        area_grp["percentage"] = (area_grp["total_accidents"] / total_accidents) * 100

        return {
            "by_region": region_sorted.to_dict(orient="records"),
            "top_blackspots": top_blackspots.to_dict(orient="records"),
            "by_area_type": area_grp.to_dict(orient="records"),
            "top_region": region_sorted.iloc[0]["region"] if not region_sorted.empty else "N/A",
            "top_region_count": int(region_sorted.iloc[0]["total_accidents"]) if not region_sorted.empty else 0
        }

    # ==========================================
    # SEVERITY CATEGORY SUMMARIES
    # ==========================================
    @classmethod
    def get_severity_analysis(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Summarizes severity categories, percentages, and cross-tabulations."""
        if df.empty:
            return {}

        total_accidents = len(df)
        order = ["Minor", "Moderate", "Severe", "Fatal"]

        # Value counts
        sev_counts = df["severity"].value_counts().reindex(order, fill_value=0)
        sev_pcts = (sev_counts / total_accidents) * 100

        # Severity across vehicle types (Crosstab)
        veh_sev_ct = pd.crosstab(df["vehicle_type"], df["severity"], margins=True, margins_name="Total")
        for col in order:
            if col not in veh_sev_ct.columns:
                veh_sev_ct[col] = 0
        veh_sev_ct = veh_sev_ct.reindex(columns=order + ["Total"])

        # Row-wise percentages
        veh_sev_pct = pd.crosstab(df["vehicle_type"], df["severity"], normalize="index") * 100
        for col in order:
            if col not in veh_sev_pct.columns:
                veh_sev_pct[col] = 0.0
        veh_sev_pct = veh_sev_pct.reindex(columns=order)

        # Casualties summary
        casualty_sum = int(df["casualty_count_optional"].sum())
        avg_casualties_per_sev = df.groupby("severity")["casualty_count_optional"].mean().reindex(order, fill_value=0)

        # Environmental factors cross-tabulation
        weather_sev = pd.crosstab(df["weather_clean"], df["severity"], normalize="index") * 100

        return {
            "distribution": {
                "categories": order,
                "counts": [int(sev_counts[c]) for c in order],
                "percentages": [round(float(sev_pcts[c]), 2) for c in order]
            },
            "vehicle_type_breakdown": {
                "counts": veh_sev_ct.reset_index().to_dict(orient="records"),
                "percentages": veh_sev_pct.round(2).reset_index().to_dict(orient="records")
            },
            "casualties": {
                "total_casualties": casualty_sum,
                "avg_per_severity": {c: round(float(avg_casualties_per_sev[c]), 2) for c in order}
            },
            "weather_breakdown": weather_sev.round(2).reset_index().to_dict(orient="records")
        }

    # ==========================================
    # NUMPY DESCRIPTIVE STATISTICS
    # ==========================================
    @classmethod
    def get_numpy_descriptive_stats(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Uses NumPy for descriptive statistical metrics across
        hourly frequencies, daily occurrences, and regional concentrations.
        """
        if df.empty:
            return {}

        # 1. Hourly Accident Distribution Stats (NumPy)
        hourly_counts = np.array(df["hour"].value_counts().reindex(range(24), fill_value=0).values, dtype=float)
        h_mean = float(np.mean(hourly_counts))
        h_median = float(np.median(hourly_counts))
        h_std = float(np.std(hourly_counts, ddof=1)) if len(hourly_counts) > 1 else 0.0
        h_var = float(np.var(hourly_counts, ddof=1)) if len(hourly_counts) > 1 else 0.0
        h_q25 = float(np.percentile(hourly_counts, 25))
        h_q75 = float(np.percentile(hourly_counts, 75))
        h_iqr = h_q75 - h_q25
        h_max = float(np.max(hourly_counts))
        h_min = float(np.min(hourly_counts))

        # Skewness calculation using NumPy
        if h_std > 0:
            h_skew = float(np.mean(((hourly_counts - h_mean) / h_std) ** 3))
        else:
            h_skew = 0.0

        # 2. Daily Accident Counts Stats (NumPy)
        daily_counts_series = df.groupby("date")["accident_id"].count()
        daily_counts = np.array(daily_counts_series.values, dtype=float)
        d_mean = float(np.mean(daily_counts))
        d_median = float(np.median(daily_counts))
        d_std = float(np.std(daily_counts, ddof=1)) if len(daily_counts) > 1 else 0.0
        d_min = float(np.min(daily_counts))
        d_max = float(np.max(daily_counts))
        # 95% Confidence interval for daily mean: mean ± 1.96 * (std / sqrt(n))
        n_days = len(daily_counts)
        margin_of_error = float(1.96 * (d_std / np.sqrt(n_days))) if n_days > 1 else 0.0
        ci_lower = max(0.0, d_mean - margin_of_error)
        ci_upper = d_mean + margin_of_error

        # 3. Location Volume Stats & Outlier Detection (NumPy)
        region_counts = np.array(df["region"].value_counts().values, dtype=float)
        r_mean = float(np.mean(region_counts))
        r_median = float(np.median(region_counts))
        r_std = float(np.std(region_counts, ddof=1)) if len(region_counts) > 1 else 0.0
        r_q75 = float(np.percentile(region_counts, 75))
        r_q25 = float(np.percentile(region_counts, 25))
        r_iqr = r_q75 - r_q25
        outlier_threshold = r_q75 + (1.5 * r_iqr)

        # Identify statistical outlier regions
        reg_series = df["region"].value_counts()
        outlier_regions = reg_series[reg_series > outlier_threshold].to_dict()

        # 4. Severity Weight Distribution Stats (NumPy)
        sev_weights = np.array(df["severity_weight"].values, dtype=float)
        s_mean = float(np.mean(sev_weights))
        s_median = float(np.median(sev_weights))
        s_std = float(np.std(sev_weights, ddof=1))

        return {
            "hourly_statistics": {
                "mean": round(h_mean, 2),
                "median": round(h_median, 2),
                "std_dev": round(h_std, 2),
                "variance": round(h_var, 2),
                "q25": round(h_q25, 2),
                "q75": round(h_q75, 2),
                "iqr": round(h_iqr, 2),
                "min": int(h_min),
                "max": int(h_max),
                "skewness": round(h_skew, 3)
            },
            "daily_statistics": {
                "total_recorded_days": n_days,
                "mean_accidents_per_day": round(d_mean, 2),
                "median_accidents_per_day": round(d_median, 2),
                "std_dev": round(d_std, 2),
                "min_per_day": int(d_min),
                "max_per_day": int(d_max),
                "ci_95_percent": {
                    "lower": round(ci_lower, 2),
                    "upper": round(ci_upper, 2),
                    "margin_of_error": round(margin_of_error, 2)
                }
            },
            "location_statistics": {
                "mean_accidents_per_region": round(r_mean, 2),
                "median_accidents_per_region": round(r_median, 2),
                "std_dev": round(r_std, 2),
                "outlier_threshold_iqr": round(outlier_threshold, 2),
                "outlier_regions": outlier_regions
            },
            "severity_index_statistics": {
                "mean_severity_score": round(s_mean, 2),
                "median_severity_score": round(s_median, 2),
                "std_dev": round(s_std, 2)
            }
        }

    # ==========================================
    # STATISTICAL ASSOCIATION VS CAUSATION
    # ==========================================
    @classmethod
    def get_association_insights(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes contingency tables and Cramér's V measure of association.
        Clearly documents that associations must NOT be construed as proof of causation.
        """
        if df.empty:
            return {}

        # Vehicle Type vs Severity contingency
        contingency = pd.crosstab(df["vehicle_type"], df["severity"])
        
        # Calculate Chi-Square statistic and Cramér's V
        try:
            from scipy.stats import chi2_contingency
            chi2, p_val, dof, _ = chi2_contingency(contingency)
            n = len(df)
            min_dim = min(contingency.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
        except Exception:
            # Fallback numpy approximation if scipy not present
            chi2, p_val, cramers_v = 0.0, 1.0, 0.0

        vulnerability_ranking = df.groupby("vehicle_type").agg(
            total=("accident_id", "count"),
            fatal_or_severe_pct=("severity", lambda s: (s.isin(["Fatal", "Severe"]).sum() / len(s)) * 100)
        ).sort_values("fatal_or_severe_pct", ascending=False).reset_index()

        return {
            "chi_square_stat": round(float(chi2), 3),
            "p_value": float(p_val),
            "cramers_v": round(float(cramers_v), 3),
            "association_strength": "Moderate" if cramers_v > 0.15 else "Weak" if cramers_v > 0.05 else "Negligible",
            "vulnerability_ranking": vulnerability_ranking.to_dict(orient="records"),
            "causation_caveat": Config.DISCLAIMER_TEXT
        }

    # ==========================================
    # COMPREHENSIVE MASTER SUMMARY
    # ==========================================
    @classmethod
    def get_full_analysis_summary(cls, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Executes all analytical routines and compiles master analysis dictionary."""
        if df is None:
            df = cls.load_data()

        if df.empty:
            return {"status": "empty", "message": "No accident records available for analysis."}

        time_res = cls.get_time_analysis(df)
        loc_res = cls.get_location_analysis(df)
        sev_res = cls.get_severity_analysis(df)
        numpy_stats = cls.get_numpy_descriptive_stats(df)
        assoc_res = cls.get_association_insights(df)

        return {
            "status": "success",
            "total_records": len(df),
            "date_range": {
                "earliest": df["accident_datetime"].min().strftime("%Y-%m-%d"),
                "latest": df["accident_datetime"].max().strftime("%Y-%m-%d")
            },
            "time_analysis": time_res,
            "location_analysis": loc_res,
            "severity_analysis": sev_res,
            "numpy_descriptive_statistics": numpy_stats,
            "association_analysis": assoc_res,
            "disclaimer": Config.DISCLAIMER_TEXT
        }
