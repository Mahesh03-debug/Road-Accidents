import os
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for server execution
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from config import Config
from analyzer import AccidentAnalyzer

# Professional color palettes
PALETTE_SEVERITY = {
    "Minor": "#2ecc71",     # Emerald green
    "Moderate": "#f39c12",  # Amber orange
    "Severe": "#e67e22",    # Dark orange
    "Fatal": "#c0392b"      # Deep crimson
}

COLOR_PRIMARY = "#2980b9"
COLOR_ACCENT = "#8e44ad"
COLOR_BG = "#fcfcfc"
COLOR_GRID = "#e0e0e0"

def apply_base_style(fig, ax):
    """Applies clean, modern typographic and grid styling."""
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

def add_disclaimer(fig):
    """Adds the mandatory analytical disclaimer footnote."""
    fig.text(
        0.5, 0.01,
        "Notice: Findings describe statistical associations and patterns in recorded data, not proof of causation.",
        ha="center", fontsize=8, color="#7f8c8d", style="italic"
    )

class AccidentVisualizer:
    """Generates publication-quality charts using Matplotlib."""

    @classmethod
    def ensure_output_dir(cls) -> Path:
        out_dir = Config.CHARTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # 1. Hourly Trend Chart
    @classmethod
    def plot_hourly_trend(cls, df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
        if save_path is None:
            save_path = cls.ensure_output_dir() / "accidents_by_hour.png"

        hourly = df["hour"].value_counts().reindex(range(24), fill_value=0).sort_index()
        peak_hour = int(hourly.idxmax())

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        apply_base_style(fig, ax)

        # Highlight morning & evening rush hours
        ax.axvspan(7.5, 10.5, color="#f39c12", alpha=0.15, label="Morning Rush (08:00 - 10:00)", zorder=1)
        ax.axvspan(16.5, 19.5, color="#e74c3c", alpha=0.15, label="Evening Rush (17:00 - 19:00)", zorder=1)

        # Bar chart + line
        ax.bar(range(24), hourly.values, color="#3498db", alpha=0.6, width=0.6, zorder=2, label="Accident Count")
        ax.plot(range(24), hourly.values, color="#1b4f72", linewidth=2.5, marker="o", markersize=6, zorder=3)

        # Annotate peak hour
        ax.annotate(
            f"Peak Hour: {peak_hour:02d}:00\n({hourly.max()} incidents)",
            xy=(peak_hour, hourly.max()),
            xytext=(peak_hour - 3.5, hourly.max() + 5),
            arrowprops=dict(facecolor="#c0392b", shrink=0.08, width=1.5, headwidth=6),
            fontsize=9, fontweight="bold", color="#922b21",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fdedec", ec="#c0392b", lw=1)
        )

        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=8)
        ax.set_title("Road Accident Frequency by Hour of Day (24-Hour Cycle)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Hour of Day", fontsize=10, labelpad=8)
        ax.set_ylabel("Total Recorded Accidents", fontsize=10, labelpad=8)
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8)

        add_disclaimer(fig)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # 2. Day of Week Chart
    @classmethod
    def plot_day_of_week(cls, df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
        if save_path is None:
            save_path = cls.ensure_output_dir() / "accidents_by_day_of_week.png"

        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = df["day_name"].value_counts().reindex(days_order, fill_value=0)
        colors = ["#2980b9" if d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"] else "#e67e22" for d in days_order]

        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        apply_base_style(fig, ax)

        bars = ax.bar(days_order, dow_counts.values, color=colors, width=0.55, zorder=2)

        # Value and percentage labels
        total = len(df)
        for bar in bars:
            height = bar.get_height()
            pct = (height / total) * 100 if total > 0 else 0
            ax.annotate(
                f"{height}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

        ax.set_ylim(0, max(dow_counts.values) * 1.18)
        ax.set_title("Accident Volume by Day of Week (Weekday vs Weekend)", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylabel("Total Recorded Accidents", fontsize=10)

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2980b9", label="Weekday"),
            Patch(facecolor="#e67e22", label="Weekend (Elevated Leisure/Night Traffic)")
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", fontsize=8)

        add_disclaimer(fig)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # 3. Severity Distribution Donut & Bar Chart
    @classmethod
    def plot_severity_distribution(cls, df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
        if save_path is None:
            save_path = cls.ensure_output_dir() / "severity_distribution.png"

        order = ["Minor", "Moderate", "Severe", "Fatal"]
        sev_counts = df["severity"].value_counts().reindex(order, fill_value=0)
        colors = [PALETTE_SEVERITY[s] for s in order]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        fig.patch.set_facecolor(COLOR_BG)

        # 1. Donut chart
        explode = (0.02, 0.02, 0.04, 0.08)
        wedges, texts, autotexts = ax1.pie(
            sev_counts.values,
            labels=order,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            explode=explode,
            pctdistance=0.75,
            textprops=dict(color="#333333", fontweight="bold", fontsize=9)
        )
        # Donut hole
        centre_circle = plt.Circle((0, 0), 0.50, fc="#ffffff")
        ax1.add_artist(centre_circle)
        ax1.set_title("Severity Proportion (%)", fontsize=12, fontweight="bold")

        # 2. Count bar chart
        apply_base_style(fig, ax2)
        bars = ax2.bar(order, sev_counts.values, color=colors, width=0.5, zorder=2)
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(
                f"{height} cases",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

        ax2.set_ylim(0, max(sev_counts.values) * 1.18)
        ax2.set_title("Accident Count by Severity Level", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Total Cases", fontsize=10)

        plt.suptitle("Accident Severity Categories Breakdown", fontsize=14, fontweight="bold", y=0.98)
        add_disclaimer(fig)
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # 4. Location Hotspots Chart
    @classmethod
    def plot_location_hotspots(cls, df: pd.DataFrame, top_n=10, save_path: Optional[Path] = None) -> Path:
        if save_path is None:
            save_path = cls.ensure_output_dir() / "location_hotspots.png"

        reg_data = df.groupby("region").agg(
            total=("accident_id", "count"),
            fatal_count=("severity", lambda s: (s == "Fatal").sum()),
            severe_count=("severity", lambda s: (s == "Severe").sum())
        ).sort_values("total", ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        apply_base_style(fig, ax)

        y_positions = range(len(reg_data))
        bars = ax.barh(y_positions, reg_data["total"], color="#16a085", height=0.6, zorder=2, label="Total Incidents")

        for idx, (bar, (_, row)) in enumerate(zip(bars, reg_data.iterrows())):
            width = bar.get_width()
            ax.annotate(
                f" {width} (Fatal/Severe: {row['fatal_count'] + row['severe_count']})",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8, fontweight="bold", color="#2c3e50"
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(reg_data.index, fontsize=9)
        ax.set_xlim(0, max(reg_data["total"]) * 1.25)
        ax.set_title(f"Top {top_n} Accident-Prone Regions (Hotspot Identification)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Recorded Accident Count", fontsize=10, labelpad=8)

        add_disclaimer(fig)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # 5. Vehicle Type vs Severity Chart
    @classmethod
    def plot_vehicle_severity_breakdown(cls, df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
        if save_path is None:
            save_path = cls.ensure_output_dir() / "vehicle_type_vs_severity.png"

        order = ["Minor", "Moderate", "Severe", "Fatal"]
        crosstab_pct = pd.crosstab(df["vehicle_type"], df["severity"], normalize="index")[order] * 100
        # Sort by fatal + severe proportion descending
        crosstab_pct["high_risk"] = crosstab_pct["Fatal"] + crosstab_pct["Severe"]
        crosstab_sorted = crosstab_pct.sort_values("high_risk", ascending=True).drop(columns=["high_risk"])

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        apply_base_style(fig, ax)

        y_positions = range(len(crosstab_sorted))
        lefts = np.zeros(len(crosstab_sorted))

        for sev in order:
            vals = crosstab_sorted[sev].values
            ax.barh(y_positions, vals, left=lefts, color=PALETTE_SEVERITY[sev], height=0.6, label=sev, zorder=2)
            lefts += vals

        ax.set_yticks(y_positions)
        ax.set_yticklabels(crosstab_sorted.index, fontsize=9, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter())
        ax.set_title("Severity Proportion (%) by Vehicle Type (Vulnerability Comparison)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Proportion of Accidents within Vehicle Category", fontsize=10, labelpad=8)
        ax.legend(title="Severity Category", loc="lower right", frameon=True, facecolor="white", fontsize=8)

        add_disclaimer(fig)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # 6. Monthly Trend Analysis
    @classmethod
    def plot_monthly_trend(cls, df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
        if save_path is None:
            save_path = cls.ensure_output_dir() / "monthly_trend_analysis.png"

        monthly = df.groupby("year_month").agg(
            total=("accident_id", "count"),
            fatal_severe=("severity", lambda s: s.isin(["Fatal", "Severe"]).sum())
        ).reset_index().sort_values("year_month")

        fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
        apply_base_style(fig, ax)

        x_indices = range(len(monthly))
        ax.plot(x_indices, monthly["total"], color="#2980b9", marker="o", linewidth=2.5, label="Total Accidents", zorder=3)
        ax.bar(x_indices, monthly["fatal_severe"], color="#c0392b", alpha=0.5, width=0.4, label="Fatal & Severe Subset", zorder=2)

        for i, row in monthly.iterrows():
            ax.annotate(str(row["total"]), xy=(i, row["total"]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

        ax.set_xticks(x_indices)
        ax.set_xticklabels(monthly["year_month"], rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, max(monthly["total"]) * 1.2)
        ax.set_title("Monthly Recorded Accident Trend Over Time", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylabel("Monthly Incidents", fontsize=10)
        ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8)

        add_disclaimer(fig)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # 7. Weather vs Severity Matrix
    @classmethod
    def plot_weather_breakdown(cls, df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
        if save_path is None:
            save_path = cls.ensure_output_dir() / "weather_severity_matrix.png"

        order = ["Minor", "Moderate", "Severe", "Fatal"]
        ct = pd.crosstab(df["weather_clean"], df["severity"])[order]
        ct["total"] = ct.sum(axis=1)
        ct_sorted = ct.sort_values("total", ascending=False).drop(columns=["total"])

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        apply_base_style(fig, ax)

        ct_sorted.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[PALETTE_SEVERITY[s] for s in order],
            width=0.6,
            zorder=2
        )

        ax.set_title("Accident Volume and Severity Distribution by Weather Condition", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Weather Condition", fontsize=10, labelpad=8)
        ax.set_ylabel("Accident Count", fontsize=10)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="Severity", loc="upper right", frameon=True, facecolor="white", fontsize=8)

        add_disclaimer(fig)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # 8. Composite Executive Dashboard (High-Resolution Multi-Panel)
    @classmethod
    def plot_master_dashboard(cls, df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
        """
        Creates a high-resolution 2x2 executive analytical dashboard chart.
        """
        if save_path is None:
            save_path = cls.ensure_output_dir() / "accident_analysis_dashboard.png"

        fig = plt.figure(figsize=(16, 12), dpi=180)
        fig.patch.set_facecolor(COLOR_BG)

        # Header Title
        fig.text(
            0.5, 0.97,
            "Road Accident Data Analysis System — Comprehensive Analytical Dashboard",
            ha="center", fontsize=16, fontweight="bold", color="#1a252f"
        )
        fig.text(
            0.5, 0.95,
            f"Based on {len(df):,} validated records | Temporal, Spatial, and Severity Pattern Evaluation",
            ha="center", fontsize=10, color="#555555"
        )

        # Subplot 1: Hourly Distribution (Top Left)
        ax1 = fig.add_subplot(2, 2, 1)
        apply_base_style(fig, ax1)
        hourly = df["hour"].value_counts().reindex(range(24), fill_value=0).sort_index()
        ax1.axvspan(7.5, 10.5, color="#f39c12", alpha=0.15, label="Morning Rush")
        ax1.axvspan(16.5, 19.5, color="#e74c3c", alpha=0.15, label="Evening Rush")
        ax1.bar(range(24), hourly.values, color="#3498db", alpha=0.6, width=0.6, zorder=2)
        ax1.plot(range(24), hourly.values, color="#1b4f72", linewidth=2, marker="o", markersize=4, zorder=3)
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 2)], fontsize=8)
        ax1.set_title("A. 24-Hour Accident Distribution", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Hour", fontsize=9)
        ax1.set_ylabel("Count", fontsize=9)
        ax1.legend(loc="upper left", fontsize=7)

        # Subplot 2: Severity Donut (Top Right)
        ax2 = fig.add_subplot(2, 2, 2)
        order = ["Minor", "Moderate", "Severe", "Fatal"]
        sev_counts = df["severity"].value_counts().reindex(order, fill_value=0)
        colors = [PALETTE_SEVERITY[s] for s in order]
        wedges, texts, autotexts = ax2.pie(
            sev_counts.values,
            labels=order,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            explode=(0.02, 0.02, 0.04, 0.08),
            pctdistance=0.75,
            textprops=dict(color="#333333", fontweight="bold", fontsize=8)
        )
        centre_circle = plt.Circle((0, 0), 0.50, fc="#ffffff")
        ax2.add_artist(centre_circle)
        ax2.set_title("B. Severity Category Breakdown", fontsize=11, fontweight="bold")

        # Subplot 3: Top Hotspots (Bottom Left)
        ax3 = fig.add_subplot(2, 2, 3)
        apply_base_style(fig, ax3)
        top_regs = df["region"].value_counts().head(7).sort_values(ascending=True)
        bars = ax3.barh(range(len(top_regs)), top_regs.values, color="#16a085", height=0.55, zorder=2)
        ax3.set_yticks(range(len(top_regs)))
        ax3.set_yticklabels(top_regs.index, fontsize=8)
        for bar in bars:
            width = bar.get_width()
            ax3.annotate(f" {width}", xy=(width, bar.get_y() + bar.get_height() / 2), ha="left", va="center", fontsize=8, fontweight="bold")
        ax3.set_xlim(0, max(top_regs.values) * 1.2)
        ax3.set_title("C. Top Incident Regions (Hotspots)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Recorded Accidents", fontsize=9)

        # Subplot 4: Vehicle Type Vulnerability (Bottom Right)
        ax4 = fig.add_subplot(2, 2, 4)
        apply_base_style(fig, ax4)
        crosstab_pct = pd.crosstab(df["vehicle_type"], df["severity"], normalize="index")[order] * 100
        crosstab_pct["high_risk"] = crosstab_pct["Fatal"] + crosstab_pct["Severe"]
        crosstab_sorted = crosstab_pct.sort_values("high_risk", ascending=True).drop(columns=["high_risk"])
        y_pos = range(len(crosstab_sorted))
        lefts = np.zeros(len(crosstab_sorted))
        for sev in order:
            vals = crosstab_sorted[sev].values
            ax4.barh(y_pos, vals, left=lefts, color=PALETTE_SEVERITY[sev], height=0.55, label=sev, zorder=2)
            lefts += vals
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(crosstab_sorted.index, fontsize=8)
        ax4.set_xlim(0, 100)
        ax4.xaxis.set_major_formatter(ticker.PercentFormatter())
        ax4.set_title("D. Severity Proportion by Vehicle Type", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Proportion within Vehicle Class", fontsize=9)
        ax4.legend(loc="lower right", fontsize=7)

        # Footnote
        fig.text(
            0.5, 0.015,
            Config.DISCLAIMER_TEXT,
            ha="center", fontsize=8, color="#555555", style="italic"
        )

        fig.subplots_adjust(left=0.08, right=0.95, top=0.91, bottom=0.06, wspace=0.35, hspace=0.25)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    @classmethod
    def generate_all_charts(cls, df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """Generates the full suite of charts and returns their file paths."""
        if df is None:
            df = AccidentAnalyzer.load_data()

        if df.empty:
            return {}

        results = {}
        results["hourly"] = str(cls.plot_hourly_trend(df))
        results["day_of_week"] = str(cls.plot_day_of_week(df))
        results["severity"] = str(cls.plot_severity_distribution(df))
        results["hotspots"] = str(cls.plot_location_hotspots(df))
        results["vehicle_severity"] = str(cls.plot_vehicle_severity_breakdown(df))
        results["monthly_trend"] = str(cls.plot_monthly_trend(df))
        results["weather"] = str(cls.plot_weather_breakdown(df))
        results["dashboard"] = str(cls.plot_master_dashboard(df))

        return results
