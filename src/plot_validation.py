"""Generate validation plots showing forecast accuracy across horizons and seasons."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"


def plot_validation_comparison() -> None:
    """Create a comprehensive validation comparison plot."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Tidal Forecast Validation: LSM vs LSM+LSTM Across Seasons", fontsize=14, fontweight="bold")

    # Define validation files and their metadata
    validations = [
        ("validation_forecast_haldia_2024-12-01_to_2024-12-08.csv", "Haldia Dec 2024\n(Post-monsoon)", 0),
        ("validation_forecast_haldia_2024-09-01_to_2024-09-15.csv", "Haldia Sep 2024\n(Monsoon)", 1),
        ("validation_forecast_haldia_2024-06-01_to_2024-06-08.csv", "Haldia Jun 2024\n(Early monsoon)", 2),
        ("validation_forecast_diamond_harbour_2023-02-01_to_2023-02-15.csv", "Diamond Harbour Feb 2023\n(Dry season)", 3),
        ("validation_forecast_diamond_harbour_2023-03-01_to_2023-03-08.csv", "Diamond Harbour Mar 2023\n(Dry season)", 4),
    ]

    for filename, title, idx in validations:
        filepath = OUTPUT_DIR / filename
        if not filepath.exists():
            print(f"Skipping {filename} (not found)")
            continue

        df = pd.read_csv(filepath)
        ax = axes.flat[idx]

        # Plot observed vs predictions
        ax.plot(df.index, df["height_m"], "k-", linewidth=1.5, label="Observed", alpha=0.7)
        ax.plot(df.index, df["lsm_predicted_m"], "b--", linewidth=1, label="LSM only", alpha=0.7)
        ax.plot(df.index, df["combined_predicted_m"], "r-", linewidth=1, label="LSM+LSTM", alpha=0.7)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Height (m)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Remove the 6th subplot (we only have 5 validations)
    fig.delaxes(axes.flat[5])

    plt.tight_layout()
    out_path = OUTPUT_DIR / "validation_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_horizon_degradation() -> None:
    """Create a plot showing how accuracy degrades over forecast horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Forecast Accuracy Degradation by Horizon", fontsize=14, fontweight="bold")

    # Haldia results
    haldia_data = {
        "Dec 2024": {"horizons": [6, 12, 24, 48, 72], "lsm": [0.060, 0.171, 0.261, 0.240, 0.256], "lstm": [0.053, 0.102, 0.125, 0.142, 0.183]},
        "Sep 2024": {"horizons": [6, 12, 24, 48, 72], "lsm": [0.119, 0.215, 0.195, 0.180, 0.174], "lstm": [0.058, 0.060, 0.064, 0.077, 0.080]},
        "Jun 2024": {"horizons": [6, 12, 24, 48, 72], "lsm": [0.155, 0.129, 0.166, 0.148, 0.143], "lstm": [0.161, 0.131, 0.160, 0.173, 0.204]},
    }

    ax = axes[0]
    for season, data in haldia_data.items():
        ax.plot(data["horizons"], data["lsm"], "o--", label=f"{season} (LSM)", linewidth=2, markersize=6)
        ax.plot(data["horizons"], data["lstm"], "s-", label=f"{season} (LSM+LSTM)", linewidth=2, markersize=6)

    ax.set_xlabel("Forecast Horizon (hours)", fontsize=11)
    ax.set_ylabel("RMSE (m)", fontsize=11)
    ax.set_title("Haldia Port", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([6, 12, 24, 48, 72])

    # Diamond Harbour results
    dh_data = {
        "Feb 2023": {"horizons": [6, 12, 24, 48, 72], "lsm": [0.118, 0.214, 0.250, 0.272, 0.326], "lstm": [0.055, 0.173, 0.194, 0.285, 0.358]},
        "Mar 2023": {"horizons": [6, 12, 24, 48, 72], "lsm": [0.127, 0.111, 0.233, 0.231, 0.248], "lstm": [0.179, 0.162, 0.280, 0.304, 0.274]},
    }

    ax = axes[1]
    for season, data in dh_data.items():
        ax.plot(data["horizons"], data["lsm"], "o--", label=f"{season} (LSM)", linewidth=2, markersize=6)
        ax.plot(data["horizons"], data["lstm"], "s-", label=f"{season} (LSM+LSTM)", linewidth=2, markersize=6)

    ax.set_xlabel("Forecast Horizon (hours)", fontsize=11)
    ax.set_ylabel("RMSE (m)", fontsize=11)
    ax.set_title("Diamond Harbour Port", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([6, 12, 24, 48, 72])

    plt.tight_layout()
    out_path = OUTPUT_DIR / "horizon_degradation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_error_distribution() -> None:
    """Create a plot showing error distributions for LSM vs LSM+LSTM."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Prediction Error Distribution: LSM vs LSM+LSTM", fontsize=14, fontweight="bold")

    validations = [
        ("validation_forecast_haldia_2024-12-01_to_2024-12-08.csv", "Haldia Dec 2024", 0),
        ("validation_forecast_haldia_2024-09-01_to_2024-09-15.csv", "Haldia Sep 2024", 1),
        ("validation_forecast_haldia_2024-06-01_to_2024-06-08.csv", "Haldia Jun 2024", 2),
        ("validation_forecast_diamond_harbour_2023-02-01_to_2023-02-15.csv", "Diamond Harbour Feb 2023", 3),
        ("validation_forecast_diamond_harbour_2023-03-01_to_2023-03-08.csv", "Diamond Harbour Mar 2023", 4),
    ]

    for filename, title, idx in validations:
        filepath = OUTPUT_DIR / filename
        if not filepath.exists():
            continue

        df = pd.read_csv(filepath)
        ax = axes.flat[idx]

        # Create box plots
        errors = [df["lsm_error_m"].values, df["combined_error_m"].values]
        bp = ax.boxplot(errors, labels=["LSM", "LSM+LSTM"], patch_artist=True)

        # Color the boxes
        colors = ["lightblue", "lightcoral"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Error (m)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    # Remove the 6th subplot
    fig.delaxes(axes.flat[5])

    plt.tight_layout()
    out_path = OUTPUT_DIR / "error_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main() -> None:
    """Generate all validation plots."""
    print("Generating validation plots...")
    plot_validation_comparison()
    plot_horizon_degradation()
    plot_error_distribution()
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
