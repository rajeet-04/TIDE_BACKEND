"""Phase 6 plots: side-by-side observed / LSM-only / combined for the test set."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def plot_port(port_slug: str) -> None:
    csv_path = OUTPUT_DIR / f"lstm_test_{port_slug}.csv"
    if not csv_path.exists():
        print(f"  skip {port_slug}: {csv_path} not found")
        return
    df = pd.read_csv(csv_path, parse_dates=["datetime_ist"])

    # Time series comparison (first 14 days of test segment)
    win = df.iloc[: 24 * 14]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(win["datetime_ist"], win["observed_m"], color="black", lw=0.9, label="Observed")
    axes[0].plot(win["datetime_ist"], win["lsm_predicted_m"], color="tab:blue", lw=0.9, ls="--", label="LSM only")
    axes[0].plot(win["datetime_ist"], win["combined_predicted_m"], color="tab:red", lw=0.9, label="LSM + LSTM")
    axes[0].set_ylabel("Height (m)")
    axes[0].set_title(f"{port_slug.replace('_', ' ').title()} — first 14 days of test set")
    axes[0].legend(loc="upper right")

    # Residuals
    axes[1].plot(win["datetime_ist"], win["observed_m"] - win["lsm_predicted_m"],
                 color="tab:blue", lw=0.9, label="LSM-only error")
    axes[1].plot(win["datetime_ist"], win["observed_m"] - win["combined_predicted_m"],
                 color="tab:red", lw=0.9, label="LSM+LSTM error")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel("Error: observed − predicted (m)")
    axes[1].legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{port_slug}_lstm_timeseries.png", dpi=120)
    plt.close(fig)

    # Error histograms
    err_lsm = (df["lsm_predicted_m"] - df["observed_m"]).to_numpy()
    err_combined = (df["combined_predicted_m"] - df["observed_m"]).to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(min(err_lsm.min(), err_combined.min()),
                       max(err_lsm.max(), err_combined.max()), 80)
    ax.hist(err_lsm, bins=bins, alpha=0.5, label="LSM only", color="tab:blue")
    ax.hist(err_combined, bins=bins, alpha=0.6, label="LSM + LSTM", color="tab:red")
    ax.set_xlabel("Error: predicted − observed (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"{port_slug.replace('_', ' ').title()} — test-set error distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{port_slug}_lstm_errors.png", dpi=120)
    plt.close(fig)

    print(f"  plots -> {PLOT_DIR}/{port_slug}_lstm_*.png")


def main() -> None:
    for csv_path in sorted(OUTPUT_DIR.glob("lstm_test_*.csv")):
        slug = csv_path.stem.replace("lstm_test_", "")
        print(f"plotting {slug}")
        plot_port(slug)


if __name__ == "__main__":
    main()
