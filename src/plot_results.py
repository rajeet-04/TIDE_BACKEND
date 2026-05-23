"""Generate diagnostic plots: predicted vs observed (15-day window),
residual histogram, and amplitude bar chart per port.
"""
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
    val = pd.read_csv(OUTPUT_DIR / f"validation_{port_slug}.csv",
                      parse_dates=["datetime_ist"])
    hc = pd.read_csv(OUTPUT_DIR / f"harmonic_constants_{port_slug}.csv")

    # 1) 15-day predicted vs observed near the start of validation
    window = val.iloc[: 24 * 15]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(window["datetime_ist"], window["height_m"], label="Observed", lw=1.0)
    ax.plot(window["datetime_ist"], window["predicted_m"], label="LSM Predicted",
            lw=1.0, ls="--")
    ax.set_title(f"{port_slug.replace('_', ' ').title()} — first 15 days of validation")
    ax.set_ylabel("Height (m)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{port_slug}_15day.png", dpi=120)
    plt.close(fig)

    # 2) Scatter predicted vs observed
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(val["height_m"], val["predicted_m"], s=1, alpha=0.2)
    lo = min(val["height_m"].min(), val["predicted_m"].min())
    hi = max(val["height_m"].max(), val["predicted_m"].max())
    ax.plot([lo, hi], [lo, hi], "r-", lw=1, label="y=x")
    ax.set_xlabel("Observed (m)")
    ax.set_ylabel("Predicted (m)")
    ax.set_title(f"{port_slug.replace('_', ' ').title()} — validation scatter")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{port_slug}_scatter.png", dpi=120)
    plt.close(fig)

    # 3) Residual histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(val["residual_m"], bins=80)
    ax.set_xlabel("Residual = Predicted - Observed (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"{port_slug.replace('_', ' ').title()} — residual distribution")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{port_slug}_residuals.png", dpi=120)
    plt.close(fig)

    # 4) Amplitudes bar chart (sorted, top 11)
    main = hc.sort_values("amplitude_m", ascending=False).head(13)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(main["name"], main["amplitude_m"])
    ax.set_ylabel("Amplitude (m)")
    ax.set_title(f"{port_slug.replace('_', ' ').title()} — constituent amplitudes")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{port_slug}_amplitudes.png", dpi=120)
    plt.close(fig)

    print(f"  plots -> {PLOT_DIR}/{port_slug}_*.png")


def main() -> None:
    for hc_path in sorted(OUTPUT_DIR.glob("harmonic_constants_*.csv")):
        slug = hc_path.stem.replace("harmonic_constants_", "")
        print(f"plotting {slug}")
        plot_port(slug)


if __name__ == "__main__":
    main()
