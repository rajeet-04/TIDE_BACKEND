"""Print a final summary of LSM-only vs LSM+LSTM test metrics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

s = json.loads((OUT / "lstm_summary.json").read_text())

print(f"{'PORT':<20} {'LSM RMSE':>10} {'LSM+LSTM':>10} {'IMPROVE':>10} {'R':>8}")
print("-" * 62)
for r in s:
    rmse_lsm = r["lsm_only_test"]["rmse_m"]
    rmse_comb = r["lsm_plus_lstm_test"]["rmse_m"]
    pct = 100 * (rmse_lsm - rmse_comb) / rmse_lsm
    r_corr = r["lsm_plus_lstm_test"]["r"]
    print(f"{r['port']:<20} {rmse_lsm:>9.3f} m {rmse_comb:>9.3f} m {pct:>+9.1f}% {r_corr:>8.4f}")
