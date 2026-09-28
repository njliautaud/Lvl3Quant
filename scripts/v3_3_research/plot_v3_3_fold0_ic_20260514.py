#!/usr/bin/env python3
"""
Visualization-only: plot v3.3 fold 0 IC (logged in trainer stdout 2026-05-14)
against v2 baseline (proven champion) across horizons, plus sigma-bar showing
which heads converged vs which the uncertainty-weighting auto-down-weighted.

No model code touched. Inputs are hard-coded numbers parsed from log/MLflow.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- Data (from /home/nick/Lvl3Quant/logs/cnn_mamba_v3_3_20260514_035133.log)
horizons = ["1s", "5s", "10s", "30s"]

# v3.3 fold 0 OOT IC (Ep 5/5 last line before crash)
v33_ic = [0.2859, 0.1418, 0.0962, 0.0589]

# v2 CNN-Mamba published concat IC (proven champion baseline)
v2_ic = [0.222, 0.141, 0.106, None]   # v2 was not trained on 30s

# v3.2 aggregate OOT 5-day (from comparison.md)
v32_ic = [0.2739, 0.1171, 0.0584, 0.0271]

# Sigma (learned uncertainty) by head — final state at end of Ep 5
sigma_heads = {
    "log_ret_60s":   0.0497,
    "log_ret_5min":  0.0497,
    "p_up_60s":      0.0497,
    "log_ret_1s":    None,   # not in σ-extremes log line
    "log_ret_5s":    3.4846,
    "log_ret_10s":   4.6791,
    "log_ret_30s":   7.4850,
}

# ---- Figure: 2 panels side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("v3.3 Fold 0 (uncertainty-weighted MTL) vs v2 baseline vs v3.2 — 2026-05-14",
             fontsize=14, fontweight="bold")

# ---------- Panel 1: IC by horizon ----------
x = np.arange(len(horizons))
w = 0.26

v2_plot = [v if v is not None else 0 for v in v2_ic]
v2_mask = [v is not None for v in v2_ic]

b1 = ax1.bar(x - w, v2_plot, w, label="v2 CNN-Mamba (published)", color="#888")
b2 = ax1.bar(x,     v32_ic,  w, label="v3.2 (5-day OOT agg)",     color="#3a86ff")
b3 = ax1.bar(x + w, v33_ic,  w, label="v3.3 fold 0 (today)",      color="#ff006e")

# Hatch missing v2 30s bar
for bar, m in zip(b1, v2_mask):
    if not m:
        bar.set_hatch("////")
        bar.set_edgecolor("white")

# Numeric labels
for bars, vals in [(b1, v2_ic), (b2, v32_ic), (b3, v33_ic)]:
    for bar, v in zip(bars, vals):
        if v is None:
            continue
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)

ax1.set_xticks(x)
ax1.set_xticklabels(horizons)
ax1.set_ylabel("Concat IC (Spearman)")
ax1.set_title("Information Coefficient by horizon")
ax1.axhline(0, color="k", lw=0.6)
ax1.legend(loc="upper right")
ax1.grid(axis="y", alpha=0.3)

# Delta annotation
ax1.text(0.02, 0.95,
         f"v3.3 vs v2:\n"
         f"  1s:  +{(v33_ic[0]-v2_ic[0])/v2_ic[0]*100:+.0f}%\n"
         f"  5s:  +{(v33_ic[1]-v2_ic[1])/v2_ic[1]*100:+.0f}%\n"
         f"  10s: {(v33_ic[2]-v2_ic[2])/v2_ic[2]*100:+.0f}%",
         transform=ax1.transAxes, fontsize=10, va="top",
         bbox=dict(boxstyle="round", facecolor="#fff4d6", alpha=0.85))

# ---------- Panel 2: Sigma (learned uncertainty per head) ----------
heads = list(sigma_heads.keys())
sigmas = [sigma_heads[h] if sigma_heads[h] is not None else np.nan for h in heads]

colors = []
for s in sigmas:
    if np.isnan(s):
        colors.append("#cccccc")
    elif s <= 0.10:
        colors.append("#06d6a0")     # high-confidence head
    elif s <= 1.0:
        colors.append("#ffd166")
    else:
        colors.append("#ef476f")     # auto-down-weighted

ax2.barh(heads, [s if not np.isnan(s) else 0 for s in sigmas], color=colors)
ax2.set_xscale("log")
ax2.set_xlabel("Learned σ  (log scale) — lower = more confident")
ax2.set_title("Uncertainty per head at end of Ep 5\n"
              "green = trusted, red = auto-down-weighted")
ax2.axvline(0.05, color="green", ls=":", alpha=0.6, label="σ-floor 0.05")
ax2.axvline(1.0,  color="orange", ls=":", alpha=0.6, label="σ=1.0 ref")
ax2.grid(axis="x", alpha=0.3)
ax2.legend(loc="lower right")

for i, s in enumerate(sigmas):
    if np.isnan(s):
        ax2.text(0.06, i, "(not in σ-extremes log line)",
                 va="center", fontsize=8, style="italic", color="#666")
    else:
        ax2.text(s * 1.15, i, f"{s:.3f}", va="center", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])

out = "/home/jupiter/Lvl3Quant/output/v3_3_fold0_ic_report_20260514.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"WROTE {out}")
