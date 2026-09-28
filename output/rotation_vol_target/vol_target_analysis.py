"""
Volatility-targeting overlay analysis for ETF rotation strategies.
HC #428 R1: regime symmetry gate (gap < 0.50)
HC #659: permutation test mandatory
"""

from __future__ import annotations
import json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
DATA_DIR = ROOT / "output/rotation_universe_exploration"
OUT_DIR = ROOT / "output/rotation_vol_target"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
N_PERM = 2000
TARGET_VOLS = [0.10, 0.12, 0.15, 0.20]
VOL_WINDOWS = [20, 60]
LEV_CAPS = [1.0, 1.5]  # no-leverage vs mild leverage

# ─────────────────────────────────────────────────────────────────────────────
# LOAD BASE RETURNS
# ─────────────────────────────────────────────────────────────────────────────
print("Loading base returns...")
v3 = pd.read_csv(DATA_DIR / "returns_U.S._Sectors_K3.csv", parse_dates=["date"], index_col="date")["return"]
v4 = pd.read_csv(DATA_DIR / "returns_Wide_Sector+Intl_K4.csv", parse_dates=["date"], index_col="date")["return"]

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD SPY + VIX
# ─────────────────────────────────────────────────────────────────────────────
print("Downloading SPY and VIX...")
START = "2015-01-01"
END   = "2026-07-11"
spy_raw = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)

spy_close = spy_raw["Close"].squeeze()
vix_close = vix_raw["Close"].squeeze()

spy_ret = spy_close.pct_change()

# Align to strategy dates
idx = v3.index
spy_ret_aligned = spy_ret.reindex(idx).fillna(0.0)
spy_close_aligned = spy_close.reindex(idx).ffill()
vix_aligned = vix_close.reindex(idx).ffill()

# ─────────────────────────────────────────────────────────────────────────────
# REGIME CLASSIFICATION (green/red/flat by SPY close-to-close)
# ─────────────────────────────────────────────────────────────────────────────
def classify_regime(spy_ret: pd.Series, flat_band: float = 0.001) -> pd.Series:
    regime = pd.Series("flat", index=spy_ret.index)
    regime[spy_ret > flat_band] = "green"
    regime[spy_ret < -flat_band] = "red"
    return regime

regime = classify_regime(spy_ret_aligned)
print(f"Regime counts: {regime.value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(r: pd.Series, label: str = "") -> dict:
    r = r.dropna()
    ann_ret = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    down    = r[r < 0]
    sortino = ann_ret / (down.std() * np.sqrt(TRADING_DAYS)) if len(down) > 0 else np.nan
    equity  = (1 + r).cumprod()
    dd      = equity / equity.cummax() - 1
    maxdd   = dd.min()
    calmar  = ann_ret / abs(maxdd) if maxdd != 0 else np.nan
    wr      = (r > 0).mean()
    return {
        "label": label,
        "CAGR": ann_ret,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": maxdd,
        "Calmar": calmar,
        "AnnVol": ann_vol,
        "WinRate": wr,
        "N_days": len(r),
    }

def regime_sharpe(r: pd.Series, reg: pd.Series) -> dict:
    r, reg = r.align(reg, join="inner")
    out = {}
    for g in ["green", "red", "flat"]:
        mask = reg == g
        sub = r[mask]
        if len(sub) < 10:
            out[g] = np.nan
            continue
        ann_r = sub.mean() * TRADING_DAYS
        ann_v = sub.std() * np.sqrt(TRADING_DAYS)
        out[g] = ann_r / ann_v if ann_v > 0 else np.nan
        out[f"{g}_n"] = int(mask.sum())
    # regime symmetry gap
    G, R = out.get("green", np.nan), out.get("red", np.nan)
    if not np.isnan(G) and not np.isnan(R):
        denom = max(abs(G), abs(R))
        out["regime_gap"] = abs(G - R) / denom if denom > 0 else np.nan
        out["regime_gate"] = "PASS" if out["regime_gap"] <= 0.50 else "FAIL"
    return out

def permutation_test(r: pd.Series, n_perm: int = N_PERM) -> dict:
    r = r.dropna().values
    ann_r = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
    ann_v = r.std() * np.sqrt(TRADING_DAYS)
    actual_sharpe = ann_r / ann_v if ann_v > 0 else 0.0
    perm_sharpes = []
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        shuf = rng.permutation(r)
        ar = (1 + shuf).prod() ** (TRADING_DAYS / len(shuf)) - 1
        av = shuf.std() * np.sqrt(TRADING_DAYS)
        perm_sharpes.append(ar / av if av > 0 else 0.0)
    perm_sharpes = np.array(perm_sharpes)
    p_val = (perm_sharpes >= actual_sharpe).mean()
    return {
        "actual_sharpe": actual_sharpe,
        "perm_p_value": p_val,
        "perm_sharpe_mean": perm_sharpes.mean(),
        "perm_sharpe_p95": np.percentile(perm_sharpes, 95),
    }

def day_concentration(r: pd.Series) -> float:
    """Fraction of cumulative PnL from single best day."""
    total = r.sum()
    if total <= 0:
        return np.nan
    return r.max() / total

# ─────────────────────────────────────────────────────────────────────────────
# VOL-TARGETING OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
def apply_vol_target(
    r: pd.Series,
    vol_window: int,
    target_vol: float,
    lev_cap: float,
    warmup_days: int = None,
) -> pd.Series:
    """
    Scale daily returns so realized portfolio vol hits target_vol.
    Scale computed from previous day's realized vol (no look-ahead).
    """
    if warmup_days is None:
        warmup_days = vol_window
    realized_vol = r.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    # scale = target / prev_day_vol (shifted by 1 to avoid look-ahead)
    scale = (target_vol / realized_vol.shift(1)).clip(upper=lev_cap)
    # during warmup period: use 1.0x (unscaled)
    scale.iloc[:warmup_days] = 1.0
    scaled = r * scale
    return scaled

# ─────────────────────────────────────────────────────────────────────────────
# VIX-BASED SCALING
# ─────────────────────────────────────────────────────────────────────────────
def apply_vix_scale(
    r: pd.Series,
    vix: pd.Series,
    target_vol: float,
    lev_cap: float,
) -> pd.Series:
    """
    Scale = min(lev_cap, target_vol / (VIX/100)).
    VIX is annualized implied vol (%), divide by 100 for decimal.
    Previous-day VIX used (shift by 1 — no look-ahead).
    """
    vix_prev = (vix / 100.0).shift(1)
    scale = (target_vol / vix_prev).clip(upper=lev_cap)
    scale = scale.reindex(r.index).ffill()
    # fallback for NaN at start
    scale = scale.fillna(1.0)
    scaled = r * scale
    return scaled

# ─────────────────────────────────────────────────────────────────────────────
# RUN FULL BATTERY
# ─────────────────────────────────────────────────────────────────────────────
results = []
all_series = {}

def run_config(label: str, r: pd.Series, reg: pd.Series, vix: pd.Series):
    """Evaluate unscaled + all vol-target variants for a given base return series."""
    configs_to_run = []

    # Baseline (unscaled)
    configs_to_run.append(("Unscaled", r))

    # Vol-target variants: window x target_vol x lev_cap
    for vw in VOL_WINDOWS:
        for tv in TARGET_VOLS:
            for lc in LEV_CAPS:
                tag = f"VT-{int(tv*100)}pct_W{vw}_cap{lc}"
                r_scaled = apply_vol_target(r, vw, tv, lc)
                configs_to_run.append((tag, r_scaled))

    # VIX-based scaling
    for tv in TARGET_VOLS:
        for lc in LEV_CAPS:
            tag = f"VIX-{int(tv*100)}pct_cap{lc}"
            r_scaled = apply_vix_scale(r, vix, tv, lc)
            configs_to_run.append((tag, r_scaled))

    print(f"\n{'='*60}")
    print(f"UNIVERSE: {label} ({len(configs_to_run)} configs)")
    print(f"{'='*60}")

    for cfg_name, r_cfg in configs_to_run:
        full_label = f"{label} | {cfg_name}"
        m = compute_metrics(r_cfg, label=full_label)
        rs = regime_sharpe(r_cfg, reg)
        perm = permutation_test(r_cfg)
        day_conc = day_concentration(r_cfg)
        rec = {
            **m,
            **{f"regime_{k}": v for k, v in rs.items()},
            "perm_p_value": perm["perm_p_value"],
            "perm_actual_sharpe": perm["actual_sharpe"],
            "perm_p95_sharpe": perm["perm_sharpe_p95"],
            "day_conc": day_conc,
            "day_conc_gate": "PASS" if day_conc is not None and not np.isnan(day_conc) and day_conc <= 0.70 else "FAIL",
            "config": cfg_name,
            "universe": label,
        }
        results.append(rec)
        all_series[full_label] = r_cfg

        # Print summary
        rgate = rs.get("regime_gate", "N/A")
        rg    = rs.get("regime_gap", np.nan)
        pgate = "PASS" if perm["perm_p_value"] < 0.05 else "FAIL"
        print(
            f"  {cfg_name:<40s}  "
            f"Sh={m['Sharpe']:+.3f}  So={m['Sortino']:+.3f}  "
            f"DD={m['MaxDD']:.1%}  CAGR={m['CAGR']:.1%}  "
            f"R1={rgate}(gap={rg:.3f})  "
            f"Perm={pgate}(p={perm['perm_p_value']:.3f})  "
            f"DayConc={day_conc:.2%}"
        )


run_config("U.S. Sectors v3", v3, regime, vix_aligned)
run_config("Wide Sector+Intl v4", v4, regime, vix_aligned)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
df_results = pd.DataFrame(results)
df_results.to_csv(OUT_DIR / "full_results.csv", index=False)

# Best configs per universe by Sharpe (excluding baseline)
for univ in ["U.S. Sectors v3", "Wide Sector+Intl v4"]:
    sub = df_results[
        (df_results["universe"] == univ) &
        (df_results["config"] != "Unscaled") &
        (df_results["regime_regime_gate"] == "PASS") &
        (df_results["perm_p_value"] < 0.05)
    ].sort_values("Sharpe", ascending=False)
    if not sub.empty:
        print(f"\n=== BEST REGIME-PASSING + SIG CONFIGS: {univ} ===")
        print(sub[["config","Sharpe","Sortino","MaxDD","CAGR","WinRate",
                    "regime_green","regime_red","regime_regime_gap","perm_p_value"]].head(5).to_string(index=False))
    else:
        print(f"\n=== NO CONFIGS PASS BOTH GATES FOR {univ} ===")

# Save summary JSON
summary = {}
for univ in ["U.S. Sectors v3", "Wide Sector+Intl v4"]:
    sub = df_results[df_results["universe"] == univ]
    baseline = sub[sub["config"] == "Unscaled"].iloc[0].to_dict() if not sub[sub["config"] == "Unscaled"].empty else {}
    best_passing = sub[
        (sub["config"] != "Unscaled") &
        (sub["regime_regime_gate"] == "PASS") &
        (sub["perm_p_value"] < 0.05)
    ].sort_values("Sharpe", ascending=False)
    summary[univ] = {
        "baseline": baseline,
        "n_configs_tested": int(len(sub)),
        "n_regime_pass": int((sub["regime_regime_gate"] == "PASS").sum()),
        "n_perm_pass": int((sub["perm_p_value"] < 0.05).sum()),
        "n_both_pass": int(
            ((sub["regime_regime_gate"] == "PASS") & (sub["perm_p_value"] < 0.05)).sum()
        ),
        "best_passing": best_passing.iloc[0].to_dict() if not best_passing.empty else None,
    }

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nResults saved to {OUT_DIR}")
print(f"Total configs evaluated: {len(df_results)}")
