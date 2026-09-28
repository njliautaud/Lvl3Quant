"""
calibrate_skew_real.py — lane A3 / HC #556 R3(5).

Calibrates the wheel skew model

    iv(K,T) = sigma_atm * (1 + a*m + b*m^2),   m = log(K/S)/sqrt(T)

against REAL option surfaces from the DOLT clone, materialized at
data/cache/options_real/chains/{TICKER}.parquet by
materialize_chains_parallel.py. Replaces the hardcoded a=-0.10, b=+0.20
in strategy/iv_skew.py (the "calibration TODO" in its docstring).

Method, per (ticker, date, expiration) surface slice:
  - restrict to wheel-relevant tenor (default 20-45 DTE, near the 30d ATM
    IV the engine uses as sigma_atm) and |m| <= 0.60 (covers 10-45Δ wings)
  - need >= 6 strikes with valid vol
  - OLS fit vol = c0 + c1*m + c2*m^2  ->  sigma_atm=c0, a=c1/c0, b=c2/c0
  - sanity drop: c0 outside (0.05, 3.0), |a|>1.0, |b|>3.0, R^2 < 0.5

Aggregation: per-ticker median (a, b) + pooled global median, plus
VIX-regime split (VIX<18 / 18-25 / >25) for diagnostics.

Output: data/cache/skew_calibration_real.json
  { "global": {"a":..., "b":...}, "per_ticker": {...}, "by_vix_regime": {...},
    "diagnostics": {...} }

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python -m data.calibrate_skew_real [--dte-min 20 --dte-max 45]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data" / "cache"
CHAINS_DIR = CACHE / "options_real" / "chains"
OUT_JSON = CACHE / "skew_calibration_real.json"


def fit_ticker(ticker: str, spots: pd.DataFrame, dte_min: int, dte_max: int,
               m_max: float) -> pd.DataFrame:
    """Return per-(date,expiration) fit rows for one ticker."""
    f = CHAINS_DIR / f"{ticker}.parquet"
    ch = pd.read_parquet(f, columns=["date", "expiration", "strike", "type",
                                     "vol", "dte"])
    ch = ch[(ch["dte"] >= dte_min) & (ch["dte"] <= dte_max)]
    ch = ch.dropna(subset=["vol", "strike"])
    ch = ch[(ch["vol"] > 0.01) & (ch["vol"] < 4.0)]
    if ch.empty:
        return pd.DataFrame()
    sp = spots[spots["ticker"] == ticker][["date", "close"]]
    ch = ch.merge(sp, on="date", how="inner")
    if ch.empty:
        return pd.DataFrame()
    T = ch["dte"].values / 365.0
    ch["m"] = np.log(ch["strike"].values / ch["close"].values) / np.sqrt(T)
    ch = ch[ch["m"].abs() <= m_max]

    # For each strike keep OTM side only (puts below spot, calls above) —
    # the cleaner half of the smile, and the side the wheel actually trades.
    otm = ch[((ch["type"] == "p") & (ch["m"] <= 0.02)) |
             ((ch["type"] == "c") & (ch["m"] >= -0.02))]

    rows = []
    for (d, e), g in otm.groupby(["date", "expiration"]):
        g = g.drop_duplicates(subset=["strike", "type"])
        if len(g) < 6 or g["m"].nunique() < 5:
            continue
        m = g["m"].values
        v = g["vol"].values
        try:
            X = np.column_stack([np.ones_like(m), m, m * m])
            coef, res, rank, _ = np.linalg.lstsq(X, v, rcond=None)
        except Exception:
            continue
        c0, c1, c2 = coef
        if not (0.05 < c0 < 3.0):
            continue
        pred = X @ coef
        ss_res = float(((v - pred) ** 2).sum())
        ss_tot = float(((v - v.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        a, b = c1 / c0, c2 / c0
        if abs(a) > 1.0 or abs(b) > 3.0 or r2 < 0.5:
            continue
        rows.append({"ticker": ticker, "date": d, "expiration": e,
                     "n": len(g), "sigma_atm": c0, "a": a, "b": b, "r2": r2})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dte-min", type=int, default=20)
    ap.add_argument("--dte-max", type=int, default=45)
    ap.add_argument("--m-max", type=float, default=0.60)
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    px = pd.read_parquet(CACHE / "prices.parquet", columns=["date", "ticker", "close"])
    px["date"] = pd.to_datetime(px["date"])

    macro = pd.read_parquet(CACHE / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"])
    vix = macro.set_index("date")["vix"] if "vix" in macro.columns else None

    files = sorted(CHAINS_DIR.glob("*.parquet"))
    print(f"[skew-cal] {len(files)} ticker chain files", flush=True)
    all_fits = []
    for f in files:
        t = f.stem
        try:
            fit = fit_ticker(t, px, args.dte_min, args.dte_max, args.m_max)
        except Exception as e:
            print(f"[skew-cal] {t}: FAIL {e}", flush=True)
            continue
        if not fit.empty:
            all_fits.append(fit)
            print(f"[skew-cal] {t}: {len(fit)} surface fits, "
                  f"median a={fit['a'].median():+.4f} b={fit['b'].median():+.4f} "
                  f"r2={fit['r2'].median():.3f}", flush=True)
        else:
            print(f"[skew-cal] {t}: 0 usable fits", flush=True)

    if not all_fits:
        raise SystemExit("[skew-cal] no fits — chains missing?")
    fits = pd.concat(all_fits, ignore_index=True)
    fits["date"] = pd.to_datetime(fits["date"])

    per_ticker = (fits.groupby("ticker")
                  .agg(a=("a", "median"), b=("b", "median"),
                       n_fits=("a", "size"), r2=("r2", "median"))
                  .round(4))

    g_a = float(fits["a"].median())
    g_b = float(fits["b"].median())

    by_regime = {}
    if vix is not None:
        fits = fits.merge(vix.rename("vix"), left_on="date", right_index=True,
                          how="left")
        for name, lo, hi in [("vix_lt18", 0, 18), ("vix_18_25", 18, 25),
                             ("vix_gt25", 25, 999)]:
            sub = fits[(fits["vix"] >= lo) & (fits["vix"] < hi)]
            if len(sub) > 50:
                by_regime[name] = {"a": round(float(sub["a"].median()), 4),
                                   "b": round(float(sub["b"].median()), 4),
                                   "n": int(len(sub))}

    out = {
        "model": "iv = sigma_atm * (1 + a*m + b*m^2), m=log(K/S)/sqrt(T)",
        "source": "DOLT post-no-preference/options option_chain (OTM side, "
                  f"{args.dte_min}-{args.dte_max} DTE, |m|<={args.m_max})",
        "prior_hardcoded": {"a": -0.10, "b": 0.20},
        "global": {"a": round(g_a, 4), "b": round(g_b, 4)},
        "by_vix_regime": by_regime,
        "per_ticker": per_ticker.to_dict(orient="index"),
        "diagnostics": {
            "n_surface_fits": int(len(fits)),
            "n_tickers": int(fits["ticker"].nunique()),
            "date_min": str(fits["date"].min().date()),
            "date_max": str(fits["date"].max().date()),
            "a_iqr": [round(float(fits["a"].quantile(q)), 4) for q in (0.25, 0.75)],
            "b_iqr": [round(float(fits["b"].quantile(q)), 4) for q in (0.25, 0.75)],
            "median_r2": round(float(fits["r2"].median()), 4),
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    fits.drop(columns=["vix"], errors="ignore").to_parquet(
        CACHE / "skew_calibration_fits.parquet", index=False)
    print(f"\n[skew-cal] GLOBAL a={g_a:+.4f} b={g_b:+.4f} "
          f"(prior hardcoded a=-0.10 b=+0.20)")
    print(f"[skew-cal] wrote {args.out} + skew_calibration_fits.parquet")


if __name__ == "__main__":
    main()
