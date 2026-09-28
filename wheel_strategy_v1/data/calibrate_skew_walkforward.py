"""
calibrate_skew_walkforward.py — lane A3 leak-free skew calibration.

Replaces the single in-sample global fit (skew_calibration_real.json) with a
walk-forward schedule: for each backtest year Y, skew (a, b) are the medians
of per-surface fits dated STRICTLY BEFORE Y-01-01 (expanding prior window).
Trade decisions in year Y therefore never see year-Y (or later) surfaces.

Reads the per-(ticker,date,expiration) surface fits already materialized by
calibrate_skew_real.py at data/cache/skew_calibration_fits.parquet.

2020 caveat: the only prior data is 2019-05 -> 2019-12 (DOLT chains are
strike-sparse in 2019), giving ~42 surface fits. Thin but leak-free; n_fits
is recorded per period so the consumer can judge.

Output: data/cache/skew_calibration_walkforward.json
    {"periods": {"2020": {"a":..,"b":..,"n_fits":..,"fit_end":"2019-12-31"},
                 ...}}

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python -m data.calibrate_skew_walkforward [--first-year 2020 --last-year 2026]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
FITS = CACHE / "skew_calibration_fits.parquet"
OUT = CACHE / "skew_calibration_walkforward.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-year", type=int, default=2020)
    ap.add_argument("--last-year", type=int, default=2026)
    ap.add_argument("--min-fits", type=int, default=30,
                    help="Below this prior-fit count, fall back to the "
                         "hardcoded a=-0.10/b=+0.20 for that period.")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--per-ticker", action="store_true",
                    help="Also emit per-ticker walk-forward (a,b): for each "
                         "year Y, a ticker gets the median of ITS OWN surface "
                         "fits dated strictly before Y-01-01, falling back to "
                         "the global-year (a,b) when it has < --min-fits "
                         "prior fits. Output goes to "
                         "skew_calibration_walkforward_perticker.json")
    args = ap.parse_args()

    fits = pd.read_parquet(FITS)
    fits["date"] = pd.to_datetime(fits["date"])

    periods = {}
    for y in range(args.first_year, args.last_year + 1):
        cutoff = pd.Timestamp(f"{y}-01-01")
        prior = fits[fits["date"] < cutoff]
        if len(prior) < args.min_fits:
            periods[str(y)] = {"a": -0.10, "b": 0.20, "n_fits": int(len(prior)),
                               "source": "hardcoded_fallback_thin_prior"}
        else:
            periods[str(y)] = {
                "a": round(float(prior["a"].median()), 4),
                "b": round(float(prior["b"].median()), 4),
                "n_fits": int(len(prior)),
                "n_tickers": int(prior["ticker"].nunique()),
                "fit_start": str(prior["date"].min().date()),
                "fit_end": str(prior["date"].max().date()),
                "source": "expanding_prior_median",
            }
        p = periods[str(y)]
        print(f"[skew-wf] {y}: a={p['a']:+.4f} b={p['b']:+.4f} "
              f"n_fits={p['n_fits']} ({p['source']})")

    out = {
        "model": "iv = sigma_atm * (1 + a*m + b*m^2), m=log(K/S)/sqrt(T)",
        "method": "walk-forward: per backtest year, median of per-surface OLS "
                  "fits dated strictly before Jan 1 of that year "
                  "(expanding prior). Leak-free for 2020+ backtests.",
        "fits_source": str(FITS.name),
        "prior_hardcoded": {"a": -0.10, "b": 0.20},
        "periods": periods,
    }

    if args.per_ticker:
        # Per-ticker walk-forward: same leak-free expanding-prior convention,
        # but medians computed within each ticker's own surface fits. Tickers
        # with < min-fits prior fits are OMITTED for that year — the consumer
        # (iv_skew per-ticker schedule) falls back to the global-year (a, b).
        per_ticker = {}
        for y in range(args.first_year, args.last_year + 1):
            cutoff = pd.Timestamp(f"{y}-01-01")
            prior = fits[fits["date"] < cutoff]
            yr = {}
            for tk, g in prior.groupby("ticker"):
                if len(g) < args.min_fits:
                    continue
                yr[str(tk)] = {
                    "a": round(float(g["a"].median()), 4),
                    "b": round(float(g["b"].median()), 4),
                    "n_fits": int(len(g)),
                }
            per_ticker[str(y)] = yr
            print(f"[skew-wf-pt] {y}: {len(yr)} tickers with >= "
                  f"{args.min_fits} prior fits (global fallback for rest)")
        out["method_per_ticker"] = (
            "per ticker per year: median of that ticker's surface fits dated "
            f"strictly before Jan 1 (expanding prior); tickers with < "
            f"{args.min_fits} prior fits omitted -> consumer falls back to "
            "the global-year (a, b) above.")
        out["per_ticker_min_fits"] = int(args.min_fits)
        out["per_ticker_periods"] = per_ticker
        pt_out = str(CACHE / "skew_calibration_walkforward_perticker.json")
        Path(pt_out).write_text(json.dumps(out, indent=2))
        print(f"[skew-wf] wrote {pt_out}")
    else:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[skew-wf] wrote {args.out}")


if __name__ == "__main__":
    main()
