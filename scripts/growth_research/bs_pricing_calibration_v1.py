#!/usr/bin/env python3
"""
BS Pricing Calibration v1 — Calibrate Black-Scholes Against Real Market Data
=============================================================================

Context: V10 adversarial audit found BS pricing underestimates real option costs
by ~61% on average. The current 15% haircut doesn't compensate.

This script:
  1. Loads real options chain snapshots from data/options_chains/
  2. Computes BS theoretical prices using our ATR-based IV estimation
  3. Compares to real market mid-prices
  4. Analyzes the BS->market mapping by moneyness, DTE, VIX
  5. Fits calibration models (linear + multivariate)
  6. Computes optimal haircut
  7. Re-runs simplified V10 backtest with calibrated pricing
  8. Logs results to MLflow

Usage:
    cd /home/jupiter/Lvl3Quant
    python -m scripts.growth_research.bs_pricing_calibration_v1
    # or
    python scripts/growth_research/bs_pricing_calibration_v1.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Path setup ──────────────────────────────────────────────────────
# Auto-detect Jupiter vs Neptune
if Path("/home/jupiter/Lvl3Quant").exists():
    BASE = Path("/home/jupiter/Lvl3Quant")
elif Path("/home/nick/Lvl3Quant").exists():
    BASE = Path("/home/nick/Lvl3Quant")
else:
    BASE = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(BASE))

from research.tools.options_pricer import (
    RISK_FREE_RATE,
    bs_call_price,
    bs_put_price,
    estimate_iv,
    compute_atr,
)

CHAINS_DIR = BASE / "data" / "options_chains"
OUTPUT_DIR = BASE / "output" / "growth_research" / "bs_pricing_calibration_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MLflow config
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "bs_pricing_calibration_v1"

# Sector ETFs used in V10
SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLY", "XLRE",
    "XLC", "SMH", "IBB", "XBI", "XHB", "XRT", "XME", "XOP", "GDX", "GDXJ",
    "KRE", "IYT", "IYR",
]

# Tickers we have chain data for
CHAIN_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AMD", "TSLA",
    "SPY", "QQQ", "IWM", "AVGO", "CRM", "NFLX", "ADBE", "INTC",
    "QCOM", "MU", "AMAT", "JPM", "BAC", "GS", "UNH", "LLY",
    "ABBV", "WMT", "COST", "HD", "XOM", "CVX", "GLD", "SLV", "TLT",
]


def fprint(msg: str) -> None:
    """Print with timestamp prefix."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# =====================================================================
# STEP 1: Load all chain snapshots
# =====================================================================

def load_all_chains() -> pd.DataFrame:
    """Load and concatenate all available chain snapshots."""
    if not CHAINS_DIR.exists():
        return pd.DataFrame()

    date_dirs = sorted([d for d in CHAINS_DIR.iterdir() if d.is_dir()])
    if not date_dirs:
        return pd.DataFrame()

    frames = []
    for d in date_dirs:
        all_file = d / "_all_tickers.parquet"
        if all_file.exists():
            df = pd.read_parquet(all_file)
            df["snapshot_date"] = d.name  # e.g. "2026-07-17"
            frames.append(df)
        else:
            # Load individual ticker files
            for pf in d.glob("*.parquet"):
                if pf.name.startswith("_"):
                    continue
                df = pd.read_parquet(pf)
                df["snapshot_date"] = d.name
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    fprint(f"Loaded {len(combined):,} option records from {len(date_dirs)} snapshot dates")
    fprint(f"  Tickers: {combined['ticker'].nunique()}, DTE range: {combined['dte_days'].min()}-{combined['dte_days'].max()}")
    return combined


# =====================================================================
# STEP 2: Fetch ATR data via yfinance for BS pricing
# =====================================================================

def fetch_atr_data(tickers: list[str], atr_period: int = 14) -> dict[str, float]:
    """Fetch ATR for each ticker using yfinance. Returns {ticker: atr_value}."""
    try:
        import yfinance as yf
    except ImportError:
        fprint("WARNING: yfinance not installed. Using fallback ATR estimation from IV.")
        return {}

    atr_map = {}
    batch_size = 10
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, period="30d", interval="1d",
                               progress=False, group_by="ticker", threads=True)
            for tk in batch:
                try:
                    if len(batch) == 1:
                        tk_data = data
                    else:
                        tk_data = data[tk] if tk in data.columns.get_level_values(0) else None
                    if tk_data is None or tk_data.empty:
                        continue
                    tk_data = tk_data.dropna()
                    if len(tk_data) < atr_period + 1:
                        continue
                    atr_val = compute_atr(
                        tk_data["High"].values,
                        tk_data["Low"].values,
                        tk_data["Close"].values,
                        period=atr_period,
                    )
                    atr_map[tk] = atr_val
                except Exception:
                    continue
        except Exception as e:
            fprint(f"  yfinance batch error: {e}")
            continue

    fprint(f"  ATR data fetched for {len(atr_map)}/{len(tickers)} tickers")
    return atr_map


# =====================================================================
# STEP 3: Compute BS prices and compare to market
# =====================================================================

def compute_comparisons(chains: pd.DataFrame, atr_map: dict[str, float],
                        vix_level: float = 20.0) -> pd.DataFrame:
    """
    For each option in the chain, compute BS theoretical price and compare
    to market mid-price.
    """
    # Filter to usable options
    df = chains.copy()

    # Basic quality filters
    df = df[df["dte_days"] > 0]  # No expired options
    df = df[df["bid"] > 0]       # Must have positive bid
    df = df[df["ask"] > 0]       # Must have positive ask
    df = df[df["mid"] > 0.05]    # Filter out penny options
    df = df[df["ask"] - df["bid"] < df["mid"] * 3]  # Filter absurd spreads

    # Compute moneyness = (strike - spot) / spot
    df["moneyness"] = (df["strike"] - df["underlying_price"]) / df["underlying_price"]
    df["abs_moneyness"] = df["moneyness"].abs()

    # Filter to reasonable moneyness range (within 15% of spot)
    df = df[df["abs_moneyness"] <= 0.15]

    fprint(f"  After filtering: {len(df):,} options to compare")

    # Compute BS price for each option
    bs_prices = []
    bs_iv_estimates = []

    for _, row in df.iterrows():
        S = row["underlying_price"]
        K = row["strike"]
        T = row["dte_days"] / 365.0
        tk = row["ticker"]

        # Method A: Use our ATR-based IV estimation
        atr = atr_map.get(tk, None)
        if atr is not None and atr > 0:
            sigma_atr = estimate_iv(atr=atr, spot=S, vix=vix_level)
        else:
            # Fallback: estimate ATR from spot price (rough 1.5% daily range)
            atr_est = S * 0.015
            sigma_atr = estimate_iv(atr=atr_est, spot=S, vix=vix_level)

        if row["option_type"] == "call":
            bs_p = bs_call_price(S, K, T, RISK_FREE_RATE, sigma_atr)
        else:
            bs_p = bs_put_price(S, K, T, RISK_FREE_RATE, sigma_atr)

        bs_prices.append(bs_p)
        bs_iv_estimates.append(sigma_atr)

    df = df.copy()
    df["bs_price"] = bs_prices
    df["bs_iv_estimate"] = bs_iv_estimates

    # Filter out near-zero BS prices (deep OTM with negligible value)
    df = df[df["bs_price"] > 0.01]

    # Compute ratio: market_mid / bs_price
    df["ratio"] = df["mid"] / df["bs_price"]
    df["abs_error"] = (df["mid"] - df["bs_price"]).abs()
    df["pct_error"] = df["abs_error"] / df["mid"] * 100

    # Compute moneyness buckets
    def moneyness_bucket(m, opt_type):
        """Classify moneyness. For calls, negative = ITM. For puts, positive = ITM."""
        if opt_type == "call":
            otm = m  # positive = OTM for calls
        else:
            otm = -m  # negative = OTM for puts (flip sign)
        abs_otm = abs(otm)
        if abs_otm <= 0.01:
            return "ATM (0-1%)"
        elif abs_otm <= 0.02:
            return "~2% OTM"
        elif abs_otm <= 0.04:
            return "~3-4% OTM"
        elif abs_otm <= 0.06:
            return "~5-6% OTM"
        elif abs_otm <= 0.10:
            return "~7-10% OTM"
        else:
            return "10%+ OTM"

    df["moneyness_bucket"] = df.apply(
        lambda r: moneyness_bucket(r["moneyness"], r["option_type"]), axis=1)

    # DTE buckets
    def dte_bucket(dte):
        if dte <= 7:
            return "0-7d"
        elif dte <= 14:
            return "8-14d"
        elif dte <= 21:
            return "15-21d"
        elif dte <= 28:
            return "22-28d"
        elif dte <= 42:
            return "29-42d"
        else:
            return "42d+"

    df["dte_bucket"] = df["dte_days"].apply(dte_bucket)

    # IV bucket (using real market IV)
    def iv_bucket(iv_val):
        if iv_val < 0.20:
            return "IV<20%"
        elif iv_val < 0.30:
            return "IV 20-30%"
        elif iv_val < 0.40:
            return "IV 30-40%"
        elif iv_val < 0.50:
            return "IV 40-50%"
        else:
            return "IV 50%+"

    df["iv_bucket"] = df["iv"].apply(iv_bucket)

    fprint(f"  Comparison dataset: {len(df):,} options, median ratio={df['ratio'].median():.3f}")
    return df


# =====================================================================
# STEP 4: Analysis — distribution, bucketed ratios, regression
# =====================================================================

def analyze_pricing_gap(comp: pd.DataFrame) -> dict:
    """Analyze the BS vs market pricing gap across multiple dimensions."""
    results = {}

    # ── 4A: Overall ratio distribution ──
    r = comp["ratio"]
    overall = {
        "count": int(len(r)),
        "mean": float(r.mean()),
        "median": float(r.median()),
        "std": float(r.std()),
        "p10": float(r.quantile(0.10)),
        "p25": float(r.quantile(0.25)),
        "p75": float(r.quantile(0.75)),
        "p90": float(r.quantile(0.90)),
        "pct_bs_underprices": float((r > 1.0).mean() * 100),
    }
    results["overall_ratio"] = overall

    fprint("\n" + "=" * 70)
    fprint("OVERALL BS vs MARKET RATIO (market_mid / bs_price)")
    fprint("=" * 70)
    fprint(f"  Count:  {overall['count']:,}")
    fprint(f"  Mean:   {overall['mean']:.3f}  (BS {'under' if overall['mean'] > 1 else 'over'}prices by {abs(overall['mean'] - 1) * 100:.1f}%)")
    fprint(f"  Median: {overall['median']:.3f}")
    fprint(f"  p10-p90: [{overall['p10']:.3f}, {overall['p90']:.3f}]")
    fprint(f"  p25-p75: [{overall['p25']:.3f}, {overall['p75']:.3f}]")
    fprint(f"  % where BS underprices: {overall['pct_bs_underprices']:.1f}%")

    # ── 4B: By moneyness bucket ──
    fprint("\n" + "-" * 50)
    fprint("RATIO BY MONEYNESS")
    fprint("-" * 50)
    money_stats = {}
    for bucket in ["ATM (0-1%)", "~2% OTM", "~3-4% OTM", "~5-6% OTM", "~7-10% OTM", "10%+ OTM"]:
        sub = comp[comp["moneyness_bucket"] == bucket]
        if len(sub) < 5:
            continue
        s = {
            "count": int(len(sub)),
            "mean_ratio": float(sub["ratio"].mean()),
            "median_ratio": float(sub["ratio"].median()),
            "mean_pct_error": float(sub["pct_error"].mean()),
        }
        money_stats[bucket] = s
        fprint(f"  {bucket:15s}: n={s['count']:5d}, median_ratio={s['median_ratio']:.3f}, "
               f"mean_pct_err={s['mean_pct_error']:.1f}%")
    results["by_moneyness"] = money_stats

    # ── 4C: By DTE bucket ──
    fprint("\n" + "-" * 50)
    fprint("RATIO BY DTE")
    fprint("-" * 50)
    dte_stats = {}
    for bucket in ["0-7d", "8-14d", "15-21d", "22-28d", "29-42d", "42d+"]:
        sub = comp[comp["dte_bucket"] == bucket]
        if len(sub) < 5:
            continue
        s = {
            "count": int(len(sub)),
            "mean_ratio": float(sub["ratio"].mean()),
            "median_ratio": float(sub["ratio"].median()),
            "mean_pct_error": float(sub["pct_error"].mean()),
        }
        dte_stats[bucket] = s
        fprint(f"  {bucket:10s}: n={s['count']:5d}, median_ratio={s['median_ratio']:.3f}, "
               f"mean_pct_err={s['mean_pct_error']:.1f}%")
    results["by_dte"] = dte_stats

    # ── 4D: By IV bucket (market IV) ──
    fprint("\n" + "-" * 50)
    fprint("RATIO BY MARKET IV LEVEL")
    fprint("-" * 50)
    iv_stats = {}
    for bucket in ["IV<20%", "IV 20-30%", "IV 30-40%", "IV 40-50%", "IV 50%+"]:
        sub = comp[comp["iv_bucket"] == bucket]
        if len(sub) < 5:
            continue
        s = {
            "count": int(len(sub)),
            "mean_ratio": float(sub["ratio"].mean()),
            "median_ratio": float(sub["ratio"].median()),
            "iv_gap_mean": float((sub["iv"] - sub["bs_iv_estimate"]).mean()),
        }
        iv_stats[bucket] = s
        fprint(f"  {bucket:12s}: n={s['count']:5d}, median_ratio={s['median_ratio']:.3f}, "
               f"IV gap (mkt-BS)={s['iv_gap_mean']:+.3f}")
    results["by_iv"] = iv_stats

    # ── 4E: By call vs put ──
    fprint("\n" + "-" * 50)
    fprint("RATIO BY OPTION TYPE")
    fprint("-" * 50)
    for otype in ["call", "put"]:
        sub = comp[comp["option_type"] == otype]
        if len(sub) < 5:
            continue
        fprint(f"  {otype:5s}: n={len(sub):5d}, median_ratio={sub['ratio'].median():.3f}, "
               f"mean_pct_err={sub['pct_error'].mean():.1f}%")

    # ── 4F: IV estimation accuracy ──
    fprint("\n" + "-" * 50)
    fprint("IV ESTIMATION ACCURACY (ATR-based vs market IV)")
    fprint("-" * 50)
    iv_diff = comp["iv"] - comp["bs_iv_estimate"]
    fprint(f"  Mean IV gap (market - BS):  {iv_diff.mean():+.4f} ({iv_diff.mean()*100:+.1f}%)")
    fprint(f"  Median IV gap:             {iv_diff.median():+.4f}")
    fprint(f"  Std IV gap:                {iv_diff.std():.4f}")
    fprint(f"  % where market IV > BS IV: {(iv_diff > 0).mean()*100:.1f}%")

    # Correlation between our IV estimate and market IV
    iv_corr = comp[["iv", "bs_iv_estimate"]].corr().iloc[0, 1]
    fprint(f"  Correlation(market IV, BS IV): {iv_corr:.4f}")
    results["iv_accuracy"] = {
        "mean_gap": float(iv_diff.mean()),
        "median_gap": float(iv_diff.median()),
        "std_gap": float(iv_diff.std()),
        "correlation": float(iv_corr),
        "pct_market_higher": float((iv_diff > 0).mean() * 100),
    }

    return results


# =====================================================================
# STEP 5: Fit calibration models
# =====================================================================

def fit_calibration_models(comp: pd.DataFrame) -> dict:
    """Fit linear and multivariate calibration models."""
    fprint("\n" + "=" * 70)
    fprint("CALIBRATION MODELS")
    fprint("=" * 70)

    results = {}

    # ── 5A: Simple linear regression: market_mid = a * bs_price + b ──
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error

    X_simple = comp[["bs_price"]].values
    y = comp["mid"].values

    lr = LinearRegression()
    lr.fit(X_simple, y)
    y_pred_lr = lr.predict(X_simple)

    results["linear"] = {
        "slope": float(lr.coef_[0]),
        "intercept": float(lr.intercept_),
        "r2": float(r2_score(y, y_pred_lr)),
        "mae": float(mean_absolute_error(y, y_pred_lr)),
        "formula": f"market_mid = {lr.coef_[0]:.4f} * bs_price + {lr.intercept_:.4f}",
    }
    fprint(f"\n  Linear: {results['linear']['formula']}")
    fprint(f"    R2={results['linear']['r2']:.4f}, MAE=${results['linear']['mae']:.4f}")

    # ── 5B: Multivariate model ──
    # Features: bs_price, abs_moneyness, dte_days, iv (market), bs_iv_estimate
    features = ["bs_price", "abs_moneyness", "dte_days", "bs_iv_estimate"]
    X_multi = comp[features].values

    lr_multi = LinearRegression()
    lr_multi.fit(X_multi, y)
    y_pred_multi = lr_multi.predict(X_multi)

    coef_dict = {f: float(c) for f, c in zip(features, lr_multi.coef_)}
    results["multivariate"] = {
        "coefficients": coef_dict,
        "intercept": float(lr_multi.intercept_),
        "r2": float(r2_score(y, y_pred_multi)),
        "mae": float(mean_absolute_error(y, y_pred_multi)),
        "features": features,
    }
    fprint(f"\n  Multivariate: market_mid = f({', '.join(features)})")
    fprint(f"    Coefficients: {coef_dict}")
    fprint(f"    Intercept: {lr_multi.intercept_:.4f}")
    fprint(f"    R2={results['multivariate']['r2']:.4f}, MAE=${results['multivariate']['mae']:.4f}")

    # ── 5C: What if we just use market IV directly? ──
    # Recompute BS prices using market IV instead of ATR-estimated IV
    bs_with_market_iv = []
    for _, row in comp.iterrows():
        S = row["underlying_price"]
        K = row["strike"]
        T = row["dte_days"] / 365.0
        sigma = row["iv"]  # Use market IV
        if row["option_type"] == "call":
            p = bs_call_price(S, K, T, RISK_FREE_RATE, sigma)
        else:
            p = bs_put_price(S, K, T, RISK_FREE_RATE, sigma)
        bs_with_market_iv.append(p)

    comp_copy = comp.copy()
    comp_copy["bs_market_iv"] = bs_with_market_iv
    ratio_miv = comp_copy["mid"] / comp_copy["bs_market_iv"].clip(lower=0.01)
    results["market_iv_sanity"] = {
        "mean_ratio": float(ratio_miv.mean()),
        "median_ratio": float(ratio_miv.median()),
        "r2_vs_mid": float(
            r2_score(comp_copy["mid"], comp_copy["bs_market_iv"])
            if comp_copy["bs_market_iv"].std() > 0 else 0.0
        ),
    }
    fprint(f"\n  Market-IV sanity check (BS with market IV vs mid):")
    fprint(f"    Median ratio: {results['market_iv_sanity']['median_ratio']:.4f}")
    fprint(f"    R2: {results['market_iv_sanity']['r2_vs_mid']:.4f}")
    fprint(f"    (If ratio~1.0 and R2~1.0, the IV estimation is the root cause)")

    return results


# =====================================================================
# STEP 6: Compute optimal haircut
# =====================================================================

def compute_optimal_haircut(comp: pd.DataFrame) -> dict:
    """
    Find the haircut value that minimizes pricing error for spreads.

    For spread trades, the haircut inflates entry cost and deflates exit value.
    We want haircut such that: bs_price * (1 + haircut) ~ market_mid (for buying)
    and: bs_price * (1 - haircut) ~ market_bid (for selling)
    """
    fprint("\n" + "=" * 70)
    fprint("OPTIMAL HAIRCUT COMPUTATION")
    fprint("=" * 70)

    # Method 1: Simple ratio-based
    # For entry (buying): you pay ask. haircut should make bs_price*(1+h) ~ ask
    # For exit (selling): you receive bid. haircut should make bs_price*(1-h) ~ bid
    # But we're comparing to mid, so mid = (bid+ask)/2
    # The "haircut" in our pricer inflates ENTRY by (1+h) and deflates EXIT by (1-h)
    # So for mid comparison: ratio = mid/bs = average of entry_ratio and exit_ratio

    # Compute what haircut would make BS match market mid exactly
    # If market_mid = bs_price * (1 + h_optimal), then h_optimal = mid/bs - 1
    h_implied = comp["ratio"] - 1.0  # per-option implied haircut

    results = {}

    # Method 1: Median implied haircut (robust to outliers)
    h_median = float(h_implied.median())
    results["median_implied_haircut"] = h_median

    # Focus on OTM options only (that's what we actually trade in spreads)
    otm_mask = comp["abs_moneyness"] >= 0.01  # exclude deep ITM
    comp_otm = comp[otm_mask] if otm_mask.sum() > 100 else comp

    # Method 2: Minimize MAE across haircuts (on OTM options)
    haircuts_to_test = np.arange(0.0, 3.0, 0.01)
    maes = []
    for h in haircuts_to_test:
        adjusted = comp_otm["bs_price"] * (1 + h)
        mae = float((adjusted - comp_otm["mid"]).abs().mean())
        maes.append(mae)

    best_idx = np.argmin(maes)
    h_optimal_mae = float(haircuts_to_test[best_idx])
    results["optimal_haircut_mae"] = h_optimal_mae
    results["min_mae"] = float(maes[best_idx])

    # Method 3: Minimize MAPE (most meaningful for option pricing)
    mapes = []
    for h in haircuts_to_test:
        adjusted = comp_otm["bs_price"] * (1 + h)
        mape = float(((adjusted - comp_otm["mid"]).abs() / comp_otm["mid"]).mean() * 100)
        mapes.append(mape)

    best_idx_mape = np.argmin(mapes)
    h_optimal_mape = float(haircuts_to_test[best_idx_mape])
    results["optimal_haircut_mape"] = h_optimal_mape
    results["min_mape"] = float(mapes[best_idx_mape])

    # Method 4: Haircut by moneyness bucket (different haircuts for different regions)
    bucket_haircuts = {}
    for bucket in comp["moneyness_bucket"].unique():
        sub = comp[comp["moneyness_bucket"] == bucket]
        if len(sub) < 10:
            continue
        bucket_h = float((sub["ratio"] - 1.0).median())
        bucket_haircuts[bucket] = round(bucket_h, 4)
    results["haircut_by_moneyness"] = bucket_haircuts

    # Method 5: Haircut by DTE
    bucket_haircuts_dte = {}
    for bucket in comp["dte_bucket"].unique():
        sub = comp[comp["dte_bucket"] == bucket]
        if len(sub) < 10:
            continue
        bucket_h = float((sub["ratio"] - 1.0).median())
        bucket_haircuts_dte[bucket] = round(bucket_h, 4)
    results["haircut_by_dte"] = bucket_haircuts_dte

    fprint(f"  Current haircut:           15.0%")
    fprint(f"  Median implied haircut:    {h_median*100:.1f}%")
    fprint(f"  Optimal haircut (MAE):     {h_optimal_mae*100:.1f}% (MAE=${results['min_mae']:.4f})")
    fprint(f"  Optimal haircut (MAPE):    {h_optimal_mape*100:.1f}% (MAPE={results['min_mape']:.1f}%)")
    fprint(f"\n  Haircut by moneyness:")
    for k, v in sorted(bucket_haircuts.items()):
        fprint(f"    {k:15s}: {v*100:+.1f}%")
    fprint(f"\n  Haircut by DTE:")
    for k, v in sorted(bucket_haircuts_dte.items()):
        fprint(f"    {k:10s}: {v*100:+.1f}%")

    # Recommendation: use MAPE-optimal for spread trades, but sanity-check
    # against median implied haircut. The MAPE-optimal is the most relevant
    # because it minimizes percentage pricing error.
    recommended = round(h_optimal_mape, 2)
    if recommended < 0.05:
        recommended = 0.05  # floor
    # Cross-check: if median implied haircut is much higher, note it
    if h_median > recommended * 2:
        fprint(f"\n  WARNING: Median implied haircut ({h_median*100:.0f}%) is much higher than")
        fprint(f"  MAPE-optimal ({recommended*100:.0f}%). This suggests heavy-tailed distribution.")
        fprint(f"  For conservative pricing, consider using median ({h_median*100:.0f}%).")
        # Use median for conservative recommendation
        recommended_conservative = round(h_median, 2)
        results["recommended_haircut_conservative"] = recommended_conservative

    results["recommended_haircut"] = recommended
    fprint(f"\n  RECOMMENDED HAIRCUT (MAPE-optimal): {recommended*100:.0f}%")
    fprint(f"  (vs current 15% — {'increase' if recommended > 0.15 else 'decrease'} "
           f"by {abs(recommended - 0.15)*100:.0f}pp)")

    return results


# =====================================================================
# STEP 7: Simplified V10 backtest with calibrated vs current pricing
# =====================================================================

def simplified_v10_backtest(comp: pd.DataFrame, haircut_current: float = 0.15,
                            haircut_calibrated: float = 0.50) -> dict:
    """
    Simplified V10-style backtest to show Sharpe impact of calibrated pricing.

    Simulates bull call spread trades using ATM + 3% OTM structure on
    available tickers, comparing current vs calibrated haircut.
    """
    fprint("\n" + "=" * 70)
    fprint("SIMPLIFIED V10 BACKTEST — CURRENT vs CALIBRATED PRICING")
    fprint("=" * 70)

    # Use chain data to simulate realistic spread trades
    # For each snapshot date + ticker, simulate entering a bull call spread
    results = {}

    for label, haircut in [("current_15pct", haircut_current),
                           ("calibrated", haircut_calibrated)]:
        trades = []

        # Group by snapshot_date and ticker
        for (date, ticker), grp in comp.groupby(["snapshot_date", "ticker"]):
            # Get ATM calls
            calls = grp[grp["option_type"] == "call"].copy()
            if calls.empty:
                continue

            S = calls["underlying_price"].iloc[0]

            # Find DTE in 20-35 range (typical V10 trade)
            suitable = calls[(calls["dte_days"] >= 20) & (calls["dte_days"] <= 45)]
            if suitable.empty:
                continue

            # Pick closest to 28 DTE
            target_dte = 28
            suitable["dte_diff"] = (suitable["dte_days"] - target_dte).abs()
            best_dte = suitable.loc[suitable["dte_diff"].idxmin(), "dte_days"]
            dte_calls = suitable[suitable["dte_days"] == best_dte]

            # Find ATM strike (closest to spot)
            dte_calls = dte_calls.copy()
            dte_calls["strike_diff"] = (dte_calls["strike"] - S).abs()
            atm_row = dte_calls.loc[dte_calls["strike_diff"].idxmin()]
            K1 = atm_row["strike"]

            # Find ~3% OTM strike
            target_K2 = S * 1.03
            otm_candidates = dte_calls[dte_calls["strike"] > K1]
            if otm_candidates.empty:
                continue
            otm_candidates = otm_candidates.copy()
            otm_candidates["k2_diff"] = (otm_candidates["strike"] - target_K2).abs()
            K2_row = otm_candidates.loc[otm_candidates["k2_diff"].idxmin()]
            K2 = K2_row["strike"]

            if K2 <= K1:
                continue

            # Real market spread cost
            real_entry = float(atm_row["mid"]) - float(K2_row["mid"])
            if real_entry <= 0:
                continue

            # BS spread cost with haircut
            bs_entry = float(atm_row["bs_price"]) - float(K2_row["bs_price"])
            if bs_entry <= 0:
                continue
            bs_entry_with_haircut = bs_entry * (1 + haircut)

            # Spread width
            width = K2 - K1

            # Simulate random outcome: uniform draw of final price
            # Use a simple model: 50% chance of +1% move, 50% chance of -1% move
            # This is simplified but shows the pricing impact
            np.random.seed(hash((date, ticker, label)) % (2**31))
            final_move_pct = np.random.normal(0.003, 0.04)  # slight upward bias
            S_final = S * (1 + final_move_pct)

            # Intrinsic at expiry
            intrinsic = max(S_final - K1, 0) - max(S_final - K2, 0)

            # PnL per share
            pnl_real = (intrinsic - real_entry) * 100 - 2.60  # real pricing
            pnl_bs = (intrinsic - bs_entry_with_haircut) * 100 - 2.60  # BS pricing

            trades.append({
                "date": date,
                "ticker": ticker,
                "S": S,
                "K1": K1,
                "K2": K2,
                "dte": int(best_dte),
                "real_entry": real_entry,
                "bs_entry": bs_entry_with_haircut,
                "entry_error_pct": (bs_entry_with_haircut - real_entry) / real_entry * 100,
                "intrinsic": intrinsic,
                "pnl_real": pnl_real,
                "pnl_bs": pnl_bs,
            })

        if not trades:
            fprint(f"  [{label}] No trades generated")
            continue

        trades_df = pd.DataFrame(trades)

        # Compute Sharpe-like metric
        pnl_real = trades_df["pnl_real"]
        pnl_bs = trades_df["pnl_bs"]
        entry_err = trades_df["entry_error_pct"]

        def sharpe(returns):
            if returns.std() == 0:
                return 0.0
            return float(returns.mean() / returns.std() * np.sqrt(252 / 7))  # weekly-ish

        results[label] = {
            "n_trades": len(trades_df),
            "mean_pnl_real": float(pnl_real.mean()),
            "mean_pnl_bs": float(pnl_bs.mean()),
            "sharpe_real": sharpe(pnl_real),
            "sharpe_bs": sharpe(pnl_bs),
            "wr_real": float((pnl_real > 0).mean()),
            "wr_bs": float((pnl_bs > 0).mean()),
            "mean_entry_error_pct": float(entry_err.mean()),
            "median_entry_error_pct": float(entry_err.median()),
            "haircut_used": haircut,
        }

        fprint(f"\n  [{label}] haircut={haircut*100:.0f}%")
        fprint(f"    Trades: {len(trades_df)}")
        fprint(f"    Mean PnL (real pricing):  ${pnl_real.mean():.2f}")
        fprint(f"    Mean PnL (BS pricing):    ${pnl_bs.mean():.2f}")
        fprint(f"    Sharpe (real):  {sharpe(pnl_real):.3f}")
        fprint(f"    Sharpe (BS):    {sharpe(pnl_bs):.3f}")
        fprint(f"    WR (real):      {(pnl_real > 0).mean()*100:.1f}%")
        fprint(f"    WR (BS):        {(pnl_bs > 0).mean()*100:.1f}%")
        fprint(f"    Entry error (BS vs real): mean={entry_err.mean():+.1f}%, "
               f"median={entry_err.median():+.1f}%")

    if "current_15pct" in results and "calibrated" in results:
        # Compare how BS pricing deviates from real pricing under each haircut
        # Use absolute Sharpe difference rather than ratio (avoids division issues)
        sharpe_real_curr = results["current_15pct"]["sharpe_real"]
        sharpe_bs_curr = results["current_15pct"]["sharpe_bs"]
        sharpe_real_cal = results["calibrated"]["sharpe_real"]
        sharpe_bs_cal = results["calibrated"]["sharpe_bs"]

        # Pricing error = how much BS Sharpe deviates from real Sharpe
        error_current = sharpe_bs_curr - sharpe_real_curr
        error_calibrated = sharpe_bs_cal - sharpe_real_cal

        fprint(f"\n  BS pricing error on Sharpe:")
        fprint(f"    Current (15%):    BS Sharpe {sharpe_bs_curr:.3f} vs real {sharpe_real_curr:.3f} "
               f"(error={error_current:+.3f})")
        fprint(f"    Calibrated ({results['calibrated']['haircut_used']*100:.0f}%): "
               f"BS Sharpe {sharpe_bs_cal:.3f} vs real {sharpe_real_cal:.3f} "
               f"(error={error_calibrated:+.3f})")
        results["sharpe_error_current"] = float(error_current)
        results["sharpe_error_calibrated"] = float(error_calibrated)

        # Mean entry cost error is the key pricing metric
        fprint(f"    Current entry cost error:    {results['current_15pct']['mean_entry_error_pct']:+.1f}%")
        fprint(f"    Calibrated entry cost error: {results['calibrated']['mean_entry_error_pct']:+.1f}%")

    return results


# =====================================================================
# STEP 8: MLflow logging
# =====================================================================

def log_to_mlflow(analysis_results: dict, calibration_results: dict,
                  haircut_results: dict, backtest_results: dict) -> None:
    """Log all results to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"calibration_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Overall ratio stats
            overall = analysis_results.get("overall_ratio", {})
            for k, v in overall.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"overall_{k}", v)

            # IV accuracy
            iv_acc = analysis_results.get("iv_accuracy", {})
            for k, v in iv_acc.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"iv_{k}", v)

            # Calibration model results
            lin = calibration_results.get("linear", {})
            mlflow.log_metric("linear_r2", lin.get("r2", 0))
            mlflow.log_metric("linear_mae", lin.get("mae", 0))
            mlflow.log_metric("linear_slope", lin.get("slope", 0))

            multi = calibration_results.get("multivariate", {})
            mlflow.log_metric("multi_r2", multi.get("r2", 0))
            mlflow.log_metric("multi_mae", multi.get("mae", 0))

            miv = calibration_results.get("market_iv_sanity", {})
            mlflow.log_metric("market_iv_ratio", miv.get("median_ratio", 0))
            mlflow.log_metric("market_iv_r2", miv.get("r2_vs_mid", 0))

            # Haircut results
            mlflow.log_metric("current_haircut", 0.15)
            mlflow.log_metric("optimal_haircut_mae", haircut_results.get("optimal_haircut_mae", 0))
            mlflow.log_metric("optimal_haircut_mape", haircut_results.get("optimal_haircut_mape", 0))
            mlflow.log_metric("median_implied_haircut", haircut_results.get("median_implied_haircut", 0))
            mlflow.log_metric("recommended_haircut", haircut_results.get("recommended_haircut", 0))

            # Backtest results
            for label in ["current_15pct", "calibrated"]:
                bt = backtest_results.get(label, {})
                for k, v in bt.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"bt_{label}_{k}", v)

            if "sharpe_error_current" in backtest_results:
                mlflow.log_metric("sharpe_error_current", backtest_results["sharpe_error_current"])
                mlflow.log_metric("sharpe_error_calibrated", backtest_results["sharpe_error_calibrated"])

            # Log params
            mlflow.log_param("chain_dates_used", len(set()))  # updated below
            mlflow.log_param("script", "bs_pricing_calibration_v1.py")

        fprint("  MLflow logging complete")
    except Exception as e:
        fprint(f"  MLflow logging failed (non-fatal): {e}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    fprint("=" * 70)
    fprint("BS PRICING CALIBRATION v1")
    fprint("Calibrating Black-Scholes against real market option chains")
    fprint("=" * 70)

    # Step 1: Load chain data
    chains = load_all_chains()
    if chains.empty:
        fprint("\nNo chain data found in %s — cannot calibrate." % CHAINS_DIR)
        fprint("To collect chain data, run: python scripts/growth_research/daily_options_collector.py")
        return

    # Step 2: Fetch ATR data
    tickers = chains["ticker"].unique().tolist()
    fprint(f"\nFetching ATR data for {len(tickers)} tickers...")
    atr_map = fetch_atr_data(tickers)

    # Get approximate VIX level (use yfinance)
    vix_level = 20.0
    try:
        import yfinance as yf
        vix_data = yf.download("^VIX", period="5d", interval="1d", progress=False)
        if not vix_data.empty:
            vix_level = float(vix_data["Close"].iloc[-1])
            # Handle multi-level columns
            if hasattr(vix_level, '__iter__'):
                vix_level = float(list(vix_level)[0]) if len(list(vix_level)) > 0 else 20.0
            fprint(f"  Current VIX: {vix_level:.1f}")
    except Exception as e:
        fprint(f"  Could not fetch VIX ({e}), using default {vix_level}")

    # Step 3: Compute BS vs market comparisons
    fprint("\nComputing BS vs market comparisons...")
    comp = compute_comparisons(chains, atr_map, vix_level)
    if comp.empty or len(comp) < 50:
        fprint("Insufficient comparison data after filtering. Cannot calibrate.")
        return

    # Step 4: Analyze pricing gap
    analysis_results = analyze_pricing_gap(comp)

    # Step 5: Fit calibration models
    calibration_results = fit_calibration_models(comp)

    # Step 6: Compute optimal haircut
    haircut_results = compute_optimal_haircut(comp)
    recommended_haircut = haircut_results["recommended_haircut"]

    # Step 7: Simplified V10 backtest
    backtest_results = simplified_v10_backtest(comp,
                                               haircut_current=0.15,
                                               haircut_calibrated=recommended_haircut)

    # Step 8: Log to MLflow
    fprint("\nLogging to MLflow...")
    log_to_mlflow(analysis_results, calibration_results, haircut_results, backtest_results)

    # Step 9: Save results JSON
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "chain_snapshot_dates": sorted(chains["snapshot_date"].unique().tolist()),
        "n_options_compared": len(comp),
        "n_tickers": int(comp["ticker"].nunique()),
        "vix_level_used": vix_level,
        "analysis": analysis_results,
        "calibration_models": calibration_results,
        "optimal_haircut": haircut_results,
        "v10_backtest": backtest_results,
    }

    output_path = OUTPUT_DIR / "calibration_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {output_path}")

    # Save comparison dataset for further analysis
    comp_path = OUTPUT_DIR / "bs_vs_market_comparisons.parquet"
    comp.to_parquet(comp_path, index=False)
    fprint(f"Comparison dataset saved to {comp_path}")

    # ── Final Summary ──
    fprint("\n" + "=" * 70)
    fprint("CALIBRATION SUMMARY")
    fprint("=" * 70)
    overall = analysis_results["overall_ratio"]
    fprint(f"  Options compared:        {overall['count']:,}")
    fprint(f"  BS underprices by:       {(overall['median'] - 1)*100:.1f}% (median)")
    fprint(f"  Current haircut:         15%")
    fprint(f"  Recommended haircut:     {recommended_haircut*100:.0f}%")
    fprint(f"  IV estimation corr:      {analysis_results['iv_accuracy']['correlation']:.3f}")

    lin = calibration_results["linear"]
    fprint(f"\n  Best simple model:       {lin['formula']}")
    fprint(f"    R2={lin['r2']:.4f}")

    multi = calibration_results["multivariate"]
    fprint(f"  Multivariate model R2:   {multi['r2']:.4f}")

    miv = calibration_results["market_iv_sanity"]
    fprint(f"\n  Root cause diagnosis:")
    if miv["r2_vs_mid"] > 0.95:
        fprint(f"    Using market IV gives R2={miv['r2_vs_mid']:.4f} — ")
        fprint(f"    IV ESTIMATION IS THE ROOT CAUSE of pricing error.")
        fprint(f"    Recommendation: Use historical IV percentiles instead of ATR-based estimation.")
    else:
        fprint(f"    Even with market IV, R2={miv['r2_vs_mid']:.4f}")
        fprint(f"    Pricing error has MULTIPLE causes beyond IV estimation.")

    if "sharpe_error_current" in backtest_results:
        fprint(f"\n  V10 BS Sharpe error (current 15%):    {backtest_results['sharpe_error_current']:+.3f}")
        fprint(f"  V10 BS Sharpe error (calibrated):      {backtest_results['sharpe_error_calibrated']:+.3f}")

    fprint("\n" + "=" * 70)
    fprint("DONE")
    fprint("=" * 70)


if __name__ == "__main__":
    main()
