#!/usr/bin/env python3
"""
V10 Agentic Signal Generator
=============================
Translates V10 LGBM sector rankings into single-leg options trades
for the Robinhood agentic account (Level 2 options: long calls/puts only).

Run daily at 4:30 PM ET. Outputs signals to state/agentic_v10_signals.json.

Account constraints:
  - ~$645 equity, Level 2 options (no spreads)
  - Max 3 concurrent positions
  - Max $150/trade, max 50% equity deployed
  - Stop loss at -50%, profit target at +100%

Usage:
  python3 engines/v10_agentic_signal_generator.py
  python3 engines/v10_agentic_signal_generator.py --dry-run
"""

import argparse
import json
import logging
import math
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path("/home/jupiter/Lvl3Quant")
SIGNAL_FILE = BASE / "state" / "agentic_v10_signals.json"
STATE_FILE = BASE / "state" / "agentic_v10_state.json"
POSITIONS_FILE = BASE / "state" / "agentic_positions.json"
LOG_DIR = BASE / "engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "v10_agentic_signals.log"

# ── Logging ────────────────────────────────────────────────────────────
logger = logging.getLogger("v10_agentic")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_FILE, mode="a")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(message)s"))
sh.setLevel(logging.INFO)
logger.addHandler(sh)

# ── Universe ───────────────────────────────────────────────────────────
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
SECTOR_NAMES = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Health Care", "XLY": "Consumer Disc", "XLP": "Consumer Staples",
    "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Svcs",
}
EXTRA_TICKERS = ["SPY", "^VIX"]

# ── Account / Position Rules ──────────────────────────────────────────
MAX_CONCURRENT_POSITIONS = 3
MAX_COST_PER_TRADE = 150.0      # dollars (option premium * 100)
MAX_EQUITY_DEPLOYED_PCT = 0.50
STOP_LOSS_PCT = -0.50
PROFIT_TARGET_PCT = 1.00
DEFAULT_EQUITY = 645.0

# ── Signal Parameters ─────────────────────────────────────────────────
TOP_K = 3               # bullish sectors
BOT_K = 3               # bearish sectors
OTM_PCT = 0.05          # 5% out of the money
TARGET_DTE = 30          # ~30 days to expiration
VIX_MAX = 30.0           # skip all trades if VIX > 30
VIX_MIN_FOR_PUTS = 15.0  # skip puts if VIX < 15

# ── LGBM Walk-Forward ─────────────────────────────────────────────────
WF_TRAIN_DAYS = 500      # ~2 years sliding window (trading days)
LGBM_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 10,
    "random_state": 42,
}

# ── V10 Legacy Features (17) ──────────────────────────────────────────
FEATURE_NAMES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "cross_sector_dispersion",
]
assert len(FEATURE_NAMES) == 17


# ═══════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════

def download_data(lookback_days=800):
    """Download price data for sectors + SPY + VIX via yfinance."""
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    logger.info("Downloading %d tickers from %s...", len(all_tickers), start)

    raw = yf.download(all_tickers, start=start, auto_adjust=True, progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.rename(columns={"^VIX": "VIX"})
    close = close.ffill().dropna(thresh=len(close.columns) - 2)

    logger.info("Data: %d days, %s to %s",
                len(close), close.index[0].date(), close.index[-1].date())
    return close


# ═══════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION (V10 LEGACY 17)
# ═══════════════════════════════════════════════════════════════════════

def compute_legacy_features(px, spy_px):
    """Compute the 15 single-asset legacy features for one sector."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()
    f["sharpe_63d"] = float(
        rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252)
    ) if len(rets) > 63 else 0.0

    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float(
        (monthly.iloc[-12:] > 0).mean()
    ) if len(monthly) >= 12 else 0.5

    dr = rets.iloc[-63:][rets.iloc[-63:] < 0]
    f["sortino_63d"] = float(
        rets.iloc[-63:].mean() / (dr.std() + 1e-10) * np.sqrt(252)
    ) if len(dr) > 3 else 0.0

    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f["calmar_1y"] = cagr / (abs(mdd) + 1e-10)

    up_days = rets[rets > 0]
    f["up_capture"] = float(
        up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)
    ) if len(up_days) > 10 else 1.0

    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
        f["trend_slope_63d"] = slope * 252
    else:
        f["trend_r2_63d"] = 0.0
        f["trend_slope_63d"] = 0.0

    return f


def compute_cross_asset_features(sector_ticker, close_df):
    """Compute the 2 cross-asset features: beta and dispersion."""
    f = {}
    spy = close_df["SPY"].dropna()
    sec = close_df[sector_ticker].dropna() if sector_ticker in close_df.columns else None

    if spy is not None and sec is not None and len(spy) > 63 and len(sec) > 63:
        spy_ret = spy.pct_change().dropna()
        sec_ret = sec.pct_change().dropna()
        common = spy_ret.index.intersection(sec_ret.index)
        if len(common) > 63:
            sr = sec_ret.loc[common].iloc[-63:]
            mr = spy_ret.loc[common].iloc[-63:]
            cov = np.cov(sr.values, mr.values)
            f["sector_spy_beta_63d"] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0

    sector_cols = [c for c in SECTORS if c in close_df.columns]
    if len(sector_cols) > 3:
        sector_rets = close_df[sector_cols].pct_change()
        daily_disp = sector_rets.std(axis=1)
        if len(daily_disp) > 21:
            f["cross_sector_dispersion"] = float(daily_disp.rolling(21).mean().iloc[-1])
        else:
            f["cross_sector_dispersion"] = 0.01
    else:
        f["cross_sector_dispersion"] = 0.01

    return f


def build_current_features(close_df):
    """Build feature vectors for all sectors at the latest date."""
    spy_px = close_df["SPY"].dropna()
    records = []
    for tk in SECTORS:
        if tk not in close_df.columns:
            continue
        px = close_df[tk].dropna()
        legacy = compute_legacy_features(px, spy_px)
        if legacy is None:
            logger.warning("Skipping %s: insufficient data for features", tk)
            continue
        cross = compute_cross_asset_features(tk, close_df)
        row = {**legacy, **cross, "ticker": tk}
        records.append(row)
    return pd.DataFrame(records)


def build_historical_features(close_df, lookback_weeks=None):
    """Build weekly feature records for LGBM training.

    For each Friday in the history (after warmup), compute features for all
    sectors and label with the forward 21-day return rank (top half = 1).
    """
    spy_px = close_df["SPY"].dropna()
    sector_cols = [c for c in SECTORS if c in close_df.columns]

    # Weekly Fridays as observation dates
    fridays = pd.DatetimeIndex(
        close_df.index.to_series().resample("W-FRI").last().dropna().values
    )

    records = []
    for dt in fridays:
        idx = close_df.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        # Need forward return for label
        fwd_idx = min(idx + 21, len(close_df) - 1)
        if fwd_idx <= idx:
            continue

        sector_fwd_rets = {}
        sector_features = {}
        for tk in sector_cols:
            px = close_df[tk].iloc[:idx + 1].dropna()
            legacy = compute_legacy_features(px, spy_px.iloc[:idx + 1])
            if legacy is None:
                continue
            # Cross-asset features using data up to this point
            slice_df = close_df.iloc[:idx + 1]
            cross = compute_cross_asset_features(tk, slice_df)
            sector_features[tk] = {**legacy, **cross}
            # Forward return for label
            p0 = close_df[tk].iloc[idx]
            p1 = close_df[tk].iloc[fwd_idx]
            if p0 > 0:
                sector_fwd_rets[tk] = float(p1 / p0 - 1)

        if len(sector_fwd_rets) < 6:
            continue

        # Rank forward returns: top half = 1, bottom half = 0
        sorted_tks = sorted(sector_fwd_rets.keys(), key=lambda t: sector_fwd_rets[t], reverse=True)
        half = len(sorted_tks) // 2
        labels = {t: (1 if i < half else 0) for i, t in enumerate(sorted_tks)}

        for tk in sector_features:
            if tk in labels:
                row = sector_features[tk].copy()
                row["ticker"] = tk
                row["date"] = dt
                row["label"] = labels[tk]
                row["fwd_ret_21d"] = sector_fwd_rets[tk]
                records.append(row)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════
# LGBM RANKING (SLIDING WALK-FORWARD)
# ═══════════════════════════════════════════════════════════════════════

def rank_sectors_lgbm(close_df):
    """Train LGBM on sliding window and rank current sectors."""
    from lightgbm import LGBMClassifier

    logger.info("Building historical features for LGBM training...")
    hist_df = build_historical_features(close_df)
    if hist_df.empty or len(hist_df) < 100:
        logger.error("Insufficient historical data for LGBM (%d records)", len(hist_df))
        return None

    logger.info("Historical records: %d", len(hist_df))

    # Use latest WF_TRAIN_DAYS worth of observations for training
    unique_dates = sorted(hist_df["date"].unique())
    if len(unique_dates) > WF_TRAIN_DAYS // 5:  # weekly observations
        train_start = unique_dates[-(WF_TRAIN_DAYS // 5):]
        hist_df = hist_df[hist_df["date"].isin(train_start)]

    X_train = hist_df[FEATURE_NAMES].values
    y_train = hist_df["label"].values

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info("Training LGBM on %d samples...", len(X_train))
    model = LGBMClassifier(**LGBM_PARAMS, verbose=-1)
    model.fit(X_train, y_train)

    # Score current sectors
    current_features = build_current_features(close_df)
    if current_features.empty:
        logger.error("No current features computed")
        return None

    X_current = current_features[FEATURE_NAMES].values
    X_current = np.nan_to_num(X_current, nan=0.0, posinf=0.0, neginf=0.0)

    probs = model.predict_proba(X_current)[:, 1]  # probability of being top-half
    current_features["lgbm_score"] = probs
    current_features = current_features.sort_values("lgbm_score", ascending=False)

    # Rank percentile (1.0 = most bullish)
    n = len(current_features)
    current_features["rank_pct"] = [(n - i) / n for i in range(n)]

    logger.info("LGBM Rankings:")
    for _, row in current_features.iterrows():
        logger.info("  %s (%s): score=%.3f rank_pct=%.2f",
                     row["ticker"], SECTOR_NAMES.get(row["ticker"], ""),
                     row["lgbm_score"], row["rank_pct"])

    # Feature importances
    imp = dict(zip(FEATURE_NAMES, model.feature_importances_))
    top_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("Top features: %s", ", ".join(f"{k}={v}" for k, v in top_feats))

    return current_features


# ═══════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES FOR STRIKE SELECTION
# ═══════════════════════════════════════════════════════════════════════

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def estimate_option_premium(spot, direction, otm_pct, dte, vol_annual, risk_free=0.05):
    """Estimate option premium using Black-Scholes.

    Returns (strike, premium_per_share, premium_per_contract).
    """
    T = dte / 365.0
    if direction == "long_call":
        K = round(spot * (1 + otm_pct), 2)
        premium = bs_call_price(spot, K, T, risk_free, vol_annual)
    elif direction == "long_put":
        K = round(spot * (1 - otm_pct), 2)
        premium = bs_put_price(spot, K, T, risk_free, vol_annual)
    else:
        return None, None, None

    premium_contract = premium * 100  # 100 shares per contract
    return K, premium, premium_contract


# ═══════════════════════════════════════════════════════════════════════
# POSITION STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

def load_state():
    """Load existing state (open positions tracked by this generator)."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass
    return {"open_positions": [], "signal_history": [], "last_run": None}


def load_agentic_positions():
    """Load positions from the main agentic position tracker."""
    if POSITIONS_FILE.exists():
        try:
            with open(POSITIONS_FILE) as f:
                data = json.load(f)
            return data.get("positions", {})
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def get_open_sector_tickers(state):
    """Get set of sector tickers that already have open positions."""
    held = set()
    # From our own state
    for pos in state.get("open_positions", []):
        held.add(pos.get("ticker", ""))
    # From main agentic tracker
    agentic_pos = load_agentic_positions()
    for sym, pos_data in agentic_pos.items():
        # Check if it's a sector ETF
        base = sym.split()[0] if isinstance(sym, str) else sym
        if base in SECTORS:
            held.add(base)
    return held


def save_state(state):
    """Save state file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_signals(close_df, rankings_df, state, equity=DEFAULT_EQUITY):
    """Generate trade recommendations from LGBM rankings."""
    now = datetime.now()
    vix = float(close_df["VIX"].dropna().iloc[-1])
    spy_price = float(close_df["SPY"].dropna().iloc[-1])

    # Market regime
    if vix > 25:
        regime = "high_vol"
    elif vix < 15:
        regime = "low_vol"
    else:
        regime = "normal"

    signals = {
        "timestamp": now.isoformat(),
        "vix_level": round(vix, 2),
        "spy_price": round(spy_price, 2),
        "market_regime": regime,
        "recommendations": [],
        "skipped": [],
        "summary": "",
    }

    # VIX filter: skip everything if too high
    if vix > VIX_MAX:
        msg = f"VIX at {vix:.1f} (>{VIX_MAX}) -- all trades skipped, premiums too expensive"
        signals["summary"] = msg
        logger.info(msg)
        return signals

    # Check position capacity
    held_tickers = get_open_sector_tickers(state)
    open_count = len(held_tickers)
    max_deployable = equity * MAX_EQUITY_DEPLOYED_PCT

    # Estimate currently deployed capital from state
    deployed = sum(
        pos.get("cost_basis", 0) for pos in state.get("open_positions", [])
    )
    remaining_capacity = max_deployable - deployed
    slots_available = MAX_CONCURRENT_POSITIONS - open_count

    if slots_available <= 0:
        msg = f"Max positions reached ({open_count}/{MAX_CONCURRENT_POSITIONS}) -- no new trades"
        signals["summary"] = msg
        logger.info(msg)
        return signals

    logger.info("Capacity: %d slots, $%.0f deployable remaining", slots_available, remaining_capacity)

    # Split rankings into bullish/bearish
    n = len(rankings_df)
    bullish = rankings_df.head(TOP_K)
    bearish = rankings_df.tail(BOT_K)

    recommendations = []

    # Process bullish sectors (long calls)
    for _, row in bullish.iterrows():
        tk = row["ticker"]
        if tk in held_tickers:
            signals["skipped"].append({"ticker": tk, "reason": "already_held"})
            continue
        if len(recommendations) >= slots_available:
            break
        if remaining_capacity < 30:
            signals["skipped"].append({"ticker": tk, "reason": "insufficient_capital"})
            continue

        spot = float(close_df[tk].dropna().iloc[-1])
        # Annualized vol from 21d returns
        rets = close_df[tk].pct_change().dropna()
        vol_21d = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.25

        strike, prem_share, prem_contract = estimate_option_premium(
            spot, "long_call", OTM_PCT, TARGET_DTE, vol_21d
        )

        if prem_contract is None or prem_contract <= 0:
            signals["skipped"].append({"ticker": tk, "reason": "pricing_failed"})
            continue

        if prem_contract > MAX_COST_PER_TRADE:
            signals["skipped"].append({
                "ticker": tk, "reason": f"too_expensive (${prem_contract:.0f})"
            })
            continue

        if prem_contract > remaining_capacity:
            signals["skipped"].append({"ticker": tk, "reason": "insufficient_capital"})
            continue

        # Suggested expiry (~30 DTE from today)
        expiry = (now + timedelta(days=TARGET_DTE)).strftime("%Y-%m-%d")

        rec = {
            "ticker": tk,
            "sector": SECTOR_NAMES.get(tk, tk),
            "direction": "long_call",
            "spot_price": round(spot, 2),
            "suggested_strike": strike,
            "suggested_expiry": expiry,
            "estimated_premium": round(prem_contract, 2),
            "max_cost": MAX_COST_PER_TRADE,
            "implied_vol": round(vol_21d, 3),
            "confidence_score": round(float(row["rank_pct"]), 3),
            "lgbm_score": round(float(row["lgbm_score"]), 3),
            "reasoning": (
                f"{SECTOR_NAMES.get(tk, tk)} ranked #{int((1 - row['rank_pct']) * n) + 1}/{n} "
                f"by LGBM (score {row['lgbm_score']:.3f}). "
                f"21d momentum {row.get('ret_21d', 0)*100:.1f}%, "
                f"63d Sharpe {row.get('sharpe_63d', 0):.2f}. "
                f"Call ~{OTM_PCT*100:.0f}% OTM at ${strike}, est. ${prem_contract:.0f}/contract."
            ),
            "stop_loss_pct": STOP_LOSS_PCT,
            "profit_target_pct": PROFIT_TARGET_PCT,
        }
        recommendations.append(rec)
        remaining_capacity -= prem_contract

    # Process bearish sectors (long puts) -- skip if VIX too low
    if vix >= VIX_MIN_FOR_PUTS:
        for _, row in bearish.iterrows():
            tk = row["ticker"]
            if tk in held_tickers:
                signals["skipped"].append({"ticker": tk, "reason": "already_held"})
                continue
            if len(recommendations) >= slots_available:
                break
            if remaining_capacity < 30:
                signals["skipped"].append({"ticker": tk, "reason": "insufficient_capital"})
                continue

            spot = float(close_df[tk].dropna().iloc[-1])
            rets = close_df[tk].pct_change().dropna()
            vol_21d = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.25

            strike, prem_share, prem_contract = estimate_option_premium(
                spot, "long_put", OTM_PCT, TARGET_DTE, vol_21d
            )

            if prem_contract is None or prem_contract <= 0:
                signals["skipped"].append({"ticker": tk, "reason": "pricing_failed"})
                continue

            if prem_contract > MAX_COST_PER_TRADE:
                signals["skipped"].append({
                    "ticker": tk, "reason": f"too_expensive (${prem_contract:.0f})"
                })
                continue

            if prem_contract > remaining_capacity:
                signals["skipped"].append({"ticker": tk, "reason": "insufficient_capital"})
                continue

            expiry = (now + timedelta(days=TARGET_DTE)).strftime("%Y-%m-%d")

            rec = {
                "ticker": tk,
                "sector": SECTOR_NAMES.get(tk, tk),
                "direction": "long_put",
                "spot_price": round(spot, 2),
                "suggested_strike": strike,
                "suggested_expiry": expiry,
                "estimated_premium": round(prem_contract, 2),
                "max_cost": MAX_COST_PER_TRADE,
                "implied_vol": round(vol_21d, 3),
                "confidence_score": round(float(1 - row["rank_pct"]), 3),  # invert for bearish
                "lgbm_score": round(float(row["lgbm_score"]), 3),
                "reasoning": (
                    f"{SECTOR_NAMES.get(tk, tk)} ranked #{int((1 - row['rank_pct']) * n) + 1}/{n} "
                    f"by LGBM (score {row['lgbm_score']:.3f}) -- bottom sector. "
                    f"21d momentum {row.get('ret_21d', 0)*100:.1f}%, "
                    f"63d Sharpe {row.get('sharpe_63d', 0):.2f}. "
                    f"Put ~{OTM_PCT*100:.0f}% OTM at ${strike}, est. ${prem_contract:.0f}/contract."
                ),
                "stop_loss_pct": STOP_LOSS_PCT,
                "profit_target_pct": PROFIT_TARGET_PCT,
            }
            recommendations.append(rec)
            remaining_capacity -= prem_contract
    else:
        logger.info("VIX at %.1f (<%s) -- skipping put recommendations", vix, VIX_MIN_FOR_PUTS)
        signals["skipped"].append({
            "ticker": "ALL_PUTS", "reason": f"VIX {vix:.1f} < {VIX_MIN_FOR_PUTS} (calm market)"
        })

    signals["recommendations"] = recommendations

    # Build summary
    if not recommendations:
        signals["summary"] = "No actionable trades today."
    else:
        calls = [r for r in recommendations if r["direction"] == "long_call"]
        puts = [r for r in recommendations if r["direction"] == "long_put"]
        parts = []
        if calls:
            tks = ", ".join(r["ticker"] for r in calls)
            parts.append(f"BULLISH: {tks} (long calls)")
        if puts:
            tks = ", ".join(r["ticker"] for r in puts)
            parts.append(f"BEARISH: {tks} (long puts)")
        total_cost = sum(r["estimated_premium"] for r in recommendations)
        signals["summary"] = (
            f"V10 Agentic Signals -- {len(recommendations)} trades | "
            + " | ".join(parts)
            + f" | Est. total cost: ${total_cost:.0f} | VIX: {vix:.1f} ({regime})"
        )

    return signals


# ═══════════════════════════════════════════════════════════════════════
# PLAIN-ENGLISH OUTPUT
# ═══════════════════════════════════════════════════════════════════════

def format_summary(signals):
    """Format signals as plain-English output for Discord."""
    lines = []
    lines.append("=== V10 Agentic Signal Generator ===")
    lines.append(f"Time: {signals['timestamp'][:19]}")
    lines.append(f"VIX: {signals['vix_level']} | SPY: ${signals['spy_price']} | Regime: {signals['market_regime']}")
    lines.append("")

    recs = signals.get("recommendations", [])
    if not recs:
        lines.append(signals.get("summary", "No signals."))
    else:
        lines.append(f"{len(recs)} Trade Recommendation(s):")
        lines.append("-" * 40)
        for i, r in enumerate(recs, 1):
            direction_label = "CALL" if r["direction"] == "long_call" else "PUT"
            lines.append(
                f"  {i}. {r['ticker']} ({r['sector']}) -- Long {direction_label}"
            )
            lines.append(
                f"     Strike ${r['suggested_strike']} exp {r['suggested_expiry']} "
                f"(~${r['estimated_premium']:.0f}/contract)"
            )
            lines.append(
                f"     Confidence: {r['confidence_score']:.0%} | "
                f"Stop: {r['stop_loss_pct']:.0%} | Target: +{r['profit_target_pct']:.0%}"
            )
            lines.append(f"     {r['reasoning']}")
            lines.append("")

    skipped = signals.get("skipped", [])
    if skipped:
        skip_summary = ", ".join(
            f"{s['ticker']}({s['reason']})" for s in skipped[:5]
        )
        lines.append(f"Skipped: {skip_summary}")

    lines.append("")
    lines.append("Position rules: max 3 open, max $150/trade, 50% equity cap")
    lines.append("Management: stop at -50%, target at +100%")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V10 Agentic Signal Generator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without saving signals to file")
    parser.add_argument("--equity", type=float, default=DEFAULT_EQUITY,
                        help="Current account equity (default: $645)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("V10 Agentic Signal Generator -- %s", datetime.now().isoformat())
    logger.info("Mode: %s | Equity: $%.0f", "DRY RUN" if args.dry_run else "LIVE", args.equity)
    logger.info("=" * 60)

    # Load state
    state = load_state()

    # Download data
    try:
        close_df = download_data()
    except Exception as e:
        logger.error("Data download failed: %s", e)
        print(f"ERROR: Data download failed -- {e}")
        sys.exit(1)

    # Check VIX is available
    if "VIX" not in close_df.columns:
        logger.error("VIX data missing from download")
        print("ERROR: VIX data not available")
        sys.exit(1)

    # Run LGBM ranking
    try:
        rankings = rank_sectors_lgbm(close_df)
    except Exception as e:
        logger.error("LGBM ranking failed: %s", e)
        print(f"ERROR: LGBM ranking failed -- {e}")
        sys.exit(1)

    if rankings is None or rankings.empty:
        logger.error("No rankings produced")
        print("ERROR: No sector rankings produced")
        sys.exit(1)

    # Generate signals
    signals = generate_signals(close_df, rankings, state, equity=args.equity)

    # Format and print summary
    summary = format_summary(signals)
    print(summary)

    # Save signal file
    if not args.dry_run:
        SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SIGNAL_FILE, "w") as f:
            json.dump(signals, f, indent=2, default=str)
        logger.info("Signals saved to %s", SIGNAL_FILE)

        # Update state
        state["last_run"] = datetime.now().isoformat()
        state["signal_history"].append({
            "timestamp": signals["timestamp"],
            "n_recommendations": len(signals["recommendations"]),
            "vix": signals["vix_level"],
            "regime": signals["market_regime"],
        })
        # Keep only last 90 days of history
        state["signal_history"] = state["signal_history"][-90:]
        save_state(state)
        logger.info("State updated")
    else:
        logger.info("DRY RUN -- signals not saved")

    logger.info("Done.")
    return signals


if __name__ == "__main__":
    main()
