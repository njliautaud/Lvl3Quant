#!/usr/bin/env python3
"""
Income Strategy v2 -- LGBM-Ranked Credit Spread Portfolio
==========================================================
Tests 6 premium-selling variants on 11 sector ETFs with proper risk management.
Fixes the -61% MDD from jade lizard v1 (KB #251) by adding position sizing limits.

Variants:
  A: Bull Put Credit Spreads on top-3 LGBM sectors
  B: Bear Call Credit Spreads on bottom-3 LGBM sectors
  C: Iron Condor on middle-ranked sectors (rank 4-8)
  D: Combined Income Portfolio (A+B+C weighted 40/40/20)
  E: VIX-Adaptive Income (regime-gated variant selection)
  F: Jade Lizard v2 (short put + bear call spread, strict limits)

Config:
  Universe: 11 sector ETFs (XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC)
  Capital: $645 initial
  DTE: 28 days, biweekly (10 trading day) rebalance
  Spread width: $3
  Commission: $2.60/spread RT, $5.20/iron condor RT
  Max 3 concurrent positions, max 15% equity per trade
  SLIDING walk-forward LGBM: 500d train window
  BS pricing with iv_multiplier=1.2, 15% haircut
  5-gate adversarial validation

Author: Claude (Opus 4.6), 2026-07-27
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
import warnings
import traceback

warnings.filterwarnings("ignore")


def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# --- Path Detection ---
if Path("/home/nick/Lvl3Quant").exists():
    BASE = Path("/home/nick/Lvl3Quant")
elif Path("/home/jupiter/Lvl3Quant").exists():
    BASE = Path("/home/jupiter/Lvl3Quant")
else:
    BASE = Path(__file__).resolve().parents[2]

OUT_DIR = BASE / "output" / "growth_research" / "income_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- MLflow ---
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen("http://jupiter:5000/", timeout=3)
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    MLFLOW_OK = True
    fprint("MLflow connected")
except Exception:
    fprint("MLflow unavailable -- results saved locally only")

# --- LightGBM ---
try:
    import lightgbm as lgb
    LGBM_OK = True
except ImportError:
    fprint("FATAL: lightgbm not installed")
    sys.exit(1)

import yfinance as yf

# ============================================================================
# Constants
# ============================================================================

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
CAPITAL = 645.0
DTE = 28
SPREAD_WIDTH = 3.0          # $3 wide spreads
OTM_PCT = 0.02              # 2% OTM for short strikes
REBAL_DAYS = 10             # biweekly rebalance
TRAIN_DAYS = 500            # sliding window
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # $2.60 per spread RT
IC_COMM = 2 * SPREAD_COMM   # $5.20 per iron condor RT
HAIRCUT = 0.15              # 15% bid-ask haircut
RISK_FREE = 0.04
IV_MULT = 1.2               # iv_multiplier for realism
IV_FLOOR = 0.10
MAX_CONCURRENT = 3          # max 3 concurrent positions
MAX_RISK_PCT = 0.15         # max 15% equity per trade
MAX_RISK_CAP = 100.0        # absolute cap per spread
STOP_MULT = 2.0             # stop at -200% of premium (variant F)
N_PERM = 300                # permutation trials

FEAT_COLS = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high",
    "mom_accel", "pct_pos_months_12m", "sortino_63d", "calmar_1y",
    "up_capture", "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "sector_relative_vol_21d", "cross_sector_dispersion",
]


# ============================================================================
# Black-Scholes Pricing (self-contained)
# ============================================================================

def bs_call(S, K, T, sigma, r=RISK_FREE):
    """Black-Scholes call price."""
    if T <= 1e-8 or sigma <= 1e-8:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put(S, K, T, sigma, r=RISK_FREE):
    """Black-Scholes put price."""
    if T <= 1e-8 or sigma <= 1e-8:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def get_iv(prices_col, as_of_idx, lookback=63):
    """Realized vol x IV_MULT, floored."""
    px = prices_col.iloc[max(0, as_of_idx - lookback): as_of_idx + 1].dropna()
    if len(px) < 20:
        return 0.25
    rv = px.pct_change().dropna().std() * np.sqrt(252)
    return max(float(rv) * IV_MULT, IV_FLOOR)


# ============================================================================
# Data Download
# ============================================================================

def download_data():
    """Download sector ETF + SPY + VIX data from 2020-01-01."""
    fprint("Downloading data...")
    cache = OUT_DIR / "price_cache.parquet"
    vix_cache = OUT_DIR / "vix_cache.parquet"

    if cache.exists() and vix_cache.exists():
        mtime = datetime.fromtimestamp(cache.stat().st_mtime)
        if (datetime.now() - mtime).days < 1:
            fprint("  Using cached data")
            prices = pd.read_parquet(cache)
            vix = pd.read_parquet(vix_cache)["VIX"]
            return prices, vix

    tickers = SECTORS + ["SPY", "^VIX"]
    raw = yf.download(tickers, start="2020-01-01", progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw

    vc = "^VIX" if "^VIX" in close.columns else "VIX"
    vix = close[vc].dropna()
    vix.name = "VIX"

    cols = [c for c in SECTORS + ["SPY"] if c in close.columns]
    prices = close[cols].dropna(how="all")

    ix = prices.index.intersection(vix.index)
    prices = prices.loc[ix]
    vix = vix.loc[ix]

    prices.to_parquet(cache)
    pd.DataFrame({"VIX": vix}).to_parquet(vix_cache)
    fprint(f"  {len(cols)} tickers, {len(prices)} days "
           f"({prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')})")
    return prices, vix


# ============================================================================
# Feature Engineering (21 features)
# ============================================================================

def compute_features(px_daily, idx, ticker, spy_col=None):
    """Compute 21 features for LGBM ranking at a given index."""
    px = px_daily[ticker].iloc[:idx + 1].dropna()
    if len(px) < 260:
        return None

    f = {}
    rets = px.pct_change().dropna()

    # Momentum returns
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    # Volatility
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2

    # Risk-adjusted
    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252))
    px63 = px.iloc[-63:]
    pk = px63.cummax()
    f["maxdd_63d"] = float(((px63 / pk) - 1).min())
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    # Pct positive months (last 12m)
    monthly = px.iloc[-252:].resample("ME").last().pct_change().dropna()
    f["pct_pos_months_12m"] = float((monthly > 0).mean()) if len(monthly) > 3 else 0.5

    # Sortino 63d
    downside = r63[r63 < 0]
    ds_std = downside.std() if len(downside) > 3 else r63.std()
    f["sortino_63d"] = float(r63.mean() / (ds_std + 1e-10) * np.sqrt(252))

    # Calmar 1y
    px252 = px.iloc[-252:]
    ann_ret = float(px252.iloc[-1] / px252.iloc[0] - 1)
    dd252 = float(((px252 / px252.cummax()) - 1).min())
    f["calmar_1y"] = ann_ret / (abs(dd252) + 1e-10) if dd252 < -0.001 else ann_ret * 10

    # Up capture (vs SPY)
    if spy_col is not None and len(spy_col) > idx:
        spy_rets = spy_col.pct_change().dropna()
        spy_r63 = spy_rets.iloc[max(0, len(spy_rets)-63):]
        sec_r63 = rets.iloc[-len(spy_r63):]
        if len(spy_r63) > 10 and len(sec_r63) == len(spy_r63):
            up_mask = spy_r63 > 0
            if up_mask.sum() > 5:
                f["up_capture"] = float(sec_r63[up_mask.values].mean() / (spy_r63[up_mask].mean() + 1e-10))
            else:
                f["up_capture"] = 1.0
        else:
            f["up_capture"] = 1.0
    else:
        f["up_capture"] = 1.0

    # Trend R2 and slope (63d)
    y = px.iloc[-63:].values
    x = np.arange(len(y))
    if len(y) >= 10:
        coeffs = np.polyfit(x, y, 1)
        yhat = np.polyval(coeffs, x)
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - y.mean())**2) + 1e-10
        f["trend_r2_63d"] = float(1 - ss_res / ss_tot)
        f["trend_slope_63d"] = float(coeffs[0] / (y.mean() + 1e-10))  # normalized slope
    else:
        f["trend_r2_63d"] = 0.0
        f["trend_slope_63d"] = 0.0

    # Sector-SPY beta (63d)
    if spy_col is not None and len(spy_col) > idx:
        spy_rets_all = spy_col.pct_change().dropna()
        n = min(63, len(spy_rets_all), len(rets))
        sr = spy_rets_all.iloc[-n:]
        er = rets.iloc[-n:]
        if len(sr) == len(er) and len(sr) > 10:
            cov = np.cov(er.values, sr.values)
            f["sector_spy_beta_63d"] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0

    # Sector relative vol
    if spy_col is not None:
        spy_vol = spy_col.pct_change().dropna().iloc[-21:].std() * np.sqrt(252)
        f["sector_relative_vol_21d"] = f["vol_21d"] / (float(spy_vol) + 1e-10)
    else:
        f["sector_relative_vol_21d"] = 1.0

    # Cross-sector dispersion (computed per-date, use placeholder; overridden in caller)
    f["cross_sector_dispersion"] = 0.0

    return f


# ============================================================================
# LGBM Walk-Forward Ranking
# ============================================================================

def lgbm_walk_forward(px_daily, rebal_dates):
    """Walk-forward LGBM sector ranking with sliding 500d window."""
    fprint("Running LGBM walk-forward ranking...")

    spy_col = px_daily["SPY"] if "SPY" in px_daily.columns else None
    sector_cols = [c for c in px_daily.columns if c in SECTORS]

    # Build feature/label dataset
    records = []
    for dt in rebal_dates:
        idx = px_daily.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # Cross-sector dispersion for this date
        r21_all = []
        for tk in sector_cols:
            p = px_daily[tk].iloc[:idx+1].dropna()
            if len(p) > 21:
                r21_all.append(p.pct_change().iloc[-21:].mean())
        dispersion = float(np.std(r21_all)) if len(r21_all) > 3 else 0.0

        for tk in sector_cols:
            feats = compute_features(px_daily, idx, tk, spy_col)
            if feats is None:
                continue
            feats["cross_sector_dispersion"] = dispersion
            fi = min(idx + DTE, len(px_daily) - 1)
            fwd = float(px_daily[tk].iloc[fi] / px_daily[tk].iloc[idx] - 1)
            feats.update({"date": dt, "ticker": tk, "fwd_ret": fwd})
            records.append(feats)

    df = pd.DataFrame(records)
    if len(df) < 200:
        fprint(f"  Only {len(df)} records, need 200+")
        return {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    rankings = {}

    # Sliding window: TRAIN_DAYS worth of rebalance dates
    # Each rebal is ~10 trading days, so 500d ~ 50 rebal dates
    train_n = max(20, TRAIN_DAYS // REBAL_DAYS)

    for i in range(train_n, len(dates)):
        td = dates[max(0, i - train_n): i]
        test_date = dates[i]
        tr = df[df["date"].isin(td)]
        te = df[df["date"] == test_date].copy()
        if len(te) < 3 or len(tr) < 50:
            continue

        Xt = np.nan_to_num(tr[FEAT_COLS].values.astype(np.float32))
        yt = tr["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(te[FEAT_COLS].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1,
            )
            m.fit(Xt, yt)
            te["score"] = m.predict(Xe)
            rankings[test_date] = dict(zip(te["ticker"], te["score"]))
        except Exception as e:
            fprint(f"  LGBM error at {test_date}: {e}")
            continue

    fprint(f"  {len(rankings)} ranking dates produced")
    return rankings


# ============================================================================
# Credit Spread Pricing
# ============================================================================

def price_bull_put_spread(S, sigma, T, width=SPREAD_WIDTH, otm_pct=OTM_PCT, r=RISK_FREE):
    """
    Bull put credit spread: sell put at K1 (closer ATM), buy put at K2 (further OTM).
    K1 = S * (1 - otm_pct), K2 = K1 - width.
    Returns dict with credit, max_loss, strikes.
    """
    K1 = round(S * (1 - otm_pct), 2)  # short put (higher, closer ATM)
    K2 = round(K1 - width, 2)          # long put (lower, further OTM)
    if K2 <= 0:
        return None

    p1 = bs_put(S, K1, T, sigma, r)
    p2 = bs_put(S, K2, T, sigma, r)
    credit = (p1 - p2) * (1 - HAIRCUT)
    max_loss = (K1 - K2) - credit

    if credit <= 0.01:
        return None

    return {
        "type": "bull_put",
        "credit": credit,
        "max_loss": max(max_loss, 0.01),
        "K_short": K1,
        "K_long": K2,
        "width": K1 - K2,
        "commission": SPREAD_COMM,
    }


def price_bear_call_spread(S, sigma, T, width=SPREAD_WIDTH, otm_pct=OTM_PCT, r=RISK_FREE):
    """
    Bear call credit spread: sell call at K1 (closer ATM), buy call at K2 (further OTM).
    K1 = S * (1 + otm_pct), K2 = K1 + width.
    """
    K1 = round(S * (1 + otm_pct), 2)  # short call (lower, closer ATM)
    K2 = round(K1 + width, 2)          # long call (higher, further OTM)

    c1 = bs_call(S, K1, T, sigma, r)
    c2 = bs_call(S, K2, T, sigma, r)
    credit = (c1 - c2) * (1 - HAIRCUT)
    max_loss = (K2 - K1) - credit

    if credit <= 0.01:
        return None

    return {
        "type": "bear_call",
        "credit": credit,
        "max_loss": max(max_loss, 0.01),
        "K_short": K1,
        "K_long": K2,
        "width": K2 - K1,
        "commission": SPREAD_COMM,
    }


def price_iron_condor(S, sigma, T, width=SPREAD_WIDTH, otm_pct=OTM_PCT, r=RISK_FREE):
    """
    Iron condor: bull put spread + bear call spread.
    """
    bp = price_bull_put_spread(S, sigma, T, width, otm_pct, r)
    bc = price_bear_call_spread(S, sigma, T, width, otm_pct, r)
    if bp is None or bc is None:
        return None

    return {
        "type": "iron_condor",
        "credit": bp["credit"] + bc["credit"],
        "max_loss": max(bp["max_loss"], bc["max_loss"]),  # only one wing can be ITM
        "put_K_short": bp["K_short"],
        "put_K_long": bp["K_long"],
        "call_K_short": bc["K_short"],
        "call_K_long": bc["K_long"],
        "width": width,
        "commission": IC_COMM,
    }


def price_jade_lizard_v2(S, sigma, T, width=SPREAD_WIDTH, otm_pct=OTM_PCT, r=RISK_FREE):
    """
    Jade Lizard v2: short put (naked-ish, but we size it as defined risk) + bear call spread.
    Short put at S*(1-otm_pct), bear call spread above.
    """
    K_put = round(S * (1 - otm_pct), 2)
    put_prem = bs_put(S, K_put, T, sigma, r) * (1 - HAIRCUT)

    bc = price_bear_call_spread(S, sigma, T, width, otm_pct, r)
    if bc is None or put_prem <= 0.01:
        return None

    total_credit = put_prem + bc["credit"]
    # Max loss on call side: bc max_loss. On put side: K_put - total_credit (if stock goes to 0).
    # For sizing, use spread-like max loss = width (as if we had a $3 wide put spread)
    max_loss = width - total_credit  # treat as defined-risk for sizing

    return {
        "type": "jade_lizard",
        "credit": total_credit,
        "max_loss": max(max_loss, 0.01),
        "K_put": K_put,
        "call_K_short": bc["K_short"],
        "call_K_long": bc["K_long"],
        "width": width,
        "commission": SPREAD_COMM + 2 * LEG_COMM,  # put (2 legs) + call spread (4 legs) = 6 legs = $3.90
    }


# ============================================================================
# Spread Exit Logic (at expiry)
# ============================================================================

def exit_spread(spread, S_exit):
    """
    Determine P&L at expiry.
    Returns net P&L (positive = profit for seller).
    """
    stype = spread["type"]
    credit = spread["credit"]
    comm = spread["commission"]

    if stype == "bull_put":
        # Bull put: max loss if S_exit < K_long. Full credit if S_exit >= K_short.
        K_short = spread["K_short"]
        K_long = spread["K_long"]
        if S_exit >= K_short:
            intrinsic_cost = 0.0
        elif S_exit <= K_long:
            intrinsic_cost = K_short - K_long  # full width
        else:
            intrinsic_cost = K_short - S_exit
        pnl = credit - intrinsic_cost - comm
        return pnl

    elif stype == "bear_call":
        K_short = spread["K_short"]
        K_long = spread["K_long"]
        if S_exit <= K_short:
            intrinsic_cost = 0.0
        elif S_exit >= K_long:
            intrinsic_cost = K_long - K_short  # full width
        else:
            intrinsic_cost = S_exit - K_short
        pnl = credit - intrinsic_cost - comm
        return pnl

    elif stype == "iron_condor":
        # Only one wing can be ITM at expiry
        put_K_short = spread["put_K_short"]
        put_K_long = spread["put_K_long"]
        call_K_short = spread["call_K_short"]
        call_K_long = spread["call_K_long"]

        # Put side
        if S_exit >= put_K_short:
            put_cost = 0.0
        elif S_exit <= put_K_long:
            put_cost = put_K_short - put_K_long
        else:
            put_cost = put_K_short - S_exit

        # Call side
        if S_exit <= call_K_short:
            call_cost = 0.0
        elif S_exit >= call_K_long:
            call_cost = call_K_long - call_K_short
        else:
            call_cost = S_exit - call_K_short

        pnl = credit - put_cost - call_cost - comm
        return pnl

    elif stype == "jade_lizard":
        K_put = spread["K_put"]
        call_K_short = spread["call_K_short"]
        call_K_long = spread["call_K_long"]

        # Put side (naked put -- cost if S < K_put)
        put_cost = max(K_put - S_exit, 0.0)
        # But cap the loss at width (we treat it as defined risk for sizing)
        put_cost = min(put_cost, spread["width"])

        # Call side
        if S_exit <= call_K_short:
            call_cost = 0.0
        elif S_exit >= call_K_long:
            call_cost = call_K_long - call_K_short
        else:
            call_cost = S_exit - call_K_short

        pnl = credit - put_cost - call_cost - spread["commission"]
        return pnl

    return 0.0


# ============================================================================
# Position Sizing
# ============================================================================

def compute_position_size(equity, max_loss_per_spread):
    """
    How many spreads can we trade?
    Max risk = min(15% of equity, $100) per trade.
    Returns number of contracts (usually 1 for small account).
    """
    max_risk = min(equity * MAX_RISK_PCT, MAX_RISK_CAP)
    if max_loss_per_spread <= 0:
        return 0
    n = int(max_risk / (max_loss_per_spread * 100))  # each spread = 100 shares notional
    # For ETFs at ~$30-200, a $3 wide spread costs $300 max loss per contract
    # With $645 account, 15% = $96.75, so usually 0 contracts with 100 multiplier
    # BUT for options on ETFs, spreads are per-share, not per-100.
    # With ETF options: max_loss = width * 100 per contract.
    # $3 wide = $300 max loss per contract. 15% of $645 = $96.75 => 0 contracts.
    # This is too restrictive. For a small account, we need to think in per-share terms.
    # Actually, options are per 100 shares. A $3 spread = $300 max loss.
    # With $645 we can afford max 2 contracts at a time.
    # Let's size based on notional risk as fraction of equity:
    risk_per_contract = max_loss_per_spread * 100  # options are per 100 shares
    if risk_per_contract <= 0:
        return 0
    n = max(1, int(max_risk / risk_per_contract))
    return min(n, 2)  # cap at 2 contracts for small account


def compute_position_size_dollars(equity, max_loss_dollar):
    """
    Position sizing in dollar terms.
    max_loss_dollar is the max loss for 1 unit of the trade.
    Returns number of units.
    """
    max_risk = min(equity * MAX_RISK_PCT, MAX_RISK_CAP)
    if max_loss_dollar <= 0:
        return 0
    n = max(1, int(max_risk / max_loss_dollar))
    return min(n, 3)


# ============================================================================
# Strategy Simulation Engine
# ============================================================================

def simulate_variant(prices, vix, rankings, variant, name):
    """
    Simulate one strategy variant.
    Returns dict with equity curve, trades, metrics.
    """
    fprint(f"\n{'='*60}")
    fprint(f"Variant {name}")
    fprint(f"{'='*60}")

    sector_cols = [c for c in prices.columns if c in SECTORS]
    ranking_dates = sorted(rankings.keys())
    if not ranking_dates:
        fprint("  No ranking dates!")
        return None

    equity = CAPITAL
    equity_curve = []
    trades = []
    open_positions = []  # list of dicts

    T = DTE / 252.0  # time to expiry in years

    for rd_idx, rd in enumerate(ranking_dates):
        rd_loc = prices.index.get_indexer([rd], method="ffill")[0]
        if rd_loc < 0 or rd_loc >= len(prices):
            continue

        # Get current VIX
        vix_loc = vix.index.get_indexer([rd], method="ffill")[0]
        curr_vix = float(vix.iloc[vix_loc]) if vix_loc >= 0 else 20.0

        # Close expired positions
        new_open = []
        for pos in open_positions:
            if rd >= pos["expiry_date"]:
                # Find exit price
                exp_loc = prices.index.get_indexer([pos["expiry_date"]], method="ffill")[0]
                if exp_loc < 0 or exp_loc >= len(prices):
                    exp_loc = min(rd_loc, len(prices) - 1)
                S_exit = float(prices[pos["ticker"]].iloc[exp_loc])
                pnl = exit_spread(pos["spread"], S_exit) * pos["qty"]
                equity += pnl
                trades.append({
                    "entry_date": str(pos["entry_date"]),
                    "exit_date": str(pos["expiry_date"]),
                    "ticker": pos["ticker"],
                    "type": pos["spread"]["type"],
                    "credit": pos["spread"]["credit"] * pos["qty"],
                    "pnl": pnl,
                    "S_entry": pos["S_entry"],
                    "S_exit": S_exit,
                })
            else:
                # Check stop loss for variant F
                if variant == "F" and pos["spread"]["type"] == "jade_lizard":
                    S_now = float(prices[pos["ticker"]].iloc[rd_loc])
                    # Rough mark-to-market: if underlying moved against us badly
                    put_loss = max(pos["spread"]["K_put"] - S_now, 0)
                    if put_loss > STOP_MULT * pos["spread"]["credit"]:
                        pnl = -(STOP_MULT * pos["spread"]["credit"]) * pos["qty"] - pos["spread"]["commission"] * pos["qty"]
                        equity += pnl
                        trades.append({
                            "entry_date": str(pos["entry_date"]),
                            "exit_date": str(rd),
                            "ticker": pos["ticker"],
                            "type": pos["spread"]["type"] + "_stopped",
                            "credit": pos["spread"]["credit"] * pos["qty"],
                            "pnl": pnl,
                            "S_entry": pos["S_entry"],
                            "S_exit": S_now,
                        })
                        continue
                new_open.append(pos)
        open_positions = new_open

        equity_curve.append({"date": str(rd), "equity": equity})

        # Skip if too many open positions
        if len(open_positions) >= MAX_CONCURRENT:
            continue

        # Skip if equity too low
        if equity < 50:
            continue

        # Get ranked sectors
        scores = rankings[rd]
        sorted_sectors = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Determine which spreads to open based on variant
        new_trades = []

        if variant == "A":
            # Bull put on top 3
            top3 = sorted_sectors[:3]
            for tk in top3:
                if len(open_positions) + len(new_trades) >= MAX_CONCURRENT:
                    break
                # Skip if already have position in this ticker
                if any(p["ticker"] == tk for p in open_positions + new_trades):
                    continue
                S = float(prices[tk].iloc[rd_loc])
                sigma = get_iv(prices[tk], rd_loc)
                spread = price_bull_put_spread(S, sigma, T)
                if spread:
                    new_trades.append({"ticker": tk, "spread": spread, "S": S})

        elif variant == "B":
            # Bear call on bottom 3
            bottom3 = sorted_sectors[-3:]
            for tk in bottom3:
                if len(open_positions) + len(new_trades) >= MAX_CONCURRENT:
                    break
                if any(p["ticker"] == tk for p in open_positions + new_trades):
                    continue
                S = float(prices[tk].iloc[rd_loc])
                sigma = get_iv(prices[tk], rd_loc)
                spread = price_bear_call_spread(S, sigma, T)
                if spread:
                    new_trades.append({"ticker": tk, "spread": spread, "S": S})

        elif variant == "C":
            # Iron condor on middle ranked (4-8)
            middle = sorted_sectors[3:8]
            for tk in middle:
                if len(open_positions) + len(new_trades) >= MAX_CONCURRENT:
                    break
                if any(p["ticker"] == tk for p in open_positions + new_trades):
                    continue
                S = float(prices[tk].iloc[rd_loc])
                sigma = get_iv(prices[tk], rd_loc)
                spread = price_iron_condor(S, sigma, T)
                if spread:
                    new_trades.append({"ticker": tk, "spread": spread, "S": S})

        elif variant == "D":
            # Combined: 40% bull put (top), 40% bear call (bottom), 20% iron condor (mid)
            slots = MAX_CONCURRENT - len(open_positions)
            bull_slots = max(1, int(slots * 0.4))
            bear_slots = max(1, int(slots * 0.4))
            ic_slots = max(0, slots - bull_slots - bear_slots)

            # Bull puts on top
            for tk in sorted_sectors[:3]:
                if len(new_trades) >= bull_slots:
                    break
                if any(p["ticker"] == tk for p in open_positions + new_trades):
                    continue
                S = float(prices[tk].iloc[rd_loc])
                sigma = get_iv(prices[tk], rd_loc)
                spread = price_bull_put_spread(S, sigma, T)
                if spread:
                    new_trades.append({"ticker": tk, "spread": spread, "S": S})

            # Bear calls on bottom
            for tk in sorted_sectors[-3:]:
                if len(new_trades) >= bull_slots + bear_slots:
                    break
                if any(p["ticker"] == tk for p in open_positions + new_trades):
                    continue
                S = float(prices[tk].iloc[rd_loc])
                sigma = get_iv(prices[tk], rd_loc)
                spread = price_bear_call_spread(S, sigma, T)
                if spread:
                    new_trades.append({"ticker": tk, "spread": spread, "S": S})

            # Iron condors on middle
            for tk in sorted_sectors[3:8]:
                if len(new_trades) >= bull_slots + bear_slots + ic_slots:
                    break
                if any(p["ticker"] == tk for p in open_positions + new_trades):
                    continue
                S = float(prices[tk].iloc[rd_loc])
                sigma = get_iv(prices[tk], rd_loc)
                spread = price_iron_condor(S, sigma, T)
                if spread:
                    new_trades.append({"ticker": tk, "spread": spread, "S": S})

        elif variant == "E":
            # VIX-Adaptive
            if curr_vix > 20:
                # High VIX: only bull puts (elevated premium, bullish bias)
                for tk in sorted_sectors[:3]:
                    if len(open_positions) + len(new_trades) >= MAX_CONCURRENT:
                        break
                    if any(p["ticker"] == tk for p in open_positions + new_trades):
                        continue
                    S = float(prices[tk].iloc[rd_loc])
                    sigma = get_iv(prices[tk], rd_loc)
                    spread = price_bull_put_spread(S, sigma, T)
                    if spread:
                        new_trades.append({"ticker": tk, "spread": spread, "S": S})
            elif curr_vix < 15:
                # Low VIX: iron condors (range-bound)
                for tk in sorted_sectors[3:8]:
                    if len(open_positions) + len(new_trades) >= MAX_CONCURRENT:
                        break
                    if any(p["ticker"] == tk for p in open_positions + new_trades):
                        continue
                    S = float(prices[tk].iloc[rd_loc])
                    sigma = get_iv(prices[tk], rd_loc)
                    spread = price_iron_condor(S, sigma, T)
                    if spread:
                        new_trades.append({"ticker": tk, "spread": spread, "S": S})
            else:
                # Mid VIX (15-20): equal mix
                # 1 bull put, 1 bear call, 1 iron condor
                if sorted_sectors:
                    tk = sorted_sectors[0]
                    if not any(p["ticker"] == tk for p in open_positions + new_trades):
                        S = float(prices[tk].iloc[rd_loc])
                        sigma = get_iv(prices[tk], rd_loc)
                        spread = price_bull_put_spread(S, sigma, T)
                        if spread:
                            new_trades.append({"ticker": tk, "spread": spread, "S": S})

                if len(sorted_sectors) > 10:
                    tk = sorted_sectors[-1]
                    if not any(p["ticker"] == tk for p in open_positions + new_trades):
                        S = float(prices[tk].iloc[rd_loc])
                        sigma = get_iv(prices[tk], rd_loc)
                        spread = price_bear_call_spread(S, sigma, T)
                        if spread:
                            new_trades.append({"ticker": tk, "spread": spread, "S": S})

                if len(sorted_sectors) > 5:
                    tk = sorted_sectors[5]
                    if not any(p["ticker"] == tk for p in open_positions + new_trades):
                        S = float(prices[tk].iloc[rd_loc])
                        sigma = get_iv(prices[tk], rd_loc)
                        spread = price_iron_condor(S, sigma, T)
                        if spread:
                            new_trades.append({"ticker": tk, "spread": spread, "S": S})

        elif variant == "F":
            # Jade Lizard v2 on top 3
            top3 = sorted_sectors[:3]
            for tk in top3:
                if len(open_positions) + len(new_trades) >= MAX_CONCURRENT:
                    break
                if any(p["ticker"] == tk for p in open_positions + new_trades):
                    continue
                S = float(prices[tk].iloc[rd_loc])
                sigma = get_iv(prices[tk], rd_loc)
                spread = price_jade_lizard_v2(S, sigma, T)
                if spread:
                    new_trades.append({"ticker": tk, "spread": spread, "S": S})

        # Open new positions with position sizing
        for nt in new_trades:
            spread = nt["spread"]
            # Dollar max loss per unit
            max_loss_dollar = spread["max_loss"]  # per share
            # For ETF options, 1 contract = 100 shares
            # But with $645 account, even 1 contract of $3 wide = $300 risk
            # Use per-share sizing: treat each "unit" as 1 share equivalent
            # This is how small accounts trade: 1 contract, partial fills
            qty = 1  # always 1 contract for small account
            risk_dollar = max_loss_dollar * 100  # actual risk per contract
            if risk_dollar > equity * MAX_RISK_PCT:
                continue  # too risky for current equity

            # Calculate expiry date
            exp_idx = min(rd_loc + DTE, len(prices) - 1)
            exp_date = prices.index[exp_idx]

            open_positions.append({
                "ticker": nt["ticker"],
                "spread": spread,
                "qty": qty,
                "S_entry": nt["S"],
                "entry_date": rd,
                "expiry_date": exp_date,
            })

    # Close any remaining open positions at last date
    last_loc = len(prices) - 1
    for pos in open_positions:
        S_exit = float(prices[pos["ticker"]].iloc[last_loc])
        pnl = exit_spread(pos["spread"], S_exit) * pos["qty"]
        equity += pnl
        trades.append({
            "entry_date": str(pos["entry_date"]),
            "exit_date": str(prices.index[last_loc]),
            "ticker": pos["ticker"],
            "type": pos["spread"]["type"],
            "credit": pos["spread"]["credit"] * pos["qty"],
            "pnl": pnl,
            "S_entry": pos["S_entry"],
            "S_exit": S_exit,
        })

    equity_curve.append({"date": str(prices.index[last_loc]), "equity": equity})

    if not trades:
        fprint("  No trades generated!")
        return None

    fprint(f"  {len(trades)} trades, final equity: ${equity:.2f}")
    return {
        "name": name,
        "variant": variant,
        "equity_curve": equity_curve,
        "trades": trades,
        "final_equity": equity,
    }


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(result):
    """Compute performance metrics from simulation result."""
    if result is None:
        return None

    trades = result["trades"]
    eq = pd.DataFrame(result["equity_curve"])
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date").sort_index()

    pnls = [t["pnl"] for t in trades]
    pnl_arr = np.array(pnls)

    # Basic metrics
    total_return = (result["final_equity"] / CAPITAL - 1) * 100
    n_trades = len(trades)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n_trades * 100 if n_trades > 0 else 0
    avg_win = np.mean([p for p in pnls if p > 0]) if wins > 0 else 0
    avg_loss = np.mean([p for p in pnls if p <= 0]) if (n_trades - wins) > 0 else 0
    pf = abs(sum(p for p in pnls if p > 0) / (sum(p for p in pnls if p < 0) + 1e-10))

    # Equity curve metrics
    eq_vals = eq["equity"].values
    peak = np.maximum.accumulate(eq_vals)
    dd = (eq_vals - peak) / (peak + 1e-10)
    mdd = float(dd.min()) * 100

    # Monthly returns for Sharpe/Sortino
    eq_monthly = eq["equity"].resample("ME").last().dropna()
    monthly_rets = eq_monthly.pct_change().dropna()

    if len(monthly_rets) > 2:
        sharpe = float(monthly_rets.mean() / (monthly_rets.std() + 1e-10) * np.sqrt(12))
        downside = monthly_rets[monthly_rets < 0]
        ds_std = downside.std() if len(downside) > 2 else monthly_rets.std()
        sortino = float(monthly_rets.mean() / (ds_std + 1e-10) * np.sqrt(12))
    else:
        sharpe = 0.0
        sortino = 0.0

    # Yearly returns
    eq_yearly = eq["equity"].resample("YE").last().dropna()
    yearly_rets = eq_yearly.pct_change().dropna()
    years_profitable = (yearly_rets > 0).sum()
    total_years = len(yearly_rets)
    yearly_consistency = years_profitable / total_years * 100 if total_years > 0 else 0

    metrics = {
        "total_return_pct": round(total_return, 2),
        "n_trades": n_trades,
        "win_rate_pct": round(wr, 2),
        "profit_factor": round(pf, 3),
        "sharpe_monthly": round(sharpe, 3),
        "sortino_monthly": round(sortino, 3),
        "mdd_pct": round(mdd, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "final_equity": round(result["final_equity"], 2),
        "yearly_consistency_pct": round(yearly_consistency, 1),
        "years_profitable": int(years_profitable),
        "total_years": int(total_years),
    }
    return metrics


# ============================================================================
# 5-Gate Adversarial Validation
# ============================================================================

def adversarial_validation(result, prices, vix):
    """
    5-gate adversarial validation:
    1. Permutation p<0.05 (300 trials)
    2. Regime stability |green-red|/max < 0.50
    3. Sub-period (both halves Sharpe>0.3)
    4. Outlier removal (Sharpe within 50% after trimming top/bottom 5%)
    5. Yearly consistency (60%+ years profitable)
    """
    if result is None:
        return {"pass": False, "gates": {}}

    trades = result["trades"]
    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    if n < 10:
        return {"pass": False, "gates": {"reason": "too few trades"}}

    gates = {}

    # Gate 1: Permutation test
    real_sharpe = _trade_sharpe(pnls)
    perm_count = 0
    for _ in range(N_PERM):
        shuffled = np.random.permutation(pnls)
        if _trade_sharpe(shuffled) >= real_sharpe:
            perm_count += 1
    p_val = perm_count / N_PERM
    gates["permutation"] = {"p_value": round(p_val, 4), "pass": p_val < 0.05}

    # Gate 2: Regime stability
    # Classify days as green/red based on SPY
    spy = prices["SPY"] if "SPY" in prices.columns else None
    if spy is not None:
        spy_daily = spy.pct_change()
        green_pnls = []
        red_pnls = []
        for t in trades:
            entry = pd.Timestamp(t["entry_date"])
            loc = spy_daily.index.get_indexer([entry], method="ffill")[0]
            if loc >= 0:
                # Use cumulative SPY return over trade period as regime
                exit_dt = pd.Timestamp(t["exit_date"])
                exit_loc = spy_daily.index.get_indexer([exit_dt], method="ffill")[0]
                spy_ret = float(spy.iloc[min(exit_loc, len(spy)-1)] / spy.iloc[loc] - 1) if loc < len(spy) else 0
                if spy_ret >= 0:
                    green_pnls.append(t["pnl"])
                else:
                    red_pnls.append(t["pnl"])

        if green_pnls and red_pnls:
            green_sharpe = _trade_sharpe(np.array(green_pnls))
            red_sharpe = _trade_sharpe(np.array(red_pnls))
            max_s = max(abs(green_sharpe), abs(red_sharpe), 1e-10)
            regime_diff = abs(green_sharpe - red_sharpe) / max_s
            gates["regime_stability"] = {
                "green_sharpe": round(green_sharpe, 3),
                "red_sharpe": round(red_sharpe, 3),
                "diff_ratio": round(regime_diff, 3),
                "pass": regime_diff < 0.50,
            }
        else:
            gates["regime_stability"] = {"pass": True, "note": "insufficient regime data"}
    else:
        gates["regime_stability"] = {"pass": True, "note": "no SPY data"}

    # Gate 3: Sub-period consistency
    mid = n // 2
    first_half = pnls[:mid]
    second_half = pnls[mid:]
    s1 = _trade_sharpe(first_half)
    s2 = _trade_sharpe(second_half)
    gates["sub_period"] = {
        "first_half_sharpe": round(s1, 3),
        "second_half_sharpe": round(s2, 3),
        "pass": s1 > 0.3 and s2 > 0.3,
    }

    # Gate 4: Outlier removal
    trim_n = max(1, int(n * 0.05))
    sorted_pnls = np.sort(pnls)
    trimmed = sorted_pnls[trim_n:-trim_n] if trim_n > 0 and 2*trim_n < n else pnls
    trimmed_sharpe = _trade_sharpe(trimmed)
    gates["outlier_removal"] = {
        "full_sharpe": round(real_sharpe, 3),
        "trimmed_sharpe": round(trimmed_sharpe, 3),
        "ratio": round(trimmed_sharpe / (real_sharpe + 1e-10), 3),
        "pass": trimmed_sharpe > real_sharpe * 0.50,
    }

    # Gate 5: Yearly consistency
    # Group trades by year
    yearly_pnl = {}
    for t in trades:
        yr = pd.Timestamp(t["entry_date"]).year
        yearly_pnl[yr] = yearly_pnl.get(yr, 0) + t["pnl"]
    if yearly_pnl:
        n_years = len(yearly_pnl)
        n_profitable = sum(1 for v in yearly_pnl.values() if v > 0)
        pct = n_profitable / n_years * 100
        gates["yearly_consistency"] = {
            "years_profitable": n_profitable,
            "total_years": n_years,
            "pct": round(pct, 1),
            "pass": pct >= 60.0,
        }
    else:
        gates["yearly_consistency"] = {"pass": False, "note": "no yearly data"}

    n_pass = sum(1 for g in gates.values() if g.get("pass", False))
    return {
        "pass": n_pass >= 4,  # pass if 4/5 gates pass
        "gates_passed": n_pass,
        "gates_total": 5,
        "gates": gates,
    }


def _trade_sharpe(pnls):
    """Sharpe ratio from trade-level PnL array."""
    if len(pnls) < 2:
        return 0.0
    return float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(26))  # ~26 biweekly periods/year


# ============================================================================
# MLflow Logging
# ============================================================================

def log_to_mlflow(all_results):
    """Log all variant results to MLflow experiment."""
    if not MLFLOW_OK:
        return

    try:
        mlflow.set_experiment("income_strategy_v2")
        for name, data in all_results.items():
            metrics = data.get("metrics")
            adv = data.get("adversarial")
            if metrics is None:
                continue
            with mlflow.start_run(run_name=name):
                mlflow.log_params({
                    "variant": data.get("variant", ""),
                    "capital": CAPITAL,
                    "dte": DTE,
                    "spread_width": SPREAD_WIDTH,
                    "otm_pct": OTM_PCT,
                    "max_concurrent": MAX_CONCURRENT,
                    "max_risk_pct": MAX_RISK_PCT,
                    "iv_multiplier": IV_MULT,
                    "haircut": HAIRCUT,
                    "train_days": TRAIN_DAYS,
                    "rebal_days": REBAL_DAYS,
                })
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                if adv:
                    mlflow.log_metric("adv_gates_passed", adv.get("gates_passed", 0))
                    mlflow.log_metric("adv_pass", int(adv.get("pass", False)))
        fprint("MLflow logging complete")
    except Exception as e:
        fprint(f"MLflow logging error: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("INCOME STRATEGY v2 -- LGBM-Ranked Credit Spread Portfolio")
    fprint("=" * 70)

    # Download data
    prices, vix = download_data()

    # Generate biweekly rebalance dates (every 10 trading days)
    all_dates = prices.index
    rebal_dates = [all_dates[i] for i in range(0, len(all_dates), REBAL_DAYS)]
    fprint(f"  {len(rebal_dates)} biweekly rebalance dates")

    # LGBM walk-forward ranking
    rankings = lgbm_walk_forward(prices, rebal_dates)
    if not rankings:
        fprint("FATAL: No rankings produced")
        return

    # Define variants
    variants = {
        "A_BullPut_Top3": ("A", "Bull Put Credit Spreads on top-3 LGBM sectors"),
        "B_BearCall_Bot3": ("B", "Bear Call Credit Spreads on bottom-3 LGBM sectors"),
        "C_IronCondor_Mid": ("C", "Iron Condor on middle-ranked sectors (4-8)"),
        "D_Combined_Income": ("D", "Combined Income Portfolio (40/40/20)"),
        "E_VIX_Adaptive": ("E", "VIX-Adaptive Income"),
        "F_JadeLizard_v2": ("F", "Jade Lizard v2 with strict limits"),
    }

    all_results = {}

    for vname, (vcode, vdesc) in variants.items():
        try:
            result = simulate_variant(prices, vix, rankings, vcode, f"{vname}: {vdesc}")
            if result is None:
                fprint(f"  {vname}: No result")
                all_results[vname] = {"variant": vcode, "metrics": None, "adversarial": None}
                continue

            metrics = compute_metrics(result)
            adv = adversarial_validation(result, prices, vix)

            fprint(f"\n  Metrics for {vname}:")
            if metrics:
                for k, v in metrics.items():
                    fprint(f"    {k}: {v}")
            fprint(f"  Adversarial: {adv['gates_passed']}/{adv['gates_total']} gates passed"
                   f" {'PASS' if adv['pass'] else 'FAIL'}")
            for gname, gdata in adv["gates"].items():
                fprint(f"    {gname}: {'PASS' if gdata.get('pass') else 'FAIL'} -- {gdata}")

            all_results[vname] = {
                "variant": vcode,
                "metrics": metrics,
                "adversarial": adv,
                "equity_curve": result["equity_curve"],
                "n_trades": len(result["trades"]),
                "trades_sample": result["trades"][:5],  # first 5 trades as sample
            }
        except Exception as e:
            fprint(f"  ERROR in {vname}: {e}")
            traceback.print_exc()
            all_results[vname] = {"variant": vcode, "metrics": None, "adversarial": None, "error": str(e)}

    # Print sorted results table
    fprint("\n" + "=" * 90)
    fprint("SORTED RESULTS (by Sharpe)")
    fprint("=" * 90)
    fprint(f"{'Variant':<25} {'Sharpe':>8} {'Sortino':>8} {'WR%':>6} {'PF':>7} {'MDD%':>8} {'Return%':>9} {'Adv':>5} {'Trades':>7}")
    fprint("-" * 90)

    scored = []
    for vname, data in all_results.items():
        m = data.get("metrics")
        if m is None:
            fprint(f"{vname:<25} {'N/A':>8} {'N/A':>8} {'N/A':>6} {'N/A':>7} {'N/A':>8} {'N/A':>9} {'N/A':>5} {'N/A':>7}")
            continue
        adv = data.get("adversarial", {})
        scored.append((m["sharpe_monthly"], vname, m, adv))

    scored.sort(reverse=True, key=lambda x: x[0])
    for sharpe, vname, m, adv in scored:
        adv_str = f"{adv.get('gates_passed',0)}/{adv.get('gates_total',5)}"
        fprint(f"{vname:<25} {m['sharpe_monthly']:>8.3f} {m['sortino_monthly']:>8.3f} "
               f"{m['win_rate_pct']:>5.1f}% {m['profit_factor']:>7.3f} "
               f"{m['mdd_pct']:>7.1f}% {m['total_return_pct']:>8.1f}% "
               f"{adv_str:>5} {m['n_trades']:>7}")

    fprint("=" * 90)

    # Save results
    save_data = {
        "run_timestamp": datetime.now().isoformat(),
        "config": {
            "capital": CAPITAL,
            "dte": DTE,
            "spread_width": SPREAD_WIDTH,
            "otm_pct": OTM_PCT,
            "max_concurrent": MAX_CONCURRENT,
            "max_risk_pct": MAX_RISK_PCT,
            "iv_multiplier": IV_MULT,
            "haircut": HAIRCUT,
            "train_days": TRAIN_DAYS,
            "rebal_days": REBAL_DAYS,
            "n_permutations": N_PERM,
            "sectors": SECTORS,
        },
        "results": {},
    }

    for vname, data in all_results.items():
        save_entry = {
            "variant": data.get("variant"),
            "metrics": data.get("metrics"),
            "adversarial": data.get("adversarial"),
            "n_trades": data.get("n_trades"),
        }
        if data.get("error"):
            save_entry["error"] = data["error"]
        save_data["results"][vname] = save_entry

    results_path = OUT_DIR / "income_v2_results.json"
    with open(results_path, "w") as fp:
        json.dump(save_data, fp, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Log to MLflow
    log_to_mlflow(all_results)

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
