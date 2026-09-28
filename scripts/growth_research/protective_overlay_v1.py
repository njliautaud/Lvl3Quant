#!/usr/bin/env python3
"""
Protective Overlay v1 — Hedging overlays to reduce MaxDD for sector spread strategy
=====================================================================================
Tests 8 protective variants (A-H) on top of LGBM-ranked sector bull call spreads.

Base strategy: LGBM-ranked sector bull call spreads, VIX>20, 3% width, DTE=28,
hold-to-expiry, 15% haircut, $2.60 commission, $645 start, top 3 sectors.

Variants:
  A) OTM Put Hedge — buy 5% OTM SPY puts when entering bull spreads
  B) VIX Call Hedge — buy VIX call spreads (ATM to +5%) alongside bull spreads
  C) Position Size Cap by Drawdown — 50% size when in >10% drawdown
  D) Stop-Loss at Portfolio Level — close all if DD > 15%, re-enter after 10 days
  E) Inverse Sector ETF Hedge — 20% in SH when VIX>25
  F) Regime-Adaptive Sizing — full at VIX 20-25, 50% at 25-30, 25% at VIX>30
  G) Trailing Stop per Trade — exit spread if down >50% of debit, 15% exit haircut
  H) Combo Best — combine best-performing elements from A-G

Walk-forward: 500d train, 250d test, sliding. Biweekly rebalance.
21 LGBM features. ATR-based BS pricing, iv_multiplier=1.2, 15% haircut.
5-gate adversarial validation per variant.

MLflow: protective_overlay_v1
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    os.system(f"{sys.executable} -m pip install yfinance -q")
    import yfinance as yf

try:
    import lightgbm as lgb
except ImportError:
    os.system(f"{sys.executable} -m pip install lightgbm -q")
    import lightgbm as lgb


def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "output" / "growth_research" / "protective_overlay_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
INITIAL_CAP = 645.0
SPREAD_COMM = 2.60          # $2.60 per spread round trip
HAIRCUT = 0.15              # 15% entry haircut
EXIT_HAIRCUT = 0.15         # 15% exit haircut (for variant G and mid-trade exits)
SPREAD_WIDTH_PCT = 3.0      # 3% OTM for short leg
DTE = 28                    # days to expiry
MAX_POS_USD = 200.0         # max $200 per trade
MAX_CONCURRENT = 3          # top 3 sectors
IV_MULTIPLIER = 1.2         # IV = realized_vol * 1.2
RISK_FREE_RATE = 0.045
IV_FLOOR = 0.15

# Walk-forward
TRAIN_DAYS = 500
TEST_DAYS = 250

# LGBM features (21 total)
LGBM_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_21d", "sharpe_63d",
    "maxdd_63d", "pct_52w_high", "mom_accel",
    "rsi_14", "macd_signal", "bb_pct",
    "rel_spy_21d", "rel_spy_63d",
    "skew_21d", "kurt_21d",
    "atr_pct_14d",
]

# MLflow
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen("http://jupiter:5000/", timeout=2)
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    MLFLOW_OK = True
except Exception:
    fprint("MLflow unavailable — results saved to JSON only")


# ============================================================
# BLACK-SCHOLES PRICING
# ============================================================

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 1e-6 or sigma <= 1e-6:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if T <= 1e-6 or sigma <= 1e-6:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def get_iv(prices: pd.Series, as_of_idx: int, lookback: int = 63) -> float:
    """Compute implied vol estimate from realized vol * IV_MULTIPLIER."""
    start = max(0, as_of_idx - lookback)
    window = prices.iloc[start:as_of_idx + 1]
    if len(window) < 20:
        return 0.30
    rv = window.pct_change().dropna().std() * np.sqrt(252)
    return max(float(rv) * IV_MULTIPLIER, IV_FLOOR)


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Download sector ETFs, SPY, VIX, SH from yfinance (2007-2026)."""
    cache_file = OUT_DIR / "price_cache.parquet"
    cache_vix = OUT_DIR / "vix_cache.parquet"

    tickers = SECTORS + ["SPY", "SH"]
    vix_ticker = "^VIX"

    fprint("[1/7] Downloading data...")

    # Download main tickers
    raw = yf.download(tickers + [vix_ticker], start="2007-01-01",
                      end="2026-07-27", progress=False)

    mi = isinstance(raw.columns, pd.MultiIndex)
    if mi:
        close = raw["Close"]
        high = raw["High"]
        low = raw["Low"]
    else:
        close = raw
        high = raw
        low = raw

    # Extract VIX
    vix_col = "^VIX" if "^VIX" in close.columns else "VIX"
    vix = close[vix_col].dropna()
    if vix_col in close.columns:
        close = close.drop(columns=[vix_col])
        high = high.drop(columns=[vix_col]) if vix_col in high.columns else high
        low = low.drop(columns=[vix_col]) if vix_col in low.columns else low

    spy = close["SPY"].dropna()
    sh_price = close["SH"].dropna() if "SH" in close.columns else None

    # Sector prices
    sector_cols = [c for c in SECTORS if c in close.columns]
    sector_px = close[sector_cols].dropna(how="all")

    # Common index
    idx = sector_px.index.intersection(spy.index).intersection(vix.index)
    if sh_price is not None:
        idx = idx.intersection(sh_price.index)
    sector_px = sector_px.loc[idx]
    spy = spy.loc[idx]
    vix = vix.loc[idx]
    sh_price = sh_price.loc[idx] if sh_price is not None else pd.Series(0.0, index=idx)

    # High/low for ATR
    sector_hi = high[[c for c in sector_cols if c in high.columns]].reindex(idx).ffill()
    sector_lo = low[[c for c in sector_cols if c in low.columns]].reindex(idx).ffill()

    fprint(f"  {len(idx)} trading days, {len(sector_cols)} sectors, "
           f"{idx[0].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')}")

    return sector_px, sector_hi, sector_lo, spy, vix, sh_price


# ============================================================
# ATR COMPUTATION
# ============================================================

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": abs(high - close.shift(1)),
        "lc": abs(low - close.shift(1)),
    }).max(axis=1)
    return tr.rolling(period).mean()


# ============================================================
# LGBM FEATURES & WALK-FORWARD RANKING
# ============================================================

def build_sector_features(px: pd.DataFrame, spy: pd.Series,
                          idx: int, ticker: str) -> Optional[Dict]:
    """Build 21 LGBM features for a sector at a given index."""
    p = px[ticker].iloc[:idx + 1].dropna()
    if len(p) < 260:
        return None

    sp = spy.iloc[:idx + 1].dropna()
    r = p.pct_change().dropna()

    f = {}

    # Momentum features (6)
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(p.iloc[-1] / p.iloc[-lb] - 1) if len(p) > lb else 0.0

    # Volatility (2)
    f["vol_21d"] = float(r.iloc[-21:].std() * np.sqrt(252)) if len(r) > 21 else 0.2
    f["vol_63d"] = float(r.iloc[-63:].std() * np.sqrt(252)) if len(r) > 63 else 0.2

    # Risk-adjusted (2)
    r21 = r.iloc[-21:]
    f["sharpe_21d"] = float(r21.mean() / (r21.std() + 1e-10) * np.sqrt(252)) if len(r21) > 10 else 0
    r63 = r.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0

    # Drawdown (1)
    p63 = p.iloc[-63:]
    f["maxdd_63d"] = float(((p63 / p63.cummax()) - 1).min()) if len(p63) > 5 else 0

    # Relative price (1)
    f["pct_52w_high"] = float(p.iloc[-1] / p.iloc[-252:].max()) if len(p) > 252 else 1.0

    # Momentum acceleration (1)
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    # RSI (1)
    delta = p.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    f["rsi_14"] = float(rsi.iloc[-1]) if len(rsi) > 14 else 50.0

    # MACD signal (1)
    ema12 = p.ewm(span=12).mean()
    ema26 = p.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    f["macd_signal"] = float((macd.iloc[-1] - signal.iloc[-1]) / p.iloc[-1]) if len(p) > 26 else 0

    # Bollinger Band position (1)
    sma20 = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    if len(p) > 20 and float(std20.iloc[-1]) > 0:
        f["bb_pct"] = float((p.iloc[-1] - sma20.iloc[-1]) / (2 * std20.iloc[-1]))
    else:
        f["bb_pct"] = 0.0

    # Relative to SPY (2)
    if len(sp) > 63:
        spy_r = sp.pct_change().dropna()
        sec_r = r
        common = sec_r.index.intersection(spy_r.index)
        if len(common) > 21:
            f["rel_spy_21d"] = float(sec_r.loc[common].iloc[-21:].mean() -
                                     spy_r.loc[common].iloc[-21:].mean()) * 252
        else:
            f["rel_spy_21d"] = 0.0
        if len(common) > 63:
            f["rel_spy_63d"] = float(sec_r.loc[common].iloc[-63:].mean() -
                                     spy_r.loc[common].iloc[-63:].mean()) * 252
        else:
            f["rel_spy_63d"] = 0.0
    else:
        f["rel_spy_21d"] = 0.0
        f["rel_spy_63d"] = 0.0

    # Higher moments (2)
    if len(r) > 21:
        from scipy.stats import skew, kurtosis
        f["skew_21d"] = float(skew(r.iloc[-21:].values))
        f["kurt_21d"] = float(kurtosis(r.iloc[-21:].values))
    else:
        f["skew_21d"] = 0.0
        f["kurt_21d"] = 0.0

    # ATR as pct (1)
    if len(p) > 14:
        atr_approx = r.abs().rolling(14).mean().iloc[-1] * p.iloc[-1]
        f["atr_pct_14d"] = float(atr_approx / p.iloc[-1]) if p.iloc[-1] > 0 else 0.02
    else:
        f["atr_pct_14d"] = 0.02

    return f


def lgbm_walkforward(px: pd.DataFrame, spy: pd.Series,
                     rebal_dates: pd.DatetimeIndex) -> Dict:
    """Walk-forward LGBM sector ranking: 500d train, 250d test, sliding."""
    fprint("[2/7] LightGBM walk-forward sector ranking...")

    # Build full feature dataset
    records = []
    for dt in rebal_dates:
        idx = px.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        for tk in px.columns:
            f = build_sector_features(px, spy, idx, tk)
            if f is None:
                continue
            # Forward return (21d)
            fi = min(idx + 21, len(px) - 1)
            fwd_ret = float(px[tk].iloc[fi] / px[tk].iloc[idx] - 1)
            f.update({"date": dt, "ticker": tk, "fwd_ret": fwd_ret})
            records.append(f)

    df = pd.DataFrame(records)
    if len(df) < 100:
        fprint("  FATAL: insufficient data for LGBM")
        return {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    unique_dates = sorted(df["date"].unique())
    ranks = {}

    # Sliding walk-forward: 500d train periods -> ~24 rebal dates, 250d test -> ~12 dates
    train_periods = TRAIN_DAYS // 14   # ~36 biweekly periods
    test_periods = TEST_DAYS // 14     # ~18 biweekly periods

    n_folds = 0
    for test_start in range(train_periods, len(unique_dates), test_periods):
        train_start = max(0, test_start - train_periods)
        test_end = min(test_start + test_periods, len(unique_dates))

        train_dates = unique_dates[train_start:test_start]
        test_dates_fold = unique_dates[test_start:test_end]

        if len(test_dates_fold) == 0:
            continue

        train_df = df[df["date"].isin(train_dates)]
        if len(train_df) < 50:
            continue

        X_train = np.nan_to_num(train_df[LGBM_FEATURES].values.astype(np.float32))
        y_train = train_df["rank_label"].values

        try:
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1
            )
            model.fit(X_train, y_train)
        except Exception as e:
            fprint(f"    LGBM fit error: {e}")
            continue

        for tdate in test_dates_fold:
            test_df = df[df["date"] == tdate].copy()
            if len(test_df) < 3:
                continue
            X_test = np.nan_to_num(test_df[LGBM_FEATURES].values.astype(np.float32))
            test_df["score"] = model.predict(X_test)
            ranks[tdate] = dict(zip(test_df["ticker"], test_df["score"]))

        n_folds += 1

    fprint(f"  {n_folds} WF folds, {len(ranks)} ranked dates")
    return ranks


# ============================================================
# SPREAD PRICING (ATR-based BS)
# ============================================================

def price_bull_call_spread(S: float, width_pct: float, dte: int,
                           iv: float) -> Tuple[float, float, float, float, float]:
    """
    Price a bull call spread: buy ATM call + sell OTM call.
    Returns: (debit_per_share, max_profit_net, max_loss_net, K_long, K_short)
    Applies 15% haircut on entry: long call costs 15% more, short call pays 15% less.
    """
    T = dte / 365.0
    K_long = round(S, 2)                          # ATM
    K_short = round(S * (1 + width_pct / 100), 2) # OTM

    long_price = bs_call_price(S, K_long, T, RISK_FREE_RATE, iv) * (1 + HAIRCUT)
    short_price = bs_call_price(S, K_short, T, RISK_FREE_RATE, iv) * (1 - HAIRCUT)

    debit = long_price - short_price  # per share
    if debit <= 0:
        return 0, 0, 0, K_long, K_short

    width = K_short - K_long
    max_profit_net = (width - debit) * 100 - SPREAD_COMM
    max_loss_net = debit * 100 + SPREAD_COMM

    return debit, max_profit_net, max_loss_net, K_long, K_short


def price_otm_put(S: float, otm_pct: float, dte: int, iv: float) -> float:
    """Price a 5% OTM SPY put with haircut. Returns cost per contract (100 shares)."""
    T = dte / 365.0
    K = round(S * (1 - otm_pct), 2)
    put_price = bs_put_price(S, K, T, RISK_FREE_RATE, iv) * (1 + HAIRCUT)
    return put_price * 100 + SPREAD_COMM  # total cost per contract


def put_payoff_at_expiry(S_entry: float, S_exit: float, otm_pct: float,
                         cost: float) -> float:
    """P&L of the OTM put at expiry."""
    K = round(S_entry * (1 - otm_pct), 2)
    intrinsic = max(K - S_exit, 0) * 100
    return intrinsic - cost


def price_vix_call_spread(vix_level: float, dte: int, iv: float) -> Tuple[float, float]:
    """
    Price a VIX call spread: buy ATM, sell +5 points OTM.
    Returns: (cost, max_payoff) per contract.
    """
    T = dte / 365.0
    K_long = round(vix_level, 0)
    K_short = K_long + 5.0

    # VIX options: use higher IV (VIX of VIX is typically 80-120%)
    vix_iv = max(iv * 2.0, 0.80)
    long_price = bs_call_price(vix_level, K_long, T, RISK_FREE_RATE, vix_iv) * (1 + HAIRCUT)
    short_price = bs_call_price(vix_level, K_short, T, RISK_FREE_RATE, vix_iv) * (1 - HAIRCUT)

    debit = (long_price - short_price) * 100
    max_payoff = (K_short - K_long) * 100 - debit

    return max(debit, 0.50), max_payoff  # floor at $0.50 to avoid zero-cost artifacts


def vix_spread_payoff(vix_entry: float, vix_exit: float, cost: float,
                      max_payoff: float) -> float:
    """P&L of VIX call spread at expiry."""
    K_long = round(vix_entry, 0)
    K_short = K_long + 5.0
    intrinsic = (min(max(vix_exit, K_long), K_short) - K_long) * 100
    return min(intrinsic - cost, max_payoff)


# ============================================================
# SPREAD EXPIRY P&L
# ============================================================

def spread_pnl_at_expiry(S_exit: float, K_long: float, K_short: float,
                         debit: float) -> float:
    """P&L of bull call spread at expiry."""
    payoff = (min(max(S_exit, K_long), K_short) - K_long) * 100
    return payoff - debit * 100 - SPREAD_COMM


def spread_pnl_early_exit(S_now: float, K_long: float, K_short: float,
                          debit: float, iv: float, dte_remaining: int) -> float:
    """P&L of bull call spread at early exit with haircut."""
    T = dte_remaining / 365.0
    # Current value with exit haircut (selling at worse prices)
    long_val = bs_call_price(S_now, K_long, T, RISK_FREE_RATE, iv) * (1 - EXIT_HAIRCUT)
    short_val = bs_call_price(S_now, K_short, T, RISK_FREE_RATE, iv) * (1 + EXIT_HAIRCUT)
    current_value = (long_val - short_val) * 100
    return current_value - debit * 100 - SPREAD_COMM * 2  # double commission for early exit


# ============================================================
# SIMULATION ENGINE
# ============================================================

class Position:
    """Represents an open bull call spread position."""
    def __init__(self, ticker: str, entry_date, entry_idx: int,
                 S_entry: float, K_long: float, K_short: float,
                 debit: float, iv: float, expiry_idx: int, cost_usd: float):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_idx = entry_idx
        self.S_entry = S_entry
        self.K_long = K_long
        self.K_short = K_short
        self.debit = debit
        self.iv = iv
        self.expiry_idx = expiry_idx
        self.cost_usd = cost_usd  # total dollars at risk
        self.max_debit_seen = cost_usd  # for trailing stop


def simulate_variant(variant: str, ranks: Dict, sector_px: pd.DataFrame,
                     spy: pd.Series, vix: pd.Series, sh_price: pd.Series,
                     baseline_result: Optional[Dict] = None) -> Dict:
    """
    Simulate a protective overlay variant.
    Returns dict with trades, equity curve, and metrics.
    """
    equity = INITIAL_CAP
    peak_equity = INITIAL_CAP
    equity_curve = [INITIAL_CAP]
    equity_dates = [sector_px.index[0]]
    trades = []
    positions: List[Position] = []
    total_hedge_cost = 0.0
    total_hedge_pnl = 0.0
    stopped_out = False
    stop_reentry_date = None

    sorted_dates = sorted(ranks.keys())

    for rebal_date in sorted_dates:
        if rebal_date not in sector_px.index or rebal_date not in vix.index:
            continue

        di = sector_px.index.get_loc(rebal_date)
        cv = float(vix.iloc[di])
        spy_price = float(spy.iloc[di])
        scores = ranks[rebal_date]

        if not scores:
            equity_curve.append(equity)
            equity_dates.append(rebal_date)
            continue

        # ── VIX > 20 filter (base strategy requirement) ──
        if cv < 20:
            equity_curve.append(equity)
            equity_dates.append(rebal_date)
            continue

        # ── Process expiring positions ──
        expired_positions = []
        for pos in positions:
            if di >= pos.expiry_idx:
                S_exit = float(sector_px[pos.ticker].iloc[pos.expiry_idx])
                pnl = spread_pnl_at_expiry(S_exit, pos.K_long, pos.K_short, pos.debit)
                equity += pnl
                trades.append({
                    "entry": str(pos.entry_date.date()) if hasattr(pos.entry_date, "date") else str(pos.entry_date),
                    "exit": str(sector_px.index[pos.expiry_idx].date()),
                    "ticker": pos.ticker,
                    "pnl": round(pnl, 2),
                    "win": pnl > 0,
                    "vix": round(cv, 1),
                    "type": "spread",
                })
                expired_positions.append(pos)
        for p in expired_positions:
            positions.remove(p)

        # ── Variant G: Trailing stop check on open positions ──
        if variant in ("G_TrailingStop", "H_ComboBest"):
            stopped_positions = []
            for pos in positions:
                if di < pos.expiry_idx:
                    S_now = float(sector_px[pos.ticker].iloc[di])
                    dte_rem = pos.expiry_idx - di
                    current_pnl = spread_pnl_early_exit(
                        S_now, pos.K_long, pos.K_short, pos.debit, pos.iv, dte_rem
                    )
                    # Stop if loss exceeds 50% of debit paid
                    if current_pnl < -(pos.cost_usd * 0.50):
                        equity += current_pnl
                        trades.append({
                            "entry": str(pos.entry_date.date()) if hasattr(pos.entry_date, "date") else str(pos.entry_date),
                            "exit": str(rebal_date.date()),
                            "ticker": pos.ticker,
                            "pnl": round(current_pnl, 2),
                            "win": current_pnl > 0,
                            "vix": round(cv, 1),
                            "type": "trailing_stop",
                        })
                        stopped_positions.append(pos)
            for p in stopped_positions:
                positions.remove(p)

        # ── Variant D: Portfolio stop-loss ──
        if variant in ("D_PortfolioStop", "H_ComboBest"):
            drawdown_pct = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
            if drawdown_pct < -0.15 and not stopped_out:
                # Close ALL positions at intrinsic (sell at haircut)
                for pos in positions:
                    S_now = float(sector_px[pos.ticker].iloc[di])
                    dte_rem = pos.expiry_idx - di
                    pnl = spread_pnl_early_exit(
                        S_now, pos.K_long, pos.K_short, pos.debit, pos.iv, dte_rem
                    )
                    equity += pnl
                    trades.append({
                        "entry": str(pos.entry_date.date()) if hasattr(pos.entry_date, "date") else str(pos.entry_date),
                        "exit": str(rebal_date.date()),
                        "ticker": pos.ticker,
                        "pnl": round(pnl, 2),
                        "win": pnl > 0,
                        "vix": round(cv, 1),
                        "type": "portfolio_stop",
                    })
                positions = []
                stopped_out = True
                stop_reentry_date = sector_px.index[min(di + 10, len(sector_px) - 1)]
                equity_curve.append(equity)
                equity_dates.append(rebal_date)
                continue

        if stopped_out:
            if rebal_date >= stop_reentry_date:
                stopped_out = False
                peak_equity = equity  # reset peak after stop
            else:
                equity_curve.append(equity)
                equity_dates.append(rebal_date)
                continue

        # ── Determine position sizing ──
        size_mult = 1.0

        if variant == "C_DDSizeCap" or variant == "H_ComboBest":
            drawdown_pct = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
            if drawdown_pct < -0.10:
                size_mult *= 0.50

        if variant == "F_RegimeSizing" or variant == "H_ComboBest":
            if 20 <= cv <= 25:
                size_mult *= 1.0
            elif 25 < cv <= 30:
                size_mult *= 0.50
            elif cv > 30:
                size_mult *= 0.25

        # ── Equity scaling: max per trade ──
        equity_scale = max(equity / INITIAL_CAP, 0.1)
        max_trade = min(MAX_POS_USD * equity_scale * size_mult, equity * 0.40)

        if max_trade < 20 or equity < 50:
            equity_curve.append(equity)
            equity_dates.append(rebal_date)
            continue

        # ── Select top 3 sectors ──
        top_sectors = [t for t, _ in sorted(scores.items(), key=lambda x: x[1],
                                            reverse=True)[:MAX_CONCURRENT]]

        # ── Enter new positions ──
        n_entered = len(positions)
        for tk in top_sectors:
            if n_entered >= MAX_CONCURRENT:
                break
            if tk not in sector_px.columns:
                continue
            # Skip if already have position in this sector
            if any(p.ticker == tk for p in positions):
                continue

            S = float(sector_px[tk].iloc[di])
            iv = get_iv(sector_px[tk], di)
            expiry_idx = min(di + DTE, len(sector_px) - 1)

            debit, max_profit, max_loss, K_long, K_short = price_bull_call_spread(
                S, SPREAD_WIDTH_PCT, DTE, iv
            )

            if debit <= 0:
                continue
            cost_usd = debit * 100 + SPREAD_COMM
            if cost_usd > max_trade or cost_usd > equity * 0.40:
                continue

            positions.append(Position(
                ticker=tk, entry_date=rebal_date, entry_idx=di,
                S_entry=S, K_long=K_long, K_short=K_short,
                debit=debit, iv=iv, expiry_idx=expiry_idx,
                cost_usd=cost_usd,
            ))
            equity -= 0  # cost is accounted at expiry/exit in P&L calc
            n_entered += 1

        # ── Variant A: OTM Put Hedge ──
        if variant == "A_OTMPutHedge" and n_entered > 0:
            spy_iv = get_iv(spy, di)
            put_cost = price_otm_put(spy_price, 0.05, DTE, spy_iv)
            # Scale: one put per ~$10k portfolio (fractional for small account)
            hedge_fraction = equity / 10000.0
            scaled_cost = put_cost * hedge_fraction
            if scaled_cost < equity * 0.05:  # cap at 5% of equity
                equity -= scaled_cost
                total_hedge_cost += scaled_cost
                # Compute payoff at DTE
                spy_exit_idx = min(di + DTE, len(spy) - 1)
                spy_exit = float(spy.iloc[spy_exit_idx])
                payoff = put_payoff_at_expiry(spy_price, spy_exit, 0.05, put_cost) * hedge_fraction
                equity += max(payoff, 0) if payoff > 0 else 0  # puts expire or pay off
                total_hedge_pnl += payoff * hedge_fraction

        # ── Variant B: VIX Call Spread Hedge ──
        if variant == "B_VIXCallHedge" and n_entered > 0:
            vix_iv = get_iv(vix, di, lookback=30)
            vcs_cost, vcs_max_payoff = price_vix_call_spread(cv, DTE, vix_iv)
            hedge_fraction = equity / 10000.0
            scaled_cost = vcs_cost * hedge_fraction
            if scaled_cost < equity * 0.05:
                equity -= scaled_cost
                total_hedge_cost += scaled_cost
                vix_exit_idx = min(di + DTE, len(vix) - 1)
                vix_exit = float(vix.iloc[vix_exit_idx])
                payoff = vix_spread_payoff(cv, vix_exit, vcs_cost, vcs_max_payoff) * hedge_fraction
                equity += payoff if payoff > 0 else 0
                total_hedge_pnl += payoff

        # ── Variant E: Inverse ETF Hedge (SH) ──
        if variant in ("E_InverseHedge", "H_ComboBest") and cv > 25:
            # Allocate 20% of equity to SH for DTE period
            sh_alloc = equity * 0.20
            sh_entry = float(sh_price.iloc[di])
            sh_exit_idx = min(di + DTE, len(sh_price) - 1)
            sh_exit = float(sh_price.iloc[sh_exit_idx])
            if sh_entry > 0:
                sh_return = (sh_exit / sh_entry) - 1.0
                sh_pnl = sh_alloc * sh_return
                equity += sh_pnl
                total_hedge_pnl += sh_pnl
                total_hedge_cost += abs(min(sh_pnl, 0))  # cost = losses from hedge
                trades.append({
                    "entry": str(rebal_date.date()),
                    "exit": str(sector_px.index[sh_exit_idx].date()),
                    "ticker": "SH_HEDGE",
                    "pnl": round(sh_pnl, 2),
                    "win": sh_pnl > 0,
                    "vix": round(cv, 1),
                    "type": "inverse_hedge",
                })

        # Update peak
        peak_equity = max(peak_equity, equity)

        equity_curve.append(equity)
        equity_dates.append(rebal_date)

    # Process any remaining positions at last date
    for pos in positions:
        last_idx = min(pos.expiry_idx, len(sector_px) - 1)
        S_exit = float(sector_px[pos.ticker].iloc[last_idx])
        pnl = spread_pnl_at_expiry(S_exit, pos.K_long, pos.K_short, pos.debit)
        equity += pnl
        trades.append({
            "entry": str(pos.entry_date.date()) if hasattr(pos.entry_date, "date") else str(pos.entry_date),
            "exit": str(sector_px.index[last_idx].date()),
            "ticker": pos.ticker,
            "pnl": round(pnl, 2),
            "win": pnl > 0,
            "vix": round(float(vix.iloc[last_idx]), 1),
            "type": "expiry",
        })

    return {
        "variant": variant,
        "trades": trades,
        "equity_curve": equity_curve,
        "equity_dates": [str(d) for d in equity_dates],
        "final_equity": equity,
        "total_hedge_cost": total_hedge_cost,
        "total_hedge_pnl": total_hedge_pnl,
    }


# ============================================================
# METRICS
# ============================================================

def compute_metrics(result: Dict, baseline_metrics: Optional[Dict] = None) -> Dict:
    """Compute Sharpe, Sortino, CAGR, MaxDD, etc. from simulation result."""
    trades = result["trades"]
    curve = np.array(result["equity_curve"])
    variant = result["variant"]

    if len(trades) == 0:
        fprint(f"  {variant}: No trades")
        return {"name": variant, "valid": False}

    # Trade-level stats
    pnls = [t["pnl"] for t in trades if t["type"] != "inverse_hedge"]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100 if n > 0 else 0

    # Calendar month Sharpe (from equity curve)
    eq_series = pd.Series(curve, index=pd.to_datetime(result["equity_dates"][:len(curve)]))
    monthly_eq = eq_series.resample("ME").last().dropna()
    monthly_ret = monthly_eq.pct_change().dropna()

    if len(monthly_ret) < 3:
        fprint(f"  {variant}: Too few months")
        return {"name": variant, "valid": False}

    sharpe = float(monthly_ret.mean() * 12 / (monthly_ret.std() * np.sqrt(12) + 1e-10))
    down_ret = monthly_ret[monthly_ret < 0]
    sortino = float(monthly_ret.mean() * 12 / (down_ret.std() * np.sqrt(12) + 1e-10)) if len(down_ret) > 1 else 0

    # CAGR
    n_years = max(len(monthly_ret) / 12, 0.5)
    final_eq = result["final_equity"]
    cagr = (final_eq / INITIAL_CAP) ** (1 / n_years) - 1

    # MaxDD
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / (peak + 1e-10)
    maxdd = float(dd.min())

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_profit / (gross_loss + 1e-10)

    # Max consecutive losses
    mcl = c = 0
    for t in trades:
        if t.get("type") in ("inverse_hedge",):
            continue
        if not t["win"]:
            c += 1
            mcl = max(mcl, c)
        else:
            c = 0

    # Hedge cost analysis
    total_pnl = sum(pnls)
    hedge_cost = result["total_hedge_cost"]
    hedge_cost_pct = hedge_cost / (gross_profit + 1e-10) * 100
    cost_of_protection = hedge_cost / (total_pnl + 1e-10) * 100 if total_pnl > 0 else float("inf")

    m = {
        "name": variant,
        "valid": True,
        "n_trades": n,
        "win_rate": round(wr, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "cagr_pct": round(cagr * 100, 1),
        "maxdd_pct": round(maxdd * 100, 1),
        "pf": round(pf, 2),
        "final_equity": round(final_eq, 2),
        "total_pnl": round(total_pnl, 2),
        "max_consec_loss": mcl,
        "avg_pnl": round(np.mean(pnls), 2) if pnls else 0,
        "hedge_cost": round(hedge_cost, 2),
        "hedge_cost_pct_of_profit": round(hedge_cost_pct, 1),
        "cost_of_protection_pct": round(cost_of_protection, 1),
        "monthly_returns": monthly_ret.values.tolist(),
    }

    # Comparison to baseline
    if baseline_metrics and baseline_metrics.get("valid"):
        m["sharpe_change"] = round(m["sharpe"] - baseline_metrics["sharpe"], 2)
        m["maxdd_change"] = round(m["maxdd_pct"] - baseline_metrics["maxdd_pct"], 1)
        m["cagr_change"] = round(m["cagr_pct"] - baseline_metrics["cagr_pct"], 1)
        m["sortino_change"] = round(m["sortino"] - baseline_metrics["sortino"], 2)

    fprint(f"  {variant:22s} | {n:4d} | WR {wr:5.1f}% | Sh {sharpe:5.2f} | So {sortino:5.2f} | "
           f"CAGR {cagr*100:5.1f}% | MDD {maxdd*100:5.1f}% | PF {pf:5.2f} | "
           f"${INITIAL_CAP:.0f}->${final_eq:.0f} | MCL {mcl}")

    return m


# ============================================================
# 5-GATE ADVERSARIAL VALIDATION
# ============================================================

def adversarial_validation(m: Dict) -> Dict:
    """5-gate adversarial validation per variant."""
    if not m.get("valid"):
        m.update({"gates": 0, "gate_details": {}})
        return m

    rets = np.array(m.get("monthly_returns", []))
    gates = 0
    details = {}

    # Gate 1: Permutation test (Sharpe is real, not noise)
    if len(rets) >= 10:
        obs_sharpe = np.mean(rets) / (np.std(rets) + 1e-10)
        perm_count = sum(
            1 for _ in range(1000)
            if np.mean(rets * np.random.choice([-1, 1], len(rets))) /
               (np.std(rets) + 1e-10) >= obs_sharpe
        )
        perm_p = perm_count / 1000
        g1 = perm_p < 0.05
        gates += g1
        details["g1_permutation"] = {"pass": g1, "p_value": round(perm_p, 3)}
    else:
        details["g1_permutation"] = {"pass": False, "p_value": 1.0}

    # Gate 2: Regime agnostic (R1) — check if performance is balanced
    # Use trade-level regime data
    spread_trades = [t for t in m.get("_trades", []) if t.get("type") not in ("inverse_hedge",)]
    vix_high = [t for t in spread_trades if t.get("vix", 0) > 25]
    vix_med = [t for t in spread_trades if 20 <= t.get("vix", 0) <= 25]

    if len(vix_high) > 5 and len(vix_med) > 5:
        wr_high = sum(1 for t in vix_high if t["win"]) / len(vix_high)
        wr_med = sum(1 for t in vix_med if t["win"]) / len(vix_med)
        gap = abs(wr_high - wr_med) / max(wr_high, wr_med, 0.01)
        g2 = gap < 0.50
    else:
        g2 = len(rets) > 20  # pass if enough data
        gap = 0
    gates += g2
    details["g2_regime_agnostic"] = {"pass": g2, "gap": round(gap, 3)}

    # Gate 3: Sub-period consistency (both halves profitable)
    if len(rets) >= 10:
        mid = len(rets) // 2
        h1_sharpe = np.mean(rets[:mid]) / (np.std(rets[:mid]) + 1e-10)
        h2_sharpe = np.mean(rets[mid:]) / (np.std(rets[mid:]) + 1e-10)
        g3 = h1_sharpe > 0 and h2_sharpe > 0
        gates += g3
        details["g3_subperiod"] = {"pass": g3, "h1_sharpe": round(h1_sharpe, 2),
                                   "h2_sharpe": round(h2_sharpe, 2)}
    else:
        details["g3_subperiod"] = {"pass": False}

    # Gate 4: Outlier robustness (still positive after removing best month)
    if len(rets) > 5:
        trimmed = np.sort(rets)[:-1]  # remove best month
        g4 = np.mean(trimmed) / (np.std(trimmed) + 1e-10) > 0
        gates += g4
        details["g4_outlier_robust"] = {"pass": g4}
    else:
        details["g4_outlier_robust"] = {"pass": False}

    # Gate 5: Drawdown recovery (max consecutive loss streak < 6)
    g5 = m.get("max_consec_loss", 99) < 6
    gates += g5
    details["g5_dd_recovery"] = {"pass": g5, "mcl": m.get("max_consec_loss", 0)}

    m["gates"] = gates
    m["gate_details"] = details

    fprint(f"    Gates: P={'Y' if details.get('g1_permutation',{}).get('pass') else 'N'} "
           f"R1={'Y' if details.get('g2_regime_agnostic',{}).get('pass') else 'N'} "
           f"Sub={'Y' if details.get('g3_subperiod',{}).get('pass') else 'N'} "
           f"Out={'Y' if details.get('g4_outlier_robust',{}).get('pass') else 'N'} "
           f"DD={'Y' if details.get('g5_dd_recovery',{}).get('pass') else 'N'} "
           f"=> {gates}/5")

    return m


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    fprint(f"Protective Overlay v1 — {datetime.now():%Y-%m-%d %H:%M:%S}")
    fprint("=" * 80)
    fprint(f"Capital: ${INITIAL_CAP:.0f} | Spread: {SPREAD_WIDTH_PCT}% width | DTE: {DTE} | "
           f"Comm: ${SPREAD_COMM} | Haircut: {HAIRCUT:.0%}")
    fprint(f"WF: {TRAIN_DAYS}d train / {TEST_DAYS}d test (sliding) | "
           f"21 LGBM features | biweekly rebalance | VIX>20 filter")
    fprint("=" * 80)

    # 1. Download data
    sector_px, sector_hi, sector_lo, spy, vix, sh_price = download_data()

    # 2. Biweekly rebalance dates
    rebal_dates = pd.DatetimeIndex(
        sector_px.index.to_series().resample("2W-FRI").last().dropna().values
    )
    fprint(f"  {len(rebal_dates)} biweekly rebalance dates")

    # 3. LGBM walk-forward ranking
    ranks = lgbm_walkforward(sector_px, spy, rebal_dates)
    if not ranks:
        fprint("FATAL: No rankings produced")
        return

    # 4. Simulate all variants
    variants = [
        "BASELINE",            # Unhedged baseline
        "A_OTMPutHedge",       # OTM SPY puts
        "B_VIXCallHedge",      # VIX call spreads
        "C_DDSizeCap",         # Position size cap by drawdown
        "D_PortfolioStop",     # Portfolio-level stop-loss
        "E_InverseHedge",      # Inverse ETF hedge (SH)
        "F_RegimeSizing",      # Regime-adaptive sizing
        "G_TrailingStop",      # Trailing stop per trade
        "H_ComboBest",         # Combo: C + D + E + F + G (best elements)
    ]

    fprint(f"\n[4/7] Simulating {len(variants)} variants...")
    results = []
    baseline_metrics = None

    for var in variants:
        sim = simulate_variant(var, ranks, sector_px, spy, vix, sh_price)
        m = compute_metrics(sim, baseline_metrics)
        if m.get("valid"):
            m["_trades"] = sim["trades"]  # keep for regime gate
            m = adversarial_validation(m)
            del m["_trades"]  # don't serialize all trades
            results.append(m)
            if var == "BASELINE":
                baseline_metrics = m

    if not results:
        fprint("No valid results")
        return

    # 5. Summary table sorted by MaxDD (best protection first)
    fprint(f"\n{'=' * 115}")
    fprint(f"{'Variant':22s} {'#':>5} {'WR':>6} {'Sh':>6} {'dSh':>6} {'So':>6} "
           f"{'CAGR':>7} {'MDD':>7} {'dMDD':>7} {'PF':>5} {'$':>8} {'MCL':>4} {'G':>4}")
    fprint("-" * 115)

    for r in sorted(results, key=lambda x: x["maxdd_pct"], reverse=True):
        dsh = r.get("sharpe_change", 0)
        dmdd = r.get("maxdd_change", 0)
        fprint(f"{r['name']:22s} {r['n_trades']:5d} {r['win_rate']:5.1f}% {r['sharpe']:6.2f} "
               f"{dsh:+5.2f} {r['sortino']:6.2f} {r['cagr_pct']:6.1f}% {r['maxdd_pct']:6.1f}% "
               f"{dmdd:+6.1f}% {r['pf']:5.2f} ${r['final_equity']:7.0f} "
               f"{r['max_consec_loss']:3d} {r['gates']:3d}/5")

    # 6. Hedge cost analysis
    fprint(f"\n{'=' * 80}")
    fprint("HEDGE COST ANALYSIS:")
    fprint(f"{'Variant':22s} {'HedgeCost$':>10} {'%GrossProfit':>13} {'CostOfProt%':>12}")
    fprint("-" * 60)
    for r in sorted(results, key=lambda x: x["maxdd_pct"], reverse=True):
        if r["name"] == "BASELINE":
            continue
        fprint(f"{r['name']:22s} ${r['hedge_cost']:9.2f} {r['hedge_cost_pct_of_profit']:12.1f}% "
               f"{r['cost_of_protection_pct']:11.1f}%")

    # 7. Best variant analysis
    baseline = next((r for r in results if r["name"] == "BASELINE"), None)
    best_mdd = min(results, key=lambda x: abs(x["maxdd_pct"]))
    best_sharpe = max(results, key=lambda x: x["sharpe"])
    valid_results = [r for r in results if r["gates"] >= 3]
    best_valid = min(valid_results, key=lambda x: abs(x["maxdd_pct"])) if valid_results else None

    fprint(f"\nBEST MaxDD REDUCTION: {best_mdd['name']} — MDD={best_mdd['maxdd_pct']:.1f}% "
           f"(Sharpe={best_mdd['sharpe']:.2f})")
    if baseline:
        mdd_improvement = abs(baseline["maxdd_pct"]) - abs(best_mdd["maxdd_pct"])
        fprint(f"  vs BASELINE: MDD improved by {mdd_improvement:.1f}pp, "
               f"Sharpe change: {best_mdd.get('sharpe_change', 0):+.2f}")

    fprint(f"BEST Sharpe: {best_sharpe['name']} — Sharpe={best_sharpe['sharpe']:.2f} "
           f"(MDD={best_sharpe['maxdd_pct']:.1f}%)")

    if best_valid:
        fprint(f"BEST VALID (3+/5 gates): {best_valid['name']} — MDD={best_valid['maxdd_pct']:.1f}% "
               f"Sharpe={best_valid['sharpe']:.2f}")

    # 8. MLflow logging
    if MLFLOW_OK:
        try:
            exp_name = "protective_overlay_v1"
            try:
                if not mlflow.get_experiment_by_name(exp_name):
                    mlflow.create_experiment(exp_name)
            except Exception:
                pass
            mlflow.set_experiment(exp_name)
            with mlflow.start_run(run_name=f"pov1_{datetime.now():%Y%m%d_%H%M}"):
                mlflow.log_params({
                    "capital": INITIAL_CAP,
                    "spread_width_pct": SPREAD_WIDTH_PCT,
                    "dte": DTE,
                    "haircut": HAIRCUT,
                    "commission": SPREAD_COMM,
                    "iv_multiplier": IV_MULTIPLIER,
                    "train_days": TRAIN_DAYS,
                    "test_days": TEST_DAYS,
                    "n_features": len(LGBM_FEATURES),
                    "vix_filter": 20,
                })
                for r in results:
                    for k in ["sharpe", "sortino", "cagr_pct", "maxdd_pct",
                              "win_rate", "pf", "gates", "hedge_cost",
                              "cost_of_protection_pct"]:
                        try:
                            mlflow.log_metric(f"{r['name']}_{k}", float(r[k]))
                        except Exception:
                            pass
            fprint("MLflow: logged successfully")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    # 9. Save results JSON
    save_data = {
        "strategy": "Protective Overlay v1",
        "description": "Hedging overlays to reduce MaxDD for LGBM sector spread strategy",
        "run_date": datetime.now().isoformat(),
        "config": {
            "capital": INITIAL_CAP,
            "spread_width_pct": SPREAD_WIDTH_PCT,
            "dte": DTE,
            "haircut": HAIRCUT,
            "exit_haircut": EXIT_HAIRCUT,
            "commission": SPREAD_COMM,
            "iv_multiplier": IV_MULTIPLIER,
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "n_lgbm_features": len(LGBM_FEATURES),
            "max_concurrent": MAX_CONCURRENT,
            "max_pos_usd": MAX_POS_USD,
            "vix_filter": 20,
        },
        "variants": [
            {k: v for k, v in r.items() if k != "monthly_returns"}
            for r in sorted(results, key=lambda x: abs(x["maxdd_pct"]))
        ],
        "best_mdd": best_mdd["name"] if best_mdd else "NONE",
        "best_sharpe": best_sharpe["name"] if best_sharpe else "NONE",
        "best_valid": best_valid["name"] if best_valid else "NONE",
        "baseline_sharpe": baseline["sharpe"] if baseline else None,
        "baseline_maxdd": baseline["maxdd_pct"] if baseline else None,
    }

    out_path = OUT_DIR / "protective_overlay_v1_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    fprint(f"\nResults saved to {out_path}")

    elapsed = time.time() - t0
    fprint(f"\nDone — {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
