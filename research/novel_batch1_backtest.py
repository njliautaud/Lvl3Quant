#!/usr/bin/env python3
"""
Novel Strategy Batch 1 — 6 strategies × 6 variants = 36 backtests
Credit Spread Velocity, Dollar Strength Reversal, Yield Curve Velocity,
Overnight Return Anomaly, VVIX/Vol-of-Vol, Systematic Put Selling

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation p < 0.05 (1000 shuffles)
  3. Regime gap < 0.50
  4. Max DD < 30%
  5. Min 20 trades
"""

import json, warnings, sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────
START = "2020-01-01"
END = "2026-07-31"
CAPITAL = 645.0
POS_SIZE = 200.0
MAX_CONCURRENT = 3
HOLD_DAYS = 10
COMMISSION_PCT = 0.001  # realistic equity commission
N_PERM = 1000

QUALITY_PATH = "/home/jupiter/Lvl3Quant/data/quality_universe.json"
OUTPUT_PATH = "/home/jupiter/Lvl3Quant/data/novel_batch1_results.json"

with open(QUALITY_PATH) as f:
    UNIVERSE = json.load(f)["tickers"]

# International-revenue-heavy subset for Strategy 2C
INTL_HEAVY = ["AAPL", "MSFT", "AVGO", "META", "GOOGL", "NVDA", "AMD", "NFLX", "CRM", "INTU"]

# ── Data Download ───────────────────────────────────────────────────────
print("Downloading data...")

def download(tickers, start=START, end=END):
    """Download daily OHLCV for a list of tickers."""
    all_data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) > 100:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                all_data[t] = df
        except Exception:
            pass
    return all_data

# Download everything we need
all_tickers = list(set(UNIVERSE + ["SPY", "HYG", "LQD", "UUP", "TLT", "SHY", "^VIX", "^VVIX"]))
stock_data = download(all_tickers)

# Rename VIX/VVIX
if "^VIX" in stock_data:
    stock_data["VIX"] = stock_data.pop("^VIX")
if "^VVIX" in stock_data:
    stock_data["VVIX"] = stock_data.pop("^VVIX")

print(f"Downloaded {len(stock_data)} tickers, universe has {len([t for t in UNIVERSE if t in stock_data])} quality stocks")

# ── SPY regime (bull/bear = green/red day) ──────────────────────────────
spy = stock_data.get("SPY")
if spy is None:
    print("FATAL: SPY data missing"); sys.exit(1)

spy_ret = spy["Close"].pct_change()
bull_days = set(spy_ret[spy_ret > 0].index)
bear_days = set(spy_ret[spy_ret <= 0].index)

# ── Indicator Helpers ───────────────────────────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)

def pct_from_high(series, window=20):
    """How far below rolling high (negative = below)."""
    rh = series.rolling(window).max()
    return (series - rh) / (rh + 1e-10)

def sma(series, window):
    return series.rolling(window).mean()

def zscore(series, window=60):
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / (s + 1e-10)

# ── Precompute Indicators ──────────────────────────────────────────────
# Credit spread = HYG yield proxy - LQD yield proxy.
# Since HYG/LQD are bond ETFs, falling price = rising yield.
# Credit spread widens when HYG falls more than LQD.
# We use LQD/HYG ratio: rising = spread tightening (risk-on)
hyg = stock_data.get("HYG")
lqd = stock_data.get("LQD")
credit_ratio = None
if hyg is not None and lqd is not None:
    # Use HYG - LQD price ratio as proxy (higher = tighter spread = risk-on)
    cr = hyg["Close"] / lqd["Close"]
    cr = cr.dropna()
    credit_ratio = cr

uup = stock_data.get("UUP")
tlt = stock_data.get("TLT")
shy = stock_data.get("SHY")
vix_df = stock_data.get("VIX")
vvix_df = stock_data.get("VVIX")

# Yield curve proxy: TLT/SHY ratio
curve_ratio = None
if tlt is not None and shy is not None:
    curve_ratio = (tlt["Close"] / shy["Close"]).dropna()

# ── Backtest Engine ─────────────────────────────────────────────────────
def run_backtest(signal_func, hold_days=HOLD_DAYS, use_overnight=False):
    """
    Generic backtest engine.
    signal_func(date, ticker, data_dict) -> bool (True = enter long)

    If use_overnight=True: buy at close, sell at next open (1 day hold).
    Otherwise: buy at next open, sell after hold_days at open.

    Returns dict with metrics.
    """
    # Collect all signals
    signals = []

    # Get common dates from SPY
    spy_dates = spy.index.tolist()

    for ticker in UNIVERSE:
        if ticker not in stock_data:
            continue
        tdata = stock_data[ticker]
        dates = tdata.index
        for i, dt in enumerate(dates):
            if dt < pd.Timestamp(START) or dt > pd.Timestamp(END):
                continue
            try:
                if signal_func(dt, ticker, stock_data):
                    signals.append((dt, ticker))
            except Exception:
                continue

    if not signals:
        return None

    # Sort by date
    signals.sort(key=lambda x: x[0])

    # Execute trades with position limits
    trades = []
    active_positions = []  # list of (exit_date, ticker)

    for dt, ticker in signals:
        tdata = stock_data[ticker]

        # Remove expired positions
        active_positions = [(ed, tk) for ed, tk in active_positions if ed > dt]

        if len(active_positions) >= MAX_CONCURRENT:
            continue

        # Already in this ticker?
        if any(tk == ticker for _, tk in active_positions):
            continue

        if use_overnight:
            # Buy at today's close, sell at tomorrow's open
            idx = tdata.index.get_loc(dt) if dt in tdata.index else None
            if idx is None or idx + 1 >= len(tdata):
                continue
            entry_price = tdata["Close"].iloc[idx]
            exit_price = tdata["Open"].iloc[idx + 1]
            exit_dt = tdata.index[idx + 1]

            shares = int(POS_SIZE / entry_price)
            if shares < 1:
                continue

            pnl = shares * (exit_price - entry_price) - 2 * COMMISSION_PCT * POS_SIZE
            ret = pnl / POS_SIZE

            trades.append({
                "entry_date": dt,
                "exit_date": exit_dt,
                "ticker": ticker,
                "entry": entry_price,
                "exit": exit_price,
                "pnl": pnl,
                "return": ret,
            })
            active_positions.append((exit_dt, ticker))
        else:
            # Buy at next day's open
            idx = tdata.index.get_loc(dt) if dt in tdata.index else None
            if idx is None or idx + 1 >= len(tdata) or idx + 1 + hold_days >= len(tdata):
                continue

            entry_price = tdata["Open"].iloc[idx + 1]
            exit_idx = min(idx + 1 + hold_days, len(tdata) - 1)
            exit_price = tdata["Open"].iloc[exit_idx]
            exit_dt = tdata.index[exit_idx]

            shares = int(POS_SIZE / entry_price)
            if shares < 1:
                continue

            pnl = shares * (exit_price - entry_price) - 2 * COMMISSION_PCT * POS_SIZE
            ret = pnl / POS_SIZE

            trades.append({
                "entry_date": dt,
                "exit_date": exit_dt,
                "ticker": ticker,
                "entry": entry_price,
                "exit": exit_price,
                "pnl": pnl,
                "return": ret,
            })
            active_positions.append((exit_dt, ticker))

    return compute_metrics(trades)


def compute_metrics(trades):
    """Compute all metrics from trade list."""
    if not trades or len(trades) < 5:
        return None

    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    returns = df["return"].values
    n_trades = len(returns)

    if n_trades < 5:
        return None

    # Basic metrics
    win_rate = np.mean(returns > 0)
    avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
    avg_loss = np.mean(returns[returns < 0]) if np.any(returns < 0) else 0
    profit_factor = (np.sum(returns[returns > 0]) / (-np.sum(returns[returns < 0]) + 1e-10)) if np.any(returns < 0) else 99.0

    # Equity curve for Sharpe/Sortino/MDD
    equity = CAPITAL
    eq_curve = [CAPITAL]
    for r in returns:
        equity += POS_SIZE * r
        eq_curve.append(equity)

    eq_curve = np.array(eq_curve)
    total_return = (eq_curve[-1] / eq_curve[0]) - 1

    # MDD
    peak = np.maximum.accumulate(eq_curve)
    dd = (eq_curve - peak) / peak
    mdd = abs(dd.min())

    # Annualized Sharpe (using trade returns)
    # Estimate trades per year
    date_range = (df["entry_date"].max() - df["entry_date"].min()).days
    if date_range < 30:
        return None
    trades_per_year = n_trades / (date_range / 365.25)

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    sharpe = (mean_ret / (std_ret + 1e-10)) * np.sqrt(trades_per_year) if std_ret > 1e-10 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / (downside_std + 1e-10)) * np.sqrt(trades_per_year) if downside_std > 1e-10 else 0

    # Regime analysis
    bull_rets = []
    bear_rets = []
    for _, row in df.iterrows():
        if row["entry_date"] in bull_days:
            bull_rets.append(row["return"])
        else:
            bear_rets.append(row["return"])

    bull_sharpe = 0
    bear_sharpe = 0
    if len(bull_rets) > 3:
        br = np.array(bull_rets)
        bull_sharpe = (np.mean(br) / (np.std(br, ddof=1) + 1e-10)) * np.sqrt(max(1, len(br)))
    if len(bear_rets) > 3:
        ber = np.array(bear_rets)
        bear_sharpe = (np.mean(ber) / (np.std(ber, ddof=1) + 1e-10)) * np.sqrt(max(1, len(ber)))

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-10)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    # Permutation test
    real_sharpe = sharpe
    perm_count = 0
    for _ in range(N_PERM):
        shuffled = returns.copy()
        np.random.shuffle(shuffled)
        s_mean = np.mean(shuffled)
        s_std = np.std(shuffled, ddof=1)
        s_sharpe = (s_mean / (s_std + 1e-10)) * np.sqrt(trades_per_year) if s_std > 1e-10 else 0
        if s_sharpe >= real_sharpe:
            perm_count += 1
    perm_p = perm_count / N_PERM

    # 5-gate validation
    g1 = sharpe > 0.5
    g2 = perm_p < 0.05
    g3 = regime_gap < 0.50
    g4 = mdd < 0.30
    g5 = n_trades >= 20
    gates_passed = sum([g1, g2, g3, g4, g5])

    return {
        "n_trades": int(n_trades),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(min(profit_factor, 99.0), 3),
        "max_dd": round(mdd, 4),
        "total_return": round(total_return, 4),
        "final_equity": round(eq_curve[-1], 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 4),
        "perm_p": round(perm_p, 4),
        "gate1_sharpe": g1,
        "gate2_perm": g2,
        "gate3_regime": g3,
        "gate4_mdd": g4,
        "gate5_trades": g5,
        "gates_passed": gates_passed,
        "pass_all_5": gates_passed == 5,
    }


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 1: CREDIT SPREAD VELOCITY
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Strategy 1: Credit Spread Velocity ===")

def s1a(dt, ticker, data):
    """Buy quality dips when credit spread TIGHTENS >0.5% in 5 days."""
    if credit_ratio is None or dt not in credit_ratio.index:
        return False
    idx = credit_ratio.index.get_loc(dt)
    if idx < 5:
        return False
    cr_chg = (credit_ratio.iloc[idx] / credit_ratio.iloc[idx-5] - 1) * 100
    if cr_chg <= 0.5:
        return False
    # Quality dip: stock down >2% from 10d high
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 10:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-20):tidx+1], 10).iloc[-1]
    return pfh < -0.02

def s1b(dt, ticker, data):
    """Buy when spread tightens >1% in 10 days + stock RSI<40."""
    if credit_ratio is None or dt not in credit_ratio.index:
        return False
    idx = credit_ratio.index.get_loc(dt)
    if idx < 10:
        return False
    cr_chg = (credit_ratio.iloc[idx] / credit_ratio.iloc[idx-10] - 1) * 100
    if cr_chg <= 1.0:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 20:
        return False
    r = rsi(tdata["Close"].iloc[max(0,tidx-30):tidx+1]).iloc[-1]
    return r < 40

def s1c(dt, ticker, data):
    """Buy when spread velocity turns negative→positive + stock >5% below high."""
    if credit_ratio is None or dt not in credit_ratio.index:
        return False
    idx = credit_ratio.index.get_loc(dt)
    if idx < 10:
        return False
    vel_now = credit_ratio.iloc[idx] - credit_ratio.iloc[idx-5]
    vel_prev = credit_ratio.iloc[idx-5] - credit_ratio.iloc[idx-10]
    if not (vel_prev < 0 and vel_now > 0):
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 20:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-25):tidx+1], 20).iloc[-1]
    return pfh < -0.05

def s1d(dt, ticker, data):
    """Dual: spread tightening + VIX declining + stock dip."""
    if credit_ratio is None or dt not in credit_ratio.index:
        return False
    if vix_df is None or dt not in vix_df.index:
        return False
    idx = credit_ratio.index.get_loc(dt)
    if idx < 5:
        return False
    cr_chg = (credit_ratio.iloc[idx] / credit_ratio.iloc[idx-5] - 1) * 100
    if cr_chg <= 0.3:
        return False
    vidx = vix_df.index.get_loc(dt)
    if vidx < 5:
        return False
    vix_chg = vix_df["Close"].iloc[vidx] - vix_df["Close"].iloc[vidx-5]
    if vix_chg >= 0:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 15:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-20):tidx+1], 10).iloc[-1]
    return pfh < -0.02

def s1e(dt, ticker, data):
    """Credit spread z-score falls below -1 + quality dip."""
    if credit_ratio is None or dt not in credit_ratio.index:
        return False
    idx = credit_ratio.index.get_loc(dt)
    if idx < 60:
        return False
    z = zscore(credit_ratio.iloc[max(0,idx-80):idx+1], 60).iloc[-1]
    # z < -1 means spread is unusually tight (ratio unusually high? No, z<-1 means ratio below mean)
    # Actually we want: spread WAS wide (ratio low), now reverting. z rising from <-1.
    # Let's use: z was <-1 yesterday, now >-1 (reverting up = tightening)
    z_prev = zscore(credit_ratio.iloc[max(0,idx-80):idx], 60).iloc[-1] if idx > 61 else 0
    if not (z_prev < -1 and z > -1):
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 15:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-20):tidx+1], 10).iloc[-1]
    return pfh < -0.03

def s1f(dt, ticker, data):
    """Multi-speed: 5d tightening + 20d tightening both positive + stock RSI<35."""
    if credit_ratio is None or dt not in credit_ratio.index:
        return False
    idx = credit_ratio.index.get_loc(dt)
    if idx < 20:
        return False
    chg5 = credit_ratio.iloc[idx] - credit_ratio.iloc[idx-5]
    chg20 = credit_ratio.iloc[idx] - credit_ratio.iloc[idx-20]
    if chg5 <= 0 or chg20 <= 0:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 20:
        return False
    r = rsi(tdata["Close"].iloc[max(0,tidx-30):tidx+1]).iloc[-1]
    return r < 35

# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 2: DOLLAR STRENGTH REVERSAL
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Strategy 2: Dollar Strength Reversal ===")

def s2a(dt, ticker, data):
    """Buy quality stocks when UUP drops >2% from 20d high."""
    if uup is None or dt not in uup.index:
        return False
    idx = uup.index.get_loc(dt)
    if idx < 20:
        return False
    pfh = pct_from_high(uup["Close"].iloc[max(0,idx-25):idx+1], 20).iloc[-1]
    if pfh >= -0.02:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 10:
        return False
    pfh_stock = pct_from_high(tdata["Close"].iloc[max(0,tidx-15):tidx+1], 10).iloc[-1]
    return pfh_stock < -0.02

def s2b(dt, ticker, data):
    """Buy after 5+ day UUP rally reverses + quality dip."""
    if uup is None or dt not in uup.index:
        return False
    idx = uup.index.get_loc(dt)
    if idx < 7:
        return False
    # UUP red today
    if uup["Close"].iloc[idx] >= uup["Close"].iloc[idx-1]:
        return False
    # 5 prior days green
    green_streak = all(uup["Close"].iloc[idx-j] > uup["Close"].iloc[idx-j-1] for j in range(1, 6))
    if not green_streak:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 15:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-20):tidx+1], 10).iloc[-1]
    return pfh < -0.03

def s2c(dt, ticker, data):
    """Buy intl-heavy stocks when UUP reverses from >1 std above mean."""
    if ticker not in INTL_HEAVY:
        return False
    if uup is None or dt not in uup.index:
        return False
    idx = uup.index.get_loc(dt)
    if idx < 60:
        return False
    z = zscore(uup["Close"].iloc[max(0,idx-80):idx+1], 60).iloc[-1]
    z_prev = zscore(uup["Close"].iloc[max(0,idx-80):idx], 60).iloc[-1] if idx > 61 else 0
    if not (z_prev > 1.0 and z < z_prev):
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 10:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-15):tidx+1], 10).iloc[-1]
    return pfh < -0.03

def s2d(dt, ticker, data):
    """Dollar-stock divergence: UUP up but quality stock also up (resilient)."""
    if uup is None or dt not in uup.index:
        return False
    idx = uup.index.get_loc(dt)
    if idx < 10:
        return False
    uup_chg = (uup["Close"].iloc[idx] / uup["Close"].iloc[idx-5] - 1)
    if uup_chg <= 0.005:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 10:
        return False
    stock_chg = (tdata["Close"].iloc[tidx] / tdata["Close"].iloc[tidx-5] - 1)
    return stock_chg > 0.01  # Stock up despite dollar strength

def s2e(dt, ticker, data):
    """UUP RSI>70 + quality stock RSI<35."""
    if uup is None or dt not in uup.index:
        return False
    idx = uup.index.get_loc(dt)
    if idx < 20:
        return False
    uup_rsi = rsi(uup["Close"].iloc[max(0,idx-30):idx+1]).iloc[-1]
    if uup_rsi <= 70:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 20:
        return False
    stock_rsi = rsi(tdata["Close"].iloc[max(0,tidx-30):tidx+1]).iloc[-1]
    return stock_rsi < 35

def s2f(dt, ticker, data):
    """Combined: UUP declining + credit tightening + quality stock dip."""
    if uup is None or dt not in uup.index:
        return False
    if credit_ratio is None or dt not in credit_ratio.index:
        return False
    idx_u = uup.index.get_loc(dt)
    idx_c = credit_ratio.index.get_loc(dt)
    if idx_u < 10 or idx_c < 10:
        return False
    uup_chg = uup["Close"].iloc[idx_u] - uup["Close"].iloc[idx_u-5]
    cr_chg = credit_ratio.iloc[idx_c] - credit_ratio.iloc[idx_c-5]
    if uup_chg >= 0 or cr_chg <= 0:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 15:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-20):tidx+1], 10).iloc[-1]
    return pfh < -0.03

# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 3: YIELD CURVE VELOCITY
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Strategy 3: Yield Curve Velocity ===")

def s3a(dt, ticker, data):
    """Buy when TLT/SHY ratio increases >1% in 5 days."""
    if curve_ratio is None or dt not in curve_ratio.index:
        return False
    idx = curve_ratio.index.get_loc(dt)
    if idx < 5:
        return False
    chg = (curve_ratio.iloc[idx] / curve_ratio.iloc[idx-5] - 1) * 100
    if chg <= 1.0:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 10:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-15):tidx+1], 10).iloc[-1]
    return pfh < -0.02

def s3b(dt, ticker, data):
    """Curve steepening acceleration (2nd derivative positive) + stock dip."""
    if curve_ratio is None or dt not in curve_ratio.index:
        return False
    idx = curve_ratio.index.get_loc(dt)
    if idx < 15:
        return False
    vel_now = curve_ratio.iloc[idx] - curve_ratio.iloc[idx-5]
    vel_prev = curve_ratio.iloc[idx-5] - curve_ratio.iloc[idx-10]
    accel = vel_now - vel_prev  # 2nd derivative
    if accel <= 0 or vel_now <= 0:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 15:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-20):tidx+1], 10).iloc[-1]
    return pfh < -0.03

def s3c(dt, ticker, data):
    """Curve was flattening, reverses + RSI<40."""
    if curve_ratio is None or dt not in curve_ratio.index:
        return False
    idx = curve_ratio.index.get_loc(dt)
    if idx < 15:
        return False
    vel_now = curve_ratio.iloc[idx] - curve_ratio.iloc[idx-5]
    vel_prev = curve_ratio.iloc[idx-5] - curve_ratio.iloc[idx-10]
    if not (vel_prev < 0 and vel_now > 0):
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 20:
        return False
    r = rsi(tdata["Close"].iloc[max(0,tidx-30):tidx+1]).iloc[-1]
    return r < 40

def s3d(dt, ticker, data):
    """Curve velocity + credit spread velocity both positive + stock dip."""
    if curve_ratio is None or dt not in curve_ratio.index:
        return False
    if credit_ratio is None or dt not in credit_ratio.index:
        return False
    idx_cv = curve_ratio.index.get_loc(dt)
    idx_cr = credit_ratio.index.get_loc(dt)
    if idx_cv < 5 or idx_cr < 5:
        return False
    cv_vel = curve_ratio.iloc[idx_cv] - curve_ratio.iloc[idx_cv-5]
    cr_vel = credit_ratio.iloc[idx_cr] - credit_ratio.iloc[idx_cr-5]
    if cv_vel <= 0 or cr_vel <= 0:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 15:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-20):tidx+1], 10).iloc[-1]
    return pfh < -0.03

def s3e(dt, ticker, data):
    """Extreme curve flattening reverses + stock >5% below high."""
    if curve_ratio is None or dt not in curve_ratio.index:
        return False
    idx = curve_ratio.index.get_loc(dt)
    if idx < 25:
        return False
    low20 = curve_ratio.iloc[idx-20:idx].min()
    # Was at/below 20d low recently, now recovering
    if not (curve_ratio.iloc[idx-2] <= low20 * 1.001 and curve_ratio.iloc[idx] > curve_ratio.iloc[idx-2]):
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 20:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-25):tidx+1], 20).iloc[-1]
    return pfh < -0.05

def s3f(dt, ticker, data):
    """Multi-timeframe: 5d, 10d, 20d curve velocity all positive + RSI<35."""
    if curve_ratio is None or dt not in curve_ratio.index:
        return False
    idx = curve_ratio.index.get_loc(dt)
    if idx < 25:
        return False
    v5 = curve_ratio.iloc[idx] - curve_ratio.iloc[idx-5]
    v10 = curve_ratio.iloc[idx] - curve_ratio.iloc[idx-10]
    v20 = curve_ratio.iloc[idx] - curve_ratio.iloc[idx-20]
    if v5 <= 0 or v10 <= 0 or v20 <= 0:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 20:
        return False
    r = rsi(tdata["Close"].iloc[max(0,tidx-30):tidx+1]).iloc[-1]
    return r < 35

# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 4: OVERNIGHT RETURN ANOMALY
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Strategy 4: Overnight Return Anomaly ===")

def s4a(dt, ticker, data):
    """Buy at close when day's return < -2%, sell next open."""
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    day_ret = (tdata["Close"].iloc[tidx] / tdata["Open"].iloc[tidx] - 1)
    return day_ret < -0.02

def s4b(dt, ticker, data):
    """Buy at close when both stock AND SPY down >1%."""
    if dt not in spy.index:
        return False
    spy_idx = spy.index.get_loc(dt)
    spy_ret = (spy["Close"].iloc[spy_idx] / spy["Open"].iloc[spy_idx] - 1)
    if spy_ret >= -0.01:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    stock_ret = (tdata["Close"].iloc[tidx] / tdata["Open"].iloc[tidx] - 1)
    return stock_ret < -0.01

def s4c(dt, ticker, data):
    """Buy when prior overnight was >1% AND stock dips during day."""
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 2:
        return False
    # Prior overnight return (yesterday close to today open)
    overnight_ret = (tdata["Open"].iloc[tidx] / tdata["Close"].iloc[tidx-1] - 1)
    if overnight_ret <= 0.01:
        return False
    # Today's intraday is down
    day_ret = (tdata["Close"].iloc[tidx] / tdata["Open"].iloc[tidx] - 1)
    return day_ret < -0.005

def s4d(dt, ticker, data):
    """Buy when 5-day cumulative overnight returns are negative."""
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 6:
        return False
    cum_overnight = 0
    for j in range(5):
        i = tidx - j
        if i < 1:
            return False
        overnight = (tdata["Open"].iloc[i] / tdata["Close"].iloc[i-1] - 1)
        cum_overnight += overnight
    return cum_overnight < -0.01

def s4e(dt, ticker, data):
    """Buy at close when VIX rose >5% + quality stock down >2%."""
    if vix_df is None or dt not in vix_df.index:
        return False
    vidx = vix_df.index.get_loc(dt)
    if vidx < 1:
        return False
    vix_chg = (vix_df["Close"].iloc[vidx] / vix_df["Close"].iloc[vidx-1] - 1)
    if vix_chg <= 0.05:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 1:
        return False
    stock_ret = (tdata["Close"].iloc[tidx] / tdata["Close"].iloc[tidx-1] - 1)
    return stock_ret < -0.02

def s4f(dt, ticker, data):
    """Buy at close when stock down >3% intraday but closed above LOD (hammer)."""
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    high = tdata["High"].iloc[tidx]
    low = tdata["Low"].iloc[tidx]
    close = tdata["Close"].iloc[tidx]
    opn = tdata["Open"].iloc[tidx]

    intraday_range = (low / opn - 1)
    if intraday_range >= -0.03:
        return False
    # Close above midpoint of range (hammer-like)
    midpoint = (high + low) / 2
    return close > midpoint

# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 5: VVIX / VOLATILITY-OF-VOLATILITY
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Strategy 5: VVIX / Vol-of-Vol ===")

def s5a(dt, ticker, data):
    """Buy quality dips when VVIX > 120."""
    if vvix_df is None or dt not in vvix_df.index:
        return False
    vidx = vvix_df.index.get_loc(dt)
    if vvix_df["Close"].iloc[vidx] <= 120:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 10:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-15):tidx+1], 10).iloc[-1]
    return pfh < -0.03

def s5b(dt, ticker, data):
    """Buy when VVIX spikes >20% in 1 day + stock RSI<40."""
    if vvix_df is None or dt not in vvix_df.index:
        return False
    vidx = vvix_df.index.get_loc(dt)
    if vidx < 1:
        return False
    vvix_chg = (vvix_df["Close"].iloc[vidx] / vvix_df["Close"].iloc[vidx-1] - 1)
    if vvix_chg <= 0.20:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 20:
        return False
    r = rsi(tdata["Close"].iloc[max(0,tidx-30):tidx+1]).iloc[-1]
    return r < 40

def s5c(dt, ticker, data):
    """Buy when VVIX/VIX ratio > 6 + quality dip."""
    if vvix_df is None or vix_df is None:
        return False
    if dt not in vvix_df.index or dt not in vix_df.index:
        return False
    vidx = vvix_df.index.get_loc(dt)
    vx_idx = vix_df.index.get_loc(dt)
    ratio = vvix_df["Close"].iloc[vidx] / (vix_df["Close"].iloc[vx_idx] + 1e-10)
    if ratio <= 6.0:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 10:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-15):tidx+1], 10).iloc[-1]
    return pfh < -0.03

def s5d(dt, ticker, data):
    """Buy when VVIX drops from >130 to <120 + stock >5% below high."""
    if vvix_df is None or dt not in vvix_df.index:
        return False
    vidx = vvix_df.index.get_loc(dt)
    if vidx < 5:
        return False
    vvix_now = vvix_df["Close"].iloc[vidx]
    vvix_recent_max = vvix_df["Close"].iloc[max(0,vidx-5):vidx].max()
    if not (vvix_recent_max > 130 and vvix_now < 120):
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 20:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-25):tidx+1], 20).iloc[-1]
    return pfh < -0.05

def s5e(dt, ticker, data):
    """VVIX > 2 std above 60d mean + quality dip."""
    if vvix_df is None or dt not in vvix_df.index:
        return False
    vidx = vvix_df.index.get_loc(dt)
    if vidx < 65:
        return False
    z = zscore(vvix_df["Close"].iloc[max(0,vidx-80):vidx+1], 60).iloc[-1]
    if z <= 2.0:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 10:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-15):tidx+1], 10).iloc[-1]
    return pfh < -0.03

def s5f(dt, ticker, data):
    """VVIX falling from spike + VIX falling + credit tightening + stock dip."""
    if vvix_df is None or vix_df is None or credit_ratio is None:
        return False
    if dt not in vvix_df.index or dt not in vix_df.index or dt not in credit_ratio.index:
        return False
    vidx = vvix_df.index.get_loc(dt)
    vx_idx = vix_df.index.get_loc(dt)
    cidx = credit_ratio.index.get_loc(dt)
    if vidx < 5 or vx_idx < 5 or cidx < 5:
        return False
    # VVIX falling
    if vvix_df["Close"].iloc[vidx] >= vvix_df["Close"].iloc[vidx-3]:
        return False
    # VIX falling
    if vix_df["Close"].iloc[vx_idx] >= vix_df["Close"].iloc[vx_idx-3]:
        return False
    # Credit tightening
    if credit_ratio.iloc[cidx] <= credit_ratio.iloc[cidx-5]:
        return False
    tdata = data[ticker]
    if dt not in tdata.index:
        return False
    tidx = tdata.index.get_loc(dt)
    if tidx < 15:
        return False
    pfh = pct_from_high(tdata["Close"].iloc[max(0,tidx-20):tidx+1], 10).iloc[-1]
    return pfh < -0.03

# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 6: SYSTEMATIC PUT SELLING (Simplified with BS proxy)
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Strategy 6: Systematic Put Selling ===")

# For options strategies, we approximate put selling returns:
# - Premium received ~ BS pricing proxy
# - P/L = premium collected - max(0, strike - stock_at_expiry)

from scipy.stats import norm

def bs_put_price(S, K, T, r=0.04, sigma=0.25):
    """Black-Scholes put price."""
    if T <= 0 or S <= 0 or K <= 0:
        return 0
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def run_put_selling_backtest(signal_func, dte=30, early_close_pct=0.5):
    """
    Specialized backtest for put selling strategies.
    signal_func(dt, ticker, data) -> (bool, strike, iv) or False
    """
    signals = []

    for ticker in UNIVERSE:
        if ticker not in stock_data:
            continue
        tdata = stock_data[ticker]
        for i, dt in enumerate(tdata.index):
            if dt < pd.Timestamp(START) or dt > pd.Timestamp(END):
                continue
            try:
                result = signal_func(dt, ticker, stock_data)
                if result and result[0]:
                    signals.append((dt, ticker, result[1], result[2]))  # dt, ticker, strike, iv
            except Exception:
                continue

    if not signals:
        return None

    signals.sort(key=lambda x: x[0])

    trades = []
    active = []

    for dt, ticker, strike, iv in signals:
        active = [(ed, tk) for ed, tk in active if ed > dt]
        if len(active) >= MAX_CONCURRENT:
            continue
        if any(tk == ticker for _, tk in active):
            continue

        tdata = stock_data[ticker]
        if dt not in tdata.index:
            continue
        tidx = tdata.index.get_loc(dt)

        S = tdata["Close"].iloc[tidx]
        T = dte / 365.0
        premium = bs_put_price(S, strike, T, sigma=iv)

        # Find expiry
        exp_idx = min(tidx + dte, len(tdata) - 1)
        S_exp = tdata["Close"].iloc[exp_idx]
        exp_dt = tdata.index[exp_idx]

        # Check for early close at 50% profit
        closed_early = False
        for check_idx in range(tidx + 1, exp_idx):
            S_check = tdata["Close"].iloc[check_idx]
            T_remaining = max((exp_idx - check_idx), 1) / 365.0
            current_put_price = bs_put_price(S_check, strike, T_remaining, sigma=iv*0.9)
            if current_put_price <= premium * (1 - early_close_pct):
                # Close at 50% profit
                pnl_per_share = premium - current_put_price
                exp_dt = tdata.index[check_idx]
                S_exp = S_check
                closed_early = True
                break

        if not closed_early:
            # Hold to expiry
            intrinsic = max(0, strike - S_exp)
            pnl_per_share = premium - intrinsic
        else:
            pass  # pnl already set

        # Scale: ~1 contract controls 100 shares, but we size to $200 risk
        # Max loss on CSP = strike * 100 - premium * 100
        # Position size: risk $200 max
        contracts = max(1, int(POS_SIZE / (strike * 100 - premium * 100 + 1)))
        # But realistically with $200, we can't sell a real put on a $200 stock
        # So let's use fractional sizing: pretend we can sell fraction of contract
        notional = strike  # per-share basis
        pnl = (pnl_per_share / notional) * POS_SIZE  # scale to position size
        ret = pnl / POS_SIZE

        trades.append({
            "entry_date": dt,
            "exit_date": exp_dt,
            "ticker": ticker,
            "entry": S,
            "exit": S_exp,
            "pnl": pnl - 2 * COMMISSION_PCT * POS_SIZE,
            "return": ret - 2 * COMMISSION_PCT,
        })
        active.append((exp_dt, ticker))

    return compute_metrics(trades)


def s6a(dt, ticker, data):
    """Sell 30-delta put at 20-SMA support, 30 DTE."""
    tdata = data[ticker]
    if dt not in tdata.index:
        return (False, 0, 0)
    tidx = tdata.index.get_loc(dt)
    if tidx < 30:
        return (False, 0, 0)
    close = tdata["Close"].iloc[tidx]
    sma20 = tdata["Close"].iloc[tidx-20:tidx+1].mean()
    # At 20-SMA support: within 1% of SMA
    if abs(close / sma20 - 1) > 0.01:
        return (False, 0, 0)
    # 30-delta put: roughly 5% OTM
    strike = close * 0.95
    # Estimate IV from recent realized vol
    rets = np.diff(np.log(tdata["Close"].iloc[max(0,tidx-30):tidx+1].values))
    iv = np.std(rets) * np.sqrt(252) if len(rets) > 5 else 0.25
    iv = max(iv, 0.15)
    return (True, strike, iv)

def s6b(dt, ticker, data):
    """Sell put when RSI<35, strike at -5% from current."""
    tdata = data[ticker]
    if dt not in tdata.index:
        return (False, 0, 0)
    tidx = tdata.index.get_loc(dt)
    if tidx < 20:
        return (False, 0, 0)
    r = rsi(tdata["Close"].iloc[max(0,tidx-30):tidx+1]).iloc[-1]
    if r >= 35:
        return (False, 0, 0)
    close = tdata["Close"].iloc[tidx]
    strike = close * 0.95
    rets = np.diff(np.log(tdata["Close"].iloc[max(0,tidx-30):tidx+1].values))
    iv = np.std(rets) * np.sqrt(252) if len(rets) > 5 else 0.30
    iv = max(iv, 0.20)  # IV likely elevated when RSI low
    return (True, strike, iv)

def s6c(dt, ticker, data):
    """Sell weekly puts (7 DTE) on quality stocks above 200-SMA, ATM-1 strike."""
    tdata = data[ticker]
    if dt not in tdata.index:
        return (False, 0, 0)
    tidx = tdata.index.get_loc(dt)
    if tidx < 200:
        return (False, 0, 0)
    close = tdata["Close"].iloc[tidx]
    sma200 = tdata["Close"].iloc[tidx-200:tidx+1].mean()
    if close <= sma200:
        return (False, 0, 0)
    # Only trigger once per week (Monday-ish)
    if dt.weekday() != 0:  # Monday
        return (False, 0, 0)
    strike = close * 0.99  # ATM-1 (slightly OTM)
    rets = np.diff(np.log(tdata["Close"].iloc[max(0,tidx-30):tidx+1].values))
    iv = np.std(rets) * np.sqrt(252) if len(rets) > 5 else 0.25
    iv = max(iv, 0.15)
    return (True, strike, iv)

def s6d(dt, ticker, data):
    """Sell put only when IV rank > 50th percentile."""
    tdata = data[ticker]
    if dt not in tdata.index:
        return (False, 0, 0)
    tidx = tdata.index.get_loc(dt)
    if tidx < 252:
        return (False, 0, 0)
    close = tdata["Close"].iloc[tidx]
    # Calculate IV rank (realized vol as proxy)
    current_vol = np.std(np.diff(np.log(tdata["Close"].iloc[tidx-21:tidx+1].values))) * np.sqrt(252)
    vols_1y = []
    for j in range(0, 252, 21):
        start_j = tidx - 252 + j
        end_j = start_j + 21
        if start_j < 0 or end_j > tidx:
            continue
        v = np.std(np.diff(np.log(tdata["Close"].iloc[start_j:end_j+1].values))) * np.sqrt(252)
        vols_1y.append(v)
    if len(vols_1y) < 5:
        return (False, 0, 0)
    iv_rank = np.mean([1 for v in vols_1y if v <= current_vol]) / len(vols_1y)
    if iv_rank <= 0.50:
        return (False, 0, 0)
    strike = close * 0.95
    iv = max(current_vol, 0.20)
    return (True, strike, iv)

def s6e(dt, ticker, data):
    """Sell put at MR entry (RSI<30 dip level)."""
    tdata = data[ticker]
    if dt not in tdata.index:
        return (False, 0, 0)
    tidx = tdata.index.get_loc(dt)
    if tidx < 30:
        return (False, 0, 0)
    r = rsi(tdata["Close"].iloc[max(0,tidx-30):tidx+1]).iloc[-1]
    if r >= 30:
        return (False, 0, 0)
    close = tdata["Close"].iloc[tidx]
    # Sell ATM put (aggressive - high premium, higher risk)
    strike = close
    rets = np.diff(np.log(tdata["Close"].iloc[max(0,tidx-30):tidx+1].values))
    iv = np.std(rets) * np.sqrt(252) if len(rets) > 5 else 0.35
    iv = max(iv, 0.25)
    return (True, strike, iv)

def s6f(dt, ticker, data):
    """Wheel: sell put at support. If assigned, sell covered call (simplified simulation)."""
    tdata = data[ticker]
    if dt not in tdata.index:
        return (False, 0, 0)
    tidx = tdata.index.get_loc(dt)
    if tidx < 50:
        return (False, 0, 0)
    close = tdata["Close"].iloc[tidx]
    sma50 = tdata["Close"].iloc[tidx-50:tidx+1].mean()
    # At or below 50-SMA
    if close > sma50 * 1.02:
        return (False, 0, 0)
    # Not too far below (avoid falling knives)
    if close < sma50 * 0.90:
        return (False, 0, 0)
    # Only trigger bi-weekly
    if dt.day not in [1, 2, 3, 15, 16, 17]:
        return (False, 0, 0)
    strike = close * 0.97
    rets = np.diff(np.log(tdata["Close"].iloc[max(0,tidx-30):tidx+1].values))
    iv = np.std(rets) * np.sqrt(252) if len(rets) > 5 else 0.25
    iv = max(iv, 0.15)
    return (True, strike, iv)


# ══════════════════════════════════════════════════════════════════════════
# RUN ALL BACKTESTS
# ══════════════════════════════════════════════════════════════════════════
strategies = {
    "1_CreditSpreadVelocity": {
        "A": ("Tightens >0.5% in 5d + dip", s1a),
        "B": ("Tightens >1% in 10d + RSI<40", s1b),
        "C": ("Velocity inflection + >5% below high", s1c),
        "D": ("Tightening + VIX declining + dip", s1d),
        "E": ("Z-score reversal from <-1 + dip", s1e),
        "F": ("Multi-speed 5d+20d + RSI<35", s1f),
    },
    "2_DollarStrengthReversal": {
        "A": ("UUP drops >2% from 20d high + dip", s2a),
        "B": ("5d UUP rally reverses + dip", s2b),
        "C": ("Intl-heavy stocks, UUP reversal from >1std", s2c),
        "D": ("Dollar-stock divergence (resilient)", s2d),
        "E": ("UUP RSI>70 + stock RSI<35", s2e),
        "F": ("UUP declining + credit tight + dip", s2f),
    },
    "3_YieldCurveVelocity": {
        "A": ("TLT/SHY ratio +1% in 5d + dip", s3a),
        "B": ("Steepening acceleration + dip", s3b),
        "C": ("Curve inflection (flat→steep) + RSI<40", s3c),
        "D": ("Curve vel + credit vel both + + dip", s3d),
        "E": ("Extreme flatten reversal + >5% below", s3e),
        "F": ("Multi-TF 5/10/20d all + + RSI<35", s3f),
    },
    "4_OvernightReturnAnomaly": {
        "A": ("Day ret < -2%, overnight bounce", s4a),
        "B": ("Stock+SPY both down >1%, overnight", s4b),
        "C": ("Prior overnight >1% + day dip, continuation", s4c),
        "D": ("5d cum overnight negative, reversal", s4d),
        "E": ("VIX +5% + stock -2%, overnight recovery", s4e),
        "F": ("Down >3% intraday, close above LOD (hammer)", s4f),
    },
    "5_VVIX_VolOfVol": {
        "A": ("VVIX > 120 + quality dip", s5a),
        "B": ("VVIX spike >20% 1d + RSI<40", s5b),
        "C": ("VVIX/VIX ratio > 6 + dip", s5c),
        "D": ("VVIX drops 130→120 + >5% below high", s5d),
        "E": ("VVIX > 2std above 60d mean + dip", s5e),
        "F": ("VVIX+VIX falling + credit tight + dip", s5f),
    },
}

put_strategies = {
    "6_SystematicPutSelling": {
        "A": ("30-delta at 20-SMA, 30 DTE, 50% close", s6a, 30),
        "B": ("RSI<35, -5% strike", s6b, 30),
        "C": ("Weekly puts, >200-SMA, ATM-1", s6c, 7),
        "D": ("IV rank > 50th pctile, -5% strike", s6d, 30),
        "E": ("MR entry (RSI<30), ATM put", s6e, 30),
        "F": ("Wheel at 50-SMA support", s6f, 30),
    },
}

results = {}
summary_lines = []

print("\n" + "="*80)
print("RUNNING 36 BACKTESTS")
print("="*80)

# Run strategies 1-5 (equity-based)
for strat_name, variants in strategies.items():
    print(f"\n--- {strat_name} ---")
    results[strat_name] = {}

    for var_label, (desc, signal_func) in variants.items():
        is_overnight = strat_name == "4_OvernightReturnAnomaly"
        hold = 1 if is_overnight else HOLD_DAYS

        result = run_backtest(signal_func, hold_days=hold, use_overnight=is_overnight)

        if result is None:
            result = {"n_trades": 0, "sharpe": 0, "gates_passed": 0, "pass_all_5": False, "error": "insufficient trades"}
            print(f"  {var_label}: {desc} → INSUFFICIENT TRADES")
        else:
            status = "PASS 5/5" if result["pass_all_5"] else f"FAIL {result['gates_passed']}/5"
            gates_detail = []
            if not result.get("gate1_sharpe", False): gates_detail.append(f"Sharpe={result['sharpe']}")
            if not result.get("gate2_perm", False): gates_detail.append(f"perm_p={result['perm_p']}")
            if not result.get("gate3_regime", False): gates_detail.append(f"regime_gap={result['regime_gap']}")
            if not result.get("gate4_mdd", False): gates_detail.append(f"MDD={result['max_dd']}")
            if not result.get("gate5_trades", False): gates_detail.append(f"trades={result['n_trades']}")

            fail_str = f" [{', '.join(gates_detail)}]" if gates_detail else ""
            print(f"  {var_label}: {desc}")
            print(f"     Sharpe={result['sharpe']:.2f} Sortino={result['sortino']:.2f} WR={result['win_rate']:.1%} PF={result['profit_factor']:.2f} MDD={result['max_dd']:.1%} Trades={result['n_trades']} → {status}{fail_str}")

        result["description"] = desc
        result["variant"] = var_label
        results[strat_name][var_label] = result

# Run strategy 6 (put selling)
for strat_name, variants in put_strategies.items():
    print(f"\n--- {strat_name} ---")
    results[strat_name] = {}

    for var_label, (desc, signal_func, dte) in variants.items():
        result = run_put_selling_backtest(signal_func, dte=dte)

        if result is None:
            result = {"n_trades": 0, "sharpe": 0, "gates_passed": 0, "pass_all_5": False, "error": "insufficient trades"}
            print(f"  {var_label}: {desc} → INSUFFICIENT TRADES")
        else:
            status = "PASS 5/5" if result["pass_all_5"] else f"FAIL {result['gates_passed']}/5"
            gates_detail = []
            if not result.get("gate1_sharpe", False): gates_detail.append(f"Sharpe={result['sharpe']}")
            if not result.get("gate2_perm", False): gates_detail.append(f"perm_p={result['perm_p']}")
            if not result.get("gate3_regime", False): gates_detail.append(f"regime_gap={result['regime_gap']}")
            if not result.get("gate4_mdd", False): gates_detail.append(f"MDD={result['max_dd']}")
            if not result.get("gate5_trades", False): gates_detail.append(f"trades={result['n_trades']}")

            fail_str = f" [{', '.join(gates_detail)}]" if gates_detail else ""
            print(f"  {var_label}: {desc}")
            print(f"     Sharpe={result['sharpe']:.2f} Sortino={result['sortino']:.2f} WR={result['win_rate']:.1%} PF={result['profit_factor']:.2f} MDD={result['max_dd']:.1%} Trades={result['n_trades']} → {status}{fail_str}")

        result["description"] = desc
        result["variant"] = var_label
        results[strat_name][var_label] = result


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("FINAL SUMMARY — 5-GATE VALIDATION")
print("="*80)

all_pass = []
for strat_name, variants in results.items():
    print(f"\n{strat_name}:")
    for var_label in sorted(variants.keys()):
        v = variants[var_label]
        desc = v.get("description", "")
        n = v.get("n_trades", 0)
        sh = v.get("sharpe", 0)
        gp = v.get("gates_passed", 0)
        pa = v.get("pass_all_5", False)

        if pa:
            tag = "★ PASS 5/5 — ADVERSARIAL NEEDED"
            all_pass.append(f"{strat_name}/{var_label}")
        elif n == 0:
            tag = "✗ NO TRADES"
        else:
            tag = f"✗ {gp}/5"

        print(f"  {var_label} ({desc}): Sharpe={sh:.2f}, Trades={n} → {tag}")

print(f"\n{'='*80}")
print(f"VARIANTS PASSING ALL 5 GATES: {len(all_pass)}")
for p in all_pass:
    print(f"  → {p} — ADVERSARIAL NEEDED")
if not all_pass:
    print("  (none)")
print(f"{'='*80}")

# Save results
# Convert any non-serializable types
def clean_for_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj

def deep_clean(d):
    if isinstance(d, dict):
        return {k: deep_clean(v) for k, v in d.items()}
    if isinstance(d, list):
        return [deep_clean(v) for v in d]
    return clean_for_json(d)

output = {
    "run_date": datetime.now().isoformat(),
    "config": {
        "start": START, "end": END, "capital": CAPITAL,
        "pos_size": POS_SIZE, "max_concurrent": MAX_CONCURRENT,
        "hold_days": HOLD_DAYS, "n_permutations": N_PERM,
    },
    "results": deep_clean(results),
    "pass_all_5_gates": all_pass,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {OUTPUT_PATH}")
