#!/usr/bin/env python3
"""
Dual Signal QMR Backtest
========================
Combine two strongest validated timing signals into one strategy:
  - QMR-A (dip buy): drop >5% from 20d high AND RSI(14) < 35
  - Recovery-E: first green day after 3+ consecutive red days, post 5% drawdown

6 Variants (+ VIX<25 filtered versions):
  A: QMR-A only (BASELINE). Hold 10 days.
  B: Recovery-E only. Hold 5 days.
  C: EITHER signal triggers (union). Hold 10 days.
  D: BOTH signals agree (intersection). Hold 10 days.
  E: QMR-A entry, but only if 2+ recent red days (combo). Hold 10 days.
  F: Staged entry — half on QMR-A, add half on Recovery-E confirm. Hold 10 days.

5-Gate Validation + Permutation Test (1000 iterations)

OOT: Jan 2022 – Jul 2026 | Capital: $645 | Max $200/trade | Max 3 concurrent
"""

import json
import warnings
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

# ── Configuration ─────────────────────────────────────────────────────────
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% each way
DATA_START = "2020-01-01"  # extra lookback for indicators
OOT_START = "2022-01-01"
END = "2026-07-31"
N_PERM = 1000
VIX_THRESHOLD = 25

TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]

ALL_TICKERS = sorted(set(TICKERS + ["SPY", "^VIX"]))

GATES = {
    "sharpe_min": 0.5,
    "perm_p_max": 0.05,
    "regime_gap_max": 0.5,
    "mdd_min": -50.0,
    "min_trades": 20,
}

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(ALL_TICKERS, start=DATA_START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        tk = ticker if ticker != "VIX" else "^VIX"
        if len(ALL_TICKERS) == 1:
            s = raw["Close"].dropna()
        else:
            s = raw[tk]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in TICKERS}
spy_close = get_close("SPY")
vix_close = get_close("^VIX")

loaded = sum(1 for t in TICKERS if len(closes.get(t, [])) > 100)
print(f"  Tickers with data: {loaded}/{len(TICKERS)}")
print(f"  SPY rows: {len(spy_close)}, VIX rows: {len(vix_close)}")

# ── Indicator Helpers ─────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def rolling_high(series, period=20):
    return series.rolling(period).max()


def consecutive_red_days(series):
    """Return a series with count of consecutive red (close < prior close) days ending at each date."""
    is_red = (series.diff() < 0).astype(int)
    # Count consecutive reds
    result = pd.Series(0, index=series.index, dtype=int)
    count = 0
    for i in range(len(is_red)):
        if is_red.iloc[i] == 1:
            count += 1
        else:
            count = 0
        result.iloc[i] = count
    return result


def is_green_day(series):
    """Return boolean series: True when close > prior close."""
    return series.diff() > 0


# ── Pre-compute indicators ───────────────────────────────────────────────
print("Computing indicators ...")
indicators = {}
for t in TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) < 50:
        continue
    indicators[t] = {
        "rsi14": calc_rsi(c, 14),
        "high20": rolling_high(c, 20),
        "consec_red": consecutive_red_days(c),
        "green": is_green_day(c),
    }

spy_sma200 = spy_close.rolling(200).mean()

# ── Signal Generation ─────────────────────────────────────────────────────
print("Generating signals ...")
oot_start_ts = pd.Timestamp(OOT_START)


def is_drawdown(ticker, date):
    """Check if stock has dropped >5% from 20-day high."""
    if ticker not in indicators:
        return False
    c = closes[ticker]
    h20 = indicators[ticker]["high20"]
    try:
        price = c.loc[date]
        high = h20.loc[date]
        if pd.isna(price) or pd.isna(high) or high == 0:
            return False
        return (price - high) / high < -0.05
    except (KeyError, IndexError):
        return False


def get_vix(date):
    """Get VIX value for a date."""
    try:
        return float(vix_close.asof(date))
    except Exception:
        return 20.0  # default neutral


def generate_qmr_a_signals():
    """QMR-A: drop >5% from 20d high AND RSI(14) < 35."""
    signals = []
    for t in TICKERS:
        if t not in indicators:
            continue
        c = closes[t]
        rsi = indicators[t]["rsi14"]
        h20 = indicators[t]["high20"]
        for date in c.index:
            if date < oot_start_ts:
                continue
            try:
                price = c.loc[date]
                high = h20.loc[date]
                r = rsi.loc[date]
                if pd.isna(price) or pd.isna(high) or pd.isna(r):
                    continue
                drop_pct = (price - high) / high
                if drop_pct < -0.05 and r < 35:
                    signals.append((date, t))
            except (KeyError, IndexError):
                continue
    return signals


def generate_recovery_e_signals():
    """Recovery-E: first green day after 3+ consecutive red days, post 5% drawdown from 20d high."""
    signals = []
    for t in TICKERS:
        if t not in indicators:
            continue
        c = closes[t]
        consec = indicators[t]["consec_red"]
        green = indicators[t]["green"]
        h20 = indicators[t]["high20"]

        for i in range(1, len(c)):
            date = c.index[i]
            if date < oot_start_ts:
                continue
            prev_date = c.index[i - 1]
            try:
                # Check: today is green
                if not green.iloc[i]:
                    continue
                # Check: yesterday had 3+ consecutive red days
                if consec.iloc[i - 1] < 3:
                    continue
                # Check: currently in drawdown (>5% from 20d high)
                price = c.iloc[i]
                high = h20.loc[date]
                if pd.isna(price) or pd.isna(high) or high == 0:
                    continue
                if (price - high) / high >= -0.05:
                    continue
                signals.append((date, t))
            except (KeyError, IndexError):
                continue
    return signals


qmr_a_signals = generate_qmr_a_signals()
recovery_e_signals = generate_recovery_e_signals()

# Build lookup sets for fast intersection/union
qmr_a_set = set(qmr_a_signals)  # (date, ticker) tuples
recovery_e_set = set(recovery_e_signals)

print(f"  QMR-A raw signals: {len(qmr_a_signals)}")
print(f"  Recovery-E raw signals: {len(recovery_e_signals)}")

# ── Build variant signal lists ────────────────────────────────────────────

def build_recent_red_lookup():
    """For variant E: check if ticker had 2+ red days in last 5 days."""
    lookup = {}
    for t in TICKERS:
        if t not in indicators:
            continue
        c = closes[t]
        consec = indicators[t]["consec_red"]
        is_red = (c.diff() < 0).astype(int)
        red_5d = is_red.rolling(5).sum()
        lookup[t] = red_5d
    return lookup

recent_red_lookup = build_recent_red_lookup()


def variant_signals():
    """Generate signals for all 6 variants."""
    variants = {}

    # A: QMR-A only, hold 10
    variants["A"] = [(d, t, 10) for d, t in qmr_a_signals]

    # B: Recovery-E only, hold 5
    variants["B"] = [(d, t, 5) for d, t in recovery_e_signals]

    # C: Union (either), hold 10
    union = sorted(set(qmr_a_signals) | set(recovery_e_signals))
    variants["C"] = [(d, t, 10) for d, t in union]

    # D: Intersection (both must agree on same date+ticker), hold 10
    intersection = sorted(qmr_a_set & recovery_e_set)
    variants["D"] = [(d, t, 10) for d, t in intersection]

    # E: QMR-A but only if 2+ recent red days, hold 10
    e_signals = []
    for d, t in qmr_a_signals:
        if t in recent_red_lookup:
            try:
                red_count = recent_red_lookup[t].asof(d)
                if not pd.isna(red_count) and red_count >= 2:
                    e_signals.append((d, t, 10))
            except Exception:
                continue
    variants["E"] = e_signals

    # F: Staged entry — will be handled specially in simulation
    # For signal generation, we track QMR-A as potential first entries
    # and Recovery-E as potential add-on entries
    variants["F"] = "staged"  # special marker

    return variants

all_variants = variant_signals()
for k, v in all_variants.items():
    if k != "F":
        print(f"  Variant {k}: {len(v)} raw signals")
    else:
        print(f"  Variant F: staged (QMR-A first half + Recovery-E add-on)")


# ── Trade Simulator ──────────────────────────────────────────────────────
def simulate_trades(signals, capital=CAPITAL, max_per_trade=MAX_PER_TRADE,
                    max_concurrent=MAX_CONCURRENT, vix_filter=False):
    """Simulate trades with position limits and concurrent position tracking.

    signals: list of (date, ticker, hold_days)
    Returns list of trade dicts.
    """
    if not signals:
        return []

    # Sort by date
    signals = sorted(signals, key=lambda x: x[0])

    # Deduplicate: no same-ticker entry within hold period
    deduped = []
    last_entry = {}
    for date, ticker, hold_days in signals:
        if ticker in last_entry:
            delta = (date - last_entry[ticker]).days
            if delta < hold_days:
                continue
        deduped.append((date, ticker, hold_days))
        last_entry[ticker] = date

    # Simulate with concurrent position limits
    trades = []
    open_positions = []  # list of (exit_date, ticker)

    for date, ticker, hold_days in deduped:
        # VIX filter
        if vix_filter and get_vix(date) >= VIX_THRESHOLD:
            continue

        # Close expired positions
        open_positions = [(ed, tk) for ed, tk in open_positions if ed > date]

        # Check concurrent limit
        if len(open_positions) >= max_concurrent:
            continue

        # Simulate trade
        c = closes.get(ticker, pd.Series(dtype=float))
        if len(c) == 0:
            continue
        try:
            loc = c.index.get_loc(date)
        except KeyError:
            mask = c.index >= date
            if mask.sum() == 0:
                continue
            loc = c.index.get_loc(c.index[mask][0])

        exit_loc = min(loc + hold_days, len(c) - 1)
        entry_price = float(c.iloc[loc]) * (1 + SLIPPAGE_PCT)
        exit_price = float(c.iloc[exit_loc]) * (1 - SLIPPAGE_PCT)
        exit_date = c.index[exit_loc]

        if entry_price <= 0:
            continue

        shares = max_per_trade / entry_price  # fractional for small account
        pnl = shares * (exit_price - entry_price)
        ret = (exit_price - entry_price) / entry_price

        trades.append({
            "ticker": ticker,
            "entry_date": str(c.index[loc].date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(float(pnl), 2),
            "return": round(float(ret), 6),
            "hold_days": hold_days,
            "shares": round(float(shares), 4),
        })

        open_positions.append((exit_date, ticker))

    return trades


def simulate_staged(capital=CAPITAL, max_per_trade=MAX_PER_TRADE,
                    max_concurrent=MAX_CONCURRENT, vix_filter=False):
    """Variant F: staged entry.

    Half position on QMR-A signal, add other half if Recovery-E fires within 10 days.
    Hold 10 days from first entry.
    """
    half_size = max_per_trade / 2.0

    # Sort QMR-A signals by date
    qmr_sorted = sorted(qmr_a_signals, key=lambda x: x[0])

    # Build Recovery-E lookup: (ticker) -> list of dates
    recovery_by_ticker = defaultdict(list)
    for d, t in recovery_e_signals:
        recovery_by_ticker[t].append(d)
    for t in recovery_by_ticker:
        recovery_by_ticker[t] = sorted(recovery_by_ticker[t])

    trades = []
    open_positions = []
    last_entry = {}

    for date, ticker in qmr_sorted:
        if date < oot_start_ts:
            continue

        if vix_filter and get_vix(date) >= VIX_THRESHOLD:
            continue

        # Dedup
        if ticker in last_entry:
            if (date - last_entry[ticker]).days < 10:
                continue

        # Expire old positions
        open_positions = [(ed, tk) for ed, tk in open_positions if ed > date]
        if len(open_positions) >= max_concurrent:
            continue

        c = closes.get(ticker, pd.Series(dtype=float))
        if len(c) == 0:
            continue
        try:
            loc = c.index.get_loc(date)
        except KeyError:
            mask = c.index >= date
            if mask.sum() == 0:
                continue
            loc = c.index.get_loc(c.index[mask][0])

        exit_loc = min(loc + 10, len(c) - 1)
        entry_price_1 = float(c.iloc[loc]) * (1 + SLIPPAGE_PCT)
        exit_price = float(c.iloc[exit_loc]) * (1 - SLIPPAGE_PCT)
        exit_date = c.index[exit_loc]

        if entry_price_1 <= 0:
            continue

        shares_1 = half_size / entry_price_1

        # Check if Recovery-E fires within next 10 days for this ticker
        recovery_dates = recovery_by_ticker.get(ticker, [])
        add_on = False
        shares_2 = 0
        entry_price_2 = 0
        for rd in recovery_dates:
            days_diff = (rd - date).days
            if 0 < days_diff <= 10:
                # Add second half
                try:
                    rd_loc = c.index.get_loc(rd)
                except KeyError:
                    continue
                entry_price_2 = float(c.iloc[rd_loc]) * (1 + SLIPPAGE_PCT)
                if entry_price_2 > 0:
                    shares_2 = half_size / entry_price_2
                    add_on = True
                break

        # Calculate PnL
        pnl_1 = shares_1 * (exit_price - entry_price_1)
        pnl_2 = shares_2 * (exit_price - entry_price_2) if add_on else 0
        total_pnl = pnl_1 + pnl_2

        total_cost = half_size + (half_size if add_on else 0)
        total_exit_val = total_cost + total_pnl
        ret = total_pnl / total_cost if total_cost > 0 else 0

        trades.append({
            "ticker": ticker,
            "entry_date": str(c.index[loc].date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price_1, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(float(total_pnl), 2),
            "return": round(float(ret), 6),
            "hold_days": 10,
            "shares": round(float(shares_1 + shares_2), 4),
            "staged": add_on,
        })

        open_positions.append((exit_date, ticker))
        last_entry[ticker] = date

    return trades


# ── Metrics Calculation ──────────────────────────────────────────────────
def calc_metrics(trades, capital=CAPITAL):
    """Calculate performance metrics from trade list."""
    if not trades:
        return {
            "total_return_pct": 0, "sharpe": 0, "sortino": 0,
            "max_drawdown_pct": 0, "win_rate": 0, "profit_factor": 0,
            "n_trades": 0, "total_pnl": 0, "avg_return_pct": 0,
        }

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    total_pnl = pnls.sum()

    wins = (pnls > 0).sum()
    win_rate = wins / len(pnls)

    gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss

    # Sharpe (trade-level, annualized)
    oot_years = 4.5  # Jan 2022 - Jul 2026
    if len(returns) > 1 and returns.std() > 0:
        trades_per_year = max(len(returns) / oot_years, 1)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        trades_per_year = max(len(returns) / oot_years, 1)
        sortino = (returns.mean() / downside.std()) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

    # Max drawdown from cumulative PnL
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl + capital)
    drawdowns = (cum_pnl + capital - peak) / peak
    max_dd = drawdowns.min() * 100 if len(drawdowns) > 0 else 0

    return {
        "total_return_pct": round(total_pnl / capital * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_drawdown_pct": round(float(max_dd), 2),
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(profit_factor), 3),
        "n_trades": len(trades),
        "total_pnl": round(float(total_pnl), 2),
        "avg_return_pct": round(float(returns.mean() * 100), 3),
    }


def regime_stratified_sharpe(trades):
    """Split trades into bull/bear based on SPY vs 200-SMA."""
    if not trades:
        return {"bull_sharpe": 0, "bear_sharpe": 0, "regime_gap": 0,
                "bull_trades": 0, "bear_trades": 0}

    bull_rets, bear_rets = [], []
    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        try:
            spy_val = spy_close.asof(entry)
            sma_val = spy_sma200.asof(entry)
            if pd.isna(spy_val) or pd.isna(sma_val):
                continue
            if spy_val > sma_val:
                bull_rets.append(t["return"])
            else:
                bear_rets.append(t["return"])
        except Exception:
            continue

    def _sharpe(rets):
        if len(rets) < 2:
            return 0.0
        arr = np.array(rets)
        if arr.std() == 0:
            return 0.0
        tpy = max(len(arr) / 4.5, 1)
        return float((arr.mean() / arr.std()) * np.sqrt(tpy))

    bull_s = _sharpe(bull_rets)
    bear_s = _sharpe(bear_rets)
    max_abs = max(abs(bull_s), abs(bear_s), 1e-9)
    gap = abs(bull_s - bear_s) / max_abs

    return {
        "bull_sharpe": round(bull_s, 3),
        "bear_sharpe": round(bear_s, 3),
        "regime_gap": round(gap, 3),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
    }


def permutation_test(trades, n_perm=N_PERM):
    """Shuffle signal timing to test if returns are due to signal or luck."""
    if len(trades) < 5:
        return 1.0

    real_returns = np.array([t["return"] for t in trades])
    real_mean = real_returns.mean()

    ticker_dates = {}
    for t in trades:
        tk = t["ticker"]
        if tk not in ticker_dates:
            c = closes.get(tk, pd.Series(dtype=float))
            valid = c.index[c.index >= oot_start_ts]
            if len(valid) > 20:
                ticker_dates[tk] = valid

    count_better = 0
    for _ in range(n_perm):
        perm_returns = []
        for t in trades:
            tk = t["ticker"]
            hold = t["hold_days"]
            if tk not in ticker_dates:
                perm_returns.append(0.0)
                continue
            dates = ticker_dates[tk]
            max_idx = max(0, len(dates) - hold - 1)
            if max_idx < 1:
                perm_returns.append(0.0)
                continue
            rand_loc = np.random.randint(0, max_idx)
            c = closes[tk]
            try:
                entry_p = float(c.iloc[c.index.get_loc(dates[rand_loc])]) * (1 + SLIPPAGE_PCT)
                exit_loc = min(c.index.get_loc(dates[rand_loc]) + hold, len(c) - 1)
                exit_p = float(c.iloc[exit_loc]) * (1 - SLIPPAGE_PCT)
                if entry_p > 0:
                    perm_returns.append((exit_p - entry_p) / entry_p)
                else:
                    perm_returns.append(0.0)
            except Exception:
                perm_returns.append(0.0)

        if np.mean(perm_returns) >= real_mean:
            count_better += 1

    return count_better / n_perm


def five_gate(metrics, regime, perm_p):
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > GATES["sharpe_min"],
        "perm_p_lt_0.05": perm_p < GATES["perm_p_max"],
        "regime_gap_lt_0.5": regime["regime_gap"] < GATES["regime_gap_max"],
        "maxdd_gt_neg50": metrics["max_drawdown_pct"] > GATES["mdd_min"],
        "trades_gte_20": metrics["n_trades"] >= GATES["min_trades"],
    }
    gates["pass_all"] = all(v for k, v in gates.items() if k != "pass_all")
    return gates


# ── Run All Variants ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("RUNNING DUAL SIGNAL QMR BACKTEST — 12 VARIANTS")
print("="*70)

results = {}

# Generate trades for each variant
variant_configs = {
    "A": ("QMR-A only (baseline)", False),
    "B": ("Recovery-E only", False),
    "C": ("Union (either signal)", False),
    "D": ("Intersection (both agree)", False),
    "E": ("QMR-A + recent red days", False),
    "F": ("Staged entry", False),
    "A_vix": ("QMR-A + VIX<25", True),
    "B_vix": ("Recovery-E + VIX<25", True),
    "C_vix": ("Union + VIX<25", True),
    "D_vix": ("Intersection + VIX<25", True),
    "E_vix": ("QMR-A + red days + VIX<25", True),
    "F_vix": ("Staged + VIX<25", True),
}

for var_name, (description, vix_filter) in variant_configs.items():
    base_var = var_name.replace("_vix", "")
    print(f"\n--- Variant {var_name}: {description} ---")

    if base_var == "F":
        trades = simulate_staged(vix_filter=vix_filter)
    else:
        sigs = all_variants[base_var]
        trades = simulate_trades(sigs, vix_filter=vix_filter)

    metrics = calc_metrics(trades)
    regime = regime_stratified_sharpe(trades)

    print(f"  Trades: {metrics['n_trades']}, Return: {metrics['total_return_pct']:.1f}%, "
          f"Sharpe: {metrics['sharpe']:.3f}, WR: {metrics['win_rate']:.1%}")

    # Permutation test
    print(f"  Running permutation test ({N_PERM} iterations) ...")
    perm_p = permutation_test(trades, N_PERM)
    print(f"  Perm p-value: {perm_p:.4f}")

    gates = five_gate(metrics, regime, perm_p)
    passed = sum(1 for k, v in gates.items() if k != "pass_all" and v)
    print(f"  Gates: {passed}/5 {'PASS' if gates['pass_all'] else 'FAIL'}")

    results[var_name] = {
        "description": description,
        "metrics": metrics,
        "regime": regime,
        "perm_p": round(perm_p, 4),
        "gates": gates,
        "vix_filtered": vix_filter,
        "top_tickers": {},
    }

    # Top tickers by trade count
    if trades:
        ticker_counts = defaultdict(int)
        ticker_pnl = defaultdict(float)
        for t in trades:
            ticker_counts[t["ticker"]] += 1
            ticker_pnl[t["ticker"]] += t["pnl"]
        top5 = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        results[var_name]["top_tickers"] = {
            tk: {"trades": cnt, "pnl": round(ticker_pnl[tk], 2)} for tk, cnt in top5
        }


# ── Print Comparison Table ──────────────────────────────────────────────
print("\n" + "="*110)
print("COMPARISON TABLE — DUAL SIGNAL QMR BACKTEST")
print("="*110)
print(f"{'Variant':<10} {'Description':<30} {'Trades':>6} {'Return%':>8} {'Sharpe':>7} "
      f"{'Sortino':>8} {'WR':>6} {'PF':>6} {'MDD%':>7} {'PermP':>6} {'Gates':>6}")
print("-"*110)

for var_name in ["A", "B", "C", "D", "E", "F", "A_vix", "B_vix", "C_vix", "D_vix", "E_vix", "F_vix"]:
    r = results[var_name]
    m = r["metrics"]
    g = r["gates"]
    passed = sum(1 for k, v in g.items() if k != "pass_all" and v)
    tag = "PASS" if g["pass_all"] else "FAIL"
    print(f"{var_name:<10} {r['description']:<30} {m['n_trades']:>6} {m['total_return_pct']:>7.1f}% "
          f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} "
          f"{m['max_drawdown_pct']:>6.1f}% {r['perm_p']:>6.4f} {passed}/5 {tag}")

# ── Regime Detail ────────────────────────────────────────────────────────
print("\n" + "="*90)
print("REGIME STRATIFICATION")
print("="*90)
print(f"{'Variant':<10} {'Bull Sharpe':>11} {'Bear Sharpe':>11} {'Gap':>6} {'Bull#':>6} {'Bear#':>6}")
print("-"*90)
for var_name in ["A", "B", "C", "D", "E", "F", "A_vix", "B_vix", "C_vix", "D_vix", "E_vix", "F_vix"]:
    reg = results[var_name]["regime"]
    print(f"{var_name:<10} {reg['bull_sharpe']:>11.3f} {reg['bear_sharpe']:>11.3f} "
          f"{reg['regime_gap']:>6.3f} {reg['bull_trades']:>6} {reg['bear_trades']:>6}")

# ── Best Variant Analysis ────────────────────────────────────────────────
print("\n" + "="*70)
print("BEST VARIANTS (by Sharpe, passing all 5 gates)")
print("="*70)

passing = {k: v for k, v in results.items() if v["gates"]["pass_all"]}
if passing:
    sorted_pass = sorted(passing.items(), key=lambda x: x[1]["metrics"]["sharpe"], reverse=True)
    for rank, (var, data) in enumerate(sorted_pass, 1):
        m = data["metrics"]
        print(f"  #{rank}: Variant {var} — Sharpe {m['sharpe']:.3f}, "
              f"Sortino {m['sortino']:.3f}, WR {m['win_rate']:.1%}, "
              f"PF {m['profit_factor']:.2f}, {m['n_trades']} trades, "
              f"Return {m['total_return_pct']:.1f}%")
else:
    print("  No variants passed all 5 gates.")
    # Show best by Sharpe regardless
    sorted_all = sorted(results.items(), key=lambda x: x[1]["metrics"]["sharpe"], reverse=True)
    for rank, (var, data) in enumerate(sorted_all[:3], 1):
        m = data["metrics"]
        g = data["gates"]
        passed = sum(1 for k, v in g.items() if k != "pass_all" and v)
        print(f"  #{rank} (best effort, {passed}/5 gates): Variant {var} — "
              f"Sharpe {m['sharpe']:.3f}, {m['n_trades']} trades")

# ── Combo vs Individual ──────────────────────────────────────────────────
print("\n" + "="*70)
print("COMBO VALUE ANALYSIS")
print("="*70)
a_sharpe = results["A"]["metrics"]["sharpe"]
b_sharpe = results["B"]["metrics"]["sharpe"]
for var in ["C", "D", "E", "F"]:
    v_sharpe = results[var]["metrics"]["sharpe"]
    better_than_both = v_sharpe > max(a_sharpe, b_sharpe)
    better_than_either = v_sharpe > min(a_sharpe, b_sharpe)
    print(f"  {var} ({results[var]['description']}): Sharpe {v_sharpe:.3f} "
          f"{'> BOTH' if better_than_both else '> weaker' if better_than_either else '< both'} "
          f"individual signals (A={a_sharpe:.3f}, B={b_sharpe:.3f})")

# ── Save Results ─────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/dual_signal_qmr_results.json")

output = {
    "backtest": "Dual Signal QMR",
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "config": {
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_pct": SLIPPAGE_PCT,
        "oot_period": f"{OOT_START} to {END}",
        "universe": TICKERS,
        "n_permutations": N_PERM,
        "vix_threshold": VIX_THRESHOLD,
    },
    "signals": {
        "qmr_a_count": len(qmr_a_signals),
        "recovery_e_count": len(recovery_e_signals),
        "intersection_count": len(qmr_a_set & recovery_e_set),
    },
    "variants": results,
    "gates": dict(GATES),
}

output_path.write_text(json.dumps(output, indent=2, default=str))
print(f"\nResults saved to {output_path}")
print("DONE.")
