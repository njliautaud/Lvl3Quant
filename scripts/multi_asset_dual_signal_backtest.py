#!/usr/bin/env python3
"""
Multi-Asset Dual Signal Portfolio Backtest
==========================================
Run Dual Signal D strategy across three asset universes simultaneously:
  1. US Quality Stocks (20 names)
  2. International ADRs (10 names)
  3. Sector ETFs (10 names)

Signal Logic (Dual Signal D = intersection):
  Stocks/ADRs: 5% dip from 20d high + RSI<35 + first green after 3+ red. Hold 10d.
  ETFs: 3% dip (ETFs move less) + RSI<35 + first green after 2+ red. Hold 10d.

6 Variants testing diversification benefit:
  A: US Quality only (BASELINE) — $645, max $200/trade, 3 concurrent
  B: All three equal ($215 each), max 2/universe, 6 total
  C: All three weighted (US 50%, Intl 30%, ETF 20%), 6 total
  D: US + Intl only, $322.50 each, 3/universe
  E: All three, prioritize strongest signal universe, 3 total
  F: All three, reduce Intl when USD strong (UUP > 20d SMA), 6 total

5-Gate Validation + 1000 permutations.
OOT: Jan 2022 – Jul 2026. Capital: $645.
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
DATA_START = "2020-01-01"
OOT_START = "2022-01-01"
END = "2026-07-31"
N_PERM = 1000
HOLD_DAYS = 10

US_TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]

INTL_TICKERS = [
    "TSM", "ASML", "NVO", "SAP", "AZN", "NVS", "SHOP", "TM", "SONY", "BHP",
]

ETF_TICKERS = [
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE",
]

UNIVERSE_CONFIG = {
    "US":   {"tickers": US_TICKERS,   "dip_pct": 0.05, "min_red": 3, "slippage": 0.0002},
    "INTL": {"tickers": INTL_TICKERS, "dip_pct": 0.05, "min_red": 3, "slippage": 0.0002},
    "ETF":  {"tickers": ETF_TICKERS,  "dip_pct": 0.03, "min_red": 2, "slippage": 0.0001},
}

ALL_STOCK_TICKERS = US_TICKERS + INTL_TICKERS + ETF_TICKERS
ALL_TICKERS = sorted(set(ALL_STOCK_TICKERS + ["SPY", "UUP"]))

GATES = {
    "sharpe_min": 0.5,
    "perm_p_max": 0.05,
    "regime_gap_max": 0.5,
    "mdd_min": -50.0,
    "min_trades": 20,
}

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data for all universes ...")
raw = yf.download(ALL_TICKERS, start=DATA_START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        if len(ALL_TICKERS) == 1:
            s = raw["Close"].dropna()
        else:
            s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in ALL_TICKERS}
spy_close = closes.get("SPY", pd.Series(dtype=float))
uup_close = closes.get("UUP", pd.Series(dtype=float))
spy_sma200 = spy_close.rolling(200).mean()
uup_sma20 = uup_close.rolling(20).mean()

for universe_name, cfg in UNIVERSE_CONFIG.items():
    loaded = sum(1 for t in cfg["tickers"] if len(closes.get(t, [])) > 100)
    print(f"  {universe_name}: {loaded}/{len(cfg['tickers'])} tickers with data")
print(f"  SPY rows: {len(spy_close)}, UUP rows: {len(uup_close)}")


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
    is_red = (series.diff() < 0).astype(int)
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
    return series.diff() > 0


# ── Pre-compute indicators ───────────────────────────────────────────────
print("Computing indicators ...")
indicators = {}
for t in ALL_STOCK_TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) < 50:
        continue
    indicators[t] = {
        "rsi14": calc_rsi(c, 14),
        "high20": rolling_high(c, 20),
        "consec_red": consecutive_red_days(c),
        "green": is_green_day(c),
    }


# ── Signal Generation (Dual Signal D per universe) ───────────────────────
print("Generating signals ...")
oot_start_ts = pd.Timestamp(OOT_START)


def generate_dual_d_signals(tickers, dip_pct, min_red):
    """Dual Signal D: dip from 20d high + RSI<35 + first green after N+ red days."""
    signals = []
    for t in tickers:
        if t not in indicators:
            continue
        c = closes[t]
        rsi = indicators[t]["rsi14"]
        h20 = indicators[t]["high20"]
        consec = indicators[t]["consec_red"]
        green = indicators[t]["green"]

        for i in range(1, len(c)):
            date = c.index[i]
            if date < oot_start_ts:
                continue
            try:
                price = c.iloc[i]
                high = h20.loc[date]
                r = rsi.loc[date]
                if pd.isna(price) or pd.isna(high) or pd.isna(r) or high == 0:
                    continue
                drop_pct = (price - high) / high
                # Gate 1: dip threshold
                if drop_pct >= -dip_pct:
                    continue
                # Gate 2: RSI < 35
                if r >= 35:
                    continue
                # Gate 3: today is green
                if not green.iloc[i]:
                    continue
                # Gate 4: yesterday had min_red+ consecutive red days
                if consec.iloc[i - 1] < min_red:
                    continue
                signals.append((date, t))
            except (KeyError, IndexError):
                continue
    return signals


# Generate signals per universe
universe_signals = {}
for universe_name, cfg in UNIVERSE_CONFIG.items():
    sigs = generate_dual_d_signals(cfg["tickers"], cfg["dip_pct"], cfg["min_red"])
    universe_signals[universe_name] = sigs
    print(f"  {universe_name}: {len(sigs)} Dual-D signals")


# ── Trade Simulator ──────────────────────────────────────────────────────
def get_slippage(ticker):
    """Return slippage based on universe membership."""
    for uname, cfg in UNIVERSE_CONFIG.items():
        if ticker in cfg["tickers"]:
            return cfg["slippage"]
    return 0.0002


def simulate_trades(signals, capital, max_per_trade, max_concurrent,
                    per_universe_limits=None):
    """Simulate trades with position limits.

    signals: list of (date, ticker) or (date, ticker, universe)
    per_universe_limits: dict {universe: max_concurrent} or None
    """
    if not signals:
        return []

    signals = sorted(signals, key=lambda x: x[0])

    # Deduplicate: no same-ticker entry within hold period
    deduped = []
    last_entry = {}
    for item in signals:
        date, ticker = item[0], item[1]
        universe = item[2] if len(item) > 2 else None
        if ticker in last_entry:
            delta = (date - last_entry[ticker]).days
            if delta < HOLD_DAYS:
                continue
        deduped.append((date, ticker, universe))
        last_entry[ticker] = date

    trades = []
    open_positions = []  # (exit_date, ticker, universe)

    for date, ticker, universe in deduped:
        # Close expired positions
        open_positions = [(ed, tk, u) for ed, tk, u in open_positions if ed > date]

        # Check total concurrent limit
        if len(open_positions) >= max_concurrent:
            continue

        # Check per-universe limit if specified
        if per_universe_limits and universe:
            universe_count = sum(1 for _, _, u in open_positions if u == universe)
            if universe_count >= per_universe_limits.get(universe, max_concurrent):
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

        exit_loc = min(loc + HOLD_DAYS, len(c) - 1)
        slip = get_slippage(ticker)
        entry_price = float(c.iloc[loc]) * (1 + slip)
        exit_price = float(c.iloc[exit_loc]) * (1 - slip)
        exit_date = c.index[exit_loc]

        if entry_price <= 0:
            continue

        shares = max_per_trade / entry_price
        pnl = shares * (exit_price - entry_price)
        ret = (exit_price - entry_price) / entry_price

        trades.append({
            "ticker": ticker,
            "universe": universe or "US",
            "entry_date": str(c.index[loc].date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(float(pnl), 2),
            "return": round(float(ret), 6),
            "hold_days": HOLD_DAYS,
            "shares": round(float(shares), 4),
        })

        open_positions.append((exit_date, ticker, universe))

    return trades


# ── Metrics ──────────────────────────────────────────────────────────────
def calc_metrics(trades, capital=CAPITAL):
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

    oot_years = 4.5
    if len(returns) > 1 and returns.std() > 0:
        trades_per_year = max(len(returns) / oot_years, 1)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        trades_per_year = max(len(returns) / oot_years, 1)
        sortino = (returns.mean() / downside.std()) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

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


def universe_breakdown(trades):
    """Break down metrics per universe."""
    by_u = defaultdict(list)
    for t in trades:
        by_u[t.get("universe", "US")].append(t)
    result = {}
    for u, utrades in by_u.items():
        result[u] = calc_metrics(utrades)
    return result


def permutation_test(trades, n_perm=N_PERM):
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
            slip = get_slippage(tk)
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
                entry_p = float(c.iloc[c.index.get_loc(dates[rand_loc])]) * (1 + slip)
                exit_loc = min(c.index.get_loc(dates[rand_loc]) + hold, len(c) - 1)
                exit_p = float(c.iloc[exit_loc]) * (1 - slip)
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
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > GATES["sharpe_min"],
        "perm_p_lt_0.05": perm_p < GATES["perm_p_max"],
        "regime_gap_lt_0.5": regime["regime_gap"] < GATES["regime_gap_max"],
        "maxdd_gt_neg50": metrics["max_drawdown_pct"] > GATES["mdd_min"],
        "trades_gte_20": metrics["n_trades"] >= GATES["min_trades"],
    }
    gates["pass_all"] = all(v for k, v in gates.items() if k != "pass_all")
    return gates


# ── Build Variant Signal Lists ───────────────────────────────────────────

def tag_signals(sigs, universe):
    """Add universe tag to signal tuples."""
    return [(d, t, universe) for d, t in sigs]


us_sigs = tag_signals(universe_signals["US"], "US")
intl_sigs = tag_signals(universe_signals["INTL"], "INTL")
etf_sigs = tag_signals(universe_signals["ETF"], "ETF")
all_sigs = sorted(us_sigs + intl_sigs + etf_sigs, key=lambda x: x[0])


def compute_signal_strength(universe, date, lookback=10):
    """Compute signal strength for a universe on a date.
    Strength = number of tickers in the universe with RSI < 40 and near 20d low.
    More signals = universe is more oversold = stronger opportunity."""
    cfg = UNIVERSE_CONFIG[universe]
    count = 0
    for t in cfg["tickers"]:
        if t not in indicators:
            continue
        c = closes[t]
        rsi = indicators[t]["rsi14"]
        h20 = indicators[t]["high20"]
        try:
            price = c.asof(date)
            r = rsi.asof(date)
            high = h20.asof(date)
            if pd.isna(price) or pd.isna(r) or pd.isna(high) or high == 0:
                continue
            drop = (price - high) / high
            if drop < -cfg["dip_pct"] * 0.5 and r < 40:
                count += 1
        except Exception:
            continue
    return count


def is_usd_strong(date):
    """Check if USD is strengthening: UUP > 20-day SMA."""
    try:
        uup = uup_close.asof(date)
        sma = uup_sma20.asof(date)
        if pd.isna(uup) or pd.isna(sma):
            return False
        return uup > sma
    except Exception:
        return False


# ── Run Variants ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RUNNING MULTI-ASSET DUAL SIGNAL PORTFOLIO BACKTEST — 6 VARIANTS")
print("=" * 70)

results = {}

# ── Variant A: US Quality only (BASELINE) ────────────────────────────────
print("\n--- Variant A: US Quality only (BASELINE) ---")
trades_a = simulate_trades(us_sigs, capital=CAPITAL, max_per_trade=200.0,
                           max_concurrent=3)
print(f"  {len(trades_a)} trades generated")

# ── Variant B: All three equal allocation ─────────────────────────────────
print("\n--- Variant B: All three equal ($215 each) ---")
# Equal allocation: $215 per universe, max 2 per universe, 6 total
# We simulate by combining signals but enforcing per-universe limits
trades_b = simulate_trades(all_sigs, capital=CAPITAL, max_per_trade=107.5,
                           max_concurrent=6,
                           per_universe_limits={"US": 2, "INTL": 2, "ETF": 2})
print(f"  {len(trades_b)} trades generated")

# ── Variant C: All three weighted ─────────────────────────────────────────
print("\n--- Variant C: All three weighted (US 50%, Intl 30%, ETF 20%) ---")
# Weighted: US $322.50 (50%), Intl $193.50 (30%), ETF $129 (20%)
# We need different max_per_trade per universe
# Implement by running separately then merging with concurrent limits
def simulate_weighted(us_sigs, intl_sigs, etf_sigs, capital=CAPITAL,
                      us_pct=0.50, intl_pct=0.30, etf_pct=0.20,
                      max_concurrent=6):
    """Simulate with different position sizes per universe."""
    us_alloc = capital * us_pct
    intl_alloc = capital * intl_pct
    etf_alloc = capital * etf_pct

    # Max per trade: allocate universe capital across ~2 concurrent positions
    us_max = us_alloc / 2
    intl_max = intl_alloc / 2
    etf_max = etf_alloc / 2

    # Combine all signals with max_per_trade encoded
    combined = []
    for d, t, u in us_sigs:
        combined.append((d, t, u, us_max))
    for d, t, u in intl_sigs:
        combined.append((d, t, u, intl_max))
    for d, t, u in etf_sigs:
        combined.append((d, t, u, etf_max))
    combined.sort(key=lambda x: x[0])

    # Deduplicate
    deduped = []
    last_entry = {}
    for date, ticker, universe, mpt in combined:
        if ticker in last_entry:
            if (date - last_entry[ticker]).days < HOLD_DAYS:
                continue
        deduped.append((date, ticker, universe, mpt))
        last_entry[ticker] = date

    trades = []
    open_positions = []

    for date, ticker, universe, max_per_trade in deduped:
        open_positions = [(ed, tk, u) for ed, tk, u in open_positions if ed > date]
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

        exit_loc = min(loc + HOLD_DAYS, len(c) - 1)
        slip = get_slippage(ticker)
        entry_price = float(c.iloc[loc]) * (1 + slip)
        exit_price = float(c.iloc[exit_loc]) * (1 - slip)
        exit_date = c.index[exit_loc]

        if entry_price <= 0:
            continue

        shares = max_per_trade / entry_price
        pnl = shares * (exit_price - entry_price)
        ret = (exit_price - entry_price) / entry_price

        trades.append({
            "ticker": ticker,
            "universe": universe,
            "entry_date": str(c.index[loc].date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(float(pnl), 2),
            "return": round(float(ret), 6),
            "hold_days": HOLD_DAYS,
            "shares": round(float(shares), 4),
        })
        open_positions.append((exit_date, ticker, universe))

    return trades

trades_c = simulate_weighted(us_sigs, intl_sigs, etf_sigs)
print(f"  {len(trades_c)} trades generated")

# ── Variant D: US + Intl only (no ETFs) ──────────────────────────────────
print("\n--- Variant D: US + Intl only ---")
us_intl_sigs = sorted(us_sigs + intl_sigs, key=lambda x: x[0])
trades_d = simulate_trades(us_intl_sigs, capital=CAPITAL, max_per_trade=161.25,
                           max_concurrent=6,
                           per_universe_limits={"US": 3, "INTL": 3})
print(f"  {len(trades_d)} trades generated")

# ── Variant E: Prioritize strongest universe ─────────────────────────────
print("\n--- Variant E: Prioritize strongest signal universe ---")


def simulate_priority(all_signals, capital=CAPITAL, max_per_trade=200.0,
                      max_concurrent=3):
    """On each signal day, score universes by signal strength and prefer strongest."""
    if not all_signals:
        return []

    signals = sorted(all_signals, key=lambda x: x[0])

    # Group by date
    by_date = defaultdict(list)
    for d, t, u in signals:
        by_date[d].append((t, u))

    deduped_signals = []
    last_entry = {}

    for date in sorted(by_date.keys()):
        items = by_date[date]
        # Score each universe present on this date
        universe_strength = {}
        for _, u in items:
            if u not in universe_strength:
                universe_strength[u] = compute_signal_strength(u, date)

        # Sort items: strongest universe first, then by signal strength within
        items_scored = [(t, u, universe_strength.get(u, 0)) for t, u in items]
        items_scored.sort(key=lambda x: -x[2])

        for ticker, universe, _ in items_scored:
            if ticker in last_entry:
                if (date - last_entry[ticker]).days < HOLD_DAYS:
                    continue
            deduped_signals.append((date, ticker, universe))
            last_entry[ticker] = date

    # Now simulate with overall 3 concurrent limit
    trades = []
    open_positions = []

    for date, ticker, universe in deduped_signals:
        open_positions = [(ed, tk, u) for ed, tk, u in open_positions if ed > date]
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

        exit_loc = min(loc + HOLD_DAYS, len(c) - 1)
        slip = get_slippage(ticker)
        entry_price = float(c.iloc[loc]) * (1 + slip)
        exit_price = float(c.iloc[exit_loc]) * (1 - slip)
        exit_date = c.index[exit_loc]

        if entry_price <= 0:
            continue

        shares = max_per_trade / entry_price
        pnl = shares * (exit_price - entry_price)
        ret = (exit_price - entry_price) / entry_price

        trades.append({
            "ticker": ticker,
            "universe": universe,
            "entry_date": str(c.index[loc].date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(float(pnl), 2),
            "return": round(float(ret), 6),
            "hold_days": HOLD_DAYS,
            "shares": round(float(shares), 4),
        })
        open_positions.append((exit_date, ticker, universe))

    return trades


trades_e = simulate_priority(all_sigs, max_per_trade=200.0, max_concurrent=3)
print(f"  {len(trades_e)} trades generated")

# ── Variant F: All three, reduce Intl when USD strong ────────────────────
print("\n--- Variant F: All three, reduce Intl when USD strong ---")


def simulate_usd_aware(us_sigs, intl_sigs, etf_sigs, capital=CAPITAL,
                       max_concurrent=6):
    """Reduce international allocation when USD is strengthening."""
    combined = []
    for d, t, u in us_sigs:
        combined.append((d, t, u))
    for d, t, u in intl_sigs:
        combined.append((d, t, u))
    for d, t, u in etf_sigs:
        combined.append((d, t, u))
    combined.sort(key=lambda x: x[0])

    deduped = []
    last_entry = {}
    for date, ticker, universe in combined:
        if ticker in last_entry:
            if (date - last_entry[ticker]).days < HOLD_DAYS:
                continue
        deduped.append((date, ticker, universe))
        last_entry[ticker] = date

    trades = []
    open_positions = []

    for date, ticker, universe in deduped:
        open_positions = [(ed, tk, u) for ed, tk, u in open_positions if ed > date]
        if len(open_positions) >= max_concurrent:
            continue

        # If USD is strong, reduce Intl: allow only 1 Intl concurrent (instead of 2)
        usd_strong = is_usd_strong(date)
        if universe == "INTL" and usd_strong:
            intl_count = sum(1 for _, _, u in open_positions if u == "INTL")
            if intl_count >= 1:  # reduced from 2 to 1 when USD strong
                continue

        # Normal per-universe limits otherwise
        per_u_limits = {"US": 3, "INTL": 2, "ETF": 2}
        u_count = sum(1 for _, _, u in open_positions if u == universe)
        if u_count >= per_u_limits.get(universe, 2):
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

        exit_loc = min(loc + HOLD_DAYS, len(c) - 1)
        slip = get_slippage(ticker)

        # When USD strong, reduce Intl position size too
        if universe == "INTL" and usd_strong:
            max_per_trade = 75.0   # reduced from ~107.5
        else:
            max_per_trade = 107.5  # $645 / 6

        entry_price = float(c.iloc[loc]) * (1 + slip)
        exit_price = float(c.iloc[exit_loc]) * (1 - slip)
        exit_date = c.index[exit_loc]

        if entry_price <= 0:
            continue

        shares = max_per_trade / entry_price
        pnl = shares * (exit_price - entry_price)
        ret = (exit_price - entry_price) / entry_price

        trades.append({
            "ticker": ticker,
            "universe": universe,
            "entry_date": str(c.index[loc].date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(float(pnl), 2),
            "return": round(float(ret), 6),
            "hold_days": HOLD_DAYS,
            "shares": round(float(shares), 4),
            "usd_strong": usd_strong,
        })
        open_positions.append((exit_date, ticker, universe))

    return trades


trades_f = simulate_usd_aware(us_sigs, intl_sigs, etf_sigs)
print(f"  {len(trades_f)} trades generated")

# ── Evaluate All Variants ────────────────────────────────────────────────
variant_trades = {
    "A": trades_a,
    "B": trades_b,
    "C": trades_c,
    "D": trades_d,
    "E": trades_e,
    "F": trades_f,
}

variant_descriptions = {
    "A": "US Quality only (BASELINE)",
    "B": "All three equal ($215 each)",
    "C": "All three weighted (50/30/20)",
    "D": "US + Intl only (no ETFs)",
    "E": "All three, prioritize strongest",
    "F": "All three, USD-aware Intl",
}

for var_name in ["A", "B", "C", "D", "E", "F"]:
    trades = variant_trades[var_name]
    desc = variant_descriptions[var_name]
    print(f"\n{'='*60}")
    print(f"Evaluating Variant {var_name}: {desc}")
    print(f"{'='*60}")

    metrics = calc_metrics(trades)
    regime = regime_stratified_sharpe(trades)

    print(f"  Trades: {metrics['n_trades']}, Return: {metrics['total_return_pct']:.1f}%, "
          f"Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}, "
          f"WR: {metrics['win_rate']:.1%}, PF: {metrics['profit_factor']:.2f}")

    # Universe breakdown
    u_breakdown = universe_breakdown(trades)
    for u, um in sorted(u_breakdown.items()):
        print(f"    {u}: {um['n_trades']} trades, Sharpe {um['sharpe']:.3f}, "
              f"WR {um['win_rate']:.1%}, PnL ${um['total_pnl']:.2f}")

    # Permutation test
    print(f"  Running permutation test ({N_PERM} iterations) ...")
    perm_p = permutation_test(trades, N_PERM)
    print(f"  Perm p-value: {perm_p:.4f}")

    gates = five_gate(metrics, regime, perm_p)
    passed = sum(1 for k, v in gates.items() if k != "pass_all" and v)
    print(f"  Gates: {passed}/5 {'PASS' if gates['pass_all'] else 'FAIL'}")

    # Correlation between universes (trade-level)
    corr_data = {}
    for t in trades:
        month = t["entry_date"][:7]
        u = t.get("universe", "US")
        if month not in corr_data:
            corr_data[month] = {}
        if u not in corr_data[month]:
            corr_data[month][u] = []
        corr_data[month][u].append(t["return"])

    # Top tickers
    ticker_counts = defaultdict(int)
    ticker_pnl = defaultdict(float)
    for t in trades:
        ticker_counts[t["ticker"]] += 1
        ticker_pnl[t["ticker"]] += t["pnl"]
    top5 = sorted(ticker_counts.items(), key=lambda x: ticker_pnl[x[0]], reverse=True)[:5]

    results[var_name] = {
        "description": desc,
        "metrics": metrics,
        "regime": regime,
        "perm_p": round(perm_p, 4),
        "gates": gates,
        "universe_breakdown": {u: um for u, um in u_breakdown.items()},
        "top_tickers": {tk: {"trades": cnt, "pnl": round(ticker_pnl[tk], 2)}
                        for tk, cnt in top5},
    }


# ── Print Comparison Table ───────────────────────────────────────────────
print("\n" + "=" * 120)
print("COMPARISON TABLE — MULTI-ASSET DUAL SIGNAL PORTFOLIO")
print("=" * 120)
print(f"{'Var':<4} {'Description':<36} {'Trades':>6} {'Return%':>8} {'Sharpe':>7} "
      f"{'Sortino':>8} {'WR':>6} {'PF':>6} {'MDD%':>7} {'PermP':>6} {'Gates':>6}")
print("-" * 120)

for var_name in ["A", "B", "C", "D", "E", "F"]:
    r = results[var_name]
    m = r["metrics"]
    g = r["gates"]
    passed = sum(1 for k, v in g.items() if k != "pass_all" and v)
    tag = "PASS" if g["pass_all"] else "FAIL"
    print(f"{var_name:<4} {r['description']:<36} {m['n_trades']:>6} {m['total_return_pct']:>7.1f}% "
          f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} "
          f"{m['max_drawdown_pct']:>6.1f}% {r['perm_p']:>6.4f} {passed}/5 {tag}")

# ── Regime Detail ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("REGIME STRATIFICATION")
print("=" * 90)
print(f"{'Var':<4} {'Bull Sharpe':>11} {'Bear Sharpe':>11} {'Gap':>6} {'Bull#':>6} {'Bear#':>6}")
print("-" * 90)
for var_name in ["A", "B", "C", "D", "E", "F"]:
    reg = results[var_name]["regime"]
    print(f"{var_name:<4} {reg['bull_sharpe']:>11.3f} {reg['bear_sharpe']:>11.3f} "
          f"{reg['regime_gap']:>6.3f} {reg['bull_trades']:>6} {reg['bear_trades']:>6}")

# ── Universe Breakdown Comparison ────────────────────────────────────────
print("\n" + "=" * 100)
print("UNIVERSE BREAKDOWN (per-universe metrics for multi-asset variants)")
print("=" * 100)
for var_name in ["B", "C", "D", "E", "F"]:
    r = results[var_name]
    print(f"\n  Variant {var_name}: {r['description']}")
    ub = r.get("universe_breakdown", {})
    for u in ["US", "INTL", "ETF"]:
        if u in ub:
            um = ub[u]
            print(f"    {u:>4}: {um['n_trades']:>3} trades, Sharpe {um['sharpe']:>6.3f}, "
                  f"WR {um['win_rate']:>5.1%}, PF {um['profit_factor']:>5.2f}, "
                  f"PnL ${um['total_pnl']:>7.2f}, MDD {um['max_drawdown_pct']:>6.1f}%")

# ── Diversification Benefit Analysis ─────────────────────────────────────
print("\n" + "=" * 80)
print("DIVERSIFICATION BENEFIT vs US-ONLY BASELINE (Variant A)")
print("=" * 80)

baseline = results["A"]["metrics"]
for var_name in ["B", "C", "D", "E", "F"]:
    m = results[var_name]["metrics"]
    sharpe_delta = m["sharpe"] - baseline["sharpe"]
    mdd_delta = m["max_drawdown_pct"] - baseline["max_drawdown_pct"]
    trade_delta = m["n_trades"] - baseline["n_trades"]
    pnl_delta = m["total_pnl"] - baseline["total_pnl"]

    print(f"  {var_name} ({results[var_name]['description']})")
    print(f"    Sharpe: {m['sharpe']:.3f} vs {baseline['sharpe']:.3f} "
          f"({'+'if sharpe_delta>=0 else ''}{sharpe_delta:.3f})")
    print(f"    MDD:    {m['max_drawdown_pct']:.1f}% vs {baseline['max_drawdown_pct']:.1f}% "
          f"({'+'if mdd_delta>=0 else ''}{mdd_delta:.1f}%)")
    print(f"    Trades: {m['n_trades']} vs {baseline['n_trades']} "
          f"({'+'if trade_delta>=0 else ''}{trade_delta})")
    print(f"    PnL:    ${m['total_pnl']:.2f} vs ${baseline['total_pnl']:.2f} "
          f"({'+'if pnl_delta>=0 else ''}${pnl_delta:.2f})")

# ── Return Correlation Between Universes ─────────────────────────────────
print("\n" + "=" * 70)
print("INTER-UNIVERSE RETURN CORRELATION (monthly avg returns)")
print("=" * 70)

# Use variant B (equal allocation) for correlation analysis
b_trades = variant_trades["B"]
monthly_rets = defaultdict(lambda: defaultdict(list))
for t in b_trades:
    month = t["entry_date"][:7]
    u = t.get("universe", "US")
    monthly_rets[month][u].append(t["return"])

# Build monthly average return series per universe
months = sorted(monthly_rets.keys())
universe_monthly = {u: [] for u in ["US", "INTL", "ETF"]}
common_months = []
for m in months:
    has_all = all(u in monthly_rets[m] for u in ["US", "INTL", "ETF"])
    if has_all:
        common_months.append(m)
        for u in ["US", "INTL", "ETF"]:
            universe_monthly[u].append(np.mean(monthly_rets[m][u]))

if len(common_months) >= 5:
    corr_df = pd.DataFrame(universe_monthly, index=common_months)
    corr_matrix = corr_df.corr()
    print(f"\n  Correlation matrix ({len(common_months)} common months):")
    print(f"  {'':>6} {'US':>8} {'INTL':>8} {'ETF':>8}")
    for u in ["US", "INTL", "ETF"]:
        vals = [f"{corr_matrix.loc[u, u2]:>8.3f}" for u2 in ["US", "INTL", "ETF"]]
        print(f"  {u:>6} {''.join(vals)}")
    avg_corr = (corr_matrix.values.sum() - 3) / 6  # off-diagonal average
    print(f"\n  Average cross-universe correlation: {avg_corr:.3f}")
    if avg_corr < 0.3:
        print("  Low correlation — diversification benefit is STRONG")
    elif avg_corr < 0.6:
        print("  Moderate correlation — some diversification benefit")
    else:
        print("  High correlation — limited diversification benefit")
else:
    print("  Insufficient common months for correlation analysis")

# ── Best Variant ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BEST VARIANTS (by Sharpe, passing all 5 gates)")
print("=" * 70)

passing = {k: v for k, v in results.items() if v["gates"]["pass_all"]}
if passing:
    sorted_pass = sorted(passing.items(), key=lambda x: x[1]["metrics"]["sharpe"], reverse=True)
    for rank, (var, data) in enumerate(sorted_pass, 1):
        m = data["metrics"]
        print(f"  #{rank}: Variant {var} — {data['description']}")
        print(f"         Sharpe {m['sharpe']:.3f}, Sortino {m['sortino']:.3f}, "
              f"WR {m['win_rate']:.1%}, PF {m['profit_factor']:.2f}, "
              f"{m['n_trades']} trades, Return {m['total_return_pct']:.1f}%, "
              f"MDD {m['max_drawdown_pct']:.1f}%")
else:
    print("  No variants passed all 5 gates.")
    sorted_all = sorted(results.items(), key=lambda x: x[1]["metrics"]["sharpe"], reverse=True)
    for rank, (var, data) in enumerate(sorted_all[:3], 1):
        m = data["metrics"]
        g = data["gates"]
        passed = sum(1 for k, v in g.items() if k != "pass_all" and v)
        print(f"  #{rank} (best effort, {passed}/5 gates): Variant {var} — "
              f"Sharpe {m['sharpe']:.3f}, {m['n_trades']} trades, "
              f"Return {m['total_return_pct']:.1f}%")

# ── Save Results ─────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/multi_asset_dual_signal_results.json")

output = {
    "backtest": "Multi-Asset Dual Signal Portfolio",
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "config": {
        "capital": CAPITAL,
        "oot_period": f"{OOT_START} to {END}",
        "hold_days": HOLD_DAYS,
        "n_permutations": N_PERM,
        "universes": {
            "US": {"tickers": US_TICKERS, "dip_pct": 0.05, "min_red": 3, "slippage": 0.0002},
            "INTL": {"tickers": INTL_TICKERS, "dip_pct": 0.05, "min_red": 3, "slippage": 0.0002},
            "ETF": {"tickers": ETF_TICKERS, "dip_pct": 0.03, "min_red": 2, "slippage": 0.0001},
        },
    },
    "signal_counts": {u: len(s) for u, s in universe_signals.items()},
    "variants": results,
    "gates": dict(GATES),
}

output_path.write_text(json.dumps(output, indent=2, default=str))
print(f"\nResults saved to {output_path}")
print("DONE.")
