#!/usr/bin/env python3
"""
Multi-Strategy Combo Backtest
=============================
Tests COMBINING 5 individually validated strategies into a single portfolio.

Strategies:
  1. Quality Mean Reversion (QMR) — buy quality stocks on dip + low RSI
  2. Adaptive RSI — buy growth stocks on very low RSI
  3. Contrarian Sector — buy sector ETFs after monthly drawdowns
  4. PEAD — buy after large positive earnings surprise
  5. IV Run-Up — buy before earnings, sell before event

6 Variants:
  A: Equal allocation ($129 each)
  B: Signal-weighted (Sharpe-based allocation)
  C: Ensemble — trade only when 2+ strategies agree on same ticker
  D: Regime-aware — VIX-based filtering
  E: Sequential — priority-ordered, max 3 concurrent positions
  F: Diversification-maximized — no 2 positions from same strategy

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation test p < 0.05
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades

OOT: Jan 2022 – Jul 2026. Starting capital: $645.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
START = "2020-01-01"    # extra lookback for indicators
END = "2026-07-31"
OOT_START = "2022-01-01"
N_PERM = 1000

# Strategy universes
QMR_TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
GROWTH_TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AVGO", "CRM", "NFLX", "AMD"]
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE"]
PEAD_TICKERS = sorted(set(QMR_TICKERS + GROWTH_TICKERS))

ALL_TICKERS = sorted(set(
    QMR_TICKERS + GROWTH_TICKERS + SECTOR_ETFS + PEAD_TICKERS + ["SPY", "^VIX"]
))

# Variant B weights
VARIANT_B_WEIGHTS = {
    "QMR": 0.25, "IV_RUNUP": 0.25, "PEAD": 0.20, "RSI": 0.15, "CONTRARIAN": 0.15
}

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(ALL_TICKERS, start=START, end=END,
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


def get_volume(ticker):
    try:
        if len(ALL_TICKERS) == 1:
            s = raw["Volume"].dropna()
        else:
            s = raw[ticker]["Volume"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in ALL_TICKERS}
volumes = {t: get_volume(t) for t in ALL_TICKERS}
spy_close = closes.get("SPY", pd.Series(dtype=float))
vix_close = closes.get("^VIX", pd.Series(dtype=float))

loaded = sum(1 for t in ALL_TICKERS if t not in ("SPY", "^VIX") and len(closes.get(t, [])) > 100)
print(f"  Tickers with sufficient data: {loaded}/{len(ALL_TICKERS) - 2}")
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


def monthly_return(series):
    """Trailing 21-day return."""
    return series.pct_change(21)


# ── Pre-compute indicators ───────────────────────────────────────────────
print("Computing indicators ...")
indicators = {}
for t in ALL_TICKERS:
    if t in ("SPY", "^VIX"):
        continue
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) < 50:
        continue
    indicators[t] = {
        "rsi14": calc_rsi(c, 14),
        "high20": rolling_high(c, 20),
        "monthly_ret": monthly_return(c),
    }

# SPY regime classification
spy_ret = spy_close.pct_change()
spy_cumret = (1 + spy_ret).cumprod()

# ── Simulated Earnings Dates ─────────────────────────────────────────────
# We approximate earnings dates as quarterly: mid-Jan, mid-Apr, mid-Jul, mid-Oct
# with some jitter per ticker to avoid exact clustering.
def generate_earnings_dates(ticker, start_date, end_date):
    """Generate approximate quarterly earnings dates for a ticker."""
    np.random.seed(hash(ticker) % 2**31)
    dates = []
    year = start_date.year
    while year <= end_date.year:
        for month in [1, 4, 7, 10]:
            base_day = 15 + np.random.randint(-5, 10)
            base_day = min(base_day, 28)
            try:
                d = datetime(year, month, base_day)
                if start_date <= d <= end_date:
                    dates.append(pd.Timestamp(d))
            except ValueError:
                pass
        year += 1
    return dates


earnings_dates = {}
start_dt = datetime.strptime(START, "%Y-%m-%d")
end_dt = datetime.strptime(END, "%Y-%m-%d")
for t in PEAD_TICKERS:
    earnings_dates[t] = generate_earnings_dates(t, start_dt, end_dt)

# ── Strategy Signal Generation ───────────────────────────────────────────
print("Generating strategy signals ...")

# Each signal: (date, ticker, strategy_name, direction, hold_days)


def generate_qmr_signals():
    """Quality Mean Reversion: buy when drop >5% from 20d high AND RSI(14) < 35. Hold 10d."""
    signals = []
    for t in QMR_TICKERS:
        if t not in indicators:
            continue
        c = closes[t]
        rsi = indicators[t]["rsi14"]
        h20 = indicators[t]["high20"]
        for date in c.index:
            if date < pd.Timestamp(OOT_START):
                continue
            try:
                price = c.loc[date]
                high = h20.loc[date]
                r = rsi.loc[date]
                if pd.isna(price) or pd.isna(high) or pd.isna(r):
                    continue
                drop_pct = (price - high) / high
                if drop_pct < -0.05 and r < 35:
                    signals.append((date, t, "QMR", "long", 10))
            except (KeyError, IndexError):
                continue
    return signals


def generate_rsi_signals():
    """Adaptive RSI: buy growth stocks when RSI(14) < 25. Hold 5d."""
    signals = []
    for t in GROWTH_TICKERS:
        if t not in indicators:
            continue
        c = closes[t]
        rsi = indicators[t]["rsi14"]
        for date in c.index:
            if date < pd.Timestamp(OOT_START):
                continue
            try:
                r = rsi.loc[date]
                if pd.isna(r):
                    continue
                if r < 25:
                    signals.append((date, t, "RSI", "long", 5))
            except (KeyError, IndexError):
                continue
    return signals


def generate_contrarian_signals():
    """Contrarian Sector: buy sector ETF when monthly return < -5%. Hold 10d."""
    signals = []
    for t in SECTOR_ETFS:
        if t not in indicators:
            continue
        c = closes[t]
        mret = indicators[t]["monthly_ret"]
        for date in c.index:
            if date < pd.Timestamp(OOT_START):
                continue
            try:
                mr = mret.loc[date]
                if pd.isna(mr):
                    continue
                if mr < -0.05:
                    signals.append((date, t, "CONTRARIAN", "long", 10))
            except (KeyError, IndexError):
                continue
    return signals


def generate_pead_signals():
    """PEAD: buy after large positive earnings surprise (gap-up >5%). Hold 40d."""
    signals = []
    for t in PEAD_TICKERS:
        if t not in indicators:
            continue
        c = closes[t]
        for ed in earnings_dates.get(t, []):
            # Find the trading day on or after earnings
            mask = c.index >= ed
            if mask.sum() == 0:
                continue
            earn_idx = c.index[mask][0]
            if earn_idx < pd.Timestamp(OOT_START):
                continue
            # Check gap-up: compare earn_day close to prev day close
            loc = c.index.get_loc(earn_idx)
            if loc < 1:
                continue
            prev_close = c.iloc[loc - 1]
            earn_close = c.iloc[loc]
            if pd.isna(prev_close) or pd.isna(earn_close) or prev_close == 0:
                continue
            gap = (earn_close - prev_close) / prev_close
            if gap > 0.05:
                # Enter next day
                if loc + 1 < len(c):
                    entry_date = c.index[loc + 1]
                    signals.append((entry_date, t, "PEAD", "long", 40))
    return signals


def generate_iv_runup_signals():
    """IV Run-Up: buy 7 days before earnings, sell 1 day before. Hold ~6d."""
    signals = []
    for t in PEAD_TICKERS:
        if t not in indicators:
            continue
        c = closes[t]
        for ed in earnings_dates.get(t, []):
            # Find 7 trading days before earnings
            mask = c.index <= ed
            if mask.sum() < 8:
                continue
            pre_dates = c.index[mask]
            if len(pre_dates) < 8:
                continue
            entry_date = pre_dates[-7]  # 7 trading days before
            if entry_date < pd.Timestamp(OOT_START):
                continue
            # Hold for 6 days (sell 1 day before earnings)
            signals.append((entry_date, t, "IV_RUNUP", "long", 6))
    return signals


# Generate all signals
all_signals = {
    "QMR": generate_qmr_signals(),
    "RSI": generate_rsi_signals(),
    "CONTRARIAN": generate_contrarian_signals(),
    "PEAD": generate_pead_signals(),
    "IV_RUNUP": generate_iv_runup_signals(),
}

for name, sigs in all_signals.items():
    print(f"  {name}: {len(sigs)} raw signals")

# ── Trade Simulator ──────────────────────────────────────────────────────
def simulate_trade(ticker, entry_date, hold_days, capital_per_trade):
    """Simulate a single trade with slippage. Returns (pnl, entry_price, exit_price, exit_date)."""
    c = closes.get(ticker, pd.Series(dtype=float))
    if len(c) == 0:
        return None
    try:
        loc = c.index.get_loc(entry_date)
    except KeyError:
        # Find nearest date
        mask = c.index >= entry_date
        if mask.sum() == 0:
            return None
        loc = c.index.get_loc(c.index[mask][0])

    if loc + hold_days >= len(c):
        exit_loc = len(c) - 1
    else:
        exit_loc = loc + hold_days

    entry_price = c.iloc[loc] * (1 + SLIPPAGE_PCT)  # buy with slippage
    exit_price = c.iloc[exit_loc] * (1 - SLIPPAGE_PCT)  # sell with slippage
    exit_date = c.index[exit_loc]

    if entry_price <= 0:
        return None

    shares = int(capital_per_trade / entry_price)
    if shares < 1:
        # Allow fractional for small accounts
        shares = capital_per_trade / entry_price

    pnl = shares * (exit_price - entry_price)
    ret = (exit_price - entry_price) / entry_price

    return {
        "ticker": ticker,
        "entry_date": str(entry_date.date()),
        "exit_date": str(exit_date.date()),
        "entry_price": round(float(entry_price), 2),
        "exit_price": round(float(exit_price), 2),
        "pnl": round(float(pnl), 2),
        "return": round(float(ret), 6),
        "strategy": "",
        "hold_days": hold_days,
    }


def deduplicate_signals(signals, min_gap_days=None):
    """Remove duplicate signals on same ticker within hold period."""
    signals = sorted(signals, key=lambda x: (x[1], x[0]))  # sort by ticker, date
    deduped = []
    last_entry = {}  # ticker -> last entry date
    for date, ticker, strategy, direction, hold_days in signals:
        gap = min_gap_days if min_gap_days else hold_days
        if ticker in last_entry:
            delta = (date - last_entry[ticker]).days
            if delta < gap:
                continue
        deduped.append((date, ticker, strategy, direction, hold_days))
        last_entry[ticker] = date
    return deduped


# ── Metrics Calculation ──────────────────────────────────────────────────
def calc_metrics(trades, capital):
    """Calculate performance metrics from a list of trade dicts."""
    if not trades:
        return {
            "total_return_pct": 0, "sharpe": 0, "sortino": 0,
            "max_drawdown_pct": 0, "win_rate": 0, "profit_factor": 0,
            "n_trades": 0, "total_pnl": 0,
        }

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    total_pnl = pnls.sum()

    # Win rate
    wins = (pnls > 0).sum()
    win_rate = wins / len(pnls) if len(pnls) > 0 else 0

    # Profit factor
    gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss

    # Sharpe (annualized, assuming ~50 trades/year for daily strategies)
    if len(returns) > 1 and returns.std() > 0:
        trades_per_year = max(len(returns) / 4.5, 1)  # ~4.5 year OOT
        sharpe = (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        trades_per_year = max(len(returns) / 4.5, 1)
        sortino = (returns.mean() / downside.std()) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

    # Max drawdown (from cumulative P&L)
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
    }


def regime_stratified_sharpe(trades):
    """Split trades into bull/bear regimes based on SPY trend at entry."""
    if not trades:
        return {"bull_sharpe": 0, "bear_sharpe": 0, "regime_gap": 0}

    spy_sma200 = spy_close.rolling(200).mean()

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
    """Shuffle signal timing to test if returns are due to signal or luck.

    For each permutation, we randomly shift each trade's entry date to a
    different random date in the OOT period (same ticker, same hold period)
    and re-simulate. This tests whether the TIMING of signals matters.
    """
    if len(trades) < 5:
        return 1.0  # Not enough trades

    real_returns = np.array([t["return"] for t in trades])
    real_mean = real_returns.mean()

    # Build tradeable date index per ticker for random sampling
    oot_start_ts = pd.Timestamp(OOT_START)
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
            entry_p = c.iloc[c.index.get_loc(dates[rand_loc])] * (1 + SLIPPAGE_PCT)
            exit_loc = min(c.index.get_loc(dates[rand_loc]) + hold, len(c) - 1)
            exit_p = c.iloc[exit_loc] * (1 - SLIPPAGE_PCT)
            if entry_p > 0:
                perm_returns.append((exit_p - entry_p) / entry_p)
            else:
                perm_returns.append(0.0)

        if np.mean(perm_returns) >= real_mean:
            count_better += 1

    return count_better / n_perm


def five_gate(metrics, regime, perm_p):
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "maxdd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["pass_all"] = all(gates.values())
    return gates


# ── Variant A: Equal Allocation ──────────────────────────────────────────
print("\n=== Variant A: Equal Allocation ===")
capital_per_strategy = CAPITAL / 5  # $129 each

variant_a_trades = []
for strat_name, sigs in all_signals.items():
    deduped = deduplicate_signals(sigs)
    for date, ticker, strategy, direction, hold_days in deduped:
        result = simulate_trade(ticker, date, hold_days, capital_per_strategy)
        if result:
            result["strategy"] = strat_name
            variant_a_trades.append(result)

variant_a_trades.sort(key=lambda x: x["entry_date"])
print(f"  Total trades: {len(variant_a_trades)}")

# ── Variant B: Signal-Weighted ───────────────────────────────────────────
print("\n=== Variant B: Signal-Weighted ===")
variant_b_trades = []
for strat_name, sigs in all_signals.items():
    weight = VARIANT_B_WEIGHTS.get(strat_name, 0.2)
    cap = CAPITAL * weight
    deduped = deduplicate_signals(sigs)
    for date, ticker, strategy, direction, hold_days in deduped:
        result = simulate_trade(ticker, date, hold_days, cap)
        if result:
            result["strategy"] = strat_name
            variant_b_trades.append(result)

variant_b_trades.sort(key=lambda x: x["entry_date"])
print(f"  Total trades: {len(variant_b_trades)}")

# ── Variant C: Ensemble Signal (2+ strategies agree) ─────────────────────
print("\n=== Variant C: Ensemble Signal ===")

# Build date-ticker map of which strategies fire
date_ticker_strategies = defaultdict(lambda: defaultdict(set))
for strat_name, sigs in all_signals.items():
    deduped = deduplicate_signals(sigs)
    for date, ticker, strategy, direction, hold_days in deduped:
        date_ticker_strategies[date][ticker].add(strat_name)

ensemble_cap = CAPITAL / 3  # double position size, fewer trades expected
variant_c_trades = []
seen_c = set()
for date in sorted(date_ticker_strategies.keys()):
    for ticker, strats in date_ticker_strategies[date].items():
        if len(strats) >= 2:
            key = (date, ticker)
            if key in seen_c:
                continue
            seen_c.add(key)
            # Use average hold of agreeing strategies
            hold_map = {"QMR": 10, "RSI": 5, "CONTRARIAN": 10, "PEAD": 40, "IV_RUNUP": 6}
            avg_hold = int(np.mean([hold_map.get(s, 10) for s in strats]))
            result = simulate_trade(ticker, date, avg_hold, ensemble_cap)
            if result:
                result["strategy"] = "+".join(sorted(strats))
                variant_c_trades.append(result)

variant_c_trades.sort(key=lambda x: x["entry_date"])
print(f"  Total trades: {len(variant_c_trades)}")

# ── Variant D: Regime-Aware Combo ────────────────────────────────────────
print("\n=== Variant D: Regime-Aware ===")
# VIX < 20: run all. VIX 20-25: 50% size. VIX > 25: only QMR + Contrarian
variant_d_trades = []
for strat_name, sigs in all_signals.items():
    deduped = deduplicate_signals(sigs)
    for date, ticker, strategy, direction, hold_days in deduped:
        try:
            vix_val = vix_close.asof(date)
        except Exception:
            vix_val = 20  # default
        if pd.isna(vix_val):
            vix_val = 20

        # Filter by VIX regime
        if vix_val > 25 and strat_name not in ("QMR", "CONTRARIAN"):
            continue

        # Position sizing by VIX
        cap = capital_per_strategy
        if 20 <= vix_val <= 25:
            cap *= 0.5

        result = simulate_trade(ticker, date, hold_days, cap)
        if result:
            result["strategy"] = strat_name
            result["vix_at_entry"] = round(float(vix_val), 1)
            variant_d_trades.append(result)

variant_d_trades.sort(key=lambda x: x["entry_date"])
print(f"  Total trades: {len(variant_d_trades)}")

# ── Variant E: Sequential Priority ──────────────────────────────────────
print("\n=== Variant E: Sequential ===")
# Priority: QMR > RSI > PEAD > IV_RUNUP > CONTRARIAN. Max 3 concurrent, $200 each.
PRIORITY_ORDER = ["QMR", "RSI", "PEAD", "IV_RUNUP", "CONTRARIAN"]
all_sigs_sorted = []
for strat_name in PRIORITY_ORDER:
    deduped = deduplicate_signals(all_signals[strat_name])
    for sig in deduped:
        all_sigs_sorted.append(sig)

all_sigs_sorted.sort(key=lambda x: (x[0], PRIORITY_ORDER.index(x[2])))

variant_e_trades = []
active_positions = []  # list of (exit_date, ticker)
MAX_CONCURRENT = 3
POS_SIZE_E = 200.0

for date, ticker, strategy, direction, hold_days in all_sigs_sorted:
    # Expire old positions
    active_positions = [(ed, t) for ed, t in active_positions if ed > date]

    if len(active_positions) >= MAX_CONCURRENT:
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

    exit_loc = min(loc + hold_days, len(c) - 1)
    exit_date = c.index[exit_loc]

    result = simulate_trade(ticker, date, hold_days, POS_SIZE_E)
    if result:
        result["strategy"] = strategy
        variant_e_trades.append(result)
        active_positions.append((exit_date, ticker))

variant_e_trades.sort(key=lambda x: x["entry_date"])
print(f"  Total trades: {len(variant_e_trades)}")

# ── Variant F: Diversification-Maximized ─────────────────────────────────
print("\n=== Variant F: Diversification-Maximized ===")
# Never 2 positions from same strategy simultaneously. Rotate by freshest signal.
variant_f_trades = []
active_strats = {}  # strategy -> exit_date
POS_SIZE_F = CAPITAL / 5

all_sigs_f = []
for strat_name, sigs in all_signals.items():
    deduped = deduplicate_signals(sigs)
    for sig in deduped:
        all_sigs_f.append(sig)

all_sigs_f.sort(key=lambda x: x[0])

for date, ticker, strategy, direction, hold_days in all_sigs_f:
    # Expire old positions
    active_strats = {s: ed for s, ed in active_strats.items() if ed > date}

    if strategy in active_strats:
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

    exit_loc = min(loc + hold_days, len(c) - 1)
    exit_date = c.index[exit_loc]

    result = simulate_trade(ticker, date, hold_days, POS_SIZE_F)
    if result:
        result["strategy"] = strategy
        variant_f_trades.append(result)
        active_strats[strategy] = exit_date

variant_f_trades.sort(key=lambda x: x["entry_date"])
print(f"  Total trades: {len(variant_f_trades)}")

# ── QMR Standalone Baseline ─────────────────────────────────────────────
print("\n=== QMR Standalone Baseline ===")
qmr_standalone_trades = []
qmr_deduped = deduplicate_signals(all_signals["QMR"])
for date, ticker, strategy, direction, hold_days in qmr_deduped:
    result = simulate_trade(ticker, date, hold_days, CAPITAL)
    if result:
        result["strategy"] = "QMR"
        qmr_standalone_trades.append(result)

qmr_standalone_trades.sort(key=lambda x: x["entry_date"])
print(f"  QMR standalone trades: {len(qmr_standalone_trades)}")

# ── Evaluate All Variants ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("EVALUATION")
print("=" * 80)

variants = {
    "A_equal_alloc": variant_a_trades,
    "B_signal_weighted": variant_b_trades,
    "C_ensemble": variant_c_trades,
    "D_regime_aware": variant_d_trades,
    "E_sequential": variant_e_trades,
    "F_diversified": variant_f_trades,
    "QMR_standalone": qmr_standalone_trades,
}

results = {}
for name, trades in variants.items():
    print(f"\n--- {name} ---")
    metrics = calc_metrics(trades, CAPITAL)
    regime = regime_stratified_sharpe(trades)

    print(f"  Running permutation test ({N_PERM} iterations) ...")
    perm_p = permutation_test(trades, N_PERM)
    gates = five_gate(metrics, regime, perm_p)

    results[name] = {
        "metrics": metrics,
        "regime": regime,
        "perm_p": round(perm_p, 4),
        "gates": gates,
    }

    # Per-strategy breakdown for combo variants
    if name != "QMR_standalone":
        strat_breakdown = defaultdict(list)
        for t in trades:
            strat_breakdown[t["strategy"]].append(t["return"])
        sub = {}
        for s, rets in strat_breakdown.items():
            arr = np.array(rets)
            sub[s] = {
                "n_trades": len(rets),
                "avg_return": round(float(arr.mean()) * 100, 3),
                "win_rate": round(float((arr > 0).mean()), 3),
            }
        results[name]["strategy_breakdown"] = sub

    print(f"  Trades:   {metrics['n_trades']}")
    print(f"  Return:   {metrics['total_return_pct']:.1f}%  (${metrics['total_pnl']:.0f})")
    print(f"  Sharpe:   {metrics['sharpe']:.3f}")
    print(f"  Sortino:  {metrics['sortino']:.3f}")
    print(f"  MaxDD:    {metrics['max_drawdown_pct']:.1f}%")
    print(f"  WinRate:  {metrics['win_rate']:.1%}")
    print(f"  PF:       {metrics['profit_factor']:.2f}")
    print(f"  Bull/Bear Sharpe: {regime['bull_sharpe']:.3f} / {regime['bear_sharpe']:.3f}")
    print(f"  Regime Gap: {regime['regime_gap']:.3f}")
    print(f"  Perm p:   {perm_p:.4f}")
    print(f"  5-Gate:   {'PASS' if gates['pass_all'] else 'FAIL'} — {gates}")

# ── Comparison Summary ───────────────────────────────────────────────────
print("\n" + "=" * 80)
print("COMPARISON: Combo Variants vs QMR Standalone")
print("=" * 80)

qmr_sharpe = results["QMR_standalone"]["metrics"]["sharpe"]

print(f"\n{'Variant':<25} {'Trades':>6} {'Return%':>8} {'Sharpe':>7} {'Sortino':>8} "
      f"{'MaxDD%':>7} {'WR':>6} {'PF':>6} {'RegGap':>7} {'p-val':>6} {'5G':>5} {'vs QMR':>8}")
print("-" * 110)

for name in ["A_equal_alloc", "B_signal_weighted", "C_ensemble", "D_regime_aware",
             "E_sequential", "F_diversified", "QMR_standalone"]:
    r = results[name]
    m = r["metrics"]
    rg = r["regime"]
    g = r["gates"]
    delta = m["sharpe"] - qmr_sharpe if name != "QMR_standalone" else 0
    flag = "PASS" if g["pass_all"] else "FAIL"
    print(f"{name:<25} {m['n_trades']:>6} {m['total_return_pct']:>7.1f}% "
          f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['max_drawdown_pct']:>6.1f}% "
          f"{m['win_rate']:>5.1%} {m['profit_factor']:>5.2f} {rg['regime_gap']:>7.3f} "
          f"{r['perm_p']:>6.4f} {flag:>5} {delta:>+7.3f}")

# ── Best variant ─────────────────────────────────────────────────────────
passing = {k: v for k, v in results.items()
           if v["gates"]["pass_all"] and k != "QMR_standalone"}

if passing:
    best = max(passing.items(), key=lambda x: x[1]["metrics"]["sharpe"])
    print(f"\nBest passing combo: {best[0]} (Sharpe {best[1]['metrics']['sharpe']:.3f})")
    better_than_qmr = best[1]["metrics"]["sharpe"] > qmr_sharpe
    print(f"Better than QMR standalone? {'YES' if better_than_qmr else 'NO'} "
          f"(delta: {best[1]['metrics']['sharpe'] - qmr_sharpe:+.3f})")
else:
    print("\nNo variant passed all 5 gates.")
    # Find best anyway
    best = max(results.items(), key=lambda x: x[1]["metrics"]["sharpe"]
               if x[0] != "QMR_standalone" else -999)
    print(f"Best combo (did not pass all gates): {best[0]} "
          f"(Sharpe {best[1]['metrics']['sharpe']:.3f})")

# ── Save Results ─────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/multi_strategy_combo_results.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Convert trades for JSON serialization (remove numpy types)
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

save_data = {
    "config": {
        "capital": CAPITAL,
        "slippage_pct": SLIPPAGE_PCT,
        "oot_period": f"{OOT_START} to {END}",
        "n_permutations": N_PERM,
    },
    "results": clean_for_json(results),
    "trade_counts": {name: len(trades) for name, trades in variants.items()},
    "generated_at": datetime.now().isoformat(),
}

with open(output_path, "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nResults saved to {output_path}")
print("Done.")
