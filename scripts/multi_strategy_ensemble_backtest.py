#!/usr/bin/env python3
"""
Multi-Strategy Ensemble Portfolio Backtest
==========================================
Combine 4 PROVEN validated strategies into an optimal portfolio allocation.

Strategies (all on same 20-stock quality universe):
  1. Dual Signal D: >5% dip from 20d high + RSI<35 + first green after 3+ red days. Hold 10d.
  2. Multi-TF L: Same as Dual Signal D + weekly RSI declining 2+ weeks. Hold 10d.
  3. Quality MR-A: >5% dip from 20d high + RSI<35. Hold 10d.
  4. RSI Divergence C: Price new 20d low + higher RSI + declining volume. Hold 10d.

Variants:
  A: Equal allocation ($161.25 each)
  B: Sharpe-weighted (proportional to validated Sharpe)
  C: Signal agreement (2+ strategies agree => full $200 position)
  D: Cascade priority (Multi-TF L > Dual Signal D > RSI Div > QMR-A)
  E: Anti-correlation (deprioritize recently-fired strategies)
  F: Regime-adaptive (bull=momentum bias, bear=MR bias)

5-Gate Validation + Permutation Test

OOT: Jan 2022 - Jul 2026 | Capital: $645 | Max $200/trade | Max 3 concurrent
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
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

# ── Configuration ────────────────────────────────────────────────────────
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 2bps each way
DATA_START = "2020-01-01"  # lookback for indicators
OOT_START = "2022-01-01"
END = "2026-07-31"
HOLD_DAYS = 10
N_PERM = 1000

# Validated Sharpe ratios for weighting
STRATEGY_SHARPES = {
    "dual_signal_d": 1.77,
    "multi_tf_l": 1.96,
    "quality_mr_a": 1.03,
    "rsi_divergence_c": 1.75,
}

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

# ── Data Download ────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(ALL_TICKERS, start=DATA_START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_ohlcv(ticker):
    """Return DataFrame with Open, High, Low, Close, Volume for a ticker."""
    try:
        tk = ticker if ticker != "VIX" else "^VIX"
        df = raw[tk][["Open", "High", "Low", "Close", "Volume"]].dropna()
        # Flatten multi-level columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def get_close(ticker):
    try:
        tk = ticker if ticker != "VIX" else "^VIX"
        s = raw[tk]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


ohlcv = {t: get_ohlcv(t) for t in TICKERS}
closes = {t: get_close(t) for t in TICKERS}
spy_close = get_close("SPY")
vix_close = get_close("^VIX")

loaded = sum(1 for t in TICKERS if len(closes.get(t, [])) > 100)
print(f"  Tickers with data: {loaded}/{len(TICKERS)}")
print(f"  SPY rows: {len(spy_close)}, VIX rows: {len(vix_close)}")

# ── Indicator Helpers ────────────────────────────────────────────────────
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


def rolling_low(series, period=20):
    return series.rolling(period).min()


def consecutive_red_days(series):
    """Count of consecutive red (close < prior close) days ending at each date."""
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


def weekly_rsi_declining(daily_close, n_weeks=2):
    """Return True on dates where weekly RSI has declined for n_weeks+ consecutive weeks."""
    weekly = daily_close.resample('W-FRI').last().dropna()
    w_rsi = calc_rsi(weekly, 14)
    w_rsi_diff = w_rsi.diff()
    # Count consecutive declining weeks
    declining_count = pd.Series(0, index=weekly.index, dtype=int)
    count = 0
    for i in range(len(w_rsi_diff)):
        if pd.notna(w_rsi_diff.iloc[i]) and w_rsi_diff.iloc[i] < 0:
            count += 1
        else:
            count = 0
        declining_count.iloc[i] = count
    # Map back to daily: each day gets the most recent weekly value
    declining_daily = declining_count.reindex(daily_close.index, method='ffill')
    return declining_daily >= n_weeks


# ── Precompute Indicators ───────────────────────────────────────────────
print("Computing indicators ...")
indicators = {}
for t in TICKERS:
    c = closes[t]
    if len(c) < 50:
        continue
    df_ind = pd.DataFrame(index=c.index)
    df_ind['close'] = c
    df_ind['rsi'] = calc_rsi(c)
    df_ind['high_20d'] = rolling_high(c, 20)
    df_ind['low_20d'] = rolling_low(c, 20)
    df_ind['dip_pct'] = (c - df_ind['high_20d']) / df_ind['high_20d']
    df_ind['consec_red'] = consecutive_red_days(c)
    df_ind['is_green'] = (c.diff() > 0).astype(int)
    df_ind['prev_consec_red'] = df_ind['consec_red'].shift(1)
    df_ind['weekly_rsi_declining'] = weekly_rsi_declining(c)

    # Volume data for RSI Divergence
    if t in ohlcv and len(ohlcv[t]) > 50 and 'Volume' in ohlcv[t].columns:
        vol = ohlcv[t]['Volume']
        vol = vol.reindex(c.index)
        df_ind['volume'] = vol
        df_ind['vol_sma20'] = vol.rolling(20).mean()
    else:
        df_ind['volume'] = np.nan
        df_ind['vol_sma20'] = np.nan

    # RSI divergence: price new 20d low but RSI higher than RSI at prev 20d low
    df_ind['is_20d_low'] = (c <= df_ind['low_20d']).astype(int)
    # Track RSI at previous 20d low
    rsi_at_low = pd.Series(np.nan, index=c.index)
    last_low_rsi = np.nan
    for i in range(len(df_ind)):
        if df_ind['is_20d_low'].iloc[i] == 1:
            if pd.notna(last_low_rsi):
                rsi_at_low.iloc[i] = last_low_rsi
            last_low_rsi = df_ind['rsi'].iloc[i]
    df_ind['prev_low_rsi'] = rsi_at_low

    indicators[t] = df_ind

# SPY for regime detection
spy_sma200 = spy_close.rolling(200).mean()

print(f"  Indicators computed for {len(indicators)} tickers")

# ── Signal Generation ────────────────────────────────────────────────────
def generate_signals(ticker, date_idx):
    """Return dict of {strategy_name: True/False} for a given ticker on a given date."""
    if ticker not in indicators:
        return {}
    ind = indicators[ticker]
    if date_idx not in ind.index:
        return {}

    try:
        row = ind.loc[date_idx]
    except KeyError:
        return {}

    signals = {}

    dip_ok = pd.notna(row['dip_pct']) and row['dip_pct'] < -0.05
    rsi_low = pd.notna(row['rsi']) and row['rsi'] < 35

    # Strategy 1: Dual Signal D
    # >5% dip + RSI<35 + first green after 3+ red days
    first_green_after_red = (row.get('is_green', 0) == 1 and
                              pd.notna(row.get('prev_consec_red')) and
                              row.get('prev_consec_red', 0) >= 3)
    signals['dual_signal_d'] = bool(dip_ok and rsi_low and first_green_after_red)

    # Strategy 2: Multi-TF L
    # Same as Dual Signal D + weekly RSI declining 2+ weeks
    weekly_declining = bool(row.get('weekly_rsi_declining', False))
    signals['multi_tf_l'] = bool(signals['dual_signal_d'] and weekly_declining)

    # Strategy 3: Quality MR-A
    # >5% dip + RSI<35 (simplest)
    signals['quality_mr_a'] = bool(dip_ok and rsi_low)

    # Strategy 4: RSI Divergence C
    # Price new 20d low + higher RSI than at prev 20d low + declining volume
    is_new_low = bool(row.get('is_20d_low', 0) == 1)
    rsi_higher = (pd.notna(row.get('prev_low_rsi')) and
                  pd.notna(row.get('rsi')) and
                  row['rsi'] > row['prev_low_rsi'])
    vol_declining = (pd.notna(row.get('volume')) and
                     pd.notna(row.get('vol_sma20')) and
                     row['volume'] < row['vol_sma20'])
    signals['rsi_divergence_c'] = bool(is_new_low and rsi_higher and vol_declining)

    return signals


# ── Generate All Signals Matrix ─────────────────────────────────────────
print("Generating signal matrix ...")
oot_dates = spy_close.loc[OOT_START:END].index
strategy_names = ['dual_signal_d', 'multi_tf_l', 'quality_mr_a', 'rsi_divergence_c']

# signals_matrix[date][ticker] = {strat: True/False}
signals_matrix = {}
for dt in oot_dates:
    signals_matrix[dt] = {}
    for t in TICKERS:
        sigs = generate_signals(t, dt)
        if sigs:
            signals_matrix[dt][t] = sigs

print(f"  Signal matrix built: {len(oot_dates)} trading days")

# Count total signals per strategy
for sn in strategy_names:
    count = sum(1 for dt in signals_matrix for t in signals_matrix[dt]
                if signals_matrix[dt][t].get(sn, False))
    print(f"    {sn}: {count} raw signals")


# ── Backtest Engine ──────────────────────────────────────────────────────
class Position:
    def __init__(self, ticker, entry_date, entry_price, shares, strategy, hold_days=HOLD_DAYS):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.strategy = strategy
        self.hold_days = hold_days
        self.exit_date = None
        self.exit_price = None
        self.pnl = None


def run_backtest(variant_name, position_selector):
    """
    Generic backtest engine.
    position_selector(date, signals_for_date, active_positions, capital_available, trade_history)
      -> list of (ticker, strategy_name, allocation_amount)
    """
    positions = []  # active
    closed = []     # completed trades
    equity_curve = []
    daily_returns = []
    prev_equity = CAPITAL

    for dt in oot_dates:
        # Check exits first
        still_active = []
        for pos in positions:
            # Count trading days held
            entry_idx = oot_dates.get_loc(pos.entry_date) if pos.entry_date in oot_dates else -1
            current_idx = oot_dates.get_loc(dt)
            days_held = current_idx - entry_idx if entry_idx >= 0 else 999

            if days_held >= pos.hold_days:
                # Exit
                c = closes.get(pos.ticker)
                if c is not None and dt in c.index:
                    exit_price = c.loc[dt] * (1 - SLIPPAGE_PCT)
                    pos.exit_date = dt
                    pos.exit_price = exit_price
                    pos.pnl = (exit_price - pos.entry_price) * pos.shares
                    closed.append(pos)
                else:
                    still_active.append(pos)
            else:
                still_active.append(pos)
        positions = still_active

        # Capital available
        invested = sum(p.entry_price * p.shares for p in positions)
        capital_available = CAPITAL - invested
        slots_available = MAX_CONCURRENT - len(positions)

        if slots_available > 0 and capital_available > 10:
            sigs = signals_matrix.get(dt, {})
            new_trades = position_selector(dt, sigs, positions, capital_available, closed)

            for ticker, strat_name, alloc in new_trades:
                if slots_available <= 0 or capital_available < 10:
                    break
                # Don't double up on same ticker
                if any(p.ticker == ticker for p in positions):
                    continue

                c = closes.get(ticker)
                if c is None or dt not in c.index:
                    continue
                entry_price = c.loc[dt] * (1 + SLIPPAGE_PCT)
                alloc = min(alloc, capital_available, MAX_PER_TRADE)
                shares = alloc / entry_price
                if shares < 0.001:
                    continue

                pos = Position(ticker, dt, entry_price, shares, strat_name)
                positions.append(pos)
                capital_available -= alloc
                slots_available -= 1

        # Mark-to-market equity
        mtm = CAPITAL - sum(p.entry_price * p.shares for p in positions)  # cash
        for p in positions:
            c = closes.get(p.ticker)
            if c is not None and dt in c.index:
                mtm += c.loc[dt] * p.shares
            else:
                mtm += p.entry_price * p.shares
        mtm += sum(t.pnl for t in closed)
        equity_curve.append((dt, mtm))

        daily_ret = (mtm - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_returns.append(daily_ret)
        prev_equity = mtm

    # Force close any remaining positions at end
    for pos in positions:
        c = closes.get(pos.ticker)
        last_date = oot_dates[-1]
        if c is not None and last_date in c.index:
            pos.exit_date = last_date
            pos.exit_price = c.loc[last_date] * (1 - SLIPPAGE_PCT)
            pos.pnl = (pos.exit_price - pos.entry_price) * pos.shares
            closed.append(pos)

    return closed, daily_returns, equity_curve


# ── Variant Selectors ────────────────────────────────────────────────────
def variant_a_equal(dt, sigs, active, cap_avail, history):
    """Equal allocation: each strategy gets $161.25."""
    alloc_per_strat = CAPITAL / 4.0  # $161.25
    trades = []
    for t, strat_sigs in sigs.items():
        for sn in strategy_names:
            if strat_sigs.get(sn, False):
                trades.append((t, sn, alloc_per_strat))
    return trades


def variant_b_sharpe_weighted(dt, sigs, active, cap_avail, history):
    """Sharpe-weighted allocation."""
    total_sharpe = sum(STRATEGY_SHARPES.values())
    trades = []
    for t, strat_sigs in sigs.items():
        for sn in strategy_names:
            if strat_sigs.get(sn, False):
                alloc = CAPITAL * (STRATEGY_SHARPES[sn] / total_sharpe)
                trades.append((t, sn, alloc))
    return trades


def variant_c_agreement(dt, sigs, active, cap_avail, history):
    """Only trade when 2+ strategies agree on same stock same day."""
    trades = []
    for t, strat_sigs in sigs.items():
        firing = [sn for sn in strategy_names if strat_sigs.get(sn, False)]
        if len(firing) >= 2:
            trades.append((t, firing[0], MAX_PER_TRADE))
    return trades


def variant_d_cascade(dt, sigs, active, cap_avail, history):
    """Priority cascade: Multi-TF L > Dual Signal D > RSI Div > QMR-A."""
    priority = ['multi_tf_l', 'dual_signal_d', 'rsi_divergence_c', 'quality_mr_a']
    for sn in priority:
        candidates = [(t, sn, MAX_PER_TRADE) for t, strat_sigs in sigs.items()
                       if strat_sigs.get(sn, False)]
        if candidates:
            return candidates[:MAX_CONCURRENT - len(active)]
    return []


def variant_e_anticorrelation(dt, sigs, active, cap_avail, history):
    """Anti-correlation: deprioritize recently-fired strategies."""
    # Track which strategies fired in last 5 days
    recent_strats = defaultdict(int)
    dt_idx = oot_dates.get_loc(dt)
    for trade in history:
        entry_idx = oot_dates.get_loc(trade.entry_date) if trade.entry_date in oot_dates else -1
        if entry_idx >= 0 and dt_idx - entry_idx <= 5:
            recent_strats[trade.strategy] += 1

    # Sort strategies by least recently used
    strat_order = sorted(strategy_names, key=lambda s: recent_strats.get(s, 0))

    trades = []
    for sn in strat_order:
        for t, strat_sigs in sigs.items():
            if strat_sigs.get(sn, False):
                trades.append((t, sn, MAX_PER_TRADE))
    return trades


def variant_f_regime_adaptive(dt, sigs, active, cap_avail, history):
    """Regime-adaptive: bull=less selective, bear=MR focus."""
    is_bull = True
    if dt in spy_sma200.index and pd.notna(spy_sma200.loc[dt]):
        spy_val = spy_close.loc[dt] if dt in spy_close.index else None
        if spy_val is not None:
            is_bull = spy_val > spy_sma200.loc[dt]

    if is_bull:
        # Bull: use all strategies but prefer dual signal / multi-tf
        priority = ['dual_signal_d', 'multi_tf_l', 'rsi_divergence_c', 'quality_mr_a']
        weights = {'dual_signal_d': 1.3, 'multi_tf_l': 1.3,
                   'rsi_divergence_c': 0.8, 'quality_mr_a': 0.8}
    else:
        # Bear: MR strategies stronger
        priority = ['quality_mr_a', 'rsi_divergence_c', 'dual_signal_d', 'multi_tf_l']
        weights = {'quality_mr_a': 1.5, 'rsi_divergence_c': 1.3,
                   'dual_signal_d': 0.8, 'multi_tf_l': 0.8}

    trades = []
    for sn in priority:
        for t, strat_sigs in sigs.items():
            if strat_sigs.get(sn, False):
                alloc = MAX_PER_TRADE * weights.get(sn, 1.0)
                alloc = min(alloc, MAX_PER_TRADE)
                trades.append((t, sn, alloc))
    return trades


VARIANTS = {
    "A_equal_alloc": variant_a_equal,
    "B_sharpe_weighted": variant_b_sharpe_weighted,
    "C_signal_agreement": variant_c_agreement,
    "D_cascade_priority": variant_d_cascade,
    "E_anti_correlation": variant_e_anticorrelation,
    "F_regime_adaptive": variant_f_regime_adaptive,
}


# ── Metrics Calculation ──────────────────────────────────────────────────
def calc_metrics(closed_trades, daily_returns, equity_curve):
    """Calculate comprehensive metrics."""
    if not closed_trades:
        return {
            "n_trades": 0, "win_rate": 0, "total_pnl": 0, "sharpe": 0,
            "sortino": 0, "profit_factor": 0, "max_drawdown_pct": 0,
            "avg_pnl_per_trade": 0, "avg_win": 0, "avg_loss": 0,
            "best_trade": 0, "worst_trade": 0, "exposure_pct": 0,
        }

    pnls = [t.pnl for t in closed_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    n_trades = len(pnls)
    win_rate = len(wins) / n_trades if n_trades > 0 else 0

    # Annualized Sharpe (daily returns)
    dr = np.array(daily_returns)
    sharpe = (np.mean(dr) / np.std(dr) * np.sqrt(252)) if np.std(dr) > 0 else 0

    # Sortino
    downside = dr[dr < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-9
    sortino = (np.mean(dr) / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Profit factor
    gross_wins = sum(wins) if wins else 0
    gross_losses = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Max drawdown
    eq_vals = [e[1] for e in equity_curve]
    peak = CAPITAL
    max_dd = 0
    for v in eq_vals:
        peak = max(peak, v)
        dd = (v - peak) / peak * 100
        max_dd = min(max_dd, dd)

    # Exposure: fraction of days with active positions
    days_with_pos = sum(1 for v in eq_vals if abs(v - CAPITAL) > 0.01)
    exposure = days_with_pos / len(eq_vals) * 100 if eq_vals else 0

    # Strategy breakdown
    strat_counts = defaultdict(int)
    strat_pnl = defaultdict(float)
    for t in closed_trades:
        strat_counts[t.strategy] += 1
        strat_pnl[t.strategy] += t.pnl

    return {
        "n_trades": n_trades,
        "win_rate": round(win_rate * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(min(profit_factor, 99.9), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_pnl_per_trade": round(total_pnl / n_trades, 2) if n_trades else 0,
        "avg_win": round(np.mean(wins), 2) if wins else 0,
        "avg_loss": round(np.mean(losses), 2) if losses else 0,
        "best_trade": round(max(pnls), 2) if pnls else 0,
        "worst_trade": round(min(pnls), 2) if pnls else 0,
        "exposure_pct": round(exposure, 1),
        "strategy_breakdown": {
            s: {"trades": strat_counts[s], "pnl": round(strat_pnl[s], 2)}
            for s in sorted(strat_counts.keys())
        }
    }


# ── Regime Analysis ──────────────────────────────────────────────────────
def regime_analysis(closed_trades, equity_curve):
    """Split performance by bull/bear regime."""
    bull_pnls = []
    bear_pnls = []

    for t in closed_trades:
        dt = t.entry_date
        if dt in spy_sma200.index and pd.notna(spy_sma200.loc[dt]):
            spy_val = spy_close.loc[dt] if dt in spy_close.index else None
            if spy_val is not None:
                if spy_val > spy_sma200.loc[dt]:
                    bull_pnls.append(t.pnl)
                else:
                    bear_pnls.append(t.pnl)

    def pnl_sharpe(pnl_list):
        if len(pnl_list) < 2:
            return 0
        return np.mean(pnl_list) / np.std(pnl_list) if np.std(pnl_list) > 0 else 0

    bull_sharpe = pnl_sharpe(bull_pnls)
    bear_sharpe = pnl_sharpe(bear_pnls)
    max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_sharpe

    return {
        "bull_trades": len(bull_pnls),
        "bear_trades": len(bear_pnls),
        "bull_pnl": round(sum(bull_pnls), 2) if bull_pnls else 0,
        "bear_pnl": round(sum(bear_pnls), 2) if bear_pnls else 0,
        "bull_wr": round(sum(1 for p in bull_pnls if p > 0) / len(bull_pnls) * 100, 1) if bull_pnls else 0,
        "bear_wr": round(sum(1 for p in bear_pnls if p > 0) / len(bear_pnls) * 100, 1) if bear_pnls else 0,
        "bull_sharpe_proxy": round(bull_sharpe, 3),
        "bear_sharpe_proxy": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
    }


# ── Permutation Test ─────────────────────────────────────────────────────
def permutation_test(closed_trades, n_perm=N_PERM):
    """Shuffle trade PnLs and compare realized Sharpe to random."""
    if len(closed_trades) < 5:
        return {"p_value": 1.0, "realized_sharpe_proxy": 0, "null_mean": 0}

    pnls = np.array([t.pnl for t in closed_trades])
    realized = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0

    rng = np.random.RandomState(42)
    count_better = 0
    null_sharpes = []
    for _ in range(n_perm):
        shuffled = rng.choice(pnls, size=len(pnls), replace=True)
        # Randomly flip signs to break date alignment
        signs = rng.choice([-1, 1], size=len(pnls))
        shuffled = pnls * signs
        s = np.mean(shuffled) / np.std(shuffled) if np.std(shuffled) > 0 else 0
        null_sharpes.append(s)
        if s >= realized:
            count_better += 1

    return {
        "p_value": round(count_better / n_perm, 4),
        "realized_sharpe_proxy": round(realized, 4),
        "null_mean": round(np.mean(null_sharpes), 4),
    }


# ── 5-Gate Validation ────────────────────────────────────────────────────
def validate_5gate(metrics, regime, perm):
    """Apply 5-gate validation."""
    gates = {}

    # Gate 1: Minimum Sharpe
    gates["G1_sharpe"] = {
        "threshold": GATES["sharpe_min"],
        "value": metrics["sharpe"],
        "pass": metrics["sharpe"] >= GATES["sharpe_min"],
    }

    # Gate 2: Permutation significance
    gates["G2_perm_test"] = {
        "threshold": GATES["perm_p_max"],
        "value": perm["p_value"],
        "pass": perm["p_value"] <= GATES["perm_p_max"],
    }

    # Gate 3: Regime robustness
    gates["G3_regime_gap"] = {
        "threshold": GATES["regime_gap_max"],
        "value": regime["regime_gap"],
        "pass": regime["regime_gap"] <= GATES["regime_gap_max"],
    }

    # Gate 4: Max drawdown
    gates["G4_max_dd"] = {
        "threshold": GATES["mdd_min"],
        "value": metrics["max_drawdown_pct"],
        "pass": metrics["max_drawdown_pct"] >= GATES["mdd_min"],
    }

    # Gate 5: Minimum trades
    gates["G5_min_trades"] = {
        "threshold": GATES["min_trades"],
        "value": metrics["n_trades"],
        "pass": metrics["n_trades"] >= GATES["min_trades"],
    }

    all_pass = all(g["pass"] for g in gates.values())
    n_pass = sum(1 for g in gates.values() if g["pass"])

    return {
        "gates": gates,
        "gates_passed": f"{n_pass}/5",
        "all_pass": all_pass,
    }


# ── Run All Variants ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MULTI-STRATEGY ENSEMBLE PORTFOLIO BACKTEST")
print("=" * 70)

results = {}

for vname, selector in VARIANTS.items():
    print(f"\n--- Variant {vname} ---")
    closed, daily_rets, eq_curve = run_backtest(vname, selector)

    metrics = calc_metrics(closed, daily_rets, eq_curve)
    regime = regime_analysis(closed, eq_curve)
    perm = permutation_test(closed)
    validation = validate_5gate(metrics, regime, perm)

    results[vname] = {
        "metrics": metrics,
        "regime_analysis": regime,
        "permutation_test": perm,
        "validation": validation,
    }

    print(f"  Trades: {metrics['n_trades']:>4}  |  WR: {metrics['win_rate']:>5.1f}%  |  "
          f"PnL: ${metrics['total_pnl']:>8.2f}  |  Sharpe: {metrics['sharpe']:>6.3f}  |  "
          f"Sortino: {metrics['sortino']:>6.3f}  |  PF: {metrics['profit_factor']:>5.2f}  |  "
          f"MDD: {metrics['max_drawdown_pct']:>6.2f}%")
    print(f"  Regime gap: {regime['regime_gap']:.3f}  |  "
          f"Perm p: {perm['p_value']:.4f}  |  "
          f"Gates: {validation['gates_passed']}  |  "
          f"{'PASS' if validation['all_pass'] else 'FAIL'}")
    if metrics.get('strategy_breakdown'):
        for s, info in metrics['strategy_breakdown'].items():
            print(f"    {s}: {info['trades']} trades, ${info['pnl']:.2f}")

# ── Ranking ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RANKING (by Sharpe ratio)")
print("=" * 70)

ranked = sorted(results.items(), key=lambda x: x[1]['metrics']['sharpe'], reverse=True)
for rank, (vname, data) in enumerate(ranked, 1):
    m = data['metrics']
    v = data['validation']
    status = "PASS" if v['all_pass'] else "FAIL"
    print(f"  #{rank}: {vname:<25} Sharpe={m['sharpe']:>6.3f}  "
          f"Sortino={m['sortino']:>6.3f}  PF={m['profit_factor']:>5.2f}  "
          f"WR={m['win_rate']:>5.1f}%  PnL=${m['total_pnl']:>8.2f}  "
          f"Trades={m['n_trades']:>3}  [{status} {v['gates_passed']}]")

# ── Summary ──────────────────────────────────────────────────────────────
passed = [v for v, d in results.items() if d['validation']['all_pass']]
best_name = ranked[0][0] if ranked else "none"
best_sharpe = ranked[0][1]['metrics']['sharpe'] if ranked else 0

summary = {
    "backtest": "Multi-Strategy Ensemble Portfolio",
    "period": f"{OOT_START} to {END}",
    "universe": TICKERS,
    "capital": CAPITAL,
    "max_per_trade": MAX_PER_TRADE,
    "max_concurrent": MAX_CONCURRENT,
    "slippage_bps": SLIPPAGE_PCT * 10000,
    "hold_days": HOLD_DAYS,
    "strategies_combined": list(STRATEGY_SHARPES.keys()),
    "strategy_validated_sharpes": STRATEGY_SHARPES,
    "variants_tested": len(VARIANTS),
    "variants_passed_5gate": len(passed),
    "passed_variants": passed,
    "best_variant": best_name,
    "best_sharpe": best_sharpe,
    "results": results,
    "timestamp": datetime.now().isoformat(),
}

print(f"\n{'=' * 70}")
print(f"SUMMARY: {len(passed)}/{len(VARIANTS)} variants passed all 5 gates")
print(f"Best variant: {best_name} (Sharpe={best_sharpe:.3f})")
print(f"{'=' * 70}")

# ── Save Results ─────────────────────────────────────────────────────────
out_path = Path("/home/jupiter/Lvl3Quant/data/multi_strategy_ensemble_results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Convert datetime keys in results for JSON serialization
def make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj

with open(out_path, "w") as f:
    json.dump(make_serializable(summary), f, indent=2, default=str)

print(f"\nResults saved to {out_path}")
