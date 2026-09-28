#!/usr/bin/env python3
"""
Sector Dispersion Timing Backtest — 6 Variants
================================================
Uses cross-sectional dispersion of sector returns as a timing signal.
High dispersion = sectors diverging (regime transition, mean reversion).
Low dispersion = sectors converging (trend continuation).

OOT: Jan 2022 - Jul 2026 | Capital: $669 | Commission: $0 (Robinhood)
5-Gate: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]
TRADE_ETFS = ["SPY", "QQQ"]
VIX_TICKER = "^VIX"
ALL_TICKERS = SECTOR_ETFS + TRADE_ETFS + [VIX_TICKER]

START = "2020-06-01"  # lookback buffer for 252d rolling + 200 SMA warmup
OOT_START = "2022-01-03"
OOT_END = "2026-07-30"
INITIAL_CAPITAL = 669.0
SLIPPAGE_BPS = 0.02 / 100  # 0.02% per trade
COMMISSION = 0.0
N_PERM = 1000
SEED = 42

DISP_WINDOW = 20       # 20-day sector returns for dispersion calc
DISP_PCTILE_WIN = 252  # rolling 252d window for percentile
HOLD_DAYS = 20          # holding period for mean reversion variants

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/sector_dispersion_timing_results.json")

# ── DATA DOWNLOAD ───────────────────────────────────────────────────────────
print("Downloading sector + market data...")
raw = yf.download(ALL_TICKERS, start=START, end=OOT_END, auto_adjust=True, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"].copy()
else:
    close = raw.copy()

if "^VIX" in close.columns:
    close.rename(columns={"^VIX": "VIX"}, inplace=True)

close = close.ffill().dropna(how="all")
print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days, {close.shape[1]} tickers")

# ── COMPUTE DISPERSION ──────────────────────────────────────────────────────
# 20-day returns for each sector ETF
sector_returns = close[SECTOR_ETFS].pct_change(DISP_WINDOW)

# Cross-sectional standard deviation of sector returns each day
dispersion = sector_returns.std(axis=1)
dispersion.name = "dispersion"

# Rolling percentile of dispersion (252-day window)
disp_pctile = dispersion.rolling(DISP_PCTILE_WIN).apply(
    lambda x: (x.values[-1] >= x.values[:-1]).mean() if len(x) > 1 else 0.5,
    raw=False
)
disp_pctile.name = "disp_pctile"

# Dispersion rate of change (20-day)
disp_roc = dispersion.pct_change(20) * 100
disp_roc.name = "disp_roc"

# Rank sectors by 20-day return (for best/worst selection)
sector_ranks = sector_returns.rank(axis=1, ascending=True)  # 1 = worst, 11 = best

# ── BUILD MASTER DataFrame ──────────────────────────────────────────────────
df = pd.DataFrame({
    'spy_close': close['SPY'],
    'qqq_close': close.get('QQQ', close['SPY']),
    'dispersion': dispersion,
    'disp_pctile': disp_pctile,
    'disp_roc': disp_roc,
})

# VIX
if 'VIX' in close.columns:
    df['vix'] = close['VIX']
else:
    df['vix'] = close['SPY'].pct_change().rolling(20).std() * np.sqrt(252) * 100

# SPY indicators
df['spy_sma200'] = df['spy_close'].rolling(200).mean()
df['spy_ret'] = df['spy_close'].pct_change()

# Add sector close prices for trading
for etf in SECTOR_ETFS:
    df[f'{etf}_close'] = close[etf]
    df[f'{etf}_ret'] = close[etf].pct_change()
    df[f'{etf}_ret20'] = close[etf].pct_change(DISP_WINDOW)

# Regime
df['regime'] = (df['spy_close'] > df['spy_sma200']).astype(int)  # 1=bull, 0=bear

# Trim to OOT
df = df.loc[OOT_START:]
df = df.dropna(subset=['dispersion', 'disp_pctile', 'spy_sma200'])
print(f"OOT period: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} trading days")
print(f"Dispersion stats — mean: {df['dispersion'].mean():.4f}, "
      f"median: {df['dispersion'].median():.4f}, "
      f"p20: {df['dispersion'].quantile(0.2):.4f}, "
      f"p80: {df['dispersion'].quantile(0.8):.4f}")


# ── BACKTEST ENGINE ─────────────────────────────────────────────────────────
def run_backtest_alloc(df, alloc_series, ret_series, initial_capital=INITIAL_CAPITAL):
    """
    Simple allocation-based backtest.
    alloc_series: daily target allocation (0 to 1).
    ret_series: daily return of the instrument being traded.
    """
    equity = initial_capital
    position_value = 0.0
    cash = initial_capital
    current_alloc = 0.0
    n_trades = 0
    equity_curve = []
    daily_returns = []
    prev_equity = initial_capital

    for i in range(len(df)):
        # Apply return to position
        if i > 0 and current_alloc > 0:
            r = ret_series.iloc[i]
            if not np.isnan(r):
                position_value *= (1 + r)

        equity = cash + position_value
        target_alloc = alloc_series.iloc[i] if i < len(alloc_series) else 0.0
        if np.isnan(target_alloc):
            target_alloc = 0.0

        # Rebalance
        if abs(target_alloc - current_alloc) > 0.01:
            target_pos = equity * target_alloc
            trade_val = abs(target_pos - position_value)
            if trade_val > 1.0:
                cost = trade_val * SLIPPAGE_BPS
                equity -= cost
                n_trades += 1
            position_value = equity * target_alloc
            cash = equity - position_value
            current_alloc = target_alloc

        equity = cash + position_value
        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0.0
        equity_curve.append(equity)
        daily_returns.append(daily_ret)
        prev_equity = equity

    return np.array(equity_curve), np.array(daily_returns), n_trades


def run_backtest_trades(df, entries, instrument_col, hold_days=HOLD_DAYS,
                        initial_capital=INITIAL_CAPITAL, short=False):
    """
    Trade-list based backtest. entries = boolean Series of entry signals.
    Goes long (or short) the instrument for hold_days.
    For long/short pair: use run_backtest_pair.
    """
    equity = initial_capital
    equity_curve = []
    daily_returns = []
    trades = []
    prev_equity = initial_capital
    in_trade = False
    trade_entry_price = 0.0
    trade_entry_date = None
    trade_bars = 0

    for i in range(len(df)):
        date = df.index[i]
        price = df[instrument_col].iloc[i]

        if in_trade:
            trade_bars += 1
            if short:
                daily_pnl = -(price / df[instrument_col].iloc[i-1] - 1) * equity if i > 0 else 0
            else:
                daily_pnl = (price / df[instrument_col].iloc[i-1] - 1) * equity if i > 0 else 0
            equity += daily_pnl

            if trade_bars >= hold_days:
                # Exit
                exit_price = price
                slippage = equity * SLIPPAGE_BPS
                equity -= slippage
                if short:
                    ret = (trade_entry_price / exit_price - 1)
                else:
                    ret = (exit_price / trade_entry_price - 1)
                trades.append({
                    'entry_date': str(trade_entry_date.date()),
                    'exit_date': str(date.date()),
                    'instrument': instrument_col,
                    'return_pct': round(ret * 100, 2),
                    'side': 'short' if short else 'long',
                })
                in_trade = False

        elif entries.iloc[i] and not in_trade:
            # Enter trade
            in_trade = True
            trade_entry_price = price
            trade_entry_date = date
            trade_bars = 0
            slippage = equity * SLIPPAGE_BPS
            equity -= slippage

        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0
        equity_curve.append(equity)
        daily_returns.append(daily_ret)
        prev_equity = equity

    return np.array(equity_curve), np.array(daily_returns), len(trades), trades


def run_backtest_pair(df, entries, long_col, short_col, hold_days=HOLD_DAYS,
                      initial_capital=INITIAL_CAPITAL):
    """
    Long worst sector, short best sector pair trade.
    Each leg gets 50% of capital. Market neutral.
    """
    equity = initial_capital
    equity_curve = []
    daily_returns = []
    trades = []
    prev_equity = initial_capital
    in_trade = False
    long_entry = short_entry = 0.0
    trade_entry_date = None
    trade_bars = 0
    cur_long_col = cur_short_col = None

    for i in range(len(df)):
        date = df.index[i]

        if in_trade:
            trade_bars += 1
            if i > 0 and cur_long_col and cur_short_col:
                long_ret = df[cur_long_col].iloc[i] / df[cur_long_col].iloc[i-1] - 1
                short_ret = -(df[cur_short_col].iloc[i] / df[cur_short_col].iloc[i-1] - 1)
                daily_pnl = (long_ret + short_ret) / 2 * equity
                equity += daily_pnl

            if trade_bars >= hold_days:
                slippage = equity * SLIPPAGE_BPS * 2  # two legs
                equity -= slippage
                long_exit = df[cur_long_col].iloc[i]
                short_exit = df[cur_short_col].iloc[i]
                long_r = long_exit / long_entry - 1
                short_r = -(short_exit / short_entry - 1)
                net_r = (long_r + short_r) / 2
                trades.append({
                    'entry_date': str(trade_entry_date.date()),
                    'exit_date': str(date.date()),
                    'long': cur_long_col,
                    'short': cur_short_col,
                    'return_pct': round(net_r * 100, 2),
                })
                in_trade = False

        elif i < len(entries) and entries.iloc[i] and not in_trade:
            in_trade = True
            cur_long_col = long_col.iloc[i] if hasattr(long_col, 'iloc') else long_col
            cur_short_col = short_col.iloc[i] if hasattr(short_col, 'iloc') else short_col
            long_entry = df[cur_long_col].iloc[i]
            short_entry = df[cur_short_col].iloc[i]
            trade_entry_date = date
            trade_bars = 0
            slippage = equity * SLIPPAGE_BPS * 2
            equity -= slippage

        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0
        equity_curve.append(equity)
        daily_returns.append(daily_ret)
        prev_equity = equity

    return np.array(equity_curve), np.array(daily_returns), len(trades), trades


# ── METRICS ─────────────────────────────────────────────────────────────────
def compute_metrics(daily_returns, equity_curve, n_trades):
    dr = daily_returns[1:]
    if len(dr) == 0 or np.std(dr) == 0:
        return {
            'sharpe': 0.0, 'sortino': 0.0, 'profit_factor': 0.0,
            'win_rate': 0.0, 'max_dd_pct': 0.0, 'total_return_pct': 0.0,
            'final_equity': float(equity_curve[-1]), 'n_trades': n_trades, 'cagr_pct': 0.0,
        }
    ann = np.sqrt(252)
    sharpe = np.mean(dr) / np.std(dr) * ann
    downside = dr[dr < 0]
    ds_std = np.std(downside) if len(downside) > 0 else 1e-8
    sortino = np.mean(dr) / ds_std * ann
    pos = dr[dr > 0].sum()
    neg = abs(dr[dr < 0].sum())
    pf = pos / neg if neg > 0 else float('inf')
    wr = (dr > 0).sum() / len(dr) * 100
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = dd.min() * 100
    total_ret = (equity_curve[-1] / equity_curve[0] - 1) * 100
    n_years = len(dr) / 252
    cagr = ((equity_curve[-1] / equity_curve[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    return {
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'profit_factor': round(float(pf), 3),
        'win_rate': round(float(wr), 1),
        'max_dd_pct': round(float(max_dd), 2),
        'total_return_pct': round(float(total_ret), 2),
        'final_equity': round(float(equity_curve[-1]), 2),
        'n_trades': int(n_trades),
        'cagr_pct': round(float(cagr), 2),
    }


# ── PERMUTATION TEST ────────────────────────────────────────────────────────
def permutation_test_alloc(df, alloc_series, ret_series, observed_sharpe, n_perms=N_PERM):
    """Shuffle allocation dates to test timing edge."""
    rng = np.random.RandomState(SEED)
    alloc_vals = alloc_series.values.copy()
    count_better = 0
    for _ in range(n_perms):
        shuffled = alloc_vals.copy()
        rng.shuffle(shuffled)
        shuf_series = pd.Series(shuffled, index=alloc_series.index)
        _, dr, _ = run_backtest_alloc(df, shuf_series, ret_series)
        dr_clean = dr[1:]
        if len(dr_clean) > 0 and np.std(dr_clean) > 0:
            ps = np.mean(dr_clean) / np.std(dr_clean) * np.sqrt(252)
        else:
            ps = 0.0
        if ps >= observed_sharpe:
            count_better += 1
    return (count_better + 1) / (n_perms + 1)


def permutation_test_entries(df, entries, instrument_col, observed_sharpe,
                             hold_days=HOLD_DAYS, n_perms=N_PERM, short=False):
    """Shuffle entry signal dates."""
    rng = np.random.RandomState(SEED)
    entry_vals = entries.values.copy()
    count_better = 0
    for _ in range(n_perms):
        shuffled = entry_vals.copy()
        rng.shuffle(shuffled)
        shuf_entries = pd.Series(shuffled, index=entries.index)
        _, dr, _, _ = run_backtest_trades(df, shuf_entries, instrument_col,
                                          hold_days=hold_days, short=short)
        dr_clean = dr[1:]
        if len(dr_clean) > 0 and np.std(dr_clean) > 0:
            ps = np.mean(dr_clean) / np.std(dr_clean) * np.sqrt(252)
        else:
            ps = 0.0
        if ps >= observed_sharpe:
            count_better += 1
    return (count_better + 1) / (n_perms + 1)


def permutation_test_pair(df, entries, long_col, short_col, observed_sharpe,
                          hold_days=HOLD_DAYS, n_perms=N_PERM):
    """Shuffle pair entry dates."""
    rng = np.random.RandomState(SEED)
    entry_vals = entries.values.copy()
    count_better = 0
    for _ in range(n_perms):
        shuffled = entry_vals.copy()
        rng.shuffle(shuffled)
        shuf_entries = pd.Series(shuffled, index=entries.index)
        _, dr, _, _ = run_backtest_pair(df, shuf_entries, long_col, short_col, hold_days=hold_days)
        dr_clean = dr[1:]
        if len(dr_clean) > 0 and np.std(dr_clean) > 0:
            ps = np.mean(dr_clean) / np.std(dr_clean) * np.sqrt(252)
        else:
            ps = 0.0
        if ps >= observed_sharpe:
            count_better += 1
    return (count_better + 1) / (n_perms + 1)


# ── REGIME STRATIFICATION ──────────────────────────────────────────────────
def regime_stratification(df, daily_returns):
    bull = df['regime'] == 1
    bear = df['regime'] == 0
    dr = daily_returns

    def safe_sharpe(r):
        if len(r) < 10 or np.std(r) == 0:
            return 0.0
        return float(np.mean(r) / np.std(r) * np.sqrt(252))

    bull_dr = dr[bull.values[:len(dr)]] if len(bull) >= len(dr) else dr
    bear_dr = dr[bear.values[:len(dr)]] if len(bear) >= len(dr) else dr
    bull_sharpe = safe_sharpe(bull_dr)
    bear_sharpe = safe_sharpe(bear_dr)
    gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-8)
    return {
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(float(gap), 3),
        'bull_days': int(bull.sum()),
        'bear_days': int(bear.sum()),
    }


# ── 5-GATE VALIDATION ──────────────────────────────────────────────────────
def validate_5gates(metrics, perm_p, regime_gap):
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime_gap < 0.5,
        'max_dd_gt_neg50': metrics['max_dd_pct'] > -50,
        'trades_gte_20': metrics['n_trades'] >= 20,
    }
    gates['all_passed'] = all(gates.values())
    return gates


# ── SIGNAL GENERATORS (6 VARIANTS) ─────────────────────────────────────────

def find_worst_best_sector(df, i):
    """Find worst and best performing sector ETF on day i based on 20d return."""
    rets = {}
    for etf in SECTOR_ETFS:
        col = f'{etf}_ret20'
        if col in df.columns and not np.isnan(df[col].iloc[i]):
            rets[etf] = df[col].iloc[i]
    if not rets:
        return None, None
    worst = min(rets, key=rets.get)
    best = max(rets, key=rets.get)
    return worst, best


# ── Variant A: High Dispersion Mean Reversion ──────────────────────────────
def run_variant_a(df):
    """When dispersion > 80th pctile, buy worst sector ETF. Hold 20 days."""
    print("\n=== Variant A: High Dispersion Mean Reversion ===")

    entries = pd.Series(False, index=df.index)
    worst_sectors = pd.Series("", index=df.index)

    cooldown = 0
    for i in range(len(df)):
        if cooldown > 0:
            cooldown -= 1
            continue
        if df['disp_pctile'].iloc[i] > 0.80:
            worst, _ = find_worst_best_sector(df, i)
            if worst:
                entries.iloc[i] = True
                worst_sectors.iloc[i] = worst
                cooldown = HOLD_DAYS

    # Run backtest for each trade individually
    equity = INITIAL_CAPITAL
    equity_curve = []
    daily_returns = []
    trades_list = []
    prev_equity = INITIAL_CAPITAL
    in_trade = False
    trade_bars = 0
    trade_col = None
    trade_entry_price = 0

    for i in range(len(df)):
        if in_trade and trade_col:
            trade_bars += 1
            if i > 0:
                r = df[f'{trade_col}_close'].iloc[i] / df[f'{trade_col}_close'].iloc[i-1] - 1
                equity *= (1 + r)
            if trade_bars >= HOLD_DAYS:
                exit_price = df[f'{trade_col}_close'].iloc[i]
                slippage = equity * SLIPPAGE_BPS
                equity -= slippage
                ret = exit_price / trade_entry_price - 1
                trades_list.append({
                    'entry_date': str(trade_entry_date.date()),
                    'exit_date': str(df.index[i].date()),
                    'instrument': trade_col,
                    'return_pct': round(ret * 100, 2),
                })
                in_trade = False

        elif entries.iloc[i] and not in_trade:
            trade_col = worst_sectors.iloc[i]
            if trade_col and f'{trade_col}_close' in df.columns:
                in_trade = True
                trade_entry_price = df[f'{trade_col}_close'].iloc[i]
                trade_entry_date = df.index[i]
                trade_bars = 0
                slippage = equity * SLIPPAGE_BPS
                equity -= slippage

        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0
        equity_curve.append(equity)
        daily_returns.append(daily_ret)
        prev_equity = equity

    eq = np.array(equity_curve)
    dr = np.array(daily_returns)
    return eq, dr, len(trades_list), trades_list, entries


# ── Variant B: Low Dispersion Momentum ─────────────────────────────────────
def run_variant_b(df):
    """When dispersion < 20th pctile, buy SPY. Go to cash when dispersion > 50th pctile."""
    print("\n=== Variant B: Low Dispersion Momentum ===")
    alloc = pd.Series(0.0, index=df.index)

    in_position = False
    for i in range(len(df)):
        pctile = df['disp_pctile'].iloc[i]
        if np.isnan(pctile):
            continue
        if not in_position and pctile < 0.20:
            in_position = True
        elif in_position and pctile > 0.50:
            in_position = False
        alloc.iloc[i] = 1.0 if in_position else 0.0

    ret_series = df['spy_ret'].fillna(0)
    eq, dr, n_trades = run_backtest_alloc(df, alloc, ret_series)
    return eq, dr, n_trades, alloc


# ── Variant C: Dispersion Regime ───────────────────────────────────────────
def run_variant_c(df):
    """
    Dispersion rising + VIX rising → defensive (XLU).
    Dispersion falling → risk-on (QQQ).
    """
    print("\n=== Variant C: Dispersion Regime ===")

    disp_rising = df['dispersion'] > df['dispersion'].shift(5)
    vix_rising = df['vix'] > df['vix'].shift(5)

    # Allocation to QQQ vs XLU
    alloc_qqq = pd.Series(0.0, index=df.index)
    alloc_xlu = pd.Series(0.0, index=df.index)

    for i in range(len(df)):
        if np.isnan(df['dispersion'].iloc[i]):
            continue
        if disp_rising.iloc[i] and vix_rising.iloc[i]:
            alloc_xlu.iloc[i] = 1.0
        elif not disp_rising.iloc[i]:
            alloc_qqq.iloc[i] = 1.0
        # else: cash

    # Combine returns
    qqq_ret = close['QQQ'].pct_change().reindex(df.index).fillna(0)
    xlu_ret = close['XLU'].pct_change().reindex(df.index).fillna(0)
    combined_ret = alloc_qqq * qqq_ret + alloc_xlu * xlu_ret
    combined_alloc = alloc_qqq + alloc_xlu

    # Simple equity sim
    equity = INITIAL_CAPITAL
    equity_curve = []
    daily_returns = []
    prev_equity = INITIAL_CAPITAL
    prev_alloc = 0.0
    n_trades = 0

    for i in range(len(df)):
        if i > 0:
            equity *= (1 + combined_ret.iloc[i])
        cur_alloc = combined_alloc.iloc[i]
        if abs(cur_alloc - prev_alloc) > 0.01:
            equity -= abs(equity * SLIPPAGE_BPS)
            n_trades += 1
            prev_alloc = cur_alloc
        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0
        equity_curve.append(equity)
        daily_returns.append(daily_ret)
        prev_equity = equity

    return np.array(equity_curve), np.array(daily_returns), n_trades, combined_alloc


# ── Variant D: Best-Worst Spread (Pair Trade) ──────────────────────────────
def run_variant_d(df):
    """Long worst sector, short best sector when dispersion > 80th pctile."""
    print("\n=== Variant D: Best-Worst Spread (Pair Trade) ===")

    entries = pd.Series(False, index=df.index)
    worst_col = pd.Series("", index=df.index)
    best_col = pd.Series("", index=df.index)

    cooldown = 0
    for i in range(len(df)):
        if cooldown > 0:
            cooldown -= 1
            continue
        if df['disp_pctile'].iloc[i] > 0.80:
            worst, best = find_worst_best_sector(df, i)
            if worst and best and worst != best:
                entries.iloc[i] = True
                worst_col.iloc[i] = f'{worst}_close'
                best_col.iloc[i] = f'{best}_close'
                cooldown = HOLD_DAYS

    eq, dr, n_trades, trades = run_backtest_pair(df, entries, worst_col, best_col)
    return eq, dr, n_trades, trades, entries


# ── Variant E: Dispersion Rate of Change ───────────────────────────────────
def run_variant_e(df):
    """
    Dispersion ROC < -30% → buy QQQ (converging = risk-on).
    Dispersion ROC > 30% → cash.
    """
    print("\n=== Variant E: Dispersion Rate of Change ===")

    alloc = pd.Series(0.0, index=df.index)
    in_position = False

    for i in range(len(df)):
        roc = df['disp_roc'].iloc[i]
        if np.isnan(roc):
            continue
        if not in_position and roc < -30:
            in_position = True
        elif in_position and roc > 30:
            in_position = False
        alloc.iloc[i] = 1.0 if in_position else 0.0

    qqq_ret = close['QQQ'].pct_change().reindex(df.index).fillna(0)
    eq, dr, n_trades = run_backtest_alloc(df, alloc, qqq_ret)
    return eq, dr, n_trades, alloc


# ── Variant F: Combined Signal ─────────────────────────────────────────────
def run_variant_f(df):
    """
    3 signals vote: dispersion pctile < 50, VIX < 20, SPY > 200-SMA.
    >= 2/3 agree → buy QQQ. Else cash.
    """
    print("\n=== Variant F: Combined (Dispersion + VIX + Trend) ===")

    alloc = pd.Series(0.0, index=df.index)

    for i in range(len(df)):
        score = 0
        # Signal 1: Low dispersion (below median) = risk-on
        if not np.isnan(df['disp_pctile'].iloc[i]) and df['disp_pctile'].iloc[i] < 0.50:
            score += 1
        # Signal 2: VIX < 20 = calm
        if not np.isnan(df['vix'].iloc[i]) and df['vix'].iloc[i] < 20:
            score += 1
        # Signal 3: SPY trend = bull
        if df['regime'].iloc[i] == 1:
            score += 1

        alloc.iloc[i] = 1.0 if score >= 2 else 0.0

    qqq_ret = close['QQQ'].pct_change().reindex(df.index).fillna(0)
    eq, dr, n_trades = run_backtest_alloc(df, alloc, qqq_ret)
    return eq, dr, n_trades, alloc


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    # Benchmark: buy & hold SPY
    print("\n=== BENCHMARK: Buy & Hold SPY ===")
    bh_alloc = pd.Series(1.0, index=df.index)
    spy_ret = df['spy_ret'].fillna(0)
    bh_eq, bh_dr, bh_trades = run_backtest_alloc(df, bh_alloc, spy_ret)
    bh_metrics = compute_metrics(bh_dr, bh_eq, bh_trades)
    print(f"  Final equity: ${bh_metrics['final_equity']:.2f} | "
          f"Sharpe: {bh_metrics['sharpe']:.3f} | MaxDD: {bh_metrics['max_dd_pct']:.1f}%")

    results = {
        'metadata': {
            'strategy': 'Sector Dispersion Timing',
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'oot_period': f"{OOT_START} to {OOT_END}",
            'initial_capital': INITIAL_CAPITAL,
            'slippage_bps': round(SLIPPAGE_BPS * 10000, 2),
            'n_permutations': N_PERM,
            'trading_days': len(df),
            'dispersion_window': DISP_WINDOW,
            'hold_days': HOLD_DAYS,
            'sector_etfs': SECTOR_ETFS,
        },
        'benchmark': bh_metrics,
        'variants': {},
    }

    # ── Variant A ──
    eq_a, dr_a, nt_a, trades_a, entries_a = run_variant_a(df)
    met_a = compute_metrics(dr_a, eq_a, nt_a)
    print(f"  A: Final=${met_a['final_equity']:.2f} Sharpe={met_a['sharpe']:.3f} "
          f"MaxDD={met_a['max_dd_pct']:.1f}% Trades={met_a['n_trades']}")
    print(f"  Running permutation test...", end='', flush=True)
    perm_a = permutation_test_entries(df, entries_a, 'spy_close', met_a['sharpe'], short=False)
    print(f" p={perm_a:.4f}")
    reg_a = regime_stratification(df, dr_a)
    gates_a = validate_5gates(met_a, perm_a, reg_a['regime_gap'])
    print(f"  Bull Sharpe={reg_a['bull_sharpe']:.3f} Bear={reg_a['bear_sharpe']:.3f} Gap={reg_a['regime_gap']:.3f}")
    print(f"  Gates: {'ALL PASS' if gates_a['all_passed'] else 'FAILED'}")
    results['variants']['A_HighDisp_MeanReversion'] = {
        'description': 'Buy worst sector when dispersion > 80th pctile, hold 20d',
        'metrics': met_a, 'permutation_p_value': round(perm_a, 4),
        'regime': reg_a, 'gates': gates_a,
        'sample_trades': trades_a[:5] if trades_a else [],
    }

    # ── Variant B ──
    eq_b, dr_b, nt_b, alloc_b = run_variant_b(df)
    met_b = compute_metrics(dr_b, eq_b, nt_b)
    print(f"  B: Final=${met_b['final_equity']:.2f} Sharpe={met_b['sharpe']:.3f} "
          f"MaxDD={met_b['max_dd_pct']:.1f}% Trades={met_b['n_trades']}")
    print(f"  Running permutation test...", end='', flush=True)
    perm_b = permutation_test_alloc(df, alloc_b, spy_ret, met_b['sharpe'])
    print(f" p={perm_b:.4f}")
    reg_b = regime_stratification(df, dr_b)
    gates_b = validate_5gates(met_b, perm_b, reg_b['regime_gap'])
    print(f"  Bull Sharpe={reg_b['bull_sharpe']:.3f} Bear={reg_b['bear_sharpe']:.3f} Gap={reg_b['regime_gap']:.3f}")
    print(f"  Gates: {'ALL PASS' if gates_b['all_passed'] else 'FAILED'}")
    results['variants']['B_LowDisp_Momentum'] = {
        'description': 'Buy SPY when dispersion < 20th pctile, cash when > 50th',
        'metrics': met_b, 'permutation_p_value': round(perm_b, 4),
        'regime': reg_b, 'gates': gates_b,
        'allocation_stats': {
            'pct_invested': round(float((alloc_b > 0).mean() * 100), 1),
            'pct_cash': round(float((alloc_b == 0).mean() * 100), 1),
        },
    }

    # ── Variant C ──
    eq_c, dr_c, nt_c, alloc_c = run_variant_c(df)
    met_c = compute_metrics(dr_c, eq_c, nt_c)
    print(f"  C: Final=${met_c['final_equity']:.2f} Sharpe={met_c['sharpe']:.3f} "
          f"MaxDD={met_c['max_dd_pct']:.1f}% Trades={met_c['n_trades']}")
    print(f"  Running permutation test...", end='', flush=True)
    qqq_ret_series = close['QQQ'].pct_change().reindex(df.index).fillna(0)
    perm_c = permutation_test_alloc(df, alloc_c, qqq_ret_series, met_c['sharpe'])
    print(f" p={perm_c:.4f}")
    reg_c = regime_stratification(df, dr_c)
    gates_c = validate_5gates(met_c, perm_c, reg_c['regime_gap'])
    print(f"  Bull Sharpe={reg_c['bull_sharpe']:.3f} Bear={reg_c['bear_sharpe']:.3f} Gap={reg_c['regime_gap']:.3f}")
    print(f"  Gates: {'ALL PASS' if gates_c['all_passed'] else 'FAILED'}")
    results['variants']['C_Dispersion_Regime'] = {
        'description': 'Disp rising + VIX rising → XLU defensive; Disp falling → QQQ risk-on',
        'metrics': met_c, 'permutation_p_value': round(perm_c, 4),
        'regime': reg_c, 'gates': gates_c,
    }

    # ── Variant D ──
    eq_d, dr_d, nt_d, trades_d, entries_d = run_variant_d(df)
    met_d = compute_metrics(dr_d, eq_d, nt_d)
    print(f"  D: Final=${met_d['final_equity']:.2f} Sharpe={met_d['sharpe']:.3f} "
          f"MaxDD={met_d['max_dd_pct']:.1f}% Trades={met_d['n_trades']}")
    print(f"  Running permutation test...", end='', flush=True)
    perm_d = permutation_test_pair(df, entries_d, pd.Series("XLU_close", index=df.index),
                                   pd.Series("XLK_close", index=df.index), met_d['sharpe'])
    print(f" p={perm_d:.4f}")
    reg_d = regime_stratification(df, dr_d)
    gates_d = validate_5gates(met_d, perm_d, reg_d['regime_gap'])
    print(f"  Bull Sharpe={reg_d['bull_sharpe']:.3f} Bear={reg_d['bear_sharpe']:.3f} Gap={reg_d['regime_gap']:.3f}")
    print(f"  Gates: {'ALL PASS' if gates_d['all_passed'] else 'FAILED'}")
    results['variants']['D_BestWorst_Spread'] = {
        'description': 'Long worst sector, short best sector when dispersion > 80th pctile',
        'metrics': met_d, 'permutation_p_value': round(perm_d, 4),
        'regime': reg_d, 'gates': gates_d,
        'sample_trades': trades_d[:5] if trades_d else [],
    }

    # ── Variant E ──
    eq_e, dr_e, nt_e, alloc_e = run_variant_e(df)
    met_e = compute_metrics(dr_e, eq_e, nt_e)
    print(f"  E: Final=${met_e['final_equity']:.2f} Sharpe={met_e['sharpe']:.3f} "
          f"MaxDD={met_e['max_dd_pct']:.1f}% Trades={met_e['n_trades']}")
    print(f"  Running permutation test...", end='', flush=True)
    perm_e = permutation_test_alloc(df, alloc_e, qqq_ret_series, met_e['sharpe'])
    print(f" p={perm_e:.4f}")
    reg_e = regime_stratification(df, dr_e)
    gates_e = validate_5gates(met_e, perm_e, reg_e['regime_gap'])
    print(f"  Bull Sharpe={reg_e['bull_sharpe']:.3f} Bear={reg_e['bear_sharpe']:.3f} Gap={reg_e['regime_gap']:.3f}")
    print(f"  Gates: {'ALL PASS' if gates_e['all_passed'] else 'FAILED'}")
    results['variants']['E_Dispersion_ROC'] = {
        'description': 'Buy QQQ when dispersion drops >30% in 20d, cash when rises >30%',
        'metrics': met_e, 'permutation_p_value': round(perm_e, 4),
        'regime': reg_e, 'gates': gates_e,
        'allocation_stats': {
            'pct_invested': round(float((alloc_e > 0).mean() * 100), 1),
            'pct_cash': round(float((alloc_e == 0).mean() * 100), 1),
        },
    }

    # ── Variant F ──
    eq_f, dr_f, nt_f, alloc_f = run_variant_f(df)
    met_f = compute_metrics(dr_f, eq_f, nt_f)
    print(f"  F: Final=${met_f['final_equity']:.2f} Sharpe={met_f['sharpe']:.3f} "
          f"MaxDD={met_f['max_dd_pct']:.1f}% Trades={met_f['n_trades']}")
    print(f"  Running permutation test...", end='', flush=True)
    perm_f = permutation_test_alloc(df, alloc_f, qqq_ret_series, met_f['sharpe'])
    print(f" p={perm_f:.4f}")
    reg_f = regime_stratification(df, dr_f)
    gates_f = validate_5gates(met_f, perm_f, reg_f['regime_gap'])
    print(f"  Bull Sharpe={reg_f['bull_sharpe']:.3f} Bear={reg_f['bear_sharpe']:.3f} Gap={reg_f['regime_gap']:.3f}")
    print(f"  Gates: {'ALL PASS' if gates_f['all_passed'] else 'FAILED'}")
    results['variants']['F_Combined_Signal'] = {
        'description': 'Dispersion pctile + VIX + SPY trend, 2/3 vote → QQQ',
        'metrics': met_f, 'permutation_p_value': round(perm_f, 4),
        'regime': reg_f, 'gates': gates_f,
        'allocation_stats': {
            'pct_invested': round(float((alloc_f > 0).mean() * 100), 1),
            'pct_cash': round(float((alloc_f == 0).mean() * 100), 1),
        },
    }

    # ── Summary ──
    print("\n" + "=" * 110)
    print("SECTOR DISPERSION TIMING — SUMMARY")
    print("=" * 110)
    header = f"{'Variant':<30} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR%':>6} {'MaxDD%':>8} {'Trades':>7} {'Final$':>9} {'Perm-p':>8} {'Gates':>7}"
    print(header)
    print("-" * 110)

    any_passed = False
    for name, data in results['variants'].items():
        m = data['metrics']
        p = data['permutation_p_value']
        status = "PASS" if data['gates']['all_passed'] else "FAIL"
        if data['gates']['all_passed']:
            any_passed = True
        print(f"{name:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.2f} "
              f"{m['win_rate']:>5.1f}% {m['max_dd_pct']:>7.1f}% {m['n_trades']:>7d} "
              f"${m['final_equity']:>8.2f} {p:>8.4f} {status:>7}")

    print("-" * 110)
    print(f"{'Benchmark (SPY B&H)':<30} {bh_metrics['sharpe']:>7.3f} {bh_metrics['sortino']:>8.3f} "
          f"{bh_metrics['profit_factor']:>6.2f} {bh_metrics['win_rate']:>5.1f}% "
          f"{bh_metrics['max_dd_pct']:>7.1f}% {'—':>7} ${bh_metrics['final_equity']:>8.2f} {'—':>8} {'—':>7}")

    n_passed = sum(1 for v in results['variants'].values() if v['gates']['all_passed'])
    print(f"\n{n_passed}/{len(results['variants'])} variants passed all 5 gates.")

    if not any_passed:
        print("\nVERDICT: Sector dispersion timing does NOT produce tradeable edge in OOT period.")
    else:
        passed = [n for n, v in results['variants'].items() if v['gates']['all_passed']]
        print(f"\nVERDICT: Passed variants: {', '.join(passed)}")

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
