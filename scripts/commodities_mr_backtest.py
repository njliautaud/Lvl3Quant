#!/usr/bin/env python3
"""
Commodities Mean Reversion Backtest (Dual Signal D Adapted)
============================================================
Commodities are fundamentally mean-reverting due to supply/demand cycles.
Tests whether Dual Signal D logic transfers to commodity ETFs.

6 Variants:
  A: Exact Dual Signal D (5% dip from 20d high, RSI<35, 3+ red days, first green, 10d hold)
  B: Adjusted for commodities (7% dip, RSI<30, 3+ red, first green, 10d hold)
  C: Adjusted + longer hold (7% dip, RSI<30, 3+ red, first green, 20d hold)
  D: Precious metals only (GLD/SLV/PPLT/PALL), 5% dip, RSI<35, 3+ red, first green, 10d hold
  E: Energy + Agri only (USO/UNG/DBA/WEAT/CORN), 7% dip, RSI<30, 3+ red, first green, 10d hold
  F: All commodities, 5% dip, RSI<35, 3+ red, first green, VIX>20 filter (flight to real assets)

OOT: Jan 2022 - Jul 2026. $645 capital. 0.02% slippage each way.
5-Gate Validation: Sharpe>0.5, permutation p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
"""

import json
import warnings
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

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
DATA_START = "2020-01-01"  # lookback for indicators
OOT_START = "2022-01-01"
END = "2026-07-31"
N_PERM = 1000

ALL_COMMODITIES = [
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PPLT", "PALL",
    "WEAT", "CORN", "CPER", "URA",
]

PRECIOUS_METALS = ["GLD", "SLV", "PPLT", "PALL"]

ENERGY_AGRI = ["USO", "UNG", "DBA", "WEAT", "CORN"]

RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/commodities_mr_results.json"

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading commodity ETF price data ...")
ALL_DOWNLOAD = sorted(set(ALL_COMMODITIES + ["^VIX", "SPY"]))
raw = yf.download(ALL_DOWNLOAD, start=DATA_START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in ALL_DOWNLOAD}
vix_close = closes.get("^VIX", pd.Series(dtype=float))

loaded = sum(1 for t in ALL_COMMODITIES if len(closes.get(t, [])) > 100)
print(f"  Commodity ETFs with data: {loaded}/{len(ALL_COMMODITIES)}")
if len(vix_close) > 100:
    print(f"  VIX data: {len(vix_close)} days")


# ── Indicator Helpers ─────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def count_consecutive_red_days(close_series, date):
    """Count consecutive red days ending on the day BEFORE date."""
    if date not in close_series.index:
        return 0
    idx = close_series.index.get_loc(date)
    if isinstance(idx, slice):
        idx = idx.start
    count = 0
    for i in range(idx - 1, max(idx - 20, 0), -1):
        if close_series.iloc[i] < close_series.iloc[i - 1]:
            count += 1
        else:
            break
    return count


def is_green_day(close_series, date):
    """Check if today is a green day (close > previous close)."""
    if date not in close_series.index:
        return False
    idx = close_series.index.get_loc(date)
    if isinstance(idx, slice):
        idx = idx.start
    if idx < 1:
        return False
    return close_series.iloc[idx] > close_series.iloc[idx - 1]


# ── Pre-compute Indicators ───────────────────────────────────────────────
print("Computing indicators ...")
indicators = {}
for t in ALL_COMMODITIES:
    s = closes.get(t, pd.Series(dtype=float))
    if len(s) < 50:
        print(f"  WARNING: {t} has insufficient data ({len(s)} bars), skipping")
        continue
    indicators[t] = {
        'close': s,
        'rsi14': calc_rsi(s, 14),
        'high20': s.rolling(20).max(),
    }

# SPY 200-SMA for regime classification
spy_close = closes.get("SPY", pd.Series(dtype=float))
spy_200sma = spy_close.rolling(200).mean() if len(spy_close) > 200 else pd.Series(dtype=float)


def get_regime(date):
    if date in spy_close.index and date in spy_200sma.index:
        sc = spy_close.loc[date]
        sm = spy_200sma.loc[date]
        if pd.notna(sc) and pd.notna(sm):
            return 'bull' if sc > sm else 'bear'
    return 'unknown'


# ── Signal Generators ────────────────────────────────────────────────────

def generate_signals_A(date, universe):
    """Exact Dual Signal D: 5% dip from 20d high, RSI<35, 3+ red days then first green."""
    signals = []
    for t in universe:
        if t not in indicators:
            continue
        ind = indicators[t]
        s = ind['close']
        if date not in s.index:
            continue
        price = s.loc[date]
        high20 = ind['high20'].loc[date] if date in ind['high20'].index else np.nan
        rsi = ind['rsi14'].loc[date] if date in ind['rsi14'].index else np.nan
        if pd.isna(price) or pd.isna(high20) or pd.isna(rsi):
            continue

        drawdown = (price - high20) / high20
        cond1 = drawdown < -0.05 and rsi < 35
        red_days = count_consecutive_red_days(s, date)
        green = is_green_day(s, date)
        cond2 = red_days >= 3 and green

        if cond1 and cond2:
            signals.append((t, float(price)))
    return signals


def generate_signals_B(date, universe):
    """Commodity-adjusted: 7% dip from 20d high, RSI<30, 3+ red days then first green."""
    signals = []
    for t in universe:
        if t not in indicators:
            continue
        ind = indicators[t]
        s = ind['close']
        if date not in s.index:
            continue
        price = s.loc[date]
        high20 = ind['high20'].loc[date] if date in ind['high20'].index else np.nan
        rsi = ind['rsi14'].loc[date] if date in ind['rsi14'].index else np.nan
        if pd.isna(price) or pd.isna(high20) or pd.isna(rsi):
            continue

        drawdown = (price - high20) / high20
        cond1 = drawdown < -0.07 and rsi < 30
        red_days = count_consecutive_red_days(s, date)
        green = is_green_day(s, date)
        cond2 = red_days >= 3 and green

        if cond1 and cond2:
            signals.append((t, float(price)))
    return signals


def generate_signals_F(date, universe):
    """Same as A but ONLY when VIX > 20 (flight to real assets during equity stress)."""
    if date in vix_close.index:
        vix = vix_close.loc[date]
        if pd.notna(vix) and vix <= 20:
            return []
    else:
        return []  # No VIX data = no signal
    return generate_signals_A(date, universe)


# ── Backtest Engine ──────────────────────────────────────────────────────

def build_close_df(universe):
    """Build a DataFrame of closes for the given universe."""
    frames = {}
    for t in universe:
        if t in closes:
            frames[t] = closes[t]
    return pd.DataFrame(frames)


def run_backtest(universe, signal_fn, hold_days):
    """Run backtest with fixed hold_days exit."""
    close_df = build_close_df(universe)
    oot_idx = close_df.loc[OOT_START:END].index
    if len(oot_idx) == 0:
        return [], pd.Series(dtype=float)

    capital = CAPITAL
    positions = []
    trades = []
    equity = {}

    for date in oot_idx:
        # ── Check exits ──
        still_open = []
        for pos in positions:
            should_exit = date >= pos['exit_target']

            if should_exit:
                exit_price_raw = close_df.loc[date, pos['ticker']] if pos['ticker'] in close_df.columns else np.nan
                if pd.isna(exit_price_raw):
                    still_open.append(pos)
                    continue
                exit_price = float(exit_price_raw) * (1 - SLIPPAGE_PCT)
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                capital += exit_price * pos['shares']
                ret = (exit_price / pos['entry_price']) - 1
                trades.append({
                    'ticker': pos['ticker'],
                    'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': date.strftime('%Y-%m-%d'),
                    'entry_price': float(pos['entry_price']),
                    'exit_price': float(exit_price),
                    'shares': int(pos['shares']),
                    'pnl': float(pnl),
                    'return': float(ret),
                    'regime': pos['regime'],
                })
            else:
                still_open.append(pos)
        positions = still_open

        # ── Generate signals & enter ──
        if len(positions) < MAX_CONCURRENT:
            signals = signal_fn(date, universe)
            held_tickers = {p['ticker'] for p in positions}
            signals = [(t, p) for t, p in signals if t not in held_tickers]

            slots = MAX_CONCURRENT - len(positions)
            for t, price in signals[:slots]:
                entry_price = price * (1 + SLIPPAGE_PCT)
                alloc = min(MAX_PER_TRADE, capital * 0.95)
                if alloc < 10 or capital < 10:
                    continue
                shares = int(alloc / entry_price)
                if shares < 1:
                    continue
                cost = shares * entry_price
                capital -= cost
                regime = get_regime(date)
                exit_loc = min(oot_idx.get_loc(date) + hold_days, len(oot_idx) - 1)
                exit_target = oot_idx[exit_loc]
                positions.append({
                    'ticker': t,
                    'entry_price': entry_price,
                    'entry_date': date,
                    'shares': shares,
                    'exit_target': exit_target,
                    'regime': regime,
                })

        # ── Mark-to-market ──
        mtm = capital
        for pos in positions:
            curr = close_df.loc[date, pos['ticker']] if pos['ticker'] in close_df.columns else np.nan
            if pd.isna(curr):
                curr = pos['entry_price']
            mtm += float(curr) * pos['shares']
        equity[date] = mtm

    # ── Close remaining positions ──
    last_date = oot_idx[-1]
    for pos in positions:
        exit_price_raw = close_df.loc[last_date, pos['ticker']] if pos['ticker'] in close_df.columns else np.nan
        if pd.isna(exit_price_raw):
            exit_price_raw = pos['entry_price']
        exit_price = float(exit_price_raw) * (1 - SLIPPAGE_PCT)
        pnl = (exit_price - pos['entry_price']) * pos['shares']
        capital += exit_price * pos['shares']
        ret = (exit_price / pos['entry_price']) - 1
        trades.append({
            'ticker': pos['ticker'],
            'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
            'exit_date': last_date.strftime('%Y-%m-%d'),
            'entry_price': float(pos['entry_price']),
            'exit_price': float(exit_price),
            'shares': int(pos['shares']),
            'pnl': float(pnl),
            'return': float(ret),
            'regime': pos['regime'],
        })

    equity_series = pd.Series(equity).sort_index()
    return trades, equity_series


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(trades, equity_series):
    if len(trades) == 0:
        return {
            'n_trades': 0, 'total_pnl': 0, 'total_return_pct': 0,
            'sharpe': 0, 'sortino': 0, 'profit_factor': 0, 'win_rate': 0,
            'max_drawdown_pct': 0, 'avg_trade_return_pct': 0,
        }

    total_pnl = sum(t['pnl'] for t in trades)
    total_return = (equity_series.iloc[-1] / CAPITAL - 1) if len(equity_series) > 0 else 0

    if len(equity_series) > 1:
        daily_rets = equity_series.pct_change().dropna()
        ann = np.sqrt(252)
        sharpe = (daily_rets.mean() / daily_rets.std() * ann) if daily_rets.std() > 0 else 0
        downside = daily_rets[daily_rets < 0].std()
        sortino = (daily_rets.mean() / downside * ann) if downside > 0 else 0
    else:
        sharpe = sortino = 0

    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    winners = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = winners / len(trades)

    if len(equity_series) > 0:
        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak
        max_dd = dd.min()
    else:
        max_dd = 0

    avg_trade_ret = np.mean([t['return'] for t in trades]) * 100

    return {
        'n_trades': len(trades),
        'total_pnl': round(float(total_pnl), 2),
        'total_return_pct': round(float(total_return * 100), 2),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'profit_factor': round(float(profit_factor), 3),
        'win_rate': round(float(win_rate), 3),
        'max_drawdown_pct': round(float(max_dd * 100), 2),
        'avg_trade_return_pct': round(float(avg_trade_ret), 3),
    }


def compute_regime_sharpe(trades):
    bull_rets = [t['return'] for t in trades if t['regime'] == 'bull']
    bear_rets = [t['return'] for t in trades if t['regime'] == 'bear']

    def _sharpe(rets):
        if len(rets) < 2:
            return 0.0
        arr = np.array(rets)
        if arr.std() == 0:
            return 0.0
        return float(arr.mean() / arr.std() * np.sqrt(25))

    s_bull = _sharpe(bull_rets)
    s_bear = _sharpe(bear_rets)
    denom = max(abs(s_bull), abs(s_bear))
    regime_gap = abs(s_bull - s_bear) / denom if denom > 0 else 0.0

    return {
        'sharpe_bull': round(s_bull, 3),
        'sharpe_bear': round(s_bear, 3),
        'n_bull': len(bull_rets),
        'n_bear': len(bear_rets),
        'regime_gap': round(regime_gap, 3),
    }


def permutation_test(trades, equity_series, n_perms=N_PERM):
    if len(trades) < 5 or len(equity_series) < 10:
        return 1.0

    observed_sharpe = 0
    if len(equity_series) > 1:
        daily_rets = equity_series.pct_change().dropna()
        if daily_rets.std() > 0:
            observed_sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)

    trade_returns = np.array([t['return'] for t in trades])
    n_better = 0
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(trade_returns))
        scrambled = trade_returns * signs
        if scrambled.std() > 0:
            perm_sharpe = scrambled.mean() / scrambled.std() * np.sqrt(25)
        else:
            perm_sharpe = 0
        if perm_sharpe >= observed_sharpe:
            n_better += 1

    return round(n_better / n_perms, 4)


def five_gate_validation(metrics, regime_metrics, perm_p):
    gates = {
        'G1_sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'G2_perm_p_lt_0.05': perm_p < 0.05,
        'G3_regime_gap_lt_0.5': regime_metrics['regime_gap'] < 0.5,
        'G4_maxdd_gt_neg50': metrics['max_drawdown_pct'] > -50,
        'G5_min_20_trades': metrics['n_trades'] >= 20,
    }
    gates['all_pass'] = all(gates.values())
    return gates


# ── Variant Definitions ──────────────────────────────────────────────────

VARIANTS = {
    'A': {
        'desc': 'Exact Dual Signal D (5% dip, RSI<35, 3+ red, first green, 10d hold)',
        'universe': ALL_COMMODITIES,
        'signal_fn': generate_signals_A,
        'hold_days': 10,
    },
    'B': {
        'desc': 'Commodity-adjusted (7% dip, RSI<30, 3+ red, first green, 10d hold)',
        'universe': ALL_COMMODITIES,
        'signal_fn': generate_signals_B,
        'hold_days': 10,
    },
    'C': {
        'desc': 'Commodity-adjusted + longer hold (7% dip, RSI<30, 3+ red, first green, 20d hold)',
        'universe': ALL_COMMODITIES,
        'signal_fn': generate_signals_B,
        'hold_days': 20,
    },
    'D': {
        'desc': 'Precious metals only (GLD/SLV/PPLT/PALL, 5% dip, RSI<35, 3+ red, first green, 10d hold)',
        'universe': PRECIOUS_METALS,
        'signal_fn': generate_signals_A,
        'hold_days': 10,
    },
    'E': {
        'desc': 'Energy + Agri only (USO/UNG/DBA/WEAT/CORN, 7% dip, RSI<30, 3+ red, first green, 10d hold)',
        'universe': ENERGY_AGRI,
        'signal_fn': generate_signals_B,
        'hold_days': 10,
    },
    'F': {
        'desc': 'All commodities, 5% dip, RSI<35, 3+ red, first green, VIX>20 filter (flight to real assets)',
        'universe': ALL_COMMODITIES,
        'signal_fn': generate_signals_F,
        'hold_days': 10,
    },
}


# ── Custom JSON encoder for numpy types ──────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)


# ── Run All Variants ─────────────────────────────────────────────────────

results = {}
print("\n" + "=" * 70)
print("COMMODITIES MEAN REVERSION BACKTEST — Dual Signal D Adapted")
print("=" * 70)

for label, cfg in VARIANTS.items():
    print(f"\n{'─' * 60}")
    print(f"[{label}] {cfg['desc']}")
    print(f"  Universe: {cfg['universe']}")

    trades, eq = run_backtest(cfg['universe'], cfg['signal_fn'], cfg['hold_days'])
    metrics = compute_metrics(trades, eq)
    regime = compute_regime_sharpe(trades)
    perm_p = permutation_test(trades, eq)
    gates = five_gate_validation(metrics, regime, perm_p)

    # Ticker breakdown
    ticker_pnl = defaultdict(float)
    ticker_cnt = defaultdict(int)
    for t in trades:
        ticker_pnl[t['ticker']] += t['pnl']
        ticker_cnt[t['ticker']] += 1

    results[label] = {
        'description': cfg['desc'],
        'universe': cfg['universe'],
        'hold_days': cfg['hold_days'],
        'metrics': metrics,
        'regime': regime,
        'permutation_p': perm_p,
        'five_gate': gates,
        'ticker_breakdown': {t: {'pnl': round(p, 2), 'n_trades': ticker_cnt[t]}
                             for t, p in sorted(ticker_pnl.items(), key=lambda x: -x[1])},
        'trades': trades[:50],  # first 50 for inspection
        'total_trades_list_truncated': len(trades) > 50,
    }

    # Print summary
    m = metrics
    g_pass = sum(1 for k, v in gates.items() if k != 'all_pass' and v)
    print(f"  Trades: {m['n_trades']}  |  PnL: ${m['total_pnl']:.2f}  |  Return: {m['total_return_pct']:.1f}%")
    print(f"  Sharpe: {m['sharpe']:.3f}  |  Sortino: {m['sortino']:.3f}  |  PF: {m['profit_factor']:.2f}  |  WR: {m['win_rate']:.1%}")
    print(f"  MaxDD: {m['max_drawdown_pct']:.1f}%  |  Avg Trade: {m['avg_trade_return_pct']:.2f}%")
    print(f"  Regime: bull={regime['sharpe_bull']:.2f} ({regime['n_bull']}), bear={regime['sharpe_bear']:.2f} ({regime['n_bear']}), gap={regime['regime_gap']:.3f}")
    print(f"  Perm p: {perm_p:.4f}  |  5-Gate: {g_pass}/5 {'PASS' if gates['all_pass'] else 'FAIL'}")

    if ticker_pnl:
        print(f"  Top tickers: ", end="")
        top3 = sorted(ticker_pnl.items(), key=lambda x: -x[1])[:3]
        print(", ".join(f"{t}: ${p:.2f} ({ticker_cnt[t]}t)" for t, p in top3))


# ── Save Results ─────────────────────────────────────────────────────────
output = {
    'strategy': 'Commodities Mean Reversion (Dual Signal D Adapted)',
    'run_date': datetime.now().isoformat(),
    'oot_period': f'{OOT_START} to {END}',
    'capital': CAPITAL,
    'max_per_trade': MAX_PER_TRADE,
    'max_concurrent': MAX_CONCURRENT,
    'slippage_pct': SLIPPAGE_PCT,
    'n_permutations': N_PERM,
    'commodities_loaded': loaded,
    'variants': results,
    'summary': {},
}

# Build summary
for label, res in results.items():
    m = res['metrics']
    output['summary'][label] = {
        'sharpe': m['sharpe'],
        'sortino': m['sortino'],
        'pf': m['profit_factor'],
        'wr': m['win_rate'],
        'n_trades': m['n_trades'],
        'total_return_pct': m['total_return_pct'],
        'max_dd_pct': m['max_drawdown_pct'],
        'five_gate_pass': res['five_gate']['all_pass'],
    }

with open(RESULTS_PATH, 'w') as f:
    json.dump(output, f, indent=2, cls=NumpyEncoder)
print(f"\nResults saved to {RESULTS_PATH}")

# ── Final Summary Table ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Var':<4} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'Trades':>7} {'Return':>8} {'MaxDD':>7} {'5-Gate':>7}")
print("-" * 70)
for label in sorted(results.keys()):
    m = results[label]['metrics']
    g = results[label]['five_gate']
    status = "PASS" if g['all_pass'] else "FAIL"
    print(f"  {label}   {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.2f} {m['win_rate']:>5.1%} {m['n_trades']:>7} {m['total_return_pct']:>7.1f}% {m['max_drawdown_pct']:>6.1f}% {status:>7}")
print("=" * 70)
