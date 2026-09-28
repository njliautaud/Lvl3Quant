#!/usr/bin/env python3
"""
ETF Mean Reversion with Dual Signal Backtest
=============================================
Tests whether the Dual Signal D logic (Sharpe 1.77 on quality stocks) transfers to major ETFs.

Dual Signal D requires BOTH conditions simultaneously:
  1. Dip from recent high + RSI oversold
  2. First green day after consecutive red days (recovery confirmation)

6 Variants:
  A: Exact Dual Signal D (5% dip from 20d high, RSI<35, 3 red days, 10d hold)
  B: Adjusted for ETFs (3% dip, RSI<35, 2 red days, 10d hold)
  C: Adjusted + longer hold (3% dip, RSI<35, 2 red days, 15d hold)
  D: Equity ETFs only (no TLT/GLD/HYG/EEM/VGK/EFA), 3% dip, 2 red, 10d
  E: All ETFs, 3% dip, 2 red, 10d hold, VIX<25 filter
  F: All ETFs, 3% dip, 2 red, dynamic exit (hold until RSI crosses above 50, max 30d)

OOT: Jan 2022 - Jul 2026. $645 capital. 0.01% slippage each way.
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
SLIPPAGE_PCT = 0.0001  # 0.01% each way (ETFs more liquid)
DATA_START = "2020-01-01"  # lookback for indicators
OOT_START = "2022-01-01"
END = "2026-07-31"
N_PERM = 1000

ALL_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLE", "XLI", "XLC",
    "XLY", "XLP", "XLU", "XLRE", "TLT", "GLD", "EEM", "VGK", "EFA", "HYG",
]

EQUITY_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLE", "XLI", "XLC",
    "XLY", "XLP", "XLU", "XLRE",
]

RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/etf_mean_reversion_results.json"

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading ETF price data ...")
ALL_DOWNLOAD = sorted(set(ALL_ETFS + ["^VIX"]))
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

loaded = sum(1 for t in ALL_ETFS if len(closes.get(t, [])) > 100)
print(f"  ETFs with data: {loaded}/{len(ALL_ETFS)}")
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
for t in ALL_ETFS:
    s = closes.get(t, pd.Series(dtype=float))
    if len(s) < 50:
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
    """Exact Dual Signal D: 5% dip from 20d high, RSI<35, 3+ red days then green."""
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
    """ETF-adjusted: 3% dip from 20d high, RSI<35, 2+ red days then green, 10d hold."""
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
        cond1 = drawdown < -0.03 and rsi < 35
        red_days = count_consecutive_red_days(s, date)
        green = is_green_day(s, date)
        cond2 = red_days >= 2 and green

        if cond1 and cond2:
            signals.append((t, float(price)))
    return signals


def generate_signals_E(date, universe):
    """Same as B but with VIX < 25 filter."""
    if date in vix_close.index:
        vix = vix_close.loc[date]
        if pd.notna(vix) and vix >= 25:
            return []
    return generate_signals_B(date, universe)


# ── Backtest Engine ──────────────────────────────────────────────────────

def build_close_df(universe):
    """Build a DataFrame of closes for the given universe."""
    frames = {}
    for t in universe:
        if t in closes:
            frames[t] = closes[t]
    return pd.DataFrame(frames)


def run_backtest(universe, signal_fn, hold_days, dynamic_exit_rsi=None):
    """
    Run backtest.
    If dynamic_exit_rsi is set (e.g. 50), hold until RSI crosses above that level (max 30 trading days).
    """
    close_df = build_close_df(universe)
    oot_idx = close_df.loc[OOT_START:END].index
    if len(oot_idx) == 0:
        return [], pd.Series(dtype=float)

    capital = CAPITAL
    positions = []
    trades = []
    equity = {}
    max_hold = 30 if dynamic_exit_rsi else hold_days

    for date in oot_idx:
        # ── Check exits ──
        still_open = []
        for pos in positions:
            should_exit = False

            # Time-based exit
            if date >= pos['exit_target']:
                should_exit = True

            # Dynamic RSI exit
            if dynamic_exit_rsi and not should_exit:
                t = pos['ticker']
                if t in indicators and date in indicators[t]['rsi14'].index:
                    current_rsi = indicators[t]['rsi14'].loc[date]
                    if pd.notna(current_rsi) and current_rsi > dynamic_exit_rsi:
                        should_exit = True

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
                exit_loc = min(oot_idx.get_loc(date) + max_hold, len(oot_idx) - 1)
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
        'desc': 'Exact Dual Signal D on ETFs (5% dip, RSI<35, 3 red, 10d hold)',
        'universe': ALL_ETFS,
        'signal_fn': generate_signals_A,
        'hold_days': 10,
        'dynamic_exit_rsi': None,
    },
    'B': {
        'desc': 'ETF-adjusted (3% dip, RSI<35, 2 red, 10d hold)',
        'universe': ALL_ETFS,
        'signal_fn': generate_signals_B,
        'hold_days': 10,
        'dynamic_exit_rsi': None,
    },
    'C': {
        'desc': 'ETF-adjusted + longer hold (3% dip, RSI<35, 2 red, 15d hold)',
        'universe': ALL_ETFS,
        'signal_fn': generate_signals_B,
        'hold_days': 15,
        'dynamic_exit_rsi': None,
    },
    'D': {
        'desc': 'Equity ETFs only (3% dip, RSI<35, 2 red, 10d hold)',
        'universe': EQUITY_ETFS,
        'signal_fn': generate_signals_B,
        'hold_days': 10,
        'dynamic_exit_rsi': None,
    },
    'E': {
        'desc': 'All ETFs + VIX<25 filter (3% dip, RSI<35, 2 red, 10d hold)',
        'universe': ALL_ETFS,
        'signal_fn': generate_signals_E,
        'hold_days': 10,
        'dynamic_exit_rsi': None,
    },
    'F': {
        'desc': 'All ETFs, dynamic exit RSI>50 (3% dip, RSI<35, 2 red, max 30d)',
        'universe': ALL_ETFS,
        'signal_fn': generate_signals_B,
        'hold_days': 30,
        'dynamic_exit_rsi': 50,
    },
}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    results = {}
    summary_rows = []

    for vname, cfg in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Running Variant {vname}: {cfg['desc']}")
        print(f"{'='*60}")

        trades, equity = run_backtest(
            cfg['universe'], cfg['signal_fn'],
            cfg['hold_days'], cfg.get('dynamic_exit_rsi'),
        )
        metrics = compute_metrics(trades, equity)
        regime = compute_regime_sharpe(trades)
        perm_p = permutation_test(trades, equity)
        gates = five_gate_validation(metrics, regime, perm_p)

        print(f"  Trades: {metrics['n_trades']}, PnL: ${metrics['total_pnl']:.2f}, "
              f"Return: {metrics['total_return_pct']:.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}, "
              f"PF: {metrics['profit_factor']:.2f}, WR: {metrics['win_rate']:.1%}")
        print(f"  MaxDD: {metrics['max_drawdown_pct']:.1f}%, AvgTrade: {metrics['avg_trade_return_pct']:.2f}%")
        print(f"  Regime: Bull Sharpe={regime['sharpe_bull']:.3f} ({regime['n_bull']}), "
              f"Bear Sharpe={regime['sharpe_bear']:.3f} ({regime['n_bear']}), Gap={regime['regime_gap']:.3f}")
        print(f"  Perm p-value: {perm_p:.4f}")
        print(f"  Gates: {' | '.join(f'{k}={v}' for k,v in gates.items())}")

        passed = 'PASS' if gates['all_pass'] else 'FAIL'
        print(f"  >>> 5-GATE: {passed}")

        # Ticker breakdown
        ticker_pnl = defaultdict(float)
        ticker_cnt = defaultdict(int)
        for tr in trades:
            ticker_pnl[tr['ticker']] += tr['pnl']
            ticker_cnt[tr['ticker']] += 1
        if ticker_pnl:
            top3 = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)[:3]
            bot3 = sorted(ticker_pnl.items(), key=lambda x: x[1])[:3]
            print(f"  Top ETFs: {', '.join(f'{t}=${p:.1f}({ticker_cnt[t]}t)' for t,p in top3)}")
            print(f"  Bot ETFs: {', '.join(f'{t}=${p:.1f}({ticker_cnt[t]}t)' for t,p in bot3)}")

        results[vname] = {
            'description': cfg['desc'],
            'hold_days': cfg['hold_days'],
            'dynamic_exit_rsi': cfg.get('dynamic_exit_rsi'),
            'universe_size': len(cfg['universe']),
            'metrics': metrics,
            'regime': regime,
            'permutation_p_value': perm_p,
            'gates': {k: bool(v) for k, v in gates.items()},
            'ticker_pnl': {t: round(p, 2) for t, p in ticker_pnl.items()},
            'ticker_count': dict(ticker_cnt),
            'sample_trades': trades[:5] if len(trades) > 5 else trades,
        }

        summary_rows.append({
            'Variant': vname,
            'Trades': metrics['n_trades'],
            'PnL': f"${metrics['total_pnl']:.0f}",
            'Return': f"{metrics['total_return_pct']:.1f}%",
            'Sharpe': metrics['sharpe'],
            'Sortino': metrics['sortino'],
            'PF': metrics['profit_factor'],
            'WR': f"{metrics['win_rate']:.0%}",
            'MaxDD': f"{metrics['max_drawdown_pct']:.1f}%",
            'RegGap': regime['regime_gap'],
            'PermP': perm_p,
            '5Gate': passed,
        })

    # ── Summary Table ─────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("ETF MEAN REVERSION WITH DUAL SIGNAL -- SUMMARY TABLE")
    print(f"OOT: {OOT_START} to {END} | Capital: ${CAPITAL} | Slippage: {SLIPPAGE_PCT*100:.2f}%/side")
    print(f"{'='*100}")

    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))

    passed_variants = [r['Variant'] for r in summary_rows if r['5Gate'] == 'PASS']
    failed_variants = [r['Variant'] for r in summary_rows if r['5Gate'] == 'FAIL']
    print(f"\nPASSED 5-GATE: {passed_variants if passed_variants else 'None'}")
    print(f"FAILED 5-GATE: {failed_variants if failed_variants else 'None'}")

    # ── Save results ──────────────────────────────────────────────────────
    results['_meta'] = {
        'strategy': 'ETF Mean Reversion with Dual Signal',
        'universe': ALL_ETFS,
        'equity_etfs': EQUITY_ETFS,
        'oot_start': OOT_START,
        'oot_end': END,
        'starting_capital': CAPITAL,
        'slippage_pct_each_way': SLIPPAGE_PCT,
        'max_positions': MAX_CONCURRENT,
        'max_per_trade': MAX_PER_TRADE,
        'n_permutations': N_PERM,
        'run_timestamp': datetime.now().isoformat(),
        'passed_variants': passed_variants,
        'failed_variants': failed_variants,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
