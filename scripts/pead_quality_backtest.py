#!/usr/bin/env python3
"""
Post-Earnings Announcement Drift (PEAD) with Quality Filter
============================================================
Academic basis: Ball & Brown (1968), Bernard & Thomas (1989).

Key design decisions to avoid prior failure modes:
1. Use large-cap S&P 500 (not speculative growth stocks) — avoids "just long high-beta"
2. Measure 2-day post-earnings gap (day-of + day-after) — captures the announcement reaction
3. Use gap magnitude as proxy for surprise — avoids needing paid estimates data
4. Quality filter (200-SMA) isolates drift from "buying beaten-down stocks"
5. Permutation test shuffles entry dates to ensure timing matters, not just being long
6. Regime stratification ensures edge isn't just bull-market beta

Variants:
  A: Long only — buy after >2% earnings gap up, hold 20d
  B: Long only — buy after >3% earnings gap up, hold 20d (stricter)
  C: Long+Short — long >2% gap up, short >2% gap down (half size shorts), hold 20d
  D: Long only with quality filter — only buy if above 200-SMA AND >2% gap up
  E: Long only — buy after >2% gap up, hold 40d (longer drift)
  F: Long only — buy after >2% gap up, hold 10d (shorter, more active)

OOT: Jan 2022 - Jul 2026
Starting capital: $645
"""

import json
import warnings
import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
CAPITAL = 645.0
MAX_POSITIONS = 5
SLIPPAGE_PCT = 0.0002  # 0.02% per side
COMMISSION = 0.0

OOT_START = pd.Timestamp('2022-01-01')
OOT_END = pd.Timestamp('2026-07-29')
PERM_ITERATIONS = 100

UNIVERSE = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'BRK-B',
    'LLY', 'AVGO', 'JPM', 'XOM', 'UNH', 'V', 'MA', 'COST', 'PG', 'JNJ',
    'HD', 'WMT', 'NFLX', 'CRM', 'ABBV', 'BAC', 'ORCL', 'CVX', 'MRK',
    'KO', 'PEP', 'AMD', 'ACN', 'ADBE', 'TMO', 'CSCO', 'LIN', 'MCD',
    'ABT', 'WFC', 'GE', 'DHR', 'PM', 'QCOM', 'TXN', 'ISRG', 'INTU',
    'AMGN', 'CAT', 'AMAT', 'BX', 'NOW',
]

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RESULTS_PATH = DATA_DIR / 'pead_quality_results.json'

# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download adjusted price data for universe + SPY."""
    print("Downloading price data...")
    all_tickers = list(set(UNIVERSE + ['SPY']))

    prices = {}
    for ticker in all_tickers:
        try:
            df = yf.download(
                ticker, start='2021-01-01',
                end=(OOT_END + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                progress=False, auto_adjust=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                prices[ticker] = df
        except Exception as e:
            print(f"  Failed {ticker}: {e}")

    print(f"  Downloaded {len(prices)} tickers")
    return prices


# ── Detect Earnings Events ────────────────────────────────────────────────
def detect_earnings_events(prices: dict, gap_threshold: float = 0.02):
    """
    Detect earnings-like events using large single-day gaps.

    For each stock, find days where the open-to-previous-close gap exceeds
    the threshold. Then measure the 2-day move (close[day+1] / close[day-1] - 1)
    as the "earnings reaction".

    To avoid non-earnings gaps (e.g., market-wide crashes), we also require:
    - The gap is at least 2x the stock's recent 20-day daily volatility
    - SPY didn't move more than 2% on the same day (filters macro events)
    """
    events = []
    spy = prices.get('SPY')

    for ticker in UNIVERSE:
        if ticker not in prices or ticker == 'SPY':
            continue

        df = prices[ticker].copy()
        if len(df) < 60:
            continue

        # Calculate gap: open vs previous close
        df['prev_close'] = df['Close'].shift(1)
        df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close']

        # 20-day rolling volatility for relative gap sizing
        df['vol_20d'] = df['Close'].pct_change().rolling(20).std()

        # 200-day SMA for quality filter
        df['sma_200'] = df['Close'].rolling(200).mean()

        # 2-day reaction: close[t+1] / close[t-1] - 1
        df['close_t_plus_1'] = df['Close'].shift(-1)
        df['reaction_2d'] = (df['close_t_plus_1'] / df['prev_close']) - 1

        for i in range(21, len(df) - 1):
            row = df.iloc[i]
            date = df.index[i]

            gap = row['gap_pct']
            if pd.isna(gap) or pd.isna(row['vol_20d']):
                continue

            abs_gap = abs(gap)

            # Must exceed absolute threshold
            if abs_gap < gap_threshold:
                continue

            # Must be at least 2x recent vol (idiosyncratic, not normal noise)
            if abs_gap < 2.0 * row['vol_20d']:
                continue

            # Filter out macro events: SPY must not have moved > 2%
            if spy is not None and date in spy.index:
                spy_ret = abs(spy.loc[date, 'Close'] / spy['Close'].shift(1).loc[date] - 1)
                if pd.notna(spy_ret) and spy_ret > 0.02:
                    continue

            # Minimum 60 days between events for same ticker (quarterly spacing)
            direction = 'up' if gap > 0 else 'down'
            reaction = row['reaction_2d'] if pd.notna(row['reaction_2d']) else gap

            events.append({
                'ticker': ticker,
                'date': date,
                'gap_pct': gap,
                'reaction_2d': reaction,
                'direction': direction,
                'above_sma200': row['Close'] > row['sma_200'] if pd.notna(row['sma_200']) else False,
                'vol_20d': row['vol_20d'],
            })

    # Deduplicate: keep at most 1 event per ticker per 60-day window
    events.sort(key=lambda x: (x['ticker'], x['date']))
    filtered = []
    last_event = {}
    for ev in events:
        tk = ev['ticker']
        if tk in last_event:
            if (ev['date'] - last_event[tk]).days < 60:
                continue
        last_event[tk] = ev['date']
        filtered.append(ev)

    print(f"  Detected {len(filtered)} earnings-like events across {len(set(e['ticker'] for e in filtered))} tickers")
    return filtered


# ── Backtest Engine ────────────────────────────────────────────────────────
def backtest_variant(events, prices, variant_name, config):
    """
    Run a single variant backtest.

    config keys:
      gap_threshold: minimum abs gap to trigger
      hold_days: how many trading days to hold
      long_only: if True, skip shorts
      quality_filter: if True, require above 200-SMA for longs
      short_half_size: if True, shorts use half position size
    """
    gap_threshold = config.get('gap_threshold', 0.02)
    hold_days = config.get('hold_days', 20)
    long_only = config.get('long_only', True)
    quality_filter = config.get('quality_filter', False)
    short_half_size = config.get('short_half_size', True)

    # Filter events to OOT period and matching criteria
    valid_events = []
    for ev in events:
        if ev['date'] < OOT_START:
            continue

        abs_gap = abs(ev['gap_pct'])
        if abs_gap < gap_threshold:
            continue

        if ev['direction'] == 'up':
            # Long signal
            if quality_filter and not ev['above_sma200']:
                continue
            valid_events.append({**ev, 'side': 'long'})
        elif ev['direction'] == 'down' and not long_only:
            valid_events.append({**ev, 'side': 'short'})

    if not valid_events:
        return None

    # Sort by date
    valid_events.sort(key=lambda x: x['date'])

    # Simulate trades
    trades = []
    equity = CAPITAL
    equity_curve = [(OOT_START, CAPITAL)]
    open_positions = []

    # Create daily timeline
    spy = prices.get('SPY')
    if spy is None:
        return None
    trading_days = spy.loc[OOT_START:OOT_END].index

    event_by_date = {}
    for ev in valid_events:
        if ev['date'] not in event_by_date:
            event_by_date[ev['date']] = []
        event_by_date[ev['date']].append(ev)

    for day in trading_days:
        # Close expired positions
        still_open = []
        for pos in open_positions:
            days_held = len(spy.loc[pos['entry_date']:day].index) - 1
            if days_held >= hold_days:
                # Exit
                ticker = pos['ticker']
                if ticker in prices and day in prices[ticker].index:
                    exit_price = prices[ticker].loc[day, 'Close']
                    slippage = exit_price * SLIPPAGE_PCT
                    if pos['side'] == 'long':
                        exit_price -= slippage
                        pnl_pct = (exit_price / pos['entry_price']) - 1
                    else:
                        exit_price += slippage
                        pnl_pct = (pos['entry_price'] / exit_price) - 1

                    size_mult = 0.5 if (pos['side'] == 'short' and short_half_size) else 1.0
                    position_value = pos['alloc'] * size_mult
                    pnl_dollar = position_value * pnl_pct

                    equity += pnl_dollar
                    trades.append({
                        'ticker': ticker,
                        'side': pos['side'],
                        'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                        'exit_date': day.strftime('%Y-%m-%d'),
                        'entry_price': round(pos['entry_price'], 2),
                        'exit_price': round(exit_price, 2),
                        'gap_pct': round(pos['gap_pct'] * 100, 2),
                        'return_pct': round(pnl_pct * 100, 2),
                        'pnl_dollar': round(pnl_dollar, 2),
                        'hold_days': days_held,
                    })
                else:
                    still_open.append(pos)
                    continue
            else:
                still_open.append(pos)
        open_positions = still_open

        # Open new positions
        if day in event_by_date:
            n_open = len(open_positions)
            for ev in event_by_date[day]:
                if n_open >= MAX_POSITIONS:
                    break

                ticker = ev['ticker']
                # Don't double up on same ticker
                if any(p['ticker'] == ticker for p in open_positions):
                    continue

                if ticker not in prices or day not in prices[ticker].index:
                    continue

                # Enter next day's open (day after event detection)
                day_idx = list(prices[ticker].index).index(day)
                if day_idx + 1 >= len(prices[ticker]):
                    continue
                entry_date = prices[ticker].index[day_idx + 1]
                entry_price = prices[ticker].iloc[day_idx + 1]['Open']

                slippage = entry_price * SLIPPAGE_PCT
                if ev['side'] == 'long':
                    entry_price += slippage
                else:
                    entry_price -= slippage

                alloc = equity / MAX_POSITIONS
                open_positions.append({
                    'ticker': ticker,
                    'side': ev['side'],
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'gap_pct': ev['gap_pct'],
                    'alloc': alloc,
                })
                n_open += 1

        equity_curve.append((day, equity))

    # Close any remaining positions at last available price
    for pos in open_positions:
        ticker = pos['ticker']
        if ticker in prices:
            last_price = prices[ticker].iloc[-1]['Close']
            slippage = last_price * SLIPPAGE_PCT
            if pos['side'] == 'long':
                last_price -= slippage
                pnl_pct = (last_price / pos['entry_price']) - 1
            else:
                last_price += slippage
                pnl_pct = (pos['entry_price'] / last_price) - 1

            size_mult = 0.5 if (pos['side'] == 'short' and short_half_size) else 1.0
            position_value = pos['alloc'] * size_mult
            pnl_dollar = position_value * pnl_pct
            equity += pnl_dollar
            trades.append({
                'ticker': ticker,
                'side': pos['side'],
                'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                'exit_date': 'STILL_OPEN',
                'entry_price': round(pos['entry_price'], 2),
                'exit_price': round(last_price, 2),
                'gap_pct': round(pos['gap_pct'] * 100, 2),
                'return_pct': round(pnl_pct * 100, 2),
                'pnl_dollar': round(pnl_dollar, 2),
                'hold_days': -1,
            })

    if len(trades) == 0:
        return None

    return {
        'trades': trades,
        'equity_curve': [(d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d), round(e, 2))
                         for d, e in equity_curve],
        'final_equity': round(equity, 2),
    }


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(result, prices):
    """Compute risk-adjusted metrics from backtest result."""
    trades = result['trades']
    if len(trades) == 0:
        return None

    returns = np.array([t['return_pct'] / 100 for t in trades])
    n_trades = len(trades)

    # Basic metrics
    avg_ret = np.mean(returns) * 100
    med_ret = np.median(returns) * 100
    std_ret = np.std(returns) * 100
    win_rate = np.mean(returns > 0) * 100
    total_return = (result['final_equity'] / CAPITAL - 1) * 100

    # Sharpe (annualized assuming avg hold ~ 20 trading days)
    avg_hold = np.mean([t['hold_days'] for t in trades if t['hold_days'] > 0])
    if avg_hold <= 0:
        avg_hold = 20
    trades_per_year = 252 / avg_hold
    if std_ret > 0:
        sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0:
        downside_std = np.std(downside) * 100
        sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0.0
    else:
        sortino = float('inf')

    # Profit factor
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown from equity curve
    eq_vals = [e for _, e in result['equity_curve']]
    peak = eq_vals[0]
    max_dd = 0
    for v in eq_vals:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    return {
        'n_trades': n_trades,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(profit_factor, 3),
        'win_rate': round(win_rate, 1),
        'max_dd_pct': round(max_dd * 100, 2),
        'total_return_pct': round(total_return, 2),
        'final_equity': result['final_equity'],
        'avg_return_pct': round(avg_ret, 2),
        'median_return_pct': round(med_ret, 2),
        'std_return_pct': round(std_ret, 2),
        'avg_hold_days': round(avg_hold, 1),
    }


# ── Regime Analysis ───────────────────────────────────────────────────────
def regime_analysis(result, prices):
    """Stratify trades by market regime (SPY > 200-SMA = bull, else bear)."""
    spy = prices.get('SPY')
    if spy is None:
        return None

    spy_sma200 = spy['Close'].rolling(200).mean()

    trades = result['trades']
    bull_rets, bear_rets = [], []

    for t in trades:
        entry = pd.Timestamp(t['entry_date'])
        if entry in spy.index:
            sma_val = spy_sma200.get(entry)
            spy_close = spy.loc[entry, 'Close']
            if pd.notna(sma_val) and spy_close > sma_val:
                bull_rets.append(t['return_pct'] / 100)
            else:
                bear_rets.append(t['return_pct'] / 100)
        else:
            # Find nearest trading day
            mask = spy.index <= entry
            if mask.any():
                nearest = spy.index[mask][-1]
                sma_val = spy_sma200.get(nearest)
                spy_close = spy.loc[nearest, 'Close']
                if pd.notna(sma_val) and spy_close > sma_val:
                    bull_rets.append(t['return_pct'] / 100)
                else:
                    bear_rets.append(t['return_pct'] / 100)

    bull_rets = np.array(bull_rets) if bull_rets else np.array([0.0])
    bear_rets = np.array(bear_rets) if bear_rets else np.array([0.0])

    bull_sharpe = (np.mean(bull_rets) / np.std(bull_rets)) * np.sqrt(12) if np.std(bull_rets) > 0 else 0.0
    bear_sharpe = (np.mean(bear_rets) / np.std(bear_rets)) * np.sqrt(12) if np.std(bear_rets) > 0 else 0.0

    max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_sharpe if max_sharpe > 0 else 0.0

    return {
        'bull_trades': len(bull_rets),
        'bear_trades': len(bear_rets),
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'bull_avg_ret': round(np.mean(bull_rets) * 100, 2),
        'bear_avg_ret': round(np.mean(bear_rets) * 100, 2),
        'regime_gap': round(regime_gap, 3),
    }


# ── Permutation Test ──────────────────────────────────────────────────────
def permutation_test(result, events, prices, config, n_perms=PERM_ITERATIONS):
    """
    Shuffle trade entry dates within the OOT period to test if timing matters.

    For each permutation:
    - Take the same number of trades
    - Randomly assign entry dates from all available trading days
    - Measure average return

    p-value = fraction of permuted averages >= actual average return.
    """
    trades = result['trades']
    actual_avg = np.mean([t['return_pct'] for t in trades])

    spy = prices.get('SPY')
    if spy is None:
        return {'perm_p_value': 1.0}

    oot_days = spy.loc[OOT_START:OOT_END].index
    n_trades = len(trades)

    # For each trade, pick a random ticker from universe and random entry day,
    # then measure the return over hold_days
    hold_days = config.get('hold_days', 20)
    long_only = config.get('long_only', True)

    tickers_with_data = [t for t in UNIVERSE if t in prices]

    perm_avgs = []
    rng = np.random.default_rng(42)

    for _ in range(n_perms):
        perm_rets = []
        for _ in range(n_trades):
            ticker = rng.choice(tickers_with_data)
            df = prices[ticker]
            valid_start = df.index[(df.index >= OOT_START) & (df.index <= OOT_END)]
            if len(valid_start) < hold_days + 2:
                continue

            idx = rng.integers(0, len(valid_start) - hold_days - 1)
            entry_date = valid_start[idx]
            exit_idx = min(idx + hold_days, len(valid_start) - 1)
            exit_date = valid_start[exit_idx]

            entry_p = df.loc[entry_date, 'Close']
            exit_p = df.loc[exit_date, 'Close']

            ret = (exit_p / entry_p - 1) * 100
            perm_rets.append(ret)

        if perm_rets:
            perm_avgs.append(np.mean(perm_rets))

    if not perm_avgs:
        return {'perm_p_value': 1.0}

    p_value = np.mean(np.array(perm_avgs) >= actual_avg)
    return {
        'perm_p_value': round(float(p_value), 4),
        'perm_avg_mean': round(float(np.mean(perm_avgs)), 2),
        'perm_avg_std': round(float(np.std(perm_avgs)), 2),
        'actual_avg': round(float(actual_avg), 2),
    }


# ── 5-Gate Validation ─────────────────────────────────────────────────────
def validate_gates(metrics, regime, perm):
    """Apply 5-gate validation."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm['perm_p_value'] < 0.05,
        'regime_gap_lt_0.5': regime['regime_gap'] < 0.5,
        'maxdd_gt_neg50': metrics['max_dd_pct'] > -50,
        'trades_gte_20': metrics['n_trades'] >= 20,
    }
    gates['all_pass'] = all(gates.values())
    gates['gates_passed'] = sum(1 for v in list(gates.values())[:-1] if v)
    return gates


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PEAD with Quality Filter Backtest")
    print("=" * 70)

    prices = download_data()

    # Detect events at the lowest threshold (2%) — we'll filter per variant
    events = detect_earnings_events(prices, gap_threshold=0.02)

    # Also detect at 3% for variant B
    events_3pct = [e for e in events if abs(e['gap_pct']) >= 0.03]
    print(f"  Events with |gap| >= 3%: {len(events_3pct)}")

    # Filter to OOT period for summary
    oot_events = [e for e in events if e['date'] >= OOT_START]
    print(f"  OOT events (>= 2%): {len(oot_events)}")
    up_events = sum(1 for e in oot_events if e['direction'] == 'up')
    down_events = sum(1 for e in oot_events if e['direction'] == 'down')
    print(f"    Up: {up_events}, Down: {down_events}")

    # Variant definitions
    variants = {
        'A_long_2pct_20d': {
            'gap_threshold': 0.02,
            'hold_days': 20,
            'long_only': True,
            'quality_filter': False,
        },
        'B_long_3pct_20d': {
            'gap_threshold': 0.03,
            'hold_days': 20,
            'long_only': True,
            'quality_filter': False,
        },
        'C_longshort_2pct_20d': {
            'gap_threshold': 0.02,
            'hold_days': 20,
            'long_only': False,
            'quality_filter': False,
            'short_half_size': True,
        },
        'D_quality_2pct_20d': {
            'gap_threshold': 0.02,
            'hold_days': 20,
            'long_only': True,
            'quality_filter': True,
        },
        'E_long_2pct_40d': {
            'gap_threshold': 0.02,
            'hold_days': 40,
            'long_only': True,
            'quality_filter': False,
        },
        'F_long_2pct_10d': {
            'gap_threshold': 0.02,
            'hold_days': 10,
            'long_only': True,
            'quality_filter': False,
        },
    }

    results = {
        'strategy': 'pead_quality_filter',
        'description': 'Post-Earnings Announcement Drift with Quality Filter',
        'academic_basis': 'Ball & Brown (1968), Bernard & Thomas (1989)',
        'oot_period': f'{OOT_START.strftime("%Y-%m-%d")} to {OOT_END.strftime("%Y-%m-%d")}',
        'capital': CAPITAL,
        'universe_size': len(UNIVERSE),
        'slippage_pct': SLIPPAGE_PCT,
        'commission': COMMISSION,
        'max_positions': MAX_POSITIONS,
        'event_detection': 'Gap > threshold AND gap > 2x 20d vol AND SPY < 2% (filters macro events)',
        'dedup': '1 event per ticker per 60 days (quarterly spacing)',
        'timestamp': dt.datetime.now().isoformat(),
        'total_events_detected': len(events),
        'oot_events': len(oot_events),
        'variants': {},
    }

    for name, config in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Variant: {name}")
        print(f"  Config: gap>={config['gap_threshold']*100:.0f}%, hold={config['hold_days']}d, "
              f"long_only={config.get('long_only', True)}, quality={config.get('quality_filter', False)}")

        bt = backtest_variant(events, prices, name, config)
        if bt is None or len(bt['trades']) == 0:
            print(f"  NO TRADES — skipping")
            results['variants'][name] = {'status': 'no_trades'}
            continue

        metrics = compute_metrics(bt, prices)
        regime = regime_analysis(bt, prices)
        perm = permutation_test(bt, events, prices, config)
        gates = validate_gates(metrics, regime, perm)

        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f}")
        print(f"  PF: {metrics['profit_factor']:.2f} | WR: {metrics['win_rate']:.1f}%")
        print(f"  MaxDD: {metrics['max_dd_pct']:.1f}% | Total Return: {metrics['total_return_pct']:.1f}%")
        print(f"  Avg Ret: {metrics['avg_return_pct']:.2f}% | Med Ret: {metrics['median_return_pct']:.2f}%")
        print(f"  Regime: bull={regime['bull_sharpe']:.3f} bear={regime['bear_sharpe']:.3f} gap={regime['regime_gap']:.3f}")
        print(f"  Perm p-value: {perm['perm_p_value']:.4f}")
        print(f"  Gates: {gates['gates_passed']}/5 {'✓ ALL PASS' if gates['all_pass'] else '✗ FAIL'}")
        for gate, passed in gates.items():
            if gate not in ('all_pass', 'gates_passed'):
                status = '✓' if passed else '✗'
                print(f"    {status} {gate}: {passed}")

        results['variants'][name] = {
            'config': config,
            'metrics': metrics,
            'regime': regime,
            'permutation': perm,
            'gates': gates,
            'sample_trades': bt['trades'][:10],
            'n_trades': metrics['n_trades'],
        }

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'PF':>6} {'WR%':>5} {'MaxDD%':>7} {'Perm-p':>7} {'RegGap':>7} {'Gates':>6}")
    print("-" * 85)

    for name, vr in results['variants'].items():
        if vr.get('status') == 'no_trades':
            print(f"{name:<25} {'NO TRADES':>6}")
            continue
        m = vr['metrics']
        r = vr['regime']
        p = vr['permutation']
        g = vr['gates']
        print(f"{name:<25} {m['n_trades']:>6} {m['sharpe']:>7.3f} {m['profit_factor']:>6.2f} "
              f"{m['win_rate']:>5.1f} {m['max_dd_pct']:>7.1f} {p['perm_p_value']:>7.4f} "
              f"{r['regime_gap']:>7.3f} {g['gates_passed']:>3}/5 {'✓' if g['all_pass'] else '✗'}")

    # Save
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
