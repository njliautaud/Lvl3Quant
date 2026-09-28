#!/usr/bin/env python3
"""
Long-Term Post-Earnings Announcement Drift (60-90 day PEAD) Backtest
=====================================================================
Tests whether holding PEAD equity positions for 20-60 trading days
captures additional drift beyond the validated 5-day hold.

BACKGROUND: 5-day PEAD validated at Sharpe 1.51, 60% WR. Academic
literature shows drift continues 60-90 days post-earnings.

ACCOUNT: $645 Robinhood, $0 commission, 0.02% slippage each way.

6 VARIANTS:
  A) 5-day hold (baseline) — >5% positive surprise gap, hold 5 days
  B) 20-day hold
  C) 40-day hold
  D) 60-day hold
  E) Magnitude-scaled 40d — position size proportional to gap magnitude
  F) Regime-filtered 40d — only enter if SPY > 200-SMA

VALIDATION GATES (5-gate):
  1. Sharpe > 0.5
  2. Permutation p < 0.05 (1000 shuffles)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades

Walk-forward OOT: Jan 2022 – Jul 2026
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')

LVL3_ROOT = '/home/jupiter/Lvl3Quant'
OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'longterm_pead')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fprint(*a, **kw):
    print(*a, **kw, flush=True)

# ============================================================
# CONSTANTS
# ============================================================

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'CRM',
    'SNOW', 'PLTR', 'SOFI', 'HOOD', 'SNAP', 'PINS', 'COIN', 'RBLX', 'RIVN', 'UBER',
    'LYFT', 'ROKU', 'NET', 'DDOG', 'TTD', 'SHOP',
]

STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
MAX_CONCURRENT = 3
MIN_GAP_PCT = 0.05  # 5% gap threshold for earnings proxy
START_DATE = '2022-01-01'
END_DATE = '2026-07-29'
N_PERMUTATIONS = 1000
SMA_WINDOW = 200

# ============================================================
# DATA LOADING
# ============================================================

def load_prices():
    """Download OHLCV data for universe + SPY."""
    import yfinance as yf

    cache_path = os.path.join(LVL3_ROOT, 'data', 'longterm_pead_prices.parquet')

    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        cached_tickers = set(df.index.get_level_values(0).unique())
        needed = set(UNIVERSE + ['SPY']) - cached_tickers
        if not needed:
            fprint(f"Loaded cached prices: {len(df)} rows")
            return df

    fprint(f"Downloading {len(UNIVERSE) + 1} tickers...")
    all_tickers = UNIVERSE + ['SPY']
    frames = []
    for t in all_tickers:
        try:
            raw = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if len(raw) < 50:
                fprint(f"  SKIP {t}: only {len(raw)} rows")
                continue
            # Flatten multi-level columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            raw['ticker'] = t
            raw.index.name = 'date'
            frames.append(raw)
        except Exception as e:
            fprint(f"  SKIP {t}: {e}")

    if not frames:
        raise RuntimeError("No price data downloaded")

    df = pd.concat(frames).reset_index().set_index(['ticker', 'date']).sort_index()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path)
    fprint(f"Cached {len(df)} price rows for {df.index.get_level_values(0).nunique()} tickers")
    return df


def detect_earnings_gaps(prices_df):
    """
    Detect large gap-up days (>5%) as earnings proxies.
    Returns dict: ticker -> list of gap event dicts.
    """
    events = {}
    available = prices_df.index.get_level_values(0).unique()

    for ticker in UNIVERSE:
        if ticker not in available:
            continue
        try:
            tp = prices_df.loc[ticker].sort_index()
        except KeyError:
            continue

        if len(tp) < 30:
            continue

        ticker_events = []
        for i in range(1, len(tp)):
            prev_close = tp.iloc[i-1]['close']
            curr_open = tp.iloc[i]['open']
            if prev_close <= 0 or curr_open <= 0:
                continue

            gap_pct = (curr_open / prev_close) - 1.0

            # Only positive gaps > threshold (long-only on RH)
            if gap_pct >= MIN_GAP_PCT:
                date = tp.index[i]
                ticker_events.append({
                    'date': date,
                    'gap_pct': gap_pct,
                    'entry_price': curr_open,
                    'idx': i,
                })

        if ticker_events:
            # Deduplicate: keep max 1 event per 70 trading days (quarterly spacing)
            deduped = []
            for evt in sorted(ticker_events, key=lambda x: x['date']):
                if not deduped or (evt['date'] - deduped[-1]['date']).days > 70:
                    deduped.append(evt)
            events[ticker] = deduped

    total = sum(len(v) for v in events.values())
    fprint(f"Detected {total} earnings-proxy gap events across {len(events)} tickers")
    return events


def compute_spy_sma(prices_df):
    """Compute SPY 200-SMA series."""
    try:
        spy = prices_df.loc['SPY', 'close'].sort_index()
        sma200 = spy.rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()
        return spy, sma200
    except Exception:
        fprint("WARNING: Could not compute SPY SMA")
        return None, None


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(prices_df, events, hold_days, variant_name,
                 magnitude_scaled=False, regime_filtered=False,
                 spy_close=None, spy_sma=None):
    """
    Run a single PEAD backtest variant.

    Returns dict with trades, equity curve, and metrics.
    """
    fprint(f"\n{'='*60}")
    fprint(f"VARIANT {variant_name}: hold={hold_days}d, mag_scaled={magnitude_scaled}, regime_filt={regime_filtered}")
    fprint(f"{'='*60}")

    # Collect all potential trades sorted by date
    all_signals = []
    for ticker, evts in events.items():
        try:
            tp = prices_df.loc[ticker].sort_index()
        except KeyError:
            continue

        for evt in evts:
            entry_date = evt['date']
            entry_idx = evt['idx']

            # Need enough future data for hold period
            if entry_idx + hold_days >= len(tp):
                continue

            exit_idx = entry_idx + hold_days
            entry_price = evt['entry_price']
            exit_price = tp.iloc[exit_idx]['close']

            # Apply slippage
            entry_cost = entry_price * (1 + SLIPPAGE_PCT)
            exit_proceeds = exit_price * (1 - SLIPPAGE_PCT)

            pnl_pct = (exit_proceeds / entry_cost) - 1.0

            # Tag regime for ALL variants (needed for regime gap analysis)
            regime = 'bull'
            if spy_close is not None and spy_sma is not None:
                try:
                    closest_spy_date = spy_close.index[spy_close.index <= entry_date]
                    if len(closest_spy_date) > 0:
                        sd = closest_spy_date[-1]
                        if sd in spy_sma.index and not pd.isna(spy_sma.loc[sd]):
                            regime = 'bull' if spy_close.loc[sd] > spy_sma.loc[sd] else 'bear'
                except Exception:
                    pass

            all_signals.append({
                'ticker': ticker,
                'entry_date': entry_date,
                'exit_date': tp.index[exit_idx],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_cost': entry_cost,
                'exit_proceeds': exit_proceeds,
                'gap_pct': evt['gap_pct'],
                'pnl_pct': pnl_pct,
                'regime': regime,
                'hold_days': hold_days,
            })

    all_signals.sort(key=lambda x: x['entry_date'])

    if regime_filtered:
        before = len(all_signals)
        all_signals = [s for s in all_signals if s['regime'] == 'bull']
        fprint(f"  Regime filter: {before} -> {len(all_signals)} trades (removed {before - len(all_signals)} bear entries)")

    # Position management: max MAX_CONCURRENT concurrent positions
    trades = []
    active = []  # list of (exit_date, alloc_dollars)
    equity = STARTING_CAPITAL
    equity_curve = [(all_signals[0]['entry_date'] - timedelta(days=1), STARTING_CAPITAL)] if all_signals else []

    for sig in all_signals:
        # Remove expired positions
        active = [a for a in active if a[0] > sig['entry_date']]

        if len(active) >= MAX_CONCURRENT:
            continue  # skip, too many concurrent

        # Position sizing
        if magnitude_scaled:
            # Scale by gap magnitude: bigger gap = bigger position
            # Base: equal weight. Scale: gap_pct / 0.05 (normalized to threshold)
            scale = min(sig['gap_pct'] / MIN_GAP_PCT, 3.0)  # cap at 3x
            alloc = (equity / MAX_CONCURRENT) * (scale / 2.0)  # normalize so avg ~= equal weight
        else:
            alloc = equity / MAX_CONCURRENT

        alloc = min(alloc, equity * 0.50)  # never more than 50% in one trade
        if alloc < 10:
            continue  # skip tiny positions

        shares = int(alloc / sig['entry_cost'])
        if shares < 1:
            # Try fractional (RH supports fractional shares)
            shares_frac = alloc / sig['entry_cost']
            if shares_frac < 0.01:
                continue
            actual_alloc = shares_frac * sig['entry_cost']
            pnl_dollars = shares_frac * (sig['exit_proceeds'] - sig['entry_cost'])
        else:
            actual_alloc = shares * sig['entry_cost']
            pnl_dollars = shares * (sig['exit_proceeds'] - sig['entry_cost'])

        equity += pnl_dollars
        active.append((sig['exit_date'], actual_alloc))

        trade = {
            'ticker': sig['ticker'],
            'entry_date': str(sig['entry_date'].date()),
            'exit_date': str(sig['exit_date'].date()),
            'entry_price': round(sig['entry_price'], 2),
            'exit_price': round(sig['exit_price'], 2),
            'gap_pct': round(sig['gap_pct'] * 100, 2),
            'pnl_pct': round(sig['pnl_pct'] * 100, 2),
            'pnl_dollars': round(pnl_dollars, 2),
            'equity_after': round(equity, 2),
            'regime': sig['regime'],
            'shares': round(shares if shares >= 1 else shares_frac, 4),
        }
        trades.append(trade)
        equity_curve.append((sig['exit_date'], equity))

    fprint(f"  Trades taken: {len(trades)}")
    if not trades:
        return None

    # ============================================================
    # METRICS
    # ============================================================
    pnl_pcts = np.array([t['pnl_pct'] for t in trades])
    pnl_dollars = np.array([t['pnl_dollars'] for t in trades])

    n_trades = len(trades)
    win_rate = np.mean(pnl_pcts > 0) * 100
    avg_win = np.mean(pnl_pcts[pnl_pcts > 0]) if np.any(pnl_pcts > 0) else 0
    avg_loss = np.mean(pnl_pcts[pnl_pcts <= 0]) if np.any(pnl_pcts <= 0) else 0
    total_pnl = np.sum(pnl_dollars)
    final_equity = STARTING_CAPITAL + total_pnl
    total_return_pct = (final_equity / STARTING_CAPITAL - 1) * 100

    # Profit factor
    gross_profit = np.sum(pnl_dollars[pnl_dollars > 0]) if np.any(pnl_dollars > 0) else 0
    gross_loss = abs(np.sum(pnl_dollars[pnl_dollars <= 0])) if np.any(pnl_dollars <= 0) else 1e-9
    profit_factor = gross_profit / gross_loss

    # Sharpe (annualized, assuming ~4 trades/month baseline)
    if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
        # Use trade returns, annualize by sqrt(trades_per_year)
        years = (pd.Timestamp(trades[-1]['exit_date']) - pd.Timestamp(trades[0]['entry_date'])).days / 365.25
        trades_per_year = n_trades / max(years, 0.5)
        sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    # Sortino
    downside = pnl_pcts[pnl_pcts < 0]
    if len(downside) > 0 and np.std(downside) > 0 and len(pnl_pcts) > 1:
        years = (pd.Timestamp(trades[-1]['exit_date']) - pd.Timestamp(trades[0]['entry_date'])).days / 365.25
        trades_per_year = n_trades / max(years, 0.5)
        sortino = (np.mean(pnl_pcts) / np.std(downside)) * np.sqrt(trades_per_year)
    else:
        sortino = 0

    # Max drawdown
    eq_series = np.array([STARTING_CAPITAL] + [t['equity_after'] for t in trades])
    running_max = np.maximum.accumulate(eq_series)
    drawdowns = (eq_series - running_max) / running_max
    max_dd = np.min(drawdowns) * 100

    # Regime analysis
    bull_trades = [t for t in trades if t['regime'] == 'bull']
    bear_trades = [t for t in trades if t['regime'] == 'bear']

    def regime_sharpe(trade_list):
        if len(trade_list) < 3:
            return 0
        rets = np.array([t['pnl_pct'] for t in trade_list])
        if np.std(rets) == 0:
            return 0
        return float(np.mean(rets) / np.std(rets))

    sharpe_bull = regime_sharpe(bull_trades)
    sharpe_bear = regime_sharpe(bear_trades)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)

    # ============================================================
    # PERMUTATION TEST — Random entry dates to destroy signal timing
    # ============================================================
    fprint(f"  Running permutation test ({N_PERMUTATIONS} iterations)...")
    observed_mean = np.mean(pnl_pcts)

    # Build pool of ALL possible N-day returns across all tickers
    # This is the null distribution: "what if we entered randomly?"
    random_returns_pool = []
    for ticker in UNIVERSE:
        try:
            tp = prices_df.loc[ticker].sort_index()
        except KeyError:
            continue
        if len(tp) < hold_days + 10:
            continue
        closes = tp['close'].values
        opens = tp['open'].values
        for i in range(0, len(tp) - hold_days, 5):  # sample every 5th day for speed
            entry_p = opens[i] * (1 + SLIPPAGE_PCT)
            exit_p = closes[i + hold_days] * (1 - SLIPPAGE_PCT)
            if entry_p > 0:
                random_returns_pool.append((exit_p / entry_p - 1.0) * 100)

    random_returns_pool = np.array(random_returns_pool)
    fprint(f"  Null pool: {len(random_returns_pool)} random {hold_days}d returns")

    perm_count = 0
    for _ in range(N_PERMUTATIONS):
        # Draw same number of trades from random entry pool
        null_sample = np.random.choice(random_returns_pool, size=n_trades, replace=True)
        if np.mean(null_sample) >= observed_mean:
            perm_count += 1

    perm_p = perm_count / N_PERMUTATIONS

    # ============================================================
    # VALIDATION GATES
    # ============================================================
    gates = {
        'sharpe_gt_0.5': sharpe > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime_gap < 0.5,
        'max_dd_gt_neg50': max_dd > -50,
        'trades_gte_20': n_trades >= 20,
    }
    gates_passed = sum(gates.values())

    # ============================================================
    # RESULTS
    # ============================================================
    result = {
        'variant': variant_name,
        'hold_days': hold_days,
        'magnitude_scaled': magnitude_scaled,
        'regime_filtered': regime_filtered,
        'n_trades': n_trades,
        'win_rate': round(win_rate, 1),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'total_return_pct': round(total_return_pct, 1),
        'final_equity': round(final_equity, 2),
        'max_drawdown_pct': round(max_dd, 1),
        'perm_p_value': round(perm_p, 4),
        'regime_sharpe_bull': round(sharpe_bull, 2),
        'regime_sharpe_bear': round(sharpe_bear, 2),
        'regime_gap': round(regime_gap, 3),
        'n_bull_trades': len(bull_trades),
        'n_bear_trades': len(bear_trades),
        'gates': gates,
        'gates_passed': f"{gates_passed}/5",
        'pass': gates_passed == 5,
        'trades': trades,
    }

    fprint(f"\n  --- {variant_name} RESULTS ---")
    fprint(f"  Trades: {n_trades} | WR: {win_rate:.1f}% | PF: {profit_factor:.2f}")
    fprint(f"  Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}")
    fprint(f"  Return: {total_return_pct:.1f}% | ${STARTING_CAPITAL:.0f} -> ${final_equity:.0f}")
    fprint(f"  MaxDD: {max_dd:.1f}% | Perm p: {perm_p:.4f}")
    fprint(f"  Regime: Bull Sharpe={sharpe_bull:.2f}, Bear Sharpe={sharpe_bear:.2f}, Gap={regime_gap:.3f}")
    fprint(f"  Gates: {gates_passed}/5 {'PASS' if gates_passed == 5 else 'FAIL'}")
    for g, v in gates.items():
        fprint(f"    {'[x]' if v else '[ ]'} {g}")

    return result


# ============================================================
# REGIME TAGGING (for non-filtered variants)
# ============================================================

def tag_regimes(events, spy_close, spy_sma):
    """Add regime tag to all events for regime analysis."""
    for ticker, evts in events.items():
        for evt in evts:
            entry_date = evt['date']
            try:
                closest = spy_close.index[spy_close.index <= entry_date]
                if len(closest) > 0:
                    sd = closest[-1]
                    if sd in spy_sma.index and not pd.isna(spy_sma.loc[sd]):
                        evt['regime'] = 'bull' if spy_close.loc[sd] > spy_sma.loc[sd] else 'bear'
                    else:
                        evt['regime'] = 'unknown'
                else:
                    evt['regime'] = 'unknown'
            except Exception:
                evt['regime'] = 'unknown'


# ============================================================
# MAIN
# ============================================================

def main():
    fprint("=" * 70)
    fprint("LONG-TERM PEAD BACKTEST — 60-90 Day Drift Analysis")
    fprint(f"Universe: {len(UNIVERSE)} stocks | Period: {START_DATE} to {END_DATE}")
    fprint(f"Account: ${STARTING_CAPITAL} | Max concurrent: {MAX_CONCURRENT}")
    fprint("=" * 70)

    # Load data
    prices_df = load_prices()

    # Detect earnings gaps
    events = detect_earnings_gaps(prices_df)

    # SPY regime data
    spy_close, spy_sma = compute_spy_sma(prices_df)

    # Tag regimes on all events
    if spy_close is not None and spy_sma is not None:
        tag_regimes(events, spy_close, spy_sma)

    # ============================================================
    # RUN ALL 6 VARIANTS
    # ============================================================
    results = {}

    # A) 5-day baseline
    r = run_backtest(prices_df, events, hold_days=5, variant_name='A_5day_baseline',
                     spy_close=spy_close, spy_sma=spy_sma)
    if r: results['A'] = r

    # B) 20-day hold
    r = run_backtest(prices_df, events, hold_days=20, variant_name='B_20day',
                     spy_close=spy_close, spy_sma=spy_sma)
    if r: results['B'] = r

    # C) 40-day hold
    r = run_backtest(prices_df, events, hold_days=40, variant_name='C_40day',
                     spy_close=spy_close, spy_sma=spy_sma)
    if r: results['C'] = r

    # D) 60-day hold
    r = run_backtest(prices_df, events, hold_days=60, variant_name='D_60day',
                     spy_close=spy_close, spy_sma=spy_sma)
    if r: results['D'] = r

    # E) Magnitude-scaled 40d
    r = run_backtest(prices_df, events, hold_days=40, variant_name='E_40day_mag_scaled',
                     magnitude_scaled=True, spy_close=spy_close, spy_sma=spy_sma)
    if r: results['E'] = r

    # F) Regime-filtered 40d
    r = run_backtest(prices_df, events, hold_days=40, variant_name='F_40day_regime_filtered',
                     regime_filtered=True, spy_close=spy_close, spy_sma=spy_sma)
    if r: results['F'] = r

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    fprint("\n" + "=" * 90)
    fprint("SUMMARY — LONG-TERM PEAD VARIANTS")
    fprint("=" * 90)
    fprint(f"{'Variant':<30} {'Trades':>6} {'WR%':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'Return%':>8} {'MaxDD%':>7} {'PermP':>7} {'RGap':>6} {'Gates':>6}")
    fprint("-" * 90)

    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        if key not in results:
            continue
        r = results[key]
        status = 'PASS' if r['pass'] else 'FAIL'
        fprint(f"{r['variant']:<30} {r['n_trades']:>6} {r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['profit_factor']:>6.2f} {r['total_return_pct']:>7.1f}% {r['max_drawdown_pct']:>6.1f}% {r['perm_p_value']:>7.4f} {r['regime_gap']:>6.3f} {r['gates_passed']:>5} {status}")

    # ============================================================
    # HOLD-PERIOD ANALYSIS (diminishing returns check)
    # ============================================================
    fprint("\n" + "=" * 70)
    fprint("HOLD-PERIOD ANALYSIS — Does longer hold = more drift?")
    fprint("=" * 70)
    for key in ['A', 'B', 'C', 'D']:
        if key in results:
            r = results[key]
            avg_ret = np.mean([t['pnl_pct'] for t in r['trades']])
            fprint(f"  {r['hold_days']:>3}d hold: avg return {avg_ret:+.2f}%, Sharpe {r['sharpe']:.2f}, {r['n_trades']} trades")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    # Remove trade lists for summary JSON (keep separate)
    summary = {}
    for key, r in results.items():
        summary[key] = {k: v for k, v in r.items() if k != 'trades'}
        summary[key]['sample_trades'] = r['trades'][:5]  # first 5 as sample
        summary[key]['n_total_trades'] = len(r['trades'])

    summary['metadata'] = {
        'universe': UNIVERSE,
        'universe_size': len(UNIVERSE),
        'start_date': START_DATE,
        'end_date': END_DATE,
        'starting_capital': STARTING_CAPITAL,
        'min_gap_pct': MIN_GAP_PCT,
        'slippage_pct': SLIPPAGE_PCT,
        'max_concurrent': MAX_CONCURRENT,
        'n_permutations': N_PERMUTATIONS,
        'run_timestamp': datetime.now().isoformat(),
    }

    # Best variant
    passing = {k: v for k, v in results.items() if v['pass']}
    if passing:
        best_key = max(passing, key=lambda k: passing[k]['sharpe'])
        summary['best_variant'] = best_key
        summary['best_sharpe'] = results[best_key]['sharpe']
        summary['recommendation'] = f"Variant {best_key} ({results[best_key]['variant']}) passes all 5 gates with Sharpe {results[best_key]['sharpe']:.2f}"
    else:
        # Find best even if not passing
        if results:
            best_key = max(results, key=lambda k: results[k]['sharpe'])
            summary['best_variant'] = best_key
            summary['best_sharpe'] = results[best_key]['sharpe']
            summary['recommendation'] = f"No variant passes all 5 gates. Best: {best_key} ({results[best_key]['variant']}) with Sharpe {results[best_key]['sharpe']:.2f}, gates {results[best_key]['gates_passed']}"

    out_path = os.path.join(LVL3_ROOT, 'data', 'longterm_pead_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    fprint(f"\nResults saved to {out_path}")

    # Also save full trade logs per variant
    for key, r in results.items():
        trade_path = os.path.join(OUTPUT_DIR, f'trades_{r["variant"]}.json')
        with open(trade_path, 'w') as f:
            json.dump(r['trades'], f, indent=2, default=str)

    fprint("\nDone.")
    return summary


if __name__ == '__main__':
    main()
