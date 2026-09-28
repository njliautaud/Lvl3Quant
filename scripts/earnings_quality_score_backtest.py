#!/usr/bin/env python3
"""
Earnings Quality Score Backtest
================================
Builds on validated Earnings Surprise Momentum (Sharpe 1.54, perm p=0.001)
by adding a QUALITY SCORE to filter entries.

Hypothesis: Not all earnings beats are equal. Better quality beats = stronger drift.

Quality Score Components (0-100):
  1. Beat Magnitude (0-30): gap_pct on earnings day
  2. Volume Confirmation (0-25): volume vs 20-day avg
  3. Price Momentum (0-20): stock vs 50-SMA / 200-SMA
  4. Consecutive Beat Streak (0-25): sequential beat count

6 Variants:
  A) Quality > 50: Only enter when quality score > 50, hold 40d
  B) Quality > 70: Higher bar, fewer trades, hold 40d
  C) Quality-Weighted Sizing: Enter all beats, size by quality/100, hold 40d
  D) Quality + Kill Switch: Q>50, skip when VIX>20 AND price<50SMA, hold 40d
  E) Sector Leader Quality: Q>50, top sector performer trailing 60d, hold 60d
  F) Adversarial Random Quality: Random scores 0-100, same rules as A

Walk-forward OOT: Jan 2022 - Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime_gap<0.5, MaxDD>-50%, >=20 trades
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
CAPITAL = 645.0
MAX_POSITIONS = 5
SLIPPAGE_PCT = 0.0002
COMMISSION = 0.0

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    'NFLX', 'CRM', 'PLTR', 'SOFI', 'HOOD', 'SNAP', 'PINS', 'UBER',
    'LYFT', 'COIN', 'RBLX', 'DDOG', 'TTD', 'SHOP', 'NET', 'ROKU'
]

SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'AMD': 'Tech',
    'CRM': 'Tech', 'DDOG': 'Tech', 'NET': 'Tech', 'SHOP': 'Tech',
    'TTD': 'Tech', 'PLTR': 'Tech',
    'GOOGL': 'Comm', 'META': 'Comm', 'NFLX': 'Comm', 'SNAP': 'Comm',
    'PINS': 'Comm', 'ROKU': 'Comm',
    'AMZN': 'Disc', 'TSLA': 'Disc', 'RBLX': 'Disc',
    'UBER': 'Disc', 'LYFT': 'Disc',
    'SOFI': 'Fin', 'HOOD': 'Fin', 'COIN': 'Fin',
}

OOT_START = pd.Timestamp('2022-01-01')
OOT_END = pd.Timestamp('2026-07-29')
HOLD_DAYS_DEFAULT = 40
HOLD_DAYS_E = 60
PERM_ITERATIONS = 1000

# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download price data for universe + SPY + VIX."""
    print("Downloading price data...")
    all_tickers = list(set(UNIVERSE + ['SPY', '^VIX']))

    prices = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start='2021-01-01', end=OOT_END.strftime('%Y-%m-%d'),
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 50:
                prices[ticker] = df
        except Exception as e:
            print(f"  Failed {ticker}: {e}")

    print(f"  Downloaded {len(prices)} tickers")
    return prices


# ── Earnings Detection ────────────────────────────────────────────────────
def detect_earnings_events(prices):
    """
    Detect earnings events via overnight gap > 1% as beat proxy.
    Track consecutive beats per ticker.
    """
    events = {}

    for ticker in UNIVERSE:
        if ticker not in prices:
            continue
        df = prices[ticker].copy()
        df['prev_close'] = df['Close'].shift(1)
        df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close']
        df['abs_gap'] = df['gap_pct'].abs()

        # Volume ratio
        df['vol_ma20'] = df['Volume'].rolling(20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_ma20']

        # 50-SMA and 200-SMA
        df['sma50'] = df['Close'].rolling(50).mean()
        df['sma200'] = df['Close'].rolling(200).mean()

        # Earnings = gap > 1% (lower threshold per spec) with volume spike
        earnings_mask = (df['abs_gap'] > 0.01) & (df['vol_ratio'] > 1.3)
        earnings_days = df[earnings_mask].copy()

        if len(earnings_days) == 0:
            continue

        # Deduplicate: 1 event per 60-day window
        deduped = []
        last_date = None
        for date, row in earnings_days.iterrows():
            if last_date is None or (date - last_date).days > 60:
                deduped.append({
                    'date': date,
                    'ticker': ticker,
                    'gap_pct': float(row['gap_pct']),
                    'abs_gap': float(row['abs_gap']),
                    'is_beat': float(row['gap_pct']) > 0,
                    'vol_ratio': float(row['vol_ratio']) if not np.isnan(row['vol_ratio']) else 1.0,
                    'close': float(row['Close']),
                    'sma50': float(row['sma50']) if not np.isnan(row['sma50']) else float(row['Close']),
                    'sma200': float(row['sma200']) if not np.isnan(row['sma200']) else float(row['Close']),
                })
                last_date = date

        events[ticker] = deduped

    # Compute consecutive beat streaks
    for ticker, evts in events.items():
        streak = 0
        for i, e in enumerate(evts):
            if e['is_beat']:
                streak += 1
            else:
                streak = 0
            e['beat_streak'] = streak

    total = sum(len(v) for v in events.values())
    beats = sum(sum(1 for e in v if e['is_beat']) for v in events.values())
    print(f"  Detected {total} earnings events ({beats} beats) across {len(events)} stocks")
    return events


# ── Quality Score ──────────────────────────────────────────────────────────
def compute_quality_score(event):
    """Compute quality score 0-100 for a beat event."""
    score = 0

    # 1. Beat Magnitude (0-30): gap_pct
    gap = abs(event['gap_pct']) * 100  # as percentage
    if gap > 5:
        score += 30
    elif gap > 3:
        score += 20
    elif gap > 1:
        score += 10

    # 2. Volume Confirmation (0-25): vol_ratio
    vr = event['vol_ratio']
    if vr > 3.0:
        score += 25
    elif vr > 2.0:
        score += 15
    elif vr > 1.5:
        score += 10

    # 3. Price Momentum (0-20): price vs SMAs
    price = event['close']
    sma50 = event['sma50']
    sma200 = event['sma200']
    if price > sma50:
        score += 20
    elif price > sma200:
        score += 10
    # below both = 0

    # 4. Consecutive Beat Streak (0-25)
    streak = event['beat_streak']
    if streak >= 4:
        score += 25
    elif streak >= 3:
        score += 15
    elif streak >= 2:
        score += 10
    # streak 1 = 0

    return score


# ── Helpers ────────────────────────────────────────────────────────────────
def get_price_at(prices_df, date, field='Open', offset_days=0):
    """Get price at date + offset trading days."""
    if date not in prices_df.index:
        mask = prices_df.index >= date
        if not mask.any():
            return None
        date = prices_df.index[mask][0]
    idx = prices_df.index.get_loc(date)
    target_idx = idx + offset_days
    if target_idx < 0 or target_idx >= len(prices_df):
        return None
    val = prices_df.iloc[target_idx][field]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return float(val)


def get_spy_regime(prices, date):
    """Bull if SPY > 200-SMA, else Bear."""
    if 'SPY' not in prices:
        return 'bull'
    spy = prices['SPY']
    if date not in spy.index:
        mask = spy.index <= date
        if not mask.any():
            return 'bull'
        date = spy.index[mask][-1]
    idx = spy.index.get_loc(date)
    if idx < 200:
        return 'bull'
    sma200 = spy['Close'].iloc[max(0, idx-199):idx+1].mean()
    close = float(spy['Close'].iloc[idx])
    if isinstance(close, pd.Series):
        close = close.iloc[0]
    return 'bull' if close > sma200 else 'bear'


def get_vix_level(prices, date):
    """Get VIX level at date."""
    vix_key = '^VIX'
    if vix_key not in prices:
        return 15.0  # default
    vix = prices[vix_key]
    if date not in vix.index:
        mask = vix.index <= date
        if not mask.any():
            return 15.0
        date = vix.index[mask][-1]
    val = float(vix['Close'].loc[date])
    return val


def is_sector_leader(ticker, date, prices, lookback=60):
    """Check if ticker is top performer in its sector over trailing lookback days."""
    sector = SECTOR_MAP.get(ticker)
    if not sector:
        return False

    sector_tickers = [t for t, s in SECTOR_MAP.items() if s == sector and t in prices]
    returns = {}

    for t in sector_tickers:
        df = prices[t]
        mask = df.index <= date
        if mask.sum() < lookback:
            continue
        end_idx = df.index[mask][-1]
        end_loc = df.index.get_loc(end_idx)
        start_loc = max(0, end_loc - lookback)
        start_price = float(df['Close'].iloc[start_loc])
        end_price = float(df['Close'].iloc[end_loc])
        if isinstance(start_price, pd.Series):
            start_price = start_price.iloc[0]
        if isinstance(end_price, pd.Series):
            end_price = end_price.iloc[0]
        if start_price > 0:
            returns[t] = (end_price - start_price) / start_price

    if not returns or ticker not in returns:
        return False

    # Top performer = rank 1 in sector
    sorted_ret = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    return sorted_ret[0][0] == ticker


# ── Trade Simulation ──────────────────────────────────────────────────────
def simulate_variant(variant_name, prices, events, hold_days=40, rng=None):
    """
    Run a single variant backtest. Returns trades list and final equity.
    Uses FIXED position sizing (fraction of starting capital) to avoid
    unrealistic compounding artifacts.
    """
    trades = []

    # Collect all beat events in OOT period
    all_beats = []
    for ticker, evts in events.items():
        for e in evts:
            if e['is_beat'] and e['date'] >= OOT_START and e['date'] <= OOT_END:
                e_copy = dict(e)
                e_copy['quality'] = compute_quality_score(e)
                all_beats.append(e_copy)

    # For adversarial: assign random quality scores
    if variant_name == 'F' and rng is not None:
        for b in all_beats:
            b['quality'] = rng.integers(0, 101)

    # Sort by date
    all_beats.sort(key=lambda x: x['date'])

    # Fixed position size based on starting capital
    base_position_size = CAPITAL / MAX_POSITIONS  # $129 per slot

    # Active positions tracking (for max position enforcement)
    active_positions = []  # list of (ticker, entry_date, hold_days)

    for beat in all_beats:
        ticker = beat['ticker']
        date = beat['date']

        if ticker not in prices:
            continue

        # Expire old positions
        active_positions = [
            p for p in active_positions
            if date < p['entry_date'] + pd.Timedelta(days=p['hold_days'])
        ]

        # Entry filters by variant
        enter = False
        sizing = 1.0
        this_hold = hold_days

        if variant_name == 'A':
            enter = beat['quality'] > 50
        elif variant_name == 'B':
            enter = beat['quality'] > 70
        elif variant_name == 'C':
            enter = True  # enter all beats
            sizing = beat['quality'] / 100.0
            sizing = max(sizing, 0.1)  # min 10% sizing
        elif variant_name == 'D':
            vix = get_vix_level(prices, date)
            price_below_sma50 = beat['close'] < beat['sma50']
            kill_switch = (vix > 20) and price_below_sma50
            enter = beat['quality'] > 50 and not kill_switch
        elif variant_name == 'E':
            enter = beat['quality'] > 50 and is_sector_leader(ticker, date, prices, lookback=60)
            this_hold = HOLD_DAYS_E
        elif variant_name == 'F':
            enter = beat['quality'] > 50  # same rules as A but random scores

        if not enter:
            continue

        if len(active_positions) >= MAX_POSITIONS:
            continue

        # Entry: buy at next day open
        entry_price = get_price_at(prices[ticker], date, 'Open', offset_days=1)
        if entry_price is None:
            continue

        entry_price *= (1 + SLIPPAGE_PCT)  # slippage on entry

        # Exit: close at hold_days trading days later
        # Use trading day offset from entry (offset_days=1 for entry, then hold_days more)
        exit_price = get_price_at(prices[ticker], date, 'Close', offset_days=1 + this_hold)
        if exit_price is None:
            # Try last available price
            df = prices[ticker]
            exit_price = float(df['Close'].iloc[-1])
            if isinstance(exit_price, pd.Series):
                exit_price = exit_price.iloc[0]

        position_size = base_position_size * sizing
        ret = (exit_price - entry_price) / entry_price
        pnl = position_size * ret

        regime = get_spy_regime(prices, date)

        trades.append({
            'ticker': ticker,
            'entry_date': str(date.date()),
            'exit_date': str((date + pd.Timedelta(days=this_hold)).date()),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'return': ret,
            'pnl': pnl,
            'quality': beat['quality'],
            'regime': regime,
        })

        active_positions.append({
            'ticker': ticker,
            'entry_date': date,
            'hold_days': this_hold,
        })

    # Compute final equity from fixed-size trades
    total_pnl = sum(t['pnl'] for t in trades)
    final_equity = CAPITAL + total_pnl

    return trades, final_equity


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(trades):
    """Compute performance metrics from trade list."""
    if not trades or len(trades) == 0:
        return {
            'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0,
            'max_dd': 0, 'n_trades': 0, 'total_return': 0,
            'sharpe_bull': 0, 'sharpe_bear': 0, 'regime_gap': 0,
            'avg_quality': 0,
        }

    returns = [t['return'] for t in trades]
    pnls = [t['pnl'] for t in trades]
    qualities = [t['quality'] for t in trades]

    n = len(returns)
    avg_ret = np.mean(returns)
    std_ret = np.std(returns) if n > 1 else 1e-9

    # Sharpe (annualized assuming ~9 trades/year avg)
    trades_per_year = max(n / 4.5, 1)  # 4.5 years of OOT
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = [r for r in returns if r < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-9
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Profit Factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)

    # Win Rate
    winners = sum(1 for r in returns if r > 0)
    wr = winners / n if n > 0 else 0

    # Max Drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative + CAPITAL)
    drawdowns = (cumulative + CAPITAL) / running_max - 1
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    # Total return
    total_pnl = sum(pnls)
    total_return = total_pnl / CAPITAL

    # Regime split
    bull_rets = [t['return'] for t in trades if t.get('regime') == 'bull']
    bear_rets = [t['return'] for t in trades if t.get('regime') == 'bear']

    def regime_sharpe(rets):
        if len(rets) < 2:
            return 0
        m = np.mean(rets)
        s = np.std(rets)
        return (m / s) * np.sqrt(len(rets)) if s > 0 else 0

    sharpe_bull = regime_sharpe(bull_rets)
    sharpe_bear = regime_sharpe(bear_rets)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 0.01)

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'wr': round(wr, 3),
        'max_dd': round(max_dd, 3),
        'n_trades': n,
        'total_return': round(total_return, 3),
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'regime_gap': round(regime_gap, 3),
        'avg_quality': round(np.mean(qualities), 1) if qualities else 0,
        'n_bull': len(bull_rets),
        'n_bear': len(bear_rets),
    }


# ── Permutation Test ──────────────────────────────────────────────────────
def permutation_test(trades, prices, events, variant_name, hold_days, n_iter=PERM_ITERATIONS):
    """Shuffle entry dates and recompute Sharpe to get p-value."""
    if len(trades) < 5:
        return 1.0

    real_metrics = compute_metrics(trades)
    real_sharpe = real_metrics['sharpe']

    # Get all valid trading dates in OOT
    spy = prices.get('SPY')
    if spy is None:
        return 1.0
    valid_dates = spy.index[(spy.index >= OOT_START) & (spy.index <= OOT_END)]

    rng = np.random.default_rng(42)
    count_better = 0

    for i in range(n_iter):
        # Shuffle: for each trade, pick a random date and random ticker
        fake_trades = []
        for t in trades:
            rand_date = rng.choice(valid_dates)
            rand_ticker = rng.choice(UNIVERSE)
            if rand_ticker not in prices:
                continue
            entry_price = get_price_at(prices[rand_ticker], rand_date, 'Open', offset_days=1)
            if entry_price is None:
                continue
            exit_price = get_price_at(prices[rand_ticker], rand_date, 'Close', offset_days=hold_days)
            if exit_price is None:
                continue
            ret = (exit_price - entry_price) / entry_price
            fake_trades.append({
                'return': ret,
                'pnl': ret * (CAPITAL / MAX_POSITIONS),
                'quality': t['quality'],
                'regime': get_spy_regime(prices, rand_date),
            })

        if len(fake_trades) >= 5:
            fake_sharpe = compute_metrics(fake_trades)['sharpe']
            if fake_sharpe >= real_sharpe:
                count_better += 1

    p_value = (count_better + 1) / (n_iter + 1)
    return round(p_value, 4)


# ── 5-Gate Validation ─────────────────────────────────────────────────────
def five_gate_check(metrics, perm_p):
    """Apply 5-gate validation framework."""
    gates = {
        'sharpe_gt_05': metrics['sharpe'] > 0.5,
        'perm_p_lt_005': perm_p < 0.05,
        'regime_gap_lt_05': metrics['regime_gap'] < 0.5,
        'mdd_gt_neg50': metrics['max_dd'] > -0.50,
        'trades_gte_20': metrics['n_trades'] >= 20,
    }
    gates['all_pass'] = all(gates.values())
    return gates


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("EARNINGS QUALITY SCORE BACKTEST")
    print("=" * 70)

    prices = download_data()
    events = detect_earnings_events(prices)

    variants = {
        'A': {'name': 'Quality > 50', 'hold': 40},
        'B': {'name': 'Quality > 70', 'hold': 40},
        'C': {'name': 'Quality-Weighted Sizing', 'hold': 40},
        'D': {'name': 'Quality + Kill Switch', 'hold': 40},
        'E': {'name': 'Sector Leader Quality', 'hold': 60},
        'F': {'name': 'Adversarial Random Quality', 'hold': 40},
    }

    results = {}
    rng_f = np.random.default_rng(12345)

    for v_key, v_info in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {v_key}: {v_info['name']}")
        print(f"{'─' * 60}")

        rng_arg = rng_f if v_key == 'F' else None
        trades, final_equity = simulate_variant(
            v_key, prices, events, hold_days=v_info['hold'], rng=rng_arg
        )

        metrics = compute_metrics(trades)
        print(f"  Trades: {metrics['n_trades']}  |  WR: {metrics['wr']:.1%}  |  PF: {metrics['pf']:.2f}")
        print(f"  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}")
        print(f"  Total Return: {metrics['total_return']:.1%}  |  MaxDD: {metrics['max_dd']:.1%}")
        print(f"  Regime: Bull={metrics['sharpe_bull']:.3f} ({metrics['n_bull']}t) "
              f"Bear={metrics['sharpe_bear']:.3f} ({metrics['n_bear']}t)  Gap={metrics['regime_gap']:.3f}")
        print(f"  Avg Quality: {metrics['avg_quality']}")
        print(f"  Final Equity: ${final_equity:.2f} (from ${CAPITAL})")

        # Permutation test
        print(f"  Running permutation test ({PERM_ITERATIONS} iterations)...", end='', flush=True)
        perm_p = permutation_test(trades, prices, events, v_key, v_info['hold'])
        print(f" p={perm_p:.4f}")

        # 5-gate
        gates = five_gate_check(metrics, perm_p)
        gate_str = " | ".join(f"{'PASS' if v else 'FAIL'}" for v in [
            gates['sharpe_gt_05'], gates['perm_p_lt_005'],
            gates['regime_gap_lt_05'], gates['mdd_gt_neg50'],
            gates['trades_gte_20']
        ])
        print(f"  5-Gate: [{gate_str}]  ALL={'PASS' if gates['all_pass'] else 'FAIL'}")

        # Quality score distribution for this variant
        qual_dist = {}
        if trades:
            qs = [t['quality'] for t in trades]
            qual_dist = {
                'min': int(min(qs)),
                'max': int(max(qs)),
                'mean': round(np.mean(qs), 1),
                'median': round(np.median(qs), 1),
            }

        results[f'variant_{v_key}'] = {
            'name': v_info['name'],
            'hold_days': v_info['hold'],
            'metrics': metrics,
            'perm_p': perm_p,
            'gates': gates,
            'final_equity': round(final_equity, 2),
            'quality_distribution': qual_dist,
            'sample_trades': trades[:5] if trades else [],
        }

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — EARNINGS QUALITY SCORE BACKTEST")
    print("=" * 70)
    print(f"{'Variant':<35} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} "
          f"{'MDD':>7} {'N':>4} {'p':>7} {'5G':>5}")
    print("-" * 95)
    for v_key in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = results[f'variant_{v_key}']
        m = r['metrics']
        tag = 'PASS' if r['gates']['all_pass'] else 'FAIL'
        print(f"  {v_key}: {r['name']:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['pf']:>6.2f} {m['wr']:>5.1%} {m['max_dd']:>6.1%} {m['n_trades']:>4} "
              f"{r['perm_p']:>7.4f} {tag:>5}")

    # ── Quality Comparison ─────────────────────────────────────────────
    print("\n── Quality Filter Impact ──")
    for pair in [('A', 'F'), ('A', 'B')]:
        a_key, b_key = pair
        a = results[f'variant_{a_key}']['metrics']
        b = results[f'variant_{b_key}']['metrics']
        print(f"  {a_key} vs {b_key}: Sharpe {a['sharpe']:.3f} vs {b['sharpe']:.3f} "
              f"| WR {a['wr']:.1%} vs {b['wr']:.1%} "
              f"| N {a['n_trades']} vs {b['n_trades']}")

    # A vs F = quality filter vs random = is quality score adding value?
    a_sharpe = results['variant_A']['metrics']['sharpe']
    f_sharpe = results['variant_F']['metrics']['sharpe']
    if f_sharpe > 0 and a_sharpe > f_sharpe:
        quality_edge = (a_sharpe - f_sharpe) / f_sharpe
        print(f"\n  Quality Score Edge over Random: +{quality_edge:.0%} Sharpe improvement")
    elif a_sharpe > f_sharpe:
        print(f"\n  Quality Score Edge: A Sharpe {a_sharpe:.3f} > F (random) {f_sharpe:.3f}")
    else:
        print(f"\n  WARNING: Random quality (F) matched or beat real quality (A). "
              f"Quality score may not add value.")

    # Save results
    output_path = Path('/home/jupiter/Lvl3Quant/data/earnings_quality_score_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Make results JSON serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    serializable_results = json.loads(
        json.dumps(results, default=make_serializable)
    )

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
