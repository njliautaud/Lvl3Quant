#!/usr/bin/env python3
"""
Adversarial Validation: Strategy D — Post-Earnings Vol Crush Put Credit Spread
6-test adversarial battery.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
import warnings, sys, time
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = ['AAPL','MSFT','GOOGL','AMZN','META','NVDA','JPM','UNH','LLY','AVGO',
            'AMD','HD','ABBV','MRK','COST','CRM','NFLX','ADBE','PG','JNJ']
START = '2020-01-01'
END   = '2026-07-01'
GAP_THRESHOLD = 0.02        # 2% gap-up
PUT_OTM_PCT   = 0.05        # short put at price - 5%
SPREAD_WIDTH_PCT = 0.05     # long put another 5% lower
PREMIUM_RATE  = 0.02        # 2% of spread width
MAX_RISK      = 150.0       # per trade
HOLD_DAYS     = 30
MAX_CONCURRENT = 2
BASELINE_SHARPE = 1.98
BASELINE_TRADES = 53
np.random.seed(42)

# ── Data download ───────────────────────────────────────────────────────────
print("Downloading data for", len(UNIVERSE), "stocks...")
data = {}
for ticker in UNIVERSE:
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
            if len(df) > 100:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                break
        except Exception:
            time.sleep(1)
    if ticker not in data:
        print(f"  WARNING: Failed to download {ticker}")

print(f"Downloaded {len(data)}/{len(UNIVERSE)} stocks")

# ── Detect earnings gap-up days ─────────────────────────────────────────────
def detect_earnings_gaps(prices, gap_thresh=GAP_THRESHOLD, direction='up'):
    """
    Detect likely earnings gap days: large overnight gaps that occur roughly quarterly.
    direction: 'up' for gap-up, 'down' for gap-down
    """
    df = prices.copy()
    df['prev_close'] = df['Close'].shift(1)
    df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close']

    if direction == 'up':
        big_gaps = df[df['gap_pct'] > gap_thresh].copy()
    else:
        big_gaps = df[df['gap_pct'] < -gap_thresh].copy()

    if len(big_gaps) == 0:
        return []

    # Filter to roughly quarterly gaps (earnings proxy):
    # Keep only gaps that are at least 45 days apart (earnings are ~quarterly)
    # This filters out non-earnings gap days
    earnings_gaps = []
    big_gaps = big_gaps.sort_index()
    last_date = None
    for date in big_gaps.index:
        if last_date is None or (date - last_date).days >= 45:
            earnings_gaps.append(date)
            last_date = date

    return earnings_gaps


def run_strategy(data, gap_thresh=GAP_THRESHOLD, put_otm=PUT_OTM_PCT,
                 spread_width=SPREAD_WIDTH_PCT, premium_rate=PREMIUM_RATE,
                 hold_days=HOLD_DAYS, direction='up', max_concurrent=MAX_CONCURRENT,
                 stock_subset=None):
    """
    Run the post-earnings vol crush put credit spread strategy.
    Returns list of trade dicts.
    """
    trades = []
    active_trades = []  # list of (exit_date,)

    universe = stock_subset if stock_subset else list(data.keys())

    # Collect all signals
    signals = []
    for ticker in universe:
        if ticker not in data:
            continue
        gaps = detect_earnings_gaps(data[ticker], gap_thresh, direction)
        for gap_date in gaps:
            # Entry is NEXT trading day after gap
            df = data[ticker]
            future = df[df.index > gap_date]
            if len(future) < 2:
                continue
            entry_date = future.index[0]
            signals.append((entry_date, ticker))

    signals.sort(key=lambda x: x[0])

    for entry_date, ticker in signals:
        # Check concurrent limit
        active_trades = [t for t in active_trades if t > entry_date]
        if len(active_trades) >= max_concurrent:
            continue

        df = data[ticker]
        entry_idx = df.index.get_loc(entry_date)

        if entry_idx + hold_days >= len(df):
            continue

        entry_price = float(df['Close'].iloc[entry_idx])
        short_put_strike = entry_price * (1 - put_otm)
        long_put_strike = entry_price * (1 - put_otm - spread_width)
        spread_width_dollar = short_put_strike - long_put_strike
        premium = spread_width_dollar * premium_rate
        max_risk_trade = spread_width_dollar - premium

        # Position sizing
        if max_risk_trade <= 0:
            continue
        contracts = max(1, int(MAX_RISK / (max_risk_trade * 100)))

        # Check outcome: does stock stay above short put strike for hold_days?
        exit_idx = min(entry_idx + hold_days, len(df) - 1)
        exit_date = df.index[exit_idx]

        # Check minimum price during hold period
        hold_slice = df.iloc[entry_idx:exit_idx+1]
        min_price = float(hold_slice['Low'].min())

        if min_price > short_put_strike:
            # Win: keep premium
            pnl = premium * contracts * 100
            outcome = 'win'
        elif min_price > long_put_strike:
            # Partial loss: stock between strikes at some point
            # Use close at expiry for settlement
            exit_price = float(df['Close'].iloc[exit_idx])
            if exit_price >= short_put_strike:
                pnl = premium * contracts * 100
                outcome = 'win'
            else:
                intrinsic = short_put_strike - exit_price
                pnl = (premium - intrinsic) * contracts * 100
                outcome = 'loss' if pnl < 0 else 'win'
        else:
            # Max loss: stock below long put
            pnl = -max_risk_trade * contracts * 100
            outcome = 'loss'

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'ticker': ticker,
            'entry_price': entry_price,
            'short_strike': short_put_strike,
            'long_strike': long_put_strike,
            'premium': premium * contracts * 100,
            'pnl': pnl,
            'outcome': outcome,
            'contracts': contracts
        })

        active_trades.append(exit_date)

    return trades


def compute_metrics(trades):
    """Compute Sharpe, WR, total PnL from trade list."""
    if not trades:
        return {'sharpe': 0, 'wr': 0, 'total_pnl': 0, 'n_trades': 0, 'pf': 0}

    pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses_vals = [-p for p in pnls if p < 0]
    gains_vals = [p for p in pnls if p > 0]

    wr = wins / len(pnls) if pnls else 0
    total = sum(pnls)

    # Sharpe: annualize assuming trades spread over period
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(len(pnls))
    else:
        sharpe = 0.0

    pf = sum(gains_vals) / sum(losses_vals) if losses_vals and sum(losses_vals) > 0 else 99.9

    return {
        'sharpe': sharpe,
        'wr': wr,
        'total_pnl': total,
        'n_trades': len(pnls),
        'pf': pf
    }


# ── RUN BASELINE ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("RUNNING BASELINE STRATEGY")
print("="*70)
baseline_trades = run_strategy(data)
baseline = compute_metrics(baseline_trades)
print(f"  Trades: {baseline['n_trades']}")
print(f"  Win Rate: {baseline['wr']:.1%}")
print(f"  Total PnL: ${baseline['total_pnl']:,.0f}")
print(f"  Sharpe: {baseline['sharpe']:.2f}")
print(f"  Profit Factor: {baseline['pf']:.2f}")

results = {}

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: RE-IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 1: RE-IMPLEMENTATION")
print("="*70)
# Already run as baseline above
t1_sharpe = baseline['sharpe']
t1_threshold = 0.70 * BASELINE_SHARPE  # 1.39
t1_pass = t1_sharpe > t1_threshold
print(f"  Re-implemented Sharpe: {t1_sharpe:.2f}")
print(f"  Threshold (70% of {BASELINE_SHARPE}): {t1_threshold:.2f}")
print(f"  RESULT: {'PASS ✓' if t1_pass else 'FAIL ✗'}")
results['Test 1: Re-Implementation'] = t1_pass

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: INVERSE SIGNAL
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 2: INVERSE SIGNAL (gap-DOWN earnings)")
print("="*70)
inverse_trades = run_strategy(data, direction='down')
inverse = compute_metrics(inverse_trades)
print(f"  Trades: {inverse['n_trades']}")
print(f"  Win Rate: {inverse['wr']:.1%}")
print(f"  Total PnL: ${inverse['total_pnl']:,.0f}")
print(f"  Sharpe: {inverse['sharpe']:.2f}")
t2_threshold = 0.50 * BASELINE_SHARPE  # 0.99
t2_pass = inverse['sharpe'] < t2_threshold
print(f"  Threshold: inverse Sharpe < {t2_threshold:.2f}")
print(f"  RESULT: {'PASS ✓' if t2_pass else 'FAIL ✗'}")
results['Test 2: Inverse Signal'] = t2_pass

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: RANDOM TIMING (300 permutations)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 3: RANDOM TIMING (300 permutations)")
print("="*70)
n_perms = 300
n_trades_target = baseline['n_trades'] if baseline['n_trades'] > 0 else BASELINE_TRADES

# Build universe of all valid trading dates per stock
all_dates = {}
for ticker in data:
    df = data[ticker]
    valid = df.index[:-HOLD_DAYS-1]  # need hold_days after entry
    all_dates[ticker] = valid

tickers_list = list(data.keys())
random_sharpes = []

for perm in range(n_perms):
    random_trades = []
    for _ in range(n_trades_target):
        ticker = np.random.choice(tickers_list)
        if len(all_dates[ticker]) == 0:
            continue
        idx = np.random.randint(0, len(all_dates[ticker]))
        entry_date = all_dates[ticker][idx]

        df = data[ticker]
        entry_loc = df.index.get_loc(entry_date)
        entry_price = float(df['Close'].iloc[entry_loc])

        short_put_strike = entry_price * (1 - PUT_OTM_PCT)
        long_put_strike = entry_price * (1 - PUT_OTM_PCT - SPREAD_WIDTH_PCT)
        spread_w = short_put_strike - long_put_strike
        prem = spread_w * PREMIUM_RATE
        max_risk_t = spread_w - prem

        if max_risk_t <= 0:
            continue
        contracts = max(1, int(MAX_RISK / (max_risk_t * 100)))

        exit_loc = min(entry_loc + HOLD_DAYS, len(df) - 1)
        hold_slice = df.iloc[entry_loc:exit_loc+1]
        min_price = float(hold_slice['Low'].min())
        exit_price = float(df['Close'].iloc[exit_loc])

        if exit_price >= short_put_strike:
            pnl = prem * contracts * 100
        elif exit_price > long_put_strike:
            intrinsic = short_put_strike - exit_price
            pnl = (prem - intrinsic) * contracts * 100
        else:
            pnl = -max_risk_t * contracts * 100

        random_trades.append({'pnl': pnl, 'outcome': 'win' if pnl > 0 else 'loss'})

    rm = compute_metrics(random_trades)
    random_sharpes.append(rm['sharpe'])

p_value = np.mean([s >= t1_sharpe for s in random_sharpes])
print(f"  Random Sharpe distribution: mean={np.mean(random_sharpes):.2f}, "
      f"median={np.median(random_sharpes):.2f}, std={np.std(random_sharpes):.2f}")
print(f"  Original Sharpe: {t1_sharpe:.2f}")
print(f"  p-value (fraction >= original): {p_value:.4f}")
t3_pass = p_value < 0.05
print(f"  Threshold: p < 0.05")
print(f"  RESULT: {'PASS ✓' if t3_pass else 'FAIL ✗'}")
results['Test 3: Random Timing'] = t3_pass

# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: SUB-PERIOD STABILITY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 4: SUB-PERIOD STABILITY (4 periods)")
print("="*70)
if baseline_trades:
    all_entry_dates = sorted([t['entry_date'] for t in baseline_trades])
    min_date = all_entry_dates[0]
    max_date = all_entry_dates[-1]
    total_days = (max_date - min_date).days
    period_len = total_days // 4

    period_sharpes = []
    for i in range(4):
        p_start = min_date + timedelta(days=i * period_len)
        p_end = min_date + timedelta(days=(i + 1) * period_len) if i < 3 else max_date + timedelta(days=1)

        period_trades = [t for t in baseline_trades if p_start <= t['entry_date'] < p_end]
        pm = compute_metrics(period_trades)
        period_sharpes.append(pm['sharpe'])
        print(f"  Period {i+1} ({p_start.strftime('%Y-%m')}-{p_end.strftime('%Y-%m')}): "
              f"Sharpe={pm['sharpe']:.2f}, Trades={pm['n_trades']}, WR={pm['wr']:.1%}, PnL=${pm['total_pnl']:,.0f}")

    positive_periods = sum(1 for s in period_sharpes if s > 0)
    min_sharpe = min(period_sharpes)
    t4_pass = positive_periods >= 3 and min_sharpe >= -0.50
    print(f"  Positive periods: {positive_periods}/4, Min Sharpe: {min_sharpe:.2f}")
    print(f"  RESULT: {'PASS ✓' if t4_pass else 'FAIL ✗'}")
else:
    t4_pass = False
    print("  No trades to analyze. FAIL")
results['Test 4: Sub-Period Stability'] = t4_pass

# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: TOP-3 STOCK REMOVAL
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 5: TOP-3 STOCK REMOVAL")
print("="*70)
if baseline_trades:
    # Find top 3 contributors
    ticker_pnl = {}
    for t in baseline_trades:
        ticker_pnl[t['ticker']] = ticker_pnl.get(t['ticker'], 0) + t['pnl']

    sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
    top3 = [t[0] for t in sorted_tickers[:3]]
    print(f"  Top 3 contributors: {top3}")
    for t, pnl in sorted_tickers[:3]:
        print(f"    {t}: ${pnl:,.0f}")

    remaining = [s for s in UNIVERSE if s not in top3]
    reduced_trades = run_strategy(data, stock_subset=remaining)
    reduced = compute_metrics(reduced_trades)

    print(f"  Reduced Sharpe: {reduced['sharpe']:.2f} (Trades: {reduced['n_trades']})")
    t5_threshold = 0.50 * BASELINE_SHARPE  # 0.99
    t5_pass = reduced['sharpe'] > t5_threshold
    print(f"  Threshold: > {t5_threshold:.2f}")
    print(f"  RESULT: {'PASS ✓' if t5_pass else 'FAIL ✗'}")
else:
    t5_pass = False
    print("  No trades. FAIL")
results['Test 5: Top-3 Stock Removal'] = t5_pass

# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: PARAMETER SENSITIVITY (150 combos)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 6: PARAMETER SENSITIVITY (150 combos)")
print("="*70)

gap_thresholds = [0.01, 0.015, 0.02, 0.03, 0.04]
put_otm_values = [0.03, 0.04, 0.05, 0.06, 0.08]
spread_widths  = [0.03, 0.04, 0.05, 0.06, 0.08]
premium_rates  = [0.01, 0.02, 0.03]
hold_days_vals = [14, 21, 30, 37, 45]

# Generate 150 random combos from the grid
combos = []
for _ in range(150):
    combos.append({
        'gap': np.random.choice(gap_thresholds),
        'otm': np.random.choice(put_otm_values),
        'sw':  np.random.choice(spread_widths),
        'pr':  np.random.choice(premium_rates),
        'hd':  int(np.random.choice(hold_days_vals))
    })

combo_sharpes = []
for i, c in enumerate(combos):
    try:
        ct = run_strategy(data, gap_thresh=c['gap'], put_otm=c['otm'],
                          spread_width=c['sw'], premium_rate=c['pr'],
                          hold_days=c['hd'])
        cm = compute_metrics(ct)
        combo_sharpes.append(cm['sharpe'])
    except Exception:
        combo_sharpes.append(0)

    if (i+1) % 30 == 0:
        print(f"  ... {i+1}/150 combos evaluated")

above_threshold = sum(1 for s in combo_sharpes if s > 0.30)
pct_above = above_threshold / len(combo_sharpes)
print(f"  Combos with Sharpe > 0.30: {above_threshold}/{len(combo_sharpes)} ({pct_above:.1%})")
print(f"  Sharpe distribution: mean={np.mean(combo_sharpes):.2f}, "
      f"median={np.median(combo_sharpes):.2f}, min={np.min(combo_sharpes):.2f}, max={np.max(combo_sharpes):.2f}")
t6_pass = pct_above > 0.50
print(f"  Threshold: >50% above 0.30")
print(f"  RESULT: {'PASS ✓' if t6_pass else 'FAIL ✗'}")
results['Test 6: Parameter Sensitivity'] = t6_pass

# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ADVERSARIAL VALIDATION SUMMARY — Strategy D: Post-Earnings Vol Crush")
print("="*70)
print(f"\nBaseline: Sharpe={baseline['sharpe']:.2f}, WR={baseline['wr']:.1%}, "
      f"Trades={baseline['n_trades']}, PnL=${baseline['total_pnl']:,.0f}")
print()

passes = 0
for test_name, passed in results.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {status}  {test_name}")
    if passed:
        passes += 1

print(f"\nOverall: {passes}/6 tests passed")
if passes >= 5:
    print("VERDICT: STRONG PASS — Strategy D is robust")
elif passes >= 4:
    print("VERDICT: CONDITIONAL PASS — Strategy D shows promise but has weaknesses")
elif passes >= 3:
    print("VERDICT: MARGINAL — Strategy D needs improvement")
else:
    print("VERDICT: FAIL — Strategy D does not survive adversarial validation")
