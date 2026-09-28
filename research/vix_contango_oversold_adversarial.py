#!/usr/bin/env python3
"""
Adversarial Validation Battery for VIX Contango + Sector Oversold Signal
=========================================================================
6 tests to stress-test the signal before promotion.

1. Re-implementation (independent rewrite, same logic)
2. Inverse signal (flip conditions)
3. Random timing (shuffle entry dates, 1000x)
4. Cost sensitivity (transaction cost sweep)
5. Sub-period stability (4 equal time periods)
6. Parameter robustness (RSI x contango threshold grid)

Uses SAME data pipeline and walk-forward approach as the original backtest.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime
import sys
import time

# ============================================================
# CONFIG — identical to original backtest
# ============================================================
SECTOR_ETFS = ['XLE', 'XLU', 'XLP', 'XLK', 'XLY', 'XLF', 'XLRE', 'XLV', 'XLI', 'XLB', 'XLC']
BENCHMARK = 'SPY'
VIX_TICKER = '^VIX'
VIX3M_TICKER = '^VIX3M'
HOLD_DAYS = 5
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
TRAIN_WINDOW = 60
START_DATE = '2018-01-01'
END_DATE = '2026-08-15'
N_PERMUTATIONS = 1000

# ============================================================
# SHARED DATA FUNCTIONS (identical to original)
# ============================================================

def download_data():
    """Download all required data."""
    print("Downloading data...")
    all_tickers = SECTOR_ETFS + [BENCHMARK, VIX_TICKER, VIX3M_TICKER]
    data = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
        except Exception as e:
            print(f"  {ticker}: DOWNLOAD FAILED - {e}")
    return data


def compute_rsi(prices, period=14):
    """Compute RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_contango_slope(data):
    """Compute VIX term structure slope."""
    vix = data[VIX_TICKER]['Close']
    if VIX3M_TICKER in data:
        vix3m = data[VIX3M_TICKER]['Close']
        common_idx = vix.index.intersection(vix3m.index)
        vix = vix.loc[common_idx]
        vix3m = vix3m.loc[common_idx]
        contango = (vix3m - vix) / vix
    else:
        vix_ma20 = vix.rolling(20).mean()
        contango = (vix_ma20 - vix) / vix
    return contango


def build_signals_df(data, contango_slope, rsi_os=RSI_OVERSOLD, rsi_ob=RSI_OVERBOUGHT):
    """Build full observation DataFrame with forward returns."""
    spy_close = data[BENCHMARK]['Close']
    spy_ret = spy_close.pct_change()
    records = []
    for ticker in SECTOR_ETFS:
        if ticker not in data:
            continue
        close = data[ticker]['Close']
        rsi = compute_rsi(close, RSI_PERIOD)
        fwd_ret = close.pct_change(HOLD_DAYS).shift(-HOLD_DAYS)
        common_idx = close.index.intersection(contango_slope.index).intersection(rsi.dropna().index)
        for dt in common_idx:
            if dt not in fwd_ret.index or pd.isna(fwd_ret.loc[dt]):
                continue
            if pd.isna(rsi.loc[dt]) or pd.isna(contango_slope.loc[dt]):
                continue
            regime = 'green' if (dt in spy_ret.index and spy_ret.loc[dt] > 0) else 'red'
            records.append({
                'date': dt, 'ticker': ticker, 'rsi': rsi.loc[dt],
                'contango': contango_slope.loc[dt], 'fwd_return': fwd_ret.loc[dt],
                'regime': regime
            })
    return pd.DataFrame(records)


def walk_forward_backtest(signals_df, rsi_os=RSI_OVERSOLD, rsi_ob=RSI_OVERBOUGHT):
    """Walk-forward sliding window backtest — identical to original."""
    signals_df = signals_df.sort_values('date').reset_index(drop=True)
    dates = sorted(signals_df['date'].unique())
    if len(dates) < TRAIN_WINDOW + 20:
        return pd.DataFrame()

    all_trades = []
    contango_thresholds = np.arange(0.02, 0.20, 0.01)

    for i in range(TRAIN_WINDOW, len(dates)):
        test_date = dates[i]
        train_start = dates[max(0, i - TRAIN_WINDOW)]
        train_end = dates[i - 1]
        train = signals_df[(signals_df['date'] >= train_start) & (signals_df['date'] <= train_end)]
        test = signals_df[signals_df['date'] == test_date]
        if len(train) < 50 or len(test) == 0:
            continue

        best_sharpe = -999
        best_thresh = 0.05
        for thresh in contango_thresholds:
            longs = train[(train['contango'] > thresh) & (train['rsi'] < rsi_os)]
            if len(longs) < 5:
                continue
            ret_mean = longs['fwd_return'].mean()
            ret_std = longs['fwd_return'].std()
            sharpe = (ret_mean / ret_std * np.sqrt(252 / HOLD_DAYS)) if ret_std > 0 else 0
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_thresh = thresh

        long_signals = test[(test['contango'] > best_thresh) & (test['rsi'] < rsi_os)]
        short_signals = test[(test['contango'] < -0.02) & (test['rsi'] > rsi_ob)]

        for _, row in long_signals.iterrows():
            all_trades.append({
                'date': row['date'], 'ticker': row['ticker'], 'direction': 'LONG',
                'rsi': row['rsi'], 'contango': row['contango'],
                'return': row['fwd_return'], 'regime': row['regime'],
                'threshold_used': best_thresh
            })
        for _, row in short_signals.iterrows():
            all_trades.append({
                'date': row['date'], 'ticker': row['ticker'], 'direction': 'SHORT',
                'rsi': row['rsi'], 'contango': row['contango'],
                'return': -row['fwd_return'], 'regime': row['regime'],
                'threshold_used': best_thresh
            })

    return pd.DataFrame(all_trades)


def compute_sharpe(returns):
    """Compute annualized Sharpe from a series of 5-day trade returns."""
    if len(returns) < 5:
        return 0.0
    mean_ret = returns.mean()
    std_ret = returns.std()
    if std_ret == 0:
        return 0.0
    trades_per_year = 252 / HOLD_DAYS
    return mean_ret / std_ret * np.sqrt(trades_per_year)


def compute_full_metrics(returns):
    """Full metrics dict for a return series."""
    if len(returns) < 3:
        return {'n_trades': len(returns), 'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0}
    returns = returns.reset_index(drop=True)
    mean_ret = returns.mean()
    std_ret = returns.std()
    tpy = 252 / HOLD_DAYS
    ann_ret = mean_ret * tpy
    ann_std = std_ret * np.sqrt(tpy) if std_ret > 0 else 1e-6
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    downside = returns[returns < 0]
    ds_std = downside.std() * np.sqrt(tpy) if len(downside) > 1 else 1e-6
    sortino = ann_ret / ds_std if ds_std > 0 else 0
    wr = (returns > 0).mean()
    gp = returns[returns > 0].sum()
    gl = abs(returns[returns < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    return {'n_trades': len(returns), 'sharpe': sharpe, 'sortino': sortino, 'win_rate': wr, 'profit_factor': pf}


# ============================================================
# TEST 1: RE-IMPLEMENTATION
# ============================================================
def test_reimplementation(data, contango_slope):
    """
    Rewrite the signal logic from scratch using vectorized pandas.
    Same logic: walk-forward 60-day sliding, optimize contango threshold,
    long when contango > thresh AND RSI < 30. Compare Sharpe.
    """
    print("\n" + "=" * 70)
    print("TEST 1: RE-IMPLEMENTATION (independent rewrite)")
    print("=" * 70)

    # Build a single wide DataFrame — completely different implementation approach
    # Use vectorized operations instead of row-by-row iteration
    spy_close = data[BENCHMARK]['Close']
    spy_ret = spy_close.pct_change()

    # Build per-ticker RSI and forward returns as columns in a single DF
    all_sector_data = []
    for ticker in SECTOR_ETFS:
        if ticker not in data:
            continue
        close = data[ticker]['Close']
        # RSI via vectorized ewm (different method than SMA-based in original)
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        # Use SMA like original for consistency
        avg_up = up.rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
        avg_down = down.rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
        rs_val = avg_up / avg_down
        rsi_val = 100.0 - 100.0 / (1.0 + rs_val)
        fwd = close.pct_change(HOLD_DAYS).shift(-HOLD_DAYS)

        ticker_df = pd.DataFrame({
            'date': close.index, 'ticker': ticker,
            'rsi': rsi_val.values, 'contango': np.nan,
            'fwd_return': fwd.values
        }).dropna(subset=['rsi', 'fwd_return'])

        # Merge contango
        ticker_df = ticker_df.set_index('date')
        ticker_df['contango'] = contango_slope
        ticker_df = ticker_df.dropna(subset=['contango']).reset_index()

        # Regime
        ticker_df = ticker_df.set_index('date')
        ticker_df['regime'] = np.where(spy_ret.reindex(ticker_df.index).fillna(0) > 0, 'green', 'red')
        ticker_df = ticker_df.reset_index()

        all_sector_data.append(ticker_df)

    reimpl_signals = pd.concat(all_sector_data, ignore_index=True).sort_values('date')
    dates = sorted(reimpl_signals['date'].unique())

    # Walk-forward: vectorized threshold search
    reimpl_trades = []
    thresholds = np.arange(0.02, 0.20, 0.01)

    for i in range(TRAIN_WINDOW, len(dates)):
        t_date = dates[i]
        t_start = dates[max(0, i - TRAIN_WINDOW)]
        t_end = dates[i - 1]

        train_mask = (reimpl_signals['date'] >= t_start) & (reimpl_signals['date'] <= t_end)
        train_chunk = reimpl_signals[train_mask]
        test_chunk = reimpl_signals[reimpl_signals['date'] == t_date]

        if len(train_chunk) < 50 or len(test_chunk) == 0:
            continue

        # Vectorized threshold search
        top_sharpe = -999
        top_thresh = 0.05
        for th in thresholds:
            cond = (train_chunk['contango'] > th) & (train_chunk['rsi'] < RSI_OVERSOLD)
            subset_ret = train_chunk.loc[cond, 'fwd_return']
            if len(subset_ret) < 5:
                continue
            mu = subset_ret.mean()
            sigma = subset_ret.std()
            sh = (mu / sigma * np.sqrt(252 / HOLD_DAYS)) if sigma > 0 else 0
            if sh > top_sharpe:
                top_sharpe = sh
                top_thresh = th

        # Apply
        long_cond = (test_chunk['contango'] > top_thresh) & (test_chunk['rsi'] < RSI_OVERSOLD)
        for _, r in test_chunk[long_cond].iterrows():
            reimpl_trades.append({'return': r['fwd_return'], 'direction': 'LONG'})

        short_cond = (test_chunk['contango'] < -0.02) & (test_chunk['rsi'] > RSI_OVERBOUGHT)
        for _, r in test_chunk[short_cond].iterrows():
            reimpl_trades.append({'return': -r['fwd_return'], 'direction': 'SHORT'})

    reimpl_df = pd.DataFrame(reimpl_trades)
    reimpl_sharpe = compute_sharpe(reimpl_df['return']) if len(reimpl_df) > 0 else 0

    # Also run original for comparison
    orig_signals = build_signals_df(data, contango_slope)
    orig_trades = walk_forward_backtest(orig_signals)
    orig_sharpe = compute_sharpe(orig_trades['return']) if len(orig_trades) > 0 else 0

    ratio = reimpl_sharpe / orig_sharpe if orig_sharpe != 0 else 0
    passed = ratio >= 0.50

    print(f"\n  Original Sharpe:        {orig_sharpe:.4f} ({len(orig_trades)} trades)")
    print(f"  Re-implementation Sharpe: {reimpl_sharpe:.4f} ({len(reimpl_df)} trades)")
    print(f"  Ratio (reimpl/orig):    {ratio:.4f}")
    print(f"  Threshold:              >= 0.50")
    print(f"\n  >>> TEST 1: {'PASS' if passed else 'FAIL'} <<<")

    return passed, orig_sharpe, orig_trades, orig_signals


# ============================================================
# TEST 2: INVERSE SIGNAL
# ============================================================
def test_inverse_signal(data, contango_slope, orig_sharpe):
    """
    Flip the entry: buy when VIX in backwardation AND RSI > 70.
    If this also works well, we have no directional edge.
    """
    print("\n" + "=" * 70)
    print("TEST 2: INVERSE SIGNAL (backwardation + overbought)")
    print("=" * 70)

    signals_df = build_signals_df(data, contango_slope)
    signals_df = signals_df.sort_values('date').reset_index(drop=True)
    dates = sorted(signals_df['date'].unique())

    inv_trades = []
    # For inverse: we BUY when backwardation (contango < -thresh) AND RSI > 70
    # This is the opposite thesis — should NOT work if original has real edge
    contango_thresholds = np.arange(0.02, 0.20, 0.01)

    for i in range(TRAIN_WINDOW, len(dates)):
        test_date = dates[i]
        train_start = dates[max(0, i - TRAIN_WINDOW)]
        train_end = dates[i - 1]
        train = signals_df[(signals_df['date'] >= train_start) & (signals_df['date'] <= train_end)]
        test = signals_df[signals_df['date'] == test_date]
        if len(train) < 50 or len(test) == 0:
            continue

        # Inverse: optimize threshold for BUYING when backwardation + overbought
        best_sharpe = -999
        best_thresh = 0.05
        for thresh in contango_thresholds:
            inv_longs = train[(train['contango'] < -thresh) & (train['rsi'] > RSI_OVERBOUGHT)]
            if len(inv_longs) < 5:
                continue
            ret_mean = inv_longs['fwd_return'].mean()
            ret_std = inv_longs['fwd_return'].std()
            sharpe = (ret_mean / ret_std * np.sqrt(252 / HOLD_DAYS)) if ret_std > 0 else 0
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_thresh = thresh

        # Apply inverse on test day
        inv_long = test[(test['contango'] < -best_thresh) & (test['rsi'] > RSI_OVERBOUGHT)]
        for _, row in inv_long.iterrows():
            inv_trades.append({'return': row['fwd_return'], 'direction': 'LONG'})

        # Also inverse short: contango + oversold => sell (opposite of original long)
        inv_short = test[(test['contango'] > 0.05) & (test['rsi'] < RSI_OVERSOLD)]
        for _, row in inv_short.iterrows():
            inv_trades.append({'return': -row['fwd_return'], 'direction': 'SHORT'})

    inv_df = pd.DataFrame(inv_trades)
    inv_sharpe = compute_sharpe(inv_df['return']) if len(inv_df) > 0 else 0

    ratio = inv_sharpe / orig_sharpe if orig_sharpe != 0 else 0
    passed = ratio < 0.50  # Inverse should be WORSE

    print(f"\n  Original Sharpe:  {orig_sharpe:.4f}")
    print(f"  Inverse Sharpe:   {inv_sharpe:.4f} ({len(inv_df)} trades)")
    print(f"  Ratio (inv/orig): {ratio:.4f}")
    print(f"  Threshold:        < 0.50 (inverse must be much worse)")
    print(f"\n  >>> TEST 2: {'PASS' if passed else 'FAIL'} <<<")

    return passed


# ============================================================
# TEST 3: RANDOM TIMING
# ============================================================
def test_random_timing(orig_trades, signals_df):
    """
    Shuffle entry dates 1000 times. Real Sharpe must be >= 95th percentile.
    """
    print("\n" + "=" * 70)
    print("TEST 3: RANDOM TIMING (1000 shuffles)")
    print("=" * 70)

    orig_sharpe = compute_sharpe(orig_trades['return'])
    n_trades = len(orig_trades)
    all_returns = signals_df['fwd_return'].dropna().values

    rng = np.random.RandomState(42)
    random_sharpes = []

    for perm_i in range(N_PERMUTATIONS):
        # Randomly sample same number of returns from the full universe
        idx = rng.choice(len(all_returns), size=n_trades, replace=True)
        sampled_returns = pd.Series(all_returns[idx])
        random_sharpes.append(compute_sharpe(sampled_returns))

    random_sharpes = np.array(random_sharpes)
    percentile = np.mean(random_sharpes < orig_sharpe) * 100

    passed = percentile >= 95.0

    print(f"\n  Original Sharpe:     {orig_sharpe:.4f}")
    print(f"  Random mean Sharpe:  {random_sharpes.mean():.4f}")
    print(f"  Random median Sharpe:{np.median(random_sharpes):.4f}")
    print(f"  Random 95th pctile:  {np.percentile(random_sharpes, 95):.4f}")
    print(f"  Random 99th pctile:  {np.percentile(random_sharpes, 99):.4f}")
    print(f"  Real Sharpe percentile: {percentile:.1f}th")
    print(f"  Threshold:           >= 95th percentile")
    print(f"\n  >>> TEST 3: {'PASS' if passed else 'FAIL'} <<<")

    return passed


# ============================================================
# TEST 4: COST SENSITIVITY
# ============================================================
def test_cost_sensitivity(orig_trades):
    """
    Add transaction costs and check if Sharpe survives.
    Costs: 0.05%, 0.10%, 0.15%, 0.20% per trade (round-trip).
    FAIL if Sharpe < 0.5 at 0.10% cost.
    """
    print("\n" + "=" * 70)
    print("TEST 4: COST SENSITIVITY")
    print("=" * 70)

    costs = [0.0, 0.0005, 0.0010, 0.0015, 0.0020]
    cost_labels = ['0.00%', '0.05%', '0.10%', '0.15%', '0.20%']

    base_returns = orig_trades['return'].values
    sharpes_at_cost = {}

    print(f"\n  {'Cost':>8s}  {'Sharpe':>8s}  {'Mean Ret':>10s}  {'WR':>6s}")
    print(f"  {'----':>8s}  {'------':>8s}  {'--------':>10s}  {'--':>6s}")

    for cost, label in zip(costs, cost_labels):
        # Subtract cost from each trade return (entry + exit = round trip)
        adj_returns = pd.Series(base_returns - cost)
        sharpe = compute_sharpe(adj_returns)
        mean_r = adj_returns.mean()
        wr = (adj_returns > 0).mean()
        sharpes_at_cost[label] = sharpe
        print(f"  {label:>8s}  {sharpe:>8.4f}  {mean_r*100:>9.4f}%  {wr*100:>5.1f}%")

    sharpe_at_10bps = sharpes_at_cost['0.10%']
    passed = sharpe_at_10bps >= 0.5

    print(f"\n  Sharpe at 0.10% cost: {sharpe_at_10bps:.4f}")
    print(f"  Threshold:            >= 0.50")
    print(f"\n  >>> TEST 4: {'PASS' if passed else 'FAIL'} <<<")

    return passed


# ============================================================
# TEST 5: SUB-PERIOD STABILITY
# ============================================================
def test_subperiod_stability(orig_trades):
    """
    Split into 4 equal time periods. ALL must have positive Sharpe.
    Best/worst ratio > 5 => WARNING.
    """
    print("\n" + "=" * 70)
    print("TEST 5: SUB-PERIOD STABILITY (4 periods)")
    print("=" * 70)

    trades = orig_trades.sort_values('date').reset_index(drop=True)
    n = len(trades)
    chunk_size = n // 4

    period_sharpes = []
    any_negative = False
    warning = False

    print(f"\n  {'Period':>10s}  {'Dates':>30s}  {'Trades':>7s}  {'Sharpe':>8s}  {'WR':>6s}")
    print(f"  {'------':>10s}  {'-----':>30s}  {'------':>7s}  {'------':>8s}  {'--':>6s}")

    for p in range(4):
        start_idx = p * chunk_size
        end_idx = (p + 1) * chunk_size if p < 3 else n
        chunk = trades.iloc[start_idx:end_idx]

        if len(chunk) < 3:
            sharpe = 0
        else:
            sharpe = compute_sharpe(chunk['return'])

        wr = (chunk['return'] > 0).mean() if len(chunk) > 0 else 0
        date_range = f"{chunk['date'].iloc[0].strftime('%Y-%m-%d')} to {chunk['date'].iloc[-1].strftime('%Y-%m-%d')}"
        period_label = f"Q{p+1}"

        print(f"  {period_label:>10s}  {date_range:>30s}  {len(chunk):>7d}  {sharpe:>8.4f}  {wr*100:>5.1f}%")

        period_sharpes.append(sharpe)
        if sharpe < 0:
            any_negative = True

    # Best/worst ratio
    pos_sharpes = [s for s in period_sharpes if s > 0]
    neg_sharpes = [s for s in period_sharpes if s <= 0]

    if len(pos_sharpes) > 0 and not any_negative:
        bw_ratio = max(period_sharpes) / min(period_sharpes) if min(period_sharpes) > 0 else float('inf')
    else:
        bw_ratio = float('inf')

    if bw_ratio > 5:
        warning = True

    passed = not any_negative

    print(f"\n  Any negative Sharpe period: {'YES' if any_negative else 'NO'}")
    print(f"  Best/Worst ratio:          {bw_ratio:.2f}")
    if warning:
        print(f"  WARNING: Best/worst ratio > 5 — performance concentration risk")
    print(f"\n  >>> TEST 5: {'PASS' if passed else 'FAIL'} <<<")

    return passed, warning


# ============================================================
# TEST 6: PARAMETER ROBUSTNESS
# ============================================================
def test_parameter_robustness(data, contango_slope):
    """
    Grid of RSI thresholds x contango thresholds.
    >= 60% of combos must produce Sharpe > 0.3.
    """
    print("\n" + "=" * 70)
    print("TEST 6: PARAMETER ROBUSTNESS (RSI x contango grid)")
    print("=" * 70)

    rsi_thresholds = [25, 30, 35, 40]
    # For contango, we test different fixed thresholds (skip walk-forward optimization)
    contango_fixed_thresholds = [0.02, 0.04, 0.06, 0.08, 0.10]

    results_grid = []
    total_combos = 0
    passing_combos = 0

    print(f"\n  {'RSI<':>6s}  {'Contango>':>10s}  {'Trades':>7s}  {'Sharpe':>8s}  {'WR':>6s}  {'Status':>8s}")
    print(f"  {'----':>6s}  {'---------':>10s}  {'------':>7s}  {'------':>8s}  {'--':>6s}  {'------':>8s}")

    for rsi_thresh in rsi_thresholds:
        for cont_thresh in contango_fixed_thresholds:
            total_combos += 1

            # Build signals with this RSI threshold
            signals_df = build_signals_df(data, contango_slope, rsi_os=rsi_thresh)

            # Simple fixed-threshold backtest (not walk-forward, for speed)
            # Apply fixed contango threshold directly
            long_signals = signals_df[(signals_df['contango'] > cont_thresh) & (signals_df['rsi'] < rsi_thresh)]

            if len(long_signals) < 10:
                sharpe = 0.0
                wr = 0.0
                n_trades = len(long_signals)
            else:
                rets = long_signals['fwd_return']
                sharpe = compute_sharpe(rets)
                wr = (rets > 0).mean()
                n_trades = len(rets)

            status = 'PASS' if sharpe > 0.3 else 'FAIL'
            if sharpe > 0.3:
                passing_combos += 1

            print(f"  {rsi_thresh:>6d}  {cont_thresh:>10.2f}  {n_trades:>7d}  {sharpe:>8.4f}  {wr*100:>5.1f}%  {status:>8s}")

            results_grid.append({
                'rsi_thresh': rsi_thresh, 'contango_thresh': cont_thresh,
                'sharpe': sharpe, 'n_trades': n_trades, 'wr': wr
            })

    pct_passing = passing_combos / total_combos if total_combos > 0 else 0
    passed = pct_passing >= 0.60

    print(f"\n  Total parameter combinations: {total_combos}")
    print(f"  Passing (Sharpe > 0.3):       {passing_combos} ({pct_passing*100:.1f}%)")
    print(f"  Threshold:                    >= 60%")
    print(f"\n  >>> TEST 6: {'PASS' if passed else 'FAIL'} <<<")

    return passed


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION BATTERY")
    print("VIX Contango + Sector Oversold Contrarian Signal")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"6 Tests | Walk-forward {TRAIN_WINDOW}-day sliding | {HOLD_DAYS}-day holds")
    print("=" * 70)

    t0 = time.time()

    # Download data once
    data = download_data()
    if VIX_TICKER not in data:
        print("FATAL: VIX data not available")
        sys.exit(1)

    contango_slope = compute_contango_slope(data)

    # Run all 6 tests
    results = {}

    # Test 1: Re-implementation
    t1_pass, orig_sharpe, orig_trades, orig_signals = test_reimplementation(data, contango_slope)
    results['1_reimplementation'] = t1_pass

    if len(orig_trades) < 10:
        print("\nFATAL: Original backtest produced too few trades for adversarial testing.")
        sys.exit(1)

    # Test 2: Inverse signal
    results['2_inverse_signal'] = test_inverse_signal(data, contango_slope, orig_sharpe)

    # Test 3: Random timing
    results['3_random_timing'] = test_random_timing(orig_trades, orig_signals)

    # Test 4: Cost sensitivity
    results['4_cost_sensitivity'] = test_cost_sensitivity(orig_trades)

    # Test 5: Sub-period stability
    t5_pass, t5_warning = test_subperiod_stability(orig_trades)
    results['5_subperiod_stability'] = t5_pass

    # Test 6: Parameter robustness
    results['6_parameter_robustness'] = test_parameter_robustness(data, contango_slope)

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    elapsed = time.time() - t0
    n_pass = sum(results.values())
    n_total = len(results)

    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n  {'Test':>30s}  {'Result':>8s}")
    print(f"  {'----':>30s}  {'------':>8s}")

    for test_name, passed in results.items():
        label = test_name.replace('_', ' ').title()
        status = 'PASS' if passed else 'FAIL'
        print(f"  {label:>30s}  {status:>8s}")

    print(f"\n  FINAL VERDICT: {n_pass}/{n_total} PASS")
    if n_pass == n_total:
        print("  >>> SIGNAL PASSES ALL ADVERSARIAL TESTS — READY FOR FORWARD TEST <<<")
    elif n_pass >= 5:
        print("  >>> SIGNAL IS PROMISING BUT HAS WEAKNESSES — PROCEED WITH CAUTION <<<")
    elif n_pass >= 4:
        print("  >>> MARGINAL — ADDRESS FAILURES BEFORE DEPLOYMENT <<<")
    else:
        print("  >>> SIGNAL FAILS ADVERSARIAL VALIDATION — DO NOT DEPLOY <<<")

    print(f"\n  Runtime: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
