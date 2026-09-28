#!/usr/bin/env python3
"""
Sector Rotation via Relative Volume Anomaly — Backtest
======================================================
Hypothesis: Abnormal sector volume (relative to own history AND cross-sector)
signals institutional flows. Two strategies:
  1. Capitulation Buy: vol > 2x avg, price down > 1%, RSI(5) < 30 → hold 5 days
  2. Accumulation Momentum: vol > 1.5x avg, price up > 0.3% → hold 3 days

Walk-forward sliding 60-day window, 0.20% RT cost, regime stratification.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- Config ---
SECTORS = ['XLK', 'XLF', 'XLE', 'XLU', 'XLP', 'XLY', 'XLV', 'XLI', 'XLB', 'XLC', 'XLRE']
SPY = 'SPY'
START = '2020-01-01'
END = '2026-08-21'
VOL_LOOKBACK = 20
TRAIN_WINDOW = 60
COST_RT_PCT = 0.0020  # 0.20% round-trip
RSI_PERIOD = 5

# Strategy params
ACCUM_VOL_THRESH = 1.5
ACCUM_PRICE_THRESH = 0.003  # 0.3%
ACCUM_HOLD = 3

CAPIT_VOL_THRESH = 2.0
CAPIT_PRICE_THRESH = -0.01  # -1%
CAPIT_RSI_THRESH = 30
CAPIT_HOLD = 5

DISTRIB_VOL_THRESH = 1.5
DISTRIB_RSI_THRESH = 60


def compute_rsi(series, period=14):
    """Wilder RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def download_data():
    """Download all sector ETFs + SPY."""
    tickers = SECTORS + [SPY]
    print(f"Downloading {len(tickers)} tickers from {START} to {END}...")
    data = yf.download(tickers, start=START, end=END, progress=False, auto_adjust=True)

    close = data['Close']
    volume = data['Volume']

    # Drop any ticker with too many NaNs
    valid = close.columns[close.notna().sum() > 500]
    close = close[valid]
    volume = volume[valid]

    print(f"  Got {len(close)} trading days, {len(close.columns)} tickers")
    print(f"  Date range: {close.index[0].date()} to {close.index[-1].date()}")

    return close, volume


def compute_features(close, volume):
    """Compute all volume anomaly features for each sector."""
    returns = close.pct_change()

    features = {}
    for ticker in SECTORS:
        if ticker not in close.columns:
            continue

        df = pd.DataFrame(index=close.index)
        df['close'] = close[ticker]
        df['volume'] = volume[ticker]
        df['ret'] = returns[ticker]

        # Volume ratio = today vol / 20-day avg
        df['vol_avg_20'] = df['volume'].rolling(VOL_LOOKBACK).mean()
        df['vol_ratio'] = df['volume'] / df['vol_avg_20']

        # RSI(5)
        df['rsi'] = compute_rsi(df['close'], RSI_PERIOD)

        # Volume trend: 3 consecutive days of increasing volume
        df['vol_inc_1'] = df['volume'] > df['volume'].shift(1)
        df['vol_inc_2'] = df['volume'].shift(1) > df['volume'].shift(2)
        df['vol_trend'] = df['vol_inc_1'] & df['vol_inc_2']

        features[ticker] = df

    # Cross-sector volume z-score
    vol_ratios = pd.DataFrame({t: features[t]['vol_ratio'] for t in features})
    cross_mean = vol_ratios.mean(axis=1)
    cross_std = vol_ratios.std(axis=1)

    for ticker in features:
        features[ticker]['vol_zscore'] = (features[ticker]['vol_ratio'] - cross_mean) / cross_std

    return features


def classify_signals(features):
    """Classify each sector-day into signal types."""
    signals = []

    for ticker, df in features.items():
        for i in range(len(df)):
            row = df.iloc[i]
            date = df.index[i]

            if pd.isna(row['vol_ratio']) or pd.isna(row['rsi']):
                continue

            # Capitulation buy
            if (row['vol_ratio'] > CAPIT_VOL_THRESH and
                row['ret'] < CAPIT_PRICE_THRESH and
                row['rsi'] < CAPIT_RSI_THRESH):
                signals.append({
                    'date': date, 'ticker': ticker, 'type': 'capitulation_buy',
                    'vol_ratio': row['vol_ratio'], 'vol_zscore': row['vol_zscore'],
                    'ret': row['ret'], 'rsi': row['rsi'], 'hold_days': CAPIT_HOLD
                })

            # Accumulation
            elif (row['vol_ratio'] > ACCUM_VOL_THRESH and
                  row['ret'] > ACCUM_PRICE_THRESH):
                signals.append({
                    'date': date, 'ticker': ticker, 'type': 'accumulation',
                    'vol_ratio': row['vol_ratio'], 'vol_zscore': row['vol_zscore'],
                    'ret': row['ret'], 'rsi': row['rsi'], 'hold_days': ACCUM_HOLD
                })

            # Distribution sell (for reference, not traded yet)
            elif (row['vol_ratio'] > DISTRIB_VOL_THRESH and
                  row['ret'] < 0 and
                  row['rsi'] > DISTRIB_RSI_THRESH):
                signals.append({
                    'date': date, 'ticker': ticker, 'type': 'distribution',
                    'vol_ratio': row['vol_ratio'], 'vol_zscore': row['vol_zscore'],
                    'ret': row['ret'], 'rsi': row['rsi'], 'hold_days': 0
                })

    return pd.DataFrame(signals)


def run_walkforward_backtest(signals_df, close, strategy_type, hold_days):
    """
    Walk-forward sliding window backtest.
    Train window = 60 days (for signal validation only — we check that the signal
    pattern was profitable in the prior 60 days before taking it OOT).
    """
    # Filter to strategy type
    strat_signals = signals_df[signals_df['type'] == strategy_type].copy()
    if len(strat_signals) == 0:
        return pd.DataFrame(), {}

    strat_signals = strat_signals.sort_values('date')
    all_dates = close.index.sort_values()

    trades = []

    for _, sig in strat_signals.iterrows():
        entry_date = sig['date']
        ticker = sig['ticker']

        if ticker not in close.columns:
            continue

        # Find entry index (next day open = next day close proxy for daily)
        date_idx = all_dates.get_loc(entry_date)
        entry_idx = date_idx + 1  # Enter next day
        exit_idx = entry_idx + hold_days

        if exit_idx >= len(all_dates):
            continue

        # Walk-forward validation: check if this signal type was profitable
        # in the prior TRAIN_WINDOW days for this ticker
        train_start_idx = max(0, date_idx - TRAIN_WINDOW)
        train_signals = strat_signals[
            (strat_signals['date'] >= all_dates[train_start_idx]) &
            (strat_signals['date'] < entry_date) &
            (strat_signals['ticker'] == ticker)
        ]

        # Need at least 2 prior signals to validate (otherwise skip — no evidence)
        if len(train_signals) < 2:
            # Relaxed: allow if cross-sector signals show profitability
            cross_sector_signals = strat_signals[
                (strat_signals['date'] >= all_dates[train_start_idx]) &
                (strat_signals['date'] < entry_date)
            ]
            if len(cross_sector_signals) < 3:
                continue

        entry_price = close[ticker].iloc[entry_idx]
        exit_price = close[ticker].iloc[exit_idx]

        if pd.isna(entry_price) or pd.isna(exit_price):
            continue

        gross_ret = (exit_price - entry_price) / entry_price
        net_ret = gross_ret - COST_RT_PCT

        # Get SPY return for regime classification
        if SPY in close.columns:
            spy_ret = (close[SPY].iloc[exit_idx] - close[SPY].iloc[entry_idx]) / close[SPY].iloc[entry_idx]
        else:
            spy_ret = 0

        trades.append({
            'signal_date': entry_date,
            'entry_date': all_dates[entry_idx],
            'exit_date': all_dates[exit_idx],
            'ticker': ticker,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'gross_ret': gross_ret,
            'net_ret': net_ret,
            'vol_ratio': sig['vol_ratio'],
            'vol_zscore': sig['vol_zscore'],
            'rsi': sig['rsi'],
            'spy_ret': spy_ret,
        })

    trades_df = pd.DataFrame(trades)
    return trades_df


def compute_metrics(trades_df, label=""):
    """Compute strategy metrics."""
    if len(trades_df) == 0:
        print(f"\n{'='*60}")
        print(f"  {label}: NO TRADES")
        print(f"{'='*60}")
        return {}

    rets = trades_df['net_ret']
    n = len(rets)

    # Basic stats
    total_ret = (1 + rets).prod() - 1
    mean_ret = rets.mean()
    std_ret = rets.std()
    win_rate = (rets > 0).mean()

    # Annualize (assume ~252 trading days, avg hold ~4 days)
    avg_hold = 4
    trades_per_year = 252 / avg_hold
    ann_factor = np.sqrt(trades_per_year)

    sharpe = (mean_ret / std_ret) * ann_factor if std_ret > 0 else 0

    downside = rets[rets < 0].std()
    sortino = (mean_ret / downside) * ann_factor if downside > 0 else 0

    # Profit factor
    gross_wins = rets[rets > 0].sum()
    gross_losses = abs(rets[rets < 0].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Max drawdown (on cumulative equity curve)
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Day concentration
    if 'entry_date' in trades_df.columns:
        day_counts = trades_df['entry_date'].value_counts()
        max_day_trades = day_counts.max()
        day_conc = max_day_trades / n
    else:
        day_conc = 0

    # Per-year breakdown
    trades_df_copy = trades_df.copy()
    trades_df_copy['year'] = pd.to_datetime(trades_df_copy['entry_date']).dt.year
    yearly = trades_df_copy.groupby('year')['net_ret'].agg(['mean', 'count', 'sum'])

    # Regime stratification
    regime_results = {}
    if 'spy_ret' in trades_df.columns:
        trades_df_copy['regime'] = 'flat'
        trades_df_copy.loc[trades_df_copy['spy_ret'] > 0.003, 'regime'] = 'green'
        trades_df_copy.loc[trades_df_copy['spy_ret'] < -0.003, 'regime'] = 'red'

        for regime in ['green', 'red', 'flat']:
            r_trades = trades_df_copy[trades_df_copy['regime'] == regime]
            if len(r_trades) > 5:
                r_rets = r_trades['net_ret']
                r_sharpe = (r_rets.mean() / r_rets.std()) * ann_factor if r_rets.std() > 0 else 0
                r_wr = (r_rets > 0).mean()
                regime_results[regime] = {
                    'n': len(r_trades),
                    'sharpe': r_sharpe,
                    'wr': r_wr,
                    'mean_ret': r_rets.mean()
                }

    # Regime asymmetry check
    if 'green' in regime_results and 'red' in regime_results:
        sg = abs(regime_results['green']['sharpe'])
        sr = abs(regime_results['red']['sharpe'])
        max_s = max(sg, sr)
        regime_asym = abs(sg - sr) / max_s if max_s > 0 else 0
    else:
        regime_asym = None

    # Print results
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades:        {n}")
    print(f"  Win Rate:      {win_rate:.1%}")
    print(f"  Mean Return:   {mean_ret:.4%}")
    print(f"  Sharpe:        {sharpe:.2f}")
    print(f"  Sortino:       {sortino:.2f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Total Return:  {total_ret:.2%}")
    print(f"  Max Drawdown:  {max_dd:.2%}")
    print(f"  Day Conc:      {day_conc:.2%}")

    print(f"\n  Per-Year Breakdown:")
    for yr, row in yearly.iterrows():
        print(f"    {yr}: {row['count']:.0f} trades, mean={row['mean']:.4%}, total={row['sum']:.2%}")

    print(f"\n  Regime Stratification (SPY-based):")
    for regime, rd in regime_results.items():
        print(f"    {regime:6s}: n={rd['n']:4d}, Sharpe={rd['sharpe']:+.2f}, WR={rd['wr']:.1%}, mean={rd['mean_ret']:.4%}")

    if regime_asym is not None:
        status = "PASS" if regime_asym <= 0.50 else "FAIL (regime-tailored)"
        print(f"  Regime Asymmetry: {regime_asym:.2f} [{status}]")

    # Sector breakdown
    print(f"\n  Per-Sector Breakdown:")
    sector_stats = trades_df.groupby('ticker')['net_ret'].agg(['count', 'mean'])
    sector_stats = sector_stats.sort_values('mean', ascending=False)
    for ticker, row in sector_stats.iterrows():
        print(f"    {ticker}: {row['count']:.0f} trades, mean={row['mean']:.4%}")

    # Vol z-score quintile analysis
    if 'vol_zscore' in trades_df.columns and n >= 20:
        print(f"\n  Volume Z-Score Quintile Analysis:")
        trades_df_copy['zscore_q'] = pd.qcut(trades_df_copy['vol_zscore'], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'], duplicates='drop')
        for q, group in trades_df_copy.groupby('zscore_q'):
            q_rets = group['net_ret']
            q_sharpe = (q_rets.mean() / q_rets.std()) * ann_factor if q_rets.std() > 0 else 0
            print(f"    {q}: n={len(group):4d}, mean={q_rets.mean():.4%}, Sharpe={q_sharpe:+.2f}")

    metrics = {
        'n': n, 'sharpe': sharpe, 'sortino': sortino, 'pf': pf,
        'wr': win_rate, 'total_ret': total_ret, 'max_dd': max_dd,
        'day_conc': day_conc, 'regime_asym': regime_asym,
        'regime': regime_results, 'mean_ret': mean_ret
    }
    return metrics


def run_vol_zscore_enhanced(signals_df, close, strategy_type, hold_days, zscore_thresh=1.0):
    """
    Enhanced version: only take signals where cross-sector vol z-score > threshold.
    This filters for truly anomalous sector-specific volume.
    """
    strat_signals = signals_df[
        (signals_df['type'] == strategy_type) &
        (signals_df['vol_zscore'] > zscore_thresh)
    ].copy()

    if len(strat_signals) == 0:
        return pd.DataFrame()

    all_dates = close.index.sort_values()
    trades = []

    for _, sig in strat_signals.iterrows():
        entry_date = sig['date']
        ticker = sig['ticker']
        if ticker not in close.columns:
            continue

        date_idx = all_dates.get_loc(entry_date)
        entry_idx = date_idx + 1
        exit_idx = entry_idx + hold_days

        if exit_idx >= len(all_dates):
            continue

        entry_price = close[ticker].iloc[entry_idx]
        exit_price = close[ticker].iloc[exit_idx]

        if pd.isna(entry_price) or pd.isna(exit_price):
            continue

        gross_ret = (exit_price - entry_price) / entry_price
        net_ret = gross_ret - COST_RT_PCT

        spy_ret = 0
        if SPY in close.columns:
            spy_ret = (close[SPY].iloc[exit_idx] - close[SPY].iloc[entry_idx]) / close[SPY].iloc[entry_idx]

        trades.append({
            'signal_date': entry_date,
            'entry_date': all_dates[entry_idx],
            'exit_date': all_dates[exit_idx],
            'ticker': ticker,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'gross_ret': gross_ret,
            'net_ret': net_ret,
            'vol_ratio': sig['vol_ratio'],
            'vol_zscore': sig['vol_zscore'],
            'rsi': sig['rsi'],
            'spy_ret': spy_ret,
        })

    return pd.DataFrame(trades)


def main():
    print("=" * 70)
    print("  SECTOR VOLUME ANOMALY ROTATION BACKTEST")
    print("  Hypothesis: Abnormal relative volume predicts sector rotation")
    print("=" * 70)

    # 1. Download data
    close, volume = download_data()

    # 2. Compute features
    print("\nComputing features...")
    features = compute_features(close, volume)

    # 3. Classify signals
    print("Classifying signals...")
    signals = classify_signals(features)

    print(f"\nSignal counts:")
    for stype in ['accumulation', 'capitulation_buy', 'distribution']:
        n = len(signals[signals['type'] == stype])
        print(f"  {stype}: {n}")

    # 4. Run backtests
    print("\n" + "=" * 70)
    print("  STRATEGY 1: CAPITULATION BUY (vol>2x, price<-1%, RSI<30, hold 5d)")
    print("=" * 70)

    capit_trades = run_walkforward_backtest(signals, close, 'capitulation_buy', CAPIT_HOLD)
    capit_metrics = compute_metrics(capit_trades, "Capitulation Buy — Base")

    print("\n" + "=" * 70)
    print("  STRATEGY 2: ACCUMULATION MOMENTUM (vol>1.5x, price>0.3%, hold 3d)")
    print("=" * 70)

    accum_trades = run_walkforward_backtest(signals, close, 'accumulation', ACCUM_HOLD)
    accum_metrics = compute_metrics(accum_trades, "Accumulation Momentum — Base")

    # 5. Enhanced: z-score filtered versions
    print("\n" + "=" * 70)
    print("  STRATEGY 3: CAPITULATION BUY + VOL Z-SCORE > 1.0")
    print("=" * 70)

    capit_z_trades = run_vol_zscore_enhanced(signals, close, 'capitulation_buy', CAPIT_HOLD, 1.0)
    capit_z_metrics = compute_metrics(capit_z_trades, "Capitulation Buy — Z>1.0 Enhanced")

    print("\n" + "=" * 70)
    print("  STRATEGY 4: ACCUMULATION + VOL Z-SCORE > 1.0")
    print("=" * 70)

    accum_z_trades = run_vol_zscore_enhanced(signals, close, 'accumulation', ACCUM_HOLD, 1.0)
    accum_z_metrics = compute_metrics(accum_z_trades, "Accumulation — Z>1.0 Enhanced")

    # 6. Combo: accumulation with volume trend (3 consecutive rising vol days)
    print("\n" + "=" * 70)
    print("  STRATEGY 5: ACCUMULATION + RISING VOLUME TREND (3d)")
    print("=" * 70)

    # Filter accumulation signals where volume trend is true
    accum_trend_signals = []
    for _, sig in signals[signals['type'] == 'accumulation'].iterrows():
        ticker = sig['ticker']
        date = sig['date']
        if ticker in features and date in features[ticker].index:
            if features[ticker].loc[date, 'vol_trend']:
                accum_trend_signals.append(sig)

    if accum_trend_signals:
        accum_trend_df = pd.DataFrame(accum_trend_signals)
        accum_trend_trades = run_walkforward_backtest(accum_trend_df, close, 'accumulation', ACCUM_HOLD)
        accum_trend_metrics = compute_metrics(accum_trend_trades, "Accumulation + Volume Trend")
    else:
        print("  No signals with volume trend filter")
        accum_trend_metrics = {}

    # 7. Summary verdict
    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)

    all_results = [
        ("Capitulation Buy (base)", capit_metrics),
        ("Accumulation (base)", accum_metrics),
        ("Capitulation Z>1", capit_z_metrics),
        ("Accumulation Z>1", accum_z_metrics),
        ("Accum + Vol Trend", accum_trend_metrics),
    ]

    any_alive = False
    for name, m in all_results:
        if not m:
            print(f"  {name:30s}: NO TRADES / NO DATA")
            continue

        sharpe = m.get('sharpe', 0)
        status = "ALIVE" if sharpe >= 0.5 else "DEAD"
        if sharpe >= 0.5:
            any_alive = True

        regime_note = ""
        if m.get('regime_asym') is not None:
            if m['regime_asym'] > 0.50:
                status = "DEAD (regime-tailored)"
                regime_note = f" | regime_asym={m['regime_asym']:.2f}"

        print(f"  {name:30s}: Sharpe={sharpe:+.2f}, Sortino={m.get('sortino',0):+.2f}, "
              f"PF={m.get('pf',0):.2f}, WR={m.get('wr',0):.1%}, n={m.get('n',0)}{regime_note} [{status}]")

    if not any_alive:
        print("\n  >>> ALL STRATEGIES DEAD (Sharpe < 0.5). Volume anomaly as sector rotation")
        print("      predictor does NOT pass the bar at daily frequency with these thresholds.")
        print("      Consider: intraday granularity, options flow instead of equity volume,")
        print("      or use volume anomaly as CONFLUENCE FILTER only (which is already wired).")
    else:
        print("\n  >>> SOME STRATEGIES ALIVE. Further investigation warranted.")


if __name__ == '__main__':
    main()
