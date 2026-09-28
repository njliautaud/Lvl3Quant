"""
Align SPY 1-Min Bars to ES MBO Feature Timestamps
===================================================
ES MBO features are 100ms bars, 234,000 per day (6.5 hrs RTH: 09:30-16:00 ET).
SPY 1-min bars from Alpaca cover extended hours (08:00-22:00 UTC = 04:00-18:00 ET).

This script:
1. Loads ES MBO bar timestamps (reconstructed from file dates + bar index)
2. Loads SPY 1-min bar data
3. Aligns by timestamp (SPY bar covering each ES bar)
4. Computes SPY returns at 1-min, 5-min, 10-min horizons
5. Saves aligned dataset for cross-venue analysis

ES bar timing: 100ms bars, RTH only (09:30:00 ET to 16:00:00 ET)
  bar_idx=0 -> 09:30:00.000 ET
  bar_idx=1 -> 09:30:00.100 ET
  ...
  bar_idx=233999 -> 15:59:59.900 ET (6.5 hrs * 36000 bars/hr = 234,000)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pytz

LVL3_ROOT = Path(__file__).parent.parent
FEAT_CACHE = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
SPY_PATH = LVL3_ROOT / 'data' / 'spy' / 'spy_1min_bars.parquet'
OUTPUT_DIR = LVL3_ROOT / 'data' / 'spy'

ET = pytz.timezone('US/Eastern')
UTC = pytz.utc


def get_es_bar_timestamps(date_str, n_bars=234000):
    """
    Reconstruct ES MBO bar timestamps from date string and bar count.
    100ms bars, RTH: 09:30:00.000 to 15:59:59.900 ET.
    """
    date = datetime.strptime(date_str, '%Y-%m-%d')
    # RTH start: 09:30:00 ET
    rth_start = ET.localize(date.replace(hour=9, minute=30, second=0, microsecond=0))

    # Generate timestamps at 100ms intervals
    timestamps = pd.date_range(
        start=rth_start,
        periods=n_bars,
        freq='100ms',
    )
    return timestamps


def load_spy_data():
    """Load SPY 1-min bars from parquet."""
    df = pd.read_parquet(SPY_PATH)
    print(f"SPY raw bars: {len(df):,}")

    # Ensure timestamp is timezone-aware UTC
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

    # Convert to ET for easier RTH filtering
    df['timestamp_et'] = df['timestamp'].dt.tz_convert(ET)

    # Compute returns
    df['ret_1min'] = df['close'].pct_change()
    df['ret_log_1min'] = np.log(df['close'] / df['close'].shift(1))

    # Forward returns at various horizons
    df['fwd_ret_1min'] = df['close'].shift(-1) / df['close'] - 1
    df['fwd_ret_5min'] = df['close'].shift(-5) / df['close'] - 1
    df['fwd_ret_10min'] = df['close'].shift(-10) / df['close'] - 1
    df['fwd_ret_30min'] = df['close'].shift(-30) / df['close'] - 1

    # Realized vol (20-bar rolling)
    df['rvol_20bar'] = df['ret_log_1min'].rolling(20).std() * np.sqrt(390)  # annualized

    return df


def align_for_date(date_str, spy_df, sample_every_n=600):
    """
    Align SPY data to ES MBO bars for a single date.

    ES bars are 100ms, SPY bars are 1-min. We sample ES at both
    start-of-minute (bar 0, 600, 1200...) and end-of-minute (bar 599, 1199...).
    SPY bar timestamp = start of bar period (Alpaca convention).
    ES start-of-min correlates with SPY open; ES end-of-min correlates with SPY close.

    Returns aligned rows with ES bar indices + SPY bar data.
    """
    # Load ES features to get bar count
    npz_path = FEAT_CACHE / f'{date_str}_mbo_features.npz'
    if not npz_path.exists():
        return None

    data = np.load(str(npz_path))
    feats = data['mbo_features']
    n_bars = len(feats)

    # ES mid prices (column 0)
    es_mid = feats[:, 0]

    # Sample at start and end of each minute
    start_indices = np.arange(0, n_bars, sample_every_n)
    end_indices = np.arange(sample_every_n - 1, n_bars, sample_every_n)
    n_min = min(len(start_indices), len(end_indices))
    start_indices = start_indices[:n_min]
    end_indices = end_indices[:n_min]

    es_start_mid = es_mid[start_indices]
    es_end_mid = es_mid[end_indices]

    # Get ES timestamps for start-of-minute bars
    es_timestamps = get_es_bar_timestamps(date_str, n_bars)
    es_start_ts = es_timestamps[start_indices]

    # Convert to UTC for matching
    es_utc = es_start_ts.tz_convert('UTC')

    # Filter SPY to this date's RTH hours
    date_dt = datetime.strptime(date_str, '%Y-%m-%d')
    rth_start_utc = ET.localize(date_dt.replace(hour=9, minute=30)).astimezone(UTC)
    rth_end_utc = ET.localize(date_dt.replace(hour=16, minute=0)).astimezone(UTC)

    spy_day = spy_df[
        (spy_df['timestamp'] >= rth_start_utc) &
        (spy_df['timestamp'] < rth_end_utc)
    ].copy()

    if len(spy_day) == 0:
        return None

    # Build ES dataframe with both start and end-of-minute mids
    es_df = pd.DataFrame({
        'timestamp': es_utc,
        'es_bar_idx_start': start_indices,
        'es_bar_idx_end': end_indices,
        'es_mid_start': es_start_mid,
        'es_mid_end': es_end_mid,
    })

    # Ensure both sorted by timestamp
    es_df = es_df.sort_values('timestamp')
    spy_day_sorted = spy_day.sort_values('timestamp')

    aligned = pd.merge_asof(
        es_df,
        spy_day_sorted[['timestamp', 'open', 'high', 'low', 'close', 'volume',
                         'vwap', 'fwd_ret_1min', 'fwd_ret_5min', 'fwd_ret_10min',
                         'fwd_ret_30min', 'rvol_20bar']],
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('61s'),
        suffixes=('', '_spy'),
    )

    # Rename SPY columns for clarity
    aligned = aligned.rename(columns={
        'open': 'spy_open',
        'high': 'spy_high',
        'low': 'spy_low',
        'close': 'spy_close',
        'volume': 'spy_volume',
        'vwap': 'spy_vwap',
    })

    aligned['date'] = date_str
    aligned['matched'] = aligned['spy_close'].notna()

    return aligned


def main():
    print("=" * 60)
    print("SPY-ES Alignment Script")
    print("=" * 60)

    # Load SPY data
    spy_df = load_spy_data()
    print(f"SPY data loaded: {len(spy_df):,} bars")
    print(f"  Date range: {spy_df['timestamp'].min()} to {spy_df['timestamp'].max()}")

    # Get ES trading dates
    es_files = sorted(FEAT_CACHE.glob('*_mbo_features.npz'))
    dates = [f.stem.replace('_mbo_features', '') for f in es_files]
    print(f"\nES MBO dates: {len(dates)} trading days")
    print(f"  First: {dates[0]}, Last: {dates[-1]}")

    # Align each date
    all_aligned = []
    matched_days = 0
    unmatched_days = []

    for i, date_str in enumerate(dates):
        result = align_for_date(date_str, spy_df)
        if result is not None and result['matched'].any():
            all_aligned.append(result)
            matched_days += 1
            n_matched = result['matched'].sum()
            n_total = len(result)
            if i % 10 == 0:
                print(f"  [{i+1}/{len(dates)}] {date_str}: {n_matched}/{n_total} bars aligned")
        else:
            unmatched_days.append(date_str)
            if i % 10 == 0:
                print(f"  [{i+1}/{len(dates)}] {date_str}: NO SPY DATA")

    if not all_aligned:
        print("ERROR: No aligned data!")
        return

    df = pd.concat(all_aligned, ignore_index=True)

    # Compute ES returns using end-of-minute mid (matches SPY close)
    df['es_ret_1min'] = df.groupby('date')['es_mid_end'].pct_change()

    # ES-SPY correlation per day
    print(f"\n{'='*60}")
    print(f"ALIGNMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total aligned bars: {len(df):,}")
    print(f"Matched days: {matched_days}/{len(dates)}")
    print(f"Unmatched days: {len(unmatched_days)}")
    if unmatched_days:
        print(f"  Missing: {unmatched_days[:5]}{'...' if len(unmatched_days) > 5 else ''}")

    matched = df[df['matched']].copy()
    print(f"\nBars with SPY match: {len(matched):,} ({100*len(matched)/len(df):.1f}%)")

    if len(matched) > 100:
        # Level correlation (should be ~0.99+)
        corr_end = matched[['es_mid_end', 'spy_close']].corr().iloc[0, 1]
        corr_start = matched[['es_mid_start', 'spy_open']].corr().iloc[0, 1]
        print(f"ES end-of-min vs SPY close level corr: {corr_end:.6f}")
        print(f"ES start-of-min vs SPY open level corr: {corr_start:.6f}")

        # 1-min change correlations (the important ones)
        daily_corrs_close = []
        daily_corrs_open = []
        for date, grp in matched.groupby('date'):
            if len(grp) > 20:
                es_close_chg = grp['es_mid_end'].diff()
                spy_close_chg = grp['spy_close'].diff()
                es_open_chg = grp['es_mid_start'].diff()
                spy_open_chg = grp['spy_open'].diff()

                valid_c = es_close_chg.notna() & spy_close_chg.notna()
                valid_o = es_open_chg.notna() & spy_open_chg.notna()

                if valid_c.sum() > 10:
                    c = np.corrcoef(es_close_chg[valid_c], spy_close_chg[valid_c])[0, 1]
                    daily_corrs_close.append({'date': date, 'corr': c, 'n_bars': valid_c.sum()})
                if valid_o.sum() > 10:
                    c = np.corrcoef(es_open_chg[valid_o], spy_open_chg[valid_o])[0, 1]
                    daily_corrs_open.append({'date': date, 'corr': c, 'n_bars': valid_o.sum()})

        if daily_corrs_close:
            dc = pd.DataFrame(daily_corrs_close)
            print(f"\nDaily ES end-of-min vs SPY close 1-min change correlation:")
            print(f"  Mean: {dc['corr'].mean():.4f}")
            print(f"  Median: {dc['corr'].median():.4f}")
            print(f"  Min: {dc['corr'].min():.4f}, Max: {dc['corr'].max():.4f}")

        if daily_corrs_open:
            dc = pd.DataFrame(daily_corrs_open)
            print(f"\nDaily ES start-of-min vs SPY open 1-min change correlation:")
            print(f"  Mean: {dc['corr'].mean():.4f}")
            print(f"  Median: {dc['corr'].median():.4f}")
            print(f"  Min: {dc['corr'].min():.4f}, Max: {dc['corr'].max():.4f}")

        # SPY summary stats
        print(f"\nSPY price range: ${matched['spy_close'].min():.2f} - ${matched['spy_close'].max():.2f}")
        print(f"ES mid range: {matched['es_mid_end'].min():.2f} - {matched['es_mid_end'].max():.2f}")
        print(f"ES/SPY ratio: {(matched['es_mid_end'] / matched['spy_close']).mean():.4f}")
        print(f"SPY avg volume/bar: {matched['spy_volume'].mean():,.0f}")

    # Save
    output_path = OUTPUT_DIR / 'spy_es_aligned.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\nSaved aligned dataset: {output_path}")
    print(f"Columns: {list(df.columns)}")

    # Also save a summary CSV for quick inspection
    summary_path = OUTPUT_DIR / 'spy_es_alignment_summary.csv'
    summary = df.groupby('date').agg(
        n_bars=('es_bar_idx_start', 'count'),
        n_matched=('matched', 'sum'),
        es_open=('es_mid_start', 'first'),
        es_close=('es_mid_end', 'last'),
        spy_open=('spy_close', 'first'),
        spy_close_last=('spy_close', 'last'),
        spy_vol_total=('spy_volume', 'sum'),
    ).reset_index()
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == '__main__':
    main()
