#!/usr/bin/env python3
"""
ETF Fund Flow & Price-Volume Microstructure Signal Research
===========================================================
Three unconventional signal ideas tested on sector ETFs (2021-2026):
1. Relative Volume Surprise as Entry Timing
2. Cross-Sector Flow Rotation Detection
3. Gap-and-Continuation vs Gap-and-Fade by Sector

All with permutation tests, regime stratification, per-sector breakdown.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services',
}
BENCHMARK = 'SPY'
START = '2021-01-01'
END = '2026-08-15'
N_PERM = 1000
HOLD_DAYS_SHORT = 1
HOLD_DAYS_LONG = 5

CACHE_DIR = Path('/home/jupiter/Lvl3Quant/research/cache')
CACHE_DIR.mkdir(exist_ok=True)

# ── Data Download ───────────────────────────────────────────────────
def download_data():
    """Download all sector ETF + SPY data."""
    tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK]
    cache_file = CACHE_DIR / 'microstructure_etf_data.pkl'

    if cache_file.exists():
        data = pd.read_pickle(cache_file)
        # Check if reasonably fresh
        if len(data) > 1000:
            print(f"Loaded cached data: {len(data)} rows, {data.index[0]} to {data.index[-1]}")
            return data

    print(f"Downloading {len(tickers)} tickers from {START} to {END}...")
    data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    data.to_pickle(cache_file)
    print(f"Downloaded: {len(data)} rows, {data.index[0]} to {data.index[-1]}")
    return data


def build_sector_frames(raw_data):
    """Build clean per-ticker DataFrames."""
    frames = {}
    tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK]

    for ticker in tickers:
        try:
            df = pd.DataFrame({
                'open': raw_data['Open'][ticker],
                'high': raw_data['High'][ticker],
                'low': raw_data['Low'][ticker],
                'close': raw_data['Close'][ticker],
                'volume': raw_data['Volume'][ticker],
            }).dropna()

            df['ret'] = df['close'].pct_change()
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            frames[ticker] = df
        except Exception as e:
            print(f"  Skipping {ticker}: {e}")

    return frames


# ── Utility Functions ───────────────────────────────────────────────
def compute_forward_returns(series, prices, hold_days):
    """Compute forward returns for signal dates."""
    fwd = prices.pct_change(hold_days).shift(-hold_days)
    aligned = fwd.reindex(series.index)
    return aligned


def sharpe_ratio(returns):
    if len(returns) < 5 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)


def sortino_ratio(returns):
    if len(returns) < 5:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float('inf') if returns.mean() > 0 else 0.0
    return returns.mean() / downside.std() * np.sqrt(252)


def profit_factor(returns):
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return gains / losses


def win_rate(returns):
    if len(returns) == 0:
        return 0.0
    return (returns > 0).mean()


def compute_metrics(returns, label=""):
    """Compute all standard metrics."""
    r = returns.dropna()
    if len(r) < 5:
        return {'n_trades': len(r), 'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0, 'mean_ret_bps': 0}
    return {
        'n_trades': int(len(r)),
        'sharpe': round(sharpe_ratio(r), 3),
        'sortino': round(sortino_ratio(r), 3),
        'pf': round(profit_factor(r), 3),
        'wr': round(win_rate(r), 4),
        'mean_ret_bps': round(r.mean() * 10000, 2),
        'median_ret_bps': round(r.median() * 10000, 2),
        'total_ret_pct': round(r.sum() * 100, 2),
    }


def permutation_test(returns, n_perm=N_PERM):
    """Run permutation test. Returns p-value for mean return > 0."""
    r = returns.dropna().values
    if len(r) < 10:
        return 1.0

    observed_mean = r.mean()
    count_ge = 0
    for _ in range(n_perm):
        perm = r.copy()
        np.random.shuffle(perm)
        # Randomly flip signs to test if the signal matters
        signs = np.random.choice([-1, 1], size=len(perm))
        perm_mean = (perm * signs).mean()
        if perm_mean >= observed_mean:
            count_ge += 1

    return round(count_ge / n_perm, 4)


def regime_stratify(returns, spy_returns):
    """Split returns into green/red SPY day regimes."""
    aligned_spy = spy_returns.reindex(returns.index)
    green_mask = aligned_spy > 0
    red_mask = aligned_spy <= 0

    green_r = returns[green_mask].dropna()
    red_r = returns[red_mask].dropna()

    return {
        'green_days': compute_metrics(green_r),
        'red_days': compute_metrics(red_r),
        'regime_sharpe_ratio': round(
            abs(sharpe_ratio(green_r) - sharpe_ratio(red_r)) /
            max(abs(sharpe_ratio(green_r)), abs(sharpe_ratio(red_r)), 0.001), 3
        ) if len(green_r) > 5 and len(red_r) > 5 else None,
    }


# ══════════════════════════════════════════════════════════════════════
# IDEA 1: Relative Volume Surprise
# ══════════════════════════════════════════════════════════════════════
def idea1_volume_surprise(frames, spy_rets):
    """Volume surprise as entry timing signal."""
    print("\n" + "="*70)
    print("IDEA 1: RELATIVE VOLUME SURPRISE")
    print("="*70)

    all_long_returns_5d = []
    all_short_returns_5d = []
    all_long_returns_1d = []
    all_short_returns_1d = []
    per_sector = {}

    for ticker, name in SECTOR_ETFS.items():
        if ticker not in frames:
            continue
        df = frames[ticker].copy()

        # Volume surprise ratio
        df['vol_avg_20d'] = df['volume'].rolling(20).mean()
        df['vol_surprise'] = df['volume'] / df['vol_avg_20d']
        df['abs_ret'] = df['ret'].abs()

        # Forward returns
        df['fwd_1d'] = df['close'].pct_change(1).shift(-1)
        df['fwd_5d'] = df['close'].pct_change(5).shift(-5)

        # Accumulation signal: high volume, small price move
        accum_mask = (df['vol_surprise'] > 2.0) & (df['abs_ret'] < 0.005)
        # Distribution signal: high volume, big price drop
        distrib_mask = (df['vol_surprise'] > 2.0) & (df['ret'] < -0.01)

        long_5d = df.loc[accum_mask, 'fwd_5d'].dropna()
        short_5d = -df.loc[distrib_mask, 'fwd_5d'].dropna()  # negate for short
        long_1d = df.loc[accum_mask, 'fwd_1d'].dropna()
        short_1d = -df.loc[distrib_mask, 'fwd_1d'].dropna()

        sector_result = {
            'long_accumulation': {
                '5d': compute_metrics(long_5d),
                '1d': compute_metrics(long_1d),
                'n_signals': int(accum_mask.sum()),
            },
            'short_distribution': {
                '5d': compute_metrics(short_5d),
                '1d': compute_metrics(short_1d),
                'n_signals': int(distrib_mask.sum()),
            }
        }
        per_sector[name] = sector_result

        all_long_returns_5d.append(long_5d)
        all_short_returns_5d.append(short_5d)
        all_long_returns_1d.append(long_1d)
        all_short_returns_1d.append(short_1d)

        print(f"  {name:25s} | Accum signals: {accum_mask.sum():4d} | Distrib signals: {distrib_mask.sum():4d}")

    # Aggregate
    agg_long_5d = pd.concat(all_long_returns_5d) if all_long_returns_5d else pd.Series(dtype=float)
    agg_short_5d = pd.concat(all_short_returns_5d) if all_short_returns_5d else pd.Series(dtype=float)
    agg_long_1d = pd.concat(all_long_returns_1d) if all_long_returns_1d else pd.Series(dtype=float)
    agg_short_1d = pd.concat(all_short_returns_1d) if all_short_returns_1d else pd.Series(dtype=float)

    # Combined long+short
    combined_5d = pd.concat([agg_long_5d, agg_short_5d])
    combined_1d = pd.concat([agg_long_1d, agg_short_1d])

    print(f"\n  AGGREGATE (all sectors):")
    print(f"    Long (accumulation) 5d: {compute_metrics(agg_long_5d)}")
    print(f"    Short (distribution) 5d: {compute_metrics(agg_short_5d)}")
    print(f"    Combined 5d: {compute_metrics(combined_5d)}")

    # Permutation tests
    perm_long_5d = permutation_test(agg_long_5d)
    perm_short_5d = permutation_test(agg_short_5d)
    perm_combined_5d = permutation_test(combined_5d)
    print(f"\n  Permutation p-values (5d):")
    print(f"    Long: {perm_long_5d} | Short: {perm_short_5d} | Combined: {perm_combined_5d}")

    # Regime
    regime_long = regime_stratify(agg_long_5d, spy_rets)
    regime_short = regime_stratify(agg_short_5d, spy_rets)
    regime_combined = regime_stratify(combined_5d, spy_rets)

    # Also test stricter thresholds
    thresholds = {}
    for vol_thresh in [1.5, 2.0, 2.5, 3.0]:
        all_r = []
        for ticker in SECTOR_ETFS:
            if ticker not in frames:
                continue
            df = frames[ticker].copy()
            df['vol_avg_20d'] = df['volume'].rolling(20).mean()
            df['vol_surprise'] = df['volume'] / df['vol_avg_20d']
            df['abs_ret'] = df['ret'].abs()
            df['fwd_5d'] = df['close'].pct_change(5).shift(-5)

            mask = (df['vol_surprise'] > vol_thresh) & (df['abs_ret'] < 0.005)
            all_r.append(df.loc[mask, 'fwd_5d'].dropna())

        combined = pd.concat(all_r) if all_r else pd.Series(dtype=float)
        thresholds[f'vol_thresh_{vol_thresh}'] = compute_metrics(combined)

    print(f"\n  Threshold sensitivity (accumulation long, 5d):")
    for k, v in thresholds.items():
        print(f"    {k}: n={v['n_trades']}, sharpe={v['sharpe']}, wr={v['wr']}")

    return {
        'aggregate': {
            'long_accumulation_5d': compute_metrics(agg_long_5d),
            'short_distribution_5d': compute_metrics(agg_short_5d),
            'combined_5d': compute_metrics(combined_5d),
            'long_accumulation_1d': compute_metrics(agg_long_1d),
            'short_distribution_1d': compute_metrics(agg_short_1d),
            'combined_1d': compute_metrics(combined_1d),
        },
        'permutation_pvalues': {
            'long_5d': perm_long_5d,
            'short_5d': perm_short_5d,
            'combined_5d': perm_combined_5d,
        },
        'regime_stratification': {
            'long': regime_long,
            'short': regime_short,
            'combined': regime_combined,
        },
        'threshold_sensitivity': thresholds,
        'per_sector': per_sector,
    }


# ══════════════════════════════════════════════════════════════════════
# IDEA 2: Cross-Sector Flow Rotation Detection
# ══════════════════════════════════════════════════════════════════════
def idea2_flow_rotation(frames, spy_rets):
    """Cross-sector money flow rotation signal."""
    print("\n" + "="*70)
    print("IDEA 2: CROSS-SECTOR FLOW ROTATION")
    print("="*70)

    # Build daily money flow for each sector
    flow_df = pd.DataFrame()
    ret_df = pd.DataFrame()
    fwd5_df = pd.DataFrame()
    fwd1_df = pd.DataFrame()

    for ticker, name in SECTOR_ETFS.items():
        if ticker not in frames:
            continue
        df = frames[ticker]
        # Money flow = price * volume * sign(close - open)
        sign_co = np.sign(df['close'] - df['open'])
        flow_df[ticker] = df['close'] * df['volume'] * sign_co
        ret_df[ticker] = df['ret']
        fwd5_df[ticker] = df['close'].pct_change(5).shift(-5)
        fwd1_df[ticker] = df['close'].pct_change(1).shift(-1)

    flow_df = flow_df.dropna()
    common_idx = flow_df.index.intersection(ret_df.dropna().index)
    common_idx = common_idx.intersection(fwd5_df.dropna().index)

    # Rank sectors by flow each day
    flow_rank = flow_df.rank(axis=1, ascending=False)

    # Long top 3, short bottom 3
    n_long = 3
    n_short = 3

    daily_returns_5d = []
    daily_returns_1d = []
    dates_used = []

    for date in common_idx:
        ranks = flow_rank.loc[date]
        if ranks.isna().all():
            continue

        long_tickers = ranks.nsmallest(n_long).index.tolist()   # rank 1,2,3 = top flow
        short_tickers = ranks.nlargest(n_short).index.tolist()  # bottom flow

        # Forward returns
        long_fwd5 = fwd5_df.loc[date, long_tickers].mean()
        short_fwd5 = fwd5_df.loc[date, short_tickers].mean()
        long_fwd1 = fwd1_df.loc[date, long_tickers].mean()
        short_fwd1 = fwd1_df.loc[date, short_tickers].mean()

        if not np.isnan(long_fwd5) and not np.isnan(short_fwd5):
            daily_returns_5d.append(long_fwd5 - short_fwd5)  # L/S portfolio
            dates_used.append(date)
        if not np.isnan(long_fwd1) and not np.isnan(short_fwd1):
            daily_returns_1d.append(long_fwd1 - short_fwd1)

    ls_5d = pd.Series(daily_returns_5d, index=dates_used[:len(daily_returns_5d)])
    ls_1d = pd.Series(daily_returns_1d, index=dates_used[:len(daily_returns_1d)])

    # Also test: long-only top 3, short-only bottom 3
    long_only_5d = []
    short_only_5d = []
    for date in common_idx:
        ranks = flow_rank.loc[date]
        if ranks.isna().all():
            continue
        long_tickers = ranks.nsmallest(n_long).index.tolist()
        short_tickers = ranks.nlargest(n_short).index.tolist()

        lo = fwd5_df.loc[date, long_tickers].mean()
        so = -fwd5_df.loc[date, short_tickers].mean()  # negate for short profit
        if not np.isnan(lo):
            long_only_5d.append(lo)
        if not np.isnan(so):
            short_only_5d.append(so)

    long_only_5d = pd.Series(long_only_5d)
    short_only_5d = pd.Series(short_only_5d)

    print(f"  Total signal days: {len(ls_5d)}")
    print(f"  L/S 5d: {compute_metrics(ls_5d)}")
    print(f"  L/S 1d: {compute_metrics(ls_1d)}")
    print(f"  Long-only top 3 (5d): {compute_metrics(long_only_5d)}")
    print(f"  Short-only bottom 3 (5d): {compute_metrics(short_only_5d)}")

    # Permutation test on L/S
    perm_ls_5d = permutation_test(ls_5d)
    perm_ls_1d = permutation_test(ls_1d)
    print(f"\n  Permutation p-values: L/S 5d={perm_ls_5d}, L/S 1d={perm_ls_1d}")

    # Regime
    regime_ls = regime_stratify(ls_5d, spy_rets)

    # Per-sector: how often each sector appears in top/bottom
    top_counts = {}
    bot_counts = {}
    top_fwd_returns = {}
    bot_fwd_returns = {}

    for ticker in SECTOR_ETFS:
        top_count = 0
        bot_count = 0
        t_rets = []
        b_rets = []
        for date in common_idx:
            if ticker not in flow_rank.columns:
                continue
            rank = flow_rank.loc[date, ticker]
            if rank <= n_long:
                top_count += 1
                fwd = fwd5_df.loc[date, ticker] if date in fwd5_df.index else np.nan
                if not np.isnan(fwd):
                    t_rets.append(fwd)
            elif rank >= len(SECTOR_ETFS) - n_short + 1:
                bot_count += 1
                fwd = fwd5_df.loc[date, ticker] if date in fwd5_df.index else np.nan
                if not np.isnan(fwd):
                    b_rets.append(fwd)

        name = SECTOR_ETFS[ticker]
        top_counts[name] = top_count
        bot_counts[name] = bot_count
        top_fwd_returns[name] = compute_metrics(pd.Series(t_rets)) if t_rets else {}
        bot_fwd_returns[name] = compute_metrics(pd.Series(b_rets)) if b_rets else {}

    print(f"\n  Sector frequency in top/bottom flow:")
    for name in sorted(top_counts.keys()):
        print(f"    {name:25s} | Top: {top_counts[name]:4d} | Bottom: {bot_counts[name]:4d}")

    # Test different numbers of top/bottom
    n_tests = {}
    for n in [1, 2, 3, 4, 5]:
        rets = []
        for date in common_idx:
            ranks = flow_rank.loc[date]
            if ranks.isna().all():
                continue
            lt = ranks.nsmallest(n).index.tolist()
            st = ranks.nlargest(n).index.tolist()
            lr = fwd5_df.loc[date, lt].mean()
            sr = fwd5_df.loc[date, st].mean()
            if not np.isnan(lr) and not np.isnan(sr):
                rets.append(lr - sr)
        n_tests[f'top_{n}_vs_bottom_{n}'] = compute_metrics(pd.Series(rets))

    print(f"\n  N-basket sensitivity (5d L/S):")
    for k, v in n_tests.items():
        print(f"    {k}: sharpe={v['sharpe']}, wr={v['wr']}, mean={v['mean_ret_bps']}bps")

    return {
        'aggregate': {
            'ls_5d': compute_metrics(ls_5d),
            'ls_1d': compute_metrics(ls_1d),
            'long_only_top3_5d': compute_metrics(long_only_5d),
            'short_only_bottom3_5d': compute_metrics(short_only_5d),
        },
        'permutation_pvalues': {
            'ls_5d': perm_ls_5d,
            'ls_1d': perm_ls_1d,
        },
        'regime_stratification': regime_ls,
        'n_basket_sensitivity': n_tests,
        'per_sector': {
            'top_frequency': top_counts,
            'bottom_frequency': bot_counts,
            'top_fwd_returns': {k: v for k, v in top_fwd_returns.items() if v},
            'bot_fwd_returns': {k: v for k, v in bot_fwd_returns.items() if v},
        },
    }


# ══════════════════════════════════════════════════════════════════════
# IDEA 3: Gap-and-Continuation vs Gap-and-Fade by Sector
# ══════════════════════════════════════════════════════════════════════
def idea3_gap_behavior(frames, spy_rets):
    """Gap continuation vs fade behavior by sector."""
    print("\n" + "="*70)
    print("IDEA 3: GAP-AND-CONTINUATION vs GAP-AND-FADE")
    print("="*70)

    GAP_THRESHOLD = 0.003  # 0.3%

    per_sector = {}
    all_continuation_returns_1d = []
    all_continuation_returns_5d = []
    all_fade_returns_1d = []
    all_fade_returns_5d = []
    all_signal_returns_1d = []
    all_signal_returns_5d = []

    for ticker, name in SECTOR_ETFS.items():
        if ticker not in frames:
            continue
        df = frames[ticker].copy()

        # Gap = open vs previous close
        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
        df['abs_gap'] = df['gap_pct'].abs()

        # Did the gap continue or fade?
        # Continue: close moved further in gap direction from open
        # Fade: close reversed back toward prev_close
        df['intraday_move'] = (df['close'] - df['open']) / df['open']
        df['gap_direction'] = np.sign(df['gap_pct'])
        df['continuation'] = df['intraday_move'] * df['gap_direction']  # positive = continued

        # Forward returns
        df['fwd_1d'] = df['close'].pct_change(1).shift(-1)
        df['fwd_5d'] = df['close'].pct_change(5).shift(-5)

        # Only look at meaningful gaps
        gap_mask = df['abs_gap'] > GAP_THRESHOLD
        gap_days = df[gap_mask].copy()

        if len(gap_days) < 20:
            print(f"  {name:25s} | Too few gaps ({len(gap_days)}), skipping")
            continue

        # Compute continuation tendency (rolling to avoid look-ahead)
        # Use ALL historical data up to each point
        continuation_rate = (gap_days['continuation'] > 0).expanding().mean()

        # For the signal, we need a LAGGED continuation tendency
        # Use 60-day rolling window of gap days
        gap_days['cont_flag'] = (gap_days['continuation'] > 0).astype(float)
        gap_days['cont_tendency_60'] = gap_days['cont_flag'].rolling(60, min_periods=20).mean()
        gap_days['cont_tendency_lag'] = gap_days['cont_tendency_60'].shift(1)  # lag to avoid look-ahead

        # Overall continuation rate
        overall_cont_rate = (gap_days['continuation'] > 0).mean()

        # Strategy:
        # If cont_tendency > 0.6: trade WITH the gap (long if gap up, short if gap down) = momentum
        # If cont_tendency < 0.4: FADE the gap (short if gap up, long if gap down) = mean reversion

        # Signal: direction to trade
        # Continuation mode: we expect gap to continue, so enter at open in gap direction
        # Our "return" is: gap_direction * fwd_return (if continuation mode)
        # Fade mode: we expect gap to reverse, so enter against gap direction
        # Our "return" is: -gap_direction * fwd_return

        cont_mask = gap_days['cont_tendency_lag'] > 0.6
        fade_mask = gap_days['cont_tendency_lag'] < 0.4

        # Continuation trade returns
        cont_days = gap_days[cont_mask]
        cont_1d = cont_days['gap_direction'] * cont_days['fwd_1d']
        cont_5d = cont_days['gap_direction'] * cont_days['fwd_5d']

        # Fade trade returns
        fade_days = gap_days[fade_mask]
        fade_1d = -fade_days['gap_direction'] * fade_days['fwd_1d']
        fade_5d = -fade_days['gap_direction'] * fade_days['fwd_5d']

        # Combined signal returns
        signal_1d = pd.concat([cont_1d, fade_1d]).dropna()
        signal_5d = pd.concat([cont_5d, fade_5d]).dropna()

        per_sector[name] = {
            'n_gap_days': int(len(gap_days)),
            'overall_continuation_rate': round(overall_cont_rate, 4),
            'n_continuation_signals': int(cont_mask.sum()),
            'n_fade_signals': int(fade_mask.sum()),
            'continuation_1d': compute_metrics(cont_1d.dropna()),
            'continuation_5d': compute_metrics(cont_5d.dropna()),
            'fade_1d': compute_metrics(fade_1d.dropna()),
            'fade_5d': compute_metrics(fade_5d.dropna()),
            'combined_1d': compute_metrics(signal_1d),
            'combined_5d': compute_metrics(signal_5d),
            # Gap size buckets
            'gap_size_stats': {
                'mean_gap_bps': round(gap_days['abs_gap'].mean() * 10000, 1),
                'median_gap_bps': round(gap_days['abs_gap'].median() * 10000, 1),
                'p90_gap_bps': round(gap_days['abs_gap'].quantile(0.9) * 10000, 1),
            }
        }

        all_continuation_returns_1d.append(cont_1d.dropna())
        all_continuation_returns_5d.append(cont_5d.dropna())
        all_fade_returns_1d.append(fade_1d.dropna())
        all_fade_returns_5d.append(fade_5d.dropna())
        all_signal_returns_1d.append(signal_1d)
        all_signal_returns_5d.append(signal_5d)

        print(f"  {name:25s} | Gaps: {len(gap_days):4d} | ContRate: {overall_cont_rate:.3f} | "
              f"ContSig: {cont_mask.sum():3d} | FadeSig: {fade_mask.sum():3d}")

    # Aggregate
    agg_cont_1d = pd.concat(all_continuation_returns_1d) if all_continuation_returns_1d else pd.Series(dtype=float)
    agg_cont_5d = pd.concat(all_continuation_returns_5d) if all_continuation_returns_5d else pd.Series(dtype=float)
    agg_fade_1d = pd.concat(all_fade_returns_1d) if all_fade_returns_1d else pd.Series(dtype=float)
    agg_fade_5d = pd.concat(all_fade_returns_5d) if all_fade_returns_5d else pd.Series(dtype=float)
    agg_signal_1d = pd.concat(all_signal_returns_1d) if all_signal_returns_1d else pd.Series(dtype=float)
    agg_signal_5d = pd.concat(all_signal_returns_5d) if all_signal_returns_5d else pd.Series(dtype=float)

    print(f"\n  AGGREGATE:")
    print(f"    Continuation trades 1d: {compute_metrics(agg_cont_1d)}")
    print(f"    Continuation trades 5d: {compute_metrics(agg_cont_5d)}")
    print(f"    Fade trades 1d: {compute_metrics(agg_fade_1d)}")
    print(f"    Fade trades 5d: {compute_metrics(agg_fade_5d)}")
    print(f"    All signals 1d: {compute_metrics(agg_signal_1d)}")
    print(f"    All signals 5d: {compute_metrics(agg_signal_5d)}")

    # Permutation tests
    perm_cont_5d = permutation_test(agg_cont_5d)
    perm_fade_5d = permutation_test(agg_fade_5d)
    perm_signal_5d = permutation_test(agg_signal_5d)
    perm_signal_1d = permutation_test(agg_signal_1d)
    print(f"\n  Permutation p-values:")
    print(f"    Continuation 5d: {perm_cont_5d}")
    print(f"    Fade 5d: {perm_fade_5d}")
    print(f"    All signals 5d: {perm_signal_5d}")
    print(f"    All signals 1d: {perm_signal_1d}")

    # Regime
    regime_signal = regime_stratify(agg_signal_5d, spy_rets)

    # Test gap size sensitivity
    gap_size_tests = {}
    for gap_thresh in [0.002, 0.003, 0.005, 0.008, 0.01]:
        all_r = []
        for ticker in SECTOR_ETFS:
            if ticker not in frames:
                continue
            df = frames[ticker].copy()
            df['prev_close'] = df['close'].shift(1)
            df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
            df['abs_gap'] = df['gap_pct'].abs()
            df['intraday_move'] = (df['close'] - df['open']) / df['open']
            df['gap_direction'] = np.sign(df['gap_pct'])
            df['continuation'] = df['intraday_move'] * df['gap_direction']
            df['fwd_5d'] = df['close'].pct_change(5).shift(-5)

            gap_mask = df['abs_gap'] > gap_thresh
            gap_days = df[gap_mask].copy()
            if len(gap_days) < 20:
                continue

            gap_days['cont_flag'] = (gap_days['continuation'] > 0).astype(float)
            gap_days['cont_tendency_60'] = gap_days['cont_flag'].rolling(60, min_periods=20).mean()
            gap_days['cont_tendency_lag'] = gap_days['cont_tendency_60'].shift(1)

            cont_mask = gap_days['cont_tendency_lag'] > 0.6
            fade_mask = gap_days['cont_tendency_lag'] < 0.4

            cont_r = gap_days.loc[cont_mask, 'gap_direction'] * gap_days.loc[cont_mask, 'fwd_5d']
            fade_r = -gap_days.loc[fade_mask, 'gap_direction'] * gap_days.loc[fade_mask, 'fwd_5d']
            all_r.extend(cont_r.dropna().tolist())
            all_r.extend(fade_r.dropna().tolist())

        gap_size_tests[f'gap_{gap_thresh*100:.1f}pct'] = compute_metrics(pd.Series(all_r))

    print(f"\n  Gap threshold sensitivity (combined signal, 5d):")
    for k, v in gap_size_tests.items():
        print(f"    {k}: n={v['n_trades']}, sharpe={v['sharpe']}, wr={v['wr']}")

    return {
        'aggregate': {
            'continuation_1d': compute_metrics(agg_cont_1d),
            'continuation_5d': compute_metrics(agg_cont_5d),
            'fade_1d': compute_metrics(agg_fade_1d),
            'fade_5d': compute_metrics(agg_fade_5d),
            'combined_signal_1d': compute_metrics(agg_signal_1d),
            'combined_signal_5d': compute_metrics(agg_signal_5d),
        },
        'permutation_pvalues': {
            'continuation_5d': perm_cont_5d,
            'fade_5d': perm_fade_5d,
            'combined_signal_5d': perm_signal_5d,
            'combined_signal_1d': perm_signal_1d,
        },
        'regime_stratification': regime_signal,
        'gap_threshold_sensitivity': gap_size_tests,
        'per_sector': per_sector,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("ETF Fund Flow & Price-Volume Microstructure Signal Research")
    print("=" * 70)

    # Download data
    raw_data = download_data()
    frames = build_sector_frames(raw_data)

    print(f"\nLoaded {len(frames)} tickers")
    for t, df in frames.items():
        print(f"  {t}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

    # SPY returns for regime stratification
    spy_rets = frames[BENCHMARK]['ret']

    # Run all three ideas
    results = {}
    results['idea1_volume_surprise'] = idea1_volume_surprise(frames, spy_rets)
    results['idea2_flow_rotation'] = idea2_flow_rotation(frames, spy_rets)
    results['idea3_gap_behavior'] = idea3_gap_behavior(frames, spy_rets)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {}

    # Idea 1
    i1 = results['idea1_volume_surprise']
    i1_comb = i1['aggregate']['combined_5d']
    i1_perm = i1['permutation_pvalues']['combined_5d']
    summary['idea1'] = {
        'name': 'Volume Surprise Entry Timing',
        'verdict': 'SIGNIFICANT' if i1_perm < 0.05 and i1_comb['sharpe'] > 0.3 else 'WEAK' if i1_perm < 0.1 else 'NO EDGE',
        'combined_5d_sharpe': i1_comb['sharpe'],
        'combined_5d_wr': i1_comb['wr'],
        'combined_5d_pf': i1_comb['pf'],
        'combined_5d_n': i1_comb['n_trades'],
        'perm_pvalue': i1_perm,
    }

    # Idea 2
    i2 = results['idea2_flow_rotation']
    i2_ls = i2['aggregate']['ls_5d']
    i2_perm = i2['permutation_pvalues']['ls_5d']
    summary['idea2'] = {
        'name': 'Cross-Sector Flow Rotation',
        'verdict': 'SIGNIFICANT' if i2_perm < 0.05 and i2_ls['sharpe'] > 0.3 else 'WEAK' if i2_perm < 0.1 else 'NO EDGE',
        'ls_5d_sharpe': i2_ls['sharpe'],
        'ls_5d_wr': i2_ls['wr'],
        'ls_5d_pf': i2_ls['pf'],
        'ls_5d_n': i2_ls['n_trades'],
        'perm_pvalue': i2_perm,
    }

    # Idea 3
    i3 = results['idea3_gap_behavior']
    i3_comb = i3['aggregate']['combined_signal_5d']
    i3_perm = i3['permutation_pvalues']['combined_signal_5d']
    summary['idea3'] = {
        'name': 'Gap Continuation/Fade by Sector',
        'verdict': 'SIGNIFICANT' if i3_perm < 0.05 and i3_comb['sharpe'] > 0.3 else 'WEAK' if i3_perm < 0.1 else 'NO EDGE',
        'combined_5d_sharpe': i3_comb['sharpe'],
        'combined_5d_wr': i3_comb['wr'],
        'combined_5d_pf': i3_comb['pf'],
        'combined_5d_n': i3_comb['n_trades'],
        'perm_pvalue': i3_perm,
    }

    results['summary'] = summary
    results['metadata'] = {
        'run_date': datetime.now().isoformat(),
        'data_range': f"{START} to {END}",
        'n_sectors': len(SECTOR_ETFS),
        'sector_etfs': SECTOR_ETFS,
        'n_permutations': N_PERM,
        'gap_threshold': 0.003,
    }

    print(f"\n  Idea 1 (Volume Surprise):    {summary['idea1']['verdict']} | Sharpe={summary['idea1']['combined_5d_sharpe']} | WR={summary['idea1']['combined_5d_wr']} | p={summary['idea1']['perm_pvalue']}")
    print(f"  Idea 2 (Flow Rotation):      {summary['idea2']['verdict']} | Sharpe={summary['idea2']['ls_5d_sharpe']} | WR={summary['idea2']['ls_5d_wr']} | p={summary['idea2']['perm_pvalue']}")
    print(f"  Idea 3 (Gap Cont/Fade):      {summary['idea3']['verdict']} | Sharpe={summary['idea3']['combined_5d_sharpe']} | WR={summary['idea3']['combined_5d_wr']} | p={summary['idea3']['perm_pvalue']}")

    # Handle inf/nan for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (float, np.floating)):
            if np.isinf(obj) or np.isnan(obj):
                return None
            return round(float(obj), 6)
        elif isinstance(obj, (int, np.integer)):
            return int(obj)
        return obj

    # Save results
    out_path = '/home/jupiter/Lvl3Quant/research/microstructure_signals_results.json'
    with open(out_path, 'w') as f:
        json.dump(clean_for_json(results), f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    return results


if __name__ == '__main__':
    main()
