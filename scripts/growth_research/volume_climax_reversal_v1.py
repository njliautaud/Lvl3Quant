#!/usr/bin/env python3
"""
Volume Climax Reversal Strategy v1
====================================
OBSERVATION (HC #735):
  Extremely high volume days (>3x 20-day average) often mark capitulation or
  distribution turning points. This is fundamentally different from exhaustion
  selling (multi-day pattern) — this is a single-day volume spike event.

HYPOTHESIS:
  After a volume climax day + price drop (capitulation), buying captures a
  mean-reversion bounce. After a volume climax + price surge (distribution),
  the move is often exhausted.

  We test BUY after capitulation (volume spike + down day) only, since our
  prior research shows short-side mean reversion strategies in equities are
  fragile.

VARIANTS: 36 combinations
  - Volume multiplier: 2x, 3x, 4x (of 20-day avg volume)
  - Price drop threshold: 1%, 2%, 3%, 5% (intraday or close-to-close)
  - Vol compression filter: with / without (20d realized vol < 10th pctl of 252d history)
  - Hold period: 5d, 10d, 21d

Wait — that's 3 × 4 × 2 × 3 = 72 combos, but user said ~36. Let's use:
  - Volume multiplier: 2x, 3x, 4x
  - Price drop threshold: 2%, 3%, 5%  (skip 1% as too noisy based on prior research)
  - Vol compression filter: with / without
  - Hold period: 5d, 10d  (skip 21d per user spec saying 5/10/21 but 36 combos = 3×3×2×2)

Actually user said: "3 vol multipliers × 3 drop thresholds × 2 vol filters × 2 hold periods"
So: 2x/3x/4x × 2%/3%/5% × with/without × 5d/10d = 36 combos. But also test 21d separately.
Let's do 3×3×2×2=36 with hold=5d,10d plus report 21d for the best variant.

UNIVERSE: S&P 500 stocks (yfinance, 10+ years)
VALIDATION: Permutation (200 shuffles, p<0.05), Regime gap <0.50, Per-year >60% profitable
WINDOW: SLIDING (HC #0)
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import json
import itertools
import time

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    print(f"GPU available: {HAS_GPU}")
    if HAS_GPU:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except:
    HAS_GPU = False

# ─── Configuration ───
START_DATE = '2014-01-01'
END_DATE = '2026-07-15'
VOL_LOOKBACK = 20       # 20-day average volume
VOL_HIST_LOOKBACK = 21  # 21-day realized vol
VOL_PCTL_HISTORY = 252  # 1-year for percentile
COMPRESSION_PCT = 10    # 10th percentile = compressed

VOL_MULTIPLIERS = [2, 3, 4]
DROP_THRESHOLDS = [0.02, 0.03, 0.05]  # 2%, 3%, 5%
VOL_FILTERS = [False, True]  # Without / with vol compression
HOLD_PERIODS = [5, 10]
EXTRA_HOLD = 21  # Test on best variant only

N_PERMS = 200
MIN_TRADES = 30
MIN_YEAR_PCT = 0.60  # Profitable in >60% of years

OUTPUT_DIR = '/home/nick/Lvl3Quant/output/volume_climax_reversal_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"{'='*70}")
print(f"VOLUME CLIMAX REVERSAL STRATEGY v1")
print(f"Observation-first research (HC #735)")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}")

# ─── Step 1: Get S&P 500 universe ───
print("\n[1/6] Building S&P 500 universe...")
import yfinance as yf

sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
try:
    tables = pd.read_html(sp500_url)
    sp500_tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
    print(f"  Got {len(sp500_tickers)} S&P 500 tickers from Wikipedia")
except Exception as e:
    print(f"  Wikipedia fetch failed ({e}), using hardcoded top 200")
    sp500_tickers = [
        'AAPL','MSFT','AMZN','NVDA','GOOGL','META','TSLA','BRK-B','UNH','JNJ',
        'V','XOM','JPM','WMT','PG','MA','HD','CVX','MRK','ABBV','LLY','PEP',
        'KO','COST','AVGO','TMO','MCD','CSCO','ACN','ABT','DHR','WFC','NEE',
        'DIS','TXN','BMY','PM','UNP','UPS','RTX','AMGN','IBM','COP','LOW','GS',
        'SPGI','MS','BLK','AXP','INTC','CAT','BA','GE','AMD','ISRG','MDLZ',
        'ADI','GILD','BKNG','SYK','ADP','VRTX','TJX','MMC','PGR','CB','REGN',
        'SCHW','ZTS','CI','EOG','SLB','MO','BSX','CME','DUK','SO','LRCX',
        'BDX','CL','FI','ANET','ITW','SHW','APD','PLD','MCK','PNC','USB',
        'HUM','TGT','ICE','SNPS','CDNS','PSA','ORLY','AON','FCX','OXY',
        'AZO','KMB','CMG','MPC','VLO','EMR','SRE','AIG','ALL','D','KHC',
        'SPG','AFL','WMB','PSX','F','GM','CTSH','FDX','NEM','EL','CCI',
        'TFC','WELL','YUM','MNST','HSY','KEYS','NDAQ','WAB','ROP','GIS',
        'DG','DLTR','HCA','FAST','CTAS','PAYX','ON','WEC','AEP','XEL',
        'EXC','ED','AWK','ES','ETR','FE','PPL','CNP','CMS','EVRG',
        'ORCL','CRM','NOW','INTU','ADBE','PYPL','NFLX','ABNB','UBER','DASH',
        'NKE','SBUX','LULU','ETSY','LMT','NOC','GD','HII','TDG','LHX',
        'WM','RSG','VRSK','MSCI','IEX','POOL','WST','IDXX','MTD','TECH',
        'A','WAT','PKI','BIO','HOLX','ALGN','TFX','STE','EW','ZBH',
        'MDT','BAX','BDX','BSX','ABT','SYK','DXCM','PODD','ISRG','GEHC',
    ]

# Download SPY for regime classification
print("  Downloading SPY for regime classification...")
spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy_ret = spy['Close'].pct_change()
print(f"  SPY: {len(spy)} days")

# Download stock data in batches
print(f"  Downloading {len(sp500_tickers)} stocks (batched)...")
BATCH_SIZE = 50
all_close = {}
all_volume = {}
all_high = {}
all_low = {}
all_open = {}
failed = []

for i in range(0, len(sp500_tickers), BATCH_SIZE):
    batch = sp500_tickers[i:i+BATCH_SIZE]
    batch_str = ' '.join(batch)
    try:
        data = yf.download(batch_str, start=START_DATE, end=END_DATE,
                          progress=False, group_by='ticker', threads=True)
        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = data
                else:
                    df = data[ticker]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df.dropna()) > 252:  # Need at least 1 year
                    all_close[ticker] = df['Close'].dropna()
                    all_volume[ticker] = df['Volume'].dropna()
                    all_high[ticker] = df['High'].dropna()
                    all_low[ticker] = df['Low'].dropna()
                    all_open[ticker] = df['Open'].dropna()
            except:
                failed.append(ticker)
    except:
        failed.extend(batch)

    done = min(i + BATCH_SIZE, len(sp500_tickers))
    print(f"  Progress: {done}/{len(sp500_tickers)} ({len(all_close)} loaded, {len(failed)} failed)")
    time.sleep(0.5)

print(f"  Final universe: {len(all_close)} stocks")

# ─── Step 2: Identify volume climax events ───
print(f"\n[2/6] Scanning for volume climax events across all parameter combos...")

close_df = pd.DataFrame(all_close)
volume_df = pd.DataFrame(all_volume)

# Align indices
common_idx = close_df.index.intersection(volume_df.index).intersection(spy_ret.index)
close_df = close_df.loc[common_idx]
volume_df = volume_df.loc[common_idx]

# Pre-compute shared quantities
print("  Computing daily returns...")
returns_df = close_df.pct_change()

print("  Computing rolling volume averages...")
vol_avg_20d = volume_df.rolling(VOL_LOOKBACK, min_periods=15).mean()

print("  Computing volume ratios...")
vol_ratio = volume_df / vol_avg_20d

print("  Computing realized volatility for compression filter...")
realized_vol = returns_df.rolling(VOL_HIST_LOOKBACK, min_periods=15).std()
vol_pctl = realized_vol.rolling(VOL_PCTL_HISTORY, min_periods=126).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
)
is_vol_compressed = vol_pctl < (COMPRESSION_PCT / 100.0)

print("  Computing SPY regime...")
spy_daily_ret = spy_ret.reindex(common_idx)

# Pre-compute forward returns for all hold periods
print("  Computing forward returns...")
fwd_returns = {}
for hp in HOLD_PERIODS + [EXTRA_HOLD]:
    fwd_returns[hp] = close_df.shift(-hp) / close_df - 1

# ─── Step 3: Generate trades for all variants ───
print(f"\n[3/6] Generating trades for 36 variant combinations...")

all_variants = []
variant_id = 0

for vol_mult, drop_thresh, use_vol_filter, hold_days in itertools.product(
    VOL_MULTIPLIERS, DROP_THRESHOLDS, VOL_FILTERS, HOLD_PERIODS
):
    variant_id += 1
    label = f"vol{vol_mult}x_drop{int(drop_thresh*100)}pct_vf{'Y' if use_vol_filter else 'N'}_hold{hold_days}d"

    # Signal: volume > vol_mult * 20d avg AND price drop > drop_thresh
    volume_spike = vol_ratio > vol_mult
    price_drop = returns_df < -drop_thresh

    # Combined signal
    signal = volume_spike & price_drop

    # Optional vol compression filter
    if use_vol_filter:
        signal = signal & is_vol_compressed

    # Extract trades
    trades = []
    for ticker in close_df.columns:
        if ticker not in fwd_returns[hold_days].columns:
            continue
        sig = signal[ticker].dropna()
        fwd = fwd_returns[hold_days][ticker]
        spy_r = spy_daily_ret

        trigger_dates = sig[sig == True].index
        for dt in trigger_dates:
            if dt not in fwd.index or pd.isna(fwd.loc[dt]):
                continue
            ret = fwd.loc[dt]
            spy_on_day = spy_r.loc[dt] if dt in spy_r.index else 0
            trades.append({
                'ticker': ticker,
                'entry_date': dt,
                'return': ret,
                'spy_ret': spy_on_day,
                'year': dt.year,
            })

    n_trades = len(trades)

    # Compute metrics if enough trades
    if n_trades < MIN_TRADES:
        all_variants.append({
            'variant_id': variant_id,
            'label': label,
            'vol_mult': vol_mult,
            'drop_thresh': drop_thresh,
            'vol_filter': use_vol_filter,
            'hold_days': hold_days,
            'n_trades': n_trades,
            'status': 'SKIP_FEW_TRADES',
        })
        print(f"  [{variant_id:2d}/36] {label}: {n_trades} trades — SKIP")
        continue

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values('entry_date')

    # ── Walk-forward sliding window validation ──
    # Group by year. Use 3-year train, 1-year test, slide by 1 year.
    years = sorted(trades_df['year'].unique())
    TRAIN_YEARS = 3
    oot_returns = []

    for test_year_idx in range(TRAIN_YEARS, len(years)):
        test_year = years[test_year_idx]
        train_years = years[test_year_idx - TRAIN_YEARS:test_year_idx]

        train_trades = trades_df[trades_df['year'].isin(train_years)]
        test_trades = trades_df[trades_df['year'] == test_year]

        if len(train_trades) < 10 or len(test_trades) < 3:
            continue

        # In-sample: check if strategy has positive expectancy
        train_mean = train_trades['return'].mean()
        if train_mean > 0:
            # Strategy is active in OOT — collect OOT returns
            oot_returns.extend(test_trades['return'].tolist())

    if len(oot_returns) < MIN_TRADES:
        all_variants.append({
            'variant_id': variant_id,
            'label': label,
            'vol_mult': vol_mult,
            'drop_thresh': drop_thresh,
            'vol_filter': use_vol_filter,
            'hold_days': hold_days,
            'n_trades': n_trades,
            'oot_trades': len(oot_returns),
            'status': 'SKIP_FEW_OOT',
        })
        print(f"  [{variant_id:2d}/36] {label}: {n_trades} total, {len(oot_returns)} OOT — SKIP")
        continue

    oot_arr = np.array(oot_returns)

    # Basic metrics on OOT
    mean_ret = oot_arr.mean()
    win_rate = (oot_arr > 0).mean()

    # Sharpe (annualized, approximate)
    std_ret = oot_arr.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252 / hold_days) if std_ret > 0 else 0

    # Sortino
    downside = oot_arr[oot_arr < 0]
    sortino_denom = downside.std() if len(downside) > 2 else std_ret
    sortino = (mean_ret / sortino_denom) * np.sqrt(252 / hold_days) if sortino_denom > 0 else 0

    # Profit factor
    gross_profit = oot_arr[oot_arr > 0].sum()
    gross_loss = abs(oot_arr[oot_arr < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Regime analysis (HC #428 R1)
    oot_trades_df = trades_df[trades_df['return'].isin(oot_returns)].copy() if len(oot_returns) < len(trades_df) else trades_df.copy()
    # More robust: rebuild OOT trades
    # Actually let's tag OOT trades properly
    oot_indices = []
    for test_year_idx in range(TRAIN_YEARS, len(years)):
        test_year = years[test_year_idx]
        train_years_list = years[test_year_idx - TRAIN_YEARS:test_year_idx]
        train_trades = trades_df[trades_df['year'].isin(train_years_list)]
        if train_trades['return'].mean() > 0:
            test_mask = trades_df['year'] == test_year
            oot_indices.extend(trades_df[test_mask].index.tolist())

    oot_df = trades_df.loc[oot_indices].copy()

    # Regime: SPY up = GREEN, SPY down = RED
    oot_df['regime'] = oot_df['spy_ret'].apply(
        lambda x: 'GREEN' if x > 0 else 'RED'
    )

    regime_sharpes = {}
    for regime in ['GREEN', 'RED']:
        r_trades = oot_df[oot_df['regime'] == regime]
        if len(r_trades) > 5:
            r_mean = r_trades['return'].mean()
            r_std = r_trades['return'].std()
            if r_std > 0:
                regime_sharpes[regime] = (r_mean / r_std) * np.sqrt(252 / hold_days)
            else:
                regime_sharpes[regime] = 0
        else:
            regime_sharpes[regime] = np.nan

    s_green = regime_sharpes.get('GREEN', 0)
    s_red = regime_sharpes.get('RED', 0)
    if pd.notna(s_green) and pd.notna(s_red):
        max_s = max(abs(s_green), abs(s_red))
        regime_gap = abs(s_green - s_red) / max_s if max_s > 0 else 0
    else:
        regime_gap = np.nan

    # Per-year consistency
    year_rets = oot_df.groupby('year')['return'].mean()
    pct_years_profitable = (year_rets > 0).mean() if len(year_rets) > 0 else 0

    # Gates
    regime_pass = pd.notna(regime_gap) and regime_gap < 0.50
    year_pass = pct_years_profitable >= MIN_YEAR_PCT

    variant_result = {
        'variant_id': variant_id,
        'label': label,
        'vol_mult': vol_mult,
        'drop_thresh': drop_thresh,
        'vol_filter': use_vol_filter,
        'hold_days': hold_days,
        'n_trades': n_trades,
        'oot_trades': len(oot_df),
        'mean_ret': float(mean_ret),
        'win_rate': float(win_rate),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'profit_factor': float(pf) if pf != float('inf') else 999,
        'regime_sharpe_green': float(s_green) if pd.notna(s_green) else None,
        'regime_sharpe_red': float(s_red) if pd.notna(s_red) else None,
        'regime_gap': float(regime_gap) if pd.notna(regime_gap) else None,
        'regime_pass': regime_pass,
        'pct_years_profitable': float(pct_years_profitable),
        'year_pass': year_pass,
        'year_rets': {str(k): float(v) for k, v in year_rets.items()},
        'status': 'EVALUATED',
    }
    all_variants.append(variant_result)

    rg = f"RG={regime_gap:.2f}" if pd.notna(regime_gap) else "RG=N/A"
    print(f"  [{variant_id:2d}/36] {label}: Sharpe {sharpe:.2f}, WR {win_rate:.0%}, "
          f"PF {pf:.2f}, {rg} {'✓' if regime_pass else '✗'}, "
          f"Yr {pct_years_profitable:.0%} {'✓' if year_pass else '✗'}, "
          f"n={len(oot_df)}")

# ─── Step 4: Filter candidates ───
print(f"\n[4/6] Filtering candidates (Regime gap < 0.50, Years profitable > 60%)...")

candidates = [v for v in all_variants
               if v.get('status') == 'EVALUATED'
               and v.get('regime_pass', False)
               and v.get('year_pass', False)
               and v.get('sharpe', 0) > 0.3]

candidates.sort(key=lambda x: x.get('sharpe', 0), reverse=True)

print(f"  {len(candidates)} variants pass regime + year consistency gates")
for c in candidates[:10]:
    print(f"    {c['label']}: Sharpe {c['sharpe']:.2f}, WR {c['win_rate']:.0%}, "
          f"PF {c['profit_factor']:.2f}, RG {c['regime_gap']:.2f}, "
          f"Yr {c['pct_years_profitable']:.0%}")

# ─── Step 5: Permutation test on top candidates ───
print(f"\n[5/6] Running permutation tests on top {min(len(candidates), 10)} candidates...")

final_results = []

for cidx, cand in enumerate(candidates[:10]):
    vol_mult = cand['vol_mult']
    drop_thresh = cand['drop_thresh']
    use_vol_filter = cand['vol_filter']
    hold_days = cand['hold_days']

    # Rebuild trades for this variant
    volume_spike = vol_ratio > vol_mult
    price_drop = returns_df < -drop_thresh
    signal = volume_spike & price_drop
    if use_vol_filter:
        signal = signal & is_vol_compressed

    trades = []
    for ticker in close_df.columns:
        if ticker not in fwd_returns[hold_days].columns:
            continue
        sig = signal[ticker].dropna()
        fwd = fwd_returns[hold_days][ticker]
        trigger_dates = sig[sig == True].index
        for dt in trigger_dates:
            if dt not in fwd.index or pd.isna(fwd.loc[dt]):
                continue
            spy_on_day = spy_daily_ret.loc[dt] if dt in spy_daily_ret.index else 0
            trades.append({
                'ticker': ticker,
                'entry_date': dt,
                'return': fwd.loc[dt],
                'spy_ret': spy_on_day,
                'year': dt.year,
            })

    trades_df_c = pd.DataFrame(trades).sort_values('entry_date')
    years = sorted(trades_df_c['year'].unique())
    TRAIN_YEARS = 3

    # Rebuild OOT
    oot_indices = []
    for test_year_idx in range(TRAIN_YEARS, len(years)):
        test_year = years[test_year_idx]
        train_years_list = years[test_year_idx - TRAIN_YEARS:test_year_idx]
        train_trades = trades_df_c[trades_df_c['year'].isin(train_years_list)]
        if train_trades['return'].mean() > 0:
            test_mask = trades_df_c['year'] == test_year
            oot_indices.extend(trades_df_c[test_mask].index.tolist())

    oot_df = trades_df_c.loc[oot_indices].copy()
    oot_rets = oot_df['return'].values

    if len(oot_rets) < MIN_TRADES:
        continue

    observed_mean = oot_rets.mean()
    observed_sharpe = (observed_mean / oot_rets.std()) * np.sqrt(252 / hold_days) if oot_rets.std() > 0 else 0

    # Permutation test: shuffle return signs
    perm_sharpes = []
    for perm_i in range(N_PERMS):
        shuffled = oot_rets.copy()
        # Random sign flip — tests if direction matters
        signs = np.random.choice([-1, 1], size=len(shuffled))
        shuffled = shuffled * signs
        s_std = shuffled.std()
        if s_std > 0:
            perm_sharpes.append((shuffled.mean() / s_std) * np.sqrt(252 / hold_days))

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= observed_sharpe).mean()

    perm_pass = p_value < 0.05

    cand['perm_p_value'] = float(p_value)
    cand['perm_null_mean'] = float(perm_sharpes.mean())
    cand['perm_null_std'] = float(perm_sharpes.std())
    cand['perm_pass'] = perm_pass
    cand['observed_sharpe_oot'] = float(observed_sharpe)

    status = "PASS" if perm_pass else "FAIL"
    print(f"  [{cidx+1}] {cand['label']}: Sharpe {observed_sharpe:.2f}, "
          f"p={p_value:.3f} → {status}")

    if perm_pass:
        final_results.append(cand)

# ─── Step 6: Summary and extended analysis on best ───
print(f"\n[6/6] Final results...")
print(f"{'='*70}")
print(f"VOLUME CLIMAX REVERSAL v1 — SUMMARY")
print(f"{'='*70}")

print(f"\nTotal variants tested: 36")
print(f"Pass regime gap (<0.50): {sum(1 for v in all_variants if v.get('regime_pass'))}")
print(f"Pass year consistency (>60%): {sum(1 for v in all_variants if v.get('year_pass'))}")
print(f"Pass both gates: {len(candidates)}")
print(f"Pass permutation test (p<0.05): {len(final_results)}")

if final_results:
    print(f"\n── PASSING VARIANTS ──")
    for r in final_results:
        print(f"\n  {r['label']}")
        print(f"    OOT trades: {r['oot_trades']}")
        print(f"    Sharpe: {r['sharpe']:.2f}  |  Sortino: {r['sortino']:.2f}")
        print(f"    Win rate: {r['win_rate']:.0%}  |  PF: {r['profit_factor']:.2f}")
        print(f"    Regime gap: {r['regime_gap']:.2f} (GREEN={r['regime_sharpe_green']:.2f}, RED={r['regime_sharpe_red']:.2f})")
        print(f"    Years profitable: {r['pct_years_profitable']:.0%}")
        print(f"    Permutation p-value: {r['perm_p_value']:.3f}")
        print(f"    Per-year returns:")
        for yr, ret in sorted(r.get('year_rets', {}).items()):
            marker = "+" if ret > 0 else "-"
            print(f"      {yr}: {ret*100:+.2f}% {marker}")

    # Extended analysis on best variant: test 21d hold too
    best = final_results[0]
    print(f"\n── BEST VARIANT EXTENDED (21d hold) ──")
    vol_mult = best['vol_mult']
    drop_thresh = best['drop_thresh']
    use_vol_filter = best['vol_filter']

    volume_spike = vol_ratio > vol_mult
    price_drop = returns_df < -drop_thresh
    signal = volume_spike & price_drop
    if use_vol_filter:
        signal = signal & is_vol_compressed

    trades_21 = []
    for ticker in close_df.columns:
        if ticker not in fwd_returns[EXTRA_HOLD].columns:
            continue
        sig = signal[ticker].dropna()
        fwd = fwd_returns[EXTRA_HOLD][ticker]
        trigger_dates = sig[sig == True].index
        for dt in trigger_dates:
            if dt not in fwd.index or pd.isna(fwd.loc[dt]):
                continue
            trades_21.append({'return': fwd.loc[dt], 'year': dt.year})

    if trades_21:
        t21 = pd.DataFrame(trades_21)
        m = t21['return'].mean()
        s = t21['return'].std()
        sh21 = (m / s) * np.sqrt(252 / EXTRA_HOLD) if s > 0 else 0
        wr21 = (t21['return'] > 0).mean()
        print(f"    21d hold: Sharpe {sh21:.2f}, WR {wr21:.0%}, n={len(t21)}")
else:
    print(f"\n  NO VARIANTS PASS ALL THREE GATES.")
    print(f"  Volume climax reversal does not appear to be a robust tradeable signal.")
    print(f"\n  Closest variants (by Sharpe, pre-permutation):")
    for v in candidates[:5]:
        print(f"    {v['label']}: Sharpe {v['sharpe']:.2f}, "
              f"RG {v.get('regime_gap', 'N/A')}, Yr {v.get('pct_years_profitable', 0):.0%}")

# ─── Save results ───
summary = {
    'strategy': 'volume_climax_reversal_v1',
    'observation': 'Single-day volume spike (>Nx 20d avg) + price drop = capitulation reversal',
    'universe': f'{len(all_close)} S&P 500 stocks',
    'date_range': f'{START_DATE} to {END_DATE}',
    'variants_tested': len(all_variants),
    'pass_regime': sum(1 for v in all_variants if v.get('regime_pass')),
    'pass_year': sum(1 for v in all_variants if v.get('year_pass')),
    'pass_both_gates': len(candidates),
    'pass_permutation': len(final_results),
    'all_variants': all_variants,
    'passing_variants': final_results,
    'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}

# Clean up for JSON serialization
def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj

summary = make_serializable(summary)

with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}/results.json")
print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
