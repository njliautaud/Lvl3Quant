#!/usr/bin/env python3
"""
Relative Strength Breakout from Vol Compression v1
====================================================
OBSERVATION:
  Stocks that break out of a vol-compressed range with above-average relative
  strength vs their sector tend to continue. This is the MOMENTUM side of vol
  compression — instead of buying weakness (contrarian), buy STRENGTH when it
  emerges from compression.

SIGNAL:
  1. Stock's 20d realized vol < Nth percentile of its trailing 252d vol history
  2. Stock breaks above its 20d high (long) OR below 20d low (short)
  3. Stock's 5d return exceeds sector ETF's 5d return by >X%

VARIANTS (~24):
  - Vol percentile: 10th, 20th
  - RS excess: >0.5%, >1%, >2%
  - Hold: 5d, 10d, 21d
  - Direction: long, short

VALIDATION: permutation (200 shuffles, p<0.05), regime gap (<0.50), year consistency (>60%)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/nick/Lvl3Quant/output/rs_breakout_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = '2015-01-01'
END_DATE = '2026-07-01'
VOL_LOOKBACK = 20
VOL_HISTORY = 252
BREAKOUT_LOOKBACK = 20
RS_LOOKBACK = 5
N_PERMUTATIONS = 200
MIN_TRADES = 30

VOL_PCTILE_THRESHOLDS = [10, 20]
RS_EXCESS_THRESHOLDS = [0.005, 0.01, 0.02]
HOLD_PERIODS = [5, 10, 21]
DIRECTIONS = ['long', 'short']

n_variants = len(VOL_PCTILE_THRESHOLDS)*len(RS_EXCESS_THRESHOLDS)*len(HOLD_PERIODS)*len(DIRECTIONS)
print(f"{'='*70}")
print(f"RELATIVE STRENGTH BREAKOUT FROM VOL COMPRESSION v1")
print(f"Variants: {n_variants}")
print(f"{'='*70}")
sys.stdout.flush()

# ─── Sector mapping ───
SECTOR_MAP = {
    'AAPL':'XLK','MSFT':'XLK','NVDA':'XLK','AVGO':'XLK','ADBE':'XLK',
    'CRM':'XLK','CSCO':'XLK','ACN':'XLK','ORCL':'XLK','IBM':'XLK',
    'INTC':'XLK','AMD':'XLK','QCOM':'XLK','TXN':'XLK','INTU':'XLK',
    'AMAT':'XLK','MU':'XLK','ADI':'XLK','LRCX':'XLK','KLAC':'XLK',
    'SNPS':'XLK','CDNS':'XLK','MCHP':'XLK','MSI':'XLK','FTNT':'XLK',
    'HPQ':'XLK','KEYS':'XLK','ON':'XLK','MPWR':'XLK','NXPI':'XLK',
    'GEN':'XLK','FSLR':'XLK','ANET':'XLK','NOW':'XLK','PANW':'XLK',
    'JPM':'XLF','BAC':'XLF','GS':'XLF','MS':'XLF','WFC':'XLF',
    'C':'XLF','BLK':'XLF','SCHW':'XLF','AXP':'XLF','CME':'XLF',
    'ICE':'XLF','CB':'XLF','PGR':'XLF','AON':'XLF',
    'AIG':'XLF','MET':'XLF','TFC':'XLF','USB':'XLF','PNC':'XLF',
    'COF':'XLF','STT':'XLF','FITB':'XLF','MTB':'XLF',
    'XOM':'XLE','CVX':'XLE','COP':'XLE','SLB':'XLE','EOG':'XLE',
    'MPC':'XLE','PSX':'XLE','VLO':'XLE','OXY':'XLE',
    'WMB':'XLE','HAL':'XLE','DVN':'XLE','FANG':'XLE','BKR':'XLE',
    'UNH':'XLV','JNJ':'XLV','LLY':'XLV','PFE':'XLV','ABT':'XLV',
    'TMO':'XLV','MRK':'XLV','ABBV':'XLV','DHR':'XLV','BMY':'XLV',
    'AMGN':'XLV','MDT':'XLV','GILD':'XLV','ISRG':'XLV','CVS':'XLV',
    'ELV':'XLV','SYK':'XLV','CI':'XLV','REGN':'XLV','VRTX':'XLV',
    'BSX':'XLV','ZTS':'XLV','BDX':'XLV','HUM':'XLV','MCK':'XLV',
    'EW':'XLV','A':'XLV','DXCM':'XLV','IQV':'XLV','IDXX':'XLV',
    'AMZN':'XLY','TSLA':'XLY','HD':'XLY','MCD':'XLY','NKE':'XLY',
    'LOW':'XLY','SBUX':'XLY','TJX':'XLY','BKNG':'XLY','CMG':'XLY',
    'MAR':'XLY','GM':'XLY','F':'XLY','ORLY':'XLY','AZO':'XLY',
    'ROST':'XLY','DHI':'XLY','LEN':'XLY','YUM':'XLY','DPZ':'XLY',
    'PG':'XLP','KO':'XLP','PEP':'XLP','COST':'XLP','WMT':'XLP',
    'PM':'XLP','MO':'XLP','CL':'XLP','MDLZ':'XLP','KMB':'XLP',
    'GIS':'XLP','STZ':'XLP','SYY':'XLP','KHC':'XLP','HSY':'XLP',
    'KDP':'XLP','ADM':'XLP','EL':'XLP','MKC':'XLP',
    'CAT':'XLI','UNP':'XLI','HON':'XLI','UPS':'XLI','BA':'XLI',
    'RTX':'XLI','DE':'XLI','LMT':'XLI','GE':'XLI','MMM':'XLI',
    'GD':'XLI','NOC':'XLI','WM':'XLI','ITW':'XLI','EMR':'XLI',
    'FDX':'XLI','CSX':'XLI','NSC':'XLI','PCAR':'XLI','TT':'XLI',
    'PH':'XLI','CTAS':'XLI','ROK':'XLI','FAST':'XLI','JCI':'XLI',
    'LIN':'XLB','APD':'XLB','SHW':'XLB','FCX':'XLB','ECL':'XLB',
    'NUE':'XLB','NEM':'XLB','DOW':'XLB','DD':'XLB','VMC':'XLB',
    'MLM':'XLB','PPG':'XLB','ALB':'XLB','CF':'XLB','CTVA':'XLB',
    'GOOG':'XLC','GOOGL':'XLC','META':'XLC','DIS':'XLC','NFLX':'XLC',
    'CMCSA':'XLC','VZ':'XLC','T':'XLC','TMUS':'XLC','CHTR':'XLC',
    'EA':'XLC','TTWO':'XLC',
    'PLD':'XLRE','AMT':'XLRE','CCI':'XLRE','EQIX':'XLRE','PSA':'XLRE',
    'SPG':'XLRE','O':'XLRE','WELL':'XLRE','DLR':'XLRE','VICI':'XLRE',
    'ARE':'XLRE','AVB':'XLRE','EQR':'XLRE','MAA':'XLRE','UDR':'XLRE',
    'NEE':'XLU','DUK':'XLU','SO':'XLU','D':'XLU','AEP':'XLU',
    'SRE':'XLU','EXC':'XLU','XEL':'XLU','ED':'XLU','WEC':'XLU',
    'AWK':'XLU','AEE':'XLU','DTE':'XLU','PPL':'XLU',
}

SECTOR_ETFS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLC','XLRE','XLU']
ALL_TICKERS = list(SECTOR_MAP.keys())
print(f"Universe: {len(ALL_TICKERS)} stocks across {len(SECTOR_ETFS)} sectors")
sys.stdout.flush()

# ─── Download data ───
print("\n[1/5] Downloading price data...")
sys.stdout.flush()
import yfinance as yf

all_symbols = ALL_TICKERS + SECTOR_ETFS + ['SPY']
all_data = {}
batch_size = 50

for i in range(0, len(all_symbols), batch_size):
    batch = all_symbols[i:i+batch_size]
    bn = i//batch_size + 1
    print(f"  Batch {bn}/{(len(all_symbols)-1)//batch_size+1} ({len(batch)} tickers)...")
    sys.stdout.flush()
    try:
        df = yf.download(batch, start=START_DATE, end=END_DATE, progress=False, threads=True)
        if isinstance(df.columns, pd.MultiIndex):
            for ticker in batch:
                try:
                    sub = df.xs(ticker, level=1, axis=1) if ticker in df.columns.get_level_values(1) else None
                    if sub is not None and len(sub.dropna()) > VOL_HISTORY + BREAKOUT_LOOKBACK + 30:
                        all_data[ticker] = sub[['Close','High','Low']].dropna().rename(
                            columns={'Close':'close','High':'high','Low':'low'})
                except:
                    pass
        elif len(batch) == 1:
            ticker = batch[0]
            if len(df.dropna()) > VOL_HISTORY + BREAKOUT_LOOKBACK + 30:
                all_data[ticker] = df[['Close','High','Low']].dropna().rename(
                    columns={'Close':'close','High':'high','Low':'low'})
    except Exception as e:
        print(f"  Warning: batch failed: {e}")
        sys.stdout.flush()

print(f"  Got data for {len(all_data)} symbols")
sys.stdout.flush()

# ─── Precompute features for all stocks (VECTORIZED) ───
print("\n[2/5] Precomputing features (vectorized)...")
sys.stdout.flush()

def fast_rolling_percentile(series, window):
    """Vectorized rolling percentile using rank within window."""
    vals = series.values
    n = len(vals)
    result = np.full(n, np.nan)
    for i in range(window-1, n):
        window_vals = vals[i-window+1:i+1]
        valid = ~np.isnan(window_vals)
        if valid.sum() >= window * 0.8:
            current = vals[i]
            result[i] = (window_vals[valid] <= current).mean() * 100
    return pd.Series(result, index=series.index)

# Precompute per-stock features
stock_features = {}
count = 0
for ticker in ALL_TICKERS:
    if ticker not in all_data:
        continue
    d = all_data[ticker]
    sector_etf = SECTOR_MAP[ticker]
    if sector_etf not in all_data:
        continue

    close = d['close']
    high = d['high']
    low = d['low']
    sector_close = all_data[sector_etf]['close']

    # Realized vol
    returns = close.pct_change()
    realized_vol = returns.rolling(VOL_LOOKBACK).std() * np.sqrt(252)

    # Rolling percentile of vol (vectorized)
    vol_pctile = fast_rolling_percentile(realized_vol, VOL_HISTORY)

    # Breakout levels (shifted to avoid look-ahead)
    rolling_high_20d = high.rolling(BREAKOUT_LOOKBACK).max().shift(1)
    rolling_low_20d = low.rolling(BREAKOUT_LOOKBACK).min().shift(1)

    # Relative strength
    stock_5d_ret = close.pct_change(RS_LOOKBACK)
    common_idx = stock_5d_ret.index.intersection(sector_close.index)
    sector_5d_ret = sector_close.reindex(common_idx).pct_change(RS_LOOKBACK)
    stock_5d_aligned = stock_5d_ret.reindex(common_idx)

    # Forward returns for various hold periods
    fwd_rets = {}
    for hp in HOLD_PERIODS:
        fwd_rets[hp] = close.pct_change(hp).shift(-hp)

    stock_features[ticker] = {
        'vol_pctile': vol_pctile,
        'close': close,
        'rolling_high': rolling_high_20d,
        'rolling_low': rolling_low_20d,
        'stock_5d_ret': stock_5d_aligned,
        'sector_5d_ret': sector_5d_ret,
        'rs_excess_long': (stock_5d_aligned - sector_5d_ret),
        'rs_excess_short': (sector_5d_ret - stock_5d_aligned),
        'fwd_rets': fwd_rets,
    }
    count += 1

print(f"  Precomputed features for {count} stocks")
sys.stdout.flush()

# ─── Regime classification (SPY) ───
print("\n[3/5] Classifying market regime...")
sys.stdout.flush()
spy = all_data['SPY']
spy_20d_ret = spy['close'].pct_change(20)
regime = (spy_20d_ret > 0).astype(int)  # 1=green, 0=red
regime.name = 'regime'
print(f"  Green days: {regime.sum()}, Red days: {(regime==0).sum()}")
sys.stdout.flush()

# ─── Generate signals and evaluate all variants ───
print("\n[4/5] Evaluating variants...")
sys.stdout.flush()

results_all = []
variant_id = 0

for vol_thresh in VOL_PCTILE_THRESHOLDS:
    for rs_thresh in RS_EXCESS_THRESHOLDS:
        for hold in HOLD_PERIODS:
            for direction in DIRECTIONS:
                variant_id += 1
                vname = f"vol{vol_thresh}_rs{int(rs_thresh*1000)}bps_hold{hold}d_{direction}"
                print(f"  V{variant_id}: {vname}", end='', flush=True)

                all_trades = []
                for ticker, feat in stock_features.items():
                    vp = feat['vol_pctile']
                    compressed = vp <= vol_thresh

                    if direction == 'long':
                        breakout = feat['close'] > feat['rolling_high']
                        rs_ok = feat['rs_excess_long'] > rs_thresh
                    else:
                        breakout = feat['close'] < feat['rolling_low']
                        rs_ok = feat['rs_excess_short'] > rs_thresh

                    # Align all signals
                    common = compressed.index
                    for s in [breakout, rs_ok]:
                        common = common.intersection(s.index)

                    compressed_a = compressed.reindex(common).fillna(False)
                    breakout_a = breakout.reindex(common).fillna(False)
                    rs_ok_a = rs_ok.reindex(common).fillna(False)

                    signal = compressed_a & breakout_a & rs_ok_a
                    signal_dates = signal[signal].index

                    if len(signal_dates) == 0:
                        continue

                    fwd = feat['fwd_rets'][hold].reindex(signal_dates).dropna()
                    for dt, ret in fwd.items():
                        r = float(ret) if direction == 'long' else float(-ret)
                        all_trades.append({'date': dt, 'return': r, 'ticker': ticker})

                n_trades = len(all_trades)
                print(f" -> {n_trades} trades", end='', flush=True)

                if n_trades < MIN_TRADES:
                    results_all.append({
                        'variant': vname, 'vol_pctile': vol_thresh,
                        'rs_excess': float(rs_thresh), 'hold_days': hold,
                        'direction': direction, 'n_trades': n_trades,
                        'status': 'SKIP_INSUFFICIENT_TRADES',
                    })
                    print(" [SKIP]")
                    sys.stdout.flush()
                    continue

                tdf = pd.DataFrame(all_trades)
                tdf['date'] = pd.to_datetime(tdf['date'])
                tdf = tdf.set_index('date').sort_index()

                mean_ret = tdf['return'].mean()
                std_ret = tdf['return'].std()
                sharpe = (mean_ret / std_ret) * np.sqrt(252 / hold) if std_ret > 0 else 0
                win_rate = (tdf['return'] > 0).mean()
                wins = tdf.loc[tdf['return'] > 0, 'return']
                losses = tdf.loc[tdf['return'] < 0, 'return']
                avg_win = wins.mean() if len(wins) > 0 else 0
                avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
                pf = (avg_win * win_rate) / (avg_loss * (1-win_rate)) if avg_loss > 0 and win_rate < 1 else 0

                # Gate 1: Permutation test
                obs_mean = mean_ret
                rets_arr = tdf['return'].values
                perm_means = np.array([np.random.permutation(rets_arr).mean() for _ in range(N_PERMUTATIONS)])
                p_value = float((perm_means >= obs_mean).mean())
                perm_pass = p_value < 0.05

                # Gate 2: Regime gap
                tw = tdf.join(regime, how='left')
                tw['regime'] = tw['regime'].fillna(0.5)
                green_t = tw[tw['regime']==1]['return']
                red_t = tw[tw['regime']==0]['return']

                if len(green_t) > 5 and len(red_t) > 5:
                    gs = green_t.std()
                    rs = red_t.std()
                    sharpe_g = (green_t.mean()/gs)*np.sqrt(252/hold) if gs>0 else 0
                    sharpe_r = (red_t.mean()/rs)*np.sqrt(252/hold) if rs>0 else 0
                    mx = max(abs(sharpe_g), abs(sharpe_r))
                    rgap = abs(sharpe_g - sharpe_r)/mx if mx>0 else 999
                    regime_pass = rgap < 0.50
                else:
                    sharpe_g = sharpe_r = rgap = float('nan')
                    regime_pass = False

                # Gate 3: Year consistency
                tdf['year'] = tdf.index.year
                yr_ret = tdf.groupby('year')['return'].mean()
                yrs_prof = int((yr_ret > 0).sum())
                tot_yrs = len(yr_ret)
                yr_cons = yrs_prof / tot_yrs if tot_yrs > 0 else 0
                cons_pass = yr_cons > 0.60

                all_pass = perm_pass and regime_pass and cons_pass

                result = {
                    'variant': vname, 'vol_pctile': vol_thresh,
                    'rs_excess': float(rs_thresh), 'hold_days': hold,
                    'direction': direction, 'n_trades': n_trades,
                    'mean_return': float(mean_ret), 'std_return': float(std_ret),
                    'sharpe': float(sharpe), 'win_rate': float(win_rate),
                    'profit_factor': float(pf),
                    'avg_win': float(avg_win), 'avg_loss': float(avg_loss),
                    'perm_p_value': float(p_value), 'perm_pass': bool(perm_pass),
                    'sharpe_green': float(sharpe_g) if not np.isnan(sharpe_g) else None,
                    'sharpe_red': float(sharpe_r) if not np.isnan(sharpe_r) else None,
                    'regime_gap': float(rgap) if not np.isnan(rgap) else None,
                    'regime_pass': bool(regime_pass),
                    'years_profitable': yrs_prof, 'total_years': tot_yrs,
                    'year_consistency': float(yr_cons), 'consistency_pass': bool(cons_pass),
                    'yearly_returns': {str(k):float(v) for k,v in yr_ret.items()},
                    'all_gates_pass': bool(all_pass),
                    'status': 'PASS_ALL' if all_pass else 'FAIL',
                }
                results_all.append(result)

                tag = "PASS" if all_pass else "FAIL"
                fails = []
                if not perm_pass: fails.append(f"perm p={p_value:.3f}")
                if not regime_pass: fails.append(f"rgap={'%.2f'%rgap if not np.isnan(rgap) else 'N/A'}")
                if not cons_pass: fails.append(f"yr={yr_cons:.0%}")
                fstr = f" ({', '.join(fails)})" if fails else ""
                print(f" | {tag} Sharpe={sharpe:.2f} WR={win_rate:.1%} PF={pf:.2f}{fstr}")
                sys.stdout.flush()

# ─── Summary ───
print(f"\n{'='*70}")
print(f"RESULTS SUMMARY")
print(f"{'='*70}")

passed = [r for r in results_all if r.get('all_gates_pass')]
failed = [r for r in results_all if r.get('status') == 'FAIL']
skipped = [r for r in results_all if r.get('status') == 'SKIP_INSUFFICIENT_TRADES']

print(f"Total: {len(results_all)} | Passed: {len(passed)} | Failed: {len(failed)} | Skipped: {len(skipped)}")

if passed:
    print(f"\n--- PASSING VARIANTS (sorted by Sharpe) ---")
    for r in sorted(passed, key=lambda x: x['sharpe'], reverse=True):
        print(f"  {r['variant']}: Sharpe={r['sharpe']:.2f} WR={r['win_rate']:.1%} "
              f"PF={r['profit_factor']:.2f} N={r['n_trades']} "
              f"perm_p={r['perm_p_value']:.3f} rgap={r['regime_gap']:.2f} "
              f"yr_cons={r['year_consistency']:.0%}")
else:
    print("\n  No variants passed all three validation gates.")
    print("  Top 5 by Sharpe (with enough trades):")
    scoreable = [r for r in results_all if r.get('sharpe') is not None]
    for r in sorted(scoreable, key=lambda x: x.get('sharpe',0), reverse=True)[:5]:
        fails = []
        if not r.get('perm_pass'): fails.append('perm')
        if not r.get('regime_pass'): fails.append('regime')
        if not r.get('consistency_pass'): fails.append('yr_cons')
        print(f"  {r['variant']}: Sharpe={r.get('sharpe',0):.2f} WR={r.get('win_rate',0):.1%} "
              f"N={r['n_trades']} Failed: {','.join(fails)}")

# ─── Save ───
print(f"\n[5/5] Saving results...")
output = {
    'experiment': 'rs_breakout_v1',
    'description': 'Relative Strength Breakout from Vol Compression',
    'run_date': datetime.now().isoformat(),
    'universe_size': len(ALL_TICKERS),
    'symbols_with_data': len(stock_features),
    'date_range': f"{START_DATE} to {END_DATE}",
    'n_variants': len(results_all),
    'n_passed': len(passed),
    'n_failed': len(failed),
    'n_skipped': len(skipped),
    'validation_gates': {
        'permutation_test': f'{N_PERMUTATIONS} shuffles, p < 0.05',
        'regime_gap': '|Sharpe_green - Sharpe_red| / max < 0.50',
        'year_consistency': 'profitable > 60% of years',
    },
    'results': results_all,
}
with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"  Saved results.json")
print(f"\nDONE. {len(passed)} of {len(results_all)} variants passed all gates.")
