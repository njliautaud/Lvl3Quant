#!/usr/bin/env python3
"""
Cross-Sectional Dispersion → Opportunity Strategy v1
=====================================================
OBSERVATION (from cross_sectional_anomaly_scanner, HC #735):
  When cross-sectional dispersion (spread of stock returns within a day) is extreme,
  SPY subsequent 3-month return averages +7.7%. High dispersion = dislocations = opportunity.

STRATEGY HYPOTHESIS:
  High dispersion creates two opportunities:
  1. MEAN REVERSION: Extreme losers during high-dispersion periods revert more strongly
  2. MOMENTUM PERSISTENCE: Winners during high-dispersion diverge from losers more persistently

  Also from observation: Skewness premium is INVERTED — positive-skew stocks outperform
  by 10% annually (opposite of academic literature which says negative-skew outperforms).

IMPLEMENTATION:
  - Compute daily cross-sectional dispersion (std of returns across all stocks)
  - When dispersion spikes (>90th percentile): rank stocks by same-day return
  - Test: buy bottom decile (mean reversion) vs top decile (momentum)
  - Also test: rank by skewness, buy positive-skew stocks

UNIVERSE: S&P 500 (~300 stocks)
VALIDATION: Permutation + regime + sub-period (HC #428)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/jupiter/Lvl3Quant')

# ─── Config ───
UNIVERSE_SIZE = 300
START_DATE = '2015-01-01'
END_DATE = '2026-07-01'
DISPERSION_LOOKBACK = 63  # Quarterly lookback for percentile
DISPERSION_PCT = 90  # Top 10% = high dispersion
HOLD_PERIODS = [5, 10, 21]
TOP_N = 0.1  # Top/bottom 10% of stocks
N_PERMS = 200
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/dispersion_momentum_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"{'='*70}")
print(f"CROSS-SECTIONAL DISPERSION → OPPORTUNITY STRATEGY v1")
print(f"Observation-first research (HC #735)")
print(f"{'='*70}")

# ─── Get data ───
print("\n[1/6] Downloading data...")
import yfinance as yf

try:
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()[:UNIVERSE_SIZE]
except:
    tickers = ['AAPL','MSFT','AMZN','NVDA','GOOGL','META','TSLA','BRK-B','UNH','JNJ',
               'V','XOM','JPM','WMT','PG','MA','HD','CVX','MRK','ABBV','LLY','PEP',
               'KO','COST','AVGO','TMO','MCD','CSCO','ACN','ABT','DHR','WFC','NEE',
               'DIS','TXN','BMY','PM','UNP','UPS','RTX','AMGN','IBM','COP','LOW','GS',
               'SPGI','MS','BLK','AXP','INTC','CAT','BA','GE','AMD','ISRG','MDLZ',
               'GILD','BKNG','SYK','ADP','VRTX','TJX','MMC','PGR','CB','REGN',
               'SCHW','ZTS','CI','EOG','SLB','MO','BSX','CME','DUK','SO','LRCX',
               'BDX','CL','FI','ANET','ITW','SHW','APD','PLD','MCK','PNC','USB',
               'TGT','ICE','SNPS','CDNS','PSA','ORLY','AON','FCX','OXY',
               'AZO','KMB','CMG','MPC','VLO','EMR','SRE','ALL','D','KHC',
               'SPG','AFL','WMB','PSX','F','GM','CTSH','FDX','NEM','CCI',
               'WELL','YUM','MNST','HSY','KEYS','NDAQ','ROP','GIS',
               'DG','DLTR','HCA','FAST','CTAS','PAYX','ON','WEC','AEP','XEL',
               'EXC','ED','AWK','ES','ETR','FE','PPL','CNP','CMS','EVRG',
               'ORCL','CRM','NOW','INTU','ADBE','PYPL','NFLX','ABNB','UBER',
               'NKE','SBUX','LULU','GD','LMT','NOC','PFE','BIIB',
               'T','VZ','TMUS','CHTR','CMCSA'][:UNIVERSE_SIZE]

# Download in batches
batch_size = 50
all_data = {}
for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    print(f"  Batch {i//batch_size + 1}: {batch[0]}..{batch[-1]}")
    try:
        data = yf.download(batch, start=START_DATE, end=END_DATE,
                          group_by='ticker', progress=False, threads=True)
        for t in batch:
            try:
                if len(batch) == 1:
                    close = data['Close'].dropna()
                else:
                    close = data[t]['Close'].dropna()
                if len(close) > 300:
                    all_data[t] = close
            except Exception as e:
                pass
    except Exception as e:
        print(f"  Download error: {e}")
        pass

print(f"  Got {len(all_data)} stocks")

# Build return matrix
returns_df = pd.DataFrame({t: c.pct_change() for t, c in all_data.items()}).dropna(how='all')
# Need at least 50 stocks per day
valid_days = returns_df.dropna(thresh=50).index
returns_df = returns_df.loc[valid_days]
print(f"  {len(returns_df)} trading days with 50+ stocks")

# Get SPY for regime classification
spy = all_data.get('SPY', pd.Series(dtype=float))

# ─── Compute dispersion ───
print(f"\n[2/6] Computing cross-sectional dispersion...")

daily_dispersion = returns_df.std(axis=1).dropna()  # Cross-sectional std each day
print(f"  Daily dispersion computed: {len(daily_dispersion)} days, mean={daily_dispersion.mean():.6f}")

# Simpler percentile: expanding window rank
disp_rank = daily_dispersion.expanding(min_periods=DISPERSION_LOOKBACK).rank(pct=True) * 100
disp_rank = disp_rank.dropna()
print(f"  Dispersion rank computed: {len(disp_rank)} days")

high_disp_days = disp_rank[disp_rank >= DISPERSION_PCT].index
print(f"  High dispersion days (>={DISPERSION_PCT}th pctile): {len(high_disp_days)}")
if len(high_disp_days) > 0:
    print(f"  Avg dispersion on signal days: {daily_dispersion.loc[high_disp_days].mean():.4f}")
print(f"  Avg dispersion normally: {daily_dispersion.mean():.4f}")

# ─── Build strategies ───
print(f"\n[3/6] Building 3 strategies...")

strategies = {
    'mean_reversion': {},   # Buy losers on high-disp days
    'momentum': {},         # Buy winners on high-disp days
    'skewness': {}          # Buy positive-skew stocks on high-disp days
}

# Pre-compute rolling skewness (63d)
skewness_df = returns_df.rolling(63).skew()

all_trades = []

for signal_day in high_disp_days:
    day_pos = returns_df.index.get_loc(signal_day)

    # Get that day's cross-sectional returns
    day_returns = returns_df.loc[signal_day].dropna()
    if len(day_returns) < 30:
        continue

    n_select = max(5, int(len(day_returns) * TOP_N))

    # Rank stocks
    sorted_rets = day_returns.sort_values()
    losers = sorted_rets.head(n_select).index.tolist()  # Bottom 10%
    winners = sorted_rets.tail(n_select).index.tolist()  # Top 10%

    # Skewness ranking
    day_skew = skewness_df.loc[signal_day].dropna()
    if len(day_skew) > 20:
        sorted_skew = day_skew.sort_values()
        pos_skew = sorted_skew.tail(n_select).index.tolist()
    else:
        pos_skew = []

    for hp in HOLD_PERIODS:
        exit_pos = day_pos + hp
        if exit_pos >= len(returns_df):
            continue

        # SPY return for regime
        spy_ret = np.nan
        if signal_day in spy.index:
            spy_pos = spy.index.get_loc(signal_day)
            if spy_pos + hp < len(spy):
                spy_ret = spy.iloc[spy_pos + hp] / spy.iloc[spy_pos] - 1

        # Mean reversion: buy losers
        for t in losers:
            if t in all_data:
                close = all_data[t]
                if signal_day in close.index:
                    entry_pos_t = close.index.get_loc(signal_day) + 1  # Enter next day
                    exit_pos_t = entry_pos_t + hp
                    if exit_pos_t < len(close) and entry_pos_t < len(close):
                        pnl = close.iloc[exit_pos_t] / close.iloc[entry_pos_t] - 1
                        all_trades.append({
                            'strategy': 'mean_reversion',
                            'ticker': t, 'signal_date': signal_day,
                            'hold_period': hp, 'pnl_pct': pnl,
                            'spy_ret': spy_ret, 'dispersion': daily_dispersion.loc[signal_day],
                            'day_rank_ret': day_returns[t],
                            'year': signal_day.year
                        })

        # Momentum: buy winners
        for t in winners:
            if t in all_data:
                close = all_data[t]
                if signal_day in close.index:
                    entry_pos_t = close.index.get_loc(signal_day) + 1
                    exit_pos_t = entry_pos_t + hp
                    if exit_pos_t < len(close) and entry_pos_t < len(close):
                        pnl = close.iloc[exit_pos_t] / close.iloc[entry_pos_t] - 1
                        all_trades.append({
                            'strategy': 'momentum',
                            'ticker': t, 'signal_date': signal_day,
                            'hold_period': hp, 'pnl_pct': pnl,
                            'spy_ret': spy_ret, 'dispersion': daily_dispersion.loc[signal_day],
                            'day_rank_ret': day_returns[t],
                            'year': signal_day.year
                        })

        # Skewness: buy positive-skew stocks
        for t in pos_skew:
            if t in all_data:
                close = all_data[t]
                if signal_day in close.index:
                    entry_pos_t = close.index.get_loc(signal_day) + 1
                    exit_pos_t = entry_pos_t + hp
                    if exit_pos_t < len(close) and entry_pos_t < len(close):
                        pnl = close.iloc[exit_pos_t] / close.iloc[entry_pos_t] - 1
                        all_trades.append({
                            'strategy': 'skewness',
                            'ticker': t, 'signal_date': signal_day,
                            'hold_period': hp, 'pnl_pct': pnl,
                            'spy_ret': spy_ret, 'dispersion': daily_dispersion.loc[signal_day],
                            'year': signal_day.year
                        })

trades_df = pd.DataFrame(all_trades)
print(f"  Total trades generated: {len(trades_df)}")

if len(trades_df) == 0:
    print("\nERROR: No trades generated. Debugging:")
    print(f"  returns_df shape: {returns_df.shape}")
    print(f"  daily_dispersion: {len(daily_dispersion)} vals, range [{daily_dispersion.min():.6f}, {daily_dispersion.max():.6f}]")
    print(f"  high_disp_days: {len(high_disp_days)}")
    if len(high_disp_days) > 0:
        print(f"  First 5 high disp days: {high_disp_days[:5].tolist()}")
    sys.exit(1)

# ─── Evaluate ───
print(f"\n[4/6] Evaluating strategies...")

all_results = {}
for strat in ['mean_reversion', 'momentum', 'skewness']:
    print(f"\n  ══ {strat.upper()} ══")
    for hp in HOLD_PERIODS:
        sub = trades_df[(trades_df['strategy'] == strat) & (trades_df['hold_period'] == hp)]
        if len(sub) < 20:
            continue

        # Equal-weight portfolio per signal day
        daily_port = sub.groupby('signal_date')['pnl_pct'].mean()

        mean_ret = daily_port.mean()
        wr = (daily_port > 0).mean()
        sharpe = daily_port.mean() / daily_port.std() * np.sqrt(252/hp) if daily_port.std() > 0 else 0
        sortino_d = daily_port[daily_port < 0].std()
        sortino = daily_port.mean() / sortino_d * np.sqrt(252/hp) if sortino_d > 0 else 0
        gross_p = daily_port[daily_port > 0].sum()
        gross_l = abs(daily_port[daily_port < 0].sum())
        pf = gross_p / gross_l if gross_l > 0 else np.inf
        cum = daily_port.cumsum()
        max_dd = (cum - cum.cummax()).min()

        # Regime test
        sub_with_regime = sub.copy()
        sub_with_regime['regime'] = sub_with_regime['spy_ret'].apply(
            lambda x: 'GREEN' if x > 0.01 else ('RED' if x < -0.01 else 'FLAT')
        )
        regime_sharpes = {}
        for reg in ['GREEN', 'RED']:
            reg_sub = sub_with_regime[sub_with_regime['regime'] == reg]
            if len(reg_sub) > 10:
                rd = reg_sub.groupby('signal_date')['pnl_pct'].mean()
                if rd.std() > 0:
                    regime_sharpes[reg] = rd.mean() / rd.std() * np.sqrt(252/hp)

        regime_gap = np.nan
        if 'GREEN' in regime_sharpes and 'RED' in regime_sharpes:
            max_s = max(abs(regime_sharpes['GREEN']), abs(regime_sharpes['RED']))
            regime_gap = abs(regime_sharpes['GREEN'] - regime_sharpes['RED']) / max_s if max_s > 0 else 0

        key = f"{strat}_{hp}d"
        all_results[key] = {
            'strategy': strat, 'hold_period': hp,
            'n_trades': len(sub), 'n_signals': len(daily_port),
            'mean_ret': mean_ret, 'win_rate': wr, 'sharpe': sharpe,
            'sortino': sortino, 'profit_factor': pf, 'max_dd': max_dd,
            'regime_gap': regime_gap, 'regime_sharpes': regime_sharpes
        }

        print(f"  {hp}d: {len(sub)} trades, {len(daily_port)} signals | "
              f"Sharpe {sharpe:.2f}, WR {wr:.0%}, PF {pf:.2f}, MaxDD {max_dd*100:.1f}% | "
              f"Regime gap: {regime_gap:.2f}" if not np.isnan(regime_gap) else
              f"  {hp}d: {len(sub)} trades | Sharpe {sharpe:.2f}, WR {wr:.0%}, PF {pf:.2f}")

# ─── Permutation tests on promising variants ───
print(f"\n[5/6] Permutation tests on promising variants...")

promising = [(k, v) for k, v in all_results.items() if v['sharpe'] > 0.3]
promising.sort(key=lambda x: -x[1]['sharpe'])

for key, r in promising[:5]:  # Top 5 only
    strat = r['strategy']
    hp = r['hold_period']
    sub = trades_df[(trades_df['strategy'] == strat) & (trades_df['hold_period'] == hp)]
    daily_port = sub.groupby('signal_date')['pnl_pct'].mean()

    obs_sharpe = daily_port.mean() / daily_port.std() * np.sqrt(252/hp) if daily_port.std() > 0 else 0

    perm_sharpes = []
    for _ in range(N_PERMS):
        # Shuffle stock assignments across signal days
        shuffled_pnl = sub['pnl_pct'].values.copy()
        np.random.shuffle(shuffled_pnl)
        sub_shuffled = sub.copy()
        sub_shuffled['pnl_pct'] = shuffled_pnl
        perm_daily = sub_shuffled.groupby('signal_date')['pnl_pct'].mean()
        if perm_daily.std() > 0:
            perm_sharpes.append(perm_daily.mean() / perm_daily.std() * np.sqrt(252/hp))

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= obs_sharpe).mean()

    all_results[key]['perm_p_value'] = p_value
    print(f"  {key}: observed Sharpe {obs_sharpe:.2f}, null mean {perm_sharpes.mean():.2f}, "
          f"p={p_value:.3f} ({'PASS' if p_value < 0.05 else 'FAIL'})")

# ─── Summary ───
print(f"\n{'='*70}")
print(f"FINAL RESULTS")
print(f"{'='*70}")

for key, r in sorted(all_results.items(), key=lambda x: -x[1]['sharpe']):
    perm = r.get('perm_p_value', 'N/T')
    perm_str = f"p={perm:.3f}" if isinstance(perm, float) else perm
    regime_str = f"gap={r['regime_gap']:.2f}" if not np.isnan(r.get('regime_gap', np.nan)) else "N/A"
    print(f"  {key}: Sharpe {r['sharpe']:.2f}, WR {r['win_rate']:.0%}, "
          f"PF {r['profit_factor']:.2f}, Perm {perm_str}, Regime {regime_str}")

# Save
trades_df.to_csv(f"{OUTPUT_DIR}/all_trades.csv", index=False)
with open(f"{OUTPUT_DIR}/summary.json", 'w') as f:
    json.dump({k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                    for kk, vv in v.items() if kk != 'regime_sharpes'}
               for k, v in all_results.items()}, f, indent=2, default=str)

print(f"\nSaved to {OUTPUT_DIR}/")
print(f"Completed: {datetime.now()}")
