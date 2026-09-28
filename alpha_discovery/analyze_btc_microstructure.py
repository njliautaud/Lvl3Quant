"""
BTC/USDT Microstructure Analysis
=================================
Pilot study to determine if crypto HFT is viable using Binance aggTrades data.

Data: BTCUSDT_aggTrades_YYYY-MM-DD.csv
Columns: agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker

Binance convention:
  is_buyer_maker=True  -> buyer placed resting order -> SELLER was aggressor -> SELL trade
  is_buyer_maker=False -> seller placed resting order -> BUYER was aggressor -> BUY trade

Costs:
  Maker fee: 0.02%
  Taker fee: 0.05%
  RT taker:  0.10%  = ~$100 per BTC RT at $100K
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = r"C:\Users\Footb\Documents\Github\Lvl3Quant\data\raw\crypto\aggTrades"
N_DAYS   = 30
BAR_MS   = 100   # 100ms bars

MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
RT_TAKER  = 2 * TAKER_FEE
RT_MIXED  = MAKER_FEE + TAKER_FEE

VPIN_WINDOW   = 50
OFI_WINDOW    = 20
ARRIVAL_WINDOW= 20

print("=" * 70)
print("BTC/USDT MICROSTRUCTURE ANALYSIS — Crypto HFT Viability Pilot")
print("=" * 70)
print(f"Bar size     : {BAR_MS}ms")
print(f"Days analyzed: {N_DAYS}")
print(f"Taker RT cost: {RT_TAKER*100:.3f}%  |  Mixed RT cost: {RT_MIXED*100:.3f}%")
print()
sys.stdout.flush()

# ── Load Data ──────────────────────────────────────────────────────────────────
t0 = time.time()
files = sorted(glob.glob(os.path.join(DATA_DIR, "BTCUSDT_aggTrades_2026-*.csv")))[:N_DAYS]
if not files:
    print(f"ERROR: No files found in {DATA_DIR}")
    sys.exit(1)

print(f"Loading {len(files)} files:")
dfs = []
for f in files:
    print(f"  {os.path.basename(f)}")
    sys.stdout.flush()
    df = pd.read_csv(f, usecols=['price', 'quantity', 'transact_time', 'is_buyer_maker'],
                     dtype={'price': 'float32', 'quantity': 'float32',
                            'transact_time': 'int64', 'is_buyer_maker': 'str'})
    dfs.append(df)

trades = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(trades):,} trades in {time.time()-t0:.1f}s")
sys.stdout.flush()

# Parse direction: is_buyer_maker='true' -> SELL aggressor (side=-1)
#                  is_buyer_maker='false'-> BUY aggressor  (side=+1)
is_bm = trades['is_buyer_maker'].str.lower() == 'true'
trades['side']       = np.where(is_bm, np.int8(-1), np.int8(1))
trades['signed_qty'] = trades['side'] * trades['quantity']
trades['notional']   = trades['price'] * trades['quantity']

print(f"Price range : ${trades['price'].min():,.2f} – ${trades['price'].max():,.2f}")
mid_price = float(trades['price'].mean())

buy_vol  = trades.loc[trades['side'] == 1,  'quantity'].sum()
sell_vol = trades.loc[trades['side'] == -1, 'quantity'].sum()
total_vol= buy_vol + sell_vol
print(f"Buy volume  : {buy_vol:,.2f} BTC ({buy_vol/total_vol*100:.1f}%)")
print(f"Sell volume : {sell_vol:,.2f} BTC ({sell_vol/total_vol*100:.1f}%)")
print(f"Total notional: ${trades['notional'].sum()/1e9:.2f}B")
print()

print("Trade size distribution (BTC):")
for q in [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]:
    v = float(trades['quantity'].quantile(q))
    print(f"  p{int(q*100):3d}: {v:.6f} BTC  (~${v*mid_price:,.0f})")
print()
sys.stdout.flush()

# ── Build 100ms Bars (Vectorized) ──────────────────────────────────────────────
print("Building 100ms bars...")
t1 = time.time()

trades['bar_key'] = (trades['transact_time'] // BAR_MS).astype('int64') * BAR_MS

# Use numpy groupby via pandas, but split buy/sell upfront to avoid lambdas
buy_mask  = trades['side'] ==  1
sell_mask = trades['side'] == -1

buy_trades  = trades[buy_mask].copy()
sell_trades = trades[sell_mask].copy()

# Aggregate all trades
g = trades.groupby('bar_key', sort=True)
bars = g['price'].agg(['first', 'max', 'min', 'last', 'count']).rename(
    columns={'first':'open','max':'high','min':'low','last':'close','count':'trade_count'})
bars['volume']   = g['quantity'].sum()
bars['notional'] = g['notional'].sum()

# Buy/sell aggregates
gb = buy_trades.groupby('bar_key', sort=True)['quantity']
gs = sell_trades.groupby('bar_key', sort=True)['quantity']
bars['buy_vol']   = gb.sum().reindex(bars.index, fill_value=0)
bars['sell_vol']  = gs.sum().reindex(bars.index, fill_value=0)
bars['buy_count'] = buy_trades.groupby('bar_key', sort=True).size().reindex(bars.index, fill_value=0)
bars['sell_count']= sell_trades.groupby('bar_key', sort=True).size().reindex(bars.index, fill_value=0)

print(f"Bars built in {time.time()-t1:.1f}s")
print(f"Total 100ms bars    : {len(bars):,}")
print(f"Bars with trades    : {(bars['trade_count']>0).sum():,}  ({(bars['trade_count']>0).mean()*100:.1f}% fill rate)")
print(f"Avg trades/bar      : {bars['trade_count'].mean():.2f}")
print(f"Avg volume/bar      : {bars['volume'].mean():.4f} BTC  (~${bars['notional'].mean():,.0f})")
print()
sys.stdout.flush()

# ── Feature Engineering (all vectorized) ──────────────────────────────────────
print("Computing features...")

# Ensure float64 for calculations
for col in ['open','high','low','close','volume','notional','buy_vol','sell_vol']:
    bars[col] = bars[col].astype('float64')

safe_vol = bars['volume'].replace(0, np.nan)

# 1. OFI normalized [-1, +1]
bars['ofi_norm']  = (bars['buy_vol'] - bars['sell_vol']) / safe_vol

# 2. VWAP deviation
bars['vwap']     = bars['notional'] / safe_vol
bars['vwap_dev'] = (bars['close'] - bars['vwap']) / bars['close']

# 3. Rolling trade imbalance (buy vol ratio - 0.5)
bars['buy_ratio']       = bars['buy_vol'] / safe_vol
bars['trade_imbalance'] = bars['buy_ratio'].rolling(OFI_WINDOW, min_periods=1).mean() - 0.5

# 4. VPIN (|buy-sell|/total, rolling)
bars['vpin_raw'] = bars['ofi_norm'].abs()
bars['vpin']     = bars['vpin_raw'].rolling(VPIN_WINDOW, min_periods=10).mean()

# 5. Arrival rate (trades/sec, rolling)
bars['arrival_rate']      = bars['trade_count'] / (BAR_MS / 1000.0)
bars['arrival_rate_roll'] = bars['arrival_rate'].rolling(ARRIVAL_WINDOW, min_periods=1).mean()

# 6. Cumulative OFI momentum
bars['ofi_roll'] = bars['ofi_norm'].rolling(OFI_WINDOW, min_periods=1).mean()

# 7. Large trade indicator
vol_mean = bars['volume'].mean()
vol_std  = bars['volume'].std()
bars['large_trade'] = (bars['volume'] > vol_mean + 2.0 * vol_std).astype(float)

# 8. HL spread proxy (bps)
bars['hl_spread_bps'] = (bars['high'] - bars['low']) / bars['close'] * 10000

# 9. Returns at multiple horizons
horizons = {'100ms':1, '1s':10, '10s':100, '30s':300, '1min':600, '5min':3000}
for label, n in horizons.items():
    bars[f'ret_{label}'] = bars['close'].pct_change(n).shift(-n)

# Convert bar_key index to datetime for date grouping
bars['ts'] = pd.to_datetime(bars.index, unit='ms', utc=True)
bars['date'] = bars['ts'].dt.date
bars['hour_utc'] = bars['ts'].dt.hour

print("Features computed.")
print()
sys.stdout.flush()

# ── IC Analysis ────────────────────────────────────────────────────────────────
feature_cols = ['ofi_norm', 'ofi_roll', 'trade_imbalance', 'vpin',
                'arrival_rate_roll', 'vwap_dev', 'large_trade']
horizon_labels = list(horizons.keys())

print("=" * 70)
print("INFORMATION COEFFICIENT (IC) ANALYSIS")
print("=" * 70)
print(f"{'Feature':<25} " + " ".join(f"{h:>7}" for h in horizon_labels))
print("-" * 70)

ic_results = {}
for feat in feature_cols:
    row = {}
    for label in horizon_labels:
        valid = bars[[feat, f'ret_{label}']].dropna()
        if len(valid) < 100:
            row[label] = np.nan
        else:
            row[label] = float(valid[feat].corr(valid[f'ret_{label}']))
    ic_results[feat] = row
    vals = [f"{row.get(h, np.nan):+.4f}" if not np.isnan(row.get(h, np.nan)) else "   nan"
            for h in horizon_labels]
    print(f"{feat:<25} " + " ".join(f"{v:>7}" for v in vals))

print()
sys.stdout.flush()

# ── T-statistics ───────────────────────────────────────────────────────────────
print("=" * 70)
print("IC T-STATISTICS (|t| > 2.0 = significant)")
print("=" * 70)
print(f"{'Feature':<25} " + " ".join(f"{h:>7}" for h in horizon_labels))
print("-" * 70)

for feat in feature_cols:
    row_t = {}
    for label in horizon_labels:
        valid = bars[[feat, f'ret_{label}']].dropna()
        n = len(valid)
        ic = ic_results[feat].get(label, np.nan)
        if np.isnan(ic) or n < 10:
            row_t[label] = np.nan
        else:
            row_t[label] = ic * np.sqrt(n-2) / np.sqrt(max(1 - ic**2, 1e-10))
    vals = [f"{row_t.get(h, np.nan):+.2f}" if not np.isnan(row_t.get(h, np.nan)) else "   nan"
            for h in horizon_labels]
    print(f"{feat:<25} " + " ".join(f"{v:>7}" for v in vals))

print()
sys.stdout.flush()

# ── Best Signal ────────────────────────────────────────────────────────────────
best_ic, best_feat, best_horizon = 0, None, None
for feat in feature_cols:
    for label in horizon_labels:
        ic = ic_results[feat].get(label, np.nan)
        if not np.isnan(ic) and abs(ic) > abs(best_ic):
            best_ic, best_feat, best_horizon = ic, feat, label

print("=" * 70)
print(f"BEST SIGNAL: {best_feat} @ {best_horizon}  IC={best_ic:+.4f}")
print("=" * 70)
print()

# ── Effective Spread ──────────────────────────────────────────────────────────
print("=" * 70)
print("EFFECTIVE SPREAD & COST ANALYSIS")
print("=" * 70)

bars_vol = bars[bars['volume'] > 0]
print(f"HL spread proxy (100ms bars):")
print(f"  Mean   : {bars_vol['hl_spread_bps'].mean():.4f} bps")
print(f"  Median : {bars_vol['hl_spread_bps'].median():.4f} bps")
print(f"  p90    : {bars_vol['hl_spread_bps'].quantile(0.90):.4f} bps")
print()

# Estimate half-spread from individual trades vs rolling mid
# Use 30s rolling mid on bar close prices as the "true mid"
t2 = time.time()
print("Estimating effective half-spread from trades vs rolling mid...")
sys.stdout.flush()

# Down-sample: use bar-level VWAP vs 30s-window mean close as proxy
bars_clean = bars[bars['volume'] > 0].copy()
bars_clean['mid_30s'] = bars_clean['close'].rolling(300, min_periods=5).mean()  # 300 bars = 30s
bars_clean['half_spread_raw'] = (bars_clean['vwap'] - bars_clean['mid_30s']).abs()
bars_clean['half_spread_bps'] = bars_clean['half_spread_raw'] / bars_clean['close'] * 10000

hs_buy  = bars_clean[bars_clean['ofi_norm'] > 0]['half_spread_bps'].mean()
hs_sell = bars_clean[bars_clean['ofi_norm'] < 0]['half_spread_bps'].mean()
hs_avg  = bars_clean['half_spread_bps'].mean()

print(f"Effective half-spread (vs 30s rolling close):")
print(f"  Buy bars  : {hs_buy:.4f} bps")
print(f"  Sell bars : {hs_sell:.4f} bps")
print(f"  Average   : {hs_avg:.4f} bps")
print(f"  Estimated in {time.time()-t2:.1f}s")
print()

rt_taker_bps = RT_TAKER * 10000
rt_mixed_bps = RT_MIXED * 10000
print(f"At BTC price ${mid_price:,.0f}:")
print(f"  RT taker fee   : {rt_taker_bps:.1f} bps = ${RT_TAKER * mid_price:.2f} per BTC RT")
print(f"  RT mixed fee   : {rt_mixed_bps:.1f} bps = ${RT_MIXED * mid_price:.2f} per BTC RT")
print(f"  Half-spread    : {hs_avg:.4f} bps = ${hs_avg/10000 * mid_price:.2f} per BTC side")
print(f"  Total RT cost  : ~{rt_taker_bps + 2*hs_avg:.2f} bps (taker fees + spread)")
print()
sys.stdout.flush()

# Compare to ES
es_rt_bps = (3.00 / (mid_price * 0.10)) * 10000  # ES $3 RT as % of ~50-pt notional
print(f"Cost comparison (per trade, at 1x leverage):")
print(f"  BTC taker RT : {rt_taker_bps:.1f} bps")
print(f"  BTC mixed RT : {rt_mixed_bps:.1f} bps")
print(f"  ES RT        : ~0.24 bps  ($3.00 per contract)")
print(f"  Ratio        : BTC costs {rt_taker_bps/0.24:.0f}x more than ES (taker)")
print()
sys.stdout.flush()

# ── Profitability Estimates ────────────────────────────────────────────────────
print("=" * 70)
print("COST-ADJUSTED PROFITABILITY ESTIMATES (Quartile Long/Short)")
print("=" * 70)
print(f"\n{'Feature':<25} {'Horizon':<8} {'IC':>7} {'Gross bps/d':>13} {'Net bps/d':>12} {'Trades/d':>10}")
print("-" * 75)

for feat in ['ofi_norm', 'ofi_roll', 'trade_imbalance']:
    for label in ['1s', '10s', '30s', '1min']:
        ret_col = f'ret_{label}'
        valid = bars[[feat, ret_col]].dropna().copy()
        ic = ic_results[feat].get(label, np.nan)
        if np.isnan(ic) or abs(ic) < 0.002 or len(valid) < 100:
            continue
        try:
            valid['signal'] = pd.qcut(valid[feat], q=4, labels=False, duplicates='drop')
            valid['signal'] = valid['signal'].map({0: -1.0, 1: 0.0, 2: 0.0, 3: 1.0})
            valid = valid[valid['signal'] != 0]
            mean_ret_bps   = (valid['signal'] * valid[ret_col]).mean() * 10000
            trd_per_day    = len(valid) / N_DAYS
            gross_bps_day  = mean_ret_bps * trd_per_day
            net_bps_day    = gross_bps_day - rt_taker_bps * trd_per_day
            print(f"{feat:<25} {label:<8} {ic:+7.4f} {gross_bps_day:>+13.1f} {net_bps_day:>+12.1f} {trd_per_day:>10.0f}")
        except Exception as e:
            print(f"{feat:<25} {label:<8} ERROR: {e}")

print()
sys.stdout.flush()

# ── Day-by-Day IC Consistency ─────────────────────────────────────────────────
print("=" * 70)
print("DAY-BY-DAY IC CONSISTENCY — ofi_norm @ 1s")
print("=" * 70)

daily_ics = []
for d, grp in bars.groupby('date'):
    valid = grp[['ofi_norm', 'ret_1s']].dropna()
    if len(valid) < 50:
        continue
    ic = valid['ofi_norm'].corr(valid['ret_1s'])
    daily_ics.append({'date': d, 'ic': ic, 'n': len(valid)})

daily_df = pd.DataFrame(daily_ics)
if len(daily_df) > 0:
    pos = (daily_df['ic'] > 0).sum()
    print(f"\n{'Date':<14} {'IC':>8} {'N bars':>10}")
    print("-" * 36)
    for _, row in daily_df.iterrows():
        star = " ***" if abs(row['ic']) > 0.01 else ""
        print(f"{str(row['date']):<14} {row['ic']:+.4f}  {int(row['n']):>10,}{star}")
    print("-" * 36)
    mean_ic = daily_df['ic'].mean()
    std_ic  = daily_df['ic'].std()
    t_stat  = mean_ic / (std_ic / np.sqrt(len(daily_df))) if std_ic > 0 else np.nan
    print(f"Mean IC    : {mean_ic:+.4f}")
    print(f"Std IC     : {std_ic:+.4f}")
    print(f"Positive % : {pos/len(daily_df)*100:.0f}%")
    print(f"t-stat     : {t_stat:+.2f}")

print()
sys.stdout.flush()

# ── Hourly Activity Profile ────────────────────────────────────────────────────
print("=" * 70)
print("HOURLY ACTIVITY PROFILE (UTC)")
print("=" * 70)
hourly = bars.groupby('hour_utc').agg(
    avg_vol    = ('volume', 'mean'),
    avg_trades = ('trade_count', 'mean'),
    avg_ofi    = ('ofi_norm', lambda x: x.abs().mean()),
    ofi_ic_1s  = ('ofi_norm', lambda x: x.corr(bars.loc[x.index, 'ret_1s']))
).reset_index()

print(f"\n{'Hour':>6} {'AvgVol':>10} {'AvgTrades':>12} {'|OFI|':>8} {'IC@1s':>8}")
print("-" * 50)
for _, row in hourly.iterrows():
    h = int(row['hour_utc'])
    ic_val = row['ofi_ic_1s']
    ic_str = f"{ic_val:+.4f}" if not np.isnan(ic_val) else "   nan"
    print(f"  {h:02d}:00  {row['avg_vol']:>10.4f}  {row['avg_trades']:>12.1f}  {row['avg_ofi']:>8.4f}  {ic_str:>8}")

print()
sys.stdout.flush()

# ── VPIN Regime Analysis ───────────────────────────────────────────────────────
print("=" * 70)
print("VPIN REGIME ANALYSIS — High vs Low Informed Trading")
print("=" * 70)

vpin_valid = bars['vpin'].dropna()
print(f"\nVPIN stats: mean={vpin_valid.mean():.4f}  p50={vpin_valid.median():.4f}  "
      f"p90={vpin_valid.quantile(0.90):.4f}  p99={vpin_valid.quantile(0.99):.4f}")
print()

med_vpin = vpin_valid.median()
for label in ['100ms', '1s', '10s']:
    ret_col = f'ret_{label}'
    hi = bars[bars['vpin'] > med_vpin][['ofi_norm', ret_col]].dropna()
    lo = bars[bars['vpin'] <= med_vpin][['ofi_norm', ret_col]].dropna()
    ic_hi = hi['ofi_norm'].corr(hi[ret_col]) if len(hi) > 50 else np.nan
    ic_lo = lo['ofi_norm'].corr(lo[ret_col]) if len(lo) > 50 else np.nan
    print(f"  ofi_norm @ {label:<6}  High VPIN: IC={ic_hi:+.4f} (n={len(hi):,})  |  "
          f"Low VPIN: IC={ic_lo:+.4f} (n={len(lo):,})")

print()
sys.stdout.flush()

# ── Final Summary ──────────────────────────────────────────────────────────────
overall_valid = bars[['ofi_norm', 'ret_1s']].dropna()
n_tot   = len(overall_valid)
overall_ic = float(overall_valid['ofi_norm'].corr(overall_valid['ret_1s']))
overall_t  = overall_ic * np.sqrt(n_tot-2) / np.sqrt(max(1-overall_ic**2, 1e-10))

print("=" * 70)
print("SUMMARY & VERDICT")
print("=" * 70)
print(f"""
Key Metrics:
  OFI @ 1s IC (overall)   : {overall_ic:+.4f}  (t={overall_t:+.2f},  n={n_tot:,})
  Best IC found            : {best_ic:+.4f}  ({best_feat} @ {best_horizon})
  Avg trades/bar (100ms)  : {bars['trade_count'].mean():.2f}
  HL spread mean           : {bars_vol['hl_spread_bps'].mean():.4f} bps
  Est. half-spread         : {hs_avg:.4f} bps
  RT taker cost            : {rt_taker_bps:.1f} bps = ${RT_TAKER * mid_price:.2f} per BTC RT

Signal Quality vs ES:
  ES microprice_dev IC@1s  : +0.1950 (t=8.57, 50 days)
  BTC OFI IC@1s            : {overall_ic:+.4f} (t={overall_t:+.2f}, {N_DAYS} days)

Cost Comparison:
  ES RT cost   : $3.00 per contract  = ~0.24 bps
  BTC RT taker : ${RT_TAKER * mid_price:.2f} per BTC  = {rt_taker_bps:.1f} bps
  BTC RT mixed : ${RT_MIXED * mid_price:.2f} per BTC  = {rt_mixed_bps:.1f} bps
  Advantage    : ES is {rt_taker_bps/0.24:.0f}x cheaper (taker) or {rt_mixed_bps/0.24:.0f}x (mixed)
""")

if abs(overall_ic) > 0.01 and overall_t > 3.0:
    print("SIGNAL: EXISTS — OFI shows statistically significant predictive power")
elif abs(overall_ic) > 0.005 and overall_t > 2.0:
    print(f"SIGNAL: MARGINAL — IC={overall_ic:+.4f} significant but weak")
else:
    print(f"SIGNAL: WEAK/NONE — IC={overall_ic:+.4f} not statistically significant")

print()
if rt_taker_bps > 5.0:
    print(f"COSTS: HIGH — {rt_taker_bps:.1f} bps RT taker ({rt_taker_bps/0.24:.0f}x ES)")
    print("  Maker entry essential. Need IC >0.02 to be net profitable.")
else:
    print(f"COSTS: ACCEPTABLE — {rt_taker_bps:.1f} bps RT taker")

print()
print("RECOMMENDATION:")
if abs(best_ic) > 0.015 and overall_t > 3.0:
    print("  PROCEED to full 30-day study. Signal is real. Key focus areas:")
    print("  1. Maker entry to cut fees from 10 bps to 7 bps RT")
    print("  2. VPIN-gated entries (higher IC in high-VPIN regime)")
    print("  3. Optimal holding period calibration to peak IC horizon")
    print("  4. Compare VWAP-deviation signal (similar to microprice_dev)")
elif abs(best_ic) > 0.005:
    print(f"  LIMITED — best IC={best_ic:+.4f} on {N_DAYS} days. Extend to 30 days.")
    print("  Costs require IC >0.02 for profitability at 1 BTC lot size.")
    print("  Consider: cross-exchange signal, VPIN-gated, larger lot sizes.")
else:
    print("  NOT VIABLE at this resolution. No significant signal found.")
    print("  Consider longer horizons (1-5 min) where fees are proportionally smaller.")

print()
print(f"Total analysis time: {time.time()-t0:.1f}s")
print("=" * 70)
print("Analysis complete.")
print("=" * 70)
