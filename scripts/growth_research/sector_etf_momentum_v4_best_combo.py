#!/usr/bin/env python3
"""
Sector ETF Momentum v4 — Best-of-Everything Combination

Combines ALL validated insights:
- LightGBM ranking (proven best ranker, Sharpe 4.63 in v2)
- Low-vol filter (best confluence signal, +0.12 Sharpe, -8.6% MaxDD)
- Defensive shift in bear markets (best R1 improvement)
- Bi-weekly rebalancing (better R1 than monthly, Sharpe 0.94)

Variants:
A. LightGBM + monthly (v2 baseline reproduction)
B. LightGBM + bi-weekly
C. LightGBM + low-vol filter + monthly
D. LightGBM + low-vol filter + bi-weekly
E. LightGBM + low-vol + defensive shift + bi-weekly (FULL COMBO)
F. LightGBM + 4-of-5 confluence + bi-weekly

Walk-forward: 504 days train, sliding, no overlap.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import json, os, sys
import yfinance as yf
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/home/jupiter/Lvl3Quant')

print(f"Sector ETF Momentum v4 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

ETFS = ['XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB',
        'XLRE', 'XLC', 'QQQ', 'DIA', 'IWM', 'EEM', 'EFA', 'GLD', 'SLV',
        'DBC', 'TLT', 'HYG', 'LQD']
DEFENSIVE = {'GLD', 'TLT', 'XLU', 'XLP', 'LQD'}

# Download data
print("Downloading data...")
raw = yf.download(ETFS + ['SPY', '^VIX'], start='2017-01-01', end='2026-07-25', progress=False)
close = raw['Close']
etf_close = close[[c for c in ETFS if c in close.columns]]
spy_close = close['SPY']
vix_close = close['^VIX']

if isinstance(spy_close, pd.DataFrame):
    spy_close = spy_close.iloc[:, 0]
if isinstance(vix_close, pd.DataFrame):
    vix_close = vix_close.iloc[:, 0]

# Monthly and bi-weekly data
monthly = etf_close.resample('ME').last().dropna(how='all')
biweekly = etf_close.resample('2W-FRI').last().dropna(how='all')

spy_monthly = spy_close.resample('ME').last()
spy_sma200 = spy_close.rolling(200).mean().resample('ME').last()
spy_biweekly = spy_close.resample('2W-FRI').last()
spy_sma200_bw = spy_close.rolling(200).mean().resample('2W-FRI').last()

# Daily returns
daily_ret = etf_close.pct_change()

print(f"Monthly data: {len(monthly)} periods, Bi-weekly: {len(biweekly)} periods")


def compute_features(prices, daily_returns, period_idx, etf):
    """Compute features for a single ETF at a given period."""
    try:
        # Get the date
        dt = prices.index[period_idx]

        # Momentum features
        feats = {}
        for lb, name in [(1, 'mom_1p'), (3, 'mom_3p'), (6, 'mom_6p'), (12, 'mom_12p')]:
            if period_idx >= lb:
                feats[name] = float(prices.iloc[period_idx][etf] / prices.iloc[period_idx - lb][etf] - 1)
            else:
                feats[name] = 0.0

        # 12-1 momentum
        if period_idx >= 12:
            feats['mom_12_1'] = feats['mom_12p'] - feats['mom_1p']
        else:
            feats['mom_12_1'] = 0.0

        # Volatility from daily data
        dt_daily = daily_returns.index.get_indexer([dt], method='ffill')[0]
        window_63 = daily_returns[etf].iloc[max(0, dt_daily-63):dt_daily+1]
        window_21 = daily_returns[etf].iloc[max(0, dt_daily-21):dt_daily+1]

        feats['vol_21d'] = float(window_21.std() * np.sqrt(252)) if len(window_21) > 5 else 0.2
        feats['vol_63d'] = float(window_63.std() * np.sqrt(252)) if len(window_63) > 10 else 0.2

        # Max drawdown 63d
        prices_63d = etf_close[etf].iloc[max(0, dt_daily-63):dt_daily+1]
        if len(prices_63d) > 1:
            peak = prices_63d.cummax()
            feats['maxdd_63d'] = float(((prices_63d / peak) - 1).min())
        else:
            feats['maxdd_63d'] = 0.0

        # Kurtosis
        feats['kurt_63d'] = float(window_63.kurtosis()) if len(window_63) > 10 else 0.0

        # Skewness
        feats['skew_63d'] = float(window_63.skew()) if len(window_63) > 10 else 0.0

        # Return/vol efficiency
        feats['ret_vol_ratio'] = feats['mom_3p'] / (feats['vol_63d'] + 1e-6)

        # Is defensive
        feats['is_defensive'] = 1.0 if etf in DEFENSIVE else 0.0

        return feats
    except:
        return None


def build_dataset(prices, daily_returns, min_periods=13):
    """Build training dataset from price history."""
    records = []

    for i in range(min_periods, len(prices) - 1):
        dt = prices.index[i]
        dt_next = prices.index[i + 1]

        for etf in prices.columns:
            try:
                price_now = float(prices.iloc[i][etf])
                price_next = float(prices.iloc[i + 1][etf])
                if pd.isna(price_now) or pd.isna(price_next):
                    continue

                feats = compute_features(prices, daily_returns, i, etf)
                if feats is None:
                    continue

                # Label: forward return rank (cross-sectional)
                fwd_ret = price_next / price_now - 1

                record = feats.copy()
                record['date'] = dt
                record['etf'] = etf
                record['fwd_ret'] = float(fwd_ret)
                records.append(record)
            except:
                continue

    df = pd.DataFrame(records)

    # Add cross-sectional rank
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)

    return df


def run_lgbm_variant(name, prices, daily_returns, spy_prices, spy_sma,
                      top_k=3, train_periods=24, vol_filter=False,
                      defensive_shift=False, confluence_min=0,
                      periods_per_year=12):
    """Run LightGBM walk-forward with specified parameters."""
    print(f"\n--- {name} ---")

    feature_cols = ['mom_1p', 'mom_3p', 'mom_6p', 'mom_12p', 'mom_12_1',
                    'vol_21d', 'vol_63d', 'maxdd_63d', 'kurt_63d', 'skew_63d',
                    'ret_vol_ratio', 'is_defensive']

    # Build full dataset
    df = build_dataset(prices, daily_returns)
    if len(df) < 100:
        print(f"  Insufficient data: {len(df)} samples")
        return pd.Series(dtype=float), []

    dates = sorted(df['date'].unique())
    rets, ret_dates, regimes = [], [], []

    for i in range(train_periods, len(dates) - 1):
        train_dates = dates[max(0, i - train_periods):i]
        test_date = dates[i]

        train_df = df[df['date'].isin(train_dates)]
        test_df = df[df['date'] == test_date].copy()

        if len(test_df) < top_k or len(train_df) < 50:
            continue

        # Train LightGBM
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df['rank_label'].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)

        X_train = np.nan_to_num(X_train, nan=0, posinf=1, neginf=-1)
        X_test = np.nan_to_num(X_test, nan=0, posinf=1, neginf=-1)

        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
            verbose=-1
        )
        model.fit(X_train, y_train)

        scores = model.predict(X_test)
        test_df['score'] = scores

        # Regime check
        is_bear = False
        if spy_prices is not None and spy_sma is not None:
            if test_date in spy_prices.index and test_date in spy_sma.index:
                is_bear = bool(spy_prices.loc[test_date] < spy_sma.loc[test_date])

        # Apply filters
        filtered_df = test_df.copy()

        # Low-vol filter
        if vol_filter:
            filtered_df = filtered_df[filtered_df['vol_63d'] < 0.40]

        # Confluence filter
        if confluence_min > 0:
            # Count confirming signals per ETF
            for idx in filtered_df.index:
                confirming = 0
                row = filtered_df.loc[idx]
                if row['mom_12_1'] > 0: confirming += 1
                if row['mom_1p'] > 0 and row['mom_3p'] > 0: confirming += 1  # short-term positive
                if row['vol_63d'] < 0.30: confirming += 1  # low vol
                if row['maxdd_63d'] > -0.10: confirming += 1  # shallow drawdown
                if not is_bear or row['is_defensive'] > 0: confirming += 1  # macro ok
                filtered_df.loc[idx, 'confirming'] = confirming

            filtered_df = filtered_df[filtered_df['confirming'] >= confluence_min]

        # Defensive shift
        if defensive_shift and is_bear:
            filtered_df.loc[filtered_df['is_defensive'] > 0, 'score'] += 0.1
            filtered_df.loc[filtered_df['is_defensive'] == 0, 'score'] -= 0.05

        if len(filtered_df) < top_k:
            # Not enough candidates after filtering — go flat
            rets.append(0.0)
            ret_dates.append(test_date)
            regimes.append(1 if is_bear else 0)
            continue

        # Select top_k
        top = filtered_df.nlargest(top_k, 'score')

        # Portfolio return
        port_ret = float(top['fwd_ret'].mean()) - 0.002  # 20bps cost

        rets.append(port_ret)
        ret_dates.append(test_date)
        regimes.append(1 if is_bear else 0)

    series = pd.Series(rets, index=ret_dates)

    if len(series) > 10:
        sharpe = series.mean() / series.std() * np.sqrt(periods_per_year) if series.std() > 0 else 0
        cum = (1 + series).cumprod()
        years = len(series) / periods_per_year
        cagr = float(cum.iloc[-1] ** (1/years) - 1) if years > 0 else 0
        maxdd = float(((cum / cum.cummax()) - 1).min())
        active = (series != 0).mean()
        print(f"  {len(series)} periods ({active:.0%} active) | Sharpe {sharpe:.2f} | CAGR {cagr:.1%} | MaxDD {maxdd:.1%}")

    return series, regimes


def adversarial_4gate(returns, regimes, name, ppy=12):
    """4-gate validation with configurable periods-per-year."""
    if len(returns) < 20:
        print(f"  INSUFFICIENT DATA")
        return {'sharpe': 0, 'gates': 0}

    gates = 0
    real_sharpe = returns.mean() / returns.std() * np.sqrt(ppy) if returns.std() > 0 else 0

    # G1: Block permutation
    vals = returns.values
    block = max(2, ppy // 4)
    n_blocks = len(vals) // block
    perm_beat = sum(1 for _ in range(1000)
                    if (lambda sh: np.mean(sh) / (np.std(sh) + 1e-10) * np.sqrt(ppy) >= real_sharpe)(
                        np.concatenate([b for b in (lambda bl: (np.random.shuffle(bl), bl)[1])(
                            [vals[i*block:(i+1)*block] for i in range(n_blocks)])] +
                            ([vals[n_blocks*block:]] if len(vals) % block else []))))
    p_val = perm_beat / 1000
    g1 = p_val < 0.05
    gates += g1

    # G2: Regime
    reg = np.array(regimes[:len(returns)])
    bull_r, bear_r = returns.values[reg == 0], returns.values[reg == 1]
    bull_sh = np.mean(bull_r) / (np.std(bull_r) + 1e-10) * np.sqrt(ppy) if len(bull_r) > 5 else 0
    bear_sh = np.mean(bear_r) / (np.std(bear_r) + 1e-10) * np.sqrt(ppy) if len(bear_r) > 5 else 0
    gap = abs(bull_sh - bear_sh) / max(abs(bull_sh), abs(bear_sh), 0.01)
    g2 = gap < 0.50
    gates += g2

    # G3: Sub-period
    mid = len(returns) // 2
    h1_sh = returns.iloc[:mid].mean() / returns.iloc[:mid].std() * np.sqrt(ppy) if returns.iloc[:mid].std() > 0 else 0
    h2_sh = returns.iloc[mid:].mean() / returns.iloc[mid:].std() * np.sqrt(ppy) if returns.iloc[mid:].std() > 0 else 0
    g3 = h1_sh > 0 and h2_sh > 0
    gates += g3

    # G4: Outlier
    trimmed = returns[(returns >= returns.quantile(0.01)) & (returns <= returns.quantile(0.99))]
    trim_sh = trimmed.mean() / trimmed.std() * np.sqrt(ppy) if len(trimmed) > 5 and trimmed.std() > 0 else 0
    g4 = trim_sh > 0
    gates += g4

    # Metrics
    cum = (1 + returns).cumprod()
    years = len(returns) / ppy
    cagr = float(cum.iloc[-1] ** (1/years) - 1) if years > 0 else 0
    maxdd = float(((cum / cum.cummax()) - 1).min())
    down = returns[returns < 0]
    sortino = float(returns.mean() / down.std() * np.sqrt(ppy)) if len(down) > 0 and down.std() > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    pf = float(returns[returns > 0].sum()) / (abs(float(returns[returns < 0].sum())) + 1e-10)
    wr = float((returns > 0).mean())
    active = float((returns != 0).mean())

    print(f"  G1={p_val:.3f}{'✅' if g1 else '❌'} G2={gap:.3f}{'✅' if g2 else '❌'} "
          f"G3={h1_sh:.1f}/{h2_sh:.1f}{'✅' if g3 else '❌'} G4={trim_sh:.1f}{'✅' if g4 else '❌'}")
    print(f"  Sharpe {real_sharpe:.2f} | Sort {sortino:.2f} | CAGR {cagr:.1%} | MaxDD {maxdd:.1%} | "
          f"Cal {calmar:.2f} | PF {pf:.2f} | WR {wr:.1%} | Act {active:.0%} | Gates {gates}/4")

    return {'sharpe': float(real_sharpe), 'sortino': sortino, 'cagr': cagr, 'max_dd': maxdd,
            'calmar': calmar, 'pf': pf, 'wr': wr, 'gates': gates, 'perm_p': float(p_val),
            'r1_gap': float(gap), 'bull_sharpe': float(bull_sh), 'bear_sharpe': float(bear_sh),
            'active_pct': active}


# ============ RUN ALL VARIANTS ============

results = {}

# A. LightGBM + monthly (v2 reproduction)
r, reg = run_lgbm_variant('A_LGBM_Monthly', monthly, daily_ret,
                           spy_monthly, spy_sma200, top_k=3, train_periods=24)
results['A_LGBM_Monthly'] = adversarial_4gate(r, reg, 'A', ppy=12)

# B. LightGBM + bi-weekly
r, reg = run_lgbm_variant('B_LGBM_Biweekly', biweekly, daily_ret,
                           spy_biweekly, spy_sma200_bw, top_k=3, train_periods=48,
                           periods_per_year=26)
results['B_LGBM_Biweekly'] = adversarial_4gate(r, reg, 'B', ppy=26)

# C. LightGBM + low-vol filter + monthly
r, reg = run_lgbm_variant('C_LGBM_LowVol_Monthly', monthly, daily_ret,
                           spy_monthly, spy_sma200, top_k=3, train_periods=24,
                           vol_filter=True)
results['C_LGBM_LowVol'] = adversarial_4gate(r, reg, 'C', ppy=12)

# D. LightGBM + low-vol + bi-weekly
r, reg = run_lgbm_variant('D_LGBM_LowVol_Biweekly', biweekly, daily_ret,
                           spy_biweekly, spy_sma200_bw, top_k=3, train_periods=48,
                           vol_filter=True, periods_per_year=26)
results['D_LGBM_LowVol_BW'] = adversarial_4gate(r, reg, 'D', ppy=26)

# E. FULL COMBO: LightGBM + low-vol + defensive shift + bi-weekly
r, reg = run_lgbm_variant('E_FULL_COMBO', biweekly, daily_ret,
                           spy_biweekly, spy_sma200_bw, top_k=3, train_periods=48,
                           vol_filter=True, defensive_shift=True, periods_per_year=26)
results['E_FullCombo'] = adversarial_4gate(r, reg, 'E', ppy=26)

# F. LightGBM + 4-of-5 confluence + bi-weekly
r, reg = run_lgbm_variant('F_LGBM_4of5_Biweekly', biweekly, daily_ret,
                           spy_biweekly, spy_sma200_bw, top_k=3, train_periods=48,
                           confluence_min=4, periods_per_year=26)
results['F_LGBM_4of5_BW'] = adversarial_4gate(r, reg, 'F', ppy=26)

# G. Top 5 version of full combo (more diversification)
r, reg = run_lgbm_variant('G_FULL_COMBO_Top5', biweekly, daily_ret,
                           spy_biweekly, spy_sma200_bw, top_k=5, train_periods=48,
                           vol_filter=True, defensive_shift=True, periods_per_year=26)
results['G_FullCombo_T5'] = adversarial_4gate(r, reg, 'G', ppy=26)

# Summary
print(f"\n{'='*70}")
print("SUMMARY — SECTOR ETF MOMENTUM v4 (BEST COMBO)")
print(f"{'='*70}")
print(f"{'Variant':<22s} {'Sharpe':>7s} {'Sort':>6s} {'CAGR':>7s} {'MaxDD':>7s} {'R1gap':>6s} {'PF':>5s} {'WR':>5s} {'G':>3s}")
print("-" * 70)
for name in sorted(results):
    r = results[name]
    print(f"{name:<22s} {r['sharpe']:7.2f} {r['sortino']:6.2f} {r['cagr']:7.1%} "
          f"{r['max_dd']:7.1%} {r.get('r1_gap',0):6.3f} {r['pf']:5.2f} {r['wr']:5.1%} {r['gates']:>2d}/4")

best = max(results.items(), key=lambda x: (x[1]['gates'], x[1]['sharpe']))
best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe'])
print(f"\n🏆 BEST VALIDATED: {best[0]} — Sharpe {best[1]['sharpe']:.2f}, R1 gap {best[1].get('r1_gap',0):.3f}, Gates {best[1]['gates']}/4")
if best_sharpe[0] != best[0]:
    print(f"📈 HIGHEST SHARPE: {best_sharpe[0]} — Sharpe {best_sharpe[1]['sharpe']:.2f}, Gates {best_sharpe[1]['gates']}/4")

# vs v2 benchmark
print(f"\nBENCHMARK: v2 LightGBM = Sharpe 4.63 (24-fold WF on monthly, R1 FAIL 0.739)")
print(f"NOTE: This simplified v4 uses fewer folds — v2's higher Sharpe partly from more training data per fold")

# Save
save_path = '/home/jupiter/Lvl3Quant/research/findings/sector_etf_momentum_v4_results.json'
with open(save_path, 'w') as f:
    json.dump({
        'strategy': 'Sector ETF Momentum v4 — Best Combo',
        'run_date': datetime.now().isoformat(),
        'variants': results,
        'best_validated': best[0],
        'best_sharpe': best_sharpe[0]
    }, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
print(f"\nSaved → {save_path}")
