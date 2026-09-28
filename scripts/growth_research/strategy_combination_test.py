#!/usr/bin/env python3
"""
Strategy Combination Analysis

Tests whether combining our two best strategies (Gameplan v3 + Vol Mean Reversion)
in a portfolio improves risk-adjusted returns vs either alone.

Approaches:
1. SIMPLE 50/50: Half capital in each strategy
2. REGIME-WEIGHTED: Use VIX regime to weight between strategies
3. CORRELATION ANALYSIS: How correlated are the two strategies' returns?
4. DIVERSIFICATION BENEFIT: Does the combination reduce MaxDD meaningfully?
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings, json, os
from datetime import datetime
from scipy.stats import percentileofscore

warnings.filterwarnings('ignore')
np.random.seed(42)

OUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_research/strategy_combination"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("STRATEGY COMBINATION: GAMEPLAN v3 + VOL MEAN REVERSION")
print("=" * 70)

# Fetch data
tickers = ['SPY', 'UPRO', 'GLD', 'TLT', '^VIX']
data = {}
for t in tickers:
    df = yf.download(t, start='2012-01-01', end='2026-07-17', progress=False)
    name = t.replace('^', '')
    data[name] = df['Close'].squeeze()
    print(f"  {name}: {len(df)} days")

common = sorted(set.intersection(*[set(data[n].index) for n in data]))
prices = pd.DataFrame({n: data[n].reindex(common) for n in data}).dropna()
returns = prices.pct_change().dropna()
print(f"\nAligned: {len(prices)} days")


# ─── Strategy 1: Gameplan v3 (simplified replica) ──────────────────────────
def run_gameplan_v3(prices_df, returns_df):
    """
    Simplified Gameplan v3:
    - Vol < 15% AND confluence score >= 2.5: UPRO
    - Vol > 30%: GLD
    - Vol > 15%: SPY
    - Confluence drops below 2.0: exit UPRO to SPY
    - September: SPY regardless
    """
    spy = prices_df['SPY']
    spy_ret = returns_df['SPY']
    upro_ret = returns_df['UPRO']
    gld_ret = returns_df['GLD']

    vol_21 = spy_ret.rolling(21).std() * np.sqrt(252) * 100

    # Confluence components
    mom_5d = spy.pct_change(5)
    rsi_10 = compute_rsi(spy, 10)
    sma_20 = spy.rolling(20).mean()
    sma_50 = spy.rolling(50).mean()
    sma_200 = spy.rolling(200).mean()
    vol_21d_raw = spy_ret.rolling(21).std() * np.sqrt(252) * 100
    slope_200 = (sma_200 / sma_200.shift(63) - 1) * 100
    vol_trend_63 = vol_21d_raw.rolling(63).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100 if x.iloc[0] > 0 else 0)

    port_rets = []
    in_upro = False

    for i in range(200, len(returns_df)):
        idx = returns_df.index[i]
        v = vol_21.iloc[i]

        # Compute confluence score
        score = 0.0
        # SHORT: 5d momentum > 0 and RSI > 50
        if not pd.isna(mom_5d.iloc[i]) and mom_5d.iloc[i] > 0:
            score += 0.5
        if not pd.isna(rsi_10.iloc[i]) and rsi_10.iloc[i] > 50:
            score += 0.5
        # MEDIUM: 20/50 cross and vol < 15
        if not pd.isna(sma_20.iloc[i]) and not pd.isna(sma_50.iloc[i]) and sma_20.iloc[i] > sma_50.iloc[i]:
            score += 0.5
        if not pd.isna(v) and v < 15:
            score += 0.5
        # LONG: 200d slope positive and vol trend declining
        if not pd.isna(slope_200.iloc[i]) and slope_200.iloc[i] > 0:
            score += 0.5
        if not pd.isna(vol_trend_63.iloc[i]) and vol_trend_63.iloc[i] < 0:
            score += 0.5

        # September hedge
        if idx.month == 9:
            holding = 'SPY'
            in_upro = False
        elif v > 30:
            holding = 'GLD'
            in_upro = False
        elif v > 15:
            holding = 'SPY'
            in_upro = False
        elif in_upro:
            if score < 2.0:
                holding = 'SPY'
                in_upro = False
            else:
                holding = 'UPRO'
        else:
            if score >= 2.5 and v < 15:
                holding = 'UPRO'
                in_upro = True
            else:
                holding = 'SPY'

        if holding == 'UPRO':
            port_rets.append(upro_ret.iloc[i])
        elif holding == 'GLD':
            port_rets.append(gld_ret.iloc[i])
        else:
            port_rets.append(spy_ret.iloc[i])

    return pd.Series(port_rets, index=returns_df.index[200:])


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ─── Strategy 2: Vol Mean Reversion ────────────────────────────────────────
def run_vol_mean_reversion(prices_df, returns_df, vix_series):
    """
    5-regime VIX system:
    - VIX < 15 and declining: UPRO
    - VIX > 20 and mean-reverting (down 15% from peak): UPRO
    - VIX > 25 and rising: 50% GLD + 50% TLT
    - VIX > 20 and rising: 50% SPY + 50% TLT
    - Otherwise: SPY
    Weekly rebalance.
    """
    spy_ret = returns_df['SPY']
    upro_ret = returns_df['UPRO']
    gld_ret = returns_df['GLD']
    tlt_ret = returns_df['TLT']

    vix_ma10 = vix_series.rolling(10).mean()
    vix_peak20 = vix_series.rolling(20).max()

    port_rets = []
    regime = 'SPY'
    last_week = None

    for i in range(252, len(returns_df)):
        idx = returns_df.index[i]

        # Weekly rebalance
        week = (idx.year, idx.isocalendar()[1])
        if week != last_week:
            last_week = week
            v = vix_series.get(idx, np.nan) if idx in vix_series.index else np.nan
            vm10 = vix_ma10.get(idx, np.nan) if idx in vix_ma10.index else np.nan
            vp20 = vix_peak20.get(idx, np.nan) if idx in vix_peak20.index else np.nan

            if pd.isna(v) or pd.isna(vm10):
                regime = 'SPY'
            elif v < 15 and v < vm10:
                regime = 'UPRO'
            elif v > 20 and v < vp20 * 0.85 and v < vm10:
                regime = 'UPRO_MR'
            elif v > 25 and v > vm10:
                regime = 'DEFENSIVE'
            elif v > 20 and v > vm10:
                regime = 'CAUTIOUS'
            else:
                regime = 'SPY'

        if regime in ('UPRO', 'UPRO_MR'):
            port_rets.append(upro_ret.iloc[i])
        elif regime == 'DEFENSIVE':
            port_rets.append(0.5 * gld_ret.iloc[i] + 0.5 * tlt_ret.iloc[i])
        elif regime == 'CAUTIOUS':
            port_rets.append(0.5 * spy_ret.iloc[i] + 0.5 * tlt_ret.iloc[i])
        else:
            port_rets.append(spy_ret.iloc[i])

    return pd.Series(port_rets, index=returns_df.index[252:])


# ─── Run both strategies ──────────────────────────────────────────────────
print("\nRunning Gameplan v3...")
gp3_rets = run_gameplan_v3(prices, returns)

print("Running Vol Mean Reversion...")
vmr_rets = run_vol_mean_reversion(prices, returns, prices['VIX'])

# Align
aligned = pd.DataFrame({
    'gp3': gp3_rets,
    'vmr': vmr_rets,
    'spy': returns['SPY'],
    'upro': returns['UPRO'],
}).dropna()

print(f"\nAligned period: {len(aligned)} days ({aligned.index[0].strftime('%Y-%m-%d')} to {aligned.index[-1].strftime('%Y-%m-%d')})")


# ─── Metrics ──────────────────────────────────────────────────────────────
def metrics(rets, label):
    ann = 252
    mu = rets.mean() * ann
    sigma = rets.std() * np.sqrt(ann)
    sharpe = mu / sigma if sigma > 0 else 0
    neg = rets[rets < 0]
    sortino = mu / (neg.std() * np.sqrt(ann)) if len(neg) > 0 and neg.std() > 0 else 0
    cum = (1 + rets).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    cagr = (cum.iloc[-1] ** (ann / len(rets))) - 1 if cum.iloc[-1] > 0 else -1
    wr = (rets > 0).mean()
    return {
        'label': label, 'sharpe': sharpe, 'sortino': sortino,
        'cagr': cagr, 'max_dd': max_dd, 'wr': wr,
        'annual_ret': mu, 'annual_vol': sigma,
    }


# Individual strategies
gp3_m = metrics(aligned['gp3'], "Gameplan v3")
vmr_m = metrics(aligned['vmr'], "Vol Mean Rev")
spy_m = metrics(aligned['spy'], "SPY")
upro_m = metrics(aligned['upro'], "UPRO")

print(f"\n{'Strategy':<25} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'WR':>6}")
print("-" * 65)
for m in [spy_m, upro_m, gp3_m, vmr_m]:
    print(f"  {m['label']:<23} {m['sharpe']:8.3f} {m['sortino']:8.3f} "
          f"{m['cagr']*100:7.1f}% {m['max_dd']*100:7.1f}% {m['wr']*100:5.1f}%")


# ─── Correlation Analysis ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

corr = aligned[['gp3', 'vmr', 'spy', 'upro']].corr()
print(f"\n  GP3 vs VMR: {corr.loc['gp3', 'vmr']:.3f}")
print(f"  GP3 vs SPY: {corr.loc['gp3', 'spy']:.3f}")
print(f"  VMR vs SPY: {corr.loc['vmr', 'spy']:.3f}")
print(f"  GP3 vs UPRO: {corr.loc['gp3', 'upro']:.3f}")

# Rolling correlation
rolling_corr = aligned['gp3'].rolling(63).corr(aligned['vmr'])
print(f"  Rolling 63d correlation: mean={rolling_corr.mean():.3f}, "
      f"min={rolling_corr.min():.3f}, max={rolling_corr.max():.3f}")

# Agreement analysis
both_positive = ((aligned['gp3'] > 0) & (aligned['vmr'] > 0)).mean()
both_negative = ((aligned['gp3'] < 0) & (aligned['vmr'] < 0)).mean()
disagree = 1 - both_positive - both_negative
print(f"\n  Both positive: {both_positive*100:.1f}%")
print(f"  Both negative: {both_negative*100:.1f}%")
print(f"  Disagree: {disagree*100:.1f}%")


# ─── Combination Strategies ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMBINATION STRATEGIES")
print("=" * 70)

combos = {
    '50/50 Equal': aligned['gp3'] * 0.5 + aligned['vmr'] * 0.5,
    '60/40 GP3-heavy': aligned['gp3'] * 0.6 + aligned['vmr'] * 0.4,
    '40/60 VMR-heavy': aligned['gp3'] * 0.4 + aligned['vmr'] * 0.6,
    '70/30 GP3-heavy': aligned['gp3'] * 0.7 + aligned['vmr'] * 0.3,
}

# Regime-weighted: use VIX to decide
vix_aligned = prices['VIX'].reindex(aligned.index)
vix_high = vix_aligned > 20

# When VIX high, favor VMR (it has explicit VIX handling)
# When VIX low, favor GP3 (it has better bull market capture)
regime_weight_gp3 = pd.Series(0.6, index=aligned.index)
regime_weight_gp3[vix_high] = 0.3
regime_weighted = aligned['gp3'] * regime_weight_gp3 + aligned['vmr'] * (1 - regime_weight_gp3)
combos['Regime-Weighted'] = regime_weighted

# Best-of: take GP3 when it's positive and VMR when GP3 is negative (next day signal)
# This is look-ahead biased for the actual selection, but tests the theoretical maximum
# In practice, use previous day's return as signal
gp3_signal = aligned['gp3'].shift(1) > 0  # Yesterday GP3 was positive
adaptive = aligned['gp3'].copy()
adaptive[~gp3_signal] = aligned['vmr'][~gp3_signal]
combos['Adaptive (prev-day signal)'] = adaptive

print(f"\n{'Combination':<30} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8}")
print("-" * 65)

combo_results = {}
for name, rets in combos.items():
    m = metrics(rets.dropna(), name)
    combo_results[name] = m
    print(f"  {name:<28} {m['sharpe']:8.3f} {m['sortino']:8.3f} "
          f"{m['cagr']*100:7.1f}% {m['max_dd']*100:7.1f}%")


# ─── Drawdown Comparison ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DRAWDOWN EVENTS COMPARISON")
print("=" * 70)

def get_drawdown_events(rets, threshold=-0.10):
    cum = (1 + rets).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    events = []
    in_dd = False
    start = None
    for i in range(len(dd)):
        if dd.iloc[i] < threshold and not in_dd:
            in_dd = True
            start = dd.index[i]
        elif dd.iloc[i] >= -0.02 and in_dd:
            in_dd = False
            events.append({
                'start': start,
                'end': dd.index[i],
                'depth': dd.loc[start:dd.index[i]].min() if start else dd.iloc[i],
                'duration': (dd.index[i] - start).days if start else 0,
            })
    return events

for name, rets in [('GP3', aligned['gp3']), ('VMR', aligned['vmr']),
                     ('50/50', combos['50/50 Equal']), ('SPY', aligned['spy'])]:
    events = get_drawdown_events(rets)
    total_dd_days = sum(e['duration'] for e in events)
    worst = min((e['depth'] for e in events), default=0)
    print(f"  {name:<10}: {len(events)} events >10%, worst={worst*100:.1f}%, "
          f"total DD days={total_dd_days}")


# ─── Diversification Benefit ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("DIVERSIFICATION BENEFIT")
print("=" * 70)

# Theoretical: if uncorrelated, portfolio vol = sqrt(0.5² * vol1² + 0.5² * vol2²)
vol_gp3 = aligned['gp3'].std() * np.sqrt(252)
vol_vmr = aligned['vmr'].std() * np.sqrt(252)
corr_val = aligned['gp3'].corr(aligned['vmr'])

theoretical_combo_vol = np.sqrt(0.5**2 * vol_gp3**2 + 0.5**2 * vol_vmr**2 +
                                2 * 0.5 * 0.5 * corr_val * vol_gp3 * vol_vmr)
actual_combo_vol = combos['50/50 Equal'].std() * np.sqrt(252)

print(f"\n  GP3 vol: {vol_gp3*100:.1f}%")
print(f"  VMR vol: {vol_vmr*100:.1f}%")
print(f"  Correlation: {corr_val:.3f}")
print(f"  Theoretical 50/50 vol: {theoretical_combo_vol*100:.1f}%")
print(f"  Actual 50/50 vol: {actual_combo_vol*100:.1f}%")
print(f"  Vol reduction vs GP3: {(1 - actual_combo_vol/vol_gp3)*100:.1f}%")
print(f"  Vol reduction vs VMR: {(1 - actual_combo_vol/vol_vmr)*100:.1f}%")

div_ratio = (0.5 * vol_gp3 + 0.5 * vol_vmr) / actual_combo_vol
print(f"  Diversification Ratio: {div_ratio:.3f} (>1.0 means diversification helps)")


# ─── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

best_combo = max(combo_results.items(), key=lambda x: x[1]['sharpe'])
print(f"\n  Best combination: {best_combo[0]}")
print(f"  Sharpe: {best_combo[1]['sharpe']:.3f} (vs GP3 {gp3_m['sharpe']:.3f}, VMR {vmr_m['sharpe']:.3f})")
print(f"  CAGR: {best_combo[1]['cagr']*100:.1f}% (vs GP3 {gp3_m['cagr']*100:.1f}%, VMR {vmr_m['cagr']*100:.1f}%)")
print(f"  MaxDD: {best_combo[1]['max_dd']*100:.1f}% (vs GP3 {gp3_m['max_dd']*100:.1f}%, VMR {vmr_m['max_dd']*100:.1f}%)")

improves_sharpe = best_combo[1]['sharpe'] > max(gp3_m['sharpe'], vmr_m['sharpe'])
improves_dd = best_combo[1]['max_dd'] > min(gp3_m['max_dd'], vmr_m['max_dd'])

if improves_sharpe and improves_dd:
    print(f"\n  ✅ COMBINATION IMPROVES BOTH Sharpe AND MaxDD — worth deploying")
elif improves_sharpe:
    print(f"\n  🟡 COMBINATION IMPROVES Sharpe but not MaxDD — conditional value")
elif improves_dd:
    print(f"\n  🟡 COMBINATION IMPROVES MaxDD but not Sharpe — risk reduction only")
else:
    print(f"\n  ❌ COMBINATION DOES NOT IMPROVE either metric — stick with individual strategies")

# Save results
summary = {
    'run_date': datetime.now().isoformat(),
    'individual': {
        'gp3': {k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in gp3_m.items()},
        'vmr': {k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in vmr_m.items()},
    },
    'combinations': {
        name: {k: float(v) if isinstance(v, (float, np.floating)) else v
               for k, v in m.items()}
        for name, m in combo_results.items()
    },
    'correlation': float(corr_val),
    'diversification_ratio': float(div_ratio),
}

with open(os.path.join(OUT_DIR, 'combination_results.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n  Results saved to {OUT_DIR}/combination_results.json")
print(f"\nDONE.")
