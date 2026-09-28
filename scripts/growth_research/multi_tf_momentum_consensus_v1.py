#!/usr/bin/env python3
"""
Multi-Timeframe Momentum Consensus v1 — HC #750 Compliant

Only trade when MULTIPLE independent signals agree. Tests:
1. Momentum consensus (1m + 3m + 6m + 12m all positive)
2. Momentum + quality (RSI not overbought + low vol)
3. Momentum + macro (VIX regime + SPY trend)
4. Full confluence (momentum + quality + macro — all must confirm)

Compared to baseline (12-1 momentum only, no filtering).
This directly tests whether HC #750's multi-signal requirement helps or hurts.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import json, os, sys
import yfinance as yf

sys.path.insert(0, '/home/jupiter/Lvl3Quant')

print(f"Multi-TF Momentum Consensus v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
spy_close = close['SPY'] if 'SPY' in close.columns else None
vix_close = close['^VIX'] if '^VIX' in close.columns else None

monthly = etf_close.resample('ME').last().dropna(how='all')
daily_ret = etf_close.pct_change()

# Compute all signals
mom_1m = monthly.pct_change(1)
mom_3m = monthly.pct_change(3)
mom_6m = monthly.pct_change(6)
mom_12m = monthly.pct_change(12)
mom_12_1 = mom_12m - mom_1m

# RSI (14-period on monthly)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

rsi_14 = pd.DataFrame({etf: compute_rsi(monthly[etf]) for etf in monthly.columns})

# Volatility (annualized from daily)
vol_21d = daily_ret.rolling(21).std().resample('ME').last() * np.sqrt(252)
vol_63d = daily_ret.rolling(63).std().resample('ME').last() * np.sqrt(252)

# SPY regime
if spy_close is not None:
    spy_monthly = spy_close.resample('ME').last()
    spy_sma200 = spy_close.rolling(200).mean().resample('ME').last()
    spy_ret = spy_monthly.pct_change()
    spy_above_sma = (spy_monthly > spy_sma200).astype(int)
else:
    spy_monthly = None

# VIX level
if vix_close is not None:
    vix_monthly = vix_close.resample('ME').last()
    vix_percentile = vix_monthly.rolling(252).rank(pct=True)
else:
    vix_monthly = None

print(f"Data: {len(monthly)} months, {len(monthly.columns)} ETFs")


def run_variant(name, top_k=3, require_all_mom_positive=False,
                require_rsi_ok=False, require_macro_ok=False,
                require_low_vol=False, defensive_shift=False,
                min_confirming_signals=0):
    """Run a variant with specified confluence filters."""
    print(f"\n--- {name} ---")

    rets, dates, regimes = [], [], []
    signals_log = []  # HC #750 R4: audit trail

    for i in range(13, len(monthly) - 1):
        dt = monthly.index[i]
        dt_next = monthly.index[i + 1]

        # Regime
        is_bear = False
        if spy_monthly is not None and dt in spy_above_sma.index:
            is_bear = bool(spy_above_sma.loc[dt] == 0)

        # Score each ETF
        candidates = {}

        for etf in monthly.columns:
            try:
                # Base: 12-1 momentum score
                m12_1 = float(mom_12_1.loc[dt, etf]) if not pd.isna(mom_12_1.loc[dt, etf]) else None
                if m12_1 is None:
                    continue

                # Count confirming signals
                confirming = 0
                total_signals = 0
                signal_detail = {}

                # Signal 1: 12-1 momentum positive
                total_signals += 1
                if m12_1 > 0:
                    confirming += 1
                    signal_detail['mom_12_1'] = 'CONFIRM'
                else:
                    signal_detail['mom_12_1'] = 'REJECT'

                # Signal 2: All timeframe momentum positive
                m1 = float(mom_1m.loc[dt, etf]) if not pd.isna(mom_1m.loc[dt, etf]) else 0
                m3 = float(mom_3m.loc[dt, etf]) if not pd.isna(mom_3m.loc[dt, etf]) else 0
                m6 = float(mom_6m.loc[dt, etf]) if not pd.isna(mom_6m.loc[dt, etf]) else 0
                m12 = float(mom_12m.loc[dt, etf]) if not pd.isna(mom_12m.loc[dt, etf]) else 0

                all_positive = m1 > 0 and m3 > 0 and m6 > 0 and m12 > 0
                total_signals += 1
                if all_positive:
                    confirming += 1
                    signal_detail['all_tf_mom'] = 'CONFIRM'
                else:
                    signal_detail['all_tf_mom'] = 'REJECT'

                # Signal 3: RSI not overbought (< 70)
                rsi_val = float(rsi_14.loc[dt, etf]) if dt in rsi_14.index and not pd.isna(rsi_14.loc[dt, etf]) else 50
                total_signals += 1
                if rsi_val < 70:
                    confirming += 1
                    signal_detail['rsi_ok'] = 'CONFIRM'
                else:
                    signal_detail['rsi_ok'] = 'REJECT'

                # Signal 4: Volatility reasonable (not extreme)
                v63 = float(vol_63d.loc[dt, etf]) if dt in vol_63d.index and not pd.isna(vol_63d.loc[dt, etf]) else 0.2
                total_signals += 1
                if v63 < 0.40:  # <40% annualized vol
                    confirming += 1
                    signal_detail['vol_ok'] = 'CONFIRM'
                else:
                    signal_detail['vol_ok'] = 'REJECT'

                # Signal 5: Macro (SPY above SMA200 OR defensive ETF)
                total_signals += 1
                macro_ok = not is_bear or etf in DEFENSIVE
                if macro_ok:
                    confirming += 1
                    signal_detail['macro_ok'] = 'CONFIRM'
                else:
                    signal_detail['macro_ok'] = 'REJECT'

                # Apply filters
                if require_all_mom_positive and not all_positive:
                    continue
                if require_rsi_ok and rsi_val >= 70:
                    continue
                if require_low_vol and v63 >= 0.40:
                    continue
                if require_macro_ok and not macro_ok:
                    continue
                if min_confirming_signals > 0 and confirming < min_confirming_signals:
                    continue

                score = m12_1

                # Defensive shift in bear markets
                if defensive_shift and is_bear:
                    if etf in DEFENSIVE:
                        score += 0.05
                    else:
                        score -= 0.02

                candidates[etf] = {
                    'score': score,
                    'confirming': confirming,
                    'total': total_signals,
                    'signals': signal_detail
                }

            except:
                continue

        if len(candidates) < top_k:
            # Not enough candidates — stay flat
            rets.append(0.0)
            dates.append(dt_next)
            regimes.append(1 if is_bear else 0)
            signals_log.append({'date': str(dt), 'action': 'FLAT', 'reason': f'Only {len(candidates)} candidates'})
            continue

        # Select top_k by score
        top = sorted(candidates.items(), key=lambda x: x[1]['score'], reverse=True)[:top_k]

        # Portfolio return
        month_rets = []
        for ticker, info in top:
            try:
                r = float(monthly.loc[dt_next, ticker] / monthly.loc[dt, ticker] - 1)
                if not np.isnan(r):
                    month_rets.append(r)
            except:
                continue

        if month_rets:
            port_ret = np.mean(month_rets) - 0.002  # 20bps cost
        else:
            port_ret = 0.0

        rets.append(port_ret)
        dates.append(dt_next)
        regimes.append(1 if is_bear else 0)

        # Log signals
        signals_log.append({
            'date': str(dt),
            'picks': [t for t, _ in top],
            'avg_confirming': np.mean([info['confirming'] for _, info in top]),
            'is_bear': is_bear
        })

    series = pd.Series(rets, index=dates)

    # Compute trade frequency (non-zero months)
    active_months = (series != 0).sum()
    total_months = len(series)

    if len(series) > 10:
        sharpe = series.mean() / series.std() * np.sqrt(12) if series.std() > 0 else 0
        cum = (1 + series).cumprod()
        years = len(series) / 12
        cagr = float(cum.iloc[-1] ** (1/years) - 1) if years > 0 else 0
        maxdd = float(((cum / cum.cummax()) - 1).min())
        wr = float((series[series != 0] > 0).mean()) if (series != 0).any() else 0
        print(f"  {len(series)} months ({active_months} active) | Sharpe {sharpe:.2f} | CAGR {cagr:.1%} | MaxDD {maxdd:.1%} | WR {wr:.1%}")

    return series, regimes, signals_log


def adversarial_4gate(returns, regimes, name):
    """Standard 4-gate validation."""
    if len(returns) < 20:
        print(f"  {name}: INSUFFICIENT DATA")
        return {'sharpe': 0, 'gates': 0}

    gates = 0
    real_sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else 0

    # G1: Block permutation
    vals = returns.values
    block = 3
    n_blocks = len(vals) // block
    perm_beat = 0
    for _ in range(1000):
        blocks = [vals[i*block:(i+1)*block] for i in range(n_blocks)]
        rem = vals[n_blocks*block:]
        np.random.shuffle(blocks)
        sh = np.concatenate(blocks + ([rem] if len(rem) > 0 else []))
        ss = np.mean(sh) / (np.std(sh) + 1e-10) * np.sqrt(12)
        if ss >= real_sharpe:
            perm_beat += 1
    p_val = perm_beat / 1000
    g1 = p_val < 0.05
    gates += g1

    # G2: Regime
    reg = np.array(regimes[:len(returns)])
    bull_r = returns.values[reg == 0]
    bear_r = returns.values[reg == 1]
    bull_sh = np.mean(bull_r) / (np.std(bull_r) + 1e-10) * np.sqrt(12) if len(bull_r) > 5 else 0
    bear_sh = np.mean(bear_r) / (np.std(bear_r) + 1e-10) * np.sqrt(12) if len(bear_r) > 5 else 0
    gap = abs(bull_sh - bear_sh) / max(abs(bull_sh), abs(bear_sh), 0.01)
    g2 = gap < 0.50
    gates += g2

    # G3: Sub-period
    mid = len(returns) // 2
    h1_sh = returns.iloc[:mid].mean() / returns.iloc[:mid].std() * np.sqrt(12) if returns.iloc[:mid].std() > 0 else 0
    h2_sh = returns.iloc[mid:].mean() / returns.iloc[mid:].std() * np.sqrt(12) if returns.iloc[mid:].std() > 0 else 0
    g3 = h1_sh > 0 and h2_sh > 0
    gates += g3

    # G4: Outlier removal
    trimmed = returns[(returns >= returns.quantile(0.01)) & (returns <= returns.quantile(0.99))]
    trim_sh = trimmed.mean() / trimmed.std() * np.sqrt(12) if len(trimmed) > 5 and trimmed.std() > 0 else 0
    g4 = trim_sh > 0
    gates += g4

    # Metrics
    cum = (1 + returns).cumprod()
    years = len(returns) / 12
    cagr = float(cum.iloc[-1] ** (1/years) - 1) if years > 0 else 0
    maxdd = float(((cum / cum.cummax()) - 1).min())
    down = returns[returns < 0]
    sortino = float(returns.mean() / down.std() * np.sqrt(12)) if len(down) > 0 and down.std() > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    pf = float(returns[returns > 0].sum()) / (abs(float(returns[returns < 0].sum())) + 1e-10)
    wr = float((returns > 0).mean())
    active = float((returns != 0).mean())

    print(f"  G1 p={p_val:.3f}{'✅' if g1 else '❌'} G2 gap={gap:.3f}{'✅' if g2 else '❌'} "
          f"G3 H1={h1_sh:.2f}/H2={h2_sh:.2f}{'✅' if g3 else '❌'} G4={trim_sh:.2f}{'✅' if g4 else '❌'}")
    print(f"  Sharpe {real_sharpe:.2f} | Sortino {sortino:.2f} | CAGR {cagr:.1%} | MaxDD {maxdd:.1%} | "
          f"PF {pf:.2f} | WR {wr:.1%} | Active {active:.0%} | Gates {gates}/4")

    return {'sharpe': float(real_sharpe), 'sortino': sortino, 'cagr': cagr, 'max_dd': maxdd,
            'calmar': calmar, 'pf': pf, 'wr': wr, 'gates': gates, 'perm_p': float(p_val),
            'r1_gap': float(gap), 'bull_sharpe': float(bull_sh), 'bear_sharpe': float(bear_sh),
            'active_pct': active}


# ============ RUN ALL VARIANTS ============

results = {}

# A. Baseline: 12-1 momentum only (no confluence)
r, reg, _ = run_variant('A_Baseline_12-1', top_k=3)
results['A_Baseline'] = adversarial_4gate(r, reg, 'A')

# B. All timeframes positive (1m + 3m + 6m + 12m)
r, reg, _ = run_variant('B_AllTF_Positive', top_k=3, require_all_mom_positive=True)
results['B_AllTF'] = adversarial_4gate(r, reg, 'B')

# C. Momentum + RSI filter (not overbought)
r, reg, _ = run_variant('C_Mom_RSI', top_k=3, require_rsi_ok=True)
results['C_MomRSI'] = adversarial_4gate(r, reg, 'C')

# D. Momentum + macro (SPY above SMA200 or defensive)
r, reg, _ = run_variant('D_Mom_Macro', top_k=3, require_macro_ok=True, defensive_shift=True)
results['D_MomMacro'] = adversarial_4gate(r, reg, 'D')

# E. Momentum + low vol filter
r, reg, _ = run_variant('E_Mom_LowVol', top_k=3, require_low_vol=True)
results['E_MomLowVol'] = adversarial_4gate(r, reg, 'E')

# F. Full confluence (≥4 of 5 signals must confirm)
r, reg, log = run_variant('F_Full_Confluence_4of5', top_k=3, min_confirming_signals=4)
results['F_4of5'] = adversarial_4gate(r, reg, 'F')

# G. Strict confluence (all 5 signals must confirm)
r, reg, _ = run_variant('G_Strict_5of5', top_k=3, min_confirming_signals=5)
results['G_5of5'] = adversarial_4gate(r, reg, 'G')

# H. All TF + defensive shift + RSI (practical best)
r, reg, _ = run_variant('H_AllTF_DefShift_RSI', top_k=3,
                         require_all_mom_positive=True, require_rsi_ok=True,
                         defensive_shift=True)
results['H_Best_Combo'] = adversarial_4gate(r, reg, 'H')

# I. Top 5 with ≥3 confirming signals (wider + medium filter)
r, reg, _ = run_variant('I_Top5_3of5', top_k=5, min_confirming_signals=3)
results['I_Top5_3of5'] = adversarial_4gate(r, reg, 'I')

# Summary
print(f"\n{'='*70}")
print("SUMMARY — MULTI-TF MOMENTUM CONSENSUS v1")
print(f"{'='*70}")
print(f"{'Variant':<20s} {'Sharpe':>7s} {'Sort':>6s} {'CAGR':>7s} {'MaxDD':>7s} {'R1gap':>6s} {'WR':>5s} {'Act%':>5s} {'G':>3s}")
print("-" * 72)
for name in sorted(results):
    r = results[name]
    print(f"{name:<20s} {r['sharpe']:7.2f} {r['sortino']:6.2f} {r['cagr']:7.1%} "
          f"{r['max_dd']:7.1%} {r.get('r1_gap',0):6.3f} {r['wr']:5.1%} {r.get('active_pct',1):5.0%} {r['gates']:>2d}/4")

best = max(results.items(), key=lambda x: (x[1]['gates'], x[1]['sharpe']))
print(f"\n🏆 BEST: {best[0]} — Sharpe {best[1]['sharpe']:.2f}, R1 gap {best[1].get('r1_gap',0):.3f}, Gates {best[1]['gates']}/4")

# Key comparison
baseline = results.get('A_Baseline', {})
print(f"\nCONFLUENCE IMPACT (vs baseline):")
for name, r in sorted(results.items()):
    if name == 'A_Baseline':
        continue
    sharpe_diff = r['sharpe'] - baseline.get('sharpe', 0)
    active_diff = r.get('active_pct', 1) - baseline.get('active_pct', 1)
    r1_diff = r.get('r1_gap', 0) - baseline.get('r1_gap', 0)
    print(f"  {name:<20s}: Sharpe {sharpe_diff:+.2f}, R1 gap {r1_diff:+.3f}, Activity {active_diff:+.0%}")

# Save
save_path = '/home/jupiter/Lvl3Quant/research/findings/multi_tf_consensus_v1_results.json'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w') as f:
    json.dump({
        'strategy': 'Multi-TF Momentum Consensus v1',
        'run_date': datetime.now().isoformat(),
        'variants': results,
        'best': best[0],
        'best_metrics': best[1]
    }, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
print(f"\nSaved → {save_path}")
