#!/usr/bin/env python3
"""
Weekly Sector ETF Momentum v1 — Rebalance weekly instead of monthly

Hypothesis: Faster rebalancing captures shorter momentum signals.
Monthly momentum has Sharpe 3.96 (v1) and 4.63 (v2 LightGBM).
Weekly rebalancing might:
- Catch momentum reversals faster (reduce drawdown)
- Increase turnover/costs BUT improve timing
- Better regime adaptation (faster defensive shift)

Tests:
A. Weekly rebalance, Top 3, 12-1 momentum
B. Weekly rebalance, Top 5, 12-1 momentum
C. Weekly rebalance, Top 3, defensive shift in bear
D. Weekly rebalance, Top 3, 4-week momentum (faster signal)
E. Bi-weekly rebalance, Top 3, 12-1 momentum
F. Weekly + trailing stop (sell if down >3% from entry)

Full adversarial validation on all variants.
Transaction costs: 20bps per round-trip (same as monthly).
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import json, os, sys
import yfinance as yf

sys.path.insert(0, '/home/jupiter/Lvl3Quant')

print(f"Weekly ETF Momentum v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

ETFS = ['XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB',
        'XLRE', 'XLC', 'QQQ', 'DIA', 'IWM', 'EEM', 'EFA', 'GLD', 'SLV',
        'DBC', 'TLT', 'HYG', 'LQD']
DEFENSIVE = {'GLD', 'TLT', 'XLU', 'XLP', 'LQD'}

# Download all data once
print("Downloading data...")
raw = yf.download(ETFS + ['SPY', '^VIX'], start='2018-01-01', end='2026-07-25', progress=False)
close = raw['Close']
spy_close = close['SPY'] if 'SPY' in close.columns else close.iloc[:, -2]
vix_close = close['^VIX'] if '^VIX' in close.columns else close.iloc[:, -1]
etf_close = close[[c for c in ETFS if c in close.columns]]

# Weekly prices (Friday close)
weekly = etf_close.resample('W-FRI').last().dropna(how='all')
spy_weekly = spy_close.resample('W-FRI').last()
spy_sma200 = spy_close.rolling(200).mean().resample('W-FRI').last()

print(f"Data: {len(weekly)} weeks, {len(weekly.columns)} ETFs")
print(f"Range: {weekly.index[0].strftime('%Y-%m-%d')} to {weekly.index[-1].strftime('%Y-%m-%d')}")

# Also keep monthly for 12-1 momentum calculation
monthly = etf_close.resample('ME').last().dropna(how='all')


def compute_momentum(weekly_prices, date, lookback_weeks=None, monthly_prices=None):
    """Compute momentum scores for each ETF at a given weekly date."""
    scores = {}

    for etf in weekly_prices.columns:
        try:
            idx = weekly_prices.index.get_loc(date)
            price_now = float(weekly_prices.iloc[idx][etf])

            if pd.isna(price_now):
                continue

            if lookback_weeks is not None:
                # Use weekly lookback
                if idx < lookback_weeks:
                    continue
                price_past = float(weekly_prices.iloc[idx - lookback_weeks][etf])
                if pd.isna(price_past) or price_past == 0:
                    continue
                # Skip last week (momentum reversal at very short horizons)
                price_1w = float(weekly_prices.iloc[idx - 1][etf])
                if pd.isna(price_1w) or price_1w == 0:
                    continue
                score = (price_1w / price_past) - 1  # lookback minus last week
            else:
                # Use 12-1 monthly momentum (standard)
                # Find closest monthly date
                if monthly_prices is None:
                    continue
                monthly_idx = monthly_prices.index.get_indexer([date], method='ffill')[0]
                if monthly_idx < 12:
                    continue
                m12 = float(monthly_prices.iloc[monthly_idx][etf])
                m0 = float(monthly_prices.iloc[monthly_idx - 12][etf])
                m1 = float(monthly_prices.iloc[monthly_idx - 1][etf])

                if pd.isna(m0) or pd.isna(m12) or pd.isna(m1) or m0 == 0:
                    continue
                score = (m1 / m0) - 1  # 12-1 momentum

            scores[etf] = score
        except:
            continue

    return scores


def run_variant(name, weekly_prices, monthly_prices, spy_weekly_prices, spy_sma,
                top_k=3, lookback_weeks=None, defensive_shift=False,
                trailing_stop=None, rebalance_freq=1):
    """Run a single variant and return weekly returns."""
    print(f"\n--- {name} ---")

    rets = []
    dates = []
    regimes = []

    # Start after enough data
    start_idx = max(52, 13 if lookback_weeks and lookback_weeks < 52 else 52)

    current_holdings = []
    entry_prices = {}  # for trailing stop

    for i in range(start_idx, len(weekly_prices) - 1):
        date = weekly_prices.index[i]
        next_date = weekly_prices.index[i + 1]

        # Check regime
        is_bear = False
        if date in spy_weekly_prices.index and date in spy_sma.index:
            s = spy_weekly_prices.loc[date]
            sma = spy_sma.loc[date]
            if not pd.isna(s) and not pd.isna(sma):
                is_bear = bool(s < sma)

        # Only rebalance on schedule
        should_rebalance = (i - start_idx) % rebalance_freq == 0

        if should_rebalance:
            scores = compute_momentum(weekly_prices, date,
                                     lookback_weeks=lookback_weeks,
                                     monthly_prices=monthly_prices)

            if len(scores) < top_k:
                rets.append(0.0)
                dates.append(next_date)
                regimes.append(1 if is_bear else 0)
                continue

            # Apply defensive shift
            if defensive_shift and is_bear:
                for etf in scores:
                    if etf in DEFENSIVE:
                        scores[etf] += 0.05
                    else:
                        scores[etf] -= 0.02

            new_holdings = [t for t, _ in sorted(scores.items(),
                           key=lambda x: x[1], reverse=True)[:top_k]]

            # Transaction cost: only pay for changes
            turnover = len(set(new_holdings) - set(current_holdings))
            cost = turnover * 0.002 / top_k  # 20bps per changed position, weighted

            current_holdings = new_holdings
            # Reset entry prices for trailing stop
            for h in current_holdings:
                entry_prices[h] = float(weekly_prices.loc[date, h]) if h in weekly_prices.columns else 0
        else:
            cost = 0.0

        # Trailing stop check
        if trailing_stop and current_holdings:
            for h in list(current_holdings):
                if h in entry_prices and entry_prices[h] > 0:
                    current_price = float(weekly_prices.loc[date, h]) if h in weekly_prices.columns and not pd.isna(weekly_prices.loc[date, h]) else entry_prices[h]
                    drawdown = (current_price / entry_prices[h]) - 1
                    if drawdown < -trailing_stop:
                        current_holdings.remove(h)
                        cost += 0.002  # sell cost

        # Calculate return
        if current_holdings:
            period_rets = []
            for h in current_holdings:
                try:
                    r = float(weekly_prices.loc[next_date, h] / weekly_prices.loc[date, h] - 1)
                    if not np.isnan(r):
                        period_rets.append(r)
                except:
                    continue

            if period_rets:
                port_ret = np.mean(period_rets) - cost
            else:
                port_ret = -cost
        else:
            port_ret = 0.0

        rets.append(port_ret)
        dates.append(next_date)
        regimes.append(1 if is_bear else 0)

    series = pd.Series(rets, index=dates)

    if len(series) > 10:
        sharpe = series.mean() / series.std() * np.sqrt(52) if series.std() > 0 else 0
        cum = (1 + series).cumprod()
        years = len(series) / 52
        cagr = float(cum.iloc[-1] ** (1/years) - 1) if years > 0 else 0
        maxdd = float(((cum / cum.cummax()) - 1).min())
        wr = float((series > 0).mean())

        # Turnover estimate
        total_trades = sum(1 for i in range(1, len(rets)) if rets[i] != 0)

        print(f"  {len(series)} weeks | Sharpe {sharpe:.2f} | CAGR {cagr:.1%} | MaxDD {maxdd:.1%} | WR {wr:.1%}")

    return series, regimes


def adversarial_validation(returns, regimes, name):
    """4-gate validation."""
    print(f"\n  GATES for {name}:")
    gates = 0

    if len(returns) < 20:
        print("    INSUFFICIENT DATA")
        return {'sharpe': 0, 'gates': 0}

    real_sharpe = returns.mean() / returns.std() * np.sqrt(52) if returns.std() > 0 else 0

    # G1: Permutation (block shuffle, 4-week blocks for weekly data)
    vals = returns.values
    block = 4
    n_blocks = len(vals) // block
    perm_beat = 0
    for _ in range(1000):
        blocks = [vals[i*block:(i+1)*block] for i in range(n_blocks)]
        rem = vals[n_blocks*block:]
        np.random.shuffle(blocks)
        shuffled = np.concatenate(blocks + ([rem] if len(rem) > 0 else []))
        ss = np.mean(shuffled) / (np.std(shuffled) + 1e-10) * np.sqrt(52)
        if ss >= real_sharpe:
            perm_beat += 1
    p_val = perm_beat / 1000
    g1 = p_val < 0.05
    gates += g1
    print(f"    G1 Perm: p={p_val:.3f} {'✅' if g1 else '❌'}")

    # G2: Regime
    reg_arr = np.array(regimes[:len(returns)])
    bull_mask = reg_arr == 0
    bear_mask = reg_arr == 1
    bull_r = returns.values[bull_mask]
    bear_r = returns.values[bear_mask]
    bull_sh = np.mean(bull_r) / (np.std(bull_r) + 1e-10) * np.sqrt(52) if len(bull_r) > 10 else 0
    bear_sh = np.mean(bear_r) / (np.std(bear_r) + 1e-10) * np.sqrt(52) if len(bear_r) > 10 else 0
    gap = abs(bull_sh - bear_sh) / max(abs(bull_sh), abs(bear_sh), 0.01)
    g2 = gap < 0.50
    gates += g2
    print(f"    G2 Regime: Bull {bull_sh:.2f} ({bull_mask.sum()}w), Bear {bear_sh:.2f} ({bear_mask.sum()}w), gap {gap:.3f} {'✅' if g2 else '❌'}")

    # G3: Sub-period
    mid = len(returns) // 2
    h1 = returns.iloc[:mid]
    h2 = returns.iloc[mid:]
    h1_sh = h1.mean() / h1.std() * np.sqrt(52) if h1.std() > 0 else 0
    h2_sh = h2.mean() / h2.std() * np.sqrt(52) if h2.std() > 0 else 0
    g3 = h1_sh > 0 and h2_sh > 0
    gates += g3
    print(f"    G3 Sub: H1 {h1_sh:.2f}, H2 {h2_sh:.2f} {'✅' if g3 else '❌'}")

    # G4: Outlier
    trimmed = returns[(returns >= returns.quantile(0.01)) & (returns <= returns.quantile(0.99))]
    trim_sh = trimmed.mean() / trimmed.std() * np.sqrt(52) if len(trimmed) > 10 and trimmed.std() > 0 else 0
    g4 = trim_sh > 0
    gates += g4
    print(f"    G4 Outlier: {trim_sh:.2f} {'✅' if g4 else '❌'}")

    # Metrics
    cum = (1 + returns).cumprod()
    years = len(returns) / 52
    cagr = float(cum.iloc[-1] ** (1/years) - 1) if years > 0 else 0
    maxdd = float(((cum / cum.cummax()) - 1).min())
    down = returns[returns < 0]
    sortino = float(returns.mean() / down.std() * np.sqrt(52)) if len(down) > 0 and down.std() > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    pf = float(returns[returns > 0].sum()) / (abs(float(returns[returns < 0].sum())) + 1e-10)
    wr = float((returns > 0).mean())

    print(f"    Sharpe {real_sharpe:.2f} | Sortino {sortino:.2f} | CAGR {cagr:.1%} | MaxDD {maxdd:.1%} | "
          f"Calmar {calmar:.2f} | PF {pf:.2f} | WR {wr:.1%} | Gates {gates}/4")

    return {'sharpe': float(real_sharpe), 'sortino': sortino, 'cagr': cagr, 'max_dd': maxdd,
            'calmar': calmar, 'pf': pf, 'wr': wr, 'gates': gates, 'perm_p': float(p_val),
            'r1_gap': float(gap), 'bull_sharpe': float(bull_sh), 'bear_sharpe': float(bear_sh)}


# Run all variants
results = {}

# A. Weekly, Top 3, 12-1 monthly momentum
rets_a, reg_a = run_variant('A_Weekly_Top3_12-1', weekly, monthly, spy_weekly, spy_sma200,
                             top_k=3, lookback_weeks=None)
results['A_Weekly_Top3'] = adversarial_validation(rets_a, reg_a, 'A')

# B. Weekly, Top 5, 12-1 monthly momentum
rets_b, reg_b = run_variant('B_Weekly_Top5_12-1', weekly, monthly, spy_weekly, spy_sma200,
                             top_k=5, lookback_weeks=None)
results['B_Weekly_Top5'] = adversarial_validation(rets_b, reg_b, 'B')

# C. Weekly, Top 3, defensive shift
rets_c, reg_c = run_variant('C_Weekly_Top3_DefShift', weekly, monthly, spy_weekly, spy_sma200,
                             top_k=3, lookback_weeks=None, defensive_shift=True)
results['C_Weekly_DefShift'] = adversarial_validation(rets_c, reg_c, 'C')

# D. Weekly, Top 3, 4-week momentum (faster signal)
rets_d, reg_d = run_variant('D_Weekly_Top3_4wMom', weekly, monthly, spy_weekly, spy_sma200,
                             top_k=3, lookback_weeks=4)
results['D_4wMom'] = adversarial_validation(rets_d, reg_d, 'D')

# E. Bi-weekly, Top 3, 12-1 momentum
rets_e, reg_e = run_variant('E_Biweekly_Top3', weekly, monthly, spy_weekly, spy_sma200,
                             top_k=3, lookback_weeks=None, rebalance_freq=2)
results['E_Biweekly'] = adversarial_validation(rets_e, reg_e, 'E')

# F. Weekly + trailing stop 3%
rets_f, reg_f = run_variant('F_Weekly_Top3_TrailStop3', weekly, monthly, spy_weekly, spy_sma200,
                             top_k=3, lookback_weeks=None, trailing_stop=0.03)
results['F_TrailStop3'] = adversarial_validation(rets_f, reg_f, 'F')

# G. Weekly, Top 3, 13-week momentum (quarterly)
rets_g, reg_g = run_variant('G_Weekly_Top3_13wMom', weekly, monthly, spy_weekly, spy_sma200,
                             top_k=3, lookback_weeks=13)
results['G_13wMom'] = adversarial_validation(rets_g, reg_g, 'G')

# H. Weekly, Top 3, defensive shift + trailing stop
rets_h, reg_h = run_variant('H_Weekly_DefShift_Trail', weekly, monthly, spy_weekly, spy_sma200,
                             top_k=3, lookback_weeks=None, defensive_shift=True, trailing_stop=0.03)
results['H_DefShift_Trail'] = adversarial_validation(rets_h, reg_h, 'H')

# Summary
print(f"\n{'='*70}")
print("SUMMARY — WEEKLY ETF MOMENTUM v1")
print(f"{'='*70}")
print(f"{'Variant':<22s} {'Sharpe':>7s} {'Sort':>6s} {'CAGR':>7s} {'MaxDD':>7s} {'R1gap':>6s} {'WR':>5s} {'G':>3s}")
print("-" * 65)
for name in sorted(results):
    r = results[name]
    print(f"{name:<22s} {r['sharpe']:7.2f} {r['sortino']:6.2f} {r['cagr']:7.1%} "
          f"{r['max_dd']:7.1%} {r.get('r1_gap',0):6.3f} {r['wr']:5.1%} {r['gates']:>2d}/4")

# Compare vs monthly baseline
print(f"\nBENCHMARK: Monthly v1 Sharpe 3.96, Monthly v2 (LightGBM) Sharpe 4.63")

best = max(results.items(), key=lambda x: (x[1]['gates'], x[1]['sharpe']))
print(f"\n🏆 BEST: {best[0]} — Sharpe {best[1]['sharpe']:.2f}, R1 gap {best[1].get('r1_gap',0):.3f}, Gates {best[1]['gates']}/4")

# Save
save_path = '/home/jupiter/Lvl3Quant/research/findings/weekly_etf_momentum_v1_results.json'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w') as f:
    json.dump({
        'strategy': 'Weekly ETF Momentum v1',
        'run_date': datetime.now().isoformat(),
        'n_weeks': len(weekly),
        'variants': results,
        'best': best[0],
        'best_metrics': best[1]
    }, f, indent=2)
print(f"\nSaved → {save_path}")
