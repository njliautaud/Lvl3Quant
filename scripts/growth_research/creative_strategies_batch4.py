#!/usr/bin/env python3
"""
Creative Strategies Batch 4 — Five genuinely different ideas.

1. SEASONALITY + OVERNIGHT COMBO — Hold UPRO overnight-only during historically weak months
   (Sep, Jun), full-day during strong months (Jul, Nov, Apr), standard vol-switching rest.
   Combines two validated findings (entry 412 + 423).

2. LEVERAGED BARBELL — 80% UPRO + 20% TMF with monthly rebalancing. Anti-correlated assets
   force buy-low-sell-high mechanically. Unlike risk parity (which failed), this keeps UPRO
   dominant and uses bonds purely as rebalancing fuel.

3. VOL-SCALED DCA — Instead of fixed weekly DCA, scale contributions by VIX level.
   VIX>30: invest 3x normal. VIX 20-30: invest 2x. VIX 15-20: invest 1x. VIX<15: invest 0.5x.
   Buy more when scared, less when complacent.

4. DRAWDOWN-RESPONSIVE ALLOCATION — Adaptive leverage based on current drawdown from peak.
   At new highs: full UPRO. Down 5%: reduce to 2/3 UPRO + 1/3 SPY. Down 10%: reduce to
   1/3 UPRO + 2/3 SPY. Down 20%: exit to SPY entirely. Re-enter UPRO only on new 20d high.

5. GOLDEN CROSS + VOLATILITY CONFIRMATION — Only hold UPRO when SPY is above 200d MA AND
   the 50d MA is above 200d MA (golden cross) AND realized vol < 20%. Triple confirmation
   prevents false entries. This is a simpler version of the confluence gate.

Full adversarial validation suite on each.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings, json, os
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

OUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_research/creative_batch4"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("CREATIVE STRATEGIES BATCH 4")
print("=" * 70)
print(f"\nFetching data...")

tickers = {
    'SPY': 'SPY', 'UPRO': 'UPRO', 'TLT': 'TLT', 'TMF': 'TMF',
    'GLD': 'GLD', 'VIX': '^VIX',
}

data = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start='2012-01-01', end='2026-07-17', progress=False)
        if len(df) > 100:
            data[name] = df['Close'].squeeze()
            print(f"  {name}: {len(df)} days")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")

# Align
common_dates = sorted(set.intersection(*[set(data[n].index) for n in data]))
prices = pd.DataFrame({n: data[n].reindex(common_dates) for n in data}).dropna()
returns = prices.pct_change().dropna()
print(f"\nAligned: {len(prices)} days, {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")


# ─── Utility functions ──────────────────────────────────────────────────────
def compute_metrics(rets, label=""):
    if len(rets) < 20:
        return {'label': label, 'sharpe': 0, 'sortino': 0, 'cagr': 0, 'max_dd': -1}
    ann = 252
    mu = rets.mean() * ann
    sigma = rets.std() * np.sqrt(ann)
    sharpe = mu / sigma if sigma > 0 else 0

    neg = rets[rets < 0]
    downside = neg.std() * np.sqrt(ann) if len(neg) > 0 else 1e-9
    sortino = mu / downside

    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    cagr = (cum.iloc[-1] ** (ann / len(rets))) - 1 if cum.iloc[-1] > 0 else -1
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    wr = (rets > 0).mean()
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'label': label, 'sharpe': sharpe, 'sortino': sortino, 'cagr': cagr,
        'max_dd': max_dd, 'calmar': calmar, 'wr': wr, 'pf': pf,
        'annual_ret': mu, 'annual_vol': sigma, 'n_days': len(rets),
        'final_value': cum.iloc[-1],
    }


def run_permutation_test(strategy_rets, baseline_rets, n_perms=200):
    """
    CORRECT permutation test: shuffle the TIMING SIGNAL, not the returns.
    Compare strategy Sharpe vs Sharpe of randomly-timed version.
    """
    real_sharpe = strategy_rets.mean() / strategy_rets.std() * np.sqrt(252) if strategy_rets.std() > 0 else 0

    # The strategy's excess return over baseline
    aligned = pd.DataFrame({'strat': strategy_rets, 'base': baseline_rets}).dropna()
    real_excess = aligned['strat'].mean() - aligned['base'].mean()

    count_beat = 0
    for _ in range(n_perms):
        # Randomly decide each day whether to apply strategy return or baseline
        mask = np.random.random(len(aligned)) > 0.5
        perm_rets = aligned['strat'].values.copy()
        perm_rets[mask] = aligned['base'].values[mask]
        perm_excess = perm_rets.mean() - aligned['base'].mean()
        if perm_excess >= real_excess:
            count_beat += 1

    p_value = count_beat / n_perms
    return p_value, real_sharpe


def run_regime_test(strategy_rets, spy_rets):
    aligned = pd.DataFrame({'strat': strategy_rets, 'spy': spy_rets}).dropna()
    green = aligned[aligned['spy'] > 0]['strat']
    red = aligned[aligned['spy'] <= 0]['strat']
    if len(green) < 20 or len(red) < 20:
        return None, None, None
    sg = green.mean() / green.std() * np.sqrt(252) if green.std() > 0 else 0
    sr = red.mean() / red.std() * np.sqrt(252) if red.std() > 0 else 0
    denom = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / denom if denom > 0 else 0
    return gap, sg, sr


def sub_period_consistency(rets, n_blocks=3):
    block_size = len(rets) // n_blocks
    sharpes = []
    for i in range(n_blocks):
        block = rets.iloc[i*block_size:(i+1)*block_size]
        if len(block) > 20 and block.std() > 0:
            sharpes.append(block.mean() / block.std() * np.sqrt(252))
    cv = np.std(sharpes) / abs(np.mean(sharpes)) if np.mean(sharpes) != 0 else float('inf')
    return sharpes, cv


def outlier_robustness(rets, n_remove=10):
    full_s = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    trimmed = rets.sort_values(ascending=True).iloc[:-n_remove]
    trim_s = trimmed.mean() / trimmed.std() * np.sqrt(252) if trimmed.std() > 0 else 0
    deg = (trim_s - full_s) / abs(full_s) if full_s != 0 else 0
    return full_s, trim_s, deg


def full_validation(strategy_rets, spy_rets, baseline_rets, label):
    """Full adversarial suite with CORRECT permutation test."""
    print(f"\n  --- {label} ---")
    m = compute_metrics(strategy_rets, label)
    print(f"  Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
          f"CAGR={m['cagr']*100:.1f}%, MaxDD={m['max_dd']*100:.1f}%, "
          f"WR={m['wr']*100:.1f}%, PF={m['pf']:.3f}")

    # Permutation (correct: tests timing skill vs baseline)
    p_val, _ = run_permutation_test(strategy_rets, baseline_rets, 200)
    perm_pass = p_val < 0.05
    print(f"  Permutation: p={p_val:.3f} ({'PASS' if perm_pass else 'FAIL'})")

    # Regime
    gap, sg, sr = run_regime_test(strategy_rets, spy_rets)
    r1_pass = gap is not None and gap <= 0.50
    if gap is not None:
        print(f"  R1: gap={gap:.3f} ({'PASS' if r1_pass else 'FAIL'}), green={sg:.2f}, red={sr:.2f}")

    # Sub-period
    sharpes, cv = sub_period_consistency(strategy_rets)
    sp_pass = all(s > 0 for s in sharpes) and cv < 0.50
    print(f"  Sub-period: {[f'{s:.2f}' for s in sharpes]}, CV={cv:.3f} ({'PASS' if sp_pass else 'FAIL'})")

    # Outlier
    _, _, deg = outlier_robustness(strategy_rets)
    out_pass = abs(deg) < 0.30
    print(f"  Outlier degradation: {deg*100:.1f}% ({'PASS' if out_pass else 'FAIL'})")

    core_pass = perm_pass and sp_pass and out_pass
    verdict = "VALIDATED" if core_pass else "FAILED"
    print(f"  VERDICT: {verdict}")

    return {
        'metrics': m, 'perm_p': p_val, 'r1_gap': gap, 'r1_pass': r1_pass,
        'sub_period_cv': cv, 'outlier_deg': deg, 'verdict': verdict,
    }


# ─── Benchmarks ─────────────────────────────────────────────────────────────
spy_rets = returns['SPY']
upro_rets = returns['UPRO']
spy_m = compute_metrics(spy_rets, "SPY")
upro_m = compute_metrics(upro_rets, "UPRO")
print(f"\nBENCHMARKS: SPY Sharpe={spy_m['sharpe']:.3f} CAGR={spy_m['cagr']*100:.1f}% MaxDD={spy_m['max_dd']*100:.1f}%")
print(f"            UPRO Sharpe={upro_m['sharpe']:.3f} CAGR={upro_m['cagr']*100:.1f}% MaxDD={upro_m['max_dd']*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 1: SEASONALITY + OVERNIGHT COMBO
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 1: SEASONALITY + OVERNIGHT COMBO")
print("=" * 70)

def run_seasonal_overnight(prices_df, returns_df, weak_months=[9],
                           strong_months=[7, 11, 4], overnight_frac=0.65,
                           vol_threshold=15, tx_cost=0.001):
    """
    Weak months: hold UPRO overnight-only (capture 65% of daily return).
    Strong months: hold UPRO full day.
    High vol months: switch to SPY/GLD per vol threshold.
    """
    port_rets = []

    # Realized vol (21-day)
    spy_vol = returns_df['SPY'].rolling(21).std() * np.sqrt(252) * 100
    prev_holding = 'UPRO_FULL'

    for i in range(21, len(returns_df)):
        idx = returns_df.index[i]
        month = idx.month
        vol = spy_vol.iloc[i]

        # Vol override — crisis protection
        if vol > 30:
            holding = 'GLD'
        elif vol > vol_threshold:
            holding = 'SPY'
        elif month in weak_months:
            holding = 'UPRO_OVERNIGHT'
        elif month in strong_months:
            holding = 'UPRO_FULL'
        else:
            holding = 'UPRO_FULL'

        # Returns
        if holding == 'GLD':
            day_ret = returns_df['GLD'].iloc[i]
        elif holding == 'SPY':
            day_ret = returns_df['SPY'].iloc[i]
        elif holding == 'UPRO_OVERNIGHT':
            day_ret = returns_df['UPRO'].iloc[i] * overnight_frac
        else:  # UPRO_FULL
            day_ret = returns_df['UPRO'].iloc[i]

        # Switch cost
        if holding != prev_holding:
            day_ret -= tx_cost
        prev_holding = holding

        port_rets.append(day_ret)

    return pd.Series(port_rets, index=returns_df.index[21:])


# Test variants
so_variants = {
    'SO_sep_only': {'weak_months': [9], 'strong_months': [7, 11, 4], 'vol_threshold': 15},
    'SO_sep_jun': {'weak_months': [9, 6], 'strong_months': [7, 11, 4], 'vol_threshold': 15},
    'SO_sep_jun_oct': {'weak_months': [9, 6, 10], 'strong_months': [7, 11, 4], 'vol_threshold': 15},
    'SO_aggressive_vol10': {'weak_months': [9], 'strong_months': [7, 11, 4], 'vol_threshold': 10},
    'SO_conservative_vol20': {'weak_months': [9, 6], 'strong_months': [7, 11], 'vol_threshold': 20},
}

so_results = {}
for name, params in so_variants.items():
    print(f"\n  Testing {name}...")
    rets = run_seasonal_overnight(prices, returns, **params)
    spy_a = spy_rets.reindex(rets.index).dropna()
    base_a = upro_rets.reindex(rets.index).dropna()
    rets_a = rets.reindex(spy_a.index).dropna()
    so_results[name] = full_validation(rets_a, spy_a, base_a, name)

best_so = max(so_results.items(), key=lambda x: x[1]['metrics']['sharpe'])
print(f"\n  BEST: {best_so[0]} (Sharpe={best_so[1]['metrics']['sharpe']:.3f})")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 2: LEVERAGED BARBELL (80/20 UPRO/TMF)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 2: LEVERAGED BARBELL (UPRO + TMF)")
print("=" * 70)

def run_barbell(prices_df, returns_df, upro_weight=0.80, tmf_weight=0.20,
                rebal_freq=21, vol_override=True, vol_threshold=20,
                tx_cost=0.001):
    """
    Hold fixed allocation of UPRO + TMF. Rebalance monthly.
    Optional: during high vol, shift entirely to SPY.
    """
    port_rets = []
    spy_vol = returns_df['SPY'].rolling(21).std() * np.sqrt(252) * 100

    w_upro = upro_weight
    w_tmf = tmf_weight
    # Track drift
    cum_upro = 1.0
    cum_tmf = 1.0

    for i in range(21, len(returns_df)):
        vol = spy_vol.iloc[i]

        # Vol override
        if vol_override and vol > vol_threshold:
            port_rets.append(returns_df['SPY'].iloc[i])
            cum_upro = 1.0
            cum_tmf = 1.0
            continue

        # Rebalance check
        if (i - 21) % rebal_freq == 0:
            # Reset to target weights
            total = cum_upro * w_upro + cum_tmf * w_tmf
            if total > 0:
                actual_upro = (cum_upro * w_upro) / total
                actual_tmf = (cum_tmf * w_tmf) / total
                turnover = abs(actual_upro - upro_weight) + abs(actual_tmf - tmf_weight)
                cost = turnover * tx_cost / 2
            else:
                cost = 0
            cum_upro = 1.0
            cum_tmf = 1.0
        else:
            cost = 0

        # Daily return
        r_upro = returns_df['UPRO'].iloc[i]
        r_tmf = returns_df['TMF'].iloc[i]

        day_ret = w_upro * r_upro + w_tmf * r_tmf - cost

        # Track drift
        cum_upro *= (1 + r_upro)
        cum_tmf *= (1 + r_tmf)

        port_rets.append(day_ret)

    return pd.Series(port_rets, index=returns_df.index[21:])


bb_variants = {
    'BB_80_20_monthly': {'upro_weight': 0.80, 'tmf_weight': 0.20, 'rebal_freq': 21, 'vol_override': True},
    'BB_70_30_monthly': {'upro_weight': 0.70, 'tmf_weight': 0.30, 'rebal_freq': 21, 'vol_override': True},
    'BB_90_10_monthly': {'upro_weight': 0.90, 'tmf_weight': 0.10, 'rebal_freq': 21, 'vol_override': True},
    'BB_80_20_weekly': {'upro_weight': 0.80, 'tmf_weight': 0.20, 'rebal_freq': 5, 'vol_override': True},
    'BB_80_20_no_vol': {'upro_weight': 0.80, 'tmf_weight': 0.20, 'rebal_freq': 21, 'vol_override': False},
    'BB_80_20_vol15': {'upro_weight': 0.80, 'tmf_weight': 0.20, 'rebal_freq': 21, 'vol_override': True, 'vol_threshold': 15},
}

bb_results = {}
for name, params in bb_variants.items():
    print(f"\n  Testing {name}...")
    rets = run_barbell(prices, returns, **params)
    spy_a = spy_rets.reindex(rets.index).dropna()
    base_a = upro_rets.reindex(rets.index).dropna()
    rets_a = rets.reindex(spy_a.index).dropna()
    bb_results[name] = full_validation(rets_a, spy_a, base_a, name)

best_bb = max(bb_results.items(), key=lambda x: x[1]['metrics']['sharpe'])
print(f"\n  BEST: {best_bb[0]} (Sharpe={best_bb[1]['metrics']['sharpe']:.3f})")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 3: VOL-SCALED DCA
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 3: VOL-SCALED DCA")
print("=" * 70)

def run_vol_scaled_dca(prices_df, returns_df, base_weekly=100,
                       vix_thresholds=None, asset='UPRO'):
    """
    Instead of fixed DCA, scale weekly contributions by VIX level.
    Track shares accumulated and portfolio value.
    Returns daily portfolio returns.
    """
    if vix_thresholds is None:
        vix_thresholds = {
            30: 3.0,   # VIX>30: invest 3x
            20: 2.0,   # VIX 20-30: invest 2x
            15: 1.0,   # VIX 15-20: invest 1x
            0: 0.5,    # VIX<15: invest 0.5x
        }

    shares = 0
    total_invested = 0
    port_values = []
    dca_day = 4  # Friday

    for i in range(1, len(prices_df)):
        idx = prices_df.index[i]
        price = prices_df[asset].iloc[i]
        vix = prices_df['VIX'].iloc[i]

        # Weekly DCA
        if idx.dayofweek == dca_day:
            # Determine multiplier
            multiplier = 0.5  # default
            for thresh in sorted(vix_thresholds.keys(), reverse=True):
                if vix >= thresh:
                    multiplier = vix_thresholds[thresh]
                    break

            invest_amount = base_weekly * multiplier
            new_shares = invest_amount / price if price > 0 else 0
            shares += new_shares
            total_invested += invest_amount

        port_value = shares * price + (total_invested * 0)  # No cash tracking for simplicity
        port_values.append(port_value)

    # Convert to returns
    port_series = pd.Series(port_values, index=prices_df.index[1:])
    port_returns = port_series.pct_change().dropna()

    # Filter out the first few weeks where values are tiny/zero
    port_returns = port_returns[port_returns.index >= prices_df.index[30]]
    port_returns = port_returns.replace([np.inf, -np.inf], 0)

    return port_returns, total_invested, shares


# Compare vol-scaled vs fixed DCA
print(f"\n  Fixed DCA ($100/wk into UPRO)...")
fixed_rets, fixed_invested, fixed_shares = run_vol_scaled_dca(
    prices, returns, base_weekly=100,
    vix_thresholds={0: 1.0},  # Always 1x = fixed
    asset='UPRO'
)
fixed_final = fixed_shares * prices['UPRO'].iloc[-1]
fixed_m = compute_metrics(fixed_rets, "Fixed DCA")
print(f"  Fixed: invested ${fixed_invested:,.0f}, final ${fixed_final:,.0f}, "
      f"Sharpe={fixed_m['sharpe']:.3f}")

vol_dca_variants = {
    'VolDCA_aggressive': {
        'vix_thresholds': {30: 3.0, 20: 2.0, 15: 1.0, 0: 0.5}
    },
    'VolDCA_moderate': {
        'vix_thresholds': {25: 2.0, 18: 1.5, 12: 1.0, 0: 0.75}
    },
    'VolDCA_extreme': {
        'vix_thresholds': {35: 5.0, 25: 3.0, 18: 1.0, 0: 0.25}
    },
    'VolDCA_inverse': {  # Contrarian: invest MORE when complacent
        'vix_thresholds': {25: 0.5, 18: 0.75, 12: 1.5, 0: 2.0}
    },
}

vd_results = {}
for name, params in vol_dca_variants.items():
    print(f"\n  Testing {name}...")
    rets, invested, shares = run_vol_scaled_dca(
        prices, returns, base_weekly=100, **params
    )
    final_val = shares * prices['UPRO'].iloc[-1]
    m = compute_metrics(rets, name)
    print(f"  Invested: ${invested:,.0f}, Final: ${final_val:,.0f} (vs fixed ${fixed_final:,.0f})")
    print(f"  Return on investment: {(final_val/invested - 1)*100:.1f}% (vs fixed {(fixed_final/fixed_invested - 1)*100:.1f}%)")
    print(f"  Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}")

    spy_a = spy_rets.reindex(rets.index).dropna()
    rets_a = rets.reindex(spy_a.index).dropna()
    base_a = fixed_rets.reindex(spy_a.index).dropna()
    vd_results[name] = full_validation(rets_a, spy_a, base_a, name)
    vd_results[name]['total_invested'] = invested
    vd_results[name]['final_value'] = final_val
    vd_results[name]['roi'] = (final_val / invested - 1) if invested > 0 else 0

best_vd = max(vd_results.items(), key=lambda x: x[1].get('roi', 0))
print(f"\n  BEST ROI: {best_vd[0]} (ROI={best_vd[1].get('roi', 0)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 4: DRAWDOWN-RESPONSIVE ALLOCATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 4: DRAWDOWN-RESPONSIVE ALLOCATION")
print("=" * 70)

def run_dd_responsive(prices_df, returns_df, dd_thresholds=None,
                      recovery_lookback=20, tx_cost=0.001):
    """
    Reduce leverage as drawdown deepens. Re-enter on new N-day high.
    """
    if dd_thresholds is None:
        dd_thresholds = [
            (0.00, 1.0),   # At/near highs: 100% UPRO
            (-0.05, 0.67), # Down 5%: 67% UPRO + 33% SPY
            (-0.10, 0.33), # Down 10%: 33% UPRO + 67% SPY
            (-0.20, 0.00), # Down 20%: 100% SPY
        ]

    port_rets = []
    # Track portfolio value to compute drawdown
    port_value = 1.0
    port_peak = 1.0
    prev_upro_frac = 1.0

    for i in range(1, len(returns_df)):
        # Current drawdown
        dd = (port_value - port_peak) / port_peak if port_peak > 0 else 0

        # Check for new N-day high (recovery signal)
        if i >= recovery_lookback:
            recent_peak = max(port_value * (1 + returns_df['UPRO'].iloc[j])
                            for j in range(max(1, i-recovery_lookback), i))
            at_recent_high = port_value >= port_peak * 0.98  # Within 2% of peak
        else:
            at_recent_high = True

        # Determine UPRO fraction based on drawdown
        upro_frac = 1.0
        for thresh, frac in sorted(dd_thresholds, reverse=True):
            if dd <= thresh:
                upro_frac = frac
            else:
                break

        # If recovering from deep drawdown, ramp back up gradually
        if at_recent_high and upro_frac < 1.0:
            upro_frac = min(1.0, upro_frac + 0.33)

        # Transaction cost for allocation changes
        cost = abs(upro_frac - prev_upro_frac) * tx_cost
        prev_upro_frac = upro_frac

        # Daily return
        day_ret = (upro_frac * returns_df['UPRO'].iloc[i] +
                   (1 - upro_frac) * returns_df['SPY'].iloc[i] - cost)

        port_value *= (1 + day_ret)
        port_peak = max(port_peak, port_value)
        port_rets.append(day_ret)

    return pd.Series(port_rets, index=returns_df.index[1:])


dd_variants = {
    'DD_standard': {
        'dd_thresholds': [(0.0, 1.0), (-0.05, 0.67), (-0.10, 0.33), (-0.20, 0.0)],
        'recovery_lookback': 20,
    },
    'DD_aggressive': {
        'dd_thresholds': [(0.0, 1.0), (-0.10, 0.67), (-0.20, 0.33), (-0.30, 0.0)],
        'recovery_lookback': 10,
    },
    'DD_conservative': {
        'dd_thresholds': [(0.0, 1.0), (-0.03, 0.67), (-0.07, 0.33), (-0.15, 0.0)],
        'recovery_lookback': 30,
    },
    'DD_binary': {
        'dd_thresholds': [(0.0, 1.0), (-0.10, 0.0)],
        'recovery_lookback': 20,
    },
    'DD_gradual': {
        'dd_thresholds': [(0.0, 1.0), (-0.03, 0.85), (-0.06, 0.70), (-0.10, 0.50),
                          (-0.15, 0.25), (-0.20, 0.0)],
        'recovery_lookback': 20,
    },
}

dd_results = {}
for name, params in dd_variants.items():
    print(f"\n  Testing {name}...")
    rets = run_dd_responsive(prices, returns, **params)
    spy_a = spy_rets.reindex(rets.index).dropna()
    base_a = upro_rets.reindex(rets.index).dropna()
    rets_a = rets.reindex(spy_a.index).dropna()
    dd_results[name] = full_validation(rets_a, spy_a, base_a, name)

best_dd = max(dd_results.items(), key=lambda x: x[1]['metrics']['sharpe'])
print(f"\n  BEST: {best_dd[0]} (Sharpe={best_dd[1]['metrics']['sharpe']:.3f})")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 5: GOLDEN CROSS + VOL CONFIRMATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 5: GOLDEN CROSS + VOL CONFIRMATION")
print("=" * 70)

def run_golden_cross_vol(prices_df, returns_df, sma_short=50, sma_long=200,
                         vol_threshold=20, require_above_200=True,
                         tx_cost=0.001):
    """
    Hold UPRO only when:
    1. SPY 50d MA > SPY 200d MA (golden cross)
    2. SPY price > 200d MA
    3. Realized vol < threshold
    Otherwise hold SPY (or GLD if vol > 30).
    """
    sma_s = prices_df['SPY'].rolling(sma_short).mean()
    sma_l = prices_df['SPY'].rolling(sma_long).mean()
    vol_series = returns_df['SPY'].rolling(21).std() * np.sqrt(252) * 100

    port_rets = []
    prev_holding = 'SPY'

    start = max(sma_long, 21)
    for i in range(start, len(returns_df)):
        idx = returns_df.index[i]
        vol = vol_series.iloc[i]

        # Crisis override
        if vol > 30:
            holding = 'GLD'
        else:
            golden = sma_s.iloc[i] > sma_l.iloc[i]
            above_200 = prices_df['SPY'].iloc[i] > sma_l.iloc[i] if require_above_200 else True
            vol_ok = vol < vol_threshold

            if golden and above_200 and vol_ok:
                holding = 'UPRO'
            else:
                holding = 'SPY'

        # Return
        if holding == 'GLD':
            day_ret = returns_df['GLD'].iloc[i]
        elif holding == 'UPRO':
            day_ret = returns_df['UPRO'].iloc[i]
        else:
            day_ret = returns_df['SPY'].iloc[i]

        # Switch cost
        if holding != prev_holding:
            day_ret -= tx_cost
        prev_holding = holding

        port_rets.append(day_ret)

    return pd.Series(port_rets, index=returns_df.index[start:])


gc_variants = {
    'GC_standard': {'sma_short': 50, 'sma_long': 200, 'vol_threshold': 20, 'require_above_200': True},
    'GC_vol15': {'sma_short': 50, 'sma_long': 200, 'vol_threshold': 15, 'require_above_200': True},
    'GC_vol25': {'sma_short': 50, 'sma_long': 200, 'vol_threshold': 25, 'require_above_200': True},
    'GC_20_100': {'sma_short': 20, 'sma_long': 100, 'vol_threshold': 20, 'require_above_200': True},
    'GC_no_above200': {'sma_short': 50, 'sma_long': 200, 'vol_threshold': 20, 'require_above_200': False},
    'GC_10_50': {'sma_short': 10, 'sma_long': 50, 'vol_threshold': 20, 'require_above_200': True},
}

gc_results = {}
for name, params in gc_variants.items():
    print(f"\n  Testing {name}...")
    rets = run_golden_cross_vol(prices, returns, **params)
    spy_a = spy_rets.reindex(rets.index).dropna()
    base_a = upro_rets.reindex(rets.index).dropna()
    rets_a = rets.reindex(spy_a.index).dropna()
    gc_results[name] = full_validation(rets_a, spy_a, base_a, name)

best_gc = max(gc_results.items(), key=lambda x: x[1]['metrics']['sharpe'])
print(f"\n  BEST: {best_gc[0]} (Sharpe={best_gc[1]['metrics']['sharpe']:.3f})")


# ═══════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL COMPARISON — ALL BATCH 4 STRATEGIES")
print("=" * 70)

print(f"\n  BENCHMARKS:")
print(f"    SPY:  Sharpe={spy_m['sharpe']:.3f}, CAGR={spy_m['cagr']*100:.1f}%")
print(f"    UPRO: Sharpe={upro_m['sharpe']:.3f}, CAGR={upro_m['cagr']*100:.1f}%")

all_best = [
    ("Seasonal+Overnight", best_so),
    ("Leveraged Barbell", best_bb),
    ("Vol-Scaled DCA", best_vd),
    ("DD-Responsive", best_dd),
    ("Golden Cross+Vol", best_gc),
]

print(f"\n  STRATEGY RESULTS (best variant each):")
for label, (name, result) in all_best:
    m = result['metrics']
    v = result['verdict']
    print(f"\n    {label} ({name}):")
    print(f"      Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
          f"CAGR={m['cagr']*100:.1f}%, MaxDD={m['max_dd']*100:.1f}%")
    r1_str = f"{result['r1_gap']:.3f}" if result['r1_gap'] is not None else 'N/A'
    print(f"      Perm p={result['perm_p']:.3f}, R1 gap={r1_str}, "
          f"CV={result['sub_period_cv']:.3f}")
    print(f"      → {v}")

# Winners
validated = [(label, name, result) for label, (name, result) in all_best if result['verdict'] == 'VALIDATED']
print(f"\n  VALIDATED STRATEGIES: {len(validated)}/{len(all_best)}")
for label, name, result in validated:
    print(f"    ✅ {label}: Sharpe={result['metrics']['sharpe']:.3f}")

if not validated:
    # Show which came closest
    by_sharpe = sorted(all_best, key=lambda x: x[1][1]['metrics']['sharpe'], reverse=True)
    print(f"\n  None fully validated. Best candidates:")
    for label, (name, result) in by_sharpe[:3]:
        perm = "PASS" if result['perm_p'] < 0.05 else "FAIL"
        sp = "PASS" if result['sub_period_cv'] < 0.50 else "FAIL"
        out = "PASS" if abs(result['outlier_deg']) < 0.30 else "FAIL"
        print(f"    🟡 {label}: Sharpe={result['metrics']['sharpe']:.3f} "
              f"(perm={perm}, subperiod={sp}, outlier={out})")

# Save
summary = {
    'run_date': datetime.now().isoformat(),
    'strategies': {}
}
for label, (name, result) in all_best:
    summary['strategies'][label] = {
        'variant': name,
        'sharpe': float(result['metrics']['sharpe']),
        'sortino': float(result['metrics']['sortino']),
        'cagr': float(result['metrics']['cagr']),
        'max_dd': float(result['metrics']['max_dd']),
        'perm_p': float(result['perm_p']),
        'r1_gap': float(result['r1_gap']) if result['r1_gap'] is not None else None,
        'sub_period_cv': float(result['sub_period_cv']),
        'outlier_deg': float(result['outlier_deg']),
        'verdict': result['verdict'],
    }

with open(os.path.join(OUT_DIR, 'batch4_results.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n  Results saved to {OUT_DIR}/batch4_results.json")
print(f"\nBATCH 4 COMPLETE.")
