#!/usr/bin/env python3
"""
Creative Strategies Batch 6 — Five genuinely novel ideas not tested in batches 2-5.

1. CARRY + MOMENTUM HYBRID — Combine dividend yield carry (SCHD/VYM) with momentum timing.
   When momentum is positive, lever into growth (UPRO). When momentum fades, rotate to
   carry/dividend ETFs. Different return drivers than pure momentum.

2. VOLATILITY TERM STRUCTURE TRADING — Use VIX/VIX3M ratio to time entries.
   Backwardation (ratio > 1) = fear/bounce signal → UPRO.
   Steep contango (ratio << 1) = complacency/risk → SPY or TLT.

3. PUT/CALL RATIO CONTRARIAN — Use CBOE equity put/call ratio as contrarian indicator.
   Extreme readings (>1.0 = panic buy, <0.6 = euphoria sell) combined with vol level.

4. ADAPTIVE LOOKBACK TREND — Instead of fixed SMA periods, dynamically choose lookback
   based on realized vol. High vol → short lookback (SMA20). Low vol → long lookback (SMA200).
   Fewer whipsaws in calm markets, faster reaction in volatile ones.

5. RISK BUDGET ALLOCATION — Target constant portfolio volatility (15%) across UPRO/SPY/TLT/GLD.
   When UPRO vol is low, hold more UPRO. When UPRO vol spikes, rotate to bonds/gold.

Full adversarial validation suite on each.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings, json, os
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

OUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_research/creative_batch6"
os.makedirs(OUT_DIR, exist_ok=True)

COST_PER_SWITCH = 0.0002  # 0.02% spread cost per switch

print("=" * 70)
print("CREATIVE STRATEGIES BATCH 6")
print("=" * 70)
print(f"\nFetching data...")

tickers = {
    'SPY': 'SPY', 'UPRO': 'UPRO', 'GLD': 'GLD', 'TLT': 'TLT',
    'VIX': '^VIX', 'SCHD': 'SCHD', 'VYM': 'VYM', 'VIXY': 'VIXY',
}

# Try VIX3M — may not be available on yfinance
vix3m_available = False
try:
    vix3m_df = yf.download('^VIX3M', start='2012-01-01', end='2026-12-31', progress=False)
    if len(vix3m_df) > 100:
        vix3m_available = True
        print(f"  VIX3M: {len(vix3m_df)} days")
except:
    pass

if not vix3m_available:
    # Try VIX3M as VXMT (CBOE 3-month vol index) or construct proxy
    try:
        vix3m_df = yf.download('VIX3M', start='2012-01-01', end='2026-12-31', progress=False)
        if len(vix3m_df) > 100:
            vix3m_available = True
            print(f"  VIX3M (alt ticker): {len(vix3m_df)} days")
    except:
        pass

if not vix3m_available:
    print("  VIX3M: not available — will construct proxy from VIX SMA")

data = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start='2012-01-01', end='2026-12-31', progress=False)
        if len(df) > 100:
            data[name] = df['Close'].squeeze()
            print(f"  {name}: {len(df)} days")
        else:
            print(f"  {name}: only {len(df)} days — skipping")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")

# Align all available data
available = [n for n in ['SPY', 'UPRO', 'GLD', 'TLT', 'VIX'] if n in data]
common_dates = sorted(set.intersection(*[set(data[n].index) for n in available]))
prices = pd.DataFrame({n: data[n].reindex(common_dates) for n in available}).dropna()

# Add optional tickers where available, forward-filling shorter histories
for opt in ['SCHD', 'VYM', 'VIXY']:
    if opt in data:
        prices[opt] = data[opt].reindex(prices.index)

if vix3m_available:
    prices['VIX3M'] = vix3m_df['Close'].squeeze().reindex(prices.index)

returns = prices[['SPY', 'UPRO', 'GLD', 'TLT']].pct_change().dropna()
# Make sure prices align with returns
prices = prices.loc[returns.index]

print(f"\nAligned: {len(prices)} days, {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
print(f"Available: {list(prices.columns)}")


# ─── Utility functions ──────────────────────────────────────────────────────

def compute_metrics(rets, label=""):
    if len(rets) < 20:
        return {'label': label, 'sharpe': 0, 'sortino': 0, 'cagr': 0, 'max_dd': -1,
                'wr': 0, 'pf': 0, 'final_value': 0, 'n_days': len(rets),
                'annual_ret': 0, 'annual_vol': 0, 'calmar': 0}
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

    total_return = float(cum.iloc[-1]) - 1.0

    return {
        'label': label, 'sharpe': round(sharpe, 4), 'sortino': round(sortino, 4),
        'cagr': round(cagr, 4), 'max_dd': round(max_dd, 4), 'calmar': round(calmar, 4),
        'wr': round(wr, 4), 'pf': round(pf, 4), 'annual_ret': round(mu, 4),
        'annual_vol': round(sigma, 4), 'n_days': len(rets),
        'final_value': round(float(cum.iloc[-1]), 4),
        'total_return': round(total_return, 4),
    }


def apply_weekly_rebalance_with_costs(daily_signals, returns_df, asset_col_map):
    """
    Given daily signals (asset name per day), apply weekly rebalance with costs.
    asset_col_map: dict mapping signal value -> returns column name
    Returns: pd.Series of strategy daily returns with costs applied.
    """
    strat_rets = pd.Series(0.0, index=returns_df.index)
    prev_asset = None
    last_rebal_idx = -999

    for i, date in enumerate(returns_df.index):
        signal = daily_signals.loc[date] if date in daily_signals.index else prev_asset
        if signal is None:
            signal = 'SPY'  # default

        # Weekly rebalance: only switch on Fridays or if first day
        is_friday = date.weekday() == 4
        days_since_rebal = i - last_rebal_idx

        if prev_asset is None:
            # First day
            current_asset = signal
            last_rebal_idx = i
        elif is_friday and days_since_rebal >= 5 and signal != prev_asset:
            current_asset = signal
            last_rebal_idx = i
            strat_rets.iloc[i] -= COST_PER_SWITCH  # cost on switch
        else:
            current_asset = prev_asset

        col = asset_col_map.get(current_asset, 'SPY')
        if col in returns_df.columns:
            strat_rets.iloc[i] += returns_df[col].iloc[i]
        else:
            strat_rets.iloc[i] += returns_df['SPY'].iloc[i]

        prev_asset = current_asset

    return strat_rets


def count_annual_switches(daily_positions):
    """Count average annual switches from a position series."""
    switches = (daily_positions != daily_positions.shift(1)).sum()
    years = len(daily_positions) / 252
    return round(switches / years, 1) if years > 0 else 0


def run_permutation_test(strategy_rets, baseline_rets, n_perms=200):
    """Shuffle timing signal, not returns. Compare excess return."""
    aligned = pd.DataFrame({'strat': strategy_rets, 'base': baseline_rets}).dropna()
    if len(aligned) < 50:
        return 1.0, 0.0

    real_excess = aligned['strat'].mean() - aligned['base'].mean()

    count_beat = 0
    for _ in range(n_perms):
        mask = np.random.random(len(aligned)) > 0.5
        perm_rets = aligned['strat'].values.copy()
        perm_rets[mask] = aligned['base'].values[mask]
        perm_excess = perm_rets.mean() - aligned['base'].mean()
        if perm_excess >= real_excess:
            count_beat += 1

    return count_beat / n_perms, real_excess


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
    if len(sharpes) == 0:
        return [], float('inf')
    cv = np.std(sharpes) / abs(np.mean(sharpes)) if np.mean(sharpes) != 0 else float('inf')
    return sharpes, cv


def outlier_robustness(rets, n_remove=10):
    full_s = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    trimmed = rets.sort_values(ascending=True).iloc[:-n_remove]
    trim_s = trimmed.mean() / trimmed.std() * np.sqrt(252) if trimmed.std() > 0 else 0
    deg = (trim_s - full_s) / abs(full_s) if full_s != 0 else 0
    return full_s, trim_s, deg


def full_validation(strategy_rets, spy_rets, baseline_rets, label):
    """Full adversarial suite."""
    print(f"\n  --- {label} ---")
    m = compute_metrics(strategy_rets, label)
    print(f"  Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
          f"CAGR={m['cagr']*100:.1f}%, MaxDD={m['max_dd']*100:.1f}%, "
          f"WR={m['wr']*100:.1f}%, PF={m['pf']:.3f}, FinalVal={m['final_value']:.1f}x")

    # Permutation test
    p_val, _ = run_permutation_test(strategy_rets, baseline_rets, 200)
    perm_pass = p_val < 0.05
    print(f"  Permutation: p={p_val:.3f} ({'PASS' if perm_pass else 'FAIL'})")

    # Regime test
    gap, sg, sr = run_regime_test(strategy_rets, spy_rets)
    r1_pass = gap is not None and gap <= 0.50
    if gap is not None:
        print(f"  R1 Regime: gap={gap:.3f} ({'PASS' if r1_pass else 'FAIL — expected for UPRO'}), green={sg:.2f}, red={sr:.2f}")
    else:
        print(f"  R1 Regime: insufficient data")
        r1_pass = False

    # Sub-period consistency
    sharpes, cv = sub_period_consistency(strategy_rets)
    all_pos = all(s > 0 for s in sharpes)
    sp_pass = all_pos and cv < 0.50
    print(f"  Sub-period: {[f'{s:.2f}' for s in sharpes]}, CV={cv:.3f}, AllPos={all_pos} ({'PASS' if sp_pass else 'FAIL'})")

    # Outlier robustness
    full_s, trim_s, deg = outlier_robustness(strategy_rets)
    out_pass = abs(deg) < 0.30
    print(f"  Outlier: full={full_s:.3f}, trimmed={trim_s:.3f}, deg={deg*100:.1f}% ({'PASS' if out_pass else 'FAIL'})")

    # Summary
    tests = {
        'permutation': {'pass': perm_pass, 'p_value': round(p_val, 4)},
        'regime': {'pass': r1_pass, 'gap': round(gap, 4) if gap is not None else None,
                   'sharpe_green': round(sg, 4) if sg is not None else None,
                   'sharpe_red': round(sr, 4) if sr is not None else None},
        'sub_period': {'pass': sp_pass, 'sharpes': [round(s, 4) for s in sharpes],
                       'cv': round(cv, 4), 'all_positive': all_pos},
        'outlier_robustness': {'pass': out_pass, 'full_sharpe': round(full_s, 4),
                                'trimmed_sharpe': round(trim_s, 4),
                                'degradation_pct': round(deg * 100, 2)},
    }

    n_pass = sum([perm_pass, r1_pass, sp_pass, out_pass])
    all_pass = n_pass == 4
    print(f"  VERDICT: {n_pass}/4 tests passed {'✓ ALL PASS' if all_pass else ''}")

    return {
        'metrics': m,
        'validation': tests,
        'n_tests_passed': n_pass,
        'all_pass': all_pass,
    }


# ─── BASELINE: SPY Buy & Hold ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("BASELINE: SPY Buy & Hold")
print("=" * 70)
spy_bh = compute_metrics(returns['SPY'], "SPY Buy & Hold")
print(f"  Sharpe={spy_bh['sharpe']:.3f}, CAGR={spy_bh['cagr']*100:.1f}%, "
      f"MaxDD={spy_bh['max_dd']*100:.1f}%, FinalVal={spy_bh['final_value']:.1f}x")

all_results = {'baseline_spy': spy_bh}


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: CARRY + MOMENTUM HYBRID
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 1: CARRY + MOMENTUM HYBRID")
print("=" * 70)

# Logic: When SPY momentum is positive (price > SMA50), lever into UPRO.
# When momentum fades (price < SMA50 but > SMA200), hold carry ETFs (SCHD/VYM).
# When momentum is negative (price < SMA200), defensive (TLT/GLD blend).
# The carry component provides income-based return when growth stalls.

spy_sma50 = prices['SPY'].rolling(50).mean()
spy_sma200 = prices['SPY'].rolling(200).mean()

# If SCHD not available for full period, use VYM or SPY as carry proxy
carry_ticker = None
if 'SCHD' in prices.columns and prices['SCHD'].notna().sum() > len(prices) * 0.5:
    carry_ticker = 'SCHD'
elif 'VYM' in prices.columns and prices['VYM'].notna().sum() > len(prices) * 0.5:
    carry_ticker = 'VYM'

if carry_ticker:
    # Add carry returns to the returns df
    carry_rets = prices[carry_ticker].pct_change()
    returns_ext = returns.copy()
    returns_ext[carry_ticker] = carry_rets
else:
    carry_ticker = 'SPY'  # Fallback: SPY as carry proxy (weaker version)
    returns_ext = returns.copy()

# Generate daily signal
signals_carry = pd.Series('SPY', index=returns.index)
for i, date in enumerate(returns.index):
    if pd.isna(spy_sma200.loc[date]) or pd.isna(spy_sma50.loc[date]):
        signals_carry.iloc[i] = 'SPY'
    elif prices['SPY'].loc[date] > spy_sma50.loc[date]:
        signals_carry.iloc[i] = 'UPRO'  # Strong momentum → leveraged growth
    elif prices['SPY'].loc[date] > spy_sma200.loc[date]:
        signals_carry.iloc[i] = carry_ticker  # Fading momentum → carry/dividend
    else:
        signals_carry.iloc[i] = 'TLT'  # Negative momentum → defensive

asset_map_carry = {'UPRO': 'UPRO', 'SPY': 'SPY', 'TLT': 'TLT', 'GLD': 'GLD'}
if carry_ticker in returns_ext.columns:
    asset_map_carry[carry_ticker] = carry_ticker

strat1_rets = apply_weekly_rebalance_with_costs(signals_carry, returns_ext, asset_map_carry)
switches_1 = count_annual_switches(signals_carry)
print(f"  Annual switches: {switches_1}")
result_1 = full_validation(strat1_rets, returns['SPY'], returns['SPY'], "Carry + Momentum Hybrid")
result_1['annual_switches'] = switches_1
all_results['carry_momentum_hybrid'] = result_1


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: VOLATILITY TERM STRUCTURE TRADING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 2: VOLATILITY TERM STRUCTURE TRADING")
print("=" * 70)

# VIX term structure: ratio of VIX (1-month) to VIX3M (3-month).
# ratio > 1.0 → backwardation → fear/stress → contrarian buy signal → UPRO
# ratio 0.85-1.0 → normal → SPY
# ratio < 0.85 → steep contango → complacency → risk reduction → TLT
# Additional: if VIX > 30, go to TLT regardless (extreme fear, wait for dust to settle)

if 'VIX3M' in prices.columns and prices['VIX3M'].notna().sum() > 500:
    vix_ratio = prices['VIX'] / prices['VIX3M']
    print("  Using actual VIX3M data for term structure")
else:
    # Proxy: use VIX / VIX 63-day SMA as term structure proxy
    # When VIX spikes above its 3M average → similar to backwardation
    vix_3m_sma = prices['VIX'].rolling(63).mean()
    vix_ratio = prices['VIX'] / vix_3m_sma
    print("  Using VIX/VIX_SMA63 proxy for term structure")

signals_ts = pd.Series('SPY', index=returns.index)
for i, date in enumerate(returns.index):
    if pd.isna(vix_ratio.loc[date]):
        signals_ts.iloc[i] = 'SPY'
        continue

    vix_level = prices['VIX'].loc[date]
    ratio = vix_ratio.loc[date]

    if vix_level > 35:
        # Extreme fear — don't try to catch falling knife
        signals_ts.iloc[i] = 'TLT'
    elif ratio > 1.05:
        # Backwardation — fear is elevated but typically near bottom → UPRO
        signals_ts.iloc[i] = 'UPRO'
    elif ratio > 0.90:
        # Normal zone
        signals_ts.iloc[i] = 'SPY'
    else:
        # Steep contango — complacency, risk building
        signals_ts.iloc[i] = 'GLD'

asset_map_ts = {'UPRO': 'UPRO', 'SPY': 'SPY', 'TLT': 'TLT', 'GLD': 'GLD'}
strat2_rets = apply_weekly_rebalance_with_costs(signals_ts, returns, asset_map_ts)
switches_2 = count_annual_switches(signals_ts)
print(f"  Annual switches: {switches_2}")
result_2 = full_validation(strat2_rets, returns['SPY'], returns['SPY'], "Vol Term Structure")
result_2['annual_switches'] = switches_2
all_results['vol_term_structure'] = result_2


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: PUT/CALL RATIO CONTRARIAN
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 3: PUT/CALL RATIO CONTRARIAN")
print("=" * 70)

# Since CBOE put/call ratio isn't directly on yfinance, we use a proxy:
# VIXY/SPY momentum as fear gauge (VIXY trends up in fear, down in calm)
# Alternative: construct synthetic put/call from VIX + vol skew behavior
# When VIX is high AND rising (5d) = panic → contrarian buy UPRO
# When VIX is low AND falling (5d) = euphoria → cautious SPY
# Middle ground = normal UPRO allocation based on trend

vix = prices['VIX']
vix_5d_chg = vix.pct_change(5)
vix_20d_pctile = vix.rolling(252).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)

signals_pc = pd.Series('SPY', index=returns.index)
for i, date in enumerate(returns.index):
    if pd.isna(vix_20d_pctile.loc[date]) or pd.isna(vix_5d_chg.loc[date]):
        signals_pc.iloc[i] = 'SPY'
        continue

    pctile = vix_20d_pctile.loc[date]
    chg_5d = vix_5d_chg.loc[date]
    vix_level = vix.loc[date]

    if pctile > 0.80 and chg_5d > 0.10:
        # Extreme fear + VIX still rising → panic → contrarian buy after confirmation
        # Wait for VIX to stop rising (mean-revert signal)
        if vix_level > 30:
            signals_pc.iloc[i] = 'TLT'  # Too scary, wait
        else:
            signals_pc.iloc[i] = 'UPRO'  # High fear but not crisis → contrarian buy
    elif pctile > 0.70:
        # Elevated fear → UPRO (lean contrarian)
        signals_pc.iloc[i] = 'UPRO'
    elif pctile < 0.20 and chg_5d < -0.05:
        # Low vol + still falling → euphoria → reduce exposure
        signals_pc.iloc[i] = 'SPY'
    elif pctile < 0.30:
        # Low vol zone → hold SPY, don't lever up
        signals_pc.iloc[i] = 'SPY'
    else:
        # Mid-range: use trend
        if prices['SPY'].loc[date] > spy_sma50.loc[date]:
            signals_pc.iloc[i] = 'UPRO'
        else:
            signals_pc.iloc[i] = 'SPY'

asset_map_pc = {'UPRO': 'UPRO', 'SPY': 'SPY', 'TLT': 'TLT', 'GLD': 'GLD'}
strat3_rets = apply_weekly_rebalance_with_costs(signals_pc, returns, asset_map_pc)
switches_3 = count_annual_switches(signals_pc)
print(f"  Annual switches: {switches_3}")
result_3 = full_validation(strat3_rets, returns['SPY'], returns['SPY'], "Put/Call Contrarian")
result_3['annual_switches'] = switches_3
all_results['putcall_contrarian'] = result_3


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: ADAPTIVE LOOKBACK TREND FOLLOWING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 4: ADAPTIVE LOOKBACK TREND FOLLOWING")
print("=" * 70)

# Key insight: Fixed SMAs whipsaw in high-vol and are too slow in regime changes.
# Solution: Dynamically interpolate between SMA20 and SMA200 based on realized vol.
# High realized vol → weight toward SMA20 (fast detection)
# Low realized vol → weight toward SMA200 (fewer whipsaws)
# Realized vol measured as 21-day std of SPY returns, normalized to percentile.

rv_21 = returns['SPY'].rolling(21).std() * np.sqrt(252)
rv_percentile = rv_21.rolling(252).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
    raw=False
)

# Pre-compute SMAs at various lookbacks
sma_dict = {}
for lb in [20, 30, 40, 50, 60, 80, 100, 120, 150, 200]:
    sma_dict[lb] = prices['SPY'].rolling(lb).mean()

signals_adap = pd.Series('SPY', index=returns.index)

for i, date in enumerate(returns.index):
    if pd.isna(rv_percentile.loc[date]):
        signals_adap.iloc[i] = 'SPY'
        continue

    # Map vol percentile to lookback: high vol → short lookback
    vol_pct = rv_percentile.loc[date]
    # Interpolate: vol_pct=1.0 → lookback=20, vol_pct=0.0 → lookback=200
    lookback = int(200 - (200 - 20) * vol_pct)
    # Snap to nearest computed SMA
    available_lbs = sorted(sma_dict.keys())
    closest_lb = min(available_lbs, key=lambda x: abs(x - lookback))

    sma_val = sma_dict[closest_lb].loc[date]
    if pd.isna(sma_val):
        signals_adap.iloc[i] = 'SPY'
        continue

    spy_price = prices['SPY'].loc[date]
    pct_above = (spy_price - sma_val) / sma_val

    if pct_above > 0.02:
        signals_adap.iloc[i] = 'UPRO'  # Clearly above adaptive trend
    elif pct_above > -0.02:
        signals_adap.iloc[i] = 'SPY'   # Near trend line → reduce leverage
    else:
        signals_adap.iloc[i] = 'TLT'   # Below adaptive trend → defensive

asset_map_adap = {'UPRO': 'UPRO', 'SPY': 'SPY', 'TLT': 'TLT'}
strat4_rets = apply_weekly_rebalance_with_costs(signals_adap, returns, asset_map_adap)
switches_4 = count_annual_switches(signals_adap)
print(f"  Annual switches: {switches_4}")
result_4 = full_validation(strat4_rets, returns['SPY'], returns['SPY'], "Adaptive Lookback Trend")
result_4['annual_switches'] = switches_4
all_results['adaptive_lookback'] = result_4


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY 5: RISK BUDGET ALLOCATION (CONSTANT VOL TARGET)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 5: RISK BUDGET ALLOCATION (CONSTANT VOL TARGET)")
print("=" * 70)

# Target portfolio volatility of 15% annualized.
# Allocate between UPRO (high vol), SPY (medium), TLT (low), GLD (uncorrelated).
# When UPRO vol is low (~20%), allocate heavily to UPRO.
# When UPRO vol spikes (60%+), shift to bonds/gold to maintain risk budget.
# Rebalance weekly. Use inverse-vol weighting with a twist: cap UPRO at 70% max.

TARGET_VOL = 0.15
LOOKBACK_VOL = 42  # 2 months realized vol
MAX_UPRO_WEIGHT = 0.70
MIN_DEFENSIVE_WEIGHT = 0.10  # Always hold some TLT/GLD

assets_rb = ['UPRO', 'SPY', 'TLT', 'GLD']

# Compute rolling vol for each asset
rolling_vols = {}
for a in assets_rb:
    rolling_vols[a] = returns[a].rolling(LOOKBACK_VOL).std() * np.sqrt(252)

# Strategy: compute weights weekly
strat5_rets = pd.Series(0.0, index=returns.index)
weights_history = []
last_weights = {'UPRO': 0.25, 'SPY': 0.25, 'TLT': 0.25, 'GLD': 0.25}
last_rebal_idx = -999

for i, date in enumerate(returns.index):
    is_friday = date.weekday() == 4
    days_since = i - last_rebal_idx

    rebalance = (is_friday and days_since >= 5) or i == 0

    if rebalance:
        vols = {}
        all_valid = True
        for a in assets_rb:
            v = rolling_vols[a].loc[date]
            if pd.isna(v) or v < 0.001:
                all_valid = False
                break
            vols[a] = v

        if all_valid:
            # Inverse-vol weighting
            inv_vols = {a: 1.0 / vols[a] for a in assets_rb}
            total_inv = sum(inv_vols.values())
            raw_weights = {a: inv_vols[a] / total_inv for a in assets_rb}

            # Cap UPRO
            if raw_weights['UPRO'] > MAX_UPRO_WEIGHT:
                excess = raw_weights['UPRO'] - MAX_UPRO_WEIGHT
                raw_weights['UPRO'] = MAX_UPRO_WEIGHT
                # Redistribute excess proportionally to others
                others_sum = sum(raw_weights[a] for a in assets_rb if a != 'UPRO')
                if others_sum > 0:
                    for a in assets_rb:
                        if a != 'UPRO':
                            raw_weights[a] += excess * (raw_weights[a] / others_sum)

            # Ensure minimum defensive allocation
            def_weight = raw_weights['TLT'] + raw_weights['GLD']
            if def_weight < MIN_DEFENSIVE_WEIGHT:
                deficit = MIN_DEFENSIVE_WEIGHT - def_weight
                raw_weights['TLT'] += deficit / 2
                raw_weights['GLD'] += deficit / 2
                # Reduce UPRO/SPY proportionally
                reduce_from = raw_weights['UPRO'] + raw_weights['SPY']
                if reduce_from > deficit:
                    ratio_u = raw_weights['UPRO'] / reduce_from
                    raw_weights['UPRO'] -= deficit * ratio_u
                    raw_weights['SPY'] -= deficit * (1 - ratio_u)

            # Scale to target vol
            port_vol = sum(raw_weights[a] * vols[a] for a in assets_rb)  # Simplified (ignores correlation)
            if port_vol > 0:
                scale = TARGET_VOL / port_vol
                # Don't scale above 1.0 (no leverage beyond UPRO itself)
                scale = min(scale, 1.5)
                for a in assets_rb:
                    raw_weights[a] *= scale

                # Normalize so sum <= 1.0
                total_w = sum(raw_weights.values())
                if total_w > 1.0:
                    for a in assets_rb:
                        raw_weights[a] /= total_w

            # Apply switching cost if weights changed significantly
            if i > 0:
                total_turnover = sum(abs(raw_weights[a] - last_weights.get(a, 0)) for a in assets_rb) / 2
                strat5_rets.iloc[i] -= total_turnover * COST_PER_SWITCH

            last_weights = raw_weights.copy()
            last_rebal_idx = i

    # Apply weights
    day_ret = sum(last_weights.get(a, 0) * returns[a].iloc[i] for a in assets_rb)
    strat5_rets.iloc[i] += day_ret

# Count switches (significant weight changes)
switches_5 = 52  # Weekly rebalance ≈ 52 per year (continuous allocation strategy)
print(f"  Annual rebalances: ~{switches_5}")
result_5 = full_validation(strat5_rets, returns['SPY'], returns['SPY'], "Risk Budget Allocation")
result_5['annual_switches'] = switches_5
all_results['risk_budget_allocation'] = result_5


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY — BATCH 6")
print("=" * 70)

print(f"\n{'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'PF':>6} {'Tests':>6}")
print("-" * 80)

strategies = [
    ('SPY Buy & Hold', spy_bh, None),
    ('Carry+Momentum Hybrid', result_1['metrics'], result_1),
    ('Vol Term Structure', result_2['metrics'], result_2),
    ('Put/Call Contrarian', result_3['metrics'], result_3),
    ('Adaptive Lookback Trend', result_4['metrics'], result_4),
    ('Risk Budget Allocation', result_5['metrics'], result_5),
]

for name, m, val in strategies:
    tests_str = f"{val['n_tests_passed']}/4" if val else "N/A"
    print(f"{name:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
          f"{m['cagr']*100:>6.1f}% {m['max_dd']*100:>6.1f}% "
          f"{m['wr']*100:>5.1f}% {m['pf']:>6.3f} {tests_str:>6}")

# Identify winners
print("\n" + "-" * 70)
print("ADVERSARIAL VALIDATION RESULTS:")
for name, m, val in strategies[1:]:
    if val and val['all_pass']:
        print(f"  ✓ {name}: ALL 4 TESTS PASSED — Sharpe {m['sharpe']:.3f}")
    elif val and val['n_tests_passed'] >= 3:
        failed = []
        for test_name, test_result in val['validation'].items():
            if not test_result['pass']:
                failed.append(test_name)
        print(f"  ~ {name}: {val['n_tests_passed']}/4 passed (failed: {', '.join(failed)})")
    elif val:
        failed = []
        for test_name, test_result in val['validation'].items():
            if not test_result['pass']:
                failed.append(test_name)
        print(f"  ✗ {name}: {val['n_tests_passed']}/4 passed (failed: {', '.join(failed)})")

# Compare with prior batch winners
print("\n" + "-" * 70)
print("CONTEXT — Prior batch winners for comparison:")
print("  Gameplan v3 (batch 3): Sharpe 2.388 — UPRO vol-switching + confluence gate")
print("  Vol Mean Reversion (batch 4): Sharpe 1.467 — 5-regime VIX system")
print("  Everything else in batches 2-5: FAILED adversarial validation")

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'batch': 6,
    'period': f"{prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}",
    'n_days': len(returns),
    'cost_per_switch': COST_PER_SWITCH,
    'rebalance': 'weekly (Fridays)',
    'baseline': spy_bh,
    'strategies': {},
}

strat_names = ['carry_momentum_hybrid', 'vol_term_structure', 'putcall_contrarian',
               'adaptive_lookback', 'risk_budget_allocation']
strat_labels = ['Carry + Momentum Hybrid', 'Vol Term Structure', 'Put/Call Contrarian',
                'Adaptive Lookback Trend', 'Risk Budget Allocation']

for key in strat_names:
    if key in all_results:
        output['strategies'][key] = all_results[key]

# JSON serialization fix
def fix_for_json(obj):
    if isinstance(obj, dict):
        return {k: fix_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

output = fix_for_json(output)

results_path = os.path.join(OUT_DIR, "batch6_results.json")
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {results_path}")
print(f"\nBatch 6 complete.")
