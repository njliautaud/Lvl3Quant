"""
Leverage Optimization Study — Kelly Criterion + Portfolio Scaling Analysis

For the user's question: "If I used 200-300% margin, would it scale linearly?"
And: "Is there anything that breaks the 25% CAGR wall?"

Tests leverage scaling on:
1. V5 CSP + Hedge (equity curve from backtest)
2. IC + Hedge (corrected equity curve)
3. ETF Rotation v3 (equity curve)
4. Combined 60/40 portfolio (V5 + ETF v3)
5. Combined 3-strategy (V5 + IC + ETF v3)

For each: optimal Kelly fraction, leverage-scaled CAGR/MaxDD/Sharpe at 1x-4x,
and the exact point where leverage becomes destructive.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/leverage_optimization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load equity curves ──
def load_equity(path, col="nav"):
    df = pd.read_parquet(path)
    if col in df.columns:
        return df[col].values
    # Try first numeric column
    for c in df.columns:
        if df[c].dtype in (np.float64, np.float32, np.int64):
            return df[c].values
    raise ValueError(f"No numeric column found in {path}")

# Load available equity curves
curves = {}

# V5 Combined Hedge
try:
    v5_path = Path("/home/jupiter/Lvl3Quant/output/v5_combined_hedge/equity_curves.parquet")
    df = pd.read_parquet(v5_path)
    print(f"V5 columns: {list(df.columns)}")
    # Find the hedged equity curve
    for col in ['hedged_nav', 'nav_hedged', 'combined_nav', 'nav', 'equity']:
        if col in df.columns:
            curves['V5_CSP_Hedge'] = df[col].values
            print(f"  Using column: {col}, length: {len(curves['V5_CSP_Hedge'])}")
            break
    if 'V5_CSP_Hedge' not in curves:
        # Use first numeric column
        for col in df.columns:
            if df[col].dtype in (np.float64, np.float32):
                curves['V5_CSP_Hedge'] = df[col].values
                print(f"  Using first numeric column: {col}")
                break
except Exception as e:
    print(f"V5 load error: {e}")

# IC Honest Recalc
try:
    ic_path = Path("/home/jupiter/Lvl3Quant/output/ic_honest_recalc/corrected_equity_curves.parquet")
    df = pd.read_parquet(ic_path)
    print(f"IC columns: {list(df.columns)}")
    for col in ['ic_hedged_corrected', 'hedged_corrected', 'corrected_nav', 'nav', 'hedged']:
        if col in df.columns:
            curves['IC_Hedge'] = df[col].values
            print(f"  Using column: {col}, length: {len(curves['IC_Hedge'])}")
            break
    if 'IC_Hedge' not in curves:
        for col in df.columns:
            if df[col].dtype in (np.float64, np.float32):
                curves['IC_Hedge'] = df[col].values
                print(f"  Using first numeric column: {col}")
                break
except Exception as e:
    print(f"IC load error: {e}")

# Multi-strategy portfolio
try:
    ms_path = Path("/home/jupiter/Lvl3Quant/output/multi_strategy_portfolio/combined_equity.parquet")
    df = pd.read_parquet(ms_path)
    print(f"Multi-strat columns: {list(df.columns)}")
    for col in df.columns:
        if df[col].dtype in (np.float64, np.float32):
            curves['Combined_60_40'] = df[col].values
            print(f"  Using column: {col}, length: {len(curves['Combined_60_40'])}")
            break
except Exception as e:
    print(f"Multi-strat load error: {e}")

# Also check for ETF v3 standalone
try:
    for p in [
        "/home/jupiter/Lvl3Quant/output/rotation_monthly_regime/etf_v3_equity.parquet",
        "/home/jupiter/Lvl3Quant/output/macro_picker/etf_rotation_v3_hedged_equity.parquet",
    ]:
        if Path(p).exists():
            df = pd.read_parquet(p)
            print(f"ETF v3 from {p}: columns {list(df.columns)}")
            for col in df.columns:
                if df[col].dtype in (np.float64, np.float32):
                    curves['ETF_v3'] = df[col].values
                    print(f"  Using column: {col}")
                    break
            break
except Exception as e:
    print(f"ETF v3 load error: {e}")

print(f"\nLoaded {len(curves)} equity curves: {list(curves.keys())}")

# ── Metrics functions ──
def daily_returns(equity):
    """Convert equity curve to daily returns."""
    eq = np.array(equity, dtype=np.float64)
    eq = eq[eq > 0]  # Remove zeros
    returns = np.diff(eq) / eq[:-1]
    return returns[np.isfinite(returns)]

def compute_metrics(equity, leverage=1.0, name=""):
    """Compute all metrics for a leveraged equity curve."""
    rets = daily_returns(equity)
    if len(rets) < 10:
        return None

    # Apply leverage to daily returns
    lev_rets = rets * leverage

    # Rebuild equity curve
    lev_equity = np.cumprod(1 + lev_rets) * 100000

    # CAGR
    years = len(lev_rets) / 252
    if years < 0.1:
        return None
    total_return = lev_equity[-1] / lev_equity[0]
    cagr = (total_return ** (1 / years) - 1) * 100

    # Sharpe
    ann_ret = np.mean(lev_rets) * 252
    ann_vol = np.std(lev_rets) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = lev_rets[lev_rets < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = ann_ret / downside_vol

    # MaxDD
    peak = np.maximum.accumulate(lev_equity)
    dd = (lev_equity - peak) / peak
    maxdd = dd.min() * 100

    # Calmar
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0

    # Win rate (daily)
    wr = (lev_rets > 0).mean() * 100

    # Kelly fraction
    # f* = (p*b - q) / b where p=win prob, b=avg win/avg loss, q=1-p
    wins = lev_rets[lev_rets > 0]
    losses = lev_rets[lev_rets < 0]
    if len(wins) > 0 and len(losses) > 0:
        p = len(wins) / len(lev_rets)
        b = np.mean(wins) / abs(np.mean(losses))
        q = 1 - p
        kelly = (p * b - q) / b
    else:
        kelly = 0

    return {
        'name': name,
        'leverage': leverage,
        'cagr_pct': round(cagr, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'maxdd_pct': round(maxdd, 1),
        'calmar': round(calmar, 2),
        'win_rate_pct': round(wr, 1),
        'kelly_fraction': round(kelly, 3),
        'ann_return_pct': round(ann_ret * 100, 1),
        'ann_vol_pct': round(ann_vol * 100, 1),
        'years': round(years, 1),
        'n_days': len(lev_rets),
    }

# ── Run leverage sweep ──
leverages = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
all_results = []

for name, equity in curves.items():
    print(f"\n{'='*60}")
    print(f"Strategy: {name}")
    print(f"{'='*60}")

    # Base metrics
    base = compute_metrics(equity, leverage=1.0, name=name)
    if base is None:
        print(f"  Insufficient data, skipping")
        continue

    print(f"  Base: CAGR {base['cagr_pct']}%, Sharpe {base['sharpe']}, MaxDD {base['maxdd_pct']}%")
    print(f"  Kelly optimal fraction: {base['kelly_fraction']}")
    print(f"  Ann vol: {base['ann_vol_pct']}%, Win rate: {base['win_rate_pct']}%")

    print(f"\n  {'Leverage':>8} | {'CAGR':>8} | {'Sharpe':>7} | {'Sortino':>8} | {'MaxDD':>8} | {'Calmar':>7}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")

    for lev in leverages:
        m = compute_metrics(equity, leverage=lev, name=name)
        if m is None:
            continue
        m['strategy'] = name
        all_results.append(m)

        marker = " ◄ KELLY" if abs(lev - base['kelly_fraction']) < 0.15 else ""
        marker = " ◄ CURRENT" if lev == 1.0 else marker
        print(f"  {lev:>7.2f}x | {m['cagr_pct']:>7.1f}% | {m['sharpe']:>6.2f} | {m['sortino']:>7.2f} | {m['maxdd_pct']:>7.1f}% | {m['calmar']:>6.2f}{marker}")

# ── Build optimal 3-strategy blend with leverage ──
print(f"\n{'='*60}")
print("OPTIMAL MULTI-STRATEGY LEVERAGE ANALYSIS")
print(f"{'='*60}")

# If we have V5 and IC, build blended portfolios
if 'V5_CSP_Hedge' in curves and 'IC_Hedge' in curves:
    v5_rets = daily_returns(curves['V5_CSP_Hedge'])
    ic_rets = daily_returns(curves['IC_Hedge'])

    # Align lengths
    min_len = min(len(v5_rets), len(ic_rets))
    v5_r = v5_rets[-min_len:]
    ic_r = ic_rets[-min_len:]

    corr = np.corrcoef(v5_r, ic_r)[0, 1]
    print(f"\nV5 vs IC correlation: {corr:.3f}")

    # Sweep blend weights AND leverage
    print(f"\n  {'Blend':>12} | {'Lev':>5} | {'CAGR':>8} | {'Sharpe':>7} | {'MaxDD':>8} | {'Calmar':>7}")
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}")

    best_calmar = {'calmar': -999}
    best_sharpe = {'sharpe': -999}

    for w_v5 in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w_ic = 1.0 - w_v5
        blended = w_v5 * v5_r + w_ic * ic_r
        blended_eq = np.cumprod(1 + blended) * 100000

        for lev in [1.0, 1.5, 2.0, 2.5, 3.0]:
            m = compute_metrics(blended_eq, leverage=lev, name=f"V5({w_v5:.0%})+IC({w_ic:.0%})")
            if m is None:
                continue

            label = f"V5:{w_v5:.0%}/IC:{w_ic:.0%}"
            print(f"  {label:>12} | {lev:>4.1f}x | {m['cagr_pct']:>7.1f}% | {m['sharpe']:>6.2f} | {m['maxdd_pct']:>7.1f}% | {m['calmar']:>6.2f}")

            if m['calmar'] > best_calmar.get('calmar', -999):
                best_calmar = {**m, 'w_v5': w_v5, 'w_ic': w_ic, 'lev': lev}
            if m['sharpe'] > best_sharpe.get('sharpe', -999):
                best_sharpe = {**m, 'w_v5': w_v5, 'w_ic': w_ic, 'lev': lev}

    print(f"\n  Best Calmar: V5:{best_calmar.get('w_v5',0):.0%}/IC:{best_calmar.get('w_ic',0):.0%} @ {best_calmar.get('lev',1)}x → CAGR {best_calmar['cagr_pct']}%, MaxDD {best_calmar['maxdd_pct']}%, Calmar {best_calmar['calmar']}")
    print(f"  Best Sharpe: V5:{best_sharpe.get('w_v5',0):.0%}/IC:{best_sharpe.get('w_ic',0):.0%} @ {best_sharpe.get('lev',1)}x → CAGR {best_sharpe['cagr_pct']}%, Sharpe {best_sharpe['sharpe']}")

# ── Save results ──
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / "leverage_sweep_results.csv", index=False)

summary = {
    'strategies_tested': list(curves.keys()),
    'leverage_range': leverages,
    'results': all_results,
}

with open(OUTPUT_DIR / "leverage_optimization_results.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}")
print("\nDONE.")
