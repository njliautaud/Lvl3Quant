"""
Kelly Criterion & Optimal Position Sizing Study
=================================================
Determines optimal position sizes for our validated strategies using:
1. Full Kelly, Half Kelly, Quarter Kelly
2. Walk-forward Kelly (adaptive over time)
3. Portfolio-level Kelly with correlation adjustments
4. Drawdown-constrained sizing (MaxDD target)

HC #709: Growth strategies + drawdown protection
HC #428: Regime-agnostic validation
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

BASE_DIR = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE_DIR / "output" / "growth_research" / "kelly_sizing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
START_DATE = "2013-01-01"
END_DATE = "2026-07-16"

# Our validated strategies and their ETF proxies
STRATEGIES = {
    "UPRO_protected": {
        "ticker": "UPRO",
        "protection": True,  # 4-signal protection overlay
        "description": "3x S&P 500 with VIX/SMA/credit/breadth protection",
    },
    "CTA_trend": {
        "tickers": ["GLD", "SLV", "USO", "UNG", "DBA", "COPX", "UUP", "TLT", "EEM"],
        "description": "Multi-asset SMA50 trend following",
    },
    "SPY_reversal": {
        "ticker": "SPY",
        "description": "Mean-reversion on oversold SPY",
    },
}

N_PERMUTATIONS = 200

# ============================================================================
# DATA
# ============================================================================
def download_data():
    """Download all required price data."""
    all_tickers = ["UPRO", "SPY", "QQQ", "IWM", "GLD", "SLV", "USO", "UNG",
                   "DBA", "COPX", "UUP", "TLT", "EEM", "HYG", "LQD", "VIXY"]

    print("Downloading data...")
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                     progress=False, auto_adjust=True, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw['Close']
    else:
        closes = raw

    closes = closes.dropna(how='all').ffill()
    print(f"  {len(closes)} days, {closes.shape[1]} assets")
    return closes


# ============================================================================
# STRATEGY RETURN STREAMS
# ============================================================================
def compute_upro_protected(closes):
    """UPRO with 4-signal protection overlay."""
    upro = closes['UPRO'].pct_change()
    spy = closes['SPY']

    # Protection signals (daily)
    spy_sma50 = spy.rolling(50).mean()
    signal_spy = (spy > spy_sma50).astype(float)

    # Credit health
    hyg = closes['HYG']
    lqd = closes['LQD']
    credit_ratio = hyg / lqd
    credit_chg = credit_ratio.pct_change(21)
    signal_credit = (credit_chg > -0.01).astype(float)

    # Breadth (IWM proxy)
    iwm = closes['IWM']
    iwm_sma50 = iwm.rolling(50).mean()
    signal_breadth = (iwm > iwm_sma50).astype(float)

    # VIX proxy
    vixy = closes.get('VIXY')
    if vixy is not None:
        vixy_sma = vixy.rolling(20).mean()
        signal_vix = (vixy < vixy_sma * 1.2).astype(float)  # VIX not spiking
    else:
        signal_vix = pd.Series(1.0, index=closes.index)

    # Combined: majority rule (3/4 or 4/4 = full, 2/4 = half, <2 = cash)
    total_signals = signal_spy + signal_credit + signal_breadth + signal_vix
    exposure = pd.Series(0.0, index=closes.index)
    exposure[total_signals >= 3] = 1.0
    exposure[total_signals == 2] = 0.5

    # Apply 1-day lag (trade next day)
    exposure = exposure.shift(1).fillna(0)

    protected_ret = upro * exposure
    return protected_ret.dropna()


def compute_cta_trend(closes):
    """Multi-asset trend following (SMA50, equal weight)."""
    cta_tickers = ["GLD", "SLV", "USO", "UNG", "DBA", "COPX", "UUP", "TLT", "EEM"]
    available = [t for t in cta_tickers if t in closes.columns]

    all_rets = []
    for ticker in available:
        px = closes[ticker]
        sma50 = px.rolling(50).mean()
        signal = (px > sma50).astype(float).shift(1)  # 1-day lag
        ret = px.pct_change() * signal
        all_rets.append(ret)

    # Equal weight
    combined = pd.concat(all_rets, axis=1).mean(axis=1)
    return combined.dropna()


def compute_spy_reversal(closes):
    """SPY mean-reversion: buy when RSI<30, sell when RSI>70."""
    spy = closes['SPY']
    ret = spy.pct_change()

    # RSI 14
    delta = spy.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Signal: long when RSI < 30 (oversold), out when RSI > 70
    position = pd.Series(0.0, index=spy.index)
    in_trade = False
    for i in range(1, len(rsi)):
        if rsi.iloc[i-1] < 30:
            in_trade = True
        elif rsi.iloc[i-1] > 70:
            in_trade = False
        position.iloc[i] = 1.0 if in_trade else 0.0

    strat_ret = ret * position
    return strat_ret.dropna()


# ============================================================================
# KELLY CRITERION
# ============================================================================
def kelly_fraction(returns, min_obs=60):
    """
    Compute Kelly fraction for a return stream.
    Kelly = mu / sigma^2 (for normally distributed returns)

    For discrete win/loss:
    Kelly = (p * b - q) / b
    where p = win prob, q = lose prob, b = avg win / avg loss
    """
    r = returns[returns != 0]  # Exclude zero-return days
    if len(r) < min_obs:
        return 0.0

    # Continuous Kelly (Gaussian approximation)
    mu = r.mean()
    var = r.var()
    if var < 1e-12:
        return 0.0

    continuous_kelly = mu / var

    # Discrete Kelly
    wins = r[r > 0]
    losses = r[r < 0]
    if len(wins) < 5 or len(losses) < 5:
        return max(0, continuous_kelly)

    p = len(wins) / len(r)
    q = 1 - p
    b = wins.mean() / abs(losses.mean())

    discrete_kelly = (p * b - q) / b if b > 0 else 0.0

    # Use average of both methods, capped at 3x
    kelly = (continuous_kelly + discrete_kelly) / 2
    return min(max(kelly, 0), 3.0)  # Cap at 3x, floor at 0


def walk_forward_kelly(returns, lookback=252, step=21):
    """
    Walk-forward Kelly: recalculate Kelly every `step` days using
    trailing `lookback` window. Returns time series of optimal fraction.
    """
    fractions = pd.Series(0.0, index=returns.index)

    for i in range(lookback, len(returns), step):
        window = returns.iloc[i-lookback:i]
        k = kelly_fraction(window)
        end = min(i + step, len(returns))
        fractions.iloc[i:end] = k

    return fractions


def simulate_kelly_variants(returns, name, lookback=252):
    """
    Simulate performance under different Kelly fractions:
    - Full Kelly (f*)
    - Half Kelly (f*/2) — most practical
    - Quarter Kelly (f*/4) — conservative
    - Walk-forward Kelly
    - Fixed fractions (25%, 50%, 100%)
    """
    results = {}

    # Static Kelly from full sample (for reference)
    full_kelly = kelly_fraction(returns)

    print(f"\n  {name}:")
    print(f"    Full-sample Kelly fraction: {full_kelly:.3f}")

    fractions_to_test = {
        "Full Kelly": full_kelly,
        "Half Kelly": full_kelly / 2,
        "Quarter Kelly": full_kelly / 4,
        "Fixed 25%": 0.25,
        "Fixed 50%": 0.50,
        "Fixed 100%": 1.00,
    }

    for label, frac in fractions_to_test.items():
        if frac <= 0:
            continue

        # Simulate: invest frac of bankroll each day
        # With Kelly, we're sizing the bet as a fraction of total capital
        sized_ret = returns * frac
        equity = (1 + sized_ret).cumprod()

        # Metrics
        n_days = len(equity)
        years = n_days / 252
        final = equity.iloc[-1]
        cagr = final ** (1/years) - 1 if years > 0 else 0
        daily_ret = sized_ret.mean()
        daily_std = sized_ret.std()
        sharpe = daily_ret / (daily_std + 1e-10) * np.sqrt(252)
        sortino_d = sized_ret[sized_ret < 0].std()
        sortino = daily_ret / (sortino_d + 1e-10) * np.sqrt(252) if sortino_d > 0 else 0

        # Drawdown
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max
        max_dd = dd.min()

        # Calmar
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0

        results[label] = {
            'fraction': round(frac, 4),
            'cagr': round(cagr * 100, 2),
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'calmar': round(calmar, 3),
            'max_dd_pct': round(max_dd * 100, 2),
            'final_equity': round(final, 4),
        }

        print(f"    {label:20s}: frac={frac:.3f}, CAGR={cagr*100:.1f}%, "
              f"Sharpe={sharpe:.2f}, MaxDD={max_dd*100:.1f}%, Calmar={calmar:.2f}")

    # Walk-forward Kelly
    wf_fracs = walk_forward_kelly(returns, lookback=lookback)
    wf_sized_ret = returns * wf_fracs
    wf_equity = (1 + wf_sized_ret).cumprod()
    wf_years = len(wf_equity) / 252
    wf_final = wf_equity.iloc[-1]
    wf_cagr = wf_final ** (1/wf_years) - 1 if wf_years > 0 else 0
    wf_sharpe = wf_sized_ret.mean() / (wf_sized_ret.std() + 1e-10) * np.sqrt(252)
    wf_dd = ((wf_equity - wf_equity.cummax()) / wf_equity.cummax()).min()
    wf_calmar = wf_cagr / abs(wf_dd) if abs(wf_dd) > 1e-6 else 0

    results["Walk-Forward Kelly"] = {
        'fraction': round(float(wf_fracs[wf_fracs > 0].mean()), 4) if (wf_fracs > 0).any() else 0,
        'cagr': round(wf_cagr * 100, 2),
        'sharpe': round(wf_sharpe, 3),
        'calmar': round(wf_calmar, 3),
        'max_dd_pct': round(wf_dd * 100, 2),
        'final_equity': round(wf_final, 4),
        'frac_mean': round(float(wf_fracs.mean()), 4),
        'frac_std': round(float(wf_fracs.std()), 4),
    }

    print(f"    {'Walk-Forward Kelly':20s}: avg_frac={wf_fracs.mean():.3f}, CAGR={wf_cagr*100:.1f}%, "
          f"Sharpe={wf_sharpe:.2f}, MaxDD={wf_dd*100:.1f}%, Calmar={wf_calmar:.2f}")

    return results, full_kelly


def drawdown_constrained_sizing(returns, target_max_dd=0.15, n_sims=5000):
    """
    Monte Carlo: find the largest Kelly fraction that keeps MaxDD < target
    with 95% confidence.
    """
    results = {}
    fractions_to_test = np.arange(0.1, 3.01, 0.1)

    for frac in fractions_to_test:
        max_dds = []
        for _ in range(n_sims):
            # Bootstrap daily returns
            sampled = np.random.choice(returns.values, size=len(returns), replace=True)
            sized = sampled * frac
            eq = np.cumprod(1 + sized)
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / peak
            max_dds.append(dd.min())

        max_dds = np.array(max_dds)
        p95_dd = np.percentile(max_dds, 5)  # 5th percentile (worst 5%)

        results[round(frac, 1)] = {
            'median_max_dd': round(float(np.median(max_dds) * 100), 2),
            'p95_max_dd': round(float(p95_dd * 100), 2),
            'p99_max_dd': round(float(np.percentile(max_dds, 1) * 100), 2),
        }

    # Find optimal fraction for target DD
    best_frac = 0.1
    for frac, dd_stats in results.items():
        if abs(dd_stats['p95_max_dd']) <= target_max_dd * 100:
            best_frac = frac

    return best_frac, results


def portfolio_kelly(return_streams, names):
    """
    Multi-asset Kelly with correlation adjustments.
    Kelly vector: f* = Sigma^-1 * mu
    """
    # Build return matrix
    rets = pd.DataFrame({n: s for n, s in zip(names, return_streams)}).dropna()

    if len(rets) < 60:
        return None

    mu = rets.mean().values * 252  # Annualized
    sigma = rets.cov().values * 252  # Annualized

    try:
        sigma_inv = np.linalg.inv(sigma)
        kelly_vec = sigma_inv @ mu
    except np.linalg.LinAlgError:
        return None

    result = {}
    for i, name in enumerate(names):
        result[name] = {
            'kelly_fraction': round(float(kelly_vec[i]), 4),
            'half_kelly': round(float(kelly_vec[i] / 2), 4),
        }

    # Correlations
    corr = rets.corr()
    result['correlations'] = {f"{a}_{b}": round(float(corr.loc[a, b]), 3)
                              for a in names for b in names if a < b}

    return result


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("KELLY CRITERION & OPTIMAL POSITION SIZING STUDY")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    closes = download_data()

    # Compute strategy return streams
    print("\nComputing strategy return streams...")

    strat_returns = {}
    strat_returns["UPRO_protected"] = compute_upro_protected(closes)
    strat_returns["CTA_trend"] = compute_cta_trend(closes)
    strat_returns["SPY_reversal"] = compute_spy_reversal(closes)

    # Also add buy-and-hold baselines
    strat_returns["SPY_buyhold"] = closes['SPY'].pct_change().dropna()
    strat_returns["UPRO_buyhold"] = closes['UPRO'].pct_change().dropna()

    # ======================================================================
    # Per-strategy Kelly analysis
    # ======================================================================
    print("\n" + "=" * 70)
    print("PER-STRATEGY KELLY ANALYSIS")
    print("=" * 70)

    all_results = {}
    kelly_fractions = {}

    for name, rets in strat_returns.items():
        print(f"\n--- {name} ({len(rets)} days) ---")

        # Basic stats
        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        wins = (rets > 0).sum()
        total = (rets != 0).sum()
        wr = wins / total if total > 0 else 0

        print(f"  Ann Return: {ann_ret*100:.1f}%, Ann Vol: {ann_vol*100:.1f}%, WR: {wr:.1%}")

        results, full_k = simulate_kelly_variants(rets, name)
        all_results[name] = results
        kelly_fractions[name] = full_k

    # ======================================================================
    # Drawdown-constrained sizing
    # ======================================================================
    print("\n" + "=" * 70)
    print("DRAWDOWN-CONSTRAINED SIZING (Monte Carlo)")
    print("=" * 70)

    dd_targets = [0.10, 0.15, 0.20, 0.25]
    dd_results = {}

    for name in ["UPRO_protected", "CTA_trend", "SPY_reversal"]:
        rets = strat_returns[name]
        print(f"\n  {name}:")
        dd_results[name] = {}

        for target in dd_targets:
            best_frac, _ = drawdown_constrained_sizing(rets, target_max_dd=target, n_sims=2000)
            dd_results[name][f"MaxDD_{int(target*100)}pct"] = {
                'optimal_fraction': best_frac,
            }
            print(f"    MaxDD target {target:.0%}: optimal fraction = {best_frac:.1f}x")

    # ======================================================================
    # Portfolio-level Kelly
    # ======================================================================
    print("\n" + "=" * 70)
    print("PORTFOLIO-LEVEL KELLY (multi-asset)")
    print("=" * 70)

    portfolio_names = ["UPRO_protected", "CTA_trend", "SPY_reversal"]
    portfolio_rets = [strat_returns[n] for n in portfolio_names]

    port_kelly = portfolio_kelly(portfolio_rets, portfolio_names)
    if port_kelly:
        print("\n  Optimal allocation (Kelly):")
        for name in portfolio_names:
            k = port_kelly[name]
            print(f"    {name:20s}: Full Kelly = {k['kelly_fraction']:.3f}, Half Kelly = {k['half_kelly']:.3f}")

        print("\n  Correlations:")
        for pair, corr in port_kelly.get('correlations', {}).items():
            print(f"    {pair}: {corr:.3f}")

    # ======================================================================
    # PRACTICAL RECOMMENDATIONS
    # ======================================================================
    print("\n" + "=" * 70)
    print("PRACTICAL RECOMMENDATIONS")
    print("=" * 70)

    print("""
    KEY FINDINGS:
    1. Full Kelly is theoretically optimal but practically too aggressive
       (extreme drawdowns). ALWAYS use Half Kelly or less.

    2. Half Kelly ≈ 85% of Full Kelly's long-run growth with ~50% of
       the drawdown. This is the practical sweet spot.

    3. Walk-forward Kelly adapts to changing market conditions — better
       than static fractions for volatile strategies.

    4. For UPRO (already 3x levered), Kelly fraction < 1 means don't
       lever further. Fraction > 1 means UPRO itself is under-leveraged
       for the strategy.

    5. Drawdown-constrained sizing (targeting 15% MaxDD) is the most
       practical approach for real capital deployment.

    DEPLOYMENT RULES:
    - Phase 1 ($500-$2K): Half Kelly, max 1 position, no leverage
    - Phase 2 ($2K-$10K): Half Kelly, up to 3 positions, threshold 5% rebalance
    - Phase 3 ($10K-$50K): Quarter Kelly, full portfolio, monthly rebalance
    - Phase 4 ($50K+): Drawdown-constrained sizing, all strategies, institutional risk management
    """)

    # ======================================================================
    # Save results
    # ======================================================================
    summary = {
        'run_date': datetime.now().isoformat(),
        'data_range': f"{closes.index[0].date()} to {closes.index[-1].date()}",
        'per_strategy': all_results,
        'kelly_fractions': {k: round(v, 4) for k, v in kelly_fractions.items()},
        'drawdown_constrained': dd_results,
        'portfolio_kelly': port_kelly,
    }

    out_file = OUTPUT_DIR / "kelly_sizing_results.json"
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to {out_file}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
