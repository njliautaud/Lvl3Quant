#!/usr/bin/env python3
"""
Cross-Sectional Momentum Long-Short Research
=============================================
Universe: 19 ETFs (11 SPDR sectors + SPY, QQQ, IWM, EFA, EEM, GLD, TLT, HYG)
Strategies: Raw momentum, risk-adjusted momentum, dual momentum
Walk-forward: sliding 252d lookback, 21d OOS step
Costs: 5bps per side at each rebalance
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = '/home/jupiter/Lvl3Quant/output/momentum_longshort_research'

# ─── Universe ───────────────────────────────────────────────────────────
UNIVERSE = [
    'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY',  # 11 SPDR sectors
    'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'GLD', 'TLT', 'HYG'  # broad + alternatives
]

COST_BPS = 5  # 5bps per side
COST_FRAC = COST_BPS / 10000  # 0.0005
K_LONG = 5   # top 25%
K_SHORT = 5  # bottom 25%
REBAL_DAYS = 21  # monthly

# ─── Data Download ──────────────────────────────────────────────────────
def download_data():
    """Download daily adj close prices for universe."""
    cache_path = os.path.join(OUT_DIR, 'price_data.parquet')
    if os.path.exists(cache_path):
        print("Loading cached price data...")
        return pd.read_parquet(cache_path)

    print(f"Downloading data for {len(UNIVERSE)} assets...")
    data = yf.download(UNIVERSE, start='2014-01-01', end='2026-07-11', auto_adjust=True)
    # yfinance returns MultiIndex columns; extract Close
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data

    prices = prices.dropna(how='all')
    prices.to_parquet(cache_path)
    print(f"Downloaded {len(prices)} days, {prices.shape[1]} assets")
    return prices


# ─── Momentum Signals ──────────────────────────────────────────────────
def compute_momentum(prices, lookback=252, skip=21):
    """12-1 momentum: trailing return excluding last month."""
    # Return from t-lookback to t-skip
    ret = prices.shift(skip) / prices.shift(lookback) - 1
    return ret


def compute_risk_adj_momentum(prices, lookback=252, skip=21):
    """Momentum / realized volatility."""
    mom = compute_momentum(prices, lookback, skip)
    vol = prices.pct_change().rolling(lookback).std() * np.sqrt(252)
    return mom / vol.replace(0, np.nan)


def compute_dual_momentum_mask(prices, sma_period=200):
    """Absolute momentum filter: long only if above SMA, short only if below."""
    sma = prices.rolling(sma_period).mean()
    above_sma = prices > sma
    below_sma = prices < sma
    return above_sma, below_sma


# ─── Portfolio Construction ─────────────────────────────────────────────
def build_ls_portfolio(signal_df, prices, k_long=K_LONG, k_short=K_SHORT,
                       rebal_days=REBAL_DAYS, cost_frac=COST_FRAC,
                       dual_mask_long=None, dual_mask_short=None,
                       long_only=False):
    """
    Walk-forward long-short portfolio.
    Returns daily returns series.
    """
    daily_ret = prices.pct_change()
    n_assets = signal_df.shape[1]

    dates = signal_df.index
    # Start after enough data
    start_idx = signal_df.first_valid_index()
    if start_idx is None:
        return pd.Series(dtype=float)

    start_loc = dates.get_loc(start_idx)
    if isinstance(start_loc, slice):
        start_loc = start_loc.start

    port_returns = []
    port_dates = []
    prev_weights = pd.Series(0.0, index=signal_df.columns)

    rebal_counter = 0

    for i in range(start_loc, len(dates)):
        dt = dates[i]
        sig = signal_df.loc[dt].dropna()
        ret_today = daily_ret.loc[dt]

        if len(sig) < k_long + k_short:
            port_returns.append(0.0)
            port_dates.append(dt)
            continue

        # Rebalance check
        if rebal_counter % rebal_days == 0:
            ranked = sig.rank(ascending=True)
            n_valid = len(sig)

            # Bottom K = short, Top K = long
            short_names = ranked.nsmallest(k_short).index.tolist()
            long_names = ranked.nlargest(k_long).index.tolist()

            # Apply dual momentum filter if provided
            if dual_mask_long is not None and dt in dual_mask_long.index:
                # Only long if above SMA
                long_names = [s for s in long_names if dual_mask_long.loc[dt].get(s, False)]
                # Only short if below SMA
                short_names = [s for s in short_names if dual_mask_short.loc[dt].get(s, False)]

            if long_only:
                short_names = []

            # Equal weight within each leg
            new_weights = pd.Series(0.0, index=signal_df.columns)
            if len(long_names) > 0:
                for s in long_names:
                    new_weights[s] = 1.0 / max(len(long_names), 1)
            if len(short_names) > 0:
                for s in short_names:
                    new_weights[s] = -1.0 / max(len(short_names), 1)

            # Turnover cost
            turnover = (new_weights - prev_weights).abs().sum()
            cost = turnover * cost_frac

            prev_weights = new_weights
        else:
            cost = 0.0

        rebal_counter += 1

        # Portfolio return
        valid_assets = prev_weights.index.intersection(ret_today.dropna().index)
        port_ret = (prev_weights[valid_assets] * ret_today[valid_assets]).sum() - cost
        port_returns.append(port_ret)
        port_dates.append(dt)

    return pd.Series(port_returns, index=port_dates)


# ─── Performance Metrics ────────────────────────────────────────────────
def compute_metrics(returns, spy_returns=None, name="Strategy"):
    """Compute comprehensive performance metrics."""
    returns = returns.dropna()
    if len(returns) < 252:
        return {"name": name, "error": "insufficient data"}

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    cagr = cum.iloc[-1] ** (252 / len(returns)) - 1

    # Monthly returns for win rate
    monthly = returns.resample('ME').sum()
    wr = (monthly > 0).mean()

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    result = {
        "name": name,
        "CAGR": f"{cagr:.1%}",
        "Ann_Vol": f"{ann_vol:.1%}",
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "MaxDD": f"{max_dd:.1%}",
        "Calmar": round(calmar, 2),
        "PF": round(pf, 2),
        "WR_monthly": f"{wr:.1%}",
        "N_days": len(returns),
    }

    # Regime analysis if SPY returns provided
    if spy_returns is not None:
        aligned = pd.DataFrame({'strat': returns, 'spy': spy_returns}).dropna()
        green = aligned['spy'] > 0.005 / 252 * 100  # roughly +0.5% daily
        red = aligned['spy'] < -0.005 / 252 * 100
        # Simpler: just use daily SPY return sign with threshold
        green_days = aligned[aligned['spy'] > 0.003]  # ~0.3% up day
        red_days = aligned[aligned['spy'] < -0.003]
        flat_days = aligned[(aligned['spy'] >= -0.003) & (aligned['spy'] <= 0.003)]

        def regime_sharpe(rets):
            if len(rets) < 20:
                return np.nan
            return rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0

        sharpe_green = regime_sharpe(green_days['strat'])
        sharpe_red = regime_sharpe(red_days['strat'])
        sharpe_flat = regime_sharpe(flat_days['strat'])

        regime_gap = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red), 0.01)

        result["Sharpe_Green"] = round(sharpe_green, 2)
        result["Sharpe_Red"] = round(sharpe_red, 2)
        result["Sharpe_Flat"] = round(sharpe_flat, 2)
        result["Regime_Gap"] = round(regime_gap, 2)
        result["N_Green"] = len(green_days)
        result["N_Red"] = len(red_days)
        result["N_Flat"] = len(flat_days)

    return result


# ─── Permutation Test ───────────────────────────────────────────────────
def permutation_test(signal_df, prices, n_perms=500, k_long=K_LONG, k_short=K_SHORT,
                     rebal_days=REBAL_DAYS, cost_frac=COST_FRAC):
    """Shuffle rankings at each rebalance date, compute Sharpe distribution."""
    print(f"  Running {n_perms} permutations...")
    np.random.seed(42)

    daily_ret = prices.pct_change()
    dates = signal_df.index
    start_idx = signal_df.first_valid_index()
    if start_idx is None:
        return []
    start_loc = dates.get_loc(start_idx)
    if isinstance(start_loc, slice):
        start_loc = start_loc.start

    perm_sharpes = []

    for p in range(n_perms):
        port_returns = []
        prev_weights = pd.Series(0.0, index=signal_df.columns)
        rebal_counter = 0

        for i in range(start_loc, len(dates)):
            dt = dates[i]
            sig = signal_df.loc[dt].dropna()
            ret_today = daily_ret.loc[dt]

            if len(sig) < k_long + k_short:
                port_returns.append(0.0)
                continue

            if rebal_counter % rebal_days == 0:
                # SHUFFLE the signal values
                shuffled = sig.copy()
                shuffled[:] = np.random.permutation(shuffled.values)
                ranked = shuffled.rank(ascending=True)

                short_names = ranked.nsmallest(k_short).index.tolist()
                long_names = ranked.nlargest(k_long).index.tolist()

                new_weights = pd.Series(0.0, index=signal_df.columns)
                for s in long_names:
                    new_weights[s] = 1.0 / k_long
                for s in short_names:
                    new_weights[s] = -1.0 / k_short

                turnover = (new_weights - prev_weights).abs().sum()
                cost = turnover * cost_frac
                prev_weights = new_weights
            else:
                cost = 0.0

            rebal_counter += 1
            valid_assets = prev_weights.index.intersection(ret_today.dropna().index)
            port_ret = (prev_weights[valid_assets] * ret_today[valid_assets]).sum() - cost
            port_returns.append(port_ret)

        rets = pd.Series(port_returns)
        if rets.std() > 0:
            perm_sharpes.append(rets.mean() / rets.std() * np.sqrt(252))

    return perm_sharpes


# ─── Main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("CROSS-SECTIONAL MOMENTUM LONG-SHORT RESEARCH")
    print("=" * 70)

    # 1. Download data
    prices = download_data()
    print(f"\nPrice data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Assets with data: {prices.columns.tolist()}")

    # Check for missing assets
    missing = [t for t in UNIVERSE if t not in prices.columns]
    if missing:
        print(f"WARNING: Missing tickers: {missing}")

    spy_ret = prices['SPY'].pct_change()

    # Trim to 2015+ for consistent universe
    prices = prices.loc['2015-01-01':]
    spy_ret = spy_ret.loc[prices.index[0]:]

    results = []
    equity_curves = {}

    # ─── Strategy 1: Raw Momentum (12-1) Long-Short ─────────────────
    print("\n" + "─" * 50)
    print("Strategy 1: Raw 12-1 Momentum Long-Short")
    for lookback, skip, label in [
        (63, 21, "3mo-1mo"),
        (126, 21, "6mo-1mo"),
        (252, 21, "12mo-1mo"),
    ]:
        print(f"  Lookback: {label}")
        mom = compute_momentum(prices, lookback=lookback, skip=skip)
        rets = build_ls_portfolio(mom, prices)
        m = compute_metrics(rets, spy_ret, f"RawMom_LS_{label}")
        results.append(m)
        if label == "12mo-1mo":
            equity_curves['raw_ls_12mo'] = rets
        print(f"    Sharpe={m.get('Sharpe','N/A')}, CAGR={m.get('CAGR','N/A')}, MaxDD={m.get('MaxDD','N/A')}, Regime_Gap={m.get('Regime_Gap','N/A')}")

    # ─── Strategy 1b: Raw Momentum Long-Only (benchmark) ───────────
    print("\n" + "─" * 50)
    print("Strategy 1b: Raw 12-1 Momentum LONG-ONLY")
    mom_12 = compute_momentum(prices, lookback=252, skip=21)
    rets_lo = build_ls_portfolio(mom_12, prices, long_only=True)
    m_lo = compute_metrics(rets_lo, spy_ret, "RawMom_LO_12mo")
    results.append(m_lo)
    equity_curves['raw_lo_12mo'] = rets_lo
    print(f"    Sharpe={m_lo.get('Sharpe','N/A')}, CAGR={m_lo.get('CAGR','N/A')}, MaxDD={m_lo.get('MaxDD','N/A')}, Regime_Gap={m_lo.get('Regime_Gap','N/A')}")

    # ─── Strategy 2: Risk-Adjusted Momentum ─────────────────────────
    print("\n" + "─" * 50)
    print("Strategy 2: Risk-Adjusted Momentum (return/vol) Long-Short")
    for lookback, skip, label in [
        (63, 21, "3mo-1mo"),
        (126, 21, "6mo-1mo"),
        (252, 21, "12mo-1mo"),
    ]:
        print(f"  Lookback: {label}")
        ra_mom = compute_risk_adj_momentum(prices, lookback=lookback, skip=skip)
        rets = build_ls_portfolio(ra_mom, prices)
        m = compute_metrics(rets, spy_ret, f"RiskAdjMom_LS_{label}")
        results.append(m)
        if label == "12mo-1mo":
            equity_curves['riskadjmom_ls_12mo'] = rets
        print(f"    Sharpe={m.get('Sharpe','N/A')}, CAGR={m.get('CAGR','N/A')}, MaxDD={m.get('MaxDD','N/A')}, Regime_Gap={m.get('Regime_Gap','N/A')}")

    # ─── Strategy 3: Dual Momentum Long-Short ───────────────────────
    print("\n" + "─" * 50)
    print("Strategy 3: Dual Momentum (relative + absolute SMA filter)")
    above_sma, below_sma = compute_dual_momentum_mask(prices, sma_period=200)
    for lookback, skip, label in [
        (63, 21, "3mo-1mo"),
        (126, 21, "6mo-1mo"),
        (252, 21, "12mo-1mo"),
    ]:
        print(f"  Lookback: {label}")
        mom = compute_momentum(prices, lookback=lookback, skip=skip)
        rets = build_ls_portfolio(mom, prices, dual_mask_long=above_sma, dual_mask_short=below_sma)
        m = compute_metrics(rets, spy_ret, f"DualMom_LS_{label}")
        results.append(m)
        if label == "12mo-1mo":
            equity_curves['dualmom_ls_12mo'] = rets
        print(f"    Sharpe={m.get('Sharpe','N/A')}, CAGR={m.get('CAGR','N/A')}, MaxDD={m.get('MaxDD','N/A')}, Regime_Gap={m.get('Regime_Gap','N/A')}")

    # ─── Strategy 3b: Dual Momentum Long-Only ───────────────────────
    print("\n" + "─" * 50)
    print("Strategy 3b: Dual Momentum LONG-ONLY (absolute filter only)")
    mom_12 = compute_momentum(prices, lookback=252, skip=21)
    rets_dual_lo = build_ls_portfolio(mom_12, prices, dual_mask_long=above_sma, dual_mask_short=below_sma, long_only=True)
    m_dual_lo = compute_metrics(rets_dual_lo, spy_ret, "DualMom_LO_12mo")
    results.append(m_dual_lo)
    equity_curves['dualmom_lo_12mo'] = rets_dual_lo
    print(f"    Sharpe={m_dual_lo.get('Sharpe','N/A')}, CAGR={m_dual_lo.get('CAGR','N/A')}, MaxDD={m_dual_lo.get('MaxDD','N/A')}, Regime_Gap={m_dual_lo.get('Regime_Gap','N/A')}")

    # ─── SPY Buy & Hold Benchmark ───────────────────────────────────
    print("\n" + "─" * 50)
    print("Benchmark: SPY Buy & Hold")
    spy_bh = spy_ret.loc[prices.index[0]:].dropna()
    m_spy = compute_metrics(spy_bh, spy_ret, "SPY_BuyHold")
    results.append(m_spy)
    equity_curves['spy_bh'] = spy_bh
    print(f"    Sharpe={m_spy.get('Sharpe','N/A')}, CAGR={m_spy.get('CAGR','N/A')}, MaxDD={m_spy.get('MaxDD','N/A')}")

    # ─── Permutation Test on Best Strategy ──────────────────────────
    print("\n" + "─" * 50)
    print("Permutation Test (500 shuffles on 12mo raw momentum)")
    mom_12 = compute_momentum(prices, lookback=252, skip=21)

    # Get actual Sharpe
    actual_rets = build_ls_portfolio(mom_12, prices)
    actual_sharpe = actual_rets.mean() / actual_rets.std() * np.sqrt(252)

    perm_sharpes = permutation_test(mom_12, prices, n_perms=500)
    p_value = np.mean([s >= actual_sharpe for s in perm_sharpes])

    perm_result = {
        "actual_sharpe": round(actual_sharpe, 3),
        "perm_mean_sharpe": round(np.mean(perm_sharpes), 3),
        "perm_std_sharpe": round(np.std(perm_sharpes), 3),
        "perm_p95_sharpe": round(np.percentile(perm_sharpes, 95), 3),
        "p_value": round(p_value, 4),
        "significant_at_5pct": p_value < 0.05,
    }
    print(f"  Actual Sharpe: {perm_result['actual_sharpe']}")
    print(f"  Random mean: {perm_result['perm_mean_sharpe']} ± {perm_result['perm_std_sharpe']}")
    print(f"  p-value: {perm_result['p_value']} ({'SIGNIFICANT' if perm_result['significant_at_5pct'] else 'NOT significant'})")

    # ─── Annual Breakdown for Key Strategies ────────────────────────
    print("\n" + "─" * 50)
    print("Annual Returns Breakdown")
    annual_data = {}
    for name, ec in [('RawMom_LS_12mo', equity_curves.get('raw_ls_12mo')),
                      ('RawMom_LO_12mo', equity_curves.get('raw_lo_12mo')),
                      ('RiskAdjMom_LS_12mo', equity_curves.get('riskadjmom_ls_12mo')),
                      ('DualMom_LS_12mo', equity_curves.get('dualmom_ls_12mo')),
                      ('SPY_BH', equity_curves.get('spy_bh'))]:
        if ec is not None:
            annual = ec.resample('YE').sum()
            annual_data[name] = {str(d.year): f"{v:.1%}" for d, v in annual.items()}
            print(f"\n  {name}:")
            for yr, ret in annual_data[name].items():
                print(f"    {yr}: {ret}")

    # ─── Drawdown Analysis ──────────────────────────────────────────
    print("\n" + "─" * 50)
    print("Drawdown Analysis (worst drawdowns for LS 12mo)")
    ls_rets = equity_curves.get('raw_ls_12mo')
    if ls_rets is not None:
        cum = (1 + ls_rets).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        # Find worst 5 drawdown troughs
        worst = dd.nsmallest(5)
        print("  Worst drawdown periods:")
        for dt, val in worst.items():
            print(f"    {dt.date()}: {val:.1%}")

    # ─── Regime Agnosticism Comparison ──────────────────────────────
    print("\n" + "=" * 70)
    print("KEY QUESTION: Is Long-Short more regime-agnostic than Long-Only?")
    print("=" * 70)

    ls_result = next((r for r in results if r['name'] == 'RawMom_LS_12mo-1mo'), None)
    lo_result = next((r for r in results if r['name'] == 'RawMom_LO_12mo'), None)

    if ls_result and lo_result and 'Regime_Gap' in ls_result and 'Regime_Gap' in lo_result:
        print(f"\n  Long-Short Regime Gap: {ls_result['Regime_Gap']}")
        print(f"  Long-Only Regime Gap:  {lo_result['Regime_Gap']}")
        if ls_result['Regime_Gap'] < lo_result['Regime_Gap']:
            print("  ✓ Long-Short IS more regime-agnostic (lower gap)")
        else:
            print("  ✗ Long-Short is NOT more regime-agnostic")

        print(f"\n  Long-Short: Sharpe_Green={ls_result.get('Sharpe_Green','N/A')}, Sharpe_Red={ls_result.get('Sharpe_Red','N/A')}")
        print(f"  Long-Only:  Sharpe_Green={lo_result.get('Sharpe_Green','N/A')}, Sharpe_Red={lo_result.get('Sharpe_Red','N/A')}")

    # ─── Final Comparison Table ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)

    # Format as table
    headers = ["Strategy", "Sharpe", "Sortino", "CAGR", "MaxDD", "PF", "WR_mo", "Regime_Gap"]
    print(f"\n{'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>8} {'PF':>6} {'WR_mo':>6} {'R.Gap':>6}")
    print("-" * 85)
    for r in results:
        if 'error' in r:
            continue
        print(f"{r['name']:<30} {r.get('Sharpe',''):>7} {r.get('Sortino',''):>8} {r.get('CAGR',''):>7} {r.get('MaxDD',''):>8} {r.get('PF',''):>6} {r.get('WR_monthly',''):>6} {r.get('Regime_Gap','N/A'):>6}")

    # ─── Save Results ───────────────────────────────────────────────
    output = {
        "timestamp": datetime.now().isoformat(),
        "universe": UNIVERSE,
        "cost_bps_per_side": COST_BPS,
        "strategies": results,
        "permutation_test": perm_result,
        "annual_returns": annual_data,
    }

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Save equity curves
    ec_df = pd.DataFrame(equity_curves)
    ec_df.to_parquet(os.path.join(OUT_DIR, 'equity_curves.parquet'))

    print(f"\nResults saved to {OUT_DIR}/results.json")
    print(f"Equity curves saved to {OUT_DIR}/equity_curves.parquet")

    # ─── Bottom Line ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BOTTOM LINE ASSESSMENT")
    print("=" * 70)

    best_ls = max([r for r in results if 'LS' in r.get('name', '') and 'error' not in r],
                  key=lambda x: x.get('Sharpe', -99))
    best_lo = max([r for r in results if 'LO' in r.get('name', '') and 'error' not in r],
                  key=lambda x: x.get('Sharpe', -99))

    print(f"\n  Best Long-Short: {best_ls['name']} (Sharpe {best_ls['Sharpe']})")
    print(f"  Best Long-Only:  {best_lo['name']} (Sharpe {best_lo['Sharpe']})")

    if best_ls['Sharpe'] > best_lo['Sharpe']:
        print(f"\n  Long-Short BEATS Long-Only by {best_ls['Sharpe'] - best_lo['Sharpe']:.2f} Sharpe units")
    else:
        print(f"\n  Long-Only BEATS Long-Short by {best_lo['Sharpe'] - best_ls['Sharpe']:.2f} Sharpe units")
        print("  NOTE: Long-short cross-sectional momentum on 19 ETFs does not add value vs long-only after costs.")

    if perm_result['significant_at_5pct']:
        print(f"  Permutation test: SIGNIFICANT (p={perm_result['p_value']})")
    else:
        print(f"  Permutation test: NOT significant (p={perm_result['p_value']}) — momentum ranking may be noise")

    return output


if __name__ == '__main__':
    main()
