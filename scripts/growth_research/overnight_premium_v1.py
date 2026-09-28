#!/usr/bin/env python3
"""
Overnight Premium Capture — v1
===============================
Tests the well-documented overnight return anomaly across SPY, UPRO, TQQQ, QQQ.

Strategies:
  1. Overnight Only (buy close, sell open)
  2. Overnight UPRO (3x leveraged overnight)
  3. Day-Avoidance (overnight long + intraday short)
  4. VIX-Filtered Overnight (hold overnight only when VIX < 25)
  5. v4.4 + Overnight (SMA timing + overnight execution)

Benchmarks: SPY buy-hold, UPRO buy-hold

Adversarial:
  - Permutation test (100 shuffles)
  - Sub-period consistency (3 blocks, Sharpe CV)
  - Walk-forward (3yr train / 1yr test)

HC #713: Fixed $100K, NO DCA.
HC #705: Adversarial validation mandatory.
"""

import os
import warnings
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/overnight_premium_v1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000
COMMISSION_RT = 4.70  # AMP round-trip
TRADES_PER_DAY = 2    # buy close + sell open = 1 RT; but day-avoidance = 2 RT
START_DATE = '2010-01-01'
END_DATE = '2026-07-18'

# ─── Data Download ────────────────────────────────────────────────────────────

def download_data():
    """Download daily OHLC for SPY, UPRO, TQQQ, QQQ, ^VIX."""
    tickers = ['SPY', 'UPRO', 'TQQQ', 'QQQ', '^VIX']
    data = {}
    for t in tickers:
        print(f"Downloading {t}...")
        df = yf.download(t, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[t.replace('^', '')] = df[['Open', 'High', 'Low', 'Close']].copy()
    return data


# ─── Return Computation ──────────────────────────────────────────────────────

def compute_returns(df):
    """Compute overnight and intraday return series."""
    # Overnight: close[t-1] -> open[t]
    overnight = df['Open'] / df['Close'].shift(1) - 1
    # Intraday: open[t] -> close[t]
    intraday = df['Close'] / df['Open'] - 1
    # Total: close[t-1] -> close[t]
    total = df['Close'] / df['Close'].shift(1) - 1
    return overnight.iloc[1:], intraday.iloc[1:], total.iloc[1:]


# ─── Strategy Implementations ────────────────────────────────────────────────

def strategy_overnight_only(df, commission_rt=COMMISSION_RT):
    """Buy at close, sell at open. 1 RT/day."""
    overnight, _, _ = compute_returns(df)
    # Commission as fraction of capital (assume ~$100K notional)
    daily_cost = commission_rt / INITIAL_CAPITAL
    net_returns = overnight - daily_cost
    return net_returns


def strategy_overnight_upro(upro_df, commission_rt=COMMISSION_RT):
    """Same as overnight only but on UPRO (3x leveraged)."""
    overnight, _, _ = compute_returns(upro_df)
    daily_cost = commission_rt / INITIAL_CAPITAL
    net_returns = overnight - daily_cost
    return net_returns


def strategy_day_avoidance(df, commission_rt=COMMISSION_RT):
    """Buy at close (overnight long) + short during day. 2 RT/day."""
    overnight, intraday, _ = compute_returns(df)
    # Long overnight + short intraday = overnight_ret - intraday_ret
    # (short intraday means you profit when intraday is negative)
    combined = overnight - intraday
    daily_cost = (2 * commission_rt) / INITIAL_CAPITAL  # 2 round trips
    # Align indices
    idx = overnight.index.intersection(intraday.index)
    combined = (overnight.loc[idx] - intraday.loc[idx]) - daily_cost
    return combined


def strategy_vix_filtered(spy_df, vix_df, commission_rt=COMMISSION_RT):
    """Overnight only when VIX < 25. Cash otherwise."""
    overnight, _, _ = compute_returns(spy_df)
    vix_close = vix_df['Close']
    # Align: VIX at close[t-1] determines if we hold overnight t-1->t
    vix_shifted = vix_close.shift(1)
    idx = overnight.index.intersection(vix_shifted.index)
    overnight = overnight.loc[idx]
    vix_at_decision = vix_shifted.loc[idx]
    mask = vix_at_decision < 25
    daily_cost = commission_rt / INITIAL_CAPITAL
    filtered = overnight.copy() * 0  # start with zeros
    filtered[mask] = overnight[mask] - daily_cost
    return filtered


def strategy_v44_overnight(spy_df, commission_rt=COMMISSION_RT):
    """v4.4 timing (200-day SMA direction) + overnight execution.
    Only hold overnight when SPY > 200-day SMA."""
    overnight, _, _ = compute_returns(spy_df)
    sma200 = spy_df['Close'].rolling(200).mean()
    # Signal: close[t-1] > SMA200[t-1] → hold overnight
    signal = (spy_df['Close'].shift(1) > sma200.shift(1))
    idx = overnight.index.intersection(signal.index)
    overnight = overnight.loc[idx]
    signal = signal.loc[idx]
    daily_cost = commission_rt / INITIAL_CAPITAL
    result = overnight.copy() * 0
    result[signal] = overnight[signal] - daily_cost
    return result


def benchmark_buy_hold(df):
    """Simple buy-and-hold total return."""
    _, _, total = compute_returns(df)
    return total


# ─── Performance Metrics ─────────────────────────────────────────────────────

def compute_metrics(returns, name="Strategy"):
    """Compute Sharpe, Sortino, CAGR, MaxDD, Calmar."""
    returns = returns.dropna()
    if len(returns) == 0:
        return {k: np.nan for k in ['Sharpe', 'Sortino', 'CAGR', 'MaxDD', 'Calmar',
                                      'WinRate', 'AvgWin', 'AvgLoss', 'PF', 'TotalReturn',
                                      'NumTradingDays', 'Name']}

    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    years = len(returns) / 252
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    # Drawdown
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Sharpe (annualized)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-10
    sortino = returns.mean() / downside_std * np.sqrt(252)

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate & profit factor
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    wr = len(wins) / len(returns) if len(returns) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else np.inf

    return {
        'Name': name,
        'Sharpe': round(sharpe, 3),
        'Sortino': round(sortino, 3),
        'CAGR': round(cagr * 100, 2),
        'MaxDD': round(max_dd * 100, 2),
        'Calmar': round(calmar, 3),
        'WinRate': round(wr * 100, 1),
        'AvgWin': round(avg_win * 10000, 2),  # in bps
        'AvgLoss': round(avg_loss * 10000, 2),  # in bps
        'PF': round(pf, 3),
        'TotalReturn': round(total_ret * 100, 1),
        'NumTradingDays': len(returns),
    }


# ─── Equity Curve & Plotting ─────────────────────────────────────────────────

def equity_curve(returns, capital=INITIAL_CAPITAL):
    """Convert return series to equity curve."""
    return capital * (1 + returns).cumprod()


def plot_equity_curves(curves_dict, filename='equity_curves.png'):
    """Plot multiple equity curves."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Top: equity curves
    ax = axes[0]
    for name, curve in curves_dict.items():
        ax.plot(curve.index, curve.values, label=name, alpha=0.8)
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title('Overnight Premium Strategies — Equity Curves')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Bottom: drawdowns
    ax = axes[1]
    for name, curve in curves_dict.items():
        peak = curve.cummax()
        dd = (curve - peak) / peak * 100
        ax.plot(dd.index, dd.values, label=name, alpha=0.7)
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdowns')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_annual_returns(returns_dict, filename='annual_returns.png'):
    """Heatmap of annual returns."""
    annual_data = {}
    for name, rets in returns_dict.items():
        annual = rets.groupby(rets.index.year).apply(lambda x: (1+x).prod() - 1) * 100
        annual_data[name] = annual

    df = pd.DataFrame(annual_data)
    fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.4)))
    im = ax.imshow(df.values, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            color = 'white' if abs(val) > 25 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=7, color=color)
    plt.colorbar(im, label='Annual Return (%)')
    ax.set_title('Annual Returns by Strategy (%)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


# ─── Cost Sensitivity ────────────────────────────────────────────────────────

def cost_sensitivity(spy_df, upro_df):
    """Test strategies at different commission levels."""
    commissions = [0, 2, 4.70, 7, 10]
    results = []
    for c in commissions:
        for name, func, args in [
            ('Overnight SPY', strategy_overnight_only, (spy_df, c)),
            ('Overnight UPRO', strategy_overnight_upro, (upro_df, c)),
            ('Day-Avoidance SPY', strategy_day_avoidance, (spy_df, c)),
        ]:
            rets = func(*args)
            m = compute_metrics(rets, name)
            m['Commission_RT'] = c
            results.append(m)
    return pd.DataFrame(results)


# ─── Correlation Analysis ────────────────────────────────────────────────────

def correlation_analysis(returns_dict, spy_total):
    """Compute correlation of each strategy with SPY buy-hold."""
    corrs = {}
    for name, rets in returns_dict.items():
        idx = rets.index.intersection(spy_total.index)
        if len(idx) > 0:
            corrs[name] = round(rets.loc[idx].corr(spy_total.loc[idx]), 3)
    return corrs


# ─── Adversarial Tests ───────────────────────────────────────────────────────

def permutation_test(returns, n_perms=100):
    """Shuffle which nights we hold. If random works equally → no edge."""
    actual_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    perm_sharpes = []
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        # Randomly select same number of days
        mask = rng.choice([True, False], size=len(returns), p=[0.5, 0.5])
        perm_rets = returns.copy()
        perm_rets[~mask] = 0
        # Scale to match same exposure
        if mask.sum() > 0:
            s = perm_rets.mean() / perm_rets.std() * np.sqrt(252) if perm_rets.std() > 0 else 0
        else:
            s = 0
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).mean()
    return {
        'actual_sharpe': round(actual_sharpe, 3),
        'perm_mean_sharpe': round(perm_sharpes.mean(), 3),
        'perm_std_sharpe': round(perm_sharpes.std(), 3),
        'p_value': round(p_value, 3),
        'edge_vs_random': actual_sharpe > perm_sharpes.mean() + 2 * perm_sharpes.std()
    }


def subperiod_consistency(returns, n_blocks=3):
    """Split into n_blocks, compute Sharpe for each. CV < 0.6 = consistent."""
    block_size = len(returns) // n_blocks
    sharpes = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else len(returns)
        block = returns.iloc[start:end]
        s = block.mean() / block.std() * np.sqrt(252) if block.std() > 0 else 0
        sharpes.append(s)
    sharpes = np.array(sharpes)
    cv = sharpes.std() / abs(sharpes.mean()) if sharpes.mean() != 0 else np.inf
    return {
        'block_sharpes': [round(s, 3) for s in sharpes],
        'cv': round(cv, 3),
        'consistent': cv < 0.6,
        'all_positive': all(s > 0 for s in sharpes)
    }


def walk_forward_test(returns, train_years=3, test_years=1):
    """Rolling walk-forward: 3yr train, 1yr test. Report OOS Sharpe stability."""
    results = []
    years = returns.index.year
    unique_years = sorted(years.unique())

    for i in range(len(unique_years) - train_years - test_years + 1):
        train_end_year = unique_years[i + train_years - 1]
        test_start_year = unique_years[i + train_years]
        test_end_year = unique_years[i + train_years + test_years - 1]

        train_mask = (years >= unique_years[i]) & (years <= train_end_year)
        test_mask = (years >= test_start_year) & (years <= test_end_year)

        train_rets = returns[train_mask]
        test_rets = returns[test_mask]

        if len(train_rets) < 100 or len(test_rets) < 50:
            continue

        train_sharpe = train_rets.mean() / train_rets.std() * np.sqrt(252) if train_rets.std() > 0 else 0
        test_sharpe = test_rets.mean() / test_rets.std() * np.sqrt(252) if test_rets.std() > 0 else 0

        results.append({
            'train_period': f"{unique_years[i]}-{train_end_year}",
            'test_period': f"{test_start_year}-{test_end_year}",
            'train_sharpe': round(train_sharpe, 3),
            'test_sharpe': round(test_sharpe, 3),
            'degradation': round(1 - test_sharpe / train_sharpe, 3) if train_sharpe != 0 else np.nan
        })

    return results


# ─── Monthly Return Table ────────────────────────────────────────────────────

def monthly_returns_table(returns, name):
    """Generate monthly return table."""
    monthly = returns.resample('ME').apply(lambda x: (1+x).prod() - 1) * 100
    pivot = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    pivot['Annual'] = returns.resample('YE').apply(lambda x: (1+x).prod() - 1).values * 100
    return pivot


# ─── Overnight vs Intraday Decomposition ─────────────────────────────────────

def return_decomposition(data):
    """Show what fraction of total return comes from overnight vs intraday."""
    results = {}
    for ticker in ['SPY', 'UPRO', 'TQQQ', 'QQQ']:
        if ticker not in data:
            continue
        df = data[ticker]
        overnight, intraday, total = compute_returns(df)

        cum_overnight = (1 + overnight).prod() - 1
        cum_intraday = (1 + intraday).prod() - 1
        cum_total = (1 + total).prod() - 1

        results[ticker] = {
            'overnight_total_return': round(cum_overnight * 100, 1),
            'intraday_total_return': round(cum_intraday * 100, 1),
            'total_return': round(cum_total * 100, 1),
            'overnight_share': round(cum_overnight / cum_total * 100, 1) if cum_total != 0 else 0,
            'overnight_sharpe': round(overnight.mean() / overnight.std() * np.sqrt(252), 3) if overnight.std() > 0 else 0,
            'intraday_sharpe': round(intraday.mean() / intraday.std() * np.sqrt(252), 3) if intraday.std() > 0 else 0,
        }
    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("OVERNIGHT PREMIUM CAPTURE — v1")
    print("=" * 80)
    print()

    # 1. Download data
    print(">>> Downloading data...")
    data = download_data()
    print()

    # Show data ranges
    for t, df in data.items():
        print(f"  {t}: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
    print()

    # 2. Return decomposition
    print(">>> Return Decomposition (Overnight vs Intraday):")
    decomp = return_decomposition(data)
    for ticker, d in decomp.items():
        print(f"  {ticker}:")
        print(f"    Overnight total: {d['overnight_total_return']:.1f}%  |  Intraday total: {d['intraday_total_return']:.1f}%")
        print(f"    Overnight share of total return: {d['overnight_share']:.1f}%")
        print(f"    Overnight Sharpe: {d['overnight_sharpe']:.3f}  |  Intraday Sharpe: {d['intraday_sharpe']:.3f}")
    print()

    # 3. Run strategies
    print(">>> Running strategies...")
    spy_df = data['SPY']
    upro_df = data['UPRO']
    tqqq_df = data['TQQQ']
    vix_df = data['VIX']

    strategies = {}
    returns_dict = {}

    # Strategies
    strat_configs = [
        ('Overnight SPY', strategy_overnight_only, (spy_df,)),
        ('Overnight UPRO', strategy_overnight_upro, (upro_df,)),
        ('Day-Avoidance SPY', strategy_day_avoidance, (spy_df,)),
        ('VIX-Filtered Overnight', strategy_vix_filtered, (spy_df, vix_df)),
        ('v4.4+Overnight SPY', strategy_v44_overnight, (spy_df,)),
    ]
    for name, func, args in strat_configs:
        rets = func(*args)
        returns_dict[name] = rets
        strategies[name] = compute_metrics(rets, name)

    # Benchmarks
    spy_total = benchmark_buy_hold(spy_df)
    upro_total = benchmark_buy_hold(upro_df)
    returns_dict['SPY Buy-Hold'] = spy_total
    returns_dict['UPRO Buy-Hold'] = upro_total
    strategies['SPY Buy-Hold'] = compute_metrics(spy_total, 'SPY Buy-Hold')
    strategies['UPRO Buy-Hold'] = compute_metrics(upro_total, 'UPRO Buy-Hold')

    # Also add Overnight TQQQ
    tqqq_overnight = strategy_overnight_upro(tqqq_df)
    returns_dict['Overnight TQQQ'] = tqqq_overnight
    strategies['Overnight TQQQ'] = compute_metrics(tqqq_overnight, 'Overnight TQQQ')

    # Performance table
    perf_df = pd.DataFrame(strategies).T
    perf_df = perf_df[['Sharpe', 'Sortino', 'CAGR', 'MaxDD', 'Calmar', 'WinRate', 'PF', 'TotalReturn', 'NumTradingDays']]
    print("\n>>> PERFORMANCE SUMMARY:")
    print(perf_df.to_string())
    print()

    # 4. Equity curves
    print(">>> Plotting equity curves...")
    curves = {}
    for name, rets in returns_dict.items():
        curves[name] = equity_curve(rets)
    plot_equity_curves(curves)

    # 5. Annual returns heatmap
    print(">>> Plotting annual returns...")
    plot_annual_returns(returns_dict)

    # 6. Correlation analysis
    print(">>> Correlation with SPY Buy-Hold:")
    corrs = correlation_analysis(returns_dict, spy_total)
    for name, c in corrs.items():
        print(f"  {name}: {c}")
    print()

    # 7. Cost sensitivity
    print(">>> Cost Sensitivity Analysis...")
    cost_df = cost_sensitivity(spy_df, upro_df)
    cost_pivot = cost_df.pivot_table(index='Name', columns='Commission_RT', values='Sharpe')
    print(cost_pivot.to_string())
    print()

    # Also show CAGR sensitivity
    cost_cagr = cost_df.pivot_table(index='Name', columns='Commission_RT', values='CAGR')
    print("CAGR (%) by Commission:")
    print(cost_cagr.to_string())
    print()

    # 8. Adversarial tests
    print("=" * 80)
    print("ADVERSARIAL VALIDATION")
    print("=" * 80)

    adversarial_results = {}
    for name in ['Overnight SPY', 'Overnight UPRO', 'Day-Avoidance SPY', 'VIX-Filtered Overnight']:
        rets = returns_dict[name]
        print(f"\n--- {name} ---")

        # Permutation test
        perm = permutation_test(rets)
        print(f"  Permutation test: actual Sharpe={perm['actual_sharpe']}, "
              f"random mean={perm['perm_mean_sharpe']}+/-{perm['perm_std_sharpe']}, "
              f"p={perm['p_value']}, edge={perm['edge_vs_random']}")

        # Sub-period consistency
        subp = subperiod_consistency(rets)
        print(f"  Sub-period: Sharpes={subp['block_sharpes']}, CV={subp['cv']}, "
              f"consistent={subp['consistent']}, all_positive={subp['all_positive']}")

        # Walk-forward
        wf = walk_forward_test(rets)
        if wf:
            oos_sharpes = [w['test_sharpe'] for w in wf]
            pct_positive = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes) * 100
            avg_deg = np.nanmean([w['degradation'] for w in wf])
            print(f"  Walk-forward: {len(wf)} windows, {pct_positive:.0f}% positive OOS, "
                  f"avg degradation={avg_deg:.1%}")
            print(f"    OOS Sharpes: {[w['test_sharpe'] for w in wf]}")

        adversarial_results[name] = {
            'permutation': perm,
            'subperiod': subp,
            'walk_forward': wf
        }

    # 9. Monthly returns for best strategy
    print("\n>>> Monthly Returns — Overnight SPY:")
    monthly = monthly_returns_table(returns_dict['Overnight SPY'], 'Overnight SPY')
    print(monthly.to_string(float_format='%.1f'))
    print()

    print("\n>>> Monthly Returns — VIX-Filtered Overnight:")
    monthly_vix = monthly_returns_table(returns_dict['VIX-Filtered Overnight'], 'VIX-Filtered')
    print(monthly_vix.to_string(float_format='%.1f'))
    print()

    # 10. Save results
    print(">>> Saving results...")

    # Save performance table
    perf_df.to_csv(OUTPUT_DIR / 'performance_summary.csv')

    # Save cost sensitivity
    cost_df.to_csv(OUTPUT_DIR / 'cost_sensitivity.csv', index=False)

    # Save adversarial results
    # Convert to serializable format
    adv_serializable = {}
    for name, res in adversarial_results.items():
        adv_serializable[name] = {
            'permutation': res['permutation'],
            'subperiod': res['subperiod'],
            'walk_forward': res['walk_forward']
        }
    with open(OUTPUT_DIR / 'adversarial_results.json', 'w') as f:
        json.dump(adv_serializable, f, indent=2, default=str)

    # Save decomposition
    with open(OUTPUT_DIR / 'return_decomposition.json', 'w') as f:
        json.dump(decomp, f, indent=2)

    # Save correlations
    with open(OUTPUT_DIR / 'correlations.json', 'w') as f:
        json.dump(corrs, f, indent=2)

    # Save monthly returns
    monthly.to_csv(OUTPUT_DIR / 'monthly_returns_overnight_spy.csv')
    monthly_vix.to_csv(OUTPUT_DIR / 'monthly_returns_vix_filtered.csv')

    # 11. Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Check adversarial gates
    all_pass = True
    for name, res in adversarial_results.items():
        perm_ok = res['permutation']['p_value'] < 0.1
        subp_ok = res['subperiod']['consistent']
        wf_data = res['walk_forward']
        wf_ok = True
        if wf_data:
            oos_sharpes = [w['test_sharpe'] for w in wf_data]
            wf_ok = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes) > 0.5

        status = "PASS" if (perm_ok and subp_ok and wf_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
        reasons = []
        if not perm_ok:
            reasons.append(f"perm p={res['permutation']['p_value']}")
        if not subp_ok:
            reasons.append(f"CV={res['subperiod']['cv']}")
        if not wf_ok:
            reasons.append("WF<50% positive")
        reason_str = f" ({', '.join(reasons)})" if reasons else ""
        print(f"  {name}: {status}{reason_str}")

    print()
    if all_pass:
        print("  ALL strategies pass adversarial validation.")
    else:
        print("  Some strategies FAIL adversarial gates. See details above.")

    print(f"\n  Output saved to: {OUTPUT_DIR}")
    print("  Done.")

    return perf_df, adversarial_results


if __name__ == '__main__':
    perf_df, adv_results = main()
