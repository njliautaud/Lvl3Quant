"""
Deployment Simulator — Phase 1 ($500 Starting Capital)
========================================================
Simulates ACTUAL week-by-week execution of our validated portfolio
from today, using real historical patterns. Shows:
1. What trades happen each week
2. When protection signals fire
3. Projected growth with confidence intervals
4. Probability of hitting capital milestones
5. Expected income at each phase

HC #709: Growth + protection portfolio
HC #711: Focus on actionable deployment
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
OUTPUT_DIR = BASE_DIR / "output" / "growth_research" / "deployment_sim"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_CAPITAL = 500.0
WEEKLY_CONTRIBUTION = 100.0  # DCA amount
SIMULATION_YEARS = 5
N_SIMULATIONS = 10000


def download_data():
    tickers = ["UPRO", "SPY", "IWM", "GLD", "SLV", "USO", "UNG", "DBA",
               "COPX", "UUP", "TLT", "EEM", "HYG", "LQD", "VIXY"]
    print("Downloading data...")
    raw = yf.download(tickers, start="2013-01-01", end="2026-07-16",
                     progress=False, auto_adjust=True, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw['Close']
    else:
        closes = raw
    return closes.dropna(how='all').ffill()


def compute_weekly_returns(closes):
    """Compute weekly (Friday-to-Friday) returns for UPRO protected strategy."""
    upro = closes['UPRO']
    spy = closes['SPY']

    # Protection signals (daily)
    spy_sma50 = spy.rolling(50).mean()
    hyg_lqd = (closes['HYG'] / closes['LQD']).pct_change(21)
    iwm_sma50 = closes['IWM'].rolling(50).mean()
    vixy = closes.get('VIXY')
    vixy_sma = vixy.rolling(20).mean() if vixy is not None else None

    # Daily exposure
    signal_count = (
        (spy > spy_sma50).astype(int) +
        (hyg_lqd > -0.01).astype(int) +
        (closes['IWM'] > iwm_sma50).astype(int) +
        ((vixy < vixy_sma * 1.2).astype(int) if vixy_sma is not None else 1)
    )
    daily_exposure = pd.Series(0.0, index=closes.index)
    daily_exposure[signal_count >= 3] = 1.0
    daily_exposure[signal_count == 2] = 0.5
    daily_exposure = daily_exposure.shift(1).fillna(0)

    # Daily protected returns
    daily_ret = upro.pct_change() * daily_exposure

    # Resample to weekly
    weekly_ret = (1 + daily_ret).resample('W-FRI').prod() - 1
    weekly_ret = weekly_ret.dropna()

    # Also compute protection status (fraction of week in protection)
    weekly_exposure = daily_exposure.resample('W-FRI').mean()

    # CTA trend weekly returns
    cta_tickers = ["GLD", "SLV", "USO", "UNG", "DBA", "COPX", "UUP", "TLT", "EEM"]
    cta_daily = []
    for t in [t for t in cta_tickers if t in closes.columns]:
        px = closes[t]
        sig = (px > px.rolling(50).mean()).astype(float).shift(1)
        cta_daily.append(px.pct_change() * sig)
    cta_daily_ret = pd.concat(cta_daily, axis=1).mean(axis=1)
    weekly_cta = (1 + cta_daily_ret).resample('W-FRI').prod() - 1

    return weekly_ret, weekly_cta, weekly_exposure


def monte_carlo_simulation(weekly_upro, weekly_cta, n_sims=10000, years=5,
                           start_capital=500, weekly_contrib=100):
    """
    Monte Carlo: bootstrap weekly returns to simulate future paths.
    Phase 1: 100% UPRO protected (account < $2K)
    Phase 2: 70% UPRO + 30% CTA (account $2K-$10K)
    Phase 3: Risk parity weights ($10K+)
    """
    n_weeks = int(years * 52)

    # Clean returns
    upro_rets = weekly_upro.dropna().values
    cta_rets = weekly_cta.dropna().values

    # Align lengths
    min_len = min(len(upro_rets), len(cta_rets))
    upro_rets = upro_rets[:min_len]
    cta_rets = cta_rets[:min_len]

    # Storage
    all_paths = np.zeros((n_sims, n_weeks))
    milestones = {1000: [], 2000: [], 5000: [], 10000: [], 25000: [], 50000: [], 100000: []}
    max_drawdowns = []
    final_values = []
    phase_transitions = {2: [], 3: []}  # Week when entering phase 2/3

    for sim in range(n_sims):
        capital = start_capital
        peak = capital
        max_dd = 0
        current_phase = 1

        for week in range(n_weeks):
            # Add weekly contribution
            capital += weekly_contrib

            # Phase-based allocation
            if capital >= 10000 and current_phase < 3:
                current_phase = 3
                phase_transitions[3].append(week)
            elif capital >= 2000 and current_phase < 2:
                current_phase = 2
                phase_transitions[2].append(week)

            # Sample random week (block bootstrap — sample 4 consecutive weeks)
            idx = np.random.randint(0, len(upro_rets) - 4)

            if current_phase == 1:
                # 100% UPRO protected
                ret = upro_rets[idx]
            elif current_phase == 2:
                # 70% UPRO + 30% CTA
                ret = 0.70 * upro_rets[idx] + 0.30 * cta_rets[idx]
            else:
                # Risk parity (roughly 40% UPRO, 40% CTA, 20% cash buffer)
                ret = 0.40 * upro_rets[idx] + 0.40 * cta_rets[idx]

            capital *= (1 + ret)
            capital = max(capital, 0)  # Can't go negative

            # Track drawdown
            peak = max(peak, capital)
            dd = (capital - peak) / peak if peak > 0 else 0
            max_dd = min(max_dd, dd)

            all_paths[sim, week] = capital

        # Record final stats
        final_values.append(capital)
        max_drawdowns.append(max_dd)

        # Record milestone timing
        for milestone in milestones:
            hit_weeks = np.where(all_paths[sim] >= milestone)[0]
            if len(hit_weeks) > 0:
                milestones[milestone].append(hit_weeks[0])

    return all_paths, final_values, max_drawdowns, milestones, phase_transitions


def analyze_results(all_paths, final_values, max_drawdowns, milestones,
                    phase_transitions, n_weeks, weekly_contrib):
    """Analyze Monte Carlo results."""
    results = {}

    # Final value distribution
    final_arr = np.array(final_values)
    results['final_value'] = {
        'median': round(float(np.median(final_arr)), 0),
        'mean': round(float(np.mean(final_arr)), 0),
        'p10': round(float(np.percentile(final_arr, 10)), 0),
        'p25': round(float(np.percentile(final_arr, 25)), 0),
        'p75': round(float(np.percentile(final_arr, 75)), 0),
        'p90': round(float(np.percentile(final_arr, 90)), 0),
        'total_contributed': round(START_CAPITAL + weekly_contrib * n_weeks, 0),
    }

    # CAGR distribution
    years = n_weeks / 52
    total_invested = START_CAPITAL + weekly_contrib * n_weeks
    # Simple CAGR approximation
    cagr_arr = (final_arr / total_invested) ** (1/years) - 1
    results['cagr'] = {
        'median': round(float(np.median(cagr_arr) * 100), 1),
        'p10': round(float(np.percentile(cagr_arr, 10) * 100), 1),
        'p90': round(float(np.percentile(cagr_arr, 90) * 100), 1),
    }

    # Max drawdown distribution
    dd_arr = np.array(max_drawdowns)
    results['max_drawdown'] = {
        'median': round(float(np.median(dd_arr) * 100), 1),
        'p5_worst': round(float(np.percentile(dd_arr, 5) * 100), 1),
        'p25': round(float(np.percentile(dd_arr, 25) * 100), 1),
    }

    # Milestone probabilities and timing
    results['milestones'] = {}
    for milestone, weeks_list in milestones.items():
        n_hit = len(weeks_list)
        prob = n_hit / len(final_values)
        if n_hit > 0:
            median_weeks = int(np.median(weeks_list))
            median_months = round(median_weeks / 4.33, 1)
        else:
            median_weeks = None
            median_months = None

        results['milestones'][f'${milestone:,}'] = {
            'probability': round(prob * 100, 1),
            'median_weeks': median_weeks,
            'median_months': median_months,
        }

    # Phase transitions
    for phase, weeks_list in phase_transitions.items():
        if weeks_list:
            results[f'phase_{phase}_entry'] = {
                'median_weeks': int(np.median(weeks_list)),
                'median_months': round(np.median(weeks_list) / 4.33, 1),
            }

    # Probability of loss (final < total invested)
    total_invested = START_CAPITAL + weekly_contrib * n_weeks
    prob_loss = np.mean(final_arr < total_invested) * 100
    results['prob_loss'] = round(prob_loss, 1)

    # Ruin probability (account drops below $100 at any point)
    ruin_count = sum(1 for path in all_paths if np.min(path) < 100)
    results['prob_ruin'] = round(ruin_count / len(all_paths) * 100, 2)

    # Equity curve percentiles (for plotting)
    weeks = np.arange(n_weeks)
    results['equity_percentiles'] = {
        'weeks': weeks.tolist(),
        'p10': np.percentile(all_paths, 10, axis=0).tolist(),
        'p25': np.percentile(all_paths, 25, axis=0).tolist(),
        'p50': np.percentile(all_paths, 50, axis=0).tolist(),
        'p75': np.percentile(all_paths, 75, axis=0).tolist(),
        'p90': np.percentile(all_paths, 90, axis=0).tolist(),
    }

    return results


def main():
    print("=" * 70)
    print("DEPLOYMENT SIMULATOR — Phase 1 Starting $500 + $100/week")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    closes = download_data()

    print("\nComputing weekly strategy returns...")
    weekly_upro, weekly_cta, weekly_exposure = compute_weekly_returns(closes)
    print(f"  UPRO protected: {len(weekly_upro)} weeks, mean={weekly_upro.mean()*100:.2f}%/wk")
    print(f"  CTA trend: {len(weekly_cta)} weeks, mean={weekly_cta.mean()*100:.2f}%/wk")
    print(f"  Avg weekly exposure: {weekly_exposure.mean():.1%}")

    # Historical weekly stats
    print(f"\n  UPRO protected weekly stats:")
    print(f"    Best week:  {weekly_upro.max()*100:+.1f}%")
    print(f"    Worst week: {weekly_upro.min()*100:+.1f}%")
    print(f"    Win rate:   {(weekly_upro > 0).mean():.1%}")
    print(f"    Sharpe (ann): {weekly_upro.mean()/weekly_upro.std()*np.sqrt(52):.2f}")

    # Monte Carlo
    print(f"\n{'='*70}")
    print(f"MONTE CARLO SIMULATION ({N_SIMULATIONS:,} paths, {SIMULATION_YEARS} years)")
    print(f"Starting: ${START_CAPITAL:.0f} + ${WEEKLY_CONTRIBUTION:.0f}/week")
    print(f"{'='*70}")

    n_weeks = int(SIMULATION_YEARS * 52)
    all_paths, final_values, max_drawdowns, milestones, phase_transitions = \
        monte_carlo_simulation(weekly_upro, weekly_cta,
                              n_sims=N_SIMULATIONS, years=SIMULATION_YEARS,
                              start_capital=START_CAPITAL,
                              weekly_contrib=WEEKLY_CONTRIBUTION)

    results = analyze_results(all_paths, final_values, max_drawdowns, milestones,
                             phase_transitions, n_weeks, WEEKLY_CONTRIBUTION)

    # Print results
    total_contributed = START_CAPITAL + WEEKLY_CONTRIBUTION * n_weeks
    print(f"\n  Total contributed over {SIMULATION_YEARS} years: ${total_contributed:,.0f}")
    print(f"  (${START_CAPITAL:.0f} initial + ${WEEKLY_CONTRIBUTION:.0f}/week × {n_weeks} weeks)")

    print(f"\n  FINAL PORTFOLIO VALUE ({SIMULATION_YEARS} years):")
    fv = results['final_value']
    print(f"    10th percentile (bad luck):  ${fv['p10']:>10,.0f}")
    print(f"    25th percentile:             ${fv['p25']:>10,.0f}")
    print(f"    MEDIAN:                      ${fv['p50' if 'p50' in fv else 'median']:>10,.0f}")
    print(f"    75th percentile:             ${fv['p75']:>10,.0f}")
    print(f"    90th percentile (good luck): ${fv['p90']:>10,.0f}")

    print(f"\n  RISK METRICS:")
    dd = results['max_drawdown']
    print(f"    Median max drawdown:         {dd['median']:>8.1f}%")
    print(f"    Worst 5% max drawdown:       {dd['p5_worst']:>8.1f}%")
    print(f"    Probability of net loss:     {results['prob_loss']:>8.1f}%")
    print(f"    Probability of ruin (<$100): {results['prob_ruin']:>8.2f}%")

    print(f"\n  MILESTONE PROBABILITIES:")
    print(f"    {'Milestone':>12s}  {'Probability':>12s}  {'Median Time':>15s}")
    print(f"    {'-'*12}  {'-'*12}  {'-'*15}")
    for milestone, stats in results['milestones'].items():
        time_str = f"{stats['median_months']:.0f} months" if stats['median_months'] else "N/A"
        print(f"    {milestone:>12s}  {stats['probability']:>10.1f}%  {time_str:>15s}")

    # Phase transitions
    if 'phase_2_entry' in results:
        p2 = results['phase_2_entry']
        print(f"\n  Phase 2 ($2K, add CTA): median {p2['median_months']:.0f} months")
    if 'phase_3_entry' in results:
        p3 = results['phase_3_entry']
        print(f"  Phase 3 ($10K, risk parity): median {p3['median_months']:.0f} months")

    # Income projections at different sizes
    print(f"\n  MONTHLY INCOME PROJECTIONS (at 1% monthly withdrawal rate):")
    for size in [5000, 10000, 25000, 50000, 100000]:
        income = size * 0.01
        prob = results['milestones'].get(f'${size:,}', {}).get('probability', 0)
        if prob > 0:
            print(f"    ${size:>7,}: ${income:>6,.0f}/mo  ({prob:.0f}% chance of reaching in {SIMULATION_YEARS}yr)")

    # Year-by-year median progression
    print(f"\n  YEAR-BY-YEAR MEDIAN PORTFOLIO VALUE:")
    for year in range(1, SIMULATION_YEARS + 1):
        week_idx = min(year * 52 - 1, n_weeks - 1)
        median_val = np.median(all_paths[:, week_idx])
        p10_val = np.percentile(all_paths[:, week_idx], 10)
        p90_val = np.percentile(all_paths[:, week_idx], 90)
        contributed = START_CAPITAL + WEEKLY_CONTRIBUTION * (year * 52)
        gain = median_val - contributed
        print(f"    Year {year}: ${median_val:>10,.0f} (range ${p10_val:,.0f}-${p90_val:,.0f}) "
              f"[contributed ${contributed:,.0f}, gain ${gain:+,.0f}]")

    # Save results
    # Remove numpy arrays from results for JSON serialization
    save_results = {k: v for k, v in results.items() if k != 'equity_percentiles'}
    save_results['config'] = {
        'start_capital': START_CAPITAL,
        'weekly_contribution': WEEKLY_CONTRIBUTION,
        'simulation_years': SIMULATION_YEARS,
        'n_simulations': N_SIMULATIONS,
        'run_date': datetime.now().isoformat(),
    }

    out_file = OUTPUT_DIR / "deployment_sim_results.json"
    with open(out_file, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Saved to {out_file}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
