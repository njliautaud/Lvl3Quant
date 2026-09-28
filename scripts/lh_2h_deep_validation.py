"""
Deep Validation Analysis for 2h ES LGBM Model
Runs 6 analyses on the walk-forward OOS predictions:
1. Hour-of-day breakdown
2. Confidence decile analysis
3. Consecutive loss analysis
4. Day-of-week effects
5. Volatility regime interaction
6. Walk-forward IC stability over time
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from pathlib import Path

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/lh_2h_deep_validation")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cost assumption (ES passive limit + commission)
COST_TICKS = 1.376
TICK_VALUE = 12.50

def load_predictions():
    df = pd.read_parquet("/home/jupiter/Lvl3Quant/output/lh_2h_intraday_clean/predictions.parquet")
    df['date_dt'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['dow'] = df['date_dt'].dt.dayofweek  # 0=Mon, 4=Fri
    df['dow_name'] = df['date_dt'].dt.day_name()
    # Direction: sign of prediction
    df['direction'] = np.sign(df['prediction'])
    # Gross ticks = actual * sign(prediction) if correct direction
    df['gross_ticks'] = df['actual'] * df['direction']
    df['net_ticks'] = df['gross_ticks'] - COST_TICKS
    df['win'] = (df['net_ticks'] > 0).astype(int)
    df['abs_pred'] = df['prediction'].abs()
    return df

def calc_metrics(subset):
    """Calculate standard metrics for a subset of trades."""
    if len(subset) == 0:
        return {'n': 0, 'ic': np.nan, 'wr': np.nan, 'avg_net_ticks': np.nan, 'sharpe': np.nan, 'pf': np.nan}

    ic = subset['prediction'].corr(subset['actual'], method='spearman')
    wr = subset['win'].mean()
    avg_net = subset['net_ticks'].mean()
    total_net = subset['net_ticks'].sum()

    # Daily aggregation for Sharpe
    daily = subset.groupby('date')['net_ticks'].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan

    # Profit factor
    wins = subset[subset['net_ticks'] > 0]['net_ticks'].sum()
    losses = abs(subset[subset['net_ticks'] < 0]['net_ticks'].sum())
    pf = wins / losses if losses > 0 else np.inf

    return {
        'n': len(subset),
        'ic': round(ic, 4),
        'wr': round(wr, 4),
        'avg_net_ticks': round(avg_net, 2),
        'total_net_ticks': round(total_net, 2),
        'sharpe': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'pf': round(pf, 2),
        'total_pnl_usd': round(total_net * TICK_VALUE, 2)
    }

def analysis_1_hour_of_day(df):
    """Which 2h prediction windows are strongest?"""
    print("\n=== ANALYSIS 1: Hour-of-Day ===")
    results = {}
    for hour in sorted(df['hour'].unique()):
        subset = df[df['hour'] == hour]
        m = calc_metrics(subset)
        results[str(hour)] = m
        print(f"  Hour {hour}:00 -> n={m['n']}, IC={m['ic']}, WR={m['wr']}, "
              f"Avg Net={m['avg_net_ticks']} ticks, Sharpe={m['sharpe']}, PF={m['pf']}")

    # Overall for reference
    results['overall'] = calc_metrics(df)
    return results

def analysis_2_confidence_deciles(df):
    """Break predictions into deciles by absolute confidence."""
    print("\n=== ANALYSIS 2: Confidence Deciles ===")
    df_sorted = df.copy()
    df_sorted['decile'] = pd.qcut(df_sorted['abs_pred'], 10, labels=False, duplicates='drop')

    results = {}
    for d in sorted(df_sorted['decile'].unique()):
        subset = df_sorted[df_sorted['decile'] == d]
        m = calc_metrics(subset)
        conf_range = f"{subset['abs_pred'].min():.1f}-{subset['abs_pred'].max():.1f}"
        results[f"D{d+1}"] = {**m, 'confidence_range': conf_range}
        print(f"  Decile {d+1} (|pred| {conf_range}): n={m['n']}, IC={m['ic']}, "
              f"WR={m['wr']}, Avg Net={m['avg_net_ticks']}, PF={m['pf']}")

    # Monotonicity test: correlation between decile and avg_net_ticks
    decile_nets = [results[f"D{d+1}"]['avg_net_ticks'] for d in range(len(results))]
    mono_corr = np.corrcoef(range(len(decile_nets)), decile_nets)[0,1]
    results['monotonicity_correlation'] = round(mono_corr, 4)
    print(f"  Monotonicity (decile vs avg_net): r={mono_corr:.4f}")

    # Recommended filter
    # Find first decile where avg_net_ticks > 0
    skip_deciles = []
    for d in range(len(decile_nets)):
        if decile_nets[d] < 20:  # marginal
            skip_deciles.append(d+1)
    results['recommendation'] = f"Skip deciles {skip_deciles} (low confidence)" if skip_deciles else "Trade all deciles"

    return results

def analysis_3_consecutive_losses(df):
    """Longest drawdown streaks."""
    print("\n=== ANALYSIS 3: Consecutive Loss Analysis ===")

    # Trade-level streaks
    losses = df['win'].values
    max_loss_streak = 0
    current_streak = 0
    streaks = []

    for w in losses:
        if w == 0:  # loss
            current_streak += 1
            max_loss_streak = max(max_loss_streak, current_streak)
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)

    # Daily-level streaks (losing days)
    daily_pnl = df.groupby('date')['net_ticks'].sum()
    daily_losses = (daily_pnl < 0).astype(int).values
    max_day_streak = 0
    current_day_streak = 0
    day_streaks = []

    for d in daily_losses:
        if d == 1:
            current_day_streak += 1
            max_day_streak = max(max_day_streak, current_day_streak)
        else:
            if current_day_streak > 0:
                day_streaks.append(current_day_streak)
            current_day_streak = 0
    if current_day_streak > 0:
        day_streaks.append(current_day_streak)

    # Drawdown analysis
    cumulative = df['net_ticks'].cumsum().values
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_dd_ticks = drawdown.min()
    max_dd_idx = np.argmin(drawdown)

    # Recovery: how many trades from max DD to new peak
    recovery_trades = 0
    for i in range(max_dd_idx, len(cumulative)):
        if cumulative[i] >= peak[max_dd_idx]:
            recovery_trades = i - max_dd_idx
            break

    results = {
        'max_consecutive_losing_trades': int(max_loss_streak),
        'avg_losing_streak': round(np.mean(streaks), 2) if streaks else 0,
        'median_losing_streak': int(np.median(streaks)) if streaks else 0,
        'num_losing_streaks': len(streaks),
        'max_consecutive_losing_days': int(max_day_streak),
        'avg_losing_day_streak': round(np.mean(day_streaks), 2) if day_streaks else 0,
        'num_losing_day_streaks': len(day_streaks),
        'max_drawdown_ticks': round(float(max_dd_ticks), 2),
        'max_drawdown_usd': round(float(max_dd_ticks) * TICK_VALUE, 2),
        'trades_to_recover_from_max_dd': int(recovery_trades),
        'pct_losing_days': round((daily_pnl < 0).mean() * 100, 1),
        'n_losing_days': int((daily_pnl < 0).sum()),
        'n_total_days': len(daily_pnl),
    }

    print(f"  Max consecutive losing trades: {results['max_consecutive_losing_trades']}")
    print(f"  Max consecutive losing days: {results['max_consecutive_losing_days']}")
    print(f"  Max drawdown: {results['max_drawdown_ticks']:.0f} ticks (${results['max_drawdown_usd']:,.0f})")
    print(f"  Trades to recover from max DD: {results['trades_to_recover_from_max_dd']}")
    print(f"  Losing days: {results['n_losing_days']}/{results['n_total_days']} ({results['pct_losing_days']}%)")

    return results

def analysis_4_day_of_week(df):
    """Day-of-week performance differences."""
    print("\n=== ANALYSIS 4: Day-of-Week ===")
    results = {}

    for dow in range(5):
        subset = df[df['dow'] == dow]
        if len(subset) == 0:
            continue
        m = calc_metrics(subset)
        day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][dow]
        results[day_name] = m
        print(f"  {day_name}: n={m['n']}, IC={m['ic']}, WR={m['wr']}, "
              f"Avg Net={m['avg_net_ticks']}, Sharpe={m['sharpe']}, PF={m['pf']}")

    # ANOVA test for significance
    dow_groups = [df[df['dow'] == d]['net_ticks'].values for d in range(5) if len(df[df['dow'] == d]) > 0]
    f_stat, p_val = stats.f_oneway(*dow_groups)
    results['anova_f_stat'] = round(float(f_stat), 4)
    results['anova_p_value'] = round(float(p_val), 4)
    results['significant_difference'] = p_val < 0.05
    print(f"  ANOVA F={f_stat:.4f}, p={p_val:.4f} {'(SIGNIFICANT)' if p_val < 0.05 else '(not significant)'}")

    return results

def analysis_5_volatility_regime(df):
    """Performance in high-vol vs low-vol environments."""
    print("\n=== ANALYSIS 5: Volatility Regime ===")

    # Use daily absolute actual move as volatility proxy
    daily_vol = df.groupby('date')['actual'].apply(lambda x: x.abs().mean()).reset_index()
    daily_vol.columns = ['date', 'daily_vol']

    # Split into terciles
    vol_thresholds = daily_vol['daily_vol'].quantile([0.33, 0.67])

    df_merged = df.merge(daily_vol, on='date')

    results = {}

    # Low vol
    low_vol = df_merged[df_merged['daily_vol'] <= vol_thresholds[0.33]]
    m = calc_metrics(low_vol)
    results['low_vol'] = {**m, 'vol_range': f"0-{vol_thresholds[0.33]:.1f} avg_abs_ticks"}
    print(f"  Low Vol (<=p33): n={m['n']}, IC={m['ic']}, WR={m['wr']}, Avg Net={m['avg_net_ticks']}, PF={m['pf']}")

    # Med vol
    med_vol = df_merged[(df_merged['daily_vol'] > vol_thresholds[0.33]) &
                        (df_merged['daily_vol'] <= vol_thresholds[0.67])]
    m = calc_metrics(med_vol)
    results['med_vol'] = {**m, 'vol_range': f"{vol_thresholds[0.33]:.1f}-{vol_thresholds[0.67]:.1f} avg_abs_ticks"}
    print(f"  Med Vol (p33-p67): n={m['n']}, IC={m['ic']}, WR={m['wr']}, Avg Net={m['avg_net_ticks']}, PF={m['pf']}")

    # High vol
    high_vol = df_merged[df_merged['daily_vol'] > vol_thresholds[0.67]]
    m = calc_metrics(high_vol)
    results['high_vol'] = {**m, 'vol_range': f">{vol_thresholds[0.67]:.1f} avg_abs_ticks"}
    print(f"  High Vol (>p67): n={m['n']}, IC={m['ic']}, WR={m['wr']}, Avg Net={m['avg_net_ticks']}, PF={m['pf']}")

    # Correlation between daily vol and daily PnL
    daily_pnl = df.groupby('date')['net_ticks'].sum().reset_index()
    daily_pnl.columns = ['date', 'daily_pnl']
    merged = daily_pnl.merge(daily_vol, on='date')
    vol_pnl_corr = merged['daily_vol'].corr(merged['daily_pnl'])
    results['vol_pnl_correlation'] = round(vol_pnl_corr, 4)
    print(f"  Vol-PnL correlation: {vol_pnl_corr:.4f}")

    return results

def analysis_6_wf_stability(df):
    """Walk-forward IC stability over time."""
    print("\n=== ANALYSIS 6: Walk-Forward Stability ===")

    # Per-day IC
    daily_ic = df.groupby('date').apply(
        lambda x: x['prediction'].corr(x['actual'], method='spearman')
    ).reset_index()
    daily_ic.columns = ['date', 'ic']
    daily_ic['date_dt'] = pd.to_datetime(daily_ic['date'], format='%Y%m%d')
    daily_ic = daily_ic.sort_values('date_dt').reset_index(drop=True)

    # Rolling 20-day IC
    daily_ic['rolling_20d_ic'] = daily_ic['ic'].rolling(20, min_periods=10).mean()

    # Trend test
    x = np.arange(len(daily_ic))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, daily_ic['ic'].values)

    # Split into quarters
    n = len(daily_ic)
    q_size = n // 4
    quarters = {
        'Q1_oldest': daily_ic.iloc[:q_size],
        'Q2': daily_ic.iloc[q_size:2*q_size],
        'Q3': daily_ic.iloc[2*q_size:3*q_size],
        'Q4_newest': daily_ic.iloc[3*q_size:]
    }

    results = {
        'trend': {
            'ic_slope_per_day': round(slope, 6),
            'p_value': round(p_value, 4),
            'r_squared': round(r_value**2, 4),
            'interpretation': 'STABLE (no significant trend)' if p_value > 0.05 else
                            ('DEGRADING' if slope < 0 else 'IMPROVING')
        },
        'quarters': {}
    }

    for qname, qdata in quarters.items():
        q_ic = qdata['ic'].mean()
        q_wr = None  # would need trade data
        results['quarters'][qname] = {
            'dates': f"{qdata['date'].iloc[0]} to {qdata['date'].iloc[-1]}",
            'mean_ic': round(q_ic, 4),
            'median_ic': round(qdata['ic'].median(), 4),
            'pct_positive': round((qdata['ic'] > 0).mean() * 100, 1),
            'min_ic': round(qdata['ic'].min(), 4),
            'max_ic': round(qdata['ic'].max(), 4)
        }
        print(f"  {qname}: IC mean={q_ic:.4f}, median={qdata['ic'].median():.4f}, "
              f"pos%={((qdata['ic']>0).mean()*100):.0f}%")

    print(f"  Trend: slope={slope:.6f}/day, p={p_value:.4f} -> {results['trend']['interpretation']}")

    # Worst/best 10 days
    worst_10 = daily_ic.nsmallest(10, 'ic')[['date', 'ic']].values.tolist()
    best_10 = daily_ic.nlargest(10, 'ic')[['date', 'ic']].values.tolist()
    results['worst_10_days'] = [{'date': d, 'ic': round(ic, 4)} for d, ic in worst_10]
    results['best_10_days'] = [{'date': d, 'ic': round(ic, 4)} for d, ic in best_10]

    return results

def main():
    print("Loading predictions...")
    df = load_predictions()
    print(f"Loaded {len(df)} predictions across {df['date'].nunique()} days")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Hours: {sorted(df['hour'].unique())}")

    all_results = {}

    # Run all analyses
    all_results['hour_of_day'] = analysis_1_hour_of_day(df)
    all_results['confidence_deciles'] = analysis_2_confidence_deciles(df)
    all_results['consecutive_losses'] = analysis_3_consecutive_losses(df)
    all_results['day_of_week'] = analysis_4_day_of_week(df)
    all_results['volatility_regime'] = analysis_5_volatility_regime(df)
    all_results['wf_stability'] = analysis_6_wf_stability(df)

    # Summary / actionable conclusions
    all_results['summary'] = {
        'total_trades': len(df),
        'total_days': int(df['date'].nunique()),
        'overall_ic': round(df['prediction'].corr(df['actual'], method='spearman'), 4),
        'overall_wr': round(df['win'].mean(), 4),
        'overall_sharpe': all_results['hour_of_day']['overall']['sharpe'],
        'overall_pf': all_results['hour_of_day']['overall']['pf'],
        'cost_assumption': COST_TICKS,
        'conclusions': []
    }

    # Build conclusions
    conclusions = all_results['summary']['conclusions']

    # Best hour
    hour_data = {k: v for k, v in all_results['hour_of_day'].items() if k != 'overall'}
    best_hour = max(hour_data, key=lambda h: hour_data[h].get('avg_net_ticks', -999))
    worst_hour = min(hour_data, key=lambda h: hour_data[h].get('avg_net_ticks', 999))
    conclusions.append(f"Best hour: {best_hour}:00 ({hour_data[best_hour]['avg_net_ticks']} avg net ticks), "
                      f"Worst: {worst_hour}:00 ({hour_data[worst_hour]['avg_net_ticks']} avg net ticks)")

    # Confidence monotonicity
    mono = all_results['confidence_deciles']['monotonicity_correlation']
    conclusions.append(f"Confidence-PnL monotonicity: r={mono:.3f} {'(STRONG)' if mono > 0.7 else '(moderate)' if mono > 0.4 else '(WEAK - investigate)'}")

    # Max DD
    max_dd = all_results['consecutive_losses']['max_drawdown_ticks']
    conclusions.append(f"Max drawdown: {max_dd:.0f} ticks (${max_dd*TICK_VALUE:,.0f}), "
                      f"max {all_results['consecutive_losses']['max_consecutive_losing_trades']} consecutive losses")

    # DOW effect
    if all_results['day_of_week']['significant_difference']:
        conclusions.append("DAY-OF-WEEK EFFECT SIGNIFICANT - some days materially better")
    else:
        conclusions.append("No significant day-of-week effect (good - edge is consistent)")

    # Vol regime
    conclusions.append(f"Vol-PnL correlation: {all_results['volatility_regime']['vol_pnl_correlation']:.3f}")

    # WF stability
    conclusions.append(f"IC trend: {all_results['wf_stability']['trend']['interpretation']}")

    # Save
    output_file = OUTPUT_DIR / "deep_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\n{'='*60}")
    print("ACTIONABLE CONCLUSIONS:")
    print('='*60)
    for c in conclusions:
        print(f"  -> {c}")
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
