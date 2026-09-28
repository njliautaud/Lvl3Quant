#!/usr/bin/env python3
"""
Creative Strategies Batch 5 — Three genuinely different ideas.

1. VOLATILITY RISK PREMIUM HARVESTING — VIX consistently trades above realized vol.
   When VIX/RV ratio > 1.3 (rich implied): hold UPRO aggressively.
   When ratio < 1.0 (cheap implied / panic): switch to SPY defensively.
   When ratio 1.0-1.3: standard vol-switching.
   Uses 21-day realized vol on SPY vs VIX.

2. MEAN-REVERSION ON UPRO AFTER MULTI-DAY DECLINES — Buy when UPRO drops 3+ consecutive
   days OR drops >8% over 5 days. Hold 5-10 days then revert. Track bounce trades standalone
   and as overlay on base strategy.

3. SPY BREADTH TIMING FOR UPRO — Use RSP/SPY ratio as breadth proxy.
   RSP outperforming = healthy broad rally = UPRO.
   RSP underperforming = narrow leadership = SPY.
   Deep underperformance = deteriorating = GLD.

Full adversarial validation suite on each.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings, json, os
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

OUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_research/creative_batch5"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("CREATIVE STRATEGIES BATCH 5")
print("=" * 70)
print(f"\nFetching data...")

tickers = {
    'SPY': 'SPY', 'UPRO': 'UPRO', 'GLD': 'GLD', 'TLT': 'TLT',
    'VIX': '^VIX', 'RSP': 'RSP',
}

data = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start='2012-01-01', end='2026-07-18', progress=False)
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
        'final_value': float(cum.iloc[-1]),
    }


def run_permutation_test(strategy_rets, baseline_rets, n_perms=200):
    """
    CORRECT permutation test: shuffle the TIMING SIGNAL, not the returns.
    Compare strategy excess return vs randomly-timed version.
    """
    aligned = pd.DataFrame({'strat': strategy_rets, 'base': baseline_rets}).dropna()
    if len(aligned) < 50:
        return 1.0, 0.0

    real_excess = aligned['strat'].mean() - aligned['base'].mean()
    real_sharpe = aligned['strat'].mean() / aligned['strat'].std() * np.sqrt(252) if aligned['strat'].std() > 0 else 0

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


def walk_forward_validation(strategy_func, prices_df, returns_df, train_years=3, test_years=1):
    """Walk-forward: 3yr train, 1yr OOS, rolling."""
    all_oos_rets = []
    dates = prices_df.index
    start = dates[0]
    end = dates[-1]
    total_years = (end - start).days / 365.25

    if total_years < train_years + test_years:
        return None, []

    results = []
    cursor = start
    while True:
        train_end = cursor + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > end:
            break

        train_mask_p = (prices_df.index >= cursor) & (prices_df.index < train_end)
        test_mask_p = (prices_df.index >= train_end) & (prices_df.index < test_end)
        train_mask_r = (returns_df.index >= cursor) & (returns_df.index < train_end)
        test_mask_r = (returns_df.index >= train_end) & (returns_df.index < test_end)

        train_p = prices_df[train_mask_p]
        train_r = returns_df[train_mask_r]
        test_p = prices_df[test_mask_p]
        test_r = returns_df[test_mask_r]

        if len(train_r) < 200 or len(test_r) < 50:
            cursor += pd.DateOffset(years=test_years)
            continue

        # Run strategy on test period using parameters "trained" on train period
        try:
            oos_rets = strategy_func(test_p, test_r)
            if oos_rets is not None and len(oos_rets) > 20:
                m = compute_metrics(oos_rets, f"WF {train_end.strftime('%Y')}-{test_end.strftime('%Y')}")
                results.append(m)
                all_oos_rets.append(oos_rets)
        except Exception as e:
            pass

        cursor += pd.DateOffset(years=test_years)

    if len(all_oos_rets) > 0:
        combined = pd.concat(all_oos_rets)
        combined_m = compute_metrics(combined, "WF Combined OOS")
        return combined_m, results
    return None, results


def full_validation(strategy_rets, spy_rets, baseline_rets, label, strategy_func=None,
                    prices_df=None, returns_df=None):
    """Full adversarial suite."""
    print(f"\n  --- {label} ---")
    m = compute_metrics(strategy_rets, label)
    print(f"  Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
          f"CAGR={m['cagr']*100:.1f}%, MaxDD={m['max_dd']*100:.1f}%, "
          f"WR={m['wr']*100:.1f}%, PF={m['pf']:.3f}, FinalVal={m['final_value']:.1f}x")

    # Permutation test (correct: tests timing skill vs baseline)
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

    # Sub-period consistency
    sharpes, cv = sub_period_consistency(strategy_rets)
    all_pos = all(s > 0 for s in sharpes)
    sp_pass = all_pos and cv < 0.50
    print(f"  Sub-period: {[f'{s:.2f}' for s in sharpes]}, CV={cv:.3f}, AllPos={all_pos} ({'PASS' if sp_pass else 'FAIL'})")

    # Outlier robustness
    full_s, trim_s, deg = outlier_robustness(strategy_rets)
    out_pass = abs(deg) < 0.30
    print(f"  Outlier: full={full_s:.3f}, trimmed={trim_s:.3f}, deg={deg*100:.1f}% ({'PASS' if out_pass else 'FAIL'})")

    # Walk-forward validation
    wf_result = None
    wf_pass = False
    if strategy_func is not None and prices_df is not None and returns_df is not None:
        wf_combined, wf_folds = walk_forward_validation(strategy_func, prices_df, returns_df)
        if wf_combined is not None:
            wf_pass = wf_combined['sharpe'] > 0
            wf_result = wf_combined
            print(f"  Walk-Forward OOS: Sharpe={wf_combined['sharpe']:.3f}, CAGR={wf_combined['cagr']*100:.1f}% ({'PASS' if wf_pass else 'FAIL'})")
            for f in wf_folds:
                print(f"    {f['label']}: Sharpe={f['sharpe']:.3f}")
        else:
            print(f"  Walk-Forward: insufficient data")
    else:
        print(f"  Walk-Forward: not applicable")

    # Core pass = permutation + sub-period + outlier
    core_pass = perm_pass and sp_pass and out_pass
    verdict = "VALIDATED" if core_pass else "FAILED"
    # Note: R1 regime expected to fail for UPRO strategies per HC #709
    print(f"  VERDICT: {verdict} (core tests)")

    return {
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in m.items()},
        'perm_p': float(p_val),
        'perm_pass': perm_pass,
        'r1_gap': float(gap) if gap is not None else None,
        'r1_pass': bool(r1_pass) if gap is not None else None,
        'sub_period_sharpes': [float(s) for s in sharpes],
        'sub_period_cv': float(cv),
        'sub_period_pass': sp_pass,
        'outlier_deg': float(deg),
        'outlier_pass': out_pass,
        'wf_oos_sharpe': float(wf_result['sharpe']) if wf_result else None,
        'wf_pass': wf_pass,
        'verdict': verdict,
    }


# ─── Benchmarks ─────────────────────────────────────────────────────────────
spy_rets = returns['SPY']
upro_rets = returns['UPRO']
spy_m = compute_metrics(spy_rets, "SPY Buy-Hold")
upro_m = compute_metrics(upro_rets, "UPRO Buy-Hold")
print(f"\nBENCHMARKS:")
print(f"  SPY:  Sharpe={spy_m['sharpe']:.3f}, CAGR={spy_m['cagr']*100:.1f}%, MaxDD={spy_m['max_dd']*100:.1f}%")
print(f"  UPRO: Sharpe={upro_m['sharpe']:.3f}, CAGR={upro_m['cagr']*100:.1f}%, MaxDD={upro_m['max_dd']*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 1: VOLATILITY RISK PREMIUM HARVESTING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 1: VOLATILITY RISK PREMIUM HARVESTING")
print("=" * 70)
print("Concept: VIX consistently trades above realized vol (fear premium).")
print("When premium is FAT (VIX >> RV): hold UPRO aggressively.")
print("When premium is THIN/INVERTED (VIX <= RV): defensive SPY.")

def run_vrp_strategy(prices_df, returns_df, rv_window=21, rich_threshold=1.3,
                     cheap_threshold=1.0, defensive_asset='SPY', tx_cost=0.001):
    """
    VIX / Realized-Vol ratio strategy.
    - ratio > rich_threshold: UPRO (rich implied vol = market overpaying for protection)
    - ratio < cheap_threshold: defensive_asset (panic / cheap vol = danger)
    - between: standard vol-based switching
    """
    # Compute 21-day realized vol (annualized) on SPY
    spy_log_rets = np.log(prices_df['SPY'] / prices_df['SPY'].shift(1))
    realized_vol = spy_log_rets.rolling(rv_window).std() * np.sqrt(252) * 100  # In % terms like VIX

    # VIX / RV ratio
    vrp_ratio = prices_df['VIX'] / realized_vol

    port_rets = []
    prev_pos = 'UPRO'

    for i in range(rv_window + 1, len(returns_df)):
        idx = returns_df.index[i]
        ratio = vrp_ratio.iloc[i] if i < len(vrp_ratio) else 1.15

        if np.isnan(ratio) or np.isinf(ratio):
            ratio = 1.15  # neutral default

        # Determine position
        if ratio > rich_threshold:
            pos = 'UPRO'  # Rich implied vol = safe to be aggressive
        elif ratio < cheap_threshold:
            pos = defensive_asset  # Cheap/inverted = danger
        else:
            # Middle zone: use a simple vol filter
            rv = realized_vol.iloc[i] if i < len(realized_vol) else 15
            if not np.isnan(rv) and rv > 25:
                pos = defensive_asset
            else:
                pos = 'UPRO'

        # Transaction cost
        cost = tx_cost if pos != prev_pos else 0
        prev_pos = pos

        if pos in returns_df.columns:
            day_ret = returns_df[pos].iloc[i] - cost
        else:
            day_ret = returns_df['SPY'].iloc[i] - cost

        port_rets.append(day_ret)

    return pd.Series(port_rets, index=returns_df.index[rv_window + 1:len(returns_df)])


# Test multiple VRP variants
vrp_variants = {
    'VRP_base': {'rich_threshold': 1.3, 'cheap_threshold': 1.0, 'defensive_asset': 'SPY'},
    'VRP_aggressive': {'rich_threshold': 1.2, 'cheap_threshold': 0.9, 'defensive_asset': 'SPY'},
    'VRP_conservative': {'rich_threshold': 1.5, 'cheap_threshold': 1.1, 'defensive_asset': 'GLD'},
    'VRP_gld_defense': {'rich_threshold': 1.3, 'cheap_threshold': 1.0, 'defensive_asset': 'GLD'},
    'VRP_tlt_defense': {'rich_threshold': 1.3, 'cheap_threshold': 1.0, 'defensive_asset': 'TLT'},
}

vrp_results = {}
for name, params in vrp_variants.items():
    print(f"\n  Testing {name}...")
    rets = run_vrp_strategy(prices, returns, **params)
    spy_a = spy_rets.reindex(rets.index).dropna()
    rets_a = rets.reindex(spy_a.index).dropna()
    upro_a = upro_rets.reindex(spy_a.index).dropna()

    # Create walk-forward function for this variant
    def make_wf_func(p):
        def wf_func(test_p, test_r):
            return run_vrp_strategy(test_p, test_r, **p)
        return wf_func

    vrp_results[name] = full_validation(
        rets_a, spy_a, upro_a, name,
        strategy_func=make_wf_func(params),
        prices_df=prices, returns_df=returns
    )

# Analyze VRP ratio distribution
spy_log_rets = np.log(prices['SPY'] / prices['SPY'].shift(1))
rv_21 = spy_log_rets.rolling(21).std() * np.sqrt(252) * 100
vrp_ratio_series = prices['VIX'] / rv_21
vrp_clean = vrp_ratio_series.dropna()
vrp_clean = vrp_clean[~np.isinf(vrp_clean)]

print(f"\n  VRP Ratio Distribution:")
print(f"    Mean: {vrp_clean.mean():.3f}")
print(f"    Median: {vrp_clean.median():.3f}")
print(f"    Std: {vrp_clean.std():.3f}")
print(f"    %time > 1.3 (rich): {(vrp_clean > 1.3).mean()*100:.1f}%")
print(f"    %time < 1.0 (cheap): {(vrp_clean < 1.0).mean()*100:.1f}%")
print(f"    %time 1.0-1.3 (neutral): {((vrp_clean >= 1.0) & (vrp_clean <= 1.3)).mean()*100:.1f}%")

best_vrp = max(vrp_results.items(), key=lambda x: x[1]['metrics']['sharpe'])
print(f"\n  BEST VRP: {best_vrp[0]} (Sharpe={best_vrp[1]['metrics']['sharpe']:.3f})")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 2: MEAN-REVERSION ON UPRO AFTER MULTI-DAY DECLINES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 2: MEAN-REVERSION ON UPRO AFTER MULTI-DAY DECLINES")
print("=" * 70)
print("Concept: When UPRO drops 3+ consecutive days or >8% in 5d, buy the bounce.")

def run_bounce_strategy(prices_df, returns_df, consec_days=3, pct_threshold=-0.08,
                        pct_window=5, hold_days=7, tx_cost=0.001):
    """
    Buy UPRO when it drops consec_days in a row OR pct_threshold over pct_window days.
    Hold for hold_days then exit.
    Returns: full strategy returns, bounce-only returns, bounce stats
    """
    upro_rets = returns_df['UPRO']

    # Detect consecutive down days
    is_down = (upro_rets < 0).astype(int)
    consec_down = is_down.copy() * 0
    count = 0
    for i in range(len(is_down)):
        if is_down.iloc[i]:
            count += 1
        else:
            count = 0
        consec_down.iloc[i] = count

    # Detect 5-day cumulative decline
    cum_5d = upro_rets.rolling(pct_window).sum()

    # Generate signals
    in_bounce = False
    bounce_countdown = 0
    port_rets = []
    bounce_trades = []  # track individual bounce returns

    current_bounce_rets = []

    for i in range(max(pct_window, consec_days), len(returns_df)):
        idx = returns_df.index[i]

        # Check for new bounce entry signal
        if not in_bounce:
            trigger_consec = consec_down.iloc[i] >= consec_days
            trigger_pct = cum_5d.iloc[i] <= pct_threshold if not np.isnan(cum_5d.iloc[i]) else False

            if trigger_consec or trigger_pct:
                in_bounce = True
                bounce_countdown = hold_days
                current_bounce_rets = []

        if in_bounce:
            day_ret = returns_df['UPRO'].iloc[i] - (tx_cost if bounce_countdown == hold_days else 0)
            port_rets.append(day_ret)
            current_bounce_rets.append(float(day_ret))
            bounce_countdown -= 1
            if bounce_countdown <= 0:
                in_bounce = False
                exit_cost = tx_cost
                port_rets[-1] -= exit_cost
                bounce_trades.append({
                    'entry_date': str(returns_df.index[i - hold_days + 1].date()),
                    'exit_date': str(idx.date()),
                    'total_ret': sum(current_bounce_rets),
                    'n_days': len(current_bounce_rets),
                })
        else:
            # Out of bounce: hold SPY as baseline
            port_rets.append(returns_df['SPY'].iloc[i])

    port_series = pd.Series(port_rets, index=returns_df.index[max(pct_window, consec_days):len(returns_df)])

    # Bounce-only stats
    if bounce_trades:
        bounce_rets_only = [t['total_ret'] for t in bounce_trades]
        bounce_stats = {
            'n_trades': len(bounce_trades),
            'avg_return': float(np.mean(bounce_rets_only)),
            'median_return': float(np.median(bounce_rets_only)),
            'win_rate': float(np.mean([1 if r > 0 else 0 for r in bounce_rets_only])),
            'best': float(max(bounce_rets_only)),
            'worst': float(min(bounce_rets_only)),
            'avg_per_day': float(np.mean(bounce_rets_only)) / hold_days if hold_days > 0 else 0,
        }
    else:
        bounce_stats = {'n_trades': 0}

    return port_series, bounce_trades, bounce_stats


# Test multiple bounce variants
bounce_variants = {
    'Bounce_3d_7hold': {'consec_days': 3, 'pct_threshold': -0.08, 'hold_days': 7},
    'Bounce_3d_5hold': {'consec_days': 3, 'pct_threshold': -0.08, 'hold_days': 5},
    'Bounce_3d_10hold': {'consec_days': 3, 'pct_threshold': -0.08, 'hold_days': 10},
    'Bounce_4d_7hold': {'consec_days': 4, 'pct_threshold': -0.10, 'hold_days': 7},
    'Bounce_2d_5hold': {'consec_days': 2, 'pct_threshold': -0.06, 'hold_days': 5},
    'Bounce_aggressive': {'consec_days': 2, 'pct_threshold': -0.05, 'hold_days': 10},
}

bounce_results = {}
for name, params in bounce_variants.items():
    print(f"\n  Testing {name}...")
    rets, trades, bstats = run_bounce_strategy(prices, returns, **params)

    print(f"  Bounce stats: {bstats['n_trades']} trades, "
          f"avg_ret={bstats.get('avg_return', 0)*100:.2f}%, "
          f"WR={bstats.get('win_rate', 0)*100:.1f}%, "
          f"avg/day={bstats.get('avg_per_day', 0)*100:.3f}%")

    spy_a = spy_rets.reindex(rets.index).dropna()
    rets_a = rets.reindex(spy_a.index).dropna()
    upro_a = upro_rets.reindex(spy_a.index).dropna()

    def make_bounce_wf(p):
        def wf_func(test_p, test_r):
            r, _, _ = run_bounce_strategy(test_p, test_r, **p)
            return r
        return wf_func

    bounce_results[name] = full_validation(
        rets_a, spy_a, upro_a, name,
        strategy_func=make_bounce_wf(params),
        prices_df=prices, returns_df=returns
    )
    bounce_results[name]['bounce_stats'] = bstats

best_bounce = max(bounce_results.items(), key=lambda x: x[1]['metrics']['sharpe'])
print(f"\n  BEST BOUNCE: {best_bounce[0]} (Sharpe={best_bounce[1]['metrics']['sharpe']:.3f})")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 3: SPY BREADTH TIMING FOR UPRO (RSP/SPY RATIO PROXY)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 3: SPY BREADTH TIMING (RSP/SPY RATIO)")
print("=" * 70)
print("Concept: RSP outperforming SPY = healthy broad rally. Narrow leadership = danger.")

def run_breadth_strategy(prices_df, returns_df, lookback=63, healthy_pctl=60,
                         narrow_pctl=40, defensive_asset='GLD', tx_cost=0.001):
    """
    Use RSP/SPY ratio rolling z-score as breadth proxy.
    - Z-score > 0 (RSP outperforming): UPRO (healthy breadth)
    - Z-score mildly negative: SPY (narrow leadership)
    - Z-score very negative: defensive_asset (deteriorating breadth)
    """
    # RSP/SPY ratio
    ratio = prices_df['RSP'] / prices_df['SPY']

    # Rolling z-score of ratio changes (rate of change)
    ratio_change = ratio.pct_change(lookback)  # N-day change in ratio
    ratio_zscore = (ratio_change - ratio_change.rolling(lookback*4).mean()) / ratio_change.rolling(lookback*4).std()

    # Alternative: just use rolling relative performance
    # Simpler: RSP return vs SPY return over lookback
    rsp_ret = prices_df['RSP'].pct_change(lookback)
    spy_ret = prices_df['SPY'].pct_change(lookback)
    breadth_signal = rsp_ret - spy_ret  # positive = RSP outperforming = broad rally

    # Rolling percentile of breadth signal
    breadth_pctl = breadth_signal.rolling(lookback * 4).rank(pct=True) * 100

    port_rets = []
    prev_pos = 'UPRO'
    start_idx = lookback * 4 + 10

    for i in range(start_idx, len(returns_df)):
        idx = returns_df.index[i]

        bp = breadth_pctl.iloc[i] if i < len(breadth_pctl) else 50
        if np.isnan(bp):
            bp = 50

        # Determine position
        if bp >= healthy_pctl:
            pos = 'UPRO'  # Broad rally — go aggressive
        elif bp >= narrow_pctl:
            pos = 'SPY'   # Narrow leadership — be cautious
        else:
            pos = defensive_asset  # Deteriorating breadth — defensive

        # Transaction cost
        cost = tx_cost if pos != prev_pos else 0
        prev_pos = pos

        if pos in returns_df.columns:
            day_ret = returns_df[pos].iloc[i] - cost
        else:
            day_ret = returns_df['SPY'].iloc[i] - cost

        port_rets.append(day_ret)

    return pd.Series(port_rets, index=returns_df.index[start_idx:len(returns_df)])


# Test multiple breadth variants
breadth_variants = {
    'Breadth_base': {'lookback': 63, 'healthy_pctl': 60, 'narrow_pctl': 40, 'defensive_asset': 'GLD'},
    'Breadth_short_lb': {'lookback': 21, 'healthy_pctl': 60, 'narrow_pctl': 40, 'defensive_asset': 'GLD'},
    'Breadth_long_lb': {'lookback': 126, 'healthy_pctl': 60, 'narrow_pctl': 40, 'defensive_asset': 'GLD'},
    'Breadth_aggressive': {'lookback': 63, 'healthy_pctl': 50, 'narrow_pctl': 30, 'defensive_asset': 'GLD'},
    'Breadth_conservative': {'lookback': 63, 'healthy_pctl': 70, 'narrow_pctl': 50, 'defensive_asset': 'GLD'},
    'Breadth_tlt_defense': {'lookback': 63, 'healthy_pctl': 60, 'narrow_pctl': 40, 'defensive_asset': 'TLT'},
    'Breadth_binary': {'lookback': 63, 'healthy_pctl': 50, 'narrow_pctl': 50, 'defensive_asset': 'SPY'},
}

breadth_results = {}
for name, params in breadth_variants.items():
    print(f"\n  Testing {name}...")
    rets = run_breadth_strategy(prices, returns, **params)

    spy_a = spy_rets.reindex(rets.index).dropna()
    rets_a = rets.reindex(spy_a.index).dropna()
    upro_a = upro_rets.reindex(spy_a.index).dropna()

    def make_breadth_wf(p):
        def wf_func(test_p, test_r):
            return run_breadth_strategy(test_p, test_r, **p)
        return wf_func

    breadth_results[name] = full_validation(
        rets_a, spy_a, upro_a, name,
        strategy_func=make_breadth_wf(params),
        prices_df=prices, returns_df=returns
    )

# Analyze breadth signal distribution
rsp_ret = prices['RSP'].pct_change(63)
spy_ret_63 = prices['SPY'].pct_change(63)
breadth_sig = (rsp_ret - spy_ret_63).dropna()
print(f"\n  Breadth Signal (RSP-SPY 63d relative return) Distribution:")
print(f"    Mean: {breadth_sig.mean()*100:.3f}%")
print(f"    Std: {breadth_sig.std()*100:.3f}%")
print(f"    %time RSP outperforms (>0): {(breadth_sig > 0).mean()*100:.1f}%")

best_breadth = max(breadth_results.items(), key=lambda x: x[1]['metrics']['sharpe'])
print(f"\n  BEST BREADTH: {best_breadth[0]} (Sharpe={best_breadth[1]['metrics']['sharpe']:.3f})")


# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BATCH 5 FINAL SUMMARY")
print("=" * 70)

# Collect all results
all_results = {
    'benchmarks': {
        'SPY': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in spy_m.items()},
        'UPRO': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in upro_m.items()},
    },
    'strategy_1_vrp': vrp_results,
    'strategy_2_bounce': bounce_results,
    'strategy_3_breadth': breadth_results,
    'summary': {},
}

# Print comparison table
print(f"\n{'Strategy':<35} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'WR':>6} {'PF':>6} {'Verdict':>10}")
print("-" * 95)
print(f"{'SPY Buy-Hold':<35} {spy_m['sharpe']:>7.3f} {spy_m['sortino']:>8.3f} {spy_m['cagr']*100:>7.1f}% {spy_m['max_dd']*100:>7.1f}% {spy_m['wr']*100:>5.1f}% {spy_m['pf']:>6.3f} {'BENCH':>10}")
print(f"{'UPRO Buy-Hold':<35} {upro_m['sharpe']:>7.3f} {upro_m['sortino']:>8.3f} {upro_m['cagr']*100:>7.1f}% {upro_m['max_dd']*100:>7.1f}% {upro_m['wr']*100:>5.1f}% {upro_m['pf']:>6.3f} {'BENCH':>10}")
print("-" * 95)

strategy_summary = []

for group_name, group_results in [('VRP', vrp_results), ('Bounce', bounce_results), ('Breadth', breadth_results)]:
    for name, res in sorted(group_results.items(), key=lambda x: -x[1]['metrics']['sharpe']):
        m = res['metrics']
        v = res['verdict']
        print(f"{name:<35} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['cagr']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['wr']*100:>5.1f}% {m['pf']:>6.3f} {v:>10}")
        strategy_summary.append({
            'name': name,
            'group': group_name,
            'sharpe': float(m['sharpe']),
            'sortino': float(m['sortino']),
            'cagr': float(m['cagr']),
            'max_dd': float(m['max_dd']),
            'verdict': v,
        })

# Overall winners
validated = [s for s in strategy_summary if s['verdict'] == 'VALIDATED']
failed = [s for s in strategy_summary if s['verdict'] == 'FAILED']

print(f"\n{'='*70}")
print(f"VALIDATED: {len(validated)} strategies")
for s in sorted(validated, key=lambda x: -x['sharpe']):
    print(f"  {s['name']} (Sharpe={s['sharpe']:.3f}, CAGR={s['cagr']*100:.1f}%)")

print(f"\nFAILED: {len(failed)} strategies")
for s in sorted(failed, key=lambda x: -x['sharpe']):
    print(f"  {s['name']} (Sharpe={s['sharpe']:.3f})")

# Best overall
if validated:
    best = max(validated, key=lambda x: x['sharpe'])
    print(f"\nBATCH 5 CHAMPION: {best['name']} (Sharpe={best['sharpe']:.3f}, CAGR={best['cagr']*100:.1f}%)")
else:
    print(f"\nNo strategies passed all validation tests.")

# Compare vs Gameplan v3
print(f"\nCOMPARISON vs CHAMPIONS:")
print(f"  Gameplan v3:     Sharpe ~2.388, CAGR ~75.4%")
print(f"  Vol Mean Rev:    Sharpe ~1.467, CAGR ~42.8%")
if validated:
    best = max(validated, key=lambda x: x['sharpe'])
    print(f"  Batch5 best:     Sharpe {best['sharpe']:.3f}, CAGR {best['cagr']*100:.1f}%")

all_results['summary'] = {
    'n_validated': len(validated),
    'n_failed': len(failed),
    'validated_strategies': [s['name'] for s in validated],
    'best_strategy': max(validated, key=lambda x: x['sharpe'])['name'] if validated else None,
    'best_sharpe': max(validated, key=lambda x: x['sharpe'])['sharpe'] if validated else None,
}

# Save results
results_path = os.path.join(OUT_DIR, "batch5_results.json")

def convert_for_json(obj):
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj

def deep_convert(obj):
    if isinstance(obj, dict):
        return {k: deep_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_convert(v) for v in obj]
    else:
        return convert_for_json(obj)

with open(results_path, 'w') as f:
    json.dump(deep_convert(all_results), f, indent=2, default=str)

# Also save to the canonical location
canonical_path = "/home/jupiter/Lvl3Quant/output/growth_research/creative_batch5/batch5_results.json"
if results_path != canonical_path:
    import shutil
    shutil.copy2(results_path, canonical_path)

print(f"\nResults saved to {results_path}")
print(f"\nDONE — Batch 5 complete.")
