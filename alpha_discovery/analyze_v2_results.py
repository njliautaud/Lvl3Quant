#!/usr/bin/env python3
"""Post-hoc analysis of queue_entry_selector_v2 results.
Run after v2 completes to produce regime-stratified report per HC #428.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/queue_entry_selector_v2")
RESULTS_PATH = OUTPUT_DIR / "results.json"
TRADES_PATH = OUTPUT_DIR / "all_oot_trades.parquet"

ES_TICK_VALUE = 12.50


def main():
    if not RESULTS_PATH.exists():
        print("v2 results not found yet — run queue_entry_selector_v2.py first")
        return

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    trades = pd.read_parquet(TRADES_PATH) if TRADES_PATH.exists() else None

    print("=" * 70)
    print(f"QUEUE ENTRY SELECTOR v2 — POST-HOC ANALYSIS")
    print(f"  Dates: {results['n_dates']}, Folds: {results['n_folds']}, Features: {results['n_features']}")
    print(f"  FIFO config: {results['fifo_config']}")
    print("=" * 70)

    # ── Baseline ──
    bl = results['baseline']
    print(f"\nBASELINE (no filter):")
    print(f"  Trades: {bl['n_trades']:,}")
    print(f"  WR: {bl['wr']:.1%}")
    print(f"  PnL: {bl['pnl']:+,.1f} ticks (${bl['pnl'] * ES_TICK_VALUE:+,.0f})")
    print(f"  Per-trade: {bl['per_trade']:+.3f} ticks")

    # ── Filtered performance ──
    print(f"\nFILTERED PERFORMANCE:")
    print(f"  {'Thresh':>7} | {'Trades':>7} | {'WR':>6} | {'PnL':>10} | {'$/trade':>8} | {'PF':>6} | {'Sharpe':>7} | {'Sortino':>8}")
    print(f"  {'-'*7} | {'-'*7} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*6} | {'-'*7} | {'-'*8}")

    for thresh, m in sorted(results['filtered_performance'].items(), key=lambda x: float(x[0])):
        dollar_per_trade = m['per_trade'] * ES_TICK_VALUE
        sortino = m.get('sortino', 'N/A')
        sortino_str = f"{sortino:8.2f}" if isinstance(sortino, (int, float)) else f"{'N/A':>8}"
        print(f"  {float(thresh):7.2f} | {m['n_trades']:7,} | {m['wr']:5.1%} | "
              f"{m['pnl']:+10,.1f} | ${dollar_per_trade:+7.2f} | {m['pf']:6.3f} | "
              f"{m['sharpe']:7.2f} | {sortino_str}")

    # ── Regime analysis ──
    ra = results['regime_analysis']
    print(f"\nREGIME ANALYSIS (HC #428):")
    print(f"  Threshold: {ra['threshold']}")
    green = ra.get('green', {})
    red = ra.get('red', {})
    print(f"  Green days: Sharpe={green.get('sharpe', 'N/A'):.2f}, n_days={green.get('n_days', 0)}, n_trades={green.get('n_trades', 0)}")
    print(f"  Red days:   Sharpe={red.get('sharpe', 'N/A'):.2f}, n_days={red.get('n_days', 0)}, n_trades={red.get('n_trades', 0)}")
    print(f"  Regime gap: {ra['gap']:.3f} {'✅ PASS (≤0.50)' if ra['pass_gate'] else '❌ FAIL (>0.50)'}")

    # ── IC ──
    ic = results['ic']
    print(f"\nINFORMATION COEFFICIENT:")
    print(f"  Pearson IC:  {ic['pearson']:.4f}")
    print(f"  Spearman IC: {ic['spearman']:.4f}")

    # ── Feature importance ──
    print(f"\nTOP 10 FEATURES (mean gain):")
    sorted_fi = sorted(results['feature_importance'].items(),
                       key=lambda x: x[1]['mean_gain'], reverse=True)[:10]
    for rank, (feat, fi) in enumerate(sorted_fi, 1):
        print(f"  {rank:2d}. {feat:35s} gain={fi['mean_gain']:.1f} ± {fi['std_gain']:.1f}")

    # ── Per-trade analysis (if parquet available) ──
    if trades is not None:
        print(f"\n{'='*70}")
        print(f"DETAILED TRADE ANALYSIS")
        print(f"{'='*70}")

        # Per-side
        best_thresh = float(ra['threshold'])
        filtered = trades[trades['pred_prob'] >= best_thresh]

        for side in ['long', 'short']:
            s = filtered[filtered['side'] == side]
            if len(s) < 5:
                continue
            wr = s['target'].mean()
            pnl = s['net_ticks'].sum()
            win_sum = s[s['net_ticks'] > 0]['net_ticks'].sum()
            loss_sum = abs(s[s['net_ticks'] < 0]['net_ticks'].sum())
            pf = win_sum / max(loss_sum, 1e-6)
            print(f"\n  {side.upper()} side @ {best_thresh}:")
            print(f"    Trades: {len(s):,}, WR: {wr:.1%}, PnL: {pnl:+,.1f}t, PF: {pf:.3f}")

        # Day-by-day P&L
        daily = filtered.groupby('date').agg(
            n_trades=('net_ticks', 'count'),
            pnl=('net_ticks', 'sum'),
            wr=('target', 'mean'),
            regime=('regime', 'first'),
        ).sort_index()

        green_days = (daily['pnl'] > 0).sum()
        red_days = (daily['pnl'] < 0).sum()
        flat_days = (daily['pnl'] == 0).sum()
        max_dd_day = daily['pnl'].min()
        best_day = daily['pnl'].max()

        print(f"\n  DAILY P&L DISTRIBUTION:")
        print(f"    Green days: {green_days}, Red days: {red_days}, Flat: {flat_days}")
        print(f"    Best day: {best_day:+.1f}t, Worst day: {max_dd_day:+.1f}t")
        print(f"    Mean: {daily['pnl'].mean():+.1f}t, Median: {daily['pnl'].median():+.1f}t")

        # Drawdown analysis
        cum_pnl = daily['pnl'].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_dd = drawdown.min()
        print(f"    Max drawdown: {max_dd:+.1f}t (${max_dd * ES_TICK_VALUE:+,.0f})")

        # Calmar ratio (annualized return / max DD)
        if max_dd < 0:
            total_pnl = daily['pnl'].sum()
            calmar = abs(total_pnl / max_dd) if max_dd != 0 else 0
            print(f"    Calmar ratio (total/maxDD): {calmar:.2f}")

    # ── v1 vs v2 comparison ──
    print(f"\n{'='*70}")
    print(f"v1 vs v2 COMPARISON")
    print(f"{'='*70}")
    print(f"  v1: 40 dates, 2 folds, 10 OOT days")
    print(f"  v1 @ thresh=0.60: WR=60.2%, PF=1.509, Sharpe=6.75")
    print(f"  v2: {results['n_dates']} dates, {results['n_folds']} folds")
    t060 = results['filtered_performance'].get('0.6', results['filtered_performance'].get('0.60', {}))
    if t060:
        print(f"  v2 @ thresh=0.60: WR={t060['wr']:.1%}, PF={t060['pf']:.3f}, Sharpe={t060['sharpe']:.2f}")
    else:
        print(f"  v2 @ thresh=0.60: no data at this threshold")

    # ── Summary verdict ──
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    best_fp = max(results['filtered_performance'].items(),
                  key=lambda x: x[1]['pf']) if results['filtered_performance'] else None
    if best_fp:
        t, m = best_fp
        passes_regime = ra['pass_gate']
        profitable = m['pnl'] > 0
        good_wr = m['wr'] > 0.50
        positive_ic = ic['pearson'] > 0
        enough_sample = m['n_trades'] > 100

        print(f"  Best threshold: {t} (PF={m['pf']:.3f})")
        print(f"  ✅ Profitable:       {profitable} ({m['pnl']:+,.1f} ticks)")
        print(f"  {'✅' if good_wr else '❌'} WR > 50%:          {m['wr']:.1%}")
        print(f"  {'✅' if passes_regime else '❌'} Regime gate:       gap={ra['gap']:.3f}")
        print(f"  {'✅' if positive_ic else '❌'} Positive IC:       {ic['pearson']:.4f}")
        print(f"  {'✅' if enough_sample else '⚠️'} Sample size > 100: {m['n_trades']:,}")

        if profitable and passes_regime and positive_ic and enough_sample:
            print(f"\n  🎯 VIABLE STRATEGY — proceed to full-dataset run + live integration planning")
        elif profitable and not passes_regime:
            print(f"\n  ⚠️ PROFITABLE BUT REGIME-DEPENDENT — needs work on regime robustness")
        elif not profitable:
            print(f"\n  ❌ NOT PROFITABLE — queue features don't provide enough edge at this config")
        else:
            print(f"\n  ⚠️ MIXED RESULTS — needs further investigation")


if __name__ == '__main__':
    main()
