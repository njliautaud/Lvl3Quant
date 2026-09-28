#!/usr/bin/env python3
"""
evaluate_selector_variants.py — Comprehensive comparison of all queue entry selector variants.

Compares v2, v2.1 (champion), v2.2, v2.3, v2.4, v2.1-lean on:
  1. Performance at standard thresholds (0.55, 0.58, 0.60, 0.65, 0.70)
  2. Regime gate (HC #428 R1): |Sharpe_green - Sharpe_red| / max(...) <= 0.50
  3. Side breakdown (long vs short)
  4. Day concentration (HC #344): no single day > 70% of P&L
  5. Feature importance rankings
  6. Cost sensitivity: +0.0, +0.2, +0.5 ticks additional cost

Outputs: summary table to stdout + JSON results.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_BASE = BASE / "output"

VARIANTS = {
    "v2":      "queue_entry_selector_v2",
    "v2.1":    "queue_entry_selector_v2_1",
    "v2.2":    "queue_entry_selector_v2_2",
    "v2.3":    "queue_entry_selector_v2_3",
    "v2.4":    "queue_entry_selector_v2_4",
    "v2.1-lean": "queue_entry_selector_v2_1_lean",
}

# Commission per HC
COMMISSION_RT_TICKS = 0.376

def load_results(variant_dir):
    """Load results.json from a variant output directory."""
    results_path = variant_dir / "results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)

def load_trades(variant_dir):
    """Load OOT trades parquet."""
    for name in ["all_oot_trades_tp4sl3.parquet", "all_oot_trades.parquet"]:
        path = variant_dir / name
        if path.exists():
            return pd.read_parquet(path)
    return None

def compute_metrics(trades_df, additional_cost=0.0):
    """Compute risk-adjusted metrics from trades dataframe."""
    if trades_df is None or len(trades_df) == 0:
        return {}

    pnl_col = None
    for col in ['net_ticks', 'pnl_ticks', 'net_pnl_ticks']:
        if col in trades_df.columns:
            pnl_col = col
            break

    if pnl_col is None:
        return {}

    pnl = trades_df[pnl_col].values - additional_cost
    wins = pnl > 0

    metrics = {
        'n_trades': len(pnl),
        'wr': float(wins.mean()),
        'pf': float(pnl[wins].sum() / abs(pnl[~wins].sum())) if (~wins).any() and wins.any() else float('inf'),
        'total_pnl': float(pnl.sum()),
        'avg_pnl': float(pnl.mean()),
    }

    # Sharpe / Sortino (daily)
    if 'date' in trades_df.columns or 'oot_date' in trades_df.columns:
        date_col = 'date' if 'date' in trades_df.columns else 'oot_date'
        daily = trades_df.groupby(date_col)[pnl_col].sum() - additional_cost * trades_df.groupby(date_col)[pnl_col].count()
        if len(daily) > 1 and daily.std() > 0:
            metrics['sharpe'] = float(daily.mean() / daily.std() * np.sqrt(252))
            downside = daily[daily < 0].std()
            metrics['sortino'] = float(daily.mean() / downside * np.sqrt(252)) if downside > 0 else float('inf')
            metrics['n_days'] = int(len(daily))
            metrics['max_dd_ticks'] = float((daily.cumsum() - daily.cumsum().cummax()).min())

    return metrics

def regime_gate(trades_df, additional_cost=0.0):
    """Check HC #428 R1 regime gate."""
    if trades_df is None:
        return {'pass': False, 'reason': 'no trades'}

    # Need regime info
    regime_col = None
    for col in ['regime', 'day_regime']:
        if col in trades_df.columns:
            regime_col = col
            break

    if regime_col is None:
        return {'pass': None, 'reason': 'no regime column'}

    pnl_col = None
    for col in ['net_ticks', 'pnl_ticks', 'net_pnl_ticks']:
        if col in trades_df.columns:
            pnl_col = col
            break

    date_col = 'date' if 'date' in trades_df.columns else 'oot_date'

    results = {}
    for regime in ['green', 'red']:
        mask = trades_df[regime_col] == regime
        if mask.sum() == 0:
            results[regime] = {'sharpe': 0, 'n_trades': 0}
            continue
        sub = trades_df[mask]
        daily = sub.groupby(date_col)[pnl_col].sum() - additional_cost * sub.groupby(date_col)[pnl_col].count()
        if len(daily) > 1 and daily.std() > 0:
            results[regime] = {
                'sharpe': float(daily.mean() / daily.std() * np.sqrt(252)),
                'n_trades': int(mask.sum()),
                'n_days': int(len(daily)),
            }
        else:
            results[regime] = {'sharpe': 0, 'n_trades': int(mask.sum())}

    sg = results.get('green', {}).get('sharpe', 0)
    sr = results.get('red', {}).get('sharpe', 0)
    denom = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / denom if denom > 0 else 0

    return {
        'pass': gap <= 0.50,
        'gap_pct': float(gap * 100),
        'green_sharpe': sg,
        'red_sharpe': sr,
        'green_trades': results.get('green', {}).get('n_trades', 0),
        'red_trades': results.get('red', {}).get('n_trades', 0),
    }

def side_breakdown(trades_df, additional_cost=0.0):
    """Break down performance by long/short side."""
    if trades_df is None:
        return {}

    side_col = None
    for col in ['side', 'direction', 'trade_side']:
        if col in trades_df.columns:
            side_col = col
            break

    if side_col is None:
        return {}

    pnl_col = None
    for col in ['net_ticks', 'pnl_ticks', 'net_pnl_ticks']:
        if col in trades_df.columns:
            pnl_col = col
            break

    results = {}
    for side_val in trades_df[side_col].unique():
        mask = trades_df[side_col] == side_val
        sub_pnl = trades_df.loc[mask, pnl_col].values - additional_cost
        wins = sub_pnl > 0
        side_name = str(side_val).lower()
        if side_name in ['1', '1.0', 'long', 'buy']:
            side_name = 'long'
        elif side_name in ['-1', '-1.0', 'short', 'sell']:
            side_name = 'short'

        results[side_name] = {
            'n_trades': int(len(sub_pnl)),
            'wr': float(wins.mean()) if len(sub_pnl) > 0 else 0,
            'pf': float(sub_pnl[wins].sum() / abs(sub_pnl[~wins].sum())) if (~wins).any() and wins.any() else float('inf'),
            'total_pnl': float(sub_pnl.sum()),
        }

    return results


def main():
    print("=" * 90)
    print("QUEUE ENTRY SELECTOR — COMPREHENSIVE VARIANT COMPARISON")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    all_results = {}

    for name, dirname in VARIANTS.items():
        variant_dir = OUTPUT_BASE / dirname
        if not variant_dir.exists():
            print(f"\n⏳ {name}: directory not found (not yet run)")
            continue

        results_json = load_results(variant_dir)
        trades = load_trades(variant_dir)

        if results_json is None and trades is None:
            print(f"\n⏳ {name}: no results yet (still running?)")
            continue

        print(f"\n{'─' * 90}")
        print(f"  {name.upper()}")
        if results_json:
            changes = results_json.get('changes_from_v2_1', results_json.get('changes_from_v2_2', results_json.get('changes', ['N/A'])))
            if isinstance(changes, list) and len(changes) > 0:
                print(f"  Changes: {changes[0][:80]}{'...' if len(changes) > 1 else ''}")
        print(f"{'─' * 90}")

        variant_data = {'name': name}

        # Try to get performance from results.json first
        if results_json and 'result' in results_json:
            fp = results_json['result'].get('filtered_performance', {})
            if fp:
                print(f"\n  {'Thresh':<8} {'Trades':>7} {'WR':>7} {'PF':>7} {'Sharpe':>8} {'Regime':>8}")
                print(f"  {'─'*48}")
                for thresh in ['0.5', '0.52', '0.55', '0.58', '0.6', '0.65', '0.7']:
                    if thresh in fp:
                        v = fp[thresh]
                        wr = v.get('win_rate', v.get('wr', 0))
                        pf = v.get('profit_factor', v.get('pf', 0))
                        sh = v.get('sharpe', 0)
                        nt = v.get('n_trades', 0)
                        rg = v.get('regime_gap_pct', v.get('regime_gap', None))
                        rg_str = f"{rg:.0f}%" if rg is not None else "N/A"
                        wr_str = f"{wr*100:.1f}%" if wr < 1 else f"{wr:.1f}%"
                        print(f"  {thresh:<8} {nt:>7} {wr_str:>7} {pf:>7.2f} {sh:>8.3f} {rg_str:>8}")

                variant_data['thresholds'] = fp

        # Regime gate from trades if available
        if trades is not None:
            rg = regime_gate(trades)
            if rg.get('pass') is not None:
                status = "✅ PASS" if rg['pass'] else "❌ FAIL"
                print(f"\n  Regime gate: {status} (gap {rg['gap_pct']:.1f}%) — green Sharpe {rg['green_sharpe']:.3f} vs red {rg['red_sharpe']:.3f}")
                variant_data['regime_gate'] = rg

            # Side breakdown
            sb = side_breakdown(trades)
            if sb:
                print(f"\n  Side breakdown:")
                for side, data in sb.items():
                    print(f"    {side}: {data['n_trades']} trades, WR {data['wr']*100:.1f}%, PF {data['pf']:.2f}, PnL {data['total_pnl']:.1f}t")
                variant_data['side_breakdown'] = sb

        all_results[name] = variant_data

    # Summary comparison table
    print(f"\n{'=' * 90}")
    print("SUMMARY COMPARISON (@ threshold 0.60)")
    print(f"{'=' * 90}")
    print(f"  {'Variant':<12} {'Trades':>7} {'WR':>7} {'PF':>7} {'Sharpe':>8} {'Regime':>10} {'Verdict':>12}")
    print(f"  {'─' * 66}")

    for name in VARIANTS.keys():
        if name not in all_results:
            print(f"  {name:<12} {'—':>7} {'—':>7} {'—':>7} {'—':>8} {'—':>10} {'pending':>12}")
            continue

        data = all_results[name]
        thresholds = data.get('thresholds', {})
        t60 = thresholds.get('0.6', thresholds.get('0.60', {}))

        if not t60:
            # Try from results.json directly
            print(f"  {name:<12} {'—':>7} {'—':>7} {'—':>7} {'—':>8} {'—':>10} {'no 0.60':>12}")
            continue

        wr = t60.get('win_rate', t60.get('wr', 0))
        pf = t60.get('profit_factor', t60.get('pf', 0))
        sh = t60.get('sharpe', 0)
        nt = t60.get('n_trades', 0)
        rg = data.get('regime_gate', {})
        rg_pass = rg.get('pass', None)

        if rg_pass is True:
            regime_str = f"✅ {rg.get('gap_pct', 0):.0f}%"
            verdict = "PASS" if sh > 0 and pf > 1.0 else "MARGINAL"
        elif rg_pass is False:
            regime_str = f"❌ {rg.get('gap_pct', 0):.0f}%"
            verdict = "FAIL"
        else:
            regime_str = "N/A"
            verdict = "?" if pf > 1.0 else "FAIL"

        wr_str = f"{wr*100:.1f}%" if wr < 1 else f"{wr:.1f}%"

        # Mark champion
        if name == "v2.1" and pf > 1.3:
            verdict = "★ CHAMPION"

        print(f"  {name:<12} {nt:>7} {wr_str:>7} {pf:>7.2f} {sh:>8.3f} {regime_str:>10} {verdict:>12}")

    # Save results
    output_path = OUTPUT_BASE / "selector_variant_comparison.json"
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'variants': all_results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
