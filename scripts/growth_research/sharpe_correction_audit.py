#!/usr/bin/env python3
"""Sharpe Correction Audit — Recalculate all top strategies with CORRECT Sharpe.

ISSUE FOUND: Original scripts compute monthly Sharpe as:
    monthly_pnl_sum / INITIAL_CAPITAL ($645)
This inflates Sharpe massively when capital compounds (e.g., $645 -> $39K).

CORRECT method:
    monthly_pnl_sum / EQUITY_AT_START_OF_MONTH

This script:
1. Loads trade logs from all validated strategies
2. Recalculates Sharpe using proper equity-based returns
3. Reports honest vs. inflated metrics
4. Flags any strategies that fail under honest calculation
"""
import json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

CAP = 645.0

def correct_sharpe_from_trades(trades, initial_capital=CAP):
    """Compute Sharpe using equity-based monthly returns (CORRECT method)."""
    if not trades or len(trades) < 10:
        return {'sharpe_correct': 0, 'sharpe_inflated': 0, 'valid': False}

    tdf = pd.DataFrame(trades)
    tdf['date'] = pd.to_datetime(tdf['entry'] if 'entry' in tdf.columns else tdf['date'])
    tdf['month'] = tdf['date'].dt.to_period('M')

    # Build equity curve
    equity = initial_capital
    equity_at_month_start = {}
    current_month = None
    monthly_pnl = {}

    for _, row in tdf.iterrows():
        m = row['month']
        if m != current_month:
            equity_at_month_start[m] = equity
            current_month = m
            monthly_pnl[m] = 0
        pnl = row['pnl']
        monthly_pnl[m] += pnl
        equity += pnl

    # INFLATED method: pnl / initial capital
    months = sorted(monthly_pnl.keys())
    inflated_rets = np.array([monthly_pnl[m] / initial_capital for m in months])

    # CORRECT method: pnl / equity at start of month
    correct_rets = np.array([
        monthly_pnl[m] / max(equity_at_month_start[m], 1.0) for m in months
    ])

    ny = max(len(months) / 12, 0.5)

    # Inflated Sharpe
    sh_inflated = (inflated_rets.mean() * 12) / (inflated_rets.std() * np.sqrt(12) + 1e-10) if len(inflated_rets) > 3 else 0

    # Correct Sharpe
    sh_correct = (correct_rets.mean() * 12) / (correct_rets.std() * np.sqrt(12) + 1e-10) if len(correct_rets) > 3 else 0

    # Also compute Sortino both ways
    dn_i = inflated_rets[inflated_rets < 0]
    so_inflated = (inflated_rets.mean() * 12) / (dn_i.std() * np.sqrt(12) + 1e-10) if len(dn_i) > 1 else 0

    dn_c = correct_rets[correct_rets < 0]
    so_correct = (correct_rets.mean() * 12) / (dn_c.std() * np.sqrt(12) + 1e-10) if len(dn_c) > 1 else 0

    # Other honest metrics
    wins = sum(1 for t in trades if t.get('win', t.get('pnl', 0) > 0))
    wr = wins / len(trades) * 100
    pnls = [t['pnl'] for t in trades]
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gp / (gl + 1e-10)
    cagr = (equity / initial_capital) ** (1 / ny) - 1

    # MaxDD on equity curve
    eq_list = [initial_capital]
    e = initial_capital
    for _, row in tdf.iterrows():
        e += row['pnl']
        eq_list.append(e)
    eq = np.array(eq_list)
    pk = np.maximum.accumulate(eq)
    maxdd = float(((eq - pk) / (pk + 1e-10)).min()) * 100

    # Regime analysis
    bull_trades = [t for t in trades if t.get('regime') == 'bull']
    bear_trades = [t for t in trades if t.get('regime') == 'bear']
    bull_wr = sum(1 for t in bull_trades if t.get('win', t.get('pnl', 0) > 0)) / max(len(bull_trades), 1) * 100
    bear_wr = sum(1 for t in bear_trades if t.get('win', t.get('pnl', 0) > 0)) / max(len(bear_trades), 1) * 100
    r1_gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr, 1)

    # Permutation test on correct returns
    perm_p = 1.0
    if len(correct_rets) >= 10:
        real_sr = correct_rets.mean() / (correct_rets.std() + 1e-10)
        n_perm = 5000
        count = sum(1 for _ in range(n_perm) if
                    np.mean(correct_rets * np.random.choice([-1, 1], len(correct_rets))) /
                    (np.std(correct_rets) + 1e-10) >= real_sr)
        perm_p = count / n_perm

    # Sub-period stability
    mid = len(correct_rets) // 2
    h1_sr = correct_rets[:mid].mean() / (correct_rets[:mid].std() + 1e-10) if mid > 3 else 0
    h2_sr = correct_rets[mid:].mean() / (correct_rets[mid:].std() + 1e-10) if len(correct_rets) - mid > 3 else 0
    stable = h1_sr > 0 and h2_sr > 0

    # Transaction cost sensitivity (would 2x cost kill it?)
    cost_pnls = [t['pnl'] - 2.60 for t in trades]  # subtract extra commission per trade
    cost_wins = sum(1 for p in cost_pnls if p > 0)
    cost_wr = cost_wins / len(cost_pnls) * 100 if cost_pnls else 0
    cost_total = sum(cost_pnls)
    cost_survives = cost_total > 0

    # Gate counting
    gates = 0
    g1 = perm_p < 0.05; gates += g1
    g2 = r1_gap < 0.50; gates += g2
    g3 = stable; gates += g3
    g4 = cost_survives; gates += g4

    return {
        'sharpe_correct': round(sh_correct, 2),
        'sharpe_inflated': round(sh_inflated, 2),
        'inflation_ratio': round(sh_inflated / (sh_correct + 1e-10), 2),
        'sortino_correct': round(so_correct, 2),
        'sortino_inflated': round(so_inflated, 2),
        'win_rate': round(wr, 1),
        'profit_factor': round(pf, 2),
        'cagr_pct': round(cagr * 100, 1),
        'maxdd_pct': round(maxdd, 1),
        'final_equity': round(equity, 2),
        'n_trades': len(trades),
        'n_months': len(months),
        'r1_gap': round(r1_gap, 3),
        'perm_p': round(perm_p, 4),
        'sub_period_stable': stable,
        'h1_sr': round(h1_sr, 3),
        'h2_sr': round(h2_sr, 3),
        'cost_survives_2x': cost_survives,
        'gates': gates,
        'gate_details': f"perm={'PASS' if g1 else 'FAIL'} R1={'PASS' if g2 else 'FAIL'} stable={'PASS' if g3 else 'FAIL'} cost={'PASS' if g4 else 'FAIL'}",
        'valid': True
    }


def load_trade_logs():
    """Find and load all trade log files from research findings."""
    findings = RESULTS_DIR
    trade_files = {}

    # Known result files with trade data
    patterns = [
        ('multi_asset_momentum_options_v1_results.json', 'Multi-Asset Momentum Options'),
        ('integrated_sector_options_v2_results.json', 'Integrated Sector Options v2'),
        ('multi_factor_sector_options_v1_results.json', 'Multi-Factor Sector Options'),
        ('sector_options_rotation_v1_results.json', 'Sector Options Rotation v1'),
        ('sensitivity_analysis_v1_results.json', 'Sensitivity Analysis v1'),
        ('small_account_options_v1_results.json', 'Small Account Options'),
        ('earnings_options_v1_results.json', 'Earnings Options'),
        ('bootstrap_stress_test_v1_results.json', 'Bootstrap Stress Test'),
        ('day_of_week_timing_v1_results.json', 'Day of Week Timing'),
        ('vix_enhanced_meanrev_v1_results.json', 'VIX Enhanced Mean-Rev'),
    ]

    for fname, label in patterns:
        fpath = findings / fname
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text())
                trade_files[label] = data
            except:
                fprint(f"  ⚠️ Could not load {fname}")

    return trade_files


def main():
    fprint("=" * 70)
    fprint(f"SHARPE CORRECTION AUDIT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)
    fprint()
    fprint("ISSUE: Original scripts compute Sharpe as monthly_pnl / INITIAL capital.")
    fprint("This inflates Sharpe when capital compounds ($645 -> $39K).")
    fprint("This audit recalculates using monthly_pnl / CURRENT equity (correct).")
    fprint()

    exp_name = "sharpe_correction_audit_v1"
    if MLFLOW_OK:
        mlflow.set_experiment(exp_name)
        run = mlflow.start_run(run_name="sharpe_correction_audit")

    # Load all result files
    data = load_trade_logs()
    fprint(f"Found {len(data)} strategy result files to audit")
    fprint()

    all_results = {}

    for label, result_data in sorted(data.items()):
        fprint(f"{'='*60}")
        fprint(f"AUDITING: {label}")
        fprint(f"{'='*60}")

        # Try to find trade data in the result
        variants = {}

        if isinstance(result_data, dict):
            # Check if it has variant results
            for key, val in result_data.items():
                if isinstance(val, dict) and 'trades' in val:
                    variants[key] = val['trades']
                elif isinstance(val, dict) and 'trade_log' in val:
                    variants[key] = val['trade_log']
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict) and 'pnl' in val[0]:
                    variants[key] = val

            # Also check top-level trades
            if 'trades' in result_data:
                variants['main'] = result_data['trades']
            if 'trade_log' in result_data:
                variants['main'] = result_data['trade_log']

            # Check for nested variants
            if 'variants' in result_data:
                for vname, vdata in result_data['variants'].items():
                    if isinstance(vdata, dict):
                        if 'trades' in vdata:
                            variants[vname] = vdata['trades']
                        elif 'trade_log' in vdata:
                            variants[vname] = vdata['trade_log']

            # Also check 'results' key
            if 'results' in result_data:
                for rname, rdata in result_data['results'].items():
                    if isinstance(rdata, dict):
                        if 'trades' in rdata:
                            variants[rname] = rdata['trades']

        if not variants:
            fprint(f"  ⚠️ No trade data found in results file")
            fprint(f"  Keys available: {list(result_data.keys()) if isinstance(result_data, dict) else 'not a dict'}")

            # If we have summary metrics but no trades, report what we know
            if isinstance(result_data, dict):
                for key, val in result_data.items():
                    if isinstance(val, dict) and 'sharpe' in val:
                        orig_sh = val.get('sharpe', 0)
                        orig_wr = val.get('win_rate', val.get('wr', 0))
                        fprint(f"  {key}: Original Sharpe={orig_sh}, WR={orig_wr}%")
                        fprint(f"    ⚠️ Cannot verify without trade-level data")
            fprint()
            continue

        strategy_results = {}
        for vname, trades in sorted(variants.items()):
            if not trades or len(trades) < 5:
                fprint(f"  {vname}: Too few trades ({len(trades) if trades else 0})")
                continue

            result = correct_sharpe_from_trades(trades)
            if not result.get('valid'):
                fprint(f"  {vname}: Invalid results")
                continue

            strategy_results[vname] = result

            inflation = result['inflation_ratio']
            verdict = "✅ HONEST" if inflation < 2.0 else "⚠️ INFLATED" if inflation < 5.0 else "❌ HIGHLY INFLATED"

            fprint(f"\n  {vname}:")
            fprint(f"    Sharpe (CORRECT):  {result['sharpe_correct']}")
            fprint(f"    Sharpe (inflated): {result['sharpe_inflated']}")
            fprint(f"    Inflation ratio:   {inflation}x {verdict}")
            fprint(f"    Sortino (correct): {result['sortino_correct']}")
            fprint(f"    WR: {result['win_rate']}% | PF: {result['profit_factor']} | CAGR: {result['cagr_pct']}%")
            fprint(f"    MaxDD: {result['maxdd_pct']}% | Final: ${result['final_equity']:.0f}")
            fprint(f"    R1 gap: {result['r1_gap']} | Perm p: {result['perm_p']}")
            fprint(f"    Sub-period: H1={result['h1_sr']}, H2={result['h2_sr']} {'✅' if result['sub_period_stable'] else '❌'}")
            fprint(f"    2x cost survival: {'✅' if result['cost_survives_2x'] else '❌'}")
            fprint(f"    Gates: {result['gates']}/4 ({result['gate_details']})")

            if MLFLOW_OK:
                try:
                    mlflow.log_metric(f"{vname}_sharpe_correct", result['sharpe_correct'])
                    mlflow.log_metric(f"{vname}_sharpe_inflated", result['sharpe_inflated'])
                    mlflow.log_metric(f"{vname}_inflation_ratio", inflation)
                    mlflow.log_metric(f"{vname}_gates", result['gates'])
                except: pass

        all_results[label] = strategy_results
        fprint()

    # SUMMARY
    fprint("\n" + "=" * 70)
    fprint("SUMMARY: HONEST vs INFLATED SHARPE ACROSS ALL STRATEGIES")
    fprint("=" * 70)
    fprint(f"\n{'Strategy':<45} {'Inflated':>10} {'Correct':>10} {'Ratio':>8} {'Gates':>7} {'Verdict':>15}")
    fprint("-" * 95)

    all_honest = []
    for label, variants in sorted(all_results.items()):
        for vname, r in sorted(variants.items(), key=lambda x: x[1].get('sharpe_correct', 0), reverse=True):
            name = f"{label[:25]}:{vname[:18]}"
            inflation = r['inflation_ratio']
            verdict = "✅ HONEST" if inflation < 2.0 else "⚠️ INFLATED" if inflation < 5.0 else "❌ BAD"
            fprint(f"{name:<45} {r['sharpe_inflated']:>10.2f} {r['sharpe_correct']:>10.2f} {inflation:>7.1f}x {r['gates']:>5}/4  {verdict:>15}")
            all_honest.append({
                'strategy': f"{label}:{vname}",
                'sharpe_correct': r['sharpe_correct'],
                'sharpe_inflated': r['sharpe_inflated'],
                'inflation': inflation,
                'gates': r['gates'],
                'win_rate': r['win_rate'],
                'cagr_pct': r['cagr_pct'],
                'maxdd_pct': r['maxdd_pct']
            })

    # Save results
    output = {
        'audit_date': datetime.now().isoformat(),
        'issue': 'Monthly Sharpe computed on initial capital inflates as account compounds',
        'fix': 'Use equity at start of month as denominator',
        'detailed_results': {k: {vn: vr for vn, vr in v.items()} for k, v in all_results.items()},
        'summary': sorted(all_honest, key=lambda x: x['sharpe_correct'], reverse=True)
    }
    out_path = RESULTS_DIR / 'sharpe_correction_audit_results.json'
    out_path.write_text(json.dumps(output, indent=2, default=str))
    fprint(f"\n\nResults saved.")

    # Key findings
    if all_honest:
        best = max(all_honest, key=lambda x: x['sharpe_correct'])
        worst_inflation = max(all_honest, key=lambda x: x['inflation'])
        avg_inflation = np.mean([x['inflation'] for x in all_honest])

        fprint(f"\n{'='*70}")
        fprint("KEY FINDINGS")
        fprint(f"{'='*70}")
        fprint(f"  Strategies audited: {len(all_honest)}")
        fprint(f"  Average inflation ratio: {avg_inflation:.1f}x")
        fprint(f"  Worst inflation: {worst_inflation['strategy']} ({worst_inflation['inflation']:.1f}x)")
        fprint(f"  Best HONEST Sharpe: {best['strategy']} = {best['sharpe_correct']:.2f}")
        fprint(f"  Strategies passing 4/4 gates (honest): {sum(1 for x in all_honest if x['gates'] >= 4)}/{len(all_honest)}")
        fprint(f"  Strategies passing 3/4+ gates: {sum(1 for x in all_honest if x['gates'] >= 3)}/{len(all_honest)}")

        if MLFLOW_OK:
            mlflow.log_metric("avg_inflation_ratio", avg_inflation)
            mlflow.log_metric("n_strategies_audited", len(all_honest))
            mlflow.log_metric("n_pass_4_gates", sum(1 for x in all_honest if x['gates'] >= 4))
            mlflow.log_metric("best_honest_sharpe", best['sharpe_correct'])

    if MLFLOW_OK:
        try: mlflow.end_run()
        except: pass

    fprint("\nAUDIT COMPLETE.")

if __name__ == '__main__':
    main()
