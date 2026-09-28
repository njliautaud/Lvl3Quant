"""
Compare Results: 16-day vs 27-day pipeline outputs.

Loads the most recent results from each pipeline run and produces
a side-by-side comparison of key metrics.

Usage:
    python alpha_discovery/compare_results.py
    python alpha_discovery/compare_results.py --baseline corrected_limit_study_20260217_101812.json
"""

import sys, json, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "alpha_discovery" / "results"


def load_latest_result(prefix: str) -> dict:
    """Load the most recent result file matching a prefix."""
    files = sorted(RESULTS_DIR.glob(f"{prefix}*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def format_pct(v, decimals=1):
    if v is None:
        return "N/A"
    return f"{v*100:.{decimals}f}%"


def format_ticks(v, decimals=3):
    if v is None:
        return "N/A"
    return f"{v:+.{decimals}f}t"


def format_dollars(v, decimals=2):
    if v is None:
        return "N/A"
    return f"${v:+.{decimals}f}"


def compare_limit_studies(baseline: dict, latest: dict):
    """Compare two corrected limit study results."""
    print("\n" + "=" * 70)
    print("CORRECTED LIMIT STUDY COMPARISON")
    print("=" * 70)

    # Data stats
    for label, d in [("Baseline (16d)", baseline), ("Latest (27d)", latest)]:
        data = d.get('data', {})
        print(f"\n  {label}:")
        print(f"    Bars: {data.get('n_bars', '?'):,}  Days: {data.get('n_days', '?')}")
        dm = d.get('direction_model', {})
        print(f"    IC: {dm.get('ic', '?'):.4f}  ICIR: {dm.get('icir', '?'):.2f}  "
              f"t-stat: {dm.get('tstat', '?'):.2f}")
        fold_ics = dm.get('fold_ics', [])
        if fold_ics:
            n_pos = sum(1 for x in fold_ics if x > 0)
            print(f"    Fold ICs: {n_pos}/{len(fold_ics)} positive  "
                  f"range=[{min(fold_ics):.3f}, {max(fold_ics):.3f}]")

    # Best configs
    print("\n  --- Best Configs by Sharpe ---")
    for label, d in [("Baseline", baseline), ("Latest", latest)]:
        opt = d.get('optimal_configs', {})
        best = opt.get('best_by_sharpe', {})
        cfg = best.get('config', {})
        print(f"\n  {label}: q{cfg.get('quantile', '?')*100:.0f} / "
              f"h{cfg.get('hold_sec', '?')}s / lat{cfg.get('latency_ms', '?')}ms"
              if cfg else f"\n  {label}: N/A")
        print(f"    PnL: {format_ticks(best.get('mean_pnl_ticks'))}  "
              f"Sharpe: {best.get('sharpe', '?'):.1f}  "
              f"TPD: {best.get('trades_per_day', '?'):.0f}  "
              f"Win: {format_pct(best.get('win_rate'))}")

    # Grid comparison: count profitable configs
    print("\n  --- Profitable Configs ---")
    for label, d in [("Baseline", baseline), ("Latest", latest)]:
        opt = d.get('optimal_configs', {})
        all_results = opt.get('all_results', [])
        if not all_results:
            print(f"  {label}: No grid results")
            continue

        mkt = [r for r in all_results if not r.get('config', {}).get('use_limit_exit', False)]
        lmt = [r for r in all_results if r.get('config', {}).get('use_limit_exit', False)]
        mkt_pos = sum(1 for r in mkt if r.get('mean_pnl_ticks', 0) > 0)
        lmt_pos = sum(1 for r in lmt if r.get('mean_pnl_ticks', 0) > 0)
        print(f"  {label}: Market exit: {mkt_pos}/{len(mkt)} profitable  "
              f"Limit exit: {lmt_pos}/{len(lmt)} profitable")

    # Hold period comparison
    print("\n  --- By Hold Period (market exit, all latencies/quantiles) ---")
    for label, d in [("Baseline", baseline), ("Latest", latest)]:
        all_results = d.get('optimal_configs', {}).get('all_results', [])
        mkt = [r for r in all_results if not r.get('config', {}).get('use_limit_exit', False)]
        holds = {}
        for r in mkt:
            h = r.get('config', {}).get('hold_sec', 0)
            if h not in holds:
                holds[h] = []
            holds[h].append(r)

        for h in sorted(holds.keys()):
            configs = holds[h]
            n_pos = sum(1 for r in configs if r.get('mean_pnl_ticks', 0) > 0)
            avg_pnl = np.mean([r.get('mean_pnl_ticks', 0) for r in configs])
            print(f"  {label} h{h}s: {n_pos}/{len(configs)} profitable  avg_pnl={format_ticks(avg_pnl)}")


def compare_queue_studies(baseline: dict, latest: dict):
    """Compare queue position studies."""
    print("\n" + "=" * 70)
    print("QUEUE POSITION STUDY COMPARISON")
    print("=" * 70)

    for label, d in [("Baseline", baseline), ("Latest", latest)]:
        if d is None:
            print(f"\n  {label}: No results available")
            continue
        p1 = d.get('part1_queue_depth', {})
        bd = p1.get('bid_depth_stats', {})
        p2 = d.get('part2_fill_probability', {})
        qs = p2.get('queue_at_entry_stats', {})
        fr = p2.get('fill_rate_estimates', {})

        print(f"\n  {label}:")
        print(f"    Bid depth: median={bd.get('median', '?'):.0f}  "
              f"mean={bd.get('mean', '?'):.1f}  "
              f"p75={bd.get('p75', '?'):.0f}")
        print(f"    Queue at entry: median={qs.get('median', '?'):.0f}  "
              f"mean={qs.get('mean', '?'):.1f}")
        print(f"    Fill rate: {fr.get('median', '?'):.1f} c/s")

        scenarios = p2.get('scenario_results', {})
        for sname, s in scenarios.items():
            fp = s.get('fill_probs', {})
            p5 = fp.get('5s', 0)
            print(f"    {sname}: P(fill,5s)={format_pct(p5)}")


def main():
    parser = argparse.ArgumentParser(description='Compare pipeline results')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Baseline result file name')
    args = parser.parse_args()

    # Load baseline (16-day)
    if args.baseline:
        baseline_path = RESULTS_DIR / args.baseline
        with open(baseline_path) as f:
            baseline_limit = json.load(f)
    else:
        # Find the first corrected_limit_study result (16-day)
        baseline_limit = None
        files = sorted(RESULTS_DIR.glob("corrected_limit_study_*.json"),
                       key=lambda f: f.stat().st_mtime)
        if files:
            with open(files[0]) as f:
                baseline_limit = json.load(f)
            print(f"Baseline: {files[0].name}")

    # Load latest (27-day)
    latest_limit = None
    files = sorted(RESULTS_DIR.glob("corrected_limit_study_*.json"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    if len(files) >= 2:
        with open(files[0]) as f:
            latest_limit = json.load(f)
        print(f"Latest: {files[0].name}")
    elif len(files) == 1:
        print(f"Only one result file found. Run pipeline with 27 days first.")
        latest_limit = baseline_limit

    if baseline_limit and latest_limit:
        compare_limit_studies(baseline_limit, latest_limit)

    # Queue position studies
    baseline_queue = load_latest_result("queue_position_study")
    compare_queue_studies(None, baseline_queue)

    # Summary
    print("\n" + "=" * 70)
    print("QUICK COMPARISON SUMMARY")
    print("=" * 70)
    if baseline_limit:
        bl_dm = baseline_limit.get('direction_model', {})
        bl_best = baseline_limit.get('optimal_configs', {}).get('best_by_sharpe', {})
        print(f"  Baseline: IC={bl_dm.get('ic', 0):.4f}  "
              f"Best: {format_ticks(bl_best.get('mean_pnl_ticks'))} @ "
              f"{bl_best.get('trades_per_day', 0):.0f} TPD  "
              f"Days: {baseline_limit.get('data', {}).get('n_days', '?')}")
    if latest_limit and latest_limit is not baseline_limit:
        lt_dm = latest_limit.get('direction_model', {})
        lt_best = latest_limit.get('optimal_configs', {}).get('best_by_sharpe', {})
        print(f"  Latest:   IC={lt_dm.get('ic', 0):.4f}  "
              f"Best: {format_ticks(lt_best.get('mean_pnl_ticks'))} @ "
              f"{lt_best.get('trades_per_day', 0):.0f} TPD  "
              f"Days: {latest_limit.get('data', {}).get('n_days', '?')}")


if __name__ == '__main__':
    main()
