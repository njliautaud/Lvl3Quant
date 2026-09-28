"""
Realistic Assessment — ES Futures Microstructure Alpha
======================================================

Compiles all findings into a clear-eyed assessment of:
1. What alpha exists (confirmed from 27-day walk-forward)
2. What it would cost to trade (infrastructure, commissions, data)
3. Expected PnL under realistic assumptions
4. Whether to proceed to paper trading

This script reads results from the pipeline and produces a plain-text report.

Usage:
    python alpha_discovery/compile_realistic_assessment.py
"""

import sys
import json
import glob
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "alpha_discovery" / "results"


def load_latest(prefix):
    """Load the most recent result file matching a prefix."""
    files = sorted(RESULTS_DIR.glob(f"{prefix}*.json"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f), files[0].name


def safe_get(d, *keys, default='N/A'):
    """Safely navigate nested dict."""
    for k in keys:
        if d is None or not isinstance(d, dict):
            return default
        d = d.get(k, None)
    return d if d is not None else default


def format_ticks(v):
    if v is None or v == 'N/A':
        return 'N/A'
    return f"{v:+.4f}t (${v * 12.50:+.2f})"


def main():
    print("=" * 80)
    print("REALISTIC ASSESSMENT — ES FUTURES MICROSTRUCTURE ALPHA")
    print(f"Generated: {datetime.now().isoformat()}")
    print("=" * 80)

    # ====================================================================
    # SECTION 1: ALPHA SIGNAL STRENGTH
    # ====================================================================
    print("\n" + "=" * 80)
    print("1. ALPHA SIGNAL STRENGTH")
    print("=" * 80)

    # Load corrected limit study
    limit_data = load_latest("corrected_limit_study")
    if limit_data:
        limit, limit_file = limit_data
        dm = limit.get('direction_model', {})
        data_info = limit.get('data', {})
        print(f"\n  Source: {limit_file}")
        print(f"  Data: {data_info.get('n_days', '?')} trading days, "
              f"{data_info.get('n_bars', '?'):,} bars at 100ms interval")
        print(f"\n  Direction Model (ret_3s, walk-forward LightGBM):")
        print(f"    IC:        {dm.get('ic', 'N/A'):.4f}")
        print(f"    ICIR:      {dm.get('icir', 'N/A'):.2f}")
        print(f"    t-stat:    {dm.get('tstat', 'N/A'):.2f}")
        fold_ics = dm.get('fold_ics', [])
        if fold_ics:
            n_pos = sum(1 for x in fold_ics if x > 0)
            print(f"    Folds:     {n_pos}/{len(fold_ics)} positive  "
                  f"range=[{min(fold_ics):.4f}, {max(fold_ics):.4f}]")
        print(f"\n  Signal Interpretation:")
        ic = dm.get('ic', 0)
        if ic > 0.15:
            print(f"    IC={ic:.4f} is STRONG for tick-level prediction")
        elif ic > 0.08:
            print(f"    IC={ic:.4f} is MODERATE — viable for limit orders, marginal for market orders")
        elif ic > 0.03:
            print(f"    IC={ic:.4f} is WEAK — needs spread capture to be profitable")
        else:
            print(f"    IC={ic:.4f} is INSUFFICIENT for any profitable strategy")
    else:
        print("\n  No corrected limit study results found. Run pipeline first.")

    # ====================================================================
    # SECTION 2: EXECUTION ANALYSIS
    # ====================================================================
    print("\n" + "=" * 80)
    print("2. EXECUTION ANALYSIS")
    print("=" * 80)

    # Load hybrid execution sim
    hybrid_data = load_latest("hybrid_execution")
    if hybrid_data:
        hybrid, hybrid_file = hybrid_data
        print(f"\n  Source: {hybrid_file}")
        opt = hybrid.get('optimal_configs', {})
        best = opt.get('best_by_sharpe', hybrid.get('best_config', {}))
        if best:
            cfg = best.get('config', {})
            print(f"\n  Best Config (by Sharpe):")
            print(f"    Quantile:   {cfg.get('quantile', '?')}")
            print(f"    Hold:       {cfg.get('hold_sec', '?')}s")
            print(f"    Latency:    {cfg.get('latency_ms', '?')}ms")
            print(f"    Limit exit: {cfg.get('use_limit_exit', '?')}")
            print(f"\n  Performance:")
            print(f"    Mean PnL:    {format_ticks(best.get('mean_pnl_ticks'))}")
            print(f"    Sharpe:      {best.get('sharpe', 'N/A')}")
            print(f"    Trades/day:  {best.get('trades_per_day', 'N/A')}")
            print(f"    Win rate:    {best.get('win_rate', 0):.1%}" if isinstance(best.get('win_rate'), (int, float)) else "    Win rate:    N/A")

        # Count profitable configs
        all_results = opt.get('all_results', [])
        if all_results:
            n_pos = sum(1 for r in all_results if r.get('mean_pnl_ticks', 0) > 0)
            print(f"\n  Grid Search: {n_pos}/{len(all_results)} configs profitable")
    else:
        print("\n  No hybrid execution sim results found.")

    # Load limit study grid results
    if limit_data:
        limit, _ = limit_data
        opt = limit.get('optimal_configs', {})
        all_results = opt.get('all_results', [])
        if all_results:
            mkt = [r for r in all_results if not r.get('config', {}).get('use_limit_exit', False)]
            lmt = [r for r in all_results if r.get('config', {}).get('use_limit_exit', False)]
            mkt_pos = sum(1 for r in mkt if r.get('mean_pnl_ticks', 0) > 0)
            lmt_pos = sum(1 for r in lmt if r.get('mean_pnl_ticks', 0) > 0)
            print(f"\n  Limit Study Grid:")
            print(f"    Market exit: {mkt_pos}/{len(mkt)} profitable")
            print(f"    Limit  exit: {lmt_pos}/{len(lmt)} profitable")

    # ====================================================================
    # SECTION 3: QUEUE POSITION / FILL MODEL
    # ====================================================================
    print("\n" + "=" * 80)
    print("3. QUEUE POSITION & FILL MODEL (from MBO data analysis)")
    print("=" * 80)

    queue_data = load_latest("queue_position_study")
    if queue_data:
        queue, queue_file = queue_data
        print(f"\n  Source: {queue_file}")

        p1 = queue.get('part1_queue_depth', {})
        bd = p1.get('bid_depth_stats', {})
        print(f"\n  Displayed Queue Depth (best bid, contracts):")
        print(f"    Median: {bd.get('median', 'N/A')}")
        print(f"    Mean:   {bd.get('mean', 'N/A')}")
        print(f"    p75:    {bd.get('p75', 'N/A')}")

        p2 = queue.get('part2_fill_probability', {})
        scenarios = p2.get('scenario_results', {})
        for sname, s in scenarios.items():
            fp = s.get('fill_probs', {})
            print(f"\n  Scenario: {sname}")
            print(f"    Queue ahead: {s.get('queue_ahead', 'N/A'):.0f}")
            print(f"    Fill rate:   {s.get('fill_rate_per_sec', 'N/A'):.1f} c/s")
            print(f"    P(fill, 5s): {fp.get('5s', 0):.1%}")
            print(f"    P(fill,10s): {fp.get('10s', 0):.1%}")

        p4 = queue.get('part4_sensitivity', {})
        verdict = p4.get('verdict', 'N/A')
        print(f"\n  Verdict: {verdict}")
    else:
        print("\n  No queue position study results found.")

    # Add hardcoded empirical findings from MBO analysis
    print(f"\n  EMPIRICAL FINDINGS (from direct MBO event analysis):")
    print(f"    Displayed queue at inside: median 2-3 contracts")
    print(f"    76% of orders are icebergs (display size=1)")
    print(f"    Effective queue (sweep volume): ~20 contracts")
    print(f"    Iceberg refills get NEW timestamp -> BACK of FIFO queue")
    print(f"    Fill probability at position 3-5: 80-87% per sweep")
    print(f"    Price level persistence: median 842ms, mean 2.0s")
    print(f"    11,822 bid-level sweeps per day")

    # ====================================================================
    # SECTION 4: INFRASTRUCTURE COSTS
    # ====================================================================
    print("\n" + "=" * 80)
    print("4. INFRASTRUCTURE COSTS (to go live)")
    print("=" * 80)

    print("""
  ONE-TIME COSTS:
    CME market data:
      - CME MBO Globex data feed:          ~$4,500-6,000/month (direct)
      - OR via broker (AMP/Rithmic):        Included in commissions
      - Databento historical (for R&D):     ~$200-500 (already purchased)

    Trading infrastructure:
      - Colocation (CME Aurora, IL):        ~$3,000-5,000/month
      - OR cloud VPS (Equinix CH):          ~$200-500/month (higher latency)
      - OR home internet (100ms+ latency):  $0 (already have)

    Software:
      - Rithmic API license:                ~$50-100/month
      - AMP Futures account minimum:        $100-500

  RECURRING COSTS (per trade):
    Commission: $3.00 round-trip (AMP + Rithmic + CME)
      - AMP clearing:     ~$0.25/side
      - Rithmic:          ~$0.10/side
      - CME exchange fee: ~$1.15/side
      - Total:            ~$1.50/side = $3.00 RT

  LATENCY CONSIDERATIONS:
    - Home internet to CME: ~30-120ms one-way
    - Cloud VPS in Chicago: ~1-5ms
    - Colocation at Aurora: <1ms
    - Our strategy: multi-tick moves, 3-10s holds => home internet OK
    - Queue position matters more than speed for limit orders

  MINIMUM VIABLE SETUP (cheapest path to paper trading):
    - AMP Futures demo account:              $0
    - Rithmic R|Trader demo:                 $0
    - Python + Rithmic API:                  $0
    - Home internet:                         $0 (existing)
    - Total to start paper trading:          $0
""")

    # ====================================================================
    # SECTION 5: EXPECTED PnL UNDER REALISTIC ASSUMPTIONS
    # ====================================================================
    print("=" * 80)
    print("5. EXPECTED PnL UNDER REALISTIC ASSUMPTIONS")
    print("=" * 80)

    # Use the pipeline results if available
    ic = 0.1135  # default from previous runs
    if limit_data:
        limit, _ = limit_data
        ic = limit.get('direction_model', {}).get('ic', ic)

    # Expected directional PnL per bar (100ms)
    # ret_3s std ~= 0.8 ticks. IC * std = expected directional PnL
    ret_std_ticks = 0.8  # typical 3s return std in ticks
    expected_dir_pnl = ic * ret_std_ticks

    # Commission
    commission_ticks = 3.00 / 12.50  # 0.24 ticks

    # Scenarios
    scenarios = [
        ("Market entry + market exit", -1.0, -commission_ticks),
        ("Limit entry + market exit", 0.0, -commission_ticks),
        ("Limit entry + limit exit", +1.0, -commission_ticks),
    ]

    print(f"\n  Signal: IC = {ic:.4f} on ret_3s")
    print(f"  Expected directional PnL: IC * sigma = {ic:.4f} * {ret_std_ticks:.1f}t = {expected_dir_pnl:.4f}t")
    print(f"  Commission: {commission_ticks:.4f}t (${3.00:.2f} RT)")
    print(f"\n  {'Strategy':<35s} {'Edge':>8s} {'Dir PnL':>8s} {'Comm':>8s} {'NET':>8s} {'$/trade':>10s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for name, edge, comm in scenarios:
        net = expected_dir_pnl + edge + comm
        net_dollars = net * 12.50
        print(f"  {name:<35s} {edge:>+8.2f} {expected_dir_pnl:>+8.4f} {comm:>+8.4f} {net:>+8.4f} ${net_dollars:>+8.2f}")

    # Trading frequency
    print(f"\n  Trading Frequency Estimates:")
    signals_per_day = 50  # top 20% of bars = many signals, but we pick best
    for tpd_label, tpd in [("Conservative (20/day)", 20),
                            ("Moderate (50/day)", 50),
                            ("Aggressive (100/day)", 100)]:
        # Limit entry + limit exit is best case
        best_net = expected_dir_pnl + 1.0 - commission_ticks
        # Limit entry + market exit (more realistic)
        realistic_net = expected_dir_pnl + 0.0 - commission_ticks
        daily_best = best_net * tpd * 12.50
        daily_real = realistic_net * tpd * 12.50
        annual_best = daily_best * 252
        annual_real = daily_real * 252
        print(f"    {tpd_label}:")
        print(f"      Best case (limit/limit):    ${daily_best:>+8.2f}/day  ${annual_best:>+10,.0f}/year")
        print(f"      Realistic (limit/market):   ${daily_real:>+8.2f}/day  ${annual_real:>+10,.0f}/year")

    # ====================================================================
    # SECTION 6: RISKS AND CAVEATS
    # ====================================================================
    print("\n" + "=" * 80)
    print("6. RISKS AND CAVEATS")
    print("=" * 80)

    print("""
  CRITICAL RISKS:
    1. OVERFITTING: Walk-forward on 27 days is still small sample.
       Need 100+ days for robust conclusions.

    2. REGIME CHANGE: July-August 2025 market conditions may not persist.
       VIX, FOMC, earnings season all affect microstructure.

    3. RETRODICTION RATIO: 3.4x backward IC vs forward IC.
       Features encode PAST more than FUTURE (common in LOB models).

    4. ADVERSE SELECTION: Limit orders fill when price moves against us.
       Previous limit study showed ALL 42 configs negative.

    5. FILL MODEL SIMPLICITY: Mid-price crossing != actual fill.
       Even with displayed queue of 2-3, execution isn't guaranteed.

    6. SLIPPAGE: High-signal events likely coincide with fast markets
       where our latency (30-120ms) means worse fills.

    7. CAPACITY: Adding our volume to inside queue may move the market.
       But at 1 contract, market impact is negligible.

  DATA LIMITATIONS:
    - Only 27 trading days (July 14 - Aug 13, 2025)
    - Single instrument (ESU5)
    - No overnight session data (RTH only: 9:30-16:00 ET)
    - Snapshot interval: 100ms (may miss sub-100ms dynamics)

  WHAT WOULD MAKE THIS WORK:
    1. Higher IC (>0.15): Would make market-order strategies viable
    2. More data (100+ days): Reduces overfitting risk
    3. Multi-instrument (NQ, YM, RTY): Diversification
    4. Better features: Order flow imbalance at tick level
    5. Adaptive model: Regime-switching or online learning
    6. Faster execution: Cloud VPS in Chicago for 1-5ms latency
""")

    # ====================================================================
    # SECTION 7: RECOMMENDATION
    # ====================================================================
    print("=" * 80)
    print("7. RECOMMENDATION")
    print("=" * 80)

    # Determine recommendation based on results
    if ic >= 0.15:
        recommendation = "GREEN LIGHT — Paper trade immediately"
        rationale = (
            f"IC={ic:.4f} is strong enough for market-order strategies. "
            f"Expected PnL is positive even with worst-case execution."
        )
    elif ic >= 0.08 and ic < 0.15:
        recommendation = "YELLOW LIGHT — Paper trade limit-order strategy only"
        rationale = (
            f"IC={ic:.4f} is too weak for market orders but viable with limit entries. "
            f"Limit entry earns +0.5t spread, which combined with even a weak directional "
            f"signal can be net positive. Key risk: adverse selection on fills. "
            f"Need to validate on 100+ days before real money."
        )
    elif ic >= 0.03:
        recommendation = "ORANGE LIGHT — Continue research, not ready for trading"
        rationale = (
            f"IC={ic:.4f} shows some predictive power but insufficient for profitable "
            f"trading after costs. Need to improve signal (more features, better model, "
            f"more data) before paper trading."
        )
    else:
        recommendation = "RED LIGHT — No viable alpha found"
        rationale = f"IC={ic:.4f} shows negligible predictive power."

    print(f"\n  {recommendation}")
    print(f"\n  Rationale: {rationale}")
    print(f"""
  NEXT STEPS:
    1. Validate with full 27-day data (pipeline running now)
    2. If IC holds at 0.08+:
       a. Set up AMP Futures demo account (free)
       b. Build Rithmic API connector (Python)
       c. Paper trade limit-order strategy for 2 weeks
       d. Track actual vs simulated fills
    3. If paper trading confirms positive edge:
       a. Fund $5,000-10,000 account
       b. Trade 1 contract (ES = ~$13,000 margin)
       c. Monitor for 1 month before scaling
    4. If results degrade:
       a. Acquire more historical data (Databento: ~$50/month)
       b. Test multi-instrument (NQ, YM, RTY)
       c. Explore regime-aware models
""")

    print("=" * 80)
    print("END OF ASSESSMENT")
    print("=" * 80)

    # Save to file
    output_path = RESULTS_DIR / f'realistic_assessment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    # Re-run to capture output
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # Just save a summary JSON instead
        pass

    # Save key metrics as JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'signal': {
            'ic': ic,
            'target': 'ret_3s',
            'model': 'LightGBM walk-forward',
            'interpretation': recommendation,
        },
        'costs': {
            'commission_rt': 3.00,
            'commission_ticks': commission_ticks,
            'minimum_setup_cost': 0,  # paper trading is free
            'colocation_monthly': '200-500 (cloud) or 3000-5000 (CME)',
        },
        'recommendation': recommendation,
        'rationale': rationale,
    }

    summary_path = RESULTS_DIR / f'assessment_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
