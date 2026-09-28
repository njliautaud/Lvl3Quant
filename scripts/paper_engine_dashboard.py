#!/usr/bin/env python3
"""
Paper Engine Dashboard — Compare V7/V8/V9 sector rotation paper engines.
Reads state files and produces a compact comparison table.
Run manually or via cron after paper engines execute (4:35 PM ET weekdays).
"""

import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path('/home/jupiter/Lvl3Quant')
STATE_DIR = ROOT / 'state'

ENGINES = {
    'V7 (2% OTM, DTE=21)': 'sector_combined_v7_paper_state.json',
    'V8 (2% OTM, DTE=14)': 'sector_combined_v8_paper_state.json',
    'V9 (3% OTM, DTE=21)': 'sector_combined_v9_paper_state.json',
    'V9.1 (3%, DTE=28)': 'sector_combined_v91_paper_state.json',
    'V9.2 (monthly, DTE=28)': 'sector_combined_v92_paper_state.json',
    'V9.3 (monthly+50%PT)': 'sector_combined_v93_paper_state.json',
    'V10 (mom+earn avg)': 'sector_combined_v10_paper_state.json',
    'V10 (8pos,4%,30%PT)': 'sector_combined_v10_optimal_paper_state.json',
    'Earn (14feat, DTE=21)': 'sector_earnings_standalone_paper_state.json',
}


def load_state(filename):
    path = STATE_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_metrics(state):
    """Compute summary metrics from state."""
    if not state:
        return None

    equity = state.get('equity', 0)
    initial = 645.0
    pnl = equity - initial
    pnl_pct = (pnl / initial) * 100 if initial > 0 else 0

    open_pos = state.get('open_positions', [])
    closed = state.get('closed_trades', [])
    risk_parity = state.get('risk_parity_positions', [])

    # Count bulls vs bears
    bulls = [p for p in open_pos if p.get('mode') == 'bull']
    bears = [p for p in open_pos if p.get('mode') == 'bear']

    # Total cost of open positions
    open_cost = sum(p.get('cost', 0) for p in open_pos)
    rp_cost = sum(p.get('cost', 0) for p in risk_parity)

    # Win rate from closed trades
    wins = state.get('wins', 0)
    losses = state.get('losses', 0)
    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0

    # Total realized PnL
    realized_pnl = state.get('total_pnl', 0)

    return {
        'equity': equity,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'open_positions': len(open_pos),
        'bulls': len(bulls),
        'bears': len(bears),
        'open_cost': open_cost,
        'rp_cost': rp_cost,
        'closed_trades': len(closed),
        'wins': wins,
        'losses': losses,
        'win_rate': wr,
        'realized_pnl': realized_pnl,
        'config_version': state.get('config_version', '?'),
        'last_rebalance': state.get('last_rebalance', '?'),
        'days_since_rebalance': state.get('days_since_rebalance', 0),
        'vix_mode': state.get('current_mode', '?'),
    }


def print_comparison():
    print(f"\n{'='*70}")
    print(f"  SECTOR ROTATION PAPER ENGINE DASHBOARD")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*70}\n")

    results = {}
    for label, filename in ENGINES.items():
        state = load_state(filename)
        metrics = compute_metrics(state)
        if metrics:
            results[label] = metrics

    if not results:
        print("No paper engine states found.")
        return

    # Header
    labels = list(results.keys())
    col_w = 22
    print(f"{'Metric':<25}", end='')
    for label in labels:
        print(f"{label:>{col_w}}", end='')
    print()
    print('-' * (25 + col_w * len(labels)))

    # Rows
    rows = [
        ('Equity', 'equity', '${:.2f}'),
        ('P&L', 'pnl', '${:.2f}'),
        ('P&L %', 'pnl_pct', '{:.1f}%'),
        ('Open Positions', 'open_positions', '{:.0f}'),
        ('  Bulls', 'bulls', '{:.0f}'),
        ('  Bears', 'bears', '{:.0f}'),
        ('Open Cost', 'open_cost', '${:.2f}'),
        ('Risk Parity Cost', 'rp_cost', '${:.2f}'),
        ('Closed Trades', 'closed_trades', '{:.0f}'),
        ('Win Rate', 'win_rate', '{:.1f}%'),
        ('Realized P&L', 'realized_pnl', '${:.2f}'),
        ('Days Since Rebal', 'days_since_rebalance', '{:.0f}'),
        ('VIX Mode', 'vix_mode', '{}'),
        ('Last Rebalance', 'last_rebalance', '{}'),
    ]

    for row_label, key, fmt in rows:
        print(f"{row_label:<25}", end='')
        for label in labels:
            val = results[label].get(key, 0)
            try:
                formatted = fmt.format(val)
            except (ValueError, TypeError):
                formatted = str(val)
            print(f"{formatted:>{col_w}}", end='')
        print()

    # Position details
    print(f"\n{'='*70}")
    print("POSITION DETAILS")
    print(f"{'='*70}")

    for label, filename in ENGINES.items():
        state = load_state(filename)
        if not state:
            continue
        print(f"\n  {label}:")
        for p in state.get('open_positions', []):
            ticker = p.get('ticker', '?')
            mode = p.get('mode', '?')
            k1 = p.get('K1', '?')
            k2 = p.get('K2', '?')
            cost = p.get('cost', 0)
            lgbm = p.get('lgbm_score', 0)
            print(f"    {ticker:>4} {mode:>4}  K1={k1} K2={k2}  cost=${cost:.2f}  lgbm={lgbm:.3f}")
        for rp in state.get('risk_parity_positions', []):
            ticker = rp.get('ticker', '?')
            shares = rp.get('shares', 0)
            cost = rp.get('cost', 0)
            print(f"    {ticker:>4}   RP  shares={shares}  cost=${cost:.2f}")

    print()


def discord_summary():
    """Return a short Discord-friendly summary."""
    results = {}
    for label, filename in ENGINES.items():
        state = load_state(filename)
        metrics = compute_metrics(state)
        if metrics:
            results[label] = metrics

    if not results:
        return "No paper engine data available."

    lines = ["**Paper Engine Comparison**"]
    for label, m in results.items():
        short = label.split('(')[0].strip()
        lines.append(
            f"• {short}: ${m['equity']:.0f} ({m['pnl_pct']:+.1f}%), "
            f"{m['closed_trades']} trades, WR {m['win_rate']:.0f}%"
        )
    return '\n'.join(lines)


if __name__ == '__main__':
    if '--discord' in sys.argv:
        print(discord_summary())
    else:
        print_comparison()
