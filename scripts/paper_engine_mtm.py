#!/usr/bin/env python3
"""
Paper Engine Mark-to-Market Dashboard
======================================
Reads all paper engine state files, fetches current prices for open positions,
and produces a consolidated performance summary.

Usage:
    python scripts/paper_engine_mtm.py              # console output
    python scripts/paper_engine_mtm.py --json        # JSON output
    python scripts/paper_engine_mtm.py --discord      # send to Discord
"""
import json
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
STATE_DIR = BASE / "state"


def load_engine_states():
    """Load all paper engine state files and extract positions."""
    engines = {}

    state_files = sorted(set(
        glob.glob(str(STATE_DIR / "*paper*.json")) +
        glob.glob(str(STATE_DIR / "*_state.json"))
    ))

    skip_names = {
        'signal_watcher', 'vmr', 'play_scanner', 'consensus',
        'expanded_alerter', 'gameplan_v2', 'gameplan_v3', 'gameplan_v4_4',
        'vol_harvest', 'contrarian_signals', 'agentic_v10',
    }

    for f in state_files:
        try:
            name = os.path.basename(f).replace('_paper_state.json', '').replace('_state.json', '').replace('.json', '')
            if name in skip_names:
                continue

            with open(f) as fh:
                d = json.load(fh)

            equity = d.get('equity', d.get('starting_equity', d.get('initial_equity', None)))
            cash = d.get('cash', None)
            total_pnl = d.get('total_pnl', d.get('realized_pnl', 0))
            wins = d.get('wins', d.get('win_count', 0))
            losses = d.get('losses', d.get('loss_count', 0))
            open_pos = d.get('open_positions', [])
            closed = d.get('closed_trades', [])

            n_open = len(open_pos) if isinstance(open_pos, (list, dict)) else 0
            n_closed = len(closed) if isinstance(closed, list) else 0

            if equity is None and total_pnl == 0 and n_open == 0 and n_closed == 0:
                continue

            # Extract open position details
            positions = []
            if isinstance(open_pos, list):
                for p in open_pos:
                    pos = {
                        'ticker': p.get('ticker', p.get('symbol', '?')),
                        'type': p.get('option_type', p.get('type', p.get('direction', 'unknown'))),
                        'entry_cost': p.get('entry_cost', p.get('cost', p.get('premium', 0))),
                        'entry_date': p.get('entry_date', '?'),
                    }
                    positions.append(pos)

            engines[name] = {
                'equity': equity,
                'cash': cash,
                'realized_pnl': total_pnl or 0,
                'wins': wins or 0,
                'losses': losses or 0,
                'n_open': n_open,
                'n_closed': n_closed,
                'positions': positions,
                'mtime': datetime.fromtimestamp(os.path.getmtime(f)).strftime('%m/%d %H:%M'),
                'file': f,
            }
        except Exception as e:
            pass

    return engines


def format_dashboard(engines):
    """Format engines into a readable dashboard."""
    lines = []
    lines.append(f"Paper Engine Dashboard — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    lines.append(f"{'='*75}")

    # Categorize
    active = {k: v for k, v in engines.items() if v['n_open'] > 0 or v['n_closed'] > 0}
    idle = {k: v for k, v in engines.items() if v['n_open'] == 0 and v['n_closed'] == 0}

    if active:
        lines.append(f"\n📈 ACTIVE ENGINES ({len(active)}):")
        lines.append(f"{'Name':<35} {'Equity':>8} {'Realized':>8} {'W/L':>6} {'Open':>4} {'Closed':>6} {'Updated':>12}")
        lines.append("-" * 85)

        for name, eng in sorted(active.items(), key=lambda x: x[1]['realized_pnl'], reverse=True):
            eq = f"${eng['equity']:.0f}" if eng['equity'] else "--"
            pnl = f"${eng['realized_pnl']:.0f}" if eng['realized_pnl'] else "$0"
            wl = f"{eng['wins']}/{eng['losses']}"
            lines.append(f"{name:<35} {eq:>8} {pnl:>8} {wl:>6} {eng['n_open']:>4} {eng['n_closed']:>6} {eng['mtime']:>12}")

            # Show open positions
            for p in eng['positions'][:3]:  # Max 3 per engine
                cost = f"${p['entry_cost']:.0f}" if p['entry_cost'] else "?"
                lines.append(f"  └─ {p['ticker']:<6} {p['type']:<15} cost={cost}")

    if idle:
        lines.append(f"\n💤 IDLE ENGINES ({len(idle)}):")
        idle_names = ', '.join(sorted(idle.keys()))
        lines.append(f"  {idle_names}")

    # Summary
    total_open = sum(v['n_open'] for v in engines.values())
    total_closed = sum(v['n_closed'] for v in engines.values())
    total_realized = sum(v['realized_pnl'] for v in engines.values())
    lines.append(f"\n{'='*75}")
    lines.append(f"TOTAL: {len(engines)} engines | {total_open} open positions | {total_closed} closed trades | ${total_realized:.0f} realized P&L")

    return '\n'.join(lines)


def main():
    engines = load_engine_states()

    if '--json' in sys.argv:
        print(json.dumps(engines, indent=2, default=str))
    else:
        print(format_dashboard(engines))


if __name__ == '__main__':
    main()
