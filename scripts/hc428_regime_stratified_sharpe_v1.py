#!/usr/bin/env python3
"""
HC #428 R1 — Regime-stratified Sharpe per (horizon, side, bucket) on the 34-day
baseline per-day stratification. Surfaces cells whose green/red Sharpe gap is
≤0.50 (the regime-agnostic gate) AND whose all-regime mean net-ticks is >0.

Reads:  output/closest_to_profit_v4/per_day_stratification.csv
Writes: output/hc428_regime_stratified_sharpe_v1/
          - regime_summary.csv (one row per cell)
          - passing_cells.csv  (cells passing both gates)
          - findings.md        (top-line summary)

Single-shot. No sub-agents. Pure pandas.
"""
import json
import math
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path('/home/jupiter/Lvl3Quant')
IN_CSV = BASE / 'output/closest_to_profit_v4/per_day_stratification.csv'
OUT = BASE / 'output/hc428_regime_stratified_sharpe_v1'
OUT.mkdir(parents=True, exist_ok=True)

COMMISSION_TICKS = 0.376  # ES passive-fill commission only (CLAUDE.md canonical)


def sharpe(x: pd.Series) -> float:
    if len(x) < 2:
        return float('nan')
    s = x.std(ddof=1)
    if s == 0 or math.isnan(s):
        return float('nan')
    return float(x.mean() / s * math.sqrt(252))  # daily-Sharpe annualized


def main() -> None:
    df = pd.read_csv(IN_CSV)
    # net_ticks already net of cost (per closest_to_profit_v4); use as-is.
    rows = []
    for (h, side, bucket), g in df.groupby(['horizon', 'side', 'bucket']):
        # Per-day net-ticks weighted by n
        per_day = g.groupby('date').apply(
            lambda d: float((d['mean_net_ticks'] * d['n']).sum() / d['n'].sum())
        )
        regime_map = g.drop_duplicates('date').set_index('date')['regime']
        per_day_df = pd.DataFrame({'net_ticks': per_day, 'regime': regime_map}).dropna()
        if per_day_df.empty:
            continue

        ovr_mean = per_day_df['net_ticks'].mean()
        ovr_sharpe = sharpe(per_day_df['net_ticks'])
        n_days = int(len(per_day_df))

        regime_stats = {}
        for r in ('green', 'red', 'flat'):
            sub = per_day_df.loc[per_day_df['regime'] == r, 'net_ticks']
            regime_stats[r] = {
                'n': int(len(sub)),
                'mean': float(sub.mean()) if len(sub) else float('nan'),
                'sharpe': sharpe(sub),
            }

        sg = regime_stats['green']['sharpe']
        sr = regime_stats['red']['sharpe']
        if not (math.isnan(sg) or math.isnan(sr)) and max(abs(sg), abs(sr)) > 0:
            regime_gap = abs(sg - sr) / max(abs(sg), abs(sr))
        else:
            regime_gap = float('nan')

        rows.append({
            'horizon': h, 'side': side, 'bucket': bucket, 'n_days': n_days,
            'mean_net_ticks': ovr_mean, 'sharpe': ovr_sharpe,
            'sharpe_green': sg, 'sharpe_red': sr, 'sharpe_flat': regime_stats['flat']['sharpe'],
            'mean_green': regime_stats['green']['mean'], 'mean_red': regime_stats['red']['mean'],
            'n_green': regime_stats['green']['n'], 'n_red': regime_stats['red']['n'], 'n_flat': regime_stats['flat']['n'],
            'regime_gap': regime_gap,
        })

    summary = pd.DataFrame(rows).sort_values('mean_net_ticks', ascending=False)
    summary.to_csv(OUT / 'regime_summary.csv', index=False)

    # HC #428 R1 gates: regime_gap ≤ 0.50 AND overall mean_net_ticks > 0 AND both regime means >= 0
    passing = summary[
        (summary['regime_gap'] <= 0.50)
        & (summary['mean_net_ticks'] > 0)
        & (summary['mean_green'] >= 0)
        & (summary['mean_red'] >= 0)
    ].copy()
    passing.to_csv(OUT / 'passing_cells.csv', index=False)

    md_lines = [
        '# HC #428 R1 — Regime-stratified Sharpe verdict',
        '',
        f'Source: 34-day baseline per-day stratification ({len(summary)} cells).',
        f'Gates: regime_gap ≤ 0.50, mean_net_ticks > 0, both regime means ≥ 0.',
        '',
        f'**Cells passing all gates: {len(passing)}**',
        '',
    ]
    if len(passing):
        md_lines.append('## Passing cells')
        md_lines.append(passing.head(10).to_markdown(index=False))
    else:
        top = summary.head(5)[['horizon', 'side', 'bucket', 'mean_net_ticks', 'sharpe', 'regime_gap']]
        md_lines.append('## Closest-misses (top-5 by mean net-ticks)')
        md_lines.append(top.to_markdown(index=False))

    (OUT / 'findings.md').write_text('\n'.join(md_lines))
    print(f'Wrote {len(summary)} cells; passing: {len(passing)}')


if __name__ == '__main__':
    main()
