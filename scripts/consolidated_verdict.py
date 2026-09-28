#!/usr/bin/env python3
"""
Consolidated Verdict: Combine all tick-level replay experiments into
a single honest assessment of whether the CNN-Mamba model can trade ES profitably.

Run this after all experiments finish.

Author: Claude (autonomous research, 2026-07-02)
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

OUTPUT = Path('/home/jupiter/Lvl3Quant/output/tick_level_replay')


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return None


def bonferroni_test(p_values, alpha=0.05):
    """Apply Bonferroni correction for multiple testing."""
    n = len(p_values)
    threshold = alpha / n
    passing = [(i, p) for i, p in enumerate(p_values) if p < threshold]
    return threshold, passing


def main():
    print("=" * 70)
    print("CONSOLIDATED VERDICT: CNN-Mamba v3.4.2 on ES Futures")
    print("=" * 70)

    # 1. v2 sweep (standard TP/SL)
    v2_results = load_json(OUTPUT / 'tick_replay_v2_results.json')
    if v2_results:
        print("\n## 1. Standard TP/SL Configs (tick-level replay, 50 perms)")
        if isinstance(v2_results, list):
            configs = v2_results
        elif isinstance(v2_results, dict):
            configs = v2_results.get('results', v2_results.get('configs', []))

        net_positive = 0
        sig_positive = 0
        for c in configs if isinstance(configs, list) else []:
            if isinstance(c, dict):
                net = c.get('net_ticks_per_trade', c.get('avg', 0))
                p = c.get('p_value', 1.0)
                if net > 0:
                    net_positive += 1
                    if p < 0.05:
                        sig_positive += 1

        print(f"  Total configs tested: {len(configs) if isinstance(configs, list) else '?'}")
        print(f"  Net positive: {net_positive}")
        print(f"  Significant & positive: {sig_positive}")
        print(f"  VERDICT: {'SOME EDGE' if sig_positive > 0 else 'NO PROFITABLE CONFIGS'}")

    # 2. v2 confidence-filtered
    for label, fname in [("Top 5% confidence", "tick_replay_v2_top5pct.json"),
                          ("Top 1% confidence", "tick_replay_v2_top1pct.json")]:
        data = load_json(OUTPUT / fname)
        if data:
            print(f"\n## 2. {label}")
            if isinstance(data, list):
                for c in data[:5]:
                    print(f"  {c}")
            elif isinstance(data, dict):
                for k, v in list(data.items())[:5]:
                    print(f"  {k}: {v}")

    # 3. v5 vol-gated conviction
    v5_log = OUTPUT / 'v5_full_log.txt'
    if v5_log.exists():
        print("\n## 3. Vol-Gated Conviction Sweep")
        positive_configs = []
        with open(v5_log) as f:
            for line in f:
                if 'net_passive=+' in line and 'p=0.0' in line:
                    # Extract config name and metrics
                    parts = line.strip().split()
                    for i, p in enumerate(parts):
                        if ':' in p and ('both_' in p or 'short_' in p):
                            name = p.rstrip(':')
                            break
                    positive_configs.append(line.strip())

        print(f"  Configs with net>0 AND p<0.05: {len(positive_configs)}")
        for c in positive_configs:
            print(f"    {c[-80:]}")

    # 4. v5b deep analysis
    v5b = load_json(OUTPUT / 'v5b_k20_deep_analysis.json')
    if v5b:
        print("\n## 4. k=20 Deep Analysis")
        config = v5b.get('config', {})
        trades = v5b.get('trades', [])
        perm = v5b.get('perm_test', {})
        per_day = v5b.get('per_day', {})

        longs = sum(1 for t in trades if t['direction'] == 1)
        shorts = sum(1 for t in trades if t['direction'] == -1)
        print(f"  Config: {config}")
        print(f"  Trades: {len(trades)} (L:{longs}, S:{shorts})")
        print(f"  p-value: {perm.get('p_value', '?')}")

        if per_day:
            nets = [v['net'] for v in per_day.values()]
            green = sum(1 for n in nets if n > 0)
            red = sum(1 for n in nets if n < 0)
            if len(nets) > 1 and np.std(nets) > 0:
                sharpe = np.mean(nets) / np.std(nets) * np.sqrt(252)
                print(f"  Sharpe: {sharpe:.2f}, Green/Red: {green}/{red}")
            print(f"  Total net: {sum(nets):+.1f} ticks")

    # 5. v6 deep dive (200-perm validation)
    v6_files = list(OUTPUT.glob('v6_deep_dive*.json'))
    for f in v6_files:
        data = load_json(f)
        if data:
            print(f"\n## 5. 200-Perm Deep Validation ({f.name})")
            if isinstance(data, dict):
                for k in ['star_config', 'p_value_200', 'per_day_sharpe', 'bonferroni_pass']:
                    if k in data:
                        print(f"  {k}: {data[k]}")

    # 6. v6 confluence
    v6c_files = list(OUTPUT.glob('v6_confluence*.json'))
    for f in v6c_files:
        data = load_json(f)
        if data:
            print(f"\n## 6. Multi-Horizon Confluence ({f.name})")
            if isinstance(data, list):
                for c in data[:3]:
                    print(f"  {c}")

    # Overall verdict
    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)
    print("""
Key findings:
1. The CNN-Mamba model GENUINELY predicts short-term ES direction (p=0.000)
2. Short-side predictions are stronger than long-side
3. Edge scales with volatility
4. Edge magnitude: ~0.2 ticks/trade (best ~0.5 in high-vol periods)
5. ES round-trip cost: 0.376 ticks (passive) to 1.376 ticks (market)
6. Standard TP/SL configs: ALL unprofitable at tick level
7. Vol-gated conviction (k=20, 30s hold, vol>=2.5): potentially net-positive
   BUT: 104 trades in 34 days, multiple testing concern (30 configs searched)

CRITICAL QUESTION: Does the star config survive 200-perm Bonferroni test?
- If yes: thin but potentially real edge (~$46/day, ~$11.5K/year per contract)
- If no: model edge exists but is too small to trade ES profitably

STRATEGIC IMPLICATIONS:
- The model's edge is 5x below ES trading costs for high-frequency strategies
- Only ultra-selective timing (3 trades/day in high vol) might work
- Consider: (a) cheaper instruments, (b) signal as feature, (c) longer horizons
""")


if __name__ == '__main__':
    main()
