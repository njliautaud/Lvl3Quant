#!/usr/bin/env python3
"""Re-summarize zscore_pred_test results with correct field names."""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/zscore_pred_test")
TICK_VALUE = 12.50  # dollars per tick

# Discover all result files
result_files = sorted(OUTPUT_DIR.glob("zscore_*.json"))
result_files = [f for f in result_files if f.name not in ("zscore_stats.json", "summary.json", "summary_v2.json")]

print(f"Found {len(result_files)} result files")

# Parse config name and date from filename
# e.g. zscore_buy_afternoon_t0.5_20260401.json
results = defaultdict(list)
for f in result_files:
    name = f.stem  # zscore_buy_afternoon_t0.5_20260401
    # Date is last 8 chars before .json
    date = name[-8:]
    config = name[:-9]  # remove _YYYYMMDD
    
    try:
        with open(f) as fh:
            data = json.load(fh)
        data["_date"] = date
        data["_file"] = f.name
        results[config].append(data)
    except Exception as e:
        print(f"  Error loading {f.name}: {e}")

print(f"Parsed {len(results)} configs")

def summarize(date_results, label="ALL"):
    if not date_results:
        return None
    
    days = len(date_results)
    total_trades = sum(r.get("total_trades", 0) for r in date_results)
    
    # Per-trade stats from individual trades
    all_trades = []
    for r in date_results:
        if "trades" in r:
            all_trades.extend(r["trades"])
    
    wins = sum(1 for t in all_trades if t.get("pnl_dollars", 0) > 0)
    losses = sum(1 for t in all_trades if t.get("pnl_dollars", 0) < 0)
    
    # Per-day P&L in ticks
    daily_pnl_ticks = []
    for r in date_results:
        pnl_dollars = r.get("total_pnl_dollars", 0)
        pnl_ticks = pnl_dollars / TICK_VALUE
        daily_pnl_ticks.append(pnl_ticks)
    
    total_pnl_ticks = sum(daily_pnl_ticks)
    daily_arr = np.array(daily_pnl_ticks)
    
    wr = wins / total_trades * 100 if total_trades > 0 else 0
    
    # Sharpe (annualized from daily)
    sharpe = (daily_arr.mean() / daily_arr.std() * np.sqrt(252)) if daily_arr.std() > 0 else 0
    
    # Sortino
    neg = daily_arr[daily_arr < 0]
    downside_std = np.sqrt((neg ** 2).mean()) if len(neg) > 0 else 1e-8
    sortino = daily_arr.mean() / downside_std * np.sqrt(252) if downside_std > 1e-8 else 0
    
    # Profit factor
    gross_wins = sum(max(0, p) for p in daily_pnl_ticks)
    gross_losses = sum(abs(min(0, p)) for p in daily_pnl_ticks)
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    green_days = sum(1 for p in daily_pnl_ticks if p > 0)
    
    # Fill rate
    total_signals = sum(r.get("total_signals", 0) for r in date_results)
    total_filled = sum(r.get("total_filled", 0) for r in date_results)
    fill_rate = total_filled / total_signals * 100 if total_signals > 0 else 0
    
    return {
        "label": label,
        "days": days,
        "trades": total_trades,
        "trades_per_day": total_trades / days if days > 0 else 0,
        "wr": wr,
        "pnl_ticks": total_pnl_ticks,
        "pnl_per_day": total_pnl_ticks / days if days > 0 else 0,
        "pnl_dollars": total_pnl_ticks * TICK_VALUE,
        "sharpe": sharpe,
        "sortino": sortino,
        "pf": pf,
        "green_days": green_days,
        "green_pct": green_days / days * 100 if days > 0 else 0,
        "fill_rate": fill_rate,
    }


print("\n" + "=" * 90)
print("RESULTS SUMMARY: Z-Score Normalization Test")
print("=" * 90)

all_summaries = {}

for config_name in sorted(results.keys()):
    date_results = results[config_name]
    
    march_results = [r for r in date_results if r["_date"].startswith("202603")]
    april_results = [r for r in date_results if r["_date"].startswith("202604")]
    
    s_all = summarize(date_results, "ALL")
    s_mar = summarize(march_results, "MARCH")
    s_apr = summarize(april_results, "APRIL")
    
    all_summaries[config_name] = {"all": s_all, "march": s_mar, "april": s_apr}
    
    print(f"\n{'─' * 90}")
    print(f"CONFIG: {config_name}")
    print(f"{'─' * 90}")
    
    for s in [s_all, s_mar, s_apr]:
        if s is None:
            continue
        print(f"  {s['label']:>6s} | {s['days']:2d}d | {s['trades']:5d} trades ({s['trades_per_day']:5.1f}/d) | "
              f"WR {s['wr']:5.1f}% | PnL {s['pnl_ticks']:+8.1f}t ({s['pnl_per_day']:+6.1f}t/d) ${s['pnl_dollars']:+8.0f} | "
              f"Sharpe {s['sharpe']:+5.2f} | Sortino {s['sortino']:+5.2f} | PF {s['pf']:5.2f} | "
              f"Green {s['green_days']:2d}/{s['days']:2d} ({s['green_pct']:3.0f}%) | Fill {s['fill_rate']:4.1f}%")
    
    if s_mar and s_apr:
        delta_pnl = (s_apr['pnl_per_day'] or 0) - (s_mar['pnl_per_day'] or 0)
        delta_sharpe = (s_apr['sharpe'] or 0) - (s_mar['sharpe'] or 0)
        print(f"  DELTA  | Apr-Mar: {delta_pnl:+.1f} ticks/day, Sharpe delta {delta_sharpe:+.2f}")

# Key question: does z-scoring equalize March vs April?
print("\n\n" + "=" * 90)
print("KEY FINDING: March vs April Regime Gap (Sharpe)")
print("=" * 90)
print(f"{'Config':<35s} | {'Mar Sharpe':>10s} | {'Apr Sharpe':>10s} | {'Gap':>8s} | {'|Gap|/max':>10s}")
print("-" * 90)
for config_name in sorted(all_summaries.keys()):
    s = all_summaries[config_name]
    if s["march"] and s["april"]:
        ms = s["march"]["sharpe"]
        as_ = s["april"]["sharpe"]
        gap = as_ - ms
        max_abs = max(abs(ms), abs(as_), 0.01)
        ratio = abs(gap) / max_abs
        print(f"  {config_name:<33s} | {ms:+10.2f} | {as_:+10.2f} | {gap:+8.2f} | {ratio:10.2f}")

with open(OUTPUT_DIR / "summary_v2.json", "w") as f:
    json.dump(all_summaries, f, indent=2, default=str)

print(f"\nSaved to {OUTPUT_DIR / 'summary_v2.json'}")
