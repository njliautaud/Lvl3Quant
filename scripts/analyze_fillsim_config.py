#!/usr/bin/env python3
"""Analyze fill sim results for a specific config — full stats + Monte Carlo."""
import json, glob, sys, os
import numpy as np

config = sys.argv[1] if len(sys.argv) > 1 else "A_tp13_sl40_t050"
results_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/jupiter/Lvl3Quant/fill_sim_test/results_v2"
initial_capital = 25000

files = sorted(glob.glob(os.path.join(results_dir, "%s_*.json" % config)))
if not files:
    print("No files found for config %s in %s" % (config, results_dir))
    sys.exit(1)

daily_pnl = []
all_trade_pnls = []
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    daily_pnl.append(d.get("total_pnl_dollars", 0))
    for t in d.get("trades", []):
        all_trade_pnls.append(t.get("pnl_dollars", 0))

pnl = np.array(daily_pnl)
cum = np.cumsum(pnl)
peak = np.maximum.accumulate(cum)
dd = cum - peak
max_dd = dd.min()
max_dd_idx = np.argmin(dd)
max_dd_pct = abs(max_dd / (initial_capital + peak[max_dd_idx])) * 100 if peak[max_dd_idx] > 0 else 0

trades = np.array(all_trade_pnls)
wins = trades[trades > 0]
losses = trades[trades <= 0]

print("=" * 60)
print("CONFIG: %s" % config)
print("=" * 60)
print("Days: %d | Active: %d | Profitable: %d" % (
    len(daily_pnl),
    sum(1 for p in daily_pnl if abs(p) > 0),
    sum(1 for p in daily_pnl if p > 0)))
print("Total PnL: $%.0f | Avg/day: $%.0f" % (sum(daily_pnl), np.mean(pnl)))
print("Total trades: %d | Win rate: %.1f%%" % (len(trades), 100*len(wins)/len(trades) if len(trades) else 0))
print("Avg win: $%.0f | Avg loss: $%.0f" % (np.mean(wins) if len(wins) else 0, np.mean(losses) if len(losses) else 0))
print("Profit factor: %.2f" % (abs(np.sum(wins)/np.sum(losses)) if np.sum(losses) != 0 else 0))
print("Max drawdown: $%.0f (%.1f%%)" % (max_dd, max_dd_pct))
daily_sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(252) if np.std(pnl) > 0 else 0
neg = pnl[pnl < 0]
downside = np.sqrt(np.mean(neg**2)) if len(neg) > 0 else 1
daily_sortino = np.mean(pnl) / downside * np.sqrt(252)
print("Sharpe: %.1f | Sortino: %.1f" % (daily_sharpe, daily_sortino))

# Monte Carlo
print("\n--- Monte Carlo (5,000 sims) ---")
n_sims = 5000
n_trades = len(all_trade_pnls)
final_eq = []
max_dds_pct = []
for _ in range(n_sims):
    sampled = np.random.choice(all_trade_pnls, size=n_trades, replace=True)
    cum_eq = initial_capital + np.cumsum(sampled)
    final_eq.append(cum_eq[-1])
    pk = np.maximum.accumulate(cum_eq)
    dd_pct = ((cum_eq - pk) / pk) * 100
    max_dds_pct.append(dd_pct.min())

fe = np.array(final_eq)
md = np.array(max_dds_pct)
print("Final equity: median=$%.0f, p5=$%.0f, p95=$%.0f" % (np.median(fe), np.percentile(fe, 5), np.percentile(fe, 95)))
print("Max DD: median=%.1f%%, p95 worst=%.1f%%" % (np.median(md), np.percentile(md, 5)))
print("Prob profit: %.1f%%" % (100 * np.mean(fe > initial_capital)))
print("Prob ruin (>50%% DD): %.1f%%" % (100 * np.mean(md < -50)))
sortinos = []
for _ in range(1000):
    sampled_daily = np.random.choice(daily_pnl, size=len(daily_pnl), replace=True)
    neg_s = sampled_daily[sampled_daily < 0]
    ds = np.sqrt(np.mean(neg_s**2)) if len(neg_s) > 0 else 1
    sortinos.append(np.mean(sampled_daily) / ds * np.sqrt(252))
sortinos = np.array(sortinos)
print("Sortino MC: median=%.1f, p5=%.1f (worst case)" % (np.median(sortinos), np.percentile(sortinos, 5)))

# Verdict
passed = True
verdicts = []
if np.percentile(sortinos, 5) < 2.0:
    verdicts.append("FAIL: Sortino p5 %.1f < 2.0" % np.percentile(sortinos, 5))
    passed = False
else:
    verdicts.append("PASS: Sortino p5 %.1f > 2.0" % np.percentile(sortinos, 5))
if 100 * np.mean(md < -50) > 5:
    verdicts.append("FAIL: Prob ruin %.1f%% > 5%%" % (100 * np.mean(md < -50)))
    passed = False
else:
    verdicts.append("PASS: Prob ruin %.1f%% < 5%%" % (100 * np.mean(md < -50)))
if 100 * np.mean(fe > initial_capital) < 80:
    verdicts.append("FAIL: Prob profit %.1f%% < 80%%" % (100 * np.mean(fe > initial_capital)))
    passed = False
else:
    verdicts.append("PASS: Prob profit %.1f%% > 80%%" % (100 * np.mean(fe > initial_capital)))

print("\n--- VERDICT ---")
for v in verdicts:
    print("  %s" % v)
print("OVERALL: %s" % ("PASS - READY FOR DEPLOYMENT" if passed else "FAIL - NOT READY"))
