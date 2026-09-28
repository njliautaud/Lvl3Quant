#!/usr/bin/env python3
"""Show day-by-day results for a config+threshold combo."""
import json, glob, os, sys

cfg = sys.argv[1] if len(sys.argv) > 1 else "h15000_t8_s8_t07"
out_dir = f"/home/jupiter/lvl3quant/production/results/rust_sim/{cfg}"
files = sorted(glob.glob(out_dir + "/*.json"))

print(f"Config: {cfg} ({len(files)} days)")
print(f"{'Date':<12} {'PnL':>10} {'Trades':>7} {'Fill':>6} {'WinR':>6} {'Sharpe':>8}")
print("-" * 55)
total_pnl = 0
total_trades = 0
total_filled = 0
total_posted = 0
total_wins = 0

for f in files:
    try:
        d = json.load(open(f))
        s = d["summary"]
        date = os.path.basename(f).replace(".json", "")
        pnl = s["total_pnl_dollars"]
        trades = s["total_trades"]
        fill = s["fill_rate"] * 100
        wr = s["win_rate"] * 100
        sharpe = s["sharpe_per_trade"]
        total_pnl += pnl
        total_trades += trades
        total_filled += s["total_filled"]
        total_posted += s["total_posted"]
        if trades > 0:
            total_wins += int(wr/100 * trades)
        marker = " ***" if abs(pnl) > 10000 else ""
        print(f"{date:<12} ${pnl:>+9,.0f} {trades:>7} {fill:>5.0f}% {wr:>5.0f}% {sharpe:>+7.3f}{marker}")
    except Exception as e:
        print(f"ERR: {f}: {e}")

print("-" * 55)
avg_pnl = total_pnl / len(files) if files else 0
fill_rate = total_filled / total_posted if total_posted else 0
overall_wr = total_wins / total_trades if total_trades else 0
print(f"{'TOTAL':<12} ${total_pnl:>+9,.0f} {total_trades:>7}")
print(f"{'AVG/DAY':<12} ${avg_pnl:>+9,.0f}")
print(f"Overall fill: {fill_rate*100:.1f}%  win_rate: {overall_wr*100:.1f}%")
