"""Re-score streaming_trade_sim_v2 results at 0.376t RT cost (HC #512)."""
import pandas as pd
import numpy as np
import json
from pathlib import Path

OLD_RT_COST = 1.752
NEW_RT_COST = 0.376
COST_DELTA = OLD_RT_COST - NEW_RT_COST  # 1.376 ticks saved per trade
ES_TICK_VALUE = 12.50

OUT = Path("/home/nick/Lvl3Quant/output/streaming_trade_sim_v2")

# --- Re-score sweep results ---
df = pd.read_csv(OUT / "sweep_results.csv")

# Recover gross from old net: gross = old_net + n_trades * OLD_RT_COST
# Then new_net = gross - n_trades * NEW_RT_COST = old_net + n_trades * COST_DELTA
df["new_total_net_ticks"] = df["total_net_ticks"] + df["n_trades"] * COST_DELTA
df["new_total_net_dollars"] = df["new_total_net_ticks"] * ES_TICK_VALUE
df["new_avg_net_per_trade"] = df["new_total_net_ticks"] / df["n_trades"]

# Re-compute profit factor from gross P&L info
# We need trade-level data for proper WR/PF recalc, but we can estimate:
# new PF = gross_wins / (gross_losses - adjusted) -- actually let's use per-trade approach
# Since net_per_trade changes by COST_DELTA for every trade:
# old avg_net = old_total_net / n_trades
# new avg_net = old_avg_net + COST_DELTA
df["old_avg_net"] = df["total_net_ticks"] / df["n_trades"]
df["new_avg_net"] = df["old_avg_net"] + COST_DELTA

# For WR: a trade flips from loss to win if old_net_trade was in (-COST_DELTA, 0]
# We can't compute exact new WR without trade-level data, but we can note the gross info

# Simple summary
print("=" * 90)
print(f"RE-SCORED STREAMING TRADE SIM v2: {OLD_RT_COST:.3f}t -> {NEW_RT_COST:.3f}t RT cost")
print(f"Cost savings: {COST_DELTA:.3f} ticks/trade")
print("=" * 90)

# Sort by new net
df_sorted = df.sort_values("new_total_net_ticks", ascending=False)

print(f"\n{'Intens%':>7} {'CNN_th':>6} {'RevStop':>7} {'Trades':>6} {'Tr/Day':>6} "
      f"{'Gross':>8} {'OldNet':>10} {'NewNet':>10} {'Net/Tr':>7} {'New$':>12} {'OldWR':>6}")
print("-" * 105)

for _, r in df_sorted.iterrows():
    marker = "+++" if r["new_total_net_ticks"] > 0 else ("+" if r["new_avg_net"] > 0 else "")
    print(f"{r['intensity_pctile']:7.0f} {r['cnn_abs_threshold']:6.2f} {r['reversal_stop_ticks']:7.1f} "
          f"{r['n_trades']:6.0f} {r['trades_per_day']:6.1f} "
          f"{r['total_gross_ticks']:8.1f} {r['total_net_ticks']:10.1f} "
          f"{r['new_total_net_ticks']:10.1f} {r['new_avg_net']:7.3f} "
          f"${r['new_total_net_dollars']:11,.0f} {r['win_rate']:6.1%} {marker}")

# Count positive configs
n_positive = (df_sorted["new_total_net_ticks"] > 0).sum()
n_positive_per_trade = (df_sorted["new_avg_net"] > 0).sum()
print(f"\n{n_positive}/{len(df)} configs net positive (total)")
print(f"{n_positive_per_trade}/{len(df)} configs positive per-trade average")

# --- Best configs ---
print("\n" + "=" * 90)
print("TOP 5 CONFIGS BY NEW NET TICKS:")
print("=" * 90)
for i, (_, r) in enumerate(df_sorted.head(5).iterrows()):
    print(f"\n#{i+1}: intensity_pctile={r['intensity_pctile']:.0f}, "
          f"cnn_threshold={r['cnn_abs_threshold']:.2f}, "
          f"reversal_stop={r['reversal_stop_ticks']:.1f}")
    print(f"  Trades: {r['n_trades']:.0f} ({r['trades_per_day']:.1f}/day over {r['n_days']:.0f} days)")
    print(f"  Gross: {r['total_gross_ticks']:+.1f}t | Old Net: {r['total_net_ticks']:+.1f}t | New Net: {r['new_total_net_ticks']:+.1f}t")
    print(f"  New Net $/day: ${r['new_total_net_dollars']/r['n_days']:+,.0f}")
    print(f"  Avg Net/Trade: {r['new_avg_net']:+.3f}t (${r['new_avg_net']*ES_TICK_VALUE:+.2f})")
    print(f"  Old WR: {r['win_rate']:.1%} | Avg Win: {r['avg_win_ticks']:+.2f}t | Avg Loss: {r['avg_loss_ticks']:+.2f}t")
    print(f"  MFE: {r['avg_mfe_ticks']:.2f}t | MAE: {r['avg_mae_ticks']:.2f}t | Hold: {r['avg_hold_s']:.1f}s")

# --- Re-score trade-level for best config ---
print("\n" + "=" * 90)
print("TRADE-LEVEL RE-SCORE (best config from original run)")
print("=" * 90)

trades = pd.read_csv(OUT / "best_trades.csv")
trades["new_net_ticks"] = trades["gross_ticks"] - NEW_RT_COST
trades["new_net_dollars"] = trades["new_net_ticks"] * ES_TICK_VALUE

n = len(trades)
wins = (trades["new_net_ticks"] > 0).sum()
losses = (trades["new_net_ticks"] <= 0).sum()
new_wr = wins / n
total_new_net = trades["new_net_ticks"].sum()

win_trades = trades[trades["new_net_ticks"] > 0]["new_net_ticks"]
loss_trades = trades[trades["new_net_ticks"] <= 0]["new_net_ticks"]
pf = win_trades.sum() / abs(loss_trades.sum()) if loss_trades.sum() != 0 else float('inf')

# Daily P&L for Sharpe
daily = trades.groupby("date")["new_net_ticks"].sum()
sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
sortino_down = daily[daily < 0].std()
sortino = daily.mean() / sortino_down * np.sqrt(252) if sortino_down > 0 else 0

print(f"Config: intensity_pctile=95, cnn_threshold=0.30, reversal_stop=6.0")
print(f"Trades: {n} over {trades['date'].nunique()} days ({n/trades['date'].nunique():.1f}/day)")
print(f"New WR: {new_wr:.1%} (was {(trades['net_ticks']>0).mean():.1%})")
print(f"New PF: {pf:.3f}")
print(f"New Sharpe (ann): {sharpe:.2f}")
print(f"New Sortino (ann): {sortino:.2f}")
print(f"Total Net: {total_new_net:+.1f} ticks (${total_new_net*ES_TICK_VALUE:+,.0f})")
print(f"Avg Net/Trade: {total_new_net/n:+.3f}t")

# Per-day breakdown
print(f"\nPer-Day Breakdown:")
print(f"{'Date':>10} {'Trades':>6} {'Gross':>8} {'NewNet':>8} {'WR':>6} {'Cum$':>10}")
cum = 0
for date, grp in trades.groupby("date"):
    day_gross = grp["gross_ticks"].sum()
    day_net = grp["new_net_ticks"].sum()
    day_wr = (grp["new_net_ticks"] > 0).mean()
    cum += day_net * ES_TICK_VALUE
    print(f"{date:>10} {len(grp):6d} {day_gross:+8.1f} {day_net:+8.1f} {day_wr:6.1%} ${cum:+10,.0f}")

# Regime test: green vs red days
# We don't have ES close-to-close here, but we can use per-day net direction as proxy
print(f"\nDay Concentration: max day = {daily.max():.1f}t of {total_new_net:.1f}t total "
      f"= {daily.max()/total_new_net:.1%}" if total_new_net != 0 else "\nDay Concentration: N/A (zero net)")

# Direction breakdown
print(f"\nDirection Breakdown:")
for direction in [-1, 1]:
    side = trades[trades["direction"] == direction]
    if len(side) == 0:
        continue
    side_net = side["new_net_ticks"].sum()
    side_wr = (side["new_net_ticks"] > 0).mean()
    side_label = "SHORT" if direction == -1 else "LONG"
    print(f"  {side_label}: {len(side)} trades, Net={side_net:+.1f}t, WR={side_wr:.1%}, "
          f"Avg={side_net/len(side):+.3f}t/trade")

# Save rescored results
df_sorted.to_csv(OUT / "sweep_results_rescored_0376.csv", index=False)
trades.to_csv(OUT / "best_trades_rescored_0376.csv", index=False)
print(f"\nSaved rescored results.")
