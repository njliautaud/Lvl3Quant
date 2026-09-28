"""
Position Sizing Optimization Analysis — Meta Production v1
===========================================================
Uses meta-model concat predictions to determine optimal position sizing.
Actuals are ALREADY net of 0.376 ticks passive commission — no additional cost subtracted.

Filtering: Top 50% of trades by meta-model prediction value (highest conviction).

IMPORTANT: Predictions are event-triggered at 250ms stride with ~10s hold period.
Raw prediction sums DOUBLE-COUNT due to overlap. We use:
  - mean P&L per prediction as the expected return per trade entry
  - Non-overlapping trade count = active_duration / hold_period for realistic daily P&L

Output: /home/jupiter/Lvl3Quant/output/sizing_analysis_v1/
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime

# ── Constants ──────────────────────────────────────────────────────────────
ES_TICK_VALUE = 12.50       # $/tick
PRED_STRIDE_S = 0.25        # seconds between predictions
HOLD_PERIOD_S = 10.0        # approximate hold period per trade (~10s per signal docs)
TRADING_DAYS_PER_WEEK = 5
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/sizing_analysis_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────
npz = np.load("/home/jupiter/Lvl3Quant/output/meta_production_v1/concat_predictions.npz")
predictions = npz["predictions"]
actuals = npz["actuals"]

with open("/home/jupiter/Lvl3Quant/output/meta_production_v1/results.json") as f:
    results = json.load(f)

per_fold = results["per_fold"]

# ── Reconstruct per-fold predictions/actuals ──────────────────────────────
fold_data = []
idx = 0
for fold_info in per_fold:
    n = fold_info["n_test"]
    fold_data.append({
        "date": fold_info["date"],
        "n_test": n,
        "predictions": predictions[idx:idx+n],
        "actuals": actuals[idx:idx+n],
        "fold_info": fold_info,
    })
    idx += n

assert idx == len(predictions), f"Fold reconstruction mismatch: {idx} vs {len(predictions)}"

# ── Apply Meta Top-50% Filter & Compute Realistic Daily P&L ──────────────
# Top 50% by prediction value per fold (highest conviction).
# Daily P&L = mean_pnl_per_pred * non_overlapping_trades_that_day

all_filtered_actuals = []    # per-prediction P&L (for trade-level stats)
daily_pnl_ticks = []         # realistic daily P&L (overlap-corrected)
daily_stats = []             # detailed per-day info

for fd in fold_data:
    preds = fd["predictions"]
    acts = fd["actuals"]

    # Top 50% filter by prediction value
    median_pred = np.median(preds)
    top50_mask = preds >= median_pred
    filtered_preds = preds[top50_mask]
    filtered_acts = acts[top50_mask]

    all_filtered_actuals.extend(filtered_acts)

    n_preds = len(filtered_acts)
    mean_pnl = float(np.mean(filtered_acts)) if n_preds > 0 else 0.0

    # Non-overlapping trade count: active predictions span n_preds * stride seconds,
    # each trade holds for hold_period, so max independent trades = duration / hold
    active_duration_s = n_preds * PRED_STRIDE_S
    non_overlap_trades = max(1, active_duration_s / HOLD_PERIOD_S)

    # Realistic daily P&L = mean P&L per entry * number of non-overlapping entries
    daily_pnl = mean_pnl * non_overlap_trades
    daily_pnl_ticks.append(daily_pnl)

    daily_stats.append({
        "date": fd["date"],
        "n_predictions": n_preds,
        "non_overlap_trades": non_overlap_trades,
        "mean_pnl_per_trade": mean_pnl,
        "daily_pnl_ticks": daily_pnl,
    })

all_filtered_actuals = np.array(all_filtered_actuals)
daily_pnl_ticks = np.array(daily_pnl_ticks)

n_trades_total = len(all_filtered_actuals)
n_days = len(daily_pnl_ticks)
avg_trades_per_day = np.mean([d["non_overlap_trades"] for d in daily_stats])

print("=" * 70)
print("POSITION SIZING ANALYSIS — Meta Production v1 (Top 50% Filter)")
print("=" * 70)
print(f"\nData: {n_trades_total} prediction events -> ~{avg_trades_per_day:.0f} non-overlapping trades/day")
print(f"Days: {n_days} OOT days ({per_fold[0]['date']} to {per_fold[-1]['date']})")
print(f"Overlap correction: {PRED_STRIDE_S}s stride, {HOLD_PERIOD_S}s hold -> "
      f"{HOLD_PERIOD_S/PRED_STRIDE_S:.0f}x overlap factor")

# Baseline comparison
print(f"\nBaseline (all {len(actuals)} preds): WR={(actuals>0).mean():.1%}, Mean={actuals.mean():.4f} ticks")
print(f"Filtered (top 50%, {n_trades_total} preds): WR={(all_filtered_actuals>0).mean():.1%}, "
      f"Mean={all_filtered_actuals.mean():.4f} ticks")

# ── 1. Per-Trade Statistics ───────────────────────────────────────────────
wins = all_filtered_actuals[all_filtered_actuals > 0]
losses = all_filtered_actuals[all_filtered_actuals <= 0]

win_rate = len(wins) / n_trades_total
loss_rate = 1 - win_rate
avg_win = np.mean(wins) if len(wins) > 0 else 0
avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 0
win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
mean_pnl = np.mean(all_filtered_actuals)
median_pnl = np.median(all_filtered_actuals)
std_pnl = np.std(all_filtered_actuals)

gross_profit = np.sum(wins)
gross_loss = np.sum(np.abs(losses))
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

print(f"\n── Per-Trade Statistics (ticks, net of passive commission) ──")
print(f"Win rate:           {win_rate:.1%}")
print(f"Avg win:            {avg_win:.3f} ticks (${avg_win * ES_TICK_VALUE:.2f})")
print(f"Avg loss:           {avg_loss:.3f} ticks (${avg_loss * ES_TICK_VALUE:.2f})")
print(f"Win/Loss ratio:     {win_loss_ratio:.3f}")
print(f"Profit factor:      {profit_factor:.3f}")
print(f"Mean P&L/trade:     {mean_pnl:.4f} ticks (${mean_pnl * ES_TICK_VALUE:.2f})")
print(f"Median P&L/trade:   {median_pnl:.4f} ticks")
print(f"Std P&L/trade:      {std_pnl:.4f} ticks")

# ── 2. Kelly Criterion ───────────────────────────────────────────────────
kelly_f = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
half_kelly = kelly_f / 2
quarter_kelly = kelly_f / 4
kelly_continuous = mean_pnl / (std_pnl ** 2) if std_pnl > 0 else 0

print(f"\n── Kelly Criterion ──")
print(f"Full Kelly fraction:    {kelly_f:.4f} ({kelly_f:.1%} of bankroll per trade)")
print(f"Half Kelly (practical): {half_kelly:.4f} ({half_kelly:.1%})")
print(f"Quarter Kelly (safe):   {quarter_kelly:.4f} ({quarter_kelly:.1%})")
print(f"Continuous Kelly:       {kelly_continuous:.4f} ({kelly_continuous:.1%})")

# Contracts per bankroll at Half Kelly
risk_per_contract = avg_loss * ES_TICK_VALUE
print(f"\nContracts per trade at various bankroll levels (Half Kelly):")
for bankroll in [25000, 50000, 100000, 200000, 500000]:
    contracts = half_kelly * bankroll / risk_per_contract if risk_per_contract > 0 else 0
    print(f"  ${bankroll:>8,} -> {contracts:.1f} contracts (risking ${half_kelly * bankroll:.0f}/trade)")

# Continuous Kelly contracts (more conservative, accounts for variance)
print(f"\nContracts per trade at various bankroll levels (Continuous Kelly):")
for bankroll in [25000, 50000, 100000, 200000, 500000]:
    contracts = kelly_continuous * bankroll / risk_per_contract if risk_per_contract > 0 else 0
    print(f"  ${bankroll:>8,} -> {contracts:.1f} contracts")

# ── 3. Daily P&L Analysis (overlap-corrected) ────────────────────────────
daily_mean = np.mean(daily_pnl_ticks)
daily_std = np.std(daily_pnl_ticks, ddof=1) if n_days > 1 else np.std(daily_pnl_ticks)
daily_sharpe = daily_mean / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_std > 0 else 0

neg_days = daily_pnl_ticks[daily_pnl_ticks < 0]
daily_downside_std = np.std(neg_days, ddof=1) if len(neg_days) > 1 else (np.std(neg_days) if len(neg_days) > 0 else daily_std)
daily_sortino = daily_mean / daily_downside_std * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_downside_std > 0 else 0

daily_gross_profit = np.sum(daily_pnl_ticks[daily_pnl_ticks > 0])
daily_gross_loss = np.sum(np.abs(daily_pnl_ticks[daily_pnl_ticks < 0]))
daily_pf = daily_gross_profit / daily_gross_loss if daily_gross_loss > 0 else float('inf')

print(f"\n── Daily P&L (1 contract, overlap-corrected) ──")
print(f"Avg non-overlap trades/day: {avg_trades_per_day:.0f}")
print(f"Daily mean P&L:     {daily_mean:.2f} ticks (${daily_mean * ES_TICK_VALUE:.2f})")
print(f"Daily std P&L:      {daily_std:.2f} ticks (${daily_std * ES_TICK_VALUE:.2f})")
print(f"Daily Sharpe (ann): {daily_sharpe:.2f}")
print(f"Daily Sortino (ann):{daily_sortino:.2f}")
print(f"Daily Profit Factor:{daily_pf:.2f}")
print(f"Daily min:          {daily_pnl_ticks.min():.2f} ticks (${daily_pnl_ticks.min() * ES_TICK_VALUE:.2f})")
print(f"Daily max:          {daily_pnl_ticks.max():.2f} ticks (${daily_pnl_ticks.max() * ES_TICK_VALUE:.2f})")
print(f"Win days:           {(daily_pnl_ticks > 0).sum()}/{n_days} ({(daily_pnl_ticks > 0).mean():.0%})")

print(f"\n  Per-day breakdown:")
for ds in daily_stats:
    pnl = ds["daily_pnl_ticks"]
    marker = "+" if pnl > 0 else "-" if pnl < 0 else " "
    print(f"    {ds['date']}: {ds['non_overlap_trades']:5.0f} trades, "
          f"mean {ds['mean_pnl_per_trade']:+.3f} t/trade, "
          f"daily {pnl:+8.2f} ticks (${pnl*ES_TICK_VALUE:+8.2f})  {marker}")

# ── 4. Position Size Scaling ─────────────────────────────────────────────
print(f"\n── Position Size Scaling (1 contract baseline) ──")
print(f"{'Contracts':>10} {'Daily $':>10} {'Weekly $':>10} {'Monthly $':>12} {'Annual $':>12} {'Sharpe':>8} {'Max DD/mo':>12}")

for nc in [1, 2, 3, 5, 10, 20]:
    dm = daily_mean * nc * ES_TICK_VALUE
    ds = daily_std * nc * ES_TICK_VALUE
    sharpe = daily_sharpe  # constant under linear scaling
    max_dd_mo = 3 * ds * np.sqrt(21)
    ann = dm * TRADING_DAYS_PER_YEAR
    print(f"{nc:>10} ${dm:>8,.0f} ${dm*5:>8,.0f} ${dm*21:>10,.0f} ${ann:>10,.0f} {sharpe:>8.2f} ${max_dd_mo:>10,.0f}")

# ── 5. Risk of Ruin (Monte Carlo) ────────────────────────────────────────
print(f"\n── Risk of Ruin (Monte Carlo: 10,000 paths, 60-day horizon) ──")

np.random.seed(42)
N_SIMS = 10000
HORIZON_DAYS = 60

def simulate_ror(n_contracts, bankroll, n_sims=N_SIMS, horizon=HORIZON_DAYS):
    daily_pnls_dollar = daily_pnl_ticks * n_contracts * ES_TICK_VALUE
    ruin_count = 0
    max_dds = []
    finals = []
    for _ in range(n_sims):
        sampled = np.random.choice(daily_pnls_dollar, size=horizon, replace=True)
        eq = bankroll + np.cumsum(sampled)
        if np.any(eq <= 0):
            ruin_count += 1
        peak = np.maximum.accumulate(np.concatenate([[bankroll], eq]))
        dd = (peak - np.concatenate([[bankroll], eq])) / peak
        max_dds.append(np.max(dd))
        finals.append(eq[-1])
    return {
        "ror": ruin_count / n_sims,
        "med_maxdd": np.median(max_dds),
        "p95_maxdd": np.percentile(max_dds, 95),
        "med_final": np.median(finals),
        "p5_final": np.percentile(finals, 5),
        "p95_final": np.percentile(finals, 95),
    }

contract_sizes = [1, 2, 3, 5, 10]
bankroll_levels = [25000, 50000, 100000, 200000]

print(f"{'Cts':>5} {'Bankroll':>10} {'RoR':>8} {'Med DD':>8} {'P95 DD':>8} {'Med End $':>12} {'P5 End $':>12}")
print("-" * 75)

ror_results = {}
for nc in contract_sizes:
    ror_results[nc] = {}
    for bk in bankroll_levels:
        r = simulate_ror(nc, bk)
        ror_results[nc][bk] = r
        print(f"{nc:>5} ${bk:>8,} {r['ror']:>7.2%} {r['med_maxdd']:>7.1%} {r['p95_maxdd']:>7.1%} "
              f"${r['med_final']:>10,.0f} ${r['p5_final']:>10,.0f}")

# ── 6. Expected P&L Table ────────────────────────────────────────────────
print(f"\n── Expected P&L by Position Size ──")
print(f"{'Cts':>5} {'Daily':>10} {'Weekly':>10} {'Monthly':>12} {'Annual':>12}")

sizing_results = {}
for nc in [1, 2, 3, 5, 10, 20]:
    d = daily_mean * nc * ES_TICK_VALUE
    sizing_results[nc] = {"daily": d, "weekly": d*5, "monthly": d*21, "annual": d*252, "sharpe": daily_sharpe}
    print(f"{nc:>5} ${d:>8,.0f} ${d*5:>8,.0f} ${d*21:>10,.0f} ${d*252:>10,.0f}")

# ── 7. Sharpe Degradation ────────────────────────────────────────────────
print(f"\n── Sharpe Degradation (market impact estimate) ──")
print(f"CAVEAT: 1-5 contracts in ES = negligible impact. 10+ = monitor fills.\n")
print(f"{'Size':>6} {'No-impact':>10} {'With-impact':>12} {'Change':>8}")
for mult in [1, 2, 3, 5, 10, 20]:
    impact_per_day = max(0, (mult - 3) * 0.01) * avg_trades_per_day
    adj_mean = daily_mean * mult - impact_per_day
    adj_std = daily_std * mult
    adj_sharpe = (adj_mean / adj_std) * np.sqrt(252) if adj_std > 0 else 0
    chg = (adj_sharpe - daily_sharpe) / abs(daily_sharpe) * 100 if daily_sharpe != 0 else 0
    print(f"{mult:>6} {daily_sharpe:>10.2f} {adj_sharpe:>12.2f} {chg:>+7.1f}%")

# ── 8. Min Bankroll for 1% RoR ───────────────────────────────────────────
print(f"\n── Minimum Bankroll for <1% Risk of Ruin (60-day horizon) ──")
print(f"{'Cts':>5} {'Min Bankroll':>14} {'AMP Margin':>12}")

ES_MARGIN = 1000
for nc in contract_sizes:
    best = nc * 200000
    for bk in [nc * x for x in [5000, 7500, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 75000, 100000]]:
        r = simulate_ror(nc, bk, n_sims=5000)
        if r["ror"] <= 0.01:
            best = bk
            break
    print(f"{nc:>5} ${best:>12,} ${nc*ES_MARGIN:>10,}")

# ── 9. Filter Threshold Sensitivity ──────────────────────────────────────
print(f"\n── Filter Selectivity Analysis ──")
print(f"{'Filter':>12} {'N Preds':>8} {'WR':>7} {'Mean PnL':>9} {'PF':>6} {'Daily ticks':>12} {'Sharpe':>8}")

for pct in [100, 70, 50, 30, 20, 10]:
    threshold = np.percentile(predictions, 100 - pct)
    mask = predictions >= threshold
    subset = actuals[mask]
    wr = (subset > 0).mean()
    mp = subset.mean()
    gp = np.sum(subset[subset > 0])
    gl = np.sum(np.abs(subset[subset <= 0]))
    pf = gp / gl if gl > 0 else float('inf')

    # Reconstruct overlap-corrected daily P&L for this filter
    d_pnl = []
    i2 = 0
    for fi in per_fold:
        n = fi["n_test"]
        fp = predictions[i2:i2+n]
        fa = actuals[i2:i2+n]
        thresh_fold = np.percentile(fp, 100 - pct)
        m = fp >= thresh_fold
        n_filtered = m.sum()
        mean_p = fa[m].mean() if n_filtered > 0 else 0
        active_s = n_filtered * PRED_STRIDE_S
        non_overlap = max(1, active_s / HOLD_PERIOD_S)
        d_pnl.append(mean_p * non_overlap)
        i2 += n
    d_pnl = np.array(d_pnl)
    d_mean = d_pnl.mean()
    d_std = np.std(d_pnl, ddof=1) if len(d_pnl) > 1 else 1
    d_sharpe = d_mean / d_std * np.sqrt(252) if d_std > 0 else 0

    print(f"{'Top '+str(pct)+'%':>12} {len(subset):>8} {wr:>6.1%} {mp:>8.4f} {pf:>6.2f} {d_mean:>11.1f} {d_sharpe:>8.2f}")

# ── 10. Summary ──────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"SUMMARY & RECOMMENDATIONS")
print(f"{'=' * 70}")

print(f"""
EDGE QUALITY (per-trade, net of passive commission):
  Win rate:          {win_rate:.1%}
  Win/Loss ratio:    {win_loss_ratio:.2f}x
  Profit factor:     {profit_factor:.2f}
  Mean P&L/trade:    {mean_pnl:.4f} ticks (${mean_pnl * ES_TICK_VALUE:.2f})

KELLY SIZING:
  Full Kelly:        {kelly_f:.1%} of bankroll per trade
  Half Kelly:        {half_kelly:.1%} (recommended max)
  Continuous Kelly:  {kelly_continuous:.1%} (variance-adjusted, more conservative)

DAILY PERFORMANCE (1 contract, overlap-corrected):
  Trades/day:        ~{avg_trades_per_day:.0f} non-overlapping
  Daily P&L:         {daily_mean:.1f} ticks (${daily_mean * ES_TICK_VALUE:.0f})
  Daily Sharpe:      {daily_sharpe:.2f} (annualized)
  Daily Sortino:     {daily_sortino:.2f} (annualized)
  Win days:          {(daily_pnl_ticks > 0).sum()}/{n_days} ({(daily_pnl_ticks > 0).mean():.0%})

EXPECTED P&L AT 1 CONTRACT:
  Daily:   ${daily_mean * ES_TICK_VALUE:+,.0f}
  Weekly:  ${daily_mean * ES_TICK_VALUE * 5:+,.0f}
  Monthly: ${daily_mean * ES_TICK_VALUE * 21:+,.0f}
  Annual:  ${daily_mean * ES_TICK_VALUE * 252:+,.0f}

RECOMMENDED APPROACH:
  1. Start with 1 contract — confirm edge over 20+ live days
  2. Scale to Half Kelly sizing after 40+ days if Sharpe holds > 1.0
  3. Use Continuous Kelly ({kelly_continuous:.1%}) for more conservative sizing
  4. Max daily loss: 3x daily std = {3*daily_std:.0f} ticks (${3*daily_std*ES_TICK_VALUE:,.0f})
  5. Cap at 5 contracts until 60+ days of live data

CAVEATS:
  - {n_days} OOT days only (HC #428 requires 40+)
  - Overlap correction is approximate (assumes {HOLD_PERIOD_S:.0f}s hold, {PRED_STRIDE_S}s stride)
  - No market impact (valid for 1-5 ES contracts)
  - Actuals net of passive commission; market orders cost ~1 tick more
  - Monte Carlo based on {n_days}-day empirical distribution (small sample)
""")

# ── Save ──────────────────────────────────────────────────────────────────
output = {
    "timestamp": datetime.now().isoformat(),
    "data_source": "meta_production_v1",
    "filter": "top 50% by prediction value",
    "overlap_correction": {
        "stride_s": PRED_STRIDE_S,
        "hold_period_s": HOLD_PERIOD_S,
        "overlap_factor": HOLD_PERIOD_S / PRED_STRIDE_S,
    },
    "n_prediction_events": int(n_trades_total),
    "avg_non_overlap_trades_per_day": float(avg_trades_per_day),
    "n_days": int(n_days),
    "date_range": [per_fold[0]["date"], per_fold[-1]["date"]],
    "per_trade": {
        "win_rate": float(win_rate),
        "avg_win_ticks": float(avg_win),
        "avg_loss_ticks": float(avg_loss),
        "win_loss_ratio": float(win_loss_ratio),
        "profit_factor": float(profit_factor),
        "mean_pnl_ticks": float(mean_pnl),
        "median_pnl_ticks": float(median_pnl),
        "std_pnl_ticks": float(std_pnl),
    },
    "kelly": {
        "full_kelly": float(kelly_f),
        "half_kelly": float(half_kelly),
        "quarter_kelly": float(quarter_kelly),
        "continuous_kelly": float(kelly_continuous),
    },
    "daily_overlap_corrected": {
        "mean_pnl_ticks": float(daily_mean),
        "std_pnl_ticks": float(daily_std),
        "sharpe_annualized": float(daily_sharpe),
        "sortino_annualized": float(daily_sortino),
        "profit_factor": float(daily_pf),
        "win_day_rate": float((daily_pnl_ticks > 0).mean()),
        "per_day": [
            {
                "date": ds["date"],
                "non_overlap_trades": ds["non_overlap_trades"],
                "mean_pnl_per_trade": ds["mean_pnl_per_trade"],
                "daily_pnl_ticks": ds["daily_pnl_ticks"],
                "daily_pnl_dollars": ds["daily_pnl_ticks"] * ES_TICK_VALUE,
            }
            for ds in daily_stats
        ],
    },
    "risk_of_ruin": {
        str(nc): {
            str(bk): ror_results[nc][bk]
            for bk in bankroll_levels
        }
        for nc in contract_sizes
    },
    "sizing_pnl": {str(nc): v for nc, v in sizing_results.items()},
    "constants": {
        "es_tick_value": ES_TICK_VALUE,
        "commission_note": "Actuals already net of 0.376 ticks passive commission",
    },
}

out_path = OUT_DIR / "sizing_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Results saved to {out_path}")
print("Done.")
