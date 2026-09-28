#!/usr/bin/env python3
"""
Dollar Smile (UUP/UDN Macro Regime) — Adversarial Validation
6 checks: inverse direction, random timing, look-ahead bias, cost sensitivity,
sub-period stability, parameter sensitivity.
"""

import json
import datetime
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
START = "2022-01-01"
END   = "2026-07-29"
CAPITAL = 645.0
SLIPPAGE_BPS = 0.0002  # 0.02%
COMMISSION = 0.0

# ── Data download ──────────────────────────────────────────────────────────
print("Downloading data …")
tickers = ["UUP", "UDN", "SPY", "QQQ", "^VIX"]
raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
close = raw["Close"].copy()
close.columns = [c if isinstance(c, str) else c[0] for c in close.columns]

# rename VIX
if "^VIX" in close.columns:
    close.rename(columns={"^VIX": "VIX"}, inplace=True)

close = close.dropna()
print(f"Data: {close.index[0].date()} → {close.index[-1].date()}, {len(close)} rows")

# ── Strategy logic ─────────────────────────────────────────────────────────
def run_strategy(df, vix_crisis=25, spy_mom_thresh=0.10, spy_mom_lookback=60,
                 slippage=SLIPPAGE_BPS, invert=False, lag_signals=False):
    """
    Returns a dict with equity curve, trades, and metrics.
    If invert=True, swap UUP↔UDN signals.
    If lag_signals=True, use previous-day VIX and SPY momentum (look-ahead fix).
    """
    d = df.copy()

    if lag_signals:
        d["vix_signal"] = d["VIX"].shift(1)
        d["spy_mom"] = d["SPY"].pct_change(spy_mom_lookback).shift(1)
    else:
        d["vix_signal"] = d["VIX"]
        d["spy_mom"] = d["SPY"].pct_change(spy_mom_lookback)

    d = d.dropna()

    # Signals
    crisis   = d["vix_signal"] > vix_crisis
    growth   = d["spy_mom"] > spy_mom_thresh
    goldilocks = (d["vix_signal"] >= 15) & (d["vix_signal"] <= vix_crisis) & (~growth)

    pos_uup = crisis | growth        # long UUP
    pos_udn = goldilocks & ~pos_uup  # long UDN (goldilocks only if not crisis/growth)

    if invert:
        pos_uup, pos_udn = pos_udn, pos_uup

    # If lag_signals, entry on NEXT day's open ≈ use next-day close as proxy
    if lag_signals:
        pos_uup = pos_uup.shift(1).fillna(False).astype(bool)
        pos_udn = pos_udn.shift(1).fillna(False).astype(bool)
        d = d.iloc[1:]
        pos_uup = pos_uup.iloc[1:]
        pos_udn = pos_udn.iloc[1:]

    # Daily returns
    ret_uup = d["UUP"].pct_change()
    ret_udn = d["UDN"].pct_change()

    # Build position series
    position = pd.Series("cash", index=d.index)
    position[pos_uup] = "UUP"
    position[pos_udn] = "UDN"

    # Strategy returns with slippage on trades
    strat_ret = pd.Series(0.0, index=d.index)
    prev_pos = "cash"
    trades = 0
    trade_rets = []
    current_trade_ret = 0.0

    for i in range(1, len(d)):
        idx = d.index[i]
        cur_pos = position.iloc[i]

        if cur_pos == "UUP":
            r = ret_uup.iloc[i]
        elif cur_pos == "UDN":
            r = ret_udn.iloc[i]
        else:
            r = 0.0

        # Slippage on entry/exit
        if cur_pos != prev_pos:
            if prev_pos != "cash":
                r -= slippage  # exit old
                if current_trade_ret != 0.0:
                    trade_rets.append(current_trade_ret)
                current_trade_ret = 0.0
            if cur_pos != "cash":
                r -= slippage  # enter new
                trades += 1
                current_trade_ret = 0.0

        if cur_pos != "cash":
            current_trade_ret += r

        strat_ret.iloc[i] = r
        prev_pos = cur_pos

    # Close last trade
    if current_trade_ret != 0.0:
        trade_rets.append(current_trade_ret)

    # Equity curve
    equity = CAPITAL * (1 + strat_ret).cumprod()

    # Metrics
    ann_ret = (equity.iloc[-1] / CAPITAL) ** (252 / len(equity)) - 1
    daily_rets = strat_ret[strat_ret != 0]  # only invested days
    if len(daily_rets) > 10:
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
        neg = daily_rets[daily_rets < 0]
        sortino = daily_rets.mean() / neg.std() * np.sqrt(252) if len(neg) > 0 and neg.std() > 0 else 0
    else:
        sharpe = sortino = 0.0

    total_return = (equity.iloc[-1] / CAPITAL - 1)
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    mdd = dd.min()

    # Win rate from trade returns
    if len(trade_rets) > 0:
        wr = sum(1 for t in trade_rets if t > 0) / len(trade_rets)
        winners = [t for t in trade_rets if t > 0]
        losers  = [abs(t) for t in trade_rets if t < 0]
        pf = (sum(winners) / sum(losers)) if losers and sum(losers) > 0 else float("inf")
    else:
        wr = 0.0
        pf = 0.0

    # QQQ correlation
    qqq_ret = d["QQQ"].pct_change().dropna()
    common = strat_ret.index.intersection(qqq_ret.index)
    qqq_corr = strat_ret.loc[common].corr(qqq_ret.loc[common])

    return {
        "equity": equity,
        "strat_ret": strat_ret,
        "position": position,
        "trades": trades,
        "trade_rets": trade_rets,
        "metrics": {
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "win_rate": round(wr, 4),
            "profit_factor": round(pf, 4) if pf != float("inf") else 999.0,
            "num_trades": trades,
            "total_return": round(total_return, 4),
            "mdd": round(mdd, 4),
            "mean_return_pct": round(daily_rets.mean() * 100, 6) if len(daily_rets) > 0 else 0,
            "ann_return": round(ann_ret, 4),
        },
        "qqq_corr": round(qqq_corr, 4) if not np.isnan(qqq_corr) else 0.0,
    }


# ── Run baseline ───────────────────────────────────────────────────────────
print("Running baseline …")
baseline = run_strategy(close)
print(f"  Baseline Sharpe: {baseline['metrics']['sharpe']}, Trades: {baseline['metrics']['num_trades']}, "
      f"Return: {baseline['metrics']['total_return']:.2%}")

results = {
    "strategy": "Dollar Smile (UUP/UDN Macro Regime)",
    "run_date": datetime.datetime.now().isoformat(),
    "oot_period": f"{close.index[0].date()} to {close.index[-1].date()}",
    "baseline": baseline["metrics"],
    "qqq_correlation": baseline["qqq_corr"],
    "checks": {},
}

# ── CHECK 1: INVERSE DIRECTION ────────────────────────────────────────────
print("\nCheck 1: Inverse direction …")
inverse = run_strategy(close, invert=True)
inv_sharpe = inverse["metrics"]["sharpe"]
base_sharpe = baseline["metrics"]["sharpe"]

inv_pass = (inv_sharpe < 0) and (base_sharpe > 2 * inv_sharpe if inv_sharpe != 0 else True)
# If inverse sharpe is negative, baseline > 2*negative is always true, so focus on inv < 0
inv_pass = inv_sharpe < 0 and base_sharpe > 0

results["checks"]["1_inverse_direction"] = {
    "pass": inv_pass,
    "verdict": f"{'PASS' if inv_pass else 'FAIL'} — Inverse Sharpe {inv_sharpe:.3f} vs Baseline {base_sharpe:.3f}. "
               f"{'Inverse loses money → directional edge confirmed.' if inv_pass else 'Inverse also profitable → may be dollar exposure, not edge.'}",
    "inverse_metrics": inverse["metrics"],
    "baseline_sharpe": base_sharpe,
    "inverse_sharpe": inv_sharpe,
}
print(f"  Inverse Sharpe: {inv_sharpe:.4f} → {'PASS' if inv_pass else 'FAIL'}")

# ── CHECK 2: RANDOM TIMING ────────────────────────────────────────────────
print("\nCheck 2: Random timing (1000 iterations) …")

# Extract trade structure from baseline
pos_series = baseline["position"]
transitions = []
current_state = pos_series.iloc[0]
start_idx = 0
for i in range(1, len(pos_series)):
    if pos_series.iloc[i] != current_state:
        transitions.append({"state": current_state, "length": i - start_idx})
        current_state = pos_series.iloc[i]
        start_idx = i
transitions.append({"state": current_state, "length": len(pos_series) - start_idx})

# Get invested segments (non-cash)
invested_segments = [t for t in transitions if t["state"] != "cash"]
segment_lengths = [s["length"] for s in invested_segments]
segment_instruments = [s["state"] for s in invested_segments]

ret_uup = close["UUP"].pct_change().dropna()
ret_udn = close["UDN"].pct_change().dropna()

n_iters = 1000
random_sharpes = []
valid_days = list(range(len(ret_uup)))

for it in range(n_iters):
    random_strat_ret = pd.Series(0.0, index=ret_uup.index)
    for seg_len, instr in zip(segment_lengths, segment_instruments):
        # Pick random start
        max_start = len(ret_uup) - seg_len
        if max_start <= 0:
            continue
        start = np.random.randint(0, max_start)
        end = start + seg_len
        if instr == "UUP":
            random_strat_ret.iloc[start:end] += ret_uup.iloc[start:end].values
        else:
            random_strat_ret.iloc[start:end] += ret_udn.iloc[start:end].values

    invested = random_strat_ret[random_strat_ret != 0]
    if len(invested) > 10 and invested.std() > 0:
        s = invested.mean() / invested.std() * np.sqrt(252)
    else:
        s = 0.0
    random_sharpes.append(s)

random_sharpes = np.array(random_sharpes)
percentile = np.mean(random_sharpes < base_sharpe) * 100
rand_pass = percentile >= 90

results["checks"]["2_random_timing"] = {
    "pass": rand_pass,
    "verdict": f"{'PASS' if rand_pass else 'FAIL'} — Real Sharpe at {percentile:.1f}th percentile of random. "
               f"Random mean={np.mean(random_sharpes):.3f}, std={np.std(random_sharpes):.3f}.",
    "percentile": round(percentile, 2),
    "random_sharpe_mean": round(np.mean(random_sharpes), 4),
    "random_sharpe_std": round(np.std(random_sharpes), 4),
    "real_sharpe": base_sharpe,
}
print(f"  Percentile: {percentile:.1f}% → {'PASS' if rand_pass else 'FAIL'}")

# ── CHECK 3: LOOK-AHEAD BIAS ──────────────────────────────────────────────
print("\nCheck 3: Look-ahead bias (point-in-time) …")
pit = run_strategy(close, lag_signals=True)
pit_sharpe = pit["metrics"]["sharpe"]
ratio = pit_sharpe / base_sharpe if base_sharpe != 0 else 0
lab_pass = pit_sharpe > 0.5 and ratio > 0.7  # within 30% of original

results["checks"]["3_look_ahead_bias"] = {
    "pass": lab_pass,
    "verdict": f"{'PASS' if lab_pass else 'FAIL'} — Point-in-time Sharpe {pit_sharpe:.3f} vs Original {base_sharpe:.3f} "
               f"(ratio {ratio:.2f}). {'Minimal look-ahead effect.' if lab_pass else 'Significant degradation with lag → possible look-ahead bias.'}",
    "point_in_time_metrics": pit["metrics"],
    "sharpe_ratio_pit_vs_original": round(ratio, 4),
}
print(f"  PIT Sharpe: {pit_sharpe:.4f}, ratio: {ratio:.2f} → {'PASS' if lab_pass else 'FAIL'}")

# ── CHECK 4: COST SENSITIVITY ─────────────────────────────────────────────
print("\nCheck 4: Cost sensitivity …")
slip_levels = [0.0005, 0.0010, 0.0015, 0.0020]
cost_results = {}
break_slip = None
cost_pass = False

for sl in slip_levels:
    r = run_strategy(close, slippage=sl)
    cost_results[f"{sl*100:.2f}%"] = {
        "sharpe": r["metrics"]["sharpe"],
        "total_return": r["metrics"]["total_return"],
    }
    if r["metrics"]["sharpe"] <= 0 and break_slip is None:
        break_slip = sl

# Pass if Sharpe > 0.5 at 0.10%
s_at_010 = cost_results["0.10%"]["sharpe"]
cost_pass = s_at_010 > 0.5

results["checks"]["4_cost_sensitivity"] = {
    "pass": cost_pass,
    "verdict": f"{'PASS' if cost_pass else 'FAIL'} — Sharpe at 0.10% slippage: {s_at_010:.3f}. "
               f"{'Strategy survives realistic costs.' if cost_pass else 'Strategy dies under moderate costs.'}",
    "break_slippage_pct": round(break_slip * 100, 2) if break_slip else None,
    "results_by_slippage": cost_results,
}
print(f"  Sharpe at 0.10%: {s_at_010:.4f} → {'PASS' if cost_pass else 'FAIL'}")

# ── CHECK 5: SUB-PERIOD STABILITY ─────────────────────────────────────────
print("\nCheck 5: Sub-period stability …")
n_periods = 4
dates = close.index
chunk = len(dates) // n_periods
sub_results = []

for i in range(n_periods):
    s = i * chunk
    e = (i + 1) * chunk if i < n_periods - 1 else len(dates)
    sub_df = close.iloc[s:e]
    if len(sub_df) < 60:
        sub_results.append({"period": f"{sub_df.index[0].date()} → {sub_df.index[-1].date()}", "sharpe": 0, "skip": True})
        continue
    r = run_strategy(sub_df)
    sub_results.append({
        "period": f"{sub_df.index[0].date()} → {sub_df.index[-1].date()}",
        "sharpe": r["metrics"]["sharpe"],
        "total_return": r["metrics"]["total_return"],
        "num_trades": r["metrics"]["num_trades"],
    })

positive_periods = sum(1 for s in sub_results if s["sharpe"] > 0)
sub_pass = positive_periods >= 3

results["checks"]["5_sub_period_stability"] = {
    "pass": sub_pass,
    "verdict": f"{'PASS' if sub_pass else 'FAIL'} — {positive_periods}/4 sub-periods have Sharpe > 0.",
    "sub_periods": sub_results,
    "positive_count": positive_periods,
}
print(f"  Positive sub-periods: {positive_periods}/4 → {'PASS' if sub_pass else 'FAIL'}")

# ── CHECK 6: PARAMETER SENSITIVITY ────────────────────────────────────────
print("\nCheck 6: Parameter sensitivity (125 combos) …")
vix_thresholds = [20, 22, 25, 28, 30]
spy_mom_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15]
spy_mom_lookbacks = [40, 50, 60, 70, 80]

grid_results = []
baseline_params = (25, 0.10, 60)

for vt in vix_thresholds:
    for smt in spy_mom_thresholds:
        for sml in spy_mom_lookbacks:
            r = run_strategy(close, vix_crisis=vt, spy_mom_thresh=smt, spy_mom_lookback=sml)
            grid_results.append({
                "vix_thresh": vt,
                "spy_mom_thresh": smt,
                "spy_mom_lookback": sml,
                "sharpe": r["metrics"]["sharpe"],
                "total_return": r["metrics"]["total_return"],
                "num_trades": r["metrics"]["num_trades"],
            })

sharpes = [g["sharpe"] for g in grid_results]
above_03 = sum(1 for s in sharpes if s > 0.3)
pct_above = above_03 / len(sharpes) * 100

best = max(grid_results, key=lambda x: x["sharpe"])
baseline_sharpe_in_grid = next(g["sharpe"] for g in grid_results
                                if g["vix_thresh"] == 25 and g["spy_mom_thresh"] == 0.10
                                and g["spy_mom_lookback"] == 60)
baseline_percentile = np.mean([s <= baseline_sharpe_in_grid for s in sharpes]) * 100

param_pass = pct_above >= 30

results["checks"]["6_parameter_sensitivity"] = {
    "pass": param_pass,
    "verdict": f"{'PASS' if param_pass else 'FAIL'} — {pct_above:.1f}% of {len(grid_results)} combos have Sharpe > 0.3. "
               f"Baseline params at {baseline_percentile:.0f}th percentile.",
    "pct_above_0_3_sharpe": round(pct_above, 2),
    "total_combos": len(grid_results),
    "above_0_3_count": above_03,
    "best_params": {
        "vix_thresh": best["vix_thresh"],
        "spy_mom_thresh": best["spy_mom_thresh"],
        "spy_mom_lookback": best["spy_mom_lookback"],
        "sharpe": best["sharpe"],
    },
    "baseline_param_percentile": round(baseline_percentile, 2),
    "sharpe_distribution": {
        "min": round(min(sharpes), 4),
        "p25": round(np.percentile(sharpes, 25), 4),
        "median": round(np.median(sharpes), 4),
        "p75": round(np.percentile(sharpes, 75), 4),
        "max": round(max(sharpes), 4),
    },
}
print(f"  {pct_above:.1f}% above Sharpe 0.3, baseline at {baseline_percentile:.0f}th pct → {'PASS' if param_pass else 'FAIL'}")

# ── Overall ────────────────────────────────────────────────────────────────
checks_passed = sum(1 for c in results["checks"].values() if c["pass"])
results["overall_pass"] = checks_passed == 6
results["checks_passed"] = f"{checks_passed}/6"

# Save
out_path = "/home/jupiter/Lvl3Quant/data/dollar_smile_adversarial_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"OVERALL: {checks_passed}/6 checks passed → {'ALL CLEAR' if checks_passed == 6 else 'FAILED'}")
print(f"Results saved to {out_path}")
print(f"{'='*60}")

# Print full JSON
print("\n" + json.dumps(results, indent=2, default=str))
