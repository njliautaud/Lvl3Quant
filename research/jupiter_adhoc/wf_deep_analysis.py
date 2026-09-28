#!/usr/bin/env python3
"""
Deep statistical analysis of walk-forward fill_sim sweep results.
Reads 60K+ per-day JSON files from wf_sweep/ and produces actionable insights.
"""
import os
import re
import json
import glob
import math
from collections import defaultdict
from datetime import datetime, timezone

SWEEP_DIR = os.path.expanduser("~/Lvl3Quant/alpha_discovery/results/wf_sweep")

# ─────────────────────────────────────────────
# 1. LOAD ALL FILES
# ─────────────────────────────────────────────
print("=" * 70)
print("LOADING FILES...")
print("=" * 70)

# Regex to parse filename: wf_v{vol}_c{conv}_h{hold}m_{mode}_ct{ct}r{r}_lat{lat}_{date}.json
FNAME_RE = re.compile(
    r"wf_v(\d+)_c([\d.]+)_h(\d+)m_(chase|passive)_ct(\d+)r(\d+)_lat(\d+)_(\d{4}-\d{2}-\d{2})\.json$"
)

records = []   # one record per per-day JSON file
skipped = 0

all_files = [f for f in os.listdir(SWEEP_DIR) if f.endswith(".json") and f.startswith("wf_v")]
print(f"Found {len(all_files):,} candidate files")

for fname in all_files:
    m = FNAME_RE.match(fname)
    if not m:
        skipped += 1
        continue
    vol, conv, hold_min, mode, ct, cr, lat, date_str = m.groups()
    fpath = os.path.join(SWEEP_DIR, fname)
    try:
        with open(fpath) as f:
            d = json.load(f)
    except Exception:
        skipped += 1
        continue

    # Skip days with 0 signals (model didn't fire at all)
    total_signals = d.get("total_signals", 0)
    if total_signals == 0:
        skipped += 1
        continue

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    rec = {
        # params
        "vol": int(vol),
        "conv": float(conv),
        "hold_min": int(hold_min),
        "mode": mode,
        "chase_ticks": int(ct),
        "chase_reprices": int(cr),
        "lat_ms": int(lat),
        # date features
        "date": date_str,
        "dow": dt.weekday(),          # 0=Mon … 4=Fri
        "month": dt.month,
        # per-day metrics
        "pnl": d.get("total_pnl_dollars", 0.0),
        "trades": d.get("total_trades", 0),
        "signals": total_signals,
        "win_rate": d.get("win_rate", 0.0),
        "fill_rate": d.get("fill_rate", 0.0),
        "sharpe": d.get("sharpe_per_trade", 0.0),
        "profit_factor": d.get("profit_factor", 0.0),
        "avg_queue_pos": d.get("avg_queue_position", 0.0),
        "avg_fill_lat_ms": d.get("avg_fill_latency_ms", 0.0),
        "avg_win": d.get("avg_win", 0.0),
        "avg_loss": d.get("avg_loss", 0.0),
        "mean_pnl_per_trade": d.get("mean_pnl_per_trade", 0.0),
        # trades list for time-of-day
        "trades_list": d.get("trades", []),
    }
    records.append(rec)

print(f"Loaded {len(records):,} valid day-records  |  skipped {skipped:,}")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")

def stdev(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def sharpe(xs):
    m = mean(xs)
    s = stdev(xs)
    if math.isnan(m) or math.isnan(s) or s == 0:
        return float("nan")
    return m / s * math.sqrt(252)

def pct_positive(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return 100.0 * sum(1 for x in xs if x > 0) / len(xs) if xs else float("nan")

def summarize_group(recs, label=""):
    pnls = [r["pnl"] for r in recs]
    if not pnls:
        return
    print(f"  {label:50s}  n={len(pnls):5d}  mean_pnl=${mean(pnls):8.1f}  "
          f"total=${sum(pnls):10.0f}  sharpe={sharpe(pnls):6.2f}  "
          f"pct+={pct_positive(pnls):5.1f}%  "
          f"wr={100*mean([r['win_rate'] for r in recs]):.1f}%  "
          f"fill={100*mean([r['fill_rate'] for r in recs]):.1f}%")


# ─────────────────────────────────────────────
# BUILD CONFIG-LEVEL AGGREGATES (for stability analysis)
# ─────────────────────────────────────────────
config_groups = defaultdict(list)
for r in records:
    key = (r["vol"], r["conv"], r["hold_min"], r["mode"],
           r["chase_ticks"], r["chase_reprices"], r["lat_ms"])
    config_groups[key].append(r)

config_stats = {}
for key, recs in config_groups.items():
    if len(recs) < 5:
        continue
    pnls = [r["pnl"] for r in recs]
    config_stats[key] = {
        "days": len(recs),
        "total_pnl": sum(pnls),
        "mean_daily": mean(pnls),
        "sharpe": sharpe(pnls),
        "pct_pos": pct_positive(pnls),
        "win_rate": mean([r["win_rate"] for r in recs]),
        "fill_rate": mean([r["fill_rate"] for r in recs]),
        "trades_per_day": mean([r["trades"] for r in recs]),
    }

print(f"\nConfigs with ≥5 days: {len(config_stats):,}")

# ─────────────────────────────────────────────
# SECTION 1 — PARAMETER STABILITY ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 1: PARAMETER STABILITY ANALYSIS")
print("=" * 70)

def param_stability(param_name, values_fn):
    """For each unique value of a param, aggregate ALL configs that use it."""
    groups = defaultdict(list)
    for key, cs in config_stats.items():
        v = values_fn(key)
        groups[v].append(cs["mean_daily"])

    print(f"\n--- {param_name} ---")
    results = []
    for v, daily_means in sorted(groups.items()):
        m = mean(daily_means)
        s = stdev(daily_means)
        pp = pct_positive(daily_means)
        stability = pp * (m / (abs(m) + abs(s) + 1e-9))  # score: pct_pos × consistency
        results.append((v, m, pp, len(daily_means), stability))
        print(f"  {param_name}={v!s:10s}  n_configs={len(daily_means):4d}  "
              f"mean_daily_pnl=${m:8.1f}  pct_pos={pp:5.1f}%  stability_score={stability:6.2f}")

    best = max(results, key=lambda x: x[4])
    worst = min(results, key=lambda x: x[4])
    print(f"  >>> BEST={best[0]} (score={best[4]:.2f})  WORST={worst[0]} (score={worst[4]:.2f})")
    return results

vol_res     = param_stability("vol_gate",     lambda k: k[0])
conv_res    = param_stability("conv_thresh",  lambda k: k[1])
hold_res    = param_stability("hold_min",     lambda k: k[2])
mode_res    = param_stability("mode",         lambda k: k[3])
ct_res      = param_stability("chase_ticks",  lambda k: k[4])
cr_res      = param_stability("chase_reprices", lambda k: k[5])
lat_res     = param_stability("latency_ms",   lambda k: k[6])

# ─────────────────────────────────────────────
# SECTION 2 — TIME-OF-DAY ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: TIME-OF-DAY ANALYSIS (from trade timestamps)")
print("=" * 70)

# signal_time_ns is Unix epoch nanoseconds
# RTH 9:30–16:00 ET = 13:30–20:00 UTC
hour_pnl = defaultdict(list)
hour_wr   = defaultdict(list)
trade_count_by_hour = defaultdict(int)

for r in records:
    for t in r["trades_list"]:
        ns = t.get("signal_time_ns", 0)
        if ns == 0:
            continue
        # Convert to ET hour (UTC - 4 or -5; Dec-Mar is EST = UTC-5)
        # Use UTC-5 for winter (Dec-Mar)
        utc_hour_frac = (ns / 1e9) % 86400 / 3600
        et_hour = int((utc_hour_frac - 5) % 24)
        pnl = t.get("pnl_dollars", 0.0)
        win = 1 if pnl > 0 else 0
        hour_pnl[et_hour].append(pnl)
        hour_wr[et_hour].append(win)
        trade_count_by_hour[et_hour] += 1

print("\nHour (ET) | Trades | Mean P&L | Total P&L | Win Rate")
print("-" * 60)
for h in sorted(hour_pnl.keys()):
    if 9 <= h <= 16:  # RTH only
        pnls = hour_pnl[h]
        wrs  = hour_wr[h]
        print(f"  {h:02d}:00     | {len(pnls):6d} | ${mean(pnls):8.1f} | "
              f"${sum(pnls):10.0f} | {100*mean(wrs):5.1f}%")

best_hour = max((h for h in hour_pnl if 9 <= h <= 16),
                key=lambda h: mean(hour_pnl[h]), default=None)
worst_hour = min((h for h in hour_pnl if 9 <= h <= 16),
                 key=lambda h: mean(hour_pnl[h]), default=None)
if best_hour is not None:
    print(f"\n>>> Best hour: {best_hour}:00 ET  (mean P&L ${mean(hour_pnl[best_hour]):.1f}/trade)")
    print(f">>> Worst hour: {worst_hour}:00 ET  (mean P&L ${mean(hour_pnl[worst_hour]):.1f}/trade)")

# ─────────────────────────────────────────────
# SECTION 3 — REGIME ANALYSIS (per-day vol proxy)
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: REGIME ANALYSIS (signal_strength as vol proxy)")
print("=" * 70)

# Use mean abs signal_strength as vol proxy per day-record
# Also use trade P&L spread as vol proxy
# Group by date-level mean_pnl_per_trade volatility
# Better: use vol_gate parameter as built-in gate — analyze how same DATE performs across vol levels

# Use signal_strength from trades as realized vol proxy
date_signal_vol = defaultdict(list)
for r in records:
    for t in r["trades_list"]:
        ss = abs(t.get("signal_strength", 0.0))
        if ss > 0:
            date_signal_vol[r["date"]].append(ss)

date_avg_signal = {d: mean(vs) for d, vs in date_signal_vol.items() if vs}

if date_avg_signal:
    sorted_dates = sorted(date_avg_signal, key=date_avg_signal.get)
    n = len(sorted_dates)
    low_vol_dates  = set(sorted_dates[:n//3])
    mid_vol_dates  = set(sorted_dates[n//3:2*n//3])
    high_vol_dates = set(sorted_dates[2*n//3:])

    def regime_stats(recs, dates_set, label):
        sub = [r for r in recs if r["date"] in dates_set and r["trades"] > 0]
        if sub:
            pnls = [r["pnl"] for r in sub]
            print(f"  {label:25s}  n_days={len(sub):5d}  "
                  f"mean_pnl=${mean(pnls):8.1f}  total=${sum(pnls):10.0f}  "
                  f"pct+={pct_positive(pnls):5.1f}%  "
                  f"sharpe={sharpe(pnls):6.2f}")

    print("\nRegime split by mean signal strength (proxy for realized vol):")
    regime_stats(records, low_vol_dates,  "LOW vol regime")
    regime_stats(records, mid_vol_dates,  "MID vol regime")
    regime_stats(records, high_vol_dates, "HIGH vol regime")

# Use vol_gate parameter itself as filter analysis
print("\nP&L by vol_gate parameter (selectivity of signal):")
for vg in sorted({r["vol"] for r in records}):
    sub = [r for r in records if r["vol"] == vg and r["trades"] > 0]
    if sub:
        pnls = [r["pnl"] for r in sub]
        print(f"  vol_gate={vg:3d}  n={len(sub):5d}  "
              f"mean_pnl=${mean(pnls):8.1f}  pct+={pct_positive(pnls):5.1f}%  "
              f"signals/day={mean([r['signals'] for r in sub]):.1f}  "
              f"fill_rate={100*mean([r['fill_rate'] for r in sub]):.1f}%")

# ─────────────────────────────────────────────
# SECTION 4 — FILL RATE vs PROFITABILITY
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: FILL RATE vs PROFITABILITY")
print("=" * 70)

# Bucket fill rates
fill_buckets = defaultdict(list)
for r in records:
    if r["trades"] < 2:
        continue
    bucket = round(r["fill_rate"] * 10) / 10  # round to nearest 0.1
    fill_buckets[bucket].append(r["pnl"])

print("\nFill Rate Bucket | N days | Mean P&L | Pct Positive")
print("-" * 55)
for bucket in sorted(fill_buckets.keys()):
    pnls = fill_buckets[bucket]
    if len(pnls) >= 10:
        print(f"  {bucket:.1f} ({bucket*100:.0f}%)     | {len(pnls):6d} | "
              f"${mean(pnls):8.1f} | {pct_positive(pnls):5.1f}%")

# Correlation fill_rate vs pnl
fr_vals = [r["fill_rate"] for r in records if r["trades"] >= 2]
pnl_vals = [r["pnl"] for r in records if r["trades"] >= 2]
if len(fr_vals) > 10:
    mf = mean(fr_vals); mp = mean(pnl_vals)
    sf = stdev(fr_vals); sp = stdev(pnl_vals)
    cov = mean([(f - mf) * (p - mp) for f, p in zip(fr_vals, pnl_vals)])
    corr = cov / (sf * sp) if sf > 0 and sp > 0 else float("nan")
    print(f"\nPearson correlation fill_rate vs daily P&L: {corr:.4f}")

# Also check: trades per day vs P&L
tp_vals = [r["trades"] for r in records if r["trades"] >= 2]
pnl_vals2 = [r["pnl"] for r in records if r["trades"] >= 2]
mt = mean(tp_vals); mp2 = mean(pnl_vals2)
st_ = stdev(tp_vals); sp2 = stdev(pnl_vals2)
if st_ > 0 and sp2 > 0:
    cov2 = mean([(t - mt) * (p - mp2) for t, p in zip(tp_vals, pnl_vals2)])
    corr2 = cov2 / (st_ * sp2)
    print(f"Pearson correlation trades_per_day vs daily P&L: {corr2:.4f}")

# ─────────────────────────────────────────────
# SECTION 5 — DAY-OF-WEEK EFFECTS
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: DAY-OF-WEEK EFFECTS")
print("=" * 70)

dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
dow_groups = defaultdict(list)
for r in records:
    if r["trades"] > 0:
        dow_groups[r["dow"]].append(r)

print("\nDay       | N days | Mean P&L | Total P&L | Sharpe | Pct+ | Win Rate")
print("-" * 75)
for dow in range(5):
    recs = dow_groups[dow]
    if recs:
        pnls = [r["pnl"] for r in recs]
        print(f"  {dow_names[dow]:9s} | {len(recs):6d} | ${mean(pnls):8.1f} | "
              f"${sum(pnls):10.0f} | {sharpe(pnls):6.2f} | "
              f"{pct_positive(pnls):5.1f}% | "
              f"{100*mean([r['win_rate'] for r in recs]):.1f}%")

# ─────────────────────────────────────────────
# SECTION 6 — PARAMETER PAIR INTERACTIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: PARAMETER PAIR INTERACTIONS (heatmaps as text)")
print("=" * 70)

def pair_heatmap(name_a, fn_a, name_b, fn_b, top_n=10):
    grid = defaultdict(list)
    for key, cs in config_stats.items():
        a = fn_a(key)
        b = fn_b(key)
        grid[(a, b)].append(cs["mean_daily"])

    print(f"\n{name_a} × {name_b} (mean daily P&L, ≥3 configs per cell):")
    vals = sorted(set(fn_a(k) for k in config_stats))
    header_vals = sorted(set(fn_b(k) for k in config_stats))

    # Print header
    header = f"  {name_a:15s} | " + "  ".join(f"{v!s:>8}" for v in header_vals)
    print(header)
    print("-" * len(header))
    for a in vals:
        row = f"  {a!s:15s} | "
        cells = []
        for b in header_vals:
            ms = grid.get((a, b), [])
            if len(ms) >= 3:
                cells.append(f"${mean(ms):7.0f}")
            else:
                cells.append(f"{'':>8}")
        row += "  ".join(cells)
        print(row)

pair_heatmap("vol",  lambda k: k[0], "conv", lambda k: k[1])
pair_heatmap("vol",  lambda k: k[0], "hold", lambda k: k[2])
pair_heatmap("conv", lambda k: k[1], "hold", lambda k: k[2])
pair_heatmap("vol",  lambda k: k[0], "lat",  lambda k: k[6])
pair_heatmap("hold", lambda k: k[2], "lat",  lambda k: k[6])

# ─────────────────────────────────────────────
# SECTION 7 — LATENCY SENSITIVITY
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 7: LATENCY SENSITIVITY")
print("=" * 70)

lat_groups = defaultdict(list)
for key, cs in config_stats.items():
    lat_groups[key[6]].append(cs["mean_daily"])

lat0_mean = mean(lat_groups.get(0, [0.0]))
print(f"\nLatency (ms) | N configs | Mean Daily P&L | Delta vs lat=0")
print("-" * 60)
for lat in sorted(lat_groups.keys()):
    ms = lat_groups[lat]
    m = mean(ms)
    delta = m - lat0_mean
    print(f"  {lat:5d} ms   | {len(ms):9d} | ${m:14.1f} | {delta:+.1f}")

# Also: correlation between fill latency and P&L (per trade level)
fill_lat_vals = []
trade_pnl_vals = []
for r in records:
    for t in r["trades_list"]:
        fl = t.get("fill_latency_ns", 0)
        pnl = t.get("pnl_dollars", 0.0)
        if fl > 0:
            fill_lat_vals.append(fl / 1e6)  # to ms
            trade_pnl_vals.append(pnl)

if len(fill_lat_vals) > 100:
    mfl = mean(fill_lat_vals); mtp = mean(trade_pnl_vals)
    sfl = stdev(fill_lat_vals); stp = stdev(trade_pnl_vals)
    if sfl > 0 and stp > 0:
        cov = mean([(f - mfl) * (p - mtp) for f, p in zip(fill_lat_vals, trade_pnl_vals)])
        corr = cov / (sfl * stp)
        print(f"\nPer-trade correlation: fill_latency_ms vs P&L: {corr:.4f}")
        print(f"(Negative = faster fills → higher P&L)")

# Fill latency buckets
fl_buckets = defaultdict(list)
for fl, pnl in zip(fill_lat_vals, trade_pnl_vals):
    if fl < 100:     b = "0-100ms"
    elif fl < 500:   b = "100-500ms"
    elif fl < 2000:  b = "500ms-2s"
    elif fl < 5000:  b = "2s-5s"
    else:            b = ">5s"
    fl_buckets[b].append(pnl)

print("\nFill Latency Bucket | Trades | Mean P&L | Pct+"  )
print("-" * 50)
for b in ["0-100ms", "100-500ms", "500ms-2s", "2s-5s", ">5s"]:
    pnls = fl_buckets.get(b, [])
    if pnls:
        print(f"  {b:18s} | {len(pnls):6d} | ${mean(pnls):8.1f} | {pct_positive(pnls):5.1f}%")

# ─────────────────────────────────────────────
# SECTION 8 — WIN RATE vs HOLD TIME
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 8: WIN RATE vs HOLD TIME")
print("=" * 70)

hold_groups = defaultdict(list)
for r in records:
    if r["trades"] >= 2:
        hold_groups[r["hold_min"]].append(r)

print("\nHold Time | N days | Mean P&L | Win Rate | Profit Factor | Trades/Day")
print("-" * 75)
for h in sorted(hold_groups.keys()):
    recs = hold_groups[h]
    pnls = [r["pnl"] for r in recs]
    pfs = [r["profit_factor"] for r in recs if r["profit_factor"] is not None and not math.isnan(r["profit_factor"]) and r["profit_factor"] < 100]
    print(f"  {h:3d} min   | {len(recs):6d} | ${mean(pnls):8.1f} | "
          f"{100*mean([r['win_rate'] for r in recs]):.1f}%    | "
          f"{mean(pfs):.2f}          | {mean([r['trades'] for r in recs]):.1f}")

# ─────────────────────────────────────────────
# SECTION 9 — TOP CONFIGS (≥15 days, full WF period)
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 9: TOP CONFIGS (≥15 trading days)")
print("=" * 70)

top = sorted(
    [(k, cs) for k, cs in config_stats.items() if cs["days"] >= 5],
    key=lambda x: x[1]["sharpe"],
    reverse=True
)[:30]

print(f"\nFound {len([k for k, cs in config_stats.items() if cs['days'] >= 5])} configs with >=5 days\n")
print(f"{'Config':55s} | days | total_$ | daily_$ | sharpe | pct+ | wr   | fill | tpd")
print("-" * 110)
for key, cs in top:
    vol, conv, hold, mode, ct, cr, lat = key
    name = f"v{vol}_c{conv}_h{hold}m_{mode}_ct{ct}r{cr}_lat{lat}"
    print(f"  {name:53s} | {cs['days']:4d} | ${cs['total_pnl']:7.0f} | "
          f"${cs['mean_daily']:7.1f} | {cs['sharpe']:6.2f} | "
          f"{cs['pct_pos']:5.1f}% | {100*cs['win_rate']:.0f}% | "
          f"{100*cs['fill_rate']:.0f}% | {cs['trades_per_day']:.1f}")

# ─────────────────────────────────────────────
# SECTION 10 — ACTIONABLE RECOMMENDATIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 10: ACTIONABLE RECOMMENDATIONS")
print("=" * 70)

# Find single best config
best_configs_22d = sorted(
    [(k, cs) for k, cs in config_stats.items() if cs["days"] >= 5],
    key=lambda x: x[1]["sharpe"],
    reverse=True
)

if best_configs_22d:
    bk, bcs = best_configs_22d[0]
    print(f"\nBest config (>=5 days by Sharpe): "
          f"v{bk[0]}_c{bk[1]}_h{bk[2]}m_{bk[3]}_ct{bk[4]}r{bk[5]}_lat{bk[6]}")
    print(f"  Total P&L: ${bcs['total_pnl']:.0f}  Days: {bcs['days']}  "
          f"Sharpe: {bcs['sharpe']:.2f}  Pct+: {bcs['pct_pos']:.1f}%  "
          f"Win Rate: {100*bcs['win_rate']:.1f}%  Fill: {100*bcs['fill_rate']:.1f}%")

# Number of profitable configs
n_total = len(config_stats)
n_profitable = sum(1 for cs in config_stats.values() if cs["total_pnl"] > 0 and cs["days"] >= 5)
print(f"\nOut of {n_total} configs (≥5 days): {n_profitable} profitable "
      f"({100*n_profitable/n_total:.1f}%)")

# Monthly breakdown (if enough months)
print("\n--- Monthly P&L breakdown (across ALL configs) ---")
month_groups = defaultdict(list)
for r in records:
    if r["trades"] > 0:
        month_groups[r["month"]].append(r["pnl"])
month_names = {12: "Dec-2025", 1: "Jan-2026", 2: "Feb-2026", 3: "Mar-2026"}
for m in sorted(month_groups.keys()):
    pnls = month_groups[m]
    print(f"  {month_names.get(m, str(m)):10s} n={len(pnls):6d}  "
          f"mean=${mean(pnls):8.1f}  pct+={pct_positive(pnls):5.1f}%")

# Parameter consensus: which params appear most in top-20 configs?
print("\n--- Consensus parameters from top-20 configs (≥15 days, by Sharpe) ---")
top20 = sorted(
    [(k, cs) for k, cs in config_stats.items() if cs["days"] >= 5],
    key=lambda x: x[1]["sharpe"], reverse=True
)[:20]
if top20:
    from collections import Counter
    cvol  = Counter(k[0] for k, _ in top20)
    cconv = Counter(k[1] for k, _ in top20)
    chold = Counter(k[2] for k, _ in top20)
    cmode = Counter(k[3] for k, _ in top20)
    clat  = Counter(k[6] for k, _ in top20)
    print(f"  vol_gate:     {dict(cvol)}")
    print(f"  conv_thresh:  {dict(cconv)}")
    print(f"  hold_min:     {dict(chold)}")
    print(f"  mode:         {dict(cmode)}")
    print(f"  latency_ms:   {dict(clat)}")

# Queue position analysis
print("\n--- Queue position vs P&L (are front-of-queue fills better?) ---")
qp_buckets = defaultdict(list)
for r in records:
    for t in r["trades_list"]:
        qp = t.get("queue_position_at_post", 0)
        pnl = t.get("pnl_dollars", 0.0)
        if qp <= 0:
            b = "1 (front)"
        elif qp <= 5:
            b = "2-5"
        elif qp <= 15:
            b = "6-15"
        elif qp <= 30:
            b = "16-30"
        elif qp <= 60:
            b = "31-60"
        else:
            b = "60+"
        qp_buckets[b].append(pnl)

for b in ["1 (front)", "2-5", "6-15", "16-30", "31-60", "60+"]:
    pnls = qp_buckets.get(b, [])
    if pnls:
        print(f"  Queue pos {b:12s}: {len(pnls):6d} trades  "
              f"mean=${mean(pnls):8.1f}  pct+={pct_positive(pnls):5.1f}%")

# Coverage summary
all_dates = sorted({r["date"] for r in records})
print(f"\n--- Data coverage ---")
print(f"  Trading days covered: {len(all_dates)}")
print(f"  Date range: {all_dates[0]} to {all_dates[-1]}")
max_days = max((cs["days"] for cs in config_stats.values()), default=0)
print(f"  Max days for any single config: {max_days}")

# Top configs by total P&L (more robust than Sharpe with few days)
print("\n--- Top 20 configs by TOTAL P&L (>=5 days) ---")
top_by_pnl = sorted(
    [(k, cs) for k, cs in config_stats.items() if cs["days"] >= 5],
    key=lambda x: x[1]["total_pnl"],
    reverse=True
)[:20]
print(f"{'Config':53s} | days | total_$ | daily_$ | sharpe | pct+ | wr | fill")
print("-" * 110)
for key, cs in top_by_pnl:
    vol, conv, hold, mode, ct, cr, lat = key
    name = f"v{vol}_c{conv}_h{hold}m_{mode}_ct{ct}r{cr}_lat{lat}"
    print(f"  {name:51s} | {cs['days']:4d} | ${cs['total_pnl']:7.0f} | "
          f"${cs['mean_daily']:7.1f} | {cs['sharpe']:6.2f} | "
          f"{cs['pct_pos']:5.1f}% | {100*cs['win_rate']:.0f}% | "
          f"{100*cs['fill_rate']:.0f}%")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
