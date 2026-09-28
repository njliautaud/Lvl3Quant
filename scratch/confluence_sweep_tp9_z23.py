"""Post-hoc confluence sweep on tp9_z2.3 trades.

Applies regime/gate filters to the per-day trade JSONs and reports
filtered Sortino_daily, total PnL, MDD ($ and %), n_trades for each
filter combination. Capital assumption: $50K/contract.

Filters tested (each binary on/off; 2^k combinations):
  vol_high   — only keep trades where rolling 5-min realized vol > median
  vol_low    — only keep trades where rolling 5-min realized vol < median
  prime_hrs  — only 09:30-11:30 ET (high-liquidity window)
  ofi_agree  — sign(z_score) == sign(OFI) at entry (proxy via signal sign)
  no_first15 — exclude first 15 min after open (high-noise window)

For each combination, compute:
  n_trades, mean_pnl_pct, sortino_daily, total_pnl, mdd_dollars, mdd_pct

We use $50K/contract MDD% denominator. If trade dict lacks regime fields,
we approximate from timestamp only (so vol filters are skipped).
"""
import json
from pathlib import Path
from itertools import product
from datetime import datetime, time as dtime
import numpy as np

CAPITAL = 50_000.0
SWEEP_DIR = Path("/home/jupiter/Lvl3Quant/execution/results/fifo_validation/sim_tp_z_finegrain_9day_20260427_101143")
OUT_PATH = Path("/home/jupiter/Lvl3Quant/scratch/confluence_tp9_z23_results.json")

def load_trades(tp_tag, z_tag):
    by_day = {}
    for fp in sorted(SWEEP_DIR.glob(f"{tp_tag}_{z_tag}_*.json")):
        date = fp.stem.split("_")[-1]
        d = json.loads(fp.read_text())
        by_day[date] = d.get("trades", [])
    return by_day

def trade_time(t):
    """Return (et_hour, et_minute) tuple from trade entry timestamp.

    Trades are tagged with 'entry_ts_ns' (epoch ns UTC). ES futures market open
    09:30 ET == 14:30 UTC (winter) / 13:30 UTC (summer DST). We approximate.
    """
    ts_ns = t.get("fill_time_ns") or t.get("post_time_ns") or t.get("signal_time_ns")
    if ts_ns is None:
        return None
    dt = datetime.utcfromtimestamp(ts_ns / 1e9)
    # Feb 23 - Mar 5 2026 is all EST (DST starts Mar 8). UTC-5.
    et_h = (dt.hour - 5) % 24
    return et_h, dt.minute

def filter_trades(trades, *, prime_hrs=False, no_first15=False):
    out = []
    for t in trades:
        if prime_hrs or no_first15:
            tm = trade_time(t)
            if tm is None:
                # without timestamps we can't filter — keep
                out.append(t)
                continue
            h, m = tm
            if prime_hrs:
                # 09:30-11:30 ET
                in_window = (h == 9 and m >= 30) or (h == 10) or (h == 11 and m < 30)
                if not in_window:
                    continue
            if no_first15:
                # exclude 09:30-09:45 ET
                if h == 9 and 30 <= m < 45:
                    continue
        out.append(t)
    return out

def metrics(by_day, capital=CAPITAL):
    daily_pnls = []
    all_pnls = []
    for date in sorted(by_day.keys()):
        ds = sum(t["pnl_dollars"] for t in by_day[date])
        daily_pnls.append(ds)
        all_pnls.extend(t["pnl_dollars"] for t in by_day[date])
    if not all_pnls:
        return None
    arr_d = np.array(daily_pnls, dtype=float)
    arr_t = np.array(all_pnls, dtype=float)
    cum = np.cumsum(arr_d)
    drawdown = cum - np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    mdd_dollars = float(-drawdown.min()) if drawdown.size else 0.0
    mdd_pct = mdd_dollars / capital * 100.0
    # sortino daily
    neg = arr_d[arr_d < 0]
    sortino_d = float(arr_d.mean() / neg.std(ddof=1)) if neg.size > 1 and neg.std(ddof=1) > 0 else float("inf") if arr_d.mean() > 0 else 0.0
    return {
        "n_trades": int(arr_t.size),
        "n_days": int(arr_d.size),
        "n_positive_days": int((arr_d > 0).sum()),
        "total_pnl": float(arr_t.sum()),
        "total_pnl_pct": float(arr_t.sum() / capital * 100.0),
        "mean_per_trade": float(arr_t.mean()),
        "sortino_daily": sortino_d,
        "mdd_dollars": mdd_dollars,
        "mdd_pct": mdd_pct,
        "win_rate": float((arr_t > 0).mean()),
    }

def main():
    base = load_trades("tp9", "z2.3")
    if not base:
        print("No trades loaded")
        return
    flat = [t for d in base.values() for t in d]
    print(f"Loaded {len(flat)} total trades across {len(base)} days from tp9_z2.3")

    # check timestamp availability
    with_ts = sum(1 for t in flat if t.get("fill_time_ns") or t.get("post_time_ns") or t.get("signal_time_ns"))
    print(f"Trades with entry timestamp: {with_ts}/{len(flat)}")
    if with_ts == 0:
        print("WARN: no timestamps in trades — time filters will keep everything")
        # fall back to printing key set
        print("Sample trade keys:", list(flat[0].keys())[:20])

    results = {}
    combos = list(product([False, True], [False, True]))   # prime_hrs, no_first15
    print(f"\n{'filter':<25}{'n_tr':>6} {'pos':>5} {'tot_pnl':>11} {'pnl_%':>8} {'sortino':>9} {'mdd_%':>8} {'wr':>7}")
    print("-" * 92)
    for prime, nf15 in combos:
        name = "+".join([n for n, on in [("prime", prime), ("no_first15", nf15)] if on]) or "baseline"
        filt = {date: filter_trades(trades, prime_hrs=prime, no_first15=nf15) for date, trades in base.items()}
        m = metrics(filt)
        if m:
            results[name] = m
            print(f"{name:<25}{m['n_trades']:>6} {m['n_positive_days']:>2}/{m['n_days']:<2}  "
                  f"${m['total_pnl']:>9,.0f} {m['total_pnl_pct']:>7.2f}% "
                  f"{m['sortino_daily']:>8.3f}  {m['mdd_pct']:>6.2f}% {m['win_rate']:>6.1%}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_PATH}")

if __name__ == "__main__":
    main()
