"""
v33_gate_combos_v4.py — Deep combo sweep on K=2 LONG with intra_vol30s as base filter.

Tests: intra_vol30s thresh × bucket × per-day cap × intraday momentum
Goal: find any combination with positive t/fill AND lowest possible max_day_conc.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/regime_analysis")
SIG = OUT / "k2_long_signals_with_regime_v2.csv"
JOUT = OUT / "gate_combos_v4.json"


def stats(d, label):
    filled = d[d.filled == True]
    n_sig = len(d); n_fill = len(filled)
    if n_fill == 0:
        return None
    net = filled.net_ticks.values
    pos = net[net>0].sum(); neg = -net[net<0].sum()
    pday = filled.groupby("date").net_ticks.agg(["count","sum","mean"])
    pday = {str(k): {"n": int(v["count"]), "t": float(v["sum"]), "tpf": float(v["mean"])}
            for k, v in pday.to_dict("index").items()}
    cnts = filled.groupby("date").size()
    return dict(
        gate=label, n_sig=int(n_sig), n_fill=int(n_fill),
        tpf=float(net.mean()), wr=float((net>0).mean()),
        pf=float(pos/neg) if neg>0 else float("inf"),
        total_t=float(net.sum()),
        max_day=float(cnts.max()/n_fill),
        n_days=len(pday), per_day=pday,
    )


def cap_per_day(df, n_max):
    """Within each day, keep at most n_max trades (chronological order)."""
    return df.sort_values(["date", "ts_ns"]).groupby("date").head(n_max).reset_index(drop=True)


def main():
    t0 = time.time()
    df = pd.read_csv(SIG)
    print(f"Loaded {len(df)} events, {int(df.filled.sum())} filled")

    results = []

    # Base intraday vol thresholds (the winning filter family)
    for vthr in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        base = df[df.vol_30s_ticks < vthr]
        s = stats(base, f"vol30s<{vthr}")
        if s: results.append(s)

        # × bucket filter
        for skip in ["open_0930_1030", None]:
            if skip:
                d2 = base[base.bucket != skip]
                lbl = f"vol30s<{vthr}_skipopen"
            else:
                continue
            s = stats(d2, lbl)
            if s: results.append(s)

        # × intraday drift positive
        d3 = base[base.intraday_drift_ticks > 0]
        s = stats(d3, f"vol30s<{vthr}_intra>0")
        if s: results.append(s)

        # × per-day caps (HC #344 attempt)
        for cap in [5, 10, 15, 20]:
            d4 = cap_per_day(base, cap)
            s = stats(d4, f"vol30s<{vthr}_cap{cap}/d")
            if s: results.append(s)

        # Composite: skipopen + intra>0 + cap
        d5 = base[(base.bucket != "open_0930_1030") & (base.intraday_drift_ticks > 0)]
        for cap in [5, 10, 15]:
            d6 = cap_per_day(d5, cap)
            s = stats(d6, f"vol30s<{vthr}_skipopen_intra>0_cap{cap}/d")
            if s: results.append(s)

    # Filter to deployable: tpf>0 + n_fill>=20
    deployable = [r for r in results if r["tpf"] > 0 and r["n_fill"] >= 20]
    deployable.sort(key=lambda r: (-r["tpf"], r["max_day"]))

    print(f"\n=== TOP 20 (tpf>0, n_fill>=20), ranked by tpf desc / max_day asc ===")
    print(f"{'gate':<46} {'n_sig':>5} {'n_fill':>6} {'tpf':>6} {'wr':>5} {'pf':>5} {'mdc':>5} {'days':>4}")
    for r in deployable[:20]:
        pf_disp = f"{r['pf']:.2f}" if r['pf'] != float("inf") else "inf"
        print(f"{r['gate']:<46} {r['n_sig']:>5} {r['n_fill']:>6} {r['tpf']:>+6.2f} {r['wr']:>5.2f} {pf_disp:>5} {r['max_day']:>5.2f} {r['n_days']:>4}")

    # Try to find ANY HC #344-passing combo
    hc344 = [r for r in results if r["max_day"] <= 0.20 and r["n_fill"] >= 20 and r["tpf"] > 0]
    print(f"\n=== HC #344 PASSING (max_day<=0.20 AND n_fill>=20 AND tpf>0): {len(hc344)} ===")
    for r in hc344:
        print(f"  {r['gate']}: n={r['n_fill']} tpf={r['tpf']:+.2f} mdc={r['max_day']:.2f} wr={r['wr']:.2f}")

    out = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        total_combos=len(results),
        deployable_count=len(deployable),
        hc344_passing=len(hc344),
        top_20=deployable[:20],
        hc344_winners=hc344,
        all_results=results,
        elapsed=round(time.time()-t0, 2),
    )
    JOUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JOUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nDone in {time.time()-t0:.2f}s — wrote {JOUT}")


if __name__ == "__main__":
    main()
