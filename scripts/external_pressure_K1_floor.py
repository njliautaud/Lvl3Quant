#!/usr/bin/env python3
"""
external_pressure_K1_floor.py — Honest deployable-floor test for the P3
book-OFI candidate alpha cell found by external_pressure_stream_v1.py.

Reproduces ONLY the single cell:
    pressure = P3 book-OFI proxy (ofi_book_5s)
    horizon  = 5s
    side     = long, policy = forward-confirmation
but with K=1 (no forward peek, no lookahead). K=1 collapses to:
    enter when entry-event book-OFI sign agrees with side AND
    |book-OFI| >= p75 of the per-day |book-OFI| distribution
    (same magnitude gate the parent used at K=4)

Also computes MIRROR (side=short, same magnitude gate) to confirm not a
sign accident.

Cost convention preserved exactly from parent: COMMISSION_TICKS_RT = 0.376
(passive limit equivalent). hold-to-h returns from target_log_ret_5s (ticks).

NO modification of any existing script. Read-only on data + parent script.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path("/home/jupiter/Lvl3Quant")
OFI_DIR  = ROOT / "data/processed/mbo_events_smart_v3_ofi_features"
PRED_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
OUT_DIR  = ROOT / "output/external_pressure_stream_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_TICKS_RT = 0.376  # preserved from parent
STRIDE = 250
OFFSET = 1499  # v4_idx = OFFSET + k * STRIDE

# Single-cell config
PRESSURE_KEY = "ofi_book_5s"     # P3 raw column in OFI npz
HORIZON      = "5s"
# Gate (HC #428 R1 + task spec)
GATE_NET       = 0.10
GATE_WR        = 0.52
GATE_REGIME    = 0.50
GATE_DAYCONC   = 0.70
GATE_PDAYS_MIN = 30  # absolute floor from task spec


def load_day(date_str: str):
    ofi_path  = OFI_DIR / f"{date_str}_ofi.npz"
    pred_path = PRED_DIR / f"oot_{date_str}.npz"
    if not (ofi_path.exists() and pred_path.exists()):
        return None
    ofi  = np.load(ofi_path)
    pred = np.load(pred_path)

    n_pred = pred["pred_log_ret_1s"].shape[0]
    v4_idx = OFFSET + np.arange(n_pred) * STRIDE
    n_ofi  = ofi[PRESSURE_KEY].shape[0]
    if v4_idx[-1] >= n_ofi:
        cap = (n_ofi - OFFSET) // STRIDE
        v4_idx = v4_idx[:cap]
        n_pred = cap

    p_book = ofi[PRESSURE_KEY][v4_idx]

    rk = f"target_log_ret_{HORIZON}"
    mk = f"mask_log_ret_{HORIZON}"
    r  = pred[rk][:n_pred].astype(np.float64)
    m  = (pred[mk][:n_pred] > 0.5) & np.isfinite(r)

    # Regime label (same convention as parent: mean log_ret_30s sign)
    r30 = pred["target_log_ret_30s"][:n_pred]
    m30 = (pred["mask_log_ret_30s"][:n_pred] > 0.5) & np.isfinite(r30)
    drift = float(np.mean(r30[m30])) if m30.sum() > 0 else 0.0
    if drift > 0.05:
        regime = "green"
    elif drift < -0.05:
        regime = "red"
    else:
        regime = "flat"

    return {
        "date": date_str,
        "p": p_book,
        "ret_h": r,
        "mask_h": m,
        "regime": regime,
        "drift": drift,
    }


def evaluate(day_rows, side: str):
    """K=1 floor: enter on entry-event when sign agrees with side AND
    |p| >= per-day p75 of |p|. No forward look-ahead at all."""
    trade_sign = +1 if side == "long" else -1
    daily_means = []
    daily_n = []
    daily_dates = []
    daily_regimes = []
    all_net = []

    for d in day_rows:
        p     = d["p"]
        r     = d["ret_h"]
        m     = d["mask_h"]
        absp  = np.abs(p)
        # Per-day magnitude p75 — same threshold spirit as parent (parent took
        # p75 of mean_abs over forward window; at K=1 mean_abs == |p[i]| so
        # the natural collapse is p75 of |p| over the day).
        finite = np.isfinite(absp)
        if finite.sum() < 100:
            continue
        thr = np.percentile(absp[finite], 75)
        if side == "long":
            sign_ok = p > 0
        else:
            sign_ok = p < 0
        strong = finite & sign_ok & (absp >= thr)
        cell = strong & m
        n_cell = int(cell.sum())
        if n_cell < 1:
            continue
        net = trade_sign * r[cell] - COMMISSION_TICKS_RT
        daily_means.append(float(np.mean(net)))
        daily_n.append(n_cell)
        daily_dates.append(d["date"])
        daily_regimes.append(d["regime"])
        all_net.append(net)

    if not all_net:
        return None
    arr = np.concatenate(all_net)
    n = int(arr.size)
    if n < 10:
        return None
    daily = np.array(daily_means)
    counts = np.array(daily_n)
    n_days = len(daily)
    prof_days = int((daily > 0).sum())
    net_mean = float(np.mean(arr))
    wr = float(np.mean(arr > 0))
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 1e-9 else 0.0

    green = daily[np.array([rg == "green" for rg in daily_regimes])]
    red   = daily[np.array([rg == "red"   for rg in daily_regimes])]
    def ss(a):
        if len(a) < 2 or a.std() < 1e-9:
            return 0.0
        return float(a.mean() / a.std() * np.sqrt(252))
    sh_g, sh_r = ss(green), ss(red)
    if max(abs(sh_g), abs(sh_r)) > 1e-9:
        regime_imb = abs(sh_g - sh_r) / max(abs(sh_g), abs(sh_r))
    else:
        regime_imb = 0.0
    day_conc = float(counts.max() / counts.sum()) if counts.sum() > 0 else 1.0

    return {
        "side": side,
        "n_events": n,
        "n_days_eval": n_days,
        "net_ticks_per_event": net_mean,
        "win_rate": wr,
        "prof_days": prof_days,
        "sharpe_daily": sharpe,
        "sharpe_green": sh_g,
        "sharpe_red": sh_r,
        "regime_imbalance": regime_imb,
        "day_concentration": day_conc,
        "daily_dates": daily_dates,
        "daily_means": daily_means,
        "daily_counts": daily_n,
    }


def main():
    t0 = time.time()
    dates = sorted([p.stem.replace("_ofi", "") for p in OFI_DIR.glob("*_ofi.npz")])
    dates = [d for d in dates if (PRED_DIR / f"oot_{d}.npz").exists()]
    print(f"[info] candidate dates: {len(dates)}", flush=True)

    day_rows = []
    for d in dates:
        try:
            dd = load_day(d)
            if dd is not None:
                day_rows.append(dd)
        except Exception as e:
            print(f"[skip] {d}: {e}", flush=True)
    print(f"[info] loaded {len(day_rows)} days", flush=True)

    res_long  = evaluate(day_rows, "long")
    res_short = evaluate(day_rows, "short")

    out = {
        "config": {
            "pressure": "P3_book_ofi_5s_proxy (ofi_book_5s)",
            "K": 1,
            "horizon": HORIZON,
            "policy": "entry-event sign + |p|>=per-day p75, no forward peek",
            "cost_ticks_rt": COMMISSION_TICKS_RT,
            "n_days_loaded": len(day_rows),
        },
        "long":  {k: v for k, v in (res_long  or {}).items()
                  if k not in ("daily_dates", "daily_means", "daily_counts")},
        "short_mirror": {k: v for k, v in (res_short or {}).items()
                        if k not in ("daily_dates", "daily_means", "daily_counts")},
        "elapsed_sec": round(time.time() - t0, 1),
    }

    # Save raw daily for long side
    if res_long is not None:
        daily_df = pd.DataFrame({
            "date":   res_long["daily_dates"],
            "n":      res_long["daily_counts"],
            "mean_net_ticks": res_long["daily_means"],
        })
        daily_df.to_csv(OUT_DIR / "K1_floor_daily_long.csv", index=False)
    if res_short is not None:
        daily_df_s = pd.DataFrame({
            "date":   res_short["daily_dates"],
            "n":      res_short["daily_counts"],
            "mean_net_ticks": res_short["daily_means"],
        })
        daily_df_s.to_csv(OUT_DIR / "K1_floor_daily_short.csv", index=False)

    with open(OUT_DIR / "K1_floor_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Verdict logic on LONG side
    verdict_lines = []
    verdict_lines.append("# K=1 Floor Verdict — P3 book-OFI, h=5s, long, forward\n")
    verdict_lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} — "
                         "no lookahead, entry-event only._\n")
    verdict_lines.append("## Config\n")
    for k, v in out["config"].items():
        verdict_lines.append(f"- **{k}**: {v}")
    verdict_lines.append("")

    def fmt(res, label):
        if res is None:
            return [f"## {label}\n_no events_\n"]
        lines = [f"## {label}\n",
                 f"- net_ticks/event: **{res['net_ticks_per_event']:+.4f}**",
                 f"- win_rate: **{res['win_rate']:.4f}**",
                 f"- sharpe_daily: **{res['sharpe_daily']:.2f}**",
                 f"- prof_days: **{res['prof_days']}/{res['n_days_eval']}**",
                 f"- n_events: **{res['n_events']:,}**",
                 f"- sharpe_green/red: {res['sharpe_green']:.2f} / {res['sharpe_red']:.2f} "
                 f"(imbalance {res['regime_imbalance']:.2f})",
                 f"- day_concentration: {res['day_concentration']:.2f}",
                 ""]
        return lines

    verdict_lines += fmt(res_long,  "LONG (deployable floor candidate)")
    verdict_lines += fmt(res_short, "SHORT MIRROR (sign-accident check)")

    # Decision
    L = res_long
    verdict_lines.append("## Verdict\n")
    if L is None:
        verdict_lines.append("**REJECT** — no events at K=1.")
    else:
        net_ok    = L["net_ticks_per_event"] >= GATE_NET
        wr_ok     = L["win_rate"]            >= GATE_WR
        pdays_ok  = L["prof_days"]           >= GATE_PDAYS_MIN
        regime_ok = L["regime_imbalance"]    <= GATE_REGIME
        conc_ok   = L["day_concentration"]   <= GATE_DAYCONC
        deployable = net_ok and wr_ok and pdays_ok and regime_ok and conc_ok

        verdict_lines.append(f"- net_ticks >= {GATE_NET}: {'PASS' if net_ok else 'FAIL'} "
                             f"({L['net_ticks_per_event']:+.4f})")
        verdict_lines.append(f"- WR >= {GATE_WR}: {'PASS' if wr_ok else 'FAIL'} "
                             f"({L['win_rate']:.4f})")
        verdict_lines.append(f"- prof_days >= {GATE_PDAYS_MIN}/{L['n_days_eval']}: "
                             f"{'PASS' if pdays_ok else 'FAIL'} ({L['prof_days']})")
        verdict_lines.append(f"- regime_imbalance <= {GATE_REGIME}: "
                             f"{'PASS' if regime_ok else 'FAIL'} ({L['regime_imbalance']:.2f})")
        verdict_lines.append(f"- day_concentration <= {GATE_DAYCONC}: "
                             f"{'PASS' if conc_ok else 'FAIL'} ({L['day_concentration']:.2f})")
        verdict_lines.append("")
        if deployable:
            verdict_lines.append("**DEPLOYABLE FLOOR CONFIRMED.** K=4's headline edge survives "
                                 "the no-lookahead K=1 collapse.")
        else:
            # If long net < 0.10, the +0.67 was lookahead-driven
            if L["net_ticks_per_event"] < GATE_NET:
                verdict_lines.append(
                    "**REJECT HEADLINE.** Edge collapses without the forward-peek window. "
                    "The +0.67 net at K=4 is lookahead-driven (forward stream-coherence is "
                    "peeking at ~1s of post-entry pressure, which is not available live).")
            else:
                verdict_lines.append(
                    "**MARGINAL.** Net survives but other gates fail — not a clean floor.")

        # Mirror sanity
        if res_short is not None:
            verdict_lines.append("")
            verdict_lines.append(
                f"_Mirror (short, same magnitude gate): "
                f"net={res_short['net_ticks_per_event']:+.4f}, "
                f"WR={res_short['win_rate']:.4f}. "
                + ("Mirror also positive — sign asymmetry weak; possible cost-related "
                   "structural effect."
                   if res_short['net_ticks_per_event'] >= GATE_NET else
                   "Mirror clearly worse, so long-side sign is not a coin-flip accident "
                   "(but that alone doesn't make long deployable).")
            )

    (OUT_DIR / "K1_floor_verdict.md").write_text("\n".join(verdict_lines))
    print("[done] wrote K1_floor_verdict.md and K1_floor_result.json", flush=True)
    print(f"[long]  net={L['net_ticks_per_event'] if L else None}")
    if res_short:
        print(f"[short] net={res_short['net_ticks_per_event']}")


if __name__ == "__main__":
    main()
