#!/usr/bin/env python3
"""
headc_confluence_v342.py — Does head-C first-passage probability add INCREMENTAL
value as a confluence/gating feature on top of CNN-Mamba v3.4.2 champion?

Head-C standalone FAILED FIFO gates (K=1 and K=2, decided 2026-06-11, see
RUN_HISTORY.md). This measures the CONFLUENCE question only.

Alignment method (EXACT, no resampling):
  Champion 47-day per-date NPZs (output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate)
  contain no timestamps, but sample positions are deterministic:
    label_idx = start + 1499 for start in range(0, n_events-1500+1, 250),
    keeping positions where labels_1s is not NaN  (verified: reconstructed count
    matches NPZ length exactly, 81526 == 81526 on 20260306).
  Head-C NPZs save ts_ns directly. Head-C grid (pos = 2999 + 2000k) is a strict
  subset of the champion grid (pos = 1499 + 250m, m = 6 + 8k), so we merge by
  EXACT int64 timestamp equality. Verified 10160/10160 exact matches on 20260306.

Champion targets in these NPZs are in TICKS despite the "log_ret" name
(verified target_log_ret_1s == raw labels_1s ticks exactly).

Part A (stats): correlation, conditional fwd returns by champion decile x
  head-C agreement, full conviction-threshold sweep, matched-count comparison.
Part B (FIFO): champion-alone vs head-C-gated vs matched-count champion-alone,
  canonical FIFOReplayEngine, TP=SL=K, cancel=5s, hold=7.5s, limit+market.
  pnl_ticks_net is net of $4.70 RT commission (0.376 ticks).

CPU only. Run:
  python3 scripts/headc_confluence_v342.py [--skip-fifo] [--k 2]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

LVL3 = Path("/home/nick/Lvl3Quant")
DATA_DIR = LVL3 / "data" / "processed" / "mbo_events_smart_v3"
CHAMP_DIR = LVL3 / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "oot_47day_perdate"
HEADC_DIRS = {2: LVL3 / "output" / "p_alpha_headc_v1",
              1: LVL3 / "output" / "p_alpha_headc_v1_K1"}
LOG_DIR = LVL3 / "logs"
W_T1, STRIDE_CHAMP = 1500, 250
HORIZON_S = 5.0
CANCEL_NS = int(HORIZON_S * 1e9)
HOLD_NS = int(1.5 * HORIZON_S * 1e9)
COST_PASSIVE = 0.376
COST_MARKET = 1.376
# head-C conviction thresholds g = |2p-1| >= tau  (tau=0.0 -> agreement-only)
G_TAUS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
CHAMP_TOP_PCTS = [10, 20]   # champion entry = top X% |pred_1s| per day

ap = argparse.ArgumentParser()
ap.add_argument("--skip-fifo", action="store_true")
ap.add_argument("--k", type=int, default=None, help="restrict to one K family")
ap.add_argument("--fifo-tau", type=float, nargs="*", default=[0.0, 0.2, 0.3],
                help="conviction taus to run through FIFO")
args = ap.parse_args()

ts_now = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"headc_confluence_v342_{ts_now}.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("headc_confl")


def load_day(k_ticks: int, headc_npz: Path):
    """Return merged dict for one day or None."""
    date = headc_npz.stem.split("_oot_")[1]
    champ_npz = CHAMP_DIR / f"oot_{date}.npz"
    if not champ_npz.exists():
        log.warning(f"{date}: no champion NPZ, skip")
        return None
    raw_path = DATA_DIR / f"{date}_mbo_events.npz"
    raw = np.load(raw_path, allow_pickle=True)
    ts_raw = raw["timestamps"].astype(np.int64)
    lab1 = raw["labels_1s"]
    n = len(ts_raw)
    starts = np.arange(0, n - W_T1 + 1, STRIDE_CHAMP)
    li = starts + W_T1 - 1
    valid = ~np.isnan(lab1[li])
    li = li[valid]
    ts_champ = ts_raw[li]
    del raw, ts_raw, lab1

    z = np.load(champ_npz)
    c1 = z["pred_log_ret_1s"]
    c5 = z["pred_log_ret_5s"]
    t1 = z["target_log_ret_1s"]   # ticks (verified)
    t5 = z["target_log_ret_5s"]   # ticks
    if len(c1) != len(ts_champ):
        log.error(f"{date}: reconstruction mismatch champ={len(c1)} recon={len(ts_champ)} — SKIP")
        return None

    h = np.load(headc_npz)
    h_ts = h["ts_ns"].astype(np.int64)
    h_p = h["predictions"].astype(np.float64)
    h_lab = h["labels"].astype(np.float64)

    # exact merge: champion index for each headc ts
    order = np.argsort(ts_champ)
    pos = np.searchsorted(ts_champ[order], h_ts)
    pos = np.clip(pos, 0, len(order) - 1)
    cidx = order[pos]
    exact = ts_champ[cidx] == h_ts
    match_rate = float(exact.mean())
    if match_rate < 0.99:
        log.warning(f"{date}: exact ts match only {match_rate:.4f}")
    cidx = cidx[exact]
    return {
        "date": date, "k": k_ticks, "match_rate": match_rate,
        "n": int(exact.sum()),
        "ts": h_ts[exact],
        "c1": c1[cidx].astype(np.float64), "c5": c5[cidx].astype(np.float64),
        "t1": t1[cidx].astype(np.float64), "t5": t5[cidx].astype(np.float64),
        "hp": h_p[exact], "hlab": h_lab[exact],
    }


def spearman(a, b):
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return float(r)


def _nancorr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 10:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def day_stats(d):
    """Per-day correlations + conditional analysis.

    SIGN ANOMALY (discovered 2026-06-11): head-C labels were built by
    cumulatively summing diff(labels_1s), which reconstructs first-passage of
    the 1s-FORWARD-RETURN process, NOT the price path. Empirically the raw
    signed head-C (2p-1) is NEGATIVELY correlated with realized forward
    returns (corr vs t1 ~ -0.15, labels vs t1 ~ -0.4 on every day). We
    therefore orient the signal EMPIRICALLY: s = -(2p-1). Raw correlations are
    reported under corr_raw for honesty.
    """
    s_raw = 2.0 * d["hp"] - 1.0
    s = -s_raw                        # empirically oriented signed head-C
    c1, c5, t1, t5 = d["c1"], d["c5"], d["t1"], d["t5"]
    out = {"date": d["date"], "n": d["n"], "match_rate": d["match_rate"],
           "corr_raw": {
               "pearson_sraw_c1": float(np.corrcoef(s_raw, c1)[0, 1]),
               "pearson_sraw_t1": _nancorr(s_raw, t1),
               "pearson_hlab_t1": _nancorr(2.0 * d["hlab"] - 1.0, t1),
           },
           "corr": {
               "pearson_s_c1": float(np.corrcoef(s, c1)[0, 1]),
               "pearson_s_c5": float(np.corrcoef(s, c5)[0, 1]),
               "spearman_s_c1": spearman(s, c1),
               "spearman_s_c5": spearman(s, c5),
               "pearson_s_t1": _nancorr(s, t1),
               "pearson_s_t5": _nancorr(s, t5),
               "pearson_c1_t1": _nancorr(c1, t1),
               "pearson_c1_t5": _nancorr(c1, t5),
           }}
    # Incremental (orthogonal) information: residualize both s and t5 on c1,
    # then correlate residuals (per-day partial correlation).
    m = ~np.isnan(t5)
    if m.sum() > 100:
        cc, ss, tt = c1[m], s[m], t5[m]
        bc = np.polyfit(cc, tt, 1)
        bs = np.polyfit(cc, ss, 1)
        rt = tt - np.polyval(bc, cc)
        rs = ss - np.polyval(bs, cc)
        out["corr"]["partial_s_t5_given_c1"] = float(np.corrcoef(rs, rt)[0, 1])
        m1 = ~np.isnan(t1)
        bc1 = np.polyfit(c1[m1], t1[m1], 1)
        bs1 = np.polyfit(c1[m1], s[m1], 1)
        out["corr"]["partial_s_t1_given_c1"] = float(np.corrcoef(
            s[m1] - np.polyval(bs1, c1[m1]), t1[m1] - np.polyval(bc1, c1[m1]))[0, 1])
    cdir = np.sign(c1)
    agree = s * cdir > 0
    g = np.abs(s)
    out["agree_rate"] = float(agree.mean())

    # deciles of |c1| (per-day)
    rank = np.argsort(np.argsort(np.abs(c1))) / max(len(c1) - 1, 1)
    dec_rows = []
    for dlo in range(10):
        m = (rank >= dlo / 10) & (rank < (dlo + 1) / 10 if dlo < 9 else rank <= 1.0)
        for cond, name in [(m, "all"), (m & agree, "agree"), (m & ~agree, "disagree")]:
            if cond.sum() == 0:
                continue
            r1 = cdir[cond] * t1[cond]
            r5 = cdir[cond] * t5[cond]
            r1v, r5v = r1[~np.isnan(r1)], r5[~np.isnan(r5)]
            dec_rows.append({"decile": dlo, "cond": name, "n": int(cond.sum()),
                             "r1_ticks": float(r1v.mean()) if len(r1v) else None,
                             "r5_ticks": float(r5v.mean()) if len(r5v) else None,
                             "hit1": float((r1v > 0).mean()) if len(r1v) else None,
                             "hit5": float((r5v > 0).mean()) if len(r5v) else None})
    out["deciles"] = dec_rows

    # sweep: champion top-X% entries, gate on agree & g>=tau, vs matched-count
    sweeps = []
    for pct in CHAMP_TOP_PCTS:
        base = rank >= 1.0 - pct / 100.0
        n_base = int(base.sum())

        def _mh(r):
            v = r[~np.isnan(r)]
            return (float(v.mean()), float((v > 0).mean())) if len(v) else (None, None)

        r5b, h5b = _mh(cdir[base] * t5[base])
        for tau in G_TAUS:
            gate = base & agree & (g >= tau)
            n_g = int(gate.sum())
            row = {"pct": pct, "tau": tau, "n_base": n_base, "n_gated": n_g,
                   "base_r5": r5b, "base_hit5": h5b}
            if n_g > 0:
                row["gated_r5"], row["gated_hit5"] = _mh(cdir[gate] * t5[gate])
                thr_idx = np.argsort(np.abs(c1))[-n_g:]
                row["matched_r5"], row["matched_hit5"] = _mh(cdir[thr_idx] * t5[thr_idx])
            sweeps.append(row)
    out["sweep"] = sweeps
    return out


def run_fifo(days, k_ticks, taus):
    """FIFO replay: champ-alone vs gated vs matched, per day, limit+market."""
    fifo_path = LVL3 / "alpha_discovery" / "deep_models" / "fifo_market_replay.py"
    spec = importlib.util.spec_from_file_location("fifo_market_replay", str(fifo_path))
    fmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fmod)
    Engine = fmod.FIFOReplayEngine

    def net_ticks(t):
        if hasattr(t, "pnl_ticks_net"):
            return float(t.pnl_ticks_net)
        if isinstance(t, dict):
            return float(t.get("pnl_ticks_net", np.nan))
        return float("nan")

    rows = []
    for d in days:
        date = d["date"]
        c1 = d["c1"]
        s = -(2.0 * d["hp"] - 1.0)   # empirically oriented (see day_stats docstring)
        g = np.abs(s)
        cdir = np.sign(c1)
        agree = s * cdir > 0
        rank = np.argsort(np.argsort(np.abs(c1))) / max(len(c1) - 1, 1)
        base = rank >= 0.90   # champion top-10%

        sets = {"champ_alone": np.where(base)[0]}
        for tau in taus:
            gate = base & agree & (g >= tau)
            n_g = int(gate.sum())
            sets[f"gated_tau{tau:.2f}"] = np.where(gate)[0]
            if n_g > 0:
                sets[f"matched_tau{tau:.2f}"] = np.argsort(np.abs(c1))[-n_g:]
        log.info(f"FIFO {date} K={k_ticks}: " +
                 ", ".join(f"{k}={len(v)}" for k, v in sets.items()))
        t0 = time.time()
        try:
            eng = Engine(date=date, cancel_after_ns=CANCEL_NS, max_hold_ns=HOLD_NS)
        except Exception as e:
            log.warning(f"FIFO {date}: engine load failed: {e}")
            continue
        log.info(f"FIFO {date}: engine loaded in {time.time()-t0:.0f}s")

        for set_name, idx in sets.items():
            if len(idx) == 0:
                continue
            sigs = [{"ts_ns": int(d["ts"][i]),
                     "direction": "long" if cdir[i] > 0 else "short",
                     "strength": float(abs(c1[i]))} for i in sorted(idx)]
            for ot in ["limit", "market"]:
                try:
                    trades = eng.simulate(signals=sigs, tp_ticks=float(k_ticks),
                                          sl_ticks=float(k_ticks), order_type=ot)
                except Exception as e:
                    log.warning(f"FIFO {date} {set_name} {ot}: {e}")
                    continue
                net = np.array([net_ticks(t) for t in trades], dtype=np.float64)
                net = net[~np.isnan(net)]
                if len(net) == 0:
                    rows.append({"date": date, "k": k_ticks, "set": set_name,
                                 "order_type": ot, "n_signals": len(sigs),
                                 "n_trades": 0})
                    continue
                std = float(net.std(ddof=1)) if len(net) > 1 else float("nan")
                rows.append({"date": date, "k": k_ticks, "set": set_name,
                             "order_type": ot, "n_signals": len(sigs),
                             "n_trades": int(len(net)),
                             "ntpt": float(net.mean()),
                             "sum_net": float(net.sum()),
                             "wr": float((net > 0).mean()),
                             "sharpe": float(net.mean() / std) if std and std > 0 else None})
                log.info(f"  {date} {set_name:>16s} {ot:>6s}: n={len(net):4d} "
                         f"ntpt={net.mean():+.3f} WR={(net>0).mean():.3f}")
        del eng
    return rows


def aggregate_fifo(rows):
    """Trade-weighted and day-weighted summaries per (set, order_type)."""
    out = {}
    keys = sorted({(r["set"], r["order_type"]) for r in rows if r.get("n_trades", 0) > 0})
    for st, ot in keys:
        rs = [r for r in rows if r["set"] == st and r["order_type"] == ot
              and r.get("n_trades", 0) > 0]
        ntpts = np.array([r["ntpt"] for r in rs])
        nts = np.array([r["n_trades"] for r in rs])
        tw = float(sum(r["sum_net"] for r in rs) / nts.sum())
        dw = float(ntpts.mean())
        ds = float(ntpts.mean() / ntpts.std(ddof=1)) if len(ntpts) > 1 and ntpts.std(ddof=1) > 0 else None
        out[f"{st}|{ot}"] = {
            "n_days": len(rs), "total_trades": int(nts.sum()),
            "ntpt_trade_weighted": tw, "ntpt_day_weighted": dw,
            "day_sharpe": ds,
            "n_days_positive": int((ntpts > 0).sum()),
            "per_day_ntpt": {r["date"]: round(r["ntpt"], 4) for r in rs},
        }
    return out


def main():
    results = {"meta": {
        "script": "headc_confluence_v342.py", "ts": ts_now,
        "alignment": "exact int64 ts match; champion positions reconstructed "
                     "deterministically (start+1499, stride 250, non-NaN labels_1s); "
                     "head-C grid is exact subset of champion grid",
        "costs": {"passive_ticks": COST_PASSIVE, "market_ticks": COST_MARKET,
                  "note": "FIFO pnl_ticks_net already net of $4.70 RT commission"},
        "sign_anomaly": "head-C labels reconstruct first-passage of the "
                        "1s-forward-return process (cumsum of diff(labels_1s)), "
                        "NOT the price path; raw signed head-C anticorrelates "
                        "with realized fwd returns. Signal oriented as -(2p-1) "
                        "throughout (raw correlations kept under corr_raw).",
        "champion_source": str(CHAMP_DIR),
        "horizon_s": HORIZON_S, "cancel_s": CANCEL_NS / 1e9, "hold_s": HOLD_NS / 1e9,
    }}

    ks = [args.k] if args.k else [2, 1]
    for k in ks:
        hdir = HEADC_DIRS[k]
        npzs = sorted(hdir.glob("fold_*_oot_*.npz"))
        log.info(f"=== K={k}: {len(npzs)} head-C OOT days from {hdir.name} ===")
        days = []
        for f in npzs:
            d = load_day(k, f)
            if d is not None:
                days.append(d)
        kres = {"n_days": len(days), "per_day": [], }
        pooled_s, pooled_c1, pooled_c5 = [], [], []
        for d in days:
            st = day_stats(d)
            kres["per_day"].append(st)
            pooled_s.append(-(2.0 * d["hp"] - 1.0))  # empirically oriented
            pooled_c1.append(d["c1"])
            pooled_c5.append(d["c5"])
        ps, pc1, pc5 = map(np.concatenate, (pooled_s, pooled_c1, pooled_c5))
        kres["pooled_corr"] = {
            "pearson_s_c1": float(np.corrcoef(ps, pc1)[0, 1]),
            "pearson_s_c5": float(np.corrcoef(ps, pc5)[0, 1]),
            "spearman_s_c1": spearman(ps, pc1),
            "spearman_s_c5": spearman(ps, pc5),
            "n": int(len(ps)),
        }
        log.info(f"K={k} pooled corr(signed headc, champ): {kres['pooled_corr']}")

        # aggregate the sweep across days: day-weighted and trade-weighted
        agg = {}
        for pct in CHAMP_TOP_PCTS:
            for tau in G_TAUS:
                cells = [r for st_ in kres["per_day"] for r in st_["sweep"]
                         if r["pct"] == pct and r["tau"] == tau
                         and r.get("gated_r5") is not None
                         and r.get("matched_r5") is not None]
                if not cells:
                    continue
                ng = np.array([c["n_gated"] for c in cells], dtype=float)
                agg[f"pct{pct}_tau{tau:.2f}"] = {
                    "n_days": len(cells),
                    "total_gated": int(ng.sum()),
                    "frac_kept": float(ng.sum() / sum(c["n_base"] for c in cells)),
                    "base_r5_dw": float(np.mean([c["base_r5"] for c in cells])),
                    "gated_r5_dw": float(np.mean([c["gated_r5"] for c in cells])),
                    "matched_r5_dw": float(np.mean([c["matched_r5"] for c in cells])),
                    "base_r5_tw": float(np.average([c["base_r5"] for c in cells],
                                                   weights=[c["n_base"] for c in cells])),
                    "gated_r5_tw": float(np.average([c["gated_r5"] for c in cells], weights=ng)),
                    "matched_r5_tw": float(np.average([c["matched_r5"] for c in cells], weights=ng)),
                    "gated_beats_matched_days": int(sum(
                        1 for c in cells if c["gated_r5"] > c["matched_r5"])),
                    "gated_hit5_dw": float(np.mean([c["gated_hit5"] for c in cells])),
                    "matched_hit5_dw": float(np.mean([c["matched_hit5"] for c in cells])),
                }
        kres["sweep_agg"] = agg
        for name, a in agg.items():
            log.info(f"K={k} {name}: kept={a['frac_kept']:.2f} "
                     f"gated_r5={a['gated_r5_dw']:+.3f} matched_r5={a['matched_r5_dw']:+.3f} "
                     f"base_r5={a['base_r5_dw']:+.3f} gated>matched on "
                     f"{a['gated_beats_matched_days']}/{a['n_days']} days")

        if not args.skip_fifo:
            fifo_rows = run_fifo(days, k, args.fifo_tau)
            kres["fifo_per_day"] = fifo_rows
            kres["fifo_agg"] = aggregate_fifo(fifo_rows)
            log.info(f"K={k} FIFO aggregate:\n" + json.dumps(kres["fifo_agg"], indent=2))

        results[f"K{k}"] = kres

    out_json = LOG_DIR / f"headc_confluence_v342_results_{ts_now}.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info(f"Wrote {out_json}")

    # MLflow (Neptune-local), best-effort
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("p_alpha_headc_firstpassage_v1")
        with mlflow.start_run(run_name=f"headc_confluence_v342_{ts_now}"):
            for k in ks:
                kr = results.get(f"K{k}", {})
                pc = kr.get("pooled_corr", {})
                if pc:
                    mlflow.log_metric(f"K{k}_pearson_s_c1", pc["pearson_s_c1"])
                    mlflow.log_metric(f"K{k}_spearman_s_c1", pc["spearman_s_c1"])
            mlflow.log_artifact(str(out_json))
            mlflow.log_artifact(str(LOG_FILE))
    except Exception as e:
        log.warning(f"MLflow logging skipped: {e}")


if __name__ == "__main__":
    main()
