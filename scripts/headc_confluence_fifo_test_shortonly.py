#!/usr/bin/env python3
"""
headc_confluence_fifo_test_shortonly.py — Follow-up to headc_confluence_fifo_test.py.

Champion baseline reconfigured as the production-style SHORT-ONLY band:
  - band05: short top 0.5% (most-negative pred_log_ret_1s per day)  [HC #441 conv]
  - band10: short top 10% (decay-analysis: short side carries the edge)
Geometry unchanged (HC #441 PRIMARY): TP=3.00 SL=0.50 hold=1.5s cancel=10s.

If a short-only baseline is negative on the 8 OOT days, that is itself the
finding. If positive, test head-C K=2 agreement (shorts need P(up-first) <=
1-tau, tau in {0.55,0.60,0.65,0.70}, staleness <= 5s).

Efficiency: band05 ⊂ band10 and all tau arms are subsets, so we simulate the
band10 signal set once per day per order type and derive every arm by
filtering trades on signal_ts_ns (orders independent in the engine).

Outputs: results_per_day_shortonly.csv + summary_shortonly.json in
output/headc_confluence_test/, MLflow run in p_alpha_headc_firstpassage_v1.
CPU only, one day at a time.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

LVL3_ROOT = Path("/home/nick/Lvl3Quant")
CHAMP_DIR = LVL3_ROOT / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "oot_47day_perdate"
HEADC_DIR = LVL3_ROOT / "output" / "p_alpha_headc_v1"
MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
OUT_DIR = LVL3_ROOT / "output" / "headc_confluence_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LVL3_ROOT / "logs" / f"headc_confluence_shortonly_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("headc_confluence_shortonly")

TP_TICKS = 3.00
SL_TICKS = 0.50
HOLD_NS = int(1.5 * 1e9)
CANCEL_NS = int(10.0 * 1e9)
ORDER_TYPES = ["limit", "market"]

CHAMP_WINDOW = 1500
CHAMP_STRIDE = 250

BANDS = {"band05": 0.005, "band10": 0.10}   # short-only fractions
TAUS = [0.55, 0.60, 0.65, 0.70]
HEADC_STALENESS_NS = int(5.0 * 1e9)

DATES = ["20260306", "20260309", "20260310", "20260311",
         "20260312", "20260313", "20260316", "20260317"]
HEADC_FOLD = {d: i for i, d in enumerate(DATES)}

ARMS = []
for b in BANDS:
    ARMS.append(f"{b}_baseline")
    for t in TAUS:
        ARMS.append(f"{b}_conf_tau{int(t*100)}")

fifo_path = LVL3_ROOT / "alpha_discovery" / "deep_models" / "fifo_market_replay.py"
spec = importlib.util.spec_from_file_location("fifo_market_replay", str(fifo_path))
fmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmod)
FIFOReplayEngine = fmod.FIFOReplayEngine


def trade_net_ticks(t) -> float:
    if hasattr(t, "pnl_ticks_net"):
        return float(t.pnl_ticks_net)
    if isinstance(t, dict):
        return float(t.get("pnl_ticks_net", np.nan))
    return float("nan")


def trade_sig_ts(t) -> int:
    if hasattr(t, "signal_ts_ns"):
        return int(t.signal_ts_ns)
    if isinstance(t, dict):
        return int(t.get("signal_ts_ns", 0))
    return 0


def champion_signal_ts(date: str, preds: np.ndarray) -> np.ndarray:
    with np.load(MBO_DIR / f"{date}_mbo_events.npz", allow_pickle=True) as z:
        ts = z["timestamps"].astype(np.int64)
        lab1 = z["labels_1s"]
    pos = np.arange(CHAMP_WINDOW - 1, len(ts), CHAMP_STRIDE)
    pos = pos[~np.isnan(lab1[pos])]
    if len(pos) != len(preds):
        raise RuntimeError(f"{date}: alignment mismatch {len(pos)} vs {len(preds)}")
    return ts[pos]


def headc_prob_at(sig_ts, hc_ts, hc_p):
    idx = np.searchsorted(hc_ts, sig_ts, side="right") - 1
    prob = np.full(len(sig_ts), np.nan)
    ok = idx >= 0
    age = np.where(ok, sig_ts - hc_ts[np.clip(idx, 0, None)], np.iinfo(np.int64).max)
    ok &= age <= HEADC_STALENESS_NS
    prob[ok] = hc_p[idx[ok]]
    return prob


def metrics_row(arm, date, ot, n_signals, net):
    net = np.asarray(net, dtype=np.float64)
    net = net[~np.isnan(net)]
    if len(net) == 0:
        return {"arm": arm, "date": date, "order_type": ot,
                "n_signals": int(n_signals), "n_trades": 0,
                "net_ticks_per_trade": float("nan"), "sum_net_ticks": 0.0,
                "win_rate": float("nan"), "pf": float("nan"), "sharpe": float("nan")}
    wins, losses = net[net > 0], net[net < 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    std = float(net.std(ddof=1)) if len(net) > 1 else float("nan")
    return {"arm": arm, "date": date, "order_type": ot,
            "n_signals": int(n_signals), "n_trades": int(len(net)),
            "net_ticks_per_trade": float(net.mean()),
            "sum_net_ticks": float(net.sum()),
            "win_rate": float((net > 0).mean()), "pf": pf,
            "sharpe": float(net.mean() / std) if std and std > 0 else float("nan")}


def main():
    rows = []
    for date in DATES:
        t0 = time.time()
        with np.load(CHAMP_DIR / f"oot_{date}.npz", allow_pickle=False) as z:
            cpred = z["pred_log_ret_1s"].astype(np.float64)
        cts = champion_signal_ts(date, cpred)

        fold = HEADC_FOLD[date]
        with np.load(HEADC_DIR / f"fold_{fold:02d}_oot_{date}.npz",
                     allow_pickle=False) as z:
            hc_ts = z["ts_ns"].astype(np.int64)
            hc_p = z["predictions"].astype(np.float64)

        # Short-only bands: most-negative predictions
        thr = {b: np.quantile(cpred, frac) for b, frac in BANDS.items()}
        masks = {b: cpred <= thr[b] for b in BANDS}
        # Superset = band10
        sup_idx = np.where(masks["band10"])[0]
        sup_ts = cts[sup_idx]
        sup_str = np.abs(cpred[sup_idx])
        prob = headc_prob_at(sup_ts, hc_ts, hc_p)
        cov = float(np.isfinite(prob).mean())
        log.info(f"{date}: n_preds={len(cpred)} band10_shorts={len(sup_idx)} "
                 f"band05_shorts={int(masks['band05'].sum())} headc_cov={cov:.3f}")

        arm_ts, arm_nsig = {}, {}
        in_band05 = masks["band05"][sup_idx]
        for b in BANDS:
            bm = in_band05 if b == "band05" else np.ones(len(sup_idx), dtype=bool)
            arm_ts[f"{b}_baseline"] = set(int(t) for t in sup_ts[bm])
            arm_nsig[f"{b}_baseline"] = int(bm.sum())
            for tau in TAUS:
                agree = bm & np.isfinite(prob) & (prob <= 1.0 - tau)
                name = f"{b}_conf_tau{int(tau*100)}"
                arm_ts[name] = set(int(t) for t in sup_ts[agree])
                arm_nsig[name] = int(agree.sum())
            log.info(f"  {b}: baseline={arm_nsig[f'{b}_baseline']} " +
                     " ".join(f"tau{int(t*100)}={arm_nsig[f'{b}_conf_tau{int(t*100)}']}"
                              for t in TAUS))

        signals = [{"ts_ns": int(t), "direction": "short", "strength": float(s)}
                   for t, s in zip(sup_ts, sup_str)]

        try:
            eng = FIFOReplayEngine(date=date, cancel_after_ns=CANCEL_NS,
                                   max_hold_ns=HOLD_NS)
        except Exception as e:
            log.warning(f"{date}: engine load error: {e}")
            continue

        for ot in ORDER_TYPES:
            try:
                trades = eng.simulate(signals=signals, tp_ticks=TP_TICKS,
                                      sl_ticks=SL_TICKS, order_type=ot)
            except Exception as e:
                log.warning(f"{date} {ot}: simulate error: {e}")
                continue
            tnet = np.array([trade_net_ticks(t) for t in trades])
            tts = np.array([trade_sig_ts(t) for t in trades], dtype=np.int64)
            for arm in ARMS:
                member = np.array([int(t) in arm_ts[arm] for t in tts], dtype=bool)
                r = metrics_row(arm, date, ot, arm_nsig[arm], tnet[member])
                rows.append(r)
                if r["n_trades"]:
                    log.info(f"  {date} {ot} {arm}: n={r['n_trades']} "
                             f"ntpt={r['net_ticks_per_trade']:+.3f} "
                             f"WR={r['win_rate']:.3f} PF={r['pf']:.2f}")
                else:
                    log.info(f"  {date} {ot} {arm}: 0 trades")

        del eng
        log.info(f"{date} done in {time.time()-t0:.0f}s")

    csv_path = OUT_DIR / "results_per_day_shortonly.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info(f"Wrote {csv_path}")

    summary = {"geometry": {"tp": TP_TICKS, "sl": SL_TICKS,
                            "hold_s": HOLD_NS / 1e9, "cancel_s": CANCEL_NS / 1e9,
                            "bands": BANDS, "side": "short_only",
                            "headc_staleness_s": HEADC_STALENESS_NS / 1e9},
               "arms": {}}
    for arm in ARMS:
        summary["arms"][arm] = {}
        for ot in ORDER_TYPES:
            rs = [r for r in rows if r["arm"] == arm and r["order_type"] == ot
                  and r.get("n_trades", 0) > 0]
            if not rs:
                summary["arms"][arm][ot] = {"status": "no_trades"}
                continue
            ntpts = np.array([r["net_ticks_per_trade"] for r in rs])
            nts = np.array([r["n_trades"] for r in rs])
            sums = np.array([r["sum_net_ticks"] for r in rs])
            wrs = np.array([r["win_rate"] for r in rs])
            summary["arms"][arm][ot] = {
                "n_days_with_trades": len(rs),
                "total_trades": int(nts.sum()),
                "trades_per_day_mean": float(nts.mean()),
                "trade_weighted_ntpt": float(sums.sum() / nts.sum()),
                "unweighted_mean_ntpt": float(ntpts.mean()),
                "median_day_ntpt": float(np.median(ntpts)),
                "total_net_ticks": float(sums.sum()),
                "n_days_positive": int((ntpts > 0).sum()),
                "zero_positive_days": bool((ntpts > 0).sum() == 0),
                "trade_weighted_wr": float((wrs * nts).sum() / nts.sum()),
                "day_sharpe_ntpt": (float(ntpts.mean() / ntpts.std(ddof=1))
                                    if len(ntpts) > 1 and ntpts.std(ddof=1) > 0 else None),
                "day_sharpe_pnl": (float(sums.mean() / sums.std(ddof=1))
                                   if len(sums) > 1 and sums.std(ddof=1) > 0 else None),
            }

    # Honest verdict construction
    verdict = {}
    for b in BANDS:
        for ot in ORDER_TYPES:
            base = summary["arms"][f"{b}_baseline"].get(ot, {})
            key = f"{b}_{ot}"
            base_ntpt = base.get("trade_weighted_ntpt")
            if base_ntpt is None:
                verdict[key] = {"result": "NO_TRADES"}
                continue
            if base_ntpt <= 0:
                verdict[key] = {"result": "BASELINE_NEGATIVE",
                                "baseline_tw_ntpt": base_ntpt,
                                "note": "Short-only champion baseline loses net of "
                                        "costs on these 8 days; confluence question moot."}
                continue
            best = None
            for tau in TAUS:
                s = summary["arms"][f"{b}_conf_tau{int(tau*100)}"].get(ot, {})
                if s.get("total_trades", 0) < 20:
                    continue
                if (s.get("trade_weighted_ntpt", -9e9) > base_ntpt
                        and (s.get("day_sharpe_pnl") or -9e9) > (base.get("day_sharpe_pnl") or -9e9)):
                    if best is None or s["trade_weighted_ntpt"] > best[1]["trade_weighted_ntpt"]:
                        best = (f"{b}_conf_tau{int(tau*100)}", s)
            verdict[key] = ({"result": "CONFLUENCE_IMPROVES", "best_arm": best[0],
                             "tw_ntpt": best[1]["trade_weighted_ntpt"],
                             "baseline_tw_ntpt": base_ntpt} if best else
                            {"result": "CONFLUENCE_NO_IMPROVEMENT",
                             "baseline_tw_ntpt": base_ntpt})
    summary["verdict"] = verdict

    sp = OUT_DIR / "summary_shortonly.json"
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Wrote {sp}")
    log.info(json.dumps(verdict, indent=2))

    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{LVL3_ROOT}/mlflow.db")
        mlflow.set_experiment("p_alpha_headc_firstpassage_v1")
        with mlflow.start_run(run_name="headc_k2_confluence_fifo_shortonly"):
            mlflow.log_params({"tp": TP_TICKS, "sl": SL_TICKS,
                               "hold_s": HOLD_NS / 1e9, "cancel_s": CANCEL_NS / 1e9,
                               "bands": str(BANDS), "taus": str(TAUS),
                               "side": "short_only"})
            for arm in ARMS:
                for ot in ORDER_TYPES:
                    s = summary["arms"][arm].get(ot, {})
                    for k in ("trade_weighted_ntpt", "day_sharpe_pnl",
                              "total_trades", "trade_weighted_wr"):
                        v = s.get(k)
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            mlflow.log_metric(f"{arm}_{ot}_{k}", float(v))
            mlflow.log_artifact(str(csv_path))
            mlflow.log_artifact(str(sp))
        log.info("Logged to MLflow (local sqlite)")
    except Exception as e:
        log.warning(f"MLflow logging failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
