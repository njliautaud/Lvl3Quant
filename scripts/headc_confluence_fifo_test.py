#!/usr/bin/env python3
"""
headc_confluence_fifo_test.py — Does gating champion v3.4.2 trades on head-C K=2
first-passage probability agreement improve risk-adjusted execution?

BASELINE: champion v3.4.2 pred_log_ret_1s, per-day symmetric top-0.5% band per side
          (HC #441 conf_band convention, extended to both sides), FIFO geometry
          TP=3.00 SL=0.50 hold=1.5s cancel=10s (HC #441 PRIMARY geometry).
CONFLUENCE: same entries, additionally require head-C K=2 P(up-first within 5s)
          agreement: long needs prob >= tau, short needs prob <= 1-tau,
          tau in {0.55, 0.60, 0.65, 0.70}. Head-C prob = latest sample at or
          before signal ts, staleness <= 5s (the head-C horizon).

Engine: canonical FIFOReplayEngine (FIFO market replay ONLY, HC #74).
Trades expose pnl_ticks_net (after 0.376 ticks RT commission; market orders
cross the spread inside the replay). Confluence arms are SUBSETS of baseline
signals, so we simulate the baseline signal set once per order type and derive
each arm by filtering trades on signal_ts_ns (orders are independent in the
engine — no cross-signal netting).

Costs reported: limit (passive, commission only) and market (commission+cross).
Aggregates are TRADE-WEIGHTED (total net / total trades) in addition to per-day
means, per HC guidance on the K=2 unweighted-mean artifact.

One day at a time to keep RAM modest (Neptune 32GB). CPU only.
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
HEADC_DIR = LVL3_ROOT / "output" / "p_alpha_headc_v1"  # K=2 OOT preds
MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
OUT_DIR = LVL3_ROOT / "output" / "headc_confluence_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LVL3_ROOT / "logs" / f"headc_confluence_fifo_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("headc_confluence")

# ── Champion gate conventions (HC #441 PRIMARY geometry) ────────────────────
CONF_BAND = 0.005            # top 0.5% per side
TP_TICKS = 3.00
SL_TICKS = 0.50
HOLD_NS = int(1.5 * 1e9)     # 1.5s
CANCEL_NS = int(10.0 * 1e9)  # 10s
ORDER_TYPES = ["limit", "market"]

# Champion dataset geometry (SmartV32Dataset: window_t1=1500, stride=250,
# samples kept where labels_1s not NaN) — verified to reproduce NPZ length.
CHAMP_WINDOW = 1500
CHAMP_STRIDE = 250

# Head-C confluence
TAUS = [0.55, 0.60, 0.65, 0.70]
HEADC_STALENESS_NS = int(5.0 * 1e9)  # head-C horizon = 5s

DATES = ["20260306", "20260309", "20260310", "20260311",
         "20260312", "20260313", "20260316", "20260317"]
HEADC_FOLD = {d: i for i, d in enumerate(DATES)}

ARMS = ["baseline"] + [f"conf_tau{int(t*100)}" for t in TAUS]

# Load canonical FIFO engine
fifo_path = LVL3_ROOT / "alpha_discovery" / "deep_models" / "fifo_market_replay.py"
spec = importlib.util.spec_from_file_location("fifo_market_replay", str(fifo_path))
fmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmod)
FIFOReplayEngine = fmod.FIFOReplayEngine


def trade_net_ticks(t) -> float:
    # TradeResult dataclass uses pnl_ticks_net (after commission) — NOT net_ticks
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
    """Reconstruct per-sample ts_ns for champion per-date NPZ (verified exact)."""
    with np.load(MBO_DIR / f"{date}_mbo_events.npz", allow_pickle=True) as z:
        ts = z["timestamps"].astype(np.int64)
        lab1 = z["labels_1s"]
    n = len(ts)
    pos = np.arange(CHAMP_WINDOW - 1, n, CHAMP_STRIDE)
    keep = ~np.isnan(lab1[pos])
    pos = pos[keep]
    if len(pos) != len(preds):
        raise RuntimeError(
            f"{date}: champion alignment mismatch {len(pos)} vs {len(preds)}")
    return ts[pos]


def headc_prob_at(sig_ts: np.ndarray, hc_ts: np.ndarray, hc_p: np.ndarray):
    """Latest head-C prob at or before each signal ts; NaN if stale (>5s) or none."""
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

        # Baseline: per-day symmetric top CONF_BAND per side
        lo = np.quantile(cpred, CONF_BAND)
        hi = np.quantile(cpred, 1.0 - CONF_BAND)
        long_m = cpred >= hi
        short_m = cpred <= lo
        sig_idx = np.where(long_m | short_m)[0]
        sig_ts = cts[sig_idx]
        sig_dir = np.where(long_m[sig_idx], 1, -1)
        sig_str = np.abs(cpred[sig_idx])

        # Head-C prob per baseline signal
        prob = headc_prob_at(sig_ts, hc_ts, hc_p)
        cov = float(np.isfinite(prob).mean())
        log.info(f"{date}: n_preds={len(cpred)} baseline_signals={len(sig_idx)} "
                 f"(long={int(long_m.sum())} short={int(short_m.sum())}) "
                 f"headc_coverage={cov:.3f}")

        # Arm membership (sets of signal ts)
        arm_ts = {"baseline": set(int(t) for t in sig_ts)}
        arm_nsig = {"baseline": len(sig_ts)}
        for tau in TAUS:
            agree = np.where(sig_dir > 0, prob >= tau, prob <= 1.0 - tau)
            agree &= np.isfinite(prob)
            name = f"conf_tau{int(tau*100)}"
            arm_ts[name] = set(int(t) for t in sig_ts[agree])
            arm_nsig[name] = int(agree.sum())
            log.info(f"  {name}: {arm_nsig[name]} signals "
                     f"({arm_nsig[name]/max(len(sig_ts),1)*100:.1f}% of baseline)")

        signals = [{"ts_ns": int(t),
                    "direction": "long" if d > 0 else "short",
                    "strength": float(s)}
                   for t, d, s in zip(sig_ts, sig_dir, sig_str)]

        try:
            eng = FIFOReplayEngine(date=date, cancel_after_ns=CANCEL_NS,
                                   max_hold_ns=HOLD_NS)
        except Exception as e:
            log.warning(f"{date}: engine load error: {e}")
            for arm in ARMS:
                for ot in ORDER_TYPES:
                    rows.append({"arm": arm, "date": date, "order_type": ot,
                                 "n_signals": arm_nsig[arm], "n_trades": 0,
                                 "error": str(e)[:80]})
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
                log.info(f"  {date} {ot} {arm}: n_trades={r['n_trades']} "
                         f"ntpt={r['net_ticks_per_trade']:+.3f} "
                         f"WR={r['win_rate']:.3f} PF={r['pf']:.2f}"
                         if r["n_trades"] else f"  {date} {ot} {arm}: 0 trades")

        del eng
        log.info(f"{date} done in {time.time()-t0:.0f}s")

    # Write per-day CSV
    csv_path = OUT_DIR / "results_per_day.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info(f"Wrote {csv_path}")

    # Aggregate: TRADE-WEIGHTED + per-day
    summary = {"geometry": {"tp": TP_TICKS, "sl": SL_TICKS,
                            "hold_s": HOLD_NS / 1e9, "cancel_s": CANCEL_NS / 1e9,
                            "conf_band": CONF_BAND,
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
            tw_ntpt = float(sums.sum() / nts.sum())
            day_sh_ntpt = (float(ntpts.mean() / ntpts.std(ddof=1))
                           if len(ntpts) > 1 and ntpts.std(ddof=1) > 0 else None)
            day_sh_pnl = (float(sums.mean() / sums.std(ddof=1))
                          if len(sums) > 1 and sums.std(ddof=1) > 0 else None)
            # pooled WR/PF (trade-weighted)
            wrs = np.array([r["win_rate"] for r in rs])
            tw_wr = float((wrs * nts).sum() / nts.sum())
            summary["arms"][arm][ot] = {
                "n_days_with_trades": len(rs),
                "total_trades": int(nts.sum()),
                "trades_per_day_mean": float(nts.mean()),
                "trade_weighted_ntpt": tw_ntpt,
                "unweighted_mean_ntpt": float(ntpts.mean()),
                "median_day_ntpt": float(np.median(ntpts)),
                "total_net_ticks": float(sums.sum()),
                "n_days_positive": int((ntpts > 0).sum()),
                "trade_weighted_wr": tw_wr,
                "day_sharpe_ntpt": day_sh_ntpt,
                "day_sharpe_pnl": day_sh_pnl,
            }

    # Verdict: confluence must beat baseline on trade-weighted ntpt AND day
    # Sharpe without collapsing trades below ~20 over 8 days.
    verdict = {}
    for ot in ORDER_TYPES:
        base = summary["arms"]["baseline"].get(ot, {})
        best = None
        for tau in TAUS:
            arm = f"conf_tau{int(tau*100)}"
            s = summary["arms"][arm].get(ot, {})
            if s.get("total_trades", 0) < 20:
                continue
            if (s.get("trade_weighted_ntpt", -9e9) > base.get("trade_weighted_ntpt", -9e9)
                    and (s.get("day_sharpe_pnl") or -9e9) > (base.get("day_sharpe_pnl") or -9e9)):
                if best is None or s["trade_weighted_ntpt"] > best[1]["trade_weighted_ntpt"]:
                    best = (arm, s)
        verdict[ot] = ({"result": "PASS", "best_arm": best[0],
                        "tw_ntpt": best[1]["trade_weighted_ntpt"],
                        "baseline_tw_ntpt": base.get("trade_weighted_ntpt")}
                       if best else
                       {"result": "FAIL",
                        "baseline_tw_ntpt": base.get("trade_weighted_ntpt")})
    summary["verdict"] = verdict

    sp = OUT_DIR / "summary.json"
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Wrote {sp}")
    log.info(json.dumps(summary["verdict"], indent=2))

    # MLflow — Neptune-local sqlite store (same as headc experiments)
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{LVL3_ROOT}/mlflow.db")
        mlflow.set_experiment("p_alpha_headc_firstpassage_v1")
        with mlflow.start_run(run_name="headc_k2_confluence_fifo_test"):
            mlflow.log_params({"tp": TP_TICKS, "sl": SL_TICKS,
                               "hold_s": HOLD_NS / 1e9, "cancel_s": CANCEL_NS / 1e9,
                               "conf_band": CONF_BAND, "taus": str(TAUS)})
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
