#!/usr/bin/env python3
"""
iceberg_highcv_fillsim.py - Test iceberg signal with higher CV thresholds.

Hypothesis: At 3x CV threshold, avg queue position = 33. At 5x-8x, fewer
participants are queued because the signal fires less frequently.
Test: cv_thresh = [3, 4, 5, 6, 8] x entry = [passive, prime, prime_chase, tp4_sl8, hold30s]

If higher threshold gives queue position < 10 AND Sortino > 0, this is viable.
Results: /home/jupiter/Lvl3Quant/data/processed/iceberg_highcv_fillsim/
"""
import glob, json, os, math, subprocess, re, time
from pathlib import Path
from collections import defaultdict
import numpy as np

BOOK_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot")
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/iceberg_highcv_fillsim")
PRED_DIR = OUTPUT_DIR / "preds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 100
CV_THRESHOLDS = [3.0, 4.0, 5.0, 6.0, 8.0]
CONFIGS = [
    {"name": "passive",      "hold_ms": 10000},
    {"name": "prime",        "hold_ms": 10000, "prime_hours": True},
    {"name": "prime_chase",  "hold_ms": 10000, "prime_hours": True, "chase": True},
    {"name": "tp4_sl8",      "hold_ms": 10000, "tp": 4, "sl": 8},
    {"name": "tp4_chase",    "hold_ms": 10000, "tp": 4, "sl": 8, "chase": True},
    {"name": "hold30s_tp4",  "hold_ms": 30000, "tp": 4, "sl": 8},
]


def rolling_mean(a, w):
    cs = np.cumsum(a.astype(np.float64))
    o = np.empty(len(a), np.float64)
    o[:w] = cs[:w] / np.arange(1, w + 1)
    o[w:] = (cs[w:] - cs[:-w]) / w
    return o.astype(np.float32)


def compute_iceberg_signal(book, cv_thresh=3.0, window=WINDOW):
    # Use volume count channels: col 3 per level
    if book.ndim == 3:
        vol_counts = book[:, :, 3] if book.shape[2] > 3 else book[:, :, -1]
    else:
        # fallback: treat as flat
        vol_counts = book
    bid_vol = vol_counts[:, :10].sum(axis=1).astype(np.float32)
    ask_vol = vol_counts[:, 10:].sum(axis=1).astype(np.float32)
    bid_rm = rolling_mean(bid_vol, window)
    ask_rm = rolling_mean(ask_vol, window)
    bid_ratio = bid_vol / (bid_rm + 1e-6)
    ask_ratio = ask_vol / (ask_rm + 1e-6)
    sig = np.zeros(len(bid_vol), dtype=np.float32)
    sig[bid_ratio >= cv_thresh] = 1.0
    sig[ask_ratio >= cv_thresh] = -1.0
    # Don't double-signal
    both = (bid_ratio >= cv_thresh) & (ask_ratio >= cv_thresh)
    sig[both] = 0.0
    return sig


def load_and_signal(book_file, cv_thresh):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', book_file)
    if not m:
        return None, None
    date = m.group(1)
    try:
        z = np.load(book_file, allow_pickle=False)
        keys = list(z.files)
        book = z[keys[0]] if len(keys) == 1 else z.get('book', z[keys[0]])
        if book.ndim == 2:
            n_levels = 20
            n_feats = book.shape[1] // n_levels
            book = book.reshape(book.shape[0], n_levels, n_feats)
    except Exception as e:
        print(f"  ERROR loading {book_file}: {e}")
        return None, date
    sig = compute_iceberg_signal(book, cv_thresh)
    n_long = int(np.sum(sig > 0))
    n_short = int(np.sum(sig < 0))
    total = n_long + n_short
    if total < 2:
        return None, date
    freq = 100.0 * total / len(sig)
    print(f"  cv={cv_thresh:.1f} {date}: +{n_long}/-{n_short} ({freq:.4f}%)")
    return sig, date


def save_pred(sig, date, cv_thresh):
    cv_str = f"{cv_thresh:.1f}".replace(".", "p")
    name = f"{date}_cv{cv_str}_preds.npz"
    pred_file = PRED_DIR / name
    np.savez_compressed(str(pred_file), predictions=sig)
    return pred_file


def run_fillsim(date, pred_file, cfg, cv_thresh):
    date_nodash = date.replace("-", "")
    mbo_file = MBO_DIR / f"glbx-mdp3-{date_nodash}.mbo.dbn.zst"
    cv_str = f"{cv_thresh:.1f}".replace(".", "p")
    out_file = OUTPUT_DIR / f"{date}_cv{cv_str}_{cfg['name']}.json"
    if not mbo_file.exists():
        return None
    if out_file.exists():
        try:
            d = json.load(open(out_file))
            d["date"] = date
            d["config"] = cfg["name"]
            d["cv_thresh"] = cv_thresh
            return d
        except Exception:
            pass
    cmd = [
        FILL_SIM,
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--signal-threshold", "0.5",
        "--latency-ms", "10",
    ]
    if cfg.get("hold_ms"):
        cmd += ["--hold-ms", str(cfg["hold_ms"])]
    if cfg.get("tp"):
        cmd += ["--take-profit-ticks", str(cfg["tp"])]
    if cfg.get("sl"):
        cmd += ["--stop-loss-ticks", str(cfg["sl"])]
    if cfg.get("prime_hours"):
        cmd += ["--prime-hours"]
    if cfg.get("chase"):
        cmd += ["--chase-entry"]
    if cfg.get("market_entry"):
        cmd += ["--market-entry"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return None
        d = json.load(open(out_file))
        d["date"] = date
        d["config"] = cfg["name"]
        d["cv_thresh"] = cv_thresh
        return d
    except Exception:
        return None


book_files = sorted(glob.glob(str(BOOK_DIR / "*_book_tensors.npz")))
print(f"Found {len(book_files)} OOT book files")
print(f"Testing CV thresholds: {CV_THRESHOLDS}")
n_total = len(CV_THRESHOLDS) * len(book_files) * len(CONFIGS)
print(f"Total jobs: {len(CV_THRESHOLDS)} x {len(book_files)} days x {len(CONFIGS)} configs = {n_total}")
t0 = time.time()

all_results = []
total_done = 0

for cv_thresh in CV_THRESHOLDS:
    print(f"\n=== CV threshold = {cv_thresh:.1f}x ===")
    for book_file in book_files:
        sig, date = load_and_signal(book_file, cv_thresh)
        if sig is None or date is None:
            continue
        pred_file = save_pred(sig, date, cv_thresh)
        for cfg_entry in CONFIGS:
            result = run_fillsim(date, pred_file, cfg_entry, cv_thresh)
            if result:
                all_results.append(result)
                total_done += 1
    elapsed = time.time() - t0
    print(f"  CV={cv_thresh:.1f} done. Total results so far: {total_done}. Elapsed: {elapsed:.0f}s")

# Aggregate
by_key = defaultdict(list)
for r in all_results:
    k = (r["cv_thresh"], r["config"])
    by_key[k].append(r)

agg = []
for (cv, cfg_name), days in sorted(by_key.items()):
    nd = len(days)
    nt = sum(d.get("total_trades", 0) for d in days)
    pnl = sum(d.get("total_pnl_dollars", 0) for d in days)
    dpnls = [d.get("total_pnl_dollars", 0) for d in days]
    sigs = sum(d.get("total_signals", 0) for d in days)
    filled = sum(d.get("total_filled", 0) for d in days)
    wins = sum(d.get("total_trades", 0) * d.get("win_rate", 0) for d in days)
    q_sum = sum(d.get("avg_queue_position", 0) * max(d.get("total_filled", 1), 1) for d in days)
    lat_sum = sum(d.get("avg_fill_latency_ms", 0) * max(d.get("total_filled", 1), 1) for d in days)
    neg = [p for p in dpnls if p < 0]
    ds = math.sqrt(sum(p ** 2 for p in neg) / len(neg)) if neg else 1e-9
    mu = pnl / nd
    sortino = mu / ds
    fr = filled / max(sigs, 1)
    wr = wins / max(nt, 1)
    aq = q_sum / max(filled, 1)
    lat = lat_sum / max(filled, 1)
    agg.append({"sortino": sortino, "cv": cv, "config": cfg_name, "n_days": nd, "n_trades": nt,
                "total_pnl": pnl, "pnl_per_day": mu, "win_rate": wr, "fill_rate": fr,
                "avg_queue": aq, "avg_lat_ms": lat})

agg.sort(key=lambda x: -x["sortino"])

print(f"\n{'CV':>5} {'Config':<22} {'Days':>5} {'Trades':>7} {'$/day':>9} {'WR':>6} {'FillR':>6} {'AvgQ':>6} {'Sortino':>9}")
print('-' * 85)
for r in agg:
    print(f"{r['cv']:>5.1f} {r['config']:<22} {r['n_days']:>5} {r['n_trades']:>7} {r['pnl_per_day']:>9,.0f} {r['win_rate']:>5.1%} {r['fill_rate']:>5.1%} {r['avg_queue']:>6.1f} {r['sortino']:>9.3f}")

summary = {"strategy": "iceberg_highcv_fillsim", "n_results": total_done, "agg": agg}
out_path = OUTPUT_DIR / "iceberg_highcv_summary.json"
with open(str(out_path), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nDone in {time.time()-t0:.0f}s. Results: {out_path}")
if agg:
    b = agg[0]
    print(f"Best: cv={b['cv']} {b['config']} -> Sortino={b['sortino']:.3f} WR={b['win_rate']:.1%} AvgQ={b['avg_queue']:.1f} n={b['n_trades']}")
