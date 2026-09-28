#!/usr/bin/env python3
"""
queue_position_v0_train.py — HC #471 R5 / HC #469 R5(a) — train queue-position model.

Trains a 1D CNN on the pre-signal book-feature window to predict the realized
queue_ahead at fill time.

Target: log1p(queue_ahead). Loss: MSE. Eval: per-day MAE, P90 error, R².

Time-based train/test split (first 70% of dates = train, last 30% = test).

Designed to run on Neptune RTX 3090 (CUDA). Falls back to CPU if no GPU.

Input:  output/queue_position_v0/dataset.npz
Output: output/queue_position_v0/REPORT.md
        output/queue_position_v0/weights.pt
        output/queue_position_v0/preds_test.npz
        output/queue_position_v0/queue_position_v0.DONE
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import os
import numpy as np

LVL3 = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
DATA = LVL3 / "output/queue_position_v0/dataset.npz"
OUT_DIR = LVL3 / "output/queue_position_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("[FATAL] pytorch not available")
    sys.exit(2)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[train] device={DEVICE}")
if DEVICE == "cuda":
    print(f"[train] cuda={torch.cuda.get_device_name(0)}")


class QPCNN(nn.Module):
    def __init__(self, n_feat=30, hidden=64):
        super().__init__()
        self.norm = nn.BatchNorm1d(n_feat)
        self.c1 = nn.Conv1d(n_feat, hidden, kernel_size=5, padding=2)
        self.c2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.c3 = nn.Conv1d(hidden, hidden // 2, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        # x: [B, K, F] -> [B, F, K]
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.act(self.c1(x)); x = self.drop(x)
        x = self.act(self.c2(x)); x = self.drop(x)
        x = self.act(self.c3(x))
        x = self.gap(x).squeeze(-1)  # [B, hidden//2]
        return self.head(x).squeeze(-1)  # [B]


def main():
    t0 = time.time()
    if not DATA.exists():
        print(f"[FATAL] {DATA} not found")
        sys.exit(3)

    ar = np.load(DATA, allow_pickle=False)
    X = ar["X"].astype(np.float32)
    y_log = ar["y_log"].astype(np.float32)
    y_raw = ar["y"].astype(np.float32)
    dates = ar["dates"]
    print(f"[train] X={X.shape} y_log range=[{y_log.min():.2f},{y_log.max():.2f}]")

    # Per-feature standardization using training-fold stats only
    unique_dates = sorted(set(dates.tolist()))
    split_n = max(1, int(len(unique_dates) * 0.70))
    train_dates = set(unique_dates[:split_n])
    test_dates = set(unique_dates[split_n:])
    print(f"[train] train_dates={len(train_dates)} test_dates={len(test_dates)}")
    print(f"[train]   train={sorted(train_dates)}")
    print(f"[train]   test ={sorted(test_dates)}")

    train_mask = np.array([d in train_dates for d in dates])
    test_mask = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y_log[train_mask], y_log[test_mask]
    y_raw_train, y_raw_test = y_raw[train_mask], y_raw[test_mask]
    dates_test = dates[test_mask]

    print(f"[train] n_train={len(X_train)} n_test={len(X_test)}")

    # Robust per-feature stats from train (across all samples × ticks for that feature)
    feat_mean = X_train.mean(axis=(0, 1), keepdims=True)
    feat_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train_n = (X_train - feat_mean) / feat_std
    X_test_n = (X_test - feat_mean) / feat_std

    # Target standardization (helps loss)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) + 1e-6
    y_train_n = (y_train - y_mean) / y_std

    # Tensors
    Xt_tr = torch.from_numpy(X_train_n).to(DEVICE)
    yt_tr = torch.from_numpy(y_train_n).to(DEVICE)
    Xt_te = torch.from_numpy(X_test_n).to(DEVICE)

    ds = TensorDataset(Xt_tr, yt_tr)
    bs = 256
    dl = DataLoader(ds, batch_size=bs, shuffle=True)

    model = QPCNN(n_feat=X.shape[2], hidden=64).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossfn = nn.MSELoss()

    EPOCHS = 25
    print(f"[train] starting {EPOCHS} epochs, bs={bs}")
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0; n = 0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = lossfn(pred, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(yb); n += len(yb)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[train] ep{ep+1:02d} train_loss={tot/n:.4f}")

    model.eval()
    with torch.no_grad():
        # batch eval to fit in VRAM
        preds_n = []
        for i in range(0, len(Xt_te), 1024):
            preds_n.append(model(Xt_te[i:i+1024]).cpu().numpy())
        preds_n = np.concatenate(preds_n)

    # Inverse-standardize back to log-space
    preds_log = preds_n * y_std + y_mean
    preds_raw = np.expm1(preds_log)

    # Metrics
    mae_log = float(np.mean(np.abs(preds_log - y_test)))
    mae_raw = float(np.mean(np.abs(preds_raw - y_raw_test)))
    p90_err_raw = float(np.percentile(np.abs(preds_raw - y_raw_test), 90))
    ss_res = float(np.sum((preds_log - y_test) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-6)
    bias_raw = float(np.mean(preds_raw - y_raw_test))

    # Baseline: predict the mean (in log space)
    baseline_pred_log = np.full_like(y_test, y_train.mean())
    baseline_mae_log = float(np.mean(np.abs(baseline_pred_log - y_test)))

    # Per-day breakdown
    perday = {}
    for d in sorted(test_dates):
        m = dates_test == d
        if m.sum() < 5:
            continue
        perday[d] = {
            "n": int(m.sum()),
            "mae_log": float(np.mean(np.abs(preds_log[m] - y_test[m]))),
            "mae_raw": float(np.mean(np.abs(preds_raw[m] - y_raw_test[m]))),
            "y_mean_raw": float(y_raw_test[m].mean()),
            "pred_mean_raw": float(preds_raw[m].mean()),
        }

    metrics = {
        "device": DEVICE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae_log": mae_log,
        "baseline_mae_log": baseline_mae_log,
        "lift_vs_mean": float((baseline_mae_log - mae_log) / baseline_mae_log * 100),
        "mae_raw": mae_raw,
        "p90_err_raw": p90_err_raw,
        "r2_log": r2,
        "bias_raw": bias_raw,
        "wall_s": time.time() - t0,
        "epochs": EPOCHS,
    }
    print(f"[eval] {json.dumps(metrics, indent=2)}")

    # Save
    np.savez_compressed(
        OUT_DIR / "preds_test.npz",
        preds_log=preds_log, preds_raw=preds_raw,
        y_log=y_test, y_raw=y_raw_test, dates=dates_test,
    )
    torch.save(model.state_dict(), str(OUT_DIR / "weights.pt"))

    lines = [
        "# Queue-Position Model v0 — HC #471 R5 / HC #469 R5(a)",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M ET')}. Wall: {metrics['wall_s']:.1f}s. Device: {DEVICE}.",
        "",
        "**Compliance**: HC #471 R5 (queue-position model prototype), HC #469 R5(a) (queue-aware fill gating).",
        "",
        "## Task",
        "Predict `queue_ahead` (realized FIFO queue position at fill time) from a pre-signal",
        f"window of K={X.shape[1]} book events × {X.shape[2]} features (5-level book + flow metrics).",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| n_test | {metrics['n_test']} |",
        f"| MAE (log queue) | {mae_log:.4f} |",
        f"| Baseline MAE (predict-mean) | {baseline_mae_log:.4f} |",
        f"| **Lift vs mean baseline** | **{metrics['lift_vs_mean']:+.1f}%** |",
        f"| MAE (raw queue count) | {mae_raw:.2f} |",
        f"| P90 abs error (raw) | {p90_err_raw:.2f} |",
        f"| R² (log space) | {r2:.4f} |",
        f"| Mean bias (raw) | {bias_raw:+.2f} |",
        "",
        "## Per-day breakdown (test)",
        "",
        "| Date | n | MAE_log | MAE_raw | y_mean | pred_mean |",
        "|---|---|---|---|---|---|",
    ]
    for d, m in sorted(perday.items()):
        lines.append(f"| {d} | {m['n']} | {m['mae_log']:.3f} | {m['mae_raw']:.2f} | {m['y_mean_raw']:.1f} | {m['pred_mean_raw']:.1f} |")
    lines.append("")
    lines.append("## Next steps")
    lines.append("- Wire as a confluence head: trade only when predicted_queue_ahead × P(joiner_drain) < threshold.")
    lines.append("- Extend window to K=500 or K=1000 events; compare lift.")
    lines.append("- Add joiner/leaver volume features (HC #469 R5(b)) and re-train.")
    lines.append("- Move to per-event sequence model (Transformer / Mamba) for sharper temporal pickup.")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT_DIR / "queue_position_v0.DONE").write_text(
        f"completed {time.strftime('%Y-%m-%d %H:%M:%S ET')} wall_s={metrics['wall_s']:.1f}\n"
    )
    print(f"[DONE] {metrics['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
