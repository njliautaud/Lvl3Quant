"""
dispatch_v34_3_pyramid_fixed.py
================================
LAUNCH-READY dispatch for v3.4.3-pyramid (spec-compliant successor to v3.4.2).

Activates the SPEC §5 book pyramid (20 levels × 6 features = 120 channels) trainer
on Neptune via a clean wrapper. Does NOT auto-launch — gated on HC #381 release.

Key differences from dispatch_v34_2_fixedmtl.py:
  - Trainer: train_cnn_mamba_v3_4_pyramid.py (NEW)
  - Data: data/derived/tier2_book_shape_pyramid_v1.parquet/*.parquet (NEW)
  - Norm stats: output/v3_4_pyramid/pyramid_norm_stats.json (bootstrap before launch)
  - Loss: FIXED MTL weights (inherited from v3.4.2 — Kendall σ collapse avoided)
  - Warmstart: v3.3 fold_00_intra_ckpt (book trunk + last d_model_book cols stay random)
  - Falsification gate (HC #366 amended): IC_1s ≥ 0.296 on 5d OOT or ≥ 0.23 on 17-day extended OOT.

PRE-LAUNCH CHECKLIST (must verify ALL before running):
  [ ] HC #381 cleared — user has released Neptune for training
  [ ] Pyramid data exists for ≥ 60 training dates AND 5 OOT dates
        ls data/derived/tier2_book_shape_pyramid_v1.parquet/*.parquet | wc -l
  [ ] Norm stats sidecar exists
        ls output/v3_4_pyramid/pyramid_norm_stats.json
  [ ] v3.3 intra-ckpt available
        ls output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt
  [ ] HC #376 head-target audit clean (all 32 heads must have nonzero target rate > 1%)
  [ ] MLflow up at http://localhost:5000
  [ ] Neptune GPU idle, ≥ 18 GB VRAM free
  [ ] No other Neptune training process

LAUNCH COMMAND (after checklist):
  cd /home/nick/Lvl3Quant
  python3 -u alpha_discovery/deep_models/train_cnn_mamba_v3_4_pyramid_dispatcher.py \\
      --fold 0 --epochs 2 --warmstart-v33

EXPECTED OUTPUTS:
  - output/cnn_mamba_v3_4_pyramid/fold_00_intra_ckpt.pt   (mid-Ep1 checkpoint)
  - output/cnn_mamba_v3_4_pyramid/fold_00_oot_predictions.npz  (Ep1 OOT)
  - MLflow run under experiment "CNNMamba_v3_4_3_pyramid"
  - Per-head IC + DA + MagCorr table (HC #382 mandatory eval before kill verdict)
"""
import sys, json, hashlib, time
from pathlib import Path

PROJECT_ROOT = Path('/home/jupiter/Lvl3Quant')
PYRAMID_DIR = PROJECT_ROOT / 'data' / 'derived' / 'tier2_book_shape_pyramid_v1.parquet'
NORM_STATS = PROJECT_ROOT / 'output' / 'v3_4_pyramid' / 'pyramid_norm_stats.json'
V33_CKPT = PROJECT_ROOT / 'output' / 'cnn_mamba_v3_3_uncertainty_weighted' / 'fold_00_intra_ckpt.pt'


def sha256_file(p: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(max_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_readiness() -> dict:
    report = {}
    # Pyramid data count
    parquets = sorted(PYRAMID_DIR.glob('*.parquet')) if PYRAMID_DIR.exists() else []
    report['pyramid_dates'] = len(parquets)
    report['pyramid_dir_exists'] = PYRAMID_DIR.exists()
    report['pyramid_first_date'] = parquets[0].stem if parquets else None
    report['pyramid_last_date'] = parquets[-1].stem if parquets else None
    # Norm stats
    report['norm_stats_exists'] = NORM_STATS.exists()
    if NORM_STATS.exists():
        with open(NORM_STATS) as f:
            stats = json.load(f)
        report['norm_stats_n_total'] = stats.get('n_total')
        report['norm_stats_mean'] = stats.get('mean')
    # v3.3 ckpt
    report['v33_ckpt_exists'] = V33_CKPT.exists()
    if V33_CKPT.exists():
        report['v33_ckpt_size_mb'] = V33_CKPT.stat().st_size / 1e6
        report['v33_ckpt_sha256'] = sha256_file(V33_CKPT)[:16]
    return report


def print_checklist():
    print("=" * 70)
    print("v3.4.3-pyramid DISPATCH READINESS REPORT")
    print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    r = check_readiness()
    for k, v in r.items():
        print(f"  {k}: {v}")
    print()
    ready = (
        r['pyramid_dates'] >= 65
        and r['norm_stats_exists']
        and r['v33_ckpt_exists']
    )
    print(f"READY TO LAUNCH: {ready}")
    if not ready:
        print()
        print("Blockers:")
        if r['pyramid_dates'] < 65:
            print(f"  - need ≥65 pyramid dates, have {r['pyramid_dates']}")
        if not r['norm_stats_exists']:
            print("  - missing norm stats sidecar; run bootstrap_pyramid_norm_stats")
        if not r['v33_ckpt_exists']:
            print("  - v3.3 fold_00_intra_ckpt.pt not found")
    print()
    print("HC #381 (Neptune gaming): NOT auto-checked. Verify with user before launch.")
    print("=" * 70)


if __name__ == "__main__":
    print_checklist()
