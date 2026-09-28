#!/usr/bin/env python3
"""
Tuesday Pre-Market Automated Checker
Validates all deploy assets and system health before paper trading.
Run from Jupiter: python3 live_trading/premarket_check.py
"""

import subprocess
import json
import sys
import os
from datetime import datetime

RAZER_HOST = "razer"
RAZER_USER = "claude"
RAZER_LVL3 = r"C:\Users\claude\Lvl3Quant"

CHECKS = []

def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    CHECKS.append((name, passed, detail))
    icon = "✅" if passed else "❌"
    print(f"  {icon} {name}: {status}" + (f" — {detail}" if detail else ""))
    return passed

def ssh_razer(cmd, timeout=15):
    """Run command on Razer via SSH, return stdout or None on failure."""
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
             f"{RAZER_USER}@{RAZER_HOST}", cmd],
            capture_output=True, text=True, timeout=timeout
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception as e:
        return None

def main():
    print(f"\n{'='*60}")
    print(f"  PRE-MARKET CHECK — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*60}\n")

    # 1. Razer SSH connectivity
    print("[1/7] Razer connectivity...")
    razer_up = ssh_razer("echo ok") == "ok"
    check("Razer SSH reachable", razer_up)
    if not razer_up:
        print("\n  ⛔ Cannot reach Razer. Fix SSH before proceeding.\n")
        return 1

    # 2. GPU health
    print("\n[2/7] GPU health...")
    gpu_out = ssh_razer('nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.free --format=csv,noheader,nounits')
    if gpu_out:
        parts = [p.strip() for p in gpu_out.split(",")]
        gpu_util, gpu_temp, gpu_free_mb = int(parts[0]), int(parts[1]), int(parts[2])
        check("GPU detected", True, f"util={gpu_util}%, temp={gpu_temp}°C, free={gpu_free_mb}MB")
        check("GPU temp < 90°C", gpu_temp < 90, f"{gpu_temp}°C")
        check("GPU VRAM free > 2GB", gpu_free_mb > 2000, f"{gpu_free_mb}MB free")
    else:
        check("GPU detected", False, "nvidia-smi failed")

    # 3. CNN-Mamba v2 weights
    print("\n[3/7] Model weights...")
    weights_path = rf"{RAZER_LVL3}\output\cnn_mamba_v2_smart_v3_mar\fold_10_best.pt"
    weights_ok = ssh_razer(f'powershell -Command "Test-Path \'{weights_path}\'"')
    check("CNN-Mamba v2 weights", weights_ok and "True" in weights_ok, "fold_10_best.pt")

    feat_stats = rf"{RAZER_LVL3}\output\cnn_mamba_v2_smart_v3_mar\fold_09_feature_stats.npz"
    feat_ok = ssh_razer(f'powershell -Command "Test-Path \'{feat_stats}\'"')
    check("Feature stats file", feat_ok and "True" in feat_ok, "fold_09_feature_stats.npz")

    # 4. Meta-model weights (15 folds)
    print("\n[4/7] Meta-model weights...")
    meta_dir = rf"{RAZER_LVL3}\output\meta_production_v1\weights"
    meta_count = ssh_razer(f'powershell -Command "(dir \'{meta_dir}\' -File -ErrorAction SilentlyContinue).Count"')
    try:
        n_meta = int(meta_count.strip())
    except:
        n_meta = 0
    check("Meta-model folds (need 15)", n_meta >= 15, f"{n_meta} files")

    # 5. Stacked filter + config
    print("\n[5/7] Config and stacked filter...")
    config_path = rf"{RAZER_LVL3}\live_trading\configs\monday_stacked_v1.json"
    config_ok = ssh_razer(f'powershell -Command "Test-Path \'{config_path}\'"')
    check("Stacked config file", config_ok and "True" in config_ok, "monday_stacked_v1.json")

    filter_path = rf"{RAZER_LVL3}\live_trading\stacked_filter.py"
    filter_ok = ssh_razer(f'powershell -Command "Test-Path \'{filter_path}\'"')
    check("Stacked filter module", filter_ok and "True" in filter_ok, "stacked_filter.py")

    inference_path = rf"{RAZER_LVL3}\live_trading\mamba_inference.py"
    inf_ok = ssh_razer(f'powershell -Command "Test-Path \'{inference_path}\'"')
    check("Inference engine (CNNMambaV2)", inf_ok and "True" in inf_ok, "mamba_inference.py")

    # 6. Disk space
    print("\n[6/7] Disk space...")
    disk = ssh_razer('powershell -Command "(Get-PSDrive C).Free / 1GB"')
    try:
        free_gb = float(disk.strip())
        check("Disk space > 10GB", free_gb > 10, f"{free_gb:.1f}GB free")
    except:
        check("Disk space check", False, "Could not read disk")

    # 7. No zombie python processes
    print("\n[7/7] Process check...")
    procs = ssh_razer('powershell -Command "(Get-Process python -ErrorAction SilentlyContinue).Count"')
    try:
        n_procs = int(procs.strip()) if procs else 0
    except:
        n_procs = 0
    # During pre-market, we want 0 stale python processes (MFE relabeling should be done by Tuesday)
    check("No stale Python processes", n_procs == 0, f"{n_procs} running" if n_procs > 0 else "clean")

    # Summary
    total = len(CHECKS)
    passed = sum(1 for _, p, _ in CHECKS if p)
    failed = total - passed

    print(f"\n{'='*60}")
    if failed == 0:
        print(f"  ✅ ALL {total} CHECKS PASSED — Ready for paper trading")
    else:
        print(f"  ⚠️  {passed}/{total} passed, {failed} FAILED")
        print(f"  Fix failed items before going live.")
        for name, p, detail in CHECKS:
            if not p:
                print(f"    ❌ {name}: {detail}")
    print(f"{'='*60}\n")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
