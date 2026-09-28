@echo off
REM HC #489 R2 — DLinear extended OOT inference batch launcher
REM Razer Windows target. Loops 20 dates, calls infer_dlinear_quantile_hc489.py
REM Output: C:\Users\claude\Lvl3Quant\output\hc489_extended_oot\fold_*.npz

setlocal enabledelayedexpansion
set DATA_DIR=C:\Users\claude\Lvl3Quant\data\processed\mbo_events_smart_v3
set OUT_DIR=C:\Users\claude\Lvl3Quant\output\hc489_extended_oot
set CKPT=C:\Users\claude\Lvl3Quant\output\hc489_dlinear_quantile_asym_long_v1\intra_ckpt.pt
set SCRIPT=C:\Users\claude\Lvl3Quant\scripts\infer_dlinear_quantile_hc489.py

if not exist %OUT_DIR% mkdir %OUT_DIR%

REM Dates (20-date list, pre-training window)
set dates=20260301 20260302 20260303 20260304 20260305 20260306 20260308 20260309 20260310 20260311 20260312 20260313 20260315 20260316 20260317 20260318 20260319 20260320 20260322 20260323

for %%D in (%dates%) do (
    set DATE=%%D
    set INPUT=!DATA_DIR!\!DATE!_mbo_events.npz
    if exist !INPUT! (
        echo [!TIME!] Starting inference for !DATE!
        python !SCRIPT! !INPUT! !OUT_DIR! !CKPT!
        if !ERRORLEVEL! neq 0 (
            echo [!TIME!] FAILED: !DATE! (exit code !ERRORLEVEL!)
        ) else (
            echo [!TIME!] OK: !DATE!
        )
    ) else (
        echo [!TIME!] MISSING: !INPUT!
    )
)

echo [!TIME!] Batch inference complete.
