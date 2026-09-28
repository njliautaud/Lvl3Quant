"""
Razer GPU launcher for deep confluence meta-model (HC #469 e1).
Run via: schtasks + pythonw.exe to survive SSH session close.
Logs to output/cm_deep_v1.log.
"""
import sys
import os
import traceback

os.chdir(r'C:\Users\claude\Lvl3Quant')
log = open('output/cm_deep_v1.log', 'w')

# Redirect all stdout/stderr to log
sys.stdout = log
sys.stderr = log

print("LAUNCHER STARTED", flush=True)

try:
    sys.argv = [
        'train_confluence_meta_deep_v1.py',
        '--device', 'cuda',
        '--wf-window', '20',
        '--epochs', '30',
        '--batch-size', '4096',
        '--dropout', '0.2',
    ]
    # Use compile+exec so __file__ is set correctly
    script_path = 'scripts/train_confluence_meta_deep_v1.py'
    with open(script_path) as f:
        code = compile(f.read(), script_path, 'exec')
    exec(code)
except Exception:
    print(f"\nCRASH:\n{traceback.format_exc()}", flush=True)

print("\nLAUNCHER DONE", flush=True)
log.close()
