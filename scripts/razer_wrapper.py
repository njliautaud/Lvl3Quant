"""Wrapper to run train_confluence_meta_deep_v1.py with error capture."""
import traceback
import sys
import os

os.chdir(r'C:\Users\claude\Lvl3Quant')
logf = open('output/cm_deep_wrapper.log', 'w')
logf.write('WRAPPER STARTED\n')
logf.flush()

try:
    sys.argv = ['x', '--device', 'cuda', '--wf-window', '20', '--epochs', '30', '--batch-size', '4096']
    exec(compile(open('scripts/train_confluence_meta_deep_v1.py').read(),
                 'scripts/train_confluence_meta_deep_v1.py', 'exec'))
except Exception as e:
    logf.write(f'ERROR:\n{traceback.format_exc()}\n')
    logf.flush()

logf.write('\nDONE\n')
logf.close()
