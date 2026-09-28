import numpy as np
from scipy import stats
import json

mamba = np.load('/tmp/mamba_fold00_oot.npz', allow_pickle=True)
print('=== Mamba fold_00 OOT ===')
print('keys:', list(mamba.files)[:20])
for k in list(mamba.files)[:8]:
    v = mamba[k]
    print(f'  {k}: shape={getattr(v,"shape","?")} dtype={getattr(v,"dtype","?")}')

cnn = np.load('/tmp/cnn_wf_incremental.npz', allow_pickle=True)
print()
print('=== CNN WF incremental ===')
ckeys = list(cnn.files)
pred_keys = [k for k in ckeys if k.endswith("_preds")]
mid_keys = [k for k in ckeys if k.endswith("_mid")]
print(f'pred keys: {len(pred_keys)} days, mid keys: {len(mid_keys)} days')
if pred_keys:
    k0 = pred_keys[0]
    print(f'  {k0}: shape={cnn[k0].shape}')
print('date range:', sorted(pred_keys)[0], '->', sorted(pred_keys)[-1])
