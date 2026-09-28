import numpy as np
oot = np.load('/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz', allow_pickle=True)
pred_shape = oot['2025-07-22_preds'].shape
targ_shape = oot['2025-07-22_targets'].shape
bc = np.load('/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot/2025-07-22_book_tensors.npz', allow_pickle=True)
ts = bc['timestamps']
msg = 'pred_shape='+str(pred_shape)+' targ_shape='+str(targ_shape)+' n_bars='+str(len(ts))+' ts_first='+str(ts[0])+' ts_last='+str(ts[-1])+' ts_dtype='+str(ts.dtype)
raise RuntimeError(msg)
