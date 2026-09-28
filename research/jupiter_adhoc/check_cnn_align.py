import numpy as np
oot = np.load('/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz', allow_pickle=True)
p = oot['2025-07-22_preds']
t = oot['2025-07-22_targets']
bc = np.load('/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot/2025-07-22_book_tensors.npz', allow_pickle=True)
mid = bc['mid_prices']
len_p = len(p)
len_m = len(mid)
offset = len_m - len_p
# Compute bar-aligned 10s (lag-100) forward return
lag = 100
fr_endalign = (mid[offset+lag:len_m] - mid[offset:len_m-lag]) / 0.25
fr_startalign = (mid[lag:len_p] - mid[:len_p-lag]) / 0.25
p_end = p[:len_m-offset-lag]
p_start = p[:len_p-lag]
ic_end = float(np.corrcoef(p_end, fr_endalign)[0,1])
ic_start = float(np.corrcoef(p_start, fr_startalign)[0,1])
# Also check target alignment - targets are stored 10s return
ic_targ_end = float(np.corrcoef(p_end, t[:len_m-offset-lag])[0,1])
ic_targ_start = float(np.corrcoef(p_start, t[:len_p-lag])[0,1])
msg = 'ic_endalign='+str(round(ic_end,4))+' ic_startalign='+str(round(ic_start,4))+' ic_targ_end='+str(round(ic_targ_end,4))+' ic_targ_start='+str(round(ic_targ_start,4))+' offset='+str(offset)
raise RuntimeError(msg)
