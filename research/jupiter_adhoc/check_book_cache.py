import numpy as np
d = np.load('/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot/2025-07-22_book_tensors.npz', allow_pickle=True)
keys = list(d.keys())
msg = 'keys='+str(keys)
for k in keys:
    msg += ' '+k+'='+str(d[k].shape)
raise RuntimeError(msg)
