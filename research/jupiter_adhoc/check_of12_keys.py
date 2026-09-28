import numpy as np
d = np.load('/home/jupiter/Lvl3Quant/data/processed/orderflow_features/20250714_orderflow.npz')
keys = list(d.keys())
shapes = str({k: d[k].shape for k in keys})
raise RuntimeError('keys=' + str(keys) + ' shapes=' + shapes)
