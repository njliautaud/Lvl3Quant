import numpy as np
d = np.load('/home/jupiter/Lvl3Quant/data/processed/orderflow_features/20250714_orderflow.npz'i
print('OF12 keys:', list(d.keys()))
print('first array shapes:', {(k: d[k].shape for k in list(d.keys())[:10]}))
