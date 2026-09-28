import numpy as np
d = np.load('/home/jupiter/Lvl3Quant/data/processed/of4_dom_depth/20250714_of4.npz')
print('OF4 keys:', list(d.keys()))
print('shapes:', {k: d[k].shape for k in list(d.keys())})
