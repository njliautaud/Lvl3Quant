import numpy as np
import os

DATA_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_events/'
f = np.load(DATA_DIR + '20250714_mbo_events.npz')
print('Keys:', list(f.keys()))
arr = f['features']
print('features shape:', arr.shape)
print('sample row 0:', arr[0])
print('event_type_id unique:', np.unique(arr[:,1]))
print('side_id unique:', np.unique(arr[:,2]))
print('labels_10s[:5]:', f['labels_10s'][:5])
