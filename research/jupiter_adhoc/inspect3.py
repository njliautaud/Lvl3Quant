import torch
d=torch.load('/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_derived/20250714_mbo_events.pt',map_location='cpu')
print('derived keys:',list(d.keys()))
print('events[:,0,:3] sample (first window, first 3 events, first 3 feats):',d['events'][0,:3,:3])
# check if timestamps are in features
m=torch.load('/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_mamba/20250714_mbo_events.pt',map_location='cpu')
print('mamba keys:',list(m.keys()))
