import torch
m=torch.load('/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_mamba/20250714_mbo_events.pt',map_location='cpu')
print('mamba events shape:',m['events'].shape)
print('mamba labels_10s shape:',m['labels_10s'].shape)
d=torch.load('/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_derived/20250714_mbo_events.pt',map_location='cpu')
print('derived events shape:',d['events'].shape)
print('derived labels_10s shape:',d['labels_10s'].shape)
