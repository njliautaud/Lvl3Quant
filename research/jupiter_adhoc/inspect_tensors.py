import torch, numpy as np
m=torch.load('/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_mamba/20250714_mbo_events.pt', map_location='cpu')
print('Mamba tensor type:', type(m))
if hasattr(m,'shape'): print('shape:',m.shape)
elif isinstance(m,dict): print('keys:',list(m.keys())[:5])
d=torch.load('/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_derived/20250714_mbo_events.pt', map_location='cpu')
print('Derived tensor type:', type(d))
if hasattr(d,'shape'): print('shape:',d.shape)
elif isinstance(d,dict): print('keys:',list(d.keys())[:5])
