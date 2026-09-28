import torch
d = torch.load('/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_derived/20250714_mbo_events.pt', map_location='cpu')
if isinstance(d, dict):
    keys = list(d.keys())
    raise RuntimeError('type=dict keys=' + str(keys))
else:
    raise RuntimeError('type=' + str(type(d)) + ' shape=' + str(getattr(d, 'shape', 'N/A')))
