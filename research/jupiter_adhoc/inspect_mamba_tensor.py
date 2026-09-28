import torch
f = '/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_mamba/20250714_mbo_events.pt'
d = torch.load(f, map_location='cpu')
if isinstance(d, dict):
    info = { k: (str(type(v)), getattr(v,'shape',None), str(getattr(v,'dtype',None))) for k,v in d.items() }
elif isinstance(d, torch.Tensor):
    info = {'shape': str(d.shape), 'dtype': str(d.dtype), 'sample[0]': str(d[0])}
else:
    info = {'type': str(type(d))}
raise RuntimeError(str(info))
