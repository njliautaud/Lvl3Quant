import torch, numpy as np
d = torch.load('/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_derived/20250714_mbo_events.pt', map_location='cpu')
l1s = d['labels_1s'].numpy()
l5s = d['labels_5s'].numpy()
l10s = d['labels_10s'].numpy()
msg = '1s_std='+repr(round(float(np.std(l1s)),4))+' 5s_std='+repr(round(float(np.std(l5s)),4))+' 10s_std='+repr(round(float(np.std(l10s)),4))+' 1s_max='+repr(round(float(np.max(np.abs(l1s))),4))+' 1s_min='+repr(round(float(np.min(l1s)),2))+' 1s_sample='+str(l1s[:10].tolist())+' 10s_sample='+str(l10s[:10].tolist())
raise RuntimeError(msg)
