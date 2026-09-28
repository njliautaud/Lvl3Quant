import torch
d = torch.load('/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_derived/20250714_mbo_events.pt', map_location='cpu')
ev = d['events']
lb1 = d['labels_1s']
lb10 = d['labels_10s']
msg = 'events_shape='+str(ev.shape)+' labels_1s_shape='+str(lb1.shape)+' labels_10s_shape='+str(lb10.shape)+' ev_sample='+str(ev[:3,:5].tolist())+' lb1_sample='+str(lb1[:5].tolist())+' lb10_sample='+str(lb10[:5].tolist())
raise RuntimeError(msg)
