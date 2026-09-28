import numpy as np, json
data = np.load('/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz', allow_pickle=True)
dates = sorted(set(k[:-6] for k in data.files if k.endswith('_preds')))
res = []
for d in dates:
    p = data[d + '_preds']; t = data[d + '_targets']
    n = min(len(p), len(t))
    ic = np.corrcoef(p[:n], t[:n])[0, 1]
    res.append((d, round(float(ic), 4)))
# First half vs second half
n h = len(res)//2
first = np.mean([r[1] for r in res[:h]])
second = np.mean([r[1] for r in res[h:	])
raise RuntimeError('first_half_IC='+str(round(first,4))+' second_half_IC='+str(round(second,4))+' dates_1st5='+str(res[:5])+' dates_last5='+str(res[-5:]))
