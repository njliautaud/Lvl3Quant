import numpy as np
data = np.load('/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz', allow_pickle=True)
dates = sorted(set(k[:-6] for k in data.files if k.endswith('_preds')))
res = []
for d in dates:
    p = data[d + '_preds']
    t = data[d + '_targets']
    n = min(len(p), len(t))
    ic = float(np.corrcoef(p[:n], t[:n])[0, 1])
    res.append((d, round(ic, 4)))
nh = len(res) // 2
first = round(float(np.mean([r[1] for r in res[:nh]])), 4)
second = round(float(np.mean([r[1] for r in res[nh:]])), 4)
msg = 'first_half=' + str(first) + ' second_half=' + str(second) + ' first5=' + str(res[:5]) + ' last5=' + str(res[-5:])
raise RuntimeError(msg)
