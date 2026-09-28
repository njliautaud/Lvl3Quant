import numpy as np, json
data = np.load('/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz', allow_pickle=True)
dates = sorted(set(k[:-6] for k in data.files if k.endswith('_preds')))
all_z = []
for d in dates:
    p = data[d + '_preds']
    if p.std(): all_z.append((p - p.mean()) / p.std())
z = np.concatenate(all_z)
g1 = round(float(np.mean(np.abs(z) >= 1.0)) * 100, 2)
g2 = round(float(np.mean(np.abs(z) >= 2.0)) * 100, 2)
g3 = round(float(np.mean(np.abs(z) >= 3.0)) * 100, 2)
kurt = round(float(np.mean(z**4)) - 3.0, 2)
pct = [round(float(x), 3) for x in np.percentile(z, [5, 10, 25, 50, 75, 90, 95])]
est_z2_day = round(g2 / 100 * (len(z) / len(dates)) * 0.2)
raise RuntimeError('g1='+str(g1)+' g2='+str(g2)+' g3='+str(g3)+' kurt='+str(kurt)+' pct='+str(pct)+' est_z2_midday/day='+str(est_z2_day))
