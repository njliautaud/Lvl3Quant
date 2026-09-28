import numpy as np, os, json

OF4_DIR = '/home/jupiter/Lvl3Quant/data/processed/of4_dom_depth'

all_poc = []
for fn in sorted(os.listdir(OF4_DIR)):
    if not fn.endswith('_of4.npz'):
        continue
    d = np.load(os.path.join(OF4_DIR, fn))
    if 'poc_dist_ticks' not in d.files:
        continue
    poc = d['poc_dist_ticks'].astype(np.float32)
    if len(poc) > 100:
        all_poc.append(poc)

print('Days loaded:', len(all_poc))

lags = [1, 3, 5, 10, 20, 50, 100, 300]
results = {}

for lag in lags:
    ac_list = []
    for poc in all_poc:
        if len(poc) <= lag:
            continue
        x1 = poc[:-lag]
        x2 = poc[lag:]
        s1 = x1.std()
        s2 = x2.std()
        if s1 > 0 and s2 > 0:
            ac = np.mean((x1 - x1.mean()) * (x2 - x2.mean())) / (s1 * s2)
            ac_list.append(float(ac))
    if ac_list:
        mean_ac = np.mean(ac_list)
        results[lag] = round(float(mean_ac), 4)
        print('lag ' + str(lag) + ' events poc_dist autocor=' + str(round(float(mean_ac), 4)))

with open('/home/jupiter/poc_dist_autocorr.json', 'w') as f:
    json.dump(results, f, indent=2)
print('DONE')
