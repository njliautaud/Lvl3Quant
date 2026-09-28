import numpy as np, os, json

OF12_DIR = '/home/jupiter/Lvl3Quant/data/processed/orderflow_features'

all_roll = []
for fn in sorted(os.listdir(OF12_DIR)):
    if not fn.endswith('_orderflow.npz'):
        continue
    d = np.load(os.path.join(OF12_DIR, fn))
    if 'roll_delta_10s' not in d.files:
        continue
    roll = d['roll_delta_10s'].astype(np.float32)
    if len(roll) > 100:
        all_roll.append(roll)

print('Days loaded:', len(all_roll))

# Compute autocorrelation at lags 1,3,5,10,20,50,100,300
lags = [1, 3, 5, 10, 20, 50, 100, 300]
results = {}

for lag in lags:
    ac_list = []
    for roll in all_roll:
        if len(roll) <= lag:
            continue
        x1 = roll[:-lag]
        x2 = roll[lag:]
        mu1 = x1.mean()
        mu2 = x2.mean()
        s1 = x1.std()
        s2 = x2.std()
        if s1 > 0 and s2 > 0:
            ac = np.mean((x1 - mu1) * (x2 - mu2)) / (s1 * s2)
            ac_list.append(float(ac))
    if ac_list:
        mean_ac = np.mean(ac_list)
        results[lag] = round(float(mean_ac), 4)
        print('lag ' + str(lag) + ' events autocor=' + str(round(float(mean_ac), 4)))

with open('/home/jupiter/roll_delta_autocorr.json', 'w') as f:
    json.dump(results, f, indent=2)
print('DONE')
