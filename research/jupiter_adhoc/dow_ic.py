import numpy as np, os, json, datetime

PREDS_NPZ = '/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz'

data = np.load(PREDS_NPZ, allow_pickle=True)
dates = sorted(set(k[:-6] for k in data.files if k.endswith('_preds')))

dow_preds = {0:[], 1:[], 2:[], 3:[], 4:[]}
dow_tgts = {0:[], 1:[], 2:[], 3:[], 4:[]}
dow_names = {0:'Mon', 1:'Tue', 2:'wed', 3:'Thu', 4:'Fri'}

for date_str in dates:
    dt = datetime.date.fromisoformat(date_str)
    dow = dt.weekday()
    preds = data[date_str + '_preds']
    targets = data[date_str + '_targets']
    dow_preds[dow].append(preds)
    dow_tgts[dow].append(targets)

results = {}
for dow in range(5):
    if not dow_preds[dow]:
        continue
    p = np.concatenate(dow_preds[dow])
    t = np.concatenate(dow_tgts[dow])
    ic = np.corrcoef(p, t)[0, 1]
    ndays = len(dow_preds[dow])
    results[dow_names[dow]] = {'ic': round(float(ic), 4), 'n_days': ndays, 'n_events': int(len(p))}
    print(dow_names[dow] + ' ndays=' + str(ndays) + ' IC=' + str(round(ic, 4)))

with open('/home/jupiter/dow_ic_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('DONE')
