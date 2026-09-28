import numpy as np, os, json

PREDS_NPZ = '/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz'
OF12_DIR = '/home/jupiter/Lvl3Quant/data/processed/orderflow_features'

data = np.load(PREDS_NPZ, allow_pickle=True)
# Keys are YYYY-MM-DD_preds, OF files are YYYYMMDD_orderflow.npz
dates = sorted(set(k[:-6] for k in data.files if k.endswith('_preds')))
print('Found', len(dates), 'pred dates, first:', dates[0])

of_files = set(os.listdir(OF12_DIR))
matches = [d for d in dates if d.replace('-', '') + '_orderflow.npz' in of_files]
print('Matching dates:', len(matches))

results = {'tod': {}, 'magnitude': {}}

for period in ['open', 'midday', 'close']:
    preds_list, targets_list = [], []
    for date_str in matches:
        of_key = date_str.replace('-', '')
        of_data = np.load(os.path.join(OF12_DIR, of_key + '_orderflow.npz'))
        if 'roll_delta_10s' not in of_data.files:
            continue
        roll = of_data['roll_delta_10s']
        preds = data[date_str + '_preds']
        targets = data[date_str + '_targets']
        n = min(len(roll), len(preds), len(targets))
        if n < 10:
            continue
        if period == 'open':
            mask = np.arange(n) < int(0.2 * n)
        elif period == 'midday':
            mask = (np.arange(n) >= int(0.4 * n)) & (np.arange(n) < int(0.6 * n))
        else:
            mask = np.arange(n) >= int(0.8 * n)
        preds_list.append(preds[:n][mask])
        targets_list.append(targets[:n][mask])
    if preds_list:
        p = np.concatenate(preds_list)
        t = np.concatenate(targets_list)
        ic = np.corrcoef(p, t)[0, 1]
        results['tod'][period] = {'ic': float(ic), 'n': int(len(p))}
        print(period + ' IC=' + str(round(ic, 4)) + ' n=' + str(len(p)))

all_roll, all_pred, all_tgt = [], [], []
for date_str in matches:
    of_key = date_str.replace('-', '')
    of_data = np.load(os.path.join(OF12_DIR, of_key + '_orderflow.npz'))
    if 'roll_delta_10s' not in of_data.files:
        continue
    roll = of_data['roll_delta_10s']
    preds = data[date_str + '_preds']
    targets = data[date_str + '_targets']
    n = min(len(roll), len(preds), len(targets))
    all_roll.append(roll[:n])
    all_pred.append(preds[:n])
    all_tgt.append(targets[:n])

if all_roll:
    roll_cat = np.concatenate(all_roll)
    pred_cat = np.concatenate(all_pred)
    tgt_cat = np.concatenate(all_tgt)
    abs_roll = np.abs(roll_cat)
    quintiles = np.percentile(abs_roll, [20, 40, 60, 80])
    bins = ['P0-20', 'P20-40', 'P40-60', 'P60-80', 'P80-100']
    for j, label in enumerate(bins):
        if j == 0:
            mask = abs_roll <= quintiles[0]
        elif j == 4:
            mask = abs_roll > quintiles[3]
        else:
            mask = (abs_roll > quintiles[j-1]) & (abs_roll <= quintiles[j])
        p_sub, t_sub = pred_cat[mask], tgt_cat[mask]
        if len(p_sub) > 100:
            ic = np.corrcoef(p_sub, t_sub)[0, 1]
            results['magnitude'][label] = {'ic': float(ic), 'n': int(len(p_sub))}
            print(label + ' IC=' + str(round(ic, 4)) + ' n=' + str(len(p_sub)))

import json
with open('/home/jupiter/roll_delta_tod_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('DONE')
