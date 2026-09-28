import numpy as np, os, json

PREDS_NPZ = '/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz'
OF12_DIR = '/home/jupiter/Lvl3Quant/data/processed/orderflow_features'
OF4_DIR = '/home/jupiter/Lvl3Quant/data/processed/of4_dom_depth'

data = np.load(PREDS_NPZ, allow_pickle=True)
dates = sorted(set(k[:-6] for k in data.files if k.endswith('_preds')))
of_files = set(os.listdir(OF12_DIR))
of4_files = set(os.listdir(OF4_DIR))

pred_all, tgt_all = [], []
pred_mid, tgt_mid = [], []
pred_near, tgt_near = [], []
pred_both, tgt_both = [], []

for date_str in dates:
    of_key = date_str.replace('-', '')
    of12_fn = of_key + '_orderflow.npz'
    of4_fn = of_key + '_of4.npz'
    if of12_fn not in of_files or of4_fn not in of4_files:
        continue
    of12 = np.load(os.path.join(OF12_DIR, of12_fn))
    of4 = np.load(os.path.join(OF4_DIR, of4_fn))
    if 'roll_delta_10s' not in of12.files or 'poc_dist_ticks' not in of4.files:
        continue
    poc = of4['poc_dist_ticks']
    preds = data[date_str + '_preds']
    targets = data[date_str + '_targets']
    n = min(len(poc), len(preds), len(targets))
    if n < 100:
        continue
    # Midday mask: 40-60% of events by position
    mid_mask = (np.arange(n) >= int(0.4 * n)) & (np.arange(n) < int(0.6 * n))
    # Near PoC mask: poc_dist below median abs
    median_poc = np.median(np.abs(poc[:n]))
    near_mask = np.abs(poc[:n]) <= median_poc
    both_mask = mid_mask & near_mask

    pred_all.append(preds[:n])
    tgt_all.append(targets[:n])
    pred_mid.append(preds[:n][mid_mask])
    tgt_mid.append(targets[:n][mid_mask])
    pred_near.append(preds[:n][near_mask])
    tgt_near.append(targets[:n][near_mask])
    pred_both.append(preds[:n][both_mask])
    tgt_both.append(targets[:n][both_mask])

if pred_all:
    pa = np.concatenate(pred_all); ta = np.concatenate(tgt_all)
    pm = np.concatenate(pred_mid); tm = np.concatenate(tgt_mid)
    pn = np.concatenate(pred_near); tn = np.concatenate(tgt_near)
    pb = np.concatenate(pred_both); tb = np.concatenate(tgt_both)

    ic_all = np.corrcoef(pa, ta)[0, 1]
    ic_mid = np.corrcoef(pm, tm)[0, 1]
    ic_near = np.corrcoef(pn, tn)[0, 1]
    ic_both = np.corrcoef(pb, tb)[0, 1]

    print('All conditions   IC=' + str(round(ic_all, 4)) + ' n=' + str(len(pa)))
    print('Midday only      IC=' + str(round(ic_mid, 4)) + ' n=' + str(len(pm)))
    print('Near PoC only    IC=' + str(round(ic_near, 4)) + ' n=' + str(len(pn)))
    print('Midday+NearPoC   IC=' + str(round(ic_both, 4)) + ' n=' + str(len(pb)))

    results = {
        'ic_all': float(ic_all), 'n_all': int(len(pa)),
        'ic_midday': float(ic_mid), 'n_midday': int(len(pm)),
        'ic_near_poc': float(ic_near), 'n_near_poc': int(len(pn)),
        'ic_midday_and_near_poc': float(ic_both), 'n_both': int(len(pb))
    }
    with open('/home/jupiter/combined_filter_ic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('DONE')
