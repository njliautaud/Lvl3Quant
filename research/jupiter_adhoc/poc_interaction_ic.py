import numpy as np, os, json

PREDS_NPZ = '/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz'
OF12_DIR = '/home/jupiter/Lvl3Quant/data/processed/orderflow_features'
OF4_DIR = '/home/jupiter/Lvl3Quant/data/processed/of4_dom_depth'

data = np.load(PREDS_NPZ, allow_pickle=True)
dates = sorted(set(k[:-6] for k in data.files if k.endswith('_preds')))
of_files = set(os.listdir(OF12_DIR))
of4_files = set(os.listdir(OF4_DIR))

# Build feature arrays across all matching dates
roll_all, poc_all, inter_all, pred_all, tgt_all = [], [], [], [], []

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
    roll = of12['roll_delta_10s']
    poc = of4['poc_dist_ticks']
    preds = data[date_str + '_preds']
    targets = data[date_str + '_targets']
    n = min(len(roll), len(poc), len(preds), len(targets))
    if n < 100:
        continue
    roll_n = roll[:n]
    poc_n = poc[:n]
    # Normalize poc to zmap to avoid scale domination
    poc_std = poc_n.std()
    poc_norm = poc_n / (poc_std + 1e-8)
    roll_norm = roll_n / (roll_n.std() + 1e-8)
    inter = roll_norm * poc_norm
    roll_all.append(roll_n[:n])
    poc_all.append(poc_n[:n])
    inter_all.append(inter[:n])
    pred_all.append(preds[:n])
    tgt_all.append(targets[:n])

if not roll_all:
    print('No matching dates found')
else:
    roll_cat = np.concatenate(roll_all)
    poc_cat = np.concatenate(poc_all)
    inter_cat = np.concatenate(inter_all)
    pred_cat = np.concatenate(pred_all)
    tgt_cat = np.concatenate(tgt_all)
    print('N events:', len(tgt_cat))

    ic_roll = np.corrcoef(roll_cat, tgt_cat)[0, 1]
    ic_poc = np.corrcoef(poc_cat, tgt_cat)[0, 1]
    ic_inter = np.corrcoef(inter_cat, tgt_cat)[0, 1]
    ic_pred = np.corrcoef(pred_cat, tgt_cat)[0, 1]

    print('IC roll_delta_10s vs target: ' + str(round(ic_roll, 4)))
    print('IC poc_dist_ticks vs target: ' + str(round(ic_poc, 4)))
    print('IC roll*poc_norm vs target: ' + str(round(ic_inter, 4)))
    print('IC CNN preds vs target: ' + str(round(ic_pred, 4)))

    # Also test if poc_dist modulates CNN pred IC
    # Split by poc_dist median: near PoC vs far from PoC
    median_poc = np.median(np.abs(poc_cat))
    near_mask = np.abs(poc_cat) <= median_poc
    far_mask = ~near_mask
    ic_near = np.corrcoef(pred_cat[near_mask], tgt_cat[near_mask])[0, 1]
    ic_far = np.corrcoef(pred_cat[far_mask], tgt_cat[far_mask])[0, 1]
    print('CNn IC near PoC (low poc_dist): ' + str(round(ic_near, 4)))
    print('CNN IC far from PoC (high poc_dist): ' + str(round(ic_far, 4)))

    results = {
        'ic_roll': float(ic_roll),
        'ic_poc': float(ic_poc),
        'ic_roll_poc_inter': float(ic_inter),
        'ic_cnn_pred': float(ic_pred),
        'ic_cnn_near_poc': float(ic_near),
        'ic_cnn_far_poc': float(ic_far),
        'n_events': int(len(tgt_cat))
    }
    with open('/home/jupiter/poc_interaction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('DONE')
