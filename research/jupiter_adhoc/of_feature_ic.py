import numpy as np, os, json
from scipy.stats import pearsonr

PREDS_NPZ = "/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz"
OF12_DIR = "/home/jupiter/Lvl3Quant/data/processed/orderflow_features"
OF4_DIR  = "/home/jupiter/Lvl3Quant/data/processed/of4_dom_depth"

OF12_KEYS = ["book_imbalance","cum_delta","roll_delta_10s","total_bid_size","total_ask_size"]
OF4_KEYS  = ["depth_imbalance_5","depth_imbalance_10","bid_depth_5","ask_depth_5","wall_imbalance","poc_dist_ticks"]

d = np.load(PREDS_NPZ)
dates = sorted(set(k.replace("_targets","") for k in d.files if k.endswith("_targets")))
print("Dates with targets: " + str(len(dates)))

all_feats = {k:[] for k in OF12_KEYS+OF4_KEYS}
all_targets = []
skipped = 0

for date in dates:
    compact = date.replace("-","")
    p12 = os.path.join(OF12_DIR, compact+"_orderflow.npz")
    p4  = os.path.join(OF4_DIR,  compact+"_of4.npz")
    if not (os.path.exists(p12) and os.path.exists(p4)):
        skipped += 1
        continue
    targets = d[date+"_targets"].astype(np.float32)
    d12 = np.load(p12)
    d4  = np.load(p4)
    n = min(len(targets), len(d12[OF12_KEYS[0]]), len(d4[OF4_KEYS[0]]))
    targets = targets[:n]
    idx = np.random.choice(n, min(5000,n), replace=False)
    all_targets.append(targets[idx])
    for k in OF12_KEYS:
        arr = d12[k].astype(np.float32)
        all_feats[k].append(arr[idx[:min(len(idx),len(arr))]])
    for k in OF4_KEYS:
        arr = d4[k].astype(np.float32)
        all_feats[k].append(arr[idx[:min(len(idx),len(arr))]])

print("Loaded " + str(len(all_targets)) + " days, skipped " + str(skipped))
y = np.concatenate(all_targets)
results = {}
for k in OF12_KEYS+OF4_KEYS:
    x = np.concatenate(all_feats[k])
    n = min(len(x),len(y))
    r,p = pearsonr(x[:n], y[:n])
    results[k] = {"ic": round(float(r),5), "pval": round(float(p),4)}

ranked = sorted(results.items(), key=lambda x: abs(x[1]["ic"]), reverse=True)
print("=== OF Feature IC vs 10s Forward Return ===")
for feat,v in ranked:
    print("  " + feat.ljust(30) + "  IC=" + str(v["ic"]) + "  p=" + str(v["pval"]))

bi = np.concatenate(all_feats["book_imbalance"])
cd = np.concatenate(all_feats["cum_delta"])
n = min(len(bi),len(cd),len(y))
interaction = bi[:n]*cd[:n]
r_int,p_int = pearsonr(interaction, y[:n])
print("  book_imb * cum_delta (interaction): IC=" + str(round(float(r_int),5)) + "  p=" + str(round(float(p_int),4)))

with open("/home/jupiter/of_feature_ic_results.json","w") as f:
    json.dump({"features":results, "interaction_book_imb_x_cum_delta":{"ic":float(r_int),"pval":float(p_int)}}, f, indent=2)
print("Saved to /home/jupiter/of_feature_ic_results.json")
