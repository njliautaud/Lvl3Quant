import sys, numpy as np
sys.path.insert(0,"/home/jupiter/Lvl3Quant/alpha_discovery/deep_models")
from fusion_dataset import FusionDataset
cnn_data = np.load("/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/oot_wf_predictions_incremental.npz")
all_dates = sorted(set(k.replace("_preds","") for k in cnn_data.files if k.endswith("_preds")))
all_dates = [d.replace(chr(45),chr(0)[:0]) for d in all_dates]
print("compact dates sample:", all_dates[:3])
dash=chr(45); empty=chr(0)[:0]
cnn_by_date = {d.replace(dash,empty): cnn_data[d.replace(empty,chr(45))+"_preds"].astype("float32") for d in all_dates if d.replace(empty,chr(45))+"_preds" in cnn_data.files}
print("cnn_by_date keys:", list(cnn_by_date.keys())[:3])
