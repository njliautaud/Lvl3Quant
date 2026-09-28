import numpy as np, os
PER_DAY_DIR = '/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/per_day_oos'
fn = sorted(os.listdir(PER_DAY_DIR))[0]
d = np.load(os.path.join(PER_DAY_DIR, fn), allow_pickle=True)
keys = list(d.files)
shapes = str({k: d[k].shape for k in keys})
raise RuntimeError(fn + ' keys=' + str(keys) + ' shapes=' + shapes)
