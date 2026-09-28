import numpy as np
d = np.load('/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz', allow_pickle=True)
keys = list(d.files)
raise RuntimeError('all_keys=' + str(keys[:30]))
