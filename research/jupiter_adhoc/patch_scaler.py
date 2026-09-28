f = "/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_fusion_mlp.py"
content = open(f).read()
old = "from fusion_dataset import FusionDataset, load_cnn_preds_from_npz, N_OF_FEATURES"
new = "from fusion_dataset import FusionDataset, load_cnn_preds_from_npz, N_OF_FEATURES, fit_scaler, apply_scaler, save_scaler"
content = content.replace(old, new)
old2 = "        model = FusionMLP(in_dim)"
new2 = "        scaler = fit_scaler(train_ds)\n        train_ds = apply_scaler(train_ds, scaler)\n        val_ds = apply_scaler(val_ds, scaler)\n        scaler_path = os.path.join(args.output_dir, chr(115)+chr(99)+chr(97)+chr(108)+chr(101)+chr(114)+chr(95)+str(fold).zfill(2)+chr(46)+chr(112)+chr(107)+chr(108))\n        save_scaler(scaler, scaler_path)\n        model = FusionMLP(in_dim)"
content = content.replace(old2, new2)
open(f,"w").write(content)
print("done")
