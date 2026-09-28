f = "/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_fusion_mlp.py"
content = open(f).read()
old = "    fold_size = max(1, len(all_dates) // args.n_folds)"
new = "    all_dates = [d.replace(chr(45), chr(0)[:0]) for d in all_dates]\n    dash = chr(45)\n    empty = chr(0)[:0]\n    fold_size = max(1, len(all_dates) // args.n_folds)"
print("found:", old in content)
content = content.replace(old, new)
open(f,"w").write(content)
print("done")
