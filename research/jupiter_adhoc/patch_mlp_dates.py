f = "/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_fusion_mlp.py"
content = open(f).read()
dash = chr(45)
empty = ""
old = "cnn_by_date = {d: cnn_data[d+"+chr(34)+"_preds"+chr(34)+"].astype(np.float32) for d in all_dates if d+"+chr(34)+"_preds"+chr(34)+" in cnn_data.files}"
new = "cnn_by_date = {d.replace(dash,empty): cnn_data[d+"+chr(34)+"_preds"+chr(34)+"].astype(np.float32) for d in all_dates if d+"+chr(34)+"_preds"+chr(34)+" in cnn_data.files}"
print("old found:", old in content)
content = content.replace(old, new)
open(f,"w").write(content)
print("done")
