f = "/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_fusion_mlp.py"
content = open(f).read()
old = "    all_dates = [d.replace(chr(45), chr(0)[:0]) for d in all_dates]"
new = "    cnn_by_date = {d.replace(chr(45),chr(0)[:0]): cnn_data[d+chr(95)+chr(112)+chr(114)+chr(101)+chr(100)+chr(115)].astype(chr(102)+chr(108)+chr(111)+chr(97)+chr(116)+chr(51)+chr(50)) for d in all_dates if d+chr(95)+chr(112)+chr(114)+chr(101)+chr(100)+chr(115) in cnn_data.files}\n    all_dates = [d.replace(chr(45), chr(0)[:0]) for d in all_dates]"
print("found:", old in content)
content = content.replace(old, new)
open(f,"w").write(content)
print("done")
