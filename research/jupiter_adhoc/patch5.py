f = "/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_fusion_mlp.py"
content = open(f).read()
old = "    cnn_by_date = {d.replace(dash,empty): cnn_data[d+"
print("dup found:", old in content)
lines = content.split("\n")
lines = [l for l in lines if not ("cnn_by_date = {d.replace(dash,empty)" in l)]
content = "\n".join(lines)
open(f,"w").write(content)
print("done")
