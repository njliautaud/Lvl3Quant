f = "/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/fusion_dataset.py"
content = open(f).read()
old = "    n = 234000"
new = "    n = len(np.load(d12p)[OF12_KEYS[0]])"
print("found:", old in content)
content = content.replace(old, new)
open(f,"w").write(content)
print("done")
