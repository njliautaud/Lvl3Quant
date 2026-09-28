import sys
sys.path.insert(0,"/home/jupiter/Lvl3Quant/alpha_discovery/deep_models")
from fusion_dataset import load_of_features_for_date
result = load_of_features_for_date("20251201")
if result is None:
    print("NONE returned")
else:
    print("shape:", result.shape)
