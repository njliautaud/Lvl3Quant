import numpy as np, os
OF12_DIR = "/home/jupiter/Lvl3Quant/data/processed/orderflow_features"
CTX_DIR = "/home/jupiter/Lvl3Quant/data/processed/fusion_context"
dates = sorted(set(f[:8] for f in os.listdir(CTX_DIR) if f.endswith("_fusion_context.npz")))
print("Fixing", len(dates), "files")
ok=0
for date in dates:
    ctx_path = os.path.join(CTX_DIR, date+"_fusion_context.npz")
    of12_path = os.path.join(OF12_DIR, date+"_orderflow.npz")
    if not os.path.exists(of12_path): continue
    d12 = np.load(of12_path)
    bid = d12["total_bid_size"]; ask = d12["total_ask_size"]
    denom = bid + ask
    size_imb = float(np.nanmean(np.where(denom>0, bid/denom, 0.5)))
    ctx = np.load(ctx_path)["of_context"].copy()
    ctx[4] = size_imb
    np.savez(ctx_path, of_context=ctx)
    ok+=1
print("Fixed", ok, "files")
