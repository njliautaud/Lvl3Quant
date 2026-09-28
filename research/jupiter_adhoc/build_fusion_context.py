import numpy as np, os
OF12_DIR = "/home/jupiter/Lvl3Quant/data/processed/orderflow_features"
OF3_DIR = "/home/jupiter/Lvl3Quant/data/processed/of3_large_order"
OF4_DIR = "/home/jupiter/Lvl3Quant/data/processed/of4_dom_depth"
OUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/fusion_context"
os.makedirs(OUT_DIR, exist_ok=True)
def smean(a): return float(np.nanmean(a)) if len(a)>0 else 0.0
def sstd(a): return float(np.nanstd(a)) if len(a)>0 else 0.0
dates = sorted(set(f[:8] for f in os.listdir(OF12_DIR) if f.endswith("_orderflow.npz")))
print("Processing", len(dates), "dates")
ok=0; skip=0; err=0
for date in dates:
    out_path = os.path.join(OUT_DIR, date+"_fusion_context.npz")
    if os.path.exists(out_path):
        ok+=1; continue
    d4p = os.path.join(OF4_DIR, date+"_of4.npz")
    d3p = os.path.join(OF3_DIR, date+".mbo.dbn_of3.npz")
    if not os.path.exists(d4p) or not os.path.exists(d3p):
        skip+=1; continue
    try:
        d12=np.load(os.path.join(OF12_DIR,date+"_orderflow.npz"))
        d4=np.load(d4p); d3=np.load(d3p)
        f1=smean(d12["book_imbalance"]); f2=sstd(d12["book_imbalance"])
        f3=float(d12["cum_delta"][-1]) if len(d12["cum_delta"])>0 else 0.0
        f4=smean(d12["roll_delta_10s"]); f5=float(np.mean(d12["large_order_mask"]))
        f6=smean(d3["large_nearby"]); f7=smean(d3["iceberg_active"])
        f8=smean(d3["large_imbalance"]); f9=smean(d3["cancel_rate"])
        f10=smean(d4["depth_imbalance_5"]); f11=smean(d4["depth_imbalance_10"])
        f12=smean(d4["wall_imbalance"]); f13=smean(d4["poc_dist_ticks"])
        ctx=np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13],dtype=np.float32)
        np.savez(out_path, of_context=ctx); ok+=1
    except Exception as e:
        print("ERR",date,e); err+=1
print("Done: ok=0 skip=0 err=0" % (ok,skip,err))
print("Done ok",ok,"skip",skip,"err",err)
