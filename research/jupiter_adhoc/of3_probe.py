import databento as db, numpy as np
store = db.DBNStore.from_file("/home/jupiter/Lvl3Quant/data/raw/mbo/glbx-mdp3-20260126.mbo.dbn.zst")
df = store.to_df().reset_index()
ts = df["ts_event"].dt.tz_convert("UTC")
rth_s = ts.dt.normalize() + np.timedelta64(52200,"s")
rth_e = ts.dt.normalize() + np.timedelta64(75600,"s")
r = df[(ts >= rth_s) & (ts < rth_e)]
print("RTH:", len(r))
print("actions:", dict(r["action"].value_counts()))
print("large50:", int((r["size"]>=50).sum()))
tc = r[r["action"]=="T"].groupby("order_id").size()
print("iceberg_cands:", int((tc>1).sum()))
