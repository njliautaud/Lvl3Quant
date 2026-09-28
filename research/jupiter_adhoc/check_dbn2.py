import databento as db
import numpy as np
f = '/home/jupiter/Lvl3Quant/data/raw/mbo/glbx-mdp3-20250722.mbo.dbn.zst'
store = db.DBNStore.from_file(f)
df = store.to_df()
cols = list(df.columns)
rows = len(df)
first = {k: str(v) for k,v in df.iloc[0].to_dict().items()}
msg = 'cols='+str(cols)+' rows='+str(rows)+' first='+str(first)
raise RuntimeError(msg)
