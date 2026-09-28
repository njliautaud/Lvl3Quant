import databento as db
import numpy as np
f = '/home/jupiter/Lvl3Quant/data/raw/mbo/glbx-mdp3-20250722.mbo.dbn.zst'
store = db.DBNStore(f)
df = store.to_df()
cols = list(df.columns)
rows = len(df)
msg = 'cols='+str(cols)+' rows='+str(rows)+' first_row='+str(df.iloc[0].to_dict())
raise RuntimeError(msg)
