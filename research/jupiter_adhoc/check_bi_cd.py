import numpy as np
of12 = np.load('/home/jupiter/Lvl3Quant/data/processed/orderflow_features/20250722_orderflow.npz')
bi = of12['book_imbalance']
cd = of12['cum_delta']
are_equal = bool(np.all(bi == cd))
corr_bi_cd = float(np.corrcoef(bi, cd)[0,1])
msg = 'are_equal='+str(are_equal)+' corr_bi_cd='+str(round(corr_bi_cd,6))+' bi_std='+str(round(float(np.std(bi)),4))+' cd_std='+str(round(float(np.std(cd)),4))+' bi_sample='+str(bi[:10].tolist())
raise RuntimeError(msg)
