import numpy as np, os, json
results = {}
for feat, d in [('OF1_OF2','/home/jupiter/Lvl3Quant/data/processed/orderflow_features'), ('OF3','/home/jupiter/Lvl3Quant/data/processed/of3_large_order'), ('OF4','/home/jupiter/Lvl3Quant/data/processed/of4_dom_depth')]:
    files = sorted([f for f in os.listdir(d) if f.endswith('.npz')])
    nan_files=0; inf_files=0; total=0
    for f in files[:10]:  # spot check first 10
        data = np.load(os.path.join(d,f))
        for k in data.files:
            arr = data[k]
            if np.isnan(arr).any(): nan_files+=1
            if np.isinf(arr).any(): inf_files+=1
            total+=1
    results[feat] = {'files_checked':len(files[:10]),'arrays_checked':total,'nan_arrays':nan_files,'inf_arrays':inf_files}
print(json.dumps(results, indent=2))
