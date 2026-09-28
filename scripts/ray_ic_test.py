import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

print('=== CNN IC TEST VIA RAY ===')

pred_dir = Path('/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions')
pred_files = sorted(pred_dir.glob('*.npz'))

results = {}
for pf in pred_files[:10]:
    data = np.load(pf)
    for k in data.keys():
        if '_preds' in k:
            date = k.replace('_preds','')
            tk = date + '_targets'
            if tk in data:
                ic, _ = spearmanr(data[k], data[tk])
                results[date] = float(ic)
                print(f'  {date}: IC={ic:+.4f}')
            break

if results:
    ics = list(results.values())
    print(f'Mean IC: {sum(ics)/len(ics):+.4f}')
    print(f'Positive: {sum(1 for x in ics if x > 0)}/{len(ics)}')

try:
    import mlflow
    mlflow.set_tracking_uri('http://neptune:5000')
    mlflow.set_experiment('Signal_Research')
    with mlflow.start_run(run_name='cnn_ic_ray'):
        if results:
            mlflow.log_metrics({'mean_ic': float(sum(ics)/len(ics)), 'n_dates': len(results)})
    print('MLflow logged')
except Exception as e:
    print(f'MLflow: {e}')
print('DONE')
