"""
Compute 3 new derived features (features 16-18) and save alongside existing feat files.
New features discovered via IC analysis:
  16: price_sign_sq  = price_rel_ticks * |price_sign_mom_200|   IC_10s=0.0626
  17: fade_side_price = -side_id * price_rel_ticks              IC_10s=-0.0499 (contrarian)
  18: restore_x_price = fill_add_restore_100 * price_rel_ticks  IC_10s=0.0317
"""
import numpy as np
import glob
import os
import time

FEAT_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_events_feat/'
OUT_DIR  = '/home/jupiter/Lvl3Quant/data/processed/mbo_events_feat18/'
LOG      = '/home/jupiter/Lvl3Quant/compute_new_features.log'

os.makedirs(OUT_DIR, exist_ok=True)

FEAT_NAMES = [
    'time_delta_log','event_type_id','side_id','price_rel_ticks',
    'qty_log','spread_ticks','cancel_side_asym_50','rolling_ofi_500',
    'event_density_20','price_mom_10','qty_price_mom_50',
    'price_sign_mom_200','event_type_entropy_200','fill_add_restore_100',
    'spread_velocity_50',
    # NEW:
    'price_sign_sq','fade_side_price','restore_x_price'
]

IDX = {n:i for i,n in enumerate(FEAT_NAMES[:15])}

def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f: f.write(msg + '\n')

files = sorted(glob.glob(FEAT_DIR + '*.npz'))
log('Computing features 16-18 | %d files' % len(files))
log('Output dir: %s' % OUT_DIR)
t0 = time.time()

for fi, fpath in enumerate(files):
    fname = os.path.basename(fpath)
    out_path = os.path.join(OUT_DIR, fname)

    if os.path.exists(out_path):
        continue  # skip already done

    try:
        d = np.load(fpath, allow_pickle=True)
        ev = d['events']  # (N, 15)
        N = ev.shape[0]

        pr  = ev[:, IDX['price_rel_ticks']].astype(np.float32)
        ps  = ev[:, IDX['price_sign_mom_200']].astype(np.float32)
        sid = ev[:, IDX['side_id']].astype(np.float32)
        far = ev[:, IDX['fill_add_restore_100']].astype(np.float32)

        f16 = (pr * np.abs(ps)).astype(np.float32)      # price_sign_sq
        f17 = (-sid * pr).astype(np.float32)             # fade_side_price
        f18 = (far * pr).astype(np.float32)              # restore_x_price

        ev18 = np.concatenate([ev, f16[:,None], f17[:,None], f18[:,None]], axis=1)
        assert ev18.shape == (N, 18), 'shape mismatch: %s' % str(ev18.shape)

        # Save all existing keys + updated events
        save_dict = {'events': ev18}
        for k in d.files:
            if k != 'events':
                save_dict[k] = d[k]
        save_dict['feature_names'] = np.array(FEAT_NAMES)

        np.savez_compressed(out_path, **save_dict)

        if (fi+1) % 25 == 0:
            log('  [%d/%d] %.0fs' % (fi+1, len(files), time.time()-t0))

    except Exception as e:
        log('  ERROR %s: %s' % (fname, e))

log('Done in %.0fs. Files in %s' % (time.time()-t0, OUT_DIR))
