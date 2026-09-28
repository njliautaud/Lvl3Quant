import numpy as np
import os
import glob
import time
from scipy.stats import spearmanr

FEAT_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_events_feat/'
STRIDE = 500  # faster pass, still ~4M samples
HORIZONS = ['labels_1s', 'labels_5s', 'labels_10s']
LOG = '/home/jupiter/Lvl3Quant/feature_deriv_run.log'

BASE_NAMES = [
    'time_delta_log','event_type_id','side_id','price_rel_ticks',
    'qty_log','spread_ticks','cancel_side_asym_50','rolling_ofi_500',
    'event_density_20','price_mom_10','qty_price_mom_50',
    'price_sign_mom_200','event_type_entropy_200','fill_add_restore_100',
    'spread_velocity_50'
]
IDX = {n:i for i,n in enumerate(BASE_NAMES)}

def log(msg):
    print(msg, flush=True)
    with open(LOG,'a') as f: f.write(msg+'\n')

files = sorted(glob.glob(FEAT_DIR + '*.npz'))
log('Feature Derivatives Discovery')
log('Files: %d | stride=%d' % (len(files), STRIDE))
t0 = time.time()

# Accumulation buffers for base features we need
base_bufs = {n:[] for n in ['price_rel_ticks','price_sign_mom_200','price_mom_10',
                              'qty_price_mom_50','cancel_side_asym_50','rolling_ofi_500',
                              'side_id','spread_ticks','event_density_20','qty_price_mom_50',
                              'fill_add_restore_100']}
lbl_bufs = {h:[] for h in HORIZONS}

for fi, fpath in enumerate(files):
    try:
        d = np.load(fpath, allow_pickle=True)
        ev = d['events']
        N = ev.shape[0]
        idx_s = np.arange(0, N, STRIDE)
        sub = ev[idx_s]
        for n in base_bufs:
            base_bufs[n].append(sub[:, IDX[n]].astype(np.float32))
        for h in HORIZONS:
            lbl_bufs[h].append(d[h][idx_s].astype(np.float32))
        if (fi+1) % 50 == 0:
            log('  [%d/%d] %.0fs' % (fi+1, len(files), time.time()-t0))
    except Exception as e:
        log('  SKIP %s: %s' % (os.path.basename(fpath), e))

log('Loaded in %.0fs. Building derived features...' % (time.time()-t0))

# Concatenate base arrays
B = {n: np.concatenate(base_bufs[n]) for n in base_bufs}
L = {h: np.concatenate(lbl_bufs[h]) for h in HORIZONS}
N_tot = len(B['price_rel_ticks'])
log('Total samples: {:,}'.format(N_tot))

# Derived features: (name, array)
def safe_div(a, b, eps=0.5): return a / (np.abs(b) + eps)

pr = B['price_rel_ticks']
ps = B['price_sign_mom_200']
pm = B['price_mom_10']
qp = B['qty_price_mom_50']
ca = B['cancel_side_asym_50']
ofi = B['rolling_ofi_500']
sid = B['side_id']
sp = B['spread_ticks']
ed = B['event_density_20']
far = B['fill_add_restore_100']

derived = [
    # Momentum alignment / amplification
    ('mom_align_10_200',      pm * ps),
    ('mom_qty_align',         pm * qp),
    ('price_mom_sq',          pr * pm),
    ('price_sign_sq',         pr * np.abs(ps)),
    # OFI contrarian combinations
    ('ofi_fade_price',        -ofi * pr),
    ('ofi_fade_sign',         -ofi * ps),
    ('ofi_vs_mom',            -ofi * pm),
    # Cancel asymmetry combinations
    ('cancel_x_price',        ca * pr),
    ('cancel_x_sign',         ca * ps),
    ('cancel_x_mom',          ca * pm),
    # Fade aggressor
    ('fade_side_price',       -sid * pr),
    ('fade_side_sign',        -sid * ps),
    # Regime gating (spread normalization)
    ('norm_price',            safe_div(pr, sp)),
    ('norm_mom',              safe_div(pm, sp)),
    # Liquidity restoration signal
    ('restore_x_price',       far * pr),
    ('restore_x_sign',        far * ps),
    # Event density amplification
    ('density_x_mom',         ed * pm),
    ('density_x_price',       ed * pr),
    # Momentum divergence (short vs long)
    ('mom_diverge',           pm - ps),
    ('mom_convergence',       pm * ps * (np.sign(pm) == np.sign(ps)).astype(np.float32)),
    # Nonlinear: squared momentum preserving sign
    ('pm_nonlinear',          np.sign(pm) * pm**2),
    ('ps_nonlinear',          np.sign(ps) * ps**2),
    ('pr_nonlinear',          np.sign(pr) * pr**2),
    # OFI divergence from momentum (fade OFI when momentum disagrees)
    ('ofi_mom_diverge',       -ofi * (np.sign(ofi) != np.sign(pm)).astype(np.float32) * np.abs(pm)),
    # Triple combos
    ('triple_mom',            pr * pm * ps),
    ('cancel_ofi_mom',        ca * (-ofi) * pm),
    ('cancel_price_sign',     ca * pr * np.sign(ps)),
    # Regime-conditioned: high event density
    ('hi_density_mom',        pm * (ed > np.median(ed)).astype(np.float32)),
    ('lo_density_price',      pr * (ed < np.median(ed)).astype(np.float32)),
    # Spread regime
    ('tight_spread_price',    pr * (sp < np.median(sp)).astype(np.float32)),
    ('wide_spread_mom',       pm * (sp > np.median(sp)).astype(np.float32)),
]

log('\nComputing IC for %d derived features...' % len(derived))
log('\n%-30s  IC_1s    IC_5s   IC_10s' % 'Feature')
log('-' * 65)

results = []
for name, arr in derived:
    arr = arr.astype(np.float32)
    ics = []
    for h in HORIZONS:
        lab = L[h]
        mask = np.isfinite(arr) & np.isfinite(lab)
        if mask.sum() < 1000:
            ics.append(0.0)
            continue
        r, _ = spearmanr(arr[mask], lab[mask])
        ics.append(float(r) if np.isfinite(r) else 0.0)
    results.append((name, ics[0], ics[1], ics[2]))
    log('%-30s  %6.4f   %6.4f   %6.4f' % (name, ics[0], ics[1], ics[2]))

# Sort by |IC_10s|
results.sort(key=lambda x: abs(x[3]), reverse=True)
log('\n=== TOP 10 by |IC_10s| ===')
for r in results[:10]:
    log('%-30s  %6.4f   %6.4f   %6.4f' % r)

log('\nDone in %.0fs' % (time.time()-t0))

# Save results
with open('/home/jupiter/Lvl3Quant/feature_deriv_results.txt','w') as f:
    f.write('Feature,IC_1s,IC_5s,IC_10s\n')
    for r in results:
        f.write('%s,%.6f,%.6f,%.6f\n' % r)
log('Saved -> /home/jupiter/Lvl3Quant/feature_deriv_results.txt')
