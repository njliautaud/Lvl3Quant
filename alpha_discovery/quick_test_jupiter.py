"""Quick test: generate 1 signal file, run 1 Rust sim combo on Jupiter."""
import numpy as np
import subprocess
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SNAP = ROOT / 'data' / 'processed' / 'medium_snapshots_cache'
SIG = ROOT / 'data' / 'processed' / 'signal_predictions'
SIG.mkdir(parents=True, exist_ok=True)
BINARY = ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO = ROOT / 'mbo'

# Generate pressure_imbalance signal for first date
f = sorted(SNAP.glob('*_snapshots.npz'))[0]
date_str = f.stem.replace('_snapshots', '')
data = np.load(str(f))
gf = data['global_features']
mid = data['mid_prices']
signal = gf[:, 20].astype(np.float64)
out = SIG / f'pressure_imbalance_{date_str}.npz'
np.savez_compressed(str(out), predictions=signal, mid_prices=mid)
print(f'Signal for {date_str}: shape={signal.shape}, min={signal.min():.4f}, max={signal.max():.4f}, std={signal.std():.4f}')
print(f'Signals > 0.1: {(np.abs(signal) > 0.1).sum()} ({(np.abs(signal) > 0.1).mean()*100:.1f}%)')
print(f'Signals > 0.3: {(np.abs(signal) > 0.3).sum()} ({(np.abs(signal) > 0.3).mean()*100:.1f}%)')

# Run Rust sim
date_nodash = date_str.replace('-', '')
mbo_file = MBO / f'glbx-mdp3-{date_nodash}.mbo.dbn.zst'
pred_file = str(out)
out_file = '/tmp/test_pressure.json'

if not mbo_file.exists():
    print(f'ERROR: MBO file not found: {mbo_file}')
    exit(1)

for thresh in [0.0, 0.05, 0.1, 0.2, 0.3]:
    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', pred_file,
        '--output', out_file,
        '--signal-threshold', str(thresh),
        '--hold-ms', '10000',
        '--trailing-ticks', '8',
        '--quiet',
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode == 0 and Path(out_file).exists():
        with open(out_file) as f2:
            result = json.load(f2)
        pnl = result.get('total_pnl_dollars', result.get('pnl_dollars', 'N/A'))
        trades = result.get('total_trades', result.get('n_trades', 'N/A'))
        fills = result.get('total_filled', result.get('n_fills', 'N/A'))
        signals = result.get('total_signals', result.get('n_orders_posted', 'N/A'))
        print(f'  thresh={thresh}: PnL=${pnl}, trades={trades}, fills={fills}, signals={signals}')
    else:
        print(f'  thresh={thresh}: ERROR — {r.stderr[:200]}')

print('\nDone! Rust MBO sim working.')
