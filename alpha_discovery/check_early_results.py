"""Quick check of early sweep results on Jupiter."""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

results_dir = Path(__file__).parent.parent / 'alpha_discovery' / 'results'
files = sorted(results_dir.glob('sig_*.json'))
print(f'Result files: {len(files)}')

combo_pnl = defaultdict(list)
for f in files:
    try:
        with open(f) as fh:
            data = json.load(fh)
        name = f.stem  # sig_{combo_id}_{date}
        # Remove 'sig_' prefix and date suffix
        parts = name[4:]  # remove 'sig_'
        # Date is last 10 chars (YYYY-MM-DD)
        combo_id = parts[:-11]  # remove _YYYY-MM-DD
        pnl = data.get('total_pnl_dollars', data.get('pnl_dollars', 0))
        combo_pnl[combo_id].append(pnl)
    except:
        pass

ranked = []
for combo, pnls in combo_pnl.items():
    if len(pnls) >= 3:
        total = sum(pnls)
        avg = np.mean(pnls)
        pos = sum(1 for p in pnls if p > 0)
        ranked.append((combo, total, avg, pos, len(pnls)))

ranked.sort(key=lambda x: x[2], reverse=True)  # Sort by avg PnL

print(f'Combos with 3+ days: {len(ranked)}')
print(f'\nTOP 15 (by avg daily PnL):')
for combo, total, avg, pos, days in ranked[:15]:
    wr = pos / max(days, 1)
    print(f'  {combo:50s}  total=${total:>+9,.2f}  avg=${avg:>+8,.2f}/day  WR={wr:.0%}  days={days}')

print(f'\nBOTTOM 5:')
for combo, total, avg, pos, days in ranked[-5:]:
    wr = pos / max(days, 1)
    print(f'  {combo:50s}  total=${total:>+9,.2f}  avg=${avg:>+8,.2f}/day  WR={wr:.0%}  days={days}')

# Count profitable combos
profitable = sum(1 for r in ranked if r[2] > 0)
print(f'\nProfitable (avg PnL > 0): {profitable}/{len(ranked)}')
