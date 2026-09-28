"""Confidence-stratified IC + directional accuracy for CNN OOT predictions.
v2: filters zero-label samples so sign(0)=0 doesn't corrupt directional accuracy.
"""
import numpy as np

PRED_FILE = "/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/oos_predictions_book_oot_20260311_092055.npz"
HORIZONS = {"1s": 10, "10s": 100, "30s": 300}
TIERS = {"all": 1.0, "top50": 0.50, "top25": 0.25, "top10": 0.10}

def pearson_ic(preds, labels):
    nz = labels != 0
    p, l = preds[nz], labels[nz]
    if len(p) < 10: return float("nan")
    p2, l2 = p - p.mean(), l - l.mean()
    denom = np.std(p2) * np.std(l2)
    return float(np.dot(p2, l2) / len(p) / denom) if denom > 1e-12 else float("nan")

def dir_acc(preds, labels):
    nz = labels != 0
    p, l = preds[nz], labels[nz]
    if len(p) < 10: return float("nan"), 0.0
    return float(np.mean(np.sign(p) == np.sign(l))), float(np.mean(~nz))

def main():
    d = np.load(PRED_FILE)
    dates = sorted(set(k.split("_")[0] for k in d.keys() if k.endswith("_preds")))
    print(f"Loaded {len(dates)} days")

    # Storage: per tier per side: list of (ic, dacc, zero_frac, n)
    R = {hz: {t: {"long": [], "short": []} for t in TIERS} for hz in HORIZONS}

    for date in dates:
        preds = d[f"{date}_preds"].astype(np.float64)
        mid   = d[f"{date}_mid"].astype(np.float64)
        abs_p = np.abs(preds)

        for hz_name, n_bars in HORIZONS.items():
            if len(mid) <= n_bars: continue
            labels  = mid[n_bars:] - mid[:-n_bars]
            p_trim  = preds[:-n_bars]
            a_trim  = abs_p[:-n_bars]

            for tier_name, frac in TIERS.items():
                cutoff = np.quantile(a_trim, 1.0 - frac)
                mask   = a_trim >= cutoff
                for side, sign_mask in [("long", p_trim > 0), ("short", p_trim < 0)]:
                    m = mask & sign_mask
                    if m.sum() < 10: continue
                    ic = pearson_ic(p_trim[m], labels[m])
                    da, zf = dir_acc(p_trim[m], labels[m])
                    R[hz_name][tier_name][side].append((ic, da, zf, m.sum()))

    print("\n" + "="*90)
    print("CNN s76 OOT — CONFIDENCE-STRATIFIED IC + DIRECTIONAL ACCURACY (zero-label filtered)")
    print("="*90)
    for hz_name in HORIZONS:
        print(f"\n  Horizon: {hz_name}")
        print(f"  {f'Tier':<10} {f'Side':<8} {f'IC':>8} {f'DirAcc':>8} {f'ZeroFrac':>10} {f'N':>12}")
        print(f"  {f'-'*60}")
        for tier_name in TIERS:
            for side in ("long", "short"):
                rows = R[hz_name][tier_name][side]
                if not rows:
                    print(f"  {tier_name:<10} {side:<8} {'n/a':>8} {'n/a':>8} {'n/a':>10} {'n/a':>12}")
                    continue
                ics  = [r[0] for r in rows if r[0] == r[0]]
                dacs = [r[1] for r in rows if r[1] == r[1]]
                zfs  = [r[2] for r in rows]
                n    = sum(r[3] for r in rows)
                print(f"  {tier_name:<10} {side:<8} {np.mean(ics):>8.4f} {np.mean(dacs):>8.3%} {np.mean(zfs):>10.3%} {n:>12,}")
    print("\n" + "="*90)
    print("TARGET: top10% DirAcc > 55% both long AND short = VIABLE | >60% = STRONG")

if __name__ == "__main__":
    main()
