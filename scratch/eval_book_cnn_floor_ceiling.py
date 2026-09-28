"""Evaluate Book CNN saved predictions against FLOOR/CEILING criteria.

NPZ structure: predictions(N,3), labels(N,3), horizons=['1s','5s','10s'], oot_files, embeddings.
Floor: concat IC>=+0.05 per horizon, no inversions (DA>=50%) at top tiers.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

PREDS_DIR = Path("/tmp/book_cnn_preds")
TIERS = [0.005, 0.01, 0.05, 0.10, 1.00]


def per_horizon_metrics(y_true, y_pred):
    if y_true.size == 0:
        return None
    ic_full, _ = spearmanr(y_pred, y_true)
    abs_pred = np.abs(y_pred)
    out = {"n": int(y_true.size), "ic_full": float(ic_full)}
    for tier in TIERS:
        k = max(1, int(round(tier * y_true.size)))
        idx = np.argpartition(-abs_pred, k - 1)[:k]
        yt, yp = y_true[idx], y_pred[idx]
        ic_tier = spearmanr(yp, yt).correlation if yt.size > 1 else float("nan")
        da = float(np.mean(np.sign(yp) == np.sign(yt))) if yt.size > 0 else float("nan")
        out[f"tier_{tier:.3f}"] = {
            "n": int(k),
            "ic": None if np.isnan(ic_tier) else float(ic_tier),
            "da": None if np.isnan(da) else da,
        }
    return out


def main():
    folds = sorted(PREDS_DIR.glob("fold_*_oot_predictions.npz"))
    if not folds:
        print(f"No predictions found in {PREDS_DIR}")
        return
    results = {}
    print(f"{'fold':<6}{'h':<5}{'n':<8}{'ic_full':<11}"
          f"{'t.5%_da':<10}{'t1%_da':<10}{'t5%_da':<10}{'tALL_da':<10}")
    print("-" * 70)
    for fp in folds:
        z = np.load(fp)
        horizons = [str(h) for h in z["horizons"].tolist()]
        preds = z["predictions"]
        labels = z["labels"]
        fold_id = fp.stem.split("_")[1]
        per_h = {}
        for i, h in enumerate(horizons):
            m = per_horizon_metrics(labels[:, i], preds[:, i])
            per_h[h] = m
            if m:
                t05 = m.get("tier_0.005", {})
                t1 = m.get("tier_0.010", {})
                t5 = m.get("tier_0.050", {})
                tA = m.get("tier_1.000", {})
                print(
                    f"{fold_id:<6}{h:<5}{m['n']:<8}{m['ic_full']:+.4f}    "
                    f"{(t05.get('da') or 0):.3f}     "
                    f"{(t1.get('da') or 0):.3f}     "
                    f"{(t5.get('da') or 0):.3f}     "
                    f"{(tA.get('da') or 0):.3f}"
                )
        results[fold_id] = {
            "horizons": per_h,
            "oot_files": [str(f) for f in z["oot_files"].tolist()],
        }

    out_path = Path("/home/jupiter/Lvl3Quant/scratch/book_cnn_floor_ceiling_eval.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote per-fold detail -> {out_path}")

    # Concat IC across folds (primary metric per directive)
    print("\n=== CONCAT IC ACROSS ALL FOLDS (primary metric) ===")
    print(f"{'horizon':<10}{'concat_ic':<12}{'t1%_ic':<11}{'t1%_da':<10}{'t5%_ic':<11}{'t5%_da':<10}{'n':<10}")
    print("-" * 75)
    summary = {}
    for h in ["1s", "5s", "10s"]:
        all_t, all_p = [], []
        for fp in folds:
            z = np.load(fp)
            horizons = [str(x) for x in z["horizons"].tolist()]
            if h not in horizons:
                continue
            i = horizons.index(h)
            all_t.append(z["labels"][:, i])
            all_p.append(z["predictions"][:, i])
        if not all_t:
            continue
        t = np.concatenate(all_t)
        p = np.concatenate(all_p)
        ic, _ = spearmanr(p, t)

        def tier(k_frac):
            k = max(1, int(round(k_frac * t.size)))
            idx = np.argpartition(-np.abs(p), k - 1)[:k]
            ic_t = spearmanr(p[idx], t[idx]).correlation if t[idx].size > 1 else float("nan")
            da_t = float(np.mean(np.sign(p[idx]) == np.sign(t[idx])))
            return ic_t, da_t

        ic_t1, da_t1 = tier(0.01)
        ic_t5, da_t5 = tier(0.05)
        print(
            f"{h:<10}{ic:+.4f}     "
            f"{ic_t1:+.4f}    {da_t1:.3f}     "
            f"{ic_t5:+.4f}    {da_t5:.3f}     {t.size}"
        )
        summary[h] = {
            "concat_ic": float(ic),
            "tier1_ic": float(ic_t1),
            "tier1_da": float(da_t1),
            "tier5_ic": float(ic_t5),
            "tier5_da": float(da_t5),
            "n": int(t.size),
        }

    # FLOOR check
    print("\n=== FLOOR CHECK (per DIRECTIVES 19:38 ET) ===")
    print("FLOOR: concat IC >= +0.05 per horizon, NO inversions (DA >= 0.50) at top tiers")
    floor_ok = True
    for h, s in summary.items():
        ic_ok = s["concat_ic"] >= 0.05
        no_inv = s["tier1_da"] >= 0.50 and s["tier5_da"] >= 0.50
        verdict = "PASS" if (ic_ok and no_inv) else "FAIL"
        print(f"  {h}: concat_ic={s['concat_ic']:+.4f} (>=0.05? {ic_ok})  "
              f"tier1_da={s['tier1_da']:.3f}  tier5_da={s['tier5_da']:.3f}  -> {verdict}")
        if not (ic_ok and no_inv):
            floor_ok = False
    print(f"\nOVERALL FLOOR: {'PASS' if floor_ok else 'FAIL'}")
    summary["floor_pass"] = floor_ok
    Path("/home/jupiter/Lvl3Quant/scratch/book_cnn_concat_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("Wrote concat summary -> /home/jupiter/Lvl3Quant/scratch/book_cnn_concat_summary.json")


if __name__ == "__main__":
    main()
