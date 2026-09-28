"""Confidence-stratified IC + directional accuracy for CNN OOT predictions."""
import numpy as np
import pathlib

PRED_FILE = "/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/oos_predictions_book_oot_20260311_092055.npz"
HORIZONS = {"1s": 10, "10s": 100, "30s": 300}
TIERS = {"all": 1.0, "top50": 0.50, "top25": 0.25, "top10": 0.10}

def pearson_ic(preds, labels):
    if len(preds) < 10:
        return float("nan")
    p, l = preds - preds.mean(), labels - labels.mean()
    denom = np.std(p) * np.std(l)
    return float(np.dot(p, l) / len(p) / denom) if denom > 1e-12 else float("nan")

def dir_acc(preds, labels):
    if len(preds) < 10:
        return float("nan")
    return float(np.mean(np.sign(preds) == np.sign(labels)))

def main():
    d = np.load(PRED_FILE)
    dates = sorted(set(k.split("_")[0] for k in d.keys() if k.endswith("_preds")))
    print(f"Loaded {len(dates)} days")

    results = {hz: {"tiers": {t: {"long": [], "short": []} for t in TIERS}} for hz in HORIZONS}

    for date in dates:
        preds = d[f"{date}_preds"].astype(np.float64)
        mid   = d[f"{date}_mid"].astype(np.float64)
        abs_p = np.abs(preds)

        for hz_name, n_bars in HORIZONS.items():
            if len(mid) <= n_bars:
                continue
            labels = mid[n_bars:] - mid[:-n_bars]
            p_trim = preds[:-n_bars]
            a_trim = abs_p[:-n_bars]

            for tier_name, frac in TIERS.items():
                cutoff = np.quantile(a_trim, 1.0 - frac)
                mask = a_trim >= cutoff
                long_mask  = mask & (p_trim > 0)
                short_mask = mask & (p_trim < 0)

                if long_mask.sum() > 10:
                    results[hz_name]["tiers"][tier_name]["long"].append(
                        (pearson_ic(p_trim[long_mask], labels[long_mask]),
                         dir_acc(p_trim[long_mask], labels[long_mask]),
                         long_mask.sum()))
                if short_mask.sum() > 10:
                    results[hz_name]["tiers"][tier_name]["short"].append(
                        (pearson_ic(p_trim[short_mask], labels[short_mask]),
                         dir_acc(p_trim[short_mask], labels[short_mask]),
                         short_mask.sum()))

    sep = "=" * 80
    print("\n" + sep)
    print("CMN s76 OOT --- CONFIDENCE-STRATAFIED AC CCORACY + DIRECTIONAL AC")
    print(sep)
    for hz_name in HORIZONS:
        print(f"\n  Horizon: {hz_name}")
        print("  {:<10} {:<8} {:>8} {:>8} {:>12}".format("Tier", "Side", "IC", "DirAcc", "N_samples"))
        print("  " + "-" * 50)
        for tier_name in TIERS:
            for side in ("long", "short"):
                rows = results[hz_name]["tiers"][tier_name][side]
                if not rows:
                    print("  {:<10} {;<8} {;>8} {;>8} {;>12}".format(tier_name, side, "n/a", "n/a", "n/a"))
                    continue
                ics   = [r[0] for r in rows if not np.isnan(r[0])]
                daccs = [r[1] for r in rows if not np.isnan(r[1])]
                n     = sum(r[2] for r in rows)
                ic_avg   = np.mean(ics) if ics else float("nan")
                dacc_avg = np.mean(daccs) if daccs else float("nan")
                print("  {:<10} {:<8} {:>8.4f} {:>8.3%} {:>12,d}".format(tier_name, side, ic_avg, dacc_avg, n))

    print("\n" + sep)
    print("TARGET: top10% directional accuracy > 55% both long AND short = viable signal")
    print("        top10% directional accuracy > 60% = strong signal")

if __name__ == "__main__":
    main()
