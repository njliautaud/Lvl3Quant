#!/usr/bin/env python3
"""
Cost-corrected regrade of composition v1 results.
All trades get proper cost: passive entry + exit cost by type.
"""
import json, glob, sys, os
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
import xgboost as xgb

BASE = Path("/home/nick/Lvl3Quant")
FILLSIM_DIR = BASE / "output/extended_oot_validation/fillsim_results"
PRED_DIR = BASE / "output/extended_oot_validation/pred_npzs"
MID_DIR = BASE / "data/derived/mid_price_bars"
FP_DIR = BASE / "output/direct_firstpassage_heads_v1"
OUT_DIR = BASE / "output/firstpassage_pressure_composition_v1"

N_BARS = 234_000
BAR_NS = 100_000_000

PASSIVE_ENTRY_COST = 0.376
MARKET_EXIT_COST = 1.376
PASSIVE_EXIT_COST = 0.376
SL_EXIT_COST = 1.376

FP_FEATURES = ['pred', 'pred_abs', 'pred_sq', 'r1', 'r5', 'r10', 'r50',
               'v10', 'v50', 'v100', 'spread', 'mom20', 'ptrend5']


def compute_rth_open_ns(date_str):
    d = datetime.strptime(date_str, "%Y%m%d")
    midnight_utc = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    month = d.month
    if 3 <= month <= 10:
        rth_open_utc = midnight_utc + timedelta(hours=13, minutes=30)
    else:
        rth_open_utc = midnight_utc + timedelta(hours=14, minutes=30)
    return int(rth_open_utc.timestamp() * 1_000_000_000)


def ns_to_bar(target_ns, rth_open_ns):
    offset_ns = target_ns - rth_open_ns
    bar = int(offset_ns / BAR_NS)
    return max(0, min(bar, N_BARS - 1))


def compute_features_at_bar(preds, mid_prices, bar_idx):
    pred = preds[bar_idx] if bar_idx < len(preds) else 0.0

    def ret(n):
        if bar_idx < n or mid_prices[bar_idx] <= 0 or mid_prices[bar_idx - n] <= 0:
            return 0.0
        return (mid_prices[bar_idx] - mid_prices[bar_idx - n]) / 0.25

    def vol(n):
        if bar_idx < n:
            return 0.0
        window = mid_prices[max(0, bar_idx - n):bar_idx + 1]
        nonzero = window[window > 0]
        if len(nonzero) < 2:
            return 0.0
        diffs = np.diff(nonzero) / 0.25
        return float(np.std(diffs)) if len(diffs) > 0 else 0.0

    def mom(n):
        if bar_idx < n:
            return 0.0
        return float(np.mean(preds[max(0, bar_idx - n):bar_idx + 1]))

    def ptrend(n):
        if bar_idx < n:
            return 0.0
        window = mid_prices[max(0, bar_idx - n):bar_idx + 1]
        nonzero = window[window > 0]
        if len(nonzero) < 2:
            return 0.0
        return float(nonzero[-1] - nonzero[0]) / 0.25

    return np.array([
        pred, abs(pred), pred ** 2,
        ret(1), ret(5), ret(10), ret(50),
        vol(10), vol(50), vol(100),
        1.0,  # spread
        mom(20), ptrend(5)
    ], dtype=np.float32)


def run_costed_sim(date_str, trades, preds, mid_prices, pcfg, fp_model=None, fp_thresh=0.5):
    rth_open_ns = compute_rth_open_ns(date_str)
    fade_thresh = pcfg['fade_thresh']
    fade_n = pcfg['fade_n']
    rev_thresh = pcfg['rev_thresh']
    rev_n = pcfg['rev_n']

    results = []
    gated_in = 0
    gated_out = 0

    for t in trades:
        entry_ns = t['fill_time_ns']
        entry_px = t['entry_price']
        side = t.get('side', 'BUY')
        entry_bar = ns_to_bar(entry_ns, rth_open_ns)

        # First-passage gate
        if fp_model is not None:
            features = compute_features_at_bar(preds, mid_prices, entry_bar)
            if side == 'SELL':
                features[0] = -features[0]
                features[3] = -features[3]
                features[4] = -features[4]
                features[5] = -features[5]
                features[6] = -features[6]
                features[11] = -features[11]
                features[12] = -features[12]
            dmat = xgb.DMatrix(features.reshape(1, -1), feature_names=FP_FEATURES)
            prob = fp_model.predict(dmat)[0]
            if prob < fp_thresh:
                gated_out += 1
                continue
            gated_in += 1
        else:
            gated_in += 1

        exit_ns = t.get('exit_time_ns', entry_ns + 1800_000_000_000)
        exit_bar = ns_to_bar(exit_ns, rth_open_ns)

        if side == 'BUY':
            tp_px = entry_px + 8 * 0.25
            sl_px = entry_px - 16 * 0.25
        else:
            tp_px = entry_px - 8 * 0.25
            sl_px = entry_px + 16 * 0.25

        fade_count = 0
        rev_count = 0
        exit_type = 'original'
        exit_bar_actual = exit_bar

        for bar in range(entry_bar + 1, min(exit_bar + 1, N_BARS)):
            mid = mid_prices[bar]
            if mid <= 0:
                continue

            if side == 'BUY':
                if mid >= tp_px:
                    exit_type = 'tp'
                    exit_bar_actual = bar
                    break
                if mid <= sl_px:
                    exit_type = 'sl'
                    exit_bar_actual = bar
                    break
            else:
                if mid <= tp_px:
                    exit_type = 'tp'
                    exit_bar_actual = bar
                    break
                if mid >= sl_px:
                    exit_type = 'sl'
                    exit_bar_actual = bar
                    break

            pred_val = preds[bar]
            if side == 'SELL':
                pred_val = -pred_val

            if pred_val < fade_thresh:
                fade_count += 1
            else:
                fade_count = 0
            if pred_val < rev_thresh:
                rev_count += 1
            else:
                rev_count = 0

            if (fade_n > 0 and fade_count >= fade_n) or (rev_n > 0 and rev_count >= rev_n):
                exit_type = 'pressure'
                exit_bar_actual = bar
                break

        # Compute PnL with proper costs
        exit_mid = mid_prices[exit_bar_actual] if exit_bar_actual < N_BARS else 0
        if exit_mid <= 0:
            exit_mid = mid_prices[min(exit_bar_actual, N_BARS - 1)]

        if exit_mid > 0 and exit_type in ('pressure', 'tp', 'sl'):
            if side == 'BUY':
                raw_ticks = (exit_mid - entry_px) / 0.25
            else:
                raw_ticks = (entry_px - exit_mid) / 0.25

            entry_cost = PASSIVE_ENTRY_COST
            if exit_type == 'tp':
                exit_cost = PASSIVE_EXIT_COST
            elif exit_type == 'sl':
                exit_cost = SL_EXIT_COST
            else:
                exit_cost = MARKET_EXIT_COST
            total_cost = entry_cost + exit_cost
            net_ticks = raw_ticks - total_cost
        else:
            raw_ticks = t['pnl_ticks']
            entry_cost = PASSIVE_ENTRY_COST
            orig_exit = t.get('exit_reason', 'unknown').lower()
            if 'tp' in orig_exit or 'take' in orig_exit:
                exit_cost = PASSIVE_EXIT_COST
            elif 'sl' in orig_exit or 'stop' in orig_exit:
                exit_cost = SL_EXIT_COST
            else:
                exit_cost = MARKET_EXIT_COST
            total_cost = entry_cost + exit_cost
            net_ticks = raw_ticks - total_cost

        results.append({
            'raw_ticks': raw_ticks,
            'net_ticks': net_ticks,
            'cost': total_cost,
            'exit_type': exit_type,
            'side': side,
        })

    return results, gated_in, gated_out


def evaluate(results, daily_pnl, label):
    if not results:
        return None
    total_net = sum(r['net_ticks'] for r in results)
    total_raw = sum(r['raw_ticks'] for r in results)
    total_cost = sum(r['cost'] for r in results)
    gross_win = sum(r['net_ticks'] for r in results if r['net_ticks'] > 0)
    gross_loss = sum(abs(r['net_ticks']) for r in results if r['net_ticks'] < 0)
    wins = sum(1 for r in results if r['net_ticks'] > 0)
    n_pressure = sum(1 for r in results if r['exit_type'] == 'pressure')
    n_tp = sum(1 for r in results if r['exit_type'] == 'tp')
    n_sl = sum(1 for r in results if r['exit_type'] == 'sl')

    pf = gross_win / gross_loss if gross_loss > 0 else 999
    wr = wins / len(results)

    daily_vals = list(daily_pnl.values())
    n_days = len(daily_vals)
    sharpe = (np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252)
              if n_days > 1 and np.std(daily_vals) > 0 else 0)
    neg = [v for v in daily_vals if v < 0]
    sortino = (np.mean(daily_vals) / np.std(neg) * np.sqrt(252)
               if neg and np.std(neg) > 0 else 0)
    green = sum(1 for v in daily_vals if v > 0)
    red = sum(1 for v in daily_vals if v <= 0)

    march_pnl = [v for k, v in daily_pnl.items() if k.startswith('202603')]
    april_pnl = [v for k, v in daily_pnl.items() if k.startswith('202604')]
    march_sharpe = (np.mean(march_pnl) / np.std(march_pnl) * np.sqrt(252)
                    if len(march_pnl) > 1 and np.std(march_pnl) > 0 else 0)
    april_sharpe = (np.mean(april_pnl) / np.std(april_pnl) * np.sqrt(252)
                    if len(april_pnl) > 1 and np.std(april_pnl) > 0 else 0)
    max_s = max(abs(march_sharpe), abs(april_sharpe))
    regime_asym = abs(march_sharpe - april_sharpe) / max_s if max_s > 0 else 999

    daily_abs = [abs(v) for v in daily_vals]
    total_abs = sum(daily_abs)
    day_conc = max(daily_abs) / total_abs if total_abs > 0 else 1.0
    tpd = len(results) / n_days if n_days > 0 else 0

    return {
        'label': label,
        'n_trades': len(results),
        'n_days': n_days,
        'trades_per_day': round(tpd, 1),
        'raw_ticks': round(total_raw, 1),
        'total_cost': round(total_cost, 1),
        'net_ticks': round(total_net, 1),
        'avg_net': round(total_net / len(results), 3),
        'pf': round(pf, 3),
        'wr': round(wr, 4),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'green': green,
        'red': red,
        'n_tp': n_tp,
        'n_sl': n_sl,
        'n_pressure': n_pressure,
        'march_sharpe': round(march_sharpe, 2),
        'april_sharpe': round(april_sharpe, 2),
        'regime_asym': round(regime_asym, 3),
        'regime_pass': regime_asym <= 0.50,
        'day_conc': round(day_conc, 3),
        'day_conc_pass': day_conc <= 0.70,
    }


def main():
    print("=" * 100)
    print("COST-CORRECTED COMPOSITION v1 REGRADE")
    print("Cost model: passive entry (0.376t) + TP exit (0.376t) / SL exit (1.376t) / pressure exit (1.376t)")
    print("=" * 100)

    # Load data
    all_data = {}
    for f in sorted(glob.glob(str(FILLSIM_DIR / "both_afternoon_*.json"))):
        date_str = Path(f).stem.replace("both_afternoon_", "")
        pred_file = PRED_DIR / f"{date_str}_unfiltered.npz"
        mid_file = MID_DIR / f"{date_str}.npz"
        if not pred_file.exists() or not mid_file.exists():
            continue
        with open(f) as fh:
            d = json.load(fh)
        trades = d.get('trades', [])
        if not trades:
            continue
        preds = np.load(pred_file)['predictions']
        mid_data = np.load(mid_file)
        mid_prices = mid_data['mid_prices'] if 'mid_prices' in mid_data else mid_data[mid_data.files[0]]
        if np.sum(mid_prices > 0) < N_BARS * 0.5:
            continue
        all_data[date_str] = {'trades': trades, 'preds': preds, 'mid_prices': mid_prices}

    total_t = sum(len(d['trades']) for d in all_data.values())
    print(f"\nLoaded {len(all_data)} dates, {total_t} trades")

    # Load FP models
    fp_models = {}
    for cell in ['TP2_SL1', 'TP3_SL1', 'TP4_SL1', 'TP5_SL1', 'TP3_SL2', 'TP4_SL2', 'TP5_SL2']:
        mp = FP_DIR / cell / "model.json"
        if mp.exists():
            m = xgb.Booster()
            m.load_model(str(mp))
            fp_models[cell] = m
    print(f"Loaded {len(fp_models)} FP models\n")

    pressure_configs = [
        {'fade_thresh': -0.3, 'fade_n': 20, 'rev_thresh': -0.5, 'rev_n': 5, 'name': 'best_prior'},
        {'fade_thresh': -0.1, 'fade_n': 160, 'rev_thresh': -0.5, 'rev_n': 8, 'name': 'best_sharpe'},
        {'fade_thresh': 0.0, 'fade_n': 20, 'rev_thresh': -0.3, 'rev_n': 5, 'name': 'tight'},
        {'fade_thresh': -0.3, 'fade_n': 80, 'rev_thresh': -0.5, 'rev_n': 15, 'name': 'loose'},
        {'fade_thresh': -999, 'fade_n': 999999, 'rev_thresh': -999, 'rev_n': 999999, 'name': 'no_pressure'},
    ]

    fp_thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    all_summaries = []

    # Baselines
    print("--- BASELINES (no FP gate, all trades get costs) ---")
    for pcfg in pressure_configs:
        all_res = []
        daily = {}
        for ds, data in sorted(all_data.items()):
            res, _, _ = run_costed_sim(ds, data['trades'], data['preds'], data['mid_prices'], pcfg)
            all_res.extend(res)
            daily[ds] = sum(r['net_ticks'] for r in res)
        m = evaluate(all_res, daily, "NoGate_" + pcfg['name'])
        if m:
            all_summaries.append(m)
            rf = "PASS" if m['regime_pass'] else "FAIL"
            tp_sl_pr = str(m['n_tp']) + "/" + str(m['n_sl']) + "/" + str(m['n_pressure'])
            print("  " + pcfg['name'].ljust(15) + ": PF=" + str(m['pf'])
                  + " WR=" + str(round(m['wr'] * 100, 1)) + "%"
                  + " Sharpe=" + str(m['sharpe'])
                  + " Net=" + str(m['net_ticks']) + "t"
                  + " (raw=" + str(m['raw_ticks']) + " cost=" + str(m['total_cost']) + ")"
                  + " TP/SL/Pr=" + tp_sl_pr
                  + " Regime=" + rf)

    # FP-gated
    print("\n--- FP-GATED + PRESSURE (cost-corrected) ---")
    for cell in ['TP5_SL2', 'TP5_SL1', 'TP4_SL2', 'TP4_SL1', 'TP3_SL2', 'TP2_SL1', 'TP3_SL1']:
        if cell not in fp_models:
            continue
        fp_model = fp_models[cell]
        print("\n  Gate: " + cell)
        for thresh in fp_thresholds:
            for pcfg in pressure_configs:
                all_res = []
                daily = {}
                gi_total = 0
                go_total = 0
                for ds, data in sorted(all_data.items()):
                    res, gi, go = run_costed_sim(
                        ds, data['trades'], data['preds'], data['mid_prices'],
                        pcfg, fp_model=fp_model, fp_thresh=thresh
                    )
                    all_res.extend(res)
                    gi_total += gi
                    go_total += go
                    daily[ds] = sum(r['net_ticks'] for r in res)

                label = cell + "_t" + str(thresh) + "_" + pcfg['name']
                m = evaluate(all_res, daily, label)
                if m:
                    passrate = gi_total / (gi_total + go_total) * 100 if (gi_total + go_total) > 0 else 0
                    m['gate_pass_rate'] = round(passrate, 1)
                    all_summaries.append(m)

                    if m['net_ticks'] > 0 and m['n_trades'] >= 30:
                        rf = "PASS" if m['regime_pass'] else "FAIL"
                        tp_sl_pr = str(m['n_tp']) + "/" + str(m['n_sl']) + "/" + str(m['n_pressure'])
                        print("    t=" + str(thresh) + " " + pcfg['name'].ljust(15)
                              + ": PF=" + str(m['pf'])
                              + " WR=" + str(round(m['wr'] * 100, 1)) + "%"
                              + " Sharpe=" + str(m['sharpe'])
                              + " Net=" + str(m['net_ticks']) + "t"
                              + " (raw=" + str(m['raw_ticks']) + " cost=" + str(m['total_cost']) + ")"
                              + " Mar/Apr=" + str(m['march_sharpe']) + "/" + str(m['april_sharpe'])
                              + " Regime=" + rf
                              + " pass=" + str(round(passrate)) + "%")

    # Acceptance test
    print("\n" + "=" * 100)
    print("HC #506 R5 ACCEPTANCE (COST-CORRECTED)")
    print("Criteria: net>0, PF>=1.2, Sharpe>=0.5, regime<=0.50, conc<=0.70, trades/day>=5")
    print("=" * 100)

    accepted = [r for r in all_summaries
                if r['net_ticks'] > 0 and r['pf'] >= 1.2
                and r['sharpe'] >= 0.5 and r['regime_pass']
                and r['day_conc_pass'] and r['trades_per_day'] >= 5]

    if accepted:
        accepted.sort(key=lambda x: x['sharpe'], reverse=True)
        print("\n  GREEN: " + str(len(accepted)) + " configs PASS:")
        for r in accepted[:15]:
            print("    " + r['label'].ljust(45)
                  + " PF=" + str(r['pf'])
                  + " Sharpe=" + str(r['sharpe'])
                  + " Sortino=" + str(r['sortino'])
                  + " Net=" + str(r['net_ticks']) + "t"
                  + " (raw=" + str(r['raw_ticks']) + " cost=" + str(r['total_cost']) + ")"
                  + " n=" + str(r['n_trades']) + " (" + str(r['trades_per_day']) + "/day)"
                  + " Mar/Apr=" + str(r['march_sharpe']) + "/" + str(r['april_sharpe']))
    else:
        print("\n  RED: NO configs pass full acceptance after costs.")
        near = sorted([r for r in all_summaries if r['net_ticks'] > 0 and r['pf'] >= 1.0],
                      key=lambda x: x['sharpe'], reverse=True)
        if near:
            print("  Top " + str(min(10, len(near))) + " near-misses (net>0, PF>=1.0):")
            for r in near[:10]:
                fails = []
                if r['pf'] < 1.2:
                    fails.append("PF=" + str(r['pf']))
                if r['sharpe'] < 0.5:
                    fails.append("Sharpe=" + str(r['sharpe']))
                if not r['regime_pass']:
                    fails.append("regime=" + str(r['regime_asym']))
                if not r['day_conc_pass']:
                    fails.append("conc=" + str(r['day_conc']))
                if r['trades_per_day'] < 5:
                    fails.append("freq=" + str(r['trades_per_day']))
                print("    " + r['label'].ljust(45)
                      + " PF=" + str(r['pf'])
                      + " Sharpe=" + str(r['sharpe'])
                      + " Net=" + str(r['net_ticks']) + "t"
                      + " n=" + str(r['n_trades'])
                      + " FAIL: " + ", ".join(fails))

    # Top 20
    print("\n" + "=" * 100)
    print("TOP 20 BY SHARPE (net>0, n>=30, COST-CORRECTED)")
    print("=" * 100)
    top = sorted([r for r in all_summaries if r['net_ticks'] > 0 and r['n_trades'] >= 30],
                 key=lambda x: x['sharpe'], reverse=True)
    for i, r in enumerate(top[:20]):
        rf = "PASS" if r['regime_pass'] else "FAIL"
        print("  " + str(i + 1).rjust(2) + ". " + r['label'].ljust(45)
              + " PF=" + str(r['pf'])
              + " WR=" + str(round(r['wr'] * 100, 1)) + "%"
              + " Sharpe=" + str(r['sharpe'])
              + " Net=" + str(r['net_ticks']) + "t"
              + " (raw=" + str(r['raw_ticks']) + " cost=" + str(r['total_cost']) + ")"
              + " n=" + str(r['n_trades'])
              + " Mar/Apr=" + str(r['march_sharpe']) + "/" + str(r['april_sharpe'])
              + " Regime=" + rf)

    # Save
    save_path = OUT_DIR / "composition_cost_corrected.json"
    with open(save_path, 'w') as f:
        json.dump(all_summaries, f, indent=2, default=str)
    n_profitable = sum(1 for r in all_summaries if r['net_ticks'] > 0)
    n_accepted = len(accepted) if accepted else 0
    print("\nSaved " + str(len(all_summaries)) + " results to " + str(save_path))
    print("Total: " + str(len(all_summaries)) + " Profitable: " + str(n_profitable) + " Accepted: " + str(n_accepted))


if __name__ == '__main__':
    main()
