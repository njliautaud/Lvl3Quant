#!/usr/bin/env python3
"""
Streaming Trade Manager v3 — HC #507 Compliant
===============================================
Key innovation: VARIABLE TP with passive-limit exits.
- TP exit = passive limit → 0.752t RT cost (cheap!)
- SL/pressure exit = market → 1.752t RT cost (expensive)
- So we WANT most trades to hit TP. Optimize TP level for best net after costs.

Architecture:
1. Entry: CNN-Mamba signal threshold + optional FP gate
2. Trade management: streaming predictions decide hold/exit
3. Exit options:
   a. Passive TP hit → cheapest exit (0.752t RT)
   b. Streaming pressure fade → market exit (1.752t RT)
   c. Stop loss hit → market exit (1.752t RT)
   d. Max hold time → market exit (1.752t RT)

Tests: TP ∈ {2,3,4,5,6,8}, SL ∈ {4,8,12,16}, max_hold ∈ {30s,60s,120s,300s}
With and without FP gate, with and without pressure exit.
On canonical OOT [20260227..20260429], both sides, afternoon.
"""
import json, glob, sys
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
import xgboost as xgb

BASE = Path("/home/nick/Lvl3Quant")
FILLSIM_DIR = BASE / "output/extended_oot_validation/fillsim_results"
PRED_DIR = BASE / "output/extended_oot_validation/pred_npzs"
MID_DIR = BASE / "data/derived/mid_price_bars"
FP_DIR = BASE / "output/direct_firstpassage_heads_v1"
OUT_DIR = BASE / "output/streaming_trade_manager_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BARS = 234_000
BAR_NS = 100_000_000  # 100ms per bar

# Costs
PASSIVE_RT = 0.752   # passive entry + passive exit (both limits filled)
MARKET_RT = 1.752    # passive entry + market exit (crossing spread)

FP_FEATURES = ['pred', 'pred_abs', 'pred_sq', 'r1', 'r5', 'r10', 'r50',
               'v10', 'v50', 'v100', 'spread', 'mom20', 'ptrend5']


def compute_rth_open_ns(date_str):
    d = datetime.strptime(date_str, "%Y%m%d")
    midnight_utc = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    if 3 <= d.month <= 10:
        rth_open_utc = midnight_utc + timedelta(hours=13, minutes=30)
    else:
        rth_open_utc = midnight_utc + timedelta(hours=14, minutes=30)
    return int(rth_open_utc.timestamp() * 1_000_000_000)


def ns_to_bar(target_ns, rth_open_ns):
    bar = int((target_ns - rth_open_ns) / BAR_NS)
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
        w = mid_prices[max(0, bar_idx - n):bar_idx + 1]
        nz = w[w > 0]
        if len(nz) < 2:
            return 0.0
        return float(np.std(np.diff(nz) / 0.25))

    def mom(n):
        if bar_idx < n:
            return 0.0
        return float(np.mean(preds[max(0, bar_idx - n):bar_idx + 1]))

    def ptrend(n):
        if bar_idx < n:
            return 0.0
        w = mid_prices[max(0, bar_idx - n):bar_idx + 1]
        nz = w[w > 0]
        if len(nz) < 2:
            return 0.0
        return float(nz[-1] - nz[0]) / 0.25

    return np.array([
        pred, abs(pred), pred ** 2,
        ret(1), ret(5), ret(10), ret(50),
        vol(10), vol(50), vol(100),
        1.0, mom(20), ptrend(5)
    ], dtype=np.float32)


def simulate_trades(trades, preds, mid_prices, rth_open_ns, cfg, fp_model=None):
    """Simulate trades with streaming prediction management."""
    tp_ticks = cfg['tp']
    sl_ticks = cfg['sl']
    max_hold_bars = cfg['max_hold_bars']
    use_pressure = cfg['use_pressure']
    fade_thresh = cfg.get('fade_thresh', 0.0)
    fade_n = cfg.get('fade_n', 20)
    fp_thresh = cfg.get('fp_thresh', 0.0)

    results = []

    for t in trades:
        entry_ns = t['fill_time_ns']
        entry_px = t['entry_price']
        side = t.get('side', 'BUY')
        entry_bar = ns_to_bar(entry_ns, rth_open_ns)

        # FP gate
        if fp_model is not None and fp_thresh > 0:
            features = compute_features_at_bar(preds, mid_prices, entry_bar)
            if side == 'SELL':
                for idx in [0, 3, 4, 5, 6, 11, 12]:
                    features[idx] = -features[idx]
            dmat = xgb.DMatrix(features.reshape(1, -1), feature_names=FP_FEATURES)
            prob = fp_model.predict(dmat)[0]
            if prob < fp_thresh:
                continue

        # Walk forward from entry
        max_bar = min(entry_bar + max_hold_bars, N_BARS)

        if side == 'BUY':
            tp_px = entry_px + tp_ticks * 0.25
            sl_px = entry_px - sl_ticks * 0.25
        else:
            tp_px = entry_px - tp_ticks * 0.25
            sl_px = entry_px + sl_ticks * 0.25

        fade_count = 0
        exit_type = 'timeout'
        exit_bar = max_bar - 1
        exit_px = 0

        for bar in range(entry_bar + 1, max_bar):
            mid = mid_prices[bar]
            if mid <= 0:
                continue

            # TP check (passive limit — cheapest exit)
            if side == 'BUY' and mid >= tp_px:
                exit_type = 'tp'
                exit_bar = bar
                exit_px = tp_px  # filled at limit price
                break
            elif side == 'SELL' and mid <= tp_px:
                exit_type = 'tp'
                exit_bar = bar
                exit_px = tp_px
                break

            # SL check (stop-market — expensive)
            if side == 'BUY' and mid <= sl_px:
                exit_type = 'sl'
                exit_bar = bar
                exit_px = mid  # market fill at current mid
                break
            elif side == 'SELL' and mid >= sl_px:
                exit_type = 'sl'
                exit_bar = bar
                exit_px = mid
                break

            # Pressure exit (market — expensive)
            if use_pressure:
                pred_val = preds[bar]
                if side == 'SELL':
                    pred_val = -pred_val

                if pred_val < fade_thresh:
                    fade_count += 1
                else:
                    fade_count = 0

                if fade_count >= fade_n:
                    exit_type = 'pressure'
                    exit_bar = bar
                    exit_px = mid
                    break

        # If timeout, exit at market
        if exit_type == 'timeout':
            exit_px = mid_prices[exit_bar] if mid_prices[exit_bar] > 0 else entry_px

        # Compute PnL
        if exit_px <= 0:
            continue

        if exit_type == 'tp':
            # TP hit — passive limit fill at tp_px
            if side == 'BUY':
                raw_ticks = (tp_px - entry_px) / 0.25
            else:
                raw_ticks = (entry_px - tp_px) / 0.25
            cost = PASSIVE_RT
        else:
            # SL, pressure, or timeout — market exit
            if side == 'BUY':
                raw_ticks = (exit_px - entry_px) / 0.25
            else:
                raw_ticks = (entry_px - exit_px) / 0.25
            cost = MARKET_RT

        net_ticks = raw_ticks - cost
        hold_seconds = (exit_bar - entry_bar) * 0.1

        results.append({
            'net_ticks': net_ticks,
            'raw_ticks': raw_ticks,
            'cost': cost,
            'exit_type': exit_type,
            'side': side,
            'hold_seconds': hold_seconds,
        })

    return results


def evaluate(results, daily_pnl, label):
    if not results:
        return None

    total_net = sum(r['net_ticks'] for r in results)
    gross_win = sum(r['net_ticks'] for r in results if r['net_ticks'] > 0)
    gross_loss = sum(abs(r['net_ticks']) for r in results if r['net_ticks'] < 0)
    wins = sum(1 for r in results if r['net_ticks'] > 0)
    n = len(results)

    pf = gross_win / gross_loss if gross_loss > 0 else 999
    wr = wins / n

    daily_vals = list(daily_pnl.values())
    n_days = len(daily_vals)
    sharpe = (np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252)
              if n_days > 1 and np.std(daily_vals) > 0 else 0)
    neg = [v for v in daily_vals if v < 0]
    sortino = (np.mean(daily_vals) / np.std(neg) * np.sqrt(252)
               if neg and np.std(neg) > 0 else 0)
    green = sum(1 for v in daily_vals if v > 0)
    red = sum(1 for v in daily_vals if v <= 0)

    march = [v for k, v in daily_pnl.items() if k.startswith('202603')]
    april = [v for k, v in daily_pnl.items() if k.startswith('202604')]
    ms = np.mean(march) / np.std(march) * np.sqrt(252) if len(march) > 1 and np.std(march) > 0 else 0
    aps = np.mean(april) / np.std(april) * np.sqrt(252) if len(april) > 1 and np.std(april) > 0 else 0
    maxs = max(abs(ms), abs(aps))
    ra = abs(ms - aps) / maxs if maxs > 0 else 999

    daily_abs = [abs(v) for v in daily_vals]
    total_abs = sum(daily_abs)
    dc = max(daily_abs) / total_abs if total_abs > 0 else 1.0

    tp_count = sum(1 for r in results if r['exit_type'] == 'tp')
    sl_count = sum(1 for r in results if r['exit_type'] == 'sl')
    pr_count = sum(1 for r in results if r['exit_type'] == 'pressure')
    to_count = sum(1 for r in results if r['exit_type'] == 'timeout')
    avg_hold = np.mean([r['hold_seconds'] for r in results])
    avg_cost = np.mean([r['cost'] for r in results])
    tpd = n / n_days if n_days > 0 else 0

    return {
        'label': label, 'n': n, 'n_days': n_days, 'tpd': round(tpd, 1),
        'net': round(total_net, 1), 'avg_net': round(total_net / n, 3),
        'pf': round(pf, 3), 'wr': round(wr, 4),
        'sharpe': round(sharpe, 2), 'sortino': round(sortino, 2),
        'green': green, 'red': red,
        'ms': round(ms, 2), 'as': round(aps, 2),
        'ra': round(ra, 3), 'regime_pass': ra <= 0.50,
        'dc': round(dc, 3), 'dc_pass': dc <= 0.70,
        'tp_pct': round(tp_count / n * 100, 1),
        'sl_pct': round(sl_count / n * 100, 1),
        'pr_pct': round(pr_count / n * 100, 1),
        'to_pct': round(to_count / n * 100, 1),
        'avg_hold_s': round(avg_hold, 1),
        'avg_cost': round(avg_cost, 3),
    }


def main():
    print("=" * 100)
    print("STREAMING TRADE MANAGER v3 — HC #507 Compliant")
    print("Key: Passive TP exits (0.752t) vs market SL/pressure exits (1.752t)")
    print("=" * 100)

    # Load data
    all_data = {}
    for f in sorted(glob.glob(str(FILLSIM_DIR / "both_afternoon_*.json"))):
        date_str = Path(f).stem.replace("both_afternoon_", "")
        pred_file = PRED_DIR / (date_str + "_unfiltered.npz")
        mid_file = MID_DIR / (date_str + ".npz")
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
        all_data[date_str] = {
            'trades': trades, 'preds': preds, 'mid_prices': mid_prices,
            'rth_open_ns': compute_rth_open_ns(date_str)
        }

    total_t = sum(len(d['trades']) for d in all_data.values())
    print("Loaded " + str(len(all_data)) + " dates, " + str(total_t) + " trades\n")

    # Load best FP model (TP5_SL2)
    fp_model = None
    fp_path = FP_DIR / "TP5_SL2" / "model.json"
    if fp_path.exists():
        fp_model = xgb.Booster()
        fp_model.load_model(str(fp_path))
        print("Loaded FP gate model: TP5_SL2\n")

    # Config grid
    tp_values = [2, 3, 4, 5, 6, 8]
    sl_values = [4, 8, 12, 16]
    max_hold_seconds = [30, 60, 120, 300]
    fp_thresholds = [0.0, 0.5]  # 0.0 = no gate
    pressure_modes = [
        {'use_pressure': False, 'name': 'no_pr'},
        {'use_pressure': True, 'fade_thresh': 0.0, 'fade_n': 20, 'name': 'tight_pr'},
        {'use_pressure': True, 'fade_thresh': -0.3, 'fade_n': 20, 'name': 'med_pr'},
    ]

    all_results = []
    total_configs = (len(tp_values) * len(sl_values) * len(max_hold_seconds)
                     * len(fp_thresholds) * len(pressure_modes))
    print("Testing " + str(total_configs) + " configs...\n")

    count = 0
    for tp in tp_values:
        for sl in sl_values:
            for mh_s in max_hold_seconds:
                mh_bars = int(mh_s / 0.1)  # convert seconds to 100ms bars
                for fp_t in fp_thresholds:
                    for pm in pressure_modes:
                        count += 1
                        cfg = {
                            'tp': tp, 'sl': sl, 'max_hold_bars': mh_bars,
                            'use_pressure': pm['use_pressure'],
                            'fade_thresh': pm.get('fade_thresh', 0.0),
                            'fade_n': pm.get('fade_n', 20),
                            'fp_thresh': fp_t,
                        }

                        gate_str = "g" + str(fp_t) if fp_t > 0 else "ng"
                        label = ("TP" + str(tp) + "_SL" + str(sl)
                                 + "_h" + str(mh_s) + "s"
                                 + "_" + gate_str
                                 + "_" + pm['name'])

                        all_trade_results = []
                        daily_pnl = {}

                        for ds, data in sorted(all_data.items()):
                            res = simulate_trades(
                                data['trades'], data['preds'], data['mid_prices'],
                                data['rth_open_ns'], cfg,
                                fp_model=fp_model if fp_t > 0 else None
                            )
                            all_trade_results.extend(res)
                            daily_pnl[ds] = sum(r['net_ticks'] for r in res)

                        m = evaluate(all_trade_results, daily_pnl, label)
                        if m:
                            m['tp'] = tp
                            m['sl'] = sl
                            m['max_hold_s'] = mh_s
                            m['fp_thresh'] = fp_t
                            m['pressure'] = pm['name']
                            all_results.append(m)

                        if count % 50 == 0:
                            print("  Progress: " + str(count) + "/" + str(total_configs))

    print("\n" + "=" * 100)
    print("ACCEPTANCE TEST (net>0, PF>=1.2, Sharpe>=0.5, regime<=0.50, conc<=0.70, freq>=5/day)")
    print("=" * 100)

    accepted = [r for r in all_results
                if r['net'] > 0 and r['pf'] >= 1.2 and r['sharpe'] >= 0.5
                and r['regime_pass'] and r['dc_pass'] and r['tpd'] >= 5]

    if accepted:
        accepted.sort(key=lambda x: x['sharpe'], reverse=True)
        print("\n  GREEN: " + str(len(accepted)) + " configs PASS:")
        for r in accepted[:20]:
            print("    " + r['label'].ljust(40)
                  + " PF=" + str(r['pf'])
                  + " WR=" + str(round(r['wr'] * 100, 1)) + "%"
                  + " Sharpe=" + str(r['sharpe'])
                  + " Net=" + str(r['net']) + "t"
                  + " n=" + str(r['n']) + " (" + str(r['tpd']) + "/d)"
                  + " TP%=" + str(r['tp_pct'])
                  + " avgCost=" + str(r['avg_cost'])
                  + " avgHold=" + str(r['avg_hold_s']) + "s"
                  + " Mar/Apr=" + str(r['ms']) + "/" + str(r['as']))
    else:
        print("\n  RED: NO configs pass full acceptance.")

    # Near misses
    near = sorted([r for r in all_results if r['net'] > 0 and r['pf'] >= 1.0 and r['n'] >= 50],
                  key=lambda x: x['sharpe'], reverse=True)
    if near:
        print("\n  Top 20 near-misses (net>0, PF>=1.0, n>=50):")
        for r in near[:20]:
            rf = "PASS" if r['regime_pass'] else "FAIL"
            print("    " + r['label'].ljust(40)
                  + " PF=" + str(r['pf'])
                  + " WR=" + str(round(r['wr'] * 100, 1)) + "%"
                  + " Sharpe=" + str(r['sharpe'])
                  + " Net=" + str(r['net']) + "t"
                  + " n=" + str(r['n']) + " (" + str(r['tpd']) + "/d)"
                  + " TP%=" + str(r['tp_pct'])
                  + " SL%=" + str(r['sl_pct'])
                  + " avgCost=" + str(r['avg_cost'])
                  + " Regime=" + rf)

    # Cost analysis: show average cost by TP% band
    print("\n" + "=" * 100)
    print("COST ANALYSIS: Average cost vs TP%")
    print("=" * 100)
    for tp_pct_band in [80, 70, 60, 50, 40, 30]:
        band = [r for r in all_results if r['tp_pct'] >= tp_pct_band and r['tp_pct'] < tp_pct_band + 10]
        if band:
            avg_cost = np.mean([r['avg_cost'] for r in band])
            avg_net = np.mean([r['net'] for r in band])
            avg_pf = np.mean([r['pf'] for r in band])
            print("  TP " + str(tp_pct_band) + "-" + str(tp_pct_band + 10) + "%:"
                  + " avgCost=" + str(round(avg_cost, 3))
                  + " avgNet=" + str(round(avg_net, 1)) + "t"
                  + " avgPF=" + str(round(avg_pf, 3))
                  + " n_configs=" + str(len(band)))

    # Best by TP level
    print("\n" + "=" * 100)
    print("BEST CONFIG BY TP LEVEL (net>0, n>=30)")
    print("=" * 100)
    for tp in tp_values:
        tp_configs = sorted([r for r in all_results if r['tp'] == tp and r['net'] > 0 and r['n'] >= 30],
                            key=lambda x: x['sharpe'], reverse=True)
        if tp_configs:
            r = tp_configs[0]
            rf = "PASS" if r['regime_pass'] else "FAIL"
            print("  TP=" + str(tp) + " best: " + r['label'].ljust(35)
                  + " PF=" + str(r['pf'])
                  + " Sharpe=" + str(r['sharpe'])
                  + " Net=" + str(r['net']) + "t"
                  + " n=" + str(r['n'])
                  + " TP%=" + str(r['tp_pct'])
                  + " avgCost=" + str(r['avg_cost'])
                  + " Regime=" + rf)

    # Save all
    save_path = OUT_DIR / "results_v3.json"
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    n_prof = sum(1 for r in all_results if r['net'] > 0)
    n_acc = len(accepted) if accepted else 0
    print("\nSaved " + str(len(all_results)) + " results. Profitable: " + str(n_prof) + ". Accepted: " + str(n_acc))


if __name__ == '__main__':
    main()
