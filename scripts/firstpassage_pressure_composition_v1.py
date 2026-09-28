#!/usr/bin/env python3
"""
First-Passage Heads + Pressure Exit Composition v1
===================================================
HC #507: Streaming prediction trade management
HC #506: First-passage heads + pressure exit composition

Combines:
1. First-passage XGBoost heads as ENTRY GATE (P(TP_K before SL_S) > threshold)
2. Streaming CNN-Mamba v2 pressure exit for TRADE MANAGEMENT
3. Both BUY and SELL sides, afternoon window

Tests on canonical 40-day OOT window [20260227..20260429].
Evaluates HC #428 regime-agnostic requirements.
"""
import json, glob, sys, os
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
import xgboost as xgb

# ============================================================
# Paths
# ============================================================
BASE = Path("/home/nick/Lvl3Quant")
FILLSIM_DIR = BASE / "output/extended_oot_validation/fillsim_results"
PRED_DIR = BASE / "output/extended_oot_validation/pred_npzs"
MID_DIR = BASE / "data/derived/mid_price_bars"
FP_DIR = BASE / "output/direct_firstpassage_heads_v1"
OUT_DIR = BASE / "output/firstpassage_pressure_composition_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BARS = 234_000
BAR_NS = 100_000_000  # 100ms

# Cost constants (HC canonical)
COMMISSION_TICKS = 0.376  # RT commission
SPREAD_TICKS = 1.0       # crossing spread (market exit)
PASSIVE_COST = COMMISSION_TICKS  # passive entry
TAKER_COST = COMMISSION_TICKS + SPREAD_TICKS  # market entry/exit

# First-passage cells to test as gates
FP_CELLS = ['TP2_SL1', 'TP3_SL1', 'TP4_SL1', 'TP5_SL1', 'TP3_SL2', 'TP4_SL2', 'TP5_SL2']
FP_FEATURES = ['pred', 'pred_abs', 'pred_sq', 'r1', 'r5', 'r10', 'r50', 'v10', 'v50', 'v100', 'spread', 'mom20', 'ptrend5']

# ============================================================
# Helpers
# ============================================================
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
    """Compute the 13 features used by first-passage heads at a given bar."""
    pred = preds[bar_idx] if bar_idx < len(preds) else 0.0

    # Price returns at various lookbacks (in ticks)
    def ret(n):
        if bar_idx < n or mid_prices[bar_idx] <= 0 or mid_prices[bar_idx - n] <= 0:
            return 0.0
        return (mid_prices[bar_idx] - mid_prices[bar_idx - n]) / 0.25  # ticks

    # Volatility (std of returns over window)
    def vol(n):
        if bar_idx < n:
            return 0.0
        window = mid_prices[max(0, bar_idx - n):bar_idx + 1]
        nonzero = window[window > 0]
        if len(nonzero) < 2:
            return 0.0
        diffs = np.diff(nonzero) / 0.25  # tick returns
        return float(np.std(diffs)) if len(diffs) > 0 else 0.0

    # Spread (approximated as 1 tick for ES during RTH)
    spread = 1.0

    # Momentum: sign of rolling prediction
    def mom(n):
        if bar_idx < n:
            return 0.0
        window = preds[max(0, bar_idx - n):bar_idx + 1]
        return float(np.mean(window))

    # Price trend (regression slope proxy)
    def ptrend(n):
        if bar_idx < n:
            return 0.0
        window = mid_prices[max(0, bar_idx - n):bar_idx + 1]
        nonzero = window[window > 0]
        if len(nonzero) < 2:
            return 0.0
        return float(nonzero[-1] - nonzero[0]) / 0.25  # ticks

    return np.array([
        pred,
        abs(pred),
        pred ** 2,
        ret(1),     # r1 = 100ms return
        ret(5),     # r5 = 500ms return
        ret(10),    # r10 = 1s return
        ret(50),    # r50 = 5s return
        vol(10),    # v10
        vol(50),    # v50
        vol(100),   # v100
        spread,
        mom(20),    # mom20
        ptrend(5),  # ptrend5
    ], dtype=np.float32)


def run_pressure_exit(date_str, trades, preds, mid_prices, cfg, fp_model=None, fp_thresh=0.5):
    """
    Run pressure exit with optional first-passage gating.

    If fp_model is provided, only enter trades where P(TP before SL) > fp_thresh.
    """
    rth_open_ns = compute_rth_open_ns(date_str)
    fade_thresh = cfg['fade_thresh']
    fade_n = cfg['fade_n']
    rev_thresh = cfg['rev_thresh']
    rev_n = cfg['rev_n']

    results = []
    gated_out = 0
    gated_in = 0

    for t in trades:
        entry_ns = t['fill_time_ns']
        entry_px = t['entry_price']
        orig_pnl = t['pnl_ticks']
        side = t.get('side', 'BUY')
        entry_bar = ns_to_bar(entry_ns, rth_open_ns)

        # First-passage gate
        if fp_model is not None:
            features = compute_features_at_bar(preds, mid_prices, entry_bar)
            # Flip prediction sign for SELL trades (model trained on long-side)
            if side == 'SELL':
                features[0] = -features[0]  # flip pred
                features[3] = -features[3]  # flip r1
                features[4] = -features[4]  # flip r5
                features[5] = -features[5]  # flip r10
                features[6] = -features[6]  # flip r50
                features[11] = -features[11]  # flip mom20
                features[12] = -features[12]  # flip ptrend5

            dmat = xgb.DMatrix(features.reshape(1, -1), feature_names=FP_FEATURES)
            prob = fp_model.predict(dmat)[0]

            if prob < fp_thresh:
                gated_out += 1
                continue  # Skip this trade
            gated_in += 1
        else:
            gated_in += 1

        # Original exit
        exit_ns = t.get('exit_time_ns', entry_ns + 1800_000_000_000)
        exit_bar = ns_to_bar(exit_ns, rth_open_ns)

        # TP/SL in price (from fillsim config — 8 tick TP, 16 tick SL)
        if side == 'BUY':
            tp_px = entry_px + 8 * 0.25
            sl_px = entry_px - 16 * 0.25
        else:
            tp_px = entry_px - 8 * 0.25
            sl_px = entry_px + 16 * 0.25

        # Walk bar by bar — pressure exit logic
        fade_count = 0
        rev_count = 0
        pressure_exit = False
        pressure_exit_bar = None

        for bar in range(entry_bar + 1, min(exit_bar + 1, N_BARS)):
            mid = mid_prices[bar]
            if mid <= 0:
                continue

            # TP/SL check
            if side == 'BUY':
                if mid >= tp_px:
                    break
                if mid <= sl_px:
                    break
            else:
                if mid <= tp_px:
                    break
                if mid >= sl_px:
                    break

            # Prediction signal (flip for sell)
            pred = preds[bar]
            if side == 'SELL':
                pred = -pred  # flip so positive = in our favor

            # Fade detection
            if pred < fade_thresh:
                fade_count += 1
            else:
                fade_count = 0

            # Reversal detection
            if pred < rev_thresh:
                rev_count += 1
            else:
                rev_count = 0

            # Trigger pressure exit
            if (fade_n > 0 and fade_count >= fade_n) or (rev_n > 0 and rev_count >= rev_n):
                pressure_exit = True
                pressure_exit_bar = bar
                break

        if pressure_exit and pressure_exit_bar is not None:
            exit_mid = mid_prices[pressure_exit_bar]
            if exit_mid > 0:
                if side == 'BUY':
                    new_pnl_ticks = (exit_mid - entry_px) / 0.25
                else:
                    new_pnl_ticks = (entry_px - exit_mid) / 0.25
                results.append({
                    'orig_pnl': orig_pnl,
                    'new_pnl': new_pnl_ticks,
                    'exit_type': 'pressure',
                    'side': side,
                    'bars_held': pressure_exit_bar - entry_bar,
                })
            else:
                results.append({
                    'orig_pnl': orig_pnl,
                    'new_pnl': orig_pnl,
                    'exit_type': 'original',
                    'side': side,
                })
        else:
            results.append({
                'orig_pnl': orig_pnl,
                'new_pnl': orig_pnl,
                'exit_type': 'original',
                'side': side,
            })

    return results, gated_in, gated_out


def evaluate_results(all_results, daily_pnl, label):
    """Compute risk-adjusted metrics."""
    if not all_results:
        return None

    total = sum(r['new_pnl'] for r in all_results)
    gross_win = sum(r['new_pnl'] for r in all_results if r['new_pnl'] > 0)
    gross_loss = sum(abs(r['new_pnl']) for r in all_results if r['new_pnl'] < 0)
    wins = sum(1 for r in all_results if r['new_pnl'] > 0)
    n_pressure = sum(1 for r in all_results if r['exit_type'] == 'pressure')

    pf = gross_win / gross_loss if gross_loss > 0 else 999
    wr = wins / len(all_results) if all_results else 0

    daily_vals = list(daily_pnl.values())
    n_days = len(daily_vals)
    sharpe = np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252) if n_days > 1 and np.std(daily_vals) > 0 else 0
    neg = [v for v in daily_vals if v < 0]
    sortino = np.mean(daily_vals) / np.std(neg) * np.sqrt(252) if neg and np.std(neg) > 0 else 0
    green = sum(1 for v in daily_vals if v > 0)
    red = sum(1 for v in daily_vals if v <= 0)

    avg_per_trade = total / len(all_results)
    trades_per_day = len(all_results) / n_days if n_days > 0 else 0

    # Regime split: March (20260301-20260331) vs April (20260401-20260430)
    march_pnl = [v for k, v in daily_pnl.items() if k.startswith('202603')]
    april_pnl = [v for k, v in daily_pnl.items() if k.startswith('202604')]

    march_sharpe = np.mean(march_pnl) / np.std(march_pnl) * np.sqrt(252) if len(march_pnl) > 1 and np.std(march_pnl) > 0 else 0
    april_sharpe = np.mean(april_pnl) / np.std(april_pnl) * np.sqrt(252) if len(april_pnl) > 1 and np.std(april_pnl) > 0 else 0

    # HC #428 R1 regime-agnostic test
    max_sharpe = max(abs(march_sharpe), abs(april_sharpe))
    regime_asymmetry = abs(march_sharpe - april_sharpe) / max_sharpe if max_sharpe > 0 else 999
    regime_pass = regime_asymmetry <= 0.50

    # Day concentration (HC #344)
    if daily_vals:
        daily_abs = [abs(v) for v in daily_vals]
        total_abs = sum(daily_abs)
        day_conc = max(daily_abs) / total_abs if total_abs > 0 else 1.0
    else:
        day_conc = 1.0

    return {
        'label': label,
        'n_trades': len(all_results),
        'n_days': n_days,
        'trades_per_day': round(trades_per_day, 1),
        'net_ticks': round(total, 1),
        'avg_per_trade': round(avg_per_trade, 3),
        'pf': round(pf, 3),
        'wr': round(wr, 4),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'green': green,
        'red': red,
        'pct_pressure': round(n_pressure / len(all_results) * 100, 1) if all_results else 0,
        'march_sharpe': round(march_sharpe, 2),
        'april_sharpe': round(april_sharpe, 2),
        'regime_asymmetry': round(regime_asymmetry, 3),
        'regime_pass': regime_pass,
        'day_conc': round(day_conc, 3),
        'day_conc_pass': day_conc <= 0.70,
    }


def main():
    print("=" * 100)
    print("FIRST-PASSAGE HEADS + PRESSURE EXIT COMPOSITION v1")
    print("HC #507: Streaming prediction trade management")
    print("=" * 100)

    # ============================================================
    # Load all date data
    # ============================================================
    all_data = {}
    trade_files = sorted(glob.glob(str(FILLSIM_DIR / "both_afternoon_*.json")))

    for f in trade_files:
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

        nonzero = mid_prices[mid_prices > 0]
        if len(nonzero) < N_BARS * 0.5:
            print(f"  SKIP {date_str}: insufficient mid price coverage")
            continue

        all_data[date_str] = {
            'trades': trades,
            'preds': preds,
            'mid_prices': mid_prices,
        }

    print(f"\nLoaded {len(all_data)} dates")
    total_trades = sum(len(d['trades']) for d in all_data.values())
    buy_trades = sum(sum(1 for t in d['trades'] if t.get('side') == 'BUY') for d in all_data.values())
    sell_trades = sum(sum(1 for t in d['trades'] if t.get('side') == 'SELL') for d in all_data.values())
    print(f"Total trades: {total_trades} (BUY: {buy_trades}, SELL: {sell_trades})")

    # ============================================================
    # Load first-passage XGBoost models
    # ============================================================
    fp_models = {}
    for cell in FP_CELLS:
        model_path = FP_DIR / cell / "model.json"
        if model_path.exists():
            model = xgb.Booster()
            model.load_model(str(model_path))
            fp_models[cell] = model
            print(f"  Loaded FP model: {cell}")

    print(f"\nLoaded {len(fp_models)} first-passage gate models")

    # ============================================================
    # Pressure exit configs (best from prior analysis + variations)
    # ============================================================
    pressure_configs = [
        # Best from prior analysis
        {'fade_thresh': -0.3, 'fade_n': 20, 'rev_thresh': -0.5, 'rev_n': 5, 'name': 'best_prior'},
        {'fade_thresh': -0.1, 'fade_n': 160, 'rev_thresh': -0.5, 'rev_n': 8, 'name': 'best_sharpe'},
        # Tighter (faster exit)
        {'fade_thresh': 0.0, 'fade_n': 20, 'rev_thresh': -0.3, 'rev_n': 5, 'name': 'tight'},
        # Looser (hold longer)
        {'fade_thresh': -0.3, 'fade_n': 80, 'rev_thresh': -0.5, 'rev_n': 15, 'name': 'loose'},
        # No pressure exit (baseline)
        {'fade_thresh': -999, 'fade_n': 999999, 'rev_thresh': -999, 'rev_n': 999999, 'name': 'no_pressure'},
    ]

    # First-passage gate thresholds
    fp_thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    # ============================================================
    # Test matrix: FP_cell × FP_threshold × pressure_config
    # Plus: no-gate baselines (pressure only)
    # ============================================================
    all_results_summary = []

    # --- BASELINE: No gate, each pressure config ---
    print("\n" + "=" * 100)
    print("PHASE 1: PRESSURE-ONLY BASELINES (no first-passage gate)")
    print("=" * 100)

    for pcfg in pressure_configs:
        all_trades_results = []
        daily_pnl = {}

        for date_str, data in sorted(all_data.items()):
            results, _, _ = run_pressure_exit(
                date_str, data['trades'], data['preds'], data['mid_prices'],
                pcfg, fp_model=None
            )
            all_trades_results.extend(results)
            daily_pnl[date_str] = sum(r['new_pnl'] for r in results)

        metrics = evaluate_results(all_trades_results, daily_pnl, f"NoGate_{pcfg['name']}")
        if metrics:
            all_results_summary.append(metrics)
            regime_flag = "✅" if metrics['regime_pass'] else "❌"
            conc_flag = "✅" if metrics['day_conc_pass'] else "❌"
            print(f"  {pcfg['name']:15s}: PF={metrics['pf']:.3f} WR={metrics['wr']:.1%} "
                  f"Sharpe={metrics['sharpe']:.1f} Sortino={metrics['sortino']:.1f} "
                  f"Net={metrics['net_ticks']:.0f}t "
                  f"Mar/Apr={metrics['march_sharpe']:.1f}/{metrics['april_sharpe']:.1f} "
                  f"Regime{regime_flag} DayConc={metrics['day_conc']:.2f}{conc_flag} "
                  f"n={metrics['n_trades']} ({metrics['trades_per_day']:.0f}/day)")

    # --- GATED: FP cell × threshold × pressure config ---
    print("\n" + "=" * 100)
    print("PHASE 2: FIRST-PASSAGE GATED + PRESSURE EXIT")
    print("=" * 100)

    for cell_name, fp_model in sorted(fp_models.items()):
        print(f"\n--- Gate: {cell_name} ---")

        for fp_thresh in fp_thresholds:
            for pcfg in pressure_configs:
                all_trades_results = []
                daily_pnl = {}
                total_gated_in = 0
                total_gated_out = 0

                for date_str, data in sorted(all_data.items()):
                    results, gi, go = run_pressure_exit(
                        date_str, data['trades'], data['preds'], data['mid_prices'],
                        pcfg, fp_model=fp_model, fp_thresh=fp_thresh
                    )
                    all_trades_results.extend(results)
                    total_gated_in += gi
                    total_gated_out += go
                    daily_pnl[date_str] = sum(r['new_pnl'] for r in results)

                label = f"{cell_name}_t{fp_thresh}_{pcfg['name']}"
                metrics = evaluate_results(all_trades_results, daily_pnl, label)
                if metrics:
                    metrics['gate_cell'] = cell_name
                    metrics['gate_thresh'] = fp_thresh
                    metrics['pressure_cfg'] = pcfg['name']
                    metrics['gated_in'] = total_gated_in
                    metrics['gated_out'] = total_gated_out
                    pass_rate = total_gated_in / (total_gated_in + total_gated_out) * 100 if (total_gated_in + total_gated_out) > 0 else 0
                    metrics['gate_pass_rate'] = round(pass_rate, 1)
                    all_results_summary.append(metrics)

                    if metrics['pf'] > 1.0 and metrics['n_trades'] >= 50:
                        regime_flag = "✅" if metrics['regime_pass'] else "❌"
                        conc_flag = "✅" if metrics['day_conc_pass'] else "❌"
                        print(f"  t={fp_thresh} {pcfg['name']:15s}: "
                              f"PF={metrics['pf']:.3f} WR={metrics['wr']:.1%} "
                              f"Sharpe={metrics['sharpe']:.1f} Sortino={metrics['sortino']:.1f} "
                              f"Net={metrics['net_ticks']:.0f}t "
                              f"Mar/Apr={metrics['march_sharpe']:.1f}/{metrics['april_sharpe']:.1f} "
                              f"Regime{regime_flag} "
                              f"n={metrics['n_trades']} pass={pass_rate:.0f}%")

    # ============================================================
    # Summary: Sort by Sharpe, filter profitable
    # ============================================================
    print("\n" + "=" * 100)
    print("TOP 20 CONFIGS BY SHARPE (PF > 1.0, n >= 30)")
    print("=" * 100)

    profitable = [r for r in all_results_summary if r['pf'] > 1.0 and r['n_trades'] >= 30]
    profitable.sort(key=lambda x: x['sharpe'], reverse=True)

    for i, r in enumerate(profitable[:20]):
        regime_flag = "✅" if r['regime_pass'] else "❌"
        conc_flag = "✅" if r['day_conc_pass'] else "❌"
        gate_info = f"gate={r.get('gate_cell','none')}_t{r.get('gate_thresh','')}" if 'gate_cell' in r else "no_gate"
        print(f"  {i+1:2d}. {r['label']:40s} "
              f"PF={r['pf']:.3f} WR={r['wr']:.1%} Sharpe={r['sharpe']:.1f} Sortino={r['sortino']:.1f} "
              f"Net={r['net_ticks']:.0f}t n={r['n_trades']} "
              f"Mar/Apr={r['march_sharpe']:.1f}/{r['april_sharpe']:.1f} "
              f"Regime{regime_flag} DConc={r['day_conc']:.2f}{conc_flag}")

    # HC #506 R5 acceptance gate
    print("\n" + "=" * 100)
    print("HC #506 R5 ACCEPTANCE TEST")
    print("Criteria: net>0, PF≥1.2, Sharpe≥0.5, regime_asym≤0.50, day_conc≤0.70, trades/day≥5")
    print("=" * 100)

    accepted = []
    for r in all_results_summary:
        if (r['net_ticks'] > 0 and r['pf'] >= 1.2 and r['sharpe'] >= 0.5
            and r['regime_pass'] and r['day_conc_pass'] and r['trades_per_day'] >= 5):
            accepted.append(r)

    if accepted:
        accepted.sort(key=lambda x: x['sharpe'], reverse=True)
        print(f"\n  🟢 {len(accepted)} configs PASS acceptance:")
        for r in accepted[:10]:
            print(f"    {r['label']:40s} PF={r['pf']:.3f} Sharpe={r['sharpe']:.1f} "
                  f"Sortino={r['sortino']:.1f} Net={r['net_ticks']:.0f}t "
                  f"n={r['n_trades']} ({r['trades_per_day']:.0f}/day) "
                  f"Mar/Apr={r['march_sharpe']:.1f}/{r['april_sharpe']:.1f}")
    else:
        print("\n  🔴 NO configs pass full acceptance criteria.")
        # Show closest misses
        near = [r for r in all_results_summary if r['net_ticks'] > 0 and r['pf'] >= 1.0]
        near.sort(key=lambda x: x['sharpe'], reverse=True)
        if near:
            print(f"  Closest {min(5, len(near))} near-misses (net>0, PF≥1.0):")
            for r in near[:5]:
                fails = []
                if r['pf'] < 1.2: fails.append(f"PF={r['pf']:.2f}<1.2")
                if r['sharpe'] < 0.5: fails.append(f"Sharpe={r['sharpe']:.1f}<0.5")
                if not r['regime_pass']: fails.append(f"regime={r['regime_asymmetry']:.2f}>0.50")
                if not r['day_conc_pass']: fails.append(f"conc={r['day_conc']:.2f}>0.70")
                if r['trades_per_day'] < 5: fails.append(f"freq={r['trades_per_day']:.1f}<5/day")
                print(f"    {r['label']:40s} Sharpe={r['sharpe']:.1f} PF={r['pf']:.3f} "
                      f"Net={r['net_ticks']:.0f}t FAIL: {', '.join(fails)}")

    # Save all results
    save_path = OUT_DIR / "composition_results.json"
    with open(save_path, 'w') as f:
        json.dump(all_results_summary, f, indent=2, default=str)
    print(f"\nSaved {len(all_results_summary)} config results to {save_path}")

    # Summary stats
    print(f"\nTotal configs tested: {len(all_results_summary)}")
    profitable_count = sum(1 for r in all_results_summary if r['net_ticks'] > 0)
    print(f"Profitable configs: {profitable_count}")
    print(f"Accepted configs: {len(accepted) if accepted else 0}")


if __name__ == '__main__':
    main()
