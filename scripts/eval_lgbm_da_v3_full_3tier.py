#!/usr/bin/env python3
"""
Comprehensive 3-Tier Analysis for LGBM DA smart_v3 model.
Tier 1: Signal Quality (IC, DA, MagCorr at confidence tiers)
Tier 2: Tradeability (MFE/MAE, win/loss ratio, long/short breakdown)
Tier 3: Profit Translation (Simulated P&L, Sortino, profit factor, equity curves)
+ Temporal Decay Analysis across Early/Mid/Late periods
"""
import numpy as np
import glob
import json
import os
import sys
from scipy.stats import spearmanr, pearsonr

PRED_DIR = "/home/jupiter/Lvl3Quant/output/lgbm_da_smart_v3_1d_oot"
OUTPUT_FILE = os.path.join(PRED_DIR, "full_3tier_analysis.json")

# Cost assumptions (ES futures)
TICK_VALUE = 12.50  # USD per tick
COST_TICKS = 2.0    # spread + slippage in ticks
AVG_MOVE_TICKS = 2.5  # avg magnitude of move per trade (conservative ES 10s)

# Confidence tiers (percentile thresholds)
CONFIDENCE_TIERS = {
    "All":     0,
    "Top50%":  50,
    "Top25%":  75,
    "Top10%":  90,
    "Top5%":   95,
    "Top1%":   99,
    "Top0.5%": 99.5,
}

# Temporal periods (fold index ranges)
TEMPORAL_PERIODS = {
    "Early (Sept-Oct 2025)": (0, 30),
    "Mid (Nov-Dec 2025)":    (31, 80),
    "Late (Jan-Mar 2026)":   (81, 142),
}


def load_all_folds():
    """Load all fold predictions, returning arrays + per-fold metadata."""
    preds_files = sorted(glob.glob(f"{PRED_DIR}/fold*_preds.npz"))
    print(f"Loading {len(preds_files)} fold predictions...")

    all_probs, all_labels, all_conf, all_fold_idx = [], [], [], []
    fold_meta = []

    for i, pf in enumerate(preds_files):
        d = np.load(pf)
        probs = d['probs']
        labels = d['labels']
        conf = d['confidence']
        n = len(labels)

        all_probs.append(probs)
        all_labels.append(labels)
        all_conf.append(conf)
        all_fold_idx.append(np.full(n, i, dtype=np.int32))

        da = np.mean((probs > 0.5) == labels)
        fold_meta.append({
            'fold_idx': i,
            'fold_name': os.path.basename(pf).replace('_preds.npz', ''),
            'n': int(n),
            'da': float(da),
        })

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    conf = np.concatenate(all_conf)
    fold_idx = np.concatenate(all_fold_idx)

    print(f"Total samples: {len(labels):,}")
    return probs, labels, conf, fold_idx, fold_meta


def get_confidence_mask(conf, percentile):
    """Return boolean mask for samples above given confidence percentile."""
    if percentile == 0:
        return np.ones(len(conf), dtype=bool)
    thresh = np.percentile(conf, percentile)
    return conf >= thresh


def compute_tier1(probs, labels, conf, mask=None):
    """Tier 1: Signal Quality metrics."""
    if mask is None:
        mask = np.ones(len(probs), dtype=bool)

    p = probs[mask]
    l = labels[mask]
    c = conf[mask]
    n = len(p)
    if n < 10:
        return None

    pred_dir = (p > 0.5).astype(float)
    correct = (pred_dir == l)

    # Directional Accuracy
    da = float(correct.mean())

    # Information Coefficient (Spearman rank corr between prob and label)
    ic, ic_pval = spearmanr(p, l)

    # Pearson correlation (for magnitude correlation)
    mag_corr, mag_pval = pearsonr(p, l)

    # Magnitude-weighted correlation: correlation between |prob - 0.5| and correctness
    signal_strength = np.abs(p - 0.5)
    mag_correct_corr, _ = spearmanr(signal_strength, correct.astype(float))

    # Calibration: mean predicted prob vs actual label rate
    mean_prob = float(p.mean())
    actual_rate = float(l.mean())

    # Long/short split
    long_mask = pred_dir == 1
    short_mask = pred_dir == 0
    long_da = float(correct[long_mask].mean()) if long_mask.sum() > 5 else None
    short_da = float(correct[short_mask].mean()) if short_mask.sum() > 5 else None

    return {
        "n": int(n),
        "DA": round(da, 6),
        "IC_spearman": round(float(ic), 6),
        "IC_pval": round(float(ic_pval), 8),
        "MagCorr_pearson": round(float(mag_corr), 6),
        "MagCorr_pval": round(float(mag_pval), 8),
        "SignalStrength_vs_Correctness": round(float(mag_correct_corr), 6),
        "mean_confidence": round(float(c.mean()), 6),
        "calibration_predicted": round(mean_prob, 6),
        "calibration_actual": round(actual_rate, 6),
        "long_DA": long_da,
        "short_DA": short_da,
        "n_long": int(long_mask.sum()),
        "n_short": int(short_mask.sum()),
    }


def simulate_trades(probs, labels, conf, mask=None):
    """Simulate trade-by-trade PnL for Tier 2 and Tier 3 analysis.

    Three cost models:
    1. Gross (no cost) - pure signal value
    2. Low cost (0.5 tick) - crossing at mid, minimal slippage
    3. Full cost (2 ticks) - aggressive crossing with slippage

    Trade model: direction correct = +AVG_MOVE, wrong = -AVG_MOVE
    Then subtract round-trip cost.
    """
    if mask is None:
        mask = np.ones(len(probs), dtype=bool)

    p = probs[mask]
    l = labels[mask]
    c = conf[mask]
    n = len(p)
    if n < 10:
        return None

    pred_dir = (p > 0.5).astype(float)  # 1=long, 0=short
    correct = (pred_dir == l)

    # Directional P&L (gross, before any costs)
    # Correct: profit of avg_move, Wrong: loss of avg_move
    direction_pnl = np.where(correct, AVG_MOVE_TICKS, -AVG_MOVE_TICKS)

    # Three cost scenarios
    net_pnl_gross = direction_pnl.copy()  # no cost
    net_pnl_low = direction_pnl - 0.5    # 0.5 tick cost
    net_pnl_full = direction_pnl - COST_TICKS  # 2 tick cost

    return {
        'pred_dir': pred_dir,
        'correct': correct,
        'confidence': c,
        'direction_pnl': direction_pnl,
        'net_pnl_gross': net_pnl_gross,
        'net_pnl_low': net_pnl_low,
        'net_pnl_full': net_pnl_full,
        'n': n,
    }


def compute_tier2(trade_data):
    """Tier 2: Tradeability metrics."""
    if trade_data is None:
        return None

    correct = trade_data['correct']
    pred_dir = trade_data['pred_dir']
    conf = trade_data['confidence']
    gross = trade_data['net_pnl_gross']
    n = trade_data['n']

    wins = int(correct.sum())
    losses = int(n - wins)
    win_rate = float(wins / n)

    # MFE/MAE: use gross PnL (no cost) to measure pure directional edge
    # MFE = avg gross profit on winners, MAE = avg gross loss on losers
    mfe = float(gross[correct].mean()) if wins > 0 else 0.0
    mae = float(abs(gross[~correct].mean())) if losses > 0 else 0.0

    # Avg winner/loser in gross and net terms
    avg_winner_gross = float(gross[correct].mean()) if wins > 0 else 0.0
    avg_loser_gross = float(gross[~correct].mean()) if losses > 0 else 0.0

    # Win/loss ratio (gross)
    wl_ratio = abs(avg_winner_gross / avg_loser_gross) if avg_loser_gross != 0 else float('inf')

    # Expectancy per trade (gross)
    expectancy_gross = float(gross.mean())

    # Confidence of winners vs losers
    winner_conf = conf[correct]
    loser_conf = conf[~correct]
    avg_winner_conf = float(winner_conf.mean()) if len(winner_conf) > 0 else 0
    avg_loser_conf = float(loser_conf.mean()) if len(loser_conf) > 0 else 0

    # Long vs Short breakdown
    long_mask = pred_dir == 1
    short_mask = pred_dir == 0

    def side_stats(side_mask):
        if side_mask.sum() < 5:
            return None
        sm_correct = correct[side_mask]
        sm_gross = gross[side_mask]
        return {
            "n": int(side_mask.sum()),
            "win_rate": round(float(sm_correct.mean()), 6),
            "avg_pnl_gross_ticks": round(float(sm_gross.mean()), 4),
            "total_pnl_gross_ticks": round(float(sm_gross.sum()), 2),
        }

    return {
        "n": int(n),
        "win_rate": round(win_rate, 6),
        "wins": wins,
        "losses": losses,
        "MFE_ticks": round(mfe, 4),
        "MAE_ticks": round(mae, 4),
        "avg_winner_gross_ticks": round(avg_winner_gross, 4),
        "avg_loser_gross_ticks": round(avg_loser_gross, 4),
        "win_loss_ratio": round(wl_ratio, 4),
        "expectancy_gross_per_trade_ticks": round(expectancy_gross, 4),
        "avg_winner_confidence": round(avg_winner_conf, 6),
        "avg_loser_confidence": round(avg_loser_conf, 6),
        "long_stats": side_stats(long_mask),
        "short_stats": side_stats(short_mask),
    }


def compute_pnl_stats(pnl_series, label):
    """Compute P&L statistics for a given cost scenario."""
    n = len(pnl_series)
    cum_pnl = np.cumsum(pnl_series)
    total_pnl = float(cum_pnl[-1])

    mean_pnl = float(pnl_series.mean())
    std_pnl = float(pnl_series.std()) if n > 1 else 1.0

    # Sortino
    downside = pnl_series[pnl_series < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 1.0
    sortino = mean_pnl / downside_std if downside_std > 0 else 0.0

    # Sharpe
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

    # Profit factor
    win_sum = float(pnl_series[pnl_series > 0].sum()) if (pnl_series > 0).any() else 0.0
    loss_sum = float(abs(pnl_series[pnl_series < 0].sum())) if (pnl_series < 0).any() else 1.0
    pf = win_sum / loss_sum if loss_sum > 0 else float('inf')

    # Max drawdown
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_max
    max_dd = float(drawdowns.min())

    # Recovery factor
    recovery = abs(total_pnl / max_dd) if max_dd != 0 else float('inf')

    # Equity curve sample (20 points)
    n_pts = min(20, len(cum_pnl))
    indices = np.linspace(0, len(cum_pnl)-1, n_pts, dtype=int)
    eq_sample = [(int(i), round(float(cum_pnl[i]), 2)) for i in indices]

    return {
        "total_pnl_ticks": round(total_pnl, 2),
        "total_pnl_usd": round(total_pnl * TICK_VALUE, 2),
        "mean_pnl_per_trade_ticks": round(mean_pnl, 4),
        "sortino": round(sortino, 4),
        "sharpe": round(sharpe, 4),
        "profit_factor": round(float(pf), 4),
        "max_drawdown_ticks": round(max_dd, 2),
        "max_drawdown_usd": round(max_dd * TICK_VALUE, 2),
        "recovery_factor": round(float(recovery), 4),
        "equity_curve_sample": eq_sample,
    }


def compute_tier3(trade_data):
    """Tier 3: Profit Translation metrics under 3 cost scenarios."""
    if trade_data is None:
        return None

    n = trade_data['n']
    correct = trade_data['correct']

    # Streak analysis (on gross)
    gross = trade_data['net_pnl_gross']
    streaks = []
    current_streak = 0
    current_sign = None
    for pnl in gross:
        s = 1 if pnl > 0 else -1
        if s == current_sign:
            current_streak += 1
        else:
            if current_sign is not None:
                streaks.append((current_sign, current_streak))
            current_streak = 1
            current_sign = s
    if current_sign is not None:
        streaks.append((current_sign, current_streak))

    win_streaks = [s[1] for s in streaks if s[0] == 1]
    loss_streaks = [s[1] for s in streaks if s[0] == -1]

    # Breakeven DA for each cost level
    # At breakeven: DA * (move - cost) = (1 - DA) * (move + cost) -- wrong, cost is flat
    # Actually: DA * move - (1-DA) * move - cost = 0 => (2*DA-1)*move = cost => DA = 0.5 + cost/(2*move)
    breakeven_gross = 0.500
    breakeven_low = 0.5 + 0.5 / (2 * AVG_MOVE_TICKS)
    breakeven_full = 0.5 + COST_TICKS / (2 * AVG_MOVE_TICKS)

    result = {
        "n_trades": n,
        "max_win_streak": max(win_streaks) if win_streaks else 0,
        "max_loss_streak": max(loss_streaks) if loss_streaks else 0,
        "avg_move_assumption_ticks": AVG_MOVE_TICKS,
        "breakeven_DA": {
            "gross": round(breakeven_gross, 4),
            "low_cost_0.5t": round(breakeven_low, 4),
            "full_cost_2t": round(breakeven_full, 4),
        },
        "scenarios": {
            "gross_no_cost": compute_pnl_stats(trade_data['net_pnl_gross'], "gross"),
            "low_cost_0.5t": compute_pnl_stats(trade_data['net_pnl_low'], "low"),
            "full_cost_2t": compute_pnl_stats(trade_data['net_pnl_full'], "full"),
        },
    }

    return result


def analyze_temporal_decay(probs, labels, conf, fold_idx, fold_meta):
    """Analyze how metrics change across time periods."""
    temporal = {}

    for period_name, (fold_start, fold_end) in TEMPORAL_PERIODS.items():
        period_mask = (fold_idx >= fold_start) & (fold_idx <= fold_end)
        n_period = period_mask.sum()
        if n_period < 50:
            temporal[period_name] = {"error": "insufficient data", "n": int(n_period)}
            continue

        p_period = probs[period_mask]
        l_period = labels[period_mask]
        c_period = conf[period_mask]

        # Compute all tiers for this period
        period_tiers = {}
        for tier_name, pct in CONFIDENCE_TIERS.items():
            # Use GLOBAL percentile thresholds for consistency
            tier_mask_local = get_confidence_mask(c_period, pct)

            t1 = compute_tier1(p_period, l_period, c_period, tier_mask_local)
            if t1 is None:
                continue

            td = simulate_trades(p_period, l_period, c_period, tier_mask_local)
            t2 = compute_tier2(td)
            t3 = compute_tier3(td)

            period_tiers[tier_name] = {
                "tier1_signal": t1,
                "tier2_tradeability": t2,
                "tier3_profit": t3,
            }

        # Per-fold DA for this period
        period_folds = [fm for fm in fold_meta if fold_start <= fm['fold_idx'] <= fold_end]
        fold_das = [fm['da'] for fm in period_folds]

        temporal[period_name] = {
            "fold_range": f"{fold_start}-{fold_end}",
            "n_folds": len(period_folds),
            "n_samples": int(n_period),
            "mean_fold_DA": round(float(np.mean(fold_das)), 6) if fold_das else None,
            "std_fold_DA": round(float(np.std(fold_das)), 6) if fold_das else None,
            "min_fold_DA": round(float(np.min(fold_das)), 6) if fold_das else None,
            "max_fold_DA": round(float(np.max(fold_das)), 6) if fold_das else None,
            "tiers": period_tiers,
        }

    return temporal


def main():
    probs, labels, conf, fold_idx, fold_meta = load_all_folds()

    results = {
        "model": "LGBM DA smart_v3",
        "analysis": "Full 3-Tier + Temporal Decay",
        "total_folds": len(fold_meta),
        "total_samples": int(len(labels)),
        "cost_model": {
            "tick_value_usd": TICK_VALUE,
            "cost_ticks": COST_TICKS,
            "avg_move_ticks": AVG_MOVE_TICKS,
        },
    }

    # ========== GLOBAL ANALYSIS (all folds) ==========
    print("\n" + "=" * 100)
    print("GLOBAL ANALYSIS (ALL 143 FOLDS)")
    print("=" * 100)

    global_tiers = {}
    for tier_name, pct in CONFIDENCE_TIERS.items():
        mask = get_confidence_mask(conf, pct)

        t1 = compute_tier1(probs, labels, conf, mask)
        td = simulate_trades(probs, labels, conf, mask)
        t2 = compute_tier2(td)
        t3 = compute_tier3(td)

        if t1 is None:
            continue

        global_tiers[tier_name] = {
            "tier1_signal": t1,
            "tier2_tradeability": t2,
            "tier3_profit": t3,
        }

        # Print summary
        print(f"\n--- {tier_name} (n={t1['n']:,}) ---")
        print(f"  T1 Signal:  DA={t1['DA']:.4f}  IC={t1['IC_spearman']:.4f}  MagCorr={t1['MagCorr_pearson']:.4f}  SigStr={t1['SignalStrength_vs_Correctness']:.4f}")
        long_da_str = f"{t1['long_DA']:.4f}" if t1['long_DA'] is not None else "N/A"
        short_da_str = f"{t1['short_DA']:.4f}" if t1['short_DA'] is not None else "N/A"
        print(f"             Long DA={long_da_str}  Short DA={short_da_str}  (L={t1['n_long']:,} S={t1['n_short']:,})")
        if t2:
            print(f"  T2 Trade:   WinRate={t2['win_rate']:.4f}  MFE={t2['MFE_ticks']:.2f}t  MAE={t2['MAE_ticks']:.2f}t  W/L={t2['win_loss_ratio']:.3f}  Expect(gross)={t2['expectancy_gross_per_trade_ticks']:.4f}t")
            if t2['long_stats']:
                ls = t2['long_stats']
                print(f"             Long:  WR={ls['win_rate']:.4f}  AvgGross={ls['avg_pnl_gross_ticks']:.4f}t  TotalGross={ls['total_pnl_gross_ticks']:.1f}t")
            if t2['short_stats']:
                ss = t2['short_stats']
                print(f"             Short: WR={ss['win_rate']:.4f}  AvgGross={ss['avg_pnl_gross_ticks']:.4f}t  TotalGross={ss['total_pnl_gross_ticks']:.1f}t")
        if t3:
            for scenario_name in ['gross_no_cost', 'low_cost_0.5t', 'full_cost_2t']:
                sc = t3['scenarios'][scenario_name]
                print(f"  T3 [{scenario_name}]:  PnL={sc['total_pnl_ticks']:.0f}t (${sc['total_pnl_usd']:,.0f})  Sortino={sc['sortino']:.4f}  PF={sc['profit_factor']:.3f}  MaxDD={sc['max_drawdown_ticks']:.0f}t")
            print(f"  Breakeven DA: gross={t3['breakeven_DA']['gross']:.3f}  low={t3['breakeven_DA']['low_cost_0.5t']:.3f}  full={t3['breakeven_DA']['full_cost_2t']:.3f}")

    results["global"] = global_tiers

    # ========== TEMPORAL DECAY ANALYSIS ==========
    print("\n" + "=" * 100)
    print("TEMPORAL DECAY ANALYSIS")
    print("=" * 100)

    temporal = analyze_temporal_decay(probs, labels, conf, fold_idx, fold_meta)

    for period_name, period_data in temporal.items():
        print(f"\n{'=' * 80}")
        print(f"  {period_name}  |  Folds: {period_data.get('fold_range', 'N/A')}  |  n={period_data.get('n_samples', 0):,}")
        print(f"  Mean Fold DA: {period_data.get('mean_fold_DA', 'N/A')}  Std: {period_data.get('std_fold_DA', 'N/A')}")
        print(f"  Min DA: {period_data.get('min_fold_DA', 'N/A')}  Max DA: {period_data.get('max_fold_DA', 'N/A')}")
        print(f"{'=' * 80}")

        if 'tiers' not in period_data:
            print("  [insufficient data]")
            continue

        for tier_name, tier_data in period_data['tiers'].items():
            t1 = tier_data['tier1_signal']
            t2 = tier_data['tier2_tradeability']
            t3 = tier_data['tier3_profit']
            if t1 is None:
                continue
            sc_g = t3['scenarios']['gross_no_cost']
            sc_l = t3['scenarios']['low_cost_0.5t']
            print(f"  {tier_name:<10} DA={t1['DA']:.4f}  IC={t1['IC_spearman']:.4f}  WR={t2['win_rate']:.4f}  Sortino(gross)={sc_g['sortino']:.4f}  PF(gross)={sc_g['profit_factor']:.3f}  PF(0.5t)={sc_l['profit_factor']:.3f}  GrossPnL={sc_g['total_pnl_ticks']:.0f}t  n={t1['n']:,}")

    results["temporal_decay"] = temporal

    # ========== TEMPORAL COMPARISON TABLE ==========
    print("\n" + "=" * 100)
    print("TEMPORAL COMPARISON: KEY METRICS ACROSS PERIODS")
    print("=" * 100)

    comparison_tiers = ["All", "Top10%", "Top5%", "Top1%"]
    for tier_name in comparison_tiers:
        print(f"\n  === {tier_name} ===")
        print(f"  {'Period':<28} {'DA':>7} {'IC':>7} {'WR':>7} {'Sort(G)':>8} {'PF(G)':>7} {'PF(0.5)':>8} {'PF(2t)':>7} {'GrossPnL':>10} {'n':>8}")
        print(f"  {'-'*100}")
        for period_name in TEMPORAL_PERIODS.keys():
            pdata = temporal.get(period_name, {})
            tiers = pdata.get('tiers', {})
            td = tiers.get(tier_name, {})
            t1 = td.get('tier1_signal', {})
            t2 = td.get('tier2_tradeability', {})
            t3 = td.get('tier3_profit', {})
            if not t1:
                print(f"  {period_name:<28} [no data]")
                continue
            sc_g = t3['scenarios']['gross_no_cost']
            sc_l = t3['scenarios']['low_cost_0.5t']
            sc_f = t3['scenarios']['full_cost_2t']
            print(f"  {period_name:<28} {t1['DA']:>7.4f} {t1['IC_spearman']:>7.4f} {t2['win_rate']:>7.4f} {sc_g['sortino']:>8.4f} {sc_g['profit_factor']:>7.3f} {sc_l['profit_factor']:>8.3f} {sc_f['profit_factor']:>7.3f} {sc_g['total_pnl_ticks']:>10.0f} {t1['n']:>8,}")

    # ========== PER-FOLD DA TREND ==========
    print("\n" + "=" * 100)
    print("PER-FOLD DA TREND (all 143 folds)")
    print("=" * 100)

    fold_das = [fm['da'] for fm in fold_meta]
    # Print in blocks of 10
    for i in range(0, len(fold_meta), 10):
        batch = fold_meta[i:i+10]
        das_str = "  ".join([f"{fm['da']:.3f}" for fm in batch])
        print(f"  Folds {i:>3}-{min(i+9, len(fold_meta)-1):>3}: {das_str}")

    # Rolling 10-fold average
    print(f"\n  Rolling 10-fold average DA:")
    for i in range(0, len(fold_das) - 9, 10):
        window = fold_das[i:i+10]
        avg = np.mean(window)
        bar = "#" * int(avg * 100 - 45)  # visual bar (centered around 50%)
        print(f"  Folds {i:>3}-{i+9:>3}: {avg:.4f}  {'|' if avg < 0.50 else ' '}{bar}")

    results["per_fold_da"] = fold_das

    # ========== SAVE ==========
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to {OUTPUT_FILE}")
    print(f"Total samples analyzed: {len(labels):,}")


if __name__ == "__main__":
    main()
