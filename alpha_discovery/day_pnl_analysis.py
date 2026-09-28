"""
Day-by-Day PnL Analysis for cancel_asym_chain signal
Parameters: threshold=3.5, hold=3000 bars (300s), no trail, market orders
ES Futures: TICK=0.25, TICK_VAL=$12.50, Commission=$3.00 RT = 0.24 ticks
"""

import numpy as np
import glob
import os
import re
import argparse
from scipy import stats
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────
SIGNAL_DIR   = r"C:\Users\Footb\Documents\Github\Lvl3Quant\data\processed\signal_predictions"
FEATURES_DIR = r"C:\Users\Footb\Documents\Github\Lvl3Quant\data\processed\mbo_features_cache"

THRESH     = 3.5
HOLD       = 3000    # bars
TRAIL      = 0       # ticks (disabled)
COOLDOWN   = 10      # bars between trades
TICK       = 0.25
TICK_VAL   = 12.50
COMM_TICKS = 0.188   # commission per side in ticks ($2.35/side = $4.70 RT, HC #52)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def simulate_day(mid: np.ndarray, spread: np.ndarray, signal: np.ndarray) -> dict:
    """
    Market-order simulation for a single day.
    Entry: abs(signal) > THRESH after COOLDOWN bars with no open position
    Exit : after HOLD bars (max hold) — no trailing stop
    Cost : spread/2 per side + COMM_TICKS per side
    """
    n = len(mid)
    trades = []
    in_trade  = False
    entry_bar = -9999
    entry_px  = 0.0
    direction = 0          # +1 long, -1 short
    last_exit = -COOLDOWN - 1

    for i in range(n):
        if in_trade:
            hold_bars = i - entry_bar
            exit_now  = (hold_bars >= HOLD)

            if exit_now:
                # Exit at market: long sells at mid - spread/2, short buys at mid + spread/2
                if direction == 1:
                    exit_px = mid[i] - spread[i] / 2.0
                else:
                    exit_px = mid[i] + spread[i] / 2.0

                # PnL: entry/exit prices already include spread (buy@ask, sell@bid)
                # So (exit_px - entry_px) already captures both half-spreads
                # Only need to subtract RT commission (0.24 ticks = $3.00)
                pnl_ticks  = direction * (exit_px - entry_px) / TICK - COMM_TICKS
                pnl_dollars = pnl_ticks * TICK_VAL

                trades.append({
                    "entry_bar": entry_bar,
                    "exit_bar": i,
                    "direction": direction,
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "pnl_ticks": pnl_ticks,
                    "pnl_dollars": pnl_dollars,
                })
                in_trade  = False
                last_exit = i

        else:
            # Check cooldown
            if (i - last_exit) < COOLDOWN:
                continue
            sig = signal[i]
            if abs(sig) > THRESH:
                direction = 1 if sig > 0 else -1
                # Entry at market: long buys at mid + spread/2, short sells at mid - spread/2
                if direction == 1:
                    entry_px = mid[i] + spread[i] / 2.0
                else:
                    entry_px = mid[i] - spread[i] / 2.0
                in_trade  = True
                entry_bar = i

    # Close any open trade at end of day at mid (no spread penalty for EOD)
    if in_trade:
        exit_px = mid[-1]
        pnl_ticks   = direction * (exit_px - entry_px) / TICK - COMM_TICKS
        pnl_dollars = pnl_ticks * TICK_VAL
        trades.append({
            "entry_bar": entry_bar,
            "exit_bar": n - 1,
            "direction": direction,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "pnl_ticks": pnl_ticks,
            "pnl_dollars": pnl_dollars,
            "eod_close": True,
        })

    total_pnl = sum(t["pnl_dollars"] for t in trades)
    wins      = sum(1 for t in trades if t["pnl_dollars"] > 0)
    n_trades  = len(trades)
    win_rate  = wins / n_trades if n_trades > 0 else float("nan")

    return {
        "pnl": total_pnl,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "trades": trades,
    }


def bucket_label(pnl):
    if pnl < -200:
        return "< -$200"
    elif pnl < -100:
        return "-$200 to -$100"
    elif pnl < 0:
        return "-$100 to $0"
    elif pnl < 100:
        return "$0 to $100"
    elif pnl < 200:
        return "$100 to $200"
    else:
        return "> $200"

BUCKET_ORDER = ["< -$200", "-$200 to -$100", "-$100 to $0", "$0 to $100", "$100 to $200", "> $200"]


def running_max_drawdown(pnl_series):
    """Maximum peak-to-trough drawdown in cumulative PnL."""
    cumulative = np.cumsum(pnl_series)
    peak = cumulative[0]
    max_dd = 0.0
    for v in cumulative:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return max_dd


def streak_analysis(binary: list):
    """Return max win streak, max loss streak."""
    max_win = max_loss = cur_win = cur_loss = 0
    for v in binary:
        if v:
            cur_win  += 1
            cur_loss  = 0
        else:
            cur_loss += 1
            cur_win   = 0
        max_win  = max(max_win,  cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default=None,
                       help='Only include days >= this date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='Only include days <= this date (YYYY-MM-DD)')
    parser.add_argument('--signal', type=str, default='cancel_asym_chain',
                       help='Signal name to analyze')
    parser.add_argument('--thresh', type=float, default=None,
                       help='Override threshold (default: use THRESH constant)')
    parser.add_argument('--hold', type=int, default=None,
                       help='Override hold bars (default: use HOLD constant)')
    parser.add_argument('--flip', action='store_true',
                       help='Flip signal direction (inverse test)')
    args = parser.parse_args()

    # Allow overrides
    global THRESH, HOLD
    if args.thresh is not None:
        THRESH = args.thresh
    if args.hold is not None:
        HOLD = args.hold

    label = f"{args.signal}  |  thresh={THRESH}  |  hold={HOLD}  |  market orders"
    if args.start_date:
        label += f"  |  from {args.start_date}"
    if args.end_date:
        label += f"  |  to {args.end_date}"

    print("=" * 70)
    print(label)
    print("=" * 70)

    # Gather files
    sig_files = sorted(glob.glob(os.path.join(SIGNAL_DIR, f"{args.signal}_*.npz")))
    if not sig_files:
        print("ERROR: No signal files found!")
        return

    day_results = []
    missing     = []

    for sig_path in sig_files:
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})\.npz", sig_path)
        if not date_match:
            continue
        date = date_match.group(1)

        # Date filtering
        if args.start_date and date < args.start_date:
            continue
        if args.end_date and date > args.end_date:
            continue

        feat_path = os.path.join(FEATURES_DIR, f"{date}_mbo_features.npz")
        if not os.path.exists(feat_path):
            missing.append(date)
            continue

        sig_data  = np.load(sig_path)
        feat_data = np.load(feat_path)

        signal = sig_data["predictions"].astype(np.float64)
        if args.flip:
            signal = -signal  # Flip direction for inverse test
        feats  = feat_data["mbo_features"]
        mid    = feats[:, 0].astype(np.float64)
        spread = feats[:, 1].astype(np.float64)

        # Align lengths (take min)
        n = min(len(signal), len(mid))
        signal = signal[:n]
        mid    = mid[:n]
        spread = spread[:n]

        result = simulate_day(mid, spread, signal)
        result["date"] = date
        day_results.append(result)

    if missing:
        print(f"WARNING: {len(missing)} feature files missing for dates: {missing[:5]} ...")

    if not day_results:
        print("ERROR: No days simulated!")
        return

    # ─── Per-day table ────────────────────────────────────────────────────────
    print(f"\n{'Date':<14}{'PnL ($)':>12}{'Trades':>8}{'Win%':>8}")
    print("-" * 44)

    pnl_list   = []
    trade_list = []
    wr_list    = []

    for r in day_results:
        pnl_list.append(r["pnl"])
        trade_list.append(r["n_trades"])
        if not np.isnan(r["win_rate"]):
            wr_list.append(r["win_rate"])

        wr_str = f"{r['win_rate']*100:.1f}%" if not np.isnan(r["win_rate"]) else "  N/A"
        print(f"{r['date']:<14}{r['pnl']:>12.2f}{r['n_trades']:>8}{wr_str:>8}")

    pnl_arr = np.array(pnl_list)

    # ─── Summary statistics ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    mean_pnl   = np.mean(pnl_arr)
    median_pnl = np.median(pnl_arr)
    std_pnl    = np.std(pnl_arr, ddof=1)
    skew_pnl   = stats.skew(pnl_arr)
    kurt_pnl   = stats.kurtosis(pnl_arr)
    max_dd     = running_max_drawdown(pnl_arr)
    sharpe     = (mean_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0
    pos_days   = np.sum(pnl_arr > 0)
    total_days = len(pnl_arr)
    t_stat, p_val = stats.ttest_1samp(pnl_arr, 0)

    print(f"  Days simulated  : {total_days}")
    print(f"  Mean PnL/day    : ${mean_pnl:.2f}")
    print(f"  Median PnL/day  : ${median_pnl:.2f}")
    print(f"  Std PnL/day     : ${std_pnl:.2f}")
    print(f"  Skewness        : {skew_pnl:.3f}")
    print(f"  Kurtosis (exc.) : {kurt_pnl:.3f}")
    print(f"  Max Drawdown    : ${max_dd:.2f}")
    print(f"  Annual Sharpe   : {sharpe:.3f}")
    print(f"  Positive days   : {pos_days}/{total_days} ({pos_days/total_days*100:.1f}%)")
    print(f"  t-stat vs 0     : {t_stat:.3f}  (p={p_val:.4f})")
    print(f"  Mean trades/day : {np.mean(trade_list):.1f}")
    print(f"  Mean win rate   : {np.mean(wr_list)*100:.1f}%" if wr_list else "  Mean win rate : N/A")

    # ─── Distribution buckets ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PnL DISTRIBUTION")
    print("=" * 70)
    bucket_counts = defaultdict(int)
    for p in pnl_arr:
        bucket_counts[bucket_label(p)] += 1

    for b in BUCKET_ORDER:
        cnt = bucket_counts[b]
        bar = "#" * cnt
        print(f"  {b:<22}: {cnt:>3}  {bar}")

    # ─── Top/Bottom 5 days ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP 5 DAYS")
    print("=" * 70)
    sorted_results = sorted(day_results, key=lambda x: x["pnl"], reverse=True)
    for r in sorted_results[:5]:
        wr_str = f"{r['win_rate']*100:.1f}%" if not np.isnan(r["win_rate"]) else "N/A"
        print(f"  {r['date']}  ${r['pnl']:>9.2f}   {r['n_trades']} trades  {wr_str} win")

    print("\n" + "=" * 70)
    print("BOTTOM 5 DAYS")
    print("=" * 70)
    for r in sorted_results[-5:]:
        wr_str = f"{r['win_rate']*100:.1f}%" if not np.isnan(r["win_rate"]) else "N/A"
        print(f"  {r['date']}  ${r['pnl']:>9.2f}   {r['n_trades']} trades  {wr_str} win")

    # ─── Clustering / streak analysis ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CLUSTERING ANALYSIS")
    print("=" * 70)
    binary = [1 if p > 0 else 0 for p in pnl_arr]
    max_win_streak, max_loss_streak = streak_analysis(binary)

    # Runs test for randomness
    runs_result = stats.runs_test(binary) if hasattr(stats, "runs_test") else None

    print(f"  Max consecutive winning days  : {max_win_streak}")
    print(f"  Max consecutive losing days   : {max_loss_streak}")

    # Autocorrelation lag-1 of daily PnL
    if len(pnl_arr) > 2:
        ac1 = np.corrcoef(pnl_arr[:-1], pnl_arr[1:])[0, 1]
        print(f"  Lag-1 autocorrelation (PnL)   : {ac1:.4f}")

    # Correlation of rank position with PnL (early vs late drift)
    rank = np.arange(len(pnl_arr))
    trend_corr, trend_p = stats.pearsonr(rank, pnl_arr)
    print(f"  Time-trend correlation        : {trend_corr:.4f}  (p={trend_p:.4f})")
    if trend_p < 0.05:
        direction_str = "degrading" if trend_corr < 0 else "improving"
        print(f"  ** SIGNIFICANT TREND: signal is {direction_str} over time! **")

    # Impact of top N days on total PnL
    total_pnl = pnl_arr.sum()
    print(f"\n  Total PnL (all {total_days} days)         : ${total_pnl:.2f}")
    for n_top in [1, 3, 5]:
        top_pnl = np.sort(pnl_arr)[::-1][:n_top].sum()
        pct     = top_pnl / total_pnl * 100 if total_pnl != 0 else float("nan")
        pnl_excl = total_pnl - top_pnl
        print(f"  Without top {n_top} days               : ${pnl_excl:.2f}  ({pct:.1f}% concentration)")

    # Same for bottom days
    print()
    for n_bot in [1, 3, 5]:
        bot_pnl  = np.sort(pnl_arr)[:n_bot].sum()
        pnl_excl = total_pnl - bot_pnl
        pct      = abs(bot_pnl) / abs(total_pnl) * 100 if total_pnl != 0 else float("nan")
        print(f"  Without bottom {n_bot} days            : ${pnl_excl:.2f}  (worst contributed {pct:.1f}%)")

    # ─── Regime check: first half vs second half ───────────────────────────────
    print("\n" + "=" * 70)
    print("STABILITY: FIRST HALF vs SECOND HALF")
    print("=" * 70)
    mid_idx = total_days // 2
    first   = pnl_arr[:mid_idx]
    second  = pnl_arr[mid_idx:]
    print(f"  First  {len(first)} days: mean=${np.mean(first):.2f}  std=${np.std(first, ddof=1):.2f}  pos={np.sum(first>0)}/{len(first)}")
    print(f"  Second {len(second)} days: mean=${np.mean(second):.2f}  std=${np.std(second, ddof=1):.2f}  pos={np.sum(second>0)}/{len(second)}")
    _, p_half = stats.ttest_ind(first, second)
    print(f"  t-test (equal means): p={p_half:.4f}" + ("  ** SIGNIFICANT REGIME CHANGE **" if p_half < 0.05 else "  (no significant change)"))

    # ─── Assessment ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)
    flags = []
    if abs(skew_pnl) > 1.0:
        flags.append(f"HIGH SKEW ({skew_pnl:.2f}) — returns are fat-tailed, outlier-driven")
    if mean_pnl > 0 and (total_pnl - np.sort(pnl_arr)[::-1][:3].sum()) < 0:
        flags.append("TOP 3 DAYS FLIP PROFITABILITY — dangerous concentration")
    if abs(ac1) > 0.3:
        flags.append(f"HIGH AUTOCORRELATION ({ac1:.3f}) — days cluster in regimes")
    if trend_p < 0.05:
        flags.append(f"SIGNIFICANT TIME TREND (corr={trend_corr:.3f}) — check for regime drift")
    if sharpe < 0.5:
        flags.append(f"LOW SHARPE ({sharpe:.3f}) — insufficient risk-adjusted return")
    if p_val > 0.1:
        flags.append(f"WEAK STATISTICAL SIGNIFICANCE (p={p_val:.3f}) — mean may not be reliably > 0")

    if not flags:
        print("  CLEAN: No major concentration or clustering issues detected.")
    else:
        for f in flags:
            print(f"  WARNING: {f}")

    print("\n  Done.")
    return {
        "mean": mean_pnl,
        "median": median_pnl,
        "std": std_pnl,
        "sharpe": sharpe,
        "pos_days": pos_days,
        "total_days": total_days,
        "skew": skew_pnl,
        "max_dd": max_dd,
        "t_stat": t_stat,
        "p_val": p_val,
        "ac1": ac1,
        "flags": flags,
        "pnl_arr": pnl_arr,
        "day_results": day_results,
    }


if __name__ == "__main__":
    main()
