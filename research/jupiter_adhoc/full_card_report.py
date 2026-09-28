#!/usr/bin/env python3
"""
Full comprehensive card report: runs fill_sim for 10 card configs across all OOT dates,
collects per-trade data, and computes detailed analytics (A-I sections).
"""
import sys, os, json, subprocess, tempfile, math, statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, date
from collections import defaultdict
import numpy as np

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ── Paths ──
FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR  = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions"
OUT_DIR  = "/home/jupiter/Lvl3Quant/data/processed/card_full_report"
WORKERS  = 14

# ── OOT date range ──
OOT_START = date(2025, 12, 1)
OOT_END   = date(2026, 3, 8)

# ── Card configs ──
# Each card: (name, pred_suffix, fill_sim extra args, post_filter)
# post_filter: None, "BUY", or "SELL"
CARDS = [
    {
        "name": "Card1",
        "pred": "book_predstdExit_conv1.5_vol50",
        "args": ["--signal-threshold", "0.1", "--take-profit-ticks", "8", "--hold-ms", "7200000"],
        "filter": None,
        "reuse": None,
    },
    {
        "name": "Card2",
        "pred": "book_predstdExit_conv1.5_vol50",
        "args": ["--signal-threshold", "0.5", "--take-profit-ticks", "15", "--hold-ms", "7200000"],
        "filter": None,
        "reuse": None,
    },
    {
        "name": "Card3",
        "pred": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "args": ["--signal-threshold", "0.3", "--take-profit-ticks", "10", "--hold-ms", "3600000",
                 "--conviction-exit-bars", "100", "--conviction-exit-mag", "0.8"],
        "filter": None,
        "reuse": None,
    },
    {
        "name": "Card4",
        "pred": "book_predstdExit_conv2.0_vol70",
        "args": ["--signal-threshold", "0.1", "--take-profit-ticks", "20", "--hold-ms", "7200000"],
        "filter": None,
        "reuse": None,
    },
    {
        "name": "Card5",
        "pred": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "args": ["--signal-threshold", "0.1", "--hold-ms", "3600000"],
        "filter": None,
        "reuse": None,
    },
    {
        "name": "Card6",
        "pred": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "args": ["--signal-threshold", "0.1", "--take-profit-ticks", "20", "--stop-loss-ticks", "25",
                 "--hold-ms", "3600000", "--conviction-exit-bars", "60", "--conviction-exit-mag", "1.5",
                 "--mae-exit-ticks", "25", "--mae-exit-hold-sec", "60"],
        "filter": None,
        "reuse": None,
    },
    {
        "name": "Card7",
        "pred": "smooth_smoothExit_conv1.5_ethr0.0_vol70",
        "args": ["--signal-threshold", "0.1", "--stop-loss-ticks", "20", "--hold-ms", "3600000"],
        "filter": None,
        "reuse": None,
    },
    {
        "name": "Card8L",
        "pred": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "args": ["--signal-threshold", "0.3", "--take-profit-ticks", "10", "--hold-ms", "3600000",
                 "--conviction-exit-bars", "100", "--conviction-exit-mag", "0.8"],
        "filter": "BUY",
        "reuse": "Card3",
    },
    {
        "name": "Card9S",
        "pred": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "args": ["--signal-threshold", "0.1", "--hold-ms", "3600000"],
        "filter": "SELL",
        "reuse": "Card5",
    },
    {
        "name": "Card10L",
        "pred": "book_predstdExit_conv2.0_vol70",
        "args": ["--signal-threshold", "0.1", "--take-profit-ticks", "20", "--hold-ms", "7200000"],
        "filter": "BUY",
        "reuse": "Card4",
    },
]


def get_oot_dates():
    """Get all OOT dates that have both MBO and prediction files."""
    dates = []
    current = OOT_START
    while current <= OOT_END:
        ds = current.strftime("%Y%m%d")
        mbo = f"{MBO_DIR}/glbx-mdp3-{ds}.mbo.dbn.zst"
        if os.path.exists(mbo):
            dates.append(current)
        current += __import__('datetime').timedelta(days=1)
    return dates


def run_fill_sim(card_name, pred_suffix, extra_args, date_str_iso, date_str_num, out_path):
    """Run fill_sim_cli for one card on one date. Returns (card, date, out_path, success, error)."""
    mbo = f"{MBO_DIR}/glbx-mdp3-{date_str_num}.mbo.dbn.zst"
    pred = f"{PRED_DIR}/{date_str_iso}_{pred_suffix}.npz"

    if not os.path.exists(mbo):
        return (card_name, date_str_iso, out_path, False, f"MBO missing: {mbo}")
    if not os.path.exists(pred):
        return (card_name, date_str_iso, out_path, False, f"Pred missing: {pred}")

    cmd = [FILL_SIM, "--mbo-file", mbo, "--predictions", pred, "--output", out_path, "--quiet"] + extra_args

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return (card_name, date_str_iso, out_path, False, result.stderr[:200])
        return (card_name, date_str_iso, out_path, True, None)
    except Exception as e:
        return (card_name, date_str_iso, out_path, False, str(e))


def pct(arr, p):
    if not arr: return 0.0
    return float(np.percentile(arr, p))


def sharpe_from_daily(daily_pnls):
    if len(daily_pnls) < 2:
        return 0.0
    m = np.mean(daily_pnls)
    s = np.std(daily_pnls, ddof=1)
    if s == 0:
        return 0.0
    return float(m / s * math.sqrt(252))


def compute_analytics(card_name, all_trades, dates_with_trades, total_dates):
    """Compute full A-I analytics from list of trade dicts."""
    if not all_trades:
        return {"card": card_name, "error": "NO TRADES", "total_dates": total_dates}

    # Basics
    pnls = [t["pnl_dollars"] for t in all_trades]
    wins = [t for t in all_trades if t["pnl_dollars"] > 0]
    losses = [t for t in all_trades if t["pnl_dollars"] <= 0]
    win_pnls = [t["pnl_dollars"] for t in wins]
    loss_pnls = [t["pnl_dollars"] for t in losses]
    maes = [t["mae_ticks"] for t in all_trades]
    mfes = [t["mfe_ticks"] for t in all_trades]

    # Daily PnL
    daily = defaultdict(float)
    for t in all_trades:
        d = t["_date"]
        daily[d] += t["pnl_dollars"]
    # Include zero days
    for d in total_dates:
        if d not in daily:
            daily[d] = 0.0
    daily_vals = [daily[d] for d in sorted(daily.keys())]
    pos_days = sum(1 for v in daily_vals if v > 0)
    neg_days = sum(1 for v in daily_vals if v < 0)
    zero_days = sum(1 for v in daily_vals if v == 0)
    trading_days = len(daily_vals)

    # Max consecutive losing days
    max_consec_loss = 0
    cur_consec = 0
    for v in daily_vals:
        if v < 0:
            cur_consec += 1
            max_consec_loss = max(max_consec_loss, cur_consec)
        else:
            cur_consec = 0

    # Max drawdown
    cum = np.cumsum(daily_vals)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(np.min(dd))
    # Drawdown duration
    in_dd = False
    max_dd_dur = 0
    cur_dur = 0
    for i, v in enumerate(dd):
        if v < 0:
            cur_dur += 1
            max_dd_dur = max(max_dd_dur, cur_dur)
        else:
            cur_dur = 0

    total_pnl = sum(pnls)
    sharpe = sharpe_from_daily(daily_vals)
    n_trades = len(all_trades)
    n_wins = len(wins)
    wr = n_wins / n_trades if n_trades else 0
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Hold times in seconds
    holds = [t["hold_duration_ns"] / 1e9 for t in all_trades]
    win_holds = [t["hold_duration_ns"] / 1e9 for t in wins]
    loss_holds = [t["hold_duration_ns"] / 1e9 for t in losses]

    # === SECTION A: Core Performance ===
    section_a = {
        "sharpe": round(sharpe, 4),
        "total_pnl": round(total_pnl, 2),
        "daily_avg_pnl": round(np.mean(daily_vals), 2),
        "trades": n_trades,
        "trades_per_day": round(n_trades / trading_days, 2) if trading_days else 0,
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 4) if pf != float('inf') else "inf",
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "wl_ratio": round(float(wl_ratio), 4) if wl_ratio != float('inf') else "inf",
        "best_trade": round(max(pnls), 2),
        "worst_trade": round(min(pnls), 2),
        "positive_days": pos_days,
        "negative_days": neg_days,
        "positive_day_pct": round(pos_days / (pos_days + neg_days) * 100, 1) if (pos_days + neg_days) > 0 else 0,
        "max_consec_losing_days": max_consec_loss,
        "max_drawdown_dollars": round(max_dd, 2),
        "max_drawdown_duration_days": max_dd_dur,
    }

    # === SECTION B: MAE Analysis ===
    win_maes = [t["mae_ticks"] for t in wins]
    loss_maes = [t["mae_ticks"] for t in losses]

    mae_cutoffs = [10, 15, 20, 25, 30, 40, 50, 75, 100]
    mae_cutoff_analysis = {}
    for thresh in mae_cutoffs:
        losers_exceed = sum(1 for m in loss_maes if m >= thresh) / len(loss_maes) * 100 if loss_maes else 0
        winners_exceed = sum(1 for m in win_maes if m >= thresh) / len(win_maes) * 100 if win_maes else 0
        mae_cutoff_analysis[str(thresh)] = {
            "losers_exceed_pct": round(losers_exceed, 2),
            "winners_exceed_pct": round(winners_exceed, 2),
        }

    section_b = {
        "all": {"avg": round(np.mean(maes), 2), "P25": round(pct(maes, 25), 2), "P50": round(pct(maes, 50), 2),
                "P75": round(pct(maes, 75), 2), "P90": round(pct(maes, 90), 2), "P95": round(pct(maes, 95), 2),
                "P99": round(pct(maes, 99), 2), "worst": round(max(maes), 2)},
        "winners": {"avg": round(np.mean(win_maes), 2) if win_maes else 0,
                    "P50": round(pct(win_maes, 50), 2) if win_maes else 0,
                    "P95": round(pct(win_maes, 95), 2) if win_maes else 0},
        "losers": {"avg": round(np.mean(loss_maes), 2) if loss_maes else 0,
                   "P50": round(pct(loss_maes, 50), 2) if loss_maes else 0,
                   "P95": round(pct(loss_maes, 95), 2) if loss_maes else 0},
        "cutoff_analysis": mae_cutoff_analysis,
    }

    # === SECTION C: MFE Analysis ===
    win_mfes = [t["mfe_ticks"] for t in wins]
    loss_mfes = [t["mfe_ticks"] for t in losses]

    # Find TP from config (approximate from card name or from trade exit reasons)
    tp_ticks = None
    tp_trades = [t for t in all_trades if t.get("exit_reason") == "TakeProfit"]
    if tp_trades:
        # TP is the mfe of TP exits (should all be the same)
        tp_vals = [t["mfe_ticks"] for t in tp_trades]
        tp_ticks = round(np.median(tp_vals), 1)

    mfe_util = round(tp_ticks / np.mean(win_mfes) * 100, 1) if (tp_ticks and win_mfes and np.mean(win_mfes) > 0) else None

    section_c = {
        "all": {"avg": round(np.mean(mfes), 2), "P25": round(pct(mfes, 25), 2), "P50": round(pct(mfes, 50), 2),
                "P75": round(pct(mfes, 75), 2), "P90": round(pct(mfes, 90), 2), "P95": round(pct(mfes, 95), 2),
                "P99": round(pct(mfes, 99), 2), "best": round(max(mfes), 2)},
        "winners": {"avg": round(np.mean(win_mfes), 2) if win_mfes else 0,
                    "P50": round(pct(win_mfes, 50), 2) if win_mfes else 0,
                    "P95": round(pct(win_mfes, 95), 2) if win_mfes else 0},
        "losers": {"avg": round(np.mean(loss_mfes), 2) if loss_mfes else 0,
                   "P50": round(pct(loss_mfes, 50), 2) if loss_mfes else 0,
                   "P95": round(pct(loss_mfes, 95), 2) if loss_mfes else 0},
        "tp_ticks": tp_ticks,
        "mfe_utilization_pct": mfe_util,
    }

    # === SECTION D: Hold Time Analysis ===
    hold_buckets = [
        ("0-60s", 0, 60),
        ("60-300s", 60, 300),
        ("300-900s", 300, 900),
        ("900-1800s", 900, 1800),
        ("1800-3600s", 1800, 3600),
        ("3600-7200s", 3600, 7200),
    ]
    hold_analysis = {}
    for label, lo, hi in hold_buckets:
        bucket_trades = [t for t in all_trades if lo <= t["hold_duration_ns"]/1e9 < hi]
        if bucket_trades:
            b_pnls = [t["pnl_dollars"] for t in bucket_trades]
            b_wins = sum(1 for p in b_pnls if p > 0)
            hold_analysis[label] = {
                "count": len(bucket_trades),
                "win_rate": round(b_wins / len(bucket_trades), 4),
                "avg_pnl": round(np.mean(b_pnls), 2),
                "total_pnl": round(sum(b_pnls), 2),
            }
        else:
            hold_analysis[label] = {"count": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}

    section_d = {
        "winners_avg_hold_sec": round(np.mean(win_holds), 1) if win_holds else 0,
        "losers_avg_hold_sec": round(np.mean(loss_holds), 1) if loss_holds else 0,
        "buckets": hold_analysis,
    }

    # === SECTION E: Edge Decay Curve ===
    # Approximate: for each hold limit, cap the hold at that limit and compute PnL
    # We use linear interpolation: if trade held T seconds and exited with pnl P,
    # at time t < T, estimated pnl = P * t / T (rough approximation)
    # For trades shorter than the limit, use actual PnL
    edge_limits = [60, 120, 300, 600, 900, 1800, 3600, 7200]
    edge_decay = {}
    for limit_sec in edge_limits:
        capped_daily = defaultdict(float)
        for t in all_trades:
            hold_sec = t["hold_duration_ns"] / 1e9
            if hold_sec <= limit_sec:
                capped_daily[t["_date"]] += t["pnl_dollars"]
            else:
                # Linear interpolation
                ratio = limit_sec / hold_sec
                capped_daily[t["_date"]] += t["pnl_dollars"] * ratio
        for d in total_dates:
            if d not in capped_daily:
                capped_daily[d] = 0.0
        vals = [capped_daily[d] for d in sorted(capped_daily.keys())]
        label = f"{limit_sec//60}min" if limit_sec >= 60 else f"{limit_sec}s"
        edge_decay[label] = round(sharpe_from_daily(vals), 4)

    section_e = {"hold_limit_sharpe": edge_decay}

    # === SECTION F: Time of Day ===
    tod_buckets = [
        ("09:30-10:30", 9*3600+30*60, 10*3600+30*60),
        ("10:30-11:30", 10*3600+30*60, 11*3600+30*60),
        ("11:30-12:30", 11*3600+30*60, 12*3600+30*60),
        ("12:30-13:30", 12*3600+30*60, 13*3600+30*60),
        ("13:30-14:30", 13*3600+30*60, 14*3600+30*60),
        ("14:30-15:30", 14*3600+30*60, 15*3600+30*60),
        ("15:30-16:00", 15*3600+30*60, 16*3600),
    ]

    section_f = {}
    for label, lo_sec, hi_sec in tod_buckets:
        bucket_trades = []
        for t in all_trades:
            # fill_time_ns is epoch nanoseconds, convert to ET time of day
            fill_ts = t["fill_time_ns"] / 1e9
            # Convert to datetime and get hour/min in ET (assume data is already ET-aligned)
            dt = datetime.utcfromtimestamp(fill_ts)
            # ES futures: times are in UTC, ET = UTC-5 (EST) or UTC-4 (EDT)
            # OOT period Dec-Mar is EST (UTC-5)
            et_dt = datetime.utcfromtimestamp(fill_ts - 5*3600)
            tod_sec = et_dt.hour * 3600 + et_dt.minute * 60 + et_dt.second
            if lo_sec <= tod_sec < hi_sec:
                bucket_trades.append(t)
        if bucket_trades:
            b_pnls = [t["pnl_dollars"] for t in bucket_trades]
            b_wins = sum(1 for p in b_pnls if p > 0)
            section_f[label] = {
                "count": len(bucket_trades),
                "win_rate": round(b_wins / len(bucket_trades), 4),
                "avg_pnl": round(np.mean(b_pnls), 2),
            }
        else:
            section_f[label] = {"count": 0, "win_rate": 0, "avg_pnl": 0}

    # === SECTION G: Side Analysis ===
    buys = [t for t in all_trades if t["side"] == "BUY"]
    sells = [t for t in all_trades if t["side"] == "SELL"]

    def side_stats(trades, label, all_dates):
        if not trades:
            return {"sharpe": 0, "total_pnl": 0, "trades": 0, "win_rate": 0, "avg_pnl": 0}
        pnl_list = [t["pnl_dollars"] for t in trades]
        sd = defaultdict(float)
        for t in trades:
            sd[t["_date"]] += t["pnl_dollars"]
        for d in all_dates:
            if d not in sd:
                sd[d] = 0.0
        sd_vals = [sd[d] for d in sorted(sd.keys())]
        w = sum(1 for p in pnl_list if p > 0)
        return {
            "sharpe": round(sharpe_from_daily(sd_vals), 4),
            "total_pnl": round(sum(pnl_list), 2),
            "trades": len(trades),
            "win_rate": round(w / len(trades), 4),
            "avg_pnl": round(np.mean(pnl_list), 2),
        }

    section_g = {
        "BUY": side_stats(buys, "BUY", total_dates),
        "SELL": side_stats(sells, "SELL", total_dates),
    }

    # === SECTION H: Signal Strength ===
    sig_buckets = [
        ("0.0-0.5", 0.0, 0.5),
        ("0.5-1.0", 0.5, 1.0),
        ("1.0-1.5", 1.0, 1.5),
        ("1.5-2.0", 1.5, 2.0),
        ("2.0+", 2.0, 1e10),
    ]
    section_h = {}
    for label, lo, hi in sig_buckets:
        bucket = [t for t in all_trades if lo <= abs(t.get("signal_strength", 0)) < hi]
        if bucket:
            b_pnls = [t["pnl_dollars"] for t in bucket]
            b_maes = [t["mae_ticks"] for t in bucket]
            b_wins = sum(1 for p in b_pnls if p > 0)
            section_h[label] = {
                "count": len(bucket),
                "win_rate": round(b_wins / len(bucket), 4),
                "avg_pnl": round(np.mean(b_pnls), 2),
                "avg_mae": round(np.mean(b_maes), 2),
            }
        else:
            section_h[label] = {"count": 0, "win_rate": 0, "avg_pnl": 0, "avg_mae": 0}

    # === SECTION I: Exit Reason Breakdown ===
    exit_counts = defaultdict(int)
    for t in all_trades:
        exit_counts[t.get("exit_reason", "Unknown")] += 1
    section_i = {}
    for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
        section_i[reason] = {"count": count, "pct": round(count / n_trades * 100, 2)}

    return {
        "card": card_name,
        "total_dates": len(total_dates),
        "dates_with_trades": dates_with_trades,
        "A_core_performance": section_a,
        "B_mae_analysis": section_b,
        "C_mfe_analysis": section_c,
        "D_hold_time": section_d,
        "E_edge_decay": section_e,
        "F_time_of_day": section_f,
        "G_side_analysis": section_g,
        "H_signal_strength": section_h,
        "I_exit_reasons": section_i,
    }


def format_report(r):
    """Format one card's analytics into a text report."""
    lines = []
    name = r["card"]
    lines.append(f"\n{'='*80}")
    lines.append(f"  {name}")
    lines.append(f"{'='*80}")

    if "error" in r:
        lines.append(f"  ERROR: {r['error']}")
        return "\n".join(lines)

    a = r["A_core_performance"]
    lines.append(f"\n  A. CORE PERFORMANCE")
    lines.append(f"  {'─'*40}")
    lines.append(f"  Sharpe:        {a['sharpe']:>10}")
    lines.append(f"  Total PnL:     ${a['total_pnl']:>10,.2f}")
    lines.append(f"  Daily Avg PnL: ${a['daily_avg_pnl']:>10,.2f}")
    lines.append(f"  Trades:        {a['trades']:>10}")
    lines.append(f"  Trades/Day:    {a['trades_per_day']:>10}")
    lines.append(f"  Win Rate:      {a['win_rate']*100:>9.1f}%")
    lines.append(f"  Profit Factor: {a['profit_factor']:>10}")
    lines.append(f"  Avg Win:       ${a['avg_win']:>10,.2f}")
    lines.append(f"  Avg Loss:      ${a['avg_loss']:>10,.2f}")
    lines.append(f"  W/L Ratio:     {a['wl_ratio']:>10}")
    lines.append(f"  Best Trade:    ${a['best_trade']:>10,.2f}")
    lines.append(f"  Worst Trade:   ${a['worst_trade']:>10,.2f}")
    lines.append(f"  Pos Days:      {a['positive_days']:>10}  ({a['positive_day_pct']:.1f}%)")
    lines.append(f"  Neg Days:      {a['negative_days']:>10}")
    lines.append(f"  Max Consec Loss Days: {a['max_consec_losing_days']:>5}")
    lines.append(f"  Max Drawdown:  ${a['max_drawdown_dollars']:>10,.2f}")
    lines.append(f"  Max DD Duration: {a['max_drawdown_duration_days']:>5} days")

    b = r["B_mae_analysis"]
    lines.append(f"\n  B. MAE ANALYSIS (ticks)")
    lines.append(f"  {'─'*40}")
    lines.append(f"  All:     avg={b['all']['avg']:>6.1f}  P25={b['all']['P25']:>5.1f}  P50={b['all']['P50']:>5.1f}  P75={b['all']['P75']:>5.1f}  P90={b['all']['P90']:>5.1f}  P95={b['all']['P95']:>5.1f}  P99={b['all']['P99']:>5.1f}  worst={b['all']['worst']:>5.1f}")
    lines.append(f"  Winners: avg={b['winners']['avg']:>6.1f}  P50={b['winners']['P50']:>5.1f}  P95={b['winners']['P95']:>5.1f}")
    lines.append(f"  Losers:  avg={b['losers']['avg']:>6.1f}  P50={b['losers']['P50']:>5.1f}  P95={b['losers']['P95']:>5.1f}")
    lines.append(f"  MAE Cutoff (% exceeding):")
    lines.append(f"  {'Thresh':>6} | {'Losers%':>8} | {'Winners%':>8}")
    for thresh, vals in b["cutoff_analysis"].items():
        lines.append(f"  {thresh:>6} | {vals['losers_exceed_pct']:>7.1f}% | {vals['winners_exceed_pct']:>7.1f}%")

    c = r["C_mfe_analysis"]
    lines.append(f"\n  C. MFE ANALYSIS (ticks)")
    lines.append(f"  {'─'*40}")
    lines.append(f"  All:     avg={c['all']['avg']:>6.1f}  P25={c['all']['P25']:>5.1f}  P50={c['all']['P50']:>5.1f}  P75={c['all']['P75']:>5.1f}  P90={c['all']['P90']:>5.1f}  P95={c['all']['P95']:>5.1f}  P99={c['all']['P99']:>5.1f}  best={c['all']['best']:>5.1f}")
    lines.append(f"  Winners: avg={c['winners']['avg']:>6.1f}  P50={c['winners']['P50']:>5.1f}  P95={c['winners']['P95']:>5.1f}")
    lines.append(f"  Losers:  avg={c['losers']['avg']:>6.1f}  P50={c['losers']['P50']:>5.1f}  P95={c['losers']['P95']:>5.1f}")
    if c["tp_ticks"]:
        lines.append(f"  TP={c['tp_ticks']} ticks, MFE utilization={c['mfe_utilization_pct']}%")

    d = r["D_hold_time"]
    lines.append(f"\n  D. HOLD TIME ANALYSIS")
    lines.append(f"  {'─'*40}")
    lines.append(f"  Winners avg hold: {d['winners_avg_hold_sec']:>8.1f}s")
    lines.append(f"  Losers avg hold:  {d['losers_avg_hold_sec']:>8.1f}s")
    lines.append(f"  {'Bucket':>14} | {'Count':>6} | {'WR':>6} | {'AvgPnL':>10} | {'TotalPnL':>10}")
    for bucket, vals in d["buckets"].items():
        lines.append(f"  {bucket:>14} | {vals['count']:>6} | {vals['win_rate']*100:>5.1f}% | ${vals['avg_pnl']:>9,.2f} | ${vals['total_pnl']:>9,.2f}")

    e = r["E_edge_decay"]
    lines.append(f"\n  E. EDGE DECAY CURVE (Sharpe at hold limit)")
    lines.append(f"  {'─'*40}")
    for limit, val in e["hold_limit_sharpe"].items():
        bar = "+" * max(0, int(val * 5)) if val > 0 else "-" * max(0, int(abs(val) * 5))
        lines.append(f"  {limit:>6}: {val:>8.4f}  {bar}")

    f = r["F_time_of_day"]
    lines.append(f"\n  F. TIME OF DAY (ET)")
    lines.append(f"  {'─'*40}")
    lines.append(f"  {'Window':>14} | {'Count':>6} | {'WR':>6} | {'AvgPnL':>10}")
    for window, vals in f.items():
        lines.append(f"  {window:>14} | {vals['count']:>6} | {vals['win_rate']*100:>5.1f}% | ${vals['avg_pnl']:>9,.2f}")

    g = r["G_side_analysis"]
    lines.append(f"\n  G. SIDE ANALYSIS")
    lines.append(f"  {'─'*40}")
    for side, vals in g.items():
        lines.append(f"  {side:>5}: Sharpe={vals['sharpe']:>7.4f}  PnL=${vals['total_pnl']:>10,.2f}  Trades={vals['trades']:>5}  WR={vals['win_rate']*100:>5.1f}%  AvgPnL=${vals['avg_pnl']:>8,.2f}")

    h = r["H_signal_strength"]
    lines.append(f"\n  H. SIGNAL STRENGTH (|z-score| at entry)")
    lines.append(f"  {'─'*40}")
    lines.append(f"  {'|z| Range':>10} | {'Count':>6} | {'WR':>6} | {'AvgPnL':>10} | {'AvgMAE':>6}")
    for bucket, vals in h.items():
        lines.append(f"  {bucket:>10} | {vals['count']:>6} | {vals['win_rate']*100:>5.1f}% | ${vals['avg_pnl']:>9,.2f} | {vals['avg_mae']:>5.1f}")

    i_data = r["I_exit_reasons"]
    lines.append(f"\n  I. EXIT REASON BREAKDOWN")
    lines.append(f"  {'─'*40}")
    for reason, vals in i_data.items():
        lines.append(f"  {reason:>20}: {vals['count']:>6} ({vals['pct']:>5.1f}%)")

    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="card_report_")

    # Get OOT dates
    dates = get_oot_dates()
    print(f"Found {len(dates)} OOT dates")
    date_strs_iso = [d.strftime("%Y-%m-%d") for d in dates]
    date_strs_num = [d.strftime("%Y%m%d") for d in dates]

    # Build task list: only unique fill_sim runs (skip reuse cards)
    tasks = []
    unique_cards = {}  # card_name -> True
    reuse_map = {}     # card_name -> source_card_name
    for card in CARDS:
        if card["reuse"]:
            reuse_map[card["name"]] = card["reuse"]
            continue
        unique_cards[card["name"]] = card
        for i, (d_iso, d_num) in enumerate(zip(date_strs_iso, date_strs_num)):
            out_path = os.path.join(tmp_dir, f"{card['name']}_{d_num}.json")
            tasks.append((card["name"], card["pred"], card["args"], d_iso, d_num, out_path))

    print(f"Submitting {len(tasks)} unique fill_sim tasks across {len(unique_cards)} cards ({WORKERS} workers)...")
    t0 = datetime.now()

    # Run fill_sim in parallel
    results = {}  # (card_name, date) -> out_path
    errors = []
    completed = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_fill_sim, *t): t for t in tasks}
        for future in as_completed(futures):
            card_name, d_iso, out_path, success, error = future.result()
            completed += 1
            if completed % 100 == 0:
                elapsed = (datetime.now() - t0).total_seconds()
                rate = completed / elapsed
                remaining = (len(tasks) - completed) / rate
                print(f"  Progress: {completed}/{len(tasks)} ({rate:.1f}/s, ~{remaining:.0f}s remaining)")
            if success:
                results[(card_name, d_iso)] = out_path
            else:
                errors.append(f"{card_name} {d_iso}: {error}")

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nfill_sim completed in {elapsed:.1f}s. {len(results)} successes, {len(errors)} errors.")
    if errors:
        print(f"First 10 errors:")
        for e in errors[:10]:
            print(f"  {e}")

    # Collect trades per card
    all_reports = []
    all_dates_set = set(date_strs_iso)

    for card in CARDS:
        card_name = card["name"]
        source = card["reuse"] or card_name
        side_filter = card["filter"]

        print(f"\nProcessing {card_name} (source={source}, filter={side_filter})...")

        all_trades = []
        dates_with = 0
        for d_iso in date_strs_iso:
            key = (source, d_iso)
            if key not in results:
                continue
            out_path = results[key]
            try:
                with open(out_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                continue

            trades = data.get("trades", [])
            if side_filter:
                trades = [t for t in trades if t["side"] == side_filter]

            if trades:
                dates_with += 1
            for t in trades:
                t["_date"] = d_iso
            all_trades.extend(trades)

        print(f"  {len(all_trades)} trades across {dates_with} active days")
        report = compute_analytics(card_name, all_trades, dates_with, all_dates_set)
        all_reports.append(report)

        # Save individual JSON
        card_json_path = os.path.join(OUT_DIR, f"{card_name}_report.json")
        with open(card_json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    # Save combined summary
    summary_path = os.path.join(OUT_DIR, "combined_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_reports, f, indent=2, default=str)

    # Print full report
    print("\n" + "="*80)
    print("  FULL CARD REPORT — 10 Cards × {} OOT Dates ({} to {})".format(
        len(dates), date_strs_iso[0], date_strs_iso[-1]))
    print("="*80)

    # Summary table first
    print(f"\n{'─'*100}")
    print(f"  SUMMARY TABLE")
    print(f"{'─'*100}")
    print(f"  {'Card':<10} | {'Sharpe':>7} | {'PnL':>12} | {'Trades':>6} | {'WR':>6} | {'PF':>7} | {'AvgWin':>8} | {'AvgLoss':>9} | {'MaxDD':>10} | {'Pos%':>5}")
    print(f"  {'─'*10}-+-{'─'*7}-+-{'─'*12}-+-{'─'*6}-+-{'─'*6}-+-{'─'*7}-+-{'─'*8}-+-{'─'*9}-+-{'─'*10}-+-{'─'*5}")
    for r in all_reports:
        if "error" in r:
            print(f"  {r['card']:<10} | {'ERROR':>7} |")
            continue
        a = r["A_core_performance"]
        print(f"  {r['card']:<10} | {a['sharpe']:>7.2f} | ${a['total_pnl']:>10,.2f} | {a['trades']:>6} | {a['win_rate']*100:>5.1f}% | {a['profit_factor']:>7} | ${a['avg_win']:>7,.2f} | ${a['avg_loss']:>8,.2f} | ${a['max_drawdown_dollars']:>9,.2f} | {a['positive_day_pct']:>4.0f}%")

    # Detailed per-card reports
    for r in all_reports:
        print(format_report(r))

    print(f"\n\nJSON reports saved to: {OUT_DIR}/")
    print(f"  Individual: {OUT_DIR}/<CardN>_report.json")
    print(f"  Combined:   {summary_path}")

    # Cleanup tmp
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
