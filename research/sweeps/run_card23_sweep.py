#!/usr/bin/env python3
"""Card 2 & Card 3 Full-Date Sweep with Deep Analysis.

Card 2: conv1.5_vol50 + TP15 + slN + wb50 + sig0.1 + lat50 + chase 1t/3r
Card 3: mom_emaExit conv0.3_ethr0.0_vol70 + TP15 + trail25 + wb50 + sig0.1 + lat50 + chase 1t/3r
"""
import sys, json, time, subprocess, os, math, statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

WORKERS = 14
LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
CARD2_OUT = LVL3_ROOT / "data" / "processed" / "card2_full"
CARD3_OUT = LVL3_ROOT / "data" / "processed" / "card3_full"
CARD2_OUT.mkdir(parents=True, exist_ok=True)
CARD3_OUT.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COMMISSION_RT = 4.70


def find_dates():
    """Find intersection of prediction dates and MBO dates for each card."""
    # Card 2: book_predstdExit_conv1.5_vol50
    card2_preds = {}
    card3_preds = {}
    mbo_dates = set()

    for f in sorted(PRED_DIR.glob("*_book_predstdExit_conv1.5_vol50.npz")):
        date = f.stem[:10]  # 2025-12-01
        card2_preds[date] = f

    for f in sorted(PRED_DIR.glob("*_mom_emaExit_conv0.3_ethr0.0_vol70.npz")):
        date = f.stem[:10]
        card3_preds[date] = f

    for f in sorted(MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst")):
        nodash = f.name.split("-")[2].split(".")[0]  # 20251201
        date = f"{nodash[:4]}-{nodash[4:6]}-{nodash[6:8]}"
        mbo_dates.add(date)

    card2_dates = sorted(set(card2_preds.keys()) & mbo_dates)
    card3_dates = sorted(set(card3_preds.keys()) & mbo_dates)

    return card2_dates, card3_dates, card2_preds, card3_preds


def get_mbo_path(date):
    nodash = date.replace("-", "")
    return MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn.zst"


def run_sim(cmd, out_path, label):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and Path(out_path).exists():
            return (label, True, None)
        return (label, False, r.stderr[:200] if r.stderr else "no output file")
    except subprocess.TimeoutExpired:
        return (label, False, "TIMEOUT")
    except Exception as e:
        return (label, False, str(e)[:200])


def build_card2_cmd(date, pred_path, out_path):
    return [
        str(BINARY),
        "--mbo-file", str(get_mbo_path(date)),
        "--predictions", str(pred_path),
        "--output", str(out_path),
        "--signal-threshold", "0.1",
        "--hold-ms", "3600000",
        "--max-wait-bars", "50",
        "--latency-ms", "50",
        "--chase-entry",
        "--chase-max-ticks", "1",
        "--chase-max-reprices", "3",
        "--take-profit-ticks", "15",
        "--quiet",
    ]


def build_card3_cmd(date, pred_path, out_path):
    return [
        str(BINARY),
        "--mbo-file", str(get_mbo_path(date)),
        "--predictions", str(pred_path),
        "--output", str(out_path),
        "--signal-threshold", "0.1",
        "--hold-ms", "3600000",
        "--max-wait-bars", "50",
        "--latency-ms", "50",
        "--chase-entry",
        "--chase-max-ticks", "1",
        "--chase-max-reprices", "3",
        "--take-profit-ticks", "15",
        "--trailing-ticks", "25",
        "--quiet",
    ]


def load_result(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None


def get_month(date_str):
    """Return regime label like Dec, Jan, Feb, Mar."""
    m = int(date_str[5:7])
    return {12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}.get(m, f"M{m}")


def analyze_card(card_name, out_dir, dates):
    """Deep analysis of a card's results."""
    results = []
    for date in sorted(dates):
        path = out_dir / f"{date}.json"
        data = load_result(path)
        if data is None:
            print(f"  WARNING: Missing result for {date}", flush=True)
            continue
        data["date"] = date
        data["month"] = get_month(date)
        results.append(data)

    if not results:
        return {"error": "No results found"}

    # === Aggregate Stats ===
    total_pnl_ticks = 0
    total_trades = 0
    total_wins = 0
    total_longs = 0
    total_shorts = 0
    long_pnl = 0
    short_pnl = 0
    daily_pnls = []
    daily_trades = []
    all_trade_pnls = []
    all_maes = []
    all_mfes = []
    all_hold_times = []
    regime_data = {}  # month -> {pnl, trades, wins}

    for r in results:
        trades = r.get("trades", [])
        day_pnl = 0
        day_wins = 0
        day_longs = 0
        day_shorts = 0
        day_long_pnl = 0
        day_short_pnl = 0

        for t in trades:
            pnl_ticks = t.get("pnl_ticks", 0)
            total_pnl_ticks += pnl_ticks
            day_pnl += pnl_ticks
            total_trades += 1

            if pnl_ticks > 0:
                total_wins += 1
                day_wins += 1

            side = t.get("side", "")
            if side == "long" or side == "Long":
                total_longs += 1
                day_longs += 1
                long_pnl += pnl_ticks
                day_long_pnl += pnl_ticks
            else:
                total_shorts += 1
                day_shorts += 1
                short_pnl += pnl_ticks
                day_short_pnl += pnl_ticks

            all_trade_pnls.append(pnl_ticks)

            mae = t.get("mae_ticks", None)
            mfe = t.get("mfe_ticks", None)
            if mae is not None:
                all_maes.append(mae)
            if mfe is not None:
                all_mfes.append(mfe)

            # Hold time
            entry_ts = t.get("entry_ts", 0)
            exit_ts = t.get("exit_ts", 0)
            if entry_ts and exit_ts:
                hold_ms = exit_ts - entry_ts
                all_hold_times.append(hold_ms / 1000.0)  # seconds

        # Also check summary-level stats if trades list empty
        if not trades:
            summary = r.get("summary", r)
            day_pnl = summary.get("total_pnl_ticks", 0)
            n_trades = summary.get("total_trades", summary.get("num_trades", 0))
            day_wins = summary.get("winning_trades", 0)
            total_pnl_ticks += day_pnl
            total_trades += n_trades
            total_wins += day_wins

        daily_pnls.append(day_pnl)
        daily_trades.append(len(trades) if trades else r.get("summary", r).get("total_trades", r.get("num_trades", 0)))

        month = r["month"]
        if month not in regime_data:
            regime_data[month] = {"pnl": 0, "trades": 0, "wins": 0, "days": 0}
        regime_data[month]["pnl"] += day_pnl
        regime_data[month]["trades"] += len(trades) if trades else daily_trades[-1]
        regime_data[month]["wins"] += day_wins
        regime_data[month]["days"] += 1

    # Sharpe
    if daily_pnls and len(daily_pnls) > 1:
        mean_daily = statistics.mean(daily_pnls)
        std_daily = statistics.stdev(daily_pnls) if len(daily_pnls) > 1 else 1
        sharpe = (mean_daily / std_daily) * math.sqrt(252) if std_daily > 0 else 0
    else:
        sharpe = 0
        mean_daily = 0
        std_daily = 0

    # Profit factor
    gross_profit = sum(p for p in all_trade_pnls if p > 0) if all_trade_pnls else 0
    gross_loss = abs(sum(p for p in all_trade_pnls if p < 0)) if all_trade_pnls else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Win rate
    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0

    # Max drawdown (in ticks)
    cumsum = 0
    peak = 0
    max_dd = 0
    for dp in daily_pnls:
        cumsum += dp
        if cumsum > peak:
            peak = cumsum
        dd = peak - cumsum
        if dd > max_dd:
            max_dd = dd

    # Average winner / average loser
    winners = [p for p in all_trade_pnls if p > 0]
    losers = [p for p in all_trade_pnls if p < 0]
    avg_win = statistics.mean(winners) if winners else 0
    avg_loss = statistics.mean(losers) if losers else 0

    # Dollar amounts
    total_pnl_usd = total_pnl_ticks * TICK_VALUE
    commission_total = total_trades * COMMISSION_RT
    net_pnl_usd = total_pnl_usd - commission_total

    # Positive days
    pos_days = sum(1 for p in daily_pnls if p > 0)

    analysis = {
        "card": card_name,
        "num_dates": len(results),
        "total_trades": total_trades,
        "total_pnl_ticks": round(total_pnl_ticks, 2),
        "total_pnl_usd_gross": round(total_pnl_usd, 2),
        "total_commission": round(commission_total, 2),
        "net_pnl_usd": round(net_pnl_usd, 2),
        "sharpe": round(sharpe, 2),
        "win_rate_pct": round(wr, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_trades_per_day": round(total_trades / len(results), 1) if results else 0,
        "positive_days": pos_days,
        "positive_day_pct": round(pos_days / len(results) * 100, 1) if results else 0,
        "max_drawdown_ticks": round(max_dd, 2),
        "max_drawdown_usd": round(max_dd * TICK_VALUE, 2),
        "avg_daily_pnl_ticks": round(mean_daily, 2),
        "std_daily_pnl_ticks": round(std_daily, 2),
        "avg_win_ticks": round(avg_win, 2),
        "avg_loss_ticks": round(avg_loss, 2),
        "win_loss_ratio": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
        "long_short_split": {
            "longs": total_longs,
            "shorts": total_shorts,
            "long_pnl_ticks": round(long_pnl, 2),
            "short_pnl_ticks": round(short_pnl, 2),
        },
        "regime_breakdown": {},
        "mae_mfe": {},
        "hold_time": {},
        "daily_results": [],
    }

    # Regime breakdown
    for month in ["Dec", "Jan", "Feb", "Mar"]:
        if month in regime_data:
            rd = regime_data[month]
            wr_m = (rd["wins"] / rd["trades"] * 100) if rd["trades"] > 0 else 0
            analysis["regime_breakdown"][month] = {
                "days": rd["days"],
                "trades": rd["trades"],
                "pnl_ticks": round(rd["pnl"], 2),
                "pnl_usd": round(rd["pnl"] * TICK_VALUE, 2),
                "win_rate": round(wr_m, 1),
                "avg_daily_pnl": round(rd["pnl"] / rd["days"], 2) if rd["days"] > 0 else 0,
            }

    # MAE/MFE stats
    if all_maes:
        analysis["mae_mfe"] = {
            "mae_mean": round(statistics.mean(all_maes), 2),
            "mae_median": round(statistics.median(all_maes), 2),
            "mae_p90": round(sorted(all_maes)[int(0.9 * len(all_maes))], 2),
            "mae_max": round(max(all_maes), 2),
            "mfe_mean": round(statistics.mean(all_mfes), 2) if all_mfes else 0,
            "mfe_median": round(statistics.median(all_mfes), 2) if all_mfes else 0,
            "mfe_p90": round(sorted(all_mfes)[int(0.9 * len(all_mfes))], 2) if all_mfes else 0,
            "mfe_max": round(max(all_mfes), 2) if all_mfes else 0,
        }
        # MAE of winners specifically
        winner_maes = [all_maes[i] for i in range(len(all_trade_pnls)) if i < len(all_maes) and all_trade_pnls[i] > 0]
        loser_maes = [all_maes[i] for i in range(len(all_trade_pnls)) if i < len(all_maes) and all_trade_pnls[i] < 0]
        if winner_maes:
            analysis["mae_mfe"]["winner_mae_mean"] = round(statistics.mean(winner_maes), 2)
            analysis["mae_mfe"]["winner_mae_median"] = round(statistics.median(winner_maes), 2)
        if loser_maes:
            analysis["mae_mfe"]["loser_mae_mean"] = round(statistics.mean(loser_maes), 2)
            analysis["mae_mfe"]["loser_mae_median"] = round(statistics.median(loser_maes), 2)

    # Hold time stats
    if all_hold_times:
        analysis["hold_time"] = {
            "mean_seconds": round(statistics.mean(all_hold_times), 1),
            "median_seconds": round(statistics.median(all_hold_times), 1),
            "p10_seconds": round(sorted(all_hold_times)[int(0.1 * len(all_hold_times))], 1),
            "p90_seconds": round(sorted(all_hold_times)[int(0.9 * len(all_hold_times))], 1),
        }

    # Daily results
    for i, r in enumerate(results):
        analysis["daily_results"].append({
            "date": r["date"],
            "month": r["month"],
            "pnl_ticks": daily_pnls[i],
            "trades": daily_trades[i],
        })

    return analysis


def format_report(a):
    """Format analysis dict into a readable report string."""
    lines = []
    lines.append(f"={'='*60}")
    lines.append(f"  {a['card']} — Full-Date Sweep ({a['num_dates']} days)")
    lines.append(f"={'='*60}")
    lines.append(f"")
    lines.append(f"  Total Trades:     {a['total_trades']}")
    lines.append(f"  Avg/Day:          {a['avg_trades_per_day']}")
    lines.append(f"  Win Rate:         {a['win_rate_pct']}%")
    lines.append(f"  Profit Factor:    {a['profit_factor']}")
    lines.append(f"  Sharpe:           {a['sharpe']}")
    lines.append(f"")
    lines.append(f"  PnL (ticks):      {a['total_pnl_ticks']}")
    lines.append(f"  PnL (gross $):    ${a['total_pnl_usd_gross']:,.2f}")
    lines.append(f"  Commission:       ${a['total_commission']:,.2f}")
    lines.append(f"  Net PnL:          ${a['net_pnl_usd']:,.2f}")
    lines.append(f"")
    lines.append(f"  Avg Daily PnL:    {a['avg_daily_pnl_ticks']} ticks")
    lines.append(f"  Std Daily PnL:    {a['std_daily_pnl_ticks']} ticks")
    lines.append(f"  Positive Days:    {a['positive_days']}/{a['num_dates']} ({a['positive_day_pct']}%)")
    lines.append(f"  Max Drawdown:     {a['max_drawdown_ticks']} ticks (${a['max_drawdown_usd']:,.2f})")
    lines.append(f"")
    lines.append(f"  Avg Win:          {a['avg_win_ticks']} ticks")
    lines.append(f"  Avg Loss:         {a['avg_loss_ticks']} ticks")
    lines.append(f"  Win/Loss Ratio:   {a['win_loss_ratio']}")
    lines.append(f"")

    # L/S split
    ls = a["long_short_split"]
    lines.append(f"  --- Long/Short Split ---")
    lines.append(f"  Longs:  {ls['longs']} trades, {ls['long_pnl_ticks']} ticks (${ls['long_pnl_ticks']*TICK_VALUE:,.2f})")
    lines.append(f"  Shorts: {ls['shorts']} trades, {ls['short_pnl_ticks']} ticks (${ls['short_pnl_ticks']*TICK_VALUE:,.2f})")
    lines.append(f"")

    # Regime
    lines.append(f"  --- Regime Breakdown ---")
    for month in ["Dec", "Jan", "Feb", "Mar"]:
        if month in a["regime_breakdown"]:
            rd = a["regime_breakdown"][month]
            lines.append(f"  {month}: {rd['days']}d, {rd['trades']}t, {rd['pnl_ticks']}tk (${rd['pnl_usd']:,.2f}), WR={rd['win_rate']}%, avg={rd['avg_daily_pnl']}tk/d")
    lines.append(f"")

    # MAE/MFE
    if a["mae_mfe"]:
        m = a["mae_mfe"]
        lines.append(f"  --- MAE/MFE ---")
        lines.append(f"  MAE: mean={m.get('mae_mean','-')}, med={m.get('mae_median','-')}, p90={m.get('mae_p90','-')}, max={m.get('mae_max','-')}")
        lines.append(f"  MFE: mean={m.get('mfe_mean','-')}, med={m.get('mfe_median','-')}, p90={m.get('mfe_p90','-')}, max={m.get('mfe_max','-')}")
        if "winner_mae_mean" in m:
            lines.append(f"  Winner MAE: mean={m['winner_mae_mean']}, med={m['winner_mae_median']}")
        if "loser_mae_mean" in m:
            lines.append(f"  Loser MAE:  mean={m['loser_mae_mean']}, med={m['loser_mae_median']}")
        lines.append(f"")

    # Hold time
    if a["hold_time"]:
        h = a["hold_time"]
        lines.append(f"  --- Hold Time ---")
        lines.append(f"  Mean: {h['mean_seconds']}s, Median: {h['median_seconds']}s, P10: {h['p10_seconds']}s, P90: {h['p90_seconds']}s")
        lines.append(f"")

    # Daily consistency
    lines.append(f"  --- Daily PnL (ticks) ---")
    for dr in a["daily_results"]:
        bar = "+" * min(int(dr["pnl_ticks"]), 50) if dr["pnl_ticks"] > 0 else "-" * min(int(abs(dr["pnl_ticks"])), 50)
        lines.append(f"  {dr['date']} [{dr['month']}] {dr['pnl_ticks']:>8.1f}tk {dr['trades']:>3}t  {bar}")

    return "\n".join(lines)


def main():
    print(f"[{datetime.now()}] Card 2 & 3 Full-Date Sweep Starting", flush=True)
    print(f"Binary: {BINARY}", flush=True)

    if not BINARY.exists():
        print(f"ERROR: Binary not found: {BINARY}", flush=True)
        sys.exit(1)

    card2_dates, card3_dates, card2_preds, card3_preds = find_dates()
    print(f"Card 2 dates (pred & MBO): {len(card2_dates)}", flush=True)
    print(f"Card 3 dates (pred & MBO): {len(card3_dates)}", flush=True)

    # Build jobs
    jobs = []
    skipped = 0

    for date in card2_dates:
        out = CARD2_OUT / f"{date}.json"
        if out.exists():
            skipped += 1
            continue
        cmd = build_card2_cmd(date, card2_preds[date], out)
        jobs.append((cmd, str(out), f"card2_{date}"))

    for date in card3_dates:
        out = CARD3_OUT / f"{date}.json"
        if out.exists():
            skipped += 1
            continue
        cmd = build_card3_cmd(date, card3_preds[date], out)
        jobs.append((cmd, str(out), f"card3_{date}"))

    print(f"Total jobs: {len(jobs)} (skipped {skipped} existing)", flush=True)

    if not jobs:
        print("All jobs already complete! Running analysis only...", flush=True)
    else:
        # Run sweep
        done = 0
        failed = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {}
            for cmd, out, label in jobs:
                f = executor.submit(run_sim, cmd, out, label)
                futures[f] = label

            for f in as_completed(futures):
                label, success, err = f.result()
                done += 1
                if not success:
                    failed += 1
                    print(f"  FAIL [{done}/{len(jobs)}] {label}: {err}", flush=True)
                elif done % 10 == 0 or done == len(jobs):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(jobs) - done) / rate if rate > 0 else 0
                    print(f"  [{done}/{len(jobs)}] {label} OK ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        print(f"\nSweep complete: {done - failed}/{len(jobs)} succeeded, {failed} failed in {time.time()-t0:.0f}s", flush=True)

    # === ANALYSIS ===
    print(f"\n{'='*60}", flush=True)
    print(f"ANALYZING RESULTS", flush=True)
    print(f"{'='*60}\n", flush=True)

    card2_analysis = analyze_card("CARD 2 (conv1.5_vol50 + TP15 + slN)", CARD2_OUT, card2_dates)
    card3_analysis = analyze_card("CARD 3 (mom_emaExit conv0.3_vol70 + TP15 + trail25)", CARD3_OUT, card3_dates)

    report2 = format_report(card2_analysis)
    report3 = format_report(card3_analysis)

    print(report2, flush=True)
    print("\n", flush=True)
    print(report3, flush=True)

    # Save analysis
    analysis_path = LVL3_ROOT / "data" / "processed" / "card23_full_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump({"card2": card2_analysis, "card3": card3_analysis, "timestamp": str(datetime.now())}, f, indent=2)
    print(f"\nAnalysis saved to: {analysis_path}", flush=True)

    # Save text report
    report_path = LVL3_ROOT / "data" / "processed" / "card23_full_report.txt"
    with open(report_path, "w") as f:
        f.write(report2 + "\n\n" + report3)
    print(f"Report saved to: {report_path}", flush=True)


if __name__ == "__main__":
    main()
