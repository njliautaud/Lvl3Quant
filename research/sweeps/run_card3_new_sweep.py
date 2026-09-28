#!/usr/bin/env python3
"""Card 3 (raw_rawExit_conv0.05_ethr0.0_vol70) Full OOT Sweep.

6 configs x 54 dates = 324 jobs:
  TP3/TP8/TP15 x slN/trail25, all with wb50+sig0.1+lat50+chase1t3r+hold60min
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
OUT_BASE = LVL3_ROOT / "data" / "processed" / "card3_new"

TICK_VALUE = 12.50
COMMISSION_RT = 4.70

CONFIGS = [
    {"name": "TP3_slN",     "tp": 3,  "trail": None},
    {"name": "TP8_slN",     "tp": 8,  "trail": None},
    {"name": "TP15_slN",    "tp": 15, "trail": None},
    {"name": "TP3_trail25", "tp": 3,  "trail": 25},
    {"name": "TP8_trail25", "tp": 8,  "trail": 25},
    {"name": "TP15_trail25","tp": 15, "trail": 25},
]


def find_dates():
    preds = {}
    mbo_dates = set()
    for f in sorted(PRED_DIR.glob("*_raw_rawExit_conv0.05_ethr0.0_vol70.npz")):
        date = f.stem[:10]
        preds[date] = f
    for f in sorted(MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst")):
        nodash = f.name.split("-")[2].split(".")[0]
        date = "{}-{}-{}".format(nodash[:4], nodash[4:6], nodash[6:8])
        mbo_dates.add(date)
    dates = sorted(set(preds.keys()) & mbo_dates)
    return dates, preds


def get_mbo_path(date):
    nodash = date.replace("-", "")
    return MBO_DIR / "glbx-mdp3-{}.mbo.dbn.zst".format(nodash)


def run_sim(cmd, out_path, label):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and Path(out_path).exists():
            return (label, True, None)
        return (label, False, r.stderr[:300] if r.stderr else "rc={}, no output".format(r.returncode))
    except subprocess.TimeoutExpired:
        return (label, False, "TIMEOUT")
    except Exception as e:
        return (label, False, str(e)[:200])


def build_cmd(date, pred_path, out_path, tp, trail):
    cmd = [
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
        "--take-profit-ticks", str(tp),
        "--quiet",
    ]
    if trail is not None:
        cmd.extend(["--trailing-ticks", str(trail)])
    return cmd


def get_month(date_str):
    m = int(date_str[5:7])
    return {12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}.get(m, "M{}".format(m))


def analyze_config(config_name, out_dir, dates):
    results = []
    for date in sorted(dates):
        path = out_dir / "{}.json".format(date)
        try:
            with open(path) as f:
                data = json.load(f)
        except:
            continue
        data["date"] = date
        data["month"] = get_month(date)
        results.append(data)

    if not results:
        return {"config": config_name, "error": "No results",
                "total_trades": 0, "avg_trades_day": 0, "total_pnl_ticks": 0,
                "gross_usd": 0, "commission": 0, "net_usd": -999999,
                "sharpe": 0, "win_rate": 0, "profit_factor": 0,
                "pos_days": 0, "pos_day_pct": 0, "max_dd_ticks": 0, "max_dd_usd": 0,
                "avg_daily_ticks": 0, "std_daily_ticks": 0, "avg_win": 0, "avg_loss": 0,
                "wl_ratio": 0, "max_consec_losers": 0,
                "longs": 0, "shorts": 0, "long_pnl": 0, "short_pnl": 0,
                "num_dates": 0, "regime": {}, "mae_mfe": {}, "hold_time": {}, "daily": []}

    total_pnl = 0; total_trades = 0; total_wins = 0
    total_longs = 0; total_shorts = 0; long_pnl = 0; short_pnl = 0
    daily_pnls = []; daily_trades_list = []; all_trade_pnls = []
    all_maes = []; all_mfes = []; all_hold_times = []
    regime = {}
    max_consec_loss = 0; cur_consec_loss = 0

    for r in results:
        trades = r.get("trades", [])
        day_pnl = 0; day_wins = 0; day_ntrades = 0

        if trades:
            day_ntrades = len(trades)
            for t in trades:
                pt = t.get("pnl_ticks", 0)
                total_pnl += pt; day_pnl += pt; total_trades += 1
                all_trade_pnls.append(pt)
                if pt > 0:
                    total_wins += 1; day_wins += 1
                    cur_consec_loss = 0
                elif pt < 0:
                    cur_consec_loss += 1
                    max_consec_loss = max(max_consec_loss, cur_consec_loss)
                side = t.get("side", "")
                if side.lower() == "long":
                    total_longs += 1; long_pnl += pt
                else:
                    total_shorts += 1; short_pnl += pt
                mae = t.get("mae_ticks"); mfe = t.get("mfe_ticks")
                if mae is not None: all_maes.append(mae)
                if mfe is not None: all_mfes.append(mfe)
                ets = t.get("entry_ts", 0); xts = t.get("exit_ts", 0)
                if ets and xts: all_hold_times.append((xts - ets) / 1000.0)
        else:
            s = r.get("summary", r)
            day_pnl = s.get("total_pnl_ticks", 0)
            day_ntrades = s.get("total_trades", s.get("num_trades", 0))
            day_wins = s.get("winning_trades", 0)
            total_pnl += day_pnl; total_trades += day_ntrades; total_wins += day_wins

        daily_pnls.append(day_pnl)
        daily_trades_list.append(day_ntrades)

        month = r["month"]
        if month not in regime:
            regime[month] = {"pnl": 0, "trades": 0, "wins": 0, "days": 0}
        regime[month]["pnl"] += day_pnl
        regime[month]["trades"] += day_ntrades
        regime[month]["wins"] += day_wins
        regime[month]["days"] += 1

    n_days = len(results)
    mean_d = statistics.mean(daily_pnls) if daily_pnls else 0
    std_d = statistics.stdev(daily_pnls) if len(daily_pnls) > 1 else 1
    sharpe = (mean_d / std_d) * math.sqrt(252) if std_d > 0 else 0

    gp = sum(p for p in all_trade_pnls if p > 0)
    gl = abs(sum(p for p in all_trade_pnls if p < 0))
    pf = gp / gl if gl > 0 else float("inf")
    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0

    cumsum = 0; peak = 0; max_dd = 0
    for dp in daily_pnls:
        cumsum += dp
        if cumsum > peak: peak = cumsum
        dd = peak - cumsum
        if dd > max_dd: max_dd = dd

    winners = [p for p in all_trade_pnls if p > 0]
    losers = [p for p in all_trade_pnls if p < 0]
    avg_win = statistics.mean(winners) if winners else 0
    avg_loss = statistics.mean(losers) if losers else 0
    pos_days = sum(1 for p in daily_pnls if p > 0)

    gross_usd = total_pnl * TICK_VALUE
    comm = total_trades * COMMISSION_RT
    net_usd = gross_usd - comm

    a = {
        "config": config_name, "num_dates": n_days,
        "total_trades": total_trades,
        "avg_trades_day": round(total_trades / n_days, 1) if n_days else 0,
        "total_pnl_ticks": round(total_pnl, 2),
        "gross_usd": round(gross_usd, 2), "commission": round(comm, 2), "net_usd": round(net_usd, 2),
        "sharpe": round(sharpe, 2), "win_rate": round(wr, 1), "profit_factor": round(pf, 2),
        "pos_days": pos_days, "pos_day_pct": round(pos_days / n_days * 100, 1) if n_days else 0,
        "max_dd_ticks": round(max_dd, 2), "max_dd_usd": round(max_dd * TICK_VALUE, 2),
        "avg_daily_ticks": round(mean_d, 2), "std_daily_ticks": round(std_d, 2),
        "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "wl_ratio": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
        "max_consec_losers": max_consec_loss,
        "longs": total_longs, "shorts": total_shorts,
        "long_pnl": round(long_pnl, 2), "short_pnl": round(short_pnl, 2),
        "regime": {},
        "mae_mfe": {},
        "hold_time": {},
        "daily": [{"date": results[i]["date"], "month": results[i]["month"],
                    "pnl": daily_pnls[i], "trades": daily_trades_list[i]}
                   for i in range(n_days)],
    }

    for month in ["Dec", "Jan", "Feb", "Mar"]:
        if month in regime:
            rd = regime[month]
            a["regime"][month] = {
                "days": rd["days"], "trades": rd["trades"],
                "pnl_ticks": round(rd["pnl"], 2),
                "pnl_usd": round(rd["pnl"] * TICK_VALUE, 2),
                "wr": round(rd["wins"] / rd["trades"] * 100, 1) if rd["trades"] > 0 else 0,
                "avg_daily": round(rd["pnl"] / rd["days"], 2) if rd["days"] > 0 else 0,
            }

    if all_maes:
        sm = sorted(all_maes); smf = sorted(all_mfes) if all_mfes else []
        a["mae_mfe"] = {
            "mae_mean": round(statistics.mean(all_maes), 2),
            "mae_med": round(statistics.median(all_maes), 2),
            "mae_p90": round(sm[int(0.9 * len(sm))], 2),
            "mae_max": round(max(all_maes), 2),
        }
        if all_mfes:
            a["mae_mfe"].update({
                "mfe_mean": round(statistics.mean(all_mfes), 2),
                "mfe_med": round(statistics.median(all_mfes), 2),
                "mfe_p90": round(smf[int(0.9 * len(smf))], 2),
                "mfe_max": round(max(all_mfes), 2),
            })
        winner_maes = [all_maes[i] for i in range(len(all_trade_pnls))
                       if i < len(all_maes) and all_trade_pnls[i] > 0]
        loser_maes = [all_maes[i] for i in range(len(all_trade_pnls))
                      if i < len(all_maes) and all_trade_pnls[i] < 0]
        if winner_maes:
            a["mae_mfe"]["winner_mae_mean"] = round(statistics.mean(winner_maes), 2)
            a["mae_mfe"]["winner_mae_med"] = round(statistics.median(winner_maes), 2)
        if loser_maes:
            a["mae_mfe"]["loser_mae_mean"] = round(statistics.mean(loser_maes), 2)
            a["mae_mfe"]["loser_mae_med"] = round(statistics.median(loser_maes), 2)

    if all_hold_times:
        sh = sorted(all_hold_times)
        a["hold_time"] = {
            "mean_s": round(statistics.mean(all_hold_times), 1),
            "med_s": round(statistics.median(all_hold_times), 1),
            "p10_s": round(sh[int(0.1 * len(sh))], 1),
            "p90_s": round(sh[int(0.9 * len(sh))], 1),
        }

    return a


def main():
    print("[{}] Card 3 New Sweep Starting".format(datetime.now()), flush=True)
    dates, preds = find_dates()
    print("Found {} dates with both pred + MBO".format(len(dates)), flush=True)

    # Create output dirs
    for cfg in CONFIGS:
        d = OUT_BASE / cfg["name"]
        d.mkdir(parents=True, exist_ok=True)

    # Build all jobs
    jobs = []
    skipped = 0
    for cfg in CONFIGS:
        out_dir = OUT_BASE / cfg["name"]
        for date in dates:
            out_path = out_dir / "{}.json".format(date)
            if out_path.exists():
                skipped += 1
                continue
            cmd = build_cmd(date, preds[date], out_path, cfg["tp"], cfg["trail"])
            jobs.append((cmd, str(out_path), "{}_{}".format(cfg["name"], date)))

    print("Total jobs: {} (skipped {} existing)".format(len(jobs), skipped), flush=True)

    if jobs:
        done = 0; failed = 0; t0 = time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(run_sim, cmd, out, lbl): lbl for cmd, out, lbl in jobs}
            for f in as_completed(futures):
                label, success, err = f.result()
                done += 1
                if not success:
                    failed += 1
                    print("  FAIL [{}/{}] {}: {}".format(done, len(jobs), label, err), flush=True)
                elif done % 20 == 0 or done == len(jobs):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(jobs) - done) / rate if rate > 0 else 0
                    print("  [{}/{}] {} OK ({:.0f}s, ETA {:.0f}s)".format(done, len(jobs), label, elapsed, eta), flush=True)
        print("\nSweep done: {}/{} OK, {} failed in {:.0f}s".format(done-failed, len(jobs), failed, time.time()-t0), flush=True)

    # === ANALYSIS ===
    print("\n" + "="*70, flush=True)
    print("DEEP ANALYSIS - Card 3 (raw_rawExit_conv0.05_ethr0.0_vol70)", flush=True)
    print("="*70 + "\n", flush=True)

    all_analyses = {}
    for cfg in CONFIGS:
        out_dir = OUT_BASE / cfg["name"]
        a = analyze_config(cfg["name"], out_dir, dates)
        all_analyses[cfg["name"]] = a

        print("\n--- {} ---".format(cfg["name"]), flush=True)
        print("  Trades: {} ({}/d)  WR: {}%  PF: {}  Sharpe: {}".format(
            a["total_trades"], a["avg_trades_day"], a["win_rate"], a["profit_factor"], a["sharpe"]), flush=True)
        print("  Net PnL: ${:,.2f} (gross ${:,.2f} - comm ${:,.2f})".format(
            a["net_usd"], a["gross_usd"], a["commission"]), flush=True)
        print("  Pos days: {}/{} ({}%)  MaxDD: {}tk (${:,.2f})".format(
            a["pos_days"], a["num_dates"], a["pos_day_pct"], a["max_dd_ticks"], a["max_dd_usd"]), flush=True)
        print("  AvgWin: {}tk  AvgLoss: {}tk  W/L: {}  MaxConsecLoss: {}".format(
            a["avg_win"], a["avg_loss"], a["wl_ratio"], a["max_consec_losers"]), flush=True)
        print("  L/S: {}L ({}tk) / {}S ({}tk)".format(
            a["longs"], a["long_pnl"], a["shorts"], a["short_pnl"]), flush=True)
        if a["regime"]:
            for m in ["Dec", "Jan", "Feb", "Mar"]:
                if m in a["regime"]:
                    rd = a["regime"][m]
                    print("  {}: {}d {}t {}tk (${:,.2f}) WR={}% avg={}tk/d".format(
                        m, rd["days"], rd["trades"], rd["pnl_ticks"], rd["pnl_usd"], rd["wr"], rd["avg_daily"]), flush=True)
        if a["mae_mfe"]:
            mm = a["mae_mfe"]
            print("  MAE: mean={} med={} p90={} max={}".format(
                mm.get("mae_mean","-"), mm.get("mae_med","-"), mm.get("mae_p90","-"), mm.get("mae_max","-")), flush=True)
            print("  MFE: mean={} med={} p90={} max={}".format(
                mm.get("mfe_mean","-"), mm.get("mfe_med","-"), mm.get("mfe_p90","-"), mm.get("mfe_max","-")), flush=True)
            if "winner_mae_mean" in mm:
                print("  Winner MAE: mean={} med={}".format(mm["winner_mae_mean"], mm["winner_mae_med"]), flush=True)
            if "loser_mae_mean" in mm:
                print("  Loser MAE: mean={} med={}".format(mm["loser_mae_mean"], mm["loser_mae_med"]), flush=True)
        if a["hold_time"]:
            h = a["hold_time"]
            print("  Hold: mean={}s med={}s p10={}s p90={}s".format(
                h["mean_s"], h["med_s"], h["p10_s"], h["p90_s"]), flush=True)

    # === HEAD-TO-HEAD COMPARISON ===
    print("\n" + "="*70, flush=True)
    print("HEAD-TO-HEAD COMPARISON", flush=True)
    print("="*70, flush=True)
    header = "{:<18} {:>7} {:>6} {:>6} {:>7} {:>10} {:>9} {:>5} {:>6}".format(
        "Config", "Trades", "WR%", "PF", "Sharpe", "Net$", "MaxDD$", "W/L", "PosD%")
    print(header, flush=True)
    print("-"*70, flush=True)
    for cfg in CONFIGS:
        a = all_analyses[cfg["name"]]
        print("{:<18} {:>7} {:>5.1f}% {:>6.2f} {:>7.2f} {:>9,.0f} {:>9,.0f} {:>5.2f} {:>5.1f}%".format(
            cfg["name"], a["total_trades"], a["win_rate"], a["profit_factor"],
            a["sharpe"], a["net_usd"], a["max_dd_usd"], a["wl_ratio"], a["pos_day_pct"]), flush=True)

    # === DAILY PNL FOR BEST CONFIG ===
    best_cfg = max(all_analyses.values(), key=lambda x: x.get("net_usd", -999999))
    print("\n" + "="*70, flush=True)
    print("BEST CONFIG: {}".format(best_cfg["config"]), flush=True)
    print("="*70, flush=True)
    print("Daily PnL (ticks):", flush=True)
    for d in best_cfg.get("daily", []):
        if d["pnl"] > 0:
            bar = "+" * min(int(d["pnl"]), 60)
        else:
            bar = "-" * min(int(abs(d["pnl"])), 60)
        print("  {} [{}] {:>8.1f}tk {:>3}t  {}".format(
            d["date"], d["month"], d["pnl"], d["trades"], bar), flush=True)

    # Save JSON
    out_path = OUT_BASE / "card3_new_analysis.json"
    with open(out_path, "w") as f:
        json.dump({"configs": all_analyses, "timestamp": str(datetime.now())}, f, indent=2)
    print("\nSaved: {}".format(out_path), flush=True)


if __name__ == "__main__":
    main()
