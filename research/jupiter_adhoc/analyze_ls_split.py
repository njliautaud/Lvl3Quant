#!/usr/bin/env python3
"""
Long vs Short Trade Split Analysis — WF Fill_Sim Sweep
Analyzes 60K+ result JSONs in ~/Lvl3Quant/alpha_discovery/results/wf_sweep/
"""

import json
import os
import re
import math
from collections import defaultdict

SWEEP_DIR = os.path.expanduser("~/Lvl3Quant/alpha_discovery/results/wf_sweep")

def parse_config_from_filename(fname):
    """Extract config params from filename like wf_v70_c2.5_h30m_chase_ct1r3_lat0_2025-12-01.json"""
    m = re.match(
        r"wf_v(\d+)_c([\d.]+)_h(\w+)_(\w+)_ct(\d+)r(\d+)_lat(\d+)_(\d{4}-\d{2}-\d{2})\.json",
        fname
    )
    if not m:
        return None
    return {
        "vol_thresh": int(m.group(1)),
        "conv_thresh": float(m.group(2)),
        "hold": m.group(3),
        "entry_mode": m.group(4),
        "chase_ticks": int(m.group(5)),
        "chase_reprices": int(m.group(6)),
        "latency": int(m.group(7)),
        "date": m.group(8),
    }

def sharpe_from_trades(pnl_list):
    """Compute Sharpe from per-trade PnL list (annualized by trade count, not time)."""
    if len(pnl_list) < 2:
        return None
    n = len(pnl_list)
    mean = sum(pnl_list) / n
    var = sum((x - mean) ** 2 for x in pnl_list) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return None
    return mean / std  # Per-trade Sharpe (not time-scaled)

def analyze_files():
    print("Scanning result files...")
    files = [f for f in os.listdir(SWEEP_DIR) if f.startswith("wf_v") and f.endswith(".json")]
    print(f"Found {len(files)} result files")

    # Aggregates across ALL configs
    all_long_trades = []
    all_short_trades = []

    # Per-config aggregates: config_key -> {long_pnl, short_pnl, long_wins, short_wins, long_n, short_n}
    config_data = defaultdict(lambda: {
        "long_pnl": [], "short_pnl": [],
        "long_wins": 0, "short_wins": 0,
        "long_n": 0, "short_n": 0,
        "dates": set()
    })

    # Per-date aggregates: date -> {long_pnl, short_pnl}
    date_data = defaultdict(lambda: {"long_pnl": [], "short_pnl": [], "long_n": 0, "short_n": 0})

    # Track all entry prices by date for market direction
    date_prices = defaultdict(list)  # date -> list of (signal_time_ns, entry_price, side)

    errors = 0
    no_trade_files = 0
    processed = 0

    for fname in files:
        cfg = parse_config_from_filename(fname)
        if cfg is None:
            errors += 1
            continue

        fpath = os.path.join(SWEEP_DIR, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            errors += 1
            continue

        trades = data.get("trades", [])
        if not trades:
            no_trade_files += 1
            processed += 1
            continue

        date = cfg["date"]
        config_key = f"v{cfg['vol_thresh']}_c{cfg['conv_thresh']}_h{cfg['hold']}_{cfg['entry_mode']}_ct{cfg['chase_ticks']}r{cfg['chase_reprices']}_lat{cfg['latency']}"

        for trade in trades:
            side = trade.get("side", "")
            pnl = trade.get("pnl_dollars", 0.0)
            entry_price = trade.get("entry_price", 0.0)
            signal_time = trade.get("signal_time_ns", 0)

            is_win = pnl > 0

            if side == "BUY":
                all_long_trades.append(pnl)
                config_data[config_key]["long_pnl"].append(pnl)
                config_data[config_key]["long_n"] += 1
                if is_win:
                    config_data[config_key]["long_wins"] += 1
                date_data[date]["long_pnl"].append(pnl)
                date_data[date]["long_n"] += 1
                date_prices[date].append((signal_time, entry_price, "LONG"))

            elif side == "SELL":
                all_short_trades.append(pnl)
                config_data[config_key]["short_pnl"].append(pnl)
                config_data[config_key]["short_n"] += 1
                if is_win:
                    config_data[config_key]["short_wins"] += 1
                date_data[date]["short_pnl"].append(pnl)
                date_data[date]["short_n"] += 1
                date_prices[date].append((signal_time, entry_price, "SHORT"))

            config_data[config_key]["dates"].add(date)

        processed += 1
        if processed % 5000 == 0:
            print(f"  Processed {processed}/{len(files)}...")

    print(f"\nDone. Processed={processed}, No-trade files={no_trade_files}, Errors={errors}")
    return all_long_trades, all_short_trades, config_data, date_data, date_prices

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def summarize_pnl(pnl_list, label):
    if not pnl_list:
        return f"  {label}: NO TRADES"
    n = len(pnl_list)
    total = sum(pnl_list)
    mean = total / n
    wins = sum(1 for x in pnl_list if x > 0)
    wr = wins / n
    sharpe = sharpe_from_trades(pnl_list)
    sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
    avg_win = sum(x for x in pnl_list if x > 0) / max(wins, 1)
    avg_loss = sum(x for x in pnl_list if x < 0) / max(n - wins, 1)
    pf_num = sum(x for x in pnl_list if x > 0)
    pf_den = abs(sum(x for x in pnl_list if x < 0))
    pf = pf_num / pf_den if pf_den > 0 else float("inf")
    return (
        f"  {label}: n={n:,}  total=${total:,.0f}  mean=${mean:.1f}"
        f"  WR={wr:.1%}  Sharpe(PT)={sharpe_str}"
        f"\n    avg_win=${avg_win:.0f}  avg_loss=${avg_loss:.0f}  PF={pf:.2f}"
    )

def main():
    all_long, all_short, config_data, date_data, date_prices = analyze_files()

    total_trades = len(all_long) + len(all_short)
    long_pct = len(all_long) / total_trades * 100 if total_trades > 0 else 0
    short_pct = len(all_short) / total_trades * 100 if total_trades > 0 else 0

    # ============================================================
    print_section("1. L/S TRADE COUNT SPLIT (ALL CONFIGS)")
    print(f"  Total trades: {total_trades:,}")
    print(f"  Long  (BUY):  {len(all_long):,}  ({long_pct:.1f}%)")
    print(f"  Short (SELL): {len(all_short):,}  ({short_pct:.1f}%)")

    # ============================================================
    print_section("2. LONG P&L vs SHORT P&L (AGGREGATE)")
    print(summarize_pnl(all_long, "LONG "))
    print(summarize_pnl(all_short, "SHORT"))

    # ============================================================
    print_section("3. WIN RATE COMPARISON")
    # Already printed in section 2 but highlight explicitly
    long_wr = sum(1 for x in all_long if x > 0) / len(all_long) if all_long else 0
    short_wr = sum(1 for x in all_short if x > 0) / len(all_short) if all_short else 0
    print(f"  Long  win rate: {long_wr:.2%}")
    print(f"  Short win rate: {short_wr:.2%}")
    print(f"  Difference:     {(long_wr - short_wr)*100:+.2f} pp")

    # ============================================================
    print_section("4. LONG SHARPE vs SHORT SHARPE (PER-TRADE)")
    long_sharpe = sharpe_from_trades(all_long)
    short_sharpe = sharpe_from_trades(all_short)
    print(f"  Long  per-trade Sharpe: {long_sharpe:.4f}" if long_sharpe else "  Long  Sharpe: N/A")
    print(f"  Short per-trade Sharpe: {short_sharpe:.4f}" if short_sharpe else "  Short Sharpe: N/A")

    # ============================================================
    print_section("5. PER-DAY L/S ANALYSIS")
    sorted_dates = sorted(date_data.keys())
    print(f"  {'Date':<12} {'L_n':>5} {'L_PnL':>9} {'L_WR':>7}  {'S_n':>5} {'S_PnL':>9} {'S_WR':>7}  {'Bias'}")
    print(f"  {'-'*12} {'-'*5} {'-'*9} {'-'*7}  {'-'*5} {'-'*9} {'-'*7}  {'-'*10}")
    for date in sorted_dates:
        d = date_data[date]
        l_pnl = sum(d["long_pnl"])
        s_pnl = sum(d["short_pnl"])
        l_n = d["long_n"]
        s_n = d["short_n"]
        l_wr = sum(1 for x in d["long_pnl"] if x > 0) / l_n if l_n > 0 else 0
        s_wr = sum(1 for x in d["short_pnl"] if x > 0) / s_n if s_n > 0 else 0
        total_day = l_n + s_n
        l_bias = l_n / total_day * 100 if total_day > 0 else 50
        if l_n > s_n * 1.5:
            bias = "LONG-heavy"
        elif s_n > l_n * 1.5:
            bias = "SHORT-heavy"
        else:
            bias = "balanced"
        print(f"  {date:<12} {l_n:>5} ${l_pnl:>8,.0f} {l_wr:>6.1%}  {s_n:>5} ${s_pnl:>8,.0f} {s_wr:>6.1%}  {bias}")

    # Aggregate per-day: days where shorts dominated
    print()
    days_short_profitable = sum(1 for d in sorted_dates if sum(date_data[d]["short_pnl"]) > 0)
    days_long_profitable = sum(1 for d in sorted_dates if sum(date_data[d]["long_pnl"]) > 0)
    days_both = sum(1 for d in sorted_dates
                    if date_data[d]["short_pnl"] and date_data[d]["long_pnl"]
                    and sum(date_data[d]["short_pnl"]) > 0 and sum(date_data[d]["long_pnl"]) > 0)
    print(f"  Days where LONGS profitable:  {days_long_profitable}/{len(sorted_dates)}")
    print(f"  Days where SHORTS profitable: {days_short_profitable}/{len(sorted_dates)}")
    print(f"  Days BOTH profitable:         {days_both}/{len(sorted_dates)}")

    # ============================================================
    print_section("6. BEST CONFIGS FOR SHORTS SPECIFICALLY")
    # Build per-config short stats, filter to configs with >= 20 short trades total
    config_short_stats = []
    for cfg_key, cd in config_data.items():
        s_pnl = cd["short_pnl"]
        if len(s_pnl) < 20:
            continue
        total_s = sum(s_pnl)
        n_s = len(s_pnl)
        wr_s = cd["short_wins"] / n_s
        sharpe_s = sharpe_from_trades(s_pnl)
        config_short_stats.append((cfg_key, n_s, total_s, wr_s, sharpe_s or -999))

    config_short_stats.sort(key=lambda x: x[2], reverse=True)  # sort by total short PnL
    print(f"  Top 20 configs by SHORT total P&L (min 20 short trades):")
    print(f"  {'Config':<45} {'S_n':>5} {'S_PnL':>10} {'S_WR':>7} {'S_Sharpe':>9}")
    print(f"  {'-'*45} {'-'*5} {'-'*10} {'-'*7} {'-'*9}")
    for row in config_short_stats[:20]:
        cfg_key, n_s, total_s, wr_s, sharpe_s = row
        print(f"  {cfg_key:<45} {n_s:>5} ${total_s:>9,.0f} {wr_s:>6.1%} {sharpe_s:>9.3f}")

    # ============================================================
    print_section("7. SHORTS-ONLY UNIVERSE")
    print(summarize_pnl(all_short, "SHORTS ONLY"))
    short_total = sum(all_short)
    print(f"\n  Per-day short P&L (avg over {len(sorted_dates)} trading days):")
    for date in sorted_dates[:5]:
        d = date_data[date]
        sp = sum(d["short_pnl"])
        print(f"    {date}: ${sp:,.0f} from {d['short_n']} short trades")
    print(f"  ...")

    # ============================================================
    print_section("8. LONGS-ONLY UNIVERSE")
    print(summarize_pnl(all_long, "LONGS ONLY"))
    long_total = sum(all_long)
    print(f"\n  Per-day long P&L (avg over {len(sorted_dates)} trading days):")
    for date in sorted_dates[:5]:
        d = date_data[date]
        lp = sum(d["long_pnl"])
        print(f"    {date}: ${lp:,.0f} from {d['long_n']} long trades")
    print(f"  ...")

    # ============================================================
    print_section("9. MARKET CONTEXT — DEC 2025 DIRECTION")
    print("  Analyzing ES price trajectory from trade entry prices...")
    date_avg_price = {}
    for date in sorted_dates:
        prices = [p for (_, p, _) in date_prices[date] if p > 0]
        if prices:
            date_avg_price[date] = sum(prices) / len(prices)

    if date_avg_price:
        dates_with_price = sorted(date_avg_price.keys())
        first_price = date_avg_price[dates_with_price[0]]
        last_price = date_avg_price[dates_with_price[-1]]
        print(f"\n  Date range: {dates_with_price[0]} to {dates_with_price[-1]}")
        print(f"  Avg entry price on first day ({dates_with_price[0]}): {first_price:.2f}")
        print(f"  Avg entry price on last day  ({dates_with_price[-1]}): {last_price:.2f}")
        print(f"  Price change: {last_price - first_price:+.2f} pts  ({(last_price/first_price - 1)*100:+.2f}%)")

        print(f"\n  {'Date':<12} {'Avg_Entry':>10} {'Long_n':>7} {'Short_n':>8} {'Net_direction'}")
        print(f"  {'-'*12} {'-'*10} {'-'*7} {'-'*8} {'-'*15}")
        for date in dates_with_price:
            avg_p = date_avg_price[date]
            change = avg_p - first_price
            l_n = date_data[date]["long_n"]
            s_n = date_data[date]["short_n"]
            net_dir = "UP" if change > 0 else "DOWN"
            print(f"  {date:<12} {avg_p:>10.2f} {l_n:>7} {s_n:>8} {change:>+8.1f}pts ({net_dir})")

    # ============================================================
    print_section("SUMMARY VERDICT")
    all_pnl = all_long + all_short
    total_pnl = sum(all_pnl)
    long_contribution = sum(all_long) / total_pnl * 100 if total_pnl != 0 else 0
    short_contribution = sum(all_short) / total_pnl * 100 if total_pnl != 0 else 0

    print(f"  Total P&L (all trades, all configs): ${total_pnl:,.0f}")
    print(f"  Long  contribution: ${sum(all_long):,.0f} ({long_contribution:.1f}%)")
    print(f"  Short contribution: ${sum(all_short):,.0f} ({short_contribution:.1f}%)")
    print()

    if sum(all_long) > 0 and sum(all_short) > 0:
        print("  VERDICT: BOTH sides are independently profitable.")
    elif sum(all_long) > 0 and sum(all_short) <= 0:
        print("  VERDICT: LONG-only is profitable; SHORTS are net negative — LONGS carry the book.")
    elif sum(all_short) > 0 and sum(all_long) <= 0:
        print("  VERDICT: SHORT-only is profitable; LONGS are net negative — SHORTS carry the book.")
    else:
        print("  VERDICT: Neither side is net profitable in aggregate.")

    if long_wr > short_wr + 0.05:
        print("  WIN RATE: Longs have significantly higher win rate — signal skews bullish.")
    elif short_wr > long_wr + 0.05:
        print("  WIN RATE: Shorts have significantly higher win rate — signal skews bearish.")
    else:
        print("  WIN RATE: Both sides have similar win rates — signal is direction-balanced.")

if __name__ == "__main__":
    main()
