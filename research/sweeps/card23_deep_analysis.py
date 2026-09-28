#!/usr/bin/env python3
"""Deep analysis of Card 2 & Card 3 results."""
import json, statistics, math
from pathlib import Path

TICK = 12.50
COMM = 4.70

for card_name, card_dir in [
    ("CARD 2 (conv1.5_vol50 + TP15 + slN + wb50 + sig0.1 + lat50 + chase)", "/home/jupiter/Lvl3Quant/data/processed/card2_full"),
    ("CARD 3 (mom_emaExit conv0.3_vol70 + TP15 + trail25 + wb50 + sig0.1 + lat50 + chase)", "/home/jupiter/Lvl3Quant/data/processed/card3_full"),
]:
    files = sorted(Path(card_dir).glob("*.json"))
    all_trades = []
    daily = []
    total_signals = 0
    total_posted = 0
    total_filled = 0

    for f in files:
        d = json.load(open(f))
        trades = d.get("trades", [])
        day_pnl = sum(t["pnl_ticks"] for t in trades)
        daily.append({"date": f.stem, "pnl": day_pnl, "trades": len(trades)})
        all_trades.extend(trades)
        total_signals += d.get("total_signals", 0)
        total_posted += d.get("total_posted", 0)
        total_filled += d.get("total_filled", d.get("total_trades", 0))

    pnls = [t["pnl_ticks"] for t in all_trades]
    daily_pnls = [d["pnl"] for d in daily]

    fill_rate = total_filled / total_posted * 100 if total_posted else 0
    signal_to_fill = total_filled / total_signals * 100 if total_signals else 0

    reasons = {}
    for t in all_trades:
        r = t.get("exit_reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1

    mean_d = statistics.mean(daily_pnls)
    std_d = statistics.stdev(daily_pnls) if len(daily_pnls) > 1 else 1
    sharpe = mean_d / std_d * math.sqrt(252)

    cum = 0
    peak = 0
    max_dd = 0
    for dp in daily_pnls:
        cum += dp
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    maes = [t["mae_ticks"] for t in all_trades if "mae_ticks" in t]
    mfes = [t["mfe_ticks"] for t in all_trades if "mfe_ticks" in t]
    winner_maes = [t["mae_ticks"] for t in all_trades if t.get("mae_ticks") is not None and t["pnl_ticks"] > 0]
    loser_maes = [t["mae_ticks"] for t in all_trades if t.get("mae_ticks") is not None and t["pnl_ticks"] < 0]
    winner_mfes = [t["mfe_ticks"] for t in all_trades if t.get("mfe_ticks") is not None and t["pnl_ticks"] > 0]
    loser_mfes = [t["mfe_ticks"] for t in all_trades if t.get("mfe_ticks") is not None and t["pnl_ticks"] < 0]

    holds = [t["hold_duration_ns"] / 1e9 for t in all_trades if "hold_duration_ns" in t]
    winner_holds = [t["hold_duration_ns"] / 1e9 for t in all_trades if "hold_duration_ns" in t and t["pnl_ticks"] > 0]
    loser_holds = [t["hold_duration_ns"] / 1e9 for t in all_trades if "hold_duration_ns" in t and t["pnl_ticks"] < 0]

    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_win / gross_loss if gross_loss else float("inf")

    total_pnl = sum(pnls)
    total_usd = total_pnl * TICK
    comm = len(all_trades) * COMM
    net = total_usd - comm

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    print()
    print("=" * 65)
    print("  {} -- DEEP ANALYSIS ({} days)".format(card_name, len(files)))
    print("=" * 65)
    print()
    print("  EXECUTION:")
    print("    Signals: {}  |  Posted: {}  |  Filled: {}".format(total_signals, total_posted, total_filled))
    print("    Fill Rate: {:.1f}% (of posted)  |  Signal-to-Fill: {:.1f}%".format(fill_rate, signal_to_fill))
    print("    Trades: {}  |  Avg/Day: {:.1f}".format(len(all_trades), len(all_trades) / len(files)))
    print()
    print("  PERFORMANCE:")
    print("    WR: {:.1f}%  |  PF: {:.2f}  |  Sharpe: {:.2f}".format(
        len(winners) / len(all_trades) * 100, pf, sharpe))
    print("    PnL: {:.1f} ticks = ${:,.0f} gross".format(total_pnl, total_usd))
    print("    Commission: ${:,.0f} ({} trades x ${:.2f})".format(comm, len(all_trades), COMM))
    print("    NET PnL: ${:,.0f}".format(net))
    print("    Annualized: ~${:,.0f}/yr".format(net / len(files) * 252))
    print("    Per-day: ${:,.0f}/day net".format(net / len(files)))
    print()
    print("  RISK:")
    print("    Max Drawdown: {:.1f} ticks (${:,.0f})".format(max_dd, max_dd * TICK))
    print("    Positive Days: {}/{} ({:.0f}%)".format(
        sum(1 for d in daily_pnls if d > 0), len(daily_pnls),
        sum(1 for d in daily_pnls if d > 0) / len(daily_pnls) * 100))
    print("    Avg Win: {:.1f}tk  |  Avg Loss: {:.1f}tk  |  Ratio: {:.2f}".format(
        statistics.mean(winners), statistics.mean(losers),
        abs(statistics.mean(winners) / statistics.mean(losers)) if losers else 0))
    print("    Largest Win: {:.1f}tk  |  Largest Loss: {:.1f}tk".format(max(pnls), min(pnls)))
    print()
    print("  EXIT REASONS: {}".format(dict(sorted(reasons.items(), key=lambda x: -x[1]))))
    print()

    if maes:
        print("  MAE/MFE ANALYSIS:")
        print("    ALL:     MAE mean={:.1f}, med={:.0f}, p90={:.0f}, max={:.0f}".format(
            statistics.mean(maes), statistics.median(maes),
            sorted(maes)[int(.9 * len(maes))], max(maes)))
        if mfes:
            print("             MFE mean={:.1f}, med={:.0f}, p90={:.0f}, max={:.0f}".format(
                statistics.mean(mfes), statistics.median(mfes),
                sorted(mfes)[int(.9 * len(mfes))], max(mfes)))
        if winner_maes:
            print("    WINNERS: MAE mean={:.1f}, med={:.0f}, p90={:.0f}  |  MFE mean={:.1f}, med={:.0f}".format(
                statistics.mean(winner_maes), statistics.median(winner_maes),
                sorted(winner_maes)[int(.9 * len(winner_maes))],
                statistics.mean(winner_mfes) if winner_mfes else 0,
                statistics.median(winner_mfes) if winner_mfes else 0))
        if loser_maes:
            print("    LOSERS:  MAE mean={:.1f}, med={:.0f}, p90={:.0f}  |  MFE mean={:.1f}, med={:.0f}".format(
                statistics.mean(loser_maes), statistics.median(loser_maes),
                sorted(loser_maes)[int(.9 * len(loser_maes))],
                statistics.mean(loser_mfes) if loser_mfes else 0,
                statistics.median(loser_mfes) if loser_mfes else 0))
        # Key question: can a tighter SL work?
        if winner_maes:
            pct_under_5 = sum(1 for m in winner_maes if m <= 5) / len(winner_maes) * 100
            pct_under_10 = sum(1 for m in winner_maes if m <= 10) / len(winner_maes) * 100
            pct_under_15 = sum(1 for m in winner_maes if m <= 15) / len(winner_maes) * 100
            pct_under_20 = sum(1 for m in winner_maes if m <= 20) / len(winner_maes) * 100
            print("    Winner MAE distribution: <=5t: {:.0f}%, <=10t: {:.0f}%, <=15t: {:.0f}%, <=20t: {:.0f}%".format(
                pct_under_5, pct_under_10, pct_under_15, pct_under_20))
        print()

    if holds:
        print("  HOLD TIME:")
        print("    All:     mean={:.0f}s, med={:.0f}s, p10={:.0f}s, p90={:.0f}s".format(
            statistics.mean(holds), statistics.median(holds),
            sorted(holds)[int(.1 * len(holds))], sorted(holds)[int(.9 * len(holds))]))
        if winner_holds:
            print("    Winners: mean={:.0f}s, med={:.0f}s".format(
                statistics.mean(winner_holds), statistics.median(winner_holds)))
        if loser_holds:
            print("    Losers:  mean={:.0f}s, med={:.0f}s".format(
                statistics.mean(loser_holds), statistics.median(loser_holds)))
        print()

    # BUY/SELL split
    buys = [t for t in all_trades if t["side"] == "BUY"]
    sells = [t for t in all_trades if t["side"] == "SELL"]
    buy_pnl = sum(t["pnl_ticks"] for t in buys)
    sell_pnl = sum(t["pnl_ticks"] for t in sells)
    buy_wr = sum(1 for t in buys if t["pnl_ticks"] > 0) / len(buys) * 100 if buys else 0
    sell_wr = sum(1 for t in sells if t["pnl_ticks"] > 0) / len(sells) * 100 if sells else 0
    print("  LONG/SHORT:")
    print("    BUY:  {}t, {:.1f}tk (${:,.0f}), WR={:.1f}%".format(len(buys), buy_pnl, buy_pnl * TICK, buy_wr))
    print("    SELL: {}t, {:.1f}tk (${:,.0f}), WR={:.1f}%".format(len(sells), sell_pnl, sell_pnl * TICK, sell_wr))
    print()

    # Regime breakdown
    print("  REGIME BREAKDOWN:")
    for month, label in [(12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar")]:
        md = [d for d in daily if int(d["date"][5:7]) == month]
        if md:
            mp = sum(d["pnl"] for d in md)
            mt = sum(d["trades"] for d in md)
            pos = sum(1 for d in md if d["pnl"] > 0)
            net_m = mp * TICK - mt * COMM
            print("    {}: {}d, {}t, {:.1f}tk (${:,.0f} net), {}/{} pos, avg={:.1f}tk/d".format(
                label, len(md), mt, mp, net_m, pos, len(md), mp / len(md)))
    print()

    # Consecutive wins/losses
    max_consec_win = 0
    max_consec_loss = 0
    cur_win = 0
    cur_loss = 0
    for p in pnls:
        if p > 0:
            cur_win += 1
            cur_loss = 0
            max_consec_win = max(max_consec_win, cur_win)
        elif p < 0:
            cur_loss += 1
            cur_win = 0
            max_consec_loss = max(max_consec_loss, cur_loss)
        else:
            cur_win = 0
            cur_loss = 0
    print("  STREAKS: Max consec wins={}, Max consec losses={}".format(max_consec_win, max_consec_loss))

    # Worst days
    worst = sorted(daily, key=lambda d: d["pnl"])[:5]
    best = sorted(daily, key=lambda d: d["pnl"], reverse=True)[:5]
    print()
    print("  WORST 5 DAYS:")
    for d in worst:
        print("    {} : {:.1f}tk ({}t) = ${:,.0f}".format(d["date"], d["pnl"], d["trades"], d["pnl"] * TICK))
    print("  BEST 5 DAYS:")
    for d in best:
        print("    {} : {:.1f}tk ({}t) = ${:,.0f}".format(d["date"], d["pnl"], d["trades"], d["pnl"] * TICK))

    print()
    print("-" * 65)
