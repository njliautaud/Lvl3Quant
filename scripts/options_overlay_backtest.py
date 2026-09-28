#!/usr/bin/env python3
"""
Options Overlay Strategy Backtest
=================================
Tests 6 options overlay variants against shares-only baselines.
Uses Black-Scholes approximations for option pricing.

Capital: $669 | Max $200/trade | Max 3 concurrent positions
Commission: $0.65/contract | 5% bid-ask haircut entry AND exit
Non-compounding: every trade uses flat $200 budget.
Fractional contracts allowed (simulating proportional exposure).
"""

import json
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "CRM"]
START_DATE = "2022-01-01"
END_DATE = "2026-07-28"
STARTING_CAPITAL = 669.0
MAX_TRADE_SIZE = 200.0
MAX_CONCURRENT = 3
COMMISSION = 0.65
BID_ASK_HAIRCUT = 0.05


def approx_call_price(stock_price, dte, otm_pct=0.0):
    """ATM call ≈ S * IV * sqrt(T) * 0.4 (Black-Scholes ATM approximation).
    Using IV=30% for growth stocks (realistic for FAANG/tech).
    S * 0.30 * sqrt(DTE/365) * 0.4 = S * 0.12 * sqrt(DTE/365)."""
    atm = stock_price * 0.12 * math.sqrt(max(dte, 1) / 365.0)
    if otm_pct > 0:
        # OTM discount: roughly exponential decay
        atm *= max(0.02, math.exp(-otm_pct * 12))
    return max(atm, 0.01)


def approx_put_price(stock_price, dte, otm_pct=0.0):
    return approx_call_price(stock_price, dte, otm_pct)


def contracts_for_budget(premium_per_share, budget=MAX_TRADE_SIZE):
    """Fractional contracts to spend exactly $budget on premium + costs.
    Returns (n_contracts, total_cost)."""
    contract_cost = premium_per_share * 100 * (1 + BID_ASK_HAIRCUT) + COMMISSION
    if contract_cost <= 0:
        return 0, 0
    n = budget / contract_cost
    total = n * contract_cost
    return n, total


def sell_value(premium_per_share, n_contracts):
    """Net proceeds from selling."""
    gross = premium_per_share * 100 * n_contracts
    return max(0, gross * (1 - BID_ASK_HAIRCUT) - COMMISSION * max(1, n_contracts))


def compute_rsi(series, period=5):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_sma(series, period=200):
    return series.rolling(window=period, min_periods=period).mean()


def download_data():
    print(f"Downloading {len(UNIVERSE)} stocks...")
    data = {}
    for t in UNIVERSE:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 200:
                df["RSI5"] = compute_rsi(df["Close"], 5)
                df["SMA200"] = compute_sma(df["Close"], 200)
                df["Return_1m"] = df["Close"].pct_change(21)
                data[t] = df
                print(f"  {t}: {len(df)} days")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")
    return data


def simulate_earnings_dates(df):
    idx = df.index
    dates = []
    for year in range(idx[0].year, idx[-1].year + 1):
        for month in [1, 4, 7, 10]:
            target = pd.Timestamp(year=year, month=month, day=20)
            mask = (idx >= target - timedelta(days=5)) & (idx <= target + timedelta(days=5))
            cands = idx[mask]
            if len(cands):
                dates.append(cands[len(cands) // 2])
    return dates


def _sharpe(pnls):
    if len(pnls) < 2:
        return 0.0
    arr = np.array(pnls, dtype=float)
    m, s = arr.mean(), arr.std(ddof=1)
    if s == 0:
        return 0.0
    tpy = max(1, len(arr) / 4.5)
    return float((m / s) * math.sqrt(tpy))


def _result(desc, opt_trades, shr_pnls):
    opt_pnls = [t["pnl"] for t in opt_trades]
    opt_total = sum(opt_pnls)
    shr_total = sum(shr_pnls)
    return {
        "desc": desc,
        "num_trades": len(opt_trades),
        "option_total_pnl": round(opt_total, 2),
        "option_return_pct": round(opt_total / STARTING_CAPITAL * 100, 2),
        "option_final": round(STARTING_CAPITAL + opt_total, 2),
        "shares_total_pnl": round(shr_total, 2),
        "shares_return_pct": round(shr_total / STARTING_CAPITAL * 100, 2),
        "shares_final": round(STARTING_CAPITAL + shr_total, 2),
        "sharpe": round(_sharpe(opt_pnls), 3),
        "shares_sharpe": round(_sharpe(shr_pnls), 3),
        "win_rate": round(sum(1 for p in opt_pnls if p > 0) / max(1, len(opt_pnls)) * 100, 1),
        "avg_pnl": round(float(np.mean(opt_pnls)), 2) if opt_pnls else 0,
        "avg_shares_pnl": round(float(np.mean(shr_pnls)), 2) if shr_pnls else 0,
        "trades": opt_trades
    }


# ═══════════════════════════════════════════════════════════════════════════
# A: RSI Oversold Calls
# ═══════════════════════════════════════════════════════════════════════════

def strategy_a(data):
    """Buy ATM calls (30 DTE) worth $200 on RSI<20 + >200SMA. Exit RSI>50 or 30d."""
    opt_trades, shr_pnls = [], []
    open_pos = {}
    all_dates = sorted(set().union(*(set(df.index) for df in data.values())))

    for date in all_dates:
        for tk in list(open_pos.keys()):
            p = open_pos[tk]
            df = data[tk]
            if date not in df.index:
                continue
            row = df.loc[date]
            days = (date - p["edate"]).days
            rsi = row["RSI5"]
            if pd.isna(rsi):
                continue
            if rsi > 50 or days >= 30:
                ex = row["Close"]
                intrinsic = max(ex - p["ep"], 0)
                days_left = max(0, 30 - days)
                if intrinsic > 0:
                    tv = p["prem"] * math.sqrt(days_left / 30) * 0.5
                    exit_val_ps = intrinsic + tv  # per-share option value
                else:
                    exit_val_ps = p["prem"] * math.sqrt(days_left / 30) * max(0.1, 1 + (ex - p["ep"]) / p["ep"] * 3)

                proceeds = sell_value(exit_val_ps, p["nc"])
                net_pnl = proceeds - p["cost"]
                net_pnl = max(net_pnl, -p["cost"])

                shr = (MAX_TRADE_SIZE / p["ep"]) * (ex - p["ep"])
                opt_trades.append({"ticker": tk, "entry": str(p["edate"].date()),
                                   "exit": str(date.date()), "days": days,
                                   "move_pct": round((ex - p["ep"]) / p["ep"] * 100, 2),
                                   "pnl": round(net_pnl, 2)})
                shr_pnls.append(round(shr, 2))
                del open_pos[tk]

        if len(open_pos) >= MAX_CONCURRENT:
            continue
        for tk, df in data.items():
            if tk in open_pos or len(open_pos) >= MAX_CONCURRENT:
                continue
            if date not in df.index:
                continue
            row = df.loc[date]
            rsi, px, sma = row["RSI5"], row["Close"], row["SMA200"]
            if pd.isna(rsi) or pd.isna(sma):
                continue
            if rsi < 20 and px > sma:
                prem = approx_call_price(px, 30)
                nc, cost = contracts_for_budget(prem, MAX_TRADE_SIZE)
                if nc <= 0 or cost < 1:
                    continue
                open_pos[tk] = {"edate": date, "ep": px, "prem": prem, "nc": nc, "cost": cost}

    return _result("Buy ATM calls ($200) on RSI<20 + >200SMA, 30 DTE", opt_trades, shr_pnls)


# ═══════════════════════════════════════════════════════════════════════════
# B: Earnings Straddle
# ═══════════════════════════════════════════════════════════════════════════

def strategy_b(data):
    """Buy ATM straddle 2d before earnings, sell day after. IV crush modeled."""
    opt_trades, shr_pnls = [], []

    for tk, df in data.items():
        for ed in simulate_earnings_dates(df):
            if ed not in df.index:
                continue
            loc = df.index.get_loc(ed)
            eloc, xloc = loc - 2, loc + 1
            if eloc < 0 or xloc >= len(df):
                continue

            ep = df.iloc[eloc]["Close"]
            xp = df.iloc[xloc]["Close"]

            # Pre-earnings IV pump: 1.5x normal
            cp = approx_call_price(ep, 7) * 1.5
            pp = approx_put_price(ep, 7) * 1.5
            straddle_prem = cp + pp

            # Size to $200 budget: buy n straddles
            nc, cost = contracts_for_budget(straddle_prem, MAX_TRADE_SIZE)
            if nc <= 0 or cost < 1:
                continue

            # Post-earnings: IV crush kills extrinsic
            call_intr = max(xp - ep, 0)
            put_intr = max(ep - xp, 0)
            exit_val_ps = call_intr + put_intr

            if exit_val_ps > 0:
                proceeds = sell_value(exit_val_ps, nc)
            else:
                proceeds = 0
            net = proceeds - cost
            net = max(net, -cost)

            opt_trades.append({"ticker": tk, "earnings": str(ed.date()),
                               "abs_move_pct": round(abs(xp - ep) / ep * 100, 2),
                               "pnl": round(net, 2)})
            shr_pnls.append(0.0)

    return _result("ATM straddle 2d pre-earnings (IV crush modeled)", opt_trades, shr_pnls)


# ═══════════════════════════════════════════════════════════════════════════
# C: Covered Call on Rotation
# ═══════════════════════════════════════════════════════════════════════════

def strategy_c(data):
    """Hold $200 shares, sell proportional OTM covered call monthly."""
    opt_trades, shr_pnls = [], []
    tickers = [t for t in ["AAPL", "MSFT", "GOOGL"] if t in data]

    for tk in tickers:
        df = data[tk]
        months = df.resample("MS").first().index
        for ms in months:
            md = df[df.index >= ms].head(22)
            if len(md) < 10:
                continue
            ep, xp = md.iloc[0]["Close"], md.iloc[-1]["Close"]

            qty = MAX_TRADE_SIZE / ep
            lots = qty / 100.0
            call_prem = approx_call_price(ep, 30, otm_pct=0.10)
            # Premium from selling proportional contracts
            prem_received = sell_value(call_prem, lots)
            strike = ep * 1.10

            stock_pnl = qty * (min(xp, strike) - ep)
            cc_pnl = stock_pnl + prem_received
            shares_pnl = qty * (xp - ep)

            opt_trades.append({"ticker": tk, "month": str(ms.date()),
                               "stock_ret_pct": round((xp - ep) / ep * 100, 2),
                               "premium": round(prem_received, 2), "pnl": round(cc_pnl, 2)})
            shr_pnls.append(round(shares_pnl, 2))

    return _result("Sell 10% OTM covered call monthly (3-stock rotation)", opt_trades, shr_pnls)


# ═══════════════════════════════════════════════════════════════════════════
# D: Cash-Secured Put for Entry
# ═══════════════════════════════════════════════════════════════════════════

def strategy_d(data):
    """Sell ATM put on RSI<20 signals. Scale to $200 collateral."""
    opt_trades, shr_pnls = [], []
    open_pos = {}
    all_dates = sorted(set().union(*(set(df.index) for df in data.values())))

    for date in all_dates:
        for tk in list(open_pos.keys()):
            p = open_pos[tk]
            df = data[tk]
            if date not in df.index:
                continue
            row = df.loc[date]
            days = (date - p["edate"]).days
            rsi = row["RSI5"]
            if pd.isna(rsi):
                continue
            if rsi > 50 or days >= 30:
                xp = row["Close"]
                if xp < p["ep"]:
                    loss = (p["ep"] - xp) * 100 * p["nc"]
                    net = p["prem_rcv"] - loss
                    assigned = True
                else:
                    net = p["prem_rcv"]
                    assigned = False
                net = max(net, -MAX_TRADE_SIZE)

                shr = (MAX_TRADE_SIZE / p["ep"]) * (xp - p["ep"])
                opt_trades.append({"ticker": tk, "entry": str(p["edate"].date()),
                                   "exit": str(date.date()), "assigned": assigned,
                                   "move_pct": round((xp - p["ep"]) / p["ep"] * 100, 2),
                                   "pnl": round(net, 2)})
                shr_pnls.append(round(shr, 2))
                del open_pos[tk]

        if len(open_pos) >= MAX_CONCURRENT:
            continue
        for tk, df in data.items():
            if tk in open_pos or len(open_pos) >= MAX_CONCURRENT:
                continue
            if date not in df.index:
                continue
            row = df.loc[date]
            rsi, px, sma = row["RSI5"], row["Close"], row["SMA200"]
            if pd.isna(rsi) or pd.isna(sma):
                continue
            if rsi < 20 and px > sma:
                pp = approx_put_price(px, 30)
                # Collateral = $200, scale contracts proportionally
                nc = MAX_TRADE_SIZE / (px * 100)  # fractional
                prem_rcv = sell_value(pp, nc)
                open_pos[tk] = {"edate": date, "ep": px, "nc": nc, "prem_rcv": prem_rcv}

    res = _result("Sell ATM cash-secured put on RSI<20", opt_trades, shr_pnls)
    res["pct_assigned"] = round(sum(1 for t in opt_trades if t.get("assigned")) / max(1, len(opt_trades)) * 100, 1)
    return res


# ═══════════════════════════════════════════════════════════════════════════
# E: Protective Put
# ═══════════════════════════════════════════════════════════════════════════

def strategy_e(data):
    """Buy $200 shares + proportional 10% OTM put on RSI<20. Hold 14d."""
    prot_trades, shr_pnls = [], []
    open_pos = {}
    all_dates = sorted(set().union(*(set(df.index) for df in data.values())))

    for date in all_dates:
        for tk in list(open_pos.keys()):
            p = open_pos[tk]
            df = data[tk]
            if date not in df.index:
                continue
            row = df.loc[date]
            days = (date - p["edate"]).days
            rsi = row["RSI5"]
            if pd.isna(rsi):
                continue
            if rsi > 50 or days >= 14:
                xp = row["Close"]
                qty = MAX_TRADE_SIZE / p["ep"]
                shares_pnl = qty * (xp - p["ep"])

                put_strike = p["ep"] * 0.90
                lots = qty / 100.0
                put_payoff = max(put_strike - xp, 0) * 100 * lots
                prot_pnl = shares_pnl + put_payoff - p["put_cost"]

                prot_trades.append({"ticker": tk, "entry": str(p["edate"].date()),
                                    "exit": str(date.date()),
                                    "move_pct": round((xp - p["ep"]) / p["ep"] * 100, 2),
                                    "pnl": round(prot_pnl, 2),
                                    "put_cost": round(p["put_cost"], 2),
                                    "put_payoff": round(put_payoff, 2)})
                shr_pnls.append(round(shares_pnl, 2))
                del open_pos[tk]

        if len(open_pos) >= MAX_CONCURRENT:
            continue
        for tk, df in data.items():
            if tk in open_pos or len(open_pos) >= MAX_CONCURRENT:
                continue
            if date not in df.index:
                continue
            row = df.loc[date]
            rsi, px, sma = row["RSI5"], row["Close"], row["SMA200"]
            if pd.isna(rsi) or pd.isna(sma):
                continue
            if rsi < 20 and px > sma:
                qty = MAX_TRADE_SIZE / px
                lots = qty / 100.0
                pp = approx_put_price(px, 14, otm_pct=0.10)
                # Proportional put cost
                _, pc = contracts_for_budget(pp, pp * 100 * lots * (1 + BID_ASK_HAIRCUT) + COMMISSION)
                put_cost = pp * 100 * lots * (1 + BID_ASK_HAIRCUT) + COMMISSION * max(1, lots)
                open_pos[tk] = {"edate": date, "ep": px, "put_cost": put_cost}

    res = _result("Shares + 10% OTM put (14 DTE) on RSI<20", prot_trades, shr_pnls)
    res["protection_cost_total"] = round(sum(t["put_cost"] for t in prot_trades), 2)
    res["protection_payoff_total"] = round(sum(t["put_payoff"] for t in prot_trades), 2)
    return res


# ═══════════════════════════════════════════════════════════════════════════
# F: Weekly Call Spread on Momentum
# ═══════════════════════════════════════════════════════════════════════════

def strategy_f(data):
    """Bull call spread (ATM/+5% OTM), 7 DTE, on momentum. $200 budget."""
    opt_trades, shr_pnls = [], []
    open_pos = {}
    all_dates = sorted(set().union(*(set(df.index) for df in data.values())))

    for date in all_dates:
        for tk in list(open_pos.keys()):
            p = open_pos[tk]
            days = (date - p["edate"]).days
            if days < 7:
                continue
            df = data[tk]
            if date not in df.index:
                continue
            xp = df.loc[date, "Close"]
            ep = p["ep"]
            move_pct = (xp - ep) / ep

            width = ep * 0.05
            if move_pct <= 0:
                exit_val_ps = 0
            elif move_pct >= 0.05:
                exit_val_ps = width
            else:
                exit_val_ps = xp - ep

            proceeds = sell_value(exit_val_ps, p["nc"]) if exit_val_ps > 0 else 0
            net = proceeds - p["cost"]
            net = max(net, -p["cost"])

            shr = (MAX_TRADE_SIZE / ep) * (xp - ep)
            opt_trades.append({"ticker": tk, "entry": str(p["edate"].date()),
                               "exit": str(date.date()),
                               "move_pct": round(move_pct * 100, 2),
                               "pnl": round(net, 2)})
            shr_pnls.append(round(shr, 2))
            del open_pos[tk]

        if len(open_pos) >= MAX_CONCURRENT:
            continue
        for tk, df in data.items():
            if tk in open_pos or len(open_pos) >= MAX_CONCURRENT:
                continue
            if date not in df.index:
                continue
            row = df.loc[date]
            rsi, px, sma = row["RSI5"], row["Close"], row["SMA200"]
            ret = row["Return_1m"]
            if pd.isna(rsi) or pd.isna(sma) or pd.isna(ret):
                continue
            if px > sma and ret > 0 and 40 <= rsi <= 60:
                lp = approx_call_price(px, 7)
                sp = approx_call_price(px, 7, otm_pct=0.05)
                debit = lp - sp
                if debit <= 0:
                    continue
                nc, cost = contracts_for_budget(debit, MAX_TRADE_SIZE)
                if nc <= 0 or cost < 1:
                    continue
                open_pos[tk] = {"edate": date, "ep": px, "nc": nc, "cost": cost}

    return _result("Bull call spread (ATM/+5%) 7 DTE on momentum", opt_trades, shr_pnls)


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("OPTIONS OVERLAY STRATEGY BACKTEST")
    print(f"  Universe: {', '.join(UNIVERSE)}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Capital: ${STARTING_CAPITAL} | Max/trade: ${MAX_TRADE_SIZE} | Max concurrent: {MAX_CONCURRENT}")
    print(f"  Commission: ${COMMISSION}/contract | Bid-ask haircut: {BID_ASK_HAIRCUT*100:.0f}%")
    print(f"  Non-compounding, fractional contracts (proportional exposure)")
    print("=" * 90)

    data = download_data()
    if not data:
        print("ERROR: No data.")
        return
    print(f"\nLoaded {len(data)} stocks\n")

    strategies = {}
    for label, fn in [("A_RSI_Oversold_Calls", strategy_a),
                      ("B_Earnings_Straddle", strategy_b),
                      ("C_Covered_Call", strategy_c),
                      ("D_CSP_Entry", strategy_d),
                      ("E_Protective_Put", strategy_e),
                      ("F_Call_Spread", strategy_f)]:
        print(f"Running {label}...")
        res = fn(data)
        strategies[label] = res
        print(f"  {res['num_trades']} trades | Opt {res['option_return_pct']:+.1f}% | "
              f"Shr {res['shares_return_pct']:+.1f}% | Sharpe {res['sharpe']:.2f} | "
              f"WR {res['win_rate']:.0f}% | Avg ${res['avg_pnl']:.1f}")

    print("\n" + "=" * 110)
    print(f"RESULTS — flat ${MAX_TRADE_SIZE}/trade, max {MAX_CONCURRENT} concurrent, ${STARTING_CAPITAL} capital")
    print("=" * 110)
    print(f"{'Strategy':<28} {'OptFinal':>9} {'Opt%':>7} {'ShrFinal':>9} {'Shr%':>7} "
          f"{'OptSh':>6} {'ShrSh':>6} {'WR':>5} {'AvgOpt':>8} {'AvgShr':>8} {'N':>4}")
    print("-" * 110)

    for nm, r in strategies.items():
        print(f"{nm:<28} ${r['option_final']:>7.0f} {r['option_return_pct']:>+6.1f}% "
              f"${r['shares_final']:>7.0f} {r['shares_return_pct']:>+6.1f}% "
              f"{r['sharpe']:>6.2f} {r['shares_sharpe']:>6.2f} {r['win_rate']:>4.0f}% "
              f"${r['avg_pnl']:>6.1f} ${r['avg_shares_pnl']:>6.1f} {r['num_trades']:>4}")

    print(f"\n{'VERDICT':=^110}")
    for nm, r in strategies.items():
        alpha = r["option_return_pct"] - r["shares_return_pct"]
        if r["shares_return_pct"] == 0:
            if r["option_return_pct"] > 5 and r["sharpe"] > 0.3:
                v = f"PROFITABLE ({r['option_return_pct']:+.1f}%, Sharpe {r['sharpe']:.2f})"
            elif r["option_return_pct"] > 0:
                v = f"MARGINAL ({r['option_return_pct']:+.1f}%)"
            else:
                v = f"UNPROFITABLE ({r['option_return_pct']:+.1f}%)"
        elif alpha > 10 and r["sharpe"] > r["shares_sharpe"]:
            v = f"ADDS ALPHA ({alpha:+.1f}% vs shares, Sharpe {r['sharpe']:.2f} vs {r['shares_sharpe']:.2f})"
        elif alpha > 0 and r["sharpe"] >= r["shares_sharpe"]:
            v = f"SLIGHT EDGE ({alpha:+.1f}%, risk-adj {r['sharpe']:.2f} vs {r['shares_sharpe']:.2f})"
        elif r["sharpe"] > r["shares_sharpe"] and alpha > -5:
            v = f"BETTER RISK-ADJ (Sharpe {r['sharpe']:.2f} > {r['shares_sharpe']:.2f}, raw {alpha:+.1f}%)"
        elif alpha > -5:
            v = f"NO BENEFIT ({alpha:+.1f}% vs shares)"
        else:
            v = f"UNDERPERFORMS ({alpha:+.1f}% vs shares)"
        print(f"  {nm}: {v}")

    if "pct_assigned" in strategies.get("D_CSP_Entry", {}):
        print(f"\n  D: {strategies['D_CSP_Entry']['pct_assigned']:.0f}% of puts assigned")
    if "protection_cost_total" in strategies.get("E_Protective_Put", {}):
        e = strategies["E_Protective_Put"]
        pc, pp = e["protection_cost_total"], e["protection_payoff_total"]
        print(f"  E: Insurance cost ${pc:.0f}, payoff ${pp:.0f} → "
              f"{'NET POSITIVE' if pp > pc else 'NET DRAG'} ${abs(pp-pc):.0f}")

    print(f"\n{'RECOMMENDATION':=^110}")
    best = max(strategies.items(), key=lambda x: x[1]["sharpe"])
    print(f"  Best risk-adjusted: {best[0]} (Sharpe {best[1]['sharpe']:.2f})")
    worthwhile = [(n, r) for n, r in strategies.items()
                  if r["sharpe"] > r.get("shares_sharpe", 0) and r["option_return_pct"] > r["shares_return_pct"]]
    if worthwhile:
        print(f"  Worth implementing: {', '.join(n for n, _ in worthwhile)}")
    else:
        print("  No option overlay clearly beats shares-only.")
    print("  Note: Options add complexity. Only implement if edge is clear and you understand the risks.")

    # Save
    output = {
        "metadata": {
            "universe": UNIVERSE, "period": f"{START_DATE} to {END_DATE}",
            "starting_capital": STARTING_CAPITAL, "max_trade_size": MAX_TRADE_SIZE,
            "max_concurrent": MAX_CONCURRENT, "commission": COMMISSION,
            "bid_ask_haircut_pct": BID_ASK_HAIRCUT * 100,
            "compounding": False, "fractional_contracts": True,
            "run_date": datetime.now().isoformat()
        },
        "strategies": {}
    }
    for nm, r in strategies.items():
        s = {
            "description": r["desc"],
            "option_return_pct": r["option_return_pct"],
            "option_final": r["option_final"],
            "option_total_pnl": r["option_total_pnl"],
            "shares_return_pct": r["shares_return_pct"],
            "shares_final": r["shares_final"],
            "num_trades": r["num_trades"],
            "option_sharpe": r["sharpe"],
            "shares_sharpe": r["shares_sharpe"],
            "win_rate_pct": r["win_rate"],
            "avg_option_pnl": r["avg_pnl"],
            "avg_shares_pnl": r["avg_shares_pnl"],
            "alpha_vs_shares_pct": round(r["option_return_pct"] - r["shares_return_pct"], 2),
            "sample_trades": r["trades"][:5]
        }
        for k in ["pct_assigned", "protection_cost_total", "protection_payoff_total"]:
            if k in r:
                s[k] = r[k]
        output["strategies"][nm] = s

    out_path = Path("/home/jupiter/Lvl3Quant/data/options_overlay_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
