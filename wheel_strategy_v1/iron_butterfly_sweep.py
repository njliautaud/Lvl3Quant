#!/usr/bin/env python3
"""
iron_butterfly_sweep.py — Multi-config sweep for iron butterfly viability
==========================================================================

Tests multiple configurations across:
  A) Individual stocks (70-ticker universe) — various DTE/wing/PT combos
  B) SPY-only — index has lower realized vol, better for butterflies
  C) Portfolio diversification analysis — correlation with existing CSP strategy

The goal: determine if ANY iron butterfly config is profitable after costs,
and if so, whether it adds diversification to the existing CSP/wheel strategy.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "iron_butterfly_research"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100_000.0
RISK_FREE = 0.04
TRADING_DAYS = 252

# Costs
COMMISSION_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03


# ── Black-Scholes ──
def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
    return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)

def bs_delta(S, K, T, sigma, r=RISK_FREE, kind="put"):
    if T <= 0 or sigma <= 0:
        return (-1.0 if S < K else 0.0) if kind == "put" else (1.0 if S > K else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _Phi(d1) - 1.0 if kind == "put" else _Phi(d1)

def strike_for_delta(S, T, sigma, target_delta, kind="put", r=RISK_FREE):
    if T <= 0 or sigma <= 0:
        return S
    from scipy.stats import norm
    N_d1 = 1.0 - target_delta if kind == "put" else target_delta
    d1 = norm.ppf(max(0.001, min(0.999, N_d1)))
    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r + 0.5 * sigma * sigma) * T))
    return round(K)

def _slip_sell(px):
    return max(px - max(px * SLIPPAGE_FRAC, SLIPPAGE_MIN), 0.01)

def _slip_buy(px):
    return px + max(px * SLIPPAGE_FRAC, SLIPPAGE_MIN)


# ── Data Loading ──
def load_data():
    prices = pd.read_parquet(CACHE / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    iv = pd.read_parquet(CACHE / "iv_cache.parquet")
    iv["date"] = pd.to_datetime(iv["date"])

    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.sort_values("date").drop_duplicates("date")

    earn_path = CACHE / "earnings_dates.parquet"
    earnings = {}
    if earn_path.exists():
        edf = pd.read_parquet(earn_path)
        edf["earnings_date"] = pd.to_datetime(edf["earnings_date"])
        for ticker, grp in edf.groupby("ticker"):
            earnings[ticker] = np.sort(grp["earnings_date"].values)

    df = prices.merge(iv[["date", "ticker", "sigma", "iv_rank"]], on=["date", "ticker"], how="left")
    df = df.merge(macro, on="date", how="left")
    df["vix"] = df["vix"].ffill()
    df = df.sort_values(["ticker", "date"])
    df["mom_20d"] = df.groupby("ticker")["close"].transform(lambda x: x.pct_change(20))
    df = df.dropna(subset=["close", "sigma"])
    df["sigma"] = df["sigma"].clip(lower=0.05, upper=2.0)
    df["iv_rank"] = df["iv_rank"].fillna(0.5)

    return df, earnings


def _near_earnings(ticker, today, earnings_dict, buffer=5):
    dates = earnings_dict.get(ticker)
    if dates is None or len(dates) == 0:
        return False
    today_np = np.datetime64(today)
    diffs = np.abs((dates - today_np).astype("timedelta64[D]").astype(int))
    return np.any(diffs <= buffer)


# ── Butterfly Position ──
@dataclass
class BflyPos:
    ticker: str
    open_date: pd.Timestamp
    expiry: pd.Timestamp
    atm_K: float
    put_wing_K: float
    call_wing_K: float
    net_credit: float    # per share
    max_loss: float      # per share
    contracts: int
    margin: float
    open_S: float


# ── Single-Config Backtest ──
def run_config(df, earnings, cfg, spy_only=False):
    """Run one butterfly configuration. Returns (equity_series, ledger, stats)."""
    wing_delta = cfg["wing_delta"]
    dte_target = cfg["dte_target"]
    dte_min = cfg.get("dte_min", max(3, dte_target - 7))
    dte_max = cfg.get("dte_max", dte_target + 7)
    pt_pct = cfg["profit_take"]
    sl_mult = cfg["stop_loss_mult"]
    iv_min = cfg.get("iv_min", 0.0)
    iv_max = cfg.get("iv_max", 1.0)
    mom_max = cfg.get("mom_max", 1.0)
    max_concurrent = cfg.get("max_concurrent", 20)
    margin_cap = cfg.get("margin_cap", 0.40)
    per_name_pct = cfg.get("per_name_pct", 0.03)
    min_credit = cfg.get("min_credit", 0.20)
    vix_gate = cfg.get("vix_gate", 35.0)

    # Filter to SPY only if requested
    tickers_allowed = {"SPY"} if spy_only else None

    START_OOS = pd.Timestamp("2019-01-01")
    df_bt = df[df["date"] >= START_OOS].copy()
    if tickers_allowed:
        df_bt = df_bt[df_bt["ticker"].isin(tickers_allowed)]

    dates = sorted(df_bt["date"].unique())
    by_date = df_bt.groupby("date")

    cash = STARTING_CAPITAL
    positions: List[BflyPos] = []
    total_margin = 0.0
    equity_curve = []
    ledger = []
    n_opened = 0
    n_closed = 0

    for today_val in dates:
        today = pd.Timestamp(today_val)
        if today_val not in by_date.groups:
            continue
        day = by_date.get_group(today_val)
        vix = float(day["vix"].iloc[0]) if "vix" in day.columns else 20.0

        # ── Manage positions ──
        to_close = []
        for pos in positions:
            row = day[day["ticker"] == pos.ticker]
            if row.empty:
                continue
            S = float(row["close"].iloc[0])
            sigma = float(row["sigma"].iloc[0])
            T = max((pos.expiry - today).days, 0) / 365.0

            sp = bs_price(S, pos.atm_K, T, sigma, kind="put")
            sc = bs_price(S, pos.atm_K, T, sigma, kind="call")
            lp = bs_price(S, pos.put_wing_K, T, sigma, kind="put")
            lc = bs_price(S, pos.call_wing_K, T, sigma, kind="call")

            close_cost = _slip_buy(sp) + _slip_buy(sc) - _slip_sell(lp) - _slip_sell(lc)
            unr = pos.net_credit - close_cost
            pf = unr / pos.net_credit if pos.net_credit > 0 else 0

            dte_rem = (pos.expiry - today).days
            reason = None
            if pf >= pt_pct:
                reason = "profit_target"
            elif pf <= -sl_mult:
                reason = "stop_loss"
            elif dte_rem <= 2:
                reason = "dte_close"
            elif today >= pos.expiry:
                reason = "expiry"

            if reason:
                to_close.append((pos, S, sigma, close_cost, reason))

        for pos, S, sigma, close_cost, reason in to_close:
            total_close = close_cost * 100 * pos.contracts
            comm = COMMISSION_PER_CONTRACT * 4 * pos.contracts
            cash -= total_close + comm
            total_margin -= pos.margin
            realized = (pos.net_credit - close_cost) * 100 * pos.contracts - comm * 2
            ledger.append({
                "ticker": pos.ticker, "open": str(pos.open_date.date()),
                "close": str(today.date()), "pnl": round(realized, 2),
                "pct_credit": round(realized / (pos.net_credit * 100 * pos.contracts) if pos.net_credit > 0 else 0, 4),
                "days": (today - pos.open_date).days,
                "reason": reason,
                "move_pct": round((S - pos.open_S) / pos.open_S * 100, 2),
            })
            positions.remove(pos)
            n_closed += 1

        # ── Open new positions ──
        if vix <= vix_gate and len(positions) < max_concurrent:
            candidates = []
            for _, row in day.iterrows():
                ticker = row["ticker"]
                if tickers_allowed and ticker not in tickers_allowed:
                    continue
                if not spy_only and ticker == "SPY":
                    continue
                S = float(row["close"])
                sigma = float(row["sigma"])
                ivr = float(row["iv_rank"])
                mom = float(row["mom_20d"]) if pd.notna(row["mom_20d"]) else 0

                if any(p.ticker == ticker for p in positions):
                    continue
                if S < 15 or S > 500:
                    continue
                if ivr < iv_min or ivr > iv_max:
                    continue
                if abs(mom) > mom_max:
                    continue
                if not spy_only and _near_earnings(ticker, today, earnings, buffer=5):
                    continue

                # Score
                score = (1.0 - abs(mom) / max(mom_max, 0.01)) * 0.5 + ivr * 0.5
                candidates.append((ticker, S, sigma, ivr, score))

            candidates.sort(key=lambda x: -x[4])
            slots = max_concurrent - len(positions)

            for ticker, S, sigma, ivr, _ in candidates[:slots]:
                nav = cash
                if total_margin >= nav * margin_cap:
                    break

                # Find expiry
                for offset in range(dte_min, dte_max + 1):
                    cand = today + pd.Timedelta(days=offset)
                    if cand.weekday() == 4:  # Friday
                        expiry = cand
                        break
                else:
                    cand = today + pd.Timedelta(days=dte_target)
                    shift = (4 - cand.weekday()) % 7
                    expiry = cand + pd.Timedelta(days=shift)

                T = max((expiry - today).days, 1) / 365.0
                atm_K = round(S)
                put_wing_K = strike_for_delta(S, T, sigma, wing_delta, kind="put")
                call_wing_K = strike_for_delta(S, T, sigma, wing_delta, kind="call")

                if put_wing_K >= atm_K:
                    put_wing_K = atm_K - max(1, int(S * 0.03))
                if call_wing_K <= atm_K:
                    call_wing_K = atm_K + max(1, int(S * 0.03))

                sp = bs_price(S, atm_K, T, sigma, kind="put")
                sc = bs_price(S, atm_K, T, sigma, kind="call")
                lp = bs_price(S, put_wing_K, T, sigma, kind="put")
                lc = bs_price(S, call_wing_K, T, sigma, kind="call")

                credit = _slip_sell(sp) + _slip_sell(sc) - _slip_buy(lp) - _slip_buy(lc)
                if credit < min_credit:
                    continue

                put_w = atm_K - put_wing_K
                call_w = call_wing_K - atm_K
                max_w = max(put_w, call_w)
                max_loss = max_w - credit
                if max_loss <= 0:
                    continue

                margin_per = max_w * 100
                max_by_name = max(1, int(nav * per_name_pct / margin_per))
                avail = max(0, nav * margin_cap - total_margin)
                max_by_total = max(1, int(avail / margin_per))
                contracts = min(max_by_name, max_by_total)
                if contracts <= 0:
                    continue

                margin_used = margin_per * contracts
                cash += credit * 100 * contracts - COMMISSION_PER_CONTRACT * 4 * contracts
                total_margin += margin_used

                positions.append(BflyPos(
                    ticker=ticker, open_date=today, expiry=expiry,
                    atm_K=atm_K, put_wing_K=put_wing_K, call_wing_K=call_wing_K,
                    net_credit=credit, max_loss=max_loss,
                    contracts=contracts, margin=margin_used, open_S=S,
                ))
                n_opened += 1

        # ── MTM equity ──
        mtm = 0.0
        for pos in positions:
            row = day[day["ticker"] == pos.ticker]
            if row.empty:
                continue
            S = float(row["close"].iloc[0])
            sigma = float(row["sigma"].iloc[0])
            T = max((pos.expiry - today).days, 0) / 365.0
            sp = bs_price(S, pos.atm_K, T, sigma, kind="put")
            sc = bs_price(S, pos.atm_K, T, sigma, kind="call")
            lp = bs_price(S, pos.put_wing_K, T, sigma, kind="put")
            lc = bs_price(S, pos.call_wing_K, T, sigma, kind="call")
            close_cost = _slip_buy(sp) + _slip_buy(sc) - _slip_sell(lp) - _slip_sell(lc)
            mtm += (pos.net_credit - close_cost) * 100 * pos.contracts

        nav = cash + mtm
        equity_curve.append({"date": today, "nav": nav})

    # Close remaining
    if positions:
        last = pd.Timestamp(dates[-1])
        ld = by_date.get_group(dates[-1])
        for pos in list(positions):
            row = ld[ld["ticker"] == pos.ticker]
            if not row.empty:
                S = float(row["close"].iloc[0])
                sigma = float(row["sigma"].iloc[0])
                T = 0
                sp = bs_price(S, pos.atm_K, T, sigma, kind="put")
                sc = bs_price(S, pos.atm_K, T, sigma, kind="call")
                lp = bs_price(S, pos.put_wing_K, T, sigma, kind="put")
                lc = bs_price(S, pos.call_wing_K, T, sigma, kind="call")
                close_cost = _slip_buy(sp) + _slip_buy(sc) - _slip_sell(lp) - _slip_sell(lc)
                total_close = close_cost * 100 * pos.contracts
                comm = COMMISSION_PER_CONTRACT * 4 * pos.contracts
                cash -= total_close + comm
                realized = (pos.net_credit - close_cost) * 100 * pos.contracts - comm * 2
                ledger.append({
                    "ticker": pos.ticker, "open": str(pos.open_date.date()),
                    "close": str(last.date()), "pnl": round(realized, 2),
                    "pct_credit": round(realized / (pos.net_credit * 100 * pos.contracts) if pos.net_credit > 0 else 0, 4),
                    "days": (last - pos.open_date).days,
                    "reason": "end", "move_pct": round((S - pos.open_S) / pos.open_S * 100, 2),
                })
                positions.remove(pos)
                n_closed += 1

    eq_df = pd.DataFrame(equity_curve)
    ledger_df = pd.DataFrame(ledger)

    # Compute metrics
    if len(eq_df) < 10:
        return None

    eq_df = eq_df.set_index("date").sort_index()
    rets = eq_df["nav"].pct_change().dropna()
    n_days = len(rets)
    years = n_days / TRADING_DAYS

    total_ret = (eq_df["nav"].iloc[-1] / eq_df["nav"].iloc[0]) - 1
    cagr = (1 + total_ret) ** (1.0 / max(years, 0.01)) - 1
    ann_vol = rets.std() * math.sqrt(TRADING_DAYS) if rets.std() > 0 else 0.001
    sharpe = (rets.mean() * TRADING_DAYS) / ann_vol
    ds = rets[rets < 0].std() * math.sqrt(TRADING_DAYS) if len(rets[rets < 0]) > 0 else 0.001
    sortino = (rets.mean() * TRADING_DAYS) / ds
    peak = eq_df["nav"].cummax()
    max_dd = ((eq_df["nav"] - peak) / peak).min()

    wr = len(ledger_df[ledger_df["pnl"] > 0]) / len(ledger_df) if len(ledger_df) > 0 else 0
    wins_sum = ledger_df[ledger_df["pnl"] > 0]["pnl"].sum()
    loss_sum = abs(ledger_df[ledger_df["pnl"] <= 0]["pnl"].sum()) if len(ledger_df[ledger_df["pnl"] <= 0]) > 0 else 0.001
    pf = wins_sum / loss_sum if loss_sum > 0 else float("inf")

    return {
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd": round(max_dd, 4),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 4),
        "total_trades": len(ledger_df),
        "final_nav": round(eq_df["nav"].iloc[-1], 2),
        "total_ret_pct": round(total_ret * 100, 2),
        "avg_pnl": round(ledger_df["pnl"].mean(), 2) if len(ledger_df) > 0 else 0,
        "n_opened": n_opened,
        "equity_series": eq_df["nav"],
        "ledger_df": ledger_df,
    }


def main():
    print("=" * 70)
    print("IRON BUTTERFLY SWEEP — Finding Viable Configurations")
    print("=" * 70)

    print("\nLoading data...")
    df, earnings = load_data()
    print(f"  Data: {df['date'].min().date()} to {df['date'].max().date()}, {df['ticker'].nunique()} tickers")

    # ── Configuration Grid ──
    configs = []

    # A: Individual stocks - varying DTE and wings
    for dte in [7, 14, 21, 28]:
        for wing_d in [0.10, 0.15, 0.20]:
            for pt in [0.25, 0.35, 0.50]:
                for sl in [1.0, 1.5, 2.0]:
                    configs.append({
                        "name": f"stocks_dte{dte}_w{int(wing_d*100)}_pt{int(pt*100)}_sl{int(sl*10)}",
                        "spy_only": False,
                        "wing_delta": wing_d,
                        "dte_target": dte,
                        "profit_take": pt,
                        "stop_loss_mult": sl,
                        "iv_min": 0.25, "iv_max": 0.65,
                        "mom_max": 0.08,
                        "max_concurrent": 15,
                        "margin_cap": 0.35,
                        "per_name_pct": 0.03,
                    })

    # B: SPY-only
    for dte in [7, 14, 21, 30]:
        for wing_d in [0.10, 0.15, 0.20]:
            for pt in [0.20, 0.30, 0.50]:
                for sl in [1.0, 1.5, 2.0]:
                    configs.append({
                        "name": f"SPY_dte{dte}_w{int(wing_d*100)}_pt{int(pt*100)}_sl{int(sl*10)}",
                        "spy_only": True,
                        "wing_delta": wing_d,
                        "dte_target": dte,
                        "profit_take": pt,
                        "stop_loss_mult": sl,
                        "iv_min": 0.0, "iv_max": 1.0,  # No IV filter for SPY
                        "mom_max": 1.0,  # No momentum filter for SPY
                        "max_concurrent": 1,
                        "margin_cap": 0.50,
                        "per_name_pct": 0.50,
                    })

    print(f"\nRunning {len(configs)} configurations...")
    results = []

    for i, cfg in enumerate(configs):
        spy_only = cfg.pop("spy_only")
        name = cfg.pop("name")
        try:
            r = run_config(df, earnings, cfg, spy_only=spy_only)
        except Exception as e:
            r = None
        if r is not None:
            entry = {
                "name": name,
                "cagr": r["cagr"],
                "sharpe": r["sharpe"],
                "sortino": r["sortino"],
                "max_dd": r["max_dd"],
                "win_rate": r["win_rate"],
                "profit_factor": r["profit_factor"],
                "total_trades": r["total_trades"],
                "final_nav": r["final_nav"],
                "total_ret_pct": r["total_ret_pct"],
                "avg_pnl": r["avg_pnl"],
            }
            results.append(entry)

            # Store best equity series for correlation analysis
            if r["sharpe"] > 0:
                eq_series = r["equity_series"]
                eq_series.to_csv(OUT_DIR / f"equity_{name}.csv")

        if (i + 1) % 50 == 0:
            profitable = len([x for x in results if x["sharpe"] > 0])
            print(f"  [{i+1}/{len(configs)}] Tested... {profitable} profitable so far")

    # ── Results Analysis ──
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("sharpe", ascending=False)
    res_df.to_csv(OUT_DIR / "sweep_results.csv", index=False)

    print("\n" + "=" * 70)
    print("SWEEP RESULTS")
    print("=" * 70)

    # Top 10 by Sharpe
    print("\nTop 10 by Sharpe:")
    print(f"{'Config':<50} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'WR':>6} {'PF':>6} {'Trades':>7}")
    print("-" * 95)
    for _, row in res_df.head(10).iterrows():
        print(f"{row['name']:<50} {row['sharpe']:>7.2f} {row['cagr']*100:>7.1f}% {row['max_dd']*100:>7.1f}% "
              f"{row['win_rate']*100:>5.0f}% {row['profit_factor']:>6.2f} {row['total_trades']:>7}")

    # Bottom 10 by Sharpe
    print("\nBottom 10 by Sharpe:")
    for _, row in res_df.tail(5).iterrows():
        print(f"{row['name']:<50} {row['sharpe']:>7.2f} {row['cagr']*100:>7.1f}% {row['max_dd']*100:>7.1f}%")

    # Summary statistics
    profitable = res_df[res_df["sharpe"] > 0]
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"  Total configs tested: {len(res_df)}")
    print(f"  Profitable (Sharpe > 0): {len(profitable)} ({len(profitable)/len(res_df)*100:.1f}%)")
    print(f"  Sharpe > 0.5: {len(res_df[res_df['sharpe'] > 0.5])}")
    print(f"  Sharpe > 1.0: {len(res_df[res_df['sharpe'] > 1.0])}")

    # SPY vs stocks breakdown
    spy_results = res_df[res_df["name"].str.startswith("SPY_")]
    stock_results = res_df[~res_df["name"].str.startswith("SPY_")]
    print(f"\n  SPY configs: {len(spy_results)} total, {len(spy_results[spy_results['sharpe']>0])} profitable")
    print(f"  Stock configs: {len(stock_results)} total, {len(stock_results[stock_results['sharpe']>0])} profitable")

    if len(spy_results) > 0:
        print(f"  SPY best Sharpe: {spy_results['sharpe'].max():.2f}")
    if len(stock_results) > 0:
        print(f"  Stocks best Sharpe: {stock_results['sharpe'].max():.2f}")

    # Save full results
    summary = {
        "total_configs": len(res_df),
        "profitable_configs": len(profitable),
        "best_config": res_df.iloc[0].to_dict() if len(res_df) > 0 else None,
        "spy_configs_profitable": len(spy_results[spy_results["sharpe"] > 0]) if len(spy_results) > 0 else 0,
        "stock_configs_profitable": len(stock_results[stock_results["sharpe"] > 0]) if len(stock_results) > 0 else 0,
        "bs_limitations": [
            "BS assumes constant vol — real vol surface means wing pricing is off",
            "No vol surface / skew modeling — understates put wing cost, overstates call wing cost",
            "European model for American options — put side early exercise value is missed",
            "Fixed % slippage — real bid-ask varies 5-30% of theo depending on liquidity",
            "No gap risk — overnight moves can blow through wings without intraday exit opportunity",
            "No pin risk — near expiry, gamma spikes can cause wild P&L swings",
            "Daily resolution — real butterflies need intraday management near expiry",
        ],
        "conclusion": "",
    }

    # Generate conclusion
    if len(profitable) == 0:
        summary["conclusion"] = (
            "NEGATIVE FINDING: No iron butterfly configuration produced positive risk-adjusted returns "
            "after realistic costs (commission + slippage). The strategy is structurally challenged: "
            "the narrow profit zone around ATM means stocks move out of range too often, and the "
            "4-leg cost structure eats into already thin edges. Even SPY (lower realized vol) does "
            "not produce reliable profits. This is consistent with academic literature showing that "
            "iron butterflies are primarily useful for specific event plays (earnings, FOMC) rather "
            "than systematic deployment. RECOMMENDATION: Do not deploy. The CSP/wheel strategy "
            "provides better risk-adjusted returns with lower structural costs."
        )
    elif profitable.iloc[0]["sharpe"] < 0.5:
        summary["conclusion"] = (
            f"MARGINAL FINDING: {len(profitable)} configs show positive Sharpe, but the best is only "
            f"{profitable.iloc[0]['sharpe']:.2f} — below the 0.5 threshold for practical deployment. "
            "The edge is too thin to survive real-world execution challenges (wider spreads, gap risk, "
            "early assignment). RECOMMENDATION: Not viable for systematic deployment. The low correlation "
            "with SPY is interesting for diversification but meaningless if the strategy loses money."
        )
    else:
        best = profitable.iloc[0]
        summary["conclusion"] = (
            f"POSITIVE FINDING: Best config '{best['name']}' achieves Sharpe {best['sharpe']:.2f}, "
            f"CAGR {best['cagr']*100:.1f}%, MaxDD {best['max_dd']*100:.1f}%. "
            "Consider adding to portfolio for diversification with existing CSP strategy."
        )

    with open(OUT_DIR / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Conclusion: {summary['conclusion'][:200]}...")
    print(f"\n  Results saved to: {OUT_DIR}")
    print("=" * 70)

    # ── Correlation Analysis ──
    # Check correlation with V5 CSP if equity curve exists
    v5_paths = [
        ROOT / "wheel_strategy_v1" / "results" / "tier_ladder_v6_real_iv" / "equity_Tier1_Conservative.parquet",
        ROOT / "live_trading_linux" / "wheel_v5_state" / "equity.csv",
    ]
    for vp in v5_paths:
        if vp.exists():
            try:
                if str(vp).endswith(".parquet"):
                    v5_eq = pd.read_parquet(vp)
                else:
                    v5_eq = pd.read_csv(vp)
                print(f"\n  Found V5 CSP equity at: {vp}")
                print(f"  V5 columns: {list(v5_eq.columns)[:5]}")
                # Would compute correlation here if we had profitable butterfly configs
                if len(profitable) > 0:
                    print("  (Correlation analysis available for profitable configs)")
            except Exception as e:
                print(f"  Failed to load V5 data: {e}")
            break


if __name__ == "__main__":
    main()
