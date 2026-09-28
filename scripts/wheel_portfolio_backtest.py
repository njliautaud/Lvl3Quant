#!/usr/bin/env python3
"""
wheel_portfolio_backtest.py — Portfolio-level wheel backtest (HC #660)

Runs the full 31-name diversified basket simultaneously with:
- Weekly DTE (14d target)
- Equal-weight capital allocation (max 10% per name)
- liq_csp_only bear regime gate (SPY < 50d SMA → close CSPs, keep shares)
- Proper portfolio-level metrics: Sharpe, Sortino, MaxDD, Calmar

This answers: "What does the PORTFOLIO return look like when all 31 names
trade together with proper capital allocation and risk management?"
"""
from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "wheel_portfolio_backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ──
START_DATE = pd.Timestamp("2019-01-01")
STARTING_CAPITAL = 100_000.0
MAX_PER_NAME_PCT = 0.10  # 10% max per name
MAX_ACTIVE_POSITIONS = 10  # Cap total positions to control leverage
TRADING_DAYS = 252
RISK_FREE = 0.04

PUT_DELTA = 0.25
CALL_DELTA = 0.30
DTE_MIN = 10
DTE_MAX = 18
DTE_TARGET = 14  # Weekly
PROFIT_TAKE = 0.50
VIX_MAX = 35.0

COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03

# The 31-name quality-filtered basket from weekly sweep
BASKET = [
    'WYNN', 'EXC', 'XOM', 'TXN', 'EA', 'IBM', 'DLR', 'AEP', 'GILD', 'CAT',
    'VZ', 'CL', 'TMUS', 'IRM', 'CVS', 'AXP', 'DUK', 'ABT', 'CVX', 'PG',
    'HON', 'CSCO', 'HD', 'VLO', 'SBUX', 'GS', 'TGT', 'SPG', 'UNP', 'V', 'LIN',
]


# ── BS pricing ──
def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
    return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)

def strike_from_delta(S, T, sigma, target_delta, kind="put", r=RISK_FREE):
    if T <= 0 or sigma <= 0:
        return S
    p = (1.0 - abs(target_delta)) if kind == "put" else abs(target_delta)
    p = min(max(p, 1e-9), 1 - 1e-9)
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d_ = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
          3.754408661907416]
    pl, pu = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d_[0]*q+d_[1])*q+d_[2])*q+d_[3])*q+1)
    elif p <= pu:
        q = p - 0.5
        rr = q*q
        z = (((((a[0]*rr+a[1])*rr+a[2])*rr+a[3])*rr+a[4])*rr+a[5])*q / (((((b[0]*rr+b[1])*rr+b[2])*rr+b[3])*rr+b[4])*rr+1)
    else:
        q = math.sqrt(-2 * math.log(1-p))
        z = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d_[0]*q+d_[1])*q+d_[2])*q+d_[3])*q+1)
    d1 = z
    K = S * math.exp((r + 0.5 * sigma**2) * T - d1 * sigma * math.sqrt(T))
    return max(0.01, round(K, 2))

def slippage(premium):
    return max(SLIPPAGE_MIN, SLIPPAGE_FRAC * premium) if premium and premium > 0 else 0.0

def find_expiry(open_date):
    best, best_dist = None, 10000
    for d_off in range(DTE_MIN, DTE_MAX + 1):
        cand = open_date + pd.Timedelta(days=d_off)
        shift = (4 - cand.weekday()) % 7
        cand_fri = cand + pd.Timedelta(days=shift)
        dte = (cand_fri - open_date).days
        if dte < DTE_MIN or dte > DTE_MAX:
            continue
        dist = abs(dte - DTE_TARGET)
        if dist < best_dist:
            best, best_dist = cand_fri, dist
    return best


# ── Position tracking ──
@dataclass
class TickerPosition:
    ticker: str
    side: str       # 'short_put' | 'long_shares' | 'short_call'
    strike: float
    expiry: pd.Timestamp
    open_date: pd.Timestamp
    open_price: float  # premium per share
    contracts: int
    share_basis: float = 0.0


def load_data():
    """Load price data for basket + SPY (for regime gate) + VIX."""
    print("[load] Loading universe metadata ...")
    universe = pd.read_parquet(CACHE / "universe_expanded.parquet")

    print("[load] Loading prices ...")
    p1 = pd.read_parquet(CACHE / "prices.parquet")[["ticker", "date", "close"]].copy()
    p1["date"] = pd.to_datetime(p1["date"], utc=False)
    if p1["date"].dt.tz is not None:
        p1["date"] = p1["date"].dt.tz_localize(None)

    p2 = pd.read_parquet(CACHE / "prices_expanded.parquet")
    p2 = p2.rename(columns={"Close": "close"})[["ticker", "date", "close"]].copy()
    p2["date"] = pd.to_datetime(p2["date"], utc=False)
    if p2["date"].dt.tz is not None:
        p2["date"] = p2["date"].dt.tz_localize(None)

    prices = pd.concat([p1, p2], ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices = prices.dropna(subset=["close"])
    prices = prices[prices["close"] > 0]
    prices = prices[prices["date"] >= START_DATE].copy()

    # Compute realized vol
    prices["log_ret"] = prices.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))
    prices["sigma"] = prices.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(20, min_periods=15).std() * np.sqrt(252)
    )
    prices["sigma"] = prices["sigma"].clip(lower=0.05, upper=2.0)
    prices = prices.dropna(subset=["sigma"]).reset_index(drop=True)

    # VIX
    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"], utc=False)
    if macro["date"].dt.tz is not None:
        macro["date"] = macro["date"].dt.tz_localize(None)
    prices = prices.merge(macro, on="date", how="left")
    prices["vix"] = prices["vix"].ffill().fillna(20.0)

    # SPY for regime gate (50d SMA)
    spy = prices[prices["ticker"] == "SPY"][["date", "close"]].copy()
    if spy.empty:
        # Try loading SPY separately
        import yfinance as yf
        print("[load] SPY not in cache, fetching from yfinance ...")
        spy_data = yf.Ticker("SPY").history(period="max", interval="1d")
        spy = pd.DataFrame({"date": spy_data.index.tz_localize(None), "close": spy_data["Close"].values})

    spy = spy.sort_values("date")
    spy["spy_sma50"] = spy["close"].rolling(50).mean()
    spy["bear"] = spy["close"] < spy["spy_sma50"]
    spy_regime = spy[["date", "bear"]].dropna()

    # Filter to basket tickers only
    basket_set = set(BASKET)
    prices = prices[prices["ticker"].isin(basket_set)]

    # Build date-aligned data
    all_dates = sorted(prices["date"].unique())

    return prices, spy_regime, all_dates


def run_portfolio_backtest():
    """Run the full portfolio backtest."""
    t0 = time.time()
    prices, spy_regime, all_dates = load_data()

    # Build lookup: {(ticker, date): (close, sigma, vix)}
    price_lookup = {}
    for _, row in prices.iterrows():
        key = (row["ticker"], row["date"])
        price_lookup[key] = (row["close"], row["sigma"], row["vix"])

    # Build regime lookup: {date: is_bear}
    bear_lookup = {}
    for _, row in spy_regime.iterrows():
        bear_lookup[row["date"]] = row["bear"]

    # Portfolio state
    cash = STARTING_CAPITAL
    positions: dict[str, TickerPosition] = {}  # ticker -> position

    equity_series = []
    trade_log = []
    daily_pnl = []
    n_trades = 0
    wins = 0
    total_premium = 0.0

    prev_equity = STARTING_CAPITAL

    print(f"[backtest] Running portfolio on {len(BASKET)} names, {len(all_dates)} dates ...")

    for di, date in enumerate(all_dates):
        if di % 250 == 0:
            print(f"  [{di}/{len(all_dates)}] {date.date()} equity=${prev_equity:,.0f}")

        # Get regime
        is_bear = bear_lookup.get(date, False)

        # ── Process existing positions ──
        tickers_to_remove = []

        for ticker, pos in list(positions.items()):
            data = price_lookup.get((ticker, date))
            if data is None:
                continue
            S, sigma, vix = data
            T = max((pos.expiry - date).days, 0) / 365.0

            if pos.side == "short_put":
                opt = bs_price(S, pos.strike, T, sigma, kind="put")
                pf = (pos.open_price - opt) / pos.open_price if pos.open_price > 0 else 0
                is_expiry = date >= pos.expiry

                # Bear regime: close CSPs early (liq_csp_only)
                if is_bear and not is_expiry:
                    slip = slippage(opt)
                    cost = (opt + slip) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    realized = pos.open_price * 100 * pos.contracts - cost
                    cash += realized
                    n_trades += 1
                    if realized > 0:
                        wins += 1
                    trade_log.append({"date": str(date.date()), "ticker": ticker, "action": "bear_close_csp", "pnl": realized})
                    tickers_to_remove.append(ticker)
                    continue

                # Profit take
                if pf >= PROFIT_TAKE and not is_expiry:
                    slip = slippage(opt)
                    cost = (opt + slip) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    realized = pos.open_price * 100 * pos.contracts - cost
                    cash += realized
                    n_trades += 1
                    if realized > 0:
                        wins += 1
                    trade_log.append({"date": str(date.date()), "ticker": ticker, "action": "close_csp_profit", "pnl": realized})
                    tickers_to_remove.append(ticker)
                    continue

                if is_expiry:
                    if S < pos.strike:
                        # Assigned
                        cost = pos.strike * 100 * pos.contracts
                        cash -= cost
                        basis = pos.strike - pos.open_price
                        n_trades += 1
                        wins += 1
                        positions[ticker] = TickerPosition(
                            ticker=ticker, side="long_shares", strike=basis,
                            expiry=date, open_date=date, open_price=basis,
                            contracts=pos.contracts, share_basis=basis,
                        )
                        trade_log.append({"date": str(date.date()), "ticker": ticker, "action": "assigned", "pnl": 0})
                    else:
                        # Expired OTM
                        realized = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                        n_trades += 1
                        if realized > 0:
                            wins += 1
                        trade_log.append({"date": str(date.date()), "ticker": ticker, "action": "expired_otm", "pnl": realized})
                        tickers_to_remove.append(ticker)
                    continue

            elif pos.side == "short_call":
                opt = bs_price(S, pos.strike, T, sigma, kind="call")
                pf = (pos.open_price - opt) / pos.open_price if pos.open_price > 0 else 0
                is_expiry = date >= pos.expiry

                if pf >= PROFIT_TAKE and not is_expiry:
                    slip = slippage(opt)
                    cost = (opt + slip) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    realized = pos.open_price * 100 * pos.contracts - cost
                    cash += realized
                    n_trades += 1
                    if realized > 0:
                        wins += 1
                    positions[ticker] = TickerPosition(
                        ticker=ticker, side="long_shares", strike=pos.share_basis,
                        expiry=date, open_date=date, open_price=pos.share_basis,
                        contracts=pos.contracts, share_basis=pos.share_basis,
                    )
                    trade_log.append({"date": str(date.date()), "ticker": ticker, "action": "close_cc_profit", "pnl": realized})
                    continue

                if is_expiry:
                    if S > pos.strike:
                        # Called away
                        proceeds = pos.strike * 100 * pos.contracts
                        premium = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                        share_pnl = (pos.strike - pos.share_basis) * 100 * pos.contracts
                        cash += proceeds + premium
                        n_trades += 1
                        if (premium + share_pnl) > 0:
                            wins += 1
                        trade_log.append({"date": str(date.date()), "ticker": ticker, "action": "called_away", "pnl": premium + share_pnl})
                        tickers_to_remove.append(ticker)
                    else:
                        # CC expired, keep shares
                        realized = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                        cash += realized
                        n_trades += 1
                        if realized > 0:
                            wins += 1
                        positions[ticker] = TickerPosition(
                            ticker=ticker, side="long_shares", strike=pos.share_basis,
                            expiry=date, open_date=date, open_price=pos.share_basis,
                            contracts=pos.contracts, share_basis=pos.share_basis,
                        )
                        trade_log.append({"date": str(date.date()), "ticker": ticker, "action": "cc_expired", "pnl": realized})
                    continue

        for t in tickers_to_remove:
            if t in positions:
                del positions[t]

        # ── Open new positions / sell CCs ──
        # Compute current NAV for position sizing
        nav = cash
        for ticker, pos in positions.items():
            data = price_lookup.get((ticker, date))
            if data is None:
                continue
            S = data[0]
            if pos.side in ("long_shares", "short_call"):
                nav += S * 100 * pos.contracts

        max_per_name = nav * MAX_PER_NAME_PCT

        # Count CSP positions (excluding long_shares and short_call which are assignment management)
        n_csp = sum(1 for p in positions.values() if p.side == "short_put")

        for ticker in BASKET:
            data = price_lookup.get((ticker, date))
            if data is None:
                continue
            S, sigma, vix = data

            if ticker not in positions:
                # Open new CSP (blocked in bear regime, cap active CSPs)
                if n_csp >= MAX_ACTIVE_POSITIONS:
                    continue
                if is_bear:
                    continue
                if vix > VIX_MAX:
                    continue

                expiry = find_expiry(date)
                if expiry is None:
                    continue
                T = (expiry - date).days / 365.0
                K = strike_from_delta(S, T, sigma, PUT_DELTA, kind="put")
                prem = bs_price(S, K, T, sigma, kind="put")
                slip = slippage(prem)
                net_prem = prem - slip

                collateral = K * 100
                if collateral > max_per_name or collateral > cash * 0.50:
                    continue
                if net_prem < 0.05:
                    continue

                # Position sizing: max 10% of NAV in collateral per name
                max_contracts = max(1, int(max_per_name / collateral))
                contracts = min(max_contracts, 1)  # cap at 1 for diversification
                credit = net_prem * 100 * contracts - COST_PER_CONTRACT * contracts
                cash += credit
                total_premium += net_prem * 100 * contracts
                positions[ticker] = TickerPosition(
                    ticker=ticker, side="short_put", strike=K, expiry=expiry,
                    open_date=date, open_price=net_prem, contracts=contracts,
                )
                n_csp += 1

            elif positions[ticker].side == "long_shares":
                # Sell covered call
                pos = positions[ticker]
                expiry = find_expiry(date)
                if expiry is None:
                    continue
                T = (expiry - date).days / 365.0
                K = strike_from_delta(S, T, sigma, CALL_DELTA, kind="call")
                prem = bs_price(S, K, T, sigma, kind="call")
                slip = slippage(prem)
                net_prem = prem - slip
                if net_prem < 0.05:
                    continue

                credit = net_prem * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                cash += credit
                total_premium += net_prem * 100 * pos.contracts
                positions[ticker] = TickerPosition(
                    ticker=ticker, side="short_call", strike=K, expiry=expiry,
                    open_date=date, open_price=net_prem, contracts=pos.contracts,
                    share_basis=pos.share_basis,
                )

        # ── Compute portfolio equity ──
        equity = cash
        for ticker, pos in positions.items():
            data = price_lookup.get((ticker, date))
            if data is None:
                continue
            S, sigma, _ = data
            T = max((pos.expiry - date).days, 0) / 365.0

            if pos.side == "short_put":
                opt = bs_price(S, pos.strike, T, sigma, kind="put")
                equity += (pos.open_price - opt) * 100 * pos.contracts
            elif pos.side == "long_shares":
                equity += (S - pos.share_basis) * 100 * pos.contracts
            elif pos.side == "short_call":
                opt = bs_price(S, pos.strike, T, sigma, kind="call")
                equity += (S - pos.share_basis) * 100 * pos.contracts
                equity += (pos.open_price - opt) * 100 * pos.contracts

        equity_series.append({"date": date, "equity": equity, "cash": cash,
                             "n_positions": len(positions), "is_bear": is_bear})

        daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_pnl.append({"date": date, "equity": equity, "daily_ret": daily_ret, "is_bear": is_bear})
        prev_equity = equity

    elapsed = time.time() - t0

    # ── Compute portfolio metrics ──
    eq_df = pd.DataFrame(equity_series)
    pnl_df = pd.DataFrame(daily_pnl)

    rets = pnl_df["daily_ret"].values[1:]  # skip first day
    rets = rets[np.isfinite(rets)]

    years = (all_dates[-1] - all_dates[0]).days / 365.25
    total_ret = eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0] - 1.0
    ann_ret = (1 + total_ret) ** (1.0 / max(years, 0.01)) - 1.0

    mu = np.mean(rets)
    sd = np.std(rets, ddof=1)
    sharpe = (mu / sd) * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan

    down_rets = rets[rets < 0]
    downside = np.std(down_rets, ddof=1) if len(down_rets) > 1 else np.nan
    sortino = (mu / downside) * np.sqrt(TRADING_DAYS) if downside and downside > 0 else np.nan

    peak = np.maximum.accumulate(eq_df["equity"].values)
    dd = eq_df["equity"].values / peak - 1.0
    max_dd = float(np.min(dd))
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    wr = float(np.mean(rets > 0))

    # Regime stratification
    bull_days = pnl_df[~pnl_df["is_bear"]]["daily_ret"].values[1:]
    bear_days = pnl_df[pnl_df["is_bear"]]["daily_ret"].values

    bull_sharpe = (np.mean(bull_days) / np.std(bull_days, ddof=1) * np.sqrt(252)) if len(bull_days) > 20 else np.nan
    bear_sharpe = (np.mean(bear_days) / np.std(bear_days, ddof=1) * np.sqrt(252)) if len(bear_days) > 20 else np.nan

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe)) if (np.isfinite(bull_sharpe) and np.isfinite(bear_sharpe)) else np.nan

    # Print results
    print(f"\n{'='*70}")
    print(f"PORTFOLIO BACKTEST RESULTS — {len(BASKET)} names, weekly wheel + liq_csp_only")
    print(f"{'='*70}")
    print(f"Period:         {all_dates[0].date()} to {all_dates[-1].date()} ({years:.1f} years)")
    print(f"Starting:       ${STARTING_CAPITAL:,.0f}")
    print(f"Ending:         ${eq_df['equity'].iloc[-1]:,.0f}")
    print(f"Total Return:   {total_ret*100:.1f}%")
    print(f"Annual Return:  {ann_ret*100:.1f}%")
    print(f"Sharpe:         {sharpe:.2f}")
    print(f"Sortino:        {sortino:.2f}")
    print(f"Max Drawdown:   {max_dd*100:.1f}%")
    print(f"Calmar:         {calmar:.2f}")
    print(f"Win Rate:       {wr:.1%}")
    print(f"Total Trades:   {n_trades}")
    print(f"Win Trades:     {wins}")
    print(f"Total Premium:  ${total_premium:,.0f}")
    print(f"")
    print(f"REGIME ANALYSIS:")
    print(f"  Bull days:    {len(bull_days)}, Sharpe={bull_sharpe:.2f}")
    print(f"  Bear days:    {len(bear_days)}, Sharpe={bear_sharpe:.2f}")
    print(f"  Regime gap:   {regime_gap:.3f} ({'PASS' if regime_gap < 0.50 else 'FAIL'} <0.50)")
    print(f"")
    print(f"Runtime: {elapsed:.1f}s")
    print(f"{'='*70}")

    # Monthly returns
    pnl_df["month"] = pnl_df["date"].dt.to_period("M")
    monthly = pnl_df.groupby("month").agg(
        ret=("daily_ret", lambda x: (1 + x).prod() - 1),
        n_days=("daily_ret", "count"),
    )
    print(f"\nMONTHLY RETURNS:")
    for m, r in monthly.iterrows():
        bar = "█" * max(0, int(r["ret"] * 200))
        neg = "▓" * max(0, int(-r["ret"] * 200))
        print(f"  {m}  {r['ret']*100:+6.1f}%  {bar}{neg}")

    # Annual returns
    pnl_df["year"] = pnl_df["date"].dt.year
    annual = pnl_df.groupby("year").agg(
        ret=("daily_ret", lambda x: (1 + x).prod() - 1),
    )
    print(f"\nANNUAL RETURNS:")
    for y, r in annual.iterrows():
        print(f"  {y}  {r['ret']*100:+6.1f}%")

    # Save results
    summary = {
        "config": {
            "n_tickers": len(BASKET),
            "tickers": BASKET,
            "starting_capital": STARTING_CAPITAL,
            "dte_target": DTE_TARGET,
            "put_delta": PUT_DELTA,
            "regime_gate": "liq_csp_only",
            "max_per_name_pct": MAX_PER_NAME_PCT,
        },
        "results": {
            "period": f"{all_dates[0].date()} to {all_dates[-1].date()}",
            "years": years,
            "total_return_pct": total_ret * 100,
            "ann_return_pct": ann_ret * 100,
            "sharpe": float(sharpe),
            "sortino": float(sortino) if np.isfinite(sortino) else None,
            "max_dd_pct": max_dd * 100,
            "calmar": float(calmar) if np.isfinite(calmar) else None,
            "win_rate": wr,
            "n_trades": n_trades,
            "wins": wins,
            "total_premium": total_premium,
            "ending_equity": float(eq_df["equity"].iloc[-1]),
        },
        "regime": {
            "bull_sharpe": float(bull_sharpe) if np.isfinite(bull_sharpe) else None,
            "bear_sharpe": float(bear_sharpe) if np.isfinite(bear_sharpe) else None,
            "regime_gap": float(regime_gap) if np.isfinite(regime_gap) else None,
            "n_bull_days": len(bull_days),
            "n_bear_days": len(bear_days),
        },
    }

    with open(OUT_DIR / "portfolio_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    eq_df.to_csv(OUT_DIR / "equity_curve.csv", index=False)
    pd.DataFrame(trade_log).to_csv(OUT_DIR / "trades.csv", index=False)
    pnl_df.to_csv(OUT_DIR / "daily_pnl.csv", index=False)

    print(f"\n[saved] Results → {OUT_DIR}")
    return summary


if __name__ == "__main__":
    run_portfolio_backtest()
