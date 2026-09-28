"""
covered_call_backtest.py -- Systematic Buy-Write / Covered Call Strategy Backtester

Strategy:
  - Buy 100 shares + immediately sell 1 OTM call (delta ~0.30, DTE 30-45)
  - Exit: At expiry or profit target (50-70% of premium captured)
  - If called away: close position, re-enter on next signal
  - Position sizing: Equal weight, max N positions
  - Walk-forward OOS only: 1y lookback for vol estimation, then trade

Uses the same BS pricing infrastructure as the wheel engine.

Variants tested:
  A) Vanilla covered call on full universe (buy-write on anything that passes gates)
  B) Momentum-filtered covered call (buy stocks with positive 6mo momentum, sell calls)
  C) High-IV filtered (sell calls on highest IV-rank names for maximum premium)
"""
from __future__ import annotations
import sys, math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "wheel_strategy_v1"
CACHE = ROOT / "data" / "cache"
OUTPUT = Path(__file__).resolve().parent

# -------- Black-Scholes (copied from wheel_engine for standalone use) --------
SQRT_2PI = math.sqrt(2 * math.pi)
def _phi(x): return math.exp(-0.5 * x * x) / SQRT_2PI
def _Phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price(S, K, T, sigma, r=0.04, q=0.0, kind="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-q * T) * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)
    return K * math.exp(-r * T) * _Phi(-d2) - S * math.exp(-q * T) * _Phi(-d1)

def bs_delta(S, K, T, sigma, r=0.04, q=0.0, kind="call"):
    if T <= 0 or sigma <= 0:
        return (1.0 if S > K else 0.0) if kind == "call" else (-1.0 if S < K else 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if kind == "call":
        return math.exp(-q * T) * _Phi(d1)
    return math.exp(-q * T) * (_Phi(d1) - 1.0)

def _ndtri(p):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def strike_from_delta(S, T, sigma, target_delta, r=0.04, q=0.0, kind="call"):
    if T <= 0 or sigma <= 0:
        return S
    target = abs(target_delta)
    p = target if kind == "call" else (1 - target)
    p = min(max(p, 1e-6), 1 - 1e-6)
    d1 = _ndtri(p)
    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r - q + 0.5 * sigma**2) * T))
    return K


# -------- Costs (Schwab/Robinhood retail) --------
OPTION_COST_PER_CONTRACT = 0.03  # reg fees only
STOCK_COMMISSION = 0.00  # $0 for Schwab/Robinhood
SLIPPAGE_FRAC = 0.025  # 2.5% of premium per leg (half-spread)
SLIPPAGE_MIN = 0.03    # $0.03/share minimum

def option_slippage(premium: float) -> float:
    if premium <= 0: return 0.0
    return max(SLIPPAGE_MIN, SLIPPAGE_FRAC * premium)

def taf_cost(notional: float) -> float:
    return max(0.0, notional * 0.0000229)


# -------- Position & Trade tracking --------
@dataclass
class CCPosition:
    ticker: str
    sector: str
    shares: int  # always 100
    share_cost: float  # per share
    call_strike: float
    call_premium: float  # per share, received
    call_expiry: pd.Timestamp
    open_date: pd.Timestamp
    call_open_sigma: float
    target_delta: float
    contracts: int = 1


@dataclass
class CCTrade:
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    ticker: str
    entry_price: float
    exit_price: float
    call_strike: float
    call_premium: float
    called_away: bool
    profit_taken: bool
    expired_otm: bool
    pnl_stock: float
    pnl_option: float
    pnl_total: float
    fees: float
    dte_open: int
    exit_reason: str


# -------- Covered Call Engine --------
@dataclass
class CCConfig:
    call_delta_target: float = 0.30
    dte_min: int = 30
    dte_max: int = 45
    profit_take_pct: float = 0.60  # close call when 60% of premium captured
    max_positions: int = 8
    min_share_price: float = 20.0  # don't buy penny stocks
    max_share_price: float = 500.0  # position size cap
    reentry_cooldown_days: int = 3  # days before re-entering same ticker
    stop_loss_pct: float = 0.15  # sell if stock drops 15% from entry
    # Filters
    min_iv_rank: float = 0.0  # minimum IV rank to sell calls
    min_momentum_6m: float = -999.0  # minimum 6-month return (for momentum variant)
    min_fund_score: float = 40.0
    vix_max: float = 40.0
    r: float = 0.04
    label: str = "vanilla"


def _select_dte(cfg: CCConfig) -> int:
    return int(round((cfg.dte_min + cfg.dte_max) / 2))


def run_covered_call(cfg: CCConfig,
                     prices: pd.DataFrame,
                     iv: pd.DataFrame,
                     macro: pd.DataFrame,
                     fundamentals: pd.DataFrame,
                     universe: pd.DataFrame,
                     starting_cash: float = 100_000.0,
                     start: str = None,
                     end: str = None) -> dict:
    """
    Walk-forward covered call backtest.
    Returns dict with equity_curve, trades, stats.
    """
    prices = prices.copy()
    iv = iv.copy()
    macro = macro.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    iv["date"] = pd.to_datetime(iv["date"])
    macro["date"] = pd.to_datetime(macro["date"])

    if start:
        s = pd.Timestamp(start)
        prices = prices[prices["date"] >= s]
        iv = iv[iv["date"] >= s]
        macro = macro[macro["date"] >= s]
    if end:
        e = pd.Timestamp(end)
        prices = prices[prices["date"] <= e]
        iv = iv[iv["date"] <= e]
        macro = macro[macro["date"] <= e]

    # Build lookups
    px_by_date = {d: g.set_index("ticker")["close"].to_dict()
                  for d, g in prices.groupby("date")}
    sigma_by_date = {d: g.set_index("ticker")["sigma"].to_dict()
                     for d, g in iv.groupby("date")}
    ivrank_by_date = {d: g.set_index("ticker")["iv_rank"].to_dict()
                      for d, g in iv.groupby("date")}
    macro_by_date = macro.set_index("date").to_dict("index")

    # 6-month momentum (rolling returns)
    mom_df = prices.pivot(index="date", columns="ticker", values="close")
    mom_6m = mom_df.pct_change(126)  # ~6 months

    # Per-ticker statics
    sector_of = dict(zip(universe["ticker"], universe.get("sector", ["Unknown"] * len(universe))))
    fund_score_of = dict(zip(fundamentals["ticker"], fundamentals.get("fund_score", [50.0] * len(fundamentals))))
    div_yield_of = dict(zip(fundamentals["ticker"], fundamentals.get("dividend_yield", [0.0] * len(fundamentals))))

    all_dates = sorted(prices["date"].unique())
    cash = starting_cash
    positions: Dict[str, CCPosition] = {}
    trades: List[CCTrade] = []
    equity_curve = []
    cooldowns: Dict[str, pd.Timestamp] = {}

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        date_ivrank = ivrank_by_date.get(dt, {})
        m = macro_by_date.get(dt, {})
        vix = m.get("vix", float("nan"))

        # ---- 1) Manage existing positions ----
        to_close = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue

            T_days = (pos.call_expiry - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos.call_open_sigma) or pos.call_open_sigma or 0.20
            q = div_yield_of.get(tk, 0.0) or 0.0

            # a) Stop loss on stock (skip if stop_loss_pct == 0)
            if cfg.stop_loss_pct > 0 and S < pos.share_cost * (1 - cfg.stop_loss_pct):
                # Buy back call + sell shares
                call_cur = bs_price(S, pos.call_strike, T, sigma_atm, r=cfg.r, q=q, kind="call")
                slip = option_slippage(call_cur) * 100
                call_buyback_cost = call_cur * 100 + OPTION_COST_PER_CONTRACT + slip
                share_proceeds = S * 100 - taf_cost(S * 100)
                cash += share_proceeds - call_buyback_cost
                pnl_stock = (S - pos.share_cost) * 100 - taf_cost(S * 100)
                pnl_option = pos.call_premium * 100 - call_buyback_cost
                trades.append(CCTrade(
                    open_date=pos.open_date, close_date=dt, ticker=tk,
                    entry_price=pos.share_cost, exit_price=S,
                    call_strike=pos.call_strike, call_premium=pos.call_premium,
                    called_away=False, profit_taken=False, expired_otm=False,
                    pnl_stock=pnl_stock, pnl_option=pnl_option,
                    pnl_total=pnl_stock + pnl_option,
                    fees=OPTION_COST_PER_CONTRACT * 2 + taf_cost(S * 100) + slip,
                    dte_open=(pos.call_expiry - pos.open_date).days,
                    exit_reason="stop_loss",
                ))
                to_close.append(tk)
                cooldowns[tk] = dt + pd.Timedelta(days=cfg.reentry_cooldown_days)
                continue

            # b) At expiry
            if T_days <= 0:
                if S > pos.call_strike:
                    # Called away - sell at strike
                    proceeds = pos.call_strike * 100
                    cash += proceeds - taf_cost(proceeds)
                    pnl_stock = (pos.call_strike - pos.share_cost) * 100 - taf_cost(proceeds)
                    pnl_option = pos.call_premium * 100 - OPTION_COST_PER_CONTRACT
                    trades.append(CCTrade(
                        open_date=pos.open_date, close_date=dt, ticker=tk,
                        entry_price=pos.share_cost, exit_price=pos.call_strike,
                        call_strike=pos.call_strike, call_premium=pos.call_premium,
                        called_away=True, profit_taken=False, expired_otm=False,
                        pnl_stock=pnl_stock, pnl_option=pnl_option,
                        pnl_total=pnl_stock + pnl_option,
                        fees=OPTION_COST_PER_CONTRACT * 2 + taf_cost(proceeds),
                        dte_open=(pos.call_expiry - pos.open_date).days,
                        exit_reason="called_away",
                    ))
                    to_close.append(tk)
                    cooldowns[tk] = dt + pd.Timedelta(days=cfg.reentry_cooldown_days)
                else:
                    # Call expired OTM - keep shares, sell new call
                    pnl_option = pos.call_premium * 100 - OPTION_COST_PER_CONTRACT
                    trades.append(CCTrade(
                        open_date=pos.open_date, close_date=dt, ticker=tk,
                        entry_price=pos.share_cost, exit_price=S,
                        call_strike=pos.call_strike, call_premium=pos.call_premium,
                        called_away=False, profit_taken=False, expired_otm=True,
                        pnl_stock=0.0, pnl_option=pnl_option,
                        pnl_total=pnl_option,
                        fees=OPTION_COST_PER_CONTRACT,
                        dte_open=(pos.call_expiry - pos.open_date).days,
                        exit_reason="expired_otm",
                    ))
                    # Re-sell a new call on same shares
                    target_dte = _select_dte(cfg)
                    T_new = target_dte / 365.0
                    K_new = strike_from_delta(S, T_new, sigma_atm, cfg.call_delta_target, r=cfg.r, q=q, kind="call")
                    prem_new = bs_price(S, K_new, T_new, sigma_atm, r=cfg.r, q=q, kind="call")
                    if prem_new > 0:
                        slip = option_slippage(prem_new) * 100
                        cash += prem_new * 100 - OPTION_COST_PER_CONTRACT - slip
                        pos.call_strike = K_new
                        pos.call_premium = prem_new
                        pos.call_expiry = dt + pd.Timedelta(days=target_dte)
                        pos.open_date = dt
                        pos.call_open_sigma = sigma_atm
                    else:
                        # Can't sell a call, just hold shares (will try next day)
                        pos.call_expiry = dt + pd.Timedelta(days=1)
                continue

            # c) Profit take on call (buy it back cheap)
            call_cur = bs_price(S, pos.call_strike, T, sigma_atm, r=cfg.r, q=q, kind="call")
            captured = (pos.call_premium - call_cur) / max(pos.call_premium, 1e-6)
            if captured >= cfg.profit_take_pct:
                slip = option_slippage(call_cur) * 100
                buyback = call_cur * 100 + OPTION_COST_PER_CONTRACT + slip
                cash -= buyback
                pnl_option = pos.call_premium * 100 - buyback
                trades.append(CCTrade(
                    open_date=pos.open_date, close_date=dt, ticker=tk,
                    entry_price=pos.share_cost, exit_price=S,
                    call_strike=pos.call_strike, call_premium=pos.call_premium,
                    called_away=False, profit_taken=True, expired_otm=False,
                    pnl_stock=0.0, pnl_option=pnl_option,
                    pnl_total=pnl_option,
                    fees=OPTION_COST_PER_CONTRACT * 2 + slip,
                    dte_open=(pos.call_expiry - pos.open_date).days,
                    exit_reason="profit_take",
                ))
                # Re-sell new call immediately
                target_dte = _select_dte(cfg)
                T_new = target_dte / 365.0
                K_new = strike_from_delta(S, T_new, sigma_atm, cfg.call_delta_target, r=cfg.r, q=q, kind="call")
                prem_new = bs_price(S, K_new, T_new, sigma_atm, r=cfg.r, q=q, kind="call")
                if prem_new > 0:
                    slip2 = option_slippage(prem_new) * 100
                    cash += prem_new * 100 - OPTION_COST_PER_CONTRACT - slip2
                    pos.call_strike = K_new
                    pos.call_premium = prem_new
                    pos.call_expiry = dt + pd.Timedelta(days=target_dte)
                    pos.open_date = dt
                    pos.call_open_sigma = sigma_atm

        for tk in to_close:
            del positions[tk]

        # ---- 2) MTM equity ----
        equity = cash
        for tk, pos in positions.items():
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos.call_expiry - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos.call_open_sigma) or pos.call_open_sigma or 0.20
            q = div_yield_of.get(tk, 0.0) or 0.0
            call_val = bs_price(S, pos.call_strike, T, sigma_atm, r=cfg.r, q=q, kind="call")
            equity += S * 100  # shares
            equity -= call_val * 100  # short call liability
        equity_curve.append((dt, equity))

        # ---- 3) Open new positions ----
        if not np.isnan(vix) and vix > cfg.vix_max:
            continue
        if len(positions) >= cfg.max_positions:
            continue

        # Available capital per new position
        n_slots = cfg.max_positions - len(positions)
        capital_per = equity / cfg.max_positions  # equal weight

        # Score candidates
        candidates = []
        univ_tickers = set(universe["ticker"])
        for tk in univ_tickers:
            if tk in positions:
                continue
            if tk in cooldowns and dt < cooldowns[tk]:
                continue
            S = date_px.get(tk)
            sigma = date_sigma.get(tk)
            if S is None or sigma is None or np.isnan(S) or np.isnan(sigma) or sigma <= 0:
                continue
            if S < cfg.min_share_price or S > cfg.max_share_price:
                continue
            # Can we afford 100 shares?
            if S * 100 > capital_per:
                continue
            # Fund score gate
            fs = fund_score_of.get(tk, 0)
            if fs < cfg.min_fund_score:
                continue
            # IV rank gate
            ivr = date_ivrank.get(tk, 0)
            if ivr is None or np.isnan(ivr):
                ivr = 0
            if ivr < cfg.min_iv_rank:
                continue
            # Momentum gate
            try:
                mom = mom_6m.at[dt, tk]
            except:
                mom = 0.0
            if np.isnan(mom):
                mom = 0.0
            if mom < cfg.min_momentum_6m:
                continue

            # Score: IV rank (premium richness) + momentum (stock quality)
            score = ivr * 0.5 + mom * 100 * 0.5  # balanced
            candidates.append((tk, S, sigma, score, ivr, mom))

        # Sort by score descending
        candidates.sort(key=lambda x: -x[3])

        for tk, S, sigma, score, ivr, mom in candidates[:n_slots]:
            q = div_yield_of.get(tk, 0.0) or 0.0
            target_dte = _select_dte(cfg)
            T = target_dte / 365.0
            K = strike_from_delta(S, T, sigma, cfg.call_delta_target, r=cfg.r, q=q, kind="call")
            premium = bs_price(S, K, T, sigma, r=cfg.r, q=q, kind="call")
            if premium <= 0:
                continue

            # Buy 100 shares + sell 1 call
            stock_cost = S * 100
            slip = option_slippage(premium) * 100
            net_credit = premium * 100 - OPTION_COST_PER_CONTRACT - slip
            cash -= stock_cost
            cash += net_credit

            positions[tk] = CCPosition(
                ticker=tk,
                sector=sector_of.get(tk, "Unknown"),
                shares=100,
                share_cost=S,
                call_strike=K,
                call_premium=premium,
                call_expiry=dt + pd.Timedelta(days=target_dte),
                open_date=dt,
                call_open_sigma=sigma,
                target_delta=cfg.call_delta_target,
            )

            if len(positions) >= cfg.max_positions:
                break

    # ---- Force-close any remaining positions at final date ----
    final_dt = all_dates[-1]
    final_px = px_by_date.get(final_dt, {})
    for tk, pos in list(positions.items()):
        S = final_px.get(tk)
        if S is None or np.isnan(S):
            continue
        T_days = (pos.call_expiry - final_dt).days
        T = max(T_days, 0) / 365.0
        sigma = date_sigma.get(tk, pos.call_open_sigma) or 0.20
        q = div_yield_of.get(tk, 0.0) or 0.0
        call_cur = bs_price(S, pos.call_strike, T, sigma, r=cfg.r, q=q, kind="call")
        slip = option_slippage(call_cur) * 100
        buyback = call_cur * 100 + OPTION_COST_PER_CONTRACT + slip
        proceeds = S * 100 - taf_cost(S * 100)
        cash += proceeds - buyback
        pnl_stock = (S - pos.share_cost) * 100 - taf_cost(S * 100)
        pnl_option = pos.call_premium * 100 - buyback
        trades.append(CCTrade(
            open_date=pos.open_date, close_date=final_dt, ticker=tk,
            entry_price=pos.share_cost, exit_price=S,
            call_strike=pos.call_strike, call_premium=pos.call_premium,
            called_away=False, profit_taken=False, expired_otm=False,
            pnl_stock=pnl_stock, pnl_option=pnl_option,
            pnl_total=pnl_stock + pnl_option,
            fees=OPTION_COST_PER_CONTRACT * 2 + taf_cost(S * 100) + slip,
            dte_open=(pos.call_expiry - pos.open_date).days,
            exit_reason="final_close",
        ))

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "starting_cash": starting_cash,
        "config": cfg,
    }


def compute_stats(result: dict, spy_prices: pd.DataFrame) -> dict:
    """Compute strategy stats + regime analysis + SPY comparison."""
    eq = pd.DataFrame(result["equity_curve"], columns=["date", "equity"])
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date").sort_index()
    eq["ret"] = eq["equity"].pct_change()

    trades_df = pd.DataFrame([t.__dict__ for t in result["trades"]]) if result["trades"] else pd.DataFrame()

    # Basic stats
    n_days = len(eq)
    n_years = n_days / 252
    total_ret = eq["equity"].iloc[-1] / result["starting_cash"] - 1
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    daily_ret = eq["ret"].dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    downside = daily_ret[daily_ret < 0].std()
    sortino = daily_ret.mean() / downside * np.sqrt(252) if downside > 0 else 0

    cummax = eq["equity"].cummax()
    drawdown = (eq["equity"] - cummax) / cummax
    max_dd = drawdown.min()

    # Win rate and profit factor
    if len(trades_df) > 0:
        wins = trades_df[trades_df["pnl_total"] > 0]
        losses = trades_df[trades_df["pnl_total"] <= 0]
        wr = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
        gross_profit = wins["pnl_total"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl_total"].sum()) if len(losses) > 0 else 1
        pf = gross_profit / max(gross_loss, 1) if gross_loss > 0 else float("inf")
        n_trades = len(trades_df)
        avg_pnl = trades_df["pnl_total"].mean()
        called_away_rate = trades_df["called_away"].mean() if len(trades_df) > 0 else 0
    else:
        wr, pf, n_trades, avg_pnl, called_away_rate = 0, 0, 0, 0, 0

    # Regime analysis: SPY daily classification
    spy = spy_prices[spy_prices["ticker"] == "SPY"][["date", "close"]].copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.set_index("date").sort_index()
    spy["spy_ret"] = spy["close"].pct_change()
    spy["regime"] = "flat"
    spy.loc[spy["spy_ret"] > 0.005, "regime"] = "green"
    spy.loc[spy["spy_ret"] < -0.005, "regime"] = "red"

    # Merge daily returns with regime
    merged = eq[["ret"]].join(spy[["regime"]], how="inner")
    regime_stats = {}
    for reg in ["green", "red", "flat"]:
        r = merged[merged["regime"] == reg]["ret"]
        if len(r) > 10:
            regime_stats[reg] = {
                "n_days": len(r),
                "mean_ret": float(r.mean()),
                "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            }

    regime_gap = 0
    if "green" in regime_stats and "red" in regime_stats:
        sg = regime_stats["green"]["sharpe"]
        sr = regime_stats["red"]["sharpe"]
        regime_gap = abs(sg - sr) / max(abs(sg), abs(sr), 0.01)

    # SPY buy-and-hold comparison
    spy_eq = spy["close"].reindex(eq.index, method="ffill")
    spy_ret_total = spy_eq.iloc[-1] / spy_eq.iloc[0] - 1
    spy_cagr = (1 + spy_ret_total) ** (1 / max(n_years, 0.01)) - 1
    spy_daily = spy_eq.pct_change().dropna()
    spy_sharpe = spy_daily.mean() / spy_daily.std() * np.sqrt(252) if spy_daily.std() > 0 else 0
    spy_dd = ((spy_eq - spy_eq.cummax()) / spy_eq.cummax()).min()

    return {
        "label": result["config"].label,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "wr": wr,
        "pf": pf,
        "n_trades": n_trades,
        "avg_pnl": avg_pnl,
        "called_away_rate": called_away_rate,
        "total_ret": total_ret,
        "final_equity": eq["equity"].iloc[-1],
        "regime_stats": regime_stats,
        "regime_gap": regime_gap,
        "spy_cagr": spy_cagr,
        "spy_sharpe": spy_sharpe,
        "spy_max_dd": spy_dd,
        "n_years": n_years,
        "equity_df": eq,
        "trades_df": trades_df,
    }


def main():
    print("Loading data...")
    prices = pd.read_parquet(CACHE / "prices.parquet")
    iv = pd.read_parquet(CACHE / "iv_cache.parquet")
    macro = pd.read_parquet(CACHE / "macro.parquet")
    fundamentals = pd.read_parquet(CACHE / "fundamentals.parquet")
    universe = pd.read_parquet(CACHE / "ga_name_table.parquet")

    # OOS period: 2020-01-01 to 2026-03-31 (6+ years, using 2015-2019 as lookback)
    oos_start = "2020-01-01"
    oos_end = "2026-03-31"
    print(f"OOS period: {oos_start} to {oos_end}")

    # Define strategy variants
    configs = [
        # A) Vanilla: buy anything, sell 0.30 delta call
        CCConfig(
            call_delta_target=0.30, dte_min=30, dte_max=45,
            profit_take_pct=0.60, max_positions=8,
            stop_loss_pct=0.15, min_iv_rank=0.0, min_momentum_6m=-999,
            min_fund_score=40, vix_max=40, label="A_Vanilla_CC",
        ),
        # B) Momentum-filtered: only buy stocks with positive 6m momentum
        CCConfig(
            call_delta_target=0.30, dte_min=30, dte_max=45,
            profit_take_pct=0.60, max_positions=8,
            stop_loss_pct=0.15, min_iv_rank=0.0, min_momentum_6m=0.05,
            min_fund_score=40, vix_max=40, label="B_Momentum_CC",
        ),
        # C) High-IV: only sell calls when IV rank > 40 (richer premiums)
        CCConfig(
            call_delta_target=0.30, dte_min=30, dte_max=45,
            profit_take_pct=0.60, max_positions=8,
            stop_loss_pct=0.15, min_iv_rank=40.0, min_momentum_6m=-999,
            min_fund_score=40, vix_max=40, label="C_HighIV_CC",
        ),
        # D) Combined: momentum + high IV (the hypothesis)
        CCConfig(
            call_delta_target=0.30, dte_min=30, dte_max=45,
            profit_take_pct=0.60, max_positions=8,
            stop_loss_pct=0.15, min_iv_rank=30.0, min_momentum_6m=0.05,
            min_fund_score=45, vix_max=40, label="D_MomentumIV_CC",
        ),
        # E) Conservative: lower delta, tighter stop
        CCConfig(
            call_delta_target=0.20, dte_min=30, dte_max=45,
            profit_take_pct=0.50, max_positions=10,
            stop_loss_pct=0.10, min_iv_rank=0.0, min_momentum_6m=-999,
            min_fund_score=50, vix_max=35, label="E_Conservative_CC",
        ),
        # F) Aggressive: higher delta, more premium
        CCConfig(
            call_delta_target=0.40, dte_min=30, dte_max=45,
            profit_take_pct=0.70, max_positions=6,
            stop_loss_pct=0.20, min_iv_rank=0.0, min_momentum_6m=-999,
            min_fund_score=35, vix_max=45, label="F_Aggressive_CC",
        ),
    ]

    all_stats = []
    for cfg in configs:
        print(f"\nRunning {cfg.label}...")
        result = run_covered_call(
            cfg, prices, iv, macro, fundamentals, universe,
            starting_cash=100_000, start=oos_start, end=oos_end,
        )
        stats = compute_stats(result, prices)
        all_stats.append(stats)

        # Save equity curve
        stats["equity_df"].to_parquet(OUTPUT / f"equity_{cfg.label}.parquet")
        if len(stats["trades_df"]) > 0:
            stats["trades_df"].to_parquet(OUTPUT / f"trades_{cfg.label}.parquet")

        print(f"  CAGR: {stats['cagr']:.1%}  Sharpe: {stats['sharpe']:.2f}  "
              f"Sortino: {stats['sortino']:.2f}  MaxDD: {stats['max_dd']:.1%}  "
              f"WR: {stats['wr']:.1%}  PF: {stats['pf']:.2f}  "
              f"Trades: {stats['n_trades']}  Called%: {stats['called_away_rate']:.1%}")
        if stats["regime_stats"]:
            for reg, rs in stats["regime_stats"].items():
                print(f"    {reg}: Sharpe={rs['sharpe']:.2f} ({rs['n_days']} days)")
        print(f"  Regime gap: {stats['regime_gap']:.2f}")

    # Summary comparison table
    print("\n" + "=" * 120)
    print("COVERED CALL STRATEGY COMPARISON (OOS {}-{})".format(oos_start, oos_end))
    print("=" * 120)
    print(f"{'Variant':<25} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'WR':>6} {'PF':>6} {'Trades':>7} {'Call%':>7} {'RegGap':>8}")
    print("-" * 120)
    for s in all_stats:
        print(f"{s['label']:<25} {s['cagr']:>7.1%} {s['sharpe']:>8.2f} {s['sortino']:>8.2f} "
              f"{s['max_dd']:>7.1%} {s['wr']:>5.1%} {s['pf']:>6.2f} {s['n_trades']:>7} "
              f"{s['called_away_rate']:>6.1%} {s['regime_gap']:>8.2f}")
    print("-" * 120)
    print(f"{'SPY Buy&Hold':<25} {all_stats[0]['spy_cagr']:>7.1%} {all_stats[0]['spy_sharpe']:>8.2f} "
          f"{'':>8} {all_stats[0]['spy_max_dd']:>7.1%}")

    # CSP comparison reference (from V6 tier ladder)
    print("\n--- CSP (V6 Fixed MTM) Reference ---")
    print(f"{'Tier2_Balanced CSP':<25} {'20.1%':>8} {'1.57':>8} {'1.48':>8} {'-24.8%':>8} {'93.8%':>6} {'3.45':>6}")
    print(f"{'Tier3_Income CSP':<25} {'21.9%':>8} {'1.76':>8} {'1.79':>8} {'-22.3%':>8} {'89.8%':>6} {'2.96':>6}")

    # Theoretical analysis
    print("\n" + "=" * 80)
    print("THEORETICAL ANALYSIS: Covered Call vs CSP")
    print("=" * 80)
    print("""
Put-Call Parity Observation:
  Covered Call = Long Stock + Short Call
  Cash-Secured Put = Short Put (with cash collateral)

  By put-call parity: Short Put = Long Stock + Short Call - Long Call + Short Put
  More precisely: Covered Call = Short Put + Long Stock exposure above strike

  Key difference: Covered call has FULL downside stock exposure from day 1.
  CSP only gets stock exposure IF assigned (which is the minority of cases).

  Result: In up-trending markets, covered calls capture more upside (stock appreciation
  + premium). In down markets, covered calls suffer more (full stock drawdown - small
  premium cushion). This is why CSP generally has better Sharpe - it avoids most
  assignment/drawdown scenarios.

  The momentum filter attempts to mitigate this: only buy stocks that are trending up.
  If it works, you get the upside participation without as much downside exposure.
""")

    # Save summary
    summary_rows = []
    for s in all_stats:
        summary_rows.append({
            "label": s["label"],
            "cagr": s["cagr"],
            "sharpe": s["sharpe"],
            "sortino": s["sortino"],
            "max_dd": s["max_dd"],
            "wr": s["wr"],
            "pf": s["pf"],
            "n_trades": s["n_trades"],
            "called_away_rate": s["called_away_rate"],
            "regime_gap": s["regime_gap"],
            "total_ret": s["total_ret"],
            "final_equity": s["final_equity"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT / "cc_strategy_comparison.csv", index=False)
    summary_df.to_parquet(OUTPUT / "cc_strategy_comparison.parquet", index=False)
    print(f"\nResults saved to {OUTPUT}/")


if __name__ == "__main__":
    main()
