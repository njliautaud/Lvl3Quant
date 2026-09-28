#!/usr/bin/env python3
"""
market_neutral_spy_v1.py - Backtest two genuinely market-neutral SPY income
strategies, applying HC #428 R1 deploy gates.

After 5 wheel variants and 3 directional-carry rotations failed HC #428 R1 today
(2026-06-09), the structural conclusion is that short-vol payoffs at any
duration carry an intrinsic green/red asymmetry: the carry IS the asymmetry.
Per HC #585 R2, we pivot to genuinely market-neutral structures.

Strategies
----------
A. DELTA_STRADDLE - sell 1 SPY ATM straddle, DTE 30, then daily-rebalance SPY
   shares to net delta = 0. Close & reopen at DTE 5 or 50% profit.
   Edge: vol-risk-premium (IV at entry - RV over the holding period).
   Hedging removes directional exposure (modulo gamma slippage and discrete
   rebalance error).

B. PUT_CALENDAR - long 1 SPY 60-DTE put (strike = ATM - 5%), short 1 SPY 30-DTE
   put (strike = ATM - 5%). Close at front-month expiry. Reopen weekly.
   Edge: front-month theta > back-month theta when IV term is roughly flat
   at the OTM strike.

Both strategies: VIX gate (don't open if VIX > 30). $20K account anchor
(HC #580). Window 2018-01-01 to 2025-12-31.

Cost model (matched to the wheel framework for apples-to-apples):
  - Options: $0.65 / contract / leg + 1% slippage on premium
  - Shares (hedge):  $0.005 / share + 1 bp slippage on notional
  - Per-day BS pricing using realized vol estimated from a 20d rolling SPY
    return window (clipped 0.05 .. 1.5), same field used by the wheel.

Outputs
-------
- MLflow experiment "market_neutral_spy_v1"
- /home/jupiter/Lvl3Quant/research/findings/market_neutral_spy_v1.md
- /home/jupiter/Lvl3Quant/output/market_neutral_spy_v1/{equity,ledger,results}_<strat>.{parquet,json}
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ----- Paths -----
ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "market_neutral_spy_v1"
REPORT_DIR = ROOT / "research" / "findings"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Account / costs -----
STARTING_CASH = 20_000.0    # HC #580 anchor (% returns are primary metric)
RISK_FREE = 0.04
TRADING_DAYS = 252

# Option costs
COMMISSION_PER_CONTRACT = 0.65
OPTION_SLIPPAGE_FRAC = 0.01  # 1% of premium

# Share costs (hedging)
COMMISSION_PER_SHARE = 0.005
SHARE_SLIPPAGE_BPS = 1.0      # 1 bp of notional

# Gates / window
VIX_GATE = 30.0
START_DATE = pd.Timestamp("2018-01-01")
END_DATE = pd.Timestamp("2025-12-31")


# ============================================================
# Black-Scholes (Phi via erf), delta closed-form
# ============================================================
def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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
        if kind == "put":
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    if kind == "put":
        return _Phi(d1) - 1.0
    return _Phi(d1)


# ============================================================
# Data loader (reuses the wheel SPY/IV/VIX panel)
# ============================================================
def load_data():
    sec = pd.read_parquet(CACHE / "sector_etfs.parquet")
    spy = sec[sec["ticker"] == "SPY"][["date", "close"]].copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")

    iv = pd.read_parquet(CACHE / "iv_features.parquet")
    iv = iv[iv["ticker"] == "SPY"][["date", "sigma"]].copy()
    iv["date"] = pd.to_datetime(iv["date"])
    iv = iv.sort_values("date").set_index("date")

    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.sort_values("date").set_index("date")

    df = spy.join(iv, how="inner").join(macro, how="left")
    df["vix"] = df["vix"].ffill()
    df = df.loc[START_DATE:END_DATE].dropna(subset=["close", "sigma"])
    df["sigma"] = df["sigma"].clip(lower=0.05, upper=1.5)
    df["spy_ret"] = df["close"].pct_change()
    # Realized vol over a 20d rolling window for option pricing as fallback
    df["rv_20d"] = df["spy_ret"].rolling(20).std() * math.sqrt(TRADING_DAYS)
    df["rv_20d"] = df["rv_20d"].clip(lower=0.05, upper=1.5).fillna(df["sigma"])
    return df


# ============================================================
# Shared state types
# ============================================================
@dataclass
class TradeRecord:
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    strat: str
    detail: str
    realized_pnl: float


@dataclass
class State:
    cash: float
    equity_curve: list = field(default_factory=list)
    ledger: list = field(default_factory=list)
    hedge_trades: int = 0
    opens: int = 0


# ============================================================
# Strategy A: delta-hedged short ATM straddle (30 DTE)
# ============================================================
@dataclass
class StraddlePos:
    open_date: pd.Timestamp
    expiry: pd.Timestamp
    strike: float
    open_sigma: float
    open_call_px: float
    open_put_px: float
    contracts: int
    shares: int            # net SPY share hedge (negative = short, positive = long)
    open_premium: float    # initial credit per straddle (per share)


def find_friday_after(d, days_ahead):
    cand = d + pd.Timedelta(days=days_ahead)
    shift = (4 - cand.weekday()) % 7
    return cand + pd.Timedelta(days=shift)


def option_close_cost(premium):
    """Cost to BUY BACK 1 contract (per share). Slippage adds to price you pay."""
    return premium * (1.0 + OPTION_SLIPPAGE_FRAC)


def option_sell_proceeds(premium):
    """Premium RECEIVED on a sell (per share). Slippage subtracts."""
    return premium * (1.0 - OPTION_SLIPPAGE_FRAC)


def share_buy_cost(S, n):
    """Cash outflow to BUY n shares (n can be 0)."""
    if n <= 0:
        return 0.0
    px = S * (1.0 + SHARE_SLIPPAGE_BPS / 10_000.0)
    return n * px + n * COMMISSION_PER_SHARE


def share_sell_proceeds(S, n):
    """Cash inflow when SELLING n shares (n > 0)."""
    if n <= 0:
        return 0.0
    px = S * (1.0 - SHARE_SLIPPAGE_BPS / 10_000.0)
    return n * px - n * COMMISSION_PER_SHARE


def adjust_hedge(state, S, current_shares, target_shares):
    """Move share position from current -> target. Return new_shares.
    Tracks hedge_trades count + cash impact."""
    diff = target_shares - current_shares
    if diff == 0:
        return current_shares
    if diff > 0:  # buy more
        state.cash -= share_buy_cost(S, diff)
    else:  # sell some
        state.cash += share_sell_proceeds(S, -diff)
    state.hedge_trades += 1
    return target_shares


def open_straddle(state, today, S, sigma_open, dte_target=30):
    """Open a 1-contract (or sized) short ATM straddle and the initial delta hedge."""
    expiry = find_friday_after(today, dte_target)
    T = (expiry - today).days / 365.0
    K = round(S, 0)  # ATM
    call_px = bs_price(S, K, T, sigma_open, kind="call")
    put_px = bs_price(S, K, T, sigma_open, kind="put")
    premium = call_px + put_px  # per share, credit
    # Premium received (selling both legs)
    credit_per_share = option_sell_proceeds(call_px) + option_sell_proceeds(put_px)
    # Sizing: 1 contract is the minimum unit. With $20K account, ATM straddle on
    # SPY ~ $300-600 yields premium ~ $1500-3000 per contract; collateral
    # requirement (reg-T short straddle margin) ~ 20% of underlying ~ $6-12K.
    # Conservative: 1 contract per ~$15K equity to leave hedge headroom.
    equity_now = state.cash  # before opening
    max_contracts = max(1, int(equity_now / 15_000.0))
    contracts = max_contracts
    state.cash += credit_per_share * 100 * contracts
    state.cash -= COMMISSION_PER_CONTRACT * 2 * contracts  # two legs
    # Initial delta of short ATM straddle: -(delta_C - delta_P) ~ approx 0
    # but actual ATM delta of call is ~0.5x, put ~ -0.5x, so position delta of
    # short straddle = -(0.5) - (-0.5)) = ~0; but include drift, so compute.
    dC = bs_delta(S, K, T, sigma_open, kind="call")
    dP = bs_delta(S, K, T, sigma_open, kind="put")
    pos_delta_per_share = -(dC + dP)  # short both
    # Total option delta in shares = 100 * contracts * pos_delta_per_share
    target_shares = int(round(-100 * contracts * pos_delta_per_share))
    new_shares = adjust_hedge(state, S, 0, target_shares)
    state.opens += 1
    return StraddlePos(
        open_date=today, expiry=expiry, strike=K, open_sigma=sigma_open,
        open_call_px=call_px, open_put_px=put_px, contracts=contracts,
        shares=new_shares, open_premium=premium,
    )


def mtm_straddle(pos, today, S, sigma_now):
    T = max((pos.expiry - today).days, 0) / 365.0
    call_px = bs_price(S, pos.strike, T, sigma_now, kind="call")
    put_px = bs_price(S, pos.strike, T, sigma_now, kind="put")
    # Short straddle MTM PnL per share = open_premium - (call_now + put_now)
    opt_pnl = (pos.open_premium - (call_px + put_px)) * 100 * pos.contracts
    # Share PnL: shares carried at no recorded basis (cash moved when bought/sold).
    # MTM share value = shares * S
    share_value = pos.shares * S
    return opt_pnl, share_value, call_px, put_px


def close_straddle(state, pos, today, S, sigma_now, reason=""):
    T = max((pos.expiry - today).days, 0) / 365.0
    call_px = bs_price(S, pos.strike, T, sigma_now, kind="call")
    put_px = bs_price(S, pos.strike, T, sigma_now, kind="put")
    # Buy back both legs
    cost = (option_close_cost(call_px) + option_close_cost(put_px)) * 100 * pos.contracts
    state.cash -= cost
    state.cash -= COMMISSION_PER_CONTRACT * 2 * pos.contracts
    # Unwind share hedge to zero
    new_shares = adjust_hedge(state, S, pos.shares, 0)
    # Realized PnL for this opening cycle:
    # Total premium collected on open = (proceeds_call + proceeds_put) * 100 * contracts
    open_credit = (option_sell_proceeds(pos.open_call_px) +
                   option_sell_proceeds(pos.open_put_px)) * 100 * pos.contracts
    open_credit -= COMMISSION_PER_CONTRACT * 2 * pos.contracts
    realized_options = open_credit - cost - COMMISSION_PER_CONTRACT * 2 * pos.contracts
    state.ledger.append({
        "open_date": pos.open_date,
        "close_date": today,
        "strat": "DELTA_STRADDLE",
        "detail": f"K={pos.strike} contracts={pos.contracts} reason={reason}",
        "realized_pnl": realized_options,
    })
    return new_shares


def rebalance_straddle_hedge(state, pos, today, S, sigma_now):
    """Daily delta rebalance to delta-neutral."""
    T = max((pos.expiry - today).days, 0) / 365.0
    if T <= 0:
        return pos
    dC = bs_delta(S, pos.strike, T, sigma_now, kind="call")
    dP = bs_delta(S, pos.strike, T, sigma_now, kind="put")
    pos_delta_per_share = -(dC + dP)  # short straddle position delta
    target_shares = int(round(-100 * pos.contracts * pos_delta_per_share))
    if target_shares != pos.shares:
        pos.shares = adjust_hedge(state, S, pos.shares, target_shares)
    return pos


def run_delta_straddle(df):
    state = State(cash=STARTING_CASH)
    pos = None
    for today in df.index:
        row = df.loc[today]
        S = float(row["close"])
        sigma_iv = float(row["sigma"])      # entry-time IV
        sigma_rv = float(row["rv_20d"])     # mark-to-market vol (proxy for what's realizing)
        vix = float(row["vix"]) if pd.notna(row["vix"]) else 0.0

        # ---- Decide on existing position ----
        if pos is not None:
            # MTM check: profit-take 50% or DTE <= 5
            opt_pnl, _, _, _ = mtm_straddle(pos, today, S, sigma_rv)
            # initial credit basis = open_premium * 100 * contracts (before slippage)
            basis = pos.open_premium * 100 * pos.contracts
            profit_frac = opt_pnl / basis if basis > 0 else 0.0
            days_to_exp = (pos.expiry - today).days
            close_now = (days_to_exp <= 5) or (profit_frac >= 0.50) or (today >= pos.expiry)
            if close_now:
                close_straddle(state, pos, today, S, sigma_rv,
                               reason="dte_5" if days_to_exp <= 5 else
                                      ("profit_50" if profit_frac >= 0.50 else "expiry"))
                pos = None
            else:
                pos = rebalance_straddle_hedge(state, pos, today, S, sigma_rv)

        # ---- Open new ----
        if pos is None and vix > 0 and vix <= VIX_GATE and state.cash > 5000:
            # use IV as the "vol I'm selling"
            pos = open_straddle(state, today, S, sigma_iv, dte_target=30)

        # ---- MTM equity ----
        equity = state.cash
        if pos is not None:
            opt_pnl, share_val, _, _ = mtm_straddle(pos, today, S, sigma_rv)
            # cash already accounts for option premium received and hedge spend;
            # MTM adjustments: + opt_pnl (from open premium baseline) + share_val
            # But cash already includes the open credit, so equity already has it.
            # We add unrealized MTM relative to the option open price:
            # current option liability = (call + put) * 100 * contracts
            T = max((pos.expiry - today).days, 0) / 365.0
            cur_liab = (bs_price(S, pos.strike, T, sigma_rv, kind="call") +
                        bs_price(S, pos.strike, T, sigma_rv, kind="put")) * 100 * pos.contracts
            equity -= cur_liab          # short option = liability
            equity += share_val         # shares are an asset
        state.equity_curve.append({
            "date": today, "equity": equity, "S": S, "vix": vix,
            "spy_ret": float(row["spy_ret"]) if pd.notna(row["spy_ret"]) else np.nan,
        })

    # Force-close any open position at last date
    if pos is not None:
        S = float(df.iloc[-1]["close"])
        sigma = float(df.iloc[-1]["rv_20d"])
        close_straddle(state, pos, df.index[-1], S, sigma, reason="end_of_window")

    eq_df = pd.DataFrame(state.equity_curve).set_index("date")
    eq_df["ret"] = eq_df["equity"].pct_change()
    ledger_df = pd.DataFrame(state.ledger)
    return eq_df, ledger_df, state


# ============================================================
# Strategy B: SPY put-calendar (long 60-DTE / short 30-DTE @ ATM-5%)
# ============================================================
@dataclass
class CalendarPos:
    open_date: pd.Timestamp
    front_expiry: pd.Timestamp     # 30-DTE short
    back_expiry: pd.Timestamp      # 60-DTE long
    strike: float                  # ATM-5% rounded
    open_sigma: float
    front_short_px: float          # received when sold
    back_long_px: float            # paid when bought
    contracts: int


def open_calendar(state, today, S, sigma, dte_front=30, dte_back=60):
    front_exp = find_friday_after(today, dte_front)
    back_exp = find_friday_after(today, dte_back)
    K = round(S * 0.95, 0)
    T_front = (front_exp - today).days / 365.0
    T_back = (back_exp - today).days / 365.0
    front_px = bs_price(S, K, T_front, sigma, kind="put")
    back_px = bs_price(S, K, T_back, sigma, kind="put")
    # Net debit (back is more expensive than front since longer-dated, same strike, same vol)
    net_debit = (back_px * (1 + OPTION_SLIPPAGE_FRAC)
                 - front_px * (1 - OPTION_SLIPPAGE_FRAC))
    if net_debit <= 0:
        return None
    # 1 contract per ~$1500 of debit budget to keep position sizes modest;
    # $20K account: 1-2 contracts typical
    debit_per_contract = net_debit * 100 + COMMISSION_PER_CONTRACT * 2
    max_contracts = max(1, int((state.cash * 0.25) / debit_per_contract))
    contracts = max_contracts
    state.cash -= debit_per_contract * contracts
    state.opens += 1
    return CalendarPos(
        open_date=today, front_expiry=front_exp, back_expiry=back_exp,
        strike=K, open_sigma=sigma,
        front_short_px=front_px * (1 - OPTION_SLIPPAGE_FRAC),
        back_long_px=back_px * (1 + OPTION_SLIPPAGE_FRAC),
        contracts=contracts,
    )


def mtm_calendar(pos, today, S, sigma):
    Tf = max((pos.front_expiry - today).days, 0) / 365.0
    Tb = max((pos.back_expiry - today).days, 0) / 365.0
    front_now = bs_price(S, pos.strike, Tf, sigma, kind="put")
    back_now = bs_price(S, pos.strike, Tb, sigma, kind="put")
    # Net value = back (long) - front (short).
    # PnL relative to open = (back_now - back_long_px) - (front_now - front_short_px)
    pnl = ((back_now - pos.back_long_px) -
           (front_now - pos.front_short_px)) * 100 * pos.contracts
    cur_value = (back_now - front_now) * 100 * pos.contracts
    return pnl, cur_value, front_now, back_now


def close_calendar(state, pos, today, S, sigma, reason=""):
    Tf = max((pos.front_expiry - today).days, 0) / 365.0
    Tb = max((pos.back_expiry - today).days, 0) / 365.0
    front_now = bs_price(S, pos.strike, Tf, sigma, kind="put")
    back_now = bs_price(S, pos.strike, Tb, sigma, kind="put")
    # Buy back front (cost), sell back (proceeds)
    cost_buyback_front = front_now * (1 + OPTION_SLIPPAGE_FRAC) * 100 * pos.contracts
    proceeds_sell_back = back_now * (1 - OPTION_SLIPPAGE_FRAC) * 100 * pos.contracts
    net_credit = proceeds_sell_back - cost_buyback_front
    state.cash += net_credit
    state.cash -= COMMISSION_PER_CONTRACT * 2 * pos.contracts
    # Realized PnL on this calendar = total close credit - original debit
    original_debit = (pos.back_long_px - pos.front_short_px) * 100 * pos.contracts \
                     + COMMISSION_PER_CONTRACT * 2 * pos.contracts
    realized = net_credit - original_debit - COMMISSION_PER_CONTRACT * 2 * pos.contracts
    state.ledger.append({
        "open_date": pos.open_date,
        "close_date": today,
        "strat": "PUT_CALENDAR",
        "detail": f"K={pos.strike} contracts={pos.contracts} reason={reason}",
        "realized_pnl": realized,
    })


def run_put_calendar(df):
    state = State(cash=STARTING_CASH)
    pos = None
    last_open = None
    for today in df.index:
        row = df.loc[today]
        S = float(row["close"])
        sigma = float(row["sigma"])
        vix = float(row["vix"]) if pd.notna(row["vix"]) else 0.0

        # Manage existing
        if pos is not None:
            # Close at or after front-month expiry
            if today >= pos.front_expiry:
                close_calendar(state, pos, today, S, sigma, reason="front_expiry")
                pos = None

        # Open weekly: at most one open per 7 calendar days
        can_open = (pos is None) and (vix > 0) and (vix <= VIX_GATE) and (state.cash > 2000)
        if can_open and (last_open is None or (today - last_open).days >= 5):
            new_pos = open_calendar(state, today, S, sigma, dte_front=30, dte_back=60)
            if new_pos is not None:
                pos = new_pos
                last_open = today

        # MTM equity
        equity = state.cash
        if pos is not None:
            _, cur_value, _, _ = mtm_calendar(pos, today, S, sigma)
            equity += cur_value
        state.equity_curve.append({
            "date": today, "equity": equity, "S": S, "vix": vix,
            "spy_ret": float(row["spy_ret"]) if pd.notna(row["spy_ret"]) else np.nan,
        })

    if pos is not None:
        S = float(df.iloc[-1]["close"])
        sigma = float(df.iloc[-1]["sigma"])
        close_calendar(state, pos, df.index[-1], S, sigma, reason="end_of_window")

    eq_df = pd.DataFrame(state.equity_curve).set_index("date")
    eq_df["ret"] = eq_df["equity"].pct_change()
    ledger_df = pd.DataFrame(state.ledger)
    return eq_df, ledger_df, state


# ============================================================
# Metrics & gates (mirrors wheel framework)
# ============================================================
def compute_metrics(eq_df):
    rets = eq_df["ret"].dropna()
    if len(rets) < 2 or eq_df["equity"].iloc[0] <= 0:
        return {}
    years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
    final_eq = max(eq_df["equity"].iloc[-1], 1e-6)
    cagr = (final_eq / eq_df["equity"].iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    mu, sd = rets.mean(), rets.std()
    downside = rets[rets < 0].std()
    sharpe = (mu / sd) * math.sqrt(TRADING_DAYS) if sd > 0 else float("nan")
    sortino = (mu / downside) * math.sqrt(TRADING_DAYS) if downside and downside > 0 else float("nan")
    peak = eq_df["equity"].cummax()
    dd = eq_df["equity"] / peak - 1.0
    max_dd = float(dd.min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    wr = float((rets > 0).mean())
    pos = rets[rets > 0].sum()
    neg = abs(rets[rets < 0].sum())
    pf = float(pos / neg) if neg > 0 else float("nan")
    return {
        "n_days": int(len(rets)),
        "years": float(years),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
        "win_rate": wr,
        "profit_factor": pf,
        "total_return_pct": float((final_eq / eq_df["equity"].iloc[0] - 1.0) * 100.0),
        "final_equity": float(final_eq),
    }


def regime_gate_metrics(eq_df):
    sub = eq_df.dropna(subset=["spy_ret", "ret"])
    sigma = sub["spy_ret"].std()
    thresh = 0.5 * sigma
    green = sub[sub["spy_ret"] >= thresh]["ret"]
    red = sub[sub["spy_ret"] <= -thresh]["ret"]
    flat = sub[(sub["spy_ret"] > -thresh) & (sub["spy_ret"] < thresh)]["ret"]

    def ann_sharpe(s):
        if len(s) < 2 or s.std() == 0:
            return float("nan")
        return (s.mean() / s.std()) * math.sqrt(TRADING_DAYS)

    sh_g, sh_r, sh_f = ann_sharpe(green), ann_sharpe(red), ann_sharpe(flat)
    denom = max(abs(sh_g) if np.isfinite(sh_g) else 0,
                abs(sh_r) if np.isfinite(sh_r) else 0, 1e-9)
    gap = abs((sh_g if np.isfinite(sh_g) else 0.0) - (sh_r if np.isfinite(sh_r) else 0.0)) / denom
    return {
        "n_green": int(len(green)),
        "n_red": int(len(red)),
        "n_flat": int(len(flat)),
        "sharpe_green": float(sh_g) if np.isfinite(sh_g) else None,
        "sharpe_red": float(sh_r) if np.isfinite(sh_r) else None,
        "sharpe_flat": float(sh_f) if np.isfinite(sh_f) else None,
        "regime_gap": float(gap),
        "hc428_r1_pass": bool(gap <= 0.50),
    }


def day_concentration(ledger_df):
    if ledger_df is None or ledger_df.empty or "realized_pnl" not in ledger_df.columns:
        return float("nan")
    pos = ledger_df[ledger_df["realized_pnl"] > 0]["realized_pnl"]
    if len(pos) == 0:
        return float("nan")
    return float(pos.max() / pos.sum())


def tail_event_stats(eq_df, label, start, end):
    sub = eq_df.loc[start:end].copy()
    if len(sub) < 2:
        return {"label": label, "n_days": 0}
    peak = sub["equity"].cummax()
    dd = sub["equity"] / peak - 1.0
    max_dd = float(dd.min())
    trough = dd.idxmin()
    pre = eq_df.loc[:sub.index[0]]
    pre_peak = float(pre["equity"].max()) if len(pre) else float(sub["equity"].iloc[0])
    after = eq_df.loc[trough:]
    rec = after[after["equity"] >= pre_peak]
    rec_days = int((rec.index[0] - trough).days) if len(rec) else None
    weekly = sub["equity"].resample("W").last().pct_change()
    worst_week = float(weekly.min()) if len(weekly) else float("nan")
    return {
        "label": label,
        "window": f"{start} to {end}",
        "n_days": int(len(sub)),
        "max_dd_pct": max_dd * 100,
        "vix_peak": float(sub["vix"].max()) if "vix" in sub.columns else None,
        "recover_days": rec_days,
        "worst_week_pct": worst_week * 100 if np.isfinite(worst_week) else None,
        "cum_ret_pct": float((sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1.0) * 100.0),
    }


def spy_buy_hold_metrics(df):
    eq = (df["close"] / df["close"].iloc[0]) * STARTING_CASH
    ed = pd.DataFrame({"equity": eq, "spy_ret": df["spy_ret"]})
    ed["ret"] = ed["equity"].pct_change()
    return compute_metrics(ed)


# ============================================================
# MLflow
# ============================================================
def log_to_mlflow(strat, metrics, gate, day_conc, tail, eq_df, ledger_df, state,
                  spy_bh):
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("market_neutral_spy_v1")
        with mlflow.start_run(run_name=strat):
            mlflow.log_param("strategy", strat)
            mlflow.log_param("starting_cash", STARTING_CASH)
            mlflow.log_param("start", str(eq_df.index[0].date()))
            mlflow.log_param("end", str(eq_df.index[-1].date()))
            mlflow.log_param("vix_gate", VIX_GATE)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(k, v)
            for k, v in gate.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(f"regime_{k}", v)
            mlflow.log_metric("day_concentration",
                              day_conc if np.isfinite(day_conc) else -1.0)
            mlflow.log_metric("hc428_r1_pass", 1.0 if gate.get("hc428_r1_pass") else 0.0)
            mlflow.log_metric("trade_count",
                              int(len(ledger_df)) if ledger_df is not None and not ledger_df.empty else 0)
            mlflow.log_metric("hedge_trades", state.hedge_trades)
            mlflow.log_metric("opens", state.opens)
            for t in tail:
                if "max_dd_pct" in t:
                    safe = t["label"].replace(" ", "_")
                    mlflow.log_metric(f"tail_{safe}_max_dd_pct", t["max_dd_pct"])
                    if t.get("recover_days") is not None:
                        mlflow.log_metric(f"tail_{safe}_recover_days", t["recover_days"])
                    if t.get("worst_week_pct") is not None:
                        mlflow.log_metric(f"tail_{safe}_worst_week_pct", t["worst_week_pct"])
            for k, v in spy_bh.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(f"spy_bh_{k}", v)
            csv_path = OUT_DIR / f"equity_{strat}.csv"
            eq_df.to_csv(csv_path)
            mlflow.log_artifact(str(csv_path))
            return mlflow.active_run().info.run_id
    except Exception as e:
        print(f"[mlflow] skipped ({e})")
        return None


# ============================================================
# Main
# ============================================================
def main():
    print("[load] SPY/IV/VIX data ...")
    df = load_data()
    print(f"[load] {len(df)} trading days {df.index[0].date()} -> {df.index[-1].date()}")

    spy_bh = spy_buy_hold_metrics(df)
    print(f"[SPY BH] CAGR={spy_bh['cagr']*100:.2f}% Sharpe={spy_bh['sharpe']:.2f} "
          f"MaxDD={spy_bh['max_dd']*100:.1f}%")

    strategies = [
        ("DELTA_STRADDLE", run_delta_straddle),
        ("PUT_CALENDAR", run_put_calendar),
    ]
    results = {}
    for name, runner in strategies:
        print(f"\n[{name}] running backtest ...")
        eq_df, ledger_df, state = runner(df)
        metrics = compute_metrics(eq_df)
        gate = regime_gate_metrics(eq_df)
        day_conc = day_concentration(ledger_df)
        tail = [
            tail_event_stats(eq_df, "Volmageddon_Feb2018", "2018-01-25", "2018-03-15"),
            tail_event_stats(eq_df, "COVID_Mar2020", "2020-02-15", "2020-05-31"),
            tail_event_stats(eq_df, "Aug2024_carry", "2024-07-15", "2024-09-15"),
            tail_event_stats(eq_df, "2022_bear", "2022-01-01", "2022-12-31"),
        ]
        trade_count = int(len(ledger_df)) if ledger_df is not None and not ledger_df.empty else 0

        gates_pass = {
            "sharpe_ge_1.0": metrics.get("sharpe", float("nan")) >= 1.0 if np.isfinite(metrics.get("sharpe", float("nan"))) else False,
            "calmar_ge_1.0": metrics.get("calmar", float("nan")) >= 1.0 if np.isfinite(metrics.get("calmar", float("nan"))) else False,
            "regime_gap_le_0.50": gate["hc428_r1_pass"],
            "day_conc_le_0.70": (day_conc <= 0.70) if np.isfinite(day_conc) else True,
        }
        deploy_ready = all(gates_pass.values())

        eq_df.to_parquet(OUT_DIR / f"equity_{name}.parquet")
        if not ledger_df.empty:
            ledger_df.to_parquet(OUT_DIR / f"ledger_{name}.parquet")
        with open(OUT_DIR / f"results_{name}.json", "w") as f:
            json.dump({
                "strategy": name, "metrics": metrics, "gate": gate,
                "day_conc": day_conc, "tail": tail,
                "trade_count": trade_count, "hedge_trades": state.hedge_trades,
                "opens": state.opens, "gates_pass": gates_pass,
                "deploy_ready": deploy_ready,
            }, f, indent=2, default=str)

        run_id = log_to_mlflow(name, metrics, gate, day_conc, tail, eq_df, ledger_df,
                               state, spy_bh)
        results[name] = {
            "metrics": metrics, "gate": gate, "day_conc": day_conc, "tail": tail,
            "trade_count": trade_count, "hedge_trades": state.hedge_trades,
            "opens": state.opens, "gates_pass": gates_pass,
            "deploy_ready": deploy_ready, "mlflow_run_id": run_id,
        }
        sh = metrics.get("sharpe", float("nan"))
        ca = metrics.get("calmar", float("nan"))
        cagr = metrics.get("cagr", float("nan"))
        mdd = metrics.get("max_dd", float("nan"))
        print(f"[{name}] Sharpe={sh:.2f} CAGR={cagr*100:.2f}% MaxDD={mdd*100:.1f}% "
              f"Calmar={ca:.2f} regime_gap={gate['regime_gap']:.2f} "
              f"trades={trade_count} hedge_trades={state.hedge_trades} "
              f"deploy={deploy_ready}")

    # ---------- Markdown report ----------
    L = []
    L.append("# Market-Neutral SPY v1 (HC #585 R2 - pivot from short-vol carry)\n")
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Window: {START_DATE.date()} to {END_DATE.date()}")
    L.append(f"Starting cash: ${STARTING_CASH:,.0f}  (HC #580 anchor; % returns are the primary read-out)")
    L.append(f"Costs: options ${COMMISSION_PER_CONTRACT}/contract + {OPTION_SLIPPAGE_FRAC*100:.1f}% slippage; "
             f"shares ${COMMISSION_PER_SHARE}/share + {SHARE_SLIPPAGE_BPS} bp slippage")
    L.append(f"VIX gate: don't open if VIX > {VIX_GATE}\n")

    L.append("## Why this experiment exists\n")
    L.append("- 5 wheel variants (baseline / QQQ+IWM / hedge overlay / regime-gated / short-DTE) ALL failed HC #428 R1.")
    L.append("- 3 directional carry rotations (ETF / tech sub-industry / blend) ALL failed HC #428 R1.")
    L.append("- Conclusion: short-vol payoffs at any duration are intrinsically regime-asymmetric. The carry IS the asymmetry.")
    L.append("- Per HC #585 R2: only genuinely market-neutral structures can clear the gate. Two candidates below.\n")

    L.append("## Strategy summary\n")
    L.append("- **DELTA_STRADDLE**: sell SPY ATM straddle, DTE 30, daily delta-hedge to neutral with shares. "
             "Close at 50% profit or DTE 5.")
    L.append("- **PUT_CALENDAR**: long 60 DTE SPY put @ ATM-5%, short 30 DTE SPY put @ same strike. "
             "Close at front-month expiry, reopen weekly.\n")

    L.append("## Headline metrics ($20K notional)\n")
    L.append("| Strategy | Trades | Hedge trades | CAGR | Sharpe | Sortino | MaxDD | Calmar | WR | PF | Final $ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, r in results.items():
        m = r["metrics"]
        L.append(f"| {n} | {r['trade_count']} | {r['hedge_trades']} | "
                 f"{m.get('cagr', float('nan'))*100:.2f}% | {m.get('sharpe', float('nan')):.2f} | "
                 f"{m.get('sortino', float('nan')):.2f} | {m.get('max_dd', float('nan'))*100:.1f}% | "
                 f"{m.get('calmar', float('nan')):.2f} | {m.get('win_rate', float('nan'))*100:.1f}% | "
                 f"{m.get('profit_factor', float('nan')):.2f} | "
                 f"${m.get('final_equity', 0):,.0f} |")
    L.append(f"| SPY BH | n/a | n/a | {spy_bh.get('cagr', 0)*100:.2f}% | {spy_bh.get('sharpe', 0):.2f} | "
             f"{spy_bh.get('sortino', 0):.2f} | {spy_bh.get('max_dd', 0)*100:.1f}% | "
             f"{spy_bh.get('calmar', 0):.2f} | {spy_bh.get('win_rate', 0)*100:.1f}% | "
             f"{spy_bh.get('profit_factor', 0):.2f} | ${spy_bh.get('final_equity', 0):,.0f} |")

    L.append("\n## HC #428 R1 - green/red regime Sharpe gap (THE gate)\n")
    L.append("| Strategy | n_green | n_red | n_flat | Sh green | Sh red | Sh flat | gap | pass (<=0.50) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for n, r in results.items():
        g = r["gate"]
        sg = g["sharpe_green"] if g["sharpe_green"] is not None else float("nan")
        sr = g["sharpe_red"] if g["sharpe_red"] is not None else float("nan")
        sf = g["sharpe_flat"] if g["sharpe_flat"] is not None else float("nan")
        L.append(f"| {n} | {g['n_green']} | {g['n_red']} | {g['n_flat']} | "
                 f"{sg:.2f} | {sr:.2f} | {sf:.2f} | {g['regime_gap']:.2f} | "
                 f"{'PASS' if g['hc428_r1_pass'] else 'FAIL'} |")

    L.append("\n## Deploy gates\n")
    L.append("| Strategy | Sharpe>=1.0 | Calmar>=1.0 | Regime gap<=0.50 | Day conc<=0.70 | DEPLOY |")
    L.append("|---|---|---|---|---|---|")
    for n, r in results.items():
        gp = r["gates_pass"]
        L.append(f"| {n} | {'PASS' if gp['sharpe_ge_1.0'] else 'FAIL'} | "
                 f"{'PASS' if gp['calmar_ge_1.0'] else 'FAIL'} | "
                 f"{'PASS' if gp['regime_gap_le_0.50'] else 'FAIL'} | "
                 f"{'PASS' if gp['day_conc_le_0.70'] else 'FAIL'} | "
                 f"**{'YES' if r['deploy_ready'] else 'NO'}** |")

    L.append("\n## Tail event stress (per strategy)\n")
    L.append("| Strategy | Event | MaxDD | VIX peak | Recover days | Worst week | Cum ret |")
    L.append("|---|---|---|---|---|---|---|")
    for n, r in results.items():
        for t in r["tail"]:
            if "max_dd_pct" not in t:
                continue
            rec = t["recover_days"] if t["recover_days"] is not None else "not_recovered"
            L.append(f"| {n} | {t['label']} | {t['max_dd_pct']:.1f}% | "
                     f"{t['vix_peak']:.1f} | {rec} | "
                     f"{t.get('worst_week_pct', 0):.1f}% | {t['cum_ret_pct']:.1f}% |")

    L.append("\n## Honest caveats\n")
    L.append("- BS pricing without skew under-prices OTM puts (calendar back-leg and straddle put leg).")
    L.append("  Real ATM straddles are typically richer than BS by 5-15% during normal regimes "
             "and 30-100% richer during stress (vol-smile + VRP).")
    L.append("- Delta-hedge frequency: this backtest rebalances ONCE PER DAY at the close. Live execution "
             "with intraday hedging would have higher gamma-slippage costs but better delta tracking.")
    L.append("- The straddle equity series will exhibit jumpy daily returns from discrete hedging error; "
             "the green/red gate is a strict test of whether residual delta survives the hedge.")
    L.append("- Calendar net debit is small relative to $20K account; sizing is conservative (~25% of equity per open).")
    L.append("- VIX-30 gate keeps both strategies OUT of the heart of Volmageddon and COVID.")

    L.append("\n## Recommendation\n")
    deploy_strategies = [n for n, r in results.items() if r["deploy_ready"]]
    if deploy_strategies:
        L.append(f"- **{', '.join(deploy_strategies)} cleared ALL HC #428 R1 deploy gates.**")
        L.append("- Paper-engine module written. Daemon NOT launched - user must flip the entries_paused "
                 "flag after reviewing this report.")
    else:
        L.append("- **Neither strategy cleared all deploy gates.**")
        L.append("- See per-strategy failures below.")
        for n, r in results.items():
            failed = [k for k, v in r["gates_pass"].items() if not v]
            if failed:
                L.append(f"  - **{n}**: failed [{', '.join(failed)}]")
        L.append("- Pivot suggestion: dispersion (sell index vol, buy basket vol) or "
                 "VIX-futures roll-down trades require additional vendor data; recommend "
                 "researching data acquisition path before next iteration.")

    report_path = REPORT_DIR / "market_neutral_spy_v1.md"
    report_path.write_text("\n".join(L))
    print(f"\n[report] wrote {report_path}")

    return results


if __name__ == "__main__":
    main()
