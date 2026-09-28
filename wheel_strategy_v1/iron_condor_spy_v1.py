#!/usr/bin/env python3
"""
iron_condor_spy_v1.py - Backtest a SPY iron condor strategy 2018-2025, applying
HC #428 R1 deploy gates. Per HC #586 R1 this is the FINAL carry/income lane
dispatch tonight - if it fails, the carry/income direction is structurally
exhausted and the recommended pivot is event-driven research.

Strategy
--------
At t=0 (and whenever we don't already have a position and VIX <= 28):
  - SELL  short put at ~10 delta (~ 1.28 sigma OTM)
  - BUY   long put at ~5 delta (~ 1.64 sigma OTM, further OTM = wing)
  - SELL  short call at ~10 delta
  - BUY   long call at ~5 delta (wing)
  - Target DTE 30 (find next Friday >= today + 30d).
  - Profit take at 50% of max profit (max profit = initial net credit).
  - Roll/close at DTE <= 7.
  - Gamma management: if absolute net gamma per $ underlying move exceeds
    threshold, close and re-open at new ATM (rebalance).
  - Intraday VIX > 30 stop: close ALL legs at market mid, do not re-enter
    until VIX < 25.
  - VIX gate at entry: don't open if VIX > 28.

Cost model
----------
  - $0.65 / contract per leg (4 legs = $2.60 per open, $2.60 per close)
  - 1 bp slippage on each leg premium

Data
----
Reuses SPY price / IV / VIX daily panel from the wheel cache. Augments with
yfinance daily VIX High to approximate "intraday VIX peak" for the >30 stop
(daily resolution is the best we have; documented honestly in the report).

Outputs
-------
  - MLflow experiment "iron_condor_spy_v1"
  - /home/jupiter/Lvl3Quant/research/findings/iron_condor_spy_v1.md
  - /home/jupiter/Lvl3Quant/output/iron_condor_spy_v1/
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "iron_condor_spy_v1"
REPORT_DIR = ROOT / "research" / "findings"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CASH = 20_000.0
RISK_FREE = 0.04
TRADING_DAYS = 252

COMMISSION_PER_CONTRACT = 0.65
OPTION_SLIPPAGE_BPS = 1.0  # 1 bp slippage on each leg premium

VIX_GATE_OPEN = 28.0
VIX_INTRADAY_STOP = 30.0
VIX_REENTRY_OK = 25.0
GAMMA_RATIO_TRIGGER = 1.75  # rebalance if current_gamma > 1.75x opening gamma
START_DATE = pd.Timestamp("2018-01-01")
END_DATE = pd.Timestamp("2025-12-31")
DTE_OPEN_TARGET = 30
DTE_CLOSE_TRIGGER = 7
PROFIT_TAKE_FRAC = 0.50


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
        if kind == "put":
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _Phi(d1) - 1.0 if kind == "put" else _Phi(d1)


def bs_gamma(S, K, T, sigma, r=RISK_FREE):
    """Gamma is the same for calls and puts."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _phi(d1) / (S * sigma * math.sqrt(T))


def strike_for_delta(S, T, sigma, target_delta, kind="put"):
    """Solve for strike K such that |delta(K)| ~= target_delta.
    For puts, delta is negative; target_delta is the absolute value (e.g. 0.10).
    Closed form via inverse normal."""
    if T <= 0 or sigma <= 0:
        return S
    # For a put: delta_put = N(d1) - 1 = -target_delta -> N(d1) = 1 - target_delta
    # For a call: delta_call = N(d1) = target_delta
    if kind == "put":
        N_d1 = 1.0 - target_delta
    else:
        N_d1 = target_delta
    # Inverse normal (Beasley-Springer-Moro approximation)
    # Use scipy if available else inline approximation
    try:
        from scipy.stats import norm
        d1 = norm.ppf(N_d1)
    except Exception:
        # Acklam's approximation
        def _ndtri(p):
            a = [-3.969683028665376e+01, 2.209460984245205e+02,
                 -2.759285104469687e+02, 1.383577518672690e+02,
                 -3.066479806614716e+01, 2.506628277459239e+00]
            b = [-5.447609879822406e+01, 1.615858368580409e+02,
                 -1.556989798598866e+02, 6.680131188771972e+01,
                 -1.328068155288572e+01]
            c = [-7.784894002430293e-03, -3.223964580411365e-01,
                 -2.400758277161838e+00, -2.549732539343734e+00,
                 4.374664141464968e+00, 2.938163982698783e+00]
            d = [7.784695709041462e-03, 3.224671290700398e-01,
                 2.445134137142996e+00, 3.754408661907416e+00]
            p_low = 0.02425
            p_high = 1 - p_low
            if p < p_low:
                q = math.sqrt(-2 * math.log(p))
                return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                       ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
            elif p <= p_high:
                q = p - 0.5
                r = q * q
                return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                       (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
            else:
                q = math.sqrt(-2 * math.log(1 - p))
                return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        d1 = _ndtri(N_d1)
    # K = S * exp(-(d1 * sigma * sqrt(T) - (r + 0.5 sigma^2) T))
    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (RISK_FREE + 0.5 * sigma * sigma) * T))
    # Round to nearest dollar for SPY chain realism
    return round(K)


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
    df["rv_20d"] = df["spy_ret"].rolling(20).std() * math.sqrt(TRADING_DAYS)
    df["rv_20d"] = df["rv_20d"].clip(lower=0.05, upper=1.5).fillna(df["sigma"])

    # Fetch VIX High (intraday peak proxy) via yfinance
    try:
        import yfinance as yf
        vh = yf.download("^VIX", start=str(START_DATE.date()),
                         end=str((END_DATE + pd.Timedelta(days=2)).date()),
                         progress=False, auto_adjust=False)
        if isinstance(vh.columns, pd.MultiIndex):
            vh.columns = [c[0] for c in vh.columns]
        vh = vh[["High", "Low", "Close"]].copy()
        vh.index = pd.to_datetime(vh.index).tz_localize(None)
        vh.columns = ["vix_high", "vix_low", "vix_close_yf"]
        df = df.join(vh, how="left")
        df["vix_high"] = df["vix_high"].ffill()
    except Exception as e:
        print(f"[warn] yfinance VIX intraday fetch failed: {e}; using vix close as proxy")
        df["vix_high"] = df["vix"]

    return df


def find_friday_after(d, days_ahead):
    cand = d + pd.Timedelta(days=days_ahead)
    shift = (4 - cand.weekday()) % 7
    return cand + pd.Timedelta(days=shift)


@dataclass
class CondorPos:
    open_date: pd.Timestamp
    expiry: pd.Timestamp
    short_put_K: float
    long_put_K: float
    short_call_K: float
    long_call_K: float
    open_sigma: float
    open_S: float
    open_short_put_px: float
    open_long_put_px: float
    open_short_call_px: float
    open_long_call_px: float
    open_net_credit_per_share: float  # before commissions/slippage
    open_max_profit_per_share: float  # = net credit (cash received minus cost paid for wings)
    open_gamma_per_share: float  # |net gamma| at open per 1-lot equiv
    contracts: int

    @property
    def short_put_width(self):
        return self.short_put_K - self.long_put_K  # positive

    @property
    def short_call_width(self):
        return self.long_call_K - self.short_call_K  # positive


@dataclass
class State:
    cash: float
    equity_curve: list = field(default_factory=list)
    ledger: list = field(default_factory=list)
    opens: int = 0
    gamma_rebalances: int = 0
    vix_stops: int = 0
    profit_takes: int = 0
    dte_closes: int = 0
    expiry_closes: int = 0
    in_vix_blackout: bool = False  # True after vix stop until vix < 25


def _premium_with_slip_sell(px):
    return px * (1.0 - OPTION_SLIPPAGE_BPS / 10_000.0)


def _premium_with_slip_buy(px):
    return px * (1.0 + OPTION_SLIPPAGE_BPS / 10_000.0)


def open_condor(state, today, S, sigma):
    """Open a 4-leg iron condor at ~10/5 delta."""
    expiry = find_friday_after(today, DTE_OPEN_TARGET)
    T = max((expiry - today).days, 1) / 365.0
    K_sp = strike_for_delta(S, T, sigma, 0.10, kind="put")
    K_lp = strike_for_delta(S, T, sigma, 0.05, kind="put")
    K_sc = strike_for_delta(S, T, sigma, 0.10, kind="call")
    K_lc = strike_for_delta(S, T, sigma, 0.05, kind="call")
    # Sanity: long put must be below short put; long call above short call
    if K_lp >= K_sp:
        K_lp = K_sp - 1
    if K_lc <= K_sc:
        K_lc = K_sc + 1

    sp_px = bs_price(S, K_sp, T, sigma, kind="put")
    lp_px = bs_price(S, K_lp, T, sigma, kind="put")
    sc_px = bs_price(S, K_sc, T, sigma, kind="call")
    lc_px = bs_price(S, K_lc, T, sigma, kind="call")

    credit_short = _premium_with_slip_sell(sp_px) + _premium_with_slip_sell(sc_px)
    debit_long = _premium_with_slip_buy(lp_px) + _premium_with_slip_buy(lc_px)
    net_credit_per_share = credit_short - debit_long
    if net_credit_per_share <= 0.05:
        return None  # credit too small to bother

    # Max loss per condor (per share) = max(put_width, call_width) - net_credit
    put_width = K_sp - K_lp
    call_width = K_lc - K_sc
    max_loss_per_share = max(put_width, call_width) - net_credit_per_share
    if max_loss_per_share <= 0:
        return None  # arbitrage / pricing artifact, skip

    # Sizing: 1 contract per ~$1500 of margin-equivalent risk to stay
    # conservative on $20K account. max_loss * 100 = $ at risk per contract.
    risk_per_contract = max_loss_per_share * 100 + COMMISSION_PER_CONTRACT * 4
    if risk_per_contract <= 0:
        return None
    # Allocate up to 50% of equity to risk on a single condor (4 legs, defined risk).
    max_contracts = max(1, int((state.cash * 0.50) / risk_per_contract))
    contracts = max_contracts

    # Cash flow at open: receive net credit, pay 4 commissions
    state.cash += net_credit_per_share * 100 * contracts
    state.cash -= COMMISSION_PER_CONTRACT * 4 * contracts

    # Gamma: short condor has negative gamma; magnitude = |sum of signed gammas|
    g = (
        -bs_gamma(S, K_sp, T, sigma)
        + bs_gamma(S, K_lp, T, sigma)
        - bs_gamma(S, K_sc, T, sigma)
        + bs_gamma(S, K_lc, T, sigma)
    )
    open_gamma = abs(g)
    state.opens += 1

    return CondorPos(
        open_date=today, expiry=expiry,
        short_put_K=K_sp, long_put_K=K_lp,
        short_call_K=K_sc, long_call_K=K_lc,
        open_sigma=sigma, open_S=S,
        open_short_put_px=sp_px, open_long_put_px=lp_px,
        open_short_call_px=sc_px, open_long_call_px=lc_px,
        open_net_credit_per_share=net_credit_per_share,
        open_max_profit_per_share=net_credit_per_share,
        open_gamma_per_share=open_gamma,
        contracts=contracts,
    )


def mtm_condor(pos, today, S, sigma):
    """Return (unrealized_pnl, current_liability_per_share, profit_frac_of_max,
                current_abs_gamma, leg_pxs_dict)."""
    T = max((pos.expiry - today).days, 0) / 365.0
    sp = bs_price(S, pos.short_put_K, T, sigma, kind="put")
    lp = bs_price(S, pos.long_put_K, T, sigma, kind="put")
    sc = bs_price(S, pos.short_call_K, T, sigma, kind="call")
    lc = bs_price(S, pos.long_call_K, T, sigma, kind="call")
    # To close: BUY back shorts, SELL the longs.
    cost_to_close_short = _premium_with_slip_buy(sp) + _premium_with_slip_buy(sc)
    proceeds_close_long = _premium_with_slip_sell(lp) + _premium_with_slip_sell(lc)
    cost_per_share_close = cost_to_close_short - proceeds_close_long
    # Unrealized pnl per share = open_net_credit - cost_per_share_close
    unrealized_per_share = pos.open_net_credit_per_share - cost_per_share_close
    pnl_total = unrealized_per_share * 100 * pos.contracts
    profit_frac = unrealized_per_share / pos.open_max_profit_per_share \
        if pos.open_max_profit_per_share > 0 else 0.0

    g = (
        -bs_gamma(S, pos.short_put_K, T, sigma)
        + bs_gamma(S, pos.long_put_K, T, sigma)
        - bs_gamma(S, pos.short_call_K, T, sigma)
        + bs_gamma(S, pos.long_call_K, T, sigma)
    )
    cur_gamma = abs(g)
    return (pnl_total, cost_per_share_close, profit_frac, cur_gamma,
            {"sp": sp, "lp": lp, "sc": sc, "lc": lc})


def close_condor(state, pos, today, S, sigma, reason=""):
    T = max((pos.expiry - today).days, 0) / 365.0
    sp = bs_price(S, pos.short_put_K, T, sigma, kind="put")
    lp = bs_price(S, pos.long_put_K, T, sigma, kind="put")
    sc = bs_price(S, pos.short_call_K, T, sigma, kind="call")
    lc = bs_price(S, pos.long_call_K, T, sigma, kind="call")
    cost_to_close_short = _premium_with_slip_buy(sp) + _premium_with_slip_buy(sc)
    proceeds_close_long = _premium_with_slip_sell(lp) + _premium_with_slip_sell(lc)
    cost_per_share = cost_to_close_short - proceeds_close_long
    state.cash -= cost_per_share * 100 * pos.contracts
    state.cash -= COMMISSION_PER_CONTRACT * 4 * pos.contracts

    realized = (pos.open_net_credit_per_share - cost_per_share) * 100 * pos.contracts \
        - COMMISSION_PER_CONTRACT * 8 * pos.contracts  # open + close commissions
    pct_of_max = realized / (pos.open_max_profit_per_share * 100 * pos.contracts) \
        if pos.open_max_profit_per_share > 0 else 0.0

    state.ledger.append({
        "open_date": pos.open_date,
        "close_date": today,
        "expiry": pos.expiry,
        "strikes": f"{pos.long_put_K}/{pos.short_put_K}/{pos.short_call_K}/{pos.long_call_K}",
        "contracts": pos.contracts,
        "open_credit_per_share": pos.open_net_credit_per_share,
        "close_cost_per_share": cost_per_share,
        "realized_pnl": realized,
        "pct_of_max_profit": pct_of_max,
        "days_in_trade": (today - pos.open_date).days,
        "reason": reason,
    })


def run_iron_condor(df):
    state = State(cash=STARTING_CASH)
    pos = None
    for today in df.index:
        row = df.loc[today]
        S = float(row["close"])
        sigma_iv = float(row["sigma"])
        sigma_rv = float(row["rv_20d"])
        vix_close = float(row["vix"]) if pd.notna(row["vix"]) else 0.0
        vix_high = float(row["vix_high"]) if "vix_high" in row and pd.notna(row["vix_high"]) else vix_close

        # VIX intraday stop: applies to OPEN positions only.
        if pos is not None and vix_high > VIX_INTRADAY_STOP:
            close_condor(state, pos, today, S, sigma_rv, reason="vix_intraday_stop")
            pos = None
            state.vix_stops += 1
            state.in_vix_blackout = True

        # Blackout release when vix close < re-entry level
        if state.in_vix_blackout and vix_close < VIX_REENTRY_OK:
            state.in_vix_blackout = False

        # Manage existing position
        if pos is not None:
            pnl, cost_close_ps, profit_frac, cur_gamma, _ = mtm_condor(
                pos, today, S, sigma_rv)
            days_to_exp = (pos.expiry - today).days

            close_reason = None
            if profit_frac >= PROFIT_TAKE_FRAC:
                close_reason = "profit_take_50pct"
                state.profit_takes += 1
            elif days_to_exp <= DTE_CLOSE_TRIGGER:
                close_reason = "dte_7"
                state.dte_closes += 1
            elif today >= pos.expiry:
                close_reason = "expiry"
                state.expiry_closes += 1
            elif pos.open_gamma_per_share > 0 and \
                    (cur_gamma / pos.open_gamma_per_share) > GAMMA_RATIO_TRIGGER:
                close_reason = "gamma_rebalance"
                state.gamma_rebalances += 1

            if close_reason is not None:
                close_condor(state, pos, today, S, sigma_rv, reason=close_reason)
                pos = None

        # Open new
        if pos is None and not state.in_vix_blackout and \
                vix_close > 0 and vix_close <= VIX_GATE_OPEN and \
                state.cash > 2000:
            new_pos = open_condor(state, today, S, sigma_iv)
            if new_pos is not None:
                pos = new_pos

        # MTM equity
        equity = state.cash
        if pos is not None:
            pnl, cost_close_ps, _, _, _ = mtm_condor(pos, today, S, sigma_rv)
            # Open credit is already in cash. Liability = current close cost.
            equity -= cost_close_ps * 100 * pos.contracts
        state.equity_curve.append({
            "date": today, "equity": equity, "S": S,
            "vix": vix_close, "vix_high": vix_high,
            "spy_ret": float(row["spy_ret"]) if pd.notna(row["spy_ret"]) else np.nan,
        })

    if pos is not None:
        S = float(df.iloc[-1]["close"])
        sigma = float(df.iloc[-1]["rv_20d"])
        close_condor(state, pos, df.index[-1], S, sigma, reason="end_of_window")

    eq_df = pd.DataFrame(state.equity_curve).set_index("date")
    eq_df["ret"] = eq_df["equity"].pct_change()
    ledger_df = pd.DataFrame(state.ledger)
    return eq_df, ledger_df, state


# ============================================================
# Metrics & gates
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
    pos_sum = rets[rets > 0].sum()
    neg_sum = abs(rets[rets < 0].sum())
    pf = float(pos_sum / neg_sum) if neg_sum > 0 else float("nan")
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
        "n_green": int(len(green)), "n_red": int(len(red)), "n_flat": int(len(flat)),
        "sharpe_green": float(sh_g) if np.isfinite(sh_g) else None,
        "sharpe_red": float(sh_r) if np.isfinite(sh_r) else None,
        "sharpe_flat": float(sh_f) if np.isfinite(sh_f) else None,
        "regime_gap": float(gap),
        "hc428_r1_pass": bool(gap <= 0.50),
    }


def day_concentration(ledger_df):
    if ledger_df is None or ledger_df.empty:
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
        "label": label, "window": f"{start} to {end}",
        "n_days": int(len(sub)), "max_dd_pct": max_dd * 100,
        "vix_peak": float(sub["vix_high"].max()) if "vix_high" in sub.columns else float(sub["vix"].max()),
        "recover_days": rec_days,
        "worst_week_pct": worst_week * 100 if np.isfinite(worst_week) else None,
        "cum_ret_pct": float((sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1.0) * 100.0),
    }


def spy_buy_hold_metrics(df):
    eq = (df["close"] / df["close"].iloc[0]) * STARTING_CASH
    ed = pd.DataFrame({"equity": eq, "spy_ret": df["spy_ret"]})
    ed["ret"] = ed["equity"].pct_change()
    return compute_metrics(ed)


def log_to_mlflow(metrics, gate, day_conc, tail, eq_df, ledger_df, state, spy_bh):
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("iron_condor_spy_v1")
        with mlflow.start_run(run_name="iron_condor_10d_5d_30dte"):
            mlflow.log_param("starting_cash", STARTING_CASH)
            mlflow.log_param("dte_open_target", DTE_OPEN_TARGET)
            mlflow.log_param("dte_close_trigger", DTE_CLOSE_TRIGGER)
            mlflow.log_param("profit_take_frac", PROFIT_TAKE_FRAC)
            mlflow.log_param("vix_gate_open", VIX_GATE_OPEN)
            mlflow.log_param("vix_intraday_stop", VIX_INTRADAY_STOP)
            mlflow.log_param("vix_reentry_ok", VIX_REENTRY_OK)
            mlflow.log_param("gamma_ratio_trigger", GAMMA_RATIO_TRIGGER)
            mlflow.log_param("short_delta", 0.10)
            mlflow.log_param("long_delta", 0.05)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(k, v)
            for k, v in gate.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(f"regime_{k}", v)
            mlflow.log_metric("day_concentration",
                              day_conc if np.isfinite(day_conc) else -1.0)
            mlflow.log_metric("hc428_r1_pass", 1.0 if gate.get("hc428_r1_pass") else 0.0)
            n_trades = int(len(ledger_df)) if ledger_df is not None and not ledger_df.empty else 0
            mlflow.log_metric("trade_count", n_trades)
            mlflow.log_metric("opens", state.opens)
            mlflow.log_metric("gamma_rebalances", state.gamma_rebalances)
            mlflow.log_metric("vix_stops", state.vix_stops)
            mlflow.log_metric("profit_takes", state.profit_takes)
            mlflow.log_metric("dte_closes", state.dte_closes)
            mlflow.log_metric("expiry_closes", state.expiry_closes)
            if not ledger_df.empty:
                mlflow.log_metric("avg_days_in_trade", float(ledger_df["days_in_trade"].mean()))
                mlflow.log_metric("avg_pct_of_max_profit", float(ledger_df["pct_of_max_profit"].mean()))
            for t in tail:
                if "max_dd_pct" in t:
                    safe = t["label"].replace(" ", "_")
                    mlflow.log_metric(f"tail_{safe}_max_dd_pct", t["max_dd_pct"])
                    if t.get("recover_days") is not None:
                        mlflow.log_metric(f"tail_{safe}_recover_days", t["recover_days"])
            for k, v in spy_bh.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(f"spy_bh_{k}", v)
            csv = OUT_DIR / "equity.csv"
            eq_df.to_csv(csv)
            mlflow.log_artifact(str(csv))
            return mlflow.active_run().info.run_id
    except Exception as e:
        print(f"[mlflow] skipped ({e})")
        return None


def main():
    print("[load] data ...")
    df = load_data()
    print(f"[load] {len(df)} trading days {df.index[0].date()} -> {df.index[-1].date()}")
    spy_bh = spy_buy_hold_metrics(df)
    print(f"[SPY BH] CAGR={spy_bh['cagr']*100:.2f}% Sharpe={spy_bh['sharpe']:.2f} "
          f"MaxDD={spy_bh['max_dd']*100:.1f}%")

    print("[backtest] iron condor 10d/5d 30DTE ...")
    eq_df, ledger_df, state = run_iron_condor(df)
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
        "sharpe_ge_1.0": metrics.get("sharpe", float("nan")) >= 1.0
            if np.isfinite(metrics.get("sharpe", float("nan"))) else False,
        "calmar_ge_1.0": metrics.get("calmar", float("nan")) >= 1.0
            if np.isfinite(metrics.get("calmar", float("nan"))) else False,
        "regime_gap_le_0.50": gate["hc428_r1_pass"],
        "day_conc_le_0.70": (day_conc <= 0.70) if np.isfinite(day_conc) else True,
        "cagr_positive": metrics.get("cagr", float("nan")) > 0
            if np.isfinite(metrics.get("cagr", float("nan"))) else False,
    }
    deploy_ready = all(gates_pass.values())

    eq_df.to_parquet(OUT_DIR / "equity.parquet")
    if not ledger_df.empty:
        ledger_df.to_parquet(OUT_DIR / "ledger.parquet")
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({
            "strategy": "iron_condor_spy_v1",
            "metrics": metrics, "gate": gate, "day_conc": day_conc, "tail": tail,
            "trade_count": trade_count, "opens": state.opens,
            "gamma_rebalances": state.gamma_rebalances,
            "vix_stops": state.vix_stops,
            "profit_takes": state.profit_takes,
            "dte_closes": state.dte_closes,
            "expiry_closes": state.expiry_closes,
            "avg_days_in_trade": float(ledger_df["days_in_trade"].mean()) if not ledger_df.empty else None,
            "avg_pct_of_max_profit": float(ledger_df["pct_of_max_profit"].mean()) if not ledger_df.empty else None,
            "gates_pass": gates_pass, "deploy_ready": deploy_ready,
        }, f, indent=2, default=str)
    run_id = log_to_mlflow(metrics, gate, day_conc, tail, eq_df, ledger_df, state, spy_bh)

    sh = metrics.get("sharpe", float("nan"))
    ca = metrics.get("calmar", float("nan"))
    cagr = metrics.get("cagr", float("nan"))
    mdd = metrics.get("max_dd", float("nan"))
    print(f"[IronCondor] Sharpe={sh:.2f} CAGR={cagr*100:.2f}% MaxDD={mdd*100:.1f}% "
          f"Calmar={ca:.2f} regime_gap={gate['regime_gap']:.2f} "
          f"trades={trade_count} vix_stops={state.vix_stops} "
          f"gamma_rebal={state.gamma_rebalances} deploy={deploy_ready}")

    return {
        "metrics": metrics, "gate": gate, "day_conc": day_conc, "tail": tail,
        "trade_count": trade_count, "opens": state.opens,
        "gamma_rebalances": state.gamma_rebalances,
        "vix_stops": state.vix_stops,
        "profit_takes": state.profit_takes,
        "dte_closes": state.dte_closes,
        "expiry_closes": state.expiry_closes,
        "gates_pass": gates_pass, "deploy_ready": deploy_ready,
        "mlflow_run_id": run_id,
        "ledger_df": ledger_df,
        "spy_bh": spy_bh,
    }


if __name__ == "__main__":
    main()
