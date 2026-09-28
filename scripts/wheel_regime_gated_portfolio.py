#!/usr/bin/env python3
"""
wheel_regime_gated_portfolio.py — Regime-gated wheel portfolio backtest.

Copies the engine from wheel_fixed_portfolio.py but adds a bear market gate
based on SPY 50-day SMA. Three modes tested:
  - "none":  no bear gate (original behavior, baseline)
  - "halt":  when SPY < 50d SMA, no new CSPs opened. Existing positions
             managed normally (profit-take, assignment, CCs).
  - "half":  when in bear regime, reduce per-name allocation to 50%.

Goal: close the regime gap (HC #428 R1 requires |Sharpe_bull - Sharpe_bear|
/ max(|Sharpe_bull|, |Sharpe_bear|) <= 0.50).
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------- paths ----------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "wheel_regime_gated"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "wheel_regime_gated_portfolio.log"

logging.basicConfig(
    format='%(asctime)s [WHEEL-RG] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE)),
    ],
)
log = logging.getLogger('WHEEL-RG')

# ----------------------------- config ---------------------------------------
BASKET = {
    'AEP':  ('Utilities',),
    'DLR':  ('Real Estate',),
    'PSX':  ('Energy',),
    'XOM':  ('Energy',),
    'QCOM': ('Technology',),
    'TSM':  ('Technology',),
    'TXN':  ('Technology',),
    'VZ':   ('Communication Services',),
    'PFE':  ('Healthcare',),
    'MDT':  ('Healthcare',),
    'UPS':  ('Industrials',),
    'WYNN': ('Consumer Cyclical',),
    'SBUX': ('Consumer Cyclical',),
    'CL':   ('Consumer Defensive',),
    'COST': ('Consumer Defensive',),
}

TICKERS = sorted(BASKET.keys())
N_NAMES = len(TICKERS)

START_DATE = pd.Timestamp("2019-01-01")
STARTING_CASH = 100_000.0
PER_NAME_ALLOC = STARTING_CASH / N_NAMES
TRADING_DAYS = 252
RISK_FREE = 0.04

PUT_DELTA = 0.25
CALL_DELTA = 0.30
DTE_MIN = 25
DTE_MAX = 35
DTE_TARGET = 30
PROFIT_TAKE = 0.50
VIX_MAX = 35.0

MAX_NOTIONAL_PCT = 0.95
MAX_PER_NAME_PCT = 1.0 / N_NAMES + 0.02

COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03

FLAT_BAND = 0.0025

SMA_PERIOD = 50  # SPY 50-day SMA for bear gate


# ----------------------------- BS pricing -----------------------------------
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
    K = S * math.exp((r + 0.5 * sigma * sigma) * T - d1 * sigma * math.sqrt(T))
    return max(0.01, round(K, 2))


def slippage(premium):
    if premium is None or premium <= 0:
        return 0.0
    return max(SLIPPAGE_MIN, SLIPPAGE_FRAC * premium)


# ----------------------------- expiry finder --------------------------------
def find_expiry(open_date):
    best, best_dist = None, 10_000
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


# ----------------------------- position tracking ----------------------------
@dataclass
class Position:
    ticker: str
    side: str       # 'short_put' | 'long_shares' | 'short_call'
    strike: float
    expiry: pd.Timestamp
    open_date: pd.Timestamp
    open_price: float
    contracts: int
    share_basis: float = 0.0


@dataclass
class TradeRecord:
    ticker: str
    kind: str
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    strike: float
    contracts: int
    premium_open: float
    premium_close: float
    realized_pnl: float
    exit_reason: str
    S_at_close: float = 0.0


# ----------------------------- portfolio engine -----------------------------
def reserved_collateral(positions: dict) -> float:
    total = 0.0
    for tk, pos in positions.items():
        if pos.side == 'short_put':
            total += pos.strike * 100 * pos.contracts
    return total


def available_cash(cash: float, positions: dict) -> float:
    return max(0.0, cash - reserved_collateral(positions))


def position_notional(pos: Position, S: float) -> float:
    if pos.side == 'short_put':
        return pos.strike * 100 * pos.contracts
    elif pos.side in ('long_shares', 'short_call'):
        return S * 100 * pos.contracts
    return 0.0


def run_portfolio_wheel(closes_by_ticker, sigmas_by_ticker,
                        vix_series, all_dates,
                        spy_closes, spy_sma50,
                        bear_mode="none",
                        spy_sma200=None):
    """
    Run portfolio wheel across all 15 names sharing capital.

    bear_mode:
      - "none": original behavior (baseline)
      - "halt": when SPY close < SPY 50d SMA, no new CSPs opened
      - "half": when in bear regime, per-name allocation halved
    """
    cash = STARTING_CASH
    positions = {}
    ledger = []
    equity_series = []
    force_close_count = 0
    bear_days_blocked = 0  # Track how many days the gate blocked new CSPs

    stats = {tk: {'csp_opened': 0, 'cc_opened': 0, 'assignments': 0,
                   'call_aways': 0, 'force_closes': 0} for tk in TICKERS}

    for dt in all_dates:
        vix = vix_series.get(dt, 20.0)

        # Determine bear regime for this date
        spy_close = spy_closes.get(dt)
        if bear_mode == "liquidate_200" and spy_sma200 is not None:
            sma_val = spy_sma200.get(dt)
        else:
            sma_val = spy_sma50.get(dt)
        is_bear = False
        if spy_close is not None and sma_val is not None:
            is_bear = spy_close < sma_val

        # ---- 1) Process existing positions (ALWAYS, regardless of bear mode) ----
        to_remove = []
        for tk, pos in list(positions.items()):
            S = closes_by_ticker.get(tk, {}).get(dt)
            sigma = sigmas_by_ticker.get(tk, {}).get(dt)
            if S is None or sigma is None or np.isnan(S) or np.isnan(sigma):
                continue

            T = max((pos.expiry - dt).days, 0) / 365.0

            if pos.side == 'short_put':
                opt = bs_price(S, pos.strike, T, sigma, kind="put")
                pnl_ps = pos.open_price - opt
                pf = pnl_ps / pos.open_price if pos.open_price > 0 else 0.0
                is_expiry = dt >= pos.expiry

                if pf >= PROFIT_TAKE and not is_expiry:
                    slip = slippage(opt)
                    cost = (opt + slip) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    realized = pos.open_price * 100 * pos.contracts - cost
                    cash += realized
                    ledger.append(TradeRecord(
                        ticker=tk, kind='CSP', open_date=pos.open_date,
                        close_date=dt, strike=pos.strike, contracts=pos.contracts,
                        premium_open=pos.open_price, premium_close=opt,
                        realized_pnl=realized, exit_reason='profit_take',
                        S_at_close=S,
                    ))
                    to_remove.append(tk)

                elif is_expiry:
                    if S < pos.strike:
                        assignment_cost = pos.strike * 100 * pos.contracts
                        if assignment_cost > cash:
                            loss_per_share = pos.strike - S
                            total_loss = loss_per_share * 100 * pos.contracts
                            premium_kept = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                            realized = premium_kept - total_loss
                            cash += realized
                            if cash < 0:
                                realized += cash
                                cash = 0.0
                            force_close_count += 1
                            stats[tk]['force_closes'] += 1
                            ledger.append(TradeRecord(
                                ticker=tk, kind='CSP_FORCE_CLOSE',
                                open_date=pos.open_date, close_date=dt,
                                strike=pos.strike, contracts=pos.contracts,
                                premium_open=pos.open_price, premium_close=0.0,
                                realized_pnl=realized, exit_reason='force_close_insufficient_cash',
                                S_at_close=S,
                            ))
                            to_remove.append(tk)
                        else:
                            cash -= assignment_cost
                            basis = pos.strike - pos.open_price
                            stats[tk]['assignments'] += 1
                            ledger.append(TradeRecord(
                                ticker=tk, kind='CSP_ASSIGNED',
                                open_date=pos.open_date, close_date=dt,
                                strike=pos.strike, contracts=pos.contracts,
                                premium_open=pos.open_price, premium_close=0.0,
                                realized_pnl=pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts,
                                exit_reason='assigned', S_at_close=S,
                            ))
                            positions[tk] = Position(
                                ticker=tk, side='long_shares', strike=basis,
                                expiry=dt, open_date=dt, open_price=basis,
                                contracts=pos.contracts, share_basis=basis,
                            )
                    else:
                        realized = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                        ledger.append(TradeRecord(
                            ticker=tk, kind='CSP', open_date=pos.open_date,
                            close_date=dt, strike=pos.strike, contracts=pos.contracts,
                            premium_open=pos.open_price, premium_close=0.0,
                            realized_pnl=realized, exit_reason='expired_worthless',
                            S_at_close=S,
                        ))
                        to_remove.append(tk)

            elif pos.side == 'short_call':
                opt = bs_price(S, pos.strike, T, sigma, kind="call")
                pnl_ps = pos.open_price - opt
                pf = pnl_ps / pos.open_price if pos.open_price > 0 else 0.0
                is_expiry = dt >= pos.expiry

                if pf >= PROFIT_TAKE and not is_expiry:
                    slip = slippage(opt)
                    cost = (opt + slip) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    realized = pos.open_price * 100 * pos.contracts - cost
                    cash += realized
                    ledger.append(TradeRecord(
                        ticker=tk, kind='CC', open_date=pos.open_date,
                        close_date=dt, strike=pos.strike, contracts=pos.contracts,
                        premium_open=pos.open_price, premium_close=opt,
                        realized_pnl=realized, exit_reason='profit_take',
                        S_at_close=S,
                    ))
                    positions[tk] = Position(
                        ticker=tk, side='long_shares', strike=pos.share_basis,
                        expiry=dt, open_date=dt, open_price=pos.share_basis,
                        contracts=pos.contracts, share_basis=pos.share_basis,
                    )

                elif is_expiry:
                    if S > pos.strike:
                        proceeds = pos.strike * 100 * pos.contracts
                        premium_kept = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                        share_pnl = (pos.strike - pos.share_basis) * 100 * pos.contracts
                        cash += proceeds + premium_kept
                        stats[tk]['call_aways'] += 1
                        ledger.append(TradeRecord(
                            ticker=tk, kind='CC_CALLED', open_date=pos.open_date,
                            close_date=dt, strike=pos.strike, contracts=pos.contracts,
                            premium_open=pos.open_price, premium_close=0.0,
                            realized_pnl=premium_kept + share_pnl,
                            exit_reason='called_away', S_at_close=S,
                        ))
                        to_remove.append(tk)
                    else:
                        realized = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                        cash += realized
                        ledger.append(TradeRecord(
                            ticker=tk, kind='CC', open_date=pos.open_date,
                            close_date=dt, strike=pos.strike, contracts=pos.contracts,
                            premium_open=pos.open_price, premium_close=0.0,
                            realized_pnl=realized, exit_reason='expired_worthless',
                            S_at_close=S,
                        ))
                        positions[tk] = Position(
                            ticker=tk, side='long_shares', strike=pos.share_basis,
                            expiry=dt, open_date=dt, open_price=pos.share_basis,
                            contracts=pos.contracts, share_basis=pos.share_basis,
                        )

        for tk in to_remove:
            if tk in positions:
                del positions[tk]

        # ---- 1b) LIQUIDATION GATE: force-sell shares & buy back CCs in bear ----
        if bear_mode in ("liquidate", "liquidate_200", "liq_csp_only", "liq_50pct") and is_bear:
            liq_remove = []
            pos_list = list(positions.items())
            # For liq_50pct: only liquidate every other position (alphabetical)
            if bear_mode == "liq_50pct":
                pos_list = pos_list[::2]  # every other position

            for tk, pos in pos_list:
                S = closes_by_ticker.get(tk, {}).get(dt)
                if S is None or np.isnan(S):
                    continue
                sigma = sigmas_by_ticker.get(tk, {}).get(dt, 0.25)

                # liq_csp_only: only close CSPs, keep shares and CCs
                if bear_mode == "liq_csp_only" and pos.side != 'short_put':
                    continue

                if pos.side == 'short_call':
                    # Buy back the covered call at market, then sell shares
                    T_cc = max((pos.expiry - dt).days, 0) / 365.0
                    cc_val = bs_price(S, pos.strike, T_cc, sigma, kind="call")
                    slip_cc = slippage(cc_val)
                    buyback_cost = (cc_val + slip_cc) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    cc_pnl = pos.open_price * 100 * pos.contracts - buyback_cost
                    cash += cc_pnl
                    ledger.append(TradeRecord(
                        ticker=tk, kind='CC_BEAR_CLOSE', open_date=pos.open_date,
                        close_date=dt, strike=pos.strike, contracts=pos.contracts,
                        premium_open=pos.open_price, premium_close=cc_val,
                        realized_pnl=cc_pnl, exit_reason='bear_liquidation',
                        S_at_close=S,
                    ))
                    # Now sell the underlying shares
                    share_proceeds = S * 100 * pos.contracts
                    share_pnl = (S - pos.share_basis) * 100 * pos.contracts
                    cash += share_proceeds
                    ledger.append(TradeRecord(
                        ticker=tk, kind='SHARES_BEAR_SOLD', open_date=pos.open_date,
                        close_date=dt, strike=pos.share_basis, contracts=pos.contracts,
                        premium_open=0.0, premium_close=0.0,
                        realized_pnl=share_pnl, exit_reason='bear_liquidation',
                        S_at_close=S,
                    ))
                    liq_remove.append(tk)

                elif pos.side == 'long_shares':
                    # Sell shares at market
                    share_proceeds = S * 100 * pos.contracts
                    share_pnl = (S - pos.share_basis) * 100 * pos.contracts
                    cash += share_proceeds
                    ledger.append(TradeRecord(
                        ticker=tk, kind='SHARES_BEAR_SOLD', open_date=pos.open_date,
                        close_date=dt, strike=pos.share_basis, contracts=pos.contracts,
                        premium_open=0.0, premium_close=0.0,
                        realized_pnl=share_pnl, exit_reason='bear_liquidation',
                        S_at_close=S,
                    ))
                    liq_remove.append(tk)

                elif pos.side == 'short_put':
                    # Buy back the CSP at market
                    T_csp = max((pos.expiry - dt).days, 0) / 365.0
                    csp_val = bs_price(S, pos.strike, T_csp, sigma, kind="put")
                    slip_csp = slippage(csp_val)
                    buyback_cost = (csp_val + slip_csp) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    csp_pnl = pos.open_price * 100 * pos.contracts - buyback_cost
                    cash += csp_pnl
                    ledger.append(TradeRecord(
                        ticker=tk, kind='CSP_BEAR_CLOSE', open_date=pos.open_date,
                        close_date=dt, strike=pos.strike, contracts=pos.contracts,
                        premium_open=pos.open_price, premium_close=csp_val,
                        realized_pnl=csp_pnl, exit_reason='bear_liquidation',
                        S_at_close=S,
                    ))
                    liq_remove.append(tk)

            for tk in liq_remove:
                if tk in positions:
                    del positions[tk]

        # ---- 2) Sell CCs on long_shares (always allowed, even in bear) ----
        for tk, pos in list(positions.items()):
            if pos.side != 'long_shares':
                continue
            if pos.expiry > dt and pos.strike > 0:
                continue
            S = closes_by_ticker.get(tk, {}).get(dt)
            sigma = sigmas_by_ticker.get(tk, {}).get(dt)
            if S is None or sigma is None or np.isnan(S) or np.isnan(sigma):
                continue
            expiry = find_expiry(dt)
            if expiry is None:
                continue
            T = (expiry - dt).days / 365.0
            K = strike_from_delta(S, T, sigma, CALL_DELTA, kind="call")
            premium = bs_price(S, K, T, sigma, kind="call")
            slip = slippage(premium)
            if premium - slip <= 0.05:
                continue
            credit = (premium - slip) * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
            cash += credit
            positions[tk] = Position(
                ticker=tk, side='short_call', strike=K, expiry=expiry,
                open_date=dt, open_price=premium - slip,
                contracts=pos.contracts, share_basis=pos.share_basis,
            )
            stats[tk]['cc_opened'] += 1

        # ---- 3) Open new CSPs (bear-gated) ----

        # BEAR GATE: blocks all new CSPs in bear regime for these modes
        if bear_mode in ("halt", "liquidate", "liquidate_200", "liq_csp_only", "liq_50pct") and is_bear:
            bear_days_blocked += 1
            # Still compute equity for the curve
            equity = cash
            for tk, pos in positions.items():
                S = closes_by_ticker.get(tk, {}).get(dt)
                if S is None or np.isnan(S):
                    continue
                T = max((pos.expiry - dt).days, 0) / 365.0
                if pos.side == 'short_put':
                    opt = bs_price(S, pos.strike, T, sigmas_by_ticker.get(tk, {}).get(dt, 0.25), kind="put")
                    equity += (pos.open_price - opt) * 100 * pos.contracts
                elif pos.side == 'long_shares':
                    equity += (S - pos.share_basis) * 100 * pos.contracts
                elif pos.side == 'short_call':
                    sigma_t = sigmas_by_ticker.get(tk, {}).get(dt, 0.25)
                    opt = bs_price(S, pos.strike, T, sigma_t, kind="call")
                    equity += (S - pos.share_basis) * 100 * pos.contracts
                    equity += (pos.open_price - opt) * 100 * pos.contracts
            equity_series.append({'date': dt, 'equity': equity, 'cash': cash,
                                  'n_positions': len(positions),
                                  'reserved_collateral': reserved_collateral(positions),
                                  'is_bear': is_bear})
            continue  # Skip CSP opening entirely

        # Compute portfolio equity for sizing
        equity = cash
        for tk, pos in positions.items():
            S = closes_by_ticker.get(tk, {}).get(dt)
            if S is None or np.isnan(S):
                continue
            T = max((pos.expiry - dt).days, 0) / 365.0
            if pos.side == 'short_put':
                opt = bs_price(S, pos.strike, T, sigmas_by_ticker.get(tk, {}).get(dt, 0.25), kind="put")
                equity += (pos.open_price - opt) * 100 * pos.contracts
            elif pos.side == 'long_shares':
                equity += (S - pos.share_basis) * 100 * pos.contracts
            elif pos.side == 'short_call':
                sigma_t = sigmas_by_ticker.get(tk, {}).get(dt, 0.25)
                opt = bs_price(S, pos.strike, T, sigma_t, kind="call")
                equity += (S - pos.share_basis) * 100 * pos.contracts
                equity += (pos.open_price - opt) * 100 * pos.contracts

        equity_series.append({'date': dt, 'equity': equity, 'cash': cash,
                              'n_positions': len(positions),
                              'reserved_collateral': reserved_collateral(positions),
                              'is_bear': is_bear})

        if equity <= 0:
            log.warning(f"[{dt.date()}] Portfolio equity <= 0 (${equity:.0f}), stopping.")
            break

        # VIX gate
        if not np.isnan(vix) and vix > VIX_MAX:
            continue

        # Available cash for new CSPs
        avail = available_cash(cash, positions)
        if avail <= 0:
            continue

        # Max notional check
        total_notional = 0.0
        for tk, pos in positions.items():
            S = closes_by_ticker.get(tk, {}).get(dt)
            if S is not None and not np.isnan(S):
                total_notional += position_notional(pos, S)
        if total_notional > MAX_NOTIONAL_PCT * equity:
            continue

        # Find candidates not already in portfolio
        candidates = []
        for tk in TICKERS:
            if tk in positions:
                continue
            S = closes_by_ticker.get(tk, {}).get(dt)
            sigma = sigmas_by_ticker.get(tk, {}).get(dt)
            if S is None or sigma is None or np.isnan(S) or np.isnan(sigma) or sigma <= 0:
                continue
            candidates.append((tk, S, sigma))

        candidates.sort(key=lambda x: x[2], reverse=True)

        # BEAR GATE: "half" mode reduces per-name allocation by 50%
        bear_alloc_mult = 0.5 if (bear_mode == "half" and is_bear) else 1.0

        for tk, S, sigma in candidates:
            avail = available_cash(cash, positions)
            if avail <= 0:
                break

            expiry = find_expiry(dt)
            if expiry is None:
                continue
            T = (expiry - dt).days / 365.0
            K = strike_from_delta(S, T, sigma, PUT_DELTA, kind="put")
            premium = bs_price(S, K, T, sigma, kind="put")
            slip = slippage(premium)
            net_prem = premium - slip
            if net_prem <= 0.05:
                continue

            # Per-name allocation cap (reduced in bear "half" mode)
            max_alloc = MAX_PER_NAME_PCT * equity * bear_alloc_mult
            contracts = int(max_alloc // (K * 100))
            if contracts < 1:
                continue

            collateral_needed = K * 100 * contracts
            if collateral_needed > avail:
                contracts = int(avail // (K * 100))
                if contracts < 1:
                    continue
                collateral_needed = K * 100 * contracts

            new_total = total_notional + collateral_needed
            if new_total > MAX_NOTIONAL_PCT * equity:
                remaining = MAX_NOTIONAL_PCT * equity - total_notional
                contracts = int(remaining // (K * 100))
                if contracts < 1:
                    continue
                collateral_needed = K * 100 * contracts

            credit = net_prem * 100 * contracts - COST_PER_CONTRACT * contracts
            cash += credit
            positions[tk] = Position(
                ticker=tk, side='short_put', strike=K, expiry=expiry,
                open_date=dt, open_price=net_prem, contracts=contracts,
                share_basis=0.0,
            )
            stats[tk]['csp_opened'] += 1
            total_notional += collateral_needed

    eq_df = pd.DataFrame(equity_series)
    led_df = pd.DataFrame([{
        'ticker': t.ticker, 'kind': t.kind, 'open_date': t.open_date,
        'close_date': t.close_date, 'strike': t.strike, 'contracts': t.contracts,
        'premium_open': t.premium_open, 'premium_close': t.premium_close,
        'realized_pnl': t.realized_pnl, 'exit_reason': t.exit_reason,
        'S_at_close': t.S_at_close,
    } for t in ledger])

    return {
        'equity_curve': eq_df,
        'ledger': led_df,
        'stats': stats,
        'force_close_count': force_close_count,
        'bear_days_blocked': bear_days_blocked,
        'final_cash': cash,
        'final_equity': eq_df['equity'].iloc[-1] if not eq_df.empty else cash,
    }


# ----------------------------- data loading ---------------------------------
def load_data():
    """Load data for the 15-name basket, including SPY SMA50."""
    log.info("Loading universe metadata ...")
    universe = pd.read_parquet(CACHE / "universe_expanded.parquet")

    log.info("Loading original prices ...")
    p1 = pd.read_parquet(CACHE / "prices.parquet")[["ticker", "date", "close"]].copy()
    p1["date"] = pd.to_datetime(p1["date"], utc=False)
    if p1["date"].dt.tz is not None:
        p1["date"] = p1["date"].dt.tz_localize(None)

    log.info("Loading expanded prices ...")
    p2 = pd.read_parquet(CACHE / "prices_expanded.parquet")
    p2 = p2.rename(columns={"Close": "close"})[["ticker", "date", "close"]].copy()
    p2["date"] = pd.to_datetime(p2["date"], utc=False)
    if p2["date"].dt.tz is not None:
        p2["date"] = p2["date"].dt.tz_localize(None)

    prices = pd.concat([p1, p2], ignore_index=True)
    prices = prices[prices["ticker"].isin(TICKERS)].copy()
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices = prices.dropna(subset=["close"])
    prices = prices[prices["close"] > 0]
    prices = prices[prices["date"] >= START_DATE].copy()

    log.info("Loading macro/VIX ...")
    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"], utc=False)
    if macro["date"].dt.tz is not None:
        macro["date"] = macro["date"].dt.tz_localize(None)

    # Load SPY for regime classification AND SMA gate
    sec = pd.read_parquet(CACHE / "sector_etfs.parquet")
    spy = sec[sec["ticker"] == "SPY"][["date", "close"]].copy()
    spy.columns = ["date", "spy_close"]
    spy["date"] = pd.to_datetime(spy["date"], utc=False)
    if spy["date"].dt.tz is not None:
        spy["date"] = spy["date"].dt.tz_localize(None)
    spy = spy.sort_values("date").drop_duplicates("date", keep="last")

    # Compute SPY 50-day SMA
    spy_indexed = spy.set_index("date").sort_index()
    spy_indexed["sma50"] = spy_indexed["spy_close"].rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
    spy_indexed["sma200"] = spy_indexed["spy_close"].rolling(200, min_periods=200).mean()

    spy_closes = spy_indexed["spy_close"].to_dict()
    spy_sma50 = spy_indexed["sma50"].dropna().to_dict()
    spy_sma200 = spy_indexed["sma200"].dropna().to_dict()

    log.info(f"SPY SMA50 computed: {len(spy_sma50)} dates with valid SMA")
    log.info(f"SPY SMA200 computed: {len(spy_sma200)} dates with valid SMA")

    prices = prices.merge(macro, on="date", how="left")
    prices["vix"] = prices["vix"].ffill().fillna(20.0)

    log.info("Computing 20-day realized vol ...")
    prices["log_ret"] = prices.groupby("ticker")["close"].transform(
        lambda x: np.log(x / x.shift(1)))
    prices["sigma"] = prices.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(20, min_periods=15).std() * np.sqrt(252))
    prices["sigma"] = prices["sigma"].clip(lower=0.05, upper=2.0)
    prices = prices.dropna(subset=["sigma"]).reset_index(drop=True)

    all_dates = sorted(prices["date"].unique())
    closes_by_ticker = {}
    sigmas_by_ticker = {}
    for tk in TICKERS:
        tk_data = prices[prices["ticker"] == tk].set_index("date")
        closes_by_ticker[tk] = tk_data["close"].to_dict()
        sigmas_by_ticker[tk] = tk_data["sigma"].to_dict()

    vix_series = prices.drop_duplicates("date").set_index("date")["vix"].to_dict()

    # SPY returns for regime classification
    spy_ret = spy_indexed["spy_close"].pct_change()

    tickers_present = [tk for tk in TICKERS if tk in closes_by_ticker and len(closes_by_ticker[tk]) > 0]
    log.info(f"Loaded {len(tickers_present)} tickers, {len(all_dates)} dates, "
             f"range {all_dates[0].date()} to {all_dates[-1].date()}")

    return closes_by_ticker, sigmas_by_ticker, vix_series, all_dates, spy_ret, spy_closes, spy_sma50, spy_sma200


# ----------------------------- metrics --------------------------------------
def compute_metrics(eq_df):
    eq = eq_df.set_index("date")["equity"].astype(float)
    rets = eq.pct_change().dropna()
    if len(rets) < 2:
        return {}
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    mu, sd = rets.mean(), rets.std()
    downside = rets[rets < 0].std()
    sharpe = (mu / sd) * math.sqrt(TRADING_DAYS) if sd > 0 else float("nan")
    sortino = (mu / downside) * math.sqrt(TRADING_DAYS) if downside and downside > 0 else float("nan")
    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = float(dd.min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    wr = float((rets > 0).mean())
    pos = rets[rets > 0].sum()
    neg = abs(rets[rets < 0].sum())
    pf = float(pos / neg) if neg > 0 else float("nan")
    pnl = eq.diff().dropna()
    tot = pnl.sum()
    day_conc = float(pnl.max() / tot) if tot > 0 else 1.0
    return {
        'n_days': int(len(rets)),
        'years': float(years),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd),
        'calmar': float(calmar),
        'win_rate': wr,
        'profit_factor': pf,
        'total_return_pct': float((eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0),
        'final_equity': float(eq.iloc[-1]),
        'initial_equity': float(eq.iloc[0]),
        'day_concentration': day_conc,
    }


def regime_analysis(eq_df, spy_ret):
    """HC #428 R1: regime-stratified Sharpe."""
    eq = eq_df.set_index("date")["equity"].astype(float)
    rets = eq.pct_change().dropna()

    spy_aligned = spy_ret.reindex(rets.index)
    valid = spy_aligned.notna()
    rets = rets[valid]
    spy_aligned = spy_aligned[valid]

    if len(rets) < 40:
        return {'hc428_r1_pass': False, 'regime_gap': float('nan'),
                'sharpe_bull': None, 'sharpe_bear': None, 'sharpe_sideways': None,
                'n_bull': 0, 'n_bear': 0, 'n_sideways': 0}

    bull = spy_aligned > FLAT_BAND
    bear = spy_aligned < -FLAT_BAND
    sideways = ~bull & ~bear

    def ann_sh(s):
        if len(s) < 2 or s.std() == 0:
            return float('nan')
        return (s.mean() / s.std()) * math.sqrt(TRADING_DAYS)

    sh_bull = ann_sh(rets[bull])
    sh_bear = ann_sh(rets[bear])
    sh_side = ann_sh(rets[sideways])

    denom = max(abs(sh_bull) if np.isfinite(sh_bull) else 0,
                abs(sh_bear) if np.isfinite(sh_bear) else 0, 1e-9)
    gap = abs((sh_bull if np.isfinite(sh_bull) else 0.0) -
              (sh_bear if np.isfinite(sh_bear) else 0.0)) / denom

    return {
        'n_bull': int(bull.sum()),
        'n_bear': int(bear.sum()),
        'n_sideways': int(sideways.sum()),
        'sharpe_bull': float(sh_bull) if np.isfinite(sh_bull) else None,
        'sharpe_bear': float(sh_bear) if np.isfinite(sh_bear) else None,
        'sharpe_sideways': float(sh_side) if np.isfinite(sh_side) else None,
        'regime_gap': float(gap),
        'hc428_r1_pass': bool(gap <= 0.50),
    }


# ----------------------------- main -----------------------------------------
def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("WHEEL REGIME-GATED PORTFOLIO BACKTEST")
    log.info(f"Basket: {', '.join(TICKERS)}")
    log.info(f"Capital: ${STARTING_CASH:,.0f}")
    log.info(f"Bear gate: SPY {SMA_PERIOD}-day SMA")
    log.info(f"Modes: none (baseline), halt, half, liquidate, liquidate_200")
    log.info("=" * 70)

    closes, sigmas, vix, all_dates, spy_ret, spy_closes, spy_sma50, spy_sma200 = load_data()

    modes = ["none", "liquidate", "liq_csp_only", "liq_50pct"]
    all_results = {}

    for mode in modes:
        log.info(f"\n{'='*50}")
        log.info(f"Running mode: {mode.upper()}")
        log.info(f"{'='*50}")

        result = run_portfolio_wheel(
            closes, sigmas, vix, all_dates,
            spy_closes, spy_sma50,
            bear_mode=mode,
            spy_sma200=spy_sma200,
        )

        eq_df = result['equity_curve']
        led_df = result['ledger']
        metrics = compute_metrics(eq_df)
        regime = regime_analysis(eq_df, spy_ret)

        total_csp = sum(s['csp_opened'] for s in result['stats'].values())
        total_cc = sum(s['cc_opened'] for s in result['stats'].values())
        total_assign = sum(s['assignments'] for s in result['stats'].values())
        total_called = sum(s['call_aways'] for s in result['stats'].values())
        total_fc = result['force_close_count']

        log.info(f"\n--- {mode.upper()} RESULTS ---")
        log.info(f"Final equity:     ${metrics.get('final_equity', 0):,.0f}")
        log.info(f"CAGR:             {metrics.get('cagr', 0)*100:.2f}%")
        log.info(f"Sharpe:           {metrics.get('sharpe', 0):.2f}")
        log.info(f"Sortino:          {metrics.get('sortino', 0):.2f}")
        log.info(f"Max Drawdown:     {metrics.get('max_dd', 0)*100:.1f}%")
        log.info(f"Calmar:           {metrics.get('calmar', 0):.2f}")
        log.info(f"Win Rate:         {metrics.get('win_rate', 0)*100:.1f}%")
        log.info(f"Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
        log.info(f"Day Concentration:{metrics.get('day_concentration', 0):.3f}")
        log.info(f"Total Trades:     {len(led_df) if not led_df.empty else 0}")
        log.info(f"Bear days blocked:{result['bear_days_blocked']}")
        log.info(f"Force closes:     {total_fc}")

        log.info(f"\nRegime Analysis (HC #428 R1):")
        log.info(f"  Bull Sharpe:  {regime['sharpe_bull']}")
        log.info(f"  Bear Sharpe:  {regime['sharpe_bear']}")
        log.info(f"  Sideways:     {regime['sharpe_sideways']}")
        log.info(f"  Regime gap:   {regime['regime_gap']:.3f}")
        log.info(f"  HC #428 R1:   {'PASS' if regime['hc428_r1_pass'] else 'FAIL'}")

        # Save per-mode outputs
        eq_df.to_parquet(OUT_DIR / f"equity_curve_{mode}.parquet", index=False)
        if not led_df.empty:
            led_df.to_parquet(OUT_DIR / f"trade_ledger_{mode}.parquet", index=False)

        all_results[mode] = {
            'metrics': metrics,
            'regime_analysis': regime,
            'activity': {
                'csp_opened': total_csp,
                'cc_opened': total_cc,
                'assignments': total_assign,
                'called_away': total_called,
                'force_closes': total_fc,
                'total_trades': len(led_df) if not led_df.empty else 0,
                'bear_days_blocked': result['bear_days_blocked'],
            },
            'gates': {
                'sharpe_ge_1.0': bool(np.isfinite(metrics.get('sharpe', float('nan'))) and metrics['sharpe'] >= 1.0),
                'calmar_ge_1.5': bool(np.isfinite(metrics.get('calmar', float('nan'))) and metrics['calmar'] >= 1.5),
                'regime_gap_le_0.50': regime['hc428_r1_pass'],
                'day_conc_le_0.70': bool(metrics.get('day_concentration', 1.0) <= 0.70),
            },
        }

    # ---- Comparison summary ----
    log.info(f"\n{'='*70}")
    log.info("COMPARISON SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"{'Metric':<20} | {'NONE':>12} | {'HALT':>12} | {'HALF':>12} | {'LIQUIDATE':>12} | {'LIQ_200':>12}")
    log.info("-" * 90)

    compare_keys = [
        ('CAGR', 'cagr', lambda v: f"{v*100:.2f}%"),
        ('Sharpe', 'sharpe', lambda v: f"{v:.2f}"),
        ('Sortino', 'sortino', lambda v: f"{v:.2f}"),
        ('Max DD', 'max_dd', lambda v: f"{v*100:.1f}%"),
        ('Calmar', 'calmar', lambda v: f"{v:.2f}"),
        ('Win Rate', 'win_rate', lambda v: f"{v*100:.1f}%"),
        ('PF', 'profit_factor', lambda v: f"{v:.2f}"),
        ('Final Equity', 'final_equity', lambda v: f"${v:,.0f}"),
        ('Day Conc', 'day_concentration', lambda v: f"{v:.3f}"),
    ]

    for label, key, fmt in compare_keys:
        vals = []
        for m in modes:
            v = all_results[m]['metrics'].get(key, float('nan'))
            vals.append(fmt(v) if np.isfinite(v) else 'N/A')
        log.info(f"{label:<20} | " + " | ".join(f"{v:>12}" for v in vals))

    log.info("-" * 90)
    for label, key in [('Bull Sharpe', 'sharpe_bull'), ('Bear Sharpe', 'sharpe_bear'),
                        ('Sideways Sharpe', 'sharpe_sideways'), ('Regime Gap', 'regime_gap')]:
        vals = []
        for m in modes:
            v = all_results[m]['regime_analysis'].get(key)
            if v is not None and np.isfinite(v):
                vals.append(f"{v:.3f}")
            else:
                vals.append('N/A')
        log.info(f"{label:<20} | " + " | ".join(f"{v:>12}" for v in vals))

    log.info("-" * 90)
    for m in modes:
        passes = all_results[m]['gates']
        pass_str = "PASS" if all(passes.values()) else "FAIL"
        log.info(f"{'HC428 R1 (' + m + ')':<24} | Regime gap: {'PASS' if passes['regime_gap_le_0.50'] else 'FAIL'} | "
                 f"Deploy: {pass_str}")

    # Save comparison JSON
    comparison = {
        'generated': datetime.now().isoformat(),
        'description': 'Regime-gated wheel portfolio: SPY 50d SMA bear gate comparison',
        'basket': TICKERS,
        'sma_period': SMA_PERIOD,
        'bear_modes': {
            'none': 'No bear gate (original baseline)',
            'liquidate': 'Liquidate all positions when SPY < 50d SMA',
            'liq_csp_only': 'Close CSPs only in bear, keep shares/CCs',
            'liq_50pct': 'Liquidate 50% of positions in bear',
        },
        'results': all_results,
    }
    with open(OUT_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    elapsed = time.time() - t0
    log.info(f"\nDone in {elapsed:.1f}s. Results saved.")
    return all_results


if __name__ == "__main__":
    main()
