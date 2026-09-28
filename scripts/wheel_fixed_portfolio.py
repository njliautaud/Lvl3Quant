#!/usr/bin/env python3
"""
wheel_fixed_portfolio.py — Fixed-leverage portfolio wheel backtest.

Fixes from wheel_expanded_sweep.py:
  1. Cash can NEVER go negative. On assignment, if buying 100 shares
     would exceed available cash, force-close (assign at market, sell
     immediately, realize the loss).
  2. max_notional_pct parameter (default 0.95) — never commit more
     than 95% of current equity to a single position.
  3. Collateral reservation: when a CSP is open, its collateral
     (strike * 100 * contracts) is reserved and unavailable for new
     positions.
  4. Portfolio mode: all 15 names share the same $100K capital pool.
  5. Per-name capital allocation: $100K / 15 ~ $6.7K max per name.

Diversification fix:
  - Replaced SO (Utilities, correlated with AEP) with DLR (REIT)
  - Replaced SWKS (Technology, 4th tech name) with COST (Consumer Defensive)
  - Effective independent bets: ~12 (up from ~10)

Regime-stratified analysis per HC #428 R1.
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
OUT_DIR = ROOT / "output" / "wheel_fixed_portfolio"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "wheel_fixed_portfolio.log"

logging.basicConfig(
    format='%(asctime)s [WHEEL-FIX] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE)),
    ],
)
log = logging.getLogger('WHEEL-FIX')

# ----------------------------- config ---------------------------------------
# Fixed 15-name basket with improved diversification
BASKET = {
    # Ticker: (sector, notes)
    'AEP':  ('Utilities',),
    'DLR':  ('Real Estate',),         # REPLACED SO (Utilities, too correlated with AEP)
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
    'COST': ('Consumer Defensive',),  # REPLACED SWKS (Technology, reduced tech concentration)
}

TICKERS = sorted(BASKET.keys())
N_NAMES = len(TICKERS)

START_DATE = pd.Timestamp("2019-01-01")
STARTING_CASH = 100_000.0
PER_NAME_ALLOC = STARTING_CASH / N_NAMES  # ~$6,667
TRADING_DAYS = 252
RISK_FREE = 0.04

PUT_DELTA = 0.25
CALL_DELTA = 0.30
DTE_MIN = 25
DTE_MAX = 35
DTE_TARGET = 30
PROFIT_TAKE = 0.50
VIX_MAX = 35.0

MAX_NOTIONAL_PCT = 0.95       # Never commit > 95% of equity to a single position
MAX_PER_NAME_PCT = 1.0 / N_NAMES + 0.02  # ~8.7% max per name (slight buffer over equal weight)

COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03

FLAT_BAND = 0.0025  # SPY daily return +/-0.25% = flat day


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
    """Total collateral reserved by open CSPs."""
    total = 0.0
    for tk, pos in positions.items():
        if pos.side == 'short_put':
            total += pos.strike * 100 * pos.contracts
    return total


def available_cash(cash: float, positions: dict) -> float:
    """Cash available for new positions after collateral reservations."""
    return max(0.0, cash - reserved_collateral(positions))


def position_notional(pos: Position, S: float) -> float:
    """Current notional value of a position."""
    if pos.side == 'short_put':
        return pos.strike * 100 * pos.contracts
    elif pos.side in ('long_shares', 'short_call'):
        return S * 100 * pos.contracts
    return 0.0


def run_portfolio_wheel(closes_by_ticker, sigmas_by_ticker,
                        vix_series, all_dates):
    """
    Run portfolio wheel across all 15 names sharing capital.

    Returns dict with equity curve, trade ledger, stats.
    """
    cash = STARTING_CASH
    positions = {}  # ticker -> Position
    ledger = []     # list of TradeRecord
    equity_series = []
    force_close_count = 0

    stats = {tk: {'csp_opened': 0, 'cc_opened': 0, 'assignments': 0,
                   'call_aways': 0, 'force_closes': 0} for tk in TICKERS}

    for dt in all_dates:
        vix = vix_series.get(dt, 20.0)

        # ---- 1) Process existing positions ----
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
                    # Profit-take: buy back
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
                        # ASSIGNMENT -- key leverage fix
                        assignment_cost = pos.strike * 100 * pos.contracts

                        if assignment_cost > cash:
                            # FORCE CLOSE: can't afford the shares.
                            # Assign at strike, immediately sell at market price.
                            # Net loss = (strike - S) * 100 * contracts - premium kept
                            loss_per_share = pos.strike - S
                            total_loss = loss_per_share * 100 * pos.contracts
                            premium_kept = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                            realized = premium_kept - total_loss
                            cash += realized  # This is negative net (a loss)
                            # Cash floor: never go below zero
                            if cash < 0:
                                log.warning(f"[{dt.date()}] {tk} force-close pushed cash to ${cash:.0f}, clamping to 0")
                                realized += cash  # Adjust realized to account for the floor
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
                            log.info(f"[{dt.date()}] {tk} FORCE CLOSE: assignment cost ${assignment_cost:,.0f} "
                                     f"> cash ${cash:,.0f}, loss ${-realized:,.0f}")
                        else:
                            # Normal assignment
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
                            # Transition to long_shares
                            positions[tk] = Position(
                                ticker=tk, side='long_shares', strike=basis,
                                expiry=dt, open_date=dt, open_price=basis,
                                contracts=pos.contracts, share_basis=basis,
                            )
                    else:
                        # Expired worthless
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
                    # Revert to long_shares
                    positions[tk] = Position(
                        ticker=tk, side='long_shares', strike=pos.share_basis,
                        expiry=dt, open_date=dt, open_price=pos.share_basis,
                        contracts=pos.contracts, share_basis=pos.share_basis,
                    )

                elif is_expiry:
                    if S > pos.strike:
                        # Called away
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
                        # CC expired worthless, keep shares
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

        # ---- 2) Sell CCs on long_shares ----
        for tk, pos in list(positions.items()):
            if pos.side != 'long_shares':
                continue
            if pos.expiry > dt and pos.strike > 0:
                continue  # Still has an active CC (shouldn't happen, but guard)
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

        # ---- 3) Open new CSPs ----
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
                              'reserved_collateral': reserved_collateral(positions)})

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

        # Don't exceed MAX_NOTIONAL_PCT of equity across all positions
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

        # Sort by IV (prefer higher IV for better premiums)
        candidates.sort(key=lambda x: x[2], reverse=True)

        for tk, S, sigma in candidates:
            # Re-check available cash (it decreases as we open positions)
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

            # Per-name allocation cap
            max_alloc = MAX_PER_NAME_PCT * equity
            contracts = int(max_alloc // (K * 100))
            if contracts < 1:
                continue

            # Collateral check: need K * 100 * contracts reserved from available cash
            collateral_needed = K * 100 * contracts
            if collateral_needed > avail:
                contracts = int(avail // (K * 100))
                if contracts < 1:
                    continue
                collateral_needed = K * 100 * contracts

            # Max notional check across portfolio
            new_total = total_notional + collateral_needed
            if new_total > MAX_NOTIONAL_PCT * equity:
                # Scale down
                remaining = MAX_NOTIONAL_PCT * equity - total_notional
                contracts = int(remaining // (K * 100))
                if contracts < 1:
                    continue
                collateral_needed = K * 100 * contracts

            # Open the CSP
            credit = net_prem * 100 * contracts - COST_PER_CONTRACT * contracts
            cash += credit
            positions[tk] = Position(
                ticker=tk, side='short_put', strike=K, expiry=expiry,
                open_date=dt, open_price=net_prem, contracts=contracts,
                share_basis=0.0,
            )
            stats[tk]['csp_opened'] += 1
            total_notional += collateral_needed

    # Build result dataframes
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
        'final_cash': cash,
        'final_equity': eq_df['equity'].iloc[-1] if not eq_df.empty else cash,
    }


# ----------------------------- data loading ---------------------------------
def load_data():
    """Load and prepare data for the 15-name basket."""
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

    # Load SPY for regime classification
    sec = pd.read_parquet(CACHE / "sector_etfs.parquet")
    spy = sec[sec["ticker"] == "SPY"][["date", "close"]].copy()
    spy.columns = ["date", "spy_close"]
    spy["date"] = pd.to_datetime(spy["date"], utc=False)
    if spy["date"].dt.tz is not None:
        spy["date"] = spy["date"].dt.tz_localize(None)
    spy = spy.sort_values("date").drop_duplicates("date", keep="last")

    prices = prices.merge(macro, on="date", how="left")
    prices["vix"] = prices["vix"].ffill().fillna(20.0)

    # Compute 20-day realized vol per ticker
    log.info("Computing 20-day realized vol ...")
    prices["log_ret"] = prices.groupby("ticker")["close"].transform(
        lambda x: np.log(x / x.shift(1)))
    prices["sigma"] = prices.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(20, min_periods=15).std() * np.sqrt(252))
    prices["sigma"] = prices["sigma"].clip(lower=0.05, upper=2.0)
    prices = prices.dropna(subset=["sigma"]).reset_index(drop=True)

    # Build lookup dicts
    all_dates = sorted(prices["date"].unique())
    closes_by_ticker = {}
    sigmas_by_ticker = {}
    for tk in TICKERS:
        tk_data = prices[prices["ticker"] == tk].set_index("date")
        closes_by_ticker[tk] = tk_data["close"].to_dict()
        sigmas_by_ticker[tk] = tk_data["sigma"].to_dict()

    vix_series = prices.drop_duplicates("date").set_index("date")["vix"].to_dict()

    # SPY returns for regime
    spy_df = spy.set_index("date")
    spy_ret = spy_df["spy_close"].pct_change()

    tickers_present = [tk for tk in TICKERS if tk in closes_by_ticker and len(closes_by_ticker[tk]) > 0]
    log.info(f"Loaded {len(tickers_present)} tickers, {len(all_dates)} dates, "
             f"range {all_dates[0].date()} to {all_dates[-1].date()}")
    for tk in TICKERS:
        n = len(closes_by_ticker.get(tk, {}))
        log.info(f"  {tk}: {n} trading days")

    return closes_by_ticker, sigmas_by_ticker, vix_series, all_dates, spy_ret


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
    # Day concentration
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

    # Align SPY returns
    spy_aligned = spy_ret.reindex(rets.index)
    valid = spy_aligned.notna()
    rets = rets[valid]
    spy_aligned = spy_aligned[valid]

    if len(rets) < 40:
        return {'hc428_r1_pass': False, 'regime_gap': float('nan'),
                'sharpe_bull': None, 'sharpe_bear': None, 'sharpe_sideways': None,
                'n_bull': 0, 'n_bear': 0, 'n_sideways': 0}

    # Classify days
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
    log.info("WHEEL FIXED PORTFOLIO BACKTEST")
    log.info(f"Basket: {', '.join(TICKERS)}")
    log.info(f"Capital: ${STARTING_CASH:,.0f}, per-name max: ${PER_NAME_ALLOC:,.0f}")
    log.info(f"Leverage fix: max_notional_pct={MAX_NOTIONAL_PCT}, collateral reservation ON")
    log.info("=" * 70)

    closes, sigmas, vix, all_dates, spy_ret = load_data()

    log.info(f"\nRunning portfolio wheel ...")
    result = run_portfolio_wheel(closes, sigmas, vix, all_dates)

    eq_df = result['equity_curve']
    led_df = result['ledger']

    # Compute metrics
    metrics = compute_metrics(eq_df)
    regime = regime_analysis(eq_df, spy_ret)

    # Per-ticker stats
    total_csp = sum(s['csp_opened'] for s in result['stats'].values())
    total_cc = sum(s['cc_opened'] for s in result['stats'].values())
    total_assign = sum(s['assignments'] for s in result['stats'].values())
    total_called = sum(s['call_aways'] for s in result['stats'].values())
    total_fc = result['force_close_count']

    # Print results
    log.info("\n" + "=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)
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
    log.info(f"CSP opened:       {total_csp}")
    log.info(f"CC opened:        {total_cc}")
    log.info(f"Assignments:      {total_assign}")
    log.info(f"Called away:      {total_called}")
    log.info(f"Force closes:     {total_fc}")

    log.info(f"\nRegime Analysis (HC #428 R1):")
    log.info(f"  Bull days:    {regime['n_bull']:>5} | Sharpe: {regime['sharpe_bull']}")
    log.info(f"  Bear days:    {regime['n_bear']:>5} | Sharpe: {regime['sharpe_bear']}")
    log.info(f"  Sideways:     {regime['n_sideways']:>5} | Sharpe: {regime['sharpe_sideways']}")
    log.info(f"  Regime gap:   {regime['regime_gap']:.3f}")
    log.info(f"  HC #428 R1:   {'PASS' if regime['hc428_r1_pass'] else 'FAIL'}")

    # Per-ticker breakdown
    log.info(f"\nPer-ticker breakdown:")
    log.info(f"{'Ticker':>6} | {'CSP':>4} | {'CC':>4} | {'Assign':>6} | {'Called':>6} | {'FClose':>6}")
    log.info("-" * 50)
    for tk in TICKERS:
        s = result['stats'][tk]
        log.info(f"{tk:>6} | {s['csp_opened']:>4} | {s['cc_opened']:>4} | "
                 f"{s['assignments']:>6} | {s['call_aways']:>6} | {s['force_closes']:>6}")

    # Trade PnL by ticker
    if not led_df.empty:
        log.info(f"\nPnL by ticker:")
        by_tk = led_df.groupby('ticker')['realized_pnl'].agg(['sum', 'count', 'mean'])
        by_tk = by_tk.sort_values('sum', ascending=False)
        log.info(f"{'Ticker':>6} | {'Total PnL':>12} | {'Trades':>6} | {'Avg PnL':>10}")
        log.info("-" * 45)
        for tk, row in by_tk.iterrows():
            log.info(f"{tk:>6} | ${row['sum']:>11,.0f} | {int(row['count']):>6} | ${row['mean']:>9,.0f}")

    # Gates
    sharpe_pass = np.isfinite(metrics.get('sharpe', float('nan'))) and metrics['sharpe'] >= 1.0
    calmar_pass = np.isfinite(metrics.get('calmar', float('nan'))) and metrics['calmar'] >= 1.5
    regime_pass = regime['hc428_r1_pass']
    dayconc_pass = metrics.get('day_concentration', 1.0) <= 0.70
    all_pass = sharpe_pass and calmar_pass and regime_pass and dayconc_pass

    log.info(f"\nDeploy Gates:")
    log.info(f"  Sharpe >= 1.0:     {'PASS' if sharpe_pass else 'FAIL'} ({metrics.get('sharpe', 0):.2f})")
    log.info(f"  Calmar >= 1.5:     {'PASS' if calmar_pass else 'FAIL'} ({metrics.get('calmar', 0):.2f})")
    log.info(f"  Regime gap <= 0.50:{'PASS' if regime_pass else 'FAIL'} ({regime['regime_gap']:.3f})")
    log.info(f"  DayConc <= 0.70:   {'PASS' if dayconc_pass else 'FAIL'} ({metrics.get('day_concentration', 0):.3f})")
    log.info(f"  DEPLOY READY:      {'YES' if all_pass else 'NO'}")

    # Save outputs
    log.info(f"\nSaving results ...")
    eq_df.to_parquet(OUT_DIR / "equity_curve.parquet", index=False)
    eq_df.to_csv(OUT_DIR / "equity_curve.csv", index=False)
    if not led_df.empty:
        led_df.to_parquet(OUT_DIR / "trade_ledger.parquet", index=False)
        led_df.to_csv(OUT_DIR / "trade_ledger.csv", index=False)

    summary = {
        'generated': datetime.now().isoformat(),
        'basket': TICKERS,
        'basket_changes': {
            'removed': ['SO (Utilities, correlated with AEP)', 'SWKS (Technology, 4th tech name)'],
            'added': ['DLR (Real Estate, low correlation to tech/energy)', 'COST (Consumer Defensive)'],
        },
        'config': {
            'starting_cash': STARTING_CASH,
            'per_name_alloc': PER_NAME_ALLOC,
            'max_notional_pct': MAX_NOTIONAL_PCT,
            'max_per_name_pct': MAX_PER_NAME_PCT,
            'put_delta': PUT_DELTA,
            'call_delta': CALL_DELTA,
            'dte_target': DTE_TARGET,
            'profit_take': PROFIT_TAKE,
            'vix_gate': VIX_MAX,
        },
        'leverage_fixes': [
            'Cash can never go negative',
            'Force-close on assignment if insufficient cash',
            'Collateral reservation for open CSPs',
            'max_notional_pct=0.95 portfolio-wide cap',
            'Per-name allocation cap at 1/N + buffer',
        ],
        'metrics': metrics,
        'regime_analysis': regime,
        'activity': {
            'csp_opened': total_csp,
            'cc_opened': total_cc,
            'assignments': total_assign,
            'called_away': total_called,
            'force_closes': total_fc,
            'total_trades': len(led_df) if not led_df.empty else 0,
        },
        'gates': {
            'sharpe_ge_1.0': sharpe_pass,
            'calmar_ge_1.5': calmar_pass,
            'regime_gap_le_0.50': regime_pass,
            'day_conc_le_0.70': dayconc_pass,
            'deploy_ready': all_pass,
        },
        'per_ticker_stats': result['stats'],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    log.info(f"\nDone in {elapsed:.1f}s. Results -> {OUT_DIR}")
    return result, metrics, regime


if __name__ == "__main__":
    main()
