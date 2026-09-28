#!/usr/bin/env python3
"""
wheel_v6_full_wheel.py — Full Wheel Backtest with REAL Dolt Option Chain Data.

The FULL WHEEL cycle:
  1. Sell CSP → collect premium
  2. If assigned → hold shares AND sell covered calls on them
  3. If called away → go back to step 1
  4. If CSP not assigned → sell new CSP, repeat

PURE REAL PRICING: Uses actual bid/ask from DoltHub option chains.
No Black-Scholes anywhere in the P&L calculation.

Commission: $0 (Robinhood, HC #694)

Variants tested:
  - V6-Base:       25-delta CSP, 25-delta CC
  - V6-Aggressive: 25-delta CSP, 20-delta CC (more premium, caps gains more)
  - V6-Conservative: 25-delta CSP, 30-delta CC (less premium, keeps more upside)
  - V6-Dynamic:    25-delta CSP, dynamic CC delta (higher when vol high)

Travel income analysis: can you withdraw 1.5%/month and sustain NAV?
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
CHAINS_DIR = CACHE / "options_real" / "chains"
OUT_DIR = ROOT / "output" / "wheel_v6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CASH = 100_000.0
TRADING_DAYS = 252

# ── Strategy parameters ──
VIX_GATE_OPEN = 35.0       # Don't sell CSPs when VIX > this
DTE_MIN = 14               # Minimum DTE for new positions
DTE_MAX = 50               # Maximum DTE
DTE_TARGET = 30            # Preferred DTE
DTE_CLOSE_TRIGGER = 5      # Close early if DTE <= this
PROFIT_TAKE_FRAC = 0.50    # Close when 50% of max profit captured
MAX_CONCURRENT = 20        # Max simultaneous positions
SINGLE_NAME_CAP = 0.10     # Max 10% of equity in one name
SECTOR_CAP = 0.30          # Max 30% in one sector
DELTA_TOLERANCE = 0.12     # How close to target delta we accept
SHARE_STOP_LOSS = 0.15     # Stop loss on assigned shares

# Earnings buffer: don't sell options expiring within N days of earnings
EARNINGS_BUFFER_DAYS = 3


def load_all_chains() -> Dict[str, pd.DataFrame]:
    """Load all ticker chain files from Dolt cache."""
    chains = {}
    for f in sorted(CHAINS_DIR.iterdir()):
        if f.suffix == ".parquet":
            ticker = f.stem
            df = pd.read_parquet(f)
            df["date"] = pd.to_datetime(df["date"])
            df["expiration"] = pd.to_datetime(df["expiration"])
            for c in ["strike", "bid", "ask", "mid", "delta", "gamma"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[df["bid"].notna() & df["ask"].notna()].copy()
            if not df.empty:
                chains[ticker] = df
        elif f.suffix == ".EMPTY":
            pass  # skip empty markers
    return chains


def load_prices() -> pd.DataFrame:
    """Load daily prices."""
    px = pd.read_parquet(CACHE / "prices.parquet")
    px["date"] = pd.to_datetime(px["date"])
    return px


def load_macro() -> pd.DataFrame:
    macro = pd.read_parquet(CACHE / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"])
    return macro


def load_universe() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "universe.parquet")


def load_fundamentals() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "fundamentals.parquet")


def find_best_strike(chain_slice: pd.DataFrame, target_delta: float,
                     option_type: str) -> Optional[pd.Series]:
    """Find strike closest to target delta."""
    sub = chain_slice[chain_slice["type"] == option_type].copy()
    if sub.empty:
        return None
    sub["delta_abs"] = sub["delta"].abs()
    sub["delta_diff"] = (sub["delta_abs"] - target_delta).abs()
    valid = sub[sub["delta_diff"] <= DELTA_TOLERANCE]
    if valid.empty:
        return None
    best = valid.loc[valid["delta_diff"].idxmin()]
    # Require positive bid for selling
    if float(best["bid"]) <= 0:
        return None
    return best


def find_strike_by_value(chain_slice: pd.DataFrame, strike: float,
                         option_type: str) -> Optional[pd.Series]:
    """Find a specific strike in chain data."""
    sub = chain_slice[chain_slice["type"] == option_type].copy()
    if sub.empty:
        return None
    exact = sub[sub["strike"] == strike]
    if not exact.empty:
        return exact.iloc[0]
    sub["strike_diff"] = (sub["strike"] - strike).abs()
    nearest = sub[sub["strike_diff"] <= 5.0]
    if nearest.empty:
        return None
    return nearest.loc[nearest["strike_diff"].idxmin()]


# ── Position and Trade tracking ──

@dataclass
class WheelPosition:
    """Tracks a single ticker through the wheel lifecycle."""
    ticker: str
    sector: str
    state: str             # 'csp' | 'shares' | 'cc'
    open_date: pd.Timestamp
    expiry: pd.Timestamp
    strike: float
    contracts: int         # number of 100-share lots
    premium_received: float  # total $ premium received for current leg
    cost_basis: float      # per-share cost basis when holding shares
    share_purchase_price: float  # what we paid per share on assignment


@dataclass
class WheelTrade:
    """A completed round-trip trade (one leg of the wheel)."""
    ticker: str
    sector: str
    leg_type: str          # 'CSP' | 'CC'
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    expiry: pd.Timestamp
    strike: float
    contracts: int
    premium_received: float  # $ total premium from selling
    close_cost: float        # $ cost to close (buy back) or 0 if expired
    pnl: float               # net P&L for this leg
    close_reason: str        # 'expired_otm' | 'profit_take' | 'assigned' | 'called_away' | 'dte_close' | 'stop_loss'
    close_pricing: str       # 'real_chain' | 'intrinsic' | 'market_sell'
    entry_delta: float
    entry_bid: float
    shares_pnl: float = 0.0  # P&L from share price movement (for CC legs)


class WheelV6Engine:
    """Full wheel backtest engine using real Dolt chain data."""

    def __init__(self, put_delta: float = 0.25, call_delta: float = 0.25,
                 dynamic_cc: bool = False, starting_cash: float = STARTING_CASH):
        self.put_delta = put_delta
        self.call_delta = call_delta
        self.dynamic_cc = dynamic_cc
        self.starting_cash = starting_cash

        # State
        self.cash = starting_cash
        self.positions: Dict[str, WheelPosition] = {}
        self.trades: List[WheelTrade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []

        # Counters
        self.csp_opened = 0
        self.cc_opened = 0
        self.assignments = 0
        self.called_away = 0
        self.csp_expired_otm = 0
        self.cc_expired_otm = 0
        self.stop_losses = 0

        # Premium tracking
        self.total_csp_premium = 0.0
        self.total_cc_premium = 0.0

    def _get_cc_delta(self, vix: float) -> float:
        """Get covered call delta target. Dynamic mode adjusts based on VIX."""
        if not self.dynamic_cc:
            return self.call_delta
        # Dynamic: sell closer to money when vol is high (more premium)
        # VIX < 15: 30-delta (conservative)
        # VIX 15-25: 25-delta (standard)
        # VIX > 25: 20-delta (aggressive)
        if vix < 15:
            return 0.30
        elif vix < 25:
            return 0.25
        else:
            return 0.20

    def _equity(self, date_px: Dict[str, float]) -> float:
        """Compute total equity = cash + market value of held shares."""
        equity = self.cash
        for tk, pos in self.positions.items():
            if pos.state in ('shares', 'cc'):
                S = date_px.get(tk)
                if S is not None and np.isfinite(S):
                    equity += S * 100 * pos.contracts
        return equity

    def _sector_exposure(self, sector: str, equity: float) -> float:
        """Current sector exposure as fraction of equity."""
        exp = 0.0
        for pos in self.positions.values():
            if pos.sector == sector:
                if pos.state == 'csp':
                    exp += pos.strike * 100 * pos.contracts
                elif pos.state in ('shares', 'cc'):
                    exp += pos.strike * 100 * pos.contracts  # notional at strike
        return exp / max(equity, 1.0)

    def run(self, chains: Dict[str, pd.DataFrame],
            prices: pd.DataFrame, macro: pd.DataFrame,
            universe: pd.DataFrame, fundamentals: pd.DataFrame,
            start: str = "2019-01-01", end: str = "2024-12-31") -> dict:
        """Run the full wheel backtest."""

        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        # Build lookup structures
        sector_of = dict(zip(universe["ticker"], universe.get("sector", "Unknown")))
        fund_score_of = dict(zip(fundamentals["ticker"],
                                  fundamentals.get("fund_score", 50.0)))

        # Price lookup by (date, ticker)
        px_sub = prices[(prices["date"] >= start_dt) & (prices["date"] <= end_dt)]
        px_by_date = {}
        for d, g in px_sub.groupby("date"):
            px_by_date[d] = dict(zip(g["ticker"], g["close"]))

        # Macro lookup
        macro_sub = macro[(macro["date"] >= start_dt) & (macro["date"] <= end_dt)]
        macro_lookup = macro_sub.set_index("date").to_dict("index")

        # Build chain indices per ticker: (obs_date, expiry) -> chain_slice
        # and obs_date -> list of available tickers with chains
        ticker_chain_idx = {}    # {ticker: {(date, exp): df}}
        ticker_exp_obs = {}      # {ticker: {exp: [obs_dates]}}
        obs_date_tickers = defaultdict(set)  # {date: set of tickers with chains}

        for ticker, chain_df in chains.items():
            chain_sub = chain_df[(chain_df["date"] >= start_dt) &
                                 (chain_df["date"] <= end_dt)]
            if chain_sub.empty:
                continue
            idx = {}
            exp_obs = defaultdict(list)
            for (d, exp), grp in chain_sub.groupby(["date", "expiration"]):
                idx[(d, exp)] = grp
                exp_obs[exp].append(d)
                obs_date_tickers[d].add(ticker)
            for exp in exp_obs:
                exp_obs[exp] = sorted(exp_obs[exp])
            ticker_chain_idx[ticker] = idx
            ticker_exp_obs[ticker] = dict(exp_obs)

        # All trading dates (union of price dates and chain observation dates)
        all_dates = sorted(set(px_by_date.keys()) |
                           set(obs_date_tickers.keys()))
        all_dates = [d for d in all_dates if start_dt <= d <= end_dt]

        print(f"  Running V6 wheel: {len(all_dates)} dates, "
              f"{len(ticker_chain_idx)} tickers with chains")
        print(f"  CSP delta={self.put_delta}, CC delta={self.call_delta}, "
              f"dynamic_cc={self.dynamic_cc}")

        for dt in all_dates:
            date_px = px_by_date.get(dt, {})
            m = macro_lookup.get(dt, {})
            vix = m.get("vix", 20.0)
            if pd.isna(vix):
                vix = 20.0

            # ── 1) Manage existing positions ──
            to_remove = []
            for tk, pos in list(self.positions.items()):
                S = date_px.get(tk)
                if S is None or not np.isfinite(S):
                    continue

                chain_idx = ticker_chain_idx.get(tk, {})
                exp_obs = ticker_exp_obs.get(tk, {})

                if pos.state == 'csp':
                    dte = (pos.expiry - dt).days

                    if dte <= 0:
                        # Expiry: check if assigned
                        if S < pos.strike:
                            # ASSIGNED — buy shares at strike
                            cost = pos.strike * 100 * pos.contracts
                            self.cash -= cost
                            self.assignments += 1
                            # Record CSP trade (premium kept)
                            self.trades.append(WheelTrade(
                                ticker=tk, sector=pos.sector, leg_type="CSP",
                                open_date=pos.open_date, close_date=dt,
                                expiry=pos.expiry, strike=pos.strike,
                                contracts=pos.contracts,
                                premium_received=pos.premium_received,
                                close_cost=0.0,
                                pnl=pos.premium_received,
                                close_reason="assigned",
                                close_pricing="intrinsic",
                                entry_delta=self.put_delta,
                                entry_bid=pos.premium_received / (100 * pos.contracts),
                            ))
                            # Transition to holding shares
                            pos.state = 'shares'
                            pos.cost_basis = pos.strike - pos.premium_received / (100 * pos.contracts)
                            pos.share_purchase_price = pos.strike
                            pos.expiry = dt  # reset, will sell CC
                        else:
                            # Expired OTM — keep premium
                            self.csp_expired_otm += 1
                            self.trades.append(WheelTrade(
                                ticker=tk, sector=pos.sector, leg_type="CSP",
                                open_date=pos.open_date, close_date=dt,
                                expiry=pos.expiry, strike=pos.strike,
                                contracts=pos.contracts,
                                premium_received=pos.premium_received,
                                close_cost=0.0,
                                pnl=pos.premium_received,
                                close_reason="expired_otm",
                                close_pricing="intrinsic",
                                entry_delta=self.put_delta,
                                entry_bid=pos.premium_received / (100 * pos.contracts),
                            ))
                            to_remove.append(tk)
                    else:
                        # Check profit take or DTE close using real chain
                        exp_chain = chain_idx.get((dt, pos.expiry))
                        if exp_chain is not None:
                            strike_row = find_strike_by_value(exp_chain, pos.strike, "p")
                            if strike_row is not None:
                                cur_ask = float(strike_row["ask"])
                                open_bid = pos.premium_received / (100 * pos.contracts)
                                profit_frac = (open_bid - cur_ask) / max(open_bid, 0.01)

                                should_close = False
                                reason = ""
                                if profit_frac >= PROFIT_TAKE_FRAC:
                                    should_close = True
                                    reason = "profit_take"
                                elif dte <= DTE_CLOSE_TRIGGER:
                                    should_close = True
                                    reason = "dte_close"

                                if should_close:
                                    close_cost = cur_ask * 100 * pos.contracts
                                    self.cash -= close_cost
                                    pnl = pos.premium_received - close_cost
                                    self.trades.append(WheelTrade(
                                        ticker=tk, sector=pos.sector, leg_type="CSP",
                                        open_date=pos.open_date, close_date=dt,
                                        expiry=pos.expiry, strike=pos.strike,
                                        contracts=pos.contracts,
                                        premium_received=pos.premium_received,
                                        close_cost=close_cost,
                                        pnl=pnl,
                                        close_reason=reason,
                                        close_pricing="real_chain",
                                        entry_delta=self.put_delta,
                                        entry_bid=open_bid,
                                    ))
                                    to_remove.append(tk)

                elif pos.state == 'cc':
                    dte = (pos.expiry - dt).days

                    if dte <= 0:
                        if S > pos.strike:
                            # CALLED AWAY — sell shares at strike
                            proceeds = pos.strike * 100 * pos.contracts
                            self.cash += proceeds
                            self.called_away += 1
                            shares_pnl = (pos.strike - pos.share_purchase_price) * 100 * pos.contracts
                            self.trades.append(WheelTrade(
                                ticker=tk, sector=pos.sector, leg_type="CC",
                                open_date=pos.open_date, close_date=dt,
                                expiry=pos.expiry, strike=pos.strike,
                                contracts=pos.contracts,
                                premium_received=pos.premium_received,
                                close_cost=0.0,
                                pnl=pos.premium_received + shares_pnl,
                                close_reason="called_away",
                                close_pricing="intrinsic",
                                entry_delta=self._get_cc_delta(vix),
                                entry_bid=pos.premium_received / (100 * pos.contracts),
                                shares_pnl=shares_pnl,
                            ))
                            to_remove.append(tk)
                        else:
                            # CC expired OTM — keep shares + premium
                            self.cc_expired_otm += 1
                            self.trades.append(WheelTrade(
                                ticker=tk, sector=pos.sector, leg_type="CC",
                                open_date=pos.open_date, close_date=dt,
                                expiry=pos.expiry, strike=pos.strike,
                                contracts=pos.contracts,
                                premium_received=pos.premium_received,
                                close_cost=0.0,
                                pnl=pos.premium_received,
                                close_reason="expired_otm",
                                close_pricing="intrinsic",
                                entry_delta=self._get_cc_delta(vix),
                                entry_bid=pos.premium_received / (100 * pos.contracts),
                            ))
                            # Back to holding shares, sell new CC
                            pos.state = 'shares'
                            pos.expiry = dt
                    else:
                        # Check profit take / DTE close for CC
                        exp_chain = chain_idx.get((dt, pos.expiry))
                        if exp_chain is not None:
                            strike_row = find_strike_by_value(exp_chain, pos.strike, "c")
                            if strike_row is not None:
                                cur_ask = float(strike_row["ask"])
                                open_bid = pos.premium_received / (100 * pos.contracts)
                                profit_frac = (open_bid - cur_ask) / max(open_bid, 0.01)

                                should_close = False
                                reason = ""
                                if profit_frac >= PROFIT_TAKE_FRAC:
                                    should_close = True
                                    reason = "profit_take"
                                elif dte <= DTE_CLOSE_TRIGGER:
                                    should_close = True
                                    reason = "dte_close"

                                if should_close:
                                    close_cost = cur_ask * 100 * pos.contracts
                                    self.cash -= close_cost
                                    pnl = pos.premium_received - close_cost
                                    self.trades.append(WheelTrade(
                                        ticker=tk, sector=pos.sector, leg_type="CC",
                                        open_date=pos.open_date, close_date=dt,
                                        expiry=pos.expiry, strike=pos.strike,
                                        contracts=pos.contracts,
                                        premium_received=pos.premium_received,
                                        close_cost=close_cost,
                                        pnl=pnl,
                                        close_reason=reason,
                                        close_pricing="real_chain",
                                        entry_delta=self._get_cc_delta(vix),
                                        entry_bid=open_bid,
                                    ))
                                    pos.state = 'shares'
                                    pos.expiry = dt

                # Share stop-loss check
                if tk not in to_remove and pos.state in ('shares', 'cc'):
                    S = date_px.get(tk)
                    if (S is not None and np.isfinite(S) and
                            pos.share_purchase_price > 0 and
                            S < pos.share_purchase_price * (1 - SHARE_STOP_LOSS)):
                        # Buy back CC if active
                        if pos.state == 'cc':
                            exp_chain = chain_idx.get((dt, pos.expiry))
                            cc_buyback = 0.0
                            if exp_chain is not None:
                                strike_row = find_strike_by_value(exp_chain, pos.strike, "c")
                                if strike_row is not None:
                                    cc_buyback = float(strike_row["ask"]) * 100 * pos.contracts
                            # If no chain data, estimate at intrinsic
                            if cc_buyback == 0.0:
                                intrinsic = max(S - pos.strike, 0) * 100 * pos.contracts
                                cc_buyback = intrinsic
                            self.cash -= cc_buyback
                            self.trades.append(WheelTrade(
                                ticker=tk, sector=pos.sector, leg_type="CC",
                                open_date=pos.open_date, close_date=dt,
                                expiry=pos.expiry, strike=pos.strike,
                                contracts=pos.contracts,
                                premium_received=pos.premium_received,
                                close_cost=cc_buyback,
                                pnl=pos.premium_received - cc_buyback,
                                close_reason="stop_loss",
                                close_pricing="real_chain" if exp_chain is not None else "intrinsic",
                                entry_delta=self._get_cc_delta(vix),
                                entry_bid=pos.premium_received / (100 * pos.contracts),
                            ))

                        # Sell shares at market
                        proceeds = S * 100 * pos.contracts
                        self.cash += proceeds
                        shares_pnl = (S - pos.share_purchase_price) * 100 * pos.contracts
                        self.trades.append(WheelTrade(
                            ticker=tk, sector=pos.sector, leg_type="SHARE_SALE",
                            open_date=pos.open_date, close_date=dt,
                            expiry=pos.expiry, strike=pos.share_purchase_price,
                            contracts=pos.contracts,
                            premium_received=0.0, close_cost=0.0,
                            pnl=shares_pnl,
                            close_reason="stop_loss",
                            close_pricing="market_sell",
                            entry_delta=0.0, entry_bid=0.0,
                            shares_pnl=shares_pnl,
                        ))
                        self.stop_losses += 1
                        to_remove.append(tk)

            for tk in set(to_remove):
                if tk in self.positions:
                    del self.positions[tk]

            # ── 2) Sell CCs on shares without active CC ──
            cc_delta = self._get_cc_delta(vix)
            for tk, pos in list(self.positions.items()):
                if pos.state != 'shares':
                    continue
                S = date_px.get(tk)
                if S is None or not np.isfinite(S):
                    continue
                chain_idx_tk = ticker_chain_idx.get(tk, {})
                exp_obs_tk = ticker_exp_obs.get(tk, {})

                # Find best expiration for CC
                available_exps = [exp for (d, exp) in chain_idx_tk if d == dt]
                best_strike_row = None
                best_exp = None
                best_dte_diff = 999

                for exp in available_exps:
                    dte = (exp - dt).days
                    if DTE_MIN <= dte <= DTE_MAX:
                        # Need future obs date to close
                        future_obs = [d for d in exp_obs_tk.get(exp, []) if d > dt]
                        diff = abs(dte - DTE_TARGET)
                        if diff < best_dte_diff:
                            exp_chain = chain_idx_tk.get((dt, exp))
                            if exp_chain is not None:
                                row = find_best_strike(exp_chain, cc_delta, "c")
                                if row is not None:
                                    # CC strike should be above current price
                                    if float(row["strike"]) > S:
                                        best_strike_row = row
                                        best_exp = exp
                                        best_dte_diff = diff

                if best_strike_row is not None:
                    cc_bid = float(best_strike_row["bid"])
                    cc_strike = float(best_strike_row["strike"])
                    premium = cc_bid * 100 * pos.contracts
                    self.cash += premium
                    self.total_cc_premium += premium
                    pos.state = 'cc'
                    pos.strike = cc_strike
                    pos.premium_received = premium
                    pos.open_date = dt
                    pos.expiry = best_exp
                    self.cc_opened += 1

            # ── 3) Open new CSPs ──
            equity = self._equity(date_px)
            if equity <= 0:
                equity = self.cash
            self.equity_curve.append((dt, equity))

            # Macro gates
            if vix > VIX_GATE_OPEN:
                continue
            if len(self.positions) >= MAX_CONCURRENT:
                continue

            # Find candidates for new CSPs
            available_tickers = obs_date_tickers.get(dt, set())
            candidates = []

            for tk in available_tickers:
                if tk in self.positions:
                    continue
                S = date_px.get(tk)
                if S is None or not np.isfinite(S):
                    continue
                sector = sector_of.get(tk, "Unknown")
                fs = fund_score_of.get(tk, 50.0)
                if fs < 35.0:
                    continue
                if self._sector_exposure(sector, equity) > SECTOR_CAP:
                    continue
                # Single name cap
                notional = S * 100  # 1 contract
                if notional / max(equity, 1) > SINGLE_NAME_CAP:
                    continue

                chain_idx_tk = ticker_chain_idx.get(tk, {})
                exp_obs_tk = ticker_exp_obs.get(tk, {})

                # Find best expiration
                available_exps = [exp for (d, exp) in chain_idx_tk if d == dt]
                best_put_row = None
                best_exp = None
                best_dte_diff = 999

                for exp in available_exps:
                    dte = (exp - dt).days
                    if DTE_MIN <= dte <= DTE_MAX:
                        future_obs = [d for d in exp_obs_tk.get(exp, []) if d > dt]
                        diff = abs(dte - DTE_TARGET)
                        if diff < best_dte_diff:
                            exp_chain = chain_idx_tk.get((dt, exp))
                            if exp_chain is not None:
                                row = find_best_strike(exp_chain, self.put_delta, "p")
                                if row is not None:
                                    # Put strike should be below current price
                                    if float(row["strike"]) < S:
                                        best_put_row = row
                                        best_exp = exp
                                        best_dte_diff = diff

                if best_put_row is not None:
                    put_bid = float(best_put_row["bid"])
                    put_strike = float(best_put_row["strike"])
                    put_delta_actual = abs(float(best_put_row["delta"]))
                    candidates.append((tk, S, sector, fs, put_bid, put_strike,
                                       best_exp, put_delta_actual))

            # Rank by premium/notional (yield)
            candidates.sort(key=lambda x: x[4] / max(x[5], 1), reverse=True)

            slots = MAX_CONCURRENT - len(self.positions)
            slots = min(slots, max(1, MAX_CONCURRENT // 5))

            for tk, S, sector, fs, put_bid, put_strike, exp, delta_actual in candidates[:slots]:
                # How many contracts?
                max_alloc = SINGLE_NAME_CAP * equity
                n_contracts = max(1, int(max_alloc // (put_strike * 100)))
                secure_needed = put_strike * 100 * n_contracts
                if secure_needed > self.cash:
                    n_contracts = int(self.cash // (put_strike * 100))
                    if n_contracts < 1:
                        continue

                # Sector cap re-check
                new_exp = (put_strike * 100 * n_contracts) / max(equity, 1)
                if self._sector_exposure(sector, equity) + new_exp > SECTOR_CAP:
                    continue

                premium = put_bid * 100 * n_contracts
                self.cash += premium
                self.total_csp_premium += premium

                self.positions[tk] = WheelPosition(
                    ticker=tk, sector=sector, state='csp',
                    open_date=dt, expiry=exp, strike=put_strike,
                    contracts=n_contracts, premium_received=premium,
                    cost_basis=0.0, share_purchase_price=0.0,
                )
                self.csp_opened += 1

                if len(self.positions) >= MAX_CONCURRENT:
                    break

        # Force close remaining positions
        if all_dates:
            last_dt = all_dates[-1]
            last_px = px_by_date.get(last_dt, {})
            for tk, pos in list(self.positions.items()):
                S = last_px.get(tk, pos.strike)
                if pos.state == 'csp':
                    # Close CSP at intrinsic
                    intrinsic = max(pos.strike - S, 0)
                    close_cost = intrinsic * 100 * pos.contracts
                    pnl = pos.premium_received - close_cost
                    self.trades.append(WheelTrade(
                        ticker=tk, sector=pos.sector, leg_type="CSP",
                        open_date=pos.open_date, close_date=last_dt,
                        expiry=pos.expiry, strike=pos.strike,
                        contracts=pos.contracts,
                        premium_received=pos.premium_received,
                        close_cost=close_cost, pnl=pnl,
                        close_reason="end_of_data", close_pricing="intrinsic",
                        entry_delta=self.put_delta, entry_bid=0.0,
                    ))
                elif pos.state in ('shares', 'cc'):
                    # Sell shares at market
                    if pos.state == 'cc':
                        intrinsic = max(S - pos.strike, 0)
                        cc_close = intrinsic * 100 * pos.contracts
                        self.cash -= cc_close
                        self.trades.append(WheelTrade(
                            ticker=tk, sector=pos.sector, leg_type="CC",
                            open_date=pos.open_date, close_date=last_dt,
                            expiry=pos.expiry, strike=pos.strike,
                            contracts=pos.contracts,
                            premium_received=pos.premium_received,
                            close_cost=cc_close,
                            pnl=pos.premium_received - cc_close,
                            close_reason="end_of_data", close_pricing="intrinsic",
                            entry_delta=self.call_delta, entry_bid=0.0,
                        ))
                    proceeds = S * 100 * pos.contracts
                    self.cash += proceeds
                    shares_pnl = (S - pos.share_purchase_price) * 100 * pos.contracts
                    self.trades.append(WheelTrade(
                        ticker=tk, sector=pos.sector, leg_type="SHARE_SALE",
                        open_date=pos.open_date, close_date=last_dt,
                        expiry=pos.expiry, strike=pos.share_purchase_price,
                        contracts=pos.contracts,
                        premium_received=0.0, close_cost=0.0,
                        pnl=shares_pnl,
                        close_reason="end_of_data", close_pricing="market_sell",
                        entry_delta=0.0, entry_bid=0.0, shares_pnl=shares_pnl,
                    ))

            final_eq = self._equity(last_px)
            self.equity_curve.append((last_dt, final_eq))

        return self._build_results()

    def _build_results(self) -> dict:
        return {
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "csp_opened": self.csp_opened,
            "cc_opened": self.cc_opened,
            "assignments": self.assignments,
            "called_away": self.called_away,
            "csp_expired_otm": self.csp_expired_otm,
            "cc_expired_otm": self.cc_expired_otm,
            "stop_losses": self.stop_losses,
            "total_csp_premium": self.total_csp_premium,
            "total_cc_premium": self.total_cc_premium,
            "final_cash": self.cash,
            "starting_cash": self.starting_cash,
        }


# ── Metrics ──

def build_equity_df(equity_curve: List[Tuple], starting_cash: float) -> pd.DataFrame:
    """Build equity DataFrame from curve."""
    df = pd.DataFrame(equity_curve, columns=["date", "equity"])
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    df["ret"] = df["equity"].pct_change().fillna(0)
    return df


def compute_metrics(eq_df: pd.DataFrame, starting_cash: float) -> dict:
    """Compute risk-adjusted metrics."""
    if eq_df.empty:
        return {}
    eq = eq_df["equity"].astype(float)
    rets = eq_df["ret"].dropna()
    years = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days / 365.25
    if years <= 0:
        return {}
    final = float(eq.iloc[-1])
    cagr = (final / starting_cash) ** (1.0 / years) - 1.0

    mu, sd = rets.mean(), rets.std()
    downside = rets[rets < 0].std()
    sharpe = (mu / sd) * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    sortino = (mu / downside) * math.sqrt(TRADING_DAYS) if downside > 0 else 0.0

    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    # Trade-level WR and PF
    return {
        "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "calmar": calmar,
        "final_equity": final, "years": years,
        "total_return_pct": (final / starting_cash - 1) * 100,
    }


def regime_gate(eq_df: pd.DataFrame, spy_close: pd.Series) -> dict:
    """HC #428 R1 regime-stratified analysis."""
    spy_ret = spy_close.pct_change()
    rets = eq_df.set_index("date")["ret"]

    # Classify days
    labels = pd.Series("flat", index=spy_ret.index)
    labels[spy_ret > 0.002] = "green"
    labels[spy_ret < -0.002] = "red"

    aligned_labels = labels.reindex(rets.index)

    result = {}
    for regime in ("green", "red", "flat"):
        sub = rets[aligned_labels == regime].dropna()
        result[f"n_{regime}"] = len(sub)
        if len(sub) >= 2 and sub.std() > 0:
            result[f"sharpe_{regime}"] = float(sub.mean() / sub.std() * math.sqrt(TRADING_DAYS))
        else:
            result[f"sharpe_{regime}"] = float("nan")

    sg = result.get("sharpe_green", 0)
    sr = result.get("sharpe_red", 0)
    if np.isfinite(sg) and np.isfinite(sr):
        denom = max(abs(sg), abs(sr), 1e-9)
        result["regime_gap"] = abs(sg - sr) / denom
        result["r1_pass"] = result["regime_gap"] <= 0.50
    else:
        result["regime_gap"] = float("nan")
        result["r1_pass"] = False
    return result


def permutation_test(trades: List[WheelTrade], n_trials: int = 100) -> dict:
    """Randomize trade assignment to dates to test if returns are real."""
    if not trades:
        return {"p_value": 1.0, "real_pnl": 0.0, "mean_random_pnl": 0.0}
    pnls = [t.pnl for t in trades]
    real_total = sum(pnls)
    rng = np.random.default_rng(42)
    count_worse = 0
    random_totals = []
    for _ in range(n_trials):
        # Shuffle trade PnLs (break time dependency)
        shuffled = rng.permutation(pnls)
        random_totals.append(sum(shuffled))
        # Compare cumulative path
        if sum(shuffled) >= real_total:
            count_worse += 1
    return {
        "p_value": count_worse / n_trials,
        "real_total_pnl": real_total,
        "mean_random_pnl": float(np.mean(random_totals)),
        "std_random_pnl": float(np.std(random_totals)),
    }


def travel_income_analysis(equity_curve: List[Tuple], starting_cash: float,
                           monthly_withdrawal_pct: float = 0.015) -> dict:
    """Model withdrawing X% per month. Can the strategy sustain NAV?"""
    if not equity_curve:
        return {}

    eq_df = pd.DataFrame(equity_curve, columns=["date", "equity"])
    eq_df = eq_df.drop_duplicates("date", keep="last").sort_values("date")

    # Build daily returns
    eq_df["ret"] = eq_df["equity"].pct_change().fillna(0)

    # Simulate with withdrawals
    nav = starting_cash
    monthly_withdrawal = starting_cash * monthly_withdrawal_pct
    last_withdrawal_month = None
    nav_history = []
    withdrawal_total = 0.0
    months_of_loss = 0
    current_loss_streak = 0
    max_loss_streak = 0
    monthly_returns = []
    month_start_nav = nav

    for _, row in eq_df.iterrows():
        dt = row["date"]
        daily_ret = row["ret"]
        nav *= (1 + daily_ret)

        current_month = (dt.year, dt.month)
        if last_withdrawal_month != current_month:
            # Record monthly return
            if last_withdrawal_month is not None:
                m_ret = (nav - month_start_nav) / max(month_start_nav, 1)
                monthly_returns.append(m_ret)
                if m_ret < 0:
                    current_loss_streak += 1
                    max_loss_streak = max(max_loss_streak, current_loss_streak)
                else:
                    current_loss_streak = 0

            # Withdraw
            withdrawal = min(monthly_withdrawal, nav * 0.5)  # Don't withdraw more than 50%
            nav -= withdrawal
            withdrawal_total += withdrawal
            last_withdrawal_month = current_month
            month_start_nav = nav

        nav_history.append((dt, nav))

    nav_df = pd.DataFrame(nav_history, columns=["date", "nav"])
    final_nav = nav_df["nav"].iloc[-1] if not nav_df.empty else 0

    # Find maximum sustainable withdrawal
    max_sustainable = None
    for test_pct in np.arange(0.005, 0.04, 0.001):
        test_nav = starting_cash
        test_withdrawal = starting_cash * test_pct
        test_month = None
        survived = True
        for _, row in eq_df.iterrows():
            test_nav *= (1 + row["ret"])
            cm = (row["date"].year, row["date"].month)
            if test_month != cm:
                test_nav -= min(test_withdrawal, test_nav * 0.5)
                test_month = cm
            if test_nav < starting_cash * 0.3:  # NAV dropped below 30%
                survived = False
                break
        if survived:
            max_sustainable = test_pct

    # Sequence of returns risk: start at worst time
    # Find the month with worst subsequent 12-month return
    worst_start_idx = 0
    worst_12m_ret = float("inf")
    monthly_eq = eq_df.set_index("date").resample("ME").last()
    if len(monthly_eq) > 12:
        for i in range(len(monthly_eq) - 12):
            ret_12m = monthly_eq["equity"].iloc[i+12] / monthly_eq["equity"].iloc[i] - 1
            if ret_12m < worst_12m_ret:
                worst_12m_ret = ret_12m
                worst_start_idx = i

    return {
        "monthly_withdrawal_pct": monthly_withdrawal_pct,
        "monthly_withdrawal_usd": monthly_withdrawal,
        "total_withdrawn": withdrawal_total,
        "final_nav": final_nav,
        "nav_preserved": final_nav >= starting_cash * 0.9,
        "max_sustainable_monthly_pct": max_sustainable,
        "max_loss_streak_months": max_loss_streak,
        "monthly_return_mean": float(np.mean(monthly_returns)) if monthly_returns else 0,
        "monthly_return_std": float(np.std(monthly_returns)) if monthly_returns else 0,
        "worst_12m_return": worst_12m_ret if worst_12m_ret != float("inf") else 0,
        "worst_12m_start": str(monthly_eq.index[worst_start_idx].date()) if len(monthly_eq) > 12 else "N/A",
    }


def run_csp_only_baseline(chains, prices, macro, universe, fundamentals,
                          start, end) -> dict:
    """Run CSP-only (no covered calls) for comparison with V5."""
    engine = WheelV6Engine(put_delta=0.25, call_delta=0.25,
                           starting_cash=STARTING_CASH)

    # Monkey-patch: skip CC selling
    original_run = engine.run

    class CSPOnlyEngine(WheelV6Engine):
        def run(self, *args, **kwargs):
            # Override: when shares assigned, immediately sell at market
            # (no CC leg)
            result = super().run(*args, **kwargs)
            return result

    csp_engine = CSPOnlyEngine(put_delta=0.25, starting_cash=STARTING_CASH)
    # For CSP-only: set share_stop_loss to 0 (sell immediately on assignment)
    # Actually, let's just run the engine without CC by making CC delta = 0
    # which will prevent any CC from being matched

    # Simpler: run the full engine but set call_delta to something impossible
    csp_engine_real = WheelV6Engine(put_delta=0.25, call_delta=0.01,
                                    starting_cash=STARTING_CASH)
    return csp_engine_real.run(chains, prices, macro, universe, fundamentals,
                               start, end)


def main():
    t0 = time.time()
    print("=" * 80)
    print("WHEEL V6 — FULL WHEEL BACKTEST WITH REAL DOLT OPTION CHAIN DATA")
    print("  Entry: REAL bid from Dolt chains")
    print("  Exit: REAL ask from Dolt chains (or intrinsic at expiry)")
    print("  Commission: $0 (Robinhood, HC #694)")
    print("  Full lifecycle: CSP → assignment → CC → called away → repeat")
    print("=" * 80)

    print("\n[1/6] Loading data...")
    chains = load_all_chains()
    prices = load_prices()
    macro = load_macro()
    universe = load_universe()
    fundamentals = load_fundamentals()
    print(f"  Loaded {len(chains)} tickers with chain data")

    # Load SPY for regime analysis
    try:
        etf = pd.read_parquet(CACHE / "sector_etfs.parquet")
        spy = etf[etf["ticker"] == "SPY"][["date", "close"]].copy()
        spy["date"] = pd.to_datetime(spy["date"])
        spy_close = spy.set_index("date")["close"].sort_index()
    except Exception:
        spy_close = None

    START = "2019-03-01"  # Dolt data starts Feb 2019
    END = "2024-12-31"

    # ── Run variants ──
    variants = {
        "V6-Base (d25/d25)": {"put_delta": 0.25, "call_delta": 0.25, "dynamic_cc": False},
        "V6-Aggressive (d25/d20)": {"put_delta": 0.25, "call_delta": 0.20, "dynamic_cc": False},
        "V6-Conservative (d25/d30)": {"put_delta": 0.25, "call_delta": 0.30, "dynamic_cc": False},
        "V6-Dynamic CC": {"put_delta": 0.25, "call_delta": 0.25, "dynamic_cc": True},
    }

    all_results = {}

    for vi, (name, params) in enumerate(variants.items()):
        print(f"\n[{vi+2}/6] Running {name}...")
        engine = WheelV6Engine(
            put_delta=params["put_delta"],
            call_delta=params["call_delta"],
            dynamic_cc=params["dynamic_cc"],
            starting_cash=STARTING_CASH,
        )
        result = engine.run(chains, prices, macro, universe, fundamentals,
                           START, END)
        eq_df = build_equity_df(result["equity_curve"], STARTING_CASH)
        metrics = compute_metrics(eq_df, STARTING_CASH)

        # Trade-level stats
        trades = result["trades"]
        csp_trades = [t for t in trades if t.leg_type == "CSP"]
        cc_trades = [t for t in trades if t.leg_type == "CC"]
        share_trades = [t for t in trades if t.leg_type == "SHARE_SALE"]

        n_trades = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in trades)
        csp_pnl = sum(t.pnl for t in csp_trades)
        cc_pnl = sum(t.pnl for t in cc_trades)
        share_pnl = sum(t.pnl for t in share_trades)

        wr = wins / n_trades if n_trades > 0 else 0
        pf_gains = sum(t.pnl for t in trades if t.pnl > 0)
        pf_losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
        pf = pf_gains / pf_losses if pf_losses > 0 else float("inf")

        # Regime gate
        regime = {}
        if spy_close is not None:
            regime = regime_gate(eq_df, spy_close)

        # Permutation test
        perm = permutation_test(trades, n_trials=100)

        # Travel income
        travel = travel_income_analysis(result["equity_curve"], STARTING_CASH,
                                        monthly_withdrawal_pct=0.015)

        # Monthly return consistency
        eq_monthly = eq_df.set_index("date").resample("ME").last()
        monthly_rets = eq_monthly["equity"].pct_change().dropna()
        monthly_std = float(monthly_rets.std()) if len(monthly_rets) > 1 else 0

        all_results[name] = {
            "metrics": metrics,
            "trade_stats": {
                "total_trades": n_trades,
                "csp_trades": len(csp_trades),
                "cc_trades": len(cc_trades),
                "share_trades": len(share_trades),
                "wins": wins,
                "win_rate": wr,
                "profit_factor": pf,
                "total_pnl": total_pnl,
                "csp_pnl": csp_pnl,
                "cc_pnl": cc_pnl,
                "share_pnl": share_pnl,
                "csp_premium_total": result["total_csp_premium"],
                "cc_premium_total": result["total_cc_premium"],
                "premium_pct_from_puts": result["total_csp_premium"] /
                    max(result["total_csp_premium"] + result["total_cc_premium"], 1) * 100,
                "premium_pct_from_calls": result["total_cc_premium"] /
                    max(result["total_csp_premium"] + result["total_cc_premium"], 1) * 100,
                "assignments": result["assignments"],
                "called_away": result["called_away"],
                "assignment_rate": result["assignments"] / max(result["csp_opened"], 1),
                "call_away_rate": result["called_away"] / max(result["cc_opened"], 1),
                "stop_losses": result["stop_losses"],
                "csp_opened": result["csp_opened"],
                "cc_opened": result["cc_opened"],
            },
            "regime": regime,
            "permutation": perm,
            "travel_income": travel,
            "monthly_return_std": monthly_std,
            "equity_curve": eq_df,
            "trades_list": trades,
        }

        # Print summary
        m = metrics
        ts = all_results[name]["trade_stats"]
        print(f"\n  ── {name} RESULTS ──")
        print(f"  CAGR:    {m['cagr']*100:+.1f}%")
        print(f"  Sharpe:  {m['sharpe']:.2f}")
        print(f"  Sortino: {m['sortino']:.2f}")
        print(f"  MaxDD:   {m['max_dd']*100:.1f}%")
        print(f"  Calmar:  {m['calmar']:.2f}")
        print(f"  WR:      {wr*100:.0f}%  PF: {pf:.2f}")
        print(f"  Total PnL: ${total_pnl:+,.0f}")
        print(f"  CSP PnL: ${csp_pnl:+,.0f}  CC PnL: ${cc_pnl:+,.0f}  Share PnL: ${share_pnl:+,.0f}")
        print(f"  Premium split: {ts['premium_pct_from_puts']:.0f}% puts / {ts['premium_pct_from_calls']:.0f}% calls")
        print(f"  Assignments: {result['assignments']}  Called away: {result['called_away']}  Stop losses: {result['stop_losses']}")
        print(f"  Assignment rate: {ts['assignment_rate']*100:.1f}%  Call-away rate: {ts['call_away_rate']*100:.1f}%")

    # ── Comparative summary ──
    print("\n" + "=" * 120)
    print("COMPARATIVE SUMMARY — ALL VARIANTS")
    print("=" * 120)
    print(f"{'Variant':<30} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Calmar':>7} "
          f"{'WR%':>5} {'PF':>5} {'CSP$':>10} {'CC$':>10} {'Total$':>10}")
    print("-" * 120)

    for name, r in all_results.items():
        m = r["metrics"]
        ts = r["trade_stats"]
        print(f"{name:<30} {m['cagr']*100:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd']*100:>6.1f}% {m['calmar']:>7.2f} "
              f"{ts['win_rate']*100:>4.0f}% {ts['profit_factor']:>5.2f} "
              f"${ts['csp_pnl']:>+9,.0f} ${ts['cc_pnl']:>+9,.0f} ${ts['total_pnl']:>+9,.0f}")

    # ── Regime Gate ──
    print("\n" + "=" * 80)
    print("REGIME GATE (HC #428 R1)")
    print("=" * 80)
    for name, r in all_results.items():
        rg = r["regime"]
        if rg:
            print(f"  {name}:")
            print(f"    Green Sharpe: {rg.get('sharpe_green', 'N/A'):.2f}  "
                  f"Red Sharpe: {rg.get('sharpe_red', 'N/A'):.2f}  "
                  f"Flat Sharpe: {rg.get('sharpe_flat', 'N/A'):.2f}")
            print(f"    Regime gap: {rg.get('regime_gap', 'N/A'):.3f}  "
                  f"R1 PASS: {rg.get('r1_pass', False)}")

    # ── Permutation Test ──
    print("\n" + "=" * 80)
    print("PERMUTATION TEST (100 trials)")
    print("=" * 80)
    for name, r in all_results.items():
        p = r["permutation"]
        print(f"  {name}: p-value={p['p_value']:.3f}  "
              f"Real PnL=${p['real_total_pnl']:+,.0f}  "
              f"Random mean=${p['mean_random_pnl']:+,.0f}")

    # ── Travel Income ──
    print("\n" + "=" * 80)
    print("TRAVEL INCOME ANALYSIS (1.5%/month = $1,500/mo on $100K)")
    print("=" * 80)
    for name, r in all_results.items():
        ti = r["travel_income"]
        print(f"  {name}:")
        print(f"    Total withdrawn: ${ti.get('total_withdrawn', 0):,.0f}")
        print(f"    Final NAV: ${ti.get('final_nav', 0):,.0f}  "
              f"(NAV preserved: {ti.get('nav_preserved', False)})")
        max_sust = ti.get('max_sustainable_monthly_pct')
        if max_sust is not None:
            print(f"    Max sustainable withdrawal: {max_sust*100:.1f}%/month "
                  f"(${max_sust * STARTING_CASH:,.0f}/mo)")
        else:
            print(f"    Max sustainable withdrawal: <0.5%/month")
        print(f"    Max consecutive losing months: {ti.get('max_loss_streak_months', 0)}")
        print(f"    Monthly return mean: {ti.get('monthly_return_mean', 0)*100:.2f}%  "
              f"std: {ti.get('monthly_return_std', 0)*100:.2f}%")
        print(f"    Worst 12-month return: {ti.get('worst_12m_return', 0)*100:.1f}% "
              f"(starting {ti.get('worst_12m_start', 'N/A')})")

    # ── Premium Breakdown ──
    print("\n" + "=" * 80)
    print("PREMIUM BREAKDOWN: PUTS vs CALLS")
    print("=" * 80)
    for name, r in all_results.items():
        ts = r["trade_stats"]
        total_prem = ts["csp_premium_total"] + ts["cc_premium_total"]
        print(f"  {name}:")
        print(f"    CSP premium: ${ts['csp_premium_total']:,.0f} ({ts['premium_pct_from_puts']:.0f}%)")
        print(f"    CC premium:  ${ts['cc_premium_total']:,.0f} ({ts['premium_pct_from_calls']:.0f}%)")
        print(f"    Total:       ${total_prem:,.0f}")

    # ── Monthly Income Consistency ──
    print("\n" + "=" * 80)
    print("MONTHLY INCOME CONSISTENCY")
    print("=" * 80)
    for name, r in all_results.items():
        print(f"  {name}: monthly return StdDev = {r['monthly_return_std']*100:.2f}%")

    # ── Save results ──
    save_results = {}
    for name, r in all_results.items():
        save_results[name] = {
            "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                       for k, v in r["metrics"].items()},
            "trade_stats": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                           for k, v in r["trade_stats"].items()},
            "regime": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                      for k, v in r.get("regime", {}).items()},
            "permutation": r["permutation"],
            "travel_income": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                             for k, v in r["travel_income"].items()},
            "monthly_return_std": float(r["monthly_return_std"]),
        }

        # Save equity curve
        r["equity_curve"].to_parquet(OUT_DIR / f"equity_{name.replace(' ', '_').replace('/', '_')}.parquet",
                                     index=False)

        # Save trade ledger
        trade_rows = []
        for t in r["trades_list"]:
            trade_rows.append({
                "ticker": t.ticker, "sector": t.sector, "leg_type": t.leg_type,
                "open_date": t.open_date, "close_date": t.close_date,
                "expiry": t.expiry, "strike": t.strike, "contracts": t.contracts,
                "premium_received": t.premium_received, "close_cost": t.close_cost,
                "pnl": t.pnl, "close_reason": t.close_reason,
                "close_pricing": t.close_pricing, "entry_delta": t.entry_delta,
                "shares_pnl": t.shares_pnl,
            })
        if trade_rows:
            trade_df = pd.DataFrame(trade_rows)
            trade_df.to_parquet(OUT_DIR / f"ledger_{name.replace(' ', '_').replace('/', '_')}.parquet",
                                index=False)

    with open(OUT_DIR / "v6_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"COMPLETE in {elapsed:.0f}s. Results saved to {OUT_DIR}")
    print(f"{'='*80}")

    return all_results


if __name__ == "__main__":
    main()
