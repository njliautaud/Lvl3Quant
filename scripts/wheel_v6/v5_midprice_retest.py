#!/usr/bin/env python3
"""
v5_midprice_retest.py — Mid-Price Execution + V5 Risk Controls Retest

Tests the honest truth between BS fantasy (25%) and worst-case bid/ask (-3.3%).

Test 1: Mid-Price Execution — sell at (bid+ask)/2, buy at (bid+ask)/2
Test 2: Conservative Mid — sell at bid + 0.4*(ask-bid), buy at ask - 0.4*(ask-bid)
Test 3: Bid-Ask Spread Analysis — how pessimistic is Dolt data?
Test 4: V5 Risk Controls — VIX gate, earnings filter, vol-sizing, exposure caps

All tests use real Dolt chain data from 69 tickers.
"""
from __future__ import annotations

import json
import math
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

# Strategy params
DTE_MIN = 14
DTE_MAX = 50
DTE_TARGET = 30
DTE_CLOSE_TRIGGER = 5
PROFIT_TAKE_FRAC = 0.50
MAX_CONCURRENT = 20
SINGLE_NAME_CAP = 0.10
SECTOR_CAP = 0.30
DELTA_TOLERANCE = 0.12


# ─── Pricing modes ───────────────────────────────────────────────────────────

def price_sell(bid: float, ask: float, mode: str) -> float:
    """Price received when SELLING an option (opening CSP/CC)."""
    mid = (bid + ask) / 2.0
    if mode == "bid":
        return bid                           # worst case (original)
    elif mode == "mid":
        return mid                           # mid-price limit
    elif mode == "conservative_mid":
        return bid + 0.4 * (ask - bid)       # 40% from bid toward mid
    elif mode == "ask":
        return ask                           # best case (never used, just for reference)
    return bid


def price_buy(bid: float, ask: float, mode: str) -> float:
    """Price paid when BUYING an option (closing CSP/CC)."""
    mid = (bid + ask) / 2.0
    if mode == "ask":
        return ask                           # worst case (original)
    elif mode == "mid":
        return mid                           # mid-price limit
    elif mode == "conservative_mid":
        return ask - 0.4 * (ask - bid)       # 40% from ask toward mid
    elif mode == "bid":
        return bid                           # best case
    return ask


# ─── Data loading ────────────────────────────────────────────────────────────

def load_all_chains() -> Dict[str, pd.DataFrame]:
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
    return chains


def find_best_strike(chain_slice: pd.DataFrame, target_delta: float,
                     option_type: str) -> Optional[pd.Series]:
    sub = chain_slice[chain_slice["type"] == option_type].copy()
    if sub.empty:
        return None
    sub["delta_abs"] = sub["delta"].abs()
    sub["delta_diff"] = (sub["delta_abs"] - target_delta).abs()
    valid = sub[sub["delta_diff"] <= DELTA_TOLERANCE]
    if valid.empty:
        return None
    best = valid.loc[valid["delta_diff"].idxmin()]
    if float(best["bid"]) <= 0:
        return None
    return best


def find_strike_by_value(chain_slice: pd.DataFrame, strike: float,
                         option_type: str) -> Optional[pd.Series]:
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


# ─── Position & Trade ────────────────────────────────────────────────────────

@dataclass
class Pos:
    ticker: str
    sector: str
    state: str  # 'csp' | 'shares' | 'cc'
    open_date: pd.Timestamp
    expiry: pd.Timestamp
    strike: float
    contracts: int
    premium_received: float
    cost_basis: float
    share_purchase_price: float


@dataclass
class Trade:
    ticker: str
    sector: str
    leg_type: str
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    expiry: pd.Timestamp
    strike: float
    contracts: int
    premium_received: float
    close_cost: float
    pnl: float
    close_reason: str
    close_pricing: str
    entry_delta: float
    shares_pnl: float = 0.0


# ─── Engine with configurable pricing + V5 risk controls ────────────────────

class MidPriceWheelEngine:
    def __init__(self, put_delta=0.25, call_delta=0.25,
                 vix_gate=28.0, share_stop_loss=0.10,
                 max_assigned_exposure=0.30, max_assigned_names=3,
                 sell_cc=True, dynamic_cc=False,
                 pricing_mode="bid",  # "bid", "mid", "conservative_mid"
                 # V5 risk controls
                 vix_hard_gate=None,  # VIX > X = no CSPs at all
                 vix_term_structure_sizing=False,
                 earnings_filter_days=0,
                 dynamic_vol_sizing=False,
                 max_portfolio_delta=None,
                 starting_cash=STARTING_CASH):
        self.put_delta = put_delta
        self.call_delta = call_delta
        self.vix_gate = vix_gate
        self.share_stop_loss = share_stop_loss
        self.max_assigned_exposure = max_assigned_exposure
        self.max_assigned_names = max_assigned_names
        self.sell_cc = sell_cc
        self.dynamic_cc = dynamic_cc
        self.pricing_mode = pricing_mode
        self.vix_hard_gate = vix_hard_gate
        self.vix_term_structure_sizing = vix_term_structure_sizing
        self.earnings_filter_days = earnings_filter_days
        self.dynamic_vol_sizing = dynamic_vol_sizing
        self.max_portfolio_delta = max_portfolio_delta
        self.starting_cash = starting_cash

        self.cash = starting_cash
        self.positions: Dict[str, Pos] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple] = []

        self.csp_opened = 0
        self.cc_opened = 0
        self.assignments = 0
        self.called_away = 0
        self.csp_expired_otm = 0
        self.cc_expired_otm = 0
        self.stop_losses = 0
        self.total_csp_premium = 0.0
        self.total_cc_premium = 0.0
        self.blocked_by_vix = 0
        self.blocked_by_earnings = 0
        self.vol_sizing_adjustments = 0

    def _get_cc_delta(self, vix):
        if not self.dynamic_cc:
            return self.call_delta
        if vix < 15:
            return 0.30
        elif vix < 25:
            return 0.25
        else:
            return 0.20

    def _equity(self, date_px):
        equity = self.cash
        for tk, pos in self.positions.items():
            if pos.state in ('shares', 'cc'):
                S = date_px.get(tk)
                if S is not None and np.isfinite(S):
                    equity += S * 100 * pos.contracts
        return equity

    def _assigned_exposure(self, date_px, equity):
        mv = 0.0
        for pos in self.positions.values():
            if pos.state in ('shares', 'cc'):
                S = date_px.get(pos.ticker)
                if S is not None and np.isfinite(S):
                    mv += S * 100 * pos.contracts
        return mv / max(equity, 1.0)

    def _assigned_count(self):
        return sum(1 for p in self.positions.values() if p.state in ('shares', 'cc'))

    def _sector_exposure(self, sector, equity):
        exp = 0.0
        for pos in self.positions.values():
            if pos.sector == sector:
                exp += pos.strike * 100 * pos.contracts
        return exp / max(equity, 1.0)

    def _vol_size_factor(self, vix, vix3m=None):
        """Dynamic vol-based position sizing. Returns multiplier 0.0 to 1.0."""
        if not self.dynamic_vol_sizing:
            return 1.0
        # Base: inverse-VIX sizing
        if vix <= 15:
            factor = 1.0
        elif vix <= 20:
            factor = 0.85
        elif vix <= 25:
            factor = 0.65
        elif vix <= 30:
            factor = 0.45
        else:
            factor = 0.25
        # Term structure inversion penalty
        if self.vix_term_structure_sizing and vix3m is not None:
            if vix > vix3m * 1.05:  # VIX > VIX3M by 5%+ = inverted = danger
                factor *= 0.5
                self.vol_sizing_adjustments += 1
        return factor

    def run(self, chains, prices, macro, universe, fundamentals,
            earnings_dates=None,
            start="2019-03-01", end="2024-12-31"):
        start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
        pm = self.pricing_mode

        sector_of = dict(zip(universe["ticker"], universe.get("sector", "Unknown")))
        fund_score_of = dict(zip(fundamentals["ticker"],
                                  fundamentals.get("fund_score", 50.0)))

        px_sub = prices[(prices["date"] >= start_dt) & (prices["date"] <= end_dt)]
        px_by_date = {}
        for d, g in px_sub.groupby("date"):
            px_by_date[d] = dict(zip(g["ticker"], g["close"]))

        macro_sub = macro[(macro["date"] >= start_dt) & (macro["date"] <= end_dt)]
        macro_lookup = macro_sub.set_index("date").to_dict("index")

        ticker_chain_idx = {}
        ticker_exp_obs = {}
        obs_date_tickers = defaultdict(set)

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

        all_dates = sorted(set(px_by_date.keys()) | set(obs_date_tickers.keys()))
        all_dates = [d for d in all_dates if start_dt <= d <= end_dt]

        for dt in all_dates:
            date_px = px_by_date.get(dt, {})
            m = macro_lookup.get(dt, {})
            vix = m.get("vix", 20.0)
            if pd.isna(vix):
                vix = 20.0
            vix3m = m.get("vix3m", m.get("vix_3m", None))
            if vix3m is not None and pd.isna(vix3m):
                vix3m = None

            # ── 1) Manage existing positions ──
            to_remove = []
            for tk, pos in list(self.positions.items()):
                S = date_px.get(tk)
                if S is None or not np.isfinite(S):
                    continue
                chain_idx = ticker_chain_idx.get(tk, {})

                if pos.state == 'csp':
                    dte = (pos.expiry - dt).days
                    if dte <= 0:
                        if S < pos.strike:
                            # ASSIGNED
                            cost = pos.strike * 100 * pos.contracts
                            self.cash -= cost
                            self.assignments += 1
                            self.trades.append(Trade(
                                ticker=tk, sector=pos.sector, leg_type="CSP",
                                open_date=pos.open_date, close_date=dt,
                                expiry=pos.expiry, strike=pos.strike,
                                contracts=pos.contracts,
                                premium_received=pos.premium_received,
                                close_cost=0.0, pnl=pos.premium_received,
                                close_reason="assigned", close_pricing="intrinsic",
                                entry_delta=self.put_delta,
                            ))
                            if self.sell_cc:
                                pos.state = 'shares'
                                pos.cost_basis = pos.strike - pos.premium_received / (100 * pos.contracts)
                                pos.share_purchase_price = pos.strike
                                pos.expiry = dt
                            else:
                                proceeds = S * 100 * pos.contracts
                                self.cash += proceeds
                                loss = (S - pos.strike) * 100 * pos.contracts
                                self.trades.append(Trade(
                                    ticker=tk, sector=pos.sector, leg_type="SHARE_SALE",
                                    open_date=dt, close_date=dt,
                                    expiry=pos.expiry, strike=pos.strike,
                                    contracts=pos.contracts,
                                    premium_received=0, close_cost=0, pnl=loss,
                                    close_reason="immediate_sell", close_pricing="market_sell",
                                    entry_delta=0, shares_pnl=loss,
                                ))
                                to_remove.append(tk)
                        else:
                            self.csp_expired_otm += 1
                            self.trades.append(Trade(
                                ticker=tk, sector=pos.sector, leg_type="CSP",
                                open_date=pos.open_date, close_date=dt,
                                expiry=pos.expiry, strike=pos.strike,
                                contracts=pos.contracts,
                                premium_received=pos.premium_received,
                                close_cost=0.0, pnl=pos.premium_received,
                                close_reason="expired_otm", close_pricing="intrinsic",
                                entry_delta=self.put_delta,
                            ))
                            to_remove.append(tk)
                    else:
                        # Check early close with pricing mode
                        exp_chain = chain_idx.get((dt, pos.expiry))
                        if exp_chain is not None:
                            strike_row = find_strike_by_value(exp_chain, pos.strike, "p")
                            if strike_row is not None:
                                cur_buy = price_buy(float(strike_row["bid"]),
                                                     float(strike_row["ask"]), pm)
                                open_sell = pos.premium_received / (100 * pos.contracts)
                                profit_frac = (open_sell - cur_buy) / max(open_sell, 0.01)
                                should_close = False
                                reason = ""
                                if profit_frac >= PROFIT_TAKE_FRAC:
                                    should_close = True
                                    reason = "profit_take"
                                elif dte <= DTE_CLOSE_TRIGGER:
                                    should_close = True
                                    reason = "dte_close"
                                if should_close:
                                    close_cost = cur_buy * 100 * pos.contracts
                                    self.cash -= close_cost
                                    pnl = pos.premium_received - close_cost
                                    self.trades.append(Trade(
                                        ticker=tk, sector=pos.sector, leg_type="CSP",
                                        open_date=pos.open_date, close_date=dt,
                                        expiry=pos.expiry, strike=pos.strike,
                                        contracts=pos.contracts,
                                        premium_received=pos.premium_received,
                                        close_cost=close_cost, pnl=pnl,
                                        close_reason=reason, close_pricing=f"real_{pm}",
                                        entry_delta=self.put_delta,
                                    ))
                                    to_remove.append(tk)

                elif pos.state == 'cc':
                    dte = (pos.expiry - dt).days
                    if dte <= 0:
                        if S > pos.strike:
                            proceeds = pos.strike * 100 * pos.contracts
                            self.cash += proceeds
                            self.called_away += 1
                            shares_pnl = (pos.strike - pos.share_purchase_price) * 100 * pos.contracts
                            self.trades.append(Trade(
                                ticker=tk, sector=pos.sector, leg_type="CC",
                                open_date=pos.open_date, close_date=dt,
                                expiry=pos.expiry, strike=pos.strike,
                                contracts=pos.contracts,
                                premium_received=pos.premium_received,
                                close_cost=0.0, pnl=pos.premium_received + shares_pnl,
                                close_reason="called_away", close_pricing="intrinsic",
                                entry_delta=self._get_cc_delta(vix),
                                shares_pnl=shares_pnl,
                            ))
                            to_remove.append(tk)
                        else:
                            self.cc_expired_otm += 1
                            self.trades.append(Trade(
                                ticker=tk, sector=pos.sector, leg_type="CC",
                                open_date=pos.open_date, close_date=dt,
                                expiry=pos.expiry, strike=pos.strike,
                                contracts=pos.contracts,
                                premium_received=pos.premium_received,
                                close_cost=0.0, pnl=pos.premium_received,
                                close_reason="expired_otm", close_pricing="intrinsic",
                                entry_delta=self._get_cc_delta(vix),
                            ))
                            pos.state = 'shares'
                            pos.expiry = dt
                    else:
                        exp_chain = chain_idx.get((dt, pos.expiry))
                        if exp_chain is not None:
                            strike_row = find_strike_by_value(exp_chain, pos.strike, "c")
                            if strike_row is not None:
                                cur_buy = price_buy(float(strike_row["bid"]),
                                                     float(strike_row["ask"]), pm)
                                open_sell = pos.premium_received / (100 * pos.contracts)
                                profit_frac = (open_sell - cur_buy) / max(open_sell, 0.01)
                                should_close = False
                                reason = ""
                                if profit_frac >= PROFIT_TAKE_FRAC:
                                    should_close = True
                                    reason = "profit_take"
                                elif dte <= DTE_CLOSE_TRIGGER:
                                    should_close = True
                                    reason = "dte_close"
                                if should_close:
                                    close_cost = cur_buy * 100 * pos.contracts
                                    self.cash -= close_cost
                                    self.trades.append(Trade(
                                        ticker=tk, sector=pos.sector, leg_type="CC",
                                        open_date=pos.open_date, close_date=dt,
                                        expiry=pos.expiry, strike=pos.strike,
                                        contracts=pos.contracts,
                                        premium_received=pos.premium_received,
                                        close_cost=close_cost,
                                        pnl=pos.premium_received - close_cost,
                                        close_reason=reason, close_pricing=f"real_{pm}",
                                        entry_delta=self._get_cc_delta(vix),
                                    ))
                                    pos.state = 'shares'
                                    pos.expiry = dt

                # Stop loss on shares
                if tk not in to_remove and pos.state in ('shares', 'cc') and self.share_stop_loss > 0:
                    S = date_px.get(tk)
                    if (S is not None and np.isfinite(S) and
                            pos.share_purchase_price > 0 and
                            S < pos.share_purchase_price * (1 - self.share_stop_loss)):
                        if pos.state == 'cc':
                            exp_chain = chain_idx.get((dt, pos.expiry))
                            cc_buyback = 0.0
                            pricing = "intrinsic"
                            if exp_chain is not None:
                                strike_row = find_strike_by_value(exp_chain, pos.strike, "c")
                                if strike_row is not None:
                                    cc_buyback = price_buy(float(strike_row["bid"]),
                                                            float(strike_row["ask"]), pm) * 100 * pos.contracts
                                    pricing = f"real_{pm}"
                            if cc_buyback == 0.0:
                                cc_buyback = max(S - pos.strike, 0) * 100 * pos.contracts
                            self.cash -= cc_buyback
                            self.trades.append(Trade(
                                ticker=tk, sector=pos.sector, leg_type="CC",
                                open_date=pos.open_date, close_date=dt,
                                expiry=pos.expiry, strike=pos.strike,
                                contracts=pos.contracts,
                                premium_received=pos.premium_received,
                                close_cost=cc_buyback,
                                pnl=pos.premium_received - cc_buyback,
                                close_reason="stop_loss", close_pricing=pricing,
                                entry_delta=self._get_cc_delta(vix),
                            ))
                        proceeds = S * 100 * pos.contracts
                        self.cash += proceeds
                        shares_pnl = (S - pos.share_purchase_price) * 100 * pos.contracts
                        self.trades.append(Trade(
                            ticker=tk, sector=pos.sector, leg_type="SHARE_SALE",
                            open_date=pos.open_date, close_date=dt,
                            expiry=pos.expiry, strike=pos.share_purchase_price,
                            contracts=pos.contracts,
                            premium_received=0, close_cost=0, pnl=shares_pnl,
                            close_reason="stop_loss", close_pricing="market_sell",
                            entry_delta=0, shares_pnl=shares_pnl,
                        ))
                        self.stop_losses += 1
                        to_remove.append(tk)

            for tk in set(to_remove):
                if tk in self.positions:
                    del self.positions[tk]

            # ── 2) Sell CCs on shares ──
            if self.sell_cc:
                cc_delta = self._get_cc_delta(vix)
                for tk, pos in list(self.positions.items()):
                    if pos.state != 'shares':
                        continue
                    S = date_px.get(tk)
                    if S is None or not np.isfinite(S):
                        continue
                    chain_idx_tk = ticker_chain_idx.get(tk, {})

                    available_exps = [exp for (d, exp) in chain_idx_tk if d == dt]
                    best_row = None
                    best_exp = None
                    best_dte_diff = 999

                    for exp in available_exps:
                        dte = (exp - dt).days
                        if DTE_MIN <= dte <= DTE_MAX:
                            diff = abs(dte - DTE_TARGET)
                            if diff < best_dte_diff:
                                exp_chain = chain_idx_tk.get((dt, exp))
                                if exp_chain is not None:
                                    row = find_best_strike(exp_chain, cc_delta, "c")
                                    if row is not None and float(row["strike"]) > S:
                                        best_row = row
                                        best_exp = exp
                                        best_dte_diff = diff

                    if best_row is not None:
                        cc_price = price_sell(float(best_row["bid"]),
                                              float(best_row["ask"]), pm)
                        premium = cc_price * 100 * pos.contracts
                        self.cash += premium
                        self.total_cc_premium += premium
                        pos.state = 'cc'
                        pos.strike = float(best_row["strike"])
                        pos.premium_received = premium
                        pos.open_date = dt
                        pos.expiry = best_exp
                        self.cc_opened += 1

            # ── 3) Open new CSPs ──
            equity = self._equity(date_px)
            if equity <= 0:
                equity = self.cash
            self.equity_curve.append((dt, equity))

            # VIX gates
            effective_vix_gate = self.vix_gate
            if self.vix_hard_gate is not None and vix > self.vix_hard_gate:
                self.blocked_by_vix += 1
                continue
            if vix > effective_vix_gate:
                self.blocked_by_vix += 1
                continue
            if len(self.positions) >= MAX_CONCURRENT:
                continue

            # Assignment exposure checks (full wheel only)
            if self.sell_cc:
                if self._assigned_exposure(date_px, equity) > self.max_assigned_exposure:
                    continue
                if self._assigned_count() >= self.max_assigned_names:
                    continue

            # Vol-based sizing factor
            size_factor = self._vol_size_factor(vix, vix3m)

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
                if (S * 100) / max(equity, 1) > SINGLE_NAME_CAP:
                    continue

                # Earnings filter
                if self.earnings_filter_days > 0 and earnings_dates is not None:
                    tk_earnings = earnings_dates.get(tk, [])
                    near_earnings = False
                    for ed in tk_earnings:
                        days_to = (ed - dt).days
                        if 0 <= days_to <= self.earnings_filter_days:
                            near_earnings = True
                            break
                    if near_earnings:
                        self.blocked_by_earnings += 1
                        continue

                chain_idx_tk = ticker_chain_idx.get(tk, {})
                available_exps = [exp for (d, exp) in chain_idx_tk if d == dt]
                best_put_row = None
                best_exp = None
                best_dte_diff = 999

                for exp in available_exps:
                    dte = (exp - dt).days
                    if DTE_MIN <= dte <= DTE_MAX:
                        diff = abs(dte - DTE_TARGET)
                        if diff < best_dte_diff:
                            exp_chain = chain_idx_tk.get((dt, exp))
                            if exp_chain is not None:
                                row = find_best_strike(exp_chain, self.put_delta, "p")
                                if row is not None and float(row["strike"]) < S:
                                    best_put_row = row
                                    best_exp = exp
                                    best_dte_diff = diff

                if best_put_row is not None:
                    sell_price = price_sell(float(best_put_row["bid"]),
                                           float(best_put_row["ask"]), pm)
                    if sell_price > 0:
                        candidates.append((tk, S, sector, sell_price,
                                           float(best_put_row["strike"]), best_exp))

            candidates.sort(key=lambda x: x[3] / max(x[4], 1), reverse=True)
            slots = min(MAX_CONCURRENT - len(self.positions),
                        max(1, MAX_CONCURRENT // 5))

            for tk, S, sector, put_price, put_strike, exp in candidates[:slots]:
                max_alloc = SINGLE_NAME_CAP * equity * size_factor
                n = max(1, int(max_alloc // (put_strike * 100)))
                if put_strike * 100 * n > self.cash:
                    n = int(self.cash // (put_strike * 100))
                    if n < 1:
                        continue
                if self._sector_exposure(sector, equity) + (put_strike * 100 * n) / max(equity, 1) > SECTOR_CAP:
                    continue

                premium = put_price * 100 * n
                self.cash += premium
                self.total_csp_premium += premium
                self.positions[tk] = Pos(
                    ticker=tk, sector=sector, state='csp',
                    open_date=dt, expiry=exp, strike=put_strike,
                    contracts=n, premium_received=premium,
                    cost_basis=0, share_purchase_price=0,
                )
                self.csp_opened += 1
                if len(self.positions) >= MAX_CONCURRENT:
                    break

        # Force close remaining
        if all_dates:
            last_dt = all_dates[-1]
            last_px = px_by_date.get(last_dt, {})
            for tk, pos in list(self.positions.items()):
                S = last_px.get(tk, pos.strike)
                if pos.state == 'csp':
                    intrinsic = max(pos.strike - S, 0)
                    close_cost = intrinsic * 100 * pos.contracts
                    self.trades.append(Trade(
                        ticker=tk, sector=pos.sector, leg_type="CSP",
                        open_date=pos.open_date, close_date=last_dt,
                        expiry=pos.expiry, strike=pos.strike,
                        contracts=pos.contracts,
                        premium_received=pos.premium_received,
                        close_cost=close_cost,
                        pnl=pos.premium_received - close_cost,
                        close_reason="end_of_data", close_pricing="intrinsic",
                        entry_delta=self.put_delta,
                    ))
                elif pos.state in ('shares', 'cc'):
                    if pos.state == 'cc':
                        intrinsic = max(S - pos.strike, 0)
                        cc_close = intrinsic * 100 * pos.contracts
                        self.cash -= cc_close
                        self.trades.append(Trade(
                            ticker=tk, sector=pos.sector, leg_type="CC",
                            open_date=pos.open_date, close_date=last_dt,
                            expiry=pos.expiry, strike=pos.strike,
                            contracts=pos.contracts,
                            premium_received=pos.premium_received,
                            close_cost=cc_close,
                            pnl=pos.premium_received - cc_close,
                            close_reason="end_of_data", close_pricing="intrinsic",
                            entry_delta=self.call_delta,
                        ))
                    proceeds = S * 100 * pos.contracts
                    self.cash += proceeds
                    self.trades.append(Trade(
                        ticker=tk, sector=pos.sector, leg_type="SHARE_SALE",
                        open_date=pos.open_date, close_date=last_dt,
                        expiry=pos.expiry, strike=pos.share_purchase_price,
                        contracts=pos.contracts,
                        premium_received=0, close_cost=0,
                        pnl=(S - pos.share_purchase_price) * 100 * pos.contracts,
                        close_reason="end_of_data", close_pricing="market_sell",
                        entry_delta=0,
                        shares_pnl=(S - pos.share_purchase_price) * 100 * pos.contracts,
                    ))
            final_eq = self._equity(last_px)
            self.equity_curve.append((last_dt, final_eq))

        return {
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "csp_opened": self.csp_opened, "cc_opened": self.cc_opened,
            "assignments": self.assignments, "called_away": self.called_away,
            "csp_expired_otm": self.csp_expired_otm,
            "cc_expired_otm": self.cc_expired_otm,
            "stop_losses": self.stop_losses,
            "total_csp_premium": self.total_csp_premium,
            "total_cc_premium": self.total_cc_premium,
            "blocked_by_vix": self.blocked_by_vix,
            "blocked_by_earnings": self.blocked_by_earnings,
            "vol_sizing_adjustments": self.vol_sizing_adjustments,
        }


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(eq_curve, starting_cash):
    df = pd.DataFrame(eq_curve, columns=["date", "equity"])
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    eq = df["equity"].astype(float)
    rets = eq.pct_change().fillna(0)
    years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    if years <= 0 or eq.iloc[0] <= 0:
        return {}
    final = float(eq.iloc[-1])
    cagr = (final / starting_cash) ** (1.0 / years) - 1.0
    mu, sd = rets.mean(), rets.std()
    ds = rets[rets < 0].std()
    sharpe = (mu / sd) * math.sqrt(TRADING_DAYS) if sd > 0 else 0
    sortino = (mu / ds) * math.sqrt(TRADING_DAYS) if ds > 0 else 0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0
    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino,
            "max_dd": max_dd, "calmar": calmar, "final_equity": final,
            "years": years, "total_return_pct": (final / starting_cash - 1) * 100}


def regime_gate(eq_curve, spy_close):
    df = pd.DataFrame(eq_curve, columns=["date", "equity"])
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    df["ret"] = df["equity"].pct_change().fillna(0)
    rets = df.set_index("date")["ret"]
    spy_ret = spy_close.pct_change()
    labels = pd.Series("flat", index=spy_ret.index)
    labels[spy_ret > 0.002] = "green"
    labels[spy_ret < -0.002] = "red"
    aligned = labels.reindex(rets.index)
    out = {}
    for regime in ("green", "red", "flat"):
        sub = rets[aligned == regime].dropna()
        out[f"n_{regime}"] = len(sub)
        if len(sub) >= 2 and sub.std() > 0:
            out[f"sharpe_{regime}"] = float(sub.mean() / sub.std() * math.sqrt(TRADING_DAYS))
        else:
            out[f"sharpe_{regime}"] = float("nan")
    sg, sr = out.get("sharpe_green", 0), out.get("sharpe_red", 0)
    if np.isfinite(sg) and np.isfinite(sr):
        denom = max(abs(sg), abs(sr), 1e-9)
        out["regime_gap"] = abs(sg - sr) / denom
        out["r1_pass"] = out["regime_gap"] <= 0.50
    else:
        out["regime_gap"] = float("nan")
        out["r1_pass"] = False
    return out


# ─── Spread Analysis (Test 3) ───────────────────────────────────────────────

def spread_analysis(chains: Dict[str, pd.DataFrame]) -> dict:
    """Analyze bid-ask spreads across all tickers to assess Dolt data quality."""
    results = {}
    all_spreads = []

    for ticker, df in chains.items():
        puts = df[df["type"] == "p"].copy()
        if puts.empty:
            continue
        puts["spread"] = puts["ask"] - puts["bid"]
        puts["spread_pct"] = puts["spread"] / ((puts["bid"] + puts["ask"]) / 2.0) * 100.0

        # Filter to ~25-delta puts (the ones we'd trade)
        tradeable = puts[(puts["delta"].abs() > 0.15) & (puts["delta"].abs() < 0.40)]
        if tradeable.empty:
            tradeable = puts

        avg_spread_pct = float(tradeable["spread_pct"].median())
        avg_spread_usd = float(tradeable["spread"].median())
        min_spread_pct = float(tradeable["spread_pct"].quantile(0.10))
        max_spread_pct = float(tradeable["spread_pct"].quantile(0.90))

        results[ticker] = {
            "median_spread_pct": avg_spread_pct,
            "median_spread_usd": avg_spread_usd,
            "p10_spread_pct": min_spread_pct,
            "p90_spread_pct": max_spread_pct,
            "n_quotes": len(tradeable),
        }
        all_spreads.extend(tradeable["spread_pct"].dropna().tolist())

    all_spreads = np.array(all_spreads)
    summary = {
        "per_ticker": results,
        "overall": {
            "median_spread_pct": float(np.median(all_spreads)),
            "mean_spread_pct": float(np.mean(all_spreads)),
            "p10_spread_pct": float(np.percentile(all_spreads, 10)),
            "p25_spread_pct": float(np.percentile(all_spreads, 25)),
            "p75_spread_pct": float(np.percentile(all_spreads, 75)),
            "p90_spread_pct": float(np.percentile(all_spreads, 90)),
            "n_total_quotes": len(all_spreads),
            "n_tickers": len(results),
        },
    }

    # Classify tickers by liquidity
    tight = [t for t, v in results.items() if v["median_spread_pct"] < 5.0]
    moderate = [t for t, v in results.items() if 5.0 <= v["median_spread_pct"] < 10.0]
    wide = [t for t, v in results.items() if v["median_spread_pct"] >= 10.0]
    summary["liquidity_groups"] = {
        "tight_spread_lt5pct": sorted(tight),
        "moderate_spread_5_10pct": sorted(moderate),
        "wide_spread_gt10pct": sorted(wide),
    }

    return summary


# ─── Build synthetic earnings calendar ───────────────────────────────────────

def build_earnings_calendar(prices: pd.DataFrame, chains: Dict[str, pd.DataFrame]) -> Dict[str, list]:
    """
    Build approximate earnings dates from price data.
    Earnings cause big moves — detect days with >5% gap for each ticker.
    This is a rough proxy when we don't have actual earnings calendar data.
    """
    earnings = {}
    for ticker in chains.keys():
        tk_prices = prices[prices["ticker"] == ticker].sort_values("date") if "ticker" in prices.columns else pd.DataFrame()
        if tk_prices.empty or len(tk_prices) < 5:
            # Use quarterly approximation: Jan/Apr/Jul/Oct
            dates = []
            for year in range(2019, 2025):
                for month in [1, 4, 7, 10]:
                    dates.append(pd.Timestamp(year, month, 25))
            earnings[ticker] = dates
            continue

        tk_prices = tk_prices.copy()
        tk_prices["ret"] = tk_prices["close"].pct_change().abs()
        # Big moves (>5% daily) are likely earnings
        big_moves = tk_prices[tk_prices["ret"] > 0.05]
        if len(big_moves) > 0:
            earnings[ticker] = list(big_moves["date"])
        else:
            # Fallback to quarterly
            dates = []
            for year in range(2019, 2025):
                for month in [1, 4, 7, 10]:
                    dates.append(pd.Timestamp(year, month, 25))
            earnings[ticker] = dates

    return earnings


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 100)
    print("V5 MID-PRICE RETEST — Finding the honest truth")
    print("  Test 1: Mid-Price Execution")
    print("  Test 2: Conservative Mid (40% toward mid)")
    print("  Test 3: Bid-Ask Spread Analysis")
    print("  Test 4: V5 Risk Controls Impact")
    print("=" * 100)

    # ── Load data ──
    print("\n[1] Loading data...")
    chains = load_all_chains()
    print(f"  Loaded {len(chains)} tickers with chain data")

    prices = pd.read_parquet(CACHE / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    macro = pd.read_parquet(CACHE / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"])
    universe = pd.read_parquet(CACHE / "universe.parquet")
    fundamentals = pd.read_parquet(CACHE / "fundamentals.parquet")

    try:
        etf = pd.read_parquet(CACHE / "sector_etfs.parquet")
        spy = etf[etf["ticker"] == "SPY"][["date", "close"]].copy()
        spy["date"] = pd.to_datetime(spy["date"])
        spy_close = spy.set_index("date")["close"].sort_index()
    except Exception:
        spy_close = None

    START = "2019-03-01"
    END = "2024-12-31"

    # Build earnings calendar
    earnings_dates = build_earnings_calendar(prices, chains)
    print(f"  Built earnings calendar for {len(earnings_dates)} tickers")

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 3: Spread Analysis (run first for context)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("TEST 3: BID-ASK SPREAD ANALYSIS")
    print("=" * 100)
    spread_info = spread_analysis(chains)
    ov = spread_info["overall"]
    print(f"  Overall median spread: {ov['median_spread_pct']:.1f}%")
    print(f"  Overall mean spread:   {ov['mean_spread_pct']:.1f}%")
    print(f"  P10-P90 range:         {ov['p10_spread_pct']:.1f}% - {ov['p90_spread_pct']:.1f}%")
    print(f"  Total quotes analyzed: {ov['n_total_quotes']:,}")

    lg = spread_info["liquidity_groups"]
    print(f"\n  Tight (<5%):    {len(lg['tight_spread_lt5pct'])} tickers — {', '.join(lg['tight_spread_lt5pct'][:15])}")
    print(f"  Moderate (5-10%): {len(lg['moderate_spread_5_10pct'])} tickers — {', '.join(lg['moderate_spread_5_10pct'][:15])}")
    print(f"  Wide (>10%):    {len(lg['wide_spread_gt10pct'])} tickers — {', '.join(lg['wide_spread_gt10pct'][:15])}")

    # What mid-price improvement means
    mid_improvement = ov['median_spread_pct'] / 2.0
    print(f"\n  Mid-price vs bid: ~{mid_improvement:.1f}% better entry per trade")
    print(f"  At typical {ov['median_spread_pct']:.1f}% spread, worst-case (bid) costs you "
          f"~{ov['median_spread_pct']/2:.1f}% vs mid on every entry AND exit")
    rh_typical = "1-3%"
    print(f"  Robinhood typical: {rh_typical} for liquid options")
    if ov['median_spread_pct'] > 5.0:
        print(f"  VERDICT: Dolt spreads ({ov['median_spread_pct']:.1f}%) are WIDER than typical Robinhood "
              f"({rh_typical}) — end-of-day snapshot penalty likely")
    else:
        print(f"  VERDICT: Dolt spreads ({ov['median_spread_pct']:.1f}%) are in line with Robinhood")

    # ─────────────────────────────────────────────────────────────────────────
    # TESTS 1, 2, 4: Engine runs
    # ─────────────────────────────────────────────────────────────────────────

    configs = {
        # ── BASELINE: Original worst-case (bid/ask) ──
        "BASELINE: Bid/Ask CSP-Only": {
            "put_delta": 0.25, "call_delta": 0.25,
            "vix_gate": 28.0, "share_stop_loss": 0.0,
            "max_assigned_exposure": 1.0, "max_assigned_names": 99,
            "sell_cc": False, "dynamic_cc": False,
            "pricing_mode": "bid",
        },
        "BASELINE: Bid/Ask Full Wheel": {
            "put_delta": 0.25, "call_delta": 0.25,
            "vix_gate": 28.0, "share_stop_loss": 0.10,
            "max_assigned_exposure": 0.30, "max_assigned_names": 5,
            "sell_cc": True, "dynamic_cc": False,
            "pricing_mode": "bid",
        },
        # ── TEST 1: Mid-Price ──
        "T1: Mid-Price CSP-Only": {
            "put_delta": 0.25, "call_delta": 0.25,
            "vix_gate": 28.0, "share_stop_loss": 0.0,
            "max_assigned_exposure": 1.0, "max_assigned_names": 99,
            "sell_cc": False, "dynamic_cc": False,
            "pricing_mode": "mid",
        },
        "T1: Mid-Price Full Wheel": {
            "put_delta": 0.25, "call_delta": 0.25,
            "vix_gate": 28.0, "share_stop_loss": 0.10,
            "max_assigned_exposure": 0.30, "max_assigned_names": 5,
            "sell_cc": True, "dynamic_cc": False,
            "pricing_mode": "mid",
        },
        # ── TEST 2: Conservative Mid ──
        "T2: Conservative Mid CSP-Only": {
            "put_delta": 0.25, "call_delta": 0.25,
            "vix_gate": 28.0, "share_stop_loss": 0.0,
            "max_assigned_exposure": 1.0, "max_assigned_names": 99,
            "sell_cc": False, "dynamic_cc": False,
            "pricing_mode": "conservative_mid",
        },
        "T2: Conservative Mid Full Wheel": {
            "put_delta": 0.25, "call_delta": 0.25,
            "vix_gate": 28.0, "share_stop_loss": 0.10,
            "max_assigned_exposure": 0.30, "max_assigned_names": 5,
            "sell_cc": True, "dynamic_cc": False,
            "pricing_mode": "conservative_mid",
        },
        # ── TEST 4: V5 Risk Controls (on mid-price) ──
        "T4: Mid + VIX Gate 30": {
            "put_delta": 0.25, "call_delta": 0.25,
            "vix_gate": 30.0, "vix_hard_gate": 30.0,
            "share_stop_loss": 0.10,
            "max_assigned_exposure": 0.30, "max_assigned_names": 5,
            "sell_cc": True, "dynamic_cc": True,
            "pricing_mode": "mid",
        },
        "T4: Mid + Full V5 Controls CSP": {
            "put_delta": 0.25, "call_delta": 0.25,
            "vix_gate": 30.0, "vix_hard_gate": 30.0,
            "share_stop_loss": 0.0,
            "max_assigned_exposure": 1.0, "max_assigned_names": 99,
            "sell_cc": False, "dynamic_cc": False,
            "pricing_mode": "mid",
            "earnings_filter_days": 7,
            "dynamic_vol_sizing": True,
            "vix_term_structure_sizing": True,
        },
        "T4: Mid + Full V5 Controls Wheel": {
            "put_delta": 0.25, "call_delta": 0.25,
            "vix_gate": 30.0, "vix_hard_gate": 30.0,
            "share_stop_loss": 0.10,
            "max_assigned_exposure": 0.30, "max_assigned_names": 5,
            "sell_cc": True, "dynamic_cc": True,
            "pricing_mode": "mid",
            "earnings_filter_days": 7,
            "dynamic_vol_sizing": True,
            "vix_term_structure_sizing": True,
        },
        "T4: ConsMid + Full V5 Controls Wheel": {
            "put_delta": 0.25, "call_delta": 0.25,
            "vix_gate": 30.0, "vix_hard_gate": 30.0,
            "share_stop_loss": 0.10,
            "max_assigned_exposure": 0.30, "max_assigned_names": 5,
            "sell_cc": True, "dynamic_cc": True,
            "pricing_mode": "conservative_mid",
            "earnings_filter_days": 7,
            "dynamic_vol_sizing": True,
            "vix_term_structure_sizing": True,
        },
    }

    all_results = {}
    for i, (name, cfg) in enumerate(configs.items()):
        print(f"\n[{i+2}] Running {name}...")
        engine = MidPriceWheelEngine(
            put_delta=cfg["put_delta"], call_delta=cfg["call_delta"],
            vix_gate=cfg["vix_gate"], share_stop_loss=cfg["share_stop_loss"],
            max_assigned_exposure=cfg["max_assigned_exposure"],
            max_assigned_names=cfg["max_assigned_names"],
            sell_cc=cfg["sell_cc"], dynamic_cc=cfg["dynamic_cc"],
            pricing_mode=cfg["pricing_mode"],
            vix_hard_gate=cfg.get("vix_hard_gate"),
            vix_term_structure_sizing=cfg.get("vix_term_structure_sizing", False),
            earnings_filter_days=cfg.get("earnings_filter_days", 0),
            dynamic_vol_sizing=cfg.get("dynamic_vol_sizing", False),
        )
        result = engine.run(chains, prices, macro, universe, fundamentals,
                            earnings_dates=earnings_dates, start=START, end=END)
        metrics = compute_metrics(result["equity_curve"], STARTING_CASH)

        trades = result["trades"]
        csp_t = [t for t in trades if t.leg_type == "CSP"]
        cc_t = [t for t in trades if t.leg_type == "CC"]
        share_t = [t for t in trades if t.leg_type == "SHARE_SALE"]
        n = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in trades)
        csp_pnl = sum(t.pnl for t in csp_t)
        cc_pnl = sum(t.pnl for t in cc_t)
        share_pnl = sum(t.pnl for t in share_t)
        wr = wins / n if n > 0 else 0
        pg = sum(t.pnl for t in trades if t.pnl > 0)
        pl = abs(sum(t.pnl for t in trades if t.pnl < 0))
        pf = pg / pl if pl > 0 else float("inf")
        total_prem = engine.total_csp_premium + engine.total_cc_premium

        regime = regime_gate(result["equity_curve"], spy_close) if spy_close is not None else {}

        all_results[name] = {
            "metrics": metrics,
            "trade_stats": {
                "total_trades": n, "csp_trades": len(csp_t), "cc_trades": len(cc_t),
                "share_trades": len(share_t), "wins": wins, "wr": wr, "pf": pf,
                "total_pnl": total_pnl, "csp_pnl": csp_pnl, "cc_pnl": cc_pnl,
                "share_pnl": share_pnl,
                "csp_premium": engine.total_csp_premium,
                "cc_premium": engine.total_cc_premium,
                "assignments": result["assignments"],
                "called_away": result["called_away"],
                "stop_losses": result["stop_losses"],
                "csp_opened": result["csp_opened"], "cc_opened": result["cc_opened"],
                "blocked_by_vix": result["blocked_by_vix"],
                "blocked_by_earnings": result["blocked_by_earnings"],
                "vol_sizing_adjustments": result["vol_sizing_adjustments"],
            },
            "regime": regime,
            "config": cfg,
        }

        m = metrics
        if m:
            print(f"  CAGR: {m['cagr']*100:+.1f}%  Sharpe: {m['sharpe']:.2f}  "
                  f"Sortino: {m['sortino']:.2f}  MaxDD: {m['max_dd']*100:.1f}%  "
                  f"Calmar: {m['calmar']:.2f}")
            print(f"  WR: {wr*100:.0f}%  PF: {pf:.2f}  Total PnL: ${total_pnl:+,.0f}")
            if result["blocked_by_vix"] > 0 or result["blocked_by_earnings"] > 0:
                print(f"  Blocked: VIX={result['blocked_by_vix']}  Earnings={result['blocked_by_earnings']}  "
                      f"VolAdj={result['vol_sizing_adjustments']}")

    # ─────────────────────────────────────────────────────────────────────────
    # COMPARATIVE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 150)
    print("COMPARATIVE SUMMARY — MID-PRICE RETEST")
    print("=" * 150)
    print(f"{'Variant':<44} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Calmar':>7} "
          f"{'WR%':>5} {'PF':>5} {'Total$':>10} {'Pricing':>12}")
    print("-" * 150)
    for name, r in all_results.items():
        m = r["metrics"]
        ts = r["trade_stats"]
        pricing = r["config"]["pricing_mode"]
        if m:
            print(f"{name:<44} {m['cagr']*100:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
                  f"{m['max_dd']*100:>6.1f}% {m['calmar']:>7.2f} "
                  f"{ts['wr']*100:>4.0f}% {ts['pf']:>5.2f} "
                  f"${ts['total_pnl']:>+9,.0f} {pricing:>12}")

    # ── Regime Gate ──
    print(f"\n{'='*100}")
    print("REGIME GATE (HC #428 R1)")
    print(f"{'='*100}")
    for name, r in all_results.items():
        rg = r["regime"]
        if rg:
            sg = rg.get('sharpe_green', float('nan'))
            sr = rg.get('sharpe_red', float('nan'))
            sf = rg.get('sharpe_flat', float('nan'))
            gap = rg.get('regime_gap', float('nan'))
            r1 = rg.get('r1_pass', False)
            print(f"  {name:<44} Green={sg:>6.2f}  Red={sr:>6.2f}  Flat={sf:>6.2f}  "
                  f"Gap={gap:.3f}  {'PASS' if r1 else 'FAIL'}")

    # ── KEY COMPARISONS ──
    print(f"\n{'='*100}")
    print("KEY COMPARISONS — PRICING IMPACT")
    print(f"{'='*100}")

    pairs = [
        ("BASELINE: Bid/Ask CSP-Only", "T1: Mid-Price CSP-Only", "CSP-Only: Bid vs Mid"),
        ("BASELINE: Bid/Ask CSP-Only", "T2: Conservative Mid CSP-Only", "CSP-Only: Bid vs Cons.Mid"),
        ("BASELINE: Bid/Ask Full Wheel", "T1: Mid-Price Full Wheel", "Full Wheel: Bid vs Mid"),
        ("BASELINE: Bid/Ask Full Wheel", "T2: Conservative Mid Full Wheel", "Full Wheel: Bid vs Cons.Mid"),
        ("T1: Mid-Price CSP-Only", "T4: Mid + Full V5 Controls CSP", "CSP Mid: Bare vs V5 Controls"),
        ("T1: Mid-Price Full Wheel", "T4: Mid + Full V5 Controls Wheel", "Wheel Mid: Bare vs V5 Controls"),
    ]

    for base_name, comp_name, label in pairs:
        if base_name in all_results and comp_name in all_results:
            bm = all_results[base_name]["metrics"]
            cm = all_results[comp_name]["metrics"]
            if bm and cm:
                delta_cagr = (cm["cagr"] - bm["cagr"]) * 100
                delta_sharpe = cm["sharpe"] - bm["sharpe"]
                delta_dd = (cm["max_dd"] - bm["max_dd"]) * 100
                print(f"  {label:<45} CAGR: {delta_cagr:+.1f}pp  "
                      f"Sharpe: {delta_sharpe:+.2f}  MaxDD: {delta_dd:+.1f}pp")

    # ── VERDICT ──
    print(f"\n{'='*100}")
    print("VERDICT")
    print(f"{'='*100}")

    # Find the most relevant configs for verdict
    mid_csp = all_results.get("T1: Mid-Price CSP-Only", {}).get("metrics", {})
    mid_wheel = all_results.get("T1: Mid-Price Full Wheel", {}).get("metrics", {})
    v5_csp = all_results.get("T4: Mid + Full V5 Controls CSP", {}).get("metrics", {})
    v5_wheel = all_results.get("T4: Mid + Full V5 Controls Wheel", {}).get("metrics", {})
    cons_v5 = all_results.get("T4: ConsMid + Full V5 Controls Wheel", {}).get("metrics", {})
    baseline_csp = all_results.get("BASELINE: Bid/Ask CSP-Only", {}).get("metrics", {})

    print(f"\n  Worst-case (bid/ask, no controls): {baseline_csp.get('cagr', 0)*100:+.1f}% CAGR")
    print(f"  Mid-price CSP-only:                {mid_csp.get('cagr', 0)*100:+.1f}% CAGR")
    print(f"  Mid-price Full Wheel:              {mid_wheel.get('cagr', 0)*100:+.1f}% CAGR")
    print(f"  Mid + V5 Controls (CSP):           {v5_csp.get('cagr', 0)*100:+.1f}% CAGR")
    print(f"  Mid + V5 Controls (Wheel):         {v5_wheel.get('cagr', 0)*100:+.1f}% CAGR")
    print(f"  Conservative Mid + V5 (Wheel):     {cons_v5.get('cagr', 0)*100:+.1f}% CAGR")

    best_realistic = cons_v5 if cons_v5 else v5_wheel
    best_cagr = best_realistic.get('cagr', 0) * 100
    best_sharpe = best_realistic.get('sharpe', 0)

    print(f"\n  MOST REALISTIC estimate (conservative mid + V5 controls):")
    print(f"    CAGR: {best_cagr:+.1f}%  Sharpe: {best_sharpe:.2f}")
    print(f"    MaxDD: {best_realistic.get('max_dd', 0)*100:.1f}%")

    if best_cagr < 5:
        print(f"\n  HONEST ASSESSMENT: Strategy is marginal at best ({best_cagr:.1f}% CAGR).")
        print(f"  Not worth the complexity and assignment risk.")
    elif best_cagr < 10:
        print(f"\n  HONEST ASSESSMENT: Modest but real return ({best_cagr:.1f}% CAGR).")
        print(f"  Competitive with high-yield bonds, but with more risk.")
    elif best_cagr < 15:
        print(f"\n  HONEST ASSESSMENT: Decent strategy ({best_cagr:.1f}% CAGR).")
        print(f"  Beats bonds, approaches equity returns with lower vol.")
    else:
        print(f"\n  HONEST ASSESSMENT: Strong strategy ({best_cagr:.1f}% CAGR).")
        print(f"  Competitive with equities on a risk-adjusted basis.")

    # ── Save results ──
    save = {}
    for name, r in all_results.items():
        save[name] = {
            "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                       for k, v in r["metrics"].items()},
            "trade_stats": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                           for k, v in r["trade_stats"].items()},
            "regime": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                      for k, v in r["regime"].items()},
            "config": r["config"],
        }
    save["spread_analysis"] = {
        "overall": spread_info["overall"],
        "liquidity_groups": spread_info["liquidity_groups"],
    }

    with open(OUT_DIR / "midprice_results.json", "w") as f:
        json.dump(save, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*100}")
    print(f"COMPLETE in {elapsed:.0f}s. Results saved.")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
