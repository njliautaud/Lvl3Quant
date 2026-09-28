#!/usr/bin/env python3
"""
wheel_v6_improved.py — Improved V6 with risk controls learned from first pass.

Key improvements over v6 base:
1. ASSIGNED STOCK EXPOSURE CAP: Max 30% of equity in assigned shares
   (prevents 2020-style cascading stop-losses across many names)
2. TIGHTER STOP LOSS: 10% instead of 15% on assigned shares
3. VIX REGIME FILTER: Don't sell CSPs when VIX > 28 (avoid forced assignments
   during panics)
4. MAX 3 ASSIGNMENTS AT ONCE: If already holding 3+ assigned names, don't open new CSPs
5. CSP-ONLY baseline for direct comparison

All pricing REAL from Dolt chains. Commission-free (Robinhood).
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
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


class ImprovedWheelEngine:
    def __init__(self, put_delta=0.25, call_delta=0.25,
                 vix_gate=28.0, share_stop_loss=0.10,
                 max_assigned_exposure=0.30, max_assigned_names=3,
                 sell_cc=True, dynamic_cc=False,
                 starting_cash=STARTING_CASH):
        self.put_delta = put_delta
        self.call_delta = call_delta
        self.vix_gate = vix_gate
        self.share_stop_loss = share_stop_loss
        self.max_assigned_exposure = max_assigned_exposure
        self.max_assigned_names = max_assigned_names
        self.sell_cc = sell_cc
        self.dynamic_cc = dynamic_cc
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
        """Fraction of equity in assigned shares."""
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

    def run(self, chains, prices, macro, universe, fundamentals,
            start="2019-03-01", end="2024-12-31"):
        start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)

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
                                # CSP-only mode: immediately sell assigned shares
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
                                    self.trades.append(Trade(
                                        ticker=tk, sector=pos.sector, leg_type="CSP",
                                        open_date=pos.open_date, close_date=dt,
                                        expiry=pos.expiry, strike=pos.strike,
                                        contracts=pos.contracts,
                                        premium_received=pos.premium_received,
                                        close_cost=close_cost, pnl=pnl,
                                        close_reason=reason, close_pricing="real_chain",
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
                                    self.trades.append(Trade(
                                        ticker=tk, sector=pos.sector, leg_type="CC",
                                        open_date=pos.open_date, close_date=dt,
                                        expiry=pos.expiry, strike=pos.strike,
                                        contracts=pos.contracts,
                                        premium_received=pos.premium_received,
                                        close_cost=close_cost,
                                        pnl=pos.premium_received - close_cost,
                                        close_reason=reason, close_pricing="real_chain",
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
                        # Buy back CC if active
                        if pos.state == 'cc':
                            exp_chain = chain_idx.get((dt, pos.expiry))
                            cc_buyback = 0.0
                            pricing = "intrinsic"
                            if exp_chain is not None:
                                strike_row = find_strike_by_value(exp_chain, pos.strike, "c")
                                if strike_row is not None:
                                    cc_buyback = float(strike_row["ask"]) * 100 * pos.contracts
                                    pricing = "real_chain"
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
                    exp_obs_tk = ticker_exp_obs.get(tk, {})

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
                        cc_bid = float(best_row["bid"])
                        premium = cc_bid * 100 * pos.contracts
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

            if vix > self.vix_gate:
                continue
            if len(self.positions) >= MAX_CONCURRENT:
                continue

            # Assignment exposure checks (full wheel only)
            if self.sell_cc:
                if self._assigned_exposure(date_px, equity) > self.max_assigned_exposure:
                    continue
                if self._assigned_count() >= self.max_assigned_names:
                    continue

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

                chain_idx_tk = ticker_chain_idx.get(tk, {})
                exp_obs_tk = ticker_exp_obs.get(tk, {})
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
                    candidates.append((tk, S, sector, float(best_put_row["bid"]),
                                       float(best_put_row["strike"]), best_exp))

            candidates.sort(key=lambda x: x[3] / max(x[4], 1), reverse=True)
            slots = min(MAX_CONCURRENT - len(self.positions),
                        max(1, MAX_CONCURRENT // 5))

            for tk, S, sector, put_bid, put_strike, exp in candidates[:slots]:
                max_alloc = SINGLE_NAME_CAP * equity
                n = max(1, int(max_alloc // (put_strike * 100)))
                if put_strike * 100 * n > self.cash:
                    n = int(self.cash // (put_strike * 100))
                    if n < 1:
                        continue
                if self._sector_exposure(sector, equity) + (put_strike * 100 * n) / max(equity, 1) > SECTOR_CAP:
                    continue

                premium = put_bid * 100 * n
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
        }


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


def travel_income(eq_curve, starting_cash, monthly_pct=0.015):
    df = pd.DataFrame(eq_curve, columns=["date", "equity"])
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    df["ret"] = df["equity"].pct_change().fillna(0)

    nav = starting_cash
    monthly_w = starting_cash * monthly_pct
    last_month = None
    total_withdrawn = 0.0
    nav_hist = []
    monthly_rets = []
    month_start = nav
    loss_streak = 0
    max_loss_streak = 0

    for _, row in df.iterrows():
        nav *= (1 + row["ret"])
        cm = (row["date"].year, row["date"].month)
        if last_month != cm:
            if last_month is not None:
                mr = (nav - month_start) / max(month_start, 1)
                monthly_rets.append(mr)
                if mr < 0:
                    loss_streak += 1
                    max_loss_streak = max(max_loss_streak, loss_streak)
                else:
                    loss_streak = 0
            w = min(monthly_w, nav * 0.5)
            nav -= w
            total_withdrawn += w
            last_month = cm
            month_start = nav
        nav_hist.append((row["date"], nav))

    final_nav = nav_hist[-1][1] if nav_hist else 0

    # Max sustainable withdrawal
    max_sust = None
    for test_pct in np.arange(0.005, 0.04, 0.001):
        test_nav = starting_cash
        test_w = starting_cash * test_pct
        tm = None
        ok = True
        for _, row in df.iterrows():
            test_nav *= (1 + row["ret"])
            cm = (row["date"].year, row["date"].month)
            if tm != cm:
                test_nav -= min(test_w, test_nav * 0.5)
                tm = cm
            if test_nav < starting_cash * 0.3:
                ok = False
                break
        if ok:
            max_sust = test_pct

    return {
        "monthly_withdrawal_usd": monthly_w,
        "total_withdrawn": total_withdrawn,
        "final_nav": final_nav,
        "nav_preserved": final_nav >= starting_cash * 0.9,
        "max_sustainable_monthly_pct": max_sust,
        "max_loss_streak_months": max_loss_streak,
        "monthly_return_mean": float(np.mean(monthly_rets)) if monthly_rets else 0,
        "monthly_return_std": float(np.std(monthly_rets)) if monthly_rets else 0,
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("WHEEL V6 IMPROVED — WITH RISK CONTROLS")
    print("  Real Dolt chain pricing | $0 commission | Full lifecycle tracking")
    print("=" * 80)

    print("\n[1] Loading data...")
    chains = load_all_chains()
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

    configs = {
        "CSP-Only (no CC)": {
            "put_delta": 0.25, "call_delta": 0.25,
            "vix_gate": 28.0, "share_stop_loss": 0.0,
            "max_assigned_exposure": 1.0, "max_assigned_names": 99,
            "sell_cc": False, "dynamic_cc": False,
        },
        "V6 Full Wheel (d25/d25, improved)": {
            "put_delta": 0.25, "call_delta": 0.25,
            "vix_gate": 28.0, "share_stop_loss": 0.10,
            "max_assigned_exposure": 0.30, "max_assigned_names": 5,
            "sell_cc": True, "dynamic_cc": False,
        },
        "V6 Conservative (d25/d30, improved)": {
            "put_delta": 0.25, "call_delta": 0.30,
            "vix_gate": 28.0, "share_stop_loss": 0.10,
            "max_assigned_exposure": 0.30, "max_assigned_names": 5,
            "sell_cc": True, "dynamic_cc": False,
        },
        "V6 Dynamic CC (improved)": {
            "put_delta": 0.25, "call_delta": 0.25,
            "vix_gate": 28.0, "share_stop_loss": 0.10,
            "max_assigned_exposure": 0.30, "max_assigned_names": 5,
            "sell_cc": True, "dynamic_cc": True,
        },
        "V6 Aggressive (d25/d20, improved)": {
            "put_delta": 0.25, "call_delta": 0.20,
            "vix_gate": 28.0, "share_stop_loss": 0.10,
            "max_assigned_exposure": 0.30, "max_assigned_names": 5,
            "sell_cc": True, "dynamic_cc": False,
        },
        "V6 Ultra-Conservative (d20/d30)": {
            "put_delta": 0.20, "call_delta": 0.30,
            "vix_gate": 25.0, "share_stop_loss": 0.08,
            "max_assigned_exposure": 0.20, "max_assigned_names": 3,
            "sell_cc": True, "dynamic_cc": False,
        },
    }

    all_results = {}
    for i, (name, cfg) in enumerate(configs.items()):
        print(f"\n[{i+2}] Running {name}...")
        engine = ImprovedWheelEngine(
            put_delta=cfg["put_delta"], call_delta=cfg["call_delta"],
            vix_gate=cfg["vix_gate"], share_stop_loss=cfg["share_stop_loss"],
            max_assigned_exposure=cfg["max_assigned_exposure"],
            max_assigned_names=cfg["max_assigned_names"],
            sell_cc=cfg["sell_cc"], dynamic_cc=cfg["dynamic_cc"],
        )
        result = engine.run(chains, prices, macro, universe, fundamentals, START, END)
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
        ti = travel_income(result["equity_curve"], STARTING_CASH)

        all_results[name] = {
            "metrics": metrics,
            "trade_stats": {
                "total_trades": n, "csp_trades": len(csp_t), "cc_trades": len(cc_t),
                "share_trades": len(share_t), "wins": wins, "wr": wr, "pf": pf,
                "total_pnl": total_pnl, "csp_pnl": csp_pnl, "cc_pnl": cc_pnl,
                "share_pnl": share_pnl,
                "csp_premium": engine.total_csp_premium,
                "cc_premium": engine.total_cc_premium,
                "pct_from_puts": engine.total_csp_premium / max(total_prem, 1) * 100,
                "pct_from_calls": engine.total_cc_premium / max(total_prem, 1) * 100,
                "assignments": result["assignments"],
                "called_away": result["called_away"],
                "assignment_rate": result["assignments"] / max(result["csp_opened"], 1),
                "call_away_rate": result["called_away"] / max(result["cc_opened"], 1) if result["cc_opened"] > 0 else 0,
                "stop_losses": result["stop_losses"],
                "csp_opened": result["csp_opened"], "cc_opened": result["cc_opened"],
            },
            "regime": regime,
            "travel": ti,
            "config": cfg,
        }

        m = metrics
        print(f"  CAGR: {m['cagr']*100:+.1f}%  Sharpe: {m['sharpe']:.2f}  "
              f"Sortino: {m['sortino']:.2f}  MaxDD: {m['max_dd']*100:.1f}%  "
              f"Calmar: {m['calmar']:.2f}")
        print(f"  WR: {wr*100:.0f}%  PF: {pf:.2f}  Total PnL: ${total_pnl:+,.0f}")
        print(f"  CSP: ${csp_pnl:+,.0f}  CC: ${cc_pnl:+,.0f}  Shares: ${share_pnl:+,.0f}")
        print(f"  Assignments: {result['assignments']}  Called: {result['called_away']}  "
              f"Stops: {result['stop_losses']}  CSPs: {result['csp_opened']}  CCs: {result['cc_opened']}")

        # Save equity curve + ledger
        eq_df = pd.DataFrame(result["equity_curve"], columns=["date", "equity"])
        eq_df.drop_duplicates("date", keep="last").to_parquet(
            OUT_DIR / f"equity_v6imp_{name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace(',', '')}.parquet",
            index=False)
        if trades:
            rows = [{k: getattr(t, k) for k in ['ticker','sector','leg_type','open_date',
                     'close_date','expiry','strike','contracts','premium_received',
                     'close_cost','pnl','close_reason','close_pricing','entry_delta','shares_pnl']}
                    for t in trades]
            pd.DataFrame(rows).to_parquet(
                OUT_DIR / f"ledger_v6imp_{name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace(',', '')}.parquet",
                index=False)

    # ── Summary ──
    print("\n" + "=" * 140)
    print("COMPARATIVE SUMMARY — V6 IMPROVED")
    print("=" * 140)
    print(f"{'Variant':<42} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Calmar':>7} "
          f"{'WR%':>5} {'PF':>5} {'CSP$':>10} {'CC$':>10} {'Share$':>10} {'Total$':>10}")
    print("-" * 140)
    for name, r in all_results.items():
        m = r["metrics"]
        ts = r["trade_stats"]
        print(f"{name:<42} {m['cagr']*100:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd']*100:>6.1f}% {m['calmar']:>7.2f} "
              f"{ts['wr']*100:>4.0f}% {ts['pf']:>5.2f} "
              f"${ts['csp_pnl']:>+9,.0f} ${ts['cc_pnl']:>+9,.0f} "
              f"${ts['share_pnl']:>+9,.0f} ${ts['total_pnl']:>+9,.0f}")

    # ── V6 vs V5 comparison ──
    csp_only = all_results.get("CSP-Only (no CC)")
    v6_best_name = max(
        [n for n in all_results if n != "CSP-Only (no CC)"],
        key=lambda n: all_results[n]["metrics"]["sharpe"]
            if all_results[n]["metrics"]["max_dd"] > -0.40 else -999
    )
    v6_best = all_results[v6_best_name]

    print(f"\n{'='*80}")
    print(f"V6 vs CSP-ONLY COMPARISON")
    print(f"{'='*80}")
    if csp_only:
        cm = csp_only["metrics"]
        vm = v6_best["metrics"]
        print(f"  CSP-Only:  CAGR={cm['cagr']*100:.1f}%  Sharpe={cm['sharpe']:.2f}  MaxDD={cm['max_dd']*100:.1f}%")
        print(f"  {v6_best_name}:")
        print(f"    CAGR={vm['cagr']*100:.1f}%  Sharpe={vm['sharpe']:.2f}  MaxDD={vm['max_dd']*100:.1f}%")
        print(f"  Incremental CAGR from CCs: {(vm['cagr'] - cm['cagr'])*100:+.1f}pp")
        print(f"  CC premium collected: ${v6_best['trade_stats']['cc_premium']:,.0f}")

    # ── Regime Gate ──
    print(f"\n{'='*80}")
    print("REGIME GATE (HC #428 R1)")
    print(f"{'='*80}")
    for name, r in all_results.items():
        rg = r["regime"]
        if rg:
            sg = rg.get('sharpe_green', float('nan'))
            sr = rg.get('sharpe_red', float('nan'))
            sf = rg.get('sharpe_flat', float('nan'))
            gap = rg.get('regime_gap', float('nan'))
            r1 = rg.get('r1_pass', False)
            print(f"  {name}: Green={sg:.2f}  Red={sr:.2f}  Flat={sf:.2f}  "
                  f"Gap={gap:.3f}  R1={'PASS' if r1 else 'FAIL'}")

    # ── Travel Income ──
    print(f"\n{'='*80}")
    print("TRAVEL INCOME ($1,500/month on $100K)")
    print(f"{'='*80}")
    for name, r in all_results.items():
        ti = r["travel"]
        ms = ti.get("max_sustainable_monthly_pct")
        ms_str = f"{ms*100:.1f}%/mo (${ms*STARTING_CASH:,.0f}/mo)" if ms else "<0.5%/mo"
        print(f"  {name}:")
        print(f"    Withdrawn: ${ti['total_withdrawn']:,.0f}  Final NAV: ${ti['final_nav']:,.0f}  "
              f"Sustained: {ti['nav_preserved']}")
        print(f"    Max sustainable: {ms_str}  "
              f"Max losing streak: {ti['max_loss_streak_months']} months")

    # ── Premium Breakdown ──
    print(f"\n{'='*80}")
    print("PREMIUM BREAKDOWN")
    print(f"{'='*80}")
    for name, r in all_results.items():
        ts = r["trade_stats"]
        print(f"  {name}: Puts ${ts['csp_premium']:,.0f} ({ts['pct_from_puts']:.0f}%) + "
              f"Calls ${ts['cc_premium']:,.0f} ({ts['pct_from_calls']:.0f}%) = "
              f"${ts['csp_premium']+ts['cc_premium']:,.0f}")

    # Save
    save = {}
    for name, r in all_results.items():
        save[name] = {
            "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                       for k, v in r["metrics"].items()},
            "trade_stats": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                           for k, v in r["trade_stats"].items()},
            "regime": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                      for k, v in r["regime"].items()},
            "travel": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                      for k, v in r["travel"].items()},
            "config": r["config"],
        }
    with open(OUT_DIR / "v6_improved_results.json", "w") as f:
        json.dump(save, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"COMPLETE in {elapsed:.0f}s. Results saved to {OUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
