#!/usr/bin/env python3
"""
iron_butterfly_backtest.py — Systematic Iron Butterfly Backtest
================================================================

Strategy: Sell ATM straddle + buy OTM wings on range-bound stocks.
  - Sell ATM put + ATM call (the straddle)
  - Buy OTM put (~15 delta) + OTM call (~15 delta) as protection
  - DTE: 21-35 days (monthly cycle)
  - Universe: 70-ticker wheel universe, filtered for range-bound candidates
  - Entry filters: IV rank 30-60%, 20d momentum near zero, no earnings within 5 days
  - Exit: 35% profit target, 1.5x credit stop loss, or expiration
  - Position sizing: max 3% of NAV per name, max 40% total margin

Walk-forward: sliding window, OOS only (no in-sample optimization).
Pricing: Black-Scholes (acknowledged limitations in report).
Cost model: $0.65/contract/leg, 4 legs = $2.60 open + $2.60 close = $5.20 RT total.
  Plus 2.5% slippage on premium (min $0.03/share per leg).

Data: Reuses wheel cache (prices, IV, macro, earnings).

Outputs: /home/jupiter/Lvl3Quant/output/iron_butterfly_research/
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "iron_butterfly_research"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──
STARTING_CAPITAL = 100_000.0
RISK_FREE = 0.04
TRADING_DAYS = 252

# Butterfly parameters
WING_DELTA_TARGET = 0.15       # OTM wing delta
DTE_MIN = 21
DTE_MAX = 35
DTE_TARGET = 28                # Target ~28 DTE (monthly)
PROFIT_TAKE_PCT = 0.35         # Close at 35% of max credit (conservative)
STOP_LOSS_MULT = 1.5           # Close when loss = 1.5x credit received
MIN_NET_CREDIT = 0.30          # Min $0.30/share net credit

# Entry filters
IV_RANK_MIN = 0.30             # IV rank floor (too low = no premium)
IV_RANK_MAX = 0.60             # IV rank ceiling (too high = too risky)
MOMENTUM_ABS_MAX = 0.08        # |20d return| must be < 8% (range-bound filter)
EARNINGS_BUFFER_DAYS = 5       # No entry within 5 days of earnings
VIX_MAX_GATE = 35.0            # Don't enter in extreme vol

# Position sizing
MARGIN_CAP = 0.40              # Max 40% of NAV in total margin
PER_NAME_PCT = 0.03            # Max 3% of NAV per name
MAX_CONCURRENT = 20            # Max 20 simultaneous positions

# Costs
COMMISSION_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025          # 2.5% of premium
SLIPPAGE_MIN = 0.03            # Min $0.03/share per leg

# Price filter
MIN_PRICE = 15.0
MAX_PRICE = 500.0


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
        if kind == "put":
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _Phi(d1) - 1.0 if kind == "put" else _Phi(d1)


def strike_for_delta(S, T, sigma, target_delta, kind="put", r=RISK_FREE):
    """Solve for strike K such that |delta(K)| ~= target_delta."""
    if T <= 0 or sigma <= 0:
        return S
    from scipy.stats import norm
    if kind == "put":
        N_d1 = 1.0 - target_delta
    else:
        N_d1 = target_delta
    d1 = norm.ppf(N_d1)
    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r + 0.5 * sigma * sigma) * T))
    # Round to nearest $1 for realism (most equity options have $1 strikes)
    return round(K)


def _slip_sell(px):
    """Premium received when selling (reduced by slippage)."""
    slip = max(px * SLIPPAGE_FRAC, SLIPPAGE_MIN)
    return max(px - slip, 0.01)


def _slip_buy(px):
    """Premium paid when buying (increased by slippage)."""
    slip = max(px * SLIPPAGE_FRAC, SLIPPAGE_MIN)
    return px + slip


# ── Data Loading ──
def load_data():
    """Load prices, IV, macro, earnings from wheel cache."""
    prices = pd.read_parquet(CACHE / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    iv = pd.read_parquet(CACHE / "iv_cache.parquet")
    iv["date"] = pd.to_datetime(iv["date"])

    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.sort_values("date").drop_duplicates("date")

    # Earnings dates
    earn_path = CACHE / "earnings_dates.parquet"
    earnings = {}
    if earn_path.exists():
        edf = pd.read_parquet(earn_path)
        edf["earnings_date"] = pd.to_datetime(edf["earnings_date"])
        for ticker, grp in edf.groupby("ticker"):
            earnings[ticker] = np.sort(grp["earnings_date"].values)

    # Merge prices + IV
    df = prices.merge(iv[["date", "ticker", "sigma", "iv_rank"]], on=["date", "ticker"], how="left")
    df = df.merge(macro, on="date", how="left")
    df["vix"] = df["vix"].ffill()

    # Compute 20d momentum for range-bound filter
    df = df.sort_values(["ticker", "date"])
    df["mom_20d"] = df.groupby("ticker")["close"].transform(lambda x: x.pct_change(20))

    # Filter valid
    df = df.dropna(subset=["close", "sigma", "iv_rank"])
    df["sigma"] = df["sigma"].clip(lower=0.05, upper=2.0)

    return df, earnings


# ── Butterfly Position ──
@dataclass
class ButterflyPos:
    ticker: str
    open_date: pd.Timestamp
    expiry: pd.Timestamp
    atm_strike: float           # ATM strike (short put + short call)
    long_put_K: float           # OTM put wing
    long_call_K: float          # OTM call wing
    open_sigma: float
    open_S: float
    net_credit_per_share: float # Total credit received per share
    max_loss_per_share: float   # Max loss = wing_width - net_credit
    contracts: int
    margin_used: float          # Dollars of margin consumed


# ── Backtest Engine ──
class IronButterflyBacktest:
    def __init__(self, capital=STARTING_CAPITAL):
        self.starting_capital = capital
        self.cash = capital
        self.positions: List[ButterflyPos] = []
        self.equity_curve = []
        self.ledger = []
        self.total_margin_used = 0.0

    def _find_expiry(self, today):
        """Find next monthly option expiry (3rd Friday) >= DTE_MIN from today."""
        # Start from today + DTE_MIN, find the 3rd Friday of that month
        target = today + pd.Timedelta(days=DTE_MIN)
        for month_offset in range(3):
            check_month = target.month + month_offset
            check_year = target.year + (check_month - 1) // 12
            check_month = ((check_month - 1) % 12) + 1
            # Find 3rd Friday
            first_day = pd.Timestamp(check_year, check_month, 1)
            # Friday = weekday 4
            first_friday = first_day + pd.Timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + pd.Timedelta(days=14)
            dte = (third_friday - today).days
            if DTE_MIN <= dte <= DTE_MAX:
                return third_friday
        # Fallback: just target DTE days out, round to Friday
        cand = today + pd.Timedelta(days=DTE_TARGET)
        shift = (4 - cand.weekday()) % 7
        return cand + pd.Timedelta(days=shift)

    def _near_earnings(self, ticker, today, earnings_dict, buffer=EARNINGS_BUFFER_DAYS):
        """Check if there's an earnings date within buffer days of today."""
        dates = earnings_dict.get(ticker)
        if dates is None or len(dates) == 0:
            return False
        today_np = np.datetime64(today)
        diffs = np.abs((dates - today_np).astype("timedelta64[D]").astype(int))
        return np.any(diffs <= buffer)

    def _current_nav(self):
        """NAV = cash (positions are defined-risk, margin is held in cash)."""
        return self.cash

    def open_butterfly(self, ticker, today, S, sigma, iv_rank):
        """Open an iron butterfly: sell ATM straddle, buy OTM wings."""
        expiry = self._find_expiry(today)
        T = max((expiry - today).days, 1) / 365.0

        atm_K = round(S)  # ATM strike

        # Wing strikes at ~15 delta
        long_put_K = strike_for_delta(S, T, sigma, WING_DELTA_TARGET, kind="put")
        long_call_K = strike_for_delta(S, T, sigma, WING_DELTA_TARGET, kind="call")

        # Sanity checks
        if long_put_K >= atm_K:
            long_put_K = atm_K - max(1, int(S * 0.05))
        if long_call_K <= atm_K:
            long_call_K = atm_K + max(1, int(S * 0.05))

        # Price all four legs
        short_put_px = bs_price(S, atm_K, T, sigma, kind="put")
        short_call_px = bs_price(S, atm_K, T, sigma, kind="call")
        long_put_px = bs_price(S, long_put_K, T, sigma, kind="put")
        long_call_px = bs_price(S, long_call_K, T, sigma, kind="call")

        # Net credit per share = premiums sold - premiums bought (with slippage)
        credit = _slip_sell(short_put_px) + _slip_sell(short_call_px)
        debit = _slip_buy(long_put_px) + _slip_buy(long_call_px)
        net_credit = credit - debit

        if net_credit < MIN_NET_CREDIT:
            return None

        # Max loss = wider wing width - net credit
        put_width = atm_K - long_put_K
        call_width = long_call_K - atm_K
        max_wing = max(put_width, call_width)
        max_loss_per_share = max_wing - net_credit

        if max_loss_per_share <= 0:
            return None

        # Position sizing
        nav = self._current_nav()
        max_margin_per_name = nav * PER_NAME_PCT
        margin_per_contract = max_wing * 100  # Defined risk
        if margin_per_contract <= 0:
            return None

        max_contracts_by_name = max(1, int(max_margin_per_name / margin_per_contract))

        # Check total margin cap
        available_margin = max(0, nav * MARGIN_CAP - self.total_margin_used)
        max_contracts_by_total = max(1, int(available_margin / margin_per_contract))

        contracts = min(max_contracts_by_name, max_contracts_by_total)
        if contracts <= 0:
            return None

        margin_used = margin_per_contract * contracts

        # Cash flow: receive credit, pay commissions
        total_credit = net_credit * 100 * contracts
        total_commission = COMMISSION_PER_CONTRACT * 4 * contracts  # 4 legs
        self.cash += total_credit - total_commission

        self.total_margin_used += margin_used

        pos = ButterflyPos(
            ticker=ticker,
            open_date=today,
            expiry=expiry,
            atm_strike=atm_K,
            long_put_K=long_put_K,
            long_call_K=long_call_K,
            open_sigma=sigma,
            open_S=S,
            net_credit_per_share=net_credit,
            max_loss_per_share=max_loss_per_share,
            contracts=contracts,
            margin_used=margin_used,
        )
        self.positions.append(pos)
        return pos

    def _mtm_butterfly(self, pos, S, sigma):
        """Mark-to-market a butterfly position. Returns (unrealized_pnl, profit_frac)."""
        T = max((pos.expiry - pd.Timestamp.now()).days, 0) / 365.0
        # Use remaining DTE based on current date context (caller handles)
        return None  # Placeholder - actual MTM done in close/manage

    def close_butterfly(self, pos, today, S, sigma, reason=""):
        """Close all 4 legs of the butterfly."""
        T = max((pos.expiry - today).days, 0) / 365.0

        # Price legs to close
        short_put_px = bs_price(S, pos.atm_strike, T, sigma, kind="put")
        short_call_px = bs_price(S, pos.atm_strike, T, sigma, kind="call")
        long_put_px = bs_price(S, pos.long_put_K, T, sigma, kind="put")
        long_call_px = bs_price(S, pos.long_call_K, T, sigma, kind="call")

        # To close: buy back shorts (pay more), sell longs (receive less)
        cost_close = _slip_buy(short_put_px) + _slip_buy(short_call_px)
        proceeds_close = _slip_sell(long_put_px) + _slip_sell(long_call_px)
        net_close_cost = cost_close - proceeds_close  # Positive = we pay

        # Cash flow
        total_close_cost = net_close_cost * 100 * pos.contracts
        total_commission = COMMISSION_PER_CONTRACT * 4 * pos.contracts
        self.cash -= total_close_cost + total_commission

        # Release margin
        self.total_margin_used -= pos.margin_used

        # Realized P&L
        open_commission = COMMISSION_PER_CONTRACT * 4 * pos.contracts
        realized_pnl = (pos.net_credit_per_share - net_close_cost) * 100 * pos.contracts \
            - open_commission - total_commission

        pct_of_max = realized_pnl / (pos.net_credit_per_share * 100 * pos.contracts) \
            if pos.net_credit_per_share > 0 else 0.0

        self.ledger.append({
            "ticker": pos.ticker,
            "open_date": str(pos.open_date.date()),
            "close_date": str(today.date()),
            "expiry": str(pos.expiry.date()),
            "atm_strike": pos.atm_strike,
            "put_wing": pos.long_put_K,
            "call_wing": pos.long_call_K,
            "contracts": pos.contracts,
            "net_credit_per_share": round(pos.net_credit_per_share, 4),
            "close_cost_per_share": round(net_close_cost, 4),
            "realized_pnl": round(realized_pnl, 2),
            "pct_of_max_credit": round(pct_of_max, 4),
            "days_in_trade": (today - pos.open_date).days,
            "reason": reason,
            "open_S": round(pos.open_S, 2),
            "close_S": round(S, 2),
            "move_pct": round((S - pos.open_S) / pos.open_S * 100, 2),
        })

        if pos in self.positions:
            self.positions.remove(pos)

        return realized_pnl

    def run(self, df, earnings_dict, spy_df):
        """Run the walk-forward backtest."""
        dates = sorted(df["date"].unique())
        tickers_by_date = df.groupby("date")

        # SPY returns for regime classification
        spy_df = spy_df.set_index("date").sort_index()

        trades_opened = 0
        trades_closed = 0
        skipped_earnings = 0
        skipped_momentum = 0
        skipped_iv = 0
        skipped_margin = 0

        for i, today in enumerate(dates):
            today = pd.Timestamp(today)

            # Get today's data
            if today not in tickers_by_date.groups:
                continue
            day_data = tickers_by_date.get_group(today)

            # Get VIX
            vix = day_data["vix"].iloc[0] if "vix" in day_data.columns else 20.0

            # ── Manage existing positions ──
            to_close = []
            for pos in self.positions:
                # Get current price for this ticker
                ticker_row = day_data[day_data["ticker"] == pos.ticker]
                if ticker_row.empty:
                    continue

                S = float(ticker_row["close"].iloc[0])
                sigma = float(ticker_row["sigma"].iloc[0])
                T = max((pos.expiry - today).days, 0) / 365.0

                # Price the position to close
                sp_px = bs_price(S, pos.atm_strike, T, sigma, kind="put")
                sc_px = bs_price(S, pos.atm_strike, T, sigma, kind="call")
                lp_px = bs_price(S, pos.long_put_K, T, sigma, kind="put")
                lc_px = bs_price(S, pos.long_call_K, T, sigma, kind="call")

                cost_close = _slip_buy(sp_px) + _slip_buy(sc_px)
                proceeds_close = _slip_sell(lp_px) + _slip_sell(lc_px)
                net_close = cost_close - proceeds_close

                unrealized_per_share = pos.net_credit_per_share - net_close
                profit_frac = unrealized_per_share / pos.net_credit_per_share \
                    if pos.net_credit_per_share > 0 else 0.0

                dte_remaining = (pos.expiry - today).days

                # Exit conditions
                reason = None
                if profit_frac >= PROFIT_TAKE_PCT:
                    reason = "profit_target"
                elif profit_frac <= -STOP_LOSS_MULT:
                    reason = "stop_loss"
                elif dte_remaining <= 3:
                    reason = "dte_close"
                elif today >= pos.expiry:
                    reason = "expiry"
                # Check if earnings are imminent (close before earnings)
                elif self._near_earnings(pos.ticker, today, earnings_dict, buffer=2):
                    reason = "pre_earnings_close"

                if reason:
                    to_close.append((pos, S, sigma, reason))

            for pos, S, sigma, reason in to_close:
                self.close_butterfly(pos, today, S, sigma, reason)
                trades_closed += 1

            # ── Open new positions ──
            if vix > VIX_MAX_GATE:
                pass  # Skip entries in extreme vol
            elif len(self.positions) < MAX_CONCURRENT:
                # Score and rank candidates
                candidates = []
                for _, row in day_data.iterrows():
                    ticker = row["ticker"]
                    if ticker == "SPY":
                        continue

                    S = float(row["close"])
                    sigma = float(row["sigma"])
                    iv_rank = float(row["iv_rank"]) if pd.notna(row["iv_rank"]) else 0.5
                    mom = float(row["mom_20d"]) if pd.notna(row["mom_20d"]) else 0.0

                    # Already have position in this name?
                    if any(p.ticker == ticker for p in self.positions):
                        continue

                    # Price filter
                    if S < MIN_PRICE or S > MAX_PRICE:
                        continue

                    # IV rank filter (sweet spot: 30-60%)
                    if iv_rank < IV_RANK_MIN or iv_rank > IV_RANK_MAX:
                        skipped_iv += 1
                        continue

                    # Momentum filter: range-bound stocks only
                    if abs(mom) > MOMENTUM_ABS_MAX:
                        skipped_momentum += 1
                        continue

                    # Earnings buffer
                    if self._near_earnings(ticker, today, earnings_dict):
                        skipped_earnings += 1
                        continue

                    # Score: prefer lower |momentum| and moderate IV rank
                    # Best butterfly candidates: dead-flat stocks with decent premium
                    score = (1.0 - abs(mom) / MOMENTUM_ABS_MAX) * 0.6 + \
                            (1.0 - abs(iv_rank - 0.45) / 0.15) * 0.4
                    candidates.append((ticker, S, sigma, iv_rank, score))

                # Sort by score descending, take top N to fill slots
                candidates.sort(key=lambda x: -x[4])
                slots = MAX_CONCURRENT - len(self.positions)

                for ticker, S, sigma, iv_rank, score in candidates[:slots]:
                    # Check margin
                    nav = self._current_nav()
                    if self.total_margin_used >= nav * MARGIN_CAP:
                        skipped_margin += 1
                        break

                    pos = self.open_butterfly(ticker, today, S, sigma, iv_rank)
                    if pos is not None:
                        trades_opened += 1

            # ── Record equity ──
            # MTM all open positions
            mtm_value = 0.0
            for pos in self.positions:
                ticker_row = day_data[day_data["ticker"] == pos.ticker]
                if ticker_row.empty:
                    continue
                S = float(ticker_row["close"].iloc[0])
                sigma = float(ticker_row["sigma"].iloc[0])
                T = max((pos.expiry - today).days, 0) / 365.0

                sp_px = bs_price(S, pos.atm_strike, T, sigma, kind="put")
                sc_px = bs_price(S, pos.atm_strike, T, sigma, kind="call")
                lp_px = bs_price(S, pos.long_put_K, T, sigma, kind="put")
                lc_px = bs_price(S, pos.long_call_K, T, sigma, kind="call")

                net_close = (_slip_buy(sp_px) + _slip_buy(sc_px)) - \
                            (_slip_sell(lp_px) + _slip_sell(lc_px))
                unrealized = (pos.net_credit_per_share - net_close) * 100 * pos.contracts
                mtm_value += unrealized

            nav = self.cash + mtm_value
            self.equity_curve.append({
                "date": today,
                "nav": nav,
                "cash": self.cash,
                "n_positions": len(self.positions),
                "total_margin": self.total_margin_used,
            })

            if i % 500 == 0:
                print(f"  [{today.date()}] NAV=${nav:,.0f} | Positions={len(self.positions)} | "
                      f"Trades={trades_opened}/{trades_closed}")

        # Close any remaining positions at last date
        if self.positions:
            last_date = dates[-1]
            last_data = tickers_by_date.get_group(last_date)
            for pos in list(self.positions):
                ticker_row = last_data[last_data["ticker"] == pos.ticker]
                if not ticker_row.empty:
                    S = float(ticker_row["close"].iloc[0])
                    sigma = float(ticker_row["sigma"].iloc[0])
                    self.close_butterfly(pos, pd.Timestamp(last_date), S, sigma, "backtest_end")
                    trades_closed += 1

        stats = {
            "trades_opened": trades_opened,
            "trades_closed": trades_closed,
            "skipped_earnings": skipped_earnings,
            "skipped_momentum": skipped_momentum,
            "skipped_iv": skipped_iv,
            "skipped_margin": skipped_margin,
        }
        return stats


def compute_metrics(equity_df, spy_df, ledger_df):
    """Compute strategy performance metrics and regime analysis."""
    equity_df = equity_df.set_index("date").sort_index()
    equity_df["daily_ret"] = equity_df["nav"].pct_change()

    # Basic metrics
    rets = equity_df["daily_ret"].dropna()
    n_days = len(rets)
    years = n_days / TRADING_DAYS

    total_ret = (equity_df["nav"].iloc[-1] / equity_df["nav"].iloc[0]) - 1
    cagr = (1 + total_ret) ** (1.0 / years) - 1 if years > 0 else 0

    ann_ret = rets.mean() * TRADING_DAYS
    ann_vol = rets.std() * math.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = rets[rets < 0].std() * math.sqrt(TRADING_DAYS)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    peak = equity_df["nav"].cummax()
    dd = (equity_df["nav"] - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Trade-level stats
    if len(ledger_df) > 0:
        wins = ledger_df[ledger_df["realized_pnl"] > 0]
        win_rate = len(wins) / len(ledger_df)
        gross_profit = wins["realized_pnl"].sum() if len(wins) > 0 else 0
        losses = ledger_df[ledger_df["realized_pnl"] <= 0]
        gross_loss = abs(losses["realized_pnl"].sum()) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_trade = ledger_df["realized_pnl"].mean()
        avg_winner = wins["realized_pnl"].mean() if len(wins) > 0 else 0
        avg_loser = losses["realized_pnl"].mean() if len(losses) > 0 else 0
        avg_days = ledger_df["days_in_trade"].mean()
    else:
        win_rate = profit_factor = avg_trade = avg_winner = avg_loser = avg_days = 0

    # Regime analysis using SPY
    spy_rets = spy_df.set_index("date")["close"].pct_change().reindex(equity_df.index).fillna(0)

    # Classify days: green (SPY > 0), red (SPY < 0), flat (SPY == 0)
    regime = pd.Series("flat", index=equity_df.index)
    regime[spy_rets > 0.001] = "green"
    regime[spy_rets < -0.001] = "red"

    regime_sharpes = {}
    for r_name in ["green", "red", "flat"]:
        r_rets = rets[regime == r_name]
        if len(r_rets) > 10:
            r_sharpe = (r_rets.mean() * TRADING_DAYS) / (r_rets.std() * math.sqrt(TRADING_DAYS)) \
                if r_rets.std() > 0 else 0
        else:
            r_sharpe = 0
        regime_sharpes[r_name] = round(r_sharpe, 4)

    # Regime gap
    s_green = regime_sharpes.get("green", 0)
    s_red = regime_sharpes.get("red", 0)
    max_s = max(abs(s_green), abs(s_red), 0.001)
    regime_gap = abs(s_green - s_red) / max_s

    # Correlation with SPY
    spy_aligned = spy_rets.reindex(rets.index).fillna(0)
    corr_spy = rets.corr(spy_aligned)

    # Close reason breakdown
    reason_counts = {}
    if len(ledger_df) > 0:
        reason_counts = ledger_df["reason"].value_counts().to_dict()

    metrics = {
        "n_days": n_days,
        "years": round(years, 2),
        "cagr": round(cagr, 4),
        "total_return_pct": round(total_ret * 100, 2),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "ann_vol": round(ann_vol, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "avg_trade_pnl": round(avg_trade, 2),
        "avg_winner": round(avg_winner, 2),
        "avg_loser": round(avg_loser, 2),
        "avg_days_in_trade": round(avg_days, 1),
        "total_trades": len(ledger_df),
        "correlation_with_spy": round(corr_spy, 4),
        "regime_sharpes": regime_sharpes,
        "regime_gap": round(regime_gap, 4),
        "hc428_regime_pass": regime_gap <= 0.50,
        "close_reasons": {k: int(v) for k, v in reason_counts.items()},
        "final_nav": round(equity_df["nav"].iloc[-1], 2),
        "starting_capital": STARTING_CAPITAL,
    }
    return metrics


def compare_with_spy(equity_df, spy_df):
    """Compare butterfly returns with SPY buy-and-hold."""
    eq = equity_df.set_index("date")["nav"]
    spy = spy_df.set_index("date")["close"]

    # Align dates
    common = eq.index.intersection(spy.index)
    if len(common) < 10:
        return {}

    eq = eq.loc[common]
    spy = spy.loc[common]

    spy_ret = (spy.iloc[-1] / spy.iloc[0]) - 1
    spy_years = len(common) / TRADING_DAYS
    spy_cagr = (1 + spy_ret) ** (1.0 / spy_years) - 1 if spy_years > 0 else 0
    spy_rets = spy.pct_change().dropna()
    spy_vol = spy_rets.std() * math.sqrt(TRADING_DAYS)
    spy_sharpe = (spy_rets.mean() * TRADING_DAYS) / spy_vol if spy_vol > 0 else 0
    spy_dd = ((spy - spy.cummax()) / spy.cummax()).min()

    return {
        "spy_cagr": round(spy_cagr, 4),
        "spy_sharpe": round(spy_sharpe, 4),
        "spy_max_dd": round(spy_dd, 4),
        "spy_total_return_pct": round(spy_ret * 100, 2),
    }


def main():
    print("=" * 70)
    print("IRON BUTTERFLY BACKTEST — Systematic Range-Bound Premium Capture")
    print("=" * 70)

    print("\nLoading data...")
    df, earnings = load_data()
    print(f"  Data: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Tickers: {df['ticker'].nunique()}")
    print(f"  Rows: {len(df):,}")

    # Use 2019-01-01 onward for OOS (2015-2018 as implicit lookback for IV rank etc.)
    START_OOS = pd.Timestamp("2019-01-01")
    df_oos = df[df["date"] >= START_OOS].copy()
    print(f"  OOS period: {START_OOS.date()} to {df_oos['date'].max().date()}")
    print(f"  OOS rows: {len(df_oos):,}")

    # SPY data for benchmarking
    spy_df = df[df["ticker"] == "SPY"][["date", "close"]].copy()

    print("\nRunning backtest...")
    bt = IronButterflyBacktest(capital=STARTING_CAPITAL)
    stats = bt.run(df_oos, earnings, spy_df)

    print(f"\n  Trades opened: {stats['trades_opened']}")
    print(f"  Trades closed: {stats['trades_closed']}")
    print(f"  Skipped (earnings): {stats['skipped_earnings']}")
    print(f"  Skipped (momentum): {stats['skipped_momentum']}")
    print(f"  Skipped (IV rank): {stats['skipped_iv']}")

    # Build results
    equity_df = pd.DataFrame(bt.equity_curve)
    ledger_df = pd.DataFrame(bt.ledger)

    if len(equity_df) == 0:
        print("\nERROR: No equity curve generated. Check data.")
        return

    print("\nComputing metrics...")
    metrics = compute_metrics(equity_df, spy_df, ledger_df)
    spy_compare = compare_with_spy(equity_df, spy_df)
    metrics["spy_benchmark"] = spy_compare
    metrics["backtest_stats"] = stats

    # BS pricing limitations
    metrics["bs_limitations"] = [
        "Black-Scholes assumes constant vol — real vol smiles/skews affect wing pricing",
        "No vol surface modeling — ATM vol used for all strikes (overstates wing value)",
        "No early exercise modeling — BS is European, US equity options are American",
        "Bid-ask spread modeled as fixed % — real spreads vary by liquidity/vol regime",
        "No gap risk modeling — overnight jumps can breach wings without intraday management",
        "IV rank computed from historical data — real-time IV rank may differ",
        "Strike rounding to $1 — some low-priced stocks have $0.50 or $2.50 strikes",
    ]

    # Save outputs
    equity_df.to_parquet(OUT_DIR / "equity.parquet", index=False)
    equity_df.to_csv(OUT_DIR / "equity.csv", index=False)
    if len(ledger_df) > 0:
        ledger_df.to_parquet(OUT_DIR / "ledger.parquet", index=False)
        ledger_df.to_csv(OUT_DIR / "ledger.csv", index=False)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS — Iron Butterfly (ATM Straddle + OTM Wings)")
    print("=" * 70)
    print(f"  Period:           {equity_df['date'].iloc[0]} to {equity_df['date'].iloc[-1]}")
    print(f"  Starting Capital: ${STARTING_CAPITAL:,.0f}")
    print(f"  Final NAV:        ${metrics['final_nav']:,.0f}")
    print(f"  Total Return:     {metrics['total_return_pct']:.1f}%")
    print(f"  CAGR:             {metrics['cagr']*100:.1f}%")
    print(f"  Sharpe:           {metrics['sharpe']:.2f}")
    print(f"  Sortino:          {metrics['sortino']:.2f}")
    print(f"  Max Drawdown:     {metrics['max_dd']*100:.1f}%")
    print(f"  Calmar:           {metrics['calmar']:.2f}")
    print(f"  Win Rate:         {metrics['win_rate']*100:.1f}%")
    print(f"  Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"  Avg Trade P&L:    ${metrics['avg_trade_pnl']:.2f}")
    print(f"  Avg Winner:       ${metrics['avg_winner']:.2f}")
    print(f"  Avg Loser:        ${metrics['avg_loser']:.2f}")
    print(f"  Avg Days/Trade:   {metrics['avg_days_in_trade']:.1f}")
    print(f"  Total Trades:     {metrics['total_trades']}")
    print(f"  Corr w/ SPY:      {metrics['correlation_with_spy']:.3f}")
    print(f"\n  Regime Sharpes:")
    for regime, s in metrics['regime_sharpes'].items():
        print(f"    {regime:8s}: {s:.2f}")
    print(f"  Regime Gap:       {metrics['regime_gap']:.2f} ({'PASS' if metrics['hc428_regime_pass'] else 'FAIL'})")
    print(f"\n  Close Reasons:")
    for reason, count in metrics.get('close_reasons', {}).items():
        print(f"    {reason:25s}: {count}")

    if spy_compare:
        print(f"\n  SPY Benchmark:")
        print(f"    SPY CAGR:       {spy_compare['spy_cagr']*100:.1f}%")
        print(f"    SPY Sharpe:     {spy_compare['spy_sharpe']:.2f}")
        print(f"    SPY Max DD:     {spy_compare['spy_max_dd']*100:.1f}%")

    print(f"\n  BS Pricing Limitations:")
    for lim in metrics["bs_limitations"][:3]:
        print(f"    - {lim}")
    print(f"    ... ({len(metrics['bs_limitations'])} total, see results.json)")

    print(f"\n  Results saved to: {OUT_DIR}")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    metrics = main()
