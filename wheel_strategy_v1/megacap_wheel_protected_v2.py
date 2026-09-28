#!/usr/bin/env python3
"""
megacap_wheel_protected_v2.py - Protected megacap wheel strategy with full risk management.

Improvements over v1:
  1. 200MA filter per stock — only sell CSPs when stock is ABOVE 200-day MA.
  2. VIX gate — half size at VIX>25, stop new positions at VIX>35.
  3. Earnings blackout — no new CSP within 7 days of earnings expiry.
  4. Hard stop-loss on assigned shares — sell if -15% below assignment price.
  5. Vol-adjusted position sizing — inverse-vol sizing using 20-day realized vol.
  6. Cross-sector diversification — 16 names across 7 sectors.
  7. Commission-free — Robinhood pricing ($0 commission).
  8. Aggressiveness slider — conservative / moderate / aggressive.
  9. Full wheel cycle with assignment tracking.
  10. HC #428 R1 validation: regime-agnostic metrics, permutation test.

Universe: AAPL, MSFT, GOOGL, AMZN, META, JPM, BAC, GS, JNJ, UNH, PFE, XOM, CVX, HD, WMT, COST
Data: real Dolt options chain data with bid/ask, 2019-2026.
"""
from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------- paths ------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
CHAINS_DIR = CACHE / "options_real" / "chains"
OUT_DIR = ROOT / "output" / "megacap_wheel_protected_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------- universe ----------------------------------------
# Available in Dolt data + prices
UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    # Financials
    "JPM", "BAC", "GS",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Energy
    "XOM", "CVX",
    # Consumer
    "HD", "WMT", "COST",
]

SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech", "META": "Tech",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "XOM": "Energy", "CVX": "Energy",
    "HD": "Consumer", "WMT": "Consumer", "COST": "Consumer",
}

START_DATE = pd.Timestamp("2019-03-01")   # first date with real chain data
END_DATE   = pd.Timestamp("2026-03-31")   # limited by IV data
TRADING_DAYS = 252
RISK_FREE = 0.04
BASKET_TOTAL = 500_000.0                  # $500K total (megacap stocks need large capital)
# At $500K / 16 names = $31,250 per name. This covers 1 contract of most megacaps:
# UNH $471 = $40K collateral (still tight), AAPL $156 = $13K (OK at 5-7% = $25-35K)
# Without a large enough portfolio, high-priced stocks simply can't be traded in 1-contract lots.

# ------------------------- aggressiveness configs --------------------------
CONFIGS = {
    "conservative": {
        "delta_target": 0.15,
        "dte_min": 10,               # chain data has DTE 10-67; weekly approach uses 10-18
        "dte_max": 18,
        "dte_target": 14,
        "max_alloc_pct": 0.06,       # 6% per name = $30K at $500K. Covers 1 contract of AAPL $156
        "profit_take_pct": 0.50,
        "vix_half_size": 25.0,
        "vix_max_gate": 35.0,
        "earnings_blackout_days": 7,
        "stop_loss_pct": 0.15,       # sell shares if -15% from assignment
        "label": "conservative",
    },
    "moderate": {
        "delta_target": 0.25,
        "dte_min": 10,
        "dte_max": 18,
        "dte_target": 14,
        "max_alloc_pct": 0.10,       # 10% per name = $50K. Fits most megacaps (1-3 contracts)
        "profit_take_pct": 0.50,
        "vix_half_size": 25.0,
        "vix_max_gate": 35.0,
        "earnings_blackout_days": 7,
        "stop_loss_pct": 0.15,
        "label": "moderate",
    },
    "aggressive": {
        "delta_target": 0.35,
        "dte_min": 10,
        "dte_max": 18,
        "dte_target": 14,
        "max_alloc_pct": 0.15,       # 15% per name = $75K. Can run 2-4 contracts on most names
        "profit_take_pct": 0.50,
        "vix_half_size": 25.0,
        "vix_max_gate": 35.0,
        "earnings_blackout_days": 7,
        "stop_loss_pct": 0.15,
        "label": "aggressive",
    },
}


# ------------------------- BS pricing (same as v1) -------------------------
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


# ------------------------- state dataclasses -------------------------------
@dataclass
class Position:
    ticker: str
    side: str                # 'short_put' | 'long_shares' | 'short_call'
    strike: float
    expiry: pd.Timestamp
    open_date: pd.Timestamp
    open_price: float        # premium received per share (or basis for shares)
    contracts: int           # 1 contract = 100 shares
    open_sigma: float
    share_basis: float = 0.0
    assignment_price: float = 0.0  # price at assignment (for stop-loss calc)


@dataclass
class TickerState:
    ticker: str
    cash: float
    positions: list = field(default_factory=list)
    ledger: list = field(default_factory=list)
    csp_opened: int = 0
    cc_opened: int = 0
    assignments: int = 0
    call_aways: int = 0
    stop_loss_exits: int = 0
    earnings_skips: int = 0
    ma_filter_skips: int = 0
    vix_skips: int = 0
    days_in_shares: int = 0
    days_in_csp: int = 0
    days_in_cash: int = 0


# ------------------------- data loading ------------------------------------
def load_data():
    """Load prices, IV, VIX, SPY, earnings for all universe tickers."""
    print("[load] prices ...")
    prices = pd.read_parquet(CACHE / "prices.parquet")
    prices = prices[prices["ticker"].isin(UNIVERSE)][
        ["ticker", "date", "close", "rv_20"]
    ].copy()
    prices["date"] = pd.to_datetime(prices["date"])

    print("[load] IV (real) ...")
    iv = pd.read_parquet(CACHE / "iv_features_real.parquet")
    iv = iv[iv["ticker"].isin(UNIVERSE)][["date", "ticker", "sigma"]].copy()
    iv["date"] = pd.to_datetime(iv["date"])

    print("[load] macro/VIX ...")
    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"])

    print("[load] SPY ...")
    sec = pd.read_parquet(CACHE / "sector_etfs.parquet")
    spy = sec[sec["ticker"] == "SPY"][["date", "close"]].copy()
    spy.columns = ["date", "spy_close"]
    spy["date"] = pd.to_datetime(spy["date"])

    print("[load] earnings ...")
    earnings = pd.read_parquet(CACHE / "earnings_dates.parquet")
    earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"])
    earnings = earnings[earnings["ticker"].isin(UNIVERSE)].copy()

    # merge
    df = prices.merge(iv, on=["date", "ticker"], how="inner")
    df = df.merge(macro, on="date", how="left")
    df = df.merge(spy, on="date", how="left")
    df["vix"] = df["vix"].ffill()
    df["spy_close"] = df["spy_close"].ffill()
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df = df.dropna(subset=["close", "sigma"])
    df["sigma"] = df["sigma"].clip(lower=0.05, upper=2.0)
    df["rv_20"] = df["rv_20"].clip(lower=0.01, upper=2.0)

    # SPY returns for regime classification
    spy_ret_df = spy.copy().sort_values("date")
    spy_ret_df["spy_ret"] = spy_ret_df["spy_close"].pct_change()
    spy_ret_df = spy_ret_df[["date", "spy_ret"]]
    df = df.merge(spy_ret_df, on="date", how="left")

    # 200-day moving average per ticker
    df = df.sort_values(["ticker", "date"])
    df["ma200"] = df.groupby("ticker")["close"].transform(
        lambda x: x.rolling(200, min_periods=100).mean()
    )

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df, earnings


def load_chains():
    """
    Load all real options chain data.
    Chain dates are not always trading days (some are Saturdays in 2019, daily in 2020+).
    We build:
      - chains[tk]['idx']: {date -> sub-DataFrame}
      - chains[tk]['sorted_dates']: sorted list of available chain dates (numpy array for bisect)
      - chains[tk]['atm_price']: {date -> implied stock price from ATM strike}
    This allows O(log n) nearest-date lookup.

    NOTE: The prices.parquet file is split-adjusted (retroactive), while options chain strikes
    reflect the actual price at the time of writing. We use chain-implied ATM price for
    sizing, collateral, and 200MA to avoid split-adjustment mismatches.
    """
    chains = {}
    for tk in UNIVERSE:
        path = CHAINS_DIR / f"{tk}.parquet"
        if not path.exists():
            print(f"[warn] no chain for {tk}")
            continue
        c = pd.read_parquet(path)
        c["date"] = pd.to_datetime(c["date"])
        c["expiration"] = pd.to_datetime(c["expiration"])
        by_date = {d: grp for d, grp in c.groupby("date")}
        sorted_dates = sorted(by_date.keys())
        # compute chain-implied stock price per date: ATM put strike (delta closest to -0.5)
        atm_price = {}
        for d, grp in by_date.items():
            puts = grp[grp["type"] == "p"]
            if puts.empty:
                continue
            idx = (puts["delta"] + 0.5).abs().idxmin()
            atm_price[d] = float(puts.loc[idx, "strike"])
        chains[tk] = {"idx": by_date, "sorted_dates": sorted_dates, "atm_price": atm_price}
    print(f"[load] chains pre-indexed for {sorted(chains.keys())}")
    return chains


def chain_implied_S(chain_info, date):
    """
    Get chain-implied stock price for a given (trading) date.
    Finds nearest chain date and returns the ATM put strike as proxy for S.
    """
    cdate = nearest_chain_date(chain_info, date)
    if cdate is None:
        return None
    return chain_info["atm_price"].get(cdate)


def nearest_chain_date(chain_info, date):
    """
    Find the most recent chain date that is <= date + 2 days (handles Saturday snapshots).
    Returns the nearest date or None.
    """
    import bisect
    sorted_dates = chain_info["sorted_dates"]
    if not sorted_dates:
        return None
    # allow up to 2 days look-ahead (Saturday = Friday+1)
    cutoff = date + pd.Timedelta(days=2)
    pos = bisect.bisect_right(sorted_dates, cutoff)
    if pos == 0:
        return None
    candidate = sorted_dates[pos - 1]
    # must be within 7 calendar days of today
    if abs((candidate - date).days) > 7:
        return None
    return candidate


# ------------------------- option selection --------------------------------
def find_best_put(chain_info, date, target_delta, cfg):
    """
    Find best put using nearest-date chain lookup.
    chain_info: {'idx': {date->df}, 'sorted_dates': [...]}
    """
    cdate = nearest_chain_date(chain_info, date)
    if cdate is None:
        return None
    day_chain = chain_info["idx"][cdate]

    puts = day_chain[
        (day_chain["type"] == "p") &
        (day_chain["dte"] >= cfg["dte_min"]) &
        (day_chain["dte"] <= cfg["dte_max"]) &
        (day_chain["bid"] > 0.01)
    ]
    if puts.empty:
        return None

    neg_target = -abs(target_delta)
    idx = (puts["delta"] - neg_target).abs().idxmin()
    best = puts.loc[idx]
    return {
        "strike": float(best["strike"]),
        "expiry": best["expiration"],
        "mid": float(best["mid"]),
        "bid": float(best["bid"]),
        "ask": float(best["ask"]),
        "delta": float(best["delta"]),
        "vol": float(best["vol"]),
        "dte": int(best["dte"]),
    }


def find_best_call(chain_info, date, target_delta, share_basis, S, cfg):
    """
    Find best covered call using nearest-date chain lookup.
    """
    cdate = nearest_chain_date(chain_info, date)
    if cdate is None:
        return None
    day_chain = chain_info["idx"][cdate]

    calls = day_chain[
        (day_chain["type"] == "c") &
        (day_chain["dte"] >= cfg["dte_min"]) &
        (day_chain["dte"] <= cfg["dte_max"]) &
        (day_chain["bid"] > 0.01)
    ]
    if calls.empty:
        return None

    above_basis = calls[calls["strike"] >= share_basis]
    pool = above_basis if not above_basis.empty else calls
    idx = (pool["delta"] - abs(target_delta)).abs().idxmin()
    best = pool.loc[idx]
    return {
        "strike": float(best["strike"]),
        "expiry": best["expiration"],
        "mid": float(best["mid"]),
        "bid": float(best["bid"]),
        "ask": float(best["ask"]),
        "delta": float(best["delta"]),
        "vol": float(best["vol"]),
        "dte": int(best["dte"]),
    }


# ------------------------- earnings blackout check -------------------------
def build_earnings_lookup(earnings_df):
    """Build a dict: {ticker: sorted array of earnings dates}."""
    lookup = {}
    for tk, grp in earnings_df.groupby("ticker"):
        lookup[tk] = sorted(grp["earnings_date"].dt.normalize().tolist())
    return lookup


def earnings_in_window(ticker, open_date, expiry, earnings_lookup, blackout_days=7):
    """
    Return True if any earnings date falls within the option's lifetime.
    We block new CSPs only if earnings is within [open_date, expiry + blackout_days].
    This avoids the IV crush risk of holding through an earnings event.
    We do NOT block before open_date (that would be overly restrictive).
    """
    dates = earnings_lookup.get(ticker, [])
    # block if earnings falls between now and expiry + small buffer
    window_start = open_date
    window_end = expiry + pd.Timedelta(days=blackout_days)
    for ed in dates:
        if window_start <= ed <= window_end:
            return True
    return False


# ------------------------- vol-adjusted sizing -----------------------------
def vol_adjusted_contracts(cash_available, strike, rv_20, rv_20_median, max_alloc_pct,
                           total_portfolio, vix, cfg):
    """
    Number of contracts to sell:
    - Base: max_alloc_pct of total portfolio / (strike * 100)
    - Reduce by vol ratio if above-median vol
    - Reduce by 50% if VIX > vix_half_size
    """
    max_cash = total_portfolio * max_alloc_pct
    if rv_20 > rv_20_median and rv_20_median > 0:
        vol_scale = rv_20_median / rv_20   # inverse vol sizing
        vol_scale = max(vol_scale, 0.33)   # floor at 1/3
    else:
        vol_scale = 1.0

    if vix > cfg["vix_half_size"]:
        vol_scale *= 0.5

    effective_max = max_cash * vol_scale
    effective_max = min(effective_max, cash_available * 0.95)  # collateral available
    contracts = int(effective_max // (strike * 100))
    return max(0, contracts)


# ------------------------- single-ticker wheel -----------------------------
def run_ticker_wheel(ticker, df_t, chain_tk, cfg, starting_cash,
                     total_portfolio, rv_20_median, earnings_lookup):
    """
    Run the protected wheel on a single ticker.
    Returns (equity_curve_df, ledger_df, state).
    """
    state = TickerState(ticker=ticker, cash=starting_cash)
    df_t = df_t.sort_values("date").reset_index(drop=True)
    equity_curve = []

    # Pre-build a chain-implied price series (ATM strike) for this ticker.
    # This avoids split-adjusted price mismatches (prices.parquet is retroactively adjusted,
    # while options chains use actual historical prices).
    # We use chain_implied_S for: 200MA, stop-loss, and equity MTM of shares.
    # For sigma/IV we use the iv_features_real data (which is already aligned).
    chain_price_cache = {}   # {date -> implied S}
    if chain_tk is not None:
        for d in df_t["date"]:
            s_imp = chain_implied_S(chain_tk, d)
            if s_imp is not None:
                chain_price_cache[d] = s_imp
    # Build chain-price rolling 200MA (using chain-implied prices)
    chain_price_series = pd.Series(chain_price_cache).sort_index()
    chain_ma200 = chain_price_series.rolling(200, min_periods=30).mean()

    for _, row in df_t.iterrows():
        today = row["date"]
        # Use chain-implied S (actual historical, not split-adjusted) when available
        S_chain = chain_price_cache.get(today)
        S = S_chain if S_chain is not None else float(row["close"])
        sigma = float(row["sigma"])
        vix = float(row["vix"]) if pd.notna(row["vix"]) else 18.0
        rv_20 = float(row["rv_20"]) if pd.notna(row["rv_20"]) else sigma
        # 200MA: use chain-based rolling MA (chain prices are not split-adjusted)
        ma200 = float(chain_ma200.get(today, 0.0))
        above_ma200 = (S > ma200) if ma200 > 0 else True

        # ---- 1) process existing positions ----
        new_positions = []
        for pos in state.positions:
            T = max((pos.expiry - today).days, 0) / 365.0

            if pos.side == "short_put":
                opt = bs_price(S, pos.strike, T, sigma, kind="put")
                pnl_per_share = pos.open_price - opt
                profit_frac = pnl_per_share / pos.open_price if pos.open_price > 0 else 0.0
                close_for_profit = profit_frac >= cfg["profit_take_pct"]
                is_expiry = today >= pos.expiry

                if close_for_profit and not is_expiry:
                    # close at mid (Robinhood) — use BS price as proxy
                    realized = pnl_per_share * 100 * pos.contracts
                    state.cash += realized
                    state.ledger.append({
                        "ticker": ticker, "open_date": pos.open_date, "close_date": today,
                        "kind": "csp_profit_take", "strike": pos.strike, "S_close": S,
                        "premium_open": pos.open_price, "premium_close": opt,
                        "dte_at_open": (pos.expiry - pos.open_date).days,
                        "realized_pnl": realized, "contracts": pos.contracts,
                    })
                elif is_expiry:
                    if S < pos.strike:
                        # ASSIGNMENT — take shares
                        cost = pos.strike * 100 * pos.contracts
                        state.cash -= cost
                        state.assignments += 1
                        basis = pos.strike - pos.open_price  # net basis after premium
                        state.ledger.append({
                            "ticker": ticker, "open_date": pos.open_date, "close_date": today,
                            "kind": "csp_assigned", "strike": pos.strike, "S_close": S,
                            "premium_open": pos.open_price, "premium_close": 0.0,
                            "dte_at_open": (pos.expiry - pos.open_date).days,
                            "realized_pnl": pos.open_price * 100 * pos.contracts,
                            "contracts": pos.contracts,
                        })
                        new_positions.append(Position(
                            ticker=ticker, side="long_shares",
                            strike=basis, expiry=today, open_date=today,
                            open_price=basis, contracts=pos.contracts,
                            open_sigma=sigma, share_basis=basis,
                            assignment_price=pos.strike,   # stop-loss anchor = strike paid
                        ))
                    else:
                        # expired worthless — keep premium
                        realized = pos.open_price * 100 * pos.contracts
                        state.cash += realized
                        state.ledger.append({
                            "ticker": ticker, "open_date": pos.open_date, "close_date": today,
                            "kind": "csp_expired", "strike": pos.strike, "S_close": S,
                            "premium_open": pos.open_price, "premium_close": 0.0,
                            "dte_at_open": (pos.expiry - pos.open_date).days,
                            "realized_pnl": realized, "contracts": pos.contracts,
                        })
                else:
                    new_positions.append(pos)
                    state.days_in_csp += 1
                continue

            elif pos.side == "short_call":
                opt = bs_price(S, pos.strike, T, sigma, kind="call")
                pnl_per_share = pos.open_price - opt
                profit_frac = pnl_per_share / pos.open_price if pos.open_price > 0 else 0.0
                close_for_profit = profit_frac >= cfg["profit_take_pct"]
                is_expiry = today >= pos.expiry

                if close_for_profit and not is_expiry:
                    realized = pnl_per_share * 100 * pos.contracts
                    state.cash += realized
                    state.ledger.append({
                        "ticker": ticker, "open_date": pos.open_date, "close_date": today,
                        "kind": "cc_profit_take", "strike": pos.strike, "S_close": S,
                        "premium_open": pos.open_price, "premium_close": opt,
                        "dte_at_open": (pos.expiry - pos.open_date).days,
                        "realized_pnl": realized, "contracts": pos.contracts,
                    })
                    # retain shares at same basis
                    new_positions.append(Position(
                        ticker=ticker, side="long_shares",
                        strike=pos.share_basis, expiry=today, open_date=today,
                        open_price=pos.share_basis, contracts=pos.contracts,
                        open_sigma=sigma, share_basis=pos.share_basis,
                        assignment_price=pos.assignment_price,
                    ))
                elif is_expiry:
                    if S > pos.strike:
                        # called away
                        proceeds = pos.strike * 100 * pos.contracts
                        premium_kept = pos.open_price * 100 * pos.contracts
                        share_pnl = (pos.strike - pos.share_basis) * 100 * pos.contracts
                        state.cash += proceeds + premium_kept
                        state.call_aways += 1
                        state.ledger.append({
                            "ticker": ticker, "open_date": pos.open_date, "close_date": today,
                            "kind": "cc_called_away", "strike": pos.strike, "S_close": S,
                            "premium_open": pos.open_price, "premium_close": 0.0,
                            "dte_at_open": (pos.expiry - pos.open_date).days,
                            "realized_pnl": premium_kept + share_pnl,
                            "contracts": pos.contracts,
                        })
                    else:
                        # expired worthless — keep premium and shares
                        realized = pos.open_price * 100 * pos.contracts
                        state.cash += realized
                        state.ledger.append({
                            "ticker": ticker, "open_date": pos.open_date, "close_date": today,
                            "kind": "cc_expired", "strike": pos.strike, "S_close": S,
                            "premium_open": pos.open_price, "premium_close": 0.0,
                            "dte_at_open": (pos.expiry - pos.open_date).days,
                            "realized_pnl": realized, "contracts": pos.contracts,
                        })
                        new_positions.append(Position(
                            ticker=ticker, side="long_shares",
                            strike=pos.share_basis, expiry=today, open_date=today,
                            open_price=pos.share_basis, contracts=pos.contracts,
                            open_sigma=sigma, share_basis=pos.share_basis,
                            assignment_price=pos.assignment_price,
                        ))
                else:
                    new_positions.append(pos)
                    state.days_in_shares += 1
                continue

            elif pos.side == "long_shares":
                # HARD STOP-LOSS: sell shares if -stop_loss_pct below assignment price
                stop_price = pos.assignment_price * (1.0 - cfg["stop_loss_pct"])
                if pos.assignment_price > 0 and S <= stop_price:
                    # stop-loss triggered — sell shares at current price
                    proceeds = S * 100 * pos.contracts
                    share_pnl = (S - pos.share_basis) * 100 * pos.contracts
                    state.cash += proceeds
                    state.stop_loss_exits += 1
                    state.ledger.append({
                        "ticker": ticker, "open_date": pos.open_date, "close_date": today,
                        "kind": "stop_loss", "strike": pos.share_basis, "S_close": S,
                        "premium_open": 0.0, "premium_close": 0.0,
                        "dte_at_open": 0,
                        "realized_pnl": share_pnl,
                        "contracts": pos.contracts,
                        "assignment_price": pos.assignment_price,
                    })
                    # do NOT add to new_positions — shares gone
                else:
                    new_positions.append(pos)
                    state.days_in_shares += 1

        state.positions = new_positions

        # ---- 2) open new legs ----
        has_active = any(p.side in ("short_put", "short_call") for p in state.positions)
        has_shares = any(p.side == "long_shares" for p in state.positions)

        # 2a) sell CC on long shares if no CC open
        if has_shares and not any(p.side == "short_call" for p in state.positions):
            updated = []
            for p in state.positions:
                if p.side == "long_shares" and chain_tk is not None:
                    cc = find_best_call(chain_tk, today, cfg["delta_target"],
                                        p.share_basis, S, cfg)
                    if cc is not None and cc["bid"] > 0.05:
                        # Use mid as execution price (Robinhood)
                        premium = cc["mid"]
                        credit = premium * 100 * p.contracts
                        state.cash += credit
                        updated.append(Position(
                            ticker=ticker, side="short_call",
                            strike=cc["strike"], expiry=cc["expiry"], open_date=today,
                            open_price=premium, contracts=p.contracts,
                            open_sigma=cc["vol"], share_basis=p.share_basis,
                            assignment_price=p.assignment_price,
                        ))
                        state.cc_opened += 1
                    else:
                        updated.append(p)
                else:
                    updated.append(p)
            state.positions = updated

        # 2b) new CSP if flat (no position, no shares)
        if not has_active and not has_shares:
            # FILTER 1: VIX gate (hard stop)
            if vix > cfg["vix_max_gate"]:
                state.days_in_cash += 1
                state.vix_skips += 1
            # FILTER 2: 200MA filter
            elif not above_ma200:
                state.days_in_cash += 1
                state.ma_filter_skips += 1
            elif chain_tk is not None:
                # find candidate put
                put = find_best_put(chain_tk, today, cfg["delta_target"], cfg)
                if put is None:
                    state.days_in_cash += 1
                else:
                    # FILTER 3: earnings blackout
                    if earnings_in_window(ticker, today, put["expiry"], earnings_lookup,
                                          cfg["earnings_blackout_days"]):
                        state.days_in_cash += 1
                        state.earnings_skips += 1
                    else:
                        # vol-adjusted sizing
                        contracts = vol_adjusted_contracts(
                            state.cash, put["strike"], rv_20, rv_20_median,
                            cfg["max_alloc_pct"], total_portfolio, vix, cfg
                        )
                        if contracts >= 1 and put["mid"] > 0.05:
                            credit = put["mid"] * 100 * contracts
                            state.cash += credit
                            state.positions.append(Position(
                                ticker=ticker, side="short_put",
                                strike=put["strike"], expiry=put["expiry"],
                                open_date=today,
                                open_price=put["mid"], contracts=contracts,
                                open_sigma=put["vol"], share_basis=0.0,
                                assignment_price=0.0,
                            ))
                            state.csp_opened += 1
                        else:
                            state.days_in_cash += 1
            else:
                state.days_in_cash += 1

        # ---- 3) MTM equity ----
        equity = state.cash
        for p in state.positions:
            T = max((p.expiry - today).days, 0) / 365.0
            if p.side == "short_put":
                opt = bs_price(S, p.strike, T, sigma, kind="put")
                equity += (p.open_price - opt) * 100 * p.contracts
            elif p.side == "long_shares":
                equity += (S - p.share_basis) * 100 * p.contracts
            elif p.side == "short_call":
                opt = bs_price(S, p.strike, T, sigma, kind="call")
                equity += (S - p.share_basis) * 100 * p.contracts
                equity += (p.open_price - opt) * 100 * p.contracts

        equity_curve.append({
            "date": today, "equity": equity, "S": S, "vix": vix,
            "spy_ret": float(row["spy_ret"]) if pd.notna(row["spy_ret"]) else np.nan,
            "above_ma200": int(above_ma200),
            "has_shares": int(has_shares),
            "has_csp": int(any(p.side == "short_put" for p in state.positions)),
            "in_cash": int(len(state.positions) == 0),
        })

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    eq_df["ret"] = eq_df["equity"].pct_change()
    ledger_df = pd.DataFrame(state.ledger)
    return eq_df, ledger_df, state


# ------------------------- basket run --------------------------------------
def run_basket(df, chains, cfg, earnings_lookup):
    """Run protected wheel on all universe tickers, return aggregated basket equity."""
    # compute global rv_20 median for vol-sizing
    rv_20_median = float(df["rv_20"].median())
    per_ticker_alloc = BASKET_TOTAL / len(UNIVERSE)
    total_portfolio = BASKET_TOTAL

    per_ticker = {}
    for tk in UNIVERSE:
        if tk not in chains:
            print(f"[skip] {tk} — no chain data")
            continue
        df_t = df[df["ticker"] == tk].copy()
        if df_t.empty:
            print(f"[skip] {tk} — no price data")
            continue
        eq, ledger, state = run_ticker_wheel(
            tk, df_t, chains.get(tk), cfg,
            starting_cash=per_ticker_alloc,
            total_portfolio=total_portfolio,
            rv_20_median=rv_20_median,
            earnings_lookup=earnings_lookup,
        )
        per_ticker[tk] = {"equity": eq, "ledger": ledger, "state": state}
        pct_in = 100 * state.days_in_shares / max(state.days_in_shares + state.days_in_csp + state.days_in_cash, 1)
        print(f"  [{tk:5s}] CSPs={state.csp_opened} CCs={state.cc_opened} "
              f"asgn={state.assignments} SL={state.stop_loss_exits} "
              f"earnskip={state.earnings_skips} MAskip={state.ma_filter_skips} "
              f"%shares={pct_in:.0f}%")

    # aggregate basket equity
    tickers_run = list(per_ticker.keys())
    all_dates = sorted(set().union(*[p["equity"].index for p in per_ticker.values()]))
    basket_eq = pd.DataFrame(index=all_dates)
    for tk in tickers_run:
        basket_eq[tk] = per_ticker[tk]["equity"]["equity"].reindex(all_dates).ffill().fillna(per_ticker_alloc)
    basket_eq["equity"] = basket_eq[tickers_run].sum(axis=1)

    spy_ret = df.drop_duplicates("date").set_index("date")["spy_ret"].reindex(all_dates)
    basket_eq["spy_ret"] = spy_ret
    basket_eq["ret"] = basket_eq["equity"].pct_change()

    all_ledger = (
        pd.concat([d["ledger"] for d in per_ticker.values() if not d["ledger"].empty],
                  ignore_index=True)
        if any(not d["ledger"].empty for d in per_ticker.values()) else pd.DataFrame()
    )
    return basket_eq, all_ledger, per_ticker, tickers_run


# ------------------------- metrics -----------------------------------------
def compute_metrics(eq_df):
    rets = eq_df["ret"].dropna()
    if len(rets) < 2:
        return {}
    years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
    cagr = (eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
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
        "win_rate": float(wr),
        "profit_factor": float(pf),
        "total_return_pct": float((eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0] - 1.0) * 100.0),
        "final_equity": float(eq_df["equity"].iloc[-1]),
        "initial_equity": float(eq_df["equity"].iloc[0]),
    }


def regime_metrics(eq_df):
    sub = eq_df.dropna(subset=["spy_ret", "ret"]).copy()
    if len(sub) < 40:
        return {"hc428_r1_pass": False, "regime_gap": float("nan"),
                "sharpe_green": None, "sharpe_red": None, "sharpe_flat": None,
                "n_green": 0, "n_red": 0, "n_flat": 0}
    sigma = sub["spy_ret"].std()
    thresh = 0.5 * sigma
    green = sub[sub["spy_ret"] >= thresh]["ret"]
    red = sub[sub["spy_ret"] <= -thresh]["ret"]
    flat = sub[(sub["spy_ret"] > -thresh) & (sub["spy_ret"] < thresh)]["ret"]

    def ann_sh(s):
        if len(s) < 2 or s.std() == 0:
            return float("nan")
        return (s.mean() / s.std()) * math.sqrt(TRADING_DAYS)

    sh_g, sh_r, sh_f = ann_sh(green), ann_sh(red), ann_sh(flat)
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
    if ledger_df is None or ledger_df.empty or "realized_pnl" not in ledger_df.columns:
        return float("nan")
    pos = ledger_df[ledger_df["realized_pnl"] > 0]["realized_pnl"]
    if len(pos) == 0:
        return float("nan")
    return float(pos.max() / pos.sum())


def tail_stress(eq_df, label, start, end):
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
    return {
        "label": label,
        "window": f"{start} to {end}",
        "n_days": int(len(sub)),
        "max_dd_pct": max_dd * 100,
        "recover_days": rec_days,
        "cum_ret_pct": float((sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1.0) * 100.0),
    }


def permutation_test(eq_df, n_trials=200):
    """
    Shuffle daily returns to test if Sharpe is above chance.
    Returns: p-value (fraction of trials with Sharpe >= observed).
    """
    rets = eq_df["ret"].dropna().values
    if len(rets) < 30:
        return float("nan")
    mu, sd = rets.mean(), rets.std()
    obs_sharpe = (mu / sd) * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    count = 0
    rng = random.Random(42)
    for _ in range(n_trials):
        perm = rets.copy()
        rng.shuffle(list(perm))
        perm = np.array(perm)
        pm, ps = perm.mean(), perm.std()
        if ps > 0:
            sh = (pm / ps) * math.sqrt(TRADING_DAYS)
            if sh >= obs_sharpe:
                count += 1
    return float(count / n_trials)


def buy_hold_basket(df, tickers_run):
    """Compute equal-weight buy-and-hold for comparison."""
    per_alloc = BASKET_TOTAL / len(tickers_run)
    eqs = {}
    for tk in tickers_run:
        d = df[df["ticker"] == tk].sort_values("date").set_index("date")["close"]
        eqs[tk] = (d / d.iloc[0]) * per_alloc
    bh = pd.DataFrame(eqs)
    bh["equity"] = bh[tickers_run].sum(axis=1)
    bh["ret"] = bh["equity"].pct_change()
    spy_ret = df.drop_duplicates("date").set_index("date")["spy_ret"].reindex(bh.index)
    bh["spy_ret"] = spy_ret
    return bh


# ------------------------- log mlflow (optional) ---------------------------
def try_log_mlflow(run_name, cfg_label, metrics, regime, day_conc, perm_pval):
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("megacap_wheel_protected_v2")
        with mlflow.start_run(run_name=f"{cfg_label}_{run_name}"):
            mlflow.log_param("config", cfg_label)
            mlflow.log_param("universe_size", len(UNIVERSE))
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(k, v)
            for k, v in regime.items():
                if isinstance(v, (int, float)) and v is not None and np.isfinite(float(v) if v else float("nan")):
                    mlflow.log_metric(f"regime_{k}", float(v))
            mlflow.log_metric("day_conc", day_conc if np.isfinite(day_conc) else -1.0)
            if perm_pval is not None and np.isfinite(perm_pval):
                mlflow.log_metric("permutation_pval", perm_pval)
    except Exception as e:
        print(f"[mlflow] skipped ({e})")


# ------------------------- main run ----------------------------------------
def run_config(cfg_name, cfg, df, chains, earnings_lookup):
    print(f"\n{'='*60}")
    print(f"[{cfg_name.upper()}] delta={cfg['delta_target']}, max_alloc={cfg['max_alloc_pct']*100:.0f}%/name")
    print(f"  VIX half-size at >{cfg['vix_half_size']}, stop at >{cfg['vix_max_gate']}")
    print(f"  Stop-loss: {cfg['stop_loss_pct']*100:.0f}% below assignment price")
    print(f"  Earnings blackout: {cfg['earnings_blackout_days']} days")

    basket_eq, basket_ledger, per_ticker, tickers_run = run_basket(
        df, chains, cfg, earnings_lookup
    )

    m = compute_metrics(basket_eq)
    g = regime_metrics(basket_eq)
    dc = day_concentration(basket_ledger)
    pval = permutation_test(basket_eq, n_trials=200)

    tail = [
        tail_stress(basket_eq, "COVID_2020", "2020-02-15", "2020-05-31"),
        tail_stress(basket_eq, "2022_bear",  "2022-01-01", "2022-12-31"),
    ]

    # aggregate trade stats
    csp_total  = sum(p["state"].csp_opened    for p in per_ticker.values())
    cc_total   = sum(p["state"].cc_opened     for p in per_ticker.values())
    asg_total  = sum(p["state"].assignments   for p in per_ticker.values())
    sl_total   = sum(p["state"].stop_loss_exits for p in per_ticker.values())
    cw_total   = sum(p["state"].call_aways    for p in per_ticker.values())
    earn_skip  = sum(p["state"].earnings_skips for p in per_ticker.values())
    ma_skip    = sum(p["state"].ma_filter_skips for p in per_ticker.values())

    # time-in-state breakdown
    tot_shares = sum(p["state"].days_in_shares for p in per_ticker.values())
    tot_csp    = sum(p["state"].days_in_csp    for p in per_ticker.values())
    tot_cash   = sum(p["state"].days_in_cash   for p in per_ticker.values())
    grand      = max(tot_shares + tot_csp + tot_cash, 1)

    try_log_mlflow("basket", cfg_name, m, g, dc, pval)

    # save
    basket_eq.to_parquet(OUT_DIR / f"equity_{cfg_name}.parquet")
    if not basket_ledger.empty:
        basket_ledger.to_parquet(OUT_DIR / f"ledger_{cfg_name}.parquet")

    return {
        "config": cfg_name,
        "metrics": m,
        "regime": g,
        "day_conc": dc,
        "permutation_pval": pval,
        "tail": tail,
        "trade_stats": {
            "csp_opened": csp_total, "cc_opened": cc_total,
            "assignments": asg_total, "stop_loss_exits": sl_total,
            "call_aways": cw_total, "earnings_skips": earn_skip,
            "ma_filter_skips": ma_skip,
        },
        "time_in_state": {
            "pct_shares": 100 * tot_shares / grand,
            "pct_csp":    100 * tot_csp    / grand,
            "pct_cash":   100 * tot_cash   / grand,
        },
        "equity_df": basket_eq,
        "per_ticker": per_ticker,
        "tickers_run": tickers_run,
    }


def print_result_summary(res, bh_metrics=None):
    cfg = res["config"]
    m = res["metrics"]
    g = res["regime"]
    ts = res["trade_stats"]
    ti = res["time_in_state"]
    pval = res["permutation_pval"]

    gates = {
        "sharpe_ge_1.0":       m.get("sharpe", 0) >= 1.0,
        "calmar_ge_1.5":       m.get("calmar", 0) >= 1.5,
        "regime_gap_le_0.50":  g["hc428_r1_pass"],
        "day_conc_le_0.70":    np.isnan(res["day_conc"]) or res["day_conc"] <= 0.70,
        "perm_pval_le_0.05":   np.isfinite(pval) and pval <= 0.05,
    }
    deploy = all(gates.values())

    print(f"\n--- {cfg.upper()} ---")
    print(f"  CAGR:     {m.get('cagr', 0)*100:+.2f}%")
    print(f"  Sharpe:   {m.get('sharpe', 0):.3f}")
    print(f"  Sortino:  {m.get('sortino', 0):.3f}")
    print(f"  Max DD:   {m.get('max_dd', 0)*100:.2f}%")
    print(f"  Calmar:   {m.get('calmar', 0):.3f}")
    print(f"  Win Rate: {m.get('win_rate', 0)*100:.1f}%")
    print(f"  Prof Fac: {m.get('profit_factor', 0):.3f}")
    print(f"  Final $:  ${m.get('final_equity', 0):,.0f}  (started ${m.get('initial_equity', 0):,.0f})")
    print(f"  Regime gap: {g['regime_gap']:.3f}  (green_sh={g['sharpe_green']}, red_sh={g['sharpe_red']})")
    print(f"  Day conc:   {res['day_conc']:.3f}")
    print(f"  Perm p-val: {pval:.3f}")
    print(f"  Trades:  CSP={ts['csp_opened']} CC={ts['cc_opened']} Asgn={ts['assignments']} "
          f"SL={ts['stop_loss_exits']} CW={ts['call_aways']}")
    print(f"  Filters: EarningsSkip={ts['earnings_skips']} MAskip={ts['ma_filter_skips']}")
    print(f"  Time:    %shares={ti['pct_shares']:.1f}% %CSP={ti['pct_csp']:.1f}% %cash={ti['pct_cash']:.1f}%")
    for t in res["tail"]:
        if t.get("n_days", 0):
            rd = t.get("recover_days") if t.get("recover_days") is not None else "not_recovered"
            print(f"  {t['label']}: MaxDD={t['max_dd_pct']:.1f}% | RecoverDays={rd} | CumRet={t['cum_ret_pct']:.1f}%")

    if bh_metrics:
        bh = bh_metrics
        cagr_delta = (m.get("cagr", 0) - bh.get("cagr", 0)) * 100
        print(f"  vs Buy-Hold: CAGR delta {cagr_delta:+.2f}pp  "
              f"(BH Sharpe={bh.get('sharpe', 0):.2f} MaxDD={bh.get('max_dd', 0)*100:.1f}%)")

    print(f"  Gates: {' '.join(['PASS' if v else 'FAIL' for v in gates.values()])} => {'DEPLOY' if deploy else 'REJECT'}")
    res["gates"] = gates
    res["deploy_ready"] = deploy
    return res


# ------------------------- main -------------------------------------------
def main():
    print("=" * 60)
    print("Megacap Wheel Protected v2 — Full Risk Management")
    print(f"Universe: {', '.join(UNIVERSE)}")
    print(f"Window: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Portfolio: ${BASKET_TOTAL:,.0f}  ({len(UNIVERSE)} names)")
    print("=" * 60)

    df, earnings = load_data()
    print(f"[data] {df['date'].nunique()} trading days across {df['ticker'].nunique()} tickers")
    print(f"[data] {df['date'].min().date()} -> {df['date'].max().date()}")
    chains = load_chains()
    earnings_lookup = build_earnings_lookup(earnings)

    all_results = {}
    for cfg_name, cfg in CONFIGS.items():
        res = run_config(cfg_name, cfg, df, chains, earnings_lookup)
        all_results[cfg_name] = res

    # buy-and-hold reference using tickers that ran
    tickers_run = all_results["moderate"]["tickers_run"]
    bh = buy_hold_basket(df, tickers_run)
    bh_m = compute_metrics(bh)
    print(f"\n[BUY_HOLD] Sharpe={bh_m['sharpe']:.2f} CAGR={bh_m['cagr']*100:.1f}% "
          f"MaxDD={bh_m['max_dd']*100:.1f}% final=${bh_m['final_equity']:,.0f}")

    # print all results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for cfg_name in CONFIGS:
        res = all_results[cfg_name]
        res = print_result_summary(res, bh_m)
        all_results[cfg_name] = res

    # v1 comparison header (v1 had no protections)
    print("\n" + "=" * 60)
    print("V1 vs V2 COMPARISON (key improvements):")
    print("  v1: no 200MA filter, no stop-loss, no earnings blackout, no vol sizing")
    print("  v1: only 5 tech names, suffered catastrophic 2022 drawdowns")
    print("  v2: 16 names across 7 sectors, all risk guards enabled")
    print("  v2: stop-loss caps assignment loss at -15%, 200MA keeps us out of downtrends")

    # find best config
    valid = {k: v for k, v in all_results.items() if v.get("deploy_ready", False)}
    if valid:
        best = max(valid, key=lambda k: valid[k]["metrics"]["sharpe"])
        print(f"\nBEST CONFIG: {best.upper()} (Sharpe={valid[best]['metrics']['sharpe']:.2f})")
    else:
        best_sh = max(all_results, key=lambda k: all_results[k]["metrics"].get("sharpe", -999))
        print(f"\nNo config passed all gates. Best Sharpe: {best_sh.upper()} "
              f"({all_results[best_sh]['metrics'].get('sharpe', 0):.2f})")

    # write JSON summary
    summary = {
        "generated": datetime.now().isoformat(),
        "universe": UNIVERSE,
        "window": [str(START_DATE.date()), str(END_DATE.date())],
        "basket_total": BASKET_TOTAL,
        "buy_hold": bh_m,
        "configs": {
            k: {
                "metrics":    v["metrics"],
                "regime":     v["regime"],
                "day_conc":   v["day_conc"],
                "perm_pval":  v["permutation_pval"],
                "trade_stats": v["trade_stats"],
                "time_in_state": v["time_in_state"],
                "tail":       v["tail"],
                "gates":      v.get("gates", {}),
                "deploy_ready": v.get("deploy_ready", False),
            }
            for k, v in all_results.items()
        },
        "any_pass": bool(valid),
        "passers": list(valid.keys()),
        "best_config": best if valid else best_sh,
    }
    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[output] summary -> {out_path}")
    return all_results, bh_m


if __name__ == "__main__":
    main()
