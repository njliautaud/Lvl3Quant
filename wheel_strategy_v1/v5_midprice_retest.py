#!/usr/bin/env python3
"""
v5_midprice_retest.py — V5 CSP backtest with REAL Dolt chains + mid-price execution.

Tests three execution price scenarios on the Tier2_Balanced config:
  A) Raw bid (worst case — what the pessimistic V6 test used)
  B) Mid price (realistic for Robinhood PFOF)
  C) Bid + 25% of spread (conservative realistic)

Uses REAL Dolt chain data for BOTH entry and exit pricing.
All V5/V8 risk controls active: VIX gate, NAAIM gate, fund_score floor,
sector cap, IV rank floor, max concurrent names, profit-take, etc.

Strategy: Cash-Secured Put (CSP) selling on ~22-delta puts, 30-45 DTE.
No assignment/wheel — CSP-only (close before expiry or let expire OTM).

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python3 v5_midprice_retest.py
"""
from __future__ import annotations

import json
import math
import time
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1")
CACHE = ROOT / "data" / "cache"
CHAINS_DIR = CACHE / "options_real" / "chains"
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/v5_midprice_retest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CASH = 100_000.0
TRADING_DAYS = 252

# ----- V5/V8 Balanced Tier config (from tiers.py) -----
PUT_DELTA_TARGET = 0.22
DTE_MIN = 30
DTE_MAX = 45
DTE_TARGET = 37  # midpoint
PROFIT_TAKE_PCT = 0.50  # Balanced scalp mode (close when 50% profit captured)
ROLL_DTE_TRIGGER = 10    # Close at DTE <= 10 if OTM
MAX_CONCURRENT_NAMES = 18
SECTOR_CAP_PCT = 0.25
VIX_MAX_GATE = 32.0
NAAIM_MIN_GATE = -50.0
FUND_SCORE_FLOOR = 45.0
IV_RANK_FLOOR = 0.25
MAX_NAME_PCT = 0.15  # max 15% of equity in any one name
DELTA_TOLERANCE = 0.12  # accept puts within this of target delta

# Commission: $0 (Robinhood) + ~$0.03 reg fees
COMMISSION_PER_CONTRACT = 0.03


# ----- Data structures -----
@dataclass
class CSPPosition:
    ticker: str
    sector: str
    open_date: pd.Timestamp
    expiration: pd.Timestamp
    strike: float
    contracts: int
    open_premium: float  # per share, at execution price
    open_bid: float
    open_ask: float
    open_mid: float
    open_underlying: float
    open_delta: float
    dte_at_open: int


@dataclass
class TradeRecord:
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    ticker: str
    strike: float
    contracts: int
    dte_at_open: int
    dte_at_close: int
    open_delta: float
    open_premium: float  # per share
    close_price: float   # per share (cost to buy back)
    realized_pnl: float
    exit_reason: str  # 'profit_take' | 'dte_trigger' | 'expired_otm' | 'expired_itm'
    sector: str


# ----- Load data -----

def load_all_chains() -> Dict[str, pd.DataFrame]:
    """Load real Dolt chains for all tickers."""
    chains = {}
    for f in sorted(CHAINS_DIR.iterdir()):
        if f.suffix != ".parquet" or f.stem.endswith("EMPTY"):
            continue
        try:
            df = pd.read_parquet(f)
            df["date"] = pd.to_datetime(df["date"])
            df["expiration"] = pd.to_datetime(df["expiration"])
            for c in ["strike", "bid", "ask", "mid", "delta"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[df["bid"].notna() & df["ask"].notna() & (df["bid"] >= 0)].copy()
            if not df.empty:
                chains[f.stem] = df
        except Exception as e:
            print(f"  Warning: failed to load {f.stem}: {e}")
    return chains


def load_macro() -> pd.DataFrame:
    macro = pd.read_parquet(CACHE / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"])
    return macro.sort_values("date").set_index("date")


def load_universe() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "universe.parquet")


def load_fundamentals() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "fundamentals.parquet")


def load_iv_features() -> pd.DataFrame:
    # Try real-blend first, then modeled
    for name in ["iv_features_real_blend.parquet", "iv_features_modeled.parquet", "iv_features.parquet"]:
        p = CACHE / name
        if p.exists():
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            print(f"  Loaded IV from {name}")
            return df
    raise FileNotFoundError("No IV features file found")


def load_prices() -> pd.DataFrame:
    px = pd.read_parquet(CACHE / "prices.parquet")
    px["date"] = pd.to_datetime(px["date"])
    return px


def load_spy_close() -> pd.Series:
    """SPY close for regime classification."""
    try:
        etf = pd.read_parquet(CACHE / "sector_etfs.parquet")
        spy = etf[etf["ticker"] == "SPY"].copy()
        spy["date"] = pd.to_datetime(spy["date"])
        return spy.set_index("date")["close"].astype(float)
    except Exception:
        return None


# ----- Option selection -----

def find_best_put(chain_day: pd.DataFrame, target_delta: float,
                  dte_min: int, dte_max: int) -> Optional[pd.Series]:
    """Find the put closest to target delta within DTE range."""
    puts = chain_day[(chain_day["type"] == "p") &
                     (chain_day["dte"] >= dte_min) &
                     (chain_day["dte"] <= dte_max) &
                     (chain_day["bid"] > 0)].copy()
    if puts.empty:
        return None
    # Delta for puts is negative; we want |delta| close to target
    puts["delta_abs"] = puts["delta"].abs()
    puts["delta_diff"] = (puts["delta_abs"] - target_delta).abs()
    # Filter to within tolerance
    puts = puts[puts["delta_diff"] <= DELTA_TOLERANCE]
    if puts.empty:
        return None
    # Prefer closest to target delta, then closest to target DTE
    puts["dte_diff"] = (puts["dte"] - DTE_TARGET).abs()
    puts = puts.sort_values(["delta_diff", "dte_diff"])
    return puts.iloc[0]


def find_close_price(chain_day: pd.DataFrame, strike: float,
                     expiration: pd.Timestamp) -> Optional[pd.Series]:
    """Find the matching put for closing a position."""
    match = chain_day[(chain_day["type"] == "p") &
                      (chain_day["strike"] == strike) &
                      (chain_day["expiration"] == expiration)]
    if match.empty:
        return None
    return match.iloc[0]


# ----- Execution price helpers -----

def exec_price_sell(bid: float, ask: float, mid: float, mode: str) -> float:
    """Price we RECEIVE when selling a put (opening CSP)."""
    if mode == "bid":
        return bid
    elif mode == "mid":
        return mid
    elif mode == "bid25":
        return bid + 0.25 * (ask - bid)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def exec_price_buy(bid: float, ask: float, mid: float, mode: str) -> float:
    """Price we PAY when buying back a put (closing CSP)."""
    if mode == "bid":
        return ask  # worst case: buy at ask
    elif mode == "mid":
        return mid
    elif mode == "bid25":
        return ask - 0.25 * (ask - bid)  # buy at ask - 25% of spread
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ----- Metrics -----

def compute_metrics(equity_curve: pd.DataFrame, ledger: pd.DataFrame,
                    starting_cash: float, spy_close: pd.Series = None) -> dict:
    eq = equity_curve["equity"].astype(float)
    if eq.empty:
        return {k: 0.0 for k in ["cagr", "sharpe", "sortino", "max_dd", "pf", "wr",
                                   "n_trades", "final_equity"]}
    daily_ret = eq.pct_change().fillna(0.0)
    days = len(eq) - 1
    yrs = days / TRADING_DAYS if days > 0 else 1.0

    # CAGR
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / yrs) - 1.0 if yrs > 0 and eq.iloc[0] > 0 else 0.0

    # Sharpe
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS) if daily_ret.std() > 0 else 0.0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    sortino = daily_ret.mean() / downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 0 and downside.std() > 0 else 0.0

    # Max DD
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    max_dd = dd.min()

    # PF / WR from ledger
    if not ledger.empty and "realized_pnl" in ledger.columns:
        wins = ledger[ledger["realized_pnl"] > 0]
        losses = ledger[ledger["realized_pnl"] < 0]
        gross_profit = wins["realized_pnl"].sum() if not wins.empty else 0.0
        gross_loss = -losses["realized_pnl"].sum() if not losses.empty else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        wr = len(wins) / len(ledger) if len(ledger) > 0 else 0.0
    else:
        pf, wr = 0.0, 0.0

    # Realized cash curve (for realized metrics)
    realized_eq = _realized_cash_curve(ledger, starting_cash,
                                        pd.DatetimeIndex(equity_curve["date"]))
    realized_ret = realized_eq.pct_change().fillna(0.0)
    r_days = len(realized_eq) - 1
    r_yrs = r_days / TRADING_DAYS if r_days > 0 else 1.0
    r_cagr = (realized_eq.iloc[-1] / realized_eq.iloc[0]) ** (1.0 / r_yrs) - 1.0 if r_yrs > 0 and realized_eq.iloc[0] > 0 else 0.0
    r_sharpe = realized_ret.mean() / realized_ret.std() * np.sqrt(TRADING_DAYS) if realized_ret.std() > 0 else 0.0
    r_downside = realized_ret[realized_ret < 0]
    r_sortino = realized_ret.mean() / r_downside.std() * np.sqrt(TRADING_DAYS) if len(r_downside) > 0 and r_downside.std() > 0 else 0.0
    r_peak = realized_eq.cummax()
    r_dd = (realized_eq / r_peak) - 1.0
    r_max_dd = r_dd.min()

    result = {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "pf": float(pf),
        "wr": float(wr),
        "n_trades": int(len(ledger)),
        "final_equity": float(eq.iloc[-1]),
        "realized_cagr": float(r_cagr),
        "realized_sharpe": float(r_sharpe),
        "realized_sortino": float(r_sortino),
        "realized_max_dd": float(r_max_dd),
        "realized_final_equity": float(realized_eq.iloc[-1]),
    }

    # Regime stratification
    if spy_close is not None:
        result.update(_regime_stratified(realized_ret, spy_close,
                                          pd.DatetimeIndex(equity_curve["date"])))

    return result


def _realized_cash_curve(ledger: pd.DataFrame, starting_cash: float,
                         dates: pd.DatetimeIndex) -> pd.Series:
    if ledger is None or ledger.empty or "realized_pnl" not in ledger.columns:
        return pd.Series([float(starting_cash)] * len(dates), index=dates)
    df = ledger[["close_date", "realized_pnl"]].copy()
    df["close_date"] = pd.to_datetime(df["close_date"])
    df = df.dropna(subset=["close_date"])
    daily_pnl = df.groupby("close_date")["realized_pnl"].sum()
    series = pd.Series(0.0, index=dates)
    series.loc[series.index.isin(daily_pnl.index)] = daily_pnl.reindex(
        series.index[series.index.isin(daily_pnl.index)]).values
    return float(starting_cash) + series.cumsum()


def _regime_stratified(realized_ret: pd.Series, spy_close: pd.Series,
                       dates: pd.DatetimeIndex) -> dict:
    out = {"regime_green_sharpe": float("nan"), "regime_red_sharpe": float("nan"),
           "regime_flat_sharpe": float("nan"), "regime_gap": float("nan"),
           "regime_n_green": 0, "regime_n_red": 0, "regime_n_flat": 0}
    if spy_close is None or len(spy_close) < 3:
        return out
    spy_ret = spy_close.sort_index().pct_change()
    labels = pd.Series("flat", index=spy_ret.index)
    labels[spy_ret > 0.002] = "green"
    labels[spy_ret < -0.002] = "red"
    aligned = labels.reindex(dates)
    for regime in ("green", "red", "flat"):
        sub = realized_ret[aligned == regime]
        out[f"regime_n_{regime}"] = int(len(sub))
        if len(sub) >= 2 and sub.std() > 0:
            out[f"regime_{regime}_sharpe"] = float(
                sub.mean() / sub.std() * np.sqrt(TRADING_DAYS))
    sg, sr = out["regime_green_sharpe"], out["regime_red_sharpe"]
    denom = max(abs(sg) if np.isfinite(sg) else 0, abs(sr) if np.isfinite(sr) else 0)
    if denom > 0 and np.isfinite(sg) and np.isfinite(sr):
        out["regime_gap"] = abs(sg - sr) / denom
    return out


# ----- Main backtest engine -----

def run_csp_backtest(chains: Dict[str, pd.DataFrame],
                     macro: pd.DataFrame,
                     universe: pd.DataFrame,
                     fundamentals: pd.DataFrame,
                     iv_features: pd.DataFrame,
                     prices: pd.DataFrame,
                     exec_mode: str = "mid",
                     start: str = "2019-03-01",
                     end: str = "2026-06-01",
                     fixed_contracts: int = 0,
                     verbose: bool = False) -> dict:
    """
    Run CSP backtest with real chain pricing.

    exec_mode: 'bid' | 'mid' | 'bid25'
    fixed_contracts: if > 0, use this many contracts per trade (no compounding).
                     0 = compound (scale with equity, 15% max per name).
                     -1 = proportional to STARTING capital (5% per name, no compound).
    """
    if fixed_contracts == -1:
        sizing_label = "PROPORTIONAL (5% of starting capital, no compound)"
    elif fixed_contracts > 0:
        sizing_label = f"FIXED {fixed_contracts} contract(s)"
    else:
        sizing_label = "COMPOUND (15% equity)"
    print(f"\n{'='*70}")
    print(f"CSP Backtest — Execution: {exec_mode.upper()} — Sizing: {sizing_label}")
    print(f"{'='*70}")

    # Build lookups
    sector_of = dict(zip(universe["ticker"], universe.get("sector", ["Unknown"] * len(universe))))
    fund_score_of = dict(zip(fundamentals["ticker"], fundamentals.get("fund_score", [50.0] * len(fundamentals))))

    # IV rank by (date, ticker)
    iv_rank_lookup = {}
    for _, row in iv_features.iterrows():
        iv_rank_lookup[(row["date"], row["ticker"])] = row.get("iv_rank", 0.5)

    # Prices by (date, ticker) -> close
    prices_lookup = {}
    for _, row in prices.iterrows():
        prices_lookup[(row["date"], row["ticker"])] = row["close"]

    # Get all unique chain observation dates across all tickers
    all_chain_dates = set()
    chain_dates_by_ticker = {}
    for tk, df in chains.items():
        dates = sorted(df["date"].unique())
        chain_dates_by_ticker[tk] = set(dates)
        all_chain_dates.update(dates)

    # Also need ALL trading dates for equity curve
    all_price_dates = sorted(prices["date"].unique())
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sim_dates = [d for d in all_price_dates if start_ts <= d <= end_ts]

    # State
    cash = STARTING_CASH
    positions: Dict[str, CSPPosition] = {}  # ticker -> position
    equity_records = []
    trade_records = []

    n_opened = 0
    n_closed = 0

    for di, dt in enumerate(sim_dates):
        # Macro gates
        m = macro.loc[:dt].iloc[-1] if dt in macro.index or len(macro.loc[:dt]) > 0 else {}
        vix = m.get("vix", float("nan")) if isinstance(m, dict) else m.get("vix", float("nan")) if hasattr(m, "get") else float("nan")
        try:
            vix = float(macro.loc[:dt, "vix"].iloc[-1])
        except:
            vix = float("nan")
        try:
            naaim = float(macro.loc[:dt, "naaim"].iloc[-1])
        except:
            naaim = float("nan")

        # --- 1. Update/close existing positions ---
        to_remove = []
        for tk, pos in list(positions.items()):
            days_to_exp = (pos.expiration - dt).days

            if days_to_exp <= 0:
                # Expiry — check if ITM
                S = prices_lookup.get((dt, tk))
                if S is None:
                    # Try nearby dates
                    for offset in range(-3, 4):
                        alt_dt = dt + pd.Timedelta(days=offset)
                        S = prices_lookup.get((alt_dt, tk))
                        if S is not None:
                            break
                if S is None:
                    S = pos.open_underlying  # fallback

                if S < pos.strike:
                    # ITM at expiry — assigned. We buy shares at strike, sell at S.
                    # Premium was already credited at open. Net assignment cost:
                    intrinsic = pos.strike - S  # positive, this is what we lose on shares
                    # Cash effect at assignment: buy at strike, immediately sell at S
                    cash_effect = (S - pos.strike) * 100 * pos.contracts  # negative
                    cash += cash_effect
                    # Realized PnL = premium kept - intrinsic loss
                    pnl = (pos.open_premium - intrinsic) * 100 * pos.contracts
                    trade_records.append(TradeRecord(
                        open_date=pos.open_date, close_date=dt, ticker=tk,
                        strike=pos.strike, contracts=pos.contracts,
                        dte_at_open=pos.dte_at_open, dte_at_close=0,
                        open_delta=pos.open_delta, open_premium=pos.open_premium,
                        close_price=intrinsic,  # cost to close = intrinsic value
                        realized_pnl=pnl, exit_reason="expired_itm",
                        sector=pos.sector,
                    ))
                else:
                    # OTM — keep full premium (already credited at open)
                    # No additional cash movement — just record the trade
                    pnl = pos.open_premium * 100 * pos.contracts - COMMISSION_PER_CONTRACT * pos.contracts
                    trade_records.append(TradeRecord(
                        open_date=pos.open_date, close_date=dt, ticker=tk,
                        strike=pos.strike, contracts=pos.contracts,
                        dte_at_open=pos.dte_at_open, dte_at_close=0,
                        open_delta=pos.open_delta, open_premium=pos.open_premium,
                        close_price=0.0, realized_pnl=pnl,
                        exit_reason="expired_otm", sector=pos.sector,
                    ))
                to_remove.append(tk)
                n_closed += 1
                continue

            # Check for real chain data on this date for this ticker
            if dt not in chain_dates_by_ticker.get(tk, set()):
                continue

            chain_day = chains[tk][chains[tk]["date"] == dt]
            close_opt = find_close_price(chain_day, pos.strike, pos.expiration)
            if close_opt is None:
                continue

            close_bid = float(close_opt["bid"])
            close_ask = float(close_opt["ask"])
            close_mid = float(close_opt["mid"])
            buy_price = exec_price_buy(close_bid, close_ask, close_mid, exec_mode)

            # Profit take check
            captured = (pos.open_premium - buy_price) / max(pos.open_premium, 0.01)
            should_close = False
            exit_reason = ""

            if captured >= PROFIT_TAKE_PCT:
                should_close = True
                exit_reason = "profit_take"
            elif days_to_exp <= ROLL_DTE_TRIGGER:
                should_close = True
                exit_reason = "dte_trigger"

            if should_close:
                cost = buy_price * 100 * pos.contracts + COMMISSION_PER_CONTRACT * pos.contracts
                pnl = pos.open_premium * 100 * pos.contracts - cost
                cash -= buy_price * 100 * pos.contracts + COMMISSION_PER_CONTRACT * pos.contracts
                trade_records.append(TradeRecord(
                    open_date=pos.open_date, close_date=dt, ticker=tk,
                    strike=pos.strike, contracts=pos.contracts,
                    dte_at_open=pos.dte_at_open, dte_at_close=days_to_exp,
                    open_delta=pos.open_delta, open_premium=pos.open_premium,
                    close_price=buy_price, realized_pnl=pnl,
                    exit_reason=exit_reason, sector=pos.sector,
                ))
                to_remove.append(tk)
                n_closed += 1

        for tk in to_remove:
            del positions[tk]

        # --- 2. Compute equity (cash + mark-to-market of open positions) ---
        equity = cash
        for tk, pos in positions.items():
            # Use real chain mid for MTM if available today
            days_to_exp = (pos.expiration - dt).days
            if days_to_exp <= 0:
                continue
            if dt in chain_dates_by_ticker.get(tk, set()):
                chain_day = chains[tk][chains[tk]["date"] == dt]
                close_opt = find_close_price(chain_day, pos.strike, pos.expiration)
                if close_opt is not None:
                    mtm_val = float(close_opt["mid"])
                    equity -= mtm_val * 100 * pos.contracts  # short put liability
                    continue
            # Fallback: use open premium decayed by fraction of time
            frac_elapsed = 1.0 - days_to_exp / max(pos.dte_at_open, 1)
            est_val = pos.open_premium * (1.0 - frac_elapsed * PROFIT_TAKE_PCT)
            equity -= est_val * 100 * pos.contracts

        equity_records.append({"date": dt, "equity": equity})

        # --- 3. Open new CSPs ---
        if not np.isnan(vix) and vix > VIX_MAX_GATE:
            continue
        if not np.isnan(naaim) and naaim < NAAIM_MIN_GATE:
            continue
        if len(positions) >= MAX_CONCURRENT_NAMES:
            continue

        # Only open on dates where we have chain data
        candidates = []
        for tk, chain_df in chains.items():
            if tk in positions:
                continue
            if dt not in chain_dates_by_ticker.get(tk, set()):
                continue
            # Fund score gate
            fs = fund_score_of.get(tk, 50.0)
            if fs < FUND_SCORE_FLOOR:
                continue
            # IV rank gate
            iv_rk = iv_rank_lookup.get((dt, tk), 0.5)
            if iv_rk < IV_RANK_FLOOR:
                continue
            # Sector cap
            sector = sector_of.get(tk, "Unknown")
            sec_exp = sum(p.strike * 100 * p.contracts
                         for p in positions.values() if p.sector == sector)
            if equity > 0 and sec_exp / equity > SECTOR_CAP_PCT:
                continue

            # Find best put
            chain_day = chain_df[chain_df["date"] == dt]
            best_put = find_best_put(chain_day, PUT_DELTA_TARGET, DTE_MIN, DTE_MAX)
            if best_put is None:
                continue

            bid = float(best_put["bid"])
            ask = float(best_put["ask"])
            mid_price = float(best_put["mid"])
            if bid <= 0:
                continue

            sell_price = exec_price_sell(bid, ask, mid_price, exec_mode)
            if sell_price <= 0:
                continue

            strike = float(best_put["strike"])
            delta_abs = abs(float(best_put["delta"]))
            dte = int(best_put["dte"])
            expiration = best_put["expiration"]

            S = prices_lookup.get((dt, tk))
            if S is None:
                S = strike / (1.0 - delta_abs)  # rough estimate

            candidates.append({
                "ticker": tk, "S": S, "strike": strike, "sell_price": sell_price,
                "bid": bid, "ask": ask, "mid": mid_price, "delta": delta_abs,
                "dte": dte, "expiration": expiration, "sector": sector,
                "iv_rank": iv_rk, "fund_score": fs,
            })

        # Rank by IV rank (prefer higher) then fund score
        candidates.sort(key=lambda c: (c["iv_rank"], c["fund_score"]), reverse=True)

        # Pace: max slots per day
        slots = min(MAX_CONCURRENT_NAMES - len(positions),
                    max(1, MAX_CONCURRENT_NAMES // 5))

        for c in candidates[:slots]:
            tk = c["ticker"]
            strike = c["strike"]
            sell_price = c["sell_price"]

            # Sizing
            if fixed_contracts == -1:
                # Proportional to starting capital — 5% per name, no compound
                alloc = 0.05 * STARTING_CASH
                n_contracts = max(1, int(alloc // (strike * 100)))
                secure_needed = strike * 100 * n_contracts
                if secure_needed > cash:
                    n_contracts = int(cash // (strike * 100))
                    if n_contracts < 1:
                        continue
            elif fixed_contracts > 0:
                n_contracts = fixed_contracts
                secure_needed = strike * 100 * n_contracts
                if secure_needed > cash:
                    continue
            else:
                # Compound: max 15% of equity per name, cash-secured
                max_alloc = MAX_NAME_PCT * max(equity, 1.0)
                n_contracts = max(1, int(max_alloc // (strike * 100)))
                secure_needed = strike * 100 * n_contracts
                if secure_needed > cash:
                    n_contracts = int(cash // (strike * 100))
                    if n_contracts < 1:
                        continue
                if (strike * 100 * n_contracts) / max(equity, 1.0) > MAX_NAME_PCT:
                    n_contracts = max(1, int(MAX_NAME_PCT * equity // (strike * 100)))
                    if n_contracts < 1:
                        continue

            # Open: credit premium
            credit = sell_price * 100 * n_contracts - COMMISSION_PER_CONTRACT * n_contracts
            cash += credit
            positions[tk] = CSPPosition(
                ticker=tk, sector=c["sector"],
                open_date=dt, expiration=c["expiration"],
                strike=strike, contracts=n_contracts,
                open_premium=sell_price,
                open_bid=c["bid"], open_ask=c["ask"], open_mid=c["mid"],
                open_underlying=c["S"], open_delta=c["delta"],
                dte_at_open=c["dte"],
            )
            n_opened += 1
            if len(positions) >= MAX_CONCURRENT_NAMES:
                break

    # Build DataFrames
    eq_df = pd.DataFrame(equity_records)
    led_df = pd.DataFrame([vars(t) for t in trade_records]) if trade_records else pd.DataFrame()

    print(f"\n  Opened: {n_opened}, Closed: {n_closed}")
    print(f"  Still open: {len(positions)}")
    if not led_df.empty:
        wins = (led_df["realized_pnl"] > 0).sum()
        print(f"  Wins: {wins}/{len(led_df)} = {wins/len(led_df)*100:.1f}%")
        print(f"  Total PnL: ${led_df['realized_pnl'].sum():,.2f}")
        print(f"  Avg PnL/trade: ${led_df['realized_pnl'].mean():,.2f}")

    return {
        "equity_curve": eq_df,
        "ledger": led_df,
        "n_opened": n_opened,
        "n_closed": n_closed,
        "exec_mode": exec_mode,
        "final_cash": cash,
    }


# ----- Permutation test -----

def permutation_test(ledger: pd.DataFrame, n_trials: int = 100) -> dict:
    """Randomize trade order to test if returns are robust to sequencing."""
    if ledger.empty or "realized_pnl" not in ledger.columns:
        return {"p_value": 1.0, "real_sharpe": 0.0, "perm_sharpes": []}

    pnls = ledger["realized_pnl"].values
    real_mean = pnls.mean()

    # Real Sharpe (of trade P&L sequence)
    real_sharpe = real_mean / pnls.std() * np.sqrt(TRADING_DAYS) if pnls.std() > 0 else 0.0

    # Permutation: shuffle trade P&Ls, compute Sharpe
    rng = np.random.default_rng(42)
    count_better = 0
    perm_sharpes = []
    for _ in range(n_trials):
        shuffled = rng.permutation(pnls)
        s_mean = shuffled.mean()
        s_std = shuffled.std()
        s_sharpe = s_mean / s_std * np.sqrt(TRADING_DAYS) if s_std > 0 else 0.0
        perm_sharpes.append(s_sharpe)
        if s_sharpe >= real_sharpe:
            count_better += 1

    p_value = count_better / n_trials
    return {
        "p_value": p_value,
        "real_sharpe": float(real_sharpe),
        "mean_perm_sharpe": float(np.mean(perm_sharpes)),
        "std_perm_sharpe": float(np.std(perm_sharpes)),
        "perm_sharpes": [float(s) for s in perm_sharpes],
    }


# ----- Main -----

def main():
    t0 = time.time()
    print("=" * 70)
    print("V5 CSP RETEST — Real Dolt Chains + Multiple Execution Prices")
    print("=" * 70)

    print("\nLoading data...")
    chains = load_all_chains()
    print(f"  {len(chains)} tickers with real chain data")
    macro = load_macro()
    universe = load_universe()
    fundamentals = load_fundamentals()
    iv_features = load_iv_features()
    prices = load_prices()
    spy_close = load_spy_close()

    # Filter chains to tickers in universe
    uni_tickers = set(universe["ticker"])
    chains = {tk: df for tk, df in chains.items() if tk in uni_tickers}
    print(f"  {len(chains)} tickers after universe filter")

    results = {}

    # ===== PROPORTIONAL SIZING (5% of starting capital, no compound) =====
    print("\n" + "#" * 70)
    print("# PART 1: PROPORTIONAL SIZING (5% starting capital per name)")
    print("#" * 70)

    for mode in ["bid", "mid", "bid25"]:
        r = run_csp_backtest(
            chains=chains, macro=macro, universe=universe,
            fundamentals=fundamentals, iv_features=iv_features, prices=prices,
            exec_mode=mode, start="2019-03-01", end="2026-06-01",
            fixed_contracts=-1,
        )
        metrics = compute_metrics(r["equity_curve"], r["ledger"],
                                   STARTING_CASH, spy_close)
        r["metrics"] = metrics
        results[f"fixed_{mode}"] = r

        print(f"\n  === PROPORTIONAL {mode.upper()} ===")
        print(f"  MTM  CAGR={metrics['cagr']*100:.2f}%  Sharpe={metrics['sharpe']:.2f}  "
              f"Sortino={metrics['sortino']:.2f}  MaxDD={metrics['max_dd']*100:.2f}%")
        print(f"  REAL CAGR={metrics['realized_cagr']*100:.2f}%  R.Sharpe={metrics['realized_sharpe']:.2f}  "
              f"R.MaxDD={metrics['realized_max_dd']*100:.2f}%")
        print(f"  PF={metrics['pf']:.2f}  WR={metrics['wr']*100:.1f}%  Trades={metrics['n_trades']}")
        print(f"  Final equity: ${metrics['final_equity']:,.2f}")

    # ===== COMPOUND (realistic with 15% per-name cap) =====
    print("\n" + "#" * 70)
    print("# PART 2: COMPOUND SIZING (15% equity per name)")
    print("#" * 70)

    for mode in ["bid", "mid", "bid25"]:
        r = run_csp_backtest(
            chains=chains, macro=macro, universe=universe,
            fundamentals=fundamentals, iv_features=iv_features, prices=prices,
            exec_mode=mode, start="2019-03-01", end="2026-06-01",
            fixed_contracts=0,
        )
        metrics = compute_metrics(r["equity_curve"], r["ledger"],
                                   STARTING_CASH, spy_close)
        r["metrics"] = metrics
        results[f"compound_{mode}"] = r

        print(f"\n  === COMPOUND {mode.upper()} ===")
        print(f"  MTM  CAGR={metrics['cagr']*100:.2f}%  Sharpe={metrics['sharpe']:.2f}  "
              f"Sortino={metrics['sortino']:.2f}  MaxDD={metrics['max_dd']*100:.2f}%")
        print(f"  REAL CAGR={metrics['realized_cagr']*100:.2f}%  R.Sharpe={metrics['realized_sharpe']:.2f}  "
              f"R.MaxDD={metrics['realized_max_dd']*100:.2f}%")
        print(f"  PF={metrics['pf']:.2f}  WR={metrics['wr']*100:.1f}%  Trades={metrics['n_trades']}")
        print(f"  Final equity: ${metrics['final_equity']:,.2f}")

    # ----- Comparative summary -----
    print("\n" + "=" * 110)
    print("COMPARATIVE SUMMARY — PROPORTIONAL SIZING (5% starting cap, no compound)")
    print("=" * 110)
    print(f"{'Scenario':<30} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'PF':>5} {'WR%':>5} {'Trades':>7} {'Final$':>12}")
    print("-" * 110)
    for key in ["fixed_bid", "fixed_mid", "fixed_bid25"]:
        r = results[key]
        m = r["metrics"]
        mode = key.split("_", 1)[1]
        label = {"bid": "A: Raw Bid (worst)", "mid": "B: Mid (Robinhood)",
                 "bid25": "C: Bid+25% (conservative)"}[mode]
        print(f"{label:<30} {m['cagr']*100:>6.2f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd']*100:>6.2f}% {m['pf']:>5.2f} {m['wr']*100:>4.1f}% {m['n_trades']:>6d}"
              f"  ${m['final_equity']:>10,.0f}")

    print(f"\n{'Scenario':<30} {'R.CAGR%':>8} {'R.Sharpe':>9} {'R.Sortino':>10} {'R.MaxDD%':>9}")
    print("-" * 75)
    for key in ["fixed_bid", "fixed_mid", "fixed_bid25"]:
        r = results[key]
        m = r["metrics"]
        mode = key.split("_", 1)[1]
        label = {"bid": "A: Raw Bid", "mid": "B: Mid", "bid25": "C: Bid+25%"}[mode]
        print(f"{label:<30} {m['realized_cagr']*100:>7.2f}% {m['realized_sharpe']:>9.2f} "
              f"{m['realized_sortino']:>10.2f} {m['realized_max_dd']*100:>8.2f}%")

    print("\n" + "=" * 110)
    print("COMPOUND SIZING RESULTS (for reference — expect inflated CAGR)")
    print("=" * 110)
    print(f"{'Scenario':<30} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'PF':>5} {'WR%':>5} {'Trades':>7} {'Final$':>12}")
    print("-" * 110)
    for key in ["compound_bid", "compound_mid", "compound_bid25"]:
        r = results[key]
        m = r["metrics"]
        mode = key.split("_", 1)[1]
        label = {"bid": "A: Raw Bid (compound)", "mid": "B: Mid (compound)",
                 "bid25": "C: Bid+25% (compound)"}[mode]
        print(f"{label:<30} {m['cagr']*100:>6.2f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd']*100:>6.2f}% {m['pf']:>5.2f} {m['wr']*100:>4.1f}% {m['n_trades']:>6d}"
              f"  ${m['final_equity']:>10,.0f}")

    # Regime test on FIXED scenarios (the honest ones)
    print("\n" + "=" * 70)
    print("REGIME STRATIFICATION (HC #428 R1) — Fixed 1-Contract")
    print("=" * 70)
    for key in ["fixed_bid", "fixed_mid", "fixed_bid25"]:
        r = results[key]
        m = r["metrics"]
        mode = key.split("_", 1)[1]
        label = {"bid": "A: Raw Bid", "mid": "B: Mid", "bid25": "C: Bid+25%"}[mode]
        gap = m.get("regime_gap", float("nan"))
        print(f"  {label}: Green Sharpe={m.get('regime_green_sharpe', float('nan')):.2f}  "
              f"Red Sharpe={m.get('regime_red_sharpe', float('nan')):.2f}  "
              f"Flat Sharpe={m.get('regime_flat_sharpe', float('nan')):.2f}  "
              f"Gap={gap:.2f}  "
              f"{'PASS' if np.isfinite(gap) and gap <= 0.50 else 'FAIL/NA'}")

    # ----- Permutation test on best fixed scenario -----
    fixed_keys = ["fixed_bid", "fixed_mid", "fixed_bid25"]
    best_key = max(fixed_keys, key=lambda k: results[k]["metrics"]["realized_sharpe"])
    best_mode = best_key.split("_", 1)[1]
    best_label = {"bid": "A: Raw Bid", "mid": "B: Mid", "bid25": "C: Bid+25%"}[best_mode]
    print(f"\n{'='*70}")
    print(f"PERMUTATION TEST — {best_label} Fixed 1-Ctr (100 trials)")
    print(f"{'='*70}")
    perm = permutation_test(results[best_key]["ledger"], n_trials=100)
    print(f"  Real Sharpe: {perm['real_sharpe']:.3f}")
    print(f"  Mean perm Sharpe: {perm['mean_perm_sharpe']:.3f} +/- {perm['std_perm_sharpe']:.3f}")
    print(f"  p-value: {perm['p_value']:.3f}")
    print(f"  Verdict: {'SIGNIFICANT (p < 0.05)' if perm['p_value'] < 0.05 else 'NOT SIGNIFICANT'}")

    # ----- Per-year breakdown for best fixed scenario -----
    print(f"\n{'='*70}")
    print(f"PER-YEAR BREAKDOWN — {best_label} Fixed 1-Ctr")
    print(f"{'='*70}")
    best_led = results[best_key]["ledger"]
    if not best_led.empty:
        best_led = best_led.copy()
        best_led["close_year"] = pd.to_datetime(best_led["close_date"]).dt.year
        for yr in sorted(best_led["close_year"].unique()):
            yr_led = best_led[best_led["close_year"] == yr]
            yr_pnl = yr_led["realized_pnl"].sum()
            yr_wr = (yr_led["realized_pnl"] > 0).sum() / len(yr_led) if len(yr_led) > 0 else 0
            yr_trades = len(yr_led)
            print(f"  {yr}: PnL=${yr_pnl:>8,.0f}  WR={yr_wr*100:.1f}%  Trades={yr_trades}")

    # ----- Save results -----
    summary = {}
    for key, r in results.items():
        m = r["metrics"]
        summary[key] = {
            "exec_mode": r["exec_mode"],
            "sizing": "fixed_1ctr" if "fixed" in key else "compound_15pct",
            "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                           and np.isfinite(v) else str(v))
                        for k, v in m.items()},
            "n_opened": r["n_opened"],
            "n_closed": r["n_closed"],
        }
        # Save equity curve + ledger
        if isinstance(r["equity_curve"], pd.DataFrame) and not r["equity_curve"].empty:
            r["equity_curve"].to_parquet(OUT_DIR / f"equity_{key}.parquet", index=False)
        if isinstance(r["ledger"], pd.DataFrame) and not r["ledger"].empty:
            r["ledger"].to_parquet(OUT_DIR / f"ledger_{key}.parquet", index=False)

    summary["permutation_test"] = {
        "best_scenario": best_key,
        "p_value": perm["p_value"],
        "real_sharpe": perm["real_sharpe"],
    }
    summary["config"] = {
        "put_delta_target": PUT_DELTA_TARGET,
        "dte_min": DTE_MIN, "dte_max": DTE_MAX,
        "profit_take_pct": PROFIT_TAKE_PCT,
        "roll_dte_trigger": ROLL_DTE_TRIGGER,
        "max_concurrent_names": MAX_CONCURRENT_NAMES,
        "sector_cap_pct": SECTOR_CAP_PCT,
        "vix_max_gate": VIX_MAX_GATE,
        "naaim_min_gate": NAAIM_MIN_GATE,
        "fund_score_floor": FUND_SCORE_FLOOR,
        "iv_rank_floor": IV_RANK_FLOOR,
        "commission_per_contract": COMMISSION_PER_CONTRACT,
        "starting_cash": STARTING_CASH,
    }

    with open(OUT_DIR / "v5_midprice_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.0f}s. Results saved.")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    main()
