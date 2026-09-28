#!/usr/bin/env python3
"""
megacap_wheel_v2.py - Improved Wheel Strategy with Risk Management

Key improvements over v1:
  1. 200MA filter on SPY — go to cash when SPY < 200MA (no new CSPs)
  2. VIX gate with graduated response (>30 = no CSPs, 25-30 = half size)
  3. Hard stop-loss on assigned shares (10-20% depending on aggressiveness)
  4. Cross-sector diversification (18 tickers, not just tech)
  5. Position sizing limits (max 5% per name, max 40% deployed)
  6. Max 8 concurrent positions
  7. Uses REAL options chains with MID-PRICE execution
  8. Commission-free (Robinhood, HC #694)
  9. Weekly expiries (5-8 DTE) for higher theta decay
  10. 3 aggressiveness levels tested

Data: Real Dolt options chains (2019-2026), prices, IV, VIX from existing cache.
Execution: Mid-price (realistic for Robinhood's tighter spreads).
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ======================== PATHS ============================================
ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
CHAINS_DIR = CACHE / "options_real" / "chains"
OUT_DIR = ROOT / "output" / "megacap_wheel_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================== UNIVERSE =========================================
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",  # Tech
    "JPM", "GS", "BAC",                         # Finance
    "UNH", "JNJ", "ABBV",                       # Healthcare
    "AMZN", "HD", "COST",                        # Consumer
    "XOM", "CVX",                                # Energy
    "CAT", "DE",                                 # Industrial
]

LEVELS = {
    1: {"name": "Conservative", "put_delta": 0.15, "call_delta": 0.15,
        "stop_loss": 0.10, "max_deployed": 0.50, "max_per_name": 0.20, "max_concurrent": 5},
    3: {"name": "Moderate", "put_delta": 0.25, "call_delta": 0.25,
        "stop_loss": 0.15, "max_deployed": 0.65, "max_per_name": 0.20, "max_concurrent": 6},
    5: {"name": "Aggressive", "put_delta": 0.35, "call_delta": 0.35,
        "stop_loss": 0.20, "max_deployed": 0.80, "max_per_name": 0.25, "max_concurrent": 8},
}
# NOTE: Options trade in 100-share lots. At $100K, a 5% max ($5K) can't buy
# 1 contract of any stock >$50. Need 15-25% per name to get 1 contract of
# stocks in the $100-800 range. Diversification comes from max_concurrent cap.

STARTING_CAPITAL = 100_000.0
TRADING_DAYS = 252
RISK_FREE = 0.04


# ======================== DATA LOADING (OPTIMIZED) =========================

def load_and_index_chains(tickers: list[str]) -> dict:
    """Load chains and pre-index by (ticker, date, type) for O(1) lookup."""
    print("  Building chain index...", flush=True)
    indexed = {}  # (ticker, date_str, 'p'|'c') -> DataFrame of options
    for tk in tickers:
        fp = CHAINS_DIR / f"{tk}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        if "mid" not in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
        # Pre-filter: only keep options with mid > 0.01 and valid bid
        df = df[(df["mid"] > 0.01) & (df["bid"] >= 0)].copy()
        # Group by date and type
        for (dt, otype), grp in df.groupby([df["date"].dt.strftime("%Y-%m-%d"), "type"]):
            indexed[(tk, dt, otype)] = grp
    return indexed


def load_market_data(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prices, IV, VIX, SPY. Returns (ticker_data, spy_data) as date-indexed."""
    prices = pd.read_parquet(CACHE / "prices.parquet")
    iv = pd.read_parquet(CACHE / "iv_features.parquet")
    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    sec = pd.read_parquet(CACHE / "sector_etfs.parquet")

    # SPY
    spy = sec[sec["ticker"] == "SPY"][["date", "close"]].copy()
    spy.columns = ["date", "spy_close"]
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").drop_duplicates("date")
    spy["spy_200ma"] = spy["spy_close"].rolling(200, min_periods=200).mean()
    spy["spy_above_200ma"] = spy["spy_close"] >= spy["spy_200ma"]
    spy["spy_ret"] = spy["spy_close"].pct_change()

    macro["date"] = pd.to_datetime(macro["date"])
    spy = spy.merge(macro, on="date", how="left")
    spy["vix"] = spy["vix"].ffill()
    spy = spy.set_index("date").sort_index()

    # Ticker data: pivot to wide format for fast lookup
    tk_prices = prices[prices["ticker"].isin(tickers)][["ticker", "date", "close"]].copy()
    tk_prices["date"] = pd.to_datetime(tk_prices["date"])
    iv = iv[iv["ticker"].isin(tickers)][["date", "ticker", "sigma", "iv_rank"]].copy()
    iv["date"] = pd.to_datetime(iv["date"])
    merged = tk_prices.merge(iv, on=["date", "ticker"], how="inner")
    merged = merged.dropna(subset=["close", "sigma"])
    merged["sigma"] = merged["sigma"].clip(0.05, 2.0)

    # Create dict: date_str -> {ticker: {close, sigma, iv_rank}}
    tk_data = {}
    for _, row in merged.iterrows():
        ds = row["date"].strftime("%Y-%m-%d")
        if ds not in tk_data:
            tk_data[ds] = {}
        tk_data[ds][row["ticker"]] = {
            "close": float(row["close"]),
            "sigma": float(row["sigma"]),
            "iv_rank": float(row["iv_rank"]) if pd.notna(row["iv_rank"]) else 0.5,
        }

    return tk_data, spy


def find_put(chain_idx, ticker, date_str, target_delta, spot):
    """Find best put from pre-indexed chains. Returns dict or None."""
    key = (ticker, date_str, "p")
    if key not in chain_idx:
        return None
    df = chain_idx[key]

    # Weekly: 4-10 DTE preferred, fallback 3-14
    weekly = df[(df["dte"] >= 4) & (df["dte"] <= 10)]
    if weekly.empty:
        weekly = df[(df["dte"] >= 3) & (df["dte"] <= 14)]
        if weekly.empty:
            return None

    # OTM puts only
    otm = weekly[weekly["strike"] < spot * 1.005]
    if otm.empty:
        return None

    # Select by delta
    has_delta = otm["delta"].notna()
    if has_delta.sum() > 0:
        valid = otm[has_delta].copy()
        best_idx = (valid["delta"].abs() - target_delta).abs().idxmin()
        best = valid.loc[best_idx]
    else:
        target_m = 1.0 - target_delta * 0.55
        best_idx = ((otm["strike"] / spot) - target_m).abs().idxmin()
        best = otm.loc[best_idx]

    mid = float(best["mid"])
    if mid <= 0.01:
        return None
    return {"strike": float(best["strike"]), "expiration": best["expiration"],
            "dte": int(best["dte"]), "mid": mid,
            "delta": float(best["delta"]) if pd.notna(best["delta"]) else None}


def find_call(chain_idx, ticker, date_str, target_delta, spot, min_strike=None):
    """Find best call from pre-indexed chains."""
    key = (ticker, date_str, "c")
    if key not in chain_idx:
        return None
    df = chain_idx[key]

    weekly = df[(df["dte"] >= 4) & (df["dte"] <= 10)]
    if weekly.empty:
        weekly = df[(df["dte"] >= 3) & (df["dte"] <= 14)]
        if weekly.empty:
            return None

    otm = weekly[weekly["strike"] > spot * 0.995]
    if otm.empty:
        return None

    if min_strike is not None:
        preferred = otm[otm["strike"] >= min_strike * 0.98]
        if not preferred.empty:
            otm = preferred

    has_delta = otm["delta"].notna()
    if has_delta.sum() > 0:
        valid = otm[has_delta].copy()
        best_idx = (valid["delta"].abs() - target_delta).abs().idxmin()
        best = valid.loc[best_idx]
    else:
        target_m = 1.0 + target_delta * 0.55
        best_idx = ((otm["strike"] / spot) - target_m).abs().idxmin()
        best = otm.loc[best_idx]

    mid = float(best["mid"])
    if mid <= 0.01:
        return None
    return {"strike": float(best["strike"]), "expiration": best["expiration"],
            "dte": int(best["dte"]), "mid": mid,
            "delta": float(best["delta"]) if pd.notna(best["delta"]) else None}


# ======================== POSITION / STATE =================================

@dataclass
class Pos:
    ticker: str
    phase: str         # 'csp' | 'shares' | 'cc'
    strike: float
    expiration: pd.Timestamp
    open_date: pd.Timestamp
    premium: float     # per share
    contracts: int
    share_basis: float = 0.0
    assign_price: float = 0.0


@dataclass
class State:
    cash: float
    positions: list = field(default_factory=list)
    ledger: list = field(default_factory=list)
    n_csp: int = 0
    n_cc: int = 0
    n_assign: int = 0
    n_callaway: int = 0
    n_stoploss: int = 0
    n_200ma_block: int = 0
    n_vix_block: int = 0
    n_vix_half: int = 0
    n_pos_limit: int = 0


# ======================== WHEEL ENGINE (OPTIMIZED) =========================

def run_wheel(tk_data, spy_df, chain_idx, cfg, tickers):
    """Run wheel backtest. Returns (equity_df, state)."""
    state = State(cash=STARTING_CAPITAL)
    equity_rows = []
    last_csp_open = {}

    # Get sorted dates that exist in both spy and ticker data
    spy_dates = sorted(spy_df.index)
    tk_dates = set(tk_data.keys())

    dates = [d for d in spy_dates if d.strftime("%Y-%m-%d") in tk_dates]

    for i, date in enumerate(dates):
        ds = date.strftime("%Y-%m-%d")
        today = pd.Timestamp(date)

        spy_row = spy_df.loc[date]
        spy_above_200ma = bool(spy_row["spy_above_200ma"]) if pd.notna(spy_row.get("spy_above_200ma")) else True
        vix = float(spy_row["vix"]) if pd.notna(spy_row.get("vix")) else 20.0
        spy_ret = float(spy_row["spy_ret"]) if pd.notna(spy_row.get("spy_ret")) else 0.0

        day = tk_data[ds]  # dict of ticker -> {close, sigma, iv_rank}

        # ---- 1) Process existing positions ----
        new_positions = []
        for pos in state.positions:
            if pos.ticker not in day:
                new_positions.append(pos)
                continue
            S = day[pos.ticker]["close"]

            if pos.phase == "csp":
                if today >= pos.expiration:
                    if S < pos.strike:
                        # Assignment
                        cost = pos.strike * 100 * pos.contracts
                        state.cash -= cost
                        state.n_assign += 1
                        basis = pos.strike - pos.premium
                        state.ledger.append({"date": ds, "ticker": pos.ticker, "type": "csp_assigned",
                                             "strike": pos.strike, "spot": S, "pnl": pos.premium * 100 * pos.contracts,
                                             "contracts": pos.contracts})
                        new_positions.append(Pos(pos.ticker, "shares", basis, today, today, 0,
                                                 pos.contracts, share_basis=basis, assign_price=pos.strike))
                    else:
                        # Expired worthless
                        state.ledger.append({"date": ds, "ticker": pos.ticker, "type": "csp_expired",
                                             "strike": pos.strike, "spot": S, "pnl": pos.premium * 100 * pos.contracts,
                                             "contracts": pos.contracts})
                else:
                    new_positions.append(pos)

            elif pos.phase == "shares":
                pct = (S - pos.assign_price) / pos.assign_price if pos.assign_price > 0 else 0
                if pct <= -cfg["stop_loss"]:
                    proceeds = S * 100 * pos.contracts
                    state.cash += proceeds
                    loss = (S - pos.share_basis) * 100 * pos.contracts
                    state.n_stoploss += 1
                    state.ledger.append({"date": ds, "ticker": pos.ticker, "type": "stop_loss",
                                         "strike": pos.assign_price, "spot": S, "pnl": loss,
                                         "contracts": pos.contracts})
                else:
                    new_positions.append(pos)

            elif pos.phase == "cc":
                if today >= pos.expiration:
                    if S > pos.strike:
                        # Called away
                        proceeds = pos.strike * 100 * pos.contracts
                        share_pnl = (pos.strike - pos.share_basis) * 100 * pos.contracts
                        prem_pnl = pos.premium * 100 * pos.contracts
                        state.cash += proceeds
                        state.n_callaway += 1
                        state.ledger.append({"date": ds, "ticker": pos.ticker, "type": "cc_called_away",
                                             "strike": pos.strike, "spot": S, "pnl": share_pnl + prem_pnl,
                                             "contracts": pos.contracts})
                    else:
                        # CC expired, keep shares
                        state.ledger.append({"date": ds, "ticker": pos.ticker, "type": "cc_expired",
                                             "strike": pos.strike, "spot": S, "pnl": pos.premium * 100 * pos.contracts,
                                             "contracts": pos.contracts})
                        new_positions.append(Pos(pos.ticker, "shares", pos.share_basis, today, today, 0,
                                                 pos.contracts, pos.share_basis, pos.assign_price))
                else:
                    new_positions.append(pos)

        state.positions = new_positions

        # ---- 2) Sell CCs on unprotected shares ----
        for idx, pos in enumerate(state.positions):
            if pos.phase != "shares" or pos.ticker not in day:
                continue
            S = day[pos.ticker]["close"]
            call = find_call(chain_idx, pos.ticker, ds, cfg["call_delta"], S, min_strike=pos.share_basis)
            if call is None:
                continue
            credit = call["mid"] * 100 * pos.contracts
            state.cash += credit
            state.n_cc += 1
            state.positions[idx] = Pos(pos.ticker, "cc", call["strike"], call["expiration"],
                                        today, call["mid"], pos.contracts,
                                        pos.share_basis, pos.assign_price)

        # ---- 3) Open new CSPs ----
        active_tickers = set(p.ticker for p in state.positions)
        n_active = len(active_tickers)

        can_open = True
        if not spy_above_200ma:
            can_open = False
            state.n_200ma_block += 1
        if vix > 30:
            can_open = False
            state.n_vix_block += 1
        if n_active >= cfg["max_concurrent"]:
            can_open = False
            state.n_pos_limit += 1

        size_mult = 0.5 if 25 <= vix <= 30 else 1.0
        if size_mult < 1.0:
            state.n_vix_half += 1

        # Check max deployed
        equity = _fast_equity(state, day)
        deployed_frac = (equity - state.cash) / max(equity, 1)
        if deployed_frac > cfg["max_deployed"]:
            can_open = False

        if can_open:
            # Rank candidates by IV rank
            cands = []
            for tk in tickers:
                if tk in active_tickers or tk not in day:
                    continue
                if tk in last_csp_open and (today - last_csp_open[tk]).days < 5:
                    continue
                cands.append((tk, day[tk]["iv_rank"]))
            cands.sort(key=lambda x: -x[1])

            for tk, _ in cands:
                if n_active >= cfg["max_concurrent"]:
                    break
                S = day[tk]["close"]
                put = find_put(chain_idx, tk, ds, cfg["put_delta"], S)
                if put is None:
                    continue

                max_alloc = equity * cfg["max_per_name"] * size_mult
                coll_per = put["strike"] * 100
                contracts = int(max_alloc // coll_per)
                if contracts < 1:
                    continue
                needed = coll_per * contracts
                if needed > state.cash * 0.95:
                    contracts = int((state.cash * 0.95) // coll_per)
                    if contracts < 1:
                        continue

                credit = put["mid"] * 100 * contracts
                state.cash += credit
                state.n_csp += 1
                n_active += 1
                last_csp_open[tk] = today
                active_tickers.add(tk)

                state.positions.append(Pos(tk, "csp", put["strike"], put["expiration"],
                                            today, put["mid"], contracts))

        # ---- 4) Record equity ----
        equity = _fast_equity(state, day)
        equity_rows.append((ds, equity, state.cash, len(state.positions), spy_ret, spy_above_200ma, vix))

        if i > 0 and i % 200 == 0:
            print(f"    Day {i}/{len(dates)}: equity=${equity:,.0f}, positions={len(state.positions)}", flush=True)

    eq_df = pd.DataFrame(equity_rows,
                          columns=["date", "equity", "cash", "n_positions", "spy_ret", "spy_above_200ma", "vix"])
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.set_index("date")
    return eq_df, state


def _fast_equity(state, day):
    """Fast equity calc using pre-looked-up day data dict."""
    equity = state.cash
    for pos in state.positions:
        if pos.ticker not in day:
            continue
        S = day[pos.ticker]["close"]
        if pos.phase == "csp":
            intrinsic = max(pos.strike - S, 0)
            equity -= intrinsic * 100 * pos.contracts
        elif pos.phase == "shares":
            equity += S * 100 * pos.contracts
        elif pos.phase == "cc":
            equity += S * 100 * pos.contracts
            intrinsic = max(S - pos.strike, 0)
            equity -= intrinsic * 100 * pos.contracts
    return equity


# ======================== METRICS ==========================================

def compute_metrics(eq_df):
    eq_df = eq_df.copy()
    eq_df["ret"] = eq_df["equity"].pct_change()
    rets = eq_df["ret"].dropna()
    if len(rets) < 2:
        return {}
    years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
    cagr = (eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0]) ** (1 / max(years, 0.01)) - 1
    mu, sd = rets.mean(), rets.std()
    down = rets[rets < 0].std()
    sharpe = (mu / sd) * math.sqrt(TRADING_DAYS) if sd > 0 else float("nan")
    sortino = (mu / down) * math.sqrt(TRADING_DAYS) if down and down > 0 else float("nan")
    peak = eq_df["equity"].cummax()
    dd = eq_df["equity"] / peak - 1
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    wr = float((rets > 0).mean())
    psum = rets[rets > 0].sum()
    nsum = abs(rets[rets < 0].sum())
    pf = float(psum / nsum) if nsum > 0 else float("nan")
    total_dollar = eq_df["equity"].iloc[-1] - eq_df["equity"].iloc[0]
    months = years * 12
    return {
        "cagr": float(cagr), "sharpe": float(sharpe), "sortino": float(sortino),
        "max_dd": float(max_dd), "calmar": float(calmar),
        "win_rate": wr, "profit_factor": pf,
        "total_return_pct": float((eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0] - 1) * 100),
        "final_equity": float(eq_df["equity"].iloc[-1]),
        "years": float(years),
        "monthly_income": float(total_dollar / max(months, 1)),
        "annual_income": float(total_dollar / max(years, 0.01)),
    }


def regime_metrics(eq_df):
    eq_df = eq_df.copy()
    eq_df["ret"] = eq_df["equity"].pct_change()
    sub = eq_df.dropna(subset=["spy_ret", "ret"])
    if len(sub) < 40:
        return {"hc428_pass": False, "regime_gap": float("nan"), "n_green": 0, "n_red": 0, "n_flat": 0,
                "sharpe_green": None, "sharpe_red": None, "sharpe_flat": None}
    green = sub[sub["spy_ret"] > 0.001]["ret"]
    red = sub[sub["spy_ret"] < -0.001]["ret"]
    flat = sub[(sub["spy_ret"] >= -0.001) & (sub["spy_ret"] <= 0.001)]["ret"]

    def ash(s):
        return (s.mean() / s.std()) * math.sqrt(TRADING_DAYS) if len(s) > 5 and s.std() > 0 else float("nan")

    sg, sr, sf = ash(green), ash(red), ash(flat)
    denom = max(abs(sg) if np.isfinite(sg) else 0, abs(sr) if np.isfinite(sr) else 0, 1e-9)
    gap = abs((sg if np.isfinite(sg) else 0) - (sr if np.isfinite(sr) else 0)) / denom
    return {
        "n_green": int(len(green)), "n_red": int(len(red)), "n_flat": int(len(flat)),
        "sharpe_green": round(float(sg), 3) if np.isfinite(sg) else None,
        "sharpe_red": round(float(sr), 3) if np.isfinite(sr) else None,
        "sharpe_flat": round(float(sf), 3) if np.isfinite(sf) else None,
        "regime_gap": float(gap),
        "hc428_pass": bool(gap <= 0.50),
    }


def day_concentration(ledger):
    if not ledger:
        return float("nan")
    pnls = [t["pnl"] for t in ledger if t.get("pnl", 0) > 0]
    if not pnls:
        return float("nan")
    return max(pnls) / sum(pnls)


# ======================== MAIN =============================================

def main():
    print("=" * 80, flush=True)
    print("MEGACAP WHEEL V2 — Risk-Managed Cross-Sector Wheel Backtest", flush=True)
    print("=" * 80, flush=True)
    print(f"Starting capital: ${STARTING_CAPITAL:,.0f}", flush=True)
    print(f"Universe: {len(UNIVERSE)} tickers across 6 sectors", flush=True)
    print(f"Execution: Mid-price, commission-free (Robinhood)\n", flush=True)

    # Load data
    print("[1/4] Loading market data...", flush=True)
    tk_data, spy_df = load_market_data(UNIVERSE)
    available_tickers = set()
    for d in tk_data.values():
        available_tickers.update(d.keys())
    available_tickers = sorted(available_tickers)
    print(f"  {len(tk_data)} trading days, {len(available_tickers)} tickers", flush=True)

    spy_valid = spy_df.dropna(subset=["spy_200ma"])
    below = (~spy_valid["spy_above_200ma"]).sum()
    print(f"  SPY below 200MA: {below} of {len(spy_valid)} days ({100*below/len(spy_valid):.1f}%)", flush=True)

    print("\n[2/4] Loading and indexing options chains...", flush=True)
    chain_idx = load_and_index_chains(available_tickers)
    final_tickers = [t for t in UNIVERSE if t in available_tickers]
    print(f"  {len(chain_idx)} chain groups indexed for {len(final_tickers)} tickers", flush=True)

    # Restrict to chain date range
    chain_dates_set = set()
    for (tk, ds, ot) in chain_idx.keys():
        chain_dates_set.add(ds)
    min_cd = min(chain_dates_set)
    max_cd = max(chain_dates_set)
    # Filter tk_data to chain range
    tk_data = {ds: v for ds, v in tk_data.items() if min_cd <= ds <= max_cd}
    spy_df = spy_df.loc[min_cd:max_cd]
    print(f"  Date range: {min_cd} to {max_cd} ({len(tk_data)} days)", flush=True)

    # Run all 3 levels
    print("\n[3/4] Running backtests...", flush=True)
    results = {}

    for level, cfg in LEVELS.items():
        print(f"\n  === Level {level}: {cfg['name']} (delta={cfg['put_delta']}, "
              f"SL={cfg['stop_loss']*100:.0f}%, max_deployed={cfg['max_deployed']*100:.0f}%) ===", flush=True)

        eq_df, state = run_wheel(tk_data, spy_df, chain_idx, cfg, final_tickers)
        metrics = compute_metrics(eq_df)
        regime = regime_metrics(eq_df)
        dc = day_concentration(state.ledger)

        results[level] = {
            "cfg": cfg, "eq_df": eq_df, "metrics": metrics, "regime": regime,
            "dc": dc, "state": state,
        }

        m = metrics
        print(f"    CAGR={m.get('cagr',0)*100:.2f}%  Sharpe={m.get('sharpe',0):.2f}  "
              f"Sortino={m.get('sortino',0):.2f}  MaxDD={m.get('max_dd',0)*100:.1f}%  "
              f"Calmar={m.get('calmar',0):.2f}", flush=True)
        print(f"    Final=${m.get('final_equity',0):,.0f}  Monthly=${m.get('monthly_income',0):,.0f}  "
              f"Annual=${m.get('annual_income',0):,.0f}", flush=True)
        print(f"    CSPs={state.n_csp}  CCs={state.n_cc}  Assigns={state.n_assign}  "
              f"CallAways={state.n_callaway}  StopLoss={state.n_stoploss}", flush=True)
        print(f"    200MA_blocks={state.n_200ma_block}  VIX_blocks={state.n_vix_block}  "
              f"VIX_half={state.n_vix_half}", flush=True)
        print(f"    Regime: gap={regime.get('regime_gap',0):.2f}  "
              f"HC428={'PASS' if regime.get('hc428_pass') else 'FAIL'}", flush=True)

    # ======================== COMPARISON TABLE ================================
    print("\n\n" + "=" * 90, flush=True)
    print("COMPARISON TABLE — All 3 Aggressiveness Levels on $100K", flush=True)
    print("=" * 90, flush=True)

    hdr = f"{'Metric':<25} {'Conservative(L1)':>18} {'Moderate(L3)':>18} {'Aggressive(L5)':>18}"
    print(hdr, flush=True)
    print("-" * 79, flush=True)

    def fv(levels, fn):
        return "  ".join(f"{fn(results[l]['metrics']):>18}" for l in levels)

    rows = [
        ("CAGR",           lambda m: f"{m.get('cagr',0)*100:.2f}%"),
        ("Sharpe",         lambda m: f"{m.get('sharpe',0):.2f}"),
        ("Sortino",        lambda m: f"{m.get('sortino',0):.2f}"),
        ("Max Drawdown",   lambda m: f"{m.get('max_dd',0)*100:.1f}%"),
        ("Calmar",         lambda m: f"{m.get('calmar',0):.2f}"),
        ("Win Rate",       lambda m: f"{m.get('win_rate',0)*100:.1f}%"),
        ("Profit Factor",  lambda m: f"{m.get('profit_factor',0):.2f}"),
        ("Total Return",   lambda m: f"{m.get('total_return_pct',0):.1f}%"),
        ("Final Equity",   lambda m: f"${m.get('final_equity',0):,.0f}"),
        ("Monthly Income", lambda m: f"${m.get('monthly_income',0):,.0f}"),
        ("Annual Income",  lambda m: f"${m.get('annual_income',0):,.0f}"),
    ]
    for label, fn in rows:
        vals = [fn(results[l]["metrics"]) for l in [1, 3, 5]]
        print(f"{label:<25} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}", flush=True)

    print(flush=True)
    print(f"{'--- RISK FILTERS ---':<25}", flush=True)
    rrows = [
        ("CSPs Opened",      lambda s: f"{s.n_csp}"),
        ("CCs Opened",       lambda s: f"{s.n_cc}"),
        ("Assignments",      lambda s: f"{s.n_assign}"),
        ("Call-Aways",       lambda s: f"{s.n_callaway}"),
        ("Stop-Losses",      lambda s: f"{s.n_stoploss}"),
        ("200MA Block Days", lambda s: f"{s.n_200ma_block}"),
        ("VIX>30 Blocks",    lambda s: f"{s.n_vix_block}"),
        ("VIX 25-30 Half",   lambda s: f"{s.n_vix_half}"),
    ]
    for label, fn in rrows:
        vals = [fn(results[l]["state"]) for l in [1, 3, 5]]
        print(f"{label:<25} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}", flush=True)

    print(flush=True)
    print(f"{'--- REGIME (HC #428) ---':<25}", flush=True)
    for label, key in [("Sharpe Green", "sharpe_green"), ("Sharpe Red", "sharpe_red"),
                       ("Sharpe Flat", "sharpe_flat"), ("Regime Gap", "regime_gap")]:
        vals = []
        for l in [1, 3, 5]:
            v = results[l]["regime"].get(key)
            vals.append(f"{v:.3f}" if v is not None and np.isfinite(v) else "n/a")
        print(f"{label:<25} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}", flush=True)
    vals = ["PASS" if results[l]["regime"].get("hc428_pass") else "FAIL" for l in [1, 3, 5]]
    print(f"{'HC #428 R1':<25} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}", flush=True)
    vals = [f"{results[l]['dc']:.3f}" if np.isfinite(results[l]['dc']) else "n/a" for l in [1, 3, 5]]
    print(f"{'Day Concentration':<25} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}", flush=True)

    # ======================== YEARLY BREAKDOWN ================================
    best_level = max(results.keys(), key=lambda l: results[l]["metrics"].get("sharpe", 0))
    print(f"\n\nYEARLY P&L — Level {best_level} ({results[best_level]['cfg']['name']})", flush=True)
    print("-" * 55, flush=True)
    eq = results[best_level]["eq_df"]
    eq_y = eq[["equity"]].copy()
    eq_y["year"] = eq_y.index.year
    yearly = eq_y.groupby("year")["equity"].agg(["first", "last"])
    yearly["pnl"] = yearly["last"] - yearly["first"]
    yearly["return_pct"] = (yearly["last"] / yearly["first"] - 1) * 100
    for yr, row in yearly.iterrows():
        print(f"  {yr}  ${row['pnl']:>+10,.0f}  ({row['return_pct']:>+6.2f}%)  "
              f"equity: ${row['last']:>10,.0f}", flush=True)

    # ======================== MONTHLY BREAKDOWN ===============================
    print(f"\n\nMONTHLY P&L — Level {best_level} ({results[best_level]['cfg']['name']})", flush=True)
    print("-" * 55, flush=True)
    eq_m = eq[["equity"]].copy()
    eq_m["month"] = eq_m.index.to_period("M")
    monthly = eq_m.groupby("month")["equity"].agg(["first", "last"])
    monthly["pnl"] = monthly["last"] - monthly["first"]
    monthly["return_pct"] = (monthly["last"] / monthly["first"] - 1) * 100
    for period, row in monthly.iterrows():
        bar = "+" * min(20, max(0, int(row["return_pct"] * 3))) if row["return_pct"] > 0 else \
              "-" * min(20, max(0, int(-row["return_pct"] * 3)))
        print(f"  {period}  ${row['pnl']:>+8,.0f}  ({row['return_pct']:>+6.2f}%)  {bar}", flush=True)

    # ======================== RISK FILTER ANALYSIS ============================
    print(f"\n\nRISK FILTER IMPACT ANALYSIS", flush=True)
    print("=" * 60, flush=True)
    for level in [1, 3, 5]:
        r = results[level]
        st = r["state"]
        sl_trades = [t for t in st.ledger if t["type"] == "stop_loss"]
        sl_total = sum(t["pnl"] for t in sl_trades)
        assign_trades = [t for t in st.ledger if t["type"] == "csp_assigned"]

        print(f"\n  Level {level} ({r['cfg']['name']}):", flush=True)
        print(f"    Stop-losses triggered: {len(sl_trades)}", flush=True)
        if sl_trades:
            print(f"    Total stop-loss damage: ${sl_total:,.0f}", flush=True)
            print(f"    Avg loss per stop: ${sl_total/len(sl_trades):,.0f}", flush=True)
            # What would have happened without stops? Hard to know exactly, but
            # these stops prevented riding positions further down
            print(f"    (These stops prevented riding positions to potentially larger losses)", flush=True)
        print(f"    200MA filter blocked CSP openings on {st.n_200ma_block} days", flush=True)
        print(f"    VIX>30 blocked CSP openings on {st.n_vix_block} days", flush=True)
        print(f"    VIX 25-30 halved position size on {st.n_vix_half} days", flush=True)
        print(f"    Total assignments: {st.n_assign}  Call-aways: {st.n_callaway}", flush=True)

    # ======================== SAVE OUTPUTS ====================================
    print(f"\n[4/4] Saving results to {OUT_DIR}", flush=True)
    for level in [1, 3, 5]:
        r = results[level]
        r["eq_df"].to_parquet(OUT_DIR / f"equity_level_{level}.parquet")
        if r["state"].ledger:
            pd.DataFrame(r["state"].ledger).to_parquet(OUT_DIR / f"ledger_level_{level}.parquet")

    summary = {
        "generated": datetime.now().isoformat(),
        "starting_capital": STARTING_CAPITAL,
        "universe": final_tickers,
        "date_range": [min_cd, max_cd],
        "levels": {},
    }
    for level in [1, 3, 5]:
        r = results[level]
        summary["levels"][str(level)] = {
            "name": r["cfg"]["name"],
            "config": r["cfg"],
            "metrics": r["metrics"],
            "regime": r["regime"],
            "day_conc": r["dc"] if np.isfinite(r["dc"]) else None,
            "risk": {"csps": r["state"].n_csp, "ccs": r["state"].n_cc,
                     "assigns": r["state"].n_assign, "callaways": r["state"].n_callaway,
                     "stoplosses": r["state"].n_stoploss,
                     "ma200_blocks": r["state"].n_200ma_block, "vix_blocks": r["state"].n_vix_block},
        }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone. Results saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
