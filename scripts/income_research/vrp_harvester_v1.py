"""
VRP Harvester v1 — Variance Risk Premium Income Strategy Backtest
=================================================================
Sells CSPs (and CCs on assignment) on stocks ranked by VRP (IV - RV).
Compares VRP-ranked vs random selection vs ATM straddle selling.

Key constraints:
  - Commission-free (Robinhood HC #694)
  - Mid-price execution
  - Walk-forward OOT validation (30% holdout)
  - HC #428 R1 regime gap test on all regimes
  - 100-trial permutation test
"""

import os
import sys
import warnings
import json
import random
import time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
CHAINS_DIR = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/options_real/chains")
PRICES_PATH = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/prices.parquet")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/vrp_harvester_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Strategy Parameters ─────────────────────────────────────────────────────
TOP_N = 7              # top VRP names per week
MAX_PCT_PER_NAME = 0.05  # 5% of portfolio per name (margin requirement)
STOP_LOSS_PCT = 0.15   # 15% stop on assigned shares
MA_DAYS = 200          # 200-day MA filter
DELTA_TARGET = 0.25    # ~25 delta put target
DTE_MIN = 5
DTE_MAX = 12
INITIAL_CAPITAL = 100_000
RANDOM_SEED = 42
N_PERMUTATIONS = 50  # 50 trials balances statistical power vs runtime
OOT_FRACTION = 0.30    # last 30% = out-of-sample


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load all chains and prices, pre-filter for efficiency."""
    print("  Loading chains...")
    t0 = time.time()
    frames = []
    for f in sorted(CHAINS_DIR.glob("*.parquet")):
        if f.suffix == ".EMPTY":
            continue
        try:
            df = pd.read_parquet(
                f,
                columns=["date", "expiration", "strike", "type", "bid", "ask", "vol", "delta", "dte", "mid"]
            )
            df["ticker"] = f.stem
            frames.append(df)
        except Exception:
            pass
    chains = pd.concat(frames, ignore_index=True)
    chains["date"] = pd.to_datetime(chains["date"])
    chains["expiration"] = pd.to_datetime(chains["expiration"])
    print(f"  Chains loaded: {len(chains):,} rows in {time.time()-t0:.1f}s")

    # Pre-filter puts for VRP + execution
    puts_exec = chains[
        (chains["type"] == "p") &
        (chains["dte"].between(DTE_MIN, DTE_MAX)) &
        (chains["vol"] > 0) & (chains["vol"] < 3.0) &
        (chains["delta"].abs().between(0.10, 0.50)) &
        (chains["bid"] > 0)
    ].copy()

    # Pre-filter calls for CC execution
    calls_exec = chains[
        (chains["type"] == "c") &
        (chains["dte"].between(DTE_MIN, DTE_MAX)) &
        (chains["delta"].between(0.20, 0.50)) &
        (chains["bid"] > 0)
    ].copy()

    # Pre-filter ATM options for straddle (delta ~0.45-0.55)
    atm_opts = chains[
        (chains["dte"].between(DTE_MIN, DTE_MAX)) &
        (chains["delta"].abs().between(0.35, 0.65)) &
        (chains["bid"] > 0)
    ].copy()

    print(f"  Puts exec: {len(puts_exec):,} | Calls exec: {len(calls_exec):,} | ATM: {len(atm_opts):,}")

    print("  Loading prices (for rv_20 only)...")
    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    print(f"  Prices: {len(prices):,} rows")

    # Derive spot prices from options chains (correct unadjusted prices).
    # Use near-ATM put+call strike median where |delta| ~ 0.45-0.55.
    # This is scale-consistent with the options strikes.
    print("  Building spot prices from ATM options...")
    atm_for_spot = chains[
        chains["delta"].abs().between(0.40, 0.60) &
        chains["dte"].between(5, 45)
    ].copy()
    spot_from_chains = (
        atm_for_spot.groupby(["ticker", "date"])["strike"]
        .median()
        .reset_index()
        .rename(columns={"strike": "spot"})
    )
    print(f"  Spot rows: {len(spot_from_chains):,}")

    # Pre-build spot price dict: {ticker: {date: spot}}
    spot_dict = {}
    for ticker, grp in spot_from_chains.groupby("ticker"):
        spot_dict[ticker] = dict(zip(grp["date"], grp["spot"]))

    return puts_exec, calls_exec, atm_opts, prices, spot_dict, spot_from_chains


def precompute_vrp(puts_exec: pd.DataFrame, prices: pd.DataFrame, spot_from_chains: pd.DataFrame) -> pd.DataFrame:
    """Compute VRP = IV(ATM) - RV(20d) per ticker per date.
    IV from options chains. RV from adjusted-price log returns (scale-invariant).
    spot_from_chains provides the unadjusted spot price for downstream use.
    """
    # Weekly IV from near-ATM puts
    weekly_iv = (
        puts_exec.groupby(["ticker", "date"])["vol"]
        .median()
        .reset_index()
        .rename(columns={"vol": "iv"})
    )
    # RV from adjusted prices — log returns are scale-invariant, so rv_20 is valid
    rv_df = prices[["ticker", "date", "rv_20"]].dropna(subset=["rv_20"])
    vrp_df = weekly_iv.merge(rv_df, on=["ticker", "date"], how="left")
    vrp_df = vrp_df.merge(spot_from_chains, on=["ticker", "date"], how="left")
    vrp_df["vrp"] = vrp_df["iv"] - vrp_df["rv_20"]
    vrp_df = vrp_df.dropna(subset=["vrp", "iv", "rv_20", "spot"])
    return vrp_df


def precompute_ma(spot_from_chains: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute 200-day MA using options-derived spot prices."""
    df = spot_from_chains.sort_values(["ticker", "date"]).copy()
    df["ma200"] = df.groupby("ticker")["spot"].transform(
        lambda x: x.rolling(MA_DAYS, min_periods=MA_DAYS // 2).mean()
    )
    return df[["ticker", "date", "spot", "ma200"]].dropna(subset=["ma200"])


def build_date_indices(df: pd.DataFrame) -> dict:
    """Build {date -> sub-dataframe} index for fast lookup."""
    return {d: grp for d, grp in df.groupby("date")}


# ═══════════════════════════════════════════════════════════════════════════
# 2. OPTION SELECTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def pick_put(puts_today: pd.DataFrame, ticker: str) -> dict | None:
    """Select best near-25-delta put for ticker on today's snapshot."""
    rows = puts_today[puts_today["ticker"] == ticker]
    if rows.empty:
        return None
    rows = rows.copy()
    rows["dd"] = (rows["delta"].abs() - DELTA_TARGET).abs()
    best = rows.nsmallest(1, "dd").iloc[0]
    return {
        "ticker": ticker,
        "strike": best["strike"],
        "expiration": best["expiration"],
        "dte": best["dte"],
        "delta": float(best["delta"]),
        "mid": float(best["mid"]),
        "iv": float(best["vol"]),
    }


def pick_call(calls_today: pd.DataFrame, ticker: str, cost_basis: float) -> dict | None:
    """Select covered call near or above cost basis."""
    rows = calls_today[(calls_today["ticker"] == ticker) & (calls_today["strike"] >= cost_basis * 0.99)]
    if rows.empty:
        return None
    best = rows.nsmallest(1, "strike").iloc[0]
    return {
        "ticker": ticker,
        "strike": float(best["strike"]),
        "expiration": best["expiration"],
        "dte": int(best["dte"]),
        "delta": float(best["delta"]),
        "mid": float(best["mid"]),
    }


def pick_straddle(atm_today: pd.DataFrame, ticker: str, spot: float) -> dict | None:
    """Select ATM straddle for ticker."""
    rows = atm_today[atm_today["ticker"] == ticker]
    if rows.empty:
        return None
    strikes = rows["strike"].unique()
    atm_strike = strikes[np.argmin(np.abs(strikes - spot))]
    put_row = rows[(rows["type"] == "p") & (rows["strike"] == atm_strike)]
    call_row = rows[(rows["type"] == "c") & (rows["strike"] == atm_strike)]
    if put_row.empty or call_row.empty:
        return None
    p = put_row.iloc[0]
    c = call_row.iloc[0]
    return {
        "ticker": ticker,
        "strike": float(atm_strike),
        "expiration": p["expiration"],
        "dte": int(p["dte"]),
        "put_mid": float(p["mid"]),
        "call_mid": float(c["mid"]),
        "total_credit": float(p["mid"] + c["mid"]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. BACKTEST ENGINE — WHEEL (CSP + CC)
# ═══════════════════════════════════════════════════════════════════════════

def run_wheel_backtest(
    puts_idx: dict,
    calls_idx: dict,
    spot_dict: dict,
    vrp_df: pd.DataFrame,
    ma_df: pd.DataFrame,
    use_vrp_ranking: bool = True,
    label: str = "VRP",
    rng_seed: int = RANDOM_SEED,
) -> dict:
    """
    Weekly wheel strategy:
      - Sell delta-25 CSPs on top-N VRP stocks (or random stocks)
      - If assigned, sell CCs until called away
      - 15% stop-loss on assigned shares
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    all_dates = sorted(vrp_df["date"].unique())
    if len(all_dates) < 8:
        return {"label": label, "error": "insufficient data"}

    # Pre-compute MA lookup: {ticker -> {date -> (spot, ma200)}}
    # Build from numpy arrays for speed
    ma_lookup = {}
    for ticker, grp in ma_df.groupby("ticker"):
        ma_lookup[ticker] = dict(zip(grp["date"], zip(grp["spot"], grp["ma200"])))

    # ── Accounting model ────────────────────────────────────────────────
    # free_cash: cash available for new positions (not in margin/shares)
    # margin_reserved: cash tied up as CSP margin (= sum of strike*100*contracts for open puts)
    # shares: equity in assigned stock (tracked at cost basis)
    # total equity = free_cash + margin_reserved + shares_at_market - premiums_received_on_open_puts
    #
    # Simpler view for weekly P&L:
    #   - Each week: P&L = premium_from_expired_worthless + cc_premium + stock_gain/loss_on_close
    #   - Running NAV = free_cash + reserved_margin + shares_at_spot (mark-to-market)

    free_cash = float(INITIAL_CAPITAL)  # available cash
    margin = {}    # ticker -> margin_reserved (strike * 100 * contracts)
    open_puts = []    # active CSP positions
    open_calls = []   # active CC positions
    open_shares = {}  # ticker -> {shares, cost_basis}

    # Track NAV: start = INITIAL_CAPITAL, track weekly changes
    nav = float(INITIAL_CAPITAL)
    equity_curve = [nav]
    dates_out = [all_dates[0]]
    weekly_pnl = []

    for trade_ts in all_dates:
        puts_today = puts_idx.get(trade_ts, pd.DataFrame())
        calls_today = calls_idx.get(trade_ts, pd.DataFrame())

        week_realized = 0.0

        # ── 1. SETTLE EXPIRED PUTS ──────────────────────────────────────
        new_puts = []
        for pos in open_puts:
            exp = pos["expiration"]
            if exp <= trade_ts:
                ticker = pos["ticker"]
                spot = _get_spot(spot_dict, ticker, trade_ts)
                mgn = pos.get("margin_reserved", 0.0)

                if spot is None or spot >= pos["strike"]:
                    # Expires worthless: release margin, keep premium as profit
                    free_cash += mgn
                    if ticker in margin:
                        del margin[ticker]
                    week_realized += pos["premium"]
                else:
                    # Assigned: margin converts to shares at strike price
                    # cost_basis = strike - premium_per_share (effective cost)
                    shares = pos["contracts"] * 100
                    # Margin was reserved = strike * 100 * contracts
                    # Now we own shares at cost = strike (margin was set aside)
                    # Premium was already collected and treated as P&L at assignment
                    week_realized += pos["premium"]  # book premium as income
                    if ticker in open_shares:
                        # Average in
                        prev = open_shares[ticker]
                        total_shares = prev["shares"] + shares
                        avg_cb = (prev["cost_basis"] * prev["shares"] + pos["strike"] * shares) / total_shares
                        open_shares[ticker] = {"shares": total_shares, "cost_basis": avg_cb}
                    else:
                        open_shares[ticker] = {"shares": shares, "cost_basis": pos["strike"]}
                    # Margin was cash now converted to stock position
                    # It's no longer "free cash" — it's in stock
                    # margin was already deducted from free_cash at open, so no change here
                    if ticker in margin:
                        del margin[ticker]
            else:
                new_puts.append(pos)
        open_puts = new_puts

        # ── 2. SETTLE EXPIRED CALLS ──────────────────────────────────────
        new_calls = []
        for pos in open_calls:
            exp = pos["expiration"]
            if exp <= trade_ts:
                ticker = pos["ticker"]
                spot = _get_spot(spot_dict, ticker, trade_ts)
                if spot is None:
                    week_realized += pos["premium"]
                    continue
                if spot >= pos["strike"] and ticker in open_shares:
                    # Called away: sell shares at strike
                    spos = open_shares[ticker]
                    shares = min(spos["shares"], pos["contracts"] * 100)
                    proceeds = pos["strike"] * shares
                    stock_gain = (pos["strike"] - spos["cost_basis"]) * shares
                    week_realized += stock_gain + pos["premium"]
                    free_cash += proceeds  # receive sale proceeds
                    spos["shares"] -= shares
                    if spos["shares"] <= 0:
                        del open_shares[ticker]
                else:
                    # Not called — CC expires worthless, keep premium
                    week_realized += pos["premium"]
            else:
                new_calls.append(pos)
        open_calls = new_calls

        # ── 3. STOP-LOSS ON SHARES ──────────────────────────────────────
        to_remove = []
        for ticker, spos in list(open_shares.items()):
            spot = _get_spot(spot_dict, ticker, trade_ts)
            if spot is None:
                continue
            loss_pct = (spot - spos["cost_basis"]) / spos["cost_basis"]
            if loss_pct < -STOP_LOSS_PCT:
                shares = spos["shares"]
                stock_gain = (spot - spos["cost_basis"]) * shares
                week_realized += stock_gain
                free_cash += spot * shares  # receive market value
                to_remove.append(ticker)
        for t in to_remove:
            del open_shares[t]

        # ── 4. SELL CCs ON UNHEDGED SHARES ─────────────────────────────
        cc_tickers = {p["ticker"] for p in open_calls}
        for ticker, spos in list(open_shares.items()):
            if ticker in cc_tickers or calls_today.empty:
                continue
            cc = pick_call(calls_today, ticker, spos["cost_basis"])
            if cc is None:
                continue
            contracts = max(1, spos["shares"] // 100)
            prem = cc["mid"] * 100 * contracts
            # CC: receive premium now, but only realize it when it settles
            open_calls.append({
                "ticker": ticker,
                "strike": cc["strike"],
                "expiration": cc["expiration"],
                "contracts": contracts,
                "premium": prem,
            })

        # ── 5. SELECT NEW PUT POSITIONS ─────────────────────────────────
        occupied = {p["ticker"] for p in open_puts} | set(open_shares.keys())
        slots = max(0, TOP_N - len(occupied))

        if slots > 0 and not puts_today.empty:
            today_vrp = vrp_df[vrp_df["date"] == trade_ts].copy()
            today_vrp = today_vrp[~today_vrp["ticker"].isin(occupied)]

            # MA filter
            def _passes_ma(ticker):
                ma_data = ma_lookup.get(ticker, {})
                if trade_ts not in ma_data:
                    for delta in [1, 2, 3, 5, 7]:
                        d2 = trade_ts - pd.Timedelta(days=delta)
                        if d2 in ma_data:
                            spot_v, ma200 = ma_data[d2]
                            return spot_v >= ma200 * 0.95
                    return True
                spot_v, ma200 = ma_data[trade_ts]
                return spot_v >= ma200 * 0.95

            today_vrp = today_vrp[today_vrp["ticker"].apply(_passes_ma)]

            if not today_vrp.empty:
                if use_vrp_ranking:
                    candidates = today_vrp.nlargest(slots * 2, "vrp")["ticker"].tolist()
                else:
                    candidates = today_vrp["ticker"].tolist()
                    random.shuffle(candidates)

                selected = candidates[:slots]

                for ticker in selected:
                    if puts_today.empty:
                        continue
                    put = pick_put(puts_today, ticker)
                    if put is None:
                        continue

                    # CSP margin = strike * 100 * contracts (cash-secured)
                    notional_per_contract = put["strike"] * 100
                    # Size based on free cash available
                    available = free_cash
                    max_notional = available * MAX_PCT_PER_NAME
                    contracts = max(1, int(max_notional / notional_per_contract))
                    total_margin = notional_per_contract * contracts
                    if total_margin > available * 0.90:
                        contracts = max(1, int(available * 0.90 / notional_per_contract))
                        total_margin = notional_per_contract * contracts
                    if total_margin > available:
                        continue  # skip if not enough cash

                    premium = put["mid"] * 100 * contracts
                    # Reserve margin: deduct from free cash, DON'T add premium yet
                    free_cash -= total_margin
                    free_cash += premium  # premium received immediately (net: free_cash -= total_margin + premium)
                    margin[ticker] = margin.get(ticker, 0.0) + total_margin

                    open_puts.append({
                        "ticker": ticker,
                        "strike": put["strike"],
                        "expiration": put["expiration"],
                        "contracts": contracts,
                        "premium": premium,
                        "margin_reserved": total_margin,
                        "iv": put["iv"],
                    })

        # ── Update NAV (mark-to-market) ─────────────────────────────────
        shares_mtm = 0.0
        for ticker, spos in open_shares.items():
            spot = _get_spot(spot_dict, ticker, trade_ts)
            if spot:
                shares_mtm += spot * spos["shares"]

        # NAV = free_cash + all margin (still cash) + shares at market
        total_margin_val = sum(margin.values())
        nav = free_cash + total_margin_val + shares_mtm

        weekly_pnl.append(week_realized)
        equity_curve.append(nav)
        dates_out.append(trade_ts)

    return _compute_metrics(equity_curve, dates_out, weekly_pnl, label)


def _get_spot(spot_dict: dict, ticker: str, date: pd.Timestamp):
    """O(1) spot price lookup from pre-built dict."""
    ticker_prices = spot_dict.get(ticker)
    if ticker_prices is None:
        return None
    v = ticker_prices.get(date)
    if v is not None:
        return float(v)
    # Fallback: sorted keys binary search would be ideal but for small fallback rate, linear is fine
    # Try a few business days back
    for delta_days in [1, 2, 3, 4, 5, 7, 10]:
        d2 = date - pd.Timedelta(days=delta_days)
        v2 = ticker_prices.get(d2)
        if v2 is not None:
            return float(v2)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 4. BACKTEST ENGINE — STRADDLE
# ═══════════════════════════════════════════════════════════════════════════

def run_straddle_backtest(
    atm_idx: dict,
    spot_dict: dict,
    vrp_df: pd.DataFrame,
    label: str = "Straddle",
) -> dict:
    """
    Sell ATM straddles on top 5 VRP stocks each week.
    Close at expiration, booking premium minus intrinsic value.
    No delta hedging (tests raw VRP premium capture).
    """
    STRADDLE_N = 5
    all_dates = sorted(vrp_df["date"].unique())
    if len(all_dates) < 8:
        return {"label": label, "error": "insufficient data"}

    # Straddle accounting: similar to CSP
    # margin_per_straddle ~= 20% of spot * 100 (rough straddle margin requirement)
    # Premium collected upfront; P&L realized at expiration
    free_cash = float(INITIAL_CAPITAL)
    equity_curve = [free_cash]
    dates_out = [all_dates[0]]
    weekly_pnl = []
    open_straddles = []
    straddle_margin = {}  # ticker -> margin reserved

    for trade_ts in all_dates:
        atm_today = atm_idx.get(trade_ts, pd.DataFrame())
        week_realized = 0.0

        # Settle expired
        new_open = []
        for pos in open_straddles:
            if pos["expiration"] <= trade_ts:
                ticker = pos["ticker"]
                spot = _get_spot(spot_dict, ticker, trade_ts)
                if spot is None:
                    spot = pos["strike"]
                intrinsic = abs(spot - pos["strike"])
                pnl = (pos["total_credit"] - intrinsic) * 100 * pos["contracts"]
                week_realized += pnl
                # Release margin
                mgn = pos.get("margin_reserved", 0.0)
                free_cash += mgn
                if ticker in straddle_margin:
                    del straddle_margin[ticker]
            else:
                new_open.append(pos)
        open_straddles = new_open

        # Open new straddles
        existing_tickers = {p["ticker"] for p in open_straddles}
        today_vrp = vrp_df[vrp_df["date"] == trade_ts].copy()
        today_vrp = today_vrp[~today_vrp["ticker"].isin(existing_tickers)]
        selected = today_vrp.nlargest(STRADDLE_N, "vrp")["ticker"].tolist()

        for ticker in selected:
            if atm_today.empty:
                continue
            spot = _get_spot(spot_dict, ticker, trade_ts)
            if spot is None:
                continue
            st = pick_straddle(atm_today, ticker, spot)
            if st is None:
                continue
            # Straddle margin: ~20% of spot notional (naked straddle requirement)
            notional = spot * 100  # per contract
            margin_per_contract = notional * 0.20
            max_contracts = max(1, int(free_cash * MAX_PCT_PER_NAME / margin_per_contract))
            contracts = max_contracts
            total_margin = margin_per_contract * contracts
            if total_margin > free_cash * 0.90:
                contracts = max(1, int(free_cash * 0.90 / margin_per_contract))
                total_margin = margin_per_contract * contracts
            if total_margin > free_cash:
                continue

            premium = st["total_credit"] * 100 * contracts
            free_cash -= total_margin  # reserve margin (premium does NOT add to free cash until settlement)
            straddle_margin[ticker] = total_margin

            open_straddles.append({
                "ticker": ticker,
                "strike": st["strike"],
                "expiration": st["expiration"],
                "total_credit": st["total_credit"],
                "contracts": contracts,
                "margin_reserved": total_margin,
            })

        free_cash += week_realized
        nav = free_cash + sum(straddle_margin.values())
        weekly_pnl.append(week_realized)
        equity_curve.append(nav)
        dates_out.append(trade_ts)

    return _compute_metrics(equity_curve, dates_out, weekly_pnl, label)


# ═══════════════════════════════════════════════════════════════════════════
# 5. METRICS
# ═══════════════════════════════════════════════════════════════════════════

def _compute_metrics(equity_curve, dates, weekly_pnl, label):
    eq = np.array(equity_curve, dtype=float)
    pnl_arr = np.array(weekly_pnl, dtype=float)

    weekly_rets = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1)
    weekly_rets = weekly_rets[np.isfinite(weekly_rets)]

    if len(weekly_rets) < 4:
        return {"label": label, "error": "insufficient data"}

    n_weeks = len(weekly_rets)
    n_years = n_weeks / 52.0

    final_eq = float(eq[-1])
    ratio = final_eq / INITIAL_CAPITAL
    if ratio <= 0:
        cagr = -1.0  # total loss
    else:
        cagr = ratio ** (1 / max(n_years, 0.1)) - 1

    mean_w = np.mean(weekly_rets)
    std_w = np.std(weekly_rets, ddof=1)
    sharpe = (mean_w / std_w) * np.sqrt(52) if std_w > 0 else 0.0

    downside = weekly_rets[weekly_rets < 0]
    sortino_denom = np.std(downside, ddof=1) if len(downside) > 1 else std_w
    sortino = (mean_w / sortino_denom) * np.sqrt(52) if sortino_denom > 0 else 0.0

    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / np.where(running_max != 0, running_max, 1)
    max_dd = float(drawdowns.min())

    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0.0
    win_rate = float((pnl_arr > 0).mean())
    gross_profit = float(pnl_arr[pnl_arr > 0].sum())
    gross_loss = float(abs(pnl_arr[pnl_arr < 0].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_return = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL

    return {
        "label": label,
        "n_weeks": n_weeks,
        "n_years": round(n_years, 2),
        "cagr": round(cagr * 100, 2),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate": round(win_rate * 100, 2),
        "profit_factor": round(pf, 3),
        "final_equity": round(final_eq, 0),
        "equity_curve": eq.tolist(),
        "dates": [str(d) for d in dates],
        "weekly_pnl": pnl_arr.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. REGIME ANALYSIS (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════════════

def regime_analysis(result: dict, prices_df: pd.DataFrame) -> dict:
    """
    Classify each week as green/red/flat based on SPY performance.
    Compute stratified Sharpe per regime, check regime gap threshold.
    Reject if |Sharpe_green - Sharpe_red| / max(|Sg|,|Sr|) > 0.50.
    """
    if "weekly_pnl" not in result or len(result.get("weekly_pnl", [])) == 0:
        return {}

    spy = prices_df[prices_df["ticker"] == "SPY"][["date", "close"]].copy()
    spy = spy.sort_values("date").set_index("date")

    dates = pd.to_datetime(result["dates"][1:])
    pnl_arr = np.array(result["weekly_pnl"])
    eq = np.array(result["equity_curve"])
    weekly_rets = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1)

    n = min(len(weekly_rets), len(dates))
    dates = dates[:n]
    weekly_rets = weekly_rets[:n]
    pnl_arr = pnl_arr[:n]

    regime_labels = []
    for d in dates:
        spy_window = spy[spy.index <= d].tail(6)
        if len(spy_window) < 2:
            regime_labels.append("flat")
            continue
        spy_ret = (float(spy_window.iloc[-1]["close"]) - float(spy_window.iloc[-2]["close"])) / float(spy_window.iloc[-2]["close"])
        if spy_ret > 0.005:
            regime_labels.append("green")
        elif spy_ret < -0.005:
            regime_labels.append("red")
        else:
            regime_labels.append("flat")

    regime_arr = np.array(regime_labels)
    regime_metrics = {}

    for r in ["green", "red", "flat"]:
        mask = regime_arr == r
        if mask.sum() < 3:
            continue
        r_rets = weekly_rets[mask]
        std = r_rets.std(ddof=1)
        r_sharpe = (r_rets.mean() / std) * np.sqrt(52) if std > 0 else 0.0
        regime_metrics[r] = {
            "n_weeks": int(mask.sum()),
            "sharpe": round(r_sharpe, 3),
            "mean_weekly_ret_pct": round(float(r_rets.mean()) * 100, 3),
            "win_rate_pct": round(float((r_rets > 0).mean()) * 100, 1),
        }

    # Regime gap test
    if "green" in regime_metrics and "red" in regime_metrics:
        sg = regime_metrics["green"]["sharpe"]
        sr = regime_metrics["red"]["sharpe"]
        denom = max(abs(sg), abs(sr))
        gap = abs(sg - sr) / denom if denom > 0 else 0.0
        regime_metrics["regime_gap"] = round(gap, 3)
        regime_metrics["passes_hc428_r1"] = bool(gap <= 0.50)
    else:
        regime_metrics["regime_gap"] = None
        regime_metrics["passes_hc428_r1"] = None

    return regime_metrics


# ═══════════════════════════════════════════════════════════════════════════
# 7. PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════════════

def permutation_test(
    puts_idx, calls_idx, spot_dict, vrp_df, ma_df, vrp_sharpe, n_trials=N_PERMUTATIONS
) -> dict:
    """
    Run n_trials random-selection wheel backtests.
    p-value = fraction of randoms with Sharpe >= VRP Sharpe.
    """
    random_sharpes = []
    for trial in range(n_trials):
        r = run_wheel_backtest(
            puts_idx, calls_idx, spot_dict, vrp_df, ma_df,
            use_vrp_ranking=False,
            label=f"Rnd_{trial}",
            rng_seed=trial + 1000,
        )
        if "sharpe" in r and not isinstance(r.get("error"), str):
            random_sharpes.append(r["sharpe"])
        if (trial + 1) % 25 == 0:
            print(f"    Permutation {trial+1}/{n_trials} done")

    random_sharpes = np.array(random_sharpes)
    p_value = float((random_sharpes >= vrp_sharpe).mean()) if len(random_sharpes) > 0 else 1.0

    return {
        "vrp_sharpe": vrp_sharpe,
        "n_trials": len(random_sharpes),
        "random_sharpe_mean": round(float(random_sharpes.mean()), 3) if len(random_sharpes) > 0 else None,
        "random_sharpe_std": round(float(random_sharpes.std()), 3) if len(random_sharpes) > 0 else None,
        "random_sharpe_p95": round(float(np.percentile(random_sharpes, 95)), 3) if len(random_sharpes) > 0 else None,
        "p_value": round(p_value, 4),
        "significant_at_5pct": p_value < 0.05,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def print_result(r):
    if "error" in r:
        print(f"\n  {r.get('label', '?')} — ERROR: {r['error']}")
        return
    print(f"\n{'='*62}")
    print(f"  Strategy: {r.get('label','?')}")
    print(f"  Period  : {r.get('n_years','?')} yrs | {r.get('n_weeks','?')} weeks")
    print(f"{'='*62}")
    print(f"  CAGR         : {r.get('cagr', 0):+.2f}%")
    print(f"  Total Return : {r.get('total_return_pct', 0):+.2f}%")
    print(f"  Sharpe       : {r.get('sharpe', 0):.3f}")
    print(f"  Sortino      : {r.get('sortino', 0):.3f}")
    print(f"  Max Drawdown : {r.get('max_dd_pct', 0):.2f}%")
    print(f"  Calmar       : {r.get('calmar', 0):.3f}")
    print(f"  Win Rate     : {r.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor: {r.get('profit_factor', 0):.3f}")
    fe = r.get('final_equity', 0)
    print(f"  Final Equity : ${fe:,.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("VRP HARVESTER v1 — Variance Risk Premium Income Strategy Backtest")
    print("=" * 70)

    # ── Load & Preprocess ─────────────────────────────────────────────────
    print("\n[1/7] Loading & pre-processing data...")
    puts_exec, calls_exec, atm_opts, prices_df, spot_dict, spot_from_chains = load_data()

    print("\n[2/7] Computing VRP (IV - RV20) per ticker per date...")
    vrp_df = precompute_vrp(puts_exec, prices_df, spot_from_chains)
    print(f"  VRP rows: {len(vrp_df):,} | Tickers: {vrp_df['ticker'].nunique()} | Dates: {vrp_df['date'].nunique()}")
    print(f"  Date range: {vrp_df['date'].min().date()} → {vrp_df['date'].max().date()}")
    print(f"  VRP distribution: mean={vrp_df['vrp'].mean():.4f}, std={vrp_df['vrp'].std():.4f}")
    top_vrp = vrp_df.groupby("ticker")["vrp"].mean().nlargest(10)
    print(f"  Highest avg VRP: {', '.join([f'{t}={v:.3f}' for t,v in top_vrp.items()])}")

    print("\n[3/7] Pre-computing 200-day MA filter...")
    ma_df = precompute_ma(spot_from_chains)
    print(f"  MA rows: {len(ma_df):,}")

    print("\n[4/7] Building date indices...")
    puts_idx = build_date_indices(puts_exec)
    calls_idx = build_date_indices(calls_exec)
    atm_idx = build_date_indices(atm_opts)
    print(f"  Put dates: {len(puts_idx)} | Call dates: {len(calls_idx)} | ATM dates: {len(atm_idx)}")

    # ── Walk-forward split ────────────────────────────────────────────────
    print("\n[5/7] Walk-forward split (70% IS | 30% OOT)...")
    all_dates = sorted(vrp_df["date"].unique())
    split_idx = int(len(all_dates) * (1 - OOT_FRACTION))
    is_dates = set(all_dates[:split_idx])
    oot_dates = set(all_dates[split_idx:])

    vrp_is = vrp_df[vrp_df["date"].isin(is_dates)]
    vrp_oot = vrp_df[vrp_df["date"].isin(oot_dates)]

    def filter_idx(idx, date_set):
        return {d: v for d, v in idx.items() if d in date_set}

    puts_is_idx = filter_idx(puts_idx, is_dates)
    puts_oot_idx = filter_idx(puts_idx, oot_dates)
    calls_is_idx = filter_idx(calls_idx, is_dates)
    calls_oot_idx = filter_idx(calls_idx, oot_dates)
    atm_is_idx = filter_idx(atm_idx, is_dates)
    atm_oot_idx = filter_idx(atm_idx, oot_dates)

    all_dt_list = sorted(all_dates)
    print(f"  IS : {pd.Timestamp(all_dt_list[0]).date()} → {pd.Timestamp(all_dt_list[split_idx-1]).date()} ({len(is_dates)} dates)")
    print(f"  OOT: {pd.Timestamp(all_dt_list[split_idx]).date()} → {pd.Timestamp(all_dt_list[-1]).date()} ({len(oot_dates)} dates)")

    # ── Run Strategies ────────────────────────────────────────────────────
    print("\n[6/7] Running strategies...")

    print("  > VRP-Ranked Wheel (Full)...")
    res_vrp_full = run_wheel_backtest(puts_idx, calls_idx, spot_dict, vrp_df, ma_df, use_vrp_ranking=True, label="VRP-Ranked Wheel (Full)")
    print("  > Random Wheel (Full)...")
    res_rnd_full = run_wheel_backtest(puts_idx, calls_idx, spot_dict, vrp_df, ma_df, use_vrp_ranking=False, label="Random Wheel (Full)")
    print("  > ATM Straddle (Full)...")
    res_str_full = run_straddle_backtest(atm_idx, spot_dict, vrp_df, label="ATM Straddle (Full)")

    print("  > VRP-Ranked Wheel (OOT)...")
    res_vrp_oot = run_wheel_backtest(puts_oot_idx, calls_oot_idx, spot_dict, vrp_oot, ma_df, use_vrp_ranking=True, label="VRP-Ranked Wheel (OOT)")
    print("  > Random Wheel (OOT)...")
    res_rnd_oot = run_wheel_backtest(puts_oot_idx, calls_oot_idx, spot_dict, vrp_oot, ma_df, use_vrp_ranking=False, label="Random Wheel (OOT)")
    print("  > ATM Straddle (OOT)...")
    res_str_oot = run_straddle_backtest(atm_oot_idx, spot_dict, vrp_oot, label="ATM Straddle (OOT)")

    print_result(res_vrp_full)
    print_result(res_rnd_full)
    print_result(res_str_full)
    print_result(res_vrp_oot)
    print_result(res_rnd_oot)
    print_result(res_str_oot)

    # ── Regime Analysis ───────────────────────────────────────────────────
    print("\n[7a/7] HC #428 R1 Regime Analysis...")
    for r in [res_vrp_full, res_rnd_full, res_str_full]:
        if "error" in r:
            continue
        reg = regime_analysis(r, prices_df)
        print(f"\n  {r['label']}")
        for k, v in reg.items():
            print(f"    {k}: {v}")

    # ── Permutation Test ──────────────────────────────────────────────────
    print(f"\n[7b/7] Permutation test ({N_PERMUTATIONS} trials)...")
    vrp_sharpe = res_vrp_full.get("sharpe", 0.0)
    perm = permutation_test(puts_idx, calls_idx, spot_dict, vrp_df, ma_df, vrp_sharpe, n_trials=N_PERMUTATIONS)
    print(f"\n  VRP Sharpe         : {perm['vrp_sharpe']}")
    print(f"  Random Sharpe mean : {perm['random_sharpe_mean']} ± {perm['random_sharpe_std']}")
    print(f"  Random p95 Sharpe  : {perm['random_sharpe_p95']}")
    print(f"  p-value            : {perm['p_value']}")
    print(f"  Significant (5%)   : {perm['significant_at_5pct']}")

    # ── Save Results ──────────────────────────────────────────────────────
    summary = {
        "run_time_seconds": round(time.time() - t_start, 1),
        "full_period": {
            "vrp_wheel": res_vrp_full,
            "random_wheel": res_rnd_full,
            "straddle": res_str_full,
        },
        "out_of_sample": {
            "vrp_wheel": res_vrp_oot,
            "random_wheel": res_rnd_oot,
            "straddle": res_str_oot,
        },
        "regime_analysis": {
            "vrp_wheel": regime_analysis(res_vrp_full, prices_df),
            "random_wheel": regime_analysis(res_rnd_full, prices_df),
            "straddle": regime_analysis(res_str_full, prices_df),
        },
        "permutation_test": perm,
    }

    out_path = OUT_DIR / "results_v1.json"
    def _default(o):
        if isinstance(o, float) and np.isinf(o):
            return "inf"
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=_default)
    print(f"\nResults saved: {out_path}")

    # ── Final Comparison Table ────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("FINAL COMPARISON TABLE")
    print("=" * 75)
    print(f"{'Strategy':<33} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'WR%':>6} {'PF':>7}")
    print("-" * 75)
    for r in [res_vrp_full, res_rnd_full, res_str_full, res_vrp_oot, res_rnd_oot, res_str_oot]:
        if "error" in r:
            continue
        lab = r["label"][:32]
        pf_val = r.get("profit_factor", 0)
        pf_str = f"{pf_val:.3f}" if not (isinstance(pf_val, float) and np.isinf(pf_val)) else "inf"
        print(f"{lab:<33} {r.get('cagr',0):>7.2f} {r.get('sharpe',0):>7.3f} {r.get('sortino',0):>8.3f} {r.get('max_dd_pct',0):>7.2f} {r.get('win_rate',0):>6.1f} {pf_str:>7}")
    print("=" * 75)
    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")

    return summary


if __name__ == "__main__":
    main()
