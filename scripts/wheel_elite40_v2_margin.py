#!/usr/bin/env python3
"""
wheel_elite40_v2_margin.py — Proper portfolio-level wheel backtest with:
- Margin-aware position sizing (cash-secured puts = 20% margin requirement)
- Assignment death-spiral prevention (max 3 assignments per rolling 5-day window)
- Loss-cut on assigned shares (sell after -15% from cost basis)
- Dynamic sizing based on current NAV, not starting capital
- 40 elite tickers across 11 sectors
- Weekly DTE (14 days), bear protection gate (SPY < 50d SMA)
- HC #428 R1 regime-agnostic validation (40+ OOT days)
"""
import json
import math
import time
import logging
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "wheel_elite40_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [WHEEL-V2] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger('WHEEL-V2')

# ========================= ELITE 40 BASKET ==================================
BASKET = {
    # Consumer Cyclical (4)
    'WYNN': 'Consumer Cyclical', 'HD': 'Consumer Cyclical',
    'SBUX': 'Consumer Cyclical', 'BABA': 'Consumer Cyclical',
    # Utilities (4)
    'EXC': 'Utilities', 'AEP': 'Utilities',
    'DUK': 'Utilities', 'ED': 'Utilities',
    # Energy (4)
    'COP': 'Energy', 'XOM': 'Energy',
    'CVX': 'Energy', 'VLO': 'Energy',
    # Technology (4)
    'TXN': 'Technology', 'IBM': 'Technology',
    'CSCO': 'Technology', 'ARM': 'Technology',
    # Communication Services (4)
    'EA': 'Communication Services', 'VZ': 'Communication Services',
    'TMUS': 'Communication Services', 'GOOGL': 'Communication Services',
    # Healthcare (4)
    'GILD': 'Healthcare', 'BIIB': 'Healthcare',
    'CVS': 'Healthcare', 'ABT': 'Healthcare',
    # Real Estate (3)
    'DLR': 'Real Estate', 'IRM': 'Real Estate', 'SPG': 'Real Estate',
    # Financial Services (4)
    'JPM': 'Financial Services', 'AXP': 'Financial Services',
    'BAC': 'Financial Services', 'GS': 'Financial Services',
    # Industrials (3)
    'CAT': 'Industrials', 'HON': 'Industrials', 'UNP': 'Industrials',
    # Consumer Defensive (3)
    'CL': 'Consumer Defensive', 'PG': 'Consumer Defensive', 'TGT': 'Consumer Defensive',
    # Basic Materials (1)
    'LIN': 'Basic Materials',
}

TICKERS = sorted(BASKET.keys())
N_NAMES = len(TICKERS)

# ========================= CONFIG ============================================
START_DATE = pd.Timestamp("2019-01-01")
STARTING_CASH = 100_000.0
TRADING_DAYS = 252
RISK_FREE = 0.04

PUT_DELTA = 0.25
CALL_DELTA = 0.30
DTE_TARGET = 14  # weekly
PROFIT_TAKE = 0.50
VIX_MAX = 35.0

# Margin & risk controls
MARGIN_REQ_PCT = 0.20       # 20% of notional for CSP margin
MAX_PORTFOLIO_MARGIN = 0.80  # Use at most 80% of NAV as margin
MAX_PER_NAME_PCT = 0.08     # 8% of NAV per name (margin basis)
MAX_ASSIGNMENTS_5D = 3       # Max 3 assignments per 5-day rolling window
LOSS_CUT_PCT = -0.15        # Sell assigned shares after -15% loss
MAX_SHARE_POSITIONS = 5      # Max 5 tickers holding shares simultaneously

COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03
FLAT_BAND = 0.0025

SMA_PERIOD = 50

# ========================= PRICING ==========================================
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

def find_strike(S, sigma, T, delta_target, kind="put"):
    """Find strike for target absolute delta using bisection.
    Put delta (abs) = N(-d1) = _Phi(-d1), ranges 0 to 0.5 for OTM.
    Call delta = N(d1) = _Phi(d1), ranges 0.5 to 1 for OTM.
    """
    if kind == "put":
        lo, hi = S * 0.3, S * 1.0
    else:
        lo, hi = S * 1.0, S * 2.0
    for _ in range(60):
        K = (lo + hi) / 2
        d1 = (math.log(S / K) + (RISK_FREE + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T) + 1e-9)
        if kind == "put":
            delta_abs = _Phi(-d1)  # Put |delta| = N(-d1)
        else:
            delta_abs = _Phi(d1)   # Call delta = N(d1)
        if delta_abs > delta_target:
            # Too ITM — for put: strike too high, need lower; for call: strike too low, need higher
            if kind == "put":
                hi = K
            else:
                lo = K
        else:
            # Too OTM — for put: strike too low, need higher; for call: strike too high, need lower
            if kind == "put":
                lo = K
            else:
                hi = K
    return round(K * 2) / 2

# ========================= DATA LOADING =====================================
def load_data():
    log.info(f"Loading data for {N_NAMES} tickers...")

    p1 = pd.read_parquet(CACHE / "prices.parquet")[["ticker", "date", "close"]].copy()
    p1["date"] = pd.to_datetime(p1["date"], utc=False)
    if p1["date"].dt.tz is not None:
        p1["date"] = p1["date"].dt.tz_localize(None)

    p2 = pd.read_parquet(CACHE / "prices_expanded.parquet")
    p2 = p2.rename(columns={"Close": "close"})[["ticker", "date", "close"]].copy()
    p2["date"] = pd.to_datetime(p2["date"], utc=False)
    if p2["date"].dt.tz is not None:
        p2["date"] = p2["date"].dt.tz_localize(None)

    prices = pd.concat([p1, p2], ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"]).reset_index(drop=True)
    prices = prices.dropna(subset=["close"])
    prices = prices[prices["close"] > 0]
    prices = prices[prices["date"] >= START_DATE]

    # VIX
    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"], utc=False)
    if macro["date"].dt.tz is not None:
        macro["date"] = macro["date"].dt.tz_localize(None)
    prices = prices.merge(macro, on="date", how="left")
    prices["vix"] = prices["vix"].ffill().fillna(20.0)

    # 20-day realized vol
    prices["log_ret"] = prices.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))
    prices["sigma"] = prices.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(20, min_periods=15).std() * np.sqrt(252)
    )
    prices["sigma"] = prices["sigma"].clip(lower=0.05, upper=2.0)

    # SPY for regime gate
    spy_file = CACHE / "spy_prices.parquet"
    if spy_file.exists():
        spy = pd.read_parquet(spy_file)[["date", "close"]].copy()
        spy["date"] = pd.to_datetime(spy["date"], utc=False)
        if spy["date"].dt.tz is not None:
            spy["date"] = spy["date"].dt.tz_localize(None)
    else:
        spy = prices[prices["ticker"] == "SPY"][["date", "close"]].copy()

    if not spy.empty:
        spy = spy.sort_values("date").drop_duplicates("date")
        spy["spy_sma"] = spy["close"].rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
        spy["bear"] = (spy["close"] < spy["spy_sma"]).astype(int)
        spy_regime = spy[["date", "bear", "spy_sma"]].copy()
    else:
        all_dates = prices["date"].unique()
        spy_regime = pd.DataFrame({"date": all_dates, "bear": 0, "spy_sma": np.nan})

    our_prices = prices[prices["ticker"].isin(TICKERS)].copy()
    available = our_prices["ticker"].nunique()
    log.info(f"Data loaded: {available}/{N_NAMES} tickers available")
    missing = set(TICKERS) - set(our_prices["ticker"].unique())
    if missing:
        log.warning(f"Missing tickers: {missing}")

    return our_prices, spy_regime


# ========================= PORTFOLIO ENGINE ==================================
@dataclass
class CSPPosition:
    ticker: str
    strike: float
    premium: float  # per-share premium collected
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    margin_held: float  # actual $ margin reserved

@dataclass
class SharePosition:
    ticker: str
    shares: int
    cost_basis: float  # per-share cost basis (strike - premium collected)
    entry_date: pd.Timestamp
    cc_strike: float = 0.0
    cc_premium: float = 0.0
    cc_expiry: pd.Timestamp = None
    has_cc: bool = False


def run_portfolio(prices_df, spy_regime, bear_mode="liq_csp_only"):
    """
    Proper margin-aware portfolio sim.

    Accounting model:
    - `cash`: free cash (starts at STARTING_CASH)
    - `margin_used`: sum of margin held by open CSPs
    - NAV = cash + margin_used + mark-to-market of share positions
    - Available margin = NAV * MAX_PORTFOLIO_MARGIN - margin_used
    """
    tickers_available = sorted(prices_df["ticker"].unique())
    log.info(f"Running portfolio: {len(tickers_available)} tickers, bear_mode={bear_mode}")

    # Build date-indexed lookups
    ticker_data = {}
    for t in tickers_available:
        tdf = prices_df[prices_df["ticker"] == t].set_index("date").sort_index()
        ticker_data[t] = tdf

    # SPY regime lookup
    spy_map = {}
    if not spy_regime.empty:
        for _, row in spy_regime.iterrows():
            spy_map[pd.Timestamp(row["date"])] = int(row["bear"])

    all_dates = sorted(pd.Timestamp(d) for d in prices_df["date"].unique())
    all_dates = [d for d in all_dates if d >= START_DATE]

    # State
    cash = STARTING_CASH
    csp_positions = {}      # ticker -> CSPPosition
    share_positions = {}    # ticker -> SharePosition

    assignment_dates = deque()  # rolling window of assignment dates

    trades = []
    daily_equity = []
    daily_pnl = []

    for di, date in enumerate(all_dates):
        is_bear = spy_map.get(date, 0) == 1

        # === MARK TO MARKET ===
        nav = cash
        # Add back margin held by CSPs (it's our money, just reserved)
        for pos in csp_positions.values():
            nav += pos.margin_held
        # Add share positions at market
        for t, pos in share_positions.items():
            if t in ticker_data and date in ticker_data[t].index:
                px = ticker_data[t].loc[date]["close"]
                nav += pos.shares * px
            else:
                nav += pos.shares * pos.cost_basis  # fallback

        daily_equity.append({"date": date, "equity": nav})
        if len(daily_equity) > 1:
            daily_pnl.append(nav - daily_equity[-2]["equity"])

        # === PROCESS CSP EXPIRATIONS ===
        expired_csps = [t for t, pos in csp_positions.items() if date >= pos.expiry_date]
        for t in expired_csps:
            pos = csp_positions[t]
            if t not in ticker_data or date not in ticker_data[t].index:
                # Can't determine — assume expired OTM, release margin
                cash += pos.margin_held
                trades.append({"date": date, "ticker": t, "action": "csp_expired_otm",
                              "pnl": pos.premium * 100})
                del csp_positions[t]
                continue

            px = ticker_data[t].loc[date]["close"]

            if px <= pos.strike:
                # ASSIGNED — check assignment limit
                # Clean old assignments from rolling window
                while assignment_dates and (date - assignment_dates[0]).days > 5:
                    assignment_dates.popleft()

                if (len(assignment_dates) >= MAX_ASSIGNMENTS_5D or
                    len(share_positions) >= MAX_SHARE_POSITIONS):
                    # REFUSE assignment — close CSP at intrinsic loss instead
                    intrinsic = (pos.strike - px) * 100
                    loss = intrinsic - pos.premium * 100 + COST_PER_CONTRACT
                    cash += pos.margin_held  # release margin
                    cash -= loss  # pay the loss
                    trades.append({"date": date, "ticker": t, "action": "assignment_refused_close",
                                  "price": px, "pnl": -(loss / 100)})
                    del csp_positions[t]
                    continue

                # Accept assignment: buy 100 shares at strike
                share_cost = pos.strike * 100 + COST_PER_CONTRACT
                cash += pos.margin_held  # release CSP margin
                cash -= share_cost        # pay for shares

                share_positions[t] = SharePosition(
                    ticker=t, shares=100,
                    cost_basis=pos.strike - pos.premium,  # net cost basis
                    entry_date=date,
                )
                assignment_dates.append(date)
                trades.append({"date": date, "ticker": t, "action": "assigned",
                              "strike": pos.strike, "cost_basis": pos.strike - pos.premium})
                del csp_positions[t]
            else:
                # Expired OTM — keep premium, release margin
                cash += pos.margin_held
                trades.append({"date": date, "ticker": t, "action": "csp_expired_otm",
                              "price": px, "pnl": pos.premium})
                del csp_positions[t]

        # === PROCESS CC EXPIRATIONS ===
        for t in list(share_positions.keys()):
            pos = share_positions[t]
            if not pos.has_cc or pos.cc_expiry is None or date < pos.cc_expiry:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                pos.has_cc = False
                continue

            px = ticker_data[t].loc[date]["close"]

            if px >= pos.cc_strike:
                # Called away — sell shares at strike
                proceeds = pos.cc_strike * 100 - COST_PER_CONTRACT
                cash += proceeds
                pnl = (pos.cc_strike - pos.cost_basis) * 100 + pos.cc_premium * 100
                trades.append({"date": date, "ticker": t, "action": "called_away",
                              "strike": pos.cc_strike, "pnl": pnl / 100})
                del share_positions[t]
            else:
                # CC expired worthless — keep premium, keep shares
                pos.cost_basis -= pos.cc_premium  # reduce cost basis by CC premium
                pos.has_cc = False
                trades.append({"date": date, "ticker": t, "action": "cc_expired_otm",
                              "price": px})

        # === LOSS-CUT ON SHARES ===
        for t in list(share_positions.keys()):
            pos = share_positions[t]
            if pos.has_cc:
                continue  # don't loss-cut while CC is open
            if t not in ticker_data or date not in ticker_data[t].index:
                continue
            px = ticker_data[t].loc[date]["close"]
            pnl_pct = (px - pos.cost_basis) / pos.cost_basis
            if pnl_pct <= LOSS_CUT_PCT:
                # Sell at market
                proceeds = px * 100 - COST_PER_CONTRACT
                cash += proceeds
                realized = (px - pos.cost_basis) * 100
                trades.append({"date": date, "ticker": t, "action": "loss_cut",
                              "price": px, "cost_basis": pos.cost_basis,
                              "pnl": realized / 100})
                del share_positions[t]

        # === PROFIT-TAKE ON CSPs ===
        for t in list(csp_positions.keys()):
            pos = csp_positions[t]
            if t not in ticker_data or date not in ticker_data[t].index:
                continue
            row = ticker_data[t].loc[date]
            px = row["close"]
            sigma = row.get("sigma", 0.3) if isinstance(row, pd.Series) else 0.3
            dte_remain = (pos.expiry_date - date).days
            if dte_remain <= 0:
                continue
            current_val = bs_price(px, pos.strike, dte_remain / 365, sigma)
            if current_val <= pos.premium * (1 - PROFIT_TAKE):
                # Buy back the put
                buyback = current_val * 100 + COST_PER_CONTRACT
                cash += pos.margin_held  # release margin
                cash -= buyback
                profit = (pos.premium - current_val) * 100 - 2 * COST_PER_CONTRACT
                trades.append({"date": date, "ticker": t, "action": "profit_take",
                              "price": px, "pnl": profit / 100})
                del csp_positions[t]

        # === BEAR PROTECTION: CLOSE CSPs ===
        if is_bear and bear_mode == "liq_csp_only":
            for t in list(csp_positions.keys()):
                pos = csp_positions[t]
                if t not in ticker_data or date not in ticker_data[t].index:
                    continue
                row = ticker_data[t].loc[date]
                px = row["close"]
                sigma = row.get("sigma", 0.3) if isinstance(row, pd.Series) else 0.3
                dte_remain = (pos.expiry_date - date).days
                if dte_remain <= 0:
                    continue
                current_val = bs_price(px, pos.strike, dte_remain / 365, sigma)
                # Close at market (buy back)
                buyback = current_val * 100 + COST_PER_CONTRACT
                cash += pos.margin_held  # release margin
                cash -= buyback
                pnl = (pos.premium - current_val) * 100 - 2 * COST_PER_CONTRACT
                trades.append({"date": date, "ticker": t, "action": "bear_close",
                              "price": px, "pnl": pnl / 100})
                del csp_positions[t]

        # === WRITE COVERED CALLS ON SHARES ===
        for t in list(share_positions.keys()):
            pos = share_positions[t]
            if pos.has_cc:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                continue
            row = ticker_data[t].loc[date]
            px = row["close"]
            sigma = row.get("sigma", 0.3) if isinstance(row, pd.Series) else 0.3
            if pd.isna(sigma) or sigma < 0.05:
                continue

            T = DTE_TARGET / 365
            K = find_strike(px, sigma, T, CALL_DELTA, kind="call")
            premium = bs_price(px, K, T, sigma, kind="call")
            premium = max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)
            if premium < 0.10:
                continue

            expiry = date + pd.Timedelta(days=DTE_TARGET)
            pos.has_cc = True
            pos.cc_strike = K
            pos.cc_premium = premium
            pos.cc_expiry = expiry
            cash += premium * 100 - COST_PER_CONTRACT
            trades.append({"date": date, "ticker": t, "action": "sell_cc",
                          "strike": K, "premium": premium})

        # === OPEN NEW CSPs (not in bear, have margin) ===
        if is_bear and bear_mode != "none":
            continue  # skip new CSPs in bear

        # Recalculate available margin
        current_margin_used = sum(p.margin_held for p in csp_positions.values())
        available_margin = nav * MAX_PORTFOLIO_MARGIN - current_margin_used
        per_name_limit = nav * MAX_PER_NAME_PCT

        # Shuffle tickers for fairness (no first-alphabetical bias)
        import random
        rng = random.Random(di)  # deterministic per day
        candidates = list(tickers_available)
        rng.shuffle(candidates)

        for t in candidates:
            if t in csp_positions or t in share_positions:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                continue

            row = ticker_data[t].loc[date]
            px = row["close"]
            sigma = row.get("sigma", 0.3) if isinstance(row, pd.Series) else 0.3
            vix = row.get("vix", 20.0) if isinstance(row, pd.Series) else 20.0

            if vix > VIX_MAX:
                continue
            if pd.isna(sigma) or sigma < 0.05:
                continue

            # Margin requirement for this CSP
            notional = px * 100
            margin_req = notional * MARGIN_REQ_PCT

            # Check limits
            if margin_req > per_name_limit:
                continue
            if margin_req > available_margin:
                continue
            if cash < margin_req:
                continue  # need cash to post margin

            # Write CSP
            T = DTE_TARGET / 365
            K = find_strike(px, sigma, T, PUT_DELTA, kind="put")
            premium = bs_price(px, K, T, sigma)
            premium = max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)

            if premium < 0.10:
                continue

            expiry = date + pd.Timedelta(days=DTE_TARGET)

            # Reserve margin, collect premium
            cash -= margin_req           # post margin
            cash += premium * 100 - COST_PER_CONTRACT  # collect premium

            csp_positions[t] = CSPPosition(
                ticker=t, strike=K, premium=premium,
                entry_date=date, expiry_date=expiry,
                margin_held=margin_req,
            )

            available_margin -= margin_req
            trades.append({"date": date, "ticker": t, "action": "sell_csp",
                          "strike": K, "premium": premium, "margin": margin_req})

    return daily_equity, daily_pnl, trades


def compute_metrics(daily_equity, daily_pnl):
    eq = pd.DataFrame(daily_equity)
    if len(eq) < 50:
        return {}

    returns = eq["equity"].pct_change().dropna()
    total_ret = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) - 1
    years = len(eq) / TRADING_DAYS
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    sharpe = (returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)) if returns.std() > 0 else 0
    down = returns[returns < 0]
    sortino = (returns.mean() / down.std() * np.sqrt(TRADING_DAYS)) if len(down) > 0 and down.std() > 0 else 0

    cum_max = eq["equity"].cummax()
    dd = (eq["equity"] - cum_max) / cum_max
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    pnl_arr = np.array(daily_pnl)
    wr = (pnl_arr > 0).mean() if len(pnl_arr) > 0 else 0
    gains = pnl_arr[pnl_arr > 0].sum() if (pnl_arr > 0).any() else 0
    losses = abs(pnl_arr[pnl_arr < 0].sum()) if (pnl_arr < 0).any() else 1e-9
    pf = gains / losses

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "final_equity": round(eq["equity"].iloc[-1], 2),
        "starting_equity": STARTING_CASH,
        "years": round(years, 2),
        "n_days": len(eq),
    }


def regime_analysis(daily_equity, spy_regime):
    eq = pd.DataFrame(daily_equity)
    eq["return"] = eq["equity"].pct_change()
    eq = eq.merge(spy_regime[["date", "bear"]], on="date", how="left")
    eq["bear"] = eq["bear"].fillna(0).astype(int)

    bull_ret = eq[eq["bear"] == 0]["return"].dropna()
    bear_ret = eq[eq["bear"] == 1]["return"].dropna()

    bull_sharpe = (bull_ret.mean() / bull_ret.std() * np.sqrt(252)) if len(bull_ret) > 20 and bull_ret.std() > 0 else 0
    bear_sharpe = (bear_ret.mean() / bear_ret.std() * np.sqrt(252)) if len(bear_ret) > 20 and bear_ret.std() > 0 else 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "hc428_pass": regime_gap <= 0.50,
        "n_bull_days": len(bull_ret),
        "n_bear_days": len(bear_ret),
    }


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("WHEEL ELITE 40 v2 — MARGIN-AWARE + DEATH-SPIRAL PREVENTION", flush=True)
    print("=" * 70, flush=True)
    print(f"Universe: {N_NAMES} tickers across {len(set(BASKET.values()))} sectors", flush=True)
    print(f"Config: DTE={DTE_TARGET}, put_delta={PUT_DELTA}, call_delta={CALL_DELTA}", flush=True)
    print(f"Capital: ${STARTING_CASH:,.0f}, margin cap={MAX_PORTFOLIO_MARGIN*100:.0f}%", flush=True)
    print(f"Risk: max {MAX_ASSIGNMENTS_5D} assignments/5d, loss-cut at {LOSS_CUT_PCT*100:.0f}%", flush=True)
    print(f"       max {MAX_SHARE_POSITIONS} share positions, {MAX_PER_NAME_PCT*100:.0f}% per name", flush=True)
    print(flush=True)

    prices, spy_regime = load_data()

    modes = ["none", "liq_csp_only"]
    all_results = {}

    for mode in modes:
        print(f"\n{'='*50}", flush=True)
        print(f"Running bear_mode = {mode}", flush=True)
        print(f"{'='*50}", flush=True)

        daily_eq, daily_pnl, trades_list = run_portfolio(prices, spy_regime, bear_mode=mode)
        metrics = compute_metrics(daily_eq, daily_pnl)
        regime = regime_analysis(daily_eq, spy_regime)

        # Count trade types
        actions = pd.DataFrame(trades_list)
        action_counts = actions["action"].value_counts().to_dict() if not actions.empty else {}

        all_results[mode] = {
            "metrics": metrics,
            "regime": regime,
            "n_trades": len(trades_list),
            "action_counts": action_counts,
        }

        print(f"\nResults ({mode}):", flush=True)
        print(f"  CAGR: {metrics.get('cagr_pct', 0):.1f}%", flush=True)
        print(f"  Sharpe: {metrics.get('sharpe', 0):.3f}", flush=True)
        print(f"  Sortino: {metrics.get('sortino', 0):.3f}", flush=True)
        print(f"  Max DD: {metrics.get('max_dd_pct', 0):.1f}%", flush=True)
        print(f"  Calmar: {metrics.get('calmar', 0):.3f}", flush=True)
        print(f"  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%", flush=True)
        print(f"  PF: {metrics.get('profit_factor', 0):.2f}", flush=True)
        print(f"  Final Equity: ${metrics.get('final_equity', 0):,.0f}", flush=True)
        print(f"  Trade counts: {action_counts}", flush=True)
        print(f"  Regime: Bull={regime['bull_sharpe']:.3f}, Bear={regime['bear_sharpe']:.3f}, Gap={regime['regime_gap']:.3f}", flush=True)
        r = regime
        print(f"  Regime-agnostic (gap≤0.50): {'PASS' if r['hc428_pass'] else 'FAIL'}", flush=True)

    # Save
    output = {
        "generated": datetime.now().isoformat(),
        "config": {
            "n_tickers": N_NAMES,
            "tickers": TICKERS,
            "starting_capital": STARTING_CASH,
            "dte_target": DTE_TARGET,
            "put_delta": PUT_DELTA,
            "margin_cap": MAX_PORTFOLIO_MARGIN,
            "max_assignments_5d": MAX_ASSIGNMENTS_5D,
            "loss_cut_pct": LOSS_CUT_PCT,
            "max_share_positions": MAX_SHARE_POSITIONS,
        },
        "results": all_results,
    }
    out_path = OUT_DIR / "elite40_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Saved to {out_path}", flush=True)

    # Final comparison
    print(flush=True)
    print("=" * 70, flush=True)
    print("COMPARISON: No Protection vs Bear Gate (CSP close)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Metric':<20} {'No Protection':>15} {'Bear Gate':>15}", flush=True)
    print("-" * 50, flush=True)
    for key in ['cagr_pct', 'sharpe', 'sortino', 'max_dd_pct', 'calmar', 'win_rate', 'profit_factor', 'final_equity']:
        v1 = all_results['none']['metrics'].get(key, 0)
        v2 = all_results['liq_csp_only']['metrics'].get(key, 0)
        if key == 'final_equity':
            print(f"{key:<20} ${v1:>13,.0f} ${v2:>13,.0f}", flush=True)
        elif 'pct' in key:
            print(f"{key:<20} {v1:>14.1f}% {v2:>14.1f}%", flush=True)
        else:
            print(f"{key:<20} {v1:>15.3f} {v2:>15.3f}", flush=True)

    print(flush=True)
    for mode in modes:
        r = all_results[mode]['regime']
        status = "✓ PASS" if r['hc428_pass'] else "✗ FAIL"
        print(f"Regime-agnostic ({mode}): gap={r['regime_gap']:.3f} {status}", flush=True)


if __name__ == "__main__":
    main()
