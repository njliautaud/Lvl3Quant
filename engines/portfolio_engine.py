"""
Unified Portfolio Engine -- Daily paper trading across validated strategies.

Combines three tiers of validated edges into a single portfolio:
  Tier 1 (Regime-Agnostic):
    1. VIX Panic Buy: Buy SPY/UPRO when VIX > 30 (or > 25 with confluence). Hold 5-20 days.
    2. ETF Short-Term Reversal: Buy bottom 5 ETFs by 5-day return, hold 5 days. Weekly rebalance.
  Tier 2 (Leveraged Beta with Timing):
    3. VIX-Threshold Leveraged Growth: Hold UPRO/TQQQ sized inversely to VIX.
    4. Dual Momentum: Switch TQQQ/QQQ/SPY monthly based on trailing 6-month returns.
  Tier 3 (Income):
    5. Diversified CSP -- runs separately, not managed here.

Capital allocation adapts to VIX regime:
  Normal (VIX < 25):  50% leveraged growth, 25% ETF reversal, 20% cash reserve, 5% panic reserve
  Elevated (25-30):   30% growth, 25% reversal, 30% cash, 15% panic reserve
  Crisis (VIX > 30):  PANIC BUY mode -- deploy panic+cash reserves into SPY/UPRO

Usage:
  python3 portfolio_engine.py             # daily run (default)
  python3 portfolio_engine.py --status    # print current state, no trades
  python3 portfolio_engine.py --backtest  # backtest last 30 trading days

Runs daily at 16:05 ET via PM2 cron.

Outputs:
  - state/portfolio_engine_state.json
  - logs/portfolio_engine.log
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pytz

ET = pytz.timezone("US/Eastern")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # /home/jupiter/Lvl3Quant
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "portfolio_engine_state.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Portfolio] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "portfolio_engine.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("portfolio_engine")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INITIAL_NAV = 100_000.0

# ETF universe for reversal strategy
REVERSAL_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP",
    "XLU", "XLRE", "XLB", "XBI", "XOP", "XHB", "XME", "KRE", "IYT", "ITB",
]
REVERSAL_TOP_N = 5
REVERSAL_LOOKBACK_DAYS = 5
REVERSAL_HOLD_DAYS = 5

# VIX thresholds
VIX_ELEVATED = 25.0
VIX_CRISIS = 30.0

# Dual momentum
DUAL_MOM_LOOKBACK_DAYS = 126  # ~6 months
DUAL_MOM_ASSETS = ["TQQQ", "QQQ", "SPY"]

# Leveraged growth
GROWTH_ASSETS = ["UPRO", "TQQQ"]

# Panic buy
PANIC_ASSETS = ["SPY", "UPRO"]

# Allocation regimes (fractions of NAV)
ALLOC_NORMAL = {
    "leveraged_growth": 0.50,
    "etf_reversal": 0.25,
    "cash_reserve": 0.20,
    "panic_reserve": 0.05,
}
ALLOC_ELEVATED = {
    "leveraged_growth": 0.30,
    "etf_reversal": 0.25,
    "cash_reserve": 0.30,
    "panic_reserve": 0.15,
}
# In crisis, panic reserve + some cash gets deployed


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def default_state() -> dict:
    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    return {
        "nav": INITIAL_NAV,
        "cash": INITIAL_NAV,
        "positions": {},  # ticker -> {shares, cost_basis, strategy, entry_date}
        "trade_log": [],
        "start_date": today_str,
        "daily_returns": [],  # [{date, nav, daily_return}]
        "peak_nav": INITIAL_NAV,
        "last_run_date": None,
        "last_reversal_rebalance": None,
        "last_dual_mom_rebalance": None,
        "panic_active": False,
        "panic_entry_date": None,
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return default_state()


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def fetch_market_data(as_of_date: str | None = None):
    """Fetch all required market data via yfinance.

    If as_of_date is provided (YYYY-MM-DD), fetch data up to that date.
    Always uses yesterday's close (no look-ahead).
    """
    import yfinance as yf

    all_tickers = list(set(
        REVERSAL_ETFS + DUAL_MOM_ASSETS + GROWTH_ASSETS + PANIC_ASSETS
        + ["^VIX", "SPY", "HYG", "IEF"]
    ))
    tickers_str = " ".join(all_tickers)

    if as_of_date:
        end_dt = datetime.strptime(as_of_date, "%Y-%m-%d") + timedelta(days=1)
        start_dt = end_dt - timedelta(days=400)  # enough for 6mo momentum + SMA
        data = yf.download(
            tickers_str, start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"), interval="1d", progress=False
        )
    else:
        data = yf.download(tickers_str, period="18mo", interval="1d", progress=False)

    if data.empty:
        raise RuntimeError("No market data returned from yfinance")

    close = data["Close"]
    trade_date = close.index[-1].strftime("%Y-%m-%d")

    # VIX
    vix_series = close["^VIX"].dropna()
    vix_level = float(vix_series.iloc[-1])

    # SPY price and SMA200
    spy_series = close["SPY"].dropna()
    spy_price = float(spy_series.iloc[-1])
    spy_sma200 = float(spy_series.rolling(200).mean().iloc[-1]) if len(spy_series) >= 200 else spy_price

    # HYG/IEF credit ratio
    hyg_s = close["HYG"].dropna()
    ief_s = close["IEF"].dropna()
    common_idx = hyg_s.index.intersection(ief_s.index)
    if len(common_idx) >= 20:
        ratio = hyg_s.loc[common_idx] / ief_s.loc[common_idx]
        credit_drop = float(ratio.iloc[-1] / ratio.iloc[-20] - 1)
    else:
        credit_drop = 0.0

    # Market breadth proxy: % of reversal ETFs above their 20-day SMA
    breadth_count = 0
    for etf in REVERSAL_ETFS:
        try:
            s = close[etf].dropna()
            if len(s) >= 20 and float(s.iloc[-1]) > float(s.rolling(20).mean().iloc[-1]):
                breadth_count += 1
        except Exception:
            pass
    breadth_pct = breadth_count / len(REVERSAL_ETFS) * 100

    # Prices for all tickers
    prices = {}
    for t in all_tickers:
        if t == "^VIX":
            continue
        try:
            s = close[t].dropna()
            if len(s) > 0:
                prices[t] = float(s.iloc[-1])
        except Exception:
            pass

    # ETF reversal: 5-day returns
    reversal_returns = {}
    for etf in REVERSAL_ETFS:
        try:
            s = close[etf].dropna()
            if len(s) >= REVERSAL_LOOKBACK_DAYS + 1:
                ret = float(s.iloc[-1] / s.iloc[-REVERSAL_LOOKBACK_DAYS - 1] - 1)
                reversal_returns[etf] = ret
        except Exception:
            pass

    # Dual momentum: 6-month returns
    dual_mom_returns = {}
    for asset in DUAL_MOM_ASSETS:
        try:
            s = close[asset].dropna()
            if len(s) >= DUAL_MOM_LOOKBACK_DAYS + 1:
                ret = float(s.iloc[-1] / s.iloc[-DUAL_MOM_LOOKBACK_DAYS - 1] - 1)
                dual_mom_returns[asset] = ret
        except Exception:
            pass

    return {
        "date": trade_date,
        "vix": vix_level,
        "spy_price": spy_price,
        "spy_sma200": spy_sma200,
        "breadth_pct": breadth_pct,
        "credit_drop": credit_drop,
        "prices": prices,
        "reversal_returns": reversal_returns,
        "dual_mom_returns": dual_mom_returns,
        "close_df": close,  # for backtest mode
    }


# ---------------------------------------------------------------------------
# Strategy signals
# ---------------------------------------------------------------------------
def get_regime(vix: float) -> str:
    if vix >= VIX_CRISIS:
        return "crisis"
    elif vix >= VIX_ELEVATED:
        return "elevated"
    return "normal"


def get_allocation(regime: str) -> dict:
    if regime == "crisis":
        return ALLOC_ELEVATED.copy()  # crisis deploy handled in panic logic
    elif regime == "elevated":
        return ALLOC_ELEVATED.copy()
    return ALLOC_NORMAL.copy()


def signal_vix_panic(mkt: dict) -> dict:
    """VIX Panic Buy: buy when VIX > 30 or > 25 with confluence."""
    vix = mkt["vix"]
    breadth = mkt["breadth_pct"]
    credit_drop = mkt["credit_drop"]

    trigger = False
    reasons = []

    if vix >= VIX_CRISIS:
        trigger = True
        reasons.append(f"VIX={vix:.1f} >= {VIX_CRISIS}")

    if vix >= VIX_ELEVATED:
        if breadth < 30:
            trigger = True
            reasons.append(f"VIX={vix:.1f} + breadth={breadth:.0f}% < 30%")
        if credit_drop < -0.03:
            trigger = True
            reasons.append(f"VIX={vix:.1f} + HYG/IEF drop={credit_drop:.2%}")

    return {
        "strategy": "vix_panic",
        "trigger": trigger,
        "reasons": reasons,
        "target_assets": PANIC_ASSETS,
    }


def signal_etf_reversal(mkt: dict) -> dict:
    """ETF reversal: buy bottom 5 by 5-day return."""
    returns = mkt["reversal_returns"]
    if len(returns) < REVERSAL_TOP_N:
        return {"strategy": "etf_reversal", "trigger": False, "reasons": ["insufficient data"], "picks": []}

    sorted_etfs = sorted(returns.items(), key=lambda x: x[1])
    bottom_n = sorted_etfs[:REVERSAL_TOP_N]

    return {
        "strategy": "etf_reversal",
        "trigger": True,
        "reasons": [f"bottom {REVERSAL_TOP_N} by 5d return"],
        "picks": [t for t, _ in bottom_n],
        "returns": {t: r for t, r in bottom_n},
    }


def signal_leveraged_growth(mkt: dict) -> dict:
    """Leveraged growth: position size inversely scaled to VIX."""
    vix = mkt["vix"]
    # Scale: at VIX=12 -> 100% of allocation, VIX=30 -> 40%, VIX=40 -> 20%
    # Linear scale: size = max(0.2, 1.0 - (vix - 12) / 35)
    size_factor = max(0.20, min(1.0, 1.0 - (vix - 12.0) / 35.0))

    return {
        "strategy": "leveraged_growth",
        "trigger": True,
        "size_factor": size_factor,
        "reasons": [f"VIX={vix:.1f}, size_factor={size_factor:.2f}"],
        "target_assets": GROWTH_ASSETS,
    }


def signal_dual_momentum(mkt: dict) -> dict:
    """Dual momentum: pick best performer over 6 months from TQQQ/QQQ/SPY."""
    returns = mkt["dual_mom_returns"]
    if not returns:
        return {"strategy": "dual_momentum", "trigger": False, "reasons": ["no data"], "pick": None}

    best = max(returns, key=returns.get)
    best_ret = returns[best]

    # If best return is negative, go to cash (SPY as proxy)
    if best_ret < 0:
        return {
            "strategy": "dual_momentum",
            "trigger": True,
            "pick": "SPY",
            "reasons": [f"all negative 6mo returns, defensive to SPY"],
            "returns": returns,
        }

    return {
        "strategy": "dual_momentum",
        "trigger": True,
        "pick": best,
        "reasons": [f"best 6mo: {best} at {best_ret:.1%}"],
        "returns": returns,
    }


# ---------------------------------------------------------------------------
# Portfolio operations
# ---------------------------------------------------------------------------
def execute_trade(state: dict, ticker: str, shares: float, price: float,
                  strategy: str, action: str, reason: str) -> None:
    """Paper trade: buy or sell shares of a ticker."""
    if shares <= 0:
        return

    trade_value = shares * price
    trade_record = {
        "date": state.get("_current_date", datetime.now(ET).strftime("%Y-%m-%d")),
        "ticker": ticker,
        "action": action,
        "shares": round(shares, 4),
        "price": round(price, 4),
        "value": round(trade_value, 2),
        "strategy": strategy,
        "reason": reason,
    }

    if action == "BUY":
        if trade_value > state["cash"]:
            # Reduce to what we can afford
            shares = math.floor(state["cash"] / price)
            if shares <= 0:
                log.warning(f"Cannot buy {ticker}: insufficient cash (${state['cash']:.2f})")
                return
            trade_value = shares * price
            trade_record["shares"] = shares
            trade_record["value"] = round(trade_value, 2)

        state["cash"] -= trade_value
        if ticker in state["positions"]:
            pos = state["positions"][ticker]
            total_shares = pos["shares"] + shares
            pos["cost_basis"] = (pos["cost_basis"] * pos["shares"] + price * shares) / total_shares
            pos["shares"] = total_shares
        else:
            state["positions"][ticker] = {
                "shares": shares,
                "cost_basis": round(price, 4),
                "strategy": strategy,
                "entry_date": trade_record["date"],
            }
        log.info(f"BUY {shares} {ticker} @ ${price:.2f} = ${trade_value:.2f} [{strategy}] {reason}")

    elif action == "SELL":
        if ticker not in state["positions"]:
            log.warning(f"Cannot sell {ticker}: no position")
            return
        pos = state["positions"][ticker]
        sell_shares = min(shares, pos["shares"])
        sell_value = sell_shares * price
        trade_record["shares"] = round(sell_shares, 4)
        trade_record["value"] = round(sell_value, 2)

        state["cash"] += sell_value
        pos["shares"] -= sell_shares
        if pos["shares"] <= 0.01:  # float cleanup
            del state["positions"][ticker]
        log.info(f"SELL {sell_shares} {ticker} @ ${price:.2f} = ${sell_value:.2f} [{strategy}] {reason}")

    state["trade_log"].append(trade_record)


def close_strategy_positions(state: dict, strategy: str, prices: dict, reason: str) -> None:
    """Close all positions belonging to a strategy."""
    to_close = [(t, p) for t, p in state["positions"].items() if p["strategy"] == strategy]
    for ticker, pos in to_close:
        if ticker in prices:
            execute_trade(state, ticker, pos["shares"], prices[ticker], strategy, "SELL", reason)


def compute_nav(state: dict, prices: dict) -> float:
    """Compute total NAV = cash + sum of position market values."""
    position_value = 0.0
    for ticker, pos in state["positions"].items():
        if ticker in prices:
            position_value += pos["shares"] * prices[ticker]
        else:
            # Use cost basis if current price unavailable
            position_value += pos["shares"] * pos["cost_basis"]
    return state["cash"] + position_value


def compute_metrics(daily_returns: list) -> dict:
    """Compute portfolio-level risk metrics from daily return history."""
    if len(daily_returns) < 2:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "total_return": 0.0}

    rets = np.array([d["daily_return"] for d in daily_returns])
    navs = np.array([d["nav"] for d in daily_returns])

    total_return = (navs[-1] / navs[0] - 1) if navs[0] > 0 else 0.0
    ann_factor = np.sqrt(252)

    mean_ret = np.mean(rets)
    std_ret = np.std(rets)
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0.0

    downside = rets[rets < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-9
    sortino = (mean_ret / downside_std * ann_factor) if downside_std > 0 else 0.0

    # Max drawdown
    peak = navs[0]
    max_dd = 0.0
    for n in navs:
        peak = max(peak, n)
        dd = (peak - n) / peak
        max_dd = max(max_dd, dd)

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "total_return": round(total_return * 100, 2),
        "trading_days": len(daily_returns),
    }


# ---------------------------------------------------------------------------
# Main daily logic
# ---------------------------------------------------------------------------
def run_daily(state: dict, mkt: dict) -> dict:
    """Execute one day of portfolio management. Returns updated state."""
    trade_date = mkt["date"]
    prices = mkt["prices"]
    state["_current_date"] = trade_date

    # Skip if already ran today
    if state["last_run_date"] == trade_date:
        log.info(f"Already ran for {trade_date}, skipping")
        return state

    prev_nav = state["nav"]
    regime = get_regime(mkt["vix"])
    alloc = get_allocation(regime)
    log.info(f"=== {trade_date} | VIX={mkt['vix']:.1f} | Regime={regime} | Breadth={mkt['breadth_pct']:.0f}% ===")

    # --- Strategy 1: VIX Panic Buy ---
    panic_sig = signal_vix_panic(mkt)
    if panic_sig["trigger"] and not state["panic_active"]:
        # Enter panic positions: deploy panic reserve + available cash into SPY/UPRO
        panic_budget = state["nav"] * alloc.get("panic_reserve", 0.05)
        if regime == "crisis":
            # In crisis, also deploy half of cash reserve
            panic_budget += state["nav"] * alloc.get("cash_reserve", 0.20) * 0.5
        # Split 60/40 between SPY and UPRO
        for asset, frac in [("SPY", 0.60), ("UPRO", 0.40)]:
            budget = panic_budget * frac
            if asset in prices and prices[asset] > 0:
                shares = math.floor(budget / prices[asset])
                if shares > 0:
                    execute_trade(state, asset, shares, prices[asset], "vix_panic", "BUY",
                                  "; ".join(panic_sig["reasons"]))
        state["panic_active"] = True
        state["panic_entry_date"] = trade_date
        log.info(f"PANIC BUY activated: {panic_sig['reasons']}")

    elif state["panic_active"] and not panic_sig["trigger"]:
        # Check hold period: exit after 5-20 days depending on conditions
        if state["panic_entry_date"]:
            entry_dt = datetime.strptime(state["panic_entry_date"], "%Y-%m-%d")
            trade_dt = datetime.strptime(trade_date, "%Y-%m-%d")
            hold_days = (trade_dt - entry_dt).days
            # Exit if held >= 10 days and VIX back below 22 (conservative)
            if hold_days >= 10 and mkt["vix"] < 22:
                close_strategy_positions(state, "vix_panic", prices, f"panic exit after {hold_days}d, VIX={mkt['vix']:.1f}")
                state["panic_active"] = False
                state["panic_entry_date"] = None
            elif hold_days >= 20:
                close_strategy_positions(state, "vix_panic", prices, f"max hold {hold_days}d reached")
                state["panic_active"] = False
                state["panic_entry_date"] = None

    # --- Strategy 2: ETF Reversal (weekly rebalance) ---
    reversal_sig = signal_etf_reversal(mkt)
    trade_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    do_reversal_rebal = False
    if state["last_reversal_rebalance"]:
        last_rebal = datetime.strptime(state["last_reversal_rebalance"], "%Y-%m-%d")
        if (trade_dt - last_rebal).days >= REVERSAL_HOLD_DAYS:
            do_reversal_rebal = True
    else:
        do_reversal_rebal = True

    if do_reversal_rebal and reversal_sig["trigger"]:
        # Close existing reversal positions
        close_strategy_positions(state, "etf_reversal", prices, "weekly rebalance")

        # Buy new picks
        reversal_budget = state["nav"] * alloc["etf_reversal"]
        per_etf = reversal_budget / REVERSAL_TOP_N
        for etf in reversal_sig["picks"]:
            if etf in prices and prices[etf] > 0:
                shares = math.floor(per_etf / prices[etf])
                if shares > 0:
                    ret_str = f"{reversal_sig['returns'].get(etf, 0):.2%}" if "returns" in reversal_sig else ""
                    execute_trade(state, etf, shares, prices[etf], "etf_reversal", "BUY",
                                  f"5d ret={ret_str}")
        state["last_reversal_rebalance"] = trade_date
        log.info(f"ETF reversal rebalanced: {reversal_sig['picks']}")

    # --- Strategy 3: Leveraged Growth ---
    growth_sig = signal_leveraged_growth(mkt)
    # Dual momentum picks which asset to hold
    dual_sig = signal_dual_momentum(mkt)

    # Monthly rebalance for leveraged growth + dual momentum
    do_growth_rebal = False
    if state["last_dual_mom_rebalance"]:
        last_mom = datetime.strptime(state["last_dual_mom_rebalance"], "%Y-%m-%d")
        # Monthly: different month
        if trade_dt.strftime("%Y-%m") != last_mom.strftime("%Y-%m"):
            do_growth_rebal = True
    else:
        do_growth_rebal = True

    if do_growth_rebal:
        # Close existing growth and dual momentum positions
        close_strategy_positions(state, "leveraged_growth", prices, "monthly rebalance")
        close_strategy_positions(state, "dual_momentum", prices, "monthly rebalance")

        growth_budget = state["nav"] * alloc["leveraged_growth"]

        if dual_sig["trigger"] and dual_sig.get("pick"):
            # Split growth budget: 60% into VIX-sized UPRO, 40% into dual momentum pick
            # Apply VIX sizing to leveraged portion
            lev_budget = growth_budget * 0.60 * growth_sig["size_factor"]
            mom_budget = growth_budget * 0.40

            # Leveraged growth: buy UPRO
            if "UPRO" in prices and prices["UPRO"] > 0:
                shares = math.floor(lev_budget / prices["UPRO"])
                if shares > 0:
                    execute_trade(state, "UPRO", shares, prices["UPRO"], "leveraged_growth", "BUY",
                                  f"VIX-sized factor={growth_sig['size_factor']:.2f}")

            # Dual momentum: buy the winner
            mom_pick = dual_sig["pick"]
            if mom_pick in prices and prices[mom_pick] > 0:
                shares = math.floor(mom_budget / prices[mom_pick])
                if shares > 0:
                    execute_trade(state, mom_pick, shares, prices[mom_pick], "dual_momentum", "BUY",
                                  dual_sig["reasons"][0])

        state["last_dual_mom_rebalance"] = trade_date
        log.info(f"Growth/momentum rebalanced: growth_factor={growth_sig['size_factor']:.2f}, "
                 f"momentum_pick={dual_sig.get('pick', 'none')}")

    # --- Update NAV ---
    nav = compute_nav(state, prices)
    daily_ret = (nav / prev_nav - 1) if prev_nav > 0 else 0.0
    state["nav"] = round(nav, 2)
    state["peak_nav"] = max(state.get("peak_nav", nav), nav)
    state["last_run_date"] = trade_date
    state["daily_returns"].append({
        "date": trade_date,
        "nav": round(nav, 2),
        "daily_return": round(daily_ret, 6),
    })

    # Cleanup
    if "_current_date" in state:
        del state["_current_date"]

    drawdown = (state["peak_nav"] - nav) / state["peak_nav"] * 100 if state["peak_nav"] > 0 else 0.0
    log.info(f"NAV=${nav:,.2f} | Daily={daily_ret:+.2%} | DD={drawdown:.1f}% | "
             f"Cash=${state['cash']:,.2f} | Positions={len(state['positions'])}")

    return state


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------
def print_status(state: dict):
    """Print current portfolio snapshot."""
    print("\n" + "=" * 70)
    print("UNIFIED PORTFOLIO ENGINE -- Status")
    print("=" * 70)
    print(f"Start Date:  {state.get('start_date', 'N/A')}")
    print(f"Last Run:    {state.get('last_run_date', 'N/A')}")
    print(f"NAV:         ${state['nav']:>12,.2f}")
    print(f"Cash:        ${state['cash']:>12,.2f}")
    peak = state.get("peak_nav", state["nav"])
    dd = (peak - state["nav"]) / peak * 100 if peak > 0 else 0.0
    print(f"Peak NAV:    ${peak:>12,.2f}")
    print(f"Drawdown:    {dd:>11.1f}%")
    print(f"Panic Mode:  {'ACTIVE' if state.get('panic_active') else 'OFF'}")

    positions = state.get("positions", {})
    if positions:
        print(f"\nPositions ({len(positions)}):")
        print(f"  {'Ticker':<8} {'Shares':>8} {'Cost':>10} {'Strategy':<20} {'Entry':<12}")
        print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*20} {'-'*12}")
        for ticker, pos in sorted(positions.items()):
            print(f"  {ticker:<8} {pos['shares']:>8.1f} ${pos['cost_basis']:>9.2f} "
                  f"{pos['strategy']:<20} {pos.get('entry_date', 'N/A'):<12}")
    else:
        print("\nNo open positions.")

    # Metrics
    daily_rets = state.get("daily_returns", [])
    if len(daily_rets) >= 2:
        metrics = compute_metrics(daily_rets)
        print(f"\nPerformance ({metrics['trading_days']} days):")
        print(f"  Total Return:  {metrics['total_return']:>8.2f}%")
        print(f"  Sharpe:        {metrics['sharpe']:>8.3f}")
        print(f"  Sortino:       {metrics['sortino']:>8.3f}")
        print(f"  Max Drawdown:  {metrics['max_drawdown']:>8.2f}%")

    # Recent trades
    trades = state.get("trade_log", [])
    if trades:
        recent = trades[-10:]
        print(f"\nRecent Trades (last {len(recent)} of {len(trades)}):")
        for t in recent:
            print(f"  {t['date']} {t['action']:<4} {t['shares']:>6.0f} {t['ticker']:<6} "
                  f"@ ${t['price']:.2f} = ${t['value']:>10,.2f} [{t['strategy']}]")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------
def run_backtest(days: int = 30):
    """Run a historical backtest over the last N trading days."""
    import yfinance as yf

    log.info(f"Starting backtest over last {days} trading days...")

    all_tickers = list(set(
        REVERSAL_ETFS + DUAL_MOM_ASSETS + GROWTH_ASSETS + PANIC_ASSETS
        + ["^VIX", "SPY", "HYG", "IEF"]
    ))
    tickers_str = " ".join(all_tickers)

    # Fetch enough data for lookbacks + backtest period
    data = yf.download(tickers_str, period="18mo", interval="1d", progress=False)
    if data.empty:
        log.error("No data for backtest")
        return

    close = data["Close"]
    dates = close.index

    # We need at least DUAL_MOM_LOOKBACK_DAYS + days of data
    min_warmup = max(DUAL_MOM_LOOKBACK_DAYS, 200) + 10
    if len(dates) < min_warmup + days:
        log.warning(f"Only {len(dates)} trading days available, need {min_warmup + days}")
        days = max(1, len(dates) - min_warmup)

    backtest_dates = dates[-days:]
    state = default_state()
    state["start_date"] = backtest_dates[0].strftime("%Y-%m-%d")

    log.info(f"Backtest period: {backtest_dates[0].strftime('%Y-%m-%d')} to {backtest_dates[-1].strftime('%Y-%m-%d')}")

    for dt in backtest_dates:
        date_str = dt.strftime("%Y-%m-%d")
        # Build market data snapshot as of this date
        mask = close.index <= dt
        hist = close[mask]
        if len(hist) < 30:
            continue

        vix_s = hist["^VIX"].dropna()
        spy_s = hist["SPY"].dropna()

        if len(vix_s) == 0 or len(spy_s) == 0:
            continue

        vix_level = float(vix_s.iloc[-1])
        spy_price = float(spy_s.iloc[-1])
        spy_sma200 = float(spy_s.rolling(200).mean().iloc[-1]) if len(spy_s) >= 200 else spy_price

        # Breadth
        breadth_count = 0
        for etf in REVERSAL_ETFS:
            try:
                s = hist[etf].dropna()
                if len(s) >= 20 and float(s.iloc[-1]) > float(s.rolling(20).mean().iloc[-1]):
                    breadth_count += 1
            except Exception:
                pass
        breadth_pct = breadth_count / len(REVERSAL_ETFS) * 100

        # Credit
        hyg_s = hist["HYG"].dropna()
        ief_s = hist["IEF"].dropna()
        ci = hyg_s.index.intersection(ief_s.index)
        credit_drop = 0.0
        if len(ci) >= 20:
            ratio = hyg_s.loc[ci] / ief_s.loc[ci]
            credit_drop = float(ratio.iloc[-1] / ratio.iloc[-20] - 1)

        # Prices
        prices = {}
        for t in all_tickers:
            if t == "^VIX":
                continue
            try:
                s = hist[t].dropna()
                if len(s) > 0:
                    prices[t] = float(s.iloc[-1])
            except Exception:
                pass

        # Reversal returns
        reversal_returns = {}
        for etf in REVERSAL_ETFS:
            try:
                s = hist[etf].dropna()
                if len(s) >= REVERSAL_LOOKBACK_DAYS + 1:
                    reversal_returns[etf] = float(s.iloc[-1] / s.iloc[-REVERSAL_LOOKBACK_DAYS - 1] - 1)
            except Exception:
                pass

        # Dual momentum
        dual_mom_returns = {}
        for asset in DUAL_MOM_ASSETS:
            try:
                s = hist[asset].dropna()
                if len(s) >= DUAL_MOM_LOOKBACK_DAYS + 1:
                    dual_mom_returns[asset] = float(s.iloc[-1] / s.iloc[-DUAL_MOM_LOOKBACK_DAYS - 1] - 1)
            except Exception:
                pass

        mkt = {
            "date": date_str,
            "vix": vix_level,
            "spy_price": spy_price,
            "spy_sma200": spy_sma200,
            "breadth_pct": breadth_pct,
            "credit_drop": credit_drop,
            "prices": prices,
            "reversal_returns": reversal_returns,
            "dual_mom_returns": dual_mom_returns,
        }

        state = run_daily(state, mkt)

    # Final summary
    print_status(state)
    metrics = compute_metrics(state.get("daily_returns", []))
    log.info(f"Backtest complete: Return={metrics['total_return']:.2f}%, "
             f"Sharpe={metrics['sharpe']:.3f}, Sortino={metrics['sortino']:.3f}, "
             f"MaxDD={metrics['max_drawdown']:.2f}%")

    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified Portfolio Engine")
    parser.add_argument("--status", action="store_true", help="Print current state without trading")
    parser.add_argument("--backtest", action="store_true", help="Run backtest over last 30 trading days")
    parser.add_argument("--backtest-days", type=int, default=30, help="Number of trading days for backtest")
    args = parser.parse_args()

    if args.status:
        state = load_state()
        print_status(state)
        return

    if args.backtest:
        run_backtest(days=args.backtest_days)
        return

    # Daily run
    log.info("Starting daily portfolio run...")
    state = load_state()
    try:
        mkt = fetch_market_data()
        state = run_daily(state, mkt)
        save_state(state)
        print_status(state)
        log.info("Daily run complete, state saved.")
    except Exception as e:
        log.error(f"Daily run failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
