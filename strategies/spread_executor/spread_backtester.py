"""
Spread Backtester — backtest validated strategies using spreads vs shares/single options.

Uses yfinance for historical equity data and Black-Scholes for synthetic option pricing.
Compares: share returns vs spread returns for each strategy.
"""

import json
import os
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("WARNING: yfinance not installed. Using synthetic data.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from spread_strategy_engine import (
    bs_call, bs_put, estimate_option_price,
    determine_strike_increment, get_atm_strike,
    ACCOUNT_EQUITY, MAX_RISK_PCT,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_DIR = os.path.join(BASE_DIR, "state")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SpreadBT] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "spread_backtester.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Default IV by sector (rough estimates)
SECTOR_IV = {
    "XLK": 0.28, "XLF": 0.22, "XLE": 0.35, "XLV": 0.20,
    "XLI": 0.22, "XLP": 0.15, "XLU": 0.18, "XLB": 0.25,
    "XLC": 0.28, "XLRE": 0.22, "XLY": 0.25,
    "AAPL": 0.28, "MSFT": 0.26, "GOOGL": 0.30, "AMZN": 0.32,
    "NVDA": 0.45, "META": 0.35, "AMD": 0.45, "NFLX": 0.38,
    "default": 0.30,
}

# Validated strategies to backtest
STRATEGIES = {
    "etf_rotation_v3": {
        "type": "momentum",
        "universe": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLC", "XLRE", "XLY"],
        "lookback_days": 21,
        "hold_days": 5,
        "direction": "long_top_short_bottom",
        "sharpe_ref": 2.39,
    },
    "post_earnings_drift": {
        "type": "event",
        "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AMD", "NFLX"],
        "hold_days": 5,
        "direction": "follow_surprise",
        "sharpe_ref": 0.58,
    },
    "vix_put_spreads": {
        "type": "mean_reversion",
        "universe": ["SPY"],
        "lookback_days": 10,
        "hold_days": 7,
        "direction": "bull_when_vix_high",
        "sharpe_ref": 1.09,
    },
    "momentum_burst": {
        "type": "momentum",
        "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
        "lookback_days": 5,
        "hold_days": 3,
        "direction": "follow_momentum",
        "sharpe_ref": None,
    },
    "sector_spreads": {
        "type": "relative_value",
        "universe": ["XLK", "XLE", "XLF", "XLV", "XLI"],
        "lookback_days": 21,
        "hold_days": 5,
        "direction": "long_strong_short_weak",
        "sharpe_ref": None,
    },
    "jade_lizard_income": {
        "type": "income",
        "universe": ["AAPL", "MSFT", "JPM", "JNJ", "PG"],
        "hold_days": 30,
        "direction": "neutral_bullish",
        "sharpe_ref": 0.95,
    },
    "iron_condor": {
        "type": "income",
        "universe": ["SPY", "QQQ"],
        "hold_days": 7,
        "direction": "neutral",
        "sharpe_ref": 2.86,
    },
    "covered_call": {
        "type": "income",
        "universe": ["AAPL", "MSFT", "JPM", "PG", "KO"],
        "hold_days": 30,
        "direction": "neutral_bullish",
        "sharpe_ref": 0.28,
    },
    "vix_contango": {
        "type": "vol_selling",
        "universe": ["SPY"],
        "hold_days": 5,
        "direction": "short_vol",
        "sharpe_ref": 0.42,
    },
    "mean_reversion": {
        "type": "mean_reversion",
        "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "lookback_days": 10,
        "hold_days": 3,
        "direction": "counter_trend",
        "sharpe_ref": None,
    },
    "quality_momentum": {
        "type": "momentum",
        "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "LLY", "UNH", "JPM", "V"],
        "lookback_days": 63,
        "hold_days": 21,
        "direction": "follow_momentum",
        "sharpe_ref": None,
    },
}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_prices(ticker: str, period: str = "1y") -> Optional[Dict[str, List]]:
    """Fetch historical prices. Returns dict with dates and closes."""
    if not HAS_YFINANCE:
        # Generate synthetic data
        return _synthetic_prices(ticker, 252)

    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            log.warning(f"No data for {ticker}")
            return None
        # Handle multi-level columns from yfinance
        if hasattr(df.columns, 'levels'):
            closes = df[("Close", ticker)].tolist() if ("Close", ticker) in df.columns else df["Close"].iloc[:, 0].tolist()
        else:
            closes = df["Close"].tolist()
        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        return {"dates": dates, "closes": closes}
    except Exception as e:
        log.warning(f"Error fetching {ticker}: {e}")
        return _synthetic_prices(ticker, 252)


def _synthetic_prices(ticker: str, n_days: int) -> Dict[str, List]:
    """Generate synthetic price series for testing."""
    import random
    random.seed(hash(ticker) % 2**31)
    price = 100.0
    closes = []
    dates = []
    base_date = datetime.now() - timedelta(days=n_days)
    for i in range(n_days):
        ret = random.gauss(0.0003, 0.015)  # slight positive drift
        price *= (1 + ret)
        closes.append(round(price, 2))
        dates.append((base_date + timedelta(days=i)).strftime("%Y-%m-%d"))
    return {"dates": dates, "closes": closes}


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    ticker: str
    entry_date: str
    exit_date: str
    direction: str
    entry_price: float
    exit_price: float
    share_return_pct: float
    spread_return_pct: float
    spread_type: str
    spread_risk: float
    spread_reward: float


def simulate_share_trade(
    entry_price: float, exit_price: float, direction: str
) -> float:
    """Compute share return percentage."""
    if direction in ("bull", "long"):
        return (exit_price / entry_price - 1) * 100
    else:
        return (1 - exit_price / entry_price) * 100


def simulate_spread_trade(
    ticker: str,
    entry_price: float,
    exit_price: float,
    direction: str,
    hold_days: int,
    iv: float = 0.30,
    dte_at_entry: int = 30,
) -> Tuple[float, str, float, float]:
    """
    Simulate spread trade return.
    Returns (return_pct, spread_type, risk, max_reward).

    Uses BS pricing at entry and exit to compute spread P&L.
    """
    increment = determine_strike_increment(entry_price)
    atm = get_atm_strike(entry_price, increment)

    # Determine spread width based on $150 max risk
    max_risk = ACCOUNT_EQUITY * MAX_RISK_PCT
    width = min(5.0, max_risk / 100.0)
    width = max(increment, round(width / increment) * increment)

    dte_exit = max(1, dte_at_entry - hold_days)

    if direction in ("bull", "long"):
        # Bull call spread
        spread_type = "bull_call_spread"
        long_k = atm
        short_k = atm + width

        # Entry premiums
        entry_long = bs_call(entry_price, long_k, dte_at_entry / 365, 0.05, iv)
        entry_short = bs_call(entry_price, short_k, dte_at_entry / 365, 0.05, iv)
        entry_debit = entry_long - entry_short

        # Exit premiums (price moved, time decayed)
        exit_long = bs_call(exit_price, long_k, dte_exit / 365, 0.05, iv)
        exit_short = bs_call(exit_price, short_k, dte_exit / 365, 0.05, iv)
        exit_value = exit_long - exit_short

        risk = entry_debit * 100
        pnl = (exit_value - entry_debit) * 100
        max_reward = (width - entry_debit) * 100

    else:
        # Bear put spread
        spread_type = "bear_put_spread"
        long_k = atm
        short_k = atm - width

        entry_long = bs_put(entry_price, long_k, dte_at_entry / 365, 0.05, iv)
        entry_short = bs_put(entry_price, short_k, dte_at_entry / 365, 0.05, iv)
        entry_debit = entry_long - entry_short

        exit_long = bs_put(exit_price, long_k, dte_exit / 365, 0.05, iv)
        exit_short = bs_put(exit_price, short_k, dte_exit / 365, 0.05, iv)
        exit_value = exit_long - exit_short

        risk = entry_debit * 100
        pnl = (exit_value - entry_debit) * 100
        max_reward = (width - entry_debit) * 100

    return_pct = (pnl / risk * 100) if risk > 0 else 0
    return return_pct, spread_type, risk, max_reward


# ---------------------------------------------------------------------------
# Strategy signal generators
# ---------------------------------------------------------------------------

def generate_momentum_signals(
    closes: List[float], dates: List[str], lookback: int, hold: int
) -> List[Dict]:
    """Generate momentum signals: buy top performers, sell bottom."""
    signals = []
    for i in range(lookback, len(closes) - hold):
        ret = (closes[i] / closes[i - lookback]) - 1
        if ret > 0.02:  # >2% momentum
            signals.append({
                "entry_idx": i, "exit_idx": i + hold,
                "direction": "bull", "strength": ret,
            })
        elif ret < -0.02:
            signals.append({
                "entry_idx": i, "exit_idx": i + hold,
                "direction": "bear", "strength": abs(ret),
            })
    return signals


def generate_mean_reversion_signals(
    closes: List[float], dates: List[str], lookback: int, hold: int
) -> List[Dict]:
    """Generate mean-reversion signals: buy oversold, sell overbought."""
    signals = []
    for i in range(lookback, len(closes) - hold):
        window = closes[i - lookback:i]
        mean = sum(window) / len(window)
        std = (sum((x - mean) ** 2 for x in window) / len(window)) ** 0.5
        if std == 0:
            continue
        z = (closes[i] - mean) / std
        if z < -1.5:  # oversold
            signals.append({
                "entry_idx": i, "exit_idx": i + hold,
                "direction": "bull", "strength": abs(z),
            })
        elif z > 1.5:  # overbought
            signals.append({
                "entry_idx": i, "exit_idx": i + hold,
                "direction": "bear", "strength": abs(z),
            })
    return signals


def generate_neutral_signals(
    closes: List[float], dates: List[str], hold: int
) -> List[Dict]:
    """Generate periodic neutral signals for income strategies."""
    signals = []
    for i in range(0, len(closes) - hold, hold):
        signals.append({
            "entry_idx": i, "exit_idx": i + hold,
            "direction": "bull",  # slightly bullish for credit spreads
            "strength": 0.5,
        })
    return signals


# ---------------------------------------------------------------------------
# Backtester core
# ---------------------------------------------------------------------------

def backtest_strategy(name: str, config: Dict) -> Dict[str, Any]:
    """Run backtest for a single strategy, comparing shares vs spreads."""
    log.info(f"Backtesting: {name}")

    strategy_type = config["type"]
    universe = config["universe"]
    hold_days = config.get("hold_days", 5)
    lookback = config.get("lookback_days", 21)

    all_trades = []

    for ticker in universe[:5]:  # Limit to 5 tickers for speed
        data = fetch_prices(ticker)
        if not data:
            continue

        closes = data["closes"]
        dates = data["dates"]
        iv = SECTOR_IV.get(ticker, SECTOR_IV["default"])

        # Generate signals based on strategy type
        if strategy_type in ("momentum",):
            signals = generate_momentum_signals(closes, dates, lookback, hold_days)
        elif strategy_type in ("mean_reversion",):
            signals = generate_mean_reversion_signals(closes, dates, lookback, hold_days)
        else:
            signals = generate_neutral_signals(closes, dates, hold_days)

        # Simulate each signal as both share trade and spread trade
        for sig in signals[:50]:  # Cap at 50 signals per ticker
            entry_idx = sig["entry_idx"]
            exit_idx = min(sig["exit_idx"], len(closes) - 1)
            direction = sig["direction"]

            entry_p = closes[entry_idx]
            exit_p = closes[exit_idx]

            share_ret = simulate_share_trade(entry_p, exit_p, direction)
            spread_ret, spread_type, risk, reward = simulate_spread_trade(
                ticker, entry_p, exit_p, direction, hold_days, iv
            )

            trade = TradeResult(
                ticker=ticker,
                entry_date=dates[entry_idx],
                exit_date=dates[exit_idx],
                direction=direction,
                entry_price=entry_p,
                exit_price=exit_p,
                share_return_pct=round(share_ret, 2),
                spread_return_pct=round(spread_ret, 2),
                spread_type=spread_type,
                spread_risk=round(risk, 2),
                spread_reward=round(reward, 2),
            )
            all_trades.append(trade)

    if not all_trades:
        return {"strategy": name, "n_trades": 0, "error": "No trades generated"}

    # Compute summary statistics
    share_rets = [t.share_return_pct for t in all_trades]
    spread_rets = [t.spread_return_pct for t in all_trades]

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0

    def safe_std(lst):
        if len(lst) < 2:
            return 0
        m = safe_mean(lst)
        return (sum((x - m) ** 2 for x in lst) / (len(lst) - 1)) ** 0.5

    def safe_sharpe(rets):
        m = safe_mean(rets)
        s = safe_std(rets)
        return m / s if s > 0 else 0

    def win_rate(rets):
        wins = sum(1 for r in rets if r > 0)
        return wins / len(rets) * 100 if rets else 0

    def profit_factor(rets):
        gains = sum(r for r in rets if r > 0)
        losses = abs(sum(r for r in rets if r < 0))
        return gains / losses if losses > 0 else float("inf")

    result = {
        "strategy": name,
        "strategy_type": strategy_type,
        "n_trades": len(all_trades),
        "shares": {
            "avg_return": round(safe_mean(share_rets), 2),
            "std_return": round(safe_std(share_rets), 2),
            "sharpe": round(safe_sharpe(share_rets), 2),
            "win_rate": round(win_rate(share_rets), 1),
            "profit_factor": round(profit_factor(share_rets), 2),
            "total_return": round(sum(share_rets), 2),
        },
        "spreads": {
            "avg_return": round(safe_mean(spread_rets), 2),
            "std_return": round(safe_std(spread_rets), 2),
            "sharpe": round(safe_sharpe(spread_rets), 2),
            "win_rate": round(win_rate(spread_rets), 1),
            "profit_factor": round(profit_factor(spread_rets), 2),
            "total_return": round(sum(spread_rets), 2),
            "avg_risk_per_trade": round(safe_mean([t.spread_risk for t in all_trades]), 2),
        },
        "spread_vs_share": {
            "sharpe_improvement": round(
                safe_sharpe(spread_rets) - safe_sharpe(share_rets), 2
            ),
            "leverage_ratio": round(
                safe_mean(spread_rets) / safe_mean(share_rets), 2
            ) if safe_mean(share_rets) != 0 else 0,
            "risk_reduction": "Yes" if safe_std(spread_rets) < safe_std(share_rets) else "No",
        },
        "reference_sharpe": config.get("sharpe_ref"),
    }

    return result


def run_full_backtest() -> Dict[str, Any]:
    """Run backtest across all validated strategies."""
    log.info("=" * 60)
    log.info("Starting full spread backtest across all strategies")

    results = {}
    for name, config in STRATEGIES.items():
        try:
            result = backtest_strategy(name, config)
            results[name] = result
        except Exception as e:
            log.error(f"Error backtesting {name}: {e}")
            results[name] = {"strategy": name, "error": str(e)}

    # Save results
    output = {
        "generated_at": datetime.now().isoformat(),
        "n_strategies": len(results),
        "results": results,
        "summary": _build_summary(results),
    }

    path = os.path.join(STATE_DIR, "spread_backtest_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved backtest results to {path}")

    return output


def _build_summary(results: Dict) -> Dict:
    """Build summary comparing shares vs spreads across strategies."""
    share_sharpes = []
    spread_sharpes = []
    improvements = []

    for name, r in results.items():
        if "error" in r:
            continue
        share_sharpes.append(r["shares"]["sharpe"])
        spread_sharpes.append(r["spreads"]["sharpe"])
        improvements.append(r["spread_vs_share"]["sharpe_improvement"])

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0

    return {
        "avg_share_sharpe": round(safe_mean(share_sharpes), 2),
        "avg_spread_sharpe": round(safe_mean(spread_sharpes), 2),
        "avg_sharpe_improvement": round(safe_mean(improvements), 2),
        "n_strategies_improved": sum(1 for x in improvements if x > 0),
        "n_strategies_total": len(improvements),
        "recommendation": (
            "Spreads improve risk-adjusted returns on average"
            if safe_mean(improvements) > 0
            else "Spreads did not consistently improve returns — review individual strategies"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output = run_full_backtest()

    print(f"\n{'='*70}")
    print("SPREAD BACKTEST RESULTS — Shares vs Spreads")
    print(f"{'='*70}")

    for name, r in output["results"].items():
        if "error" in r and "n_trades" not in r:
            print(f"\n{name}: ERROR — {r['error']}")
            continue

        n = r.get("n_trades", 0)
        if n == 0:
            print(f"\n{name}: No trades")
            continue

        sh = r["shares"]
        sp = r["spreads"]
        vs = r["spread_vs_share"]

        print(f"\n{name} ({r['strategy_type']}, {n} trades)")
        print(f"  Shares:  Sharpe={sh['sharpe']:+.2f}  WR={sh['win_rate']:.0f}%  "
              f"PF={sh['profit_factor']:.2f}  AvgRet={sh['avg_return']:+.2f}%")
        print(f"  Spreads: Sharpe={sp['sharpe']:+.2f}  WR={sp['win_rate']:.0f}%  "
              f"PF={sp['profit_factor']:.2f}  AvgRet={sp['avg_return']:+.2f}%  "
              f"AvgRisk=${sp['avg_risk_per_trade']:.0f}")
        print(f"  Delta:   Sharpe {vs['sharpe_improvement']:+.2f}  "
              f"Leverage={vs['leverage_ratio']:.1f}x  "
              f"Risk reduction: {vs['risk_reduction']}")

    s = output["summary"]
    print(f"\n{'='*70}")
    print(f"OVERALL: Avg share Sharpe={s['avg_share_sharpe']:.2f}  "
          f"Avg spread Sharpe={s['avg_spread_sharpe']:.2f}  "
          f"Improved {s['n_strategies_improved']}/{s['n_strategies_total']}")
    print(f"Verdict: {s['recommendation']}")


if __name__ == "__main__":
    main()
