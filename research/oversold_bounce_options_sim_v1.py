"""
Oversold Bounce Options P&L Simulator v1
==========================================
Validated signal: Oversold bounce passes all 4 gates (perm p=0.01,
69% WR at 10d, PF 3.22, works in bear markets).

Now simulate ACTUAL options P&L with realistic pricing:
- Buy ATM calls when oversold bounce fires
- 10-14 DTE (matching signal's optimal horizon)
- BS pricing with sector-specific IV (VIX * sector beta)
- 15% bid-ask spread cost (conservative)
- 40% stop loss, 80% take profit on option value
- Exit at min(signal horizon, DTE-3) to avoid last-week theta
- Daily mark-to-market with realistic theta/delta/vega

Key question: Does +2.6% avg stock move at 10d overcome theta + BA costs?

Target: agentic account ($645, max $200-300/trade, HC #749)
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")

# ---------------------------------------------------------------------------
# Config variants to test
# ---------------------------------------------------------------------------
CONFIGS = [
    # (name, delta_target, dte, stop_loss, take_profit, max_hold_days)
    ("ATM_10DTE_40SL_80TP", 0.50, 10, 0.40, 0.80, 7),
    ("ATM_14DTE_40SL_80TP", 0.50, 14, 0.40, 0.80, 10),
    ("ATM_14DTE_50SL_100TP", 0.50, 14, 0.50, 1.00, 10),
    ("30D_14DTE_40SL_80TP", 0.30, 14, 0.40, 0.80, 10),
    ("ATM_21DTE_40SL_80TP", 0.50, 21, 0.40, 0.80, 14),
    ("ATM_10DTE_50SL_60TP", 0.50, 10, 0.50, 0.60, 7),  # Tighter TP
    ("ITM_14DTE_30SL_50TP", 0.70, 14, 0.30, 0.50, 10),  # Deep ITM
    ("ATM_14DTE_no_SL_100TP", 0.50, 14, 1.00, 1.00, 10),  # No stop, let it ride
]

BA_SPREAD_PCT = 0.15  # 15% bid-ask
RISK_FREE = 0.045
INITIAL_CAPITAL = 645  # Match agentic account
MAX_TRADE_SIZE = 300  # HC #749

SECTOR_BETA_IV = {
    "XLK": 1.15, "XLF": 1.05, "XLE": 1.25, "XLY": 1.15,
    "XLP": 0.70, "XLU": 0.75, "XLI": 0.95, "XLV": 0.85,
    "XLB": 1.05, "XLC": 1.10, "XLRE": 1.00,
    # Single stocks get higher IV multiplier
    "AAPL": 1.10, "MSFT": 1.05, "AMZN": 1.25, "NVDA": 1.60,
    "GOOGL": 1.15, "META": 1.35, "JPM": 1.00, "TSLA": 1.80,
    "V": 0.90, "XOM": 1.10, "MA": 0.90, "COST": 0.85,
    "HD": 1.00, "WMT": 0.75, "BAC": 1.10, "CRM": 1.20,
    "NFLX": 1.40, "AMD": 1.50, "KO": 0.65, "PEP": 0.65,
    "SPY": 0.85, "QQQ": 1.10, "IWM": 1.15,
}


# ---------------------------------------------------------------------------
# Black-Scholes with Greeks
# ---------------------------------------------------------------------------
def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0), (1.0 if S > K else 0.0), 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    # Theta per calendar day
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% vol change
    return price, delta, theta, vega


def find_strike_for_delta(S, T, r, sigma, target_delta):
    lo, hi = S * 0.70, S * 1.40
    for _ in range(60):
        mid = (lo + hi) / 2
        _, d, _, _ = bs_call(S, mid, T, r, sigma)
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return round(mid, 2)


# ---------------------------------------------------------------------------
# Indicators (same as backtest v1)
# ---------------------------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_mfi(high, low, close, volume, period=14):
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    delta = typical_price.diff()
    pos_flow = raw_money_flow.where(delta > 0, 0.0)
    neg_flow = raw_money_flow.where(delta <= 0, 0.0)
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def compute_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    cache = ROOT / "research/cache/scanner_backtest_prices_v2.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    raise FileNotFoundError("Run play_scanner_backtest_v1.py first to create price cache")


def load_vix():
    vix_path = ROOT / "wheel_strategy_v1/data/cache/vix_history.parquet"
    if vix_path.exists():
        df = pd.read_parquet(vix_path)
        df["date"] = pd.to_datetime(df["date"])
        for c in ["Close", "close"]:
            if c in df.columns:
                return df.set_index("date")[c]
    return None


# ---------------------------------------------------------------------------
# Detect oversold bounces (same logic as backtest v1)
# ---------------------------------------------------------------------------
def detect_oversold_bounces(df):
    """Detect oversold bounce setups. Returns list of (date_idx, date) tuples."""
    MIN_HIST = 252
    if len(df) < MIN_HIST:
        return []

    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)

    rsi = compute_rsi(pd.Series(close)).values
    sma_20 = pd.Series(close).rolling(20).mean().values
    sma_200 = pd.Series(close).rolling(200).mean().values
    vol_sma_20 = pd.Series(volume).rolling(20).mean().values

    signals = []
    for i in range(MIN_HIST, len(df)):
        if np.isnan(rsi[i]) or np.isnan(sma_20[i]) or np.isnan(sma_200[i]):
            continue
        if vol_sma_20[i] == 0:
            continue

        oversold = rsi[i] < 35
        below_20 = close[i] < sma_20[i]
        above_200 = close[i] > sma_200[i]
        vol_spike = volume[i] / vol_sma_20[i] > 1.5

        if oversold and below_20 and above_200 and vol_spike:
            signals.append((i, df.index[i]))

    return signals


# ---------------------------------------------------------------------------
# Simulate single option trade
# ---------------------------------------------------------------------------
def simulate_option_trade(close_series, entry_idx, vix_at_entry, ticker,
                          delta_target, dte, stop_loss_pct, take_profit_pct,
                          max_hold_days):
    """
    Simulate buying a call option on oversold bounce signal.
    Returns trade result dict.
    """
    S_entry = close_series[entry_idx]
    if S_entry <= 0 or np.isnan(S_entry):
        return None

    # IV from VIX * sector beta
    beta = SECTOR_BETA_IV.get(ticker, 1.0)
    iv = (vix_at_entry / 100) * beta
    iv = max(iv, 0.12)  # Floor at 12%

    T_entry = dte / 365

    # Find strike for target delta
    K = find_strike_for_delta(S_entry, T_entry, RISK_FREE, iv, delta_target)

    # Price the option at entry (buy at ask = mid * (1 + BA/2))
    theo_price, entry_delta, entry_theta, entry_vega = bs_call(
        S_entry, K, T_entry, RISK_FREE, iv)
    ask_price = theo_price * (1 + BA_SPREAD_PCT / 2)

    if ask_price < 0.20:  # Skip very cheap options
        return None

    cost_per_contract = ask_price * 100
    if cost_per_contract > MAX_TRADE_SIZE:
        return None  # Too expensive for agentic account

    # Daily mark-to-market
    max_idx = min(entry_idx + max_hold_days + 1, len(close_series))
    exit_reason = "max_hold"
    exit_day = max_hold_days
    exit_price = 0

    peak_option_value = ask_price
    trough_option_value = ask_price

    for day in range(1, max_idx - entry_idx):
        t_remaining = max((dte - day) / 365, 0.001)
        S_now = close_series[entry_idx + day]
        if np.isnan(S_now):
            continue

        opt_price, _, _, _ = bs_call(S_now, K, t_remaining, RISK_FREE, iv)

        peak_option_value = max(peak_option_value, opt_price)
        trough_option_value = min(trough_option_value, opt_price)

        # Check stop loss
        if opt_price <= ask_price * (1 - stop_loss_pct):
            exit_reason = "stop_loss"
            exit_day = day
            exit_price = opt_price * (1 - BA_SPREAD_PCT / 2)  # Sell at bid
            break

        # Check take profit
        if opt_price >= ask_price * (1 + take_profit_pct):
            exit_reason = "take_profit"
            exit_day = day
            exit_price = opt_price * (1 - BA_SPREAD_PCT / 2)
            break

        # Last day
        if day == max_hold_days or entry_idx + day == len(close_series) - 1:
            exit_price = opt_price * (1 - BA_SPREAD_PCT / 2)
            exit_day = day
            break

    pnl_per_contract = (exit_price - ask_price) * 100
    pnl_pct = (exit_price / ask_price - 1) * 100

    # Underlying move
    S_exit = close_series[min(entry_idx + exit_day, len(close_series) - 1)]
    underlying_move = (S_exit / S_entry - 1) * 100

    return {
        "ticker": ticker,
        "entry_price": round(S_entry, 2),
        "strike": K,
        "dte": dte,
        "iv": round(iv * 100, 1),
        "option_entry": round(ask_price, 4),
        "option_exit": round(exit_price, 4),
        "cost_per_contract": round(cost_per_contract, 2),
        "pnl_per_contract": round(pnl_per_contract, 2),
        "pnl_pct": round(pnl_pct, 1),
        "underlying_move_pct": round(underlying_move, 2),
        "exit_reason": exit_reason,
        "hold_days": exit_day,
        "entry_delta": round(entry_delta, 3),
        "peak_option_value": round(peak_option_value, 4),
        "trough_option_value": round(trough_option_value, 4),
    }


# ---------------------------------------------------------------------------
# Run simulation for one config
# ---------------------------------------------------------------------------
def run_config(config_name, delta_target, dte, stop_loss, take_profit,
               max_hold, prices, vix_series):
    """Run full simulation for a single config."""
    tickers = prices["ticker"].unique()
    all_trades = []

    for ticker in tickers:
        tdf = prices[prices["ticker"] == ticker].copy()
        if len(tdf) < 252:
            continue

        close = tdf["Close"].values.astype(float)

        # Detect signals
        signals = detect_oversold_bounces(tdf)

        for entry_idx, entry_date in signals:
            # Get VIX at entry
            vix_val = 20.0
            if vix_series is not None:
                vix_near = vix_series[vix_series.index <= entry_date]
                if len(vix_near) > 0:
                    vix_val = float(vix_near.iloc[-1])

            result = simulate_option_trade(
                close, entry_idx, vix_val, ticker,
                delta_target, dte, stop_loss, take_profit, max_hold
            )
            if result:
                result["entry_date"] = str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date)[:10]
                all_trades.append(result)

    return all_trades


# ---------------------------------------------------------------------------
# Compute config metrics
# ---------------------------------------------------------------------------
def compute_config_metrics(trades, config_name):
    """Compute performance metrics for a config."""
    if not trades:
        return {"config": config_name, "n_trades": 0}

    pnls = [t["pnl_per_contract"] for t in trades]
    pnl_pcts = [t["pnl_pct"] for t in trades]
    costs = [t["cost_per_contract"] for t in trades]

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    wr = len(winners) / len(pnls) * 100
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0
    pf = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float("inf")

    # Simulate sequential trading with $645 account
    capital = INITIAL_CAPITAL
    capital_curve = [capital]
    for t in sorted(trades, key=lambda x: x["entry_date"]):
        cost = t["cost_per_contract"]
        if cost > capital:
            continue  # Skip if we can't afford
        capital += t["pnl_per_contract"]
        capital = max(capital, 0)  # Can't go negative
        capital_curve.append(capital)

    final_capital = capital_curve[-1]
    max_capital = max(capital_curve)
    min_capital = min(capital_curve)
    max_dd = min((c - max(capital_curve[:i+1])) / max(capital_curve[:i+1])
                 for i, c in enumerate(capital_curve) if max(capital_curve[:i+1]) > 0)

    # Exit reason distribution
    exit_dist = {}
    for t in trades:
        r = t["exit_reason"]
        exit_dist[r] = exit_dist.get(r, 0) + 1

    # Affordable trades (within $300 budget)
    affordable = [t for t in trades if t["cost_per_contract"] <= MAX_TRADE_SIZE]

    return {
        "config": config_name,
        "n_trades": len(trades),
        "n_affordable": len(affordable),
        "win_rate": round(wr, 1),
        "avg_win_$": round(avg_win, 2),
        "avg_loss_$": round(avg_loss, 2),
        "avg_pnl_$": round(np.mean(pnls), 2),
        "avg_pnl_pct": round(np.mean(pnl_pcts), 1),
        "profit_factor": round(pf, 2),
        "total_pnl_$": round(sum(pnls), 2),
        "avg_cost_$": round(np.mean(costs), 2),
        "median_cost_$": round(np.median(costs), 2),
        "avg_hold_days": round(np.mean([t["hold_days"] for t in trades]), 1),
        "exit_distribution": exit_dist,
        "account_sim": {
            "start": INITIAL_CAPITAL,
            "final": round(final_capital, 2),
            "return_pct": round((final_capital / INITIAL_CAPITAL - 1) * 100, 1),
            "max_dd_pct": round(max_dd * 100, 1),
            "trades_taken": sum(1 for c in capital_curve[1:] if c != capital_curve[capital_curve.index(c)]),
        },
    }


# ---------------------------------------------------------------------------
# Permutation test for best config
# ---------------------------------------------------------------------------
def permutation_test_options(prices, vix_series, best_config, real_avg_pnl, n_perms=200):
    """Shuffle entry dates for the best config and compare option P&L."""
    delta_target, dte, stop_loss, take_profit, max_hold = best_config

    tickers = prices["ticker"].unique()
    real_entries = {}  # Count signals per ticker

    # Count real signals per ticker
    for ticker in tickers:
        tdf = prices[prices["ticker"] == ticker].copy()
        if len(tdf) < 252:
            continue
        signals = detect_oversold_bounces(tdf)
        if signals:
            real_entries[ticker] = len(signals)

    perm_avg_pnls = []
    rng = np.random.default_rng(42)

    for p in range(n_perms):
        if p % 50 == 0:
            print(f"  Perm {p}/{n_perms}...")

        perm_pnls = []
        for ticker, n_sigs in real_entries.items():
            tdf = prices[prices["ticker"] == ticker].copy()
            if len(tdf) < 252 + max_hold:
                continue

            close = tdf["Close"].values.astype(float)
            valid_range = range(252, len(close) - max_hold - 1)
            if len(valid_range) < n_sigs:
                continue

            random_entries = rng.choice(list(valid_range), size=n_sigs, replace=False)

            for entry_idx in random_entries:
                entry_date = tdf.index[entry_idx]
                vix_val = 20.0
                if vix_series is not None:
                    vix_near = vix_series[vix_series.index <= entry_date]
                    if len(vix_near) > 0:
                        vix_val = float(vix_near.iloc[-1])

                result = simulate_option_trade(
                    close, entry_idx, vix_val, ticker,
                    delta_target, dte, stop_loss, take_profit, max_hold
                )
                if result:
                    perm_pnls.append(result["pnl_per_contract"])

        if perm_pnls:
            perm_avg_pnls.append(np.mean(perm_pnls))

    p_value = np.mean([p >= real_avg_pnl for p in perm_avg_pnls]) if perm_avg_pnls else 1.0

    return {
        "real_avg_pnl": round(real_avg_pnl, 2),
        "random_avg_pnl": round(np.mean(perm_avg_pnls), 2) if perm_avg_pnls else 0,
        "random_std": round(np.std(perm_avg_pnls), 2) if perm_avg_pnls else 0,
        "p_value": round(p_value, 4),
        "pass": p_value < 0.05,
        "n_perms": n_perms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("OVERSOLD BOUNCE OPTIONS P&L SIMULATOR v1")
    print("=" * 70)
    print(f"Start: {datetime.now()}")
    print(f"Account: ${INITIAL_CAPITAL}, max ${MAX_TRADE_SIZE}/trade")
    print(f"Testing {len(CONFIGS)} configs")
    print()

    # Load data
    prices = load_data()
    vix = load_vix()
    print(f"Data: {len(prices['ticker'].unique())} tickers")
    if vix is not None:
        print(f"VIX: {len(vix)} days")

    # Run all configs
    all_results = []
    best_config = None
    best_avg_pnl = -999

    for config_name, delta, dte, sl, tp, hold in CONFIGS:
        print(f"\n{'─'*50}")
        print(f"Config: {config_name}")
        print(f"  Delta={delta}, DTE={dte}, SL={sl*100:.0f}%, TP={tp*100:.0f}%, MaxHold={hold}d")

        trades = run_config(config_name, delta, dte, sl, tp, hold, prices, vix)
        metrics = compute_config_metrics(trades, config_name)

        print(f"  Trades: {metrics['n_trades']} ({metrics.get('n_affordable', 0)} affordable)")
        print(f"  Win rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"  Avg P&L: ${metrics.get('avg_pnl_$', 0):.2f} ({metrics.get('avg_pnl_pct', 0):+.1f}%)")
        print(f"  Profit factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"  Avg cost: ${metrics.get('avg_cost_$', 0):.2f}")
        print(f"  Exit dist: {metrics.get('exit_distribution', {})}")
        if "account_sim" in metrics:
            sim = metrics["account_sim"]
            print(f"  Account: ${sim['start']} → ${sim['final']} ({sim['return_pct']:+.1f}%)")

        all_results.append(metrics)

        avg_pnl = metrics.get("avg_pnl_$", -999)
        if avg_pnl > best_avg_pnl and metrics["n_trades"] >= 10:
            best_avg_pnl = avg_pnl
            best_config = (delta, dte, sl, tp, hold)
            best_config_name = config_name

    # Permutation test on best config
    print(f"\n{'='*50}")
    print(f"BEST CONFIG: {best_config_name} (avg P&L ${best_avg_pnl:.2f})")

    if best_config and best_avg_pnl > 0:
        print(f"\nRunning permutation test...")
        perm = permutation_test_options(prices, vix, best_config, best_avg_pnl, n_perms=200)
        print(f"  Real avg P&L: ${perm['real_avg_pnl']}")
        print(f"  Random avg P&L: ${perm['random_avg_pnl']} ± ${perm['random_std']}")
        print(f"  p-value: {perm['p_value']} ({'PASS' if perm['pass'] else 'FAIL'})")
    else:
        perm = {"pass": False, "note": "Best config has negative avg P&L, skipping perm test"}
        print(f"  Best config has negative P&L — no permutation test needed.")

    # Save results
    output = {
        "strategy": "oversold_bounce_options_sim_v1",
        "timestamp": datetime.now().isoformat(),
        "account_size": INITIAL_CAPITAL,
        "max_trade_size": MAX_TRADE_SIZE,
        "ba_spread_pct": BA_SPREAD_PCT,
        "configs": all_results,
        "best_config": best_config_name if best_config else None,
        "permutation": perm,
    }

    out_path = ROOT / "research/findings/oversold_bounce_options_sim_v1_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Verdict
    print(f"\n{'='*50}")
    print("VERDICT:")

    profitable_configs = [r for r in all_results if r.get("avg_pnl_$", 0) > 0]
    print(f"  Profitable configs: {len(profitable_configs)}/{len(all_results)}")

    for r in sorted(all_results, key=lambda x: x.get("avg_pnl_$", -999), reverse=True)[:3]:
        wr = r.get("win_rate", 0)
        pnl = r.get("avg_pnl_$", 0)
        pf = r.get("profit_factor", 0)
        n = r.get("n_trades", 0)
        print(f"  {r['config']}: WR={wr:.0f}%, avg=${pnl:.2f}, PF={pf:.2f} ({n} trades)")

    if perm.get("pass", False):
        print(f"\n  ✅ OVERSOLD BOUNCE OPTIONS: VALIDATED — deploy to agentic account")
    elif best_avg_pnl > 0:
        print(f"\n  ⚠️ OVERSOLD BOUNCE OPTIONS: Positive but fails permutation")
    else:
        print(f"\n  ❌ OVERSOLD BOUNCE OPTIONS: Not profitable with options")
        print(f"     Signal is real but theta/BA costs eat the edge")

    print(f"\nDone: {datetime.now()}")


if __name__ == "__main__":
    main()
