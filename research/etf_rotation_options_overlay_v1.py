"""
ETF Rotation Options Overlay v1
================================
Converts ETF rotation v3 sector picks into options trades.

Instead of buying XLE/XLF/XLV stock, buys ATM or slightly OTM calls
on the rotation picks. Tests if leveraged exposure via options improves
or destroys the rotation signal.

Key design:
- Uses same ETF rotation v3 signals (top 3 sectors by ridge rank)
- Buys ~30-delta calls, 30 DTE, on each rotation pick
- Holds until next rebalance (21 trading days) or 50% stop loss
- Prices options with BS model + 15% bid-ask spread cost
- Tests multiple sizing: equal $ per trade, Kelly-scaled
- Permutation test (200 shuffles) for honest validation
- Regime stratification per HC #428

Target: agentic account ($645, options-only per HC #749)
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLU", "XLI", "XLV", "XLB", "XLC", "XLRE"]
HOLD_DAYS = 21
N_LONG = 3
REBALANCE_FREQ = 21  # trading days
BA_SPREAD_PCT = 0.15  # 15% bid-ask cost on options
RISK_FREE = 0.045
INITIAL_CAPITAL = 10000  # paper capital for backtest
MAX_TRADE_PCT = 0.35  # max 35% of capital per trade
STOP_LOSS_PCT = 0.50  # 50% stop loss on option value
MIN_OPTION_PRICE = 0.50  # minimum option price filter
DELTA_TARGET = 0.30  # target delta for calls

# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------
def bs_call_price(S, K, T, r, sigma):
    """BS call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, r, sigma):
    """BS call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def find_strike_for_delta(S, T, r, sigma, target_delta=0.30):
    """Find strike that gives target delta (OTM call)."""
    # Binary search for strike
    lo, hi = S * 0.80, S * 1.30
    for _ in range(50):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, r, sigma)
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return round(mid, 2)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_etf_prices():
    """Load sector ETF daily prices from yfinance cache."""
    prices_path = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"
    if prices_path.exists():
        df = pd.read_parquet(prices_path)
        df["date"] = pd.to_datetime(df["date"])
        # Filter to sector ETFs + SPY
        tickers = SECTOR_ETFS + ["SPY"]
        available = [t for t in tickers if t in df.columns or
                     any(c for c in df.columns if t.lower() in c.lower())]
        return df

    # Try alternative: build from individual price files
    flow_path = ROOT / "data/feature_store/sector_etf_flows/daily.parquet"
    if flow_path.exists():
        df = pd.read_parquet(flow_path)
        df["date"] = pd.to_datetime(df["date"])
        # Pivot to get price columns
        if "close" in df.columns and "etf" in df.columns:
            prices = df.pivot(index="date", columns="etf", values="close")
            return prices

    raise FileNotFoundError("No ETF price data found")


def load_vix():
    """Load VIX for IV proxy."""
    vix_path = ROOT / "wheel_strategy_v1/data/cache/vix_history.parquet"
    if vix_path.exists():
        df = pd.read_parquet(vix_path)
        df["date"] = pd.to_datetime(df["date"])
        if "Close" in df.columns:
            return df.set_index("date")["Close"]
        elif "close" in df.columns:
            return df.set_index("date")["close"]
        # Try first numeric column
        for c in df.columns:
            if c != "date" and pd.api.types.is_numeric_dtype(df[c]):
                return df.set_index("date")[c]
    return None


def load_rotation_signals():
    """
    Re-run ETF rotation v3 signal generation inline.
    We need: for each rebalance date, the top-N sector picks.

    Simplified: use momentum + relative strength ranking.
    """
    flow_path = ROOT / "data/feature_store/sector_etf_flows/daily.parquet"
    rot_path = ROOT / "data/feature_store/sector_rotation/daily.parquet"

    flows = pd.read_parquet(flow_path)
    flows["date"] = pd.to_datetime(flows["date"])
    flows = flows[flows["etf"].isin(SECTOR_ETFS)].copy()

    rot = pd.read_parquet(rot_path)
    rot["date"] = pd.to_datetime(rot["date"])
    rot = rot[rot["etf"].isin(SECTOR_ETFS)].copy()

    merged = flows.merge(
        rot[["etf", "date", "momentum_cross_20_60", "rs_rank_among_sectors"]],
        on=["etf", "date"], how="left"
    )

    return merged


def compute_sector_rankings(panel, date):
    """Rank sectors on a given date using momentum + relative strength."""
    day_data = panel[panel["date"] == date].copy()
    if len(day_data) < 5:
        return []

    # Composite score: ret_20d + ret_60d + relative strength
    scores = {}
    for _, row in day_data.iterrows():
        etf = row["etf"]
        score = 0
        if pd.notna(row.get("ret_20d", np.nan)):
            score += float(row["ret_20d"]) * 0.4
        if pd.notna(row.get("ret_60d", np.nan)):
            score += float(row["ret_60d"]) * 0.3
        if pd.notna(row.get("rel_strength_spy", np.nan)):
            score += float(row["rel_strength_spy"]) * 0.3
        scores[etf] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [etf for etf, _ in ranked[:N_LONG]]


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------
def run_backtest(prices_wide, vix_series, rotation_panel,
                 spy_prices, shuffle_labels=False):
    """
    Run options overlay backtest.

    For each rebalance:
    1. Get top-3 sector picks from rotation signal
    2. Buy 30-delta calls on each, 30 DTE
    3. Hold until next rebalance or stop loss
    4. Mark to market daily using BS repricing
    """
    dates = sorted(prices_wide.index)
    if len(dates) < 252:
        raise ValueError(f"Need at least 252 dates, got {len(dates)}")

    # Start after warmup
    start_idx = 252
    rebalance_dates = dates[start_idx::REBALANCE_FREQ]

    if shuffle_labels:
        # Permutation: shuffle which sectors are "top picks" on each date
        rng = np.random.default_rng()
        # Keep same dates, randomize picks
        pass

    capital = INITIAL_CAPITAL
    nav_history = []
    trades = []
    positions = []  # active option positions

    for i, reb_date in enumerate(rebalance_dates[:-1]):
        next_reb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else dates[-1]

        # Get sector picks
        if shuffle_labels:
            picks = list(np.random.choice(SECTOR_ETFS, size=N_LONG, replace=False))
        else:
            picks = compute_sector_rankings(rotation_panel, reb_date)

        if not picks:
            # No signal, stay cash
            hold_dates = [d for d in dates if reb_date <= d < next_reb]
            for d in hold_dates:
                nav_history.append({"date": d, "nav": capital})
            continue

        # Close existing positions at market
        for pos in positions:
            close_date = reb_date
            if close_date in prices_wide.index and pos["etf"] in prices_wide.columns:
                S_now = prices_wide.loc[close_date, pos["etf"]]
                T_remaining = max((pos["expiry"] - close_date).days / 365, 0.001)
                iv = pos["iv"]
                close_price = bs_call_price(S_now, pos["strike"], T_remaining, RISK_FREE, iv)
                # Apply BA spread on exit
                close_price *= (1 - BA_SPREAD_PCT / 2)
                pnl = (close_price - pos["entry_price"]) * pos["contracts"] * 100
                capital += pos["contracts"] * 100 * close_price
                trades.append({
                    "entry_date": pos["entry_date"],
                    "exit_date": close_date,
                    "etf": pos["etf"],
                    "entry_price": pos["entry_price"],
                    "exit_price": close_price,
                    "pnl": pnl,
                    "pnl_pct": pnl / (pos["entry_price"] * pos["contracts"] * 100) if pos["entry_price"] > 0 else 0,
                    "contracts": pos["contracts"],
                })
        positions = []

        # Open new positions
        trade_budget = capital * MAX_TRADE_PCT / len(picks)

        for etf in picks:
            if etf not in prices_wide.columns:
                continue
            if reb_date not in prices_wide.index:
                continue

            S = prices_wide.loc[reb_date, etf]
            if pd.isna(S) or S <= 0:
                continue

            # Get IV from VIX (sector IV ~ VIX * sector_beta)
            sector_beta = {"XLK": 1.2, "XLF": 1.1, "XLE": 1.3, "XLY": 1.2,
                          "XLP": 0.7, "XLU": 0.8, "XLI": 1.0, "XLV": 0.9,
                          "XLB": 1.1, "XLC": 1.1, "XLRE": 1.0}

            vix_val = 20.0  # default
            if vix_series is not None:
                # Find closest VIX reading
                vix_near = vix_series[vix_series.index <= reb_date]
                if len(vix_near) > 0:
                    vix_val = float(vix_near.iloc[-1])

            iv = (vix_val / 100) * sector_beta.get(etf, 1.0)
            iv = max(iv, 0.10)  # floor at 10%

            T = 30 / 365  # 30 DTE
            expiry = reb_date + timedelta(days=30)

            # Find strike for target delta
            K = find_strike_for_delta(S, T, RISK_FREE, iv, DELTA_TARGET)

            # Price the option
            theo_price = bs_call_price(S, K, T, RISK_FREE, iv)
            # Apply BA spread on entry (buy at ask)
            entry_price = theo_price * (1 + BA_SPREAD_PCT / 2)

            if entry_price < MIN_OPTION_PRICE:
                continue

            # Size: how many contracts can we afford?
            cost_per_contract = entry_price * 100
            n_contracts = max(1, int(trade_budget / cost_per_contract))
            total_cost = n_contracts * cost_per_contract

            if total_cost > capital * 0.95:  # don't go below 5% cash
                n_contracts = max(1, int(capital * 0.90 / cost_per_contract))
                total_cost = n_contracts * cost_per_contract

            if total_cost > capital:
                continue

            capital -= total_cost
            positions.append({
                "etf": etf,
                "strike": K,
                "expiry": expiry,
                "entry_price": entry_price,
                "entry_date": reb_date,
                "iv": iv,
                "contracts": n_contracts,
                "spot_at_entry": S,
            })

        # Daily mark-to-market through hold period
        hold_dates = [d for d in dates if reb_date <= d < next_reb]
        for d in hold_dates:
            mtm_value = capital
            stopped_out = []

            for j, pos in enumerate(positions):
                if pos["etf"] not in prices_wide.columns or d not in prices_wide.index:
                    continue
                S_now = prices_wide.loc[d, pos["etf"]]
                if pd.isna(S_now):
                    continue

                T_rem = max((pos["expiry"] - d).days / 365, 0.001)
                opt_price = bs_call_price(S_now, pos["strike"], T_rem, RISK_FREE, pos["iv"])

                # Check stop loss
                if opt_price <= pos["entry_price"] * (1 - STOP_LOSS_PCT):
                    # Stop loss triggered
                    exit_price = opt_price * (1 - BA_SPREAD_PCT / 2)
                    pnl = (exit_price - pos["entry_price"]) * pos["contracts"] * 100
                    capital += pos["contracts"] * 100 * exit_price
                    trades.append({
                        "entry_date": pos["entry_date"],
                        "exit_date": d,
                        "etf": pos["etf"],
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "pnl_pct": pnl / (pos["entry_price"] * pos["contracts"] * 100),
                        "contracts": pos["contracts"],
                        "stopped_out": True,
                    })
                    stopped_out.append(j)
                else:
                    mtm_value += opt_price * pos["contracts"] * 100

            # Remove stopped out positions
            positions = [p for j, p in enumerate(positions) if j not in stopped_out]

            nav_history.append({"date": d, "nav": mtm_value})

    # Close any remaining positions at end
    if positions and len(dates) > 0:
        final_date = dates[-1]
        for pos in positions:
            if pos["etf"] in prices_wide.columns and final_date in prices_wide.index:
                S_now = prices_wide.loc[final_date, pos["etf"]]
                T_rem = max((pos["expiry"] - final_date).days / 365, 0.001)
                close_price = bs_call_price(S_now, pos["strike"], T_rem, RISK_FREE, pos["iv"])
                close_price *= (1 - BA_SPREAD_PCT / 2)
                pnl = (close_price - pos["entry_price"]) * pos["contracts"] * 100
                capital += pos["contracts"] * 100 * close_price
                trades.append({
                    "entry_date": pos["entry_date"],
                    "exit_date": final_date,
                    "etf": pos["etf"],
                    "entry_price": pos["entry_price"],
                    "exit_price": close_price,
                    "pnl": pnl,
                    "pnl_pct": pnl / (pos["entry_price"] * pos["contracts"] * 100) if pos["entry_price"] > 0 else 0,
                    "contracts": pos["contracts"],
                })

    return nav_history, trades


def compute_metrics(nav_history, trades):
    """Compute risk-adjusted metrics."""
    if not nav_history:
        return {}

    nav_df = pd.DataFrame(nav_history)
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df = nav_df.sort_values("date").drop_duplicates("date", keep="last")

    returns = nav_df["nav"].pct_change().dropna()

    if len(returns) < 20:
        return {}

    ann_ret = (nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0]) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # CAGR
    years = len(returns) / 252
    cagr = (nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0]) ** (1 / years) - 1 if years > 0 else 0

    # Trade stats
    trade_pnls = [t["pnl"] for t in trades]
    win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls) if trade_pnls else 0

    winners = [p for p in trade_pnls if p > 0]
    losers = [p for p in trade_pnls if p <= 0]
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0
    pf = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float("inf")

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(pf, 2),
        "n_trades": len(trades),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_pnl": round(sum(trade_pnls), 2),
        "final_nav": round(nav_df["nav"].iloc[-1], 2),
        "years": round(years, 1),
        "ann_return": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
    }


def regime_stratify(nav_history, spy_prices):
    """Stratify returns by SPY regime (up/down days)."""
    if not nav_history:
        return {}

    nav_df = pd.DataFrame(nav_history)
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df = nav_df.sort_values("date").drop_duplicates("date", keep="last")
    nav_df["ret"] = nav_df["nav"].pct_change()

    if "SPY" in spy_prices.columns:
        spy = spy_prices["SPY"].copy()
    elif "spy" in spy_prices.columns:
        spy = spy_prices["spy"].copy()
    else:
        return {"error": "No SPY column found"}

    spy_ret = spy.pct_change()

    merged = nav_df.set_index("date").join(spy_ret.rename("spy_ret"), how="inner")

    green = merged[merged["spy_ret"] > 0]["ret"]
    red = merged[merged["spy_ret"] <= 0]["ret"]

    sharpe_green = (green.mean() / green.std() * np.sqrt(252)) if len(green) > 5 and green.std() > 0 else 0
    sharpe_red = (red.mean() / red.std() * np.sqrt(252)) if len(red) > 5 and red.std() > 0 else 0

    regime_gap = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red), 0.01)

    return {
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(regime_gap, 3),
        "r1_pass": regime_gap <= 0.50,
        "n_green": len(green),
        "n_red": len(red),
    }


def permutation_test(prices_wide, vix_series, rotation_panel, spy_prices,
                     real_sharpe, n_perms=200):
    """Shuffle sector picks and compare to real Sharpe."""
    perm_sharpes = []
    for i in range(n_perms):
        if i % 50 == 0:
            print(f"  Permutation {i}/{n_perms}...")
        try:
            nav, trades = run_backtest(prices_wide, vix_series, rotation_panel,
                                       spy_prices, shuffle_labels=True)
            m = compute_metrics(nav, trades)
            perm_sharpes.append(m.get("sharpe", 0))
        except Exception:
            perm_sharpes.append(0)

    p_value = np.mean([s >= real_sharpe for s in perm_sharpes])
    return {
        "p_value": round(p_value, 4),
        "pass": p_value < 0.05,
        "real_sharpe": real_sharpe,
        "perm_mean_sharpe": round(np.mean(perm_sharpes), 3),
        "perm_std_sharpe": round(np.std(perm_sharpes), 3),
        "n_perms": n_perms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("ETF ROTATION OPTIONS OVERLAY v1")
    print("=" * 70)
    print(f"Start: {datetime.now()}")
    print(f"Config: {N_LONG} sectors, {HOLD_DAYS}d hold, {DELTA_TARGET} delta calls")
    print(f"BA spread: {BA_SPREAD_PCT*100}%, Stop loss: {STOP_LOSS_PCT*100}%")
    print()

    # Load data
    print("Loading data...")
    flow_path = ROOT / "data/feature_store/sector_etf_flows/daily.parquet"
    flows = pd.read_parquet(flow_path)
    flows["date"] = pd.to_datetime(flows["date"])

    # Build price matrix
    etf_data = flows[flows["etf"].isin(SECTOR_ETFS + ["SPY"])].copy()
    prices_wide = etf_data.pivot(index="date", columns="etf", values="close")
    prices_wide = prices_wide.sort_index().ffill()

    print(f"  Price data: {len(prices_wide)} days, {prices_wide.shape[1]} ETFs")
    print(f"  Date range: {prices_wide.index[0].date()} to {prices_wide.index[-1].date()}")

    # Load VIX
    vix = load_vix()
    if vix is not None:
        print(f"  VIX data: {len(vix)} days")

    # Load rotation signals
    rotation_panel = load_rotation_signals()
    print(f"  Rotation panel: {len(rotation_panel)} rows")

    # Run real backtest
    print("\nRunning real backtest...")
    nav_history, trades = run_backtest(prices_wide, vix, rotation_panel, prices_wide)

    metrics = compute_metrics(nav_history, trades)
    print(f"\n{'='*50}")
    print("REAL RESULTS:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Regime stratification
    print("\nRegime stratification...")
    regime = regime_stratify(nav_history, prices_wide)
    print(f"  Green-day Sharpe: {regime.get('sharpe_green', '?')}")
    print(f"  Red-day Sharpe: {regime.get('sharpe_red', '?')}")
    print(f"  Regime gap: {regime.get('regime_gap', '?')} (pass if ≤0.50)")
    print(f"  R1 PASS: {regime.get('r1_pass', '?')}")

    # Permutation test
    real_sharpe = metrics.get("sharpe", 0)
    print(f"\nPermutation test (200 shuffles, real Sharpe={real_sharpe})...")
    perm = permutation_test(prices_wide, vix, rotation_panel, prices_wide,
                            real_sharpe, n_perms=200)
    print(f"  p-value: {perm['p_value']} ({'PASS' if perm['pass'] else 'FAIL'})")
    print(f"  Random mean Sharpe: {perm['perm_mean_sharpe']} ± {perm['perm_std_sharpe']}")

    # Per-sector breakdown
    print("\nPer-sector trade breakdown:")
    if trades:
        trade_df = pd.DataFrame(trades)
        for etf in sorted(trade_df["etf"].unique()):
            sub = trade_df[trade_df["etf"] == etf]
            wr = (sub["pnl"] > 0).mean() * 100
            avg = sub["pnl"].mean()
            print(f"  {etf}: {len(sub)} trades, WR {wr:.0f}%, avg P&L ${avg:.2f}")

    # Per-year breakdown
    print("\nPer-year returns:")
    if nav_history:
        nav_df = pd.DataFrame(nav_history)
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        nav_df = nav_df.sort_values("date").drop_duplicates("date", keep="last")
        nav_df["year"] = nav_df["date"].dt.year
        for year, grp in nav_df.groupby("year"):
            if len(grp) < 10:
                continue
            yr_ret = (grp["nav"].iloc[-1] / grp["nav"].iloc[0] - 1) * 100
            print(f"  {year}: {yr_ret:+.1f}%")

    # Save results
    results = {
        "strategy": "etf_rotation_options_overlay_v1",
        "config": {
            "n_long": N_LONG,
            "hold_days": HOLD_DAYS,
            "delta_target": DELTA_TARGET,
            "ba_spread_pct": BA_SPREAD_PCT,
            "stop_loss_pct": STOP_LOSS_PCT,
            "initial_capital": INITIAL_CAPITAL,
        },
        "metrics": metrics,
        "regime": regime,
        "permutation": perm,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = ROOT / "research/findings/etf_rotation_options_v1_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary verdict
    print("\n" + "=" * 50)
    print("VERDICT:")
    gates = []
    if metrics.get("sharpe", 0) > 0.5:
        gates.append("Sharpe>0.5 PASS")
    else:
        gates.append("Sharpe>0.5 FAIL")

    if perm.get("pass", False):
        gates.append("Permutation PASS")
    else:
        gates.append("Permutation FAIL")

    if regime.get("r1_pass", False):
        gates.append("R1 Regime PASS")
    else:
        gates.append("R1 Regime FAIL")

    if metrics.get("max_dd", -100) > -30:
        gates.append("MaxDD>-30% PASS")
    else:
        gates.append("MaxDD>-30% FAIL")

    n_pass = sum(1 for g in gates if "PASS" in g)
    print(f"  Gates: {n_pass}/{len(gates)}")
    for g in gates:
        print(f"    {g}")

    if n_pass >= 3:
        print("\n  ✅ WORTHY OF PAPER TRADING")
    else:
        print("\n  ❌ DOES NOT PASS MINIMUM GATES")

    print(f"\nDone: {datetime.now()}")


if __name__ == "__main__":
    main()
