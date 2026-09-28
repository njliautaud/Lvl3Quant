#!/usr/bin/env python3
"""
BPS Liquidity & Capacity Study — How Much Capital Can BPS Actually Deploy?
==========================================================================

HC #664 R4 gap: "liquidity validation, fill quality testing, position size caps"

The current BPS backtest assumes:
  - Every limit order fills at theoretical BS price
  - No limit on contracts per name
  - Spreads are always tight (2.5% of premium)

Reality:
  - Small-cap equity options have wide bid-ask spreads (10-30% of premium)
  - Open interest / daily volume limits how many contracts you can trade
  - Fill rate on limit orders is <100% (especially at mid-price)
  - As capital grows, you run out of liquid tickers and must either concentrate or skip

This study models realistic liquidity constraints at different capital levels
($50K, $100K, $250K, $500K, $1M) and shows honest performance degradation.

Liquidity model (based on empirical options market data):
  - Large cap ($50B+): avg daily OI ~5000, bid-ask ~3-5% of premium
  - Mid cap ($10-50B): avg daily OI ~1000, bid-ask ~5-10% of premium
  - Small cap ($2-10B): avg daily OI ~200, bid-ask ~10-20% of premium
  - Micro cap (<$2B): avg daily OI ~50, bid-ask ~20-40% of premium

Position sizing rule: never take >5% of daily volume (avoid moving the market).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))
from higher_returns_study import (
    load_data, bs_price, bs_delta, strike_from_delta, trade_cost,
    compute_metrics, COST_PER_CONTRACT
)

OUTPUT = ROOT / "output" / "bps_liquidity_capacity"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ── Liquidity Model ──
# Market cap tier → (avg daily put OI for 30-delta weekly, bid-ask as fraction of premium)
LIQUIDITY_TIERS = {
    "mega":  {"mcap_min": 100e9, "avg_oi": 8000, "ba_frac": 0.03, "fill_rate": 0.85},
    "large": {"mcap_min": 50e9,  "avg_oi": 4000, "ba_frac": 0.05, "fill_rate": 0.75},
    "mid":   {"mcap_min": 10e9,  "avg_oi": 800,  "ba_frac": 0.10, "fill_rate": 0.60},
    "small": {"mcap_min": 2e9,   "avg_oi": 150,  "ba_frac": 0.18, "fill_rate": 0.45},
    "micro": {"mcap_min": 0,     "avg_oi": 30,   "ba_frac": 0.30, "fill_rate": 0.25},
}

# Approximate market cap for our universe (rough tiers based on stock price as proxy)
# In reality we'd use actual market cap data. Using price * avg volume as crude proxy.
def classify_ticker_tier(price, avg_volume=None):
    """Classify ticker into liquidity tier based on price (crude proxy for cap)."""
    # Higher price stocks tend to be larger cap, but this is rough
    # We'll use price buckets as proxy since we don't have actual cap data
    if price >= 200:
        return "mega"
    elif price >= 100:
        return "large"
    elif price >= 50:
        return "mid"
    elif price >= 20:
        return "small"
    else:
        return "micro"


def realistic_spread_cost(premium, tier_info):
    """Calculate realistic execution cost including bid-ask spread."""
    ba_frac = tier_info["ba_frac"]
    # You cross half the spread on entry and half on exit (if market order)
    # With limit orders, you save the spread but have fill risk
    # Realistic: you capture ~50% of theoretical edge due to partial fills
    half_spread = premium * ba_frac / 2
    return half_spread  # per share, one leg


def max_contracts_for_ticker(tier_info, max_oi_frac=0.05):
    """Max contracts you can trade without moving the market."""
    return max(1, int(tier_info["avg_oi"] * max_oi_frac))


def run_capacity_study(prices, iv, macro, fund, universe, earnings,
                       starting_capital=100_000.0,
                       spread_width=10.0, put_delta=0.30,
                       dte_target=7, profit_take=0.50,
                       margin_cap=0.30, max_concurrent=30,
                       per_name_pct=0.025, vix_gate=35.0,
                       dd_lookback=3, dd_threshold=-0.05,
                       # Liquidity constraints
                       apply_liquidity=True,
                       label="BPS"):
    """Run BPS backtest with realistic liquidity constraints."""

    prices_df = prices.copy()
    iv_df = iv.copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()

    macro_by_date = macro.set_index("date").to_dict("index")
    sector_of = dict(zip(fund["ticker"], fund.get("sector", pd.Series(["Unknown"]*len(fund)))))

    # Earnings
    earnings_set = {}
    for _, row in earnings.iterrows():
        tk = row["ticker"]
        ed = pd.Timestamp(row["earnings_date"])
        earnings_set.setdefault(tk, set()).add(ed)

    # SPY SMA50
    spy_sma50 = {}
    spy = prices_df[prices_df["ticker"] == "SPY"].sort_values("date")
    if len(spy) > 0:
        spy["sma50"] = spy["close"].rolling(50).mean()
        for _, row in spy.iterrows():
            spy_sma50[row["date"]] = (row["close"], row["sma50"] if pd.notna(row["sma50"]) else 0)

    all_dates = sorted(prices_df["date"].unique())

    cash = starting_capital
    positions = {}
    equity_curve = []
    ledger = []
    dd_trigger_active = 0

    # Track liquidity stats
    rejected_liquidity = 0
    rejected_fill = 0
    total_attempted = 0
    contracts_capped = 0
    ba_cost_total = 0.0

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        m = macro_by_date.get(dt, {})
        vix = m.get("vix", float("nan"))

        # ── Update positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            if T_days <= 0:
                # Expiry
                short_itm = S < pos["short_strike"]
                long_itm = S < pos["long_strike"]
                close_cost = COST_PER_CONTRACT * 2 * pos["contracts"]

                if not short_itm:
                    realized = pos["net_credit"] - close_cost
                    cash -= close_cost
                elif short_itm and not long_itm:
                    loss = (pos["short_strike"] - S) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost
                else:
                    loss = (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost

                ledger.append({"date": dt, "ticker": tk, "kind": "expire", "pnl": realized})
                to_remove.append(tk)
            else:
                # Profit take check
                short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
                long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
                spread_val = (short_val - long_val) * 100 * pos["contracts"]
                initial_credit = pos["net_credit"]
                current_cost_to_close = spread_val + trade_cost(short_val, pos["contracts"]) + trade_cost(long_val, pos["contracts"])

                # Add realistic close slippage if liquidity model active
                if apply_liquidity:
                    tier = classify_ticker_tier(S)
                    ti = LIQUIDITY_TIERS[tier]
                    close_ba = realistic_spread_cost(short_val, ti) * 100 * pos["contracts"]
                    close_ba += realistic_spread_cost(long_val, ti) * 100 * pos["contracts"]
                    current_cost_to_close += close_ba
                    ba_cost_total += close_ba

                captured = (initial_credit - current_cost_to_close) / max(initial_credit, 1e-6)
                if captured >= profit_take:
                    realized = initial_credit - current_cost_to_close
                    cash -= current_cost_to_close
                    ledger.append({"date": dt, "ticker": tk, "kind": "pt", "pnl": realized})
                    to_remove.append(tk)

        for tk in to_remove:
            del positions[tk]

        # ── MTM equity ──
        equity = cash
        for tk, pos in positions.items():
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T = max((pos["expiry"] - dt).days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
            long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
            equity -= (short_val - long_val) * 100 * pos["contracts"]

        equity_curve.append({"date": dt, "equity": equity})

        # ── Drawdown trigger ──
        if dd_lookback > 0 and len(equity_curve) >= dd_lookback + 1:
            lookback_equity = equity_curve[-(dd_lookback+1)]["equity"]
            trailing_ret = (equity - lookback_equity) / max(lookback_equity, 1)
            if trailing_ret < dd_threshold:
                dd_trigger_active = dd_lookback

        if dd_trigger_active > 0:
            dd_trigger_active -= 1
            continue

        # ── Gates ──
        if not np.isnan(vix) and vix > vix_gate:
            continue
        if dt in spy_sma50:
            spy_close, spy_sma = spy_sma50[dt]
            if spy_sma > 0 and spy_close < spy_sma:
                continue
        if len(positions) >= max_concurrent:
            continue

        current_margin = sum(
            (p["short_strike"] - p["long_strike"]) * 100 * p["contracts"]
            for p in positions.values()
        )
        if current_margin >= margin_cap * equity:
            continue

        # ── Select candidates ──
        candidates = []
        for tk in universe:
            if tk in positions or tk == "SPY":
                continue
            S = date_px.get(tk)
            if S is None or np.isnan(S) or S < 10 or S > 500:
                continue
            sigma = date_sigma.get(tk)
            if sigma is None or np.isnan(sigma) or sigma <= 0:
                continue

            # Earnings check
            if tk in earnings_set:
                e_dates = earnings_set[tk]
                check = pd.Timestamp(dt)
                expiry = check + pd.Timedelta(days=dte_target)
                if any(check - pd.Timedelta(days=2) <= pd.Timestamp(ed) <= expiry + pd.Timedelta(days=2)
                       for ed in e_dates):
                    continue

            K_short = strike_from_delta(S, dte_target / 365.0, sigma, put_delta)
            K_long = K_short - spread_width
            if K_long <= 0:
                continue

            T = dte_target / 365.0
            p_short = bs_price(S, K_short, T, sigma, kind="put")
            p_long = bs_price(S, K_long, T, sigma, kind="put")
            net_premium = p_short - p_long

            if net_premium < 0.10:
                continue

            # Liquidity filter
            tier = classify_ticker_tier(S)
            ti = LIQUIDITY_TIERS[tier]
            max_cts = max_contracts_for_ticker(ti) if apply_liquidity else 999

            # Score: premium / risk (credit efficiency)
            score = net_premium / spread_width
            candidates.append((tk, S, sigma, K_short, K_long, net_premium, score, tier, ti, max_cts))

        # Sort by score (best risk/reward first)
        candidates.sort(key=lambda x: -x[6])

        # Open positions
        for (tk, S, sigma, K_short, K_long, net_premium, score, tier, ti, max_cts) in candidates:
            if len(positions) >= max_concurrent:
                break

            current_margin = sum(
                (p["short_strike"] - p["long_strike"]) * 100 * p["contracts"]
                for p in positions.values()
            )
            if current_margin >= margin_cap * equity:
                break

            # Position sizing
            max_by_nav = int((equity * per_name_pct) / (spread_width * 100))
            max_by_margin = int((equity * margin_cap - current_margin) / (spread_width * 100))
            desired_contracts = min(max_by_nav, max_by_margin)
            desired_contracts = max(desired_contracts, 1)

            total_attempted += 1

            # Apply liquidity cap
            if apply_liquidity:
                if max_cts < 1:
                    rejected_liquidity += 1
                    continue

                if desired_contracts > max_cts:
                    contracts_capped += 1
                    desired_contracts = max_cts

                # Simulate fill probability
                if np.random.random() > ti["fill_rate"]:
                    rejected_fill += 1
                    continue

            contracts = desired_contracts

            # Calculate costs with realistic spreads
            comm = COST_PER_CONTRACT * 2 * contracts
            if apply_liquidity:
                ba_entry = realistic_spread_cost(net_premium, ti) * 100 * contracts * 2  # both legs
                ba_cost_total += ba_entry
            else:
                ba_entry = net_premium * 0.025 * 100 * contracts  # original 2.5% slippage

            total_cost = comm + ba_entry
            net_credit = net_premium * 100 * contracts - total_cost

            if net_credit <= 0:
                continue

            expiry = dt + pd.Timedelta(days=dte_target)
            positions[tk] = {
                "short_strike": K_short,
                "long_strike": K_long,
                "contracts": contracts,
                "net_credit": net_credit,
                "open_sigma": sigma,
                "expiry": expiry,
                "open_price": S,
                "tier": tier,
            }
            cash += net_credit  # net_credit already has costs subtracted

    # Compute final metrics
    eq_df = pd.DataFrame(equity_curve)
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.set_index("date")

    daily_returns = eq_df["equity"].pct_change().dropna()

    n_days = len(daily_returns)
    years = n_days / 252
    total_return = (eq_df["equity"].iloc[-1] / starting_capital - 1)
    cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    ann_ret = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg_vol = daily_returns[daily_returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / neg_vol if neg_vol > 0 else 0

    # Drawdown
    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Win rate from ledger
    trade_pnls = [t["pnl"] for t in ledger]
    n_trades = len(trade_pnls)
    n_wins = sum(1 for p in trade_pnls if p > 0)
    wr = n_wins / n_trades if n_trades > 0 else 0

    gross_win = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    results = {
        "label": label,
        "starting_capital": starting_capital,
        "liquidity_constrained": apply_liquidity,
        "final_equity": eq_df["equity"].iloc[-1],
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
        "win_rate_pct": round(wr * 100, 1),
        "profit_factor": round(pf, 2),
        "n_trades": n_trades,
        "n_days": n_days,
        "years": round(years, 1),
        "avg_pnl_per_trade": round(np.mean(trade_pnls), 2) if trade_pnls else 0,
        "liquidity_stats": {
            "total_attempted": total_attempted,
            "rejected_liquidity": rejected_liquidity,
            "rejected_fill": rejected_fill,
            "contracts_capped": contracts_capped,
            "total_ba_cost": round(ba_cost_total, 2),
            "ba_cost_pct_of_starting": round(ba_cost_total / starting_capital * 100, 2),
        },
    }

    eq_df.to_parquet(OUTPUT / f"eq_{label.replace(' ', '_')}.parquet")

    return results


def main():
    print("=" * 70)
    print("BPS LIQUIDITY & CAPACITY STUDY")
    print("HC #664 R4 — Liquidity validation")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    prices, iv, macro, fund, universe_from_data, earnings = load_data()

    # Get universe
    cache_dir = ROOT / "wheel_strategy_v1" / "data" / "cache"
    tickers = set()
    for pf in ["prices.parquet", "prices_expanded.parquet", "prices_v3_expansion.parquet"]:
        path = cache_dir / pf
        if path.exists():
            df = pd.read_parquet(path)
            if "ticker" in df.columns:
                tickers |= set(df["ticker"].unique())
    tickers -= {"SPY", "^VIX", "VIX"}
    universe = sorted(tickers)
    print(f"Universe: {len(universe)} tickers")

    # Classify universe by tier
    latest_prices = prices.sort_values("date").groupby("ticker")["close"].last()
    tier_counts = defaultdict(int)
    for tk in universe:
        if tk in latest_prices.index:
            tier = classify_ticker_tier(latest_prices[tk])
            tier_counts[tier] += 1

    print("\nUniverse by liquidity tier:")
    for tier, count in sorted(tier_counts.items(), key=lambda x: -LIQUIDITY_TIERS[x[0]]["mcap_min"]):
        ti = LIQUIDITY_TIERS[tier]
        print(f"  {tier:>6}: {count:3d} tickers | avg OI ~{ti['avg_oi']:,} | ba spread ~{ti['ba_frac']*100:.0f}% | fill rate ~{ti['fill_rate']*100:.0f}%")

    results_all = {}

    # ── Test 1: Original backtest (no liquidity constraints) ──
    print("\n" + "-" * 50)
    print("TEST 1: ORIGINAL (no liquidity constraints)")
    print("-" * 50)
    np.random.seed(42)
    r = run_capacity_study(
        prices, iv, macro, fund, universe, earnings,
        starting_capital=100_000.0,
        apply_liquidity=False,
        label="Original_100K"
    )
    print(f"  Sharpe={r['sharpe']}, CAGR={r['cagr_pct']:.0f}%, MaxDD={r['max_dd_pct']}%, WR={r['win_rate_pct']}%, PF={r['profit_factor']}, Trades={r['n_trades']}")
    results_all["original_100k"] = r

    # ── Test 2: With realistic liquidity at different capital levels ──
    capital_levels = [50_000, 100_000, 250_000, 500_000, 1_000_000]

    for cap in capital_levels:
        cap_label = f"${cap/1000:.0f}K" if cap < 1_000_000 else f"${cap/1_000_000:.0f}M"
        print(f"\n{'-'*50}")
        print(f"TEST: REALISTIC LIQUIDITY @ {cap_label}")
        print(f"{'-'*50}")

        np.random.seed(42)
        r = run_capacity_study(
            prices, iv, macro, fund, universe, earnings,
            starting_capital=float(cap),
            apply_liquidity=True,
            label=f"Realistic_{cap_label}"
        )
        print(f"  Sharpe={r['sharpe']}, CAGR={r['cagr_pct']:.0f}%, MaxDD={r['max_dd_pct']}%, WR={r['win_rate_pct']}%, PF={r['profit_factor']}, Trades={r['n_trades']}")
        liq = r["liquidity_stats"]
        print(f"  Liquidity: {liq['rejected_liquidity']} rejected (low OI), {liq['rejected_fill']} rejected (no fill), {liq['contracts_capped']} capped")
        print(f"  Total bid-ask cost: ${liq['total_ba_cost']:,.0f} ({liq['ba_cost_pct_of_starting']:.1f}% of starting)")
        results_all[f"realistic_{cap_label}"] = r

    # ── Test 3: Conservative config with realistic liquidity ──
    print(f"\n{'-'*50}")
    print(f"TEST: CONSERVATIVE + REALISTIC @ $100K")
    print(f"{'-'*50}")

    np.random.seed(42)
    r = run_capacity_study(
        prices, iv, macro, fund, universe, earnings,
        starting_capital=100_000.0,
        put_delta=0.25,  # more OTM = safer
        spread_width=15.0,  # wider spread = more protection
        margin_cap=0.15,  # lower margin = less leverage
        max_concurrent=15,
        per_name_pct=0.02,
        profit_take=0.40,
        apply_liquidity=True,
        label="Conservative_Realistic_100K"
    )
    print(f"  Sharpe={r['sharpe']}, CAGR={r['cagr_pct']:.0f}%, MaxDD={r['max_dd_pct']}%, WR={r['win_rate_pct']}%, PF={r['profit_factor']}, Trades={r['n_trades']}")
    liq = r["liquidity_stats"]
    print(f"  Liquidity: {liq['rejected_liquidity']} rejected (low OI), {liq['rejected_fill']} rejected (no fill), {liq['contracts_capped']} capped")
    results_all["conservative_realistic_100k"] = r

    # ── Summary comparison ──
    print("\n" + "=" * 70)
    print("SUMMARY — BPS CAPACITY ANALYSIS")
    print("=" * 70)
    print(f"\n{'Config':<35} {'Sharpe':>7} {'CAGR%':>8} {'MaxDD%':>8} {'WR%':>6} {'PF':>6} {'Trades':>7}")
    print("-" * 80)
    for key, r in results_all.items():
        print(f"{r['label']:<35} {r['sharpe']:>7.2f} {r['cagr_pct']:>7.0f}% {r['max_dd_pct']:>7.1f}% {r['win_rate_pct']:>5.1f}% {r['profit_factor']:>5.2f} {r['n_trades']:>7}")

    # ── Capacity verdict ──
    print("\n" + "=" * 70)
    print("CAPACITY VERDICT")
    print("=" * 70)

    realistic_results = {k: v for k, v in results_all.items() if "realistic" in k.lower() and "conservative" not in k.lower()}
    if realistic_results:
        print("\nPerformance degradation with realistic liquidity:")
        orig = results_all.get("original_100k", {})
        for key, r in sorted(realistic_results.items()):
            sharpe_loss = ((r["sharpe"] - orig.get("sharpe", r["sharpe"])) / max(abs(orig.get("sharpe", 1)), 0.01)) * 100
            print(f"  {r['label']:<30} Sharpe={r['sharpe']:.2f} (vs orig {orig.get('sharpe', 'N/A')}, Δ={sharpe_loss:+.0f}%)")

    # Save results
    with open(OUTPUT / "capacity_results.json", "w") as f:
        json.dump({
            "generated": pd.Timestamp.now().isoformat(),
            "liquidity_tiers": {k: {kk: vv for kk, vv in v.items()} for k, v in LIQUIDITY_TIERS.items()},
            "universe_tier_distribution": dict(tier_counts),
            "results": results_all,
        }, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT}/capacity_results.json")

    return results_all


if __name__ == "__main__":
    main()
