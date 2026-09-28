"""
Walk-Forward Validation of Wheel Strategy Ticker Selection
-----------------------------------------------------------
Tests whether selecting the top-20 tickers by in-sample Sharpe
produces meaningfully better out-of-sample performance than
random 20-ticker baskets.

Simulates simplified weekly CSP (cash-secured put, 30-delta)
for each ticker, then runs the WF test.
"""

import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = "/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache"
OUT_DIR = "/home/jupiter/Lvl3Quant/output/wf_ticker_selection"

IS_START = pd.Timestamp("2019-01-01")
IS_END = pd.Timestamp("2023-12-31")
OOS_START = pd.Timestamp("2024-01-01")
OOS_END = pd.Timestamp("2025-12-31")

TOP_N = 20
N_RANDOM = 50
MIN_IS_WEEKS = 200
R_FREE = 0.04
DIV_YIELD = 0.0
COMMISSION_PER_CONTRACT = 0.65  # dollars
SLIPPAGE_PCT = 0.025  # 2.5% of premium
WEEKS_PER_YEAR = 52
SEED = 42


# ── Black-Scholes helpers ──────────────────────────────────────────────
def bs_put_price(S, K, T, sigma, r=R_FREE, q=DIV_YIELD):
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_put_delta(S, K, T, sigma, r=R_FREE, q=DIV_YIELD):
    """Black-Scholes put delta (negative)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * (norm.cdf(d1) - 1)


def find_30delta_strike(S, T, sigma, r=R_FREE, q=DIV_YIELD, target_delta=-0.30):
    """Find strike K such that put delta ~ target_delta via bisection."""
    if sigma <= 0 or S <= 0:
        return S * 0.95  # fallback
    K_lo = S * 0.70
    K_hi = S * 1.05
    for _ in range(50):
        K_mid = (K_lo + K_hi) / 2
        delta = bs_put_delta(S, K_mid, T, sigma, r, q)
        if delta < target_delta:  # delta more negative => K too high
            K_hi = K_mid
        else:
            K_lo = K_mid
    return (K_lo + K_hi) / 2


# ── Data loading ────────────────────────────────────────────────────────
def load_data():
    """Load IV features and prices, merge on (date, ticker)."""
    iv = pd.read_parquet(os.path.join(DATA_DIR, "iv_features_modeled.parquet"))
    prices = pd.read_parquet(os.path.join(DATA_DIR, "prices.parquet"))

    # Merge
    df = prices[["date", "ticker", "close"]].merge(
        iv[["date", "ticker", "sigma_atm_30d", "sigma", "sigma_rv"]],
        on=["date", "ticker"],
        how="inner",
    )

    # Best available IV: sigma_atm_30d > sigma > sigma_rv
    df["iv"] = df["sigma_atm_30d"].fillna(df["sigma"]).fillna(df["sigma_rv"])
    df = df.dropna(subset=["iv", "close"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ── Weekly CSP simulation per ticker ────────────────────────────────────
def simulate_weekly_csp(ticker_df):
    """
    Simulate selling a 30-delta put each Friday (or last trading day of week).
    Returns a DataFrame of weekly returns.
    """
    ticker_df = ticker_df.copy()
    ticker_df["week"] = ticker_df["date"].dt.isocalendar().week.astype(int)
    ticker_df["year"] = ticker_df["date"].dt.year

    # Get last trading day of each week
    weekly = ticker_df.groupby(["year", "week"]).last().reset_index()
    weekly = weekly.sort_values("date").reset_index(drop=True)

    results = []
    T = 7 / 365  # 1-week expiry

    for i in range(len(weekly) - 1):
        row = weekly.iloc[i]
        next_row = weekly.iloc[i + 1]

        S_entry = row["close"]
        iv = row["iv"]
        S_expiry = next_row["close"]

        if iv <= 0 or S_entry <= 0:
            continue

        # Find 30-delta strike
        K = find_30delta_strike(S_entry, T, iv)

        # Put premium received
        premium = bs_put_price(S_entry, K, T, iv)
        if premium <= 0:
            continue

        # Apply slippage to premium (we receive less)
        premium_net = premium * (1 - SLIPPAGE_PCT)

        # Commission as fraction of stock price
        commission = COMMISSION_PER_CONTRACT / (S_entry * 100)  # per-share basis

        # P&L at expiry
        if S_expiry >= K:
            # OTM: keep premium
            pnl_per_share = premium_net - commission
        else:
            # ITM: assigned, lose (K - S_expiry), keep premium
            pnl_per_share = premium_net - (K - S_expiry) - commission

        # Return = pnl / notional (cash-secured = K per share)
        notional = K
        weekly_return = pnl_per_share / notional

        results.append({
            "date": next_row["date"],  # date of expiry/settlement
            "weekly_return": weekly_return,
            "premium": premium_net,
            "pnl": pnl_per_share,
            "S_entry": S_entry,
            "K": K,
            "S_expiry": S_expiry,
            "iv": iv,
        })

    return pd.DataFrame(results)


def compute_sharpe(returns, annualize=True):
    """Annualized Sharpe from weekly returns."""
    if len(returns) < 10:
        return np.nan
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0:
        return 0.0
    sharpe = mu / sigma
    if annualize:
        sharpe *= np.sqrt(WEEKS_PER_YEAR)
    return sharpe


def compute_sortino(returns, annualize=True):
    """Annualized Sortino from weekly returns."""
    if len(returns) < 10:
        return np.nan
    mu = returns.mean()
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.inf
    down_std = np.sqrt((downside**2).mean())
    if down_std == 0:
        return 0.0
    sortino = mu / down_std
    if annualize:
        sortino *= np.sqrt(WEEKS_PER_YEAR)
    return sortino


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = load_data()
    print(f"  {df['ticker'].nunique()} tickers, {len(df):,} rows")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print()

    # Simulate CSP for each ticker
    tickers = sorted(df["ticker"].unique())
    print(f"Simulating weekly CSP for {len(tickers)} tickers...")

    ticker_weekly = {}
    for t in tickers:
        tdf = df[df["ticker"] == t]
        wdf = simulate_weekly_csp(tdf)
        if len(wdf) > 0:
            ticker_weekly[t] = wdf

    print(f"  {len(ticker_weekly)} tickers with valid CSP simulations")
    print()

    # Split IS / OOS
    ticker_stats = {}
    for t, wdf in ticker_weekly.items():
        is_mask = (wdf["date"] >= IS_START) & (wdf["date"] <= IS_END)
        oos_mask = (wdf["date"] >= OOS_START) & (wdf["date"] <= OOS_END)

        is_returns = wdf.loc[is_mask, "weekly_return"]
        oos_returns = wdf.loc[oos_mask, "weekly_return"]

        if len(is_returns) < MIN_IS_WEEKS:
            continue

        ticker_stats[t] = {
            "is_sharpe": compute_sharpe(is_returns),
            "is_sortino": compute_sortino(is_returns),
            "is_mean_ret": float(is_returns.mean()),
            "is_weeks": len(is_returns),
            "oos_sharpe": compute_sharpe(oos_returns) if len(oos_returns) >= 10 else np.nan,
            "oos_sortino": compute_sortino(oos_returns) if len(oos_returns) >= 10 else np.nan,
            "oos_mean_ret": float(oos_returns.mean()) if len(oos_returns) > 0 else np.nan,
            "oos_weeks": len(oos_returns),
            "is_returns": is_returns.values.tolist(),
            "oos_returns": oos_returns.values.tolist(),
        }

    eligible = {t: s for t, s in ticker_stats.items()
                if not np.isnan(s["is_sharpe"]) and not np.isnan(s["oos_sharpe"])}

    print(f"  {len(eligible)} tickers with sufficient IS ({MIN_IS_WEEKS}+ weeks) and OOS data")
    print()

    if len(eligible) < TOP_N:
        print(f"ERROR: Only {len(eligible)} eligible tickers, need at least {TOP_N}")
        return

    # ── Select top-20 by IS Sharpe ──────────────────────────────────────
    ranked = sorted(eligible.items(), key=lambda x: x[1]["is_sharpe"], reverse=True)
    selected_tickers = [t for t, _ in ranked[:TOP_N]]
    remaining_tickers = [t for t, _ in ranked[TOP_N:]]
    all_eligible_tickers = [t for t, _ in ranked]

    # Compute IS-selected basket OOS Sharpe (equal-weight portfolio)
    selected_oos_returns = []
    for t in selected_tickers:
        oos_ret = np.array(eligible[t]["oos_returns"])
        selected_oos_returns.append(oos_ret)

    # Align by taking the minimum length and averaging
    min_oos_len = min(len(r) for r in selected_oos_returns)
    selected_oos_matrix = np.column_stack([r[:min_oos_len] for r in selected_oos_returns])
    selected_portfolio_returns = selected_oos_matrix.mean(axis=1)
    selected_oos_sharpe = compute_sharpe(selected_portfolio_returns)
    selected_oos_sortino = compute_sortino(selected_portfolio_returns)
    selected_oos_total_ret = float(np.sum(selected_portfolio_returns))

    # IS portfolio Sharpe for the selected basket
    selected_is_returns_list = []
    for t in selected_tickers:
        is_ret = np.array(eligible[t]["is_returns"])
        selected_is_returns_list.append(is_ret)
    min_is_len = min(len(r) for r in selected_is_returns_list)
    selected_is_matrix = np.column_stack([r[:min_is_len] for r in selected_is_returns_list])
    selected_is_portfolio = selected_is_matrix.mean(axis=1)
    selected_is_sharpe = compute_sharpe(selected_is_portfolio)

    # ── Random baskets ──────────────────────────────────────────────────
    rng = np.random.RandomState(SEED)
    random_oos_sharpes = []

    for _ in range(N_RANDOM):
        rand_tickers = rng.choice(all_eligible_tickers, size=TOP_N, replace=False)
        rand_returns = []
        for t in rand_tickers:
            oos_ret = np.array(eligible[t]["oos_returns"])
            rand_returns.append(oos_ret)

        min_len = min(len(r) for r in rand_returns)
        rand_matrix = np.column_stack([r[:min_len] for r in rand_returns])
        rand_portfolio = rand_matrix.mean(axis=1)
        random_oos_sharpes.append(compute_sharpe(rand_portfolio))

    random_oos_sharpes = np.array(random_oos_sharpes)

    # ── Statistics ──────────────────────────────────────────────────────
    pct_rank = float(np.mean(random_oos_sharpes < selected_oos_sharpe) * 100)
    p_value = float(np.mean(random_oos_sharpes >= selected_oos_sharpe))

    # Sharpe decay (IS to OOS)
    sharpe_decay = (selected_is_sharpe - selected_oos_sharpe) / abs(selected_is_sharpe) if selected_is_sharpe != 0 else np.nan

    # ── Per-ticker detail ───────────────────────────────────────────────
    selected_detail = []
    for t in selected_tickers:
        s = eligible[t]
        selected_detail.append({
            "ticker": t,
            "is_sharpe": round(s["is_sharpe"], 3),
            "oos_sharpe": round(s["oos_sharpe"], 3),
            "is_sortino": round(s["is_sortino"], 3),
            "oos_sortino": round(s["oos_sortino"], 3),
            "is_weeks": s["is_weeks"],
            "oos_weeks": s["oos_weeks"],
        })

    # ── Print results ───────────────────────────────────────────────────
    print("=" * 70)
    print("WALK-FORWARD VALIDATION: WHEEL STRATEGY TICKER SELECTION")
    print("=" * 70)
    print()
    print(f"Universe: {len(eligible)} eligible tickers (>={MIN_IS_WEEKS} IS weeks + OOS data)")
    print(f"In-Sample:  {IS_START.date()} to {IS_END.date()}")
    print(f"Out-of-Sample: {OOS_START.date()} to {OOS_END.date()}")
    print()

    print("── IS-SELECTED TOP-20 BASKET ──")
    print(f"Tickers: {', '.join(selected_tickers)}")
    print()
    print(f"{'Ticker':<8} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'IS Sortino':>11} {'OOS Sortino':>12}")
    print("-" * 54)
    for d in selected_detail:
        print(f"{d['ticker']:<8} {d['is_sharpe']:>10.3f} {d['oos_sharpe']:>11.3f} {d['is_sortino']:>11.3f} {d['oos_sortino']:>12.3f}")
    print()

    print("── PORTFOLIO-LEVEL RESULTS ──")
    print(f"IS-selected basket (equal-weight):")
    print(f"  IS  Sharpe:  {selected_is_sharpe:.3f}")
    print(f"  OOS Sharpe:  {selected_oos_sharpe:.3f}")
    print(f"  OOS Sortino: {selected_oos_sortino:.3f}")
    print(f"  Sharpe decay (IS->OOS): {sharpe_decay:.1%}")
    print(f"  OOS cumulative return:  {selected_oos_total_ret:.1%}")
    print()

    print("── RANDOM BASKET COMPARISON (50 random baskets of 20) ──")
    print(f"  Mean OOS Sharpe: {random_oos_sharpes.mean():.3f} +/- {random_oos_sharpes.std():.3f}")
    print(f"  Min OOS Sharpe:  {random_oos_sharpes.min():.3f}")
    print(f"  Max OOS Sharpe:  {random_oos_sharpes.max():.3f}")
    print(f"  Median:          {np.median(random_oos_sharpes):.3f}")
    print()

    print("── VERDICT ──")
    print(f"  IS-selected basket OOS Sharpe: {selected_oos_sharpe:.3f}")
    print(f"  Percentile rank vs random:     {pct_rank:.1f}th percentile")
    print(f"  p-value (random >= selected):  {p_value:.3f}")
    print()

    if p_value < 0.05:
        verdict = "PASS: IS selection produces statistically significant OOS outperformance."
    elif p_value < 0.20:
        verdict = "MARGINAL: Some evidence of selection skill, but not statistically significant."
    else:
        verdict = "FAIL: IS ticker selection does NOT produce meaningful OOS outperformance. Likely forward-looking bias."

    print(f"  >> {verdict}")
    print()

    # Bottom / worst tickers for reference
    bottom_5 = ranked[-5:]
    print("── WORST 5 TICKERS BY IS SHARPE (for reference) ──")
    for t, s in bottom_5:
        print(f"  {t:<8} IS={s['is_sharpe']:.3f}  OOS={s['oos_sharpe']:.3f}")
    print()

    # ── Save results ────────────────────────────────────────────────────
    results = {
        "run_date": datetime.now().isoformat(),
        "config": {
            "is_period": f"{IS_START.date()} to {IS_END.date()}",
            "oos_period": f"{OOS_START.date()} to {OOS_END.date()}",
            "top_n": TOP_N,
            "n_random_baskets": N_RANDOM,
            "min_is_weeks": MIN_IS_WEEKS,
            "r_free": R_FREE,
            "commission": COMMISSION_PER_CONTRACT,
            "slippage_pct": SLIPPAGE_PCT,
        },
        "universe_size": len(eligible),
        "selected_basket": {
            "tickers": selected_tickers,
            "is_portfolio_sharpe": round(selected_is_sharpe, 4),
            "oos_portfolio_sharpe": round(selected_oos_sharpe, 4),
            "oos_portfolio_sortino": round(selected_oos_sortino, 4),
            "sharpe_decay_pct": round(sharpe_decay * 100, 1),
            "oos_cumulative_return_pct": round(selected_oos_total_ret * 100, 2),
            "ticker_detail": selected_detail,
        },
        "random_baskets": {
            "n": N_RANDOM,
            "oos_sharpe_mean": round(float(random_oos_sharpes.mean()), 4),
            "oos_sharpe_std": round(float(random_oos_sharpes.std()), 4),
            "oos_sharpe_min": round(float(random_oos_sharpes.min()), 4),
            "oos_sharpe_max": round(float(random_oos_sharpes.max()), 4),
            "oos_sharpe_median": round(float(np.median(random_oos_sharpes)), 4),
        },
        "test_result": {
            "percentile_rank": round(pct_rank, 1),
            "p_value": round(p_value, 3),
            "verdict": verdict,
        },
        "all_ticker_stats": {
            t: {
                "is_sharpe": round(s["is_sharpe"], 4),
                "oos_sharpe": round(s["oos_sharpe"], 4),
                "is_sortino": round(s["is_sortino"], 4),
                "oos_sortino": round(s["oos_sortino"], 4),
                "is_weeks": s["is_weeks"],
                "oos_weeks": s["oos_weeks"],
            }
            for t, s in eligible.items()
        },
    }

    out_path = os.path.join(OUT_DIR, "wf_ticker_selection_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
