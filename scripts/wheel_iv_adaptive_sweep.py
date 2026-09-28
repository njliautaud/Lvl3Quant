#!/usr/bin/env python3
"""
wheel_iv_adaptive_sweep.py — IV-Rank Adaptive Delta Selection

HYPOTHESIS: Instead of a fixed put delta (e.g., 0.30), dynamically choose delta
based on each ticker's IV rank (current realized vol vs its own 1-year history):

  - High IV rank (>75th pctile): sell closer to ATM (25-delta) → richer premiums
  - Normal IV rank (25-75th):    standard (30-delta)
  - Low IV rank (<25th):          further OTM (35-delta) → more protection

Also tests VIX-adaptive overlay: when VIX > 25, bump delta 5pts closer to ATM
across the board (vol crush = premium opportunity).

Compares against fixed-delta baselines to quantify the improvement.
"""
import sys
import json
import time
import math
import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from dataclasses import dataclass

sys.path.insert(0, str(Path("/home/jupiter/Lvl3Quant/scripts")))
from wheel_universe_v3_expand import load_all_data, compute_metrics, regime_analysis

ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = ROOT / "output" / "wheel_iv_adaptive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [IV-ADAPT] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger('IV-ADAPT')

# ========================= BS HELPERS =========================================

RISK_FREE = 0.04

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
    if T <= 0 or sigma <= 0:
        return S
    if kind == "put":
        lo, hi = S * 0.3, S * 1.0
    else:
        lo, hi = S * 1.0, S * 2.0
    for _ in range(60):
        K = (lo + hi) / 2
        d1 = (math.log(S / K) + (RISK_FREE + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T) + 1e-9)
        if kind == "put":
            delta_abs = _Phi(-d1)
        else:
            delta_abs = _Phi(d1)
        if delta_abs > delta_target:
            if kind == "put":
                hi = K
            else:
                lo = K
        else:
            if kind == "put":
                lo = K
            else:
                hi = K
    return round(K * 2) / 2


# ========================= IV RANK COMPUTATION ================================

def compute_iv_rank(prices_df, lookback=252):
    """
    For each ticker-date, compute the IV rank: percentile of current 20-day
    realized vol relative to the past `lookback` trading days of realized vol.

    IV rank = (current_vol - min_vol) / (max_vol - min_vol) over lookback window.
    """
    log.info(f"Computing IV rank (lookback={lookback} days)...")

    def _rank(group):
        vol = group["sigma"]
        roll_min = vol.rolling(lookback, min_periods=60).min()
        roll_max = vol.rolling(lookback, min_periods=60).max()
        rank = (vol - roll_min) / (roll_max - roll_min + 1e-9)
        return rank.clip(0, 1)

    prices_df["iv_rank"] = prices_df.groupby("ticker", group_keys=False).apply(
        lambda g: _rank(g)
    )
    prices_df["iv_rank"] = prices_df["iv_rank"].fillna(0.5)

    log.info(f"IV rank computed. Mean={prices_df['iv_rank'].mean():.3f}, "
             f"Std={prices_df['iv_rank'].std():.3f}")
    return prices_df


# ========================= ADAPTIVE DELTA STRATEGIES ==========================

def fixed_delta_strategy(iv_rank, vix, base_delta):
    """Baseline: always use fixed delta regardless of IV."""
    return base_delta

def iv_rank_3tier(iv_rank, vix, high_delta=0.25, mid_delta=0.30, low_delta=0.35,
                  high_threshold=0.75, low_threshold=0.25):
    """
    3-tier IV rank adaptive:
    - IV rank > 0.75 → sell 25-delta (closer to ATM, richer premiums)
    - IV rank 0.25-0.75 → sell 30-delta (standard)
    - IV rank < 0.25 → sell 35-delta (further OTM, cheaper)
    """
    if iv_rank > high_threshold:
        return high_delta
    elif iv_rank < low_threshold:
        return low_delta
    return mid_delta

def iv_rank_continuous(iv_rank, vix, min_delta=0.20, max_delta=0.40):
    """
    Continuous interpolation: delta = max_delta - iv_rank * (max_delta - min_delta)
    Higher IV rank → lower delta (closer to ATM) → richer premiums.
    """
    return max_delta - iv_rank * (max_delta - min_delta)

def vix_overlay(iv_rank, vix, base_func, vix_threshold=25, vix_bump=0.05):
    """
    VIX overlay: when VIX > threshold, bump delta closer to ATM by `vix_bump`.
    Applied on top of any base strategy.
    """
    delta = base_func(iv_rank, vix)
    if vix > vix_threshold:
        delta = max(0.15, delta - vix_bump)  # Closer to ATM
    return delta

def iv_rank_5tier(iv_rank, vix):
    """
    5-tier for finer granularity:
    >0.80 → 0.22 (aggressive, very rich premiums)
    0.60-0.80 → 0.27
    0.40-0.60 → 0.30
    0.20-0.40 → 0.33
    <0.20 → 0.38 (conservative, cheap premiums)
    """
    if iv_rank > 0.80:
        return 0.22
    elif iv_rank > 0.60:
        return 0.27
    elif iv_rank > 0.40:
        return 0.30
    elif iv_rank > 0.20:
        return 0.33
    return 0.38

def iv_rank_aggressive(iv_rank, vix):
    """
    Aggressive variant: wider swings in delta.
    >0.75 → 0.20 (nearly ATM)
    0.25-0.75 → 0.30
    <0.25 → 0.40 (far OTM, barely any premium)
    """
    if iv_rank > 0.75:
        return 0.20
    elif iv_rank < 0.25:
        return 0.40
    return 0.30


# ========================= PORTFOLIO ENGINE (ADAPTIVE) ========================

def run_adaptive_portfolio(prices_df, spy_regime, sector_map,
                          delta_func, delta_func_name="custom",
                          starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
                          dte_target=14, profit_take=0.65,
                          bear_mode="liq_csp_only", max_assignments_5d=3,
                          max_share_positions=5, loss_cut_pct=-0.15,
                          min_price=10.0, max_price=500.0):
    """
    Same as run_portfolio_v3 but with dynamic delta selection via delta_func.
    delta_func(iv_rank, vix) -> put_delta for that ticker-date.
    """
    MARGIN_REQ_PCT = 0.20
    COST_PER_CONTRACT = 0.65
    SLIPPAGE_FRAC = 0.025
    SLIPPAGE_MIN = 0.03
    VIX_MAX = 35.0
    CALL_DELTA = 0.30

    # Filter to stocks in price range
    valid_tickers = set()
    for t, grp in prices_df.groupby("ticker"):
        median_price = grp["close"].median()
        if min_price <= median_price <= max_price:
            valid_tickers.add(t)

    pdf = prices_df[prices_df["ticker"].isin(valid_tickers)].copy()
    tickers_available = sorted(pdf["ticker"].unique())

    # Build date-indexed lookups
    ticker_data = {}
    for t in tickers_available:
        tdf = pdf[pdf["ticker"] == t].set_index("date").sort_index()
        ticker_data[t] = tdf

    spy_map = {}
    if not spy_regime.empty:
        for _, row in spy_regime.iterrows():
            spy_map[pd.Timestamp(row["date"])] = int(row["bear"])

    all_dates = sorted(pd.Timestamp(d) for d in pdf["date"].unique())

    # State
    cash = float(starting_cash)
    csp_positions = {}
    share_positions = {}
    assignment_dates = deque()

    trades = []
    daily_equity = []
    delta_log = []  # Track what deltas were actually used

    @dataclass
    class CSPPos:
        ticker: str
        strike: float
        premium: float
        entry_date: object
        expiry_date: object
        margin_held: float
        put_delta_used: float = 0.30

    @dataclass
    class SharePos:
        ticker: str
        shares: int
        cost_basis: float
        entry_date: object
        cc_strike: float = 0.0
        cc_premium: float = 0.0
        cc_expiry: object = None
        has_cc: bool = False

    for di, date in enumerate(all_dates):
        is_bear = spy_map.get(date, 0) == 1

        # Mark to market
        nav = cash
        for pos in csp_positions.values():
            nav += pos.margin_held
        for t, pos in share_positions.items():
            if t in ticker_data and date in ticker_data[t].index:
                px = ticker_data[t].loc[date]["close"]
                nav += pos.shares * px
            else:
                nav += pos.shares * pos.cost_basis

        daily_equity.append({"date": date, "equity": nav})

        # Process CSP expirations
        expired_csps = [t for t, pos in csp_positions.items() if date >= pos.expiry_date]
        for t in expired_csps:
            pos = csp_positions[t]
            if t not in ticker_data or date not in ticker_data[t].index:
                cash += pos.margin_held
                trades.append({"date": date, "ticker": t, "action": "csp_expired_otm",
                              "pnl": pos.premium * 100})
                del csp_positions[t]
                continue

            px = ticker_data[t].loc[date]["close"]
            if px <= pos.strike:
                while assignment_dates and (date - assignment_dates[0]).days > 5:
                    assignment_dates.popleft()
                if (len(assignment_dates) >= max_assignments_5d or
                    len(share_positions) >= max_share_positions):
                    intrinsic = (pos.strike - px) * 100
                    loss = intrinsic - pos.premium * 100 + COST_PER_CONTRACT
                    cash += pos.margin_held
                    cash -= loss
                    trades.append({"date": date, "ticker": t, "action": "assignment_refused",
                                  "pnl": -(loss / 100)})
                    del csp_positions[t]
                    continue

                share_cost = pos.strike * 100 + COST_PER_CONTRACT
                cash += pos.margin_held
                cash -= share_cost
                share_positions[t] = SharePos(
                    ticker=t, shares=100,
                    cost_basis=pos.strike - pos.premium,
                    entry_date=date,
                )
                assignment_dates.append(date)
                trades.append({"date": date, "ticker": t, "action": "assigned"})
                del csp_positions[t]
            else:
                cash += pos.margin_held
                trades.append({"date": date, "ticker": t, "action": "csp_expired_otm",
                              "pnl": pos.premium})
                del csp_positions[t]

        # Process CC expirations
        for t in list(share_positions.keys()):
            pos = share_positions[t]
            if not pos.has_cc or pos.cc_expiry is None or date < pos.cc_expiry:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                pos.has_cc = False
                continue
            px = ticker_data[t].loc[date]["close"]
            if px >= pos.cc_strike:
                proceeds = pos.cc_strike * 100 - COST_PER_CONTRACT
                cash += proceeds
                pnl = (pos.cc_strike - pos.cost_basis) * 100 + pos.cc_premium * 100
                trades.append({"date": date, "ticker": t, "action": "called_away", "pnl": pnl / 100})
                del share_positions[t]
            else:
                pos.cost_basis -= pos.cc_premium
                pos.has_cc = False
                trades.append({"date": date, "ticker": t, "action": "cc_expired_otm"})

        # Loss-cut on shares
        for t in list(share_positions.keys()):
            pos = share_positions[t]
            if pos.has_cc:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                continue
            px = ticker_data[t].loc[date]["close"]
            pnl_pct = (px - pos.cost_basis) / pos.cost_basis if pos.cost_basis > 0 else 0
            if pnl_pct <= loss_cut_pct:
                proceeds = px * 100 - COST_PER_CONTRACT
                cash += proceeds
                realized = (px - pos.cost_basis) * 100
                trades.append({"date": date, "ticker": t, "action": "loss_cut", "pnl": realized / 100})
                del share_positions[t]

        # Profit-take on CSPs
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
            if current_val <= pos.premium * (1 - profit_take):
                buyback = current_val * 100 + COST_PER_CONTRACT
                cash += pos.margin_held
                cash -= buyback
                profit = (pos.premium - current_val) * 100 - 2 * COST_PER_CONTRACT
                trades.append({"date": date, "ticker": t, "action": "profit_take", "pnl": profit / 100})
                del csp_positions[t]

        # Bear protection
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
                buyback = current_val * 100 + COST_PER_CONTRACT
                cash += pos.margin_held
                cash -= buyback
                pnl = (pos.premium - current_val) * 100 - 2 * COST_PER_CONTRACT
                trades.append({"date": date, "ticker": t, "action": "bear_close", "pnl": pnl / 100})
                del csp_positions[t]

        # Write CCs on shares
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
            T = dte_target / 365
            K = find_strike(px, sigma, T, CALL_DELTA, kind="call")
            premium = bs_price(px, K, T, sigma, kind="call")
            premium = max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)
            if premium < 0.10:
                continue
            expiry = date + pd.Timedelta(days=dte_target)
            pos.has_cc = True
            pos.cc_strike = K
            pos.cc_premium = premium
            pos.cc_expiry = expiry
            cash += premium * 100 - COST_PER_CONTRACT
            trades.append({"date": date, "ticker": t, "action": "sell_cc", "premium": premium})

        # Open new CSPs (THIS IS WHERE ADAPTIVE DELTA HAPPENS)
        if is_bear and bear_mode != "none":
            continue

        current_margin_used = sum(p.margin_held for p in csp_positions.values())
        available_margin = nav * margin_cap - current_margin_used
        per_name_limit = nav * per_name_pct

        rng = random.Random(di)
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
            iv_rank = row.get("iv_rank", 0.5) if isinstance(row, pd.Series) else 0.5

            if vix > VIX_MAX:
                continue
            if pd.isna(sigma) or sigma < 0.05:
                continue

            notional = px * 100
            margin_req = notional * MARGIN_REQ_PCT

            if margin_req > per_name_limit:
                continue
            if margin_req > available_margin:
                continue
            if cash < margin_req:
                continue

            # ADAPTIVE DELTA SELECTION
            put_delta = delta_func(iv_rank, vix)

            T = dte_target / 365
            K = find_strike(px, sigma, T, put_delta, kind="put")
            premium = bs_price(px, K, T, sigma)
            premium = max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)

            if premium < 0.10:
                continue

            expiry = date + pd.Timedelta(days=dte_target)
            cash -= margin_req
            cash += premium * 100 - COST_PER_CONTRACT

            csp_positions[t] = CSPPos(
                ticker=t, strike=K, premium=premium,
                entry_date=date, expiry_date=expiry,
                margin_held=margin_req,
                put_delta_used=put_delta,
            )
            available_margin -= margin_req
            delta_log.append({"date": date, "ticker": t, "iv_rank": iv_rank, "vix": vix,
                            "delta_used": put_delta, "premium": premium, "sigma": sigma})
            trades.append({"date": date, "ticker": t, "action": "sell_csp",
                          "strike": K, "premium": premium, "margin": margin_req,
                          "delta": put_delta, "iv_rank": iv_rank})

    return daily_equity, trades, delta_log


# ========================= MAIN SWEEP =========================================

def main():
    log.info("=" * 70)
    log.info("WHEEL IV-RANK ADAPTIVE DELTA SWEEP")
    log.info("=" * 70)

    t0_total = time.time()

    # Load data
    prices, spy_regime, sector_map = load_all_data()

    # Compute IV rank
    prices = compute_iv_rank(prices, lookback=252)

    # Define strategies to test
    strategies = {
        # Baselines (fixed delta)
        "baseline_d25": lambda iv, vix: 0.25,
        "baseline_d30": lambda iv, vix: 0.30,
        "baseline_d35": lambda iv, vix: 0.35,

        # 3-tier IV rank adaptive
        "ivrank_3tier_standard": lambda iv, vix: iv_rank_3tier(iv, vix, 0.25, 0.30, 0.35),
        "ivrank_3tier_tight": lambda iv, vix: iv_rank_3tier(iv, vix, 0.22, 0.28, 0.35),

        # 5-tier IV rank adaptive
        "ivrank_5tier": lambda iv, vix: iv_rank_5tier(iv, vix),

        # Continuous interpolation
        "ivrank_continuous_20_40": lambda iv, vix: iv_rank_continuous(iv, vix, 0.20, 0.40),
        "ivrank_continuous_22_38": lambda iv, vix: iv_rank_continuous(iv, vix, 0.22, 0.38),

        # Aggressive variant
        "ivrank_aggressive": lambda iv, vix: iv_rank_aggressive(iv, vix),

        # VIX overlay on top of 3-tier
        "ivrank_3tier_vix_overlay": lambda iv, vix: vix_overlay(
            iv, vix,
            base_func=lambda i, v: iv_rank_3tier(i, v, 0.25, 0.30, 0.35),
            vix_threshold=25, vix_bump=0.05
        ),

        # VIX overlay on top of continuous
        "ivrank_continuous_vix_overlay": lambda iv, vix: vix_overlay(
            iv, vix,
            base_func=lambda i, v: iv_rank_continuous(i, v, 0.22, 0.38),
            vix_threshold=25, vix_bump=0.05
        ),
    }

    results = {}

    for name, delta_func in strategies.items():
        log.info(f"\n--- Running: {name} ---")
        t0 = time.time()

        daily_eq, trades, delta_log = run_adaptive_portfolio(
            prices, spy_regime, sector_map,
            delta_func=delta_func, delta_func_name=name,
            starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
            dte_target=14, profit_take=0.65, bear_mode="liq_csp_only",
        )

        elapsed = time.time() - t0
        metrics = compute_metrics(daily_eq, 100_000)
        regime = regime_analysis(daily_eq, spy_regime)

        # Delta distribution stats
        dl = pd.DataFrame(delta_log)
        delta_stats = {}
        if len(dl) > 0:
            delta_stats = {
                "mean_delta": round(dl["delta_used"].mean(), 4),
                "std_delta": round(dl["delta_used"].std(), 4),
                "mean_premium": round(dl["premium"].mean(), 4),
                "mean_iv_rank": round(dl["iv_rank"].mean(), 3),
                "n_trades": len(dl),
            }

        results[name] = {
            "metrics": metrics,
            "regime": regime,
            "delta_stats": delta_stats,
            "elapsed_s": round(elapsed, 1),
        }

        m = metrics
        log.info(f"  CAGR={m.get('cagr_pct', 0):.1f}% | Sharpe={m.get('sharpe', 0):.2f} | "
                f"Sortino={m.get('sortino', 0):.2f} | DD={m.get('max_dd_pct', 0):.1f}% | "
                f"Calmar={m.get('calmar', 0):.2f}")
        if delta_stats:
            log.info(f"  Avg delta={delta_stats.get('mean_delta', 0):.3f} | "
                    f"Avg premium=${delta_stats.get('mean_premium', 0):.3f} | "
                    f"Trades={delta_stats.get('n_trades', 0)}")
        log.info(f"  Regime: bull_sharpe={regime.get('bull_sharpe', 0):.2f} | "
                f"bear_sharpe={regime.get('bear_sharpe', 0):.2f} | "
                f"gap={regime.get('regime_gap', 0):.2f}")
        log.info(f"  ({elapsed:.1f}s)")

    # ========================= RANKING =========================================
    log.info("\n" + "=" * 90)
    log.info("RANKED BY CALMAR (risk-adjusted return per unit drawdown)")
    log.info("=" * 90)
    log.info(f"{'Strategy':35s} {'CAGR':>6s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s} {'Calmar':>7s} {'AvgΔ':>6s} {'AvgPrem':>8s}")
    log.info("-" * 90)

    ranked = sorted(results.items(), key=lambda x: x[1]["metrics"].get("calmar", 0), reverse=True)
    for name, r in ranked:
        m = r["metrics"]
        ds = r.get("delta_stats", {})
        log.info(f"  {name:33s} {m.get('cagr_pct',0):5.1f}% {m.get('sharpe',0):6.2f} "
                f"{m.get('sortino',0):7.2f} {m.get('max_dd_pct',0):6.1f}% {m.get('calmar',0):6.2f} "
                f"{ds.get('mean_delta',0):5.3f} ${ds.get('mean_premium',0):6.3f}")

    # ========================= VS BASELINE COMPARISON ===========================
    baseline_calmar = results.get("baseline_d30", {}).get("metrics", {}).get("calmar", 0)
    baseline_sharpe = results.get("baseline_d30", {}).get("metrics", {}).get("sharpe", 0)
    baseline_cagr = results.get("baseline_d30", {}).get("metrics", {}).get("cagr_pct", 0)

    log.info(f"\n{'=' * 70}")
    log.info(f"IMPROVEMENT VS BASELINE (d30 fixed)")
    log.info(f"{'=' * 70}")
    for name, r in ranked:
        if "baseline" in name:
            continue
        m = r["metrics"]
        calmar_diff = m.get("calmar", 0) - baseline_calmar
        sharpe_diff = m.get("sharpe", 0) - baseline_sharpe
        cagr_diff = m.get("cagr_pct", 0) - baseline_cagr
        log.info(f"  {name:33s}: Calmar {calmar_diff:+.2f} | Sharpe {sharpe_diff:+.2f} | CAGR {cagr_diff:+.1f}%")

    # Save results
    out_file = OUT_DIR / "iv_adaptive_sweep_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {out_file}")

    total_elapsed = time.time() - t0_total
    log.info(f"\nTotal sweep time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    return results


if __name__ == "__main__":
    results = main()
