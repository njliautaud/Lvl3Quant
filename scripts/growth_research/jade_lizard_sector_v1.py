#!/usr/bin/env python3
"""
Jade Lizard Sector Income v1 — LGBM-Ranked Premium Selling
============================================================
HC #746 R3 — Income strategies judged on consistency + NAV preservation,
not strict R1 regime gates.

Jade Lizard = Short Put + Short Call Spread (bear call spread).
Collect premium from both bullish put-selling and bearish call-spread-selling.
LGBM sector ranking decides which side to sell on which sector.

Variants:
  A: Classic Jade Lizard — Top 3: sell ATM put + 5% OTM call spread, DTE=21
  B: Defined Risk Jade  — Same + long put 5% below short put (defined risk)
  C: VIX-Adaptive Jade  — Only sell when VIX > 18, else cash
  D: Iron Condor Bottom — Bottom 3: sell iron condor (range-bound bet)

Config:
  Universe: 11 sector ETFs (XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC)
  Capital: $645 initial
  DTE: 21 days
  Commission: $2.60 per spread (4 legs × $0.65)
  Weekly Friday rebalance
  Walk-forward LGBM with 52-week training window
  BS pricing with 15% bid-ask haircut
  2008–2026 data from yfinance

Author: Claude (Opus 4.6), 2026-07-27
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
import traceback

warnings.filterwarnings("ignore")

def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ─── Path Detection (Jupiter vs Neptune) ───
if Path("/home/nick/Lvl3Quant").exists():
    BASE = Path("/home/nick/Lvl3Quant")
elif Path("/home/jupiter/Lvl3Quant").exists():
    BASE = Path("/home/jupiter/Lvl3Quant")
else:
    BASE = Path(__file__).resolve().parents[2]

OUT_DIR = BASE / "output" / "jade_lizard_sector_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = BASE / "research" / "findings"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "jade_lizard_sector_v1_results.json"

# ─── MLflow ───
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen("http://jupiter:5000/", timeout=3)
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    MLFLOW_OK = True
    fprint("MLflow connected")
except Exception:
    fprint("MLflow unavailable — results will be saved locally only")

# ─── LightGBM ───
try:
    import lightgbm as lgb
    LGBM_OK = True
except ImportError:
    fprint("FATAL: lightgbm not installed")
    sys.exit(1)

import yfinance as yf

# ─── Constants ───
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
CAPITAL = 645.0
DTE = 21
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # $2.60 round-trip for a 4-leg structure
HAIRCUT = 0.15  # 15% bid-ask haircut on BS premiums
RISK_FREE = 0.04
IV_PREMIUM = 1.15  # IV = realized_vol × 1.15
IV_FLOOR = 0.12
MAX_RISK_PCT = 0.30  # Max 30% of equity per trade

FEAT_COLS = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high",
    "mom_accel", "rsi_14", "vol_ratio", "ret_std_21d", "skew_21d", "kurt_21d",
]


# ═══════════════════════════════════════════════════════════════════════════
# Black-Scholes Pricing
# ═══════════════════════════════════════════════════════════════════════════

def bs_call(S, K, T, sigma, r=RISK_FREE):
    """Black-Scholes call price."""
    if T <= 1e-8 or sigma <= 1e-8:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put(S, K, T, sigma, r=RISK_FREE):
    """Black-Scholes put price."""
    if T <= 1e-8 or sigma <= 1e-8:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def get_iv(prices_col, as_of_idx, lookback=63):
    """Realized vol × IV premium, floored."""
    px = prices_col.iloc[max(0, as_of_idx - lookback) : as_of_idx + 1].dropna()
    if len(px) < 20:
        return 0.25
    rv = px.pct_change().dropna().std() * np.sqrt(252)
    return max(float(rv) * IV_PREMIUM, IV_FLOOR)


# ═══════════════════════════════════════════════════════════════════════════
# Data Download
# ═══════════════════════════════════════════════════════════════════════════

def download_data():
    """Download sector ETF + SPY + VIX data."""
    fprint("Downloading data...")
    cache = OUT_DIR / "price_cache.parquet"
    vix_cache = OUT_DIR / "vix_cache.parquet"

    if cache.exists() and vix_cache.exists():
        mtime = datetime.fromtimestamp(cache.stat().st_mtime)
        if (datetime.now() - mtime).days < 1:
            fprint("  Using cached data")
            prices = pd.read_parquet(cache)
            vix = pd.read_parquet(vix_cache)["VIX"]
            return prices, vix

    tickers = SECTORS + ["SPY", "^VIX"]
    raw = yf.download(tickers, start="2008-01-01", end="2026-07-27", progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw

    # VIX
    vc = "^VIX" if "^VIX" in close.columns else "VIX"
    vix = close[vc].dropna()
    vix.name = "VIX"

    # Sector + SPY prices
    cols = [c for c in SECTORS + ["SPY"] if c in close.columns]
    prices = close[cols].dropna(how="all")

    # Align indices
    ix = prices.index.intersection(vix.index)
    prices = prices.loc[ix]
    vix = vix.loc[ix]

    prices.to_parquet(cache)
    pd.DataFrame({"VIX": vix}).to_parquet(vix_cache)
    fprint(f"  Downloaded {len(cols)} tickers, {len(prices)} days ({prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')})")
    return prices, vix


# ═══════════════════════════════════════════════════════════════════════════
# Feature Engineering (17 production features)
# ═══════════════════════════════════════════════════════════════════════════

def compute_features(px_daily, idx, ticker):
    """Compute features for LGBM ranking at a given index."""
    px = px_daily[ticker].iloc[: idx + 1].dropna()
    if len(px) < 260:
        return None

    f = {}
    # Momentum returns
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()

    # Volatility
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2

    # Risk-adjusted
    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
    px63 = px.iloc[-63:]
    pk = px63.cummax()
    f["maxdd_63d"] = float(((px63 / pk) - 1).min())
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    # RSI
    r14 = rets.iloc[-14:]
    gains = r14[r14 > 0].sum()
    losses = abs(r14[r14 < 0].sum())
    f["rsi_14"] = 100.0 - 100.0 / (1.0 + gains / (losses + 1e-10)) if len(r14) >= 14 else 50.0

    # Vol ratio (short vs long)
    f["vol_ratio"] = f["vol_21d"] / (f["vol_63d"] + 1e-10)

    # Higher moments
    r21 = rets.iloc[-21:]
    f["ret_std_21d"] = float(r21.std()) if len(r21) > 5 else 0.01
    f["skew_21d"] = float(r21.skew()) if len(r21) > 5 else 0.0
    f["kurt_21d"] = float(r21.kurt()) if len(r21) > 5 else 0.0

    return f


# ═══════════════════════════════════════════════════════════════════════════
# LGBM Walk-Forward Ranking
# ═══════════════════════════════════════════════════════════════════════════

def lgbm_walk_forward(px_daily, rebal_dates, train_weeks=52):
    """Walk-forward LGBM sector ranking with sliding 52-week window."""
    fprint("Running LGBM walk-forward ranking...")

    # Build feature/label dataset
    records = []
    for dt in rebal_dates:
        idx = px_daily.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        for tk in [c for c in px_daily.columns if c != "SPY"]:
            feats = compute_features(px_daily, idx, tk)
            if feats is None:
                continue
            fi = min(idx + DTE, len(px_daily) - 1)
            fwd = float(px_daily[tk].iloc[fi] / px_daily[tk].iloc[idx] - 1)
            feats.update({"date": dt, "ticker": tk, "fwd_ret": fwd})
            records.append(feats)

    df = pd.DataFrame(records)
    if len(df) < 200:
        fprint(f"  Only {len(df)} records, need 200+")
        return {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    rankings = {}

    train_periods = train_weeks  # Each rebal date ~ 1 week apart

    for i in range(train_periods, len(dates)):
        # Sliding window: use most recent train_periods dates for training
        td = dates[max(0, i - train_periods) : i]
        test_date = dates[i]
        tr = df[df["date"].isin(td)]
        te = df[df["date"] == test_date].copy()
        if len(te) < 3 or len(tr) < 50:
            continue

        Xt = np.nan_to_num(tr[FEAT_COLS].values.astype(np.float32))
        yt = tr["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(te[FEAT_COLS].values.astype(np.float32))

        try:
            # Try GPU first, fall back to CPU
            try:
                m = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                    device="gpu", verbose=-1,
                )
                m.fit(Xt, yt)
            except Exception:
                m = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                    verbose=-1,
                )
                m.fit(Xt, yt)

            te["score"] = m.predict(Xe)
            rankings[test_date] = dict(zip(te["ticker"], te["score"]))
        except Exception as e:
            fprint(f"  LGBM error at {test_date}: {e}")
            continue

    fprint(f"  {len(rankings)} ranking dates produced")
    return rankings


# ═══════════════════════════════════════════════════════════════════════════
# Trade Simulation
# ═══════════════════════════════════════════════════════════════════════════

def price_jade_lizard(S, sigma, T, variant, r=RISK_FREE):
    """
    Price a jade lizard or variant and return credit, max_loss, strikes.

    A: Classic Jade Lizard — sell ATM put + sell 5% OTM call spread (5% wide)
    B: Defined Risk Jade  — same + long put 5% below short put
    C: VIX-Adaptive Jade  — same as A (caller handles VIX filter)
    D: Iron Condor        — sell put spread + call spread centered at current price
    """
    put_strike = round(S, 2)           # ATM put
    call_short = round(S * 1.05, 2)    # 5% OTM call
    call_long = round(S * 1.10, 2)     # 10% OTM call (defines call spread)

    if variant in ("A", "C"):
        # Classic jade lizard: short put + short call spread
        put_prem = bs_put(S, put_strike, T, sigma, r)
        call_short_prem = bs_call(S, call_short, T, sigma, r)
        call_long_prem = bs_call(S, call_long, T, sigma, r)
        call_spread_credit = call_short_prem - call_long_prem

        credit = put_prem + call_spread_credit
        credit *= (1 - HAIRCUT)  # Bid-ask haircut

        # Max loss on call side = call spread width - credit (defined)
        # Max loss on put side = put_strike × 100 (undefined, but floored at 0)
        # For sizing: use call spread max loss as proxy since put side = happy to own
        call_max_loss = (call_long - call_short) - call_spread_credit
        # But real max loss on put side is if stock goes to 0
        # Size based on put assignment risk: max tolerable loss
        max_loss = put_strike * 0.15  # Assume max 15% drop from ATM = max pain

        return {
            "credit": credit,
            "max_loss": max_loss,
            "put_strike": put_strike,
            "call_short": call_short,
            "call_long": call_long,
            "put_long": None,
            "call_spread_width": call_long - call_short,
        }

    elif variant == "B":
        # Defined risk: add long put 5% below short put
        put_long_strike = round(S * 0.95, 2)

        put_short_prem = bs_put(S, put_strike, T, sigma, r)
        put_long_prem = bs_put(S, put_long_strike, T, sigma, r)
        call_short_prem = bs_call(S, call_short, T, sigma, r)
        call_long_prem = bs_call(S, call_long, T, sigma, r)

        put_spread_credit = put_short_prem - put_long_prem
        call_spread_credit = call_short_prem - call_long_prem
        credit = (put_spread_credit + call_spread_credit) * (1 - HAIRCUT)

        # Max loss = wider of (put spread width, call spread width) - credit
        put_width = put_strike - put_long_strike
        call_width = call_long - call_short
        max_loss = max(put_width, call_width) - credit

        return {
            "credit": credit,
            "max_loss": max(max_loss, 0.01),
            "put_strike": put_strike,
            "put_long": put_long_strike,
            "call_short": call_short,
            "call_long": call_long,
            "call_spread_width": call_width,
        }

    elif variant == "D":
        # Iron condor on bottom sectors (range-bound bet)
        put_short = round(S * 0.97, 2)   # 3% OTM put
        put_long_k = round(S * 0.92, 2)  # 8% OTM put (5% wide)
        call_short_k = round(S * 1.03, 2)  # 3% OTM call
        call_long_k = round(S * 1.08, 2)   # 8% OTM call (5% wide)

        sp = bs_put(S, put_short, T, sigma, r)
        lp = bs_put(S, put_long_k, T, sigma, r)
        sc = bs_call(S, call_short_k, T, sigma, r)
        lc = bs_call(S, call_long_k, T, sigma, r)

        credit = ((sp - lp) + (sc - lc)) * (1 - HAIRCUT)
        max_loss = (put_short - put_long_k) - credit  # Spread width - credit

        return {
            "credit": credit,
            "max_loss": max(max_loss, 0.01),
            "put_strike": put_short,
            "put_long": put_long_k,
            "call_short": call_short_k,
            "call_long": call_long_k,
            "call_spread_width": call_long_k - call_short_k,
        }

    return None


def payoff_at_exit(S_exit, strikes, variant, sigma_exit, T_remaining, r=RISK_FREE):
    """Compute position value at exit (cost to close)."""

    if variant in ("A", "C"):
        # Short put + short call spread
        put_val = bs_put(S_exit, strikes["put_strike"], T_remaining, sigma_exit, r)
        cs_val = bs_call(S_exit, strikes["call_short"], T_remaining, sigma_exit, r)
        cl_val = bs_call(S_exit, strikes["call_long"], T_remaining, sigma_exit, r)
        # Cost to close = buy back put + buy back call spread
        cost_to_close = put_val + (cs_val - cl_val)
        return cost_to_close

    elif variant == "B":
        # Short put spread + short call spread
        ps_val = bs_put(S_exit, strikes["put_strike"], T_remaining, sigma_exit, r)
        pl_val = bs_put(S_exit, strikes["put_long"], T_remaining, sigma_exit, r)
        cs_val = bs_call(S_exit, strikes["call_short"], T_remaining, sigma_exit, r)
        cl_val = bs_call(S_exit, strikes["call_long"], T_remaining, sigma_exit, r)
        cost_to_close = (ps_val - pl_val) + (cs_val - cl_val)
        return cost_to_close

    elif variant == "D":
        # Iron condor: short put spread + short call spread
        sp_val = bs_put(S_exit, strikes["put_strike"], T_remaining, sigma_exit, r)
        lp_val = bs_put(S_exit, strikes["put_long"], T_remaining, sigma_exit, r)
        sc_val = bs_call(S_exit, strikes["call_short"], T_remaining, sigma_exit, r)
        lc_val = bs_call(S_exit, strikes["call_long"], T_remaining, sigma_exit, r)
        cost_to_close = (sp_val - lp_val) + (sc_val - lc_val)
        return cost_to_close

    return 0.0


def simulate_variant(prices, vix, rankings, variant="A", name="", vix_filter=None):
    """
    Simulate one strategy variant across all dates.

    variant: A, B, C, D
    vix_filter: if set, only trade when VIX > this level
    """
    fprint(f"\n{'='*60}")
    fprint(f"Variant {name}")
    fprint(f"{'='*60}")

    sector_cols = [c for c in prices.columns if c in SECTORS]
    spy = prices["SPY"] if "SPY" in prices.columns else None
    spy_sma200 = spy.rolling(200).mean() if spy is not None else None

    ranking_dates = sorted(rankings.keys())
    if not ranking_dates:
        fprint("  No ranking dates!")
        return None

    equity = CAPITAL
    equity_curve = [CAPITAL]
    equity_dates = [ranking_dates[0]]
    trades = []
    monthly_pnl = {}

    for rd in ranking_dates:
        rd_idx = prices.index.get_indexer([rd], method="ffill")[0]
        if rd_idx < 0 or rd_idx + DTE >= len(prices):
            continue

        # VIX filter for variant C
        if vix_filter is not None:
            vix_val = float(vix.iloc[rd_idx]) if rd_idx < len(vix) else 15.0
            if vix_val < vix_filter:
                continue

        scores = rankings[rd]
        sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if variant in ("A", "B", "C"):
            # Top 3 sectors: sell jade lizard (bullish on them)
            top3 = [s[0] for s in sorted_sectors[:3] if s[0] in sector_cols]
            targets = top3
        elif variant == "D":
            # Bottom 3 sectors: sell iron condor (range-bound bet)
            bot3 = [s[0] for s in sorted_sectors[-3:] if s[0] in sector_cols]
            targets = bot3
        else:
            targets = []

        for ticker in targets:
            S = float(prices[ticker].iloc[rd_idx])
            if np.isnan(S) or S <= 0:
                continue

            sigma = get_iv(prices[ticker], rd_idx)
            T = DTE / 252.0

            pricing = price_jade_lizard(S, sigma, T, variant)
            if pricing is None or pricing["credit"] <= 0:
                continue

            credit = pricing["credit"]
            max_loss = pricing["max_loss"]

            # Position sizing: max risk per trade ≤ 30% of equity
            # For options: 1 contract = 100 shares multiplier
            max_loss_dollar = max_loss * 100
            if max_loss_dollar > equity * MAX_RISK_PCT:
                continue  # Skip if risk too large

            if max_loss_dollar <= 0:
                continue

            credit_dollar = credit * 100

            # Simulate exit at DTE or early (50% profit / breach)
            entry_date = prices.index[rd_idx]
            exit_pnl_dollar = None
            exit_date = None

            for d in range(1, DTE + 1):
                check_idx = rd_idx + d
                if check_idx >= len(prices):
                    break

                S_now = float(prices[ticker].iloc[check_idx])
                if np.isnan(S_now):
                    continue

                days_left = DTE - d
                T_now = max(days_left / 252.0, 1e-6)
                sigma_now = get_iv(prices[ticker], check_idx, lookback=21)

                cost_to_close = payoff_at_exit(S_now, pricing, variant, sigma_now, T_now)
                current_pnl = credit - cost_to_close

                # Early exit: 50% of max profit captured
                if current_pnl >= credit * 0.50:
                    exit_pnl_dollar = current_pnl * 100 - SPREAD_COMM
                    exit_date = prices.index[check_idx]
                    break

                # Stop: if losing > 2× credit (rolling risk management)
                if current_pnl < -credit * 2.0:
                    exit_pnl_dollar = current_pnl * 100 - SPREAD_COMM
                    exit_date = prices.index[check_idx]
                    break

                # Close at 3 DTE to avoid gamma risk
                if days_left <= 3:
                    exit_pnl_dollar = current_pnl * 100 - SPREAD_COMM
                    exit_date = prices.index[check_idx]
                    break

            # Expired — compute final payoff
            if exit_pnl_dollar is None:
                exp_idx = min(rd_idx + DTE, len(prices) - 1)
                S_exp = float(prices[ticker].iloc[exp_idx])
                # At expiry, T=0 → intrinsic value only
                cost_at_exp = payoff_at_exit(S_exp, pricing, variant, sigma, 1e-8)
                exit_pnl_dollar = (credit - cost_at_exp) * 100 - SPREAD_COMM
                exit_date = prices.index[exp_idx]

            # Regime classification
            if spy is not None and spy_sma200 is not None:
                spy_px = float(spy.iloc[rd_idx])
                sma_val = float(spy_sma200.iloc[rd_idx]) if not np.isnan(spy_sma200.iloc[rd_idx]) else spy_px
                regime = "bear" if spy_px < sma_val else "bull"
            else:
                regime = "unknown"

            # Update equity
            equity += exit_pnl_dollar
            equity_curve.append(equity)
            equity_dates.append(exit_date)

            # Monthly tracking
            month_key = entry_date.strftime("%Y-%m")
            monthly_pnl[month_key] = monthly_pnl.get(month_key, 0.0) + exit_pnl_dollar

            trade = {
                "entry_date": str(entry_date.date()),
                "exit_date": str(exit_date.date()) if exit_date is not None else "expired",
                "ticker": ticker,
                "spot": round(S, 2),
                "credit_per_share": round(credit, 3),
                "credit_dollar": round(credit_dollar, 2),
                "pnl_dollar": round(exit_pnl_dollar, 2),
                "pct_return": round(exit_pnl_dollar / max_loss_dollar * 100, 1),
                "regime": regime,
                "win": exit_pnl_dollar > 0,
                "sigma": round(sigma, 3),
            }
            trades.append(trade)

    if not trades:
        fprint("  No trades generated!")
        return None

    # ─── Metrics ───
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["win"])
    wr = wins / n_trades * 100

    total_pnl = sum(t["pnl_dollar"] for t in trades)
    avg_win = np.mean([t["pnl_dollar"] for t in trades if t["win"]]) if wins > 0 else 0
    avg_loss = np.mean([t["pnl_dollar"] for t in trades if not t["win"]]) if wins < n_trades else 0

    # Monthly return series
    months_sorted = sorted(monthly_pnl.keys())
    monthly_rets = np.array([monthly_pnl[m] / CAPITAL for m in months_sorted])
    n_months = len(monthly_rets)
    n_years = n_months / 12.0

    if n_months < 6:
        fprint(f"  Only {n_months} months — too few")
        return None

    pct_months_positive = sum(1 for r in monthly_rets if r > 0) / n_months * 100
    ann_ret = np.mean(monthly_rets) * 12
    ann_vol = np.std(monthly_rets) * np.sqrt(12)
    sharpe = ann_ret / (ann_vol + 1e-10)

    downside = monthly_rets[monthly_rets < 0]
    sortino = ann_ret / (downside.std() * np.sqrt(12) + 1e-10) if len(downside) > 0 else 99.9

    cagr = (equity / CAPITAL) ** (1.0 / max(n_years, 0.01)) - 1.0

    # Max drawdown from equity curve
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    maxdd = float(dd.min())

    # Profit factor
    gross_wins = sum(t["pnl_dollar"] for t in trades if t["win"])
    gross_losses = abs(sum(t["pnl_dollar"] for t in trades if not t["win"]))
    pf = gross_wins / (gross_losses + 1e-10)

    # Annualized yield
    ann_yield = total_pnl / max(n_years, 0.01) / CAPITAL * 100

    # Regime analysis
    bull_trades = [t for t in trades if t["regime"] == "bull"]
    bear_trades = [t for t in trades if t["regime"] == "bear"]
    bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
    bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100
    r1_gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr, 1)

    # Sector breakdown
    sector_pnl = {}
    for t in trades:
        tk = t["ticker"]
        sector_pnl[tk] = sector_pnl.get(tk, 0) + t["pnl_dollar"]

    result = {
        "name": name,
        "variant": variant,
        "n_trades": n_trades,
        "win_rate": round(wr, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(equity, 2),
        "cagr_pct": round(cagr * 100, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "maxdd_pct": round(maxdd * 100, 1),
        "pf": round(pf, 2),
        "ann_yield_pct": round(ann_yield, 1),
        "pct_months_positive": round(pct_months_positive, 1),
        "r1_gap": round(r1_gap, 3),
        "r1_pass": r1_gap <= 0.50,
        "bull_wr": round(bull_wr, 1),
        "bear_wr": round(bear_wr, 1),
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
        "n_months": n_months,
        "sector_pnl": {k: round(v, 2) for k, v in sorted(sector_pnl.items(), key=lambda x: x[1], reverse=True)},
        "monthly_returns": monthly_rets.tolist(),
    }

    fprint(f"  Trades: {n_trades} | WR: {wr:.1f}% | Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}")
    fprint(f"  CAGR: {cagr*100:.1f}% | MaxDD: {maxdd*100:.1f}% | PF: {pf:.2f}")
    fprint(f"  Total P&L: ${total_pnl:.0f} | Final equity: ${equity:.0f} | Ann yield: {ann_yield:.1f}%")
    fprint(f"  Monthly consistency: {pct_months_positive:.0f}% months positive")
    fprint(f"  Bull WR: {bull_wr:.0f}% ({len(bull_trades)}t) | Bear WR: {bear_wr:.0f}% ({len(bear_trades)}t)")
    fprint(f"  R1 regime gap: {r1_gap:.3f} ({'PASS' if r1_gap <= 0.5 else 'FAIL'})")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Adversarial Validation (5 Gates)
# ═══════════════════════════════════════════════════════════════════════════

def permutation_test(returns, n_perms=1000):
    """Gate 1: Permutation test — is timing edge real?"""
    if len(returns) < 5:
        return 1.0
    real_sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
    count = 0
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(returns))
        perm = returns * signs
        if np.mean(perm) / (np.std(perm) + 1e-10) >= real_sharpe:
            count += 1
    return count / n_perms


def adversarial_validate(results):
    """Run 5-gate adversarial validation on all results."""
    fprint(f"\n{'='*60}")
    fprint("ADVERSARIAL VALIDATION (5 GATES)")
    fprint(f"{'='*60}")

    for r in results:
        rets = np.array(r["monthly_returns"])
        n = len(rets)

        # G1: Permutation test (p < 0.05)
        perm_p = permutation_test(rets)
        r["g1_perm_p"] = round(perm_p, 3)
        r["g1_pass"] = perm_p < 0.05

        # G2: Regime agnostic (R1 gap ≤ 0.50)
        r["g2_pass"] = r["r1_pass"]

        # G3: Sub-period consistency (all thirds Sharpe > 0)
        chunk = max(n // 3, 1)
        sub_sharpes = []
        for i in range(3):
            sub = rets[i * chunk : (i + 1) * chunk]
            if len(sub) > 1:
                sub_sharpes.append(round(np.mean(sub) * 12 / (np.std(sub) * np.sqrt(12) + 1e-10), 2))
            else:
                sub_sharpes.append(0.0)
        r["g3_sub_sharpes"] = sub_sharpes
        r["g3_pass"] = all(s > 0 for s in sub_sharpes)

        # G4: Outlier robustness (trimmed Sharpe > 50% of full)
        if n > 10:
            n_trim = max(1, int(n * 0.05))
            trimmed = np.sort(rets)[n_trim:-n_trim] if 2 * n_trim < n else rets
            orig_s = np.mean(rets) / (np.std(rets) + 1e-10)
            trim_s = np.mean(trimmed) / (np.std(trimmed) + 1e-10)
            r["g4_pass"] = trim_s > 0 and (trim_s / (orig_s + 1e-10)) > 0.5
        else:
            r["g4_pass"] = False

        # G5: Income consistency (≥55% months positive for premium selling)
        r["g5_pass"] = r["pct_months_positive"] >= 55.0

        gates = sum([r["g1_pass"], r["g2_pass"], r["g3_pass"], r["g4_pass"], r["g5_pass"]])
        r["gates_passed"] = gates

        fprint(f"\n{r['name']}:")
        fprint(f"  G1 Permutation:  {'PASS' if r['g1_pass'] else 'FAIL'} (p={r['g1_perm_p']})")
        fprint(f"  G2 Regime:       {'PASS' if r['g2_pass'] else 'FAIL'} (gap={r['r1_gap']})")
        fprint(f"  G3 Sub-period:   {'PASS' if r['g3_pass'] else 'FAIL'} ({r['g3_sub_sharpes']})")
        fprint(f"  G4 Outlier:      {'PASS' if r['g4_pass'] else 'FAIL'}")
        fprint(f"  G5 Consistency:  {'PASS' if r['g5_pass'] else 'FAIL'} ({r['pct_months_positive']:.0f}% months +)")
        fprint(f"  TOTAL: {gates}/5 gates")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint(f"Jade Lizard Sector Income v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"Capital: ${CAPITAL} | DTE: {DTE}d | Commission: ${SPREAD_COMM:.2f}/trade")
    fprint(f"Haircut: {HAIRCUT*100:.0f}% | Max risk/trade: {MAX_RISK_PCT*100:.0f}%")
    fprint("=" * 60)

    # Download data
    prices, vix = download_data()

    # Weekly Friday rebalance dates
    all_dates = prices.index
    fridays = all_dates[all_dates.dayofweek == 4]
    # Start after 1 year of data for features
    rebal_dates = fridays[fridays >= all_dates[0] + timedelta(days=365)]
    fprint(f"Rebalance dates: {len(rebal_dates)} Fridays")

    # LGBM rankings
    rankings = lgbm_walk_forward(prices, rebal_dates)
    if not rankings:
        fprint("FATAL: No LGBM rankings produced")
        return

    # MLflow experiment
    if MLFLOW_OK:
        exp_name = "jade_lizard_sector_v1"
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                mlflow.create_experiment(exp_name)
        except Exception:
            pass
        mlflow.set_experiment(exp_name)

    # Run all variants
    variants = [
        ("A", "A_Classic_Jade_Lizard", None),
        ("B", "B_Defined_Risk_Jade", None),
        ("C", "C_VIX_Adaptive_Jade", 18.0),  # VIX > 18 filter
        ("D", "D_Iron_Condor_Bottom", None),
    ]

    results = []
    for v_code, v_name, vix_filt in variants:
        try:
            if MLFLOW_OK:
                with mlflow.start_run(run_name=v_name):
                    r = simulate_variant(prices, vix, rankings,
                                         variant=v_code, name=v_name, vix_filter=vix_filt)
                    if r:
                        # Log params
                        mlflow.log_params({
                            "variant": v_code,
                            "dte": DTE,
                            "capital": CAPITAL,
                            "commission": SPREAD_COMM,
                            "haircut": HAIRCUT,
                            "max_risk_pct": MAX_RISK_PCT,
                            "vix_filter": vix_filt or "none",
                        })
                        # Log metrics (exclude lists/dicts)
                        metrics = {k: v for k, v in r.items()
                                   if isinstance(v, (int, float)) and k not in ("monthly_returns",)}
                        mlflow.log_metrics(metrics)
                        results.append(r)
            else:
                r = simulate_variant(prices, vix, rankings,
                                     variant=v_code, name=v_name, vix_filter=vix_filt)
                if r:
                    results.append(r)
        except Exception as e:
            fprint(f"  ERROR in {v_name}: {e}")
            traceback.print_exc()

    if not results:
        fprint("\nNo results produced!")
        return

    # Adversarial validation
    adversarial_validate(results)

    # ─── Summary ───
    fprint(f"\n{'='*80}")
    fprint("SUMMARY — JADE LIZARD SECTOR INCOME v1")
    fprint(f"{'='*80}")
    fprint(f"{'Name':<28} {'Trades':>6} {'WR':>6} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} "
           f"{'MaxDD':>7} {'PF':>5} {'Mo+%':>5} {'Gates':>6}")
    fprint("-" * 95)
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        fprint(f"{r['name']:<28} {r['n_trades']:>6} {r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} "
               f"{r['sortino']:>8.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['pf']:>5.2f} {r['pct_months_positive']:>4.0f}% {r['gates_passed']:>4}/5")

    # Income-specific metrics
    fprint(f"\n{'='*60}")
    fprint("INCOME METRICS")
    fprint(f"{'='*60}")
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        fprint(f"\n{r['name']}:")
        fprint(f"  Annualized yield on ${CAPITAL}: {r['ann_yield_pct']:.1f}%")
        fprint(f"  Monthly consistency: {r['pct_months_positive']:.0f}% months positive")
        fprint(f"  Final equity: ${r['final_equity']:.0f} ({r['cagr_pct']:.1f}% CAGR)")
        fprint(f"  Max drawdown: {r['maxdd_pct']:.1f}%")
        top_sectors = list(r["sector_pnl"].items())[:3]
        fprint(f"  Top sectors: {', '.join(f'{s}=${p:.0f}' for s, p in top_sectors)}")

    # Save results (strip monthly_returns for JSON)
    save_data = [{k: v for k, v in r.items() if k != "monthly_returns"} for r in results]
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # Save equity curves
    eq_path = OUT_DIR / "summary.json"
    with open(eq_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nDone in {elapsed:.0f}s — {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
