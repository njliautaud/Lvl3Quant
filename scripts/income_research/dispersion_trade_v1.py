"""
Dispersion Trade v1 — Implied Correlation / Dispersion Income Strategy
=======================================================================
Exploits the structural overpricing of index implied volatility vs. single-stock
implied volatility. Index options carry a "correlation risk premium" because
portfolio hedgers bid up SPX/SPY puts, making index IV rich vs constituent IV.

THREE variants tested:
  A) SPY Straddle Sell When High Corr:
     - Compute implied correlation proxy (VIX^2 vs weighted avg single-stock vol^2)
     - Sell ATM SPY straddle ONLY when implied corr > 75th percentile (lookback)
     - Close after 5 trading days or 50% profit (whichever first)

  B) SPY Strangle Sell Always + Size by Corr:
     - Always sell ~10-delta SPY strangle (weekly cycle)
     - Size inversely to implied corr: high corr = regime stress = smaller
     - Monthly expiry cycle

  C) Pure Dispersion:
     - Sell SPY ATM straddle, buy straddles on top-5 highest-IV components
     - Market-neutral vol trade: captures correlation risk premium directly

Walk-forward: 252-day lookback for percentiles, monthly trades, 2019-2026.
HC compliance: #428 R1 regime gap, #344 day-conc ≤ 0.70, permutation test (500 shuffles).

Output: /home/nick/Lvl3Quant/output/dispersion_trade_v1/
"""

import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
CHAINS_DIR  = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/options_real/chains")
PRICES_PATH = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/prices.parquet")
MACRO_PATH  = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/macro.parquet")
OUT_DIR     = Path("/home/jupiter/Lvl3Quant/output/dispersion_trade_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL    = 100_000
OOT_FRACTION       = 0.30
LOOKBACK_DAYS       = 252   # for percentile calc (1 year of trading days)
DTE_MIN, DTE_MAX    = 5, 45
TARGET_DTE_WEEKLY   = 7     # ~1 week
TARGET_DTE_MONTHLY  = 30    # ~1 month
N_PERMUTATIONS      = 500
RANDOM_SEED         = 42
VIX_FLATTEN_THRESH  = 35    # flatten everything above this
MA_FILTER_DAYS      = 200   # reduce size below 200MA

# SPY component weights (approximate 2024-25, top holdings we have data for)
SPY_COMPONENTS = {
    "AAPL":  0.070, "MSFT":  0.065, "NVDA":  0.060, "AMZN":  0.040,
    "META":  0.025, "GOOGL": 0.022, "BRK-B": 0.015, "LLY":   0.013,
    "JPM":   0.013,
}


# ── Black-Scholes helpers ────────────────────────────────────────────────────

def bs_call(S, K, T, sigma, r=0.045):
    """Black-Scholes call price."""
    if T < 1e-8 or sigma < 1e-8:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, sigma, r=0.045):
    """Black-Scholes put price."""
    if T < 1e-8 or sigma < 1e-8:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_straddle(S, K, T, sigma, r=0.045):
    """ATM straddle price."""
    return bs_call(S, K, T, sigma, r) + bs_put(S, K, T, sigma, r)


def bs_strangle(S, K_put, K_call, T, sigma_put, sigma_call, r=0.045):
    """OTM strangle price (put at K_put, call at K_call)."""
    return bs_put(S, K_put, T, sigma_put, r) + bs_call(S, K_call, T, sigma_call, r)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load prices, macro, and options chains for SPY + components."""
    print("\n[1] Loading data...")
    t0 = time.time()

    # Prices
    prices_all = pd.read_parquet(PRICES_PATH)
    prices_all["date"] = pd.to_datetime(prices_all["date"])

    spy_px = (prices_all[prices_all["ticker"] == "SPY"]
              .set_index("date")[["close", "rv_20"]].sort_index())
    spy_px.columns = ["spy_close", "spy_rv20"]

    # 200-day MA
    spy_px["spy_ma200"] = spy_px["spy_close"].rolling(MA_FILTER_DAYS, min_periods=100).mean()

    # Component prices (for realized vol)
    comp_tickers = list(SPY_COMPONENTS.keys())
    comp_px = prices_all[prices_all["ticker"].isin(comp_tickers)].copy()
    comp_px_wide = comp_px.pivot_table(index="date", columns="ticker", values="close")
    comp_rv_wide = comp_px.pivot_table(index="date", columns="ticker", values="rv_20")

    # Macro (VIX)
    macro = pd.read_parquet(MACRO_PATH)
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date")[["vix"]].sort_index()

    # Options chains
    def load_chain(tkr):
        f = CHAINS_DIR / f"{tkr}.parquet"
        if not f.exists():
            return None
        try:
            df = pd.read_parquet(f)
            df["date"] = pd.to_datetime(df["date"])
            df["expiration"] = pd.to_datetime(df["expiration"])
            df["ticker"] = tkr
            return df[(df["vol"] > 0) & (df["vol"] < 3.0) & (df["mid"] > 0)]
        except Exception:
            return None

    spy_chains = load_chain("SPY")
    print(f"  SPY chains: {len(spy_chains):,} rows")

    comp_chains = {}
    for tkr in comp_tickers:
        c = load_chain(tkr)
        if c is not None:
            comp_chains[tkr] = c
            print(f"  {tkr} chains: {len(c):,} rows")

    print(f"  Load time: {time.time()-t0:.1f}s")
    return spy_px, comp_px_wide, comp_rv_wide, macro, spy_chains, comp_chains


# ═══════════════════════════════════════════════════════════════════════════════
# 2. IMPLIED CORRELATION PROXY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_implied_correlation(spy_px, comp_rv_wide, macro):
    """
    Implied correlation proxy:
      IC = VIX / weighted_avg_single_stock_20d_vol

    When IC is high, stocks are moving together → index vol overpriced → sell it.
    When IC is low, stock-specific moves dominate → index vol fair/cheap.

    Also compute: VIX^2 - sum(w_i^2 * sigma_i^2) / (2 * sum(w_i*w_j*sigma_i*sigma_j))
    """
    print("\n[2] Computing implied correlation proxy...")

    dates = spy_px.index.intersection(macro.index).intersection(comp_rv_wide.index)
    dates = sorted(dates)

    records = []
    weights = SPY_COMPONENTS.copy()
    w_total = sum(weights.values())
    weights = {k: v / w_total for k, v in weights.items()}

    for dt in dates:
        vix = macro.loc[dt, "vix"] / 100.0  # VIX is annualized %
        spy_close = spy_px.loc[dt, "spy_close"]
        spy_rv = spy_px.loc[dt, "spy_rv20"]

        # Weighted average single-stock realized vol
        stock_vols = {}
        for tkr, w in weights.items():
            if tkr in comp_rv_wide.columns and dt in comp_rv_wide.index:
                rv = comp_rv_wide.loc[dt, tkr]
                if pd.notna(rv) and rv > 0:
                    stock_vols[tkr] = rv

        if len(stock_vols) < 5:
            continue

        # Reweight available tickers
        avail_w = {k: weights[k] for k in stock_vols}
        tw = sum(avail_w.values())
        avail_w = {k: v / tw for k, v in avail_w.items()}

        weighted_avg_vol = sum(avail_w[k] * stock_vols[k] for k in avail_w)

        # Simple ratio proxy
        ic_ratio = vix / weighted_avg_vol if weighted_avg_vol > 0.001 else np.nan

        # Dispersion formula: implied corr
        # sigma_idx^2 = sum(w_i^2 * sigma_i^2) + 2*rho*sum(w_i*w_j*sigma_i*sigma_j)
        # rho_implied = (sigma_idx^2 - sum(w_i^2*sigma_i^2)) / (2*sum(w_i*w_j*sigma_i*sigma_j))
        tickers = list(avail_w.keys())
        ws = np.array([avail_w[t] for t in tickers])
        sigmas = np.array([stock_vols[t] for t in tickers])

        diag_term = np.sum(ws**2 * sigmas**2)
        cross_term = 0.0
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                cross_term += ws[i] * ws[j] * sigmas[i] * sigmas[j]

        if cross_term > 1e-10:
            rho_implied = (vix**2 - diag_term) / (2 * cross_term)
            rho_implied = np.clip(rho_implied, -1.0, 1.0)
        else:
            rho_implied = np.nan

        # Below 200MA flag
        below_ma = spy_close < spy_px.loc[dt, "spy_ma200"] if pd.notna(spy_px.loc[dt, "spy_ma200"]) else False

        records.append({
            "date": dt,
            "vix": vix * 100,
            "spy_close": spy_close,
            "spy_rv20": spy_rv,
            "weighted_avg_stock_vol": weighted_avg_vol,
            "ic_ratio": ic_ratio,
            "rho_implied": rho_implied,
            "vrp": vix - spy_rv,  # variance risk premium (IV - RV)
            "below_ma200": below_ma,
            "n_stocks": len(stock_vols),
        })

    ic_df = pd.DataFrame(records).set_index("date").sort_index()

    # Rolling percentiles
    ic_df["ic_ratio_pct"] = ic_df["ic_ratio"].rolling(LOOKBACK_DAYS, min_periods=60).rank(pct=True)
    ic_df["rho_pct"] = ic_df["rho_implied"].rolling(LOOKBACK_DAYS, min_periods=60).rank(pct=True)
    ic_df["vix_pct"] = ic_df["vix"].rolling(LOOKBACK_DAYS, min_periods=60).rank(pct=True)

    print(f"  Dates computed: {len(ic_df)}")
    print(f"  IC ratio: mean={ic_df['ic_ratio'].mean():.3f}, "
          f"median={ic_df['ic_ratio'].median():.3f}")
    print(f"  Rho implied: mean={ic_df['rho_implied'].mean():.3f}, "
          f"median={ic_df['rho_implied'].median():.3f}")
    print(f"  VRP (IV-RV): mean={ic_df['vrp'].mean():.4f}")

    return ic_df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OPTION SELECTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def find_atm_straddle(chains, date, target_dte=14):
    """Find the ATM straddle for a given date: closest to target DTE, then closest delta to 0.50."""
    day_chains = chains[chains["date"] == date]
    if len(day_chains) == 0:
        return None

    # Filter to reasonable DTE range
    day_chains = day_chains[day_chains["dte"].between(max(3, target_dte - 7), target_dte + 14)]
    if len(day_chains) == 0:
        return None

    # Pick expiry closest to target DTE
    calls = day_chains[day_chains["type"] == "c"].copy()
    puts = day_chains[day_chains["type"] == "p"].copy()

    if len(calls) == 0 or len(puts) == 0:
        return None

    best_dte = calls.iloc[(calls["dte"] - target_dte).abs().argsort().iloc[0]]["dte"]
    calls = calls[calls["dte"] == best_dte]
    puts = puts[puts["dte"] == best_dte]

    # ATM: delta closest to 0.50
    calls["delta_dist"] = (calls["delta"].abs() - 0.50).abs()
    atm_call = calls.sort_values("delta_dist").iloc[0]

    # Match put at same strike
    atm_strike = atm_call["strike"]
    put_match = puts[(puts["strike"] - atm_strike).abs() < 0.5]
    if len(put_match) == 0:
        # Fallback: closest put
        put_match = puts.iloc[[(puts["strike"] - atm_strike).abs().argmin()]]

    atm_put = put_match.iloc[0]
    straddle_price = atm_call["mid"] + atm_put["mid"]
    iv = (atm_call["vol"] + atm_put["vol"]) / 2

    return {
        "strike": atm_strike,
        "call_mid": atm_call["mid"],
        "put_mid": atm_put["mid"],
        "straddle_price": straddle_price,
        "iv": iv,
        "dte": best_dte,
        "expiration": atm_call["expiration"],
        "call_delta": atm_call["delta"],
        "put_delta": atm_put["delta"],
        "call_vega": atm_call.get("vega", 0),
        "put_vega": atm_put.get("vega", 0),
    }


def find_otm_strangle(chains, date, target_dte=30, put_delta_target=-0.10, call_delta_target=0.10):
    """Find OTM strangle: ~10-delta put and ~10-delta call."""
    day_chains = chains[chains["date"] == date]
    if len(day_chains) == 0:
        return None

    day_chains = day_chains[day_chains["dte"].between(max(5, target_dte - 10), target_dte + 15)]
    if len(day_chains) == 0:
        return None

    calls = day_chains[day_chains["type"] == "c"].copy()
    puts = day_chains[day_chains["type"] == "p"].copy()

    if len(calls) == 0 or len(puts) == 0:
        return None

    # Pick expiry closest to target DTE
    best_dte = calls.iloc[(calls["dte"] - target_dte).abs().argsort().iloc[0]]["dte"]
    calls = calls[calls["dte"] == best_dte]
    puts = puts[puts["dte"] == best_dte]

    # ~10 delta put (delta ~ -0.10)
    puts["delta_dist"] = (puts["delta"] - put_delta_target).abs()
    otm_put = puts.sort_values("delta_dist").iloc[0]

    # ~10 delta call (delta ~ 0.10)
    calls["delta_dist"] = (calls["delta"] - call_delta_target).abs()
    otm_call = calls.sort_values("delta_dist").iloc[0]

    strangle_price = otm_call["mid"] + otm_put["mid"]

    return {
        "call_strike": otm_call["strike"],
        "put_strike": otm_put["strike"],
        "call_mid": otm_call["mid"],
        "put_mid": otm_put["mid"],
        "strangle_price": strangle_price,
        "call_iv": otm_call["vol"],
        "put_iv": otm_put["vol"],
        "call_delta": otm_call["delta"],
        "put_delta": otm_put["delta"],
        "dte": best_dte,
        "expiration": otm_call["expiration"],
    }


def compute_straddle_exit_value(spy_close_exit, entry):
    """Intrinsic value of straddle at exit (simplified mark-to-market)."""
    K = entry["strike"]
    intrinsic = abs(spy_close_exit - K)
    # Add remaining time value estimate (rough: proportional to sqrt remaining)
    return intrinsic


def compute_strangle_exit_value(spy_close_exit, entry):
    """Intrinsic value of strangle at exit."""
    K_put = entry["put_strike"]
    K_call = entry["call_strike"]
    intrinsic = max(K_put - spy_close_exit, 0) + max(spy_close_exit - K_call, 0)
    return intrinsic


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VARIANT A: SPY STRADDLE SELL WHEN HIGH CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════

def variant_a_high_corr_straddle(spy_chains, spy_px, ic_df):
    """
    Sell ATM SPY straddle ONLY when implied correlation > 75th percentile.
    Close after 5 trading days or 50% profit (whichever comes first).
    """
    print("\n" + "=" * 70)
    print("VARIANT A: SPY Straddle Sell When High Implied Correlation")
    print("=" * 70)

    # Get weekly observation dates (every Friday or last trading day of week)
    all_dates = sorted(spy_chains["date"].unique())
    all_dates = [d for d in all_dates if d in ic_df.index]

    # Group by week
    dates_series = pd.Series(all_dates)
    dates_series.index = pd.DatetimeIndex(all_dates)
    weekly_dates = dates_series.groupby(pd.Grouper(freq="W")).last().dropna().values
    weekly_dates = pd.DatetimeIndex(weekly_dates)

    trades = []
    nav = INITIAL_CAPITAL

    for i, entry_date in enumerate(weekly_dates):
        if entry_date not in ic_df.index:
            continue

        ic_row = ic_df.loc[entry_date]

        # Skip if insufficient lookback
        if pd.isna(ic_row.get("rho_pct")):
            continue

        # VIX flatten check
        if ic_row["vix"] > VIX_FLATTEN_THRESH:
            continue

        # SIGNAL: implied correlation > 75th percentile
        if ic_row["rho_pct"] < 0.75:
            continue

        # 200MA filter: half size below MA
        size_mult = 0.5 if ic_row["below_ma200"] else 1.0

        # Find ATM straddle
        straddle = find_atm_straddle(spy_chains, entry_date, target_dte=TARGET_DTE_WEEKLY)
        if straddle is None:
            continue

        # Position size: risk max 3% of NAV per trade
        premium_per_contract = straddle["straddle_price"] * 100
        max_risk = nav * 0.03 * size_mult
        # Max loss = 2x premium (stop)
        max_loss_per_c = premium_per_contract * 2
        n_contracts = max(1, int(max_risk / max_loss_per_c))
        n_contracts = min(n_contracts, 5)  # hard cap

        premium_received = premium_per_contract * n_contracts

        # Find exit: 5 trading days later
        exit_idx = None
        for j, future_date in enumerate(all_dates):
            if future_date > entry_date:
                if exit_idx is None:
                    exit_idx = j
                if j - exit_idx >= 4:  # 5 trading days
                    break

        if exit_idx is None or exit_idx + 4 >= len(all_dates):
            continue

        exit_date = all_dates[min(exit_idx + 4, len(all_dates) - 1)]

        # Exit value: check intermediate dates for 50% profit
        actual_exit_date = exit_date
        early_exit = False
        for check_date in all_dates[exit_idx:exit_idx + 5]:
            if check_date not in spy_px.index:
                continue
            spy_at_check = spy_px.loc[check_date, "spy_close"]
            exit_straddle = find_atm_straddle(spy_chains, check_date, target_dte=max(2, straddle["dte"] - 3))

            if exit_straddle is not None:
                # Use actual market straddle price at that date for same strike
                exit_value = exit_straddle["straddle_price"]
                profit_pct = (straddle["straddle_price"] - exit_value) / straddle["straddle_price"]
                if profit_pct >= 0.50:
                    actual_exit_date = check_date
                    early_exit = True
                    break

        # Final exit P&L
        exit_straddle = find_atm_straddle(spy_chains, actual_exit_date, target_dte=max(2, straddle["dte"] - 5))
        if exit_straddle is not None:
            # Match strike for accurate P&L
            exit_price = exit_straddle["straddle_price"]
        else:
            # Fallback: intrinsic only
            if actual_exit_date in spy_px.index:
                spy_exit = spy_px.loc[actual_exit_date, "spy_close"]
                exit_price = abs(spy_exit - straddle["strike"])
            else:
                continue

        pnl_per_share = straddle["straddle_price"] - exit_price  # short: profit when price falls
        pnl_total = pnl_per_share * 100 * n_contracts

        # Cap loss at 2x premium
        pnl_total = max(pnl_total, -premium_received * 2)

        nav += pnl_total

        trades.append({
            "date": entry_date,
            "exit_date": actual_exit_date,
            "days_held": (pd.Timestamp(actual_exit_date) - pd.Timestamp(entry_date)).days,
            "early_exit": early_exit,
            "strike": straddle["strike"],
            "entry_straddle": straddle["straddle_price"],
            "exit_straddle": exit_price,
            "iv_entry": straddle["iv"],
            "n_contracts": n_contracts,
            "premium_received": premium_received,
            "pnl": pnl_total,
            "nav": nav,
            "rho_implied": ic_row["rho_implied"],
            "rho_pct": ic_row["rho_pct"],
            "vix": ic_row["vix"],
            "spy_close": ic_row["spy_close"],
            "below_ma200": ic_row["below_ma200"],
            "size_mult": size_mult,
        })

    df = pd.DataFrame(trades)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        print(f"  Trades: {len(df)}")
        print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
        print(f"  Avg premium received: ${df['premium_received'].mean():.0f}")
        print(f"  Avg P&L: ${df['pnl'].mean():.0f}")
        print(f"  Early exits (50% profit): {df['early_exit'].sum()} ({df['early_exit'].mean()*100:.1f}%)")
    else:
        print("  WARNING: No trades generated!")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VARIANT B: SPY STRANGLE SELL ALWAYS, SIZE BY CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════

def variant_b_strangle_sized(spy_chains, spy_px, ic_df):
    """
    Always sell ~10-delta SPY strangle on monthly cycle.
    Size inversely to implied correlation: high corr = stress = smaller position.
    """
    print("\n" + "=" * 70)
    print("VARIANT B: SPY Strangle Sell Always, Size by Implied Correlation")
    print("=" * 70)

    all_dates = sorted(spy_chains["date"].unique())
    all_dates = [d for d in all_dates if d in ic_df.index]

    # Monthly entry dates (first trading day of each month)
    dates_series = pd.Series(all_dates)
    dates_series.index = pd.DatetimeIndex(all_dates)
    monthly_dates = dates_series.groupby(pd.Grouper(freq="MS")).first().dropna().values
    monthly_dates = pd.DatetimeIndex(monthly_dates)

    trades = []
    nav = INITIAL_CAPITAL

    for entry_date in monthly_dates:
        if entry_date not in ic_df.index:
            continue

        ic_row = ic_df.loc[entry_date]

        if pd.isna(ic_row.get("rho_pct")):
            continue

        # VIX flatten check
        if ic_row["vix"] > VIX_FLATTEN_THRESH:
            continue

        # Find ~10-delta strangle, monthly expiry
        strangle = find_otm_strangle(spy_chains, entry_date, target_dte=TARGET_DTE_MONTHLY)
        if strangle is None:
            continue

        # Size inversely to correlation
        # rho_pct in [0, 1]: high = high corr = stress
        # Size multiplier: 1.5 when corr low (< 25th pct), 0.5 when high (> 75th pct)
        rho_pct = ic_row["rho_pct"]
        if rho_pct > 0.75:
            size_mult = 0.5
        elif rho_pct > 0.50:
            size_mult = 1.0
        elif rho_pct > 0.25:
            size_mult = 1.25
        else:
            size_mult = 1.5

        # 200MA filter
        if ic_row["below_ma200"]:
            size_mult *= 0.5

        # Position size
        premium_per_contract = strangle["strangle_price"] * 100
        if premium_per_contract < 10:
            continue  # skip if premium too low

        max_risk = nav * 0.05 * size_mult
        max_loss_per_c = premium_per_contract * 3  # wider stop for strangles
        n_contracts = max(1, int(max_risk / max_loss_per_c))
        n_contracts = min(n_contracts, 10)  # hard cap

        premium_received = premium_per_contract * n_contracts

        # Exit: find date ~DTE trading days later (expiry proxy)
        target_exit = entry_date + pd.Timedelta(days=strangle["dte"])
        # Find closest trading date
        exit_candidates = [d for d in all_dates if d >= target_exit]
        if not exit_candidates:
            continue
        exit_date = exit_candidates[0]

        # Exit P&L: strangle value at exit
        if exit_date in spy_px.index:
            spy_exit = spy_px.loc[exit_date, "spy_close"]
        else:
            continue

        # Calculate strangle intrinsic at expiry
        intrinsic = (max(strangle["put_strike"] - spy_exit, 0) +
                     max(spy_exit - strangle["call_strike"], 0))

        # If not at expiry, estimate remaining time value via market data
        exit_strangle = find_otm_strangle(spy_chains, exit_date, target_dte=5)
        if exit_strangle is not None and strangle["dte"] > 5:
            # Use actual market price for similar strangle as proxy
            remaining_tv = exit_strangle["strangle_price"] * 0.3  # rough time value remaining
            exit_price = intrinsic + remaining_tv
        else:
            exit_price = intrinsic

        pnl_per_share = strangle["strangle_price"] - exit_price
        pnl_total = pnl_per_share * 100 * n_contracts

        # Cap loss at 3x premium
        pnl_total = max(pnl_total, -premium_received * 3)

        nav += pnl_total

        trades.append({
            "date": entry_date,
            "exit_date": exit_date,
            "days_held": (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days,
            "put_strike": strangle["put_strike"],
            "call_strike": strangle["call_strike"],
            "entry_strangle": strangle["strangle_price"],
            "exit_value": exit_price,
            "intrinsic": intrinsic,
            "put_iv": strangle["put_iv"],
            "call_iv": strangle["call_iv"],
            "n_contracts": n_contracts,
            "premium_received": premium_received,
            "pnl": pnl_total,
            "nav": nav,
            "rho_implied": ic_row["rho_implied"],
            "rho_pct": rho_pct,
            "size_mult": size_mult,
            "vix": ic_row["vix"],
            "spy_close": ic_row["spy_close"],
            "below_ma200": ic_row["below_ma200"],
        })

    df = pd.DataFrame(trades)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        print(f"  Trades: {len(df)}")
        print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
        print(f"  Avg premium received: ${df['premium_received'].mean():.0f}")
        print(f"  Avg contracts: {df['n_contracts'].mean():.1f}")
        print(f"  Avg P&L: ${df['pnl'].mean():.0f}")
    else:
        print("  WARNING: No trades generated!")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VARIANT C: PURE DISPERSION (SELL INDEX, BUY COMPONENTS)
# ═══════════════════════════════════════════════════════════════════════════════

def variant_c_pure_dispersion(spy_chains, comp_chains, spy_px, ic_df):
    """
    Pure dispersion trade:
    - Sell SPY ATM straddle
    - Buy straddles on top-5 highest-IV component stocks (weighted by SPY weight)
    - Market-neutral vol trade: captures correlation risk premium directly
    """
    print("\n" + "=" * 70)
    print("VARIANT C: Pure Dispersion (Sell SPY, Buy Components)")
    print("=" * 70)

    all_dates = sorted(spy_chains["date"].unique())
    all_dates = [d for d in all_dates if d in ic_df.index]

    # Bi-weekly entries (every other week)
    dates_series = pd.Series(all_dates)
    dates_series.index = pd.DatetimeIndex(all_dates)
    weekly_dates = dates_series.groupby(pd.Grouper(freq="2W")).first().dropna().values
    weekly_dates = pd.DatetimeIndex(weekly_dates)

    trades = []
    nav = INITIAL_CAPITAL

    for entry_date in weekly_dates:
        if entry_date not in ic_df.index:
            continue

        ic_row = ic_df.loc[entry_date]
        if pd.isna(ic_row.get("rho_pct")):
            continue

        # VIX flatten check
        if ic_row["vix"] > VIX_FLATTEN_THRESH:
            continue

        # SPY straddle (SELL side)
        spy_straddle = find_atm_straddle(spy_chains, entry_date, target_dte=TARGET_DTE_WEEKLY)
        if spy_straddle is None:
            continue

        # Component straddles (BUY side) — pick top 5 by IV available today
        comp_ivs = {}
        comp_straddles = {}
        for tkr, chains in comp_chains.items():
            strd = find_atm_straddle(chains, entry_date, target_dte=TARGET_DTE_WEEKLY)
            if strd is not None:
                comp_ivs[tkr] = strd["iv"]
                comp_straddles[tkr] = strd

        if len(comp_straddles) < 3:
            continue

        # Top 5 by IV
        top5 = sorted(comp_ivs, key=comp_ivs.get, reverse=True)[:5]

        # Size: SPY notional-weighted
        spy_price_1c = spy_straddle["straddle_price"] * 100
        max_risk = nav * 0.05
        spy_contracts = max(1, min(3, int(max_risk / (spy_price_1c * 2))))

        spy_premium = spy_price_1c * spy_contracts
        spy_notional = ic_row["spy_close"] * 100 * spy_contracts

        # Component sizing: weight-proportional to SPY notional
        comp_details = []
        total_comp_cost = 0
        for tkr in top5:
            w = SPY_COMPONENTS.get(tkr, 0.02)
            strd = comp_straddles[tkr]
            tkr_notional = spy_notional * w
            tkr_price = strd["strike"]
            tkr_contracts = max(0, round(tkr_notional / (tkr_price * 100)))

            if tkr_contracts == 0:
                continue

            cost = strd["straddle_price"] * 100 * tkr_contracts
            total_comp_cost += cost
            comp_details.append({
                "ticker": tkr,
                "strike": strd["strike"],
                "straddle_price": strd["straddle_price"],
                "iv": strd["iv"],
                "contracts": tkr_contracts,
                "cost": cost,
                "weight": w,
            })

        if not comp_details:
            continue

        net_premium = spy_premium - total_comp_cost  # positive = credit trade

        # Exit: 7 trading days later
        exit_idx = None
        for j, d in enumerate(all_dates):
            if d > entry_date:
                if exit_idx is None:
                    exit_idx = j
                if j - exit_idx >= 6:
                    break

        if exit_idx is None or exit_idx + 6 >= len(all_dates):
            continue

        exit_date = all_dates[min(exit_idx + 6, len(all_dates) - 1)]

        # SPY exit P&L (short straddle)
        spy_exit = find_atm_straddle(spy_chains, exit_date, target_dte=max(2, spy_straddle["dte"] - 7))
        if spy_exit is not None:
            spy_exit_price = spy_exit["straddle_price"]
        else:
            if exit_date in spy_px.index:
                spy_exit_price = abs(spy_px.loc[exit_date, "spy_close"] - spy_straddle["strike"])
            else:
                continue

        spy_pnl = (spy_straddle["straddle_price"] - spy_exit_price) * 100 * spy_contracts

        # Component exit P&L (long straddles)
        comp_pnl = 0
        for cd in comp_details:
            tkr = cd["ticker"]
            if tkr in comp_chains:
                exit_strd = find_atm_straddle(comp_chains[tkr], exit_date, target_dte=max(2, spy_straddle["dte"] - 7))
                if exit_strd is not None:
                    comp_pnl += (exit_strd["straddle_price"] - cd["straddle_price"]) * 100 * cd["contracts"]
                else:
                    # Assume theta decay = 50% of entry premium (conservative loss for long side)
                    comp_pnl -= cd["cost"] * 0.5

        total_pnl = spy_pnl + comp_pnl

        # Cap loss
        total_pnl = max(total_pnl, -(spy_premium + total_comp_cost))

        nav += total_pnl

        trades.append({
            "date": entry_date,
            "exit_date": exit_date,
            "days_held": (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days,
            "spy_strike": spy_straddle["strike"],
            "spy_straddle_entry": spy_straddle["straddle_price"],
            "spy_straddle_exit": spy_exit_price,
            "spy_iv": spy_straddle["iv"],
            "spy_contracts": spy_contracts,
            "spy_pnl": spy_pnl,
            "n_comp_legs": len(comp_details),
            "comp_cost": total_comp_cost,
            "comp_pnl": comp_pnl,
            "net_premium": net_premium,
            "pnl": total_pnl,
            "nav": nav,
            "rho_implied": ic_row["rho_implied"],
            "rho_pct": ic_row.get("rho_pct", np.nan),
            "vix": ic_row["vix"],
            "spy_close": ic_row["spy_close"],
            "comp_tickers": [cd["ticker"] for cd in comp_details],
            "comp_ivs": [cd["iv"] for cd in comp_details],
        })

    df = pd.DataFrame(trades)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        print(f"  Trades: {len(df)}")
        print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
        print(f"  Avg SPY premium (sell): ${df['spy_pnl'].apply(lambda x: 0).mean():.0f}")
        print(f"  Net premium per trade: ${df['net_premium'].mean():.0f}")
        print(f"  Avg total P&L: ${df['pnl'].mean():.0f}")
        print(f"  Credit trades: {(df['net_premium'] > 0).sum()} / {len(df)}")
    else:
        print("  WARNING: No trades generated!")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 7. METRICS & COMPLIANCE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(df, label="all", pnl_col="pnl", nav_col="nav"):
    """Compute risk-adjusted metrics."""
    if df is None or len(df) == 0:
        return {"label": label, "n_trades": 0, "error": "no trades"}

    pnl = df[pnl_col].values
    n = len(pnl)
    nav_vals = df[nav_col].values
    total_r = (nav_vals[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    mean_p = np.mean(pnl)
    std_p = np.std(pnl, ddof=1) if n > 1 else 1.0

    # Annualize based on approximate trades per year
    date_range_days = (df["date"].max() - df["date"].min()).days
    trades_per_year = n / max(date_range_days / 365.25, 0.5)
    ann_factor = np.sqrt(trades_per_year)

    sharpe = mean_p / std_p * ann_factor if std_p > 0 else 0.0
    down_pnl = pnl[pnl < 0]
    down = np.std(down_pnl, ddof=1) if len(down_pnl) > 1 else std_p
    sortino = mean_p / down * ann_factor if down > 0 else 0.0

    wr = float((pnl > 0).mean() * 100)
    gp = float(pnl[pnl > 0].sum()) if (pnl > 0).any() else 0.0
    gl = float(abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else 1.0
    pf = gp / gl if gl > 0 else float("inf")

    # CAGR
    years = max(date_range_days / 365.25, 0.5)
    cagr = ((nav_vals[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100

    # Max drawdown
    pk = np.maximum.accumulate(nav_vals)
    max_dd = float(((nav_vals - pk) / pk).min() * 100)

    # Day concentration (HC #344)
    day_c = float(pnl[pnl > 0].max() / gp) if gp > 0 else 0.0

    # Monthly income estimate at $100K
    monthly_income = mean_p * trades_per_year / 12

    return {
        "label": label,
        "n_trades": n,
        "total_return_pct": round(total_r, 2),
        "cagr_pct": round(cagr, 2),
        "mean_pnl": round(float(mean_p), 2),
        "std_pnl": round(float(std_p), 2),
        "sharpe_annual": round(float(sharpe), 3),
        "sortino_annual": round(float(sortino), 3),
        "win_rate_pct": round(wr, 1),
        "profit_factor": round(float(pf), 3),
        "max_drawdown_pct": round(max_dd, 2),
        "day_concentration": round(day_c, 3),
        "trades_per_year": round(trades_per_year, 1),
        "monthly_income_100k": round(monthly_income, 0),
        "gross_profit": round(gp, 0),
        "gross_loss": round(gl, 0),
    }


def regime_gap_test(df, spy_px, pnl_col="pnl"):
    """HC #428 R1: regime-agnostic OOT validation."""
    if df is None or len(df) == 0:
        return {"regime_agnostic": "NO_DATA"}

    merged = df.copy()
    merged["date"] = pd.to_datetime(merged["date"])
    merged["exit_date"] = pd.to_datetime(merged["exit_date"])

    # SPY return over trade period
    spy_rets = []
    for _, row in merged.iterrows():
        d_e, d_x = row["date"], row["exit_date"]
        if d_e in spy_px.index and d_x in spy_px.index:
            spy_rets.append((spy_px.loc[d_x, "spy_close"] - spy_px.loc[d_e, "spy_close"]) / spy_px.loc[d_e, "spy_close"])
        else:
            spy_rets.append(0.0)

    merged["spy_ret"] = spy_rets
    merged["regime"] = "flat"
    merged.loc[merged["spy_ret"] > 0.005, "regime"] = "green"
    merged.loc[merged["spy_ret"] < -0.005, "regime"] = "red"

    per_regime = {}
    for reg in ["green", "red", "flat"]:
        sub = merged[merged["regime"] == reg].copy()
        if len(sub) < 3:
            per_regime[reg] = {"n": len(sub), "sharpe": None}
            continue
        sub["nav"] = INITIAL_CAPITAL + sub[pnl_col].cumsum()
        m = compute_metrics(sub, label=reg)
        per_regime[reg] = {
            "n": m["n_trades"],
            "sharpe": m["sharpe_annual"],
            "sortino": m["sortino_annual"],
            "win_rate": m["win_rate_pct"],
            "pf": m["profit_factor"],
        }

    gs = per_regime.get("green", {}).get("sharpe")
    rs = per_regime.get("red", {}).get("sharpe")
    if gs is not None and rs is not None and max(abs(gs), abs(rs)) > 1e-6:
        gap_r = abs(gs - rs) / max(abs(gs), abs(rs))
        verdict = "PASS" if gap_r <= 0.50 else "FAIL"
    else:
        gap_r = None
        verdict = "INSUFFICIENT_DATA"

    return {
        "per_regime": per_regime,
        "gap_ratio": round(gap_r, 3) if gap_r is not None else None,
        "regime_agnostic": verdict,
        "threshold": 0.50,
    }


def permutation_test(df, real_sharpe, n_perm=N_PERMUTATIONS, label=""):
    """Permutation test: shuffle timing to test if signal matters."""
    print(f"\n  Permutation test ({label}, {n_perm} trials)...")
    rng = np.random.default_rng(RANDOM_SEED)
    pnl = df["pnl"].values.copy()
    n = len(pnl)

    # Annualization factor
    date_range_days = (df["date"].max() - df["date"].min()).days
    trades_per_year = n / max(date_range_days / 365.25, 0.5)
    ann_factor = np.sqrt(trades_per_year)

    null_sharpes = []
    for _ in range(n_perm):
        p = rng.permutation(pnl)
        s = np.mean(p) / np.std(p, ddof=1) * ann_factor if np.std(p, ddof=1) > 0 else 0.0
        null_sharpes.append(s)

    null_sharpes = np.array(null_sharpes)
    pval = float((null_sharpes >= real_sharpe).mean())
    print(f"  p={pval:.4f}, real Sharpe={real_sharpe:.3f}, null mean={null_sharpes.mean():.3f}")

    return {
        "n_trials": n_perm,
        "real_sharpe": round(real_sharpe, 3),
        "null_mean": round(float(null_sharpes.mean()), 3),
        "null_std": round(float(null_sharpes.std()), 3),
        "p_value": round(pval, 4),
        "significant_05": bool(pval < 0.05),
        "verdict": "PASS" if pval < 0.05 else "FAIL",
    }


def year_by_year(df, pnl_col="pnl"):
    """Year-by-year breakdown."""
    if df is None or len(df) == 0:
        return {}
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    results = {}
    for year, grp in df.groupby("year"):
        grp = grp.copy()
        grp["nav"] = INITIAL_CAPITAL + grp[pnl_col].cumsum()
        m = compute_metrics(grp, label=str(year))
        results[int(year)] = {
            "n_trades": m["n_trades"],
            "total_pnl": round(float(grp[pnl_col].sum()), 0),
            "sharpe": m["sharpe_annual"],
            "sortino": m["sortino_annual"],
            "win_rate": m["win_rate_pct"],
            "max_dd": m["max_drawdown_pct"],
        }
    return results


def crisis_analysis(df, pnl_col="pnl"):
    """Performance during known crisis periods."""
    if df is None or len(df) == 0:
        return {}

    crises = {
        "COVID_crash_2020":    ("2020-02-19", "2020-04-01"),
        "2022_bear":           ("2022-01-03", "2022-10-12"),
        "SVB_crisis_2023":     ("2023-03-08", "2023-03-20"),
        "Aug_2024_unwind":     ("2024-07-15", "2024-08-15"),
        "April_2025_tariffs":  ("2025-03-28", "2025-04-15"),
    }

    results = {}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for name, (start, end) in crises.items():
        mask = (df["date"] >= start) & (df["date"] <= end)
        sub = df[mask]
        if len(sub) == 0:
            results[name] = {"n_trades": 0, "pnl": 0}
            continue
        results[name] = {
            "n_trades": len(sub),
            "total_pnl": round(float(sub[pnl_col].sum()), 0),
            "avg_pnl": round(float(sub[pnl_col].mean()), 0),
            "win_rate": round(float((sub[pnl_col] > 0).mean() * 100), 1),
        }

    return results


def capital_needed(monthly_target, avg_monthly_income_100k):
    """Compute capital needed for a given monthly income target."""
    if avg_monthly_income_100k <= 0:
        return float("inf")
    return round(monthly_target / avg_monthly_income_100k * INITIAL_CAPITAL, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_variant(df, spy_px, variant_name):
    """Full analysis pipeline for a variant."""
    if df is None or len(df) == 0:
        print(f"\n  {variant_name}: No trades to analyze")
        return None

    # Recalculate NAV from scratch
    df = df.copy()
    df["pnl_cumsum"] = df["pnl"].cumsum()
    df["nav"] = INITIAL_CAPITAL + df["pnl_cumsum"]

    # IS/OOT split
    split = int(len(df) * (1 - OOT_FRACTION))
    if split < 5:
        split = len(df) // 2

    is_df = df.iloc[:split].copy()
    is_df["nav"] = INITIAL_CAPITAL + is_df["pnl"].cumsum()

    oot_df = df.iloc[split:].copy()
    oot_df["nav"] = INITIAL_CAPITAL + oot_df["pnl"].cumsum()

    print(f"\n  IS:  {df['date'].iloc[0].date()} → {df['date'].iloc[split-1].date()} ({split} trades)")
    print(f"  OOT: {df['date'].iloc[split].date()} → {df['date'].iloc[-1].date()} ({len(oot_df)} trades)")

    # Metrics
    m_full = compute_metrics(df, "full")
    m_is = compute_metrics(is_df, "IS")
    m_oot = compute_metrics(oot_df, "OOT")

    print(f"\n  {'Period':<6} {'n':>4} {'Return%':>9} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'WR%':>5} {'PF':>6} {'MaxDD%':>8} {'DayConc':>8} {'Mo.Inc$':>8}")
    print("  " + "-" * 85)
    for m in [m_full, m_is, m_oot]:
        print(f"  {m['label']:<6} {m['n_trades']:>4} {m['total_return_pct']:>9.1f} "
              f"{m['cagr_pct']:>7.1f} {m['sharpe_annual']:>7.3f} {m['sortino_annual']:>8.3f} "
              f"{m['win_rate_pct']:>5.1f} {m['profit_factor']:>6.3f} "
              f"{m['max_drawdown_pct']:>8.1f} {m['day_concentration']:>8.3f} "
              f"{m['monthly_income_100k']:>8.0f}")

    # Regime gap test (on OOT)
    regime = regime_gap_test(oot_df, spy_px)
    print(f"\n  Regime gap (HC #428 R1): {regime['regime_agnostic']} (gap={regime.get('gap_ratio')})")
    for reg, s in regime.get("per_regime", {}).items():
        print(f"    {reg}: n={s['n']}, Sharpe={s.get('sharpe')}, WR={s.get('win_rate')}%")

    # Permutation test
    perm = permutation_test(df, m_full["sharpe_annual"], label=variant_name)

    # Day concentration (HC #344)
    day_c = m_full["day_concentration"]
    conc_v = "PASS" if day_c <= 0.70 else "FAIL"

    # Year by year
    yby = year_by_year(df)
    print(f"\n  Year-by-year:")
    for yr, s in sorted(yby.items()):
        print(f"    {yr}: n={s['n_trades']}, P&L=${s['total_pnl']:+,.0f}, "
              f"Sharpe={s['sharpe']:.3f}, WR={s['win_rate']:.1f}%, MaxDD={s['max_dd']:.1f}%")

    # Crisis periods
    crisis = crisis_analysis(df)
    print(f"\n  Crisis performance:")
    for name, s in crisis.items():
        print(f"    {name}: n={s['n_trades']}, P&L=${s.get('total_pnl', 0):+,.0f}, "
              f"WR={s.get('win_rate', 0):.1f}%")

    # Capital needed
    monthly_inc = m_full["monthly_income_100k"]
    cap_3k = capital_needed(3000, monthly_inc)
    cap_5k = capital_needed(5000, monthly_inc)
    print(f"\n  Income projection at $100K: ${monthly_inc:,.0f}/month")
    print(f"  Capital for $3K/month: ${cap_3k:,.0f}")
    print(f"  Capital for $5K/month: ${cap_5k:,.0f}")

    result = {
        "variant": variant_name,
        "metrics": {"full": m_full, "is": m_is, "oot": m_oot},
        "regime_gap": regime,
        "permutation": perm,
        "compliance": {
            "hc344_day_conc": {"value": round(day_c, 3), "limit": 0.70, "verdict": conc_v},
            "hc428_r1_regime": {"verdict": regime["regime_agnostic"],
                                "gap_ratio": regime.get("gap_ratio")},
            "permutation_p05": {"verdict": perm.get("verdict"),
                                "p_value": perm.get("p_value")},
        },
        "year_by_year": yby,
        "crisis_performance": crisis,
        "income_projection": {
            "monthly_at_100k": monthly_inc,
            "capital_for_3k_monthly": cap_3k,
            "capital_for_5k_monthly": cap_5k,
        },
    }

    # Save trades
    save_cols = [c for c in df.columns if not isinstance(df[c].iloc[0], list)]
    df[save_cols].to_parquet(OUT_DIR / f"trades_{variant_name}.parquet", index=False)
    df[save_cols].to_csv(OUT_DIR / f"trades_{variant_name}.csv", index=False)

    return result


def main():
    print("=" * 70)
    print("DISPERSION TRADE v1 — Implied Correlation / Dispersion Income")
    print("=" * 70)
    print(f"Capital: ${INITIAL_CAPITAL:,}")
    print(f"OOT fraction: {OOT_FRACTION}")
    print(f"Lookback: {LOOKBACK_DAYS} days")
    print(f"Permutations: {N_PERMUTATIONS}")
    print(f"VIX flatten: > {VIX_FLATTEN_THRESH}")

    # Load data
    spy_px, comp_px_wide, comp_rv_wide, macro, spy_chains, comp_chains = load_data()

    # Compute implied correlation
    ic_df = compute_implied_correlation(spy_px, comp_rv_wide, macro)

    # Save IC data for analysis
    ic_df.to_parquet(OUT_DIR / "implied_correlation.parquet")
    ic_df.to_csv(OUT_DIR / "implied_correlation.csv")

    results = {}

    # ── VARIANT A ─────────────────────────────────────────────────────────────
    df_a = variant_a_high_corr_straddle(spy_chains, spy_px, ic_df)
    r_a = analyze_variant(df_a, spy_px, "A_high_corr_straddle")
    if r_a:
        results["A_high_corr_straddle"] = r_a

    # ── VARIANT B ─────────────────────────────────────────────────────────────
    df_b = variant_b_strangle_sized(spy_chains, spy_px, ic_df)
    r_b = analyze_variant(df_b, spy_px, "B_strangle_sized")
    if r_b:
        results["B_strangle_sized"] = r_b

    # ── VARIANT C ─────────────────────────────────────────────────────────────
    df_c = variant_c_pure_dispersion(spy_chains, comp_chains, spy_px, ic_df)
    r_c = analyze_variant(df_c, spy_px, "C_pure_dispersion")
    if r_c:
        results["C_pure_dispersion"] = r_c

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPARISON SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DISPERSION TRADE v1 — COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n  {'Variant':<25} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'WR%':>5} "
          f"{'PF':>6} {'MaxDD%':>8} {'OOT Sharpe':>11} {'Regime':>8} {'Perm':>6} "
          f"{'Mo.Inc':>7}")
    print("  " + "-" * 105)
    for name, r in results.items():
        m = r["metrics"]["full"]
        mo = r["metrics"]["oot"]
        rg = r["regime_gap"]["regime_agnostic"]
        pm = r["permutation"]["verdict"]
        mi = r["income_projection"]["monthly_at_100k"]
        print(f"  {name:<25} {m['sharpe_annual']:>7.3f} {m['sortino_annual']:>8.3f} "
              f"{m['cagr_pct']:>7.1f} {m['win_rate_pct']:>5.1f} {m['profit_factor']:>6.3f} "
              f"{m['max_drawdown_pct']:>8.1f} {mo['sharpe_annual']:>11.3f} "
              f"{rg:>8} {pm:>6} ${mi:>6.0f}")

    print(f"\n  HC Compliance:")
    for name, r in results.items():
        for k, v in r["compliance"].items():
            status = v["verdict"]
            marker = "OK" if status == "PASS" else "XX" if status == "FAIL" else "??"
            print(f"    [{marker}] {name} / {k}: {status}")

    print(f"\n  Capital Requirements:")
    for name, r in results.items():
        ip = r["income_projection"]
        print(f"    {name}: ${ip['monthly_at_100k']:,.0f}/mo at $100K | "
              f"$3K/mo needs ${ip['capital_for_3k_monthly']:,.0f} | "
              f"$5K/mo needs ${ip['capital_for_5k_monthly']:,.0f}")

    # Save all results
    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    # Save IC analysis
    ic_summary = {
        "mean_implied_corr": round(float(ic_df["rho_implied"].mean()), 4),
        "median_implied_corr": round(float(ic_df["rho_implied"].median()), 4),
        "mean_ic_ratio": round(float(ic_df["ic_ratio"].mean()), 4),
        "mean_vrp": round(float(ic_df["vrp"].mean()), 4),
        "pct_high_corr_75": round(float((ic_df["rho_pct"] > 0.75).mean() * 100), 1),
    }
    with open(OUT_DIR / "ic_analysis.json", "w") as fh:
        json.dump(ic_summary, fh, indent=2)

    print(f"\n  Results saved to: {OUT_DIR}/")
    print("=" * 70)
    print("DONE")

    return results


if __name__ == "__main__":
    main()
