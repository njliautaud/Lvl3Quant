"""
Dispersion Trading v1 — Correlation Risk Premium Research
==========================================================
Tests TWO dispersion variants:
  A) CLASSIC DISPERSION: short SPY straddle, long single-name straddles
  B) REVERSE DISPERSION: long SPY straddle, short single-name straddles

Pre-analysis findings:
  - SPY IV = 17%, single-name basket IV = 30-35% → index is CHEAP vs components
  - Classic dispersion is a DEBIT trade (pays more for singles than receives from SPY)
  - Reverse dispersion COLLECTS the vol premium (single-name IV overpriced vs SPY)
  - Both tested to understand the correlation risk premium direction in this dataset

Sizing: NOTIONAL-WEIGHTED (matches academic and institutional practice)
  - SPY notional = SPY_price × 100 × n_contracts
  - Single-name i notional = component_weight_i × SPY_notional
  - Single-name contracts = notional_i / (single_price × 100)
  - Allow 0 contracts when single name is too expensive (no min=1 forcing)
  - This properly replicates the correlation structure of SPX

Hold: 1 week (mark to next weekly snapshot)
P&L: Market mid prices at entry + exit (actual data, not BS approximation)

HC compliance:
  - HC #428 R1: regime-agnostic OOT gap test
  - HC #344: day-concentration ≤ 0.70
  - HC #0: last 30% OOT
  - 100-trial permutation test
"""

import sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
CHAINS_DIR  = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/options_real/chains")
PRICES_PATH = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/prices.parquet")
MACRO_PATH  = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/macro.parquet")
OUT_DIR     = Path("/home/jupiter/Lvl3Quant/output/dispersion_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL   = 100_000
OOT_FRACTION      = 0.30
DTE_MIN, DTE_MAX  = 11, 16
TARGET_DTE        = 14
N_SINGLE_NAMES    = 8
MAX_NAV_PCT       = 0.10     # max NAV at risk per cycle (SPY straddle premium)
MAX_SPY_CONTRACTS = 6        # hard cap
N_PERMUTATIONS    = 100
RANDOM_SEED       = 42

# SPY component weights (approximate 2024-25, top 27 names we have data for)
SPY_WEIGHTS = {
    "AAPL": 0.070, "MSFT": 0.065, "NVDA": 0.060, "AMZN": 0.040,
    "META": 0.025, "GOOGL": 0.022, "TSLA": 0.018, "UNH": 0.014,
    "LLY":  0.013, "JPM":  0.013, "V":    0.011, "XOM":  0.011,
    "COST": 0.010, "MA":   0.009, "HD":   0.008, "PG":   0.007,
    "JNJ":  0.007, "ABBV": 0.007, "BAC":  0.006, "WMT":  0.006,
    "CVX":  0.005, "KO":   0.005, "GS":   0.004, "ADBE": 0.004,
    "MS":   0.004, "CRM":  0.004, "AMD":  0.003,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def bs_straddle_vec(S, K, sigma, T, r=0.045):
    S, K, sigma, T = [np.asarray(x, dtype=float) for x in (S, K, sigma, T)]
    out = np.maximum(np.abs(S - K), 0.0)
    v   = (T > 1e-6) & (sigma > 1e-6) & (S > 0) & (K > 0)
    if v.any():
        Sv, Kv, sv, Tv = S[v], K[v], sigma[v], T[v]
        d1 = (np.log(Sv/Kv) + (r + 0.5*sv**2)*Tv) / (sv*np.sqrt(Tv))
        d2 = d1 - sv*np.sqrt(Tv)
        out[v] = (Sv*norm.cdf(d1) - Kv*np.exp(-r*Tv)*norm.cdf(d2) +
                  Kv*np.exp(-r*Tv)*norm.cdf(-d2) - Sv*norm.cdf(-d1))
    return np.maximum(out, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD + PRECOMPUTE ATM STRADDLES
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_precompute():
    print("\n[1] Loading options chains and pre-computing ATM straddles...")
    t0 = time.time()

    # Prices
    prices_all = pd.read_parquet(PRICES_PATH)
    prices_all["date"] = pd.to_datetime(prices_all["date"])
    spy_px = (prices_all[prices_all["ticker"] == "SPY"]
              .set_index("date")[["close"]].sort_index())

    # Per-ticker price at each date (for notional calc)
    px_wide = prices_all.pivot_table(index="date", columns="ticker", values="close")

    macro = pd.read_parquet(MACRO_PATH)
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date")[["vix"]].sort_index()

    def load_chains(tkr):
        f = CHAINS_DIR / f"{tkr}.parquet"
        if not f.exists():
            return None
        try:
            df = pd.read_parquet(f, columns=[
                "date","expiration","strike","type","vol","delta","vega","dte","mid"
            ])
            df["date"]       = pd.to_datetime(df["date"])
            df["expiration"] = pd.to_datetime(df["expiration"])
            return df[df["dte"].between(DTE_MIN, DTE_MAX) & (df["vol"] > 0) & (df["vol"] < 3.0)]
        except Exception:
            return None

    def pick_atm(df, key_cols):
        """Vectorized: per group, pick expiry closest to TARGET_DTE, then call with delta~0.5."""
        c = df[df["type"] == "c"].copy()
        c["dte_dist"]   = (c["dte"] - TARGET_DTE).abs()
        c["delta_dist"] = (c["delta"].abs() - 0.50).abs()
        mn1 = c.groupby(key_cols)["dte_dist"].transform("min")
        c   = c[c["dte_dist"] == mn1]
        mn2 = c.groupby(key_cols)["delta_dist"].transform("min")
        c   = c[c["delta_dist"] == mn2]
        return c.groupby(key_cols).first().reset_index()

    # SPY
    spy_raw = load_chains("SPY")
    spy_raw["ticker"] = "SPY"
    spy_atm = pick_atm(spy_raw, ["ticker", "date"])
    spy_puts = (spy_raw[spy_raw["type"] == "p"]
                .groupby(["date", "strike"]).agg(put_mid=("mid", "mean")).reset_index())
    spy_atm = spy_atm.merge(spy_puts, on=["date","strike"], how="left")
    spy_atm["put_mid"].fillna(spy_atm["mid"], inplace=True)
    spy_atm["straddle_price"] = spy_atm["mid"] + spy_atm["put_mid"]
    spy_atm["atm_iv"]         = spy_atm["vol"]
    print(f"  SPY ATM: {len(spy_atm)} obs | avg straddle=${spy_atm['straddle_price'].mean():.2f}, IV={spy_atm['atm_iv'].mean():.3f}")

    # Single names
    sn_frames, avail = [], []
    for f in sorted(CHAINS_DIR.glob("*.parquet")):
        tkr = f.stem
        if tkr == "SPY":
            continue
        df = load_chains(tkr)
        if df is not None and len(df) > 0:
            df["ticker"] = tkr
            sn_frames.append(df)
            avail.append(tkr)

    sn_raw = pd.concat(sn_frames, ignore_index=True)
    sn_atm = pick_atm(sn_raw, ["ticker", "date"])
    sn_puts = (sn_raw[sn_raw["type"] == "p"]
               .groupby(["ticker","date","strike"]).agg(put_mid=("mid","mean")).reset_index())
    sn_atm = sn_atm.merge(sn_puts, on=["ticker","date","strike"], how="left")
    sn_atm["put_mid"].fillna(sn_atm["mid"], inplace=True)
    sn_atm["straddle_price"] = sn_atm["mid"] + sn_atm["put_mid"]
    sn_atm["atm_iv"]         = sn_atm["vol"]
    print(f"  SN ATM: {len(sn_atm)} obs across {sn_atm['ticker'].nunique()} tickers | avg straddle=${sn_atm['straddle_price'].mean():.2f}, IV={sn_atm['atm_iv'].mean():.3f}")
    print(f"  Load time: {time.time()-t0:.1f}s")

    return spy_atm, sn_atm, spy_px, px_wide, macro, avail


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BUILD CYCLES
# ═══════════════════════════════════════════════════════════════════════════════

def build_cycles(spy_atm, sn_atm, spy_px, px_wide, avail, mode="reverse"):
    """
    mode='classic':  SHORT SPY straddle, LONG single-name straddles
    mode='reverse':  LONG SPY straddle, SHORT single-name straddles

    Sizing: NOTIONAL-WEIGHTED
      - SPY contracts: floor(NAV × MAX_NAV_PCT / spy_price_per_contract)
      - SPY notional = SPY_price × 100 × spy_contracts
      - SN notional_i = sn_weight_i × SPY_notional
      - SN contracts_i = round(SN_notional_i / (sn_price_i × 100))
      - Allow 0 contracts (no min=1 floor)

    P&L:
      Market mid at entry and exit (next week snapshot of SAME-UNDERLYING ATM straddle)
    """
    print(f"\n[2] Building cycles (mode={mode})...")
    t0 = time.time()

    # Component pool
    avail_set = set(avail)
    pool = {t: w for t, w in SPY_WEIGHTS.items() if t in avail_set}
    if len(pool) < 3:
        pool = {t: 1.0 for t in avail}
    tw = sum(pool.values())
    pool = {t: w/tw for t, w in pool.items()}
    top_n = sorted(pool, key=pool.get, reverse=True)[:N_SINGLE_NAMES]
    top_w = {t: pool[t] for t in top_n}
    tw2 = sum(top_w.values())
    top_w = {t: w/tw2 for t, w in top_w.items()}
    print(f"  Mode: {mode} | Components: {top_n}")

    # Date sequence
    spy_d = spy_atm.copy()
    spy_d["date"] = pd.to_datetime(spy_d["date"])
    obs_dates = sorted(spy_d["date"].unique())
    dates_df = pd.DataFrame({
        "date":      pd.Series(obs_dates),
        "exit_date": pd.Series(obs_dates).shift(-1),
    }).dropna()
    dates_df = dates_df.astype({"date": "datetime64[ns]", "exit_date": "datetime64[ns]"})

    # SPY entry
    spy_e = spy_d.rename(columns={
        "strike": "spy_K", "atm_iv": "spy_iv_e",
        "straddle_price": "spy_strd_e", "dte": "spy_dte_e"
    })[["date","spy_K","spy_iv_e","spy_strd_e","spy_dte_e"]]

    # SPY exit (next-week's ATM straddle on same underlying)
    spy_x = spy_d.rename(columns={
        "atm_iv": "spy_iv_x",
        "straddle_price": "spy_strd_x",
        "dte": "spy_dte_x",
        "strike": "spy_K_x",
    })[["date","spy_iv_x","spy_strd_x","spy_dte_x","spy_K_x"]]
    spy_x = spy_x.rename(columns={"date": "exit_date"})

    # SPY spot
    spy_spot   = spy_px["close"].reset_index().rename(columns={"close":"spy_S"})
    spy_spot_x = spy_spot.rename(columns={"date":"exit_date","spy_S":"spy_S_x"})

    # Main cycles
    cyc = (dates_df
           .merge(spy_e,      on="date",      how="inner")
           .merge(spy_x,      on="exit_date", how="inner")
           .merge(spy_spot,   on="date",      how="left")
           .merge(spy_spot_x, on="exit_date", how="left"))
    cyc["spy_S"].fillna(cyc["spy_K"], inplace=True)
    cyc["spy_S_x"].fillna(cyc["spy_K_x"], inplace=True)
    cyc["days_held"] = (cyc["exit_date"] - cyc["date"]).dt.days

    # SPY sizing
    spy_price_1c = (cyc["spy_strd_e"] * 100).clip(lower=1.0)
    spy_contr = np.minimum(
        MAX_SPY_CONTRACTS,
        np.maximum(1, np.floor(INITIAL_CAPITAL * MAX_NAV_PCT / spy_price_1c))
    ).astype(int)
    cyc["spy_contracts"]  = spy_contr
    cyc["spy_premium_1c"] = spy_price_1c
    cyc["spy_notional"]   = cyc["spy_S"] * 100 * spy_contr

    # SPY P&L: (strd_entry - strd_exit) per share × 100 × contracts
    # For mode=classic: short straddle → gain when price falls
    # For mode=reverse: long straddle → gain when price rises
    spy_raw_pnl_pc = (cyc["spy_strd_e"] - cyc["spy_strd_x"]) * 100  # short = positive when decay
    if mode == "reverse":
        spy_raw_pnl_pc = -spy_raw_pnl_pc  # long = positive when price rises
    # Cap at ±3× entry premium
    cap_spy = spy_price_1c * 3
    cyc["spy_pnl_per_c"] = spy_raw_pnl_pc.clip(-cap_spy, cap_spy)
    cyc["spy_pnl_total"]  = cyc["spy_pnl_per_c"] * spy_contr

    # Single-name legs
    sn_d = sn_atm[sn_atm["ticker"].isin(top_n)].copy()
    sn_d["date"] = pd.to_datetime(sn_d["date"])

    per_ticker_frames = []
    for tkr in top_n:
        w = top_w[tkr]
        t_df = sn_d[sn_d["ticker"] == tkr]
        t_e = (t_df[["date","strike","atm_iv","straddle_price","dte"]]
               .rename(columns={"strike":"sn_K","atm_iv":"sn_iv_e",
                                "straddle_price":"sn_strd_e","dte":"sn_dte_e"}))
        t_x = (t_df[["date","atm_iv","straddle_price","dte"]]
               .rename(columns={"atm_iv":"sn_iv_x","straddle_price":"sn_strd_x","dte":"sn_dte_x",
                                 "date":"exit_date"}))
        t_x = t_df[["date","atm_iv","straddle_price","dte"]].copy()
        t_x = t_x.rename(columns={"date":"exit_date","atm_iv":"sn_iv_x",
                                   "straddle_price":"sn_strd_x","dte":"sn_dte_x"})
        tc = (dates_df
              .merge(t_e, on="date",      how="inner")
              .merge(t_x, on="exit_date", how="inner"))
        tc["ticker"] = tkr
        tc["weight"] = w
        per_ticker_frames.append(tc)

    sn_cyc = pd.concat(per_ticker_frames, ignore_index=True)

    # Merge spy sizing
    sn_cyc = sn_cyc.merge(
        cyc[["date","spy_contracts","spy_notional","spy_S"]],
        on="date", how="inner"
    )

    # Notional-weighted contracts
    sn_cyc["sn_notional_i"] = sn_cyc["spy_notional"] * sn_cyc["weight"]
    # Get single-name spot price for notional calc (use ATM strike as proxy)
    sn_cyc["sn_price_1c"]   = (sn_cyc["sn_strd_e"] * 100).clip(lower=0.01)
    # Contracts: notional_i / (single_stock_price × 100)
    # single_stock_price ≈ sn_K (ATM strike ≈ current price)
    sn_cyc["sn_stock_price"] = sn_cyc["sn_K"]
    sn_cyc["sn_contracts"]   = np.maximum(0, np.round(
        sn_cyc["sn_notional_i"] / (sn_cyc["sn_stock_price"] * 100)
    )).astype(int)

    # P&L per leg
    sn_raw_pnl_pc = (sn_cyc["sn_strd_x"] - sn_cyc["sn_strd_e"]) * 100  # long = gain when rises
    if mode == "classic":
        pass  # long singles: gain when straddle price rises (big moves)
    else:
        sn_raw_pnl_pc = -sn_raw_pnl_pc  # short singles in reverse mode

    cap_sn = sn_cyc["sn_price_1c"] * 3
    sn_cyc["sn_pnl_per_c"]  = sn_raw_pnl_pc.clip(-cap_sn, cap_sn)
    sn_cyc["sn_pnl_total"]  = sn_cyc["sn_pnl_per_c"] * sn_cyc["sn_contracts"]
    sn_cyc["sn_premium_1c"] = sn_cyc["sn_price_1c"]
    sn_cyc["sn_premium"]    = sn_cyc["sn_price_1c"] * sn_cyc["sn_contracts"]
    sn_cyc = sn_cyc.reset_index(drop=True)
    # sn_atm has 'vega' column from load; sn_cyc was built from pick_atm which includes vega
    # If vega column got dropped in merge, use proxy: straddle_price / (2 * strike * sqrt(T/365))
    if "vega" not in sn_cyc.columns:
        sn_cyc["sn_vega_contrib"] = (sn_cyc["sn_strd_e"] / sn_cyc["sn_K"].clip(lower=0.01)) * sn_cyc["sn_contracts"] * 100
    else:
        sn_cyc["sn_vega_contrib"] = sn_cyc["vega"] * 2 * sn_cyc["sn_contracts"] * 100

    # Aggregate
    sn_agg = sn_cyc.groupby("date").agg(
        single_pnl=("sn_pnl_total", "sum"),
        single_premium=("sn_premium", "sum"),
        single_vega=("sn_vega_contrib", "sum"),
        n_legs=("sn_contracts", lambda x: (x > 0).sum()),
        n_active=("sn_contracts", "sum"),
    ).reset_index()

    # Final cycles
    cyc = cyc.merge(sn_agg, on="date", how="inner")
    cyc["net_pnl"]    = cyc["spy_pnl_total"] + cyc["single_pnl"]
    cyc["spy_premium"] = cyc["spy_premium_1c"] * cyc["spy_contracts"]
    cyc["net_premium"] = cyc["spy_premium"] - cyc["single_premium"]  # SPY received - SN paid (classic)
    if mode == "reverse":
        cyc["net_premium"] = -cyc["net_premium"]  # SN received - SPY paid
    cyc["vega_ratio"]  = cyc["single_vega"] / (
        cyc["spy_S"] * 100 * cyc["spy_contracts"]  # SPY notional proxy for vega denominator
    ).clip(lower=0.01)

    # Basket IV vs SPY IV
    basket_iv = sn_d[sn_d["ticker"].isin(top_n)].groupby("date")["atm_iv"].mean()
    cyc["basket_iv"] = cyc["date"].map(basket_iv)
    cyc["iv_spread"]  = cyc["spy_iv_e"] - cyc["basket_iv"]  # negative = singles richer

    cyc = cyc.sort_values("date").reset_index(drop=True)
    cyc["nav"] = INITIAL_CAPITAL + cyc["net_pnl"].cumsum()

    # Diagnostics
    avg_sn_contr = sn_cyc.groupby("date")["sn_contracts"].sum().mean()
    print(f"  Cycles: {len(cyc)}")
    print(f"  Avg SPY contracts: {cyc['spy_contracts'].mean():.1f}")
    print(f"  Avg SPY premium: ${cyc['spy_premium'].mean():.0f}")
    print(f"  Avg SN premium: ${cyc['single_premium'].mean():.0f}")
    print(f"  Avg net premium: ${cyc['net_premium'].mean():.0f} ({'credit' if cyc['net_premium'].mean() > 0 else 'debit'})")
    print(f"  Active legs/cycle: {sn_agg['n_legs'].mean():.1f} tickers, {avg_sn_contr:.0f} total SN contracts")
    print(f"  Avg net P&L/cycle: ${cyc['net_pnl'].mean():.0f}")
    print(f"  Built in {time.time()-t0:.1f}s")

    return cyc, sn_cyc, top_n, top_w


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(df, label="all"):
    if df is None or len(df) == 0:
        return {"label": label, "n_trades": 0, "error": "no trades"}
    pnl     = df["net_pnl"].values
    n       = len(pnl)
    total_r = (df["nav"].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    mean_p  = np.mean(pnl)
    std_p   = np.std(pnl, ddof=1) if n > 1 else 1.0
    sharpe  = mean_p / std_p * np.sqrt(52) if std_p > 0 else 0.0
    down    = np.std(pnl[pnl < 0], ddof=1) if (pnl < 0).sum() > 1 else std_p
    sortino = mean_p / down * np.sqrt(52) if down > 0 else 0.0
    wr      = float((pnl > 0).mean() * 100)
    gp      = float(pnl[pnl > 0].sum()) if (pnl > 0).any() else 0.0
    gl      = float(abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else 1.0
    pf      = gp / gl if gl > 0 else float("inf")
    nav_a   = df["nav"].values
    pk      = np.maximum.accumulate(nav_a)
    max_dd  = float(((nav_a - pk) / pk).min() * 100)
    day_c   = float(pnl[pnl > 0].max() / gp) if gp > 0 else 0.0
    return {
        "label": label, "n_trades": n,
        "total_return_pct": round(total_r, 2),
        "mean_pnl": round(float(mean_p), 2),
        "std_pnl":  round(float(std_p), 2),
        "sharpe_annual":    round(float(sharpe), 3),
        "sortino_annual":   round(float(sortino), 3),
        "win_rate_pct":     round(wr, 1),
        "profit_factor":    round(float(pf), 3),
        "max_drawdown_pct": round(max_dd, 2),
        "day_concentration": round(day_c, 3),
        "gross_profit": round(gp, 0),
        "gross_loss":   round(gl, 0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. REGIME GAP TEST (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════════════════

def regime_gap_test(oot_df, spy_px):
    oot = oot_df.merge(spy_px["close"].rename("S_e").reset_index(), on="date", how="left")
    oot = oot.merge(
        spy_px["close"].rename("S_x").reset_index().rename(columns={"date":"exit_date"}),
        on="exit_date", how="left"
    )
    oot["spy_ret"] = (oot["S_x"] - oot["S_e"]) / oot["S_e"]
    oot["regime"] = "flat"
    oot.loc[oot["spy_ret"] > 0.005, "regime"]  = "green"
    oot.loc[oot["spy_ret"] < -0.005, "regime"] = "red"

    per_regime = {}
    for reg in ["green", "red", "flat"]:
        sub = oot[oot["regime"] == reg].copy()
        if len(sub) < 5:
            per_regime[reg] = {"n": len(sub), "sharpe": None, "win_rate": None}
            continue
        sub["nav"] = INITIAL_CAPITAL + sub["net_pnl"].cumsum()
        m = compute_metrics(sub, label=reg)
        per_regime[reg] = {"n": m["n_trades"], "sharpe": m["sharpe_annual"],
                           "sortino": m["sortino_annual"], "win_rate": m["win_rate_pct"],
                           "pf": m["profit_factor"]}

    gs = per_regime.get("green", {}).get("sharpe")
    rs = per_regime.get("red",   {}).get("sharpe")
    if gs is not None and rs is not None and max(abs(gs), abs(rs)) > 1e-6:
        gap_r  = abs(gs - rs) / max(abs(gs), abs(rs))
        verdict = "PASS" if gap_r <= 0.50 else "FAIL"
    else:
        gap_r   = None
        verdict = "INSUFFICIENT_DATA"

    return {"per_regime": per_regime,
            "gap_ratio":  round(gap_r, 3) if gap_r is not None else None,
            "regime_agnostic": verdict, "threshold": 0.50}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════════════════

def permutation_test(cyc, real_sharpe, n_perm=N_PERMUTATIONS, label=""):
    print(f"\n  Permutation test ({label}, {n_perm} trials)...")
    rng  = np.random.default_rng(RANDOM_SEED)
    pnl  = cyc["net_pnl"].values.copy()
    null = []
    for _ in range(n_perm):
        p = rng.permutation(pnl)
        s = np.mean(p) / np.std(p, ddof=1) * np.sqrt(52) if np.std(p) > 0 else 0.0
        null.append(s)
    null = np.array(null)
    pval = float((null >= real_sharpe).mean())
    print(f"  p={pval:.4f}, real={real_sharpe:.3f}, null_mean={null.mean():.3f}")
    return {"n_trials": n_perm, "real_sharpe": round(real_sharpe,3),
            "null_mean": round(float(null.mean()),3), "null_std": round(float(null.std()),3),
            "p_value": round(pval,4), "significant_05": bool(pval<0.05),
            "verdict": "PASS" if pval < 0.05 else "FAIL"}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CORRELATION PREMIUM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def corr_analysis(cyc, sn_atm, top_n, top_w):
    """Implied correlation via dispersion formula, correlation with P&L."""
    sn_t = sn_atm[sn_atm["ticker"].isin(top_n)][["date","ticker","atm_iv"]].copy()
    sn_t["date"]   = pd.to_datetime(sn_t["date"])
    sn_t["weight"] = sn_t["ticker"].map(top_w)

    rows = []
    for obs, grp in sn_t.groupby("date"):
        cyc_r = cyc[cyc["date"] == obs]
        if len(cyc_r) == 0:
            continue
        iv_idx = cyc_r["spy_iv_e"].iloc[0]
        ivs = grp["atm_iv"].values
        ws  = grp["weight"].values / grp["weight"].sum()
        bvar  = np.sum(ws**2 * ivs**2)
        cross = sum(ws[i]*ws[j]*ivs[i]*ivs[j]
                    for i in range(len(ws)) for j in range(i+1, len(ws)))
        ic = (iv_idx**2 - bvar)/(2*cross) if cross > 1e-10 else np.nan
        rows.append({"date": obs, "implied_corr": float(np.clip(ic,-1,1)),
                     "basket_iv": float(np.sqrt(np.sum(ws*ivs**2))),
                     "iv_spread": iv_idx - float(np.sqrt(np.sum(ws*ivs**2)))})

    if not rows:
        return {}
    cd = pd.DataFrame(rows)
    merged = cyc[["date","net_pnl"]].merge(cd, on="date", how="inner")
    iv_r = float(merged[["iv_spread","net_pnl"]].corr().iloc[0,1]) if len(merged) > 5 else None
    return {
        "mean_implied_corr": round(float(cd["implied_corr"].mean()), 4),
        "mean_basket_iv":    round(float(cd["basket_iv"].mean()), 4),
        "mean_iv_spread":    round(float(cd["iv_spread"].mean()), 4),
        "iv_spread_vs_pnl_r": round(iv_r, 3) if iv_r is not None else None,
        "pct_dates_index_cheap": round(float((cd["iv_spread"] < 0).mean()*100), 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_variant(mode, spy_atm, sn_atm, spy_px, px_wide, macro, avail, suffix=""):
    print(f"\n{'='*70}")
    print(f"DISPERSION {mode.upper()}{suffix}")
    print(f"{'='*70}")

    cyc, sn_cyc, top_n, top_w = build_cycles(spy_atm, sn_atm, spy_px, px_wide, avail, mode=mode)

    if len(cyc) < 20:
        print("ERROR: Too few cycles.")
        return None

    split = int(len(cyc) * (1 - OOT_FRACTION))
    is_df  = cyc.iloc[:split].copy();  is_df["nav"]  = INITIAL_CAPITAL + is_df["net_pnl"].cumsum()
    oot_df = cyc.iloc[split:].copy();  oot_df["nav"] = INITIAL_CAPITAL + oot_df["net_pnl"].cumsum()

    print(f"\n  IS:  {cyc['date'].iloc[0].date()} → {cyc['date'].iloc[split-1].date()} ({split} cycles)")
    print(f"  OOT: {cyc['date'].iloc[split].date()} → {cyc['date'].iloc[-1].date()} ({len(oot_df)} cycles)")

    m_full = compute_metrics(cyc,    "full")
    m_is   = compute_metrics(is_df,  "IS")
    m_oot  = compute_metrics(oot_df, "OOT")

    print(f"\n  Metrics:")
    for m in [m_full, m_is, m_oot]:
        print(f"    [{m['label'].upper()}] n={m['n_trades']} | Return={m['total_return_pct']:+.1f}% | "
              f"Sharpe={m['sharpe_annual']:.3f} | Sortino={m['sortino_annual']:.3f} | "
              f"WR={m['win_rate_pct']:.1f}% | PF={m['profit_factor']:.3f} | "
              f"MaxDD={m['max_drawdown_pct']:.1f}% | DayConc={m['day_concentration']:.3f}")

    regime_res = regime_gap_test(oot_df, spy_px)
    print(f"\n  Regime gap (HC #428 R1): {regime_res['regime_agnostic']} (gap={regime_res['gap_ratio']})")
    for reg, s in regime_res["per_regime"].items():
        print(f"    {reg}: n={s['n']}, Sharpe={s.get('sharpe')}, WR={s.get('win_rate')}%")

    perm = permutation_test(cyc, m_full["sharpe_annual"], label=mode)
    c_an = corr_analysis(cyc, sn_atm, top_n, top_w)
    print(f"\n  Correlation analysis:")
    print(f"    Mean implied corr: {c_an.get('mean_implied_corr')} (>0.3 = index richer)")
    print(f"    Mean IV spread (idx-basket): {c_an.get('mean_iv_spread')} (neg = singles richer)")
    print(f"    IV spread vs PnL r: {c_an.get('iv_spread_vs_pnl_r')}")
    print(f"    % dates index is cheap: {c_an.get('pct_dates_index_cheap')}%")

    day_c = m_full["day_concentration"]
    conc_v = "PASS" if day_c <= 0.70 else "FAIL"

    result = {
        "mode":          mode,
        "metrics":       {"full": m_full, "is": m_is, "oot": m_oot},
        "regime_gap":    regime_res,
        "permutation":   perm,
        "corr_analysis": c_an,
        "compliance": {
            "hc344_day_conc":  {"value": round(day_c,3), "limit": 0.70, "verdict": conc_v},
            "hc428_r1_regime": {"verdict": regime_res["regime_agnostic"], "gap_ratio": regime_res["gap_ratio"]},
            "permutation_p05": {"verdict": perm.get("verdict"), "p_value": perm.get("p_value")},
        },
        "data": {
            "n_cycles": len(cyc),
            "date_range": f"{cyc['date'].min().date()} → {cyc['date'].max().date()}",
            "oot_start": str(cyc["date"].iloc[split].date()),
            "top_n_tickers": top_n,
            "avg_spy_premium": round(float(cyc["spy_premium"].mean()),0),
            "avg_sn_premium":  round(float(cyc["single_premium"].mean()),0),
            "avg_net_premium": round(float(cyc["net_premium"].mean()),0),
        }
    }

    # Save trades
    cyc.to_parquet(OUT_DIR / f"trades_{mode}.parquet", index=False)
    cyc.to_csv(OUT_DIR / f"trades_{mode}.csv", index=False)

    return result


def main():
    print("=" * 70)
    print("DISPERSION TRADING v1 — Research Backtest")
    print("Testing CLASSIC and REVERSE dispersion variants")
    print("=" * 70)

    spy_atm, sn_atm, spy_px, px_wide, macro, avail = load_and_precompute()

    results = {}

    # VARIANT A: Classic dispersion (short SPY, long singles)
    r_classic = run_variant("classic", spy_atm, sn_atm, spy_px, px_wide, macro, avail)
    if r_classic:
        results["classic"] = r_classic

    # VARIANT B: Reverse dispersion (long SPY, short singles)
    r_reverse = run_variant("reverse", spy_atm, sn_atm, spy_px, px_wide, macro, avail)
    if r_reverse:
        results["reverse"] = r_reverse

    # Comparison summary
    print("\n" + "=" * 70)
    print("DISPERSION v1 — COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n  {'Mode':<10} {'Sharpe':>8} {'Sortino':>8} {'WR%':>6} {'PF':>6} {'Return%':>9} {'MaxDD%':>8} {'OOT Sharpe':>11} {'Regime':>8}")
    print("  " + "-" * 75)
    for mode, r in results.items():
        m  = r["metrics"]["full"]
        mo = r["metrics"]["oot"]
        rg = r["regime_gap"]["regime_agnostic"]
        print(f"  {mode:<10} {m['sharpe_annual']:>8.3f} {m['sortino_annual']:>8.3f} "
              f"{m['win_rate_pct']:>6.1f} {m['profit_factor']:>6.3f} "
              f"{m['total_return_pct']:>9.1f} {m['max_drawdown_pct']:>8.1f} "
              f"{mo['sharpe_annual']:>11.3f} {rg:>8}")
    print()
    print("  HC compliance:")
    for mode, r in results.items():
        for k, v in r["compliance"].items():
            print(f"    [{mode}] {k}: {v['verdict']}")

    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    print(f"\n  Results saved to: {OUT_DIR}/")
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()
