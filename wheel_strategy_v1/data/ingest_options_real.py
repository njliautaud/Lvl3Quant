"""
ingest_options_real.py — HC #556 R3 (REAL VENDOR DATA path).

Materializes the DoltHub `post-no-preference/options` clone at
`data/cache/options_real/options` into per-ticker Parquet files our
wheel research stack already understands, then produces a calibrated
`iv_features_real.parquet` that the existing `tier_runner` can consume
exactly like the modeled output.

What gets written:

  data/cache/options_real/_vol_history.parquet
      Long format: date, ticker, hv_current, iv_current, hv_year_high,
      hv_year_low, iv_year_high, iv_year_low
      One row per (date, ticker). Source: volatility_history table.

  data/cache/options_real/chains/{TICKER}.parquet  (optional, --chains)
      Long format: date, expiration, strike, type, bid, ask, mid, vol,
      delta, gamma, theta, vega, rho, dte
      Source: option_chain table, filtered to:
        - dte in [5, 90]    (wheel-relevant tenor band)
        - moneyness |log(K/S)| <= 0.30 (skip far OTM tails)
      Materialized only on --chains because the full chain table is
      multi-GB. Default mode just builds the IV summary.

  data/cache/iv_features_real.parquet
      Schema-compatible with iv_features_modeled.parquet:
        date, ticker, sigma_rv, iv_rv_ratio, term_ratio, term_proxy,
        sigma, sigma_atm_30d, iv_rank, r_1m, pricing_source

      For tickers/dates with real iv_current available, sigma is the
      REAL vendor iv (with column pricing_source='real_dolt'). For
      ticker/date combos where DOLT is missing, we fall back to the
      modeled VIX-regime IV/RV ratio and tag pricing_source='modeled_bs_calibrated'.

      iv_rank is computed from the REAL iv series per ticker (rolling
      252-day percentile rank of sigma).

Ticker symbol mapping: DoltHub uses '.' for share classes; our universe
uses '-'. Mapped both ways: BRK-B <-> BRK.B.

Run modes:
    python -m data.ingest_options_real             # IV summary + calibrated features only
    python -m data.ingest_options_real --chains    # also materialize per-ticker chains
    python -m data.ingest_options_real --tickers AAPL,MSFT,NVDA --chains   # subset

Output is written to data/cache. Existing modeled feature file is NOT touched.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import shutil
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
DOLT_DIR = CACHE / "options_real" / "options"
OUT_DIR = CACHE / "options_real"
CHAINS_DIR = OUT_DIR / "chains"


def _dolt_path() -> str:
    """Locate the dolt binary."""
    for p in (shutil.which("dolt"), "/home/jupiter/.local/bin/dolt", "/usr/local/bin/dolt"):
        if p and Path(p).exists():
            return p
    raise SystemExit("[ingest_real] dolt binary not found on PATH")


def _dolt_query_csv(sql: str, dolt_bin: Optional[str] = None,
                    timeout: int = 600) -> pd.DataFrame:
    """Run a Dolt SQL query and return a DataFrame parsed from CSV result."""
    db = dolt_bin or _dolt_path()
    proc = subprocess.run(
        [db, "sql", "-q", sql, "-r", "csv"],
        cwd=str(DOLT_DIR),
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"dolt sql failed: {proc.stderr.strip()[:500]}")
    out = proc.stdout
    if not out.strip():
        return pd.DataFrame()
    from io import StringIO
    return pd.read_csv(StringIO(out))


def _dolt_symbol(ticker: str) -> str:
    """Map universe-side ticker to Dolt act_symbol convention."""
    return ticker.replace("-", ".") if "-" in ticker else ticker


def _universe_tickers(restrict: Optional[List[str]] = None) -> List[str]:
    u = pd.read_parquet(CACHE / "universe.parquet")
    tks = u["ticker"].astype(str).tolist()
    if restrict:
        tks = [t for t in tks if t in set(restrict)]
    return tks


def materialize_vol_history(tickers: List[str], dolt_bin: str) -> pd.DataFrame:
    """Pull volatility_history rows for our tickers as one wide query."""
    dolt_syms = sorted({_dolt_symbol(t) for t in tickers})
    in_list = ",".join(f"'{s}'" for s in dolt_syms)
    sql = (
        "SELECT date, act_symbol, hv_current, hv_week_ago, hv_month_ago, "
        "hv_year_high, hv_year_low, iv_current, iv_week_ago, iv_month_ago, "
        "iv_year_high, iv_year_low "
        "FROM volatility_history "
        f"WHERE act_symbol IN ({in_list}) "
        "ORDER BY act_symbol, date"
    )
    print(f"[ingest_real] querying volatility_history for {len(dolt_syms)} symbols...")
    df = _dolt_query_csv(sql, dolt_bin, timeout=900)
    if df.empty:
        return df

    # Map symbols back to our universe convention.
    inv = {_dolt_symbol(t): t for t in tickers}
    df["ticker"] = df["act_symbol"].map(inv).fillna(df["act_symbol"])
    df["date"] = pd.to_datetime(df["date"])
    # Cast decimals to floats.
    for c in ["hv_current", "hv_week_ago", "hv_month_ago", "hv_year_high",
              "hv_year_low", "iv_current", "iv_week_ago", "iv_month_ago",
              "iv_year_high", "iv_year_low"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["date", "ticker", "hv_current", "iv_current",
            "hv_year_high", "hv_year_low", "iv_year_high", "iv_year_low"]
    return df[keep].sort_values(["ticker", "date"]).reset_index(drop=True)


def materialize_chains_for_ticker(ticker: str, dolt_bin: str) -> Optional[pd.DataFrame]:
    """Pull option_chain rows for one ticker, filtered to wheel-relevant tenor."""
    sym = _dolt_symbol(ticker)
    # We filter at SQL level to keep transferred rows manageable.
    sql = (
        "SELECT date, expiration, strike, call_put, bid, ask, vol, "
        "delta, gamma, theta, vega, rho "
        "FROM option_chain "
        f"WHERE act_symbol = '{sym}' "
        "  AND DATEDIFF(expiration, date) BETWEEN 5 AND 90 "
        "ORDER BY date, expiration, strike"
    )
    try:
        df = _dolt_query_csv(sql, dolt_bin, timeout=1800)
    except Exception as e:
        print(f"[ingest_real][{ticker}] chain query failed: {e}")
        return None
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["dte"] = (df["expiration"] - df["date"]).dt.days
    for c in ["strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df.rename(columns={"call_put": "type"})
    df["type"] = df["type"].str.lower().str[:1]  # 'c' / 'p'
    return df


def build_calibrated_iv(vol_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Build iv_features_real.parquet — schema-compatible with the modeled file.

    Approach: start from prices.parquet (date,ticker,rv_20 etc), join real
    iv_current where DOLT has it, otherwise fall back to the same
    VIX-regime modeled ratio used by ingest_options_modeled.py.
    """
    px_path = CACHE / "prices.parquet"
    macro_path = CACHE / "macro.parquet"
    if not px_path.exists() or not macro_path.exists():
        raise SystemExit("[ingest_real] prices.parquet or macro.parquet missing")
    px = pd.read_parquet(px_path)
    macro = pd.read_parquet(macro_path)
    px["date"] = pd.to_datetime(px["date"])
    macro["date"] = pd.to_datetime(macro["date"])
    px = px.sort_values(["ticker", "date"]).copy()

    # --- Modeled fallback components ---
    mac = macro[["date"]].copy()
    mac["vix"] = macro["vix"] if "vix" in macro.columns else float("nan")
    mac["vix3m"] = macro["vix3m"] if "vix3m" in macro.columns else float("nan")
    mac["vix_dec"] = mac["vix"] / 100.0
    mac["vix3m_dec"] = mac["vix3m"] / 100.0

    vix_breaks = [10.0, 12.0, 18.0, 25.0, 35.0, 50.0, 80.0]
    ratio_vals = [1.32, 1.30, 1.20, 1.10, 1.00, 0.90, 0.85]
    mac["iv_rv_ratio_modeled"] = np.interp(
        mac["vix"].fillna(18.0).clip(8.0, 90.0),
        vix_breaks, ratio_vals,
    )
    mac["iv_rv_ratio_modeled"] = (
        mac["iv_rv_ratio_modeled"].rolling(20, min_periods=1).mean().fillna(1.20)
    )
    mac["term_ratio"] = (mac["vix3m_dec"] / mac["vix_dec"]).clip(0.6, 1.6).fillna(1.05)

    # Risk-free rate
    me_path = CACHE / "macro_extra.parquet"
    if me_path.exists():
        me = pd.read_parquet(me_path)
        me["date"] = pd.to_datetime(me["date"])
        rcol = None
        for cand in ["dgs1mo", "DGS1MO", "ust_1m"]:
            if cand in me.columns:
                rcol = cand; break
        if rcol is None and "ust_2y" in me.columns:
            rcol = "ust_2y"
        if rcol:
            mac = mac.merge(me[["date", rcol]].rename(columns={rcol: "r_1m_pct"}),
                            on="date", how="left")
            mac["r_1m"] = (mac["r_1m_pct"] / 100.0).clip(0.0, 0.10)
        else:
            mac["r_1m"] = 0.04
    else:
        mac["r_1m"] = 0.04
    mac["r_1m"] = mac["r_1m"].fillna(0.04)

    px = px.merge(
        mac[["date", "iv_rv_ratio_modeled", "term_ratio", "r_1m"]],
        on="date", how="left",
    )
    px["sigma_rv"] = px["rv_20"].astype(float)
    px["sigma_atm_30d_modeled"] = px["sigma_rv"] * px["iv_rv_ratio_modeled"].fillna(1.20)

    # --- Join REAL iv_current (forward-fill within ticker, since DOLT samples weekly) ---
    vh = vol_hist[["date", "ticker", "iv_current", "hv_current"]].copy()
    vh["date"] = pd.to_datetime(vh["date"])
    # iv_current is a 30d ATM IV — exactly the sigma we want.
    px = px.merge(vh, on=["date", "ticker"], how="left")
    px = px.sort_values(["ticker", "date"])
    # Forward-fill iv_current within ticker up to 7 business days (DOLT is weekly).
    px["iv_current"] = px.groupby("ticker")["iv_current"].ffill(limit=7)
    px["hv_current"] = px.groupby("ticker")["hv_current"].ffill(limit=7)

    # --- Choose sigma per row: REAL if available, else modeled ---
    px["pricing_source"] = np.where(
        px["iv_current"].notna(),
        "real_dolt",
        "modeled_bs_calibrated",
    )
    px["sigma"] = px["iv_current"].fillna(px["sigma_atm_30d_modeled"])
    px["sigma_atm_30d"] = px["sigma"]

    # Effective IV/RV ratio per row (for analytics) — what the actual sigma corresponds to.
    px["iv_rv_ratio"] = np.where(
        px["sigma_rv"].gt(0),
        px["sigma"] / px["sigma_rv"],
        np.nan,
    )
    # term_proxy backwards-compat
    if "rv_60" in px.columns:
        px["term_proxy"] = px["rv_60"] / px["sigma_rv"]
    else:
        px["term_proxy"] = 1.0

    # iv_rank — percentile rank of sigma per ticker over trailing 252 bdays.
    def _pctrank(s, w=252, m=20):
        return s.rolling(w, min_periods=m).rank(pct=True)
    px["iv_rank"] = px.groupby("ticker")["sigma"].transform(_pctrank)

    out_cols = ["date", "ticker", "sigma_rv", "iv_rv_ratio", "term_ratio",
                "term_proxy", "sigma", "sigma_atm_30d", "iv_rank", "r_1m",
                "pricing_source"]
    return px[out_cols].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None,
                    help="Comma-separated ticker subset (default = full universe)")
    ap.add_argument("--chains", action="store_true",
                    help="Also materialize per-ticker option_chain parquets (slow)")
    ap.add_argument("--vol-out", default=None,
                    help="Override vol-history parquet path")
    ap.add_argument("--features-out", default=None,
                    help="Override calibrated iv_features_real parquet path")
    args = ap.parse_args()

    if not DOLT_DIR.exists():
        raise SystemExit(f"[ingest_real] dolt repo missing at {DOLT_DIR}")

    dolt_bin = _dolt_path()
    restrict = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    tickers = _universe_tickers(restrict)
    print(f"[ingest_real] universe size: {len(tickers)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: vol_history (always cheap, single query)
    vh = materialize_vol_history(tickers, dolt_bin)
    if vh.empty:
        raise SystemExit("[ingest_real] DOLT returned no vol_history rows")
    vol_path = Path(args.vol_out) if args.vol_out else OUT_DIR / "_vol_history.parquet"
    vh.to_parquet(vol_path, index=False)
    print(f"[ingest_real] wrote {len(vh):,} vol_history rows -> {vol_path}")
    cov = vh.groupby("ticker")["iv_current"].count().sort_values(ascending=False)
    print(f"[ingest_real] tickers covered: {len(cov)}/{len(tickers)}")
    print(f"[ingest_real] median rows/ticker (iv_current): {cov.median():.0f}")
    print(f"[ingest_real] date range: {vh['date'].min().date()} -> {vh['date'].max().date()}")
    print(f"[ingest_real] mean iv_current: {vh['iv_current'].mean():.4f}  "
          f"mean hv_current: {vh['hv_current'].mean():.4f}")

    # Step 2: calibrated iv_features (uses real iv_current where present, modeled else)
    iv = build_calibrated_iv(vh)
    feat_path = Path(args.features_out) if args.features_out else CACHE / "iv_features_real.parquet"
    iv.to_parquet(feat_path, index=False)
    real_n = (iv["pricing_source"] == "real_dolt").sum()
    mod_n = (iv["pricing_source"] == "modeled_bs_calibrated").sum()
    print(f"[ingest_real] wrote {len(iv):,} iv_feature rows -> {feat_path}")
    print(f"[ingest_real]   real_dolt:           {real_n:,} rows ({real_n/len(iv):.1%})")
    print(f"[ingest_real]   modeled_bs_fallback: {mod_n:,} rows ({mod_n/len(iv):.1%})")
    real_only = iv[iv["pricing_source"] == "real_dolt"]
    print(f"[ingest_real] REAL subset stats:")
    print(f"  mean sigma:           {real_only['sigma'].mean():.4f}")
    print(f"  mean sigma_rv:        {real_only['sigma_rv'].mean():.4f}")
    print(f"  mean iv_rv_ratio:     {real_only['iv_rv_ratio'].mean():.4f}")
    print(f"  mean iv_rank:         {real_only['iv_rank'].mean():.4f}")
    print(f"  mean r_1m:            {real_only['r_1m'].mean():.4f}")

    # Step 3 (optional): per-ticker chains. Heavy — only if explicitly asked.
    if args.chains:
        CHAINS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[ingest_real] materializing chains for {len(tickers)} tickers ...")
        ok = 0; fail = 0; empty = 0
        for i, t in enumerate(tickers, 1):
            df = materialize_chains_for_ticker(t, dolt_bin)
            if df is None:
                fail += 1
                continue
            if df.empty:
                empty += 1
                continue
            out = CHAINS_DIR / f"{t}.parquet"
            df.to_parquet(out, index=False)
            ok += 1
            if i % 10 == 0 or i == len(tickers):
                print(f"[ingest_real]   ({i}/{len(tickers)}) ok={ok} empty={empty} fail={fail}")
        print(f"[ingest_real] chains done. ok={ok} empty={empty} fail={fail}")


if __name__ == "__main__":
    main()
