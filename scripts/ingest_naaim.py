#!/usr/bin/env python3
"""
ingest_naaim.py — Idempotent ingest of the NAAIM Exposure Index.

NAAIM = National Association of Active Investment Managers. Weekly survey of
active manager equity exposure, released Thursday afternoons (data point
covers the prior Wed-Wed window). Range -200% to +200%.

Outputs:
    data/raw/naaim_exposure_history.csv          (date, exposure_pct, source, sp500)
    data/derived/naaim_weekly.parquet            (date, naaim,
                                                  naaim_change_1w,
                                                  naaim_change_4w,
                                                  naaim_zscore_52w)
    output/naaim_v1/naaim_exploratory.txt        (exploratory report)

Strategy:
    1) Fetch the NAAIM exposure index landing page.
    2) Scrape it for the canonical XLSX link
       (URL changes weekly: .../wp-content/uploads/<yyyy>/<mm>/USE_Data-since-Inception_<yyyy-mm-dd>.xlsx).
    3) Download the XLSX and parse with pandas/openpyxl.
    4) Fallback A: if XLSX cannot be reached, scrape the HTML table on the page
       (it carries the full history).
    5) Fallback B: if both blocked, reuse any cached CSV under data/raw/.

Idempotent: re-running overwrites raw CSV and derived parquet with the
freshest data. MLflow run name: naaim_ingest_v1.
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------- Paths ----------
PROJECT_ROOT = Path("/home/jupiter/Lvl3Quant")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DERIVED_DIR = PROJECT_ROOT / "data" / "derived"
OUTPUT_DIR = PROJECT_ROOT / "output" / "naaim_v1"

RAW_CSV = RAW_DIR / "naaim_exposure_history.csv"
WEEKLY_PARQUET = DERIVED_DIR / "naaim_weekly.parquet"
REPORT_TXT = OUTPUT_DIR / "naaim_exploratory.txt"

NAAIM_PAGE = "https://www.naaim.org/programs/naaim-exposure-index/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
}

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("naaim")


# ----------------------------------------------------------------------------
# Fetchers
# ----------------------------------------------------------------------------
def _http_get(url: str, timeout: int = 30) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r


def fetch_xlsx_url_from_page() -> Optional[str]:
    """Scrape the NAAIM page and return the most recent XLSX link."""
    try:
        resp = _http_get(NAAIM_PAGE)
    except Exception as e:
        log.warning("NAAIM page fetch failed: %s", e)
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    candidates = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"\.xlsx($|\?)", href, re.I):
            candidates.append(href)
    # Also scan raw text in case the link is not in an <a>.
    candidates += re.findall(r"https?://[^\"'\s)]+\.xlsx", resp.text, flags=re.I)
    if not candidates:
        return None
    # Pick the newest-looking one (by URL date if present, else alphabetical max).
    def _key(u: str) -> str:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", u)
        return m.group(1) if m else u
    return sorted(set(candidates), key=_key)[-1]


def load_from_xlsx(url: str) -> pd.DataFrame:
    log.info("Downloading XLSX: %s", url)
    resp = _http_get(url, timeout=45)
    df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
    # Standardise columns.
    df.columns = [str(c).strip() for c in df.columns]
    # NAAIM Number == Mean/Average historically; prefer 'NAAIM Number'.
    if "NAAIM Number" in df.columns:
        expo = df["NAAIM Number"]
    elif "Mean/Average" in df.columns:
        expo = df["Mean/Average"]
    else:
        raise ValueError(f"Could not locate NAAIM exposure column. cols={list(df.columns)}")
    sp = df["S&P 500"] if "S&P 500" in df.columns else pd.Series(np.nan, index=df.index)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], errors="coerce"),
        "exposure_pct": pd.to_numeric(expo, errors="coerce"),
        "sp500": pd.to_numeric(sp, errors="coerce"),
        "source": "naaim_xlsx",
    })
    return out


def load_from_html_table() -> pd.DataFrame:
    """Fallback: parse the on-page HTML table."""
    log.info("Falling back to HTML table scrape")
    resp = _http_get(NAAIM_PAGE)
    # pd.read_html handles all <table> elements.
    tables = pd.read_html(io.StringIO(resp.text))
    best = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        joined = " ".join(cols)
        if "date" in joined and ("naaim" in joined or "mean" in joined or "average" in joined):
            best = t
            break
    if best is None:
        raise RuntimeError("No suitable NAAIM HTML table found")
    # Some pages bury the real header in the first row.
    if "Date" not in best.columns:
        best.columns = best.iloc[0]
        best = best.iloc[1:].reset_index(drop=True)
    expo_col = None
    for c in best.columns:
        if str(c).strip().lower() in {"naaim number mean/average", "naaim number",
                                       "mean/average", "naaim exposure index"}:
            expo_col = c
            break
    if expo_col is None:
        # Heuristic: second numeric column.
        expo_col = best.columns[1]
    out = pd.DataFrame({
        "date": pd.to_datetime(best["Date"], errors="coerce"),
        "exposure_pct": pd.to_numeric(best[expo_col], errors="coerce"),
        "sp500": np.nan,
        "source": "naaim_html",
    })
    return out


# ----------------------------------------------------------------------------
# Cleaning & derived features
# ----------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["date", "exposure_pct"]).copy()
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    # Clamp to documented bounds.
    df["exposure_pct"] = df["exposure_pct"].clip(-200, 200)
    return df


def build_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build week-aligned features.
    NAAIM is published Thursdays; we anchor each value to the Friday of that
    week and forward-fill Mon-Fri so a daily trading-feature join is trivial.
    """
    s = df.set_index("date")["exposure_pct"].sort_index()

    # 1) Build a Friday-anchored weekly series (label='right'). This is the
    # canonical NAAIM weekly value -- if multiple prints fall in one ISO week
    # (rare cleanups in the source file), keep the last.
    weekly = s.resample("W-FRI").last().dropna()

    # 2) Lagged changes & rolling z-score (52-week window).
    feat = pd.DataFrame({"naaim": weekly})
    feat["naaim_change_1w"] = feat["naaim"].diff(1)
    feat["naaim_change_4w"] = feat["naaim"].diff(4)
    roll = feat["naaim"].rolling(window=52, min_periods=12)
    feat["naaim_zscore_52w"] = (feat["naaim"] - roll.mean()) / roll.std(ddof=0)

    # 3) Re-index to business-day grid and forward-fill within each ISO week
    #    so any trading-day join "asof" gets the most recent print.
    bdays = pd.bdate_range(feat.index.min(), feat.index.max(), freq="B")
    daily = feat.reindex(feat.index.union(bdays)).sort_index().ffill()
    daily = daily.loc[bdays]
    daily.index.name = "date"
    return daily.reset_index()


# ----------------------------------------------------------------------------
# Exploratory analysis
# ----------------------------------------------------------------------------
def exploratory_report(raw: pd.DataFrame) -> str:
    """
    Forward-return correlation analysis using the SP500 column embedded in
    NAAIM's own XLSX (Friday close at the time of survey).
    """
    df = raw.dropna(subset=["sp500", "exposure_pct"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["naaim"] = df["exposure_pct"]
    df["naaim_chg_1w"] = df["naaim"].diff(1)

    # Forward returns -- weekly cadence so N forward weeks = N rows ahead.
    df["fwd_1w_ret"] = df["sp500"].shift(-1) / df["sp500"] - 1.0
    df["fwd_2w_ret"] = df["sp500"].shift(-2) / df["sp500"] - 1.0
    df["fwd_4w_ret"] = df["sp500"].shift(-4) / df["sp500"] - 1.0

    n = len(df)
    lines = []
    lines.append("NAAIM Exploratory Report (v1)")
    lines.append("=" * 60)
    lines.append(f"Rows (weekly prints): {n}")
    lines.append(f"Date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    lines.append(f"Latest NAAIM exposure: {df['naaim'].iloc[-1]:.2f} "
                 f"on {df['date'].iloc[-1].date()}")
    lines.append(f"All-time mean: {df['naaim'].mean():.2f} | "
                 f"std: {df['naaim'].std():.2f} | "
                 f"min: {df['naaim'].min():.2f} | max: {df['naaim'].max():.2f}")
    lines.append("")

    # 1) Pearson correlations: level vs fwd returns.
    lines.append("Forward S&P 500 return correlations (Pearson):")
    lines.append("-" * 60)
    for col, label in [("fwd_1w_ret", "1-week"),
                       ("fwd_2w_ret", "2-week"),
                       ("fwd_4w_ret", "4-week")]:
        d = df[["naaim", "naaim_chg_1w", col]].dropna()
        r_level = d["naaim"].corr(d[col])
        r_chg = d["naaim_chg_1w"].corr(d[col])
        lines.append(f"  Horizon {label:<7s} | level r = {r_level:+.4f} "
                     f"| 1w-change r = {r_chg:+.4f} | N = {len(d)}")
    lines.append("")

    # 2) Quantile / regime analysis on the level.
    lines.append("Regime analysis on fwd 4-week S&P 500 return:")
    lines.append("-" * 60)
    bins = [
        ("NAAIM < 30  (defensive)",   df[df["naaim"] < 30]),
        ("NAAIM 30-60 (neutral-low)", df[(df["naaim"] >= 30) & (df["naaim"] < 60)]),
        ("NAAIM 60-80 (neutral-hi)",  df[(df["naaim"] >= 60) & (df["naaim"] < 80)]),
        ("NAAIM >= 80 (aggressive)",  df[df["naaim"] >= 80]),
        ("NAAIM > 100 (leveraged)",   df[df["naaim"] > 100]),
    ]
    lines.append(f"  {'bucket':<30s} {'N':>5s} {'mean%':>8s} "
                 f"{'std%':>8s} {'sharpe':>8s} {'hit%':>7s}")
    for label, sub in bins:
        r = sub["fwd_4w_ret"].dropna()
        if len(r) < 3:
            lines.append(f"  {label:<30s} {len(r):>5d}  (insufficient sample)")
            continue
        mean = r.mean() * 100
        std = r.std() * 100
        sharpe = (r.mean() / r.std()) * np.sqrt(13.0) if r.std() > 0 else float("nan")
        hit = (r > 0).mean() * 100
        lines.append(f"  {label:<30s} {len(r):>5d} {mean:>8.3f} "
                     f"{std:>8.3f} {sharpe:>8.3f} {hit:>7.2f}")
    lines.append("")
    lines.append("(Sharpe annualized assuming 13 non-overlapping 4w windows / yr.")
    lines.append(" Buckets overlap forward windows so SHARPE is descriptive, not tradable.)")
    lines.append("")

    # 3) Tails on the change.
    lines.append("Big weekly changes (>1 std deviation move in NAAIM) "
                 "vs fwd 4-week S&P:")
    lines.append("-" * 60)
    sd = df["naaim_chg_1w"].std()
    up = df[df["naaim_chg_1w"] > sd]["fwd_4w_ret"].dropna()
    dn = df[df["naaim_chg_1w"] < -sd]["fwd_4w_ret"].dropna()
    if len(up) > 3:
        lines.append(f"  Big-up weeks   (chg > +{sd:.1f}): "
                     f"N={len(up)} mean={up.mean()*100:+.3f}% "
                     f"hit={(up>0).mean()*100:.1f}%")
    if len(dn) > 3:
        lines.append(f"  Big-down weeks (chg < -{sd:.1f}): "
                     f"N={len(dn)} mean={dn.mean()*100:+.3f}% "
                     f"hit={(dn>0).mean()*100:.1f}%")
    lines.append("")

    # 4) Plain-English summary.
    d_all = df[["naaim", "fwd_4w_ret"]].dropna()
    r_level_4w = d_all["naaim"].corr(d_all["fwd_4w_ret"])
    low = df[df["naaim"] < 30]["fwd_4w_ret"].dropna()
    high = df[df["naaim"] >= 80]["fwd_4w_ret"].dropna()
    diff_pp = (low.mean() - high.mean()) * 100 if len(low) > 3 and len(high) > 3 else float("nan")
    verdict_lines = []
    verdict_lines.append("Plain-English verdict:")
    verdict_lines.append("-" * 60)
    if not np.isfinite(diff_pp):
        verdict_lines.append("Insufficient sample in the extreme buckets to draw a conclusion.")
    else:
        sign = "higher" if diff_pp > 0 else "lower"
        verdict_lines.append(
            f"  Over the full sample, weeks when NAAIM was <30 produced an "
            f"average forward-4-week S&P return that was {abs(diff_pp):.2f} "
            f"percentage points {sign} than weeks when NAAIM was >=80. "
            f"Raw level-to-4w-return correlation is {r_level_4w:+.3f}."
        )
        if abs(r_level_4w) < 0.05 and abs(diff_pp) < 1.0:
            verdict_lines.append("  Verdict: WEAK relationship at the studied horizons. "
                                 "Useful at best as a slow regime gate, not a directional signal.")
        elif diff_pp > 0 and abs(r_level_4w) >= 0.05:
            verdict_lines.append("  Verdict: classic contrarian fingerprint -- low NAAIM (capitulated) "
                                 "tends to be followed by stronger forward returns than high NAAIM "
                                 "(crowded long). Looks useful as a slow regime/sentiment gate, "
                                 "weighted lightly given the modest correlation.")
        else:
            verdict_lines.append("  Verdict: inverse to the usual contrarian story in this sample -- "
                                 "treat with caution, likely regime-dependent.")
    lines.extend(verdict_lines)
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# MLflow
# ----------------------------------------------------------------------------
def _try_mlflow_log(raw: pd.DataFrame, weekly: pd.DataFrame, report: str) -> None:
    try:
        import mlflow  # noqa
    except Exception as e:
        log.info("mlflow not available (%s) -- skipping tracking", e)
        return
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("macro_features")
        with mlflow.start_run(run_name="naaim_ingest_v1"):
            mlflow.log_param("source", "naaim.org XLSX (fallback HTML)")
            mlflow.log_param("rows_raw", len(raw))
            mlflow.log_param("rows_weekly", len(weekly))
            mlflow.log_param("date_start", str(raw["date"].min().date()))
            mlflow.log_param("date_end", str(raw["date"].max().date()))
            mlflow.log_metric("latest_naaim", float(raw["exposure_pct"].iloc[-1]))
            mlflow.log_metric("naaim_mean", float(raw["exposure_pct"].mean()))
            mlflow.log_metric("naaim_std", float(raw["exposure_pct"].std()))
            mlflow.log_artifact(str(RAW_CSV))
            mlflow.log_artifact(str(WEEKLY_PARQUET))
            mlflow.log_artifact(str(REPORT_TXT))
        log.info("MLflow run logged as naaim_ingest_v1")
    except Exception as e:
        log.warning("MLflow logging failed: %s", e)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    for d in (RAW_DIR, DERIVED_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Fetch.
    raw = None
    url = fetch_xlsx_url_from_page()
    if url:
        try:
            raw = load_from_xlsx(url)
        except Exception as e:
            log.warning("XLSX path failed: %s", e)
    if raw is None or raw["exposure_pct"].dropna().empty:
        try:
            raw = load_from_html_table()
        except Exception as e:
            log.error("HTML fallback failed: %s", e)
            if RAW_CSV.exists():
                log.warning("Using cached CSV at %s", RAW_CSV)
                raw = pd.read_csv(RAW_CSV, parse_dates=["date"])
            else:
                log.error("No NAAIM data available and no cache. Aborting.")
                return 2

    raw = clean(raw)
    log.info("Cleaned NAAIM rows: %d (%s .. %s)",
             len(raw), raw["date"].min().date(), raw["date"].max().date())

    # 2) Write raw CSV (idempotent overwrite).
    raw_out = raw[["date", "exposure_pct", "source", "sp500"]].copy()
    raw_out.to_csv(RAW_CSV, index=False)
    log.info("Wrote %s", RAW_CSV)

    # 3) Weekly / daily-ffill feature parquet.
    weekly = build_weekly_features(raw)
    weekly.to_parquet(WEEKLY_PARQUET, index=False)
    log.info("Wrote %s (rows=%d)", WEEKLY_PARQUET, len(weekly))

    # 4) Exploratory report.
    report = exploratory_report(raw)
    REPORT_TXT.write_text(report + "\n", encoding="utf-8")
    log.info("Wrote %s", REPORT_TXT)

    # 5) MLflow.
    _try_mlflow_log(raw, weekly, report)

    # 6) Console summary (for caller).
    print("\n=== NAAIM INGEST SUMMARY ===")
    print(f"rows={len(raw)} range={raw['date'].min().date()}..{raw['date'].max().date()}")
    print(f"latest={raw['exposure_pct'].iloc[-1]:.2f} "
          f"on {raw['date'].iloc[-1].date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
