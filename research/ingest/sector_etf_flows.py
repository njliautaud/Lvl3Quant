"""
Family: sector_flows  (HC #563 R2 — sector ETF AUM + flow proxies)

WHAT: Sector ETF flow signals — daily close, volume, dollar_volume (flow proxy),
AUM estimate (sharesOutstanding × close), multi-horizon returns, relative
strength vs SPY (etf_ret_20d − spy_ret_20d).

Universe: 11 S&P sector SPDRs (XLK XLF XLE XLV XLY XLP XLI XLB XLU XLRE XLC)
+ sub-sectors / thematics (SMH IGV KRE OIH ITB KWEB ARKK) + SPY benchmark.

SOURCE: yfinance (free).

WHY: HC #563 R2 — sector flows mandatory. Replaces the prior smoke stub.

OUTPUT: data/feature_store/sector_etf_flows/daily.parquet
Schema: (etf, date, close, volume, dollar_volume, shares_out, aum_proxy,
         ret_1d, ret_20d, ret_60d, rel_strength_spy)
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/home/jupiter/Lvl3Quant/research/ingest")
from _common import write_parquet, smoke_log  # type: ignore

FAMILY = "sector_etf_flows"

SECTOR_SPDRS = ["XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLB","XLU","XLRE","XLC"]
SUBSECTORS   = ["SMH","IGV","KRE","OIH","ITB","KWEB","ARKK"]
BENCHMARK    = ["SPY"]
ETFS = SECTOR_SPDRS + SUBSECTORS + BENCHMARK


def _fetch_history(yf, tickers, period="5y"):
    """Bulk download via yfinance.download — multi-ticker, single call."""
    df = yf.download(
        tickers,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance returned empty")
    return df


def _to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-index columns (field, ticker) → tidy long (date, etf, close, volume)."""
    close = df["Close"].stack(dropna=True).rename("close").reset_index()
    close.columns = ["date","etf","close"]
    vol   = df["Volume"].stack(dropna=True).rename("volume").reset_index()
    vol.columns = ["date","etf","volume"]
    out = close.merge(vol, on=["date","etf"], how="left")
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values(["etf","date"]).reset_index(drop=True)
    return out


def _add_returns(out: pd.DataFrame) -> pd.DataFrame:
    """Per-etf returns 1d / 20d / 60d using groupby pct_change."""
    g = out.groupby("etf", group_keys=False)
    out["ret_1d"]  = g["close"].pct_change(1)
    out["ret_20d"] = g["close"].pct_change(20)
    out["ret_60d"] = g["close"].pct_change(60)
    return out


def _add_rel_strength_spy(out: pd.DataFrame) -> pd.DataFrame:
    """rel_strength_spy = etf_ret_20d − spy_ret_20d (joined by date)."""
    spy = out.loc[out["etf"] == "SPY", ["date","ret_20d"]].rename(
        columns={"ret_20d": "spy_ret_20d"})
    out = out.merge(spy, on="date", how="left")
    out["rel_strength_spy"] = out["ret_20d"] - out["spy_ret_20d"]
    out = out.drop(columns=["spy_ret_20d"])
    return out


def _add_aum_proxy(yf, out: pd.DataFrame) -> pd.DataFrame:
    """shares_out from yf.Ticker.fast_info (best-effort); aum_proxy = shares*close."""
    shares = {}
    for t in out["etf"].unique():
        try:
            tk = yf.Ticker(t)
            so = None
            try:
                so = tk.fast_info.get("shares")
            except Exception:
                so = None
            if not so:
                try:
                    so = tk.info.get("sharesOutstanding")
                except Exception:
                    so = None
            shares[t] = float(so) if so else float("nan")
            time.sleep(0.05)
        except Exception:
            shares[t] = float("nan")
    out["shares_out"] = out["etf"].map(shares)
    out["aum_proxy"]  = out["shares_out"] * out["close"]
    return out


def main():
    try:
        import yfinance as yf
        print(f"[{FAMILY}] fetching {len(ETFS)} ETFs (5y daily) via yfinance ...")
        df = _fetch_history(yf, ETFS, period="5y")
        out = _to_tidy(df)
        print(f"[{FAMILY}] tidy frame: {len(out)} rows, {out['etf'].nunique()} ETFs")

        out["dollar_volume"] = out["close"] * out["volume"]
        out = _add_returns(out)
        out = _add_rel_strength_spy(out)
        out = _add_aum_proxy(yf, out)

        # Column order matches docstring schema
        cols = ["etf","date","close","volume","dollar_volume",
                "shares_out","aum_proxy","ret_1d","ret_20d","ret_60d",
                "rel_strength_spy"]
        out = out[cols]

        p = write_parquet(out, FAMILY, "daily.parquet")
        smoke_log(FAMILY, True, f"{len(out)} rows, {out['etf'].nunique()} etfs -> {p}")
        print(f"OK {FAMILY}: {len(out)} rows -> {p}")
        print(out.tail(5).to_string())
        return p
    except Exception as e:
        smoke_log(FAMILY, False, repr(e))
        print(f"FAIL {FAMILY}: {e}")
        raise


def run_full():
    return main()


if __name__ == "__main__":
    main()
