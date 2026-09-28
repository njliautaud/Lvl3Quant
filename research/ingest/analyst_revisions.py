"""
Family: analyst_revisions  (HC #563 R2 — analyst consensus + revisions)

WHAT: Per-ticker analyst recommendation history. yfinance exposes a quarterly
recommendation trend table (strongBuy / buy / hold / sell / strongSell counts
per "period") and a longer recommendations DataFrame depending on the ticker.
We pull whatever's available, normalise to (ticker, period_end, ...) and
compute QoQ deltas.

SOURCE: yfinance.Ticker(t).recommendations_summary  (and .recommendations as
fallback). Free.

OUTPUT: data/feature_store/analyst_revisions/daily.parquet
Schema (per (ticker, period_end)):
  ticker, period_end, strong_buy, buy, hold, sell, strong_sell,
  net_score (strongBuy*2 + buy - sell - strongSell*2),
  net_score_delta_qoq
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/home/jupiter/Lvl3Quant/research/ingest")
from _common import write_parquet, smoke_log  # type: ignore

FAMILY = "analyst_revisions"

# Universe = full master_panel coverage (~247 tickers as of HC #568 / 2026-06-08).
# Falls back to a hard-coded core list if the panel isn't available.
def _load_universe():
    fallback = [
        "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA",
        "JPM","BAC","WFC","GS","MS",
        "UNH","JNJ","PFE","LLY","ABBV",
        "XOM","CVX","COP",
        "WMT","HD","COST","MCD","NKE",
        "BA","CAT","AMD","INTC","CRM",
        "DIS","NFLX",
    ]
    try:
        panel = Path("/home/jupiter/Lvl3Quant/data/feature_store/master_panel/master_panel_v2.parquet")
        if panel.exists():
            uni = sorted(pd.read_parquet(panel, columns=["ticker"])["ticker"].unique().tolist())
            if len(uni) >= 50:
                return uni
    except Exception as e:
        print(f"[analyst_revisions] could not load master_panel universe ({e!r}) — falling back")
    return fallback

UNIVERSE = _load_universe()
print(f"[analyst_revisions] UNIVERSE size = {len(UNIVERSE)}")


def _pull_one(yf, ticker: str) -> pd.DataFrame | None:
    try:
        t = yf.Ticker(ticker)
        # Newer yfinance — recommendations is sometimes a summary table
        rs = None
        try:
            rs = t.recommendations_summary
        except Exception:
            rs = None
        if rs is None or (hasattr(rs, "empty") and rs.empty):
            try:
                rs = t.recommendations
            except Exception:
                rs = None
        if rs is None or rs.empty:
            return None
        rs = rs.copy()
        # Normalise column names
        rs.columns = [c.lower() for c in rs.columns]
        col_map = {
            "strongbuy": "strong_buy",
            "buy":       "buy",
            "hold":      "hold",
            "sell":      "sell",
            "strongsell":"strong_sell",
        }
        rename = {k: v for k, v in col_map.items() if k in rs.columns}
        rs = rs.rename(columns=rename)
        keep = [v for v in col_map.values() if v in rs.columns]
        if not keep:
            return None
        # period_end: either index (DateTimeIndex) or a "period" column
        if isinstance(rs.index, pd.DatetimeIndex):
            rs = rs.reset_index().rename(columns={rs.index.name or "index": "period_end"})
        elif "period" in rs.columns:
            # period like "0m", "-1m", "-2m" = relative months ago; map to month_end
            now = pd.Timestamp.utcnow().normalize()
            def _to_dt(p):
                try:
                    m = int(str(p).replace("m", ""))
                    return (now + pd.DateOffset(months=m)).normalize()
                except Exception:
                    return pd.NaT
            rs["period_end"] = rs["period"].map(_to_dt)
        else:
            rs["period_end"] = pd.NaT
        rs = rs[["period_end"] + keep].copy()
        rs["ticker"] = ticker
        return rs
    except Exception as e:
        print(f"  {ticker} ERR {e!r}")
        return None


def main():
    import yfinance as yf
    parts = []
    ok, fail = 0, 0
    for t in UNIVERSE:
        df = _pull_one(yf, t)
        if df is None or df.empty:
            print(f"[{FAMILY}] {t} no data")
            fail += 1
            continue
        parts.append(df)
        ok += 1
        print(f"[{FAMILY}] {t} ok {len(df)} rows")
        time.sleep(0.4)

    if not parts:
        smoke_log(FAMILY, False, "no analyst data")
        raise RuntimeError("yfinance returned no analyst recs for any ticker")

    out = pd.concat(parts, ignore_index=True)
    for c in ["strong_buy","buy","hold","sell","strong_sell"]:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out["net_score"] = (
        2 * out["strong_buy"] + out["buy"] - out["sell"] - 2 * out["strong_sell"]
    )
    out = out.sort_values(["ticker","period_end"]).reset_index(drop=True)
    out["net_score_delta_qoq"] = out.groupby("ticker")["net_score"].diff()

    out = out[["ticker","period_end",
               "strong_buy","buy","hold","sell","strong_sell",
               "net_score","net_score_delta_qoq"]]
    p = write_parquet(out, FAMILY, "daily.parquet")
    smoke_log(FAMILY, True, f"{len(out)} rows ok={ok} fail={fail} -> {p}")
    print(f"OK {FAMILY}: {len(out)} rows ({ok} tickers ok, {fail} failed) -> {p}")
    print(out.head(5).to_string())
    return p


def run_full():
    return main()


if __name__ == "__main__":
    main()
