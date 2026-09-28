"""
materialize_chains_parallel.py — HC #556 R3(5) / lane A3.

Resumable, parallel materialization of per-ticker option chains from the
local DOLT `post-no-preference/options` clone into
data/cache/options_real/chains/{TICKER}.parquet.

- Skips tickers whose parquet already exists (resume-safe).
- Runs N dolt-sql subprocesses concurrently (read-only queries; default 4).
- Reuses query/filter logic from ingest_options_real.materialize_chains_for_ticker
  (dte 5-90 band, wheel-relevant tenor).

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python -m data.materialize_chains_parallel [--workers 4] [--tickers A,B]
"""
from __future__ import annotations
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.ingest_options_real import (  # noqa: E402
    CHAINS_DIR, DOLT_DIR, _dolt_path, _universe_tickers,
    materialize_chains_for_ticker,
)


def _do_one(ticker: str, dolt_bin: str) -> str:
    out = CHAINS_DIR / f"{ticker}.parquet"
    if out.exists():
        return f"{ticker}: SKIP (exists)"
    t0 = time.time()
    df = materialize_chains_for_ticker(ticker, dolt_bin)
    if df is None or df.empty:
        # touch a sentinel so we don't re-query empties on resume
        (CHAINS_DIR / f"{ticker}.EMPTY").write_text("no rows")
        return f"{ticker}: EMPTY ({time.time()-t0:.0f}s)"
    df.to_parquet(out, index=False)
    return (f"{ticker}: OK {len(df):,} rows "
            f"{df['date'].min().date()}->{df['date'].max().date()} "
            f"({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tickers", default=None)
    args = ap.parse_args()

    if not DOLT_DIR.exists():
        raise SystemExit(f"dolt repo missing at {DOLT_DIR}")
    CHAINS_DIR.mkdir(parents=True, exist_ok=True)
    dolt_bin = _dolt_path()

    restrict = ([t.strip().upper() for t in args.tickers.split(",")]
                if args.tickers else None)
    tickers = _universe_tickers(restrict)
    pending = [t for t in tickers
               if not (CHAINS_DIR / f"{t}.parquet").exists()
               and not (CHAINS_DIR / f"{t}.EMPTY").exists()]
    print(f"[chains-par] universe={len(tickers)} pending={len(pending)} "
          f"workers={args.workers}", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_do_one, t, dolt_bin): t for t in pending}
        for fut in as_completed(futs):
            done += 1
            try:
                msg = fut.result()
            except Exception as e:
                msg = f"{futs[fut]}: FAIL {e}"
            print(f"[chains-par] ({done}/{len(pending)}) {msg}", flush=True)
    print("[chains-par] DONE", flush=True)


if __name__ == "__main__":
    main()
