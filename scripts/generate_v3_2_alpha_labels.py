"""
Generate v3.2 ALPHA-FIRST labels (HC #475 R4 + HC #432 R2 + HC #485 R1-R5).

Adds MFE-within-horizon and MAE-within-horizon labels for horizons
h in {1s, 5s, 10s, 30s, 60s}, computed from an actual MBO trade-stream
replay (HC #74) with top-of-book fall-back when no trade is present in
the window.

For each event i at price p_i, for each horizon h:
  target_mfe_<h>s_ticks = max(p_k - p_i)  over k in (i, j_h]    (long-side)
  target_mae_<h>s_ticks = min(p_k - p_i)  over k in (i, j_h]    (long-side)

Short-side targets are just the negatives — the trainer can flip sign
per side. We write the long-side convention only (less storage; same
information).

Constants: TICK_SIZE = 0.25, TICK_VALUE = $12.50. Prices in v3 corpus
are already in TICK UNITS (mid = (bid+ask)/2 where bid/ask are tick
offsets around an unknown session anchor) — the anchor cancels in
(p_k - p_i), so MFE/MAE are correct in ticks without further
normalization.

Day-boundary handling (HC #485 R1, R2):
  If t_i + h > ts[-1] (event near end of session — horizon would cross
  session close), the label is NaN. We DO NOT zero-fill (HC #477 bug).
  Per-date NaN audit printed to stdout AND appended to
  per_date_nan_audit.tsv for every (date, new_column) pair.

Trade-stream price series (HC #74):
  p[k] = trade_price if event_type_raw[k] ∈ {3 (TRADE), 4 (FILL)},
         else top-of-book mid (last-known TOB carried through).
  trade_price = mid[k] + ev[k, 3] * 25.0   (price_rel_ticks * 25 to
  un-normalize the smart-treated price column).

Outputs:
  data/processed/mbo_events_smart_v3_alpha_labels_v4/<date>_alpha_labels.npz
    — all v3 columns + 10 new MFE/MAE columns (5 horizons x 2)
  data/processed/mbo_events_smart_v3_alpha_labels_v4/per_date_nan_audit.tsv
  data/processed/mbo_events_smart_v3_alpha_labels_v4/summary_nan_audit.txt
  data/processed/mbo_events_smart_v3_alpha_labels_v4/.regen_complete.json
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

# Reuse v3.1's windowed min/max + two-pointer helpers (no functional changes).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_v3_1_alpha_labels import (  # type: ignore
    _two_pointer_indices,
    _compute_min_max_windowed,
    SEC_NS,
)

# ── Constants ───────────────────────────────────────────────────────
TICK_SIZE = 0.25
TICK_VALUE = 12.50

# New horizons to label
HORIZONS_SECS = [1, 5, 10, 30, 60]
HORIZON_WIN_NS = {h: h * SEC_NS for h in HORIZONS_SECS}
NEW_COLS = []
for h in HORIZONS_SECS:
    NEW_COLS.append(f"target_mfe_{h}s_ticks")
    NEW_COLS.append(f"target_mae_{h}s_ticks")

# Event-type encoding (from precompute_features_smart_v3.py)
ET_TRADE = 3
ET_FILL = 4

DEFAULT_V3_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_alpha_labels_v3")
DEFAULT_EVENTS_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
DEFAULT_BOOK_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_book_features")
DEFAULT_OUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_alpha_labels_v4")
DEFAULT_REPORT_DIR = Path("/home/jupiter/Lvl3Quant/output/v3_2_alpha_labels_report")


def _build_price_series(ev_array: np.ndarray, etr: np.ndarray, mids: np.ndarray) -> np.ndarray:
    """
    HC #74 FIFO trade-stream replay price path.

    For TRADE (3) / FILL (4) events: p = mid + price_rel_ticks*25 (i.e.
    actual trade execution price in ticks).
    For all other events: p = last-known top-of-book mid (we use the
    per-event mid which is already a TOB snapshot — it IS the last-known
    TOB at event i, so no forward-fill needed beyond NaN handling).

    Returns price array in tick units. NaN where mid is NaN (invalid spread).
    """
    trade_mask = (etr == ET_TRADE) | (etr == ET_FILL)
    # ev_array col 3 = price_rel_ticks AFTER smart-treatment (clipped to ±50
    # and divided by 25.0). To recover ticks: multiply by 25.
    trade_offset_ticks = ev_array[:, 3].astype(np.float64) * 25.0
    p = mids.copy()  # already in ticks
    # Where it's a trade and mid is valid, use trade price
    use_trade = trade_mask & np.isfinite(mids)
    p[use_trade] = mids[use_trade] + trade_offset_ticks[use_trade]
    return p


def _label_within_horizon(p_path: np.ndarray, ts_ns: np.ndarray, h_secs: int) -> tuple:
    """
    Compute MFE_h, MAE_h (long-side) per event i over window (i, j_h].

    Boundary rule (HC #485 R1): if t_i + h > ts[-1] (no event observed
    h seconds into future), label is NaN. We detect this via searchsorted:
    the two-pointer j_h returns len(ts) ONLY when target > ts[-1], i.e.
    the horizon crosses end-of-data → boundary NaN.
    """
    horizon_ns = h_secs * SEC_NS
    j_h = _two_pointer_indices(ts_ns, horizon_ns)
    N = len(ts_ns)

    # NaN-safe sentinels for monotonic-deque min/max
    p_for_max = np.where(np.isnan(p_path), -np.inf, p_path)
    p_for_min = np.where(np.isnan(p_path), np.inf, p_path)

    _, max_in_win = _compute_min_max_windowed(p_for_max, j_h, return_argmax=False)
    min_in_win, _ = _compute_min_max_windowed(p_for_min, j_h, return_argmax=False)

    mfe = max_in_win - p_path  # long-side: positive = favorable
    mae = min_in_win - p_path  # long-side: negative = adverse

    # Propagate NaN where signal price invalid or window empty / out-of-data
    sig_nan = np.isnan(p_path)
    boundary_nan = j_h >= N           # crosses session close → NaN per spec
    empty_win = j_h <= np.arange(N) + 1  # j_h didn't advance past i+1

    invalid = sig_nan | boundary_nan | empty_win
    mfe = np.where(invalid | np.isinf(mfe), np.nan, mfe).astype(np.float32)
    mae = np.where(invalid | np.isinf(mae), np.nan, mae).astype(np.float32)

    return mfe, mae


def process_one_date(date_str: str,
                     v3_label_dir: Path,
                     events_dir: Path,
                     book_dir: Path,
                     out_dir: Path,
                     audit_tsv_fp,
                     overwrite: bool = False) -> dict:
    out_path = out_dir / f"{date_str}_alpha_labels.npz"
    if out_path.exists() and not overwrite:
        return {"date": date_str, "status": "skipped_exists", "out": str(out_path)}

    v3_path = v3_label_dir / f"{date_str}_alpha_labels.npz"
    events_path = events_dir / f"{date_str}_mbo_events.npz"
    book_path = book_dir / f"{date_str}_book_features.npz"
    for p in (v3_path, events_path, book_path):
        if not p.exists():
            return {"date": date_str, "status": "missing_inputs", "missing": str(p)}

    t0 = time.time()

    # Load v3 labels (pass-through). Keep references so we can re-emit them.
    v3 = np.load(v3_path, allow_pickle=True)
    v3_cols = {k: v3[k] for k in v3.files}

    # Load events + book for price-path reconstruction
    ev_npz = np.load(events_path, allow_pickle=True)
    ev_array = ev_npz["events"]
    etr = ev_npz["event_type_raw"]
    ts_ns = ev_npz["timestamps"].astype(np.int64)
    N = len(ts_ns)

    bk = np.load(book_path, allow_pickle=True)
    feats = bk["features"]
    bid = feats[:, 0].astype(np.float64)
    ask = feats[:, 5].astype(np.float64)
    mids = (bid + ask) / 2.0  # ticks (signed offsets — anchor cancels in diffs)

    if len(mids) != N or len(etr) != N:
        return {"date": date_str, "status": "length_mismatch",
                "N_events": N, "N_book": len(mids), "N_etr": len(etr)}

    # Same spread-validity gate as v3.1 (HC #485 corrected)
    spread = ask - bid
    invalid_mask = (spread <= 0) | (spread >= 20.0) | ~np.isfinite(spread)
    mids_clean = mids.copy()
    mids_clean[invalid_mask] = np.nan

    # Build trade-stream price path (HC #74)
    p_path = _build_price_series(ev_array, etr, mids_clean)

    # Compute MFE/MAE for each horizon
    new_arrays = {}
    audit_lines = []
    for h in HORIZONS_SECS:
        mfe, mae = _label_within_horizon(p_path, ts_ns, h)
        mfe_col = f"target_mfe_{h}s_ticks"
        mae_col = f"target_mae_{h}s_ticks"
        new_arrays[mfe_col] = mfe
        new_arrays[mae_col] = mae
        for col, arr in ((mfe_col, mfe), (mae_col, mae)):
            nan_frac = float(np.isnan(arr).mean())
            line = f"{date_str}: col={col} nan_frac={nan_frac:.6f} N={N}"
            print(line, flush=True)
            audit_tsv_fp.write(f"{date_str}\t{col}\t{nan_frac:.6f}\t{N}\n")
            audit_lines.append((col, nan_frac))

    # Merge & write
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = dict(v3_cols)
    merged.update(new_arrays)
    np.savez_compressed(out_path, **merged)

    elapsed = time.time() - t0
    return {
        "date": date_str, "status": "ok", "N": N, "elapsed_s": elapsed,
        "out": str(out_path),
        "nan_fracs": {c: f for c, f in audit_lines},
    }


def _get_git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=str(Path(__file__).resolve().parent.parent),
                                       stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=None,
                    help="Optional date subset; default = all dates in v3 label dir")
    ap.add_argument("--v3-dir", type=Path, default=DEFAULT_V3_DIR)
    ap.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR)
    ap.add_argument("--book-dir", type=Path, default=DEFAULT_BOOK_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke-only", action="store_true",
                    help="Run 3-date smoke test only, no bulk regen")
    ap.add_argument("--smoke-dates", nargs="*", default=None,
                    help="Override the auto-picked smoke-test dates")
    ap.add_argument("--max-walltime-min", type=float, default=85.0,
                    help="Walltime budget; will write PARTIAL status if exceeded")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.time()
    walltime_budget = args.max_walltime_min * 60.0
    git_sha = _get_git_sha()

    # Resolve date list
    if args.dates:
        all_dates = sorted(args.dates)
    else:
        import re
        all_dates = sorted([
            re.match(r"(\d{8})_alpha_labels.npz", f).group(1)
            for f in os.listdir(args.v3_dir)
            if re.match(r"\d{8}_alpha_labels.npz", f)
        ])

    print(f"[v3.2-labels] {len(all_dates)} dates in v3 corpus", flush=True)

    # ── HC #485 R2: Smoke-test (3 representative dates) ───────────
    if args.smoke_dates:
        smoke_dates = args.smoke_dates
    else:
        # Pick smoke-test trio:
        #  - largest by events (RTH-heavy)
        #  - smallest by events (Sunday-ish)
        #  - first date (boundary)
        sizes = []
        for d in all_dates:
            ev_p = args.events_dir / f"{d}_mbo_events.npz"
            if ev_p.exists():
                sz = ev_p.stat().st_size
                sizes.append((d, sz))
        sizes.sort(key=lambda x: x[1])
        smallest = sizes[0][0] if sizes else all_dates[0]
        largest = sizes[-1][0] if sizes else all_dates[-1]
        boundary = all_dates[0]
        # Dedup while keeping order
        smoke_dates = []
        for d in (largest, smallest, boundary):
            if d not in smoke_dates:
                smoke_dates.append(d)

    print(f"[v3.2-labels] smoke-test dates: {smoke_dates}", flush=True)

    audit_tsv_path = args.out_dir / "per_date_nan_audit.tsv"
    if args.overwrite or not audit_tsv_path.exists():
        audit_tsv_path.write_text("date\tcol\tnan_frac\tN\n")
    audit_fp = open(audit_tsv_path, "a")

    smoke_results = []
    smoke_abort = False
    smoke_abort_msg = ""
    for d in smoke_dates:
        r = process_one_date(d, args.v3_dir, args.events_dir, args.book_dir,
                             args.out_dir, audit_fp, overwrite=args.overwrite)
        smoke_results.append(r)
        print(f"[smoke] {d}: {r.get('status')} elapsed={r.get('elapsed_s', -1):.1f}s",
              flush=True)
        if r["status"] != "ok":
            smoke_abort = True
            smoke_abort_msg = f"smoke-test date {d} failed: {r}"
            break
        for col, nf in r.get("nan_fracs", {}).items():
            # ALLOW high NaN on boundary date for short horizons? No — even
            # boundary day has only the last h seconds in NaN territory.
            # For RTH dates of full session (~6h+), nan_frac should be << 0.10.
            # Tolerate >0.10 ONLY if it's the smallest (Sunday) date — those
            # are partial-session and may have a higher boundary share.
            if nf > 0.10 and d != smoke_dates[1]:  # smallest = Sunday
                smoke_abort = True
                smoke_abort_msg = (f"smoke-test FAIL: {d} {col} nan_frac={nf:.4f} > 0.10 "
                                    "(not explainable by session close)")
                break
        if smoke_abort:
            break

    if smoke_abort:
        print(f"[v3.2-labels] ABORT (smoke): {smoke_abort_msg}", flush=True)
        audit_fp.close()
        # Write partial report
        _write_report(args.report_dir, status="ABORT",
                      reason=smoke_abort_msg, smoke_results=smoke_results,
                      corpus_results=[], smoke_dates=smoke_dates,
                      started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                      n_files_in=len(all_dates), worst_nan={}, avg_nan={}, git_sha=git_sha)
        # Done marker even on abort (negative outcome captured)
        (args.out_dir / ".regen_complete.json").write_text(json.dumps({
            "status": "ABORT_SMOKE",
            "reason": smoke_abort_msg,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "n_files_in": len(all_dates),
            "n_files_out": len(smoke_results),
            "fix_commit_sha": git_sha,
        }, indent=2))
        return 2

    print(f"[v3.2-labels] smoke-test PASS — proceeding to bulk regen", flush=True)
    if args.smoke_only:
        audit_fp.close()
        _write_report(args.report_dir, status="SMOKE_ONLY",
                      reason="--smoke-only flag", smoke_results=smoke_results,
                      corpus_results=[], smoke_dates=smoke_dates,
                      started_at=started_at,
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      n_files_in=len(all_dates), worst_nan={}, avg_nan={},
                      git_sha=git_sha)
        return 0

    # ── Bulk regen ────────────────────────────────────────────────
    remaining = [d for d in all_dates if d not in {r["date"] for r in smoke_results}]
    corpus_results = list(smoke_results)
    partial = False
    for d in remaining:
        if (time.time() - wall_start) > walltime_budget:
            print(f"[v3.2-labels] WALLTIME EXCEEDED ({walltime_budget/60:.0f}min) — stopping early",
                  flush=True)
            partial = True
            break
        r = process_one_date(d, args.v3_dir, args.events_dir, args.book_dir,
                             args.out_dir, audit_fp, overwrite=args.overwrite)
        corpus_results.append(r)
        print(f"  {d}: status={r['status']} elapsed={r.get('elapsed_s', -1):.1f}s",
              flush=True)

    audit_fp.close()

    # ── HC #485 R3 post-regen summary audit ───────────────────────
    worst_nan = {c: 0.0 for c in NEW_COLS}
    sum_nan = {c: 0.0 for c in NEW_COLS}
    cnt_nan = {c: 0 for c in NEW_COLS}
    bad_dates = []
    for r in corpus_results:
        if r.get("status") != "ok":
            continue
        for c, nf in r.get("nan_fracs", {}).items():
            worst_nan[c] = max(worst_nan[c], nf)
            sum_nan[c] += nf
            cnt_nan[c] += 1
            if nf > 0.50:
                bad_dates.append((r["date"], c, nf))
    avg_nan = {c: (sum_nan[c] / cnt_nan[c]) if cnt_nan[c] else float("nan") for c in NEW_COLS}

    summary_lines = ["v3.2 alpha labels — corpus NaN audit"]
    summary_lines.append(f"n_files_in={len(all_dates)} n_files_out={sum(1 for r in corpus_results if r.get('status')=='ok')}")
    summary_lines.append(f"started_at={started_at}")
    summary_lines.append(f"finished_at={datetime.now(timezone.utc).isoformat()}")
    summary_lines.append("col\tavg_nan_frac\tworst_nan_frac")
    for c in NEW_COLS:
        summary_lines.append(f"{c}\t{avg_nan[c]:.6f}\t{worst_nan[c]:.6f}")
    if bad_dates:
        summary_lines.append("\nDATES WITH >0.50 NaN ON OPEN-MARKET DAY:")
        for d, c, nf in bad_dates:
            summary_lines.append(f"  {d} {c} {nf:.4f}")
    (args.out_dir / "summary_nan_audit.txt").write_text("\n".join(summary_lines) + "\n")

    abort_post = False
    abort_reasons = []
    for c, a in avg_nan.items():
        if a > 0.10:
            abort_post = True
            abort_reasons.append(f"avg nan_frac for {c} = {a:.4f} > 0.10")
    if bad_dates:
        # Per spec: any single open-market date > 0.50 → ABORT.
        # Filter ALL closed-market dates first:
        #   (a) Sundays (CME ES closed pre-Mon-open)
        #   (b) known US holidays in dataset range (Christmas, New Year's Day, etc.)
        # Then count UNIQUE bad dates, not (date,col) tuples.
        CME_HOLIDAYS = {
            "20251225",  # Christmas Day
            "20260101",  # New Year's Day
            "20260119",  # MLK Day (US bank holiday)
            "20260216",  # Presidents Day
            # add other CME-closed sessions here as needed
        }
        def _is_closed_market(d_str: str) -> bool:
            if d_str in CME_HOLIDAYS:
                return True
            try:
                wd = datetime.strptime(d_str, "%Y%m%d").weekday()
                return wd == 6  # Sunday
            except Exception:
                return False
        bad_open_dates = {b[0] for b in bad_dates if not _is_closed_market(b[0])}
        if bad_open_dates:
            abort_post = True
            abort_reasons.append(
                f"{len(bad_open_dates)} open-market dates with NaN>0.50 "
                f"(unique dates, after filtering closed-market sessions)"
            )

    n_ok = sum(1 for r in corpus_results if r.get("status") == "ok")
    finished_at = datetime.now(timezone.utc).isoformat()
    status_str = "PARTIAL" if partial else ("ABORT" if abort_post else "COMPLETE")

    # ── HC #485 R5 done-marker ─────────────────────────────────────
    done_marker = {
        "status": status_str,
        "started_at": started_at,
        "finished_at": finished_at,
        "n_files_in": len(all_dates),
        "n_files_out": n_ok,
        "n_corrupt": sum(1 for r in corpus_results if r.get("status") not in ("ok", "skipped_exists")),
        "worst_nan_frac": worst_nan,
        "avg_nan_frac": avg_nan,
        "fix_commit_sha": git_sha,
        "abort_reasons": abort_reasons,
    }
    (args.out_dir / ".regen_complete.json").write_text(json.dumps(done_marker, indent=2))

    _write_report(args.report_dir, status=status_str,
                  reason="; ".join(abort_reasons) if abort_reasons else "ok",
                  smoke_results=smoke_results, corpus_results=corpus_results,
                  smoke_dates=smoke_dates, started_at=started_at,
                  finished_at=finished_at, n_files_in=len(all_dates),
                  worst_nan=worst_nan, avg_nan=avg_nan, git_sha=git_sha)

    print(f"[v3.2-labels] {status_str} n_ok={n_ok}/{len(all_dates)} "
          f"elapsed={(time.time()-wall_start)/60:.1f}min", flush=True)
    return 0 if status_str == "COMPLETE" else (1 if status_str == "PARTIAL" else 2)


def _write_report(report_dir: Path, status: str, reason: str,
                  smoke_results: list, corpus_results: list,
                  smoke_dates: list, started_at: str, finished_at: str,
                  n_files_in: int, worst_nan: dict, avg_nan: dict,
                  git_sha: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# v3.2 Alpha Labels (MFE/MAE within horizon) — STATUS={status}")
    lines.append("")
    lines.append(f"- started_at: {started_at}")
    lines.append(f"- finished_at: {finished_at}")
    lines.append(f"- git_sha: {git_sha}")
    lines.append(f"- n_files_in: {n_files_in}")
    n_ok = sum(1 for r in corpus_results if r.get("status") == "ok")
    lines.append(f"- n_files_out (ok): {n_ok}")
    lines.append(f"- reason: {reason}")
    lines.append("")
    lines.append("## Smoke-test (HC #485 R2)")
    lines.append("")
    lines.append(f"Dates: {smoke_dates}")
    lines.append("")
    lines.append("| date | status | N | elapsed_s |")
    lines.append("|------|--------|---|-----------|")
    for r in smoke_results:
        lines.append(f"| {r['date']} | {r.get('status')} | {r.get('N','?')} | {r.get('elapsed_s',-1):.1f} |")
    lines.append("")
    lines.append("### Smoke-test NaN fractions")
    lines.append("")
    lines.append("| date | column | nan_frac |")
    lines.append("|------|--------|----------|")
    for r in smoke_results:
        for c, nf in r.get("nan_fracs", {}).items():
            lines.append(f"| {r['date']} | {c} | {nf:.6f} |")
    lines.append("")
    lines.append("## Corpus summary (HC #485 R3)")
    lines.append("")
    lines.append("| column | avg_nan_frac | worst_nan_frac |")
    lines.append("|--------|--------------|----------------|")
    for c in NEW_COLS:
        lines.append(f"| {c} | {avg_nan.get(c, float('nan')):.6f} | {worst_nan.get(c, float('nan')):.6f} |")
    lines.append("")

    # Sample MFE/MAE distribution medians (from smoke-test largest day,
    # if available, since we still have it on disk).
    if smoke_results and smoke_results[0].get("status") == "ok":
        out_path = Path(smoke_results[0]["out"])
        if out_path.exists():
            try:
                z = np.load(out_path, allow_pickle=True)
                lines.append("## Sample distribution medians "
                             f"(date={smoke_results[0]['date']})")
                lines.append("")
                lines.append("| col | median | p10 | p90 |")
                lines.append("|-----|--------|-----|-----|")
                for c in NEW_COLS:
                    a = z[c]
                    fa = a[np.isfinite(a)]
                    if fa.size:
                        med = float(np.median(fa))
                        p10 = float(np.percentile(fa, 10))
                        p90 = float(np.percentile(fa, 90))
                        lines.append(f"| {c} | {med:.4f} | {p10:.4f} | {p90:.4f} |")
            except Exception as e:
                lines.append(f"(could not load sample distributions: {e})")

    (report_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
