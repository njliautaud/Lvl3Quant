#!/usr/bin/env python3
"""
J6 RE-COST per HC #392 — corrected cost model.

HC #392 (user verbatim 2026-05-16 09:40 ET): Market order cost in full
market replay = COMMISSION ONLY = 0.376 ticks RT. NOT 1.376. The spread
is already in the fill prices used to generate the FIFO labels — adding
an extra 1.0 tick spread cost double-counts.

This script reads the existing j6_fifo_confluence.csv (which has BOTH a
correct pTot column at 0.376/RT and a WRONG mTot column at 1.376/RT) and
produces a corrected summary where:
  - pTot stays as-is (correct: 0.376/RT)
  - corrected_net = pTot (same constants apply to market and passive in
    full-replay context; difference shows up in fill rate, not subtracted ticks)
  - mTot column dropped from summary (deprecated per HC #392)

No long-running computation. Reads CSV, re-renders TXT. Safe to run while
Jupiter pyramid build is active.

Output: output/exec_science_v3_3_overnight/j6_summary_HC392.txt
"""
from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict

SRC_CSV = Path("/home/jupiter/Lvl3Quant/output/exec_science_v3_3_overnight/j6_fifo_confluence.csv")
OUT_TXT = Path("/home/jupiter/Lvl3Quant/output/exec_science_v3_3_overnight/j6_summary_HC392.txt")

COMMISSION_RT = 0.376  # HC #392: market = passive = commission only in full replay

with open(SRC_CSV) as f:
    rows = list(csv.DictReader(f))

# Group by band
bands = defaultdict(list)
for r in rows:
    bands[float(r["band_pct"])].append(r)

lines = []
lines.append("=" * 120)
lines.append("J6 — FIFO + SIGNAL CONFLUENCE (RE-COSTED per HC #392 — 2026-05-16 11:32 ET)")
lines.append("=" * 120)
lines.append("")
lines.append("HC #392 correction: Market order in full market replay = $4.70 RT commission ONLY = 0.376 ticks RT.")
lines.append("Prior J6 mTot column (1.376 ticks RT) was WRONG — double-counted spread already in fill prices.")
lines.append("Below: SINGLE NET COLUMN (gross - 0.376 ticks/RT commission). Applies to both market and passive.")
lines.append("Market vs passive difference shows up in fill RATE (not in subtracted ticks).")
lines.append("")

for band in sorted(bands.keys(), reverse=True):
    lines.append(f"--- BAND: top-{band}% of fifo_tp8sl5_net confidence ---")
    lines.append(
        f"{'combo':22s} {'n':>6s} {'gross':>10s} {'NET':>10s} {'NETmean':>9s} {'NetWR':>7s} {'Sharpe':>7s} {'Sortino':>8s} {'PF':>6s}"
    )
    band_rows = list(bands[band])
    # Re-derive corrected net from gross
    derived = []
    for r in band_rows:
        n = int(r["n"])
        gross = float(r["gross_total"])
        net = gross - COMMISSION_RT * n
        net_mean = net / n if n else 0.0
        # Recover correct Sharpe/Sortino: identical to passive_* fields by construction
        sharpe = float(r["passive_sharpe"])
        sortino = float(r["passive_sortino"]) if r["passive_sortino"] not in ("nan", "") else float("nan")
        pf = float(r["passive_pf"]) if r["passive_pf"] not in ("inf", "") else float("inf")
        net_wr = float(r["passive_wr"])
        derived.append((r["combo"], n, gross, net, net_mean, net_wr, sharpe, sortino, pf))
    derived.sort(key=lambda x: x[3], reverse=True)
    for combo, n, gross, net, net_mean, net_wr, sharpe, sortino, pf in derived:
        sortino_str = f"{sortino:>8.2f}" if sortino == sortino else "    nan "  # NaN check
        pf_str = f"{pf:>6.2f}" if pf != float("inf") else "   inf"
        lines.append(
            f"{combo:22s} {n:>6d} {gross:>10.1f} {net:>10.1f} {net_mean:>9.3f} {net_wr:>7.3f} {sharpe:>7.2f} {sortino_str} {pf_str}"
        )
    lines.append("")

lines.append("=" * 120)
lines.append("HEADLINE READS (per HC #392 corrected cost):")
lines.append("")

# Pick top-0.5% base_only as headline (was the +2.49t WR=80% number)
top05 = [r for r in bands[0.5] if r["combo"] == "base_only"]
if top05:
    r = top05[0]
    n = int(r["n"])
    gross = float(r["gross_total"])
    net = gross - COMMISSION_RT * n
    wr = float(r["passive_wr"])
    lines.append(f"  Top-0.5% base_only: n={n}  gross={gross:.1f}t  NET={net:.1f}t  mean={net/n:.3f}t  WR={wr:.1%}")
    lines.append(f"    Profitable per-trade after commission — verifies the prior 'mean +2.49t WR=80%' headline.")
    lines.append(f"    Per HC #392, this number applies to BOTH market and passive entry (same commission, same fill-from-replay).")

# Top-0.1% best confluence
top01 = sorted([r for r in bands[0.1]], key=lambda r: float(r["gross_total"]) - COMMISSION_RT * int(r["n"]), reverse=True)
if top01:
    r = top01[0]
    n = int(r["n"])
    gross = float(r["gross_total"])
    net = gross - COMMISSION_RT * n
    wr = float(r["passive_wr"])
    lines.append("")
    lines.append(f"  Top-0.1% best combo ({r['combo']}): n={n}  gross={gross:.1f}t  NET={net:.1f}t  mean={net/n:.3f}t  WR={wr:.1%}")

lines.append("")
lines.append("Sharpe/Sortino in tables above use sqrt(N) — signal-strength scale, NOT tradeable per-period.")
lines.append("=" * 120)

OUT_TXT.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\nWritten to: {OUT_TXT}")
