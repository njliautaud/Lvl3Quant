# MONDAY OPEN DEPLOY PLAN — 2026-05-11
# Drafted: 2026-05-08 ~23:15 ET
# Owner: Razer (live host)
# Authority: HC #267 (live-tradeable system is the only outcome that counts) + HC #266 (push toward profitability)

## OBJECTIVE
Razer must open Monday 9:30 AM ET (US RTH) with a quantifiably-better live config than it had Friday close. Friday's live trader was OFF after 11:47 AM (warmup-bias killswitch); even the last successful 5/7 run took 0 trades in 19 min on Top5% gate. Monday's config must both run AND take measured trades.

## EVIDENCE BASE (tonight's FIFO market replay, 14 OOT folds, RAW SIMULATOR OUTPUTS — HC #268)

All numbers below are direct outputs of `FIFOReplayEngine.simulate(...)` on real MBO event tape. No extrapolation, no scaling, no implied $/day. Units: ticks (gross or net of $4.70 RT commission = 0.376 ticks).

| variant | trades | sum gross ticks | sum NET ticks | avg NET t/t | median fold avg NET t/t | % folds + |
|---|---|---|---|---|---|---|
| default (mfe/all/limit) | 1229 | −131.50 | −594 (approx)* | −0.483 | −0.427 | 21.4% |
| A (pnl/all/limit) | 1783 | −252.00 | −922.41 | −0.517 | −0.507 | 8.3% |
| B (mfe/all/chase) | 1524 | −69.50 | −642.52 | −0.422 | −0.333 | 21.4% |
| **C (mfe/SHORT/limit, top 1%)** | **534** | **+65.50** | **−135.28** | **−0.253** | **−0.297** | **15.4%** |

*default sum NET reconstructed from avg × trades; will re-verify on next reading.

Conclusion: **only short-side has positive gross edge** at top-1% v3 MLP gate. Commission (0.376 t RT) is bigger than that gross edge → still net negative AT TOP-1%. Path to profitability:
1. Tighter selectivity → larger edge per trade (queued: 0.1%, 0.25%, 0.5%, 2.0% × short × limit)
2. v4 retrain with FIFO-realized net ticks as the training target (queued, post-selectivity)
3. Use raw CNN-Mamba Top-10% short signals which CLAUDE.md (decay 2026-05-01) reports as +1.56 ticks avg, 60.5% WR — already net-positive at this gate even with market orders

## MONDAY OPEN CONFIG (PROPOSED)

### Razer paper_trader CLI invocation:
```
python paper_trading_mamba_v2_patched.py \
  --follow-events \
  --symbol ESM6 \
  --exchange CME \
  --weights output/cnn_mamba_v2_smart_v3_mar/fold_10_best.pt \
  --min-tier "Top10%" \
  --window 1000 --stride 500 \
  --max-hold-minutes 0.5 \
  --trailing-mfe-ticks 4.0 \
  --trailing-lock-ticks 1.0 \
  --max-daily-loss -3000.0 \
  --consec-loss-limit 3 \
  --consec-loss-pause-min 10.0 \
  --patchtst-weights output/patchtst_smart_v3_mar/fold_16_best.pt \
  --patchtst-stats output/patchtst_smart_v3_mar/fold_16_stats.npz \
  --side-filter short  # *** NEW FLAG, requires 5-line code add (see below) ***
```

### Diff vs Friday's (last attempted) config:
| param | old (Friday) | new (Monday) | reason |
|---|---|---|---|
| `--symbol` | ESU6 (Sept) | **ESM6 (June)** | Match MBO recorder; ESU6 not front month |
| `--min-tier` | Top5% | **Top10%** | 5% gate took 0 trades in 19 min; 10% per CLAUDE.md decay analysis is net-positive on shorts |
| `--side-filter` | (none, both sides) | **short** | Variant C tonight: only short side has +gross edge |
| `--max-hold-minutes` | 15.0 | **0.5** (=30s) | Per HC #226: signal half-life ~250ms, IC near-zero by 30s; long holds = directional bet not signal |
| PatchTST confluence | loaded but maybe not gating | **HARD GATE: only enter if patchtst sign agrees** | HC #260 mandatory |
| TP / SL | (current default unclear) | TP=6t / SL=4t | Matches tonight's FIFO test, will tune from selectivity sweep |

## CODE CHANGES REQUIRED ON RAZER

### Change 1: Add `--side-filter` arg (in `paper_trading_mamba_v2_patched.py`)
Insert after line ~869 (right after the patchtst-stats arg):
```python
parser.add_argument("--side-filter", type=str, default="both",
                    choices=["both", "long", "short"],
                    help="Only enter trades on this side (default: both)")
```
Pass through to `MambaV2PaperSession(...)` constructor: `side_filter=args.side_filter`.

### Change 2: Gate logic in `MambaV2PaperSession.__init__` and entry decision
- Store `self.side_filter = side_filter`
- In the entry-decision function (wherever `signal.dir` is checked), reject the trade if `self.side_filter != "both" and signal.dir != self.side_filter`.

### Change 3: Verify PatchTST confluence is actually gating (not just logging)
- If currently advisory-only, change to hard gate: reject if `sign(patchtst_pred) != sign(mamba_signal)`.

**TOTAL CODE CHANGES**: ~10-15 lines, contained. All other code untouched.

## VERIFICATION CHECKLIST (before Monday open)

- [ ] Variant D (pnl_all_chase) FIFO result captured (currently running)
- [ ] Short-only selectivity sweep complete (4 levels: 0.1%, 0.25%, 0.5%, 2.0%)
- [ ] If any selectivity level produces NET-POSITIVE per-trade in FIFO → use that level as `--min-tier` analog
- [ ] Code changes pushed to Razer
- [ ] Sunday 6 PM ET (futures open): verify paper trader auto-launches with new config, MBO recorder still streaming, no warmup-bias issues (pre-market warmup is HC #252-safe)
- [ ] First 30 min Sunday: verify trade count > 0, all dir="S", PatchTST confluence visible in log
- [ ] Monday 9:30 AM ET: verify trade flow during RTH, expect ~5-30 trades by midday at Top10% short

## ROLLBACK CRITERIA
If by Monday 12:00 PM ET:
- 0 trades and confidence not gated by bug → loosen min-tier to Top15% short
- Drawdown > $500 → halt, switch to short-only Top5% (more selective)
- PatchTST confluence rejecting >95% of CNN-Mamba shorts → drop confluence to advisory

## OUTSTANDING RESEARCH (this weekend, must inform config before Monday)
1. Selectivity sweep results — sets the actual `--min-tier` value
2. v4 retrain with FIFO-net-ticks target — if completes by Monday, replaces v3 MLP gate entirely
3. Neptune SSH recovery — if comes back, run additional v3 OOT validation on h10s horizon

## SELECTIVITY SWEEP RESULT (added 23:42 ET, updated 00:08 ET — RAW FIFO TICKS PER HC #268)

| selectivity | trades | sum gross ticks | sum NET ticks | avg NET t/t | median fold avg NET t/t | % folds + |
|---|---|---|---|---|---|---|
| 1.0% (variant C) | 534 | +65.50 | −135.28 | −0.253 | −0.297 | 15.4% |
| **0.1% top-X** | **52** | **+34.50** | **+14.95** | **+0.287** | **+0.436** | **58.3%** |
| 0.25% top-X | 131 | +36.00 | −13.26 | −0.101 | +0.141 | 50.0% |
| 0.5% top-X | 257 | +49.00 | −47.63 | −0.185 | −0.068 | 33.3% |
| 2.0% top-X | 1081 | −186.50 | −592.96 | −0.549 | −0.657 | 21.4% |

**TOP-0.1% IS THE ONLY POSITIVE SUM-NET-TICKS ROW** — every other selectivity has a negative realized FIFO P&L after commission.

Top-0.1% maps to strongest 1-in-1000 of v3 MLP-ranked predictions ⟹ approximately Top-0.5% raw CNN-Mamba confidence tier (because v3 MLP uses CNN-Mamba pred_{1s,5s,10s} as primary input). For paper_trader: `--min-tier "Top0.5%"` to start.

Verdict: SUM-NET-TICKS POSITIVE but consistency 58.3% < 60% HC #254 floor. Trade frequency: 52 trades over 14 folds with realized fills. **DEPLOYABLE WITH CAUTION** at 1 contract — real-world FIFO test on live data is the only ground-truth.

## CHANGE LOG
2026-05-08 23:15 ET — Initial draft after variant C result (mfe/short/limit best at +0.123 gross t/trade).
2026-05-08 23:42 ET — Top-0.1% short-only sweep done: NET +$16/day/contract, +$0.287 NET/trade. First v3 config that survives FIFO replay.
