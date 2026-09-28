# PATCH SPEC — paper_trading_mamba_v2_patched.py
# Add `--side-filter` arg + gate logic for short-only Monday deploy
# Drafted: 2026-05-09 00:18 ET
# Authority: HC #267 (live-tradeable system) + HC #266 (push toward profitability)
# Status: SPEC ONLY — awaiting user authorization before applying

## SCOPE
Three logical changes, ~10 lines total. Adds the ability to restrict trades to one side without removing the existing both-sides default behavior.

## CHANGES

### Change 1 — Add `_side_allowed` helper (around line 308, just after `_tier_meets_min`)
```python
def _side_allowed(self, direction: int) -> bool:
    """Return True if trade direction is allowed by side_filter."""
    if self.side_filter == "both":
        return True
    if self.side_filter == "long":
        return direction > 0
    if self.side_filter == "short":
        return direction < 0
    return True  # unknown filter → permissive default
```

### Change 2 — Add constructor arg + store (around line 199-213)
In `MambaV2PaperSession.__init__`, add to signature:
```python
side_filter: str = "both",
```
After existing `self.min_tier = min_tier`:
```python
self.side_filter = side_filter
log.info("  Side filter: %s", side_filter)
```

### Change 3 — Gate the three entry sites
Three places enter trades. Add `and self._side_allowed(direction)` to each entry guard:

a) Line ~420 (re-entry after risk exit):
```python
if meets_threshold and self.features.is_warm() and self._side_allowed(direction):
```

b) Line ~444 (re-entry after signal flip):
```python
if trade and meets_threshold and self.features.is_warm() and self._side_allowed(direction):
```

c) Line ~458 (flat entry):
```python
elif self.pos.is_flat and meets_threshold and self.features.is_warm() and self._side_allowed(direction):
```

### Change 4 — Argparse + session wiring
Around line 843 (after `--min-tier`):
```python
parser.add_argument("--side-filter", type=str, default="both",
                    choices=["both", "long", "short"],
                    help="Restrict entries to this side (default: both)")
```
Around line 891 (in `MambaV2PaperSession(...)` construction):
```python
side_filter=args.side_filter,
```

## VERIFICATION (DRY-RUN BEFORE LIVE)
1. Apply patch.
2. Run replay against a recent OOT session, e.g. 5/7 events:
   ```
   python paper_trading_mamba_v2_patched.py \
     --replay logs/live_events_20260507.npz \
     --side-filter short --min-tier "Top1%" \
     --weights output/cnn_mamba_v2_smart_v3_mar/fold_10_best.pt \
     --patchtst-weights output/patchtst_smart_v3_mar/fold_16_best.pt
   ```
3. Verify in log: ENTRY records all `SHORT` (no LONG entries).
4. Verify trade count > 0 (else gate is too strict).
5. If both pass → patch is safe to deploy.

## FAIL-CLOSED BEHAVIOR
Default `side_filter="both"` preserves existing behavior — no change unless explicitly invoked.

## ROLLBACK
Revert this single file to git HEAD; no schema changes, no DB writes, no side effects beyond per-session log.

## OWNERSHIP
Once approved: apply patch on Razer at `C:\Users\claude\Lvl3Quant\paper_trading_mamba_v2_patched.py` and Jupiter staging copy. Ship to git so it's durable across sessions.
