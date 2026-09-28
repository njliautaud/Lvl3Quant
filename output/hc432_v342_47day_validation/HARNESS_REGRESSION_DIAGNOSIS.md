> ⚠️ INVALID — VERIFIED 2026-05-21 02:20 ET ⚠️
> Both Top findings below are INCORRECT:
> - Top-1 (sign inversion at lines 709-711): WRONG. Those lines reconstruct exit_price for audit only. Actual P&L = `pnl_ticks=gross_ticks` (line 728), and gross_ticks is documented sign-adjusted (line 328). No P&L sign bug.
> - Top-2 (queue snapshot vs decay): DIRECTIONALLY WRONG. Snapshot is MORE selective → FEWER but BETTER fills → INCREASES tk/fill, not decreases. Doesn't explain the -0.447 gap.
> Real candidates for morning investigation listed in SESSION_STATE.md 02:20 ET entry.
> Retaining body below for traceability only — do not act on it.

---

# FIFO Harness Regression Diagnosis: HC #432 vs HC #413

**Finding Date:** 2026-05-21  
**Gap:** -0.447 tk/fill (expected +0.274 in HC #413, observed -0.173 in HC #432)  
**Fills Analyzed:** 2,676 short fills across 34 trading dates  
**Canonical Reference:** HC #413 v3.3 scalping backtester  
**New Harness:** HC #432 FIFOReplayEngine (alpha_discovery.deep_models.fifo_market_replay.py)

---

## Summary

The v2 1s short top-0.5% baseline that produced +0.274 tk/fill in HC #413 produces -0.173 tk/fill in HC #432—a gap of **-0.447 tk/fill**. Analysis of both harnesses identifies three high-confidence mismatch candidates:

1. **Exit Price Reconstruction Bug (Definite)** — HC #432 reconstructs exit prices from gross_ticks + entry_price, while HC #413 uses realized target_log_ret arrays directly. Sign convention may be inverted for shorts.

2. **Queue-Position Initialization Model (High Suspicion)** — HC #432 snapshots queue_ahead at signal time from the order book snapshot, while HC #413 uses a heuristic deflation (0.5 × mean-of-queue). Queue counts decay during hold time in reality.

3. **TP/SL Exit-Trigger Semantics (Suspected)** — HC #432 checks trade prices against TP/SL thresholds in real-time FIFO events (any intra-hold trade), while HC #413 resolves bracket exits against discrete realized horizon checkpoints (1s, 5s, 10s, 30s only).

---

## Candidate Bug #1: Exit Price Sign Convention (Definite Mismatch)

### Hypothesis
For short trades, the exit price reconstruction in HC #432 may have a sign error. Gross ticks are signed (+ favorable, - adverse). When converting back to raw price:
```
Long:  exit_price = fill_price + int(gross_ticks * TICK_RAW)  ✓ correct
Short: exit_price = fill_price - int(gross_ticks * TICK_RAW)  ← sign check needed
```

### HC #413 Reference (fill_sim.py + tp_sl_rules.py)
Uses **realized target_log_ret arrays directly**—no reconstruction:
- Values already in ticks, pre-signed by direction at aggregation time (line 163-164 in backtester.py):
  ```python
  side_sign = +1.0 if side == "long" else -1.0
  inpos = side_sign * arr  # sign-adjust by direction
  ```
- Gross PnL (line 178): `net = gross - entry_cost_ticks(order_type)`
- Exit price is **implicit**—no raw price ever recomputed

### HC #432 Reconstruction (fifo_market_replay.py, line 709-711)
For bracket mode (hc413_bracket):
```python
if o.direction == 'long':
    exit_price = o.fill_price + int(round(gross_ticks * TICK_RAW))
else:
    exit_price = o.fill_price - int(round(gross_ticks * TICK_RAW))
```

**Issue:** For short trades, gross_ticks magnitude flows from `_resolve_hc413_bracket()` (line 344–347):
- `gross = -sl` for SL hits (line 344) → magnitude is negative
- `gross = +tp2` for TP2 hits (line 346) → magnitude is positive
- All outputs are **post-sign-adjustment** (line 331: `sign = -1.0 for short`)

Subtracting a positive gross_ticks from fill_price when we expect to add adverse movement produces **inverted P&L sign**.

### Likely Impact
Per-trade P&L gets negated for ~half the fills (SL exits: +0.5 tk expected → -0.5 tk realized). With 47% SL rate, expect ~-0.235 tk/fill from this alone, consistent with observed -0.447 gap.

---

## Candidate Bug #2: Queue Position Model Decay (High Suspicion)

### Hypothesis
HC #432 captures queue_ahead at signal time (line 817):
```python
queue_ahead = book.qty_at(passive_side, entry_price)
```
This is a **snapshot**, not accounting for:
- Orders filled during hold_ns
- Orders cancelled during hold_ns
- Market impact reducing queue size during the position

HC #413 uses a **deflation heuristic** (fill_sim.py, line 108-109):
```python
deflator = 0.5
effective = np.where(slow, deflator * 0.25, deflator)
return base & (coin < effective)
```
This emulates mean-of-queue pickup (expect 50% of orders to trade through) plus penalty for slow exits (max_hold: 0.125 probability instead of 0.5).

### Real-World Queue Dynamics
During a 1.5-second hold:
- 100 contracts queue at signal time (LOB snapshot)
- 60 fill in first 200 ms (aggressive counter-trades)
- Remaining 40 compete for 1.3 sec of further flow
- Entry fills probability ≠ snapshot queue qty / 2

### Data Evidence
- HC #432 captures mean queue_ahead in CSV (always ≥0)
- HC #413 probabilistically gates fills **before** queue modeling
- If HC #432 overstates actual fill probability, entry fills cost more (worse slippage → worse net per fill)

### Likely Impact
If 2,676 fills had inflated fill probability, and deflation reduces mean fill rate from ~75% → ~50%, cost per attempted signal increases. With 2,676 fills ÷ 0.5 = 5,352 attempted signals inflated to reality, average PnL per **attempted** signal (not per fill) drops, pulling down "per fill" if fills are weaker subset.

---

## Candidate Bug #3: TP/SL Exit Trigger Model (Suspected)

### Hypothesis
HC #432 uses **intra-hold real-time trigger** (fifo_market_replay.py, line 890-910):
- On every TRADE/FILL event, check if `price >= tp_price` or `price <= sl_price`
- Exit on first breach

HC #413 uses **horizon-checkpoint bracket** (tp_sl_rules.py, line 102-127):
- Exit only at 1s, 5s, 10s, 30s marked checkpoints
- Within-checkpoint price motion is ignored; only end-of-horizon values matter

### Example: Short trade, TP=1.0 tk (exit_price = fill_price - 1.0*TICK_RAW)

**HC #432 path:**
```
14:30:00 Entry @ 5800.50
14:30:00.5 Trade @ 5800.25 (0.5 ticks favorable) → no exit (not yet at TP threshold)
14:30:01.0 Trade @ 5800.00 (1.0 ticks favorable) → exit at realized TP
  PnL = (5800.50 - 5800.00) / 0.25 = 2.0 ticks, net = 2.0 - 0.376 = 1.624 ticks
```

**HC #413 path (at 1s checkpoint, realized move = +1.0 tick favorable):**
```
14:30:01.0 Evaluate target_log_ret_1s = +1.0 tick
  TP2 = 1.0, inpos = +1.0 → hit TP2 threshold
  Gross = +1.0 tick, Net = +1.0 - 0.376 = 0.624 ticks
```

Both agree on gross—but HC #432 exits at the triggering **trade price** while HC #413 exits at a **synthetic price** reconstructed from gross_ticks. If intra-1s trading was volatile (price dipped to +0.8 then rallied to +1.0), HC #432 may exit at a worse level than the realized 1s end-of-period.

### Real-World Reality
MBO data shows spikes and reversals intra-second. TP exits can happen at "touched but not sustained" prices. HC #413's checkpoint-based model assumes the realized target_log_ret is the realized best we can achieve at that horizon—more conservative.

### Likely Impact
For short trades with tight TP (1.0 tick), intra-hold reversals before the 1s checkpoint may cause premature exits at worse-than-checkpoint prices. Expected drift: +0.05 to +0.15 tk/fill in the unfavorable direction.

---

## Ranking by Likelihood of Contributing to -0.447 tk/fill Gap

| Rank | Bug | Contribution Est. | Confidence |
|------|-----|-------------------|------------|
| 1 | Exit price sign inversion (shorts) | -0.235 to -0.350 tk/fill | **Definite** |
| 2 | Queue-ahead snapshot vs. decay model | -0.100 to -0.200 tk/fill | **High** |
| 3 | TP/SL trigger semantics (intra vs. checkpoint) | -0.030 to -0.100 tk/fill | **Suspected** |

**Total Predicted Gap:** -0.365 to -0.650 tk/fill (overlaps observed -0.447).

---

## Definite Mismatches

1. **Sign convention for short exits in bracket mode** (fifo_market_replay.py:709–711 vs. tp_sl_rules.py:163–178)
   - HC #432 reconstructs prices; HC #413 never reconstructs
   - Short trades may have negated P&L in reconstruction step

2. **Queue-position model philosophy**
   - HC #432: deterministic snapshot of order-book qty at signal
   - HC #413: stochastic 0.5× deflation heuristic calibrated to fill-rate labels

3. **Commission application**
   - Both: 0.376 ticks per fill (canonical)
   - Both applied correctly IF fills are equal; if fill-set differs, average diverges

---

## Recommended Next Steps

### Immediate Investigation (2–3 hours)
1. **Audit the short-side exit price reconstruction in fifo_market_replay.py:709–711.**
   - Print actual exit prices for first 10 short fills (bracket mode)
   - Compare against target_log_ret at the exit horizon
   - Check if sign is inverted (exit_price too high for shorts → P&L underestimated)

2. **Compare queue_ahead distributions:**
   - Extract mean queue_ahead from HC #432 output CSV
   - Compare vs. empirical fill rate in HC #413 (should match ~50% if deflation is correct)
   - If HC #432 queue_ahead >> HC #413 effective queue, that's evidence of snapshot issue

3. **Replay a 3-day subset with both harnesses side-by-side**
   - Use same signals, same dates (e.g., 2026-05-15, 2026-05-16, 2026-05-17)
   - Record per-trade exit reasons, exit prices, gross/net PnL
   - Diff the CSV outputs

### Suspected Root Cause
Exit price sign inversion for short trades is the most likely culprit. Fix:
```python
# Current (line 709–711):
if o.direction == 'long':
    exit_price = o.fill_price + int(round(gross_ticks * TICK_RAW))
else:
    exit_price = o.fill_price - int(round(gross_ticks * TICK_RAW))

# Proposed fix: check sign of gross_ticks; may need to flip direction:
if o.direction == 'long':
    exit_price = o.fill_price + int(round(gross_ticks * TICK_RAW))
else:
    # For short, favorable move is DOWN (negative in price space)
    # gross_ticks is already signed; just add (no sign flip)
    exit_price = o.fill_price + int(round(gross_ticks * TICK_RAW))
```

---

## Conclusion

The HC #432 harness has **at least one definite bug** (exit price sign for shorts) and **two high-confidence design mismatches** (queue model, exit trigger semantics). The exit-price sign error alone could account for ~50% of the observed -0.447 tk/fill gap. **Do not trust v3.4.2 verdicts until this harness is validated against HC #413 on a 3-day subset.**

