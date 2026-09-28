# Split DQN Reward Design — Audit + Redesign

**Date:** 2026-05-06
**Author:** Claude (Head of Quant)
**Trigger:** HC #221 (per-network reward engineering), HC #230 (overtrading + reward audit), HC #231 (intentionality mandate, no spread-crossing-cost), HC #219 (no raw P&L), HC #211 (training stability)

This document audits the **current** reward functions in `fifo_rl_env.py` + `train_split_dqn.py` for the three split-DQN heads (ENTRY, CANCEL, EXIT) and specifies a **redesign** with bounds, rationale, and interaction analysis. Every constant is justified — no legacy values carried over without intent.

---

## 1. Current state — as of 2026-05-06

### 1.1 ENTRY head (current)
- **Source:** `train_split_dqn.py:540-547` and `:654-660`.
- **Reward computation:** Just propagates the env step's reward. Per `:546` comment: "Entry reward: 0 during flat, P&L comes later via n-step." Means: until a position opens and closes, ENTRY's chosen-action transitions accumulate ~0 reward; the trade's eventual close P&L back-propagates through n-step (`n=50`) returns into the entry transition.
- **What it actually rewards:** Trade-close P&L (delta-Sortino) attributable through 50-step return chain to the entry decision.
- **Costs deducted:** Commission only (0.376 ticks RT), inside `pnl_ticks` at `fifo_rl_env.py:1004`.
- **Bounds:** Unbounded in either direction. Delta-Sortino has no clip.
- **Failure modes:**
  - **No price-improvement-at-fill bonus** — agent has zero incentive to prefer limit-at-bid over market-at-ask when both would work; both reward identically modulo commission.
  - **No order-type-vs-regime penalty** — agent isn't taught "thin book → don't market order" or "wide spread → must use limit."
  - **No per-entry activity cost** — early in training, picking ANY entry costs ~0 reward expectation, and the eventual trade may even pay positive. Bias toward entering everything.
  - **n-step credit assignment ambiguous** — ENTRY's contribution is mixed with EXIT's quality across the 50 steps. Bad exit pollutes good-entry signal.

### 1.2 CANCEL head (current)
- **Source:** `train_split_dqn.py:553-561` and `:668-674`.
- **Reward computation:** Inherits env step reward like ENTRY. No counterfactual.
- **What it actually rewards:** Whatever the env happens to be paying out at this step — generally near-zero unless a position closes simultaneously.
- **Failure modes — CATASTROPHIC:**
  - **Zero counterfactual signal.** Cancel net never sees "you canceled and the price moved against where your fill would have been (good cancel)" vs "you canceled and the price moved through your level (missed fill, bad cancel)."
  - The cancel head is effectively training on noise. Q-values will converge to ~mean reward of the step, learning nothing about cancel quality.

### 1.3 EXIT head (current)
- **Source:** `train_split_dqn.py:580-591` and `:692-701`.
- **Reward computation:** `total_reward = env_reward + 0.01 * (current_unrealized - prev_unrealized)`. The shaping term gives a small reward for unrealized P&L improvement step-by-step.
- **Env reward at close:** `delta_sortino` (from running Sortino tracker, `fifo_rl_env.py:1026-1039`) + `SIGNAL_ALIGNMENT_BONUS` (0.05) when entry signal aligned, − `SIGNAL_MISALIGNMENT_PENALTY` (0.08) when anti-aligned, − consecutive-loss penalty (0.05/loss after 3rd), − overhold penalty (0.01/sec over 30s, capped), − overtrading penalty (0.05/excess trade over 10/min).
- **What it actually rewards:** Risk-adjusted P&L contribution (better than raw P&L per HC #219), with weak shaping toward unrealized-favorable-direction.
- **Bounds:** Delta-Sortino unbounded. Shaping bounded.
- **Failure modes:**
  - **No MFE-capture metric** — explicitly required by HC #221 as PRIMARY signal. Exit net has no idea whether it left 80% of the move on the table.
  - **No premature-cut penalty** that scales with hold time vs MFE — agent can panic-exit at +1 tick and never learn the trade had +9 ticks of MFE remaining.
  - **No late-hold penalty** beyond the 30s overhold — by 30s the signal is dead (signal horizon ≤10s); penalty kicks in too late and with weak gradient.
  - **`MAX_HOLD_SECS = 30.0`** (line 135) is reasonable for the env, but the LIVE Razer paper trader uses `max_hold_seconds=900` (15min, line 791 of paper_trading_mamba_v2_patched.py). The training env max_hold (30s) and the live paper_trader max_hold (900s) are inconsistent — model trained on 30s holds, deployed with 900s holds. Bug.

### 1.4 Cross-cutting constants

| Constant | Value | Location | Status |
|---|---|---|---|
| `ALPHA_GATE_THRESHOLD` | 0.05 | fifo_rl_env.py:156 | **TOO LOW.** CNN-Mamba pred range is ~[-3, +3]. 0.05 = "any signal at all" = no real gate. Should align with Top5% threshold (~0.50-0.66). |
| `OVERTRADING_LIMIT` | 10/min | :144 | **TOO PERMISSIVE** + penalty (0.05/excess) is **TOO WEAK**. Agent generates 200k+ trades/epoch with this. |
| `CONSECUTIVE_LOSS_PENALTY` | 0.05 | :145 | OK — kicks in after 3rd consecutive loss. Low impact. |
| `SIGNAL_ALIGNMENT_BONUS` | 0.05 | :166 | OK — small nudge, not a hard gate. |
| `SIGNAL_MISALIGNMENT_PENALTY` | 0.08 | :167 | OK — slightly stronger than bonus. |
| `MARKET_ORDER_COST_TICKS` | 1.376 | :108 | **DELETE per HC #231(A).** Theoretical "spread crossing cost" doesn't exist. Fill price already encodes whether you crossed. |
| `LIMIT_ORDER_COST_TICKS` | 0.376 | :109 | **DELETE name; keep value as `COMMISSION_RT_TICKS` only.** Single cost = commission. |

---

## 2. Proposed redesign

### 2.1 Design principles

1. **Each head gets a reward signal that targets ITS decision quality, not the trade-close-P&L bucket.**
2. **Bounded rewards.** Use clipping or ratios to keep gradients stable (HC #219, HC #211).
3. **Costs are real, not theoretical.** Commission only. No spread-crossing model. Real measured slippage tracked separately, not as reward shaping.
4. **Activity costs.** Every action that opens or maintains a position should pay something so the no-trade baseline isn't pessimal by construction.
5. **Tier gate at the env level.** ENTRY net only sees signals that pass the live-deployment tier threshold. Otherwise: training distribution ≠ deployment distribution.

### 2.2 ENTRY head — proposed reward

```
r_entry = price_improvement_bonus          # reward for filling better than market at order time
        + edge_captured_bonus              # reward for the realized trade edge net of commission
        - per_entry_activity_cost          # small fixed cost per entry, regardless of outcome
        - signal_misalignment_penalty      # if entered against signal direction
        - regime_mismatch_penalty          # if order type wrong for liquidity regime
```

**Components:**
- `price_improvement_bonus` = `(mid_at_order − fill_price) × side_sign` measured in ticks, clipped to `[0, 1.5]`. Positive only when you fill better than mid. Limit-at-bid for a buy gets +0.5 (half-spread improvement). Market-buy at ask gets ~0.
- `edge_captured_bonus` = `(realized_pnl_ticks − commission_ticks)` propagated via n-step from trade close. Clipped `[-5, +5]` per HC #211/#219.
- `per_entry_activity_cost` = `0.10 ticks` flat, deducted at entry. Justification: forces the agent to expect at least 0.10 ticks of edge to break even. If edge < 0.10, no-op is preferred. Tunable.
- `signal_misalignment_penalty` = `-0.5 × min(|signal|, 1.0)` if `sign(direction) ≠ sign(signal)` and `|signal| > tier_threshold`. Heavy penalty for trading against alpha; HC #114 weak version is too soft.
- `regime_mismatch_penalty` = `-0.3 ticks` if market order placed when `spread > 1.5 × median_spread` (you're crossing a wide book = real cost will hurt). Tunable.

**Bounds:** approximately `[-7, +7]` ticks. Clipped explicitly to `[-5, +5]` for Q-learning stability.
**Failure modes addressed:** overtrading (per_entry_activity_cost), no-improvement-incentive, anti-signal entries, market-in-thin-book.

### 2.3 CANCEL head — proposed reward (counterfactual)

The CANCEL head requires N-event lookahead during episode replay to compute the counterfactual: "what if you had NOT canceled?"

```
For each cancel decision at time t with pending limit at price P, side S:
    Look ahead K events (K=200, ~30-60 seconds wallclock at typical event rate).
    Compute would_have_filled = (cumulative_traded_through_P[t..t+K] >= queue_position_at_t)
    If would_have_filled:
        Compute counterfactual_pnl = sign of move over [fill_ts, fill_ts + 5s_horizon] × side_sign × move_ticks
        If counterfactual_pnl > 0:  r_cancel = -counterfactual_pnl   # bad cancel — missed profit
        If counterfactual_pnl < 0:  r_cancel = -counterfactual_pnl   # good cancel — avoided loss (positive reward)
        # Both are -counterfactual_pnl; sign of pnl determines reward sign
    Else:
        r_cancel = +0.05    # small bonus for canceling a non-filling order (saved no-op queue spot)

For each NO-cancel decision at time t with pending limit, t+K close:
    Same lookahead. If filled in [t, t+K] AND filled to adverse selection: r_no_cancel = -|adverse_pnl|
    If filled to favorable: r_no_cancel = +favorable_pnl
    If still pending at t+K: r_no_cancel = -0.02 (small staleness cost)
```

**Implementation note:** This requires a "counterfactual buffer" pass after each episode — replay the episode events to compute what would have happened. Adds O(K × pending_orders) work per episode, where K ≈ 200. Acceptable cost.

**Bounds:** clipped `[-5, +5]`.
**Failure modes addressed:** zero-counterfactual problem.

### 2.4 EXIT head — proposed reward (MFE-capture primary)

```
r_exit_at_close = mfe_capture_ratio × MFE_CAPTURE_WEIGHT
               - mae_drawdown_penalty
               - premature_cut_penalty
               - late_hold_penalty
               + edge_realized_bonus

mfe_capture_ratio = clip(realized_ticks / max(mfe_ticks, 0.5), 0.0, 1.0)
                  # If MFE was 8 ticks and you exited at 6, ratio = 0.75. PRIMARY signal.

mae_drawdown_penalty = max(0, mae_ticks - mfe_ticks) × 0.1
                     # Penalty for letting MAE exceed MFE (round-tripped through profit).

premature_cut_penalty:
    If realized_ticks > 0 AND realized_ticks < 0.5 × mfe_ticks AND hold_secs < tau_premature:
        = (mfe_ticks - realized_ticks) × 0.05
    Else: 0.0
    # tau_premature = data-driven, the time at which p50 trade's MFE is still rising.

late_hold_penalty:
    If hold_secs > tau_late:
        = (hold_secs - tau_late) × 0.02
    # tau_late = data-driven, the p99 MFE-peak time. From MFE/MAE analysis ≈ 60-90s.

edge_realized_bonus = realized_ticks × 0.05
                   # Mild reward for net positive trades (commission already deducted in realized).
```

**Per-step shaping during open position:**
```
r_exit_per_step = 0.005 × (current_unrealized - prev_unrealized)   # half the current 0.01
                + 0.001 × max(0, current_unrealized - rolling_max_unrealized)   # MFE-tracking bonus
                - 0.002 × max(0, prev_max_unrealized - current_unrealized)   # MFE-giveback penalty
```

**Bounds:** terminal `[-3, +3]`, per-step `[-0.5, +0.5]`. Clipped explicitly.
**Failure modes addressed:** no MFE-capture metric (now PRIMARY), premature panic exits, late stale holds.

### 2.5 Tier gate at env level

Currently `ALPHA_GATE_THRESHOLD = 0.05` is way below tier thresholds. Required:

```python
# fifo_rl_env.py — replace constants
ALPHA_GATE_THRESHOLD = 0.50    # Top10% (calibrated from historical CNN-Mamba pred dist)
TIER_THRESHOLDS = {
    'Top10%': 0.50,
    'Top5%':  0.66,
    'Top1%':  0.94,
    'Top0.5%': 1.10,
    'Top0.1%': 1.50,
}
ACTIVE_TIER = 'Top5%'   # match Razer deployment
```

ENTRY net only gets to choose entry actions when `|signal| > TIER_THRESHOLDS[ACTIVE_TIER]`. Otherwise action space collapses to {no-op, exit-if-positioned, cancel-if-pending}. This single change eliminates the bulk of the overtrading.

### 2.6 Activity cost (per-step)

```python
PER_STEP_OPEN_COST = 0.001   # ticks per env step while position is open
                            # ~3 ticks/second at 3000 events/sec — bounds hold time naturally
```

Applied each step a position is held. Tiny per-step but accumulates: 60s hold ≈ 0.18 ticks of cost. Forces the agent to pay for time-in-market.

### 2.7 Removed: spread crossing cost

Per HC #231(A): `MARKET_ORDER_COST_TICKS = 1.376` and `LIMIT_ORDER_COST_TICKS = 0.376` are **DELETED**. Replaced with single constant `COMMISSION_RT_TICKS = 0.376`. P&L formula at `fifo_rl_env.py:1002-1006` already uses commission only — that part was correct. What gets removed is the dead-weight constants and any analyzer that subtracts a theoretical spread cost on top of fill prices.

---

## 3. Interaction analysis

| Interaction | Risk | Mitigation |
|---|---|---|
| ENTRY's `per_entry_activity_cost` + EXIT's `late_hold_penalty` | Double-penalizing trades that both happen and hold long | Acceptable: both are intended. Activity cost discourages low-edge entries, late-hold discourages stale exits. |
| CANCEL counterfactual + ENTRY tier-gate | Cancel net only sees pending orders that came from above-tier signals | Good: cancel is trained on the same distribution it'll see in deployment. |
| EXIT `mfe_capture_ratio` + ENTRY's `edge_captured` (n-step) | Exit good MFE-capture amplifies entry's edge_captured | Good: this is the alignment we want. Bad exit punishes both heads, good exit rewards both. |
| Per-step open cost (0.001) + EXIT shaping (0.005·Δunrealized) | Per-step costs > shaping for sideways movement | Intentional: sitting in a sideways position bleeds the agent toward exit. |
| Tier gate (0.50) + signal_alignment penalty | Misalignment penalty only applies when above-tier signal exists | Good: no penalty for entering on weak signals (those are filtered out). |

---

## 4. Implementation order

1. Patch `fifo_rl_env.py`:
   - Remove `MARKET_ORDER_COST_TICKS`, `LIMIT_ORDER_COST_TICKS`. Keep `COMMISSION_COST = COMMISSION_RT_TICKS = 0.376`.
   - Raise `ALPHA_GATE_THRESHOLD` to 0.50; add `TIER_THRESHOLDS` dict + `ACTIVE_TIER`.
   - Add `PER_STEP_OPEN_COST = 0.001`.
   - Rewrite `_force_close_position` reward (line 1026+) to use MFE-capture as primary, MAE-drawdown penalty, premature/late penalties.
   - Add `price_improvement_bonus` computation in `_apply_action` for entry actions.
   - Add `per_entry_activity_cost` deduction at entry.
   - Tighten signal_alignment penalties (×10 from current).
   - Add per-step open cost in `step()`.
   - Hook a counterfactual buffer for cancel decisions (post-episode pass).
2. Patch `train_split_dqn.py`:
   - Use Huber loss (already? verify) per HC #211.
   - Add reward clipping `[-5, +5]` before storing in replay buffer.
   - Reduce `n_step` from 50 to 20 (10s of events at typical rate) — current 50 is too long for 1-10s signal horizons.
   - Per-head GPU update logging + Q-loss surfacing.
3. Kill polluted Neptune run cleanly. Relaunch from CLEAN state (NOT warm-start).
4. First-epoch sanity: trade count must drop ≥10× (target: thousands per epoch, not 200k).
5. If sanity fails → re-audit. If passes → continue.

---

## 5. Open questions / requires user input

- **`per_entry_activity_cost = 0.10 ticks`** — tunable. Could be 0.05 or 0.20. Justification: Top5% tier signals should have edge ≥ 0.5 ticks easily; 0.10 is a small "must beat trivial" cost. Open to user feedback.
- **Activity cost per-step `0.001`** — could be 0.0005 or 0.002. Same logic.
- **`ACTIVE_TIER = 'Top5%'`** — should match Razer deployment. If user wants Top1% or Top0.5% in deployment, change here too.
- **Reward clip range `[-5, +5]`** — could be `[-10, +10]` for fewer cap events. 5 chosen because 99% of single-trade P&L is < 5 ticks per training data.

These are all tunable; values above are defensible defaults, not sacred.
