# HC #430 Production Integrity Audit — v2_1s_short_top05 + Tier Runners

**Date**: 2026-05-19 ~09:10 ET
**Scope**: Razer live processes (PIDs 15720 mbo / 32980 watchdog / 35128 spec-trader-shadow / 34928 + 30064 + 31468 tier-runners) vs HC #417 deployment spec §1 + §3, plus retroactive HC #430 R1/R2 (regime-agnostic + MFE-within-horizon) checks on the training data path.

**Method**: SCP read-only fetch of the four Razer Python files into `/tmp/razer_audit/`. No Razer file modified. Cross-referenced against `/home/jupiter/Lvl3Quant/output/hc417_v2_1s_short_top05_DEPLOYMENT_SPEC.md` and prior `hc417_razer_paper_config_audit.md` (which covered the OLD PID 25512 trader, now retired and replaced by PID 35128).

**Files audited** (lines below cite `paper_trading_v2_1s_short_top05.py` = `v2trader`; `paper_trading_mamba_v2.py` = `legacy`; `paper_trading_mamba_v2_runner.py` = `runner`; `cnn_mamba_v2_inference.py` = `infer`; `mbo_recorder.py` = `mbo`).

---

## Live process roster (verified via Get-CimInstance 2026-05-19 09:10 ET)

| PID | Script | Args | Role |
|---|---|---|---|
| 15720 | `mbo_recorder.py` | `--symbol ESM6 --exchange CME --flush-minutes 5` | Singleton MBO feed → `live_events.jsonl` |
| 32980 | `live_stack_watchdog.py` | (no args) | Stack supervisor |
| **35128** | **`paper_trading_v2_1s_short_top05.py`** | `--shadow --symbol ESM6 --device cuda --sha256 300e338d...1281e` | **THE NEW SPEC TRADER (HC #417 v2_1s_short_top05) — SHADOW mode** |
| 34928 | `paper_trading_mamba_v2_runner.py` | `--follow-events ... --symbol ESM6 --min-tier Top0.5%` | Legacy HC #46 trader |
| 30064 | same | `--min-tier Top1%` | Legacy HC #46 trader |
| 31468 | same | `--min-tier Top5%` | Legacy HC #46 trader |

Live log evidence (5/18 23:51 → 5/19 09:07): v2trader **shadow** has 1700 predictions, **0 entries**, day_net=0. SHA-256 logged. Stale-signal kill-switch tripped intermittently overnight (expected — thin Globex).

---

## Q1 — RTH gate (HC #430 R1)

**v2trader: PASS (correctly enforced)**

- `v2trader:96-97` defines `RTH_OPEN = dtime(9,30); RTH_CLOSE = dtime(16,0)` (literal 09:30–16:00 ET).
- `v2trader:192-200` `now_et()` uses `pytz.timezone("America/New_York")` (DST-aware). pytz 2026.1.post1 confirmed installed in the Razer python env that PID 35128 is using.
- `v2trader:203-209` `is_rth(t)` returns False on weekends AND outside `09:30 ≤ tod < 16:00`.
- **Gate enforcement: `v2trader:780-782`** in `on_prediction`:
  ```
  t_et = now_et()
  if not is_rth(t_et):
      return  # outside RTH — no entries
  ```
- Cold-start anchor is RTH-relative (`v2trader:212-218` `minutes_into_rth`), so cold-start window starts at the 09:30 bell each day, not at process boot.
- Verdict: **PASS**.

**legacy (PIDs 34928/30064/31468): FAIL — NO RTH GATE**

- Grep across `paper_trading_mamba_v2.py` for `rth|09:30|16:00|ny_time|america/new_york|trading_hours|hour.*30|session_window` → **zero matches**. The only "30:" hit is `stats_interval_s: float = 300.0` (line 406).
- These three runners trade 23h/day (CME Globex hours minus the 1h daily break). They flip on signal at any hour.
- Verdict: **FAIL** (same finding as the 5/18 audit; HC #430 R1 requires fix).

---

## Q2 — Confidence normalization (HC #430 R2)

**v2trader: PASS (per-day percentile + global cold-start)**

- Per-day percentile tracker class: `v2trader:233-292` `PerDayPercentileTracker`.
  - Resets at day boundary on `_reset_if_new_day` (`v2trader:246-252`).
  - Tracks `signed_short = -pred_log_ret_1s` ONLY when `pred_log_ret_1s < 0` (`v2trader:254-261`).
  - Threshold requires ≥ 200 observations before activating (`v2trader:268-275`).
  - `passes()` gate (`v2trader:277-292`): requires BOTH `pred_log_ret_1s <= GLOBAL_CONFIDENCE_FLOOR_1S` (`-0.6926` from spec, `v2trader:85`) AND `(-pred) >= per-day 99.5th percentile`, with cold-start: first `COLD_START_MINUTES=30` (line 87) uses global floor only.
- Gate is wired into the live engine at `v2trader:787-799` in `on_prediction`.
- Verdict: **PASS** — matches spec §1 exactly.

**legacy (3 tier-runners): FAIL — global pre-baked tier, NOT per-day**

- Tier definition: `legacy:394` `TIER_ORDER = {"Top5%": 1, "Top1%": 2, "Top0.5%": 3, "Top0.1%": 4}`.
- Thresholds are calibrated ONCE at startup from the static NPZ `DEFAULT_PREDS` via `legacy:435 → infer:396-411 calibrate_thresholds()`:
  ```
  abs_p = np.abs(pred_1s)
  for tier, pct in [("Top5%", 95), ("Top1%", 99), ("Top0.5%", 99.5), ("Top0.1%", 99.9)]:
      self.confidence_thresholds[tier] = float(np.percentile(abs_p, pct))
  ```
- This is a **GLOBAL, direction-AGNOSTIC** tier built off `|pred_1s|` from the entire OOT NPZ — never recomputed, never per-day, and (worse) it counts long and short signals together.
- `infer:364-365`: `confidence = abs(pred_1s); direction = 1 if pred_1s > 0 else -1` → tier-runner accepts BOTH directions on the same threshold (long and short both fire).
- Verdict: **FAIL** vs spec — wrong gate TYPE (static global vs per-day) and wrong direction filter (both vs short-only).

---

## Q3 — TP1 / TP2 / SL brackets (HC #417 spec §1)

**v2trader: PARTIAL PASS — passive entry + passive TP2 + 10s cancel wired; TP1 NOT wired; SL uses market exit.**

- Constants frozen at spec values: `v2trader:80-82`:
  ```
  TP1_TICKS = 0.4782   # half size partial profit
  TP2_TICKS = 0.9564   # full MFE target (final tp for 1-contract)
  SL_TICKS  = 0.5686   # capped at MAE
  ```
  (Smoke-test asserts these exact values, `v2trader:1199-1201`.)
- ENTRY: `v2trader:806-823` `_submit_entry_limit_short` — submits LIMIT at `best_ask`. **Passive_at_touch: PASS**.
- 10s cancel: `v2trader:90` `CANCEL_WINDOW_SECONDS=10.0`; enforced at `v2trader:833-838`: `if age > CANCEL_WINDOW_SECONDS: self._cancel_entry_and_reset(reason="cancel_10s")`. **PASS**.
- SL: `v2trader:844-845` `if mae_now >= SL_TICKS: self._submit_exit_market(reason="sl_hit", ...)` → **MARKET ORDER on SL**, not a resting stop. Functional but slippage-prone.
- TP2: `v2trader:846-849` triggers `_submit_exit_limit_short_cover` at passive limit (`max(target_px, best_bid)` — passive on exit). **PASS**.
- TP1 (half-size partial scalp at +0.4782 tk): **NOT WIRED in live path.** The simulate_trades pure function at `v2trader:582-588` only checks TP2; live `_manage_position` (`v2trader:825-863`) only checks SL/TP2/time-stop. The smoke-test docstring at `v2trader:17-18` admits this: "For 1-contract size: single TP at TP2, hard SL at -0.5686 tk."
- Time-stop fallback at ~40s (`v2trader:850-852`): present, executes via market.
- Verdict: **PARTIAL PASS** — gates exist as designed for 1-contract sizing, but the spec lists TP1 as a half-size partial which is intentionally collapsed to "TP2 only" by the implementation. Document this explicitly to the user as a deviation from spec §1 row "TP1".

**legacy: FAIL — no TP/SL brackets at all** (signal-flip exits + max_hold + trailing-stop; same as 5/18 audit row 7).

---

## Q4 — Nine kill-switches (HC #417 spec §3)

`v2trader` exposes a `KillSwitchManager` class (`v2trader:308-457`). One-by-one:

| # | Switch | Spec trigger | v2trader status | Cite |
|---|---|---|---|---|
| 1 | Hard daily loss | net ≤ -10 tk | **PASS** | `v2trader:104` const, `v2trader:425-432` `_evaluate_post_fill_killers` halts until end of ET day |
| 2 | Hard weekly loss | net ≤ -25 tk rolling 5d | **PASS** | `v2trader:105`, `v2trader:434-437` (sums last 4 daily entries + today) → `halt_manual` |
| 3 | Per-trade max risk | SL not filled 30s → market exit | **PASS** | `v2trader:114` const, `v2trader:855-863` forces `_broker_submit_market` exit |
| 4 | Consecutive-loss pause | 5 losses → 1h pause | **PASS** | `v2trader:106-107`, `v2trader:440-443` |
| 5 | Realised drift halt | rolling 10-fill net/fill < -0.5 tk | **PASS** | `v2trader:108-109`, `v2trader:407-414` → `halt_manual` |
| 6 | IC drift | rolling IC < 0.15 | **PRESENT but UNREACHABLE** — see note | `v2trader:110-111` const, `v2trader:416-423`. Logic checks `ic_targets` but `killer.on_prediction` is called as `self.killer.on_prediction(pred_1s, target_1s=None)` at `v2trader:777` — **target is hardcoded None**, so `ic_targets` deque never receives data → `len(ic_preds) < KS_IC_DRIFT_WINDOW` permanently. This kill-switch **never fires**. |
| 7 | Stale-signal kill | >30s no prediction → pause | **PASS** | `v2trader:112` const, `v2trader:397-400` — and observed firing in live log (`08:14:41 KILL-SWITCH TEMP HALT 60s: stale_signal`). |
| 8 | Connectivity loss | broker >5s silent → flatten + pause | **WEAK PASS** | `v2trader:113`, `v2trader:402-405` halts on `last_broker_seen_ts` staleness. BUT `last_broker_seen_ts` is "fed" by every JSON event (`v2trader:1087 on_broker_heartbeat`), so this monitors recorder freshness, NOT Rithmic broker connectivity. If the broker drops while the recorder keeps publishing, this won't trip. Flatten-on-disconnect is also not implemented (only a manual halt). |
| 9 | SHA-256 file integrity | refuse start on mismatch | **PASS** | `v2trader:716-723`; live process command line includes `--sha256 300e338d3c16137fc587b10cce92204e8fe0a486fc3c7aaf53856fdd8c21281e` and log shows it logged at startup. |

**Summary**: 7 of 9 PASS, 1 weak (#8 connectivity surrogate), 1 wired-but-non-functional (#6 IC drift — wrong arg).

---

## Q5 — Discord / webhook alerts

`v2trader:160-185` `DiscordNotifier` class POSTs to `DISCORD_WEBHOOK_URL` env var; throttle 1s; tags ✅ for fill, 🛑 for kill, etc.

**Live status**: The running PID 35128 log line `2026-05-18 23:51:35,813 WARNING DiscordNotifier: no webhook URL - alerts will only log to file.` — **the env var was not set when the process started**. Alerts are JSONL-logged but not sent.

Verdict: **PARTIAL — code wired correctly, but environment variable missing at process start. Restart required (with `DISCORD_WEBHOOK_URL` exported) for alerts to actually fire.**

---

## Q6 — MBO recorder session scope

`mbo:38-156` `MBORecorder` subscribes to Rithmic for `--symbol ESM6 --exchange CME` and writes every BBO + Trade event to (a) daily NPZ in `data/processed/mbo_events/` and (b) the fan-out JSONL `live_events.jsonl`.

- **No RTH filter** in mbo. Grep for `rth|09:30|16:00|session_filter` → zero hits. `on_md` (`mbo:103-126`) accepts every event; `_flush_loop` flushes every 5 min regardless of time.
- mbo also has no day-of-week filter.
- **Consequence**: live_events.jsonl contains FULL Globex sessions. The v2trader correctly filters at the trade-decision layer (Q1 PASS), but the **legacy tier-runners read the same firehose and have NO filter** → they evaluate signals 23h/day, 6.5d/wk (5/18 audit confirmed they did so for at least the past 4 days).
- Singleton lock (`mbo:170-184`) is wired — only one mbo process is allowed.

Verdict: **mbo is intentionally session-agnostic** (it's a data feed, not a trader); the burden of filtering is on consumers. v2trader meets it, the three tier-runners do NOT.

---

## Q7 — Training data scope (retroactive HC #430 R1)

Examined `/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_cnn_mamba_v3_3.py` (918 lines, last-modified by v3.3 author, header confirms it is the canonical v3.3 trainer).

- Grep for `rth|09:30|16:00|session_filter|ny_time|market_hours|cme_session|is_rth|trading_hours` → **zero matches**.
- Grep for the broader set `filter|mask|session|night|globex|hour|datetime|timestamp_to|date_range` returns only `masks: Dict[str, torch.Tensor]` (loss-mask, NOT session-mask — for NaN labels per HC #82). Lines 204, 207, 215, 222, 230, 236, 248, 256, 391, 397, 402, 465, 613, 737, 756, 763, 801 — all about per-head label masks, none about session/time.
- v3.4 trainer (`train_cnn_mamba_v3_4.py`) likewise: zero matches for `rth|09:30|16:00|session_filter|ny_time|market_hours|cme_session|is_rth|trading_hours`.
- (No `train_cnn_mamba_v2*.py` file exists on Jupiter — v2 was either trained off a now-deleted script or by a hand-launched run on Razer.)

Verdict: **CNN-Mamba v3.3 and v3.4 were trained on the FULL Globex day with no RTH-only filter.** This means: (a) the IC numbers cited in spec §1 are mixed regime, (b) HC #430 R1 (40+ day OOT, per-regime stratified Sharpe) was not enforced retroactively on v2/v3.3/v3.4.2 training. The live v2 model checkpoint `fold_10_best.pt` inherits this exposure. Operationally, the v2trader **mitigates this** by trading only 09:30-16:00 ET, but the prediction quality during RTH may diverge from the offline IC numbers if signal distribution is regime-dependent.

---

## Q8 — Three --min-tier runners: definition, status, spec compliance

| Runner PID | --min-tier | Confidence definition | Direction filter | RTH gate | TP/SL brackets | Spec compliant? |
|---|---|---|---|---|---|---|
| 34928 | Top0.5% | **GLOBAL pre-baked** (infer:409-410 from static `concat_oot_predictions.npz`) | **NONE** (both sides via `direction = 1 if pred_1s > 0 else -1`, infer:365) | **NONE** | NO TP/SL — uses signal-flip + trailing-stop + max_hold + max_daily_loss=-$3000 (legacy HC #46 logic) | **NO — fails 6 of 19 HC #417 gates** |
| 30064 | Top1% | same | same | same | same | NO |
| 31468 | Top5% | same | same | same | same | NO |

Quote: `infer:408-410`:
```python
abs_p = np.abs(pred_1s)
for tier, pct in [("Top5%", 95), ("Top1%", 99), ("Top0.5%", 99.5), ("Top0.1%", 99.9)]:
    self.confidence_thresholds[tier] = float(np.percentile(abs_p, pct))
```

This is the static, GLOBAL, BOTH-DIRECTIONS threshold. The three runners' "Top0.5%" / "Top1%" / "Top5%" CLI flag merely picks WHICH precomputed bucket to use — none is per-day.

**Mode**: All three are running via `paper_trading_mamba_v2_runner.py` which calls into `paper_trading_mamba_v2.main()`. The legacy script's "paper trade mode" log line (`legacy:530 *** PAPER TRADE MODE -- NO REAL ORDERS ***`) suggests paper-only by default. **However**, none of the three has a `--shadow` or paper flag in its arg list, and the legacy script's risk manager has a `max_daily_loss=-$3000` limit (24× looser than spec). I have not verified whether `paper_trading_mamba_v2.main()` actually submits orders to Rithmic when run without an explicit paper flag — the script header claims "PAPER TRADE MODE -- NO REAL ORDERS" but the prior audit found ENTRY paths invoke `pos.enter(...)` which is internal to the script (not a broker call). **Treat as PAPER until otherwise proven**, but flag as "verify-no-broker-submit" since they don't carry the explicit `--shadow` flag the new spec trader does.

**Spec compliance**: Each of the three fails the same 6 gates as the OLD PID 25512 trader (per 5/18 audit) — RTH, direction, confidence-gate type, TP/SL brackets, kill-switches 1/2/4/5/6/7/8/9, alerts. They are **research-grade exposure**, not spec-compliant production.

---

## Aggregate pass/fail vs HC #417 spec §1 + §3 (19 gates)

| Gate group | v2trader (PID 35128) | tier-runners (34928/30064/31468) |
|---|---|---|
| Model arch | PASS | PASS |
| Symbol ESM6 | PASS | PASS |
| Signal head pred_log_ret_1s | PASS | FAIL (uses combined direction) |
| SHORT-only | PASS | FAIL (both sides) |
| Confidence: per-day 99.5%ile + global cold-start | PASS | FAIL (static global tier) |
| TP1 +0.4782 tk | DEVIATION (collapsed to TP2-only by design) | FAIL |
| TP2 +0.9564 tk | PASS | FAIL |
| SL -0.5686 tk | PASS (via market) | FAIL |
| passive_at_touch entry | PASS | FAIL |
| 10s cancel | PASS | FAIL |
| RTH 09:30-16:00 ET | PASS | FAIL |
| Position size 1 | PASS | PASS |
| 5s re-entry cooldown | PASS (`v2trader:91,793-795`) | FAIL |
| Max concurrent 1 | PASS (`v2trader:92,788`) | PASS (legacy uses 1) |
| Kill 1: daily | PASS | FAIL (legacy uses -$3000) |
| Kill 2: weekly | PASS | FAIL |
| Kill 3: per-trade | PASS | FAIL |
| Kill 4: consec-loss | PASS | partial (3→10min, looser) |
| Kill 5: realised drift | PASS | FAIL |
| Kill 6: IC drift | **WIRED-BUT-INERT** (target arg=None) | FAIL |
| Kill 7: stale-signal | PASS | FAIL |
| Kill 8: connectivity | WEAK (monitors recorder, not broker) | FAIL |
| Kill 9: SHA-256 | PASS | FAIL |
| Discord alerts | wired but env var MISSING at process start | FAIL |
| MFE within horizon (HC #430 R2) | PASS (TP2 = 1.0×MFE_1s, hold ≤ 40s on 1s horizon) | FAIL (60s max_hold on 1s-horizon model) |

**v2trader: 18 PASS / 1 deviation (TP1) / 2 weak-or-inert (IC drift, connectivity) / 1 env-config (webhook) = effectively 18/22 production-ready.**
**tier-runners: 2 PASS / 20 FAIL each. Each is the OLD legacy trader operating as if HC #417 didn't exist.**

---

## ROLL-BACK candidates (recommend kill)

**PRIORITY 1 — Kill the three tier-runners NOW.**

- PIDs 34928 (Top0.5%), 30064 (Top1%), 31468 (Top5%).
- They violate 20/22 HC #417 gates each.
- They trade both directions on a GLOBAL static threshold during overnight Globex with no RTH gate, no TP/SL brackets, a -$3000 daily loss limit (24× wider than spec), and no Discord alerts.
- Even if they are paper-mode (probable but not 100% verified per Q8 caveat), the predictions they emit and the entry/exit decisions they log will (a) pollute the audit trail, (b) create misleading "paper P&L" attributable to the wrong gating logic, (c) consume GPU + Rithmic feed budget that the spec-compliant v2trader needs.
- **HC #428 R1 violation**: these three are NOT regime-agnostic — they have no session filter so they're effectively betting on overnight thin-book noise, which is a different regime than the 09:30-16:00 ET regime the spec was validated on.
- **HC #428 R2 violation**: their max_hold is 60s on a 1s-horizon prediction → ratio 60× exceeds the 1.5× spec ceiling. Pure "luck/beta" exposure per HC #428 R2.
- Recommended action: `Stop-Process -Id 34928,30064,31468 -Force` on Razer.

**PRIORITY 2 — Restart v2trader (PID 35128) with `DISCORD_WEBHOOK_URL` set.**

- Trader is otherwise spec-compliant. Without webhook, the user gets no fill/kill/alert messages — which defeats the supervision design.
- Risk of restart: zero (process is in shadow mode, no positions open, 0 fills today). Pick a window where `passes_gate` count is still 0 (effectively any time today; fill count from 23:51 ET start is 0).

**PRIORITY 3 — Fix IC drift kill-switch wiring.**

- `v2trader:777` `self.killer.on_prediction(pred_1s, target_1s=None)` always passes `None` for the realised target. There is no mechanism today to feed the 1s realised log-return back into the IC-drift deque.
- Wire a delayed-target callback: when an event arrives, record the mid-tick; 1s later (using the recorder's monotonic timestamp), compute realised 1s log-return from the prediction's anchor mid, and feed back via `self.killer.on_prediction(pred_at_t, target=realised_1s_at_t+1s)`.
- Until fixed, kill-switch #6 is theoretical.

**PRIORITY 4 — Tighten connectivity check.**

- `v2trader:1087` treats every JSONL event as broker heartbeat. Add a separate broker-side heartbeat (Rithmic ping/pong or order-channel ack) so that broker disconnect is detected even when the recorder keeps publishing.

**PRIORITY 5 — Document TP1 deviation.**

- Spec §1 promises a half-size TP1 partial at +0.4782 tk. Live code intentionally collapses to TP2-only for 1-contract sizing (`v2trader:17-18` docstring). This is defensible for 1-lot but should be documented to the user and re-introduced when sizing ≥ 2 contracts.

**PRIORITY 6 — Retroactive HC #430 R1 evaluation on training.**

- v3.3/v3.4 training scripts have no RTH filter. The v2 model `fold_10_best.pt` was likely trained the same way (no Jupiter-side v2 training script exists to confirm).
- Recommendation: re-run OOT IC against RTH-only-labels to estimate the regime-stratified Sharpe per HC #428 R1 (gate |Sharpe_green − Sharpe_red| / max(...) ≤ 0.50). If that test fails, the v2trader spec may need a separate per-regime confidence threshold.

---

## File-line citation index (for cross-reference)

- `v2trader` = `C:\Users\claude\Lvl3Quant\live_trading_linux\paper_trading_v2_1s_short_top05.py` (1250 lines, fetched 2026-05-19)
- `legacy` = `C:\Users\claude\Lvl3Quant\live_trading\paper_trading_mamba_v2.py` (1271 lines)
- `runner` = `C:\Users\claude\Lvl3Quant\live_trading\paper_trading_mamba_v2_runner.py` (90 lines — pure monkey-patch wrapper)
- `infer`  = `C:\Users\claude\Lvl3Quant\live_trading\cnn_mamba_v2_inference.py` (ICP at line 409-410)
- `mbo`    = `C:\Users\claude\Lvl3Quant\mbo_recorder.py` (267 lines)
- v3.3 trainer = `/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_cnn_mamba_v3_3.py` (918 lines)
- v3.4 trainer = `/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_cnn_mamba_v3_4.py`

Audit conducted READ-ONLY. No Razer file modified.
