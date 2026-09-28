# HC #423 §2 — v3.4.2 `book_gate` Collapse RCA

**Run** `e5f0f79b313d4ac4aa461df8b7af2385` · exp `CNNMamba_v3_4_2_fixed_mtl` · model `CNNMambaV341BookResidual` · halt state `tanh(book_gate)=0.0158` (spatial trunk = 1.6% of fused).

## 1. Trajectory (MLflow `get_metric_history`)

| step | `book_gate_tanh` | IC_1s | IC_5s | IC_10s | total loss |
|-----:|-----:|-----:|-----:|-----:|-----:|
| 0a (init) | -0.0027 | 0.280 | 0.115 | 0.062 | 49.21 |
| 0b (HC #409 raw bumped→0.549) | +0.0254 | 0.277 | 0.126 | 0.077 | 49.22 |
| 1 ep-1 end | +0.00198 | 0.271 | 0.111 | 0.050 | 49.27 |
| 2 ep-3 end | +0.01583 | 0.274 | 0.128 | 0.090 | 49.30 |

Gate did NOT start high and decay. It started ~0, was manually shoved to tanh≈0.5 mid-ep-1 (HC #409 patch, snapshot `fold_00_intra_ckpt.book_gate_fix.pt`). Optimizer reverted it to ~0.002 within one epoch. HC #422's "init=0.5" is that post-shove snapshot, not steady state. Per-head losses flat ep-1→ep-3 — model parked in a local min with book pathway pruned.

## 2. Code findings (`scripts/v3_4_research/dispatch_v34_1_residual.py`, reused by v3.4.2)

```python
self.v32_core   = CNNMambaV32()                # FULL v3.3 backbone+heads, warmstarted
self.book_cnn   = Book2DCNN(out_dim=TRUNK_DIM) # ~157K params, RANDOM init
self.book_gate  = nn.Parameter(torch.zeros(1)) # init=0 ⇒ tanh(0)=0
# forward:
trunk_out = c.trunk(emb)
book_emb  = self.book_cnn(batch["book_pyramid"])
trunk_out = trunk_out + tanh(self.book_gate) * book_emb
return {n: head(trunk_out) for n, head in c.heads.items()}
```

Warmstart loads 245/245 v3.3 tensors into `v32_core.*`; `book_cnn`+`book_gate` stay random. v3.4.2 uses fixed-weight MTL: `hw_log_ret_1s=1.0/_5s=0.7/_10s=0.5/_30s=0.3`. **No aux head reads `book_emb` directly** — only grad path into BookCNN is `gate · book_emb`.

## 3. Checkpoint forensics (`scripts/diag_v342_inspect_ckpt.py`)

| group | params | RMS |
|---|---:|---:|
| v32_core.t1/t2/t3 (warmstarted) | ~1.46M | 1.05–1.27 |
| v32_core.trunk | 62K | 0.404 |
| **book_cnn (all)** | **157K** | **0.063** |

All `book_cnn.bnN.bias` exactly 0; all `bn.running_mean=0`; all `bn.running_var≈5.6e-45` (denormal — never updated). BN `weight`s uniformly 0.9874 (WD pull from 1.0, no per-channel structure). **BookCNN has barely trained.** Even at gate=0.5: `‖gate·book_emb‖ ≈ 0.033` vs `‖trunk.W‖ ≈ 0.40` — spatial ≤8% of temporal at the "fixed" gate.

## 4. Root-cause ranking

1. **(HIGHEST) Chicken-and-egg cold-start asymmetry.** Warm v3.3 trunk (RMS≈1.07) + random BookCNN (RMS≈0.06) → adding book_emb to a calibrated trunk RAISES loss; gradient on `book_gate` points toward zero, scaling BookCNN's gradient to ~0, keeping it random. Self-reinforcing.
2. **No aux supervision on `book_emb`.** All loss flows through the gated residual; with gate≈0 there's no signal for BookCNN to escape.
3. **Init=0 + residual addition** is a stable fixed point. Escaping needs BookCNN to first produce useful features — which it cannot, per #1/#2.
4. **Loss weighting dominated by 1s/5s** (hw 1.0/0.7) — the horizons the temporal trunk already serves. Spatial features are conceptually more useful for 30s+/MFE/MAE (hw 0.1–0.3); dominant gradients don't "want" book features.
5. **(LOW) Book-feature alignment.** BN running_mean=0 is consistent with — but doesn't prove — book inputs being noise/misaligned. Needs a one-off audit.

## 5. Fix list for v3.4.x relaunch (by leverage)

(a) **Aux head on `book_emb`** (λ≈0.1, on `log_ret_5s`+`log_ret_30s`+`pred_mfe_30s_ticks`). Forces gradient into BookCNN regardless of gate. **Top leverage — kills the chicken-and-egg.**
(b) **Two-phase warmup.** Phase-1: freeze v32_core, train BookCNN+aux ~10K steps until BookCNN RMS ≥ 0.3 and BN running stats populated. Phase-2: unfreeze, raw=0.5, resume joint.
(c) **Gate ramp**: `gate(t)=ramp(t)·tanh(learned)`, ramp 0→1 over first 30%. Loss can't punish book_emb before it's trained.
(d) **Confirm BookCNN is alive**: log per-batch `‖book_emb‖`, `std`. `std→0` ⇒ dead-channel bug; `std≈1` ⇒ pure gradient-routing fix (a/b/c).
(e) **Audit `mbo_book_features/*.npz` ↔ event-window alignment** (~30 min). Sample windows; compare `book_pyramid[:,-1,:,:]` against event-ts bid/ask.
(f) (LOW) Up-weight longer-horizon heads (`hw_log_ret_30s` 0.3→0.5, `hw_pred_mfe_30s` 0.2→0.4) so gradient *wants* spatial features.

**For v3.4.3**: (e) first; then (a)+(b)+(d). (c)+(f) second pass.
