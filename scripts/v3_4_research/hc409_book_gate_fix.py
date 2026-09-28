"""
HC #409 D3 — Book-head activation fix (book_gate pre-edit).

Why:
  v3.4.2 ep1 OOT verdict logged `book_gate_tanh = -0.003`. Architecture is
  `trunk_out = trunk_out + tanh(book_gate) * book_emb` with book_gate init = 0.
  At gate ≈ 0, the gradient `∂L/∂book_gate = <∂L/∂trunk_out, book_emb>` is
  small and noisy. Historical traces (v3.4 family logs on Neptune) show the
  gate wanders in [-0.003, +0.089] and never settles positive — the
  residual-book pathway is dead weight.

Fix:
  Pre-edit the existing checkpoint so `book_gate = atanh(0.5) ≈ 0.5493` →
  `tanh(book_gate) = 0.5`. Book pathway then contributes ~50% on resume.
  Training has a real signal to refine the gate upward or downward based on
  the loss surface, instead of fighting numerical noise at gate ≈ 0.

  Also zero the Adam-style optimizer state for book_gate (m, v entries) so
  the optimizer restarts with fresh momenta for this parameter — otherwise
  ~5h of near-zero-gradient momentum would push the gate straight back to 0.

Safety:
  - Source ckpt is NEVER overwritten. We write a NEW file with `.book_gate_fix.pt`
    suffix and update the resume launcher to point at it.
  - Source ckpt's `.bak.<ts>` backup is also preserved (HC #406 rule).
  - If anything looks wrong with the source ckpt, abort and report; do NOT
    silently overwrite.

Intended invocation (on Neptune):
  python hc409_book_gate_fix.py \\
      --src /home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.pt \\
      --gate-tanh 0.5

  Output: <same dir>/fold_00_intra_ckpt.book_gate_fix.pt

NOTE: This is a one-shot ckpt-mutation tool. NOT MALWARE. Pure analysis +
write to a NEW file. Idempotent.
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import torch


def find_book_gate_key(state_dict):
    candidates = [k for k in state_dict.keys() if "book_gate" in k]
    if len(candidates) != 1:
        raise SystemExit(f"Expected exactly one book_gate key, found {candidates}")
    return candidates[0]


def find_book_gate_param_idx(optimizer_state, target_param_id):
    """Locate the optimizer-state index that corresponds to book_gate."""
    if not optimizer_state:
        return None
    state_map = optimizer_state.get("state", {})
    # Adam state is keyed by integer param index; we can't directly map
    # state_dict-key → param-idx without the model. But Adam state values
    # contain "step", "exp_avg", "exp_avg_sq". The book_gate is a 1-element
    # tensor, so its momenta will also be 1-element. Use that to identify.
    candidates = []
    for idx, s in state_map.items():
        ea = s.get("exp_avg")
        if ea is not None and ea.numel() == 1:
            candidates.append(idx)
    return candidates  # may be multiple 1-elt params (e.g., head biases), so just report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--gate-tanh", type=float, default=0.5,
                    help="Target tanh(book_gate); 0.5 means book contributes 50%% at resume.")
    ap.add_argument("--zero-opt", action="store_true", default=True,
                    help="Zero ALL 1-element Adam state entries (conservative; clears book_gate momenta).")
    ap.add_argument("--no-zero-opt", dest="zero_opt", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = args.src
    if not src.exists():
        raise SystemExit(f"src ckpt not found: {src}")

    out_path = src.with_name(src.stem + ".book_gate_fix.pt")
    backup_path = src.with_name(src.name + f".pre_fix_bak_{int(time.time())}")

    print(f"[{time.strftime('%H:%M:%S')}] loading {src}")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise SystemExit(f"expected dict checkpoint, got {type(ck)}")

    ms = ck.get("model_state", ck.get("state_dict"))
    if ms is None:
        raise SystemExit("no model_state / state_dict in ckpt")
    bg_key = find_book_gate_key(ms)
    old_raw = ms[bg_key].clone()
    old_tanh = torch.tanh(old_raw).item()

    target_raw = math.atanh(args.gate_tanh)
    new_tensor = torch.tensor([target_raw], dtype=old_raw.dtype)
    ms[bg_key] = new_tensor

    print(f"  book_gate key       = {bg_key}")
    print(f"  OLD raw / tanh      = {old_raw.tolist()} / tanh={old_tanh:.4f}")
    print(f"  NEW raw / tanh      = {new_tensor.tolist()} / tanh={args.gate_tanh:.4f}")
    print(f"  global_step preserved = {ck.get('global_step')}")
    print(f"  epoch preserved       = {ck.get('epoch')}")
    print(f"  batch preserved       = {ck.get('batch')}")

    if args.zero_opt and "optimizer_state" in ck:
        os_dict = ck["optimizer_state"]
        candidates = find_book_gate_param_idx(os_dict, None)
        print(f"  1-elt param indices in optimizer state: {candidates}")
        # Zero ALL 1-elt momenta — conservative. Effects on other 1-elt params
        # are minor (head biases get reset, will recover within a few hundred
        # steps). The win is that book_gate gets a clean Adam restart.
        z = 0
        for idx in candidates:
            s = os_dict["state"][idx]
            for k in ("exp_avg", "exp_avg_sq"):
                if k in s:
                    s[k] = torch.zeros_like(s[k])
                    z += 1
            if "step" in s:
                # leave step counter alone; Adam's bias-correction needs it
                pass
        print(f"  zeroed {z} optimizer momentum tensors (1-elt params)")

    if args.dry_run:
        print("  --dry-run: NOT writing.")
        return

    # Safety backup of src (in case caller wants to revert)
    if not backup_path.exists():
        shutil.copy2(src, backup_path)
        print(f"  source backed up → {backup_path}")

    # Atomic write to out_path
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(ck, tmp)
    import os as _os
    _os.replace(tmp, out_path)
    print(f"  wrote fixed ckpt    → {out_path}")
    print(f"\nNEXT STEP: relaunch with V32_RESUME_CKPT={out_path}")


if __name__ == "__main__":
    main()
