"""HC #423 §2 — v3.4.2 checkpoint forensic inspector.

Loads /tmp/v342_book_gate_fix_ckpt.pt (epoch-3-end ckpt, copied from Neptune
/home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.book_gate_fix.pt)
and prints:
  - book_gate raw value + tanh
  - per-parameter norms for v32_core.* vs book_cnn.*
  - log_sigma per head (uncertainty weights) — N/A if fixed-MTL run
  - sanity check: are book_cnn weights still ~kaiming-init-magnitude (untrained-like)?

READ-ONLY. Does not load model. Pure state_dict inspection.
"""
import sys
import torch
from collections import defaultdict

CKPT = "/tmp/v342_book_gate_fix_ckpt.pt"

obj = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"top-level keys: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}")
state = obj.get("model_state", obj) if isinstance(obj, dict) else obj

# 1) book_gate
for k in state:
    if "book_gate" in k:
        v = state[k]
        print(f"\n[book_gate] {k} = raw {v.item():.6f}  tanh {torch.tanh(v).item():.6f}")

# 2) per-group norm summary
groups = defaultdict(list)
for k, v in state.items():
    if not torch.is_tensor(v) or v.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        continue
    if k.startswith("v32_core.t1_"):
        groups["v32_core.t1_*"].append((k, v))
    elif k.startswith("v32_core.t2_"):
        groups["v32_core.t2_*"].append((k, v))
    elif k.startswith("v32_core.t3_"):
        groups["v32_core.t3_*"].append((k, v))
    elif k.startswith("v32_core.trunk"):
        groups["v32_core.trunk"].append((k, v))
    elif k.startswith("v32_core.heads"):
        groups["v32_core.heads"].append((k, v))
    elif k.startswith("book_cnn") or k.startswith("book_trunk"):
        groups["book_cnn/book_trunk"].append((k, v))
    elif k.startswith("book_gate"):
        groups["book_gate"].append((k, v))
    elif "log_sigma" in k:
        groups["log_sigma"].append((k, v))
    else:
        groups["other"].append((k, v))

print("\n=== Group norms (RMS over flattened tensor) ===")
for g, items in groups.items():
    n = sum(t.numel() for _, t in items)
    if n == 0:
        continue
    cat = torch.cat([t.flatten().float() for _, t in items])
    print(f"  {g:30s} | params={n:>9,} | mean={cat.mean():.4e} | rms={cat.pow(2).mean().sqrt():.4e} | max|.|={cat.abs().max():.4e}")

# 3) book_cnn detailed weights (are they still small / kaiming-like?)
print("\n=== book_cnn detailed weight stats ===")
for k, v in state.items():
    if (k.startswith("book_cnn") or k.startswith("book_trunk")) and v.dtype in (torch.float32, torch.float16, torch.bfloat16):
        f = v.flatten().float()
        print(f"  {k:55s} shape={tuple(v.shape)} | mean={f.mean():+.4e} | std={f.std():.4e} | max|.|={f.abs().max():.4e}")

# 4) compare v32_core.trunk weight magnitude vs book_cnn final proj
print("\n=== Fusion-side comparison ===")
for k, v in state.items():
    if k in ("v32_core.trunk.0.weight",
             "book_cnn.proj_to_emb.weight",
             "book_trunk.proj_to_emb.weight"):
        f = v.flatten().float()
        print(f"  {k:55s} rms={f.pow(2).mean().sqrt():.4e}")
