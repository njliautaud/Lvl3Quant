"""Check 5: book head gate scalar (v3.4.2 only).

Reads the checkpoint and looks for `book_gate` (HC #409). Must be > 0.3.
If no ckpt provided or scalar absent for v3.4.2 → FAIL.
For v3.3 → automatic SKIP (no book head).
"""
from __future__ import annotations
import os


GATE_MIN = 0.3


def _read_gate(ckpt_path: str) -> tuple[float | None, str | None]:
    try:
        import torch  # type: ignore
    except Exception as e:  # noqa
        return None, f"torch import failed: {e!r}"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa
        return None, f"torch.load failed: {e!r}"

    # Walk nested dicts. Gate may be stored as state_dict tensor or scalar.
    candidates = []

    def _walk(obj, prefix=""):
        if hasattr(obj, "items"):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(k, str) and "book_gate" in k.lower():
                    candidates.append((key, v))
                _walk(v, key)

    _walk(ckpt)
    if not candidates:
        return None, "no book_gate key found in checkpoint"

    # Prefer the deepest/most specific match.
    key, val = candidates[-1]
    try:
        import torch  # type: ignore
        if isinstance(val, torch.Tensor):
            return float(val.detach().float().squeeze().item()), key
    except Exception:
        pass
    try:
        return float(val), key
    except Exception as e:  # noqa
        return None, f"could not coerce {key} to float: {e!r}"


def run(ckpt_path: str | None, model_family: str) -> dict:
    if model_family == "v3_3":
        return {
            "check": "book_gate",
            "passed": True,
            "skipped": True,
            "failures": [],
            "details": {"reason": "v3.3 has no book head"},
        }
    if not ckpt_path:
        return {
            "check": "book_gate",
            "passed": False,
            "failures": ["no --ckpt provided for v3.4.2"],
            "details": {},
        }
    if not os.path.exists(ckpt_path):
        return {
            "check": "book_gate",
            "passed": False,
            "failures": [f"ckpt not found: {ckpt_path}"],
            "details": {},
        }
    gate, info = _read_gate(ckpt_path)
    if gate is None:
        return {
            "check": "book_gate",
            "passed": False,
            "failures": [f"could not read book_gate: {info}"],
            "details": {"ckpt": ckpt_path},
        }
    failures = []
    if gate <= GATE_MIN:
        failures.append(f"book_gate={gate:.4f} <= {GATE_MIN} (HC #409 floor)")
    return {
        "check": "book_gate",
        "passed": len(failures) == 0,
        "failures": failures,
        "details": {"ckpt": ckpt_path, "key": info, "book_gate": gate, "min": GATE_MIN},
    }
