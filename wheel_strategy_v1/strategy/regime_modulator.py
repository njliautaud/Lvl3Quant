"""
Wheel-side import shim. Canonical implementation lives at
/home/jupiter/Lvl3Quant/strategy/macro_picker/regime_modulator.py
(HC #561 R2 — "implement once, expose to both consumers").

Symlinks are avoided here because /home/jupiter/Lvl3Quant/strategy is a
sibling package, not on sys.path of the wheel package. A thin re-export
via absolute path import is the most portable option in this monorepo.
"""
from __future__ import annotations
import sys
from pathlib import Path

_CANONICAL_DIR = Path("/home/jupiter/Lvl3Quant/strategy/macro_picker")
if str(_CANONICAL_DIR) not in sys.path:
    sys.path.insert(0, str(_CANONICAL_DIR))

from regime_modulator import (  # noqa: E402,F401
    RegimeState,
    classify_regime,
    regime_series,
)
