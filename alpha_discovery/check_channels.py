#!/usr/bin/env python3
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from alpha_discovery.multi_channel_alpha import CHANNEL_DEFINITIONS
for k, v in CHANNEL_DEFINITIONS.items():
    print(f"{k}: {len(v)} features")
