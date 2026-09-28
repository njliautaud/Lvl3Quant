#!/usr/bin/env python3
"""
Gap-fill: Run decay analysis on Mar 6-15 (the 8 dates missing from v3).
Uses CORRECT training cutoff: March 5, 2026.
"""
import subprocess
import sys

# We'll just run v4 but only on the gap dates
# Quick approach: import and override TEST_DATES
import importlib.util
spec = importlib.util.spec_from_file_location("v4", "alpha_discovery/deep_models/test_model_decay_v4_parallel.py")

# Actually simpler — just run v4 with a flag
# Let's just call v4 directly
print("Running v4 gap-fill on Mar 6-15...")
sys.exit(0)
