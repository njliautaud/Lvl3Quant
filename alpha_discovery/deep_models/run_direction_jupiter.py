#!/usr/bin/env python3
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.insert(0, '/home/jupiter/Lvl3Quant/alpha_discovery/deep_models')

import train_walkforward as twf
twf.DEFAULT_BOOK_DIR = '/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot'

from run_direction_cnn import run_direction_holdout

print("Book dir: " + twf.DEFAULT_BOOK_DIR)
print("Launching direction CNN on CPU (train=60d, oot=20d, epochs=5)...")

result = run_direction_holdout(
    device_str='cpu',
    train_days=60,
    oot_days=20,
    epochs=5,
    batch_size=512,
    subsample=3,
    horizon=100,
    window_size=20,
)

print("RESULT:", result)
