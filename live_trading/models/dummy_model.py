"""
Dummy Model for Infrastructure Testing
=======================================

Ultra-fast model that returns random predictions instantly.
Used for testing trading logic without CPU bottleneck.
"""

import numpy as np
import torch
import torch.nn as nn


class DummyEventCNN1D(nn.Module):
    """Minimal model that returns random predictions"""

    def __init__(self):
        super().__init__()
        # Single tiny layer - just for valid state_dict
        self.dummy = nn.Linear(1, 3)

    def forward(self, x):
        # Ignore input, return random predictions scaled to reasonable range
        batch_size = 1 if x.dim() == 2 else x.shape[0]
        # Return predictions in range [-5, +5] ticks (typical for ES)
        return torch.randn(batch_size, 3) * 2.0


if __name__ == "__main__":
    # Create and save dummy model
    model = DummyEventCNN1D()

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'fold': 0,
        'ic_10s': 0.0,  # Dummy, no real IC
        'timestamp': '2026-04-17_dummy',
        'note': 'Ultra-fast dummy model for infrastructure testing only'
    }

    torch.save(checkpoint, '/home/jupiter/Lvl3Quant/live_trading/models/dummy_event_cnn1d.pt')
    print('✓ Dummy model created')
    print('  Parameters:', sum(p.numel() for p in model.parameters()))
    print('  Speed: ~1M predictions/sec (instant)')
