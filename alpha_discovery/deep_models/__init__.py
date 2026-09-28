"""
Deep Learning Models for Lvl3Quant MBO Data.

Three model architectures designed to learn directly from raw MBO data
rather than pre-computed tabular features:

1. EventTransformer - Processes tokenized event sequences (adds, cancels, trades)
2. BookSpatialCNN  - Processes raw 10-level order book snapshots as 2D tensors
3. TradeSequenceLSTM - Processes raw trade-by-trade flow sequences

Each model takes NPZ files produced by the Rust cache builder
(lob_cache_builder --mode events/book/trades) as input.
"""

from .event_transformer import EventTransformer
from .book_spatial_cnn import BookSpatialCNN
from .trade_sequence_lstm import TradeSequenceLSTM

__all__ = ['EventTransformer', 'BookSpatialCNN', 'TradeSequenceLSTM']
