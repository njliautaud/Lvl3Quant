#!/usr/bin/env python3
"""
LGBM Book Features - Live Trading Inference

Generates predictions from trained LGBM model for live trading.
Designed to run on Saturn/Jupiter CPU nodes.

Usage:
  python lgbm_book_inference.py --model-path results/lgbm_model.pkl --data-path /tmp/current_book_features.npz
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    'bid_price_1','bid_price_2','bid_price_3','bid_price_4','bid_price_5',
    'ask_price_1','ask_price_2','ask_price_3','ask_price_4','ask_price_5',
    'bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5',
    'ask_size_1', 'ask_size_2', 'ask_size_3', 'ask_size_4', 'ask_size_5',
    'cum_delta','rolling_imbalance_100','trade_intensity_100',
    'depth_imbalance_5','spread_ticks',
    'bid_size_change','ask_size_change','mid_price_change_ticks',
    'spread_change_ticks','net_order_flow',
]


class LGBMPredictor:
    """Live prediction wrapper for LGBM book feature model."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

        logger.info(f"Loaded model from {model_path}")

    def predict(self, features):
        """
        Generate predictions from book features.

        Args:
            features: (N, 30) numpy array of book features

        Returns:
            predictions: (N,) array of predicted price changes (ticks, 10s horizon)
        """
        if features.shape[1] != 30:
            raise ValueError(f"Expected 30 features, got {features.shape[1]}")

        preds = self.model.predict(features)
        return preds

    def predict_from_file(self, npz_path):
        """Load features from NPZ and predict."""
        data = np.load(npz_path, allow_pickle=True)
        features = data['features'].astype(np.float32)

        logger.info(f"Loaded {len(features)} samples from {npz_path}")

        predictions = self.predict(features)

        return {
            'predictions': predictions,
            'timestamps': data.get('timestamps', None),
            'n_samples': len(predictions),
        }

    def get_top_signals(self, predictions, timestamps=None, top_n=10, min_confidence=0.5):
        """
        Extract top trading signals.

        Args:
            predictions: (N,) array
            timestamps: (N,) array of timestamps (optional)
            top_n: number of top signals to return
            min_confidence: minimum absolute prediction value

        Returns:
            List of dicts with {idx, prediction, timestamp, direction}
        """
        abs_preds = np.abs(predictions)
        confident_mask = abs_preds >= min_confidence

        if confident_mask.sum() == 0:
            logger.warning("No predictions above confidence threshold")
            return []

        # Get top N by absolute prediction
        top_indices = np.argsort(abs_preds)[-top_n:][::-1]

        signals = []
        for idx in top_indices:
            if abs_preds[idx] < min_confidence:
                continue

            signal = {
                'index': int(idx),
                'prediction': float(predictions[idx]),
                'confidence': float(abs_preds[idx]),
                'direction': 'LONG' if predictions[idx] > 0 else 'SHORT',
            }

            if timestamps is not None:
                signal['timestamp'] = int(timestamps[idx])

            signals.append(signal)

        return signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True, help='Path to trained LGBM model (.pkl)')
    parser.add_argument('--data-path', required=True, help='Path to NPZ with book features')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top signals to show')
    parser.add_argument('--min-confidence', type=float, default=0.5, help='Minimum signal confidence (ticks)')
    parser.add_argument('--output', help='Optional output path for predictions')
    args = parser.parse_args()

    # Load model and predict
    predictor = LGBMPredictor(args.model_path)
    result = predictor.predict_from_file(args.data_path)

    logger.info(f"Generated {result['n_samples']} predictions")

    # Get top signals
    signals = predictor.get_top_signals(
        result['predictions'],
        timestamps=result.get('timestamps'),
        top_n=args.top_n,
        min_confidence=args.min_confidence,
    )

    logger.info(f"\nTop {len(signals)} Signals:")
    logger.info("=" * 80)
    for sig in signals:
        logger.info(
            f"  [{sig['direction']:5s}] idx={sig['index']:6d} "
            f"pred={sig['prediction']:+.3f} conf={sig['confidence']:.3f}"
        )

    # Save if requested
    if args.output:
        np.savez_compressed(
            args.output,
            predictions=result['predictions'],
            timestamps=result.get('timestamps'),
            top_signals=signals,
        )
        logger.info(f"\nSaved predictions to {args.output}")

    return signals


if __name__ == "__main__":
    main()
