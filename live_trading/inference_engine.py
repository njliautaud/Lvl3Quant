"""
Real-Time Inference Engine
===========================

Converts streaming MBO events → features → model predictions.

Supports:
- Event-based models (CNN, Transformer, Mamba): windowed event sequences
- Bar-based models (LGBM): derived features from StreamingFeatures
- Multiple models running in parallel
- Feature caching for efficiency
"""

import logging
import time
from collections import deque
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

from live_trading.data_feed import MBOEvent
from live_trading.model_registry import ModelBase, get_registry

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Model prediction with metadata"""
    model_name: str
    value: float  # Raw prediction (ticks)
    timestamp: float
    features_hash: str
    latency_ms: float


class InferenceEngine:
    """
    Real-time inference engine.

    Maintains separate feature buffers per model type:
    - Event models: rolling window of raw events (W x 6)
    - LGBM models: 21-dim derived features via StreamingFeatures
    """

    def __init__(
        self,
        event_window_size: int = 500,
        bar_window_size: int = 100,  # For bar-based features
        warmup_events: int = 500,
        prediction_stride: int = 1,  # Only predict every N events (1 = every event)
    ):
        self.event_window_size = event_window_size
        self.bar_window_size = bar_window_size
        self.warmup_events = warmup_events
        self.prediction_stride = prediction_stride

        # Event buffer for event-driven models
        self._event_buffer: deque = deque(maxlen=event_window_size)

        # Streaming features for bar-based models
        self._streaming_features: Optional[Any] = None
        try:
            from live_trading_linux.streaming_features import StreamingFeatures
            self._streaming_features = StreamingFeatures()
        except ImportError:
            logger.warning("StreamingFeatures not available — LGBM models will not work")

        # Model registry
        self._registry = get_registry()

        # Callbacks for predictions
        self._callbacks: Dict[str, list] = {}  # model_name -> [callbacks]

        # State
        self._event_count = 0
        self._last_mid = 0.0

    def on_prediction(self, model_name: str, callback: Callable[[Prediction], None]):
        """Register callback for model predictions"""
        if model_name not in self._callbacks:
            self._callbacks[model_name] = []
        self._callbacks[model_name].append(callback)

    async def process_event(self, event: MBOEvent) -> Dict[str, Prediction]:
        """
        Process incoming MBO event and generate predictions.

        Args:
            event: MBOEvent from data feed

        Returns:
            Dict of {model_name: prediction}
        """
        self._event_count += 1

        # Update mid-price estimate (simple approximation)
        if event.price > 0:
            if event.side == 0:  # bid
                self._last_mid = event.price + event.spread / 2
            elif event.side == 1:  # ask
                self._last_mid = event.price - event.spread / 2

        # Build feature vector for event buffer
        event_features = np.array([
            np.log1p(event.time_delta * 1000),  # time_delta_log (ms)
            float(event.event_type),
            float(event.side),
            0.0,  # price_rel_ticks (relative to mid — TODO: compute properly)
            np.log1p(event.qty),
            event.spread,  # already in ticks
        ], dtype=np.float32)

        # Update event buffer
        self._event_buffer.append(event_features)

        # Update streaming features (for LGBM)
        lgbm_features = None
        if self._streaming_features is not None:
            lgbm_features = self._streaming_features.update(
                event_type=event.event_type,
                side=event.side,
                price=event.price,
                qty=event.qty,
                spread=event.spread,
                time_delta=event.time_delta,
            )

        # Skip warmup period
        if self._event_count < self.warmup_events:
            return {}

        # Skip if not on prediction stride
        if (self._event_count - self.warmup_events) % self.prediction_stride != 0:
            return {}

        # Run inference on all registered models
        predictions = {}
        t0 = time.perf_counter()

        for model_name in self._registry.list_models():
            model = self._registry.get_model(model_name)
            if model is None:
                continue

            try:
                # Select appropriate features
                if model.metadata.architecture == "lgbm":
                    if lgbm_features is None:
                        continue
                    features = lgbm_features
                else:
                    # Event-based model: use event buffer
                    features = np.array(self._event_buffer, dtype=np.float32)

                # Run prediction
                pred_value = model.predict(features)

                # Compute features hash for tracking
                import hashlib
                features_hash = hashlib.sha1(features.tobytes()).hexdigest()[:12]

                latency_ms = (time.perf_counter() - t0) * 1000

                prediction = Prediction(
                    model_name=model_name,
                    value=float(pred_value),
                    timestamp=event.timestamp,
                    features_hash=features_hash,
                    latency_ms=latency_ms,
                )

                predictions[model_name] = prediction

                # Trigger callbacks
                if model_name in self._callbacks:
                    for callback in self._callbacks[model_name]:
                        try:
                            callback(prediction)
                        except Exception as e:
                            logger.error(f"Callback error for {model_name}: {e}")

            except Exception as e:
                logger.error(f"Inference error for {model_name}: {e}", exc_info=True)
                continue

        return predictions

    def get_last_mid(self) -> float:
        """Get last estimated mid price"""
        return self._last_mid

    def get_event_count(self) -> int:
        """Get total events processed"""
        return self._event_count

    def reset(self):
        """Reset engine state"""
        self._event_buffer.clear()
        if self._streaming_features is not None:
            # Reinitialize streaming features
            try:
                from live_trading_linux.streaming_features import StreamingFeatures
                self._streaming_features = StreamingFeatures()
            except ImportError:
                pass
        self._event_count = 0
        self._last_mid = 0.0
        logger.info("InferenceEngine reset")
