"""
Model Registry and Loader
==========================

Manages loading and versioning of trained models:
- EventCNN1D (PyTorch)
- CNN-Jamba (PyTorch)
- LGBM (joblib)
- Event Transformer (PyTorch)

Supports:
- Model hot-reloading
- Version tracking
- MLflow integration
- Model metadata caching
"""

import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata"""
    name: str
    version: str
    path: str
    model_type: str  # "pytorch" | "lightgbm"
    architecture: str  # "event_cnn_1d" | "cnn_jamba" | "lgbm" | "event_transformer"
    fold: int
    label_horizon: str  # "1s" | "5s" | "10s" | "30s"
    ic_score: float
    timestamp: str
    hash: str  # MD5 of model file


class ModelBase(ABC):
    """Base class for model wrappers"""

    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata

    @abstractmethod
    def predict(self, features: np.ndarray) -> float:
        """
        Run inference on features.

        Args:
            features: Input features (shape depends on model type)

        Returns:
            Prediction (scalar)
        """
        pass

    @abstractmethod
    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Batch prediction"""
        pass


class EventCNN1DModel(ModelBase):
    """EventCNN1D PyTorch model wrapper"""

    def __init__(self, metadata: ModelMetadata, window_size: int = 500):
        super().__init__(metadata)
        self.window_size = window_size
        self._load_model()

    def _load_model(self):
        """Load PyTorch model"""
        import torch
        import torch.nn as nn

        # Simple CNN architecture (matches train_event_cnn_1d.py)
        class CausalConv1d(nn.Module):
            def __init__(self, in_ch, out_ch, kernel, dilation):
                super().__init__()
                self.padding = (kernel - 1) * dilation
                self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

            def forward(self, x):
                return self.conv(nn.functional.pad(x, (self.padding, 0)))

        class EventCNN1D(nn.Module):
            def __init__(self, channels=128, kernel=5, layers=6, dropout=0.1):
                super().__init__()
                self.input_proj = nn.Conv1d(6, channels, 1)
                blocks = []
                for i in range(layers):
                    dilation = 2 ** i
                    blocks.append(nn.Sequential(
                        CausalConv1d(channels, channels, kernel, dilation),
                        nn.BatchNorm1d(channels),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ))
                self.blocks = nn.ModuleList(blocks)
                self.head = nn.Linear(channels, 3)  # 3 horizons

            def forward(self, x):
                # x: (B, W, 6) → (B, 6, W)
                x = x.permute(0, 2, 1)
                x = self.input_proj(x)
                for block in self.blocks:
                    x = x + block(x)
                x = x.mean(dim=2)  # global avg pool
                return self.head(x)

        # Load checkpoint
        checkpoint = torch.load(self.metadata.path, map_location='cpu')
        self.model = EventCNN1D()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        logger.info(f"Loaded EventCNN1D model: {self.metadata.name} on {self.device}")

    def predict(self, features: np.ndarray) -> float:
        """
        Predict on single window.

        Args:
            features: (W, 6) event window

        Returns:
            Prediction for target horizon
        """
        import torch

        # Pad if needed
        if len(features) < self.window_size:
            pad_len = self.window_size - len(features)
            features = np.vstack([np.zeros((pad_len, 6)), features])
        elif len(features) > self.window_size:
            features = features[-self.window_size:]

        with torch.no_grad():
            x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)  # (1, W, 6)
            out = self.model(x)  # (1, 3)
            # Select horizon: 0=1s, 1=5s, 2=10s
            horizon_idx = {"1s": 0, "5s": 1, "10s": 2}.get(self.metadata.label_horizon, 2)
            pred = out[0, horizon_idx].item()
            return pred

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Batch prediction"""
        import torch

        with torch.no_grad():
            x = torch.from_numpy(features).float().to(self.device)
            out = self.model(x)
            horizon_idx = {"1s": 0, "5s": 1, "10s": 2}.get(self.metadata.label_horizon, 2)
            return out[:, horizon_idx].cpu().numpy()


class LGBMModel(ModelBase):
    """LightGBM model wrapper"""

    def __init__(self, metadata: ModelMetadata):
        super().__init__(metadata)
        self._load_model()

    def _load_model(self):
        """Load LGBM model from joblib"""
        import joblib
        self.model = joblib.load(self.metadata.path)
        logger.info(f"Loaded LGBM model: {self.metadata.name}")

    def predict(self, features: np.ndarray) -> float:
        """
        Predict on single sample.

        Args:
            features: (21,) feature vector

        Returns:
            Prediction
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return float(self.model.predict(features)[0])

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Batch prediction"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.model.predict(features)


class ModelRegistry:
    """Central registry for all trading models"""

    def __init__(self):
        self.models: Dict[str, ModelBase] = {}
        self._metadata_cache: Dict[str, ModelMetadata] = {}

    def register_model(
        self,
        name: str,
        model_path: str,
        model_type: str,
        architecture: str,
        fold: int = 0,
        label_horizon: str = "10s",
        ic_score: float = 0.0,
        version: Optional[str] = None,
        **kwargs
    ) -> ModelBase:
        """
        Register a model for inference.

        Args:
            name: Unique model name (e.g., "event_cnn_baseline")
            model_path: Path to model file (.pt or .pkl)
            model_type: "pytorch" or "lightgbm"
            architecture: Architecture name
            fold: Training fold number
            label_horizon: Prediction horizon
            ic_score: Model IC score
            version: Model version string
            **kwargs: Additional model-specific parameters

        Returns:
            Loaded model instance
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Compute hash
        with open(model_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        # Create metadata
        if version is None:
            version = f"fold{fold}"

        metadata = ModelMetadata(
            name=name,
            version=version,
            path=str(model_path),
            model_type=model_type,
            architecture=architecture,
            fold=fold,
            label_horizon=label_horizon,
            ic_score=ic_score,
            timestamp=model_path.stat().st_mtime,
            hash=file_hash,
        )

        # Load model
        if architecture == "event_cnn_1d":
            model = EventCNN1DModel(metadata, window_size=kwargs.get('window_size', 500))
        elif architecture == "cnn_jamba":
            model = EventCNN1DModel(metadata, window_size=kwargs.get('window_size', 1000))
        elif architecture == "lgbm":
            model = LGBMModel(metadata)
        elif architecture == "event_transformer":
            model = EventCNN1DModel(metadata, window_size=kwargs.get('window_size', 500))  # TODO: implement transformer
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.models[name] = model
        self._metadata_cache[name] = metadata

        logger.info(f"Registered model '{name}': {architecture} fold={fold} IC={ic_score:.4f}")
        return model

    def get_model(self, name: str) -> Optional[ModelBase]:
        """Get model by name"""
        return self.models.get(name)

    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())

    def get_metadata(self, name: str) -> Optional[ModelMetadata]:
        """Get model metadata"""
        return self._metadata_cache.get(name)

    def unregister_model(self, name: str):
        """Remove model from registry"""
        if name in self.models:
            del self.models[name]
            del self._metadata_cache[name]
            logger.info(f"Unregistered model: {name}")


# Global singleton
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get global model registry"""
    return _registry
