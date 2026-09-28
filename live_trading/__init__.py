"""
Live Paper Trading Engine for Model Deployment
===============================================

Production-grade paper trading infrastructure for real-time model inference
and simulated execution with MBO market data.

Components:
- data_feed.py: Real-time MBO data ingestion (live feed + replay)
- model_registry.py: Model loading and management
- inference_engine.py: Real-time model inference pipeline
- card_engine.py: Trading card execution system
- position_manager.py: Position tracking and PnL
- fill_simulator.py: Realistic fill modeling
- risk_manager.py: Risk limits and stop losses
- paper_broker.py: Simulated order execution
- monitoring.py: Logging, metrics, Discord alerts
- main.py: Main orchestrator

Architecture:
  MBO Feed → Feature Engine → Models → Cards → Orders → Fill Sim → Positions → PnL
"""

__version__ = "1.0.0"
__author__ = "Claude Opus 4.6"
