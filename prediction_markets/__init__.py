"""Prediction Markets Trading System - Kalshi + Polymarket"""
from .kalshi_client import KalshiTrader, BracketPricer, VolModel
from .polymarket_client import PolymarketScanner, PolymarketTrader
from .edge_detector import EdgeDetector
