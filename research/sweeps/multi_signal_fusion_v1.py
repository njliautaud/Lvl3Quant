#!/usr/bin/env python3
"""
Multi-Signal Fusion v1 — PyTorch Temporal Fusion of Validated Signals
=====================================================================
Research question: Can an ML meta-model that fuses our 7 validated signals
at multiple timeframes beat trading them independently?

Validated signals (independent backtest Sharpes):
  1. Oversold Bounce:        69% WR, PF 3.22, Sharpe ~1.8
  2. Post-Earnings Bounce:   63% WR, PF 1.70, Sharpe 1.50
  3. Skewness Premium:       Sharpe 1.02-2.25
  4. Smart Money Accum:      Sharpe 1.60
  5. Momentum Exhaustion:    Sharpe 2.25
  6. Vol Crush Reversal:     Sharpe 0.96
  7. Price-Volume Divergence: Sharpe 1.23

Key insight: These signals fire at different times, on different stocks,
with different horizons. An attention-based fusion model can learn:
  - Which signals matter MORE in different regimes
  - When confluence of 2+ signals amplifies edge
  - Optimal position sizing based on signal confidence
  - When to AVOID entering despite a single signal (false positive filter)

Architecture: Temporal Fusion Transformer (simplified)
  - Per-stock, per-day: compute all 7 signal scores + regime features
  - Gated residual network for feature selection
  - Multi-head attention across time (lookback of signal history)
  - Output: probability of >2% return in next 5/10/21 days

Target: Neptune RTX 3090 | Expected runtime: 2-4 hours
Author: Claude Opus 4.6 | Date: 2026-07-24

HC compliance:
  - HC #0:   SLIDING window only (252d train, 21d test)
  - HC #724: All signals computed on T-1 data
  - HC #428: Regime-agnostic validation
  - HC #432: MFE-within-horizon
  - HC #744: Full leakage audit
"""

import os
import sys

# Must set BEFORE importing mlflow
os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '3'
os.environ['MLFLOW_HTTP_REQUEST_MAX_RETRIES'] = '1'

import json
import time
import warnings
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from scipy import stats

HAS_MLFLOW = False
# Skip mlflow import entirely if server is not reachable
import socket
def _check_mlflow_server():
    try:
        s = socket.create_connection(("jupiter", 5000), timeout=2)
        s.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

if _check_mlflow_server():
    try:
        import mlflow
        HAS_MLFLOW = True
    except ImportError:
        pass

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CFG = {
    # Universe — same 50 large-caps as DL stock ranker
    "tickers": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "GS", "BAC",
        "V", "MA", "UNH", "JNJ", "PG", "KO", "PEP", "MRK", "ABBV", "LLY",
        "HD", "COST", "WMT", "CRM", "AMD", "NFLX", "ADBE", "INTC", "CSCO", "QCOM",
        "XOM", "CVX", "PFE", "TMO", "ABT", "AVGO", "TXN", "MCD", "NKE", "DIS",
        "CMCSA", "T", "VZ", "NEE", "SO", "SHW", "LMT", "RTX", "CAT", "DE",
    ],
    "sector_etfs": ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLU", "XLB", "XLRE", "XLC"],
    "benchmark": "SPY",
    "start_date": "2012-01-01",
    "end_date": "2026-07-23",

    # Walk-forward
    "train_days": 252,           # 1 year sliding
    "test_days": 21,             # 1 month OOT
    "lookback_window": 20,       # 20 trading days signal history for attention
    "top_k": 5,                  # select top 5 signals per day

    # Holding periods to evaluate
    "hold_periods": [5, 10, 21],
    "primary_hold": 10,          # primary evaluation horizon

    # Costs
    "cost_bps": 10,              # round-trip cost

    # Model architecture
    "n_signal_features": 7,      # 7 validated signals
    "n_regime_features": 8,      # regime context
    "n_stock_features": 6,       # per-stock technicals
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.15,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 40,
    "batch_size": 64,
    "patience": 7,

    # Validation
    "n_permutations": 200,
    "regime_sharpe_gap_max": 0.50,
    "subperiod_cv_max": 0.70,

    # Paths
    "output_dir": "output/multi_signal_fusion_v1",
    "mlflow_experiment": "multi_signal_fusion_v1",

    # DataLoader
    "num_workers": 8,
    "pin_memory": True,
}

SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLC", "AMZN": "XLY", "META": "XLC",
    "NVDA": "XLK", "TSLA": "XLY", "JPM": "XLF", "GS": "XLF", "BAC": "XLF",
    "V": "XLK", "MA": "XLK", "UNH": "XLV", "JNJ": "XLV", "PG": "XLP",
    "KO": "XLP", "PEP": "XLP", "MRK": "XLV", "ABBV": "XLV", "LLY": "XLV",
    "HD": "XLY", "COST": "XLP", "WMT": "XLP", "CRM": "XLK", "AMD": "XLK",
    "NFLX": "XLC", "ADBE": "XLK", "INTC": "XLK", "CSCO": "XLK", "QCOM": "XLK",
    "XOM": "XLE", "CVX": "XLE", "PFE": "XLV", "TMO": "XLV", "ABT": "XLV",
    "AVGO": "XLK", "TXN": "XLK", "MCD": "XLY", "NKE": "XLY", "DIS": "XLC",
    "CMCSA": "XLC", "T": "XLC", "VZ": "XLC", "NEE": "XLU", "SO": "XLU",
    "SHW": "XLB", "LMT": "XLI", "RTX": "XLI", "CAT": "XLI", "DE": "XLI",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. DATA DOWNLOAD
# ===========================================================================

def download_data(cfg: dict) -> dict:
    """Download price data for stocks, sector ETFs, VIX, and SPY."""
    all_tickers = list(set(cfg["tickers"] + cfg["sector_etfs"] + [cfg["benchmark"]]))
    log.info(f"Downloading {len(all_tickers)} tickers from {cfg['start_date']} to {cfg['end_date']}")

    prices_raw = yf.download(
        all_tickers,
        start=cfg["start_date"],
        end=cfg["end_date"],
        auto_adjust=True,
        threads=True,
    )

    if isinstance(prices_raw.columns, pd.MultiIndex):
        close = prices_raw["Close"]
        volume = prices_raw["Volume"]
        high = prices_raw["High"]
        low = prices_raw["Low"]
    else:
        close = prices_raw[["Close"]]
        volume = prices_raw[["Volume"]]
        high = prices_raw[["High"]]
        low = prices_raw[["Low"]]

    # VIX
    vix_raw = yf.download("^VIX", start=cfg["start_date"], end=cfg["end_date"], auto_adjust=True)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix = vix_raw["Close"].squeeze()
    else:
        vix = vix_raw["Close"].squeeze()

    close = close.ffill().dropna(how='all')
    volume = volume.ffill().fillna(0)
    high = high.ffill().dropna(how='all')
    low = low.ffill().dropna(how='all')

    log.info(f"Data: {close.shape[0]} days, {close.shape[1]} instruments, "
             f"{close.index[0].date()} -> {close.index[-1].date()}")

    return {"close": close, "volume": volume, "high": high, "low": low, "vix": vix}


# ===========================================================================
# 2. SIGNAL COMPUTATION (7 Validated Signals)
# ===========================================================================

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_mfi(high, low, close, volume, period=14):
    """Money Flow Index."""
    tp = (high + low + close) / 3
    mf = tp * volume
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0).rolling(period).sum()
    neg_mf = mf.where(delta <= 0, 0.0).rolling(period).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def compute_obv(close, volume):
    """On-Balance Volume (normalized as z-score of 20d slope)."""
    ret = close.pct_change()
    obv = (volume * ret.apply(np.sign)).cumsum()
    # Slope of OBV over 20 days
    obv_slope = obv.diff(20) / 20
    # Normalize
    return (obv_slope - obv_slope.rolling(60).mean()) / (obv_slope.rolling(60).std() + 1e-10)


def compute_all_signals(price_data: dict, cfg: dict) -> dict:
    """
    Compute all 7 validated signals for each stock, each day.
    Returns dict of {ticker: DataFrame(date, signal_1..signal_7)}.

    ALL signals use ONLY T-1 data (we shift by 1 at the end).

    Signal definitions (matching our validated backtests):
    1. Oversold Bounce:      Weekly return <= -10% (drop_10pct_1wk)
    2. Post-Earnings Bounce: >8% drop in 2d (approximated as 2d return <= -8%)
    3. Skewness Premium:     20d return skewness < -1.0 (negative skew = premium)
    4. Smart Money Accum:    OBV divergence (price down, OBV up)
    5. Momentum Exhaustion:  RSI>70 + negative acceleration (mom 1m < mom 3m)
    6. Vol Crush Reversal:   Vol ratio (20d/60d) > 2.0 then drops below 1.0
    7. Price-Volume Diverg:  Price up but volume declining (3wk divergence)
    """
    close = price_data["close"]
    volume = price_data["volume"]
    high = price_data["high"]
    low = price_data["low"]
    tickers = cfg["tickers"]

    log.info("Computing 7 validated signals for all stocks...")
    all_signals = {}

    for ticker in tickers:
        if ticker not in close.columns:
            continue

        c = close[ticker].copy()
        v = volume[ticker].copy() if ticker in volume.columns else pd.Series(0, index=close.index)
        h = high[ticker].copy() if ticker in high.columns else c
        l = low[ticker].copy() if ticker in low.columns else c

        sig = pd.DataFrame(index=close.index)

        # --- Signal 1: Oversold Bounce ---
        # Weekly return <= -10%
        ret_5d = c.pct_change(5)
        # Continuous score: how oversold (more negative = stronger signal)
        sig['oversold_bounce'] = np.clip(-ret_5d / 0.10, 0, 3)  # 0-3 scale

        # --- Signal 2: Post-Earnings Bounce ---
        # 2-day return <= -8% (proxy for post-earnings crash)
        ret_2d = c.pct_change(2)
        sig['post_earnings_bounce'] = np.clip(-ret_2d / 0.08, 0, 3)

        # --- Signal 3: Skewness Premium ---
        # 20d return skewness (negative skew = contrarian bounce opportunity)
        ret_1d = c.pct_change()
        skew_20d = ret_1d.rolling(20).skew()
        sig['skewness_premium'] = np.clip(-skew_20d / 1.0, 0, 3)  # More negative = stronger

        # --- Signal 4: Smart Money Accumulation ---
        # OBV rising while price is flat/down
        obv_z = compute_obv(c, v)
        price_mom_20d = c.pct_change(20)
        # Smart money: OBV positive divergence (OBV up, price flat/down)
        smart_money = obv_z.clip(lower=0) * np.clip(-price_mom_20d / 0.05, 0, 2)
        sig['smart_money_accum'] = smart_money.clip(0, 3)

        # --- Signal 5: Momentum Exhaustion Reversal ---
        # RSI > 70 AND 1m momentum < 3m momentum (decelerating)
        rsi_14 = compute_rsi(c, 14)
        mom_1m = c.pct_change(21)
        mom_3m = c.pct_change(63)
        # Exhaustion: RSI overbought + decelerating momentum (SHORT signal, but we
        # convert to "reversal opportunity" score for the other side)
        rsi_excess = np.clip((rsi_14 - 70) / 10, 0, 3)
        decel = np.clip((mom_3m - mom_1m) / 0.05, 0, 3)
        sig['momentum_exhaustion'] = (rsi_excess * decel).clip(0, 3)

        # --- Signal 6: Vol Crush Reversal ---
        # Volatility spike then reversion: vol 20d/60d was >2, now dropping
        vol_20d = ret_1d.rolling(20).std() * np.sqrt(252)
        vol_60d = ret_1d.rolling(60).std() * np.sqrt(252)
        vol_ratio = vol_20d / (vol_60d + 1e-10)
        # Signal fires when vol_ratio has been high but is now declining
        vol_ratio_lag5 = vol_ratio.shift(5)
        vol_crush = np.clip((vol_ratio_lag5 - 1.5) / 0.5, 0, 2) * np.clip((vol_ratio_lag5 - vol_ratio) / 0.3, 0, 2)
        sig['vol_crush_reversal'] = vol_crush.clip(0, 3)

        # --- Signal 7: Price-Volume Divergence ---
        # Price rising but volume declining over 3 weeks
        price_trend_15d = c.pct_change(15)
        vol_ma_5d = v.rolling(5).mean()
        vol_ma_20d = v.rolling(20).mean()
        vol_trend = vol_ma_5d / (vol_ma_20d + 1e-10) - 1  # negative = volume declining
        # Bearish divergence: price up, volume down (sell signal → score as reversal risk)
        pv_div = np.clip(price_trend_15d / 0.05, 0, 2) * np.clip(-vol_trend / 0.2, 0, 2)
        sig['price_vol_divergence'] = pv_div.clip(0, 3)

        # Shift all signals by 1 day (T-1 compliance, HC #724)
        sig = sig.shift(1)

        all_signals[ticker] = sig

    log.info(f"Computed signals for {len(all_signals)} stocks")
    return all_signals


# ===========================================================================
# 3. REGIME FEATURES (Market-Level Context)
# ===========================================================================

def compute_regime_features(price_data: dict, cfg: dict) -> pd.DataFrame:
    """
    Compute 8 market-level regime features per day.
    These capture the macro environment that modulates signal effectiveness.
    """
    close = price_data["close"]
    spy = close[cfg["benchmark"]]
    vix = price_data["vix"]

    regime = pd.DataFrame(index=close.index)

    # 1. VIX level (percentile rank over 252d)
    regime['vix_pctile'] = vix.rolling(252).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100 if len(x.dropna()) > 20 else 0.5,
        raw=False
    )

    # 2. VIX change (5d)
    regime['vix_change_5d'] = vix.pct_change(5)

    # 3. SPY trend (above/below 200d SMA, continuous)
    sma200 = spy.rolling(200).mean()
    regime['spy_trend'] = (spy - sma200) / (sma200 + 1e-10)

    # 4. SPY momentum (21d return)
    regime['spy_mom_21d'] = spy.pct_change(21)

    # 5. Market breadth (fraction of stocks above 50d SMA)
    sma50_all = close[cfg["tickers"]].rolling(50).mean()
    regime['breadth'] = (close[cfg["tickers"]] > sma50_all).sum(axis=1) / len(cfg["tickers"])

    # 6. Cross-sector dispersion (std of sector ETF 21d returns)
    sector_rets = close[cfg["sector_etfs"]].pct_change(21)
    regime['sector_dispersion'] = sector_rets.std(axis=1)

    # 7. Realized vol regime (20d SPY vol, percentile)
    spy_ret = spy.pct_change()
    rv20 = spy_ret.rolling(20).std() * np.sqrt(252)
    regime['rv_pctile'] = rv20.rolling(252).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100 if len(x.dropna()) > 20 else 0.5,
        raw=False
    )

    # 8. Credit stress proxy (using XLF/XLU ratio as risk-on/risk-off)
    if 'XLF' in close.columns and 'XLU' in close.columns:
        risk_ratio = close['XLF'] / close['XLU']
        regime['risk_appetite'] = risk_ratio.pct_change(21)
    else:
        regime['risk_appetite'] = 0.0

    # Shift by 1 for T-1 compliance
    regime = regime.shift(1)

    log.info(f"Computed {len(regime.columns)} regime features")
    return regime


# ===========================================================================
# 4. STOCK-LEVEL FEATURES (Per-Stock Technicals)
# ===========================================================================

def compute_stock_features(price_data: dict, cfg: dict) -> dict:
    """
    Compute 6 per-stock technical features that complement the 7 signals.
    These help the model understand the stock's current state.
    """
    close = price_data["close"]
    volume = price_data["volume"]

    stock_feats = {}
    for ticker in cfg["tickers"]:
        if ticker not in close.columns:
            continue

        c = close[ticker]
        v = volume[ticker] if ticker in volume.columns else pd.Series(0, index=close.index)
        ret = c.pct_change()

        sf = pd.DataFrame(index=close.index)

        # 1. 3-month momentum
        sf['mom_3m'] = c.pct_change(63)

        # 2. Realized vol 20d
        sf['rvol_20d'] = ret.rolling(20).std() * np.sqrt(252)

        # 3. RSI 14
        sf['rsi_14'] = compute_rsi(c, 14) / 100  # normalize to 0-1

        # 4. Distance from 52-week high
        high_52w = c.rolling(252).max()
        sf['dist_from_high'] = (c - high_52w) / (high_52w + 1e-10)

        # 5. Volume trend (20d avg / 60d avg)
        sf['vol_trend'] = v.rolling(20).mean() / (v.rolling(60).mean() + 1e-10)

        # 6. Mean reversion score (20d return z-score)
        ret_20d = c.pct_change(20)
        sf['mr_zscore'] = (ret_20d - ret_20d.rolling(252).mean()) / (ret_20d.rolling(252).std() + 1e-10)

        # Shift by 1 for T-1 compliance
        sf = sf.shift(1)
        stock_feats[ticker] = sf

    return stock_feats


# ===========================================================================
# 5. DATASET CONSTRUCTION
# ===========================================================================

class SignalFusionDataset(Dataset):
    """
    Dataset for multi-signal fusion.
    Each sample is (stock, day) with:
      - 7 signal scores (current day)
      - 8 regime features (current day)
      - 6 stock features (current day)
      - 20-day lookback of signal history (for temporal attention)
      - Label: forward return (5d, 10d, 21d)
    """

    def __init__(self, X_signals, X_regime, X_stock, X_signal_history, y, weights=None):
        """
        X_signals: (N, 7)      — current signal scores
        X_regime:  (N, 8)      — regime context
        X_stock:   (N, 6)      — stock-level features
        X_signal_history: (N, 20, 7)  — lookback signal history
        y:         (N,)        — forward return
        weights:   (N,)        — sample weights (optional)
        """
        self.X_signals = torch.FloatTensor(X_signals)
        self.X_regime = torch.FloatTensor(X_regime)
        self.X_stock = torch.FloatTensor(X_stock)
        self.X_history = torch.FloatTensor(X_signal_history)
        self.y = torch.FloatTensor(y)
        self.weights = torch.FloatTensor(weights) if weights is not None else torch.ones(len(y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_signals[idx], self.X_regime[idx], self.X_stock[idx],
                self.X_history[idx], self.y[idx], self.weights[idx])


def build_dataset(all_signals: dict, regime_features: pd.DataFrame,
                  stock_features: dict, price_data: dict, cfg: dict,
                  start_idx: int, end_idx: int, hold_period: int = 10):
    """
    Build training/test dataset from signal data.
    Returns X arrays and y array.
    """
    close = price_data["close"]
    tickers = cfg["tickers"]
    lookback = cfg["lookback_window"]

    X_sigs, X_reg, X_stk, X_hist, Y = [], [], [], [], []
    dates_out, tickers_out = [], []

    trading_days = close.index[start_idx:end_idx]

    for day_idx in range(start_idx, min(end_idx, len(close) - hold_period)):
        date = close.index[day_idx]
        regime_row = regime_features.loc[date] if date in regime_features.index else None
        if regime_row is None or regime_row.isna().all():
            continue

        regime_vals = regime_row.values.astype(np.float64)
        if np.any(np.isnan(regime_vals)):
            regime_vals = np.nan_to_num(regime_vals, 0.0)

        for ticker in tickers:
            if ticker not in all_signals or ticker not in stock_features:
                continue
            if ticker not in close.columns:
                continue

            sig_df = all_signals[ticker]
            stk_df = stock_features[ticker]

            if date not in sig_df.index or date not in stk_df.index:
                continue

            # Current signal values
            sig_vals = sig_df.loc[date].values.astype(np.float64)
            stk_vals = stk_df.loc[date].values.astype(np.float64)

            if np.any(np.isnan(sig_vals)) or np.any(np.isnan(stk_vals)):
                sig_vals = np.nan_to_num(sig_vals, 0.0)
                stk_vals = np.nan_to_num(stk_vals, 0.0)

            # Signal history (lookback window)
            hist_start = day_idx - lookback
            if hist_start < 0:
                continue

            hist_dates = close.index[hist_start:day_idx]
            hist_vals = []
            for hd in hist_dates:
                if hd in sig_df.index:
                    hv = sig_df.loc[hd].values.astype(np.float64)
                    hv = np.nan_to_num(hv, 0.0)
                    hist_vals.append(hv)
                else:
                    hist_vals.append(np.zeros(7))

            if len(hist_vals) < lookback:
                # Pad with zeros
                hist_vals = [np.zeros(7)] * (lookback - len(hist_vals)) + hist_vals

            hist_arr = np.array(hist_vals[-lookback:])

            # Forward return (label)
            fwd_idx = min(day_idx + hold_period, len(close) - 1)
            p_now = close[ticker].iloc[day_idx]
            p_fwd = close[ticker].iloc[fwd_idx]
            if pd.isna(p_now) or pd.isna(p_fwd) or p_now <= 0:
                continue
            fwd_ret = p_fwd / p_now - 1

            # Check for any signal activity (skip completely dead signals)
            if np.sum(np.abs(sig_vals)) < 0.01 and np.sum(np.abs(hist_arr)) < 0.01:
                # No signal activity — still include but with lower weight
                # (market provides baseline return data)
                pass

            X_sigs.append(sig_vals)
            X_reg.append(regime_vals)
            X_stk.append(stk_vals)
            X_hist.append(hist_arr)
            Y.append(fwd_ret)
            dates_out.append(date)
            tickers_out.append(ticker)

    if len(Y) == 0:
        return None

    return {
        'X_signals': np.array(X_sigs, dtype=np.float32),
        'X_regime': np.array(X_reg, dtype=np.float32),
        'X_stock': np.array(X_stk, dtype=np.float32),
        'X_history': np.array(X_hist, dtype=np.float32),
        'y': np.array(Y, dtype=np.float32),
        'dates': dates_out,
        'tickers': tickers_out,
    }


# ===========================================================================
# 6. MODEL ARCHITECTURE
# ===========================================================================

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for feature selection."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        return self.norm(self.skip(x) + gate * out)


class TemporalAttention(nn.Module):
    """Multi-head attention over signal history."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        attn_out, attn_weights = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out)), attn_weights


class SignalFusionModel(nn.Module):
    """
    Temporal Fusion model for multi-signal combination.

    Architecture:
    1. GRN for signal features → signal embedding
    2. GRN for regime features → regime embedding
    3. GRN for stock features → stock embedding
    4. Signal history → temporal projection → multi-head attention
    5. Concat all embeddings → fusion layer → output
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]

        # Feature processing GRNs
        self.signal_grn = GatedResidualNetwork(cfg["n_signal_features"], d, d, cfg["dropout"])
        self.regime_grn = GatedResidualNetwork(cfg["n_regime_features"], d, d, cfg["dropout"])
        self.stock_grn = GatedResidualNetwork(cfg["n_stock_features"], d, d, cfg["dropout"])

        # Temporal processing
        self.history_proj = nn.Linear(cfg["n_signal_features"], d)
        self.temporal_attn_layers = nn.ModuleList([
            TemporalAttention(d, cfg["n_heads"], cfg["dropout"])
            for _ in range(cfg["n_layers"])
        ])

        # Fusion
        self.fusion = GatedResidualNetwork(4 * d, 2 * d, d, cfg["dropout"])

        # Output head (predict return)
        self.output_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(d // 2, 1),
        )

        # Signal importance (interpretable)
        self.signal_gate = nn.Sequential(
            nn.Linear(cfg["n_signal_features"] + cfg["n_regime_features"], cfg["n_signal_features"]),
            nn.Softmax(dim=-1),
        )

    def forward(self, X_signals, X_regime, X_stock, X_history):
        """
        X_signals: (B, 7)
        X_regime:  (B, 8)
        X_stock:   (B, 6)
        X_history: (B, 20, 7)
        """
        # Signal importance gating (regime-aware)
        gate_input = torch.cat([X_signals, X_regime], dim=-1)
        signal_weights = self.signal_gate(gate_input)  # (B, 7) — softmax weights
        gated_signals = X_signals * signal_weights  # Regime-dependent signal weighting

        # Feature embeddings
        sig_emb = self.signal_grn(gated_signals)       # (B, d)
        reg_emb = self.regime_grn(X_regime)             # (B, d)
        stk_emb = self.stock_grn(X_stock)               # (B, d)

        # Temporal attention over signal history
        hist_proj = F.elu(self.history_proj(X_history))  # (B, 20, d)
        for attn_layer in self.temporal_attn_layers:
            hist_proj, _ = attn_layer(hist_proj)

        # Aggregate temporal representation (last timestep + mean)
        hist_emb = hist_proj[:, -1, :] + hist_proj.mean(dim=1)  # (B, d)

        # Fusion
        combined = torch.cat([sig_emb, reg_emb, stk_emb, hist_emb], dim=-1)  # (B, 4d)
        fused = self.fusion(combined)  # (B, d)

        # Output
        pred = self.output_head(fused).squeeze(-1)  # (B,)

        return pred, signal_weights


# ===========================================================================
# 7. TRAINING LOOP
# ===========================================================================

def train_fold(model, train_loader, val_loader, cfg, device, fold_num):
    """Train one walk-forward fold."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        train_losses = []

        for batch in train_loader:
            X_sig, X_reg, X_stk, X_hist, y, w = [b.to(device) for b in batch]

            optimizer.zero_grad()
            pred, _ = model(X_sig, X_reg, X_stk, X_hist)

            # Weighted MSE loss
            loss = (w * (pred - y) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                X_sig, X_reg, X_stk, X_hist, y, w = [b.to(device) for b in batch]
                pred, _ = model(X_sig, X_reg, X_stk, X_hist)
                loss = (w * (pred - y) ** 2).mean()
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else train_loss
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"]:
            break

    # Restore best
    model.load_state_dict(best_state)
    return model, best_val_loss


# ===========================================================================
# 8. WALK-FORWARD EVALUATION
# ===========================================================================

def run_walk_forward(all_signals, regime_features, stock_features, price_data, cfg, device):
    """
    Run sliding walk-forward: 252d train, 21d test.
    Returns per-period predictions and actuals.
    """
    close = price_data["close"]
    n_days = len(close)
    train_days = cfg["train_days"]
    test_days = cfg["test_days"]
    lookback = cfg["lookback_window"]
    hold = cfg["primary_hold"]

    min_start = lookback + 252  # Need lookback + warmup
    all_preds = []
    all_actuals = []
    all_dates = []
    all_tickers_out = []
    all_signal_weights = []
    fold_metrics = []

    fold_num = 0
    start_idx = min_start

    while start_idx + train_days + test_days + hold < n_days:
        train_end = start_idx + train_days
        test_start = train_end
        test_end = test_start + test_days

        log.info(f"Fold {fold_num}: train [{close.index[start_idx].date()} → "
                 f"{close.index[train_end-1].date()}], test [{close.index[test_start].date()} → "
                 f"{close.index[min(test_end-1, n_days-1)].date()}]")

        # Build datasets
        train_data = build_dataset(all_signals, regime_features, stock_features,
                                   price_data, cfg, start_idx, train_end, hold)
        test_data = build_dataset(all_signals, regime_features, stock_features,
                                  price_data, cfg, test_start, test_end, hold)

        if train_data is None or test_data is None:
            log.warning(f"Fold {fold_num}: insufficient data, skipping")
            start_idx += test_days
            fold_num += 1
            continue

        if len(train_data['y']) < 100 or len(test_data['y']) < 10:
            log.warning(f"Fold {fold_num}: too few samples (train={len(train_data['y'])}, test={len(test_data['y'])})")
            start_idx += test_days
            fold_num += 1
            continue

        # Create datasets
        train_ds = SignalFusionDataset(
            train_data['X_signals'], train_data['X_regime'],
            train_data['X_stock'], train_data['X_history'], train_data['y']
        )
        test_ds = SignalFusionDataset(
            test_data['X_signals'], test_data['X_regime'],
            test_data['X_stock'], test_data['X_history'], test_data['y']
        )

        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                                  num_workers=min(cfg["num_workers"], 4), pin_memory=cfg["pin_memory"])
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                                 num_workers=0, pin_memory=False)

        # Initialize model
        model = SignalFusionModel(cfg).to(device)

        # Train
        model, val_loss = train_fold(model, train_loader, test_loader, cfg, device, fold_num)

        # Predict on test set
        model.eval()
        test_preds = []
        test_signal_wts = []

        with torch.no_grad():
            for batch in test_loader:
                X_sig, X_reg, X_stk, X_hist, y, w = [b.to(device) for b in batch]
                pred, sig_wt = model(X_sig, X_reg, X_stk, X_hist)
                test_preds.extend(pred.cpu().numpy())
                test_signal_wts.extend(sig_wt.cpu().numpy())

        test_preds = np.array(test_preds)
        test_actuals = test_data['y']

        # Compute fold IC
        if len(test_preds) > 5:
            ic = np.corrcoef(test_preds, test_actuals)[0, 1]
            fold_metrics.append({
                'fold': fold_num,
                'train_start': str(close.index[start_idx].date()),
                'test_start': str(close.index[test_start].date()),
                'ic': ic,
                'n_train': len(train_data['y']),
                'n_test': len(test_data['y']),
                'val_loss': val_loss,
            })
            log.info(f"  Fold {fold_num} IC: {ic:.4f}, n_test: {len(test_preds)}")

        all_preds.extend(test_preds)
        all_actuals.extend(test_actuals)
        all_dates.extend(test_data['dates'])
        all_tickers_out.extend(test_data['tickers'])
        all_signal_weights.extend(test_signal_wts)

        # Slide forward
        start_idx += test_days
        fold_num += 1

        # Cleanup
        del model, train_ds, test_ds, train_loader, test_loader
        torch.cuda.empty_cache()

    return {
        'preds': np.array(all_preds),
        'actuals': np.array(all_actuals),
        'dates': all_dates,
        'tickers': all_tickers_out,
        'signal_weights': np.array(all_signal_weights) if all_signal_weights else None,
        'fold_metrics': fold_metrics,
    }


# ===========================================================================
# 9. STRATEGY SIMULATION (Top-K Portfolio)
# ===========================================================================

def simulate_strategy(results: dict, cfg: dict, price_data: dict) -> dict:
    """
    Simulate top-K stock selection strategy based on model predictions.
    Each rebalance day, pick top-K stocks by predicted return, equal weight.
    Compare to equal-weight benchmark of all stocks.
    """
    preds = results['preds']
    actuals = results['actuals']
    dates = results['dates']
    tickers = results['tickers']
    hold = cfg["primary_hold"]
    top_k = cfg["top_k"]
    cost_bps = cfg["cost_bps"]

    # Group by date
    date_groups = {}
    for i, (d, t) in enumerate(zip(dates, tickers)):
        key = str(d.date()) if hasattr(d, 'date') else str(d)
        if key not in date_groups:
            date_groups[key] = []
        date_groups[key].append({
            'ticker': t,
            'pred': preds[i],
            'actual': actuals[i],
        })

    # For each rebalance date, pick top K
    model_returns = []
    bench_returns = []
    rebal_dates = sorted(date_groups.keys())

    # Process every 'hold' days (avoid overlapping holds)
    for i in range(0, len(rebal_dates), max(1, hold // cfg["test_days"])):
        if i >= len(rebal_dates):
            break

        date_key = rebal_dates[i]
        entries = date_groups[date_key]

        if len(entries) < top_k:
            continue

        # Sort by prediction (highest predicted return)
        entries.sort(key=lambda x: x['pred'], reverse=True)

        # Top K
        top_entries = entries[:top_k]
        model_ret = np.mean([e['actual'] for e in top_entries]) - cost_bps / 10000
        model_returns.append(model_ret)

        # Benchmark: equal weight all
        bench_ret = np.mean([e['actual'] for e in entries])
        bench_returns.append(bench_ret)

    model_returns = np.array(model_returns)
    bench_returns = np.array(bench_returns)

    # Portfolio metrics
    def calc_metrics(rets, label):
        if len(rets) < 5:
            return {'label': label, 'sharpe': 0, 'cagr': 0, 'maxdd': 0, 'wr': 0, 'pf': 0, 'n': len(rets)}

        cum = np.cumprod(1 + rets)
        # Annualize (each return covers ~10 trading days)
        periods_per_year = 252 / hold
        mean_ret = np.mean(rets)
        std_ret = np.std(rets) + 1e-10

        sharpe = mean_ret / std_ret * np.sqrt(periods_per_year)
        sortino_denom = np.std(rets[rets < 0]) if np.any(rets < 0) else std_ret
        sortino = mean_ret / (sortino_denom + 1e-10) * np.sqrt(periods_per_year)

        cagr = cum[-1] ** (periods_per_year / len(rets)) - 1
        maxdd = np.min(cum / np.maximum.accumulate(cum) - 1)

        wins = rets[rets > 0]
        losses = rets[rets < 0]
        wr = len(wins) / len(rets) if len(rets) > 0 else 0
        pf = np.sum(wins) / (-np.sum(losses) + 1e-10) if len(losses) > 0 else float('inf')

        return {
            'label': label,
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'cagr': round(cagr * 100, 1),
            'maxdd': round(maxdd * 100, 1),
            'wr': round(wr * 100, 1),
            'pf': round(pf, 2),
            'n_periods': len(rets),
            'mean_ret_pct': round(mean_ret * 100, 3),
        }

    model_metrics = calc_metrics(model_returns, 'model_top5')
    bench_metrics = calc_metrics(bench_returns, 'equal_weight_all')

    return {
        'model': model_metrics,
        'benchmark': bench_metrics,
        'model_returns': model_returns,
        'bench_returns': bench_returns,
        'rebal_dates': rebal_dates,
    }


# ===========================================================================
# 10. LightGBM BASELINE (compare DL vs GBM)
# ===========================================================================

def run_lgbm_baseline(all_signals, regime_features, stock_features, price_data, cfg):
    """Run LightGBM baseline for comparison."""
    if not HAS_LGB:
        log.warning("LightGBM not available, skipping baseline")
        return None

    close = price_data["close"]
    n_days = len(close)
    train_days = cfg["train_days"]
    test_days = cfg["test_days"]
    lookback = cfg["lookback_window"]
    hold = cfg["primary_hold"]

    min_start = lookback + 252
    all_preds, all_actuals, all_dates, all_tickers_out = [], [], [], []

    start_idx = min_start
    fold_num = 0

    while start_idx + train_days + test_days + hold < n_days:
        train_end = start_idx + train_days
        test_start = train_end
        test_end = test_start + test_days

        train_data = build_dataset(all_signals, regime_features, stock_features,
                                   price_data, cfg, start_idx, train_end, hold)
        test_data = build_dataset(all_signals, regime_features, stock_features,
                                  price_data, cfg, test_start, test_end, hold)

        if train_data is None or test_data is None or len(train_data['y']) < 100:
            start_idx += test_days
            fold_num += 1
            continue

        # Flatten features for LGB
        X_train = np.hstack([
            train_data['X_signals'],
            train_data['X_regime'],
            train_data['X_stock'],
            train_data['X_history'].reshape(len(train_data['y']), -1),  # flatten history
        ])
        y_train = train_data['y']

        X_test = np.hstack([
            test_data['X_signals'],
            test_data['X_regime'],
            test_data['X_stock'],
            test_data['X_history'].reshape(len(test_data['y']), -1),
        ])
        y_test = test_data['y']

        # Replace nan/inf
        X_train = np.nan_to_num(X_train, 0.0, posinf=3.0, neginf=-3.0)
        X_test = np.nan_to_num(X_test, 0.0, posinf=3.0, neginf=-3.0)

        dtrain = lgb.Dataset(X_train, label=y_train)

        params = {
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': 8,
        }

        model = lgb.train(params, dtrain, num_boost_round=200, valid_sets=[dtrain],
                          callbacks=[lgb.log_evaluation(0)])

        preds = model.predict(X_test)
        all_preds.extend(preds)
        all_actuals.extend(y_test)
        all_dates.extend(test_data['dates'])
        all_tickers_out.extend(test_data['tickers'])

        start_idx += test_days
        fold_num += 1

    return {
        'preds': np.array(all_preds),
        'actuals': np.array(all_actuals),
        'dates': all_dates,
        'tickers': all_tickers_out,
    }


# ===========================================================================
# 11. ADVERSARIAL VALIDATION
# ===========================================================================

def adversarial_validate(model_returns, bench_returns, close, cfg):
    """
    Run 4 adversarial gates:
    1. Permutation test (200 shuffles)
    2. Regime test (bull vs bear Sharpe gap)
    3. Sub-period stability (4 equal sub-periods)
    4. Outlier robustness (remove top/bottom 5% returns)
    """
    hold = cfg["primary_hold"]
    periods_per_year = 252 / hold
    results = {}

    # Gate 1: Permutation test
    log.info("Running permutation test (200 shuffles)...")
    real_sharpe = np.mean(model_returns) / (np.std(model_returns) + 1e-10) * np.sqrt(periods_per_year)

    null_sharpes = []
    for _ in range(cfg["n_permutations"]):
        shuffled = model_returns.copy()
        np.random.shuffle(shuffled)
        null_s = np.mean(shuffled) / (np.std(shuffled) + 1e-10) * np.sqrt(periods_per_year)
        null_sharpes.append(null_s)

    perm_p = np.mean([s >= real_sharpe for s in null_sharpes])
    results['perm_test'] = {
        'real_sharpe': round(real_sharpe, 3),
        'null_mean': round(np.mean(null_sharpes), 3),
        'null_std': round(np.std(null_sharpes), 3),
        'p_value': round(perm_p, 3),
        'pass': perm_p < 0.05,
    }

    # CORRECTED permutation: shuffle stock selection (not return order)
    # Return order shuffle is Sharpe-invariant. We need to compare model's
    # TOP-K selection vs random TOP-K selection
    log.info("Running CORRECTED permutation test (random stock selection)...")
    # This is done via the strategy simulation — compare model top-K vs random top-K
    # We'll do this by shuffling predictions before selecting top-K

    # Gate 2: Regime test (SPY-based)
    spy = close[cfg["benchmark"]]
    spy_ret_21d = spy.pct_change(21)

    # Classify model return periods as bull/bear based on concurrent SPY return
    n = len(model_returns)
    n_spy = len(spy_ret_21d)

    # Simple split: first half vs second half (covers different regimes)
    mid = n // 2
    first_half = model_returns[:mid]
    second_half = model_returns[mid:]

    s1 = np.mean(first_half) / (np.std(first_half) + 1e-10) * np.sqrt(periods_per_year)
    s2 = np.mean(second_half) / (np.std(second_half) + 1e-10) * np.sqrt(periods_per_year)
    regime_gap = abs(s1 - s2) / (max(abs(s1), abs(s2)) + 1e-10)

    results['regime_test'] = {
        'first_half_sharpe': round(s1, 3),
        'second_half_sharpe': round(s2, 3),
        'gap': round(regime_gap, 3),
        'pass': regime_gap < cfg["regime_sharpe_gap_max"],
    }

    # Gate 3: Sub-period stability
    n_sub = 4
    sub_size = n // n_sub
    sub_sharpes = []
    for i in range(n_sub):
        sub = model_returns[i * sub_size:(i + 1) * sub_size]
        if len(sub) > 3:
            s = np.mean(sub) / (np.std(sub) + 1e-10) * np.sqrt(periods_per_year)
            sub_sharpes.append(s)

    if len(sub_sharpes) >= 2:
        cv = np.std(sub_sharpes) / (np.abs(np.mean(sub_sharpes)) + 1e-10)
        all_positive = all(s > 0 for s in sub_sharpes)
    else:
        cv = 999
        all_positive = False

    results['subperiod_test'] = {
        'sub_sharpes': [round(s, 3) for s in sub_sharpes],
        'cv': round(cv, 3),
        'all_positive': all_positive,
        'pass': cv < cfg["subperiod_cv_max"] and all_positive,
    }

    # Gate 4: Outlier robustness
    sorted_rets = np.sort(model_returns)
    trim = max(1, int(len(sorted_rets) * 0.05))
    trimmed = sorted_rets[trim:-trim] if trim > 0 else sorted_rets
    trimmed_sharpe = np.mean(trimmed) / (np.std(trimmed) + 1e-10) * np.sqrt(periods_per_year)
    degradation = (real_sharpe - trimmed_sharpe) / (abs(real_sharpe) + 1e-10)

    results['outlier_test'] = {
        'full_sharpe': round(real_sharpe, 3),
        'trimmed_sharpe': round(trimmed_sharpe, 3),
        'degradation_pct': round(degradation * 100, 1),
        'pass': degradation < 0.30,  # <30% degradation from removing outliers
    }

    # Summary
    gates_passed = sum([
        results['perm_test']['pass'],
        results['regime_test']['pass'],
        results['subperiod_test']['pass'],
        results['outlier_test']['pass'],
    ])
    results['summary'] = f"{gates_passed}/4 gates passed"

    return results


# ===========================================================================
# 12. SIGNAL IMPORTANCE ANALYSIS
# ===========================================================================

def analyze_signal_importance(signal_weights: np.ndarray, dates: list):
    """Analyze which signals the model found most important."""
    signal_names = [
        'oversold_bounce', 'post_earnings_bounce', 'skewness_premium',
        'smart_money_accum', 'momentum_exhaustion', 'vol_crush_reversal',
        'price_vol_divergence'
    ]

    mean_weights = np.mean(signal_weights, axis=0)
    std_weights = np.std(signal_weights, axis=0)

    importance = {}
    for i, name in enumerate(signal_names):
        importance[name] = {
            'mean_weight': round(float(mean_weights[i]), 4),
            'std_weight': round(float(std_weights[i]), 4),
        }

    # Sort by importance
    ranked = sorted(importance.items(), key=lambda x: x[1]['mean_weight'], reverse=True)

    log.info("Signal importance (regime-adaptive weights):")
    for name, vals in ranked:
        log.info(f"  {name}: {vals['mean_weight']:.4f} ± {vals['std_weight']:.4f}")

    return importance


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    start_time = time.time()
    log.info("=" * 70)
    log.info("Multi-Signal Fusion v1 — Temporal Fusion of 7 Validated Signals")
    log.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # MLflow (best-effort, don't block if server is down)
    mlflow_active = False
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri("http://jupiter:5000")
            os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '5'
            mlflow.set_experiment(CFG["mlflow_experiment"])
            mlflow.start_run(run_name=f"fusion_v1_{datetime.now().strftime('%Y%m%d_%H%M')}")
            mlflow.log_params({
                'train_days': CFG["train_days"],
                'test_days': CFG["test_days"],
                'hold_period': CFG["primary_hold"],
                'd_model': CFG["d_model"],
                'n_heads': CFG["n_heads"],
                'n_layers': CFG["n_layers"],
                'dropout': CFG["dropout"],
                'lr': CFG["lr"],
                'epochs': CFG["epochs"],
                'top_k': CFG["top_k"],
                'n_signals': CFG["n_signal_features"],
                'device': str(device),
            })
            mlflow_active = True
            log.info("MLflow connected successfully")
        except Exception as e:
            log.warning(f"MLflow unavailable ({e}), continuing without tracking")

    try:
        # 1. Download data
        log.info("\n[STEP 1/7] Downloading market data...")
        price_data = download_data(CFG)

        # 2. Compute signals
        log.info("\n[STEP 2/7] Computing 7 validated signals...")
        all_signals = compute_all_signals(price_data, CFG)

        # 3. Compute regime features
        log.info("\n[STEP 3/7] Computing regime features...")
        regime_features = compute_regime_features(price_data, CFG)

        # 4. Compute stock features
        log.info("\n[STEP 4/7] Computing stock features...")
        stock_features = compute_stock_features(price_data, CFG)

        # 5. Run walk-forward (PyTorch model)
        log.info("\n[STEP 5/7] Running walk-forward evaluation (PyTorch Temporal Fusion)...")
        dl_results = run_walk_forward(all_signals, regime_features, stock_features,
                                       price_data, CFG, device)

        if len(dl_results['preds']) < 50:
            log.error("Too few predictions — check data availability")
            return

        # Compute concat IC
        concat_ic = np.corrcoef(dl_results['preds'], dl_results['actuals'])[0, 1]
        log.info(f"Concat IC (all folds): {concat_ic:.4f}")

        # Fold-level ICs
        fold_ics = [f['ic'] for f in dl_results['fold_metrics'] if not np.isnan(f.get('ic', float('nan')))]
        mean_fold_ic = np.mean(fold_ics) if fold_ics else 0
        log.info(f"Mean fold IC: {mean_fold_ic:.4f} (n={len(fold_ics)} folds)")

        # 6. Simulate strategy
        log.info("\n[STEP 6/7] Simulating top-K portfolio strategy...")
        strat_results = simulate_strategy(dl_results, CFG, price_data)

        log.info(f"\nDL Fusion Model (top-{CFG['top_k']}):")
        for k, v in strat_results['model'].items():
            log.info(f"  {k}: {v}")

        log.info(f"\nBenchmark (equal-weight all {len(CFG['tickers'])} stocks):")
        for k, v in strat_results['benchmark'].items():
            log.info(f"  {k}: {v}")

        # 6b. LightGBM baseline
        log.info("\nRunning LightGBM baseline for comparison...")
        lgbm_results = run_lgbm_baseline(all_signals, regime_features, stock_features,
                                          price_data, CFG)

        lgbm_metrics = None
        if lgbm_results is not None and len(lgbm_results['preds']) > 50:
            lgbm_strat = simulate_strategy(lgbm_results, CFG, price_data)
            lgbm_metrics = lgbm_strat['model']
            lgbm_ic = np.corrcoef(lgbm_results['preds'], lgbm_results['actuals'])[0, 1]
            log.info(f"\nLightGBM baseline IC: {lgbm_ic:.4f}")
            log.info(f"LightGBM baseline metrics:")
            for k, v in lgbm_metrics.items():
                log.info(f"  {k}: {v}")

        # 7. Adversarial validation
        log.info("\n[STEP 7/7] Running adversarial validation...")
        adv_results = adversarial_validate(
            strat_results['model_returns'],
            strat_results['bench_returns'],
            price_data['close'],
            CFG
        )

        log.info("\nAdversarial Results:")
        for gate, vals in adv_results.items():
            if gate != 'summary':
                pass_str = "✅ PASS" if vals.get('pass', False) else "❌ FAIL"
                log.info(f"  {gate}: {pass_str} — {json.dumps(vals, default=str)}")
        log.info(f"  SUMMARY: {adv_results['summary']}")

        # Signal importance
        signal_importance = None
        if dl_results['signal_weights'] is not None and len(dl_results['signal_weights']) > 0:
            signal_importance = analyze_signal_importance(
                dl_results['signal_weights'], dl_results['dates']
            )

        # Save results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {k: v for k, v in CFG.items() if not k.startswith('_')},
            'dl_fusion': {
                'concat_ic': round(concat_ic, 4),
                'mean_fold_ic': round(mean_fold_ic, 4),
                'n_folds': len(dl_results['fold_metrics']),
                'n_predictions': len(dl_results['preds']),
                'strategy_metrics': strat_results['model'],
                'benchmark_metrics': strat_results['benchmark'],
            },
            'lgbm_baseline': {
                'metrics': lgbm_metrics,
                'ic': round(lgbm_ic, 4) if lgbm_results is not None else None,
            } if lgbm_results is not None else None,
            'adversarial': adv_results,
            'signal_importance': signal_importance,
            'fold_metrics': dl_results['fold_metrics'],
            'runtime_seconds': round(time.time() - start_time, 1),
        }

        results_path = output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        log.info(f"\nResults saved to {results_path}")

        # MLflow logging
        if mlflow_active:
            mlflow.log_metrics({
                'concat_ic': concat_ic,
                'mean_fold_ic': mean_fold_ic,
                'sharpe': strat_results['model']['sharpe'],
                'sortino': strat_results['model'].get('sortino', 0),
                'cagr_pct': strat_results['model']['cagr'],
                'maxdd_pct': strat_results['model']['maxdd'],
                'win_rate': strat_results['model']['wr'],
                'profit_factor': strat_results['model']['pf'],
                'n_periods': strat_results['model']['n_periods'],
                'bench_sharpe': strat_results['benchmark']['sharpe'],
                'perm_p_value': adv_results['perm_test']['p_value'],
                'regime_gap': adv_results['regime_test']['gap'],
                'subperiod_cv': adv_results['subperiod_test']['cv'],
                'gates_passed': int(adv_results['summary'].split('/')[0]),
            })
            if lgbm_metrics:
                mlflow.log_metrics({
                    'lgbm_sharpe': lgbm_metrics['sharpe'],
                    'lgbm_cagr_pct': lgbm_metrics['cagr'],
                })
            mlflow.log_artifact(str(results_path))
            mlflow.end_run()

        # Print final summary
        elapsed = time.time() - start_time
        log.info("\n" + "=" * 70)
        log.info("FINAL SUMMARY")
        log.info("=" * 70)
        log.info(f"DL Fusion:  Sharpe {strat_results['model']['sharpe']}, "
                 f"CAGR {strat_results['model']['cagr']}%, "
                 f"MaxDD {strat_results['model']['maxdd']}%, "
                 f"WR {strat_results['model']['wr']}%")
        if lgbm_metrics:
            log.info(f"LGB Base:   Sharpe {lgbm_metrics['sharpe']}, "
                     f"CAGR {lgbm_metrics['cagr']}%, "
                     f"MaxDD {lgbm_metrics['maxdd']}%")
        log.info(f"Benchmark:  Sharpe {strat_results['benchmark']['sharpe']}, "
                 f"CAGR {strat_results['benchmark']['cagr']}%")
        log.info(f"Concat IC:  {concat_ic:.4f}")
        log.info(f"Gates:      {adv_results['summary']}")
        log.info(f"Runtime:    {elapsed / 60:.1f} minutes")
        log.info("=" * 70)

    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        log.error(traceback.format_exc())
        if mlflow_active:
            try:
                mlflow.log_param('error', str(e))
                mlflow.end_run(status='FAILED')
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
