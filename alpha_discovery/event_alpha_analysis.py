"""
Event-Triggered Alpha Analysis — Beyond Direction + Magnitude

Instead of continuous direction prediction, this script identifies SPECIFIC
microstructure events and measures their conditional alpha:

1. ABSORPTION: Aggressive volume absorbed without price move → snap-back
2. WALL BOUNCE: Large resting order detected → price deflects off wall
3. SWEEP + ABSORPTION: Post-sweep absorption → reversal signal
4. BOOK THINNING BREAKOUT: One side thins out → breakout
5. TOXICITY BURST: VPIN spike → informed flow → continuation
6. ICEBERG DETECTION: Hidden liquidity → fade the aggressor
7. CROSS-EVENT COMPOSITE: Multiple events → strongest signal

Key difference: These are EVENT-TRIGGERED (wait for setup, then trade)
vs continuous prediction (trade every bar). Event rate = 5-50/day instead of 2000+.

Usage:
    # Full analysis (loads feature cache, computes events, measures alpha)
    python alpha_discovery/event_alpha_analysis.py --n-days 70

    # Quick mode (20 days)
    python alpha_discovery/event_alpha_analysis.py --n-days 20 --quick

    # With existing predictions for comparison
    python alpha_discovery/event_alpha_analysis.py --n-days 70 \
        --load-predictions results/predictions_mfe_path_20260223_145840.npz
"""

import gc
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

from alpha_discovery.mbo_features import get_feature_names, TOTAL_FEATURES
from alpha_discovery.event_detector import (
    EventDetectionPipeline,
    EVENT_FEATURE_NAMES,
    N_EVENT_FEATURES,
)


# ---------------------------------------------------------------------------
# NaN-safe rolling statistics (the ones in mbo_features.py propagate NaN
# through cumsum, making everything NaN if any value is NaN)
# ---------------------------------------------------------------------------

def _nan_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """NaN-safe causal rolling mean using cumsum with NaN fill."""
    x = arr.astype(np.float64).copy()
    mask = np.isnan(x)
    x[mask] = 0.0

    cs = np.cumsum(x)
    cn = np.cumsum(~mask)

    result = np.empty(len(x), dtype=np.float64)

    # Warmup period
    for i in range(min(window, len(x))):
        c = cn[i]
        result[i] = cs[i] / max(c, 1)

    # Main period
    if len(x) > window:
        num = cs[window:] - cs[:-window]
        den = cn[window:] - cn[:-window]
        den = np.maximum(den, 1)
        result[window:] = num / den

    return result.astype(np.float32)


def _nan_rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """NaN-safe causal rolling standard deviation."""
    mean = _nan_rolling_mean(arr, window)
    sq_mean = _nan_rolling_mean(arr.astype(np.float64) ** 2, window)
    var = sq_mean - mean.astype(np.float64) ** 2
    var = np.maximum(var, 0.0)
    return np.sqrt(var).astype(np.float32)

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = RESULTS_DIR / f"event_alpha_{_ts}.log"
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers[:]:
    _root.removeHandler(_h)
_fmt = logging.Formatter('%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
_fh = logging.FileHandler(str(_log_file), mode='w')
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_fh)
_root.addHandler(_sh)
logger = logging.getLogger('event_alpha')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24t
BARS_PER_SEC = 10  # 100ms bars

# Forward return horizons to test (in bars)
HORIZONS = {
    '1s': 10,
    '3s': 30,
    '5s': 50,
    '10s': 100,
    '20s': 200,
    '30s': 300,
}

# ---------------------------------------------------------------------------
# Feature name → index mapping
# ---------------------------------------------------------------------------
_FEATURE_NAMES = get_feature_names()
_FIDX = {name: idx for idx, name in enumerate(_FEATURE_NAMES)}


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_features_and_prices(
    feature_cache_dir: str,
    n_days: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Load feature matrix and mid prices from cache.

    Returns:
        features: (N, 340) float32
        mid_prices: (N,) float64
        day_boundaries: list of start indices (len = n_days + 1)
        dates: list of date strings
    """
    from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner

    scanner = MBOAlphaScanner()
    info = scanner.load_precomputed_features(
        feature_cache_dir=feature_cache_dir,
        n_days=n_days,
        extra_cols=0,
    )

    features = scanner.features[:info['n_snapshots'], :TOTAL_FEATURES]
    mid_prices = scanner.mid_prices[:info['n_snapshots']]
    day_boundaries = scanner.day_boundaries
    dates = info.get('dates_loaded', [f'day_{i}' for i in range(len(day_boundaries) - 1)])

    return features, mid_prices, day_boundaries, dates


# ===========================================================================
# FORWARD RETURN COMPUTATION
# ===========================================================================

def compute_forward_returns(
    mid_prices: np.ndarray,
    day_boundaries: List[int],
    horizon_bars: int,
) -> np.ndarray:
    """
    Compute forward return in ticks for each bar, respecting day boundaries.
    Vectorized: no per-bar loops.

    Returns NaN for bars where the horizon extends beyond the day.
    """
    N = len(mid_prices)
    fwd_ret = np.full(N, np.nan, dtype=np.float32)

    for i in range(len(day_boundaries) - 1):
        start = day_boundaries[i]
        end = day_boundaries[i + 1]

        # Vectorized: all bars in day that have room for horizon
        valid_end = end - horizon_bars
        if valid_end <= start:
            continue
        idx = np.arange(start, valid_end)
        fwd_ret[idx] = (mid_prices[idx + horizon_bars] - mid_prices[idx]) / TICK_SIZE

    return fwd_ret


# ===========================================================================
# EVENT DETECTION — ENHANCED
# ===========================================================================

class EnhancedEventDetector:
    """
    Detects 7 distinct microstructure event types using existing 340 features.

    CRITICAL: At 100ms resolution, most per-bar features are zero (trades happen
    only ~0.5% of bars). Uses ROLLING features and BOOK STATE features instead
    of per-bar trade counts. All detectors are calibrated for actual data ranges.
    """

    def __init__(self, features: np.ndarray, day_boundaries: List[int]):
        self.features = features
        self.N = features.shape[0]
        self.day_boundaries = day_boundaries

        # Pre-extract key features
        self.f = {}
        for name in _FEATURE_NAMES:
            idx = _FIDX[name]
            self.f[name] = features[:, idx]

    def _safe_zscore(self, arr: np.ndarray, window: int) -> np.ndarray:
        """NaN-safe rolling z-score."""
        mu = _nan_rolling_mean(arr, window)
        sigma = _nan_rolling_std(arr, window)
        z = (arr - mu) / np.maximum(sigma, 1e-8)
        z[np.isnan(z)] = 0.0
        return z

    def _percentile_score(self, arr: np.ndarray, window: int = 500) -> np.ndarray:
        """Rolling percentile rank (0-1). More robust than z-score for skewed data."""
        mu = _nan_rolling_mean(arr, window)
        above = (arr > mu).astype(np.float32)
        return _nan_rolling_mean(above, window)

    def detect_absorption(self, threshold: float = 0.3) -> Dict:
        """
        ABSORPTION: OFI strongly negative (sell pressure) but depth_ratio stays high
        (bid is absorbing). Uses ROLLING features that are nonzero at most bars.

        Signal: OFI negative (selling) + depth_ratio still favoring bid → bid absorbing
        """
        # OFI captures net order flow (negative = sell pressure) — 95% nonzero
        ofi_5 = self.f['ofi_5']
        ofi_z = self._safe_zscore(ofi_5, 500)

        # Depth ratio captures which side has more depth — 100% nonzero
        depth_ratio = self.f['depth_ratio_l1']
        dr_z = self._safe_zscore(depth_ratio, 500)

        # Queue depletion: bid being depleted (negative) — 76% nonzero
        depl_bid = self.f['queue_depletion_rate_bid_5']
        depl_ask = self.f['queue_depletion_rate_ask_5']

        # BID ABSORPTION: sell pressure (OFI negative) but bid depth holds
        # OFI strongly negative (sell flow) — use z-score
        sell_pressure = np.clip(-ofi_z - 1.0, 0, 4) / 4.0     # >1 sigma below mean
        buy_pressure = np.clip(ofi_z - 1.0, 0, 4) / 4.0       # >1 sigma above mean

        # Bid still strong despite sell pressure
        bid_holding = np.clip(dr_z + 0.5, 0, 3) / 3.0         # Depth ratio above mean-0.5σ
        ask_holding = np.clip(-dr_z + 0.5, 0, 3) / 3.0

        # Queue not being rapidly depleted (bid refilling)
        bid_refill = np.clip(-depl_bid / 5.0, 0, 1)    # depletion_rate near 0 = holding
        ask_refill = np.clip(-depl_ask / 5.0, 0, 1)

        # Combined: sell pressure + bid holding + bid not depleting
        conf_bid = (sell_pressure * bid_holding) ** 0.5
        conf_ask = (buy_pressure * ask_holding) ** 0.5

        mask_bid = conf_bid > threshold
        mask_ask = conf_ask > threshold

        direction = np.zeros(self.N, dtype=np.int8)
        direction[mask_bid] = +1   # Bid absorbed sell flow → UP
        direction[mask_ask] = -1   # Ask absorbed buy flow → DOWN

        return {
            'mask_bid': mask_bid, 'mask_ask': mask_ask,
            'confidence_bid': conf_bid, 'confidence_ask': conf_ask,
            'direction': direction,
        }

    def detect_wall_bounce(self, threshold: float = 0.3) -> Dict:
        """
        WALL BOUNCE: Abnormally large resting order + pressure approaching.

        Uses wall_frac (100% nonzero, P99=0.23 bid, P99=0.43 ask) and
        queue_depletion_rate (76% nonzero).
        """
        bid_wall = self.f['max_bid_wall_frac']
        ask_wall = self.f['max_ask_wall_frac']

        # Wall abnormality: how far above session mean
        bid_wall_z = self._safe_zscore(bid_wall, 500)
        ask_wall_z = self._safe_zscore(ask_wall, 500)

        # Wall confidence: >1σ above mean
        bid_wall_conf = np.clip(bid_wall_z - 1.0, 0, 4) / 4.0
        ask_wall_conf = np.clip(ask_wall_z - 1.0, 0, 4) / 4.0

        # Depth ratio confirming wall side has more depth
        depth_ratio = self.f['depth_ratio_l1']
        bid_depth_conf = np.clip((depth_ratio - 0.5) / 0.4, 0, 1)
        ask_depth_conf = np.clip((0.5 - depth_ratio) / 0.4, 0, 1)

        # OFI showing pressure against the wall (testing it)
        ofi_z = self._safe_zscore(self.f['ofi_5'], 500)
        # Sell pressure against bid wall:
        sell_testing_bid = np.clip(-ofi_z - 0.5, 0, 3) / 3.0
        buy_testing_ask = np.clip(ofi_z - 0.5, 0, 3) / 3.0

        # Combined: wall present + depth confirms + pressure testing wall
        conf_bid = (bid_wall_conf * bid_depth_conf * sell_testing_bid) ** (1/3)
        conf_ask = (ask_wall_conf * ask_depth_conf * buy_testing_ask) ** (1/3)

        mask_bid = conf_bid > threshold
        mask_ask = conf_ask > threshold

        direction = np.zeros(self.N, dtype=np.int8)
        direction[mask_bid] = +1
        direction[mask_ask] = -1

        return {
            'mask_bid': mask_bid, 'mask_ask': mask_ask,
            'confidence_bid': conf_bid, 'confidence_ask': conf_ask,
            'direction': direction,
        }

    def detect_book_thinning_breakout(self, threshold: float = 0.3) -> Dict:
        """
        BOOK THINNING: Depth asymmetry + low thinning ratio → breakout.

        book_thinning is centered around 1.0 (ratio of L1 depth to rolling mean).
        depth_ratio captures L1 imbalance. Combined: when one side thins AND
        the other side is strong → breakout through the thin side.
        """
        book_thin = self.f['book_thinning']   # ~1.0 centered, 100% nonzero
        depth_ratio = self.f['depth_ratio_l1']  # 0-1, 100% nonzero

        book_thin_z = self._safe_zscore(book_thin, 500)
        depth_z = self._safe_zscore(depth_ratio, 500)

        # Low thinning (book getting thin overall = breakout potential)
        thin_conf = np.clip(-book_thin_z - 0.5, 0, 4) / 4.0

        # Directional: which side is thin?
        # High depth_ratio = bid strong / ask weak → breakout UP (through thin ask)
        up_dir = np.clip(depth_z - 0.5, 0, 3) / 3.0
        # Low depth_ratio = ask strong / bid weak → breakout DOWN
        down_dir = np.clip(-depth_z - 0.5, 0, 3) / 3.0

        # Spread expansion confirms uncertainty
        spread_exp = self.f['spread_expansion']
        spread_z = self._safe_zscore(spread_exp, 500)
        spread_conf = np.clip(spread_z - 0.5, 0, 3) / 3.0

        conf_up = (thin_conf * up_dir * spread_conf) ** (1/3)
        conf_down = (thin_conf * down_dir * spread_conf) ** (1/3)

        mask_up = conf_up > threshold
        mask_down = conf_down > threshold

        direction = np.zeros(self.N, dtype=np.int8)
        direction[mask_up] = +1
        direction[mask_down] = -1

        return {
            'mask_up': mask_up, 'mask_down': mask_down,
            'confidence_up': conf_up, 'confidence_down': conf_down,
            'direction': direction,
        }

    def detect_toxicity_burst(self, threshold: float = 0.3) -> Dict:
        """
        TOXICITY BURST: Elevated VPIN/toxicity + directional OFI → informed flow.

        toxicity_spike is P50=1.0, P99=25.0. Use z-score.
        Direction from OFI (95% nonzero) instead of aggressive_imbalance (1% nonzero).
        """
        tox_spike = self.f['toxicity_spike']   # 59% nonzero, P50=1.0, P99=25.0
        tox_z = self._safe_zscore(tox_spike, 500)

        # Toxicity elevated: >1σ above rolling mean
        tox_conf = np.clip(tox_z - 1.0, 0, 4) / 4.0

        # Direction from OFI (most reliable directional signal that's always nonzero)
        ofi_z = self._safe_zscore(self.f['ofi_5'], 500)
        buy_dir = np.clip(ofi_z - 0.5, 0, 3) / 3.0
        sell_dir = np.clip(-ofi_z - 0.5, 0, 3) / 3.0

        # Size imbalance confirms institutional direction — 100% nonzero
        size_imb = self.f['size_imbalance_5']
        size_buy = np.clip(size_imb / 0.3, 0, 1)
        size_sell = np.clip(-size_imb / 0.3, 0, 1)

        # Combined: toxicity + OFI direction + size confirmation
        conf_up = (tox_conf * buy_dir * size_buy) ** (1/3)
        conf_down = (tox_conf * sell_dir * size_sell) ** (1/3)

        mask_up = conf_up > threshold
        mask_down = conf_down > threshold

        direction = np.zeros(self.N, dtype=np.int8)
        direction[mask_up] = +1
        direction[mask_down] = -1

        return {
            'mask_up': mask_up, 'mask_down': mask_down,
            'confidence_up': conf_up, 'confidence_down': conf_down,
            'direction': direction,
        }

    def detect_sweep_reversal(self, threshold: float = 0.3) -> Dict:
        """
        POST-SWEEP REVERSAL: Price moved but is now stabilizing → mean-revert.

        Uses mid_ret_5 (price change over 0.5s) + current price stability +
        depth rebuilding on the swept side.
        """
        # 5-bar return: a proxy for recent sweep (nonzero when price moved)
        ret_5 = self.f['mid_ret_5']
        ret_5_z = self._safe_zscore(np.abs(ret_5), 500)

        # Large recent move detected
        big_move = np.clip(ret_5_z - 1.0, 0, 4) / 4.0

        # Current bar stability (1-bar return near zero = price stabilizing)
        ret_1 = self.f['mid_ret_1']
        ret_std = _nan_rolling_std(ret_1, 100)
        stability = np.clip(1.0 - np.abs(ret_1) / np.maximum(ret_std, 1e-8), 0, 1)

        # Reversal direction: opposite of the recent move
        # If ret_5 was positive (price went up) → expect reversal DOWN
        ret_5_scale = max(np.nanpercentile(np.abs(ret_5), 99) * 0.5, 1e-8)
        rev_down = np.clip(np.nan_to_num(ret_5) / ret_5_scale, 0, 1)
        rev_up = np.clip(-np.nan_to_num(ret_5) / ret_5_scale, 0, 1)

        # Depth rebuilding on swept side (depth ratio recovering toward 0.5)
        depth = self.f['depth_ratio_l1']
        depth_neutral = 1.0 - 2.0 * np.abs(depth - 0.5)  # 1.0 at 0.5, 0.0 at extremes
        rebuild_conf = np.clip(depth_neutral, 0, 1)

        conf_up = (big_move * stability * rev_up * rebuild_conf) ** 0.25
        conf_down = (big_move * stability * rev_down * rebuild_conf) ** 0.25

        mask_up = conf_up > threshold
        mask_down = conf_down > threshold

        direction = np.zeros(self.N, dtype=np.int8)
        direction[mask_up] = +1
        direction[mask_down] = -1

        return {
            'mask_up': mask_up, 'mask_down': mask_down,
            'confidence_up': conf_up, 'confidence_down': conf_down,
            'direction': direction,
        }

    def detect_iceberg_fade(self, threshold: float = 0.3) -> Dict:
        """
        ICEBERG / HIDDEN LIQUIDITY: L1 concentration abnormally high (replenishment)
        + OFI pushing against it → hidden buyer/seller absorbing.

        bid_L1_conc is 100% nonzero (P50=0.056).
        """
        bid_conc = self.f['bid_L1_conc']
        ask_conc = self.f['ask_L1_conc']

        # L1 concentration abnormally high = someone refreshing orders there
        bid_conc_z = self._safe_zscore(bid_conc, 500)
        ask_conc_z = self._safe_zscore(ask_conc, 500)

        bid_replenish = np.clip(bid_conc_z - 1.0, 0, 4) / 4.0
        ask_replenish = np.clip(ask_conc_z - 1.0, 0, 4) / 4.0

        # OFI pushing against the concentrated side (testing the hidden liquidity)
        ofi_z = self._safe_zscore(self.f['ofi_5'], 500)
        sell_flow = np.clip(-ofi_z - 0.5, 0, 3) / 3.0
        buy_flow = np.clip(ofi_z - 0.5, 0, 3) / 3.0

        # Bid iceberg: bid concentrated + sell flow testing it → absorbing → UP
        conf_up = (bid_replenish * sell_flow) ** 0.5
        conf_down = (ask_replenish * buy_flow) ** 0.5

        mask_up = conf_up > threshold
        mask_down = conf_down > threshold

        direction = np.zeros(self.N, dtype=np.int8)
        direction[mask_up] = +1
        direction[mask_down] = -1

        return {
            'mask_up': mask_up, 'mask_down': mask_down,
            'confidence_up': conf_up, 'confidence_down': conf_down,
            'direction': direction,
        }

    def detect_institutional_pressure(self, threshold: float = 0.3) -> Dict:
        """
        INSTITUTIONAL PRESSURE: Size imbalance + OFI alignment + flow persistence.

        Uses size_imbalance_5 (100% nonzero) + ofi_5 (95% nonzero) +
        burst_same_dir_20 (100% nonzero, P50=94.5 — high values!).
        """
        size_imb = self.f['size_imbalance_5']   # 100% nonzero, centered near 0
        ofi_z = self._safe_zscore(self.f['ofi_5'], 500)

        # Size imbalance: institutional buying (positive) or selling (negative)
        size_z = self._safe_zscore(size_imb, 500)
        buy_size = np.clip(size_z - 0.5, 0, 3) / 3.0
        sell_size = np.clip(-size_z - 0.5, 0, 3) / 3.0

        # OFI alignment
        buy_flow = np.clip(ofi_z - 0.5, 0, 3) / 3.0
        sell_flow = np.clip(-ofi_z - 0.5, 0, 3) / 3.0

        # Depth ratio confirms pressure direction
        depth_z = self._safe_zscore(self.f['depth_ratio_l1'], 500)
        buy_depth = np.clip(depth_z - 0.3, 0, 3) / 3.0
        sell_depth = np.clip(-depth_z - 0.3, 0, 3) / 3.0

        # Combined: size + flow + depth all agree
        conf_up = (buy_size * buy_flow * buy_depth) ** (1/3)
        conf_down = (sell_size * sell_flow * sell_depth) ** (1/3)

        mask_up = conf_up > threshold
        mask_down = conf_down > threshold

        direction = np.zeros(self.N, dtype=np.int8)
        direction[mask_up] = +1
        direction[mask_down] = -1

        return {
            'mask_up': mask_up, 'mask_down': mask_down,
            'confidence_up': conf_up, 'confidence_down': conf_down,
            'direction': direction,
        }


# ===========================================================================
# ALPHA MEASUREMENT
# ===========================================================================

def measure_event_alpha(
    event_result: Dict,
    fwd_returns: Dict[str, np.ndarray],
    mid_prices: np.ndarray,
    day_boundaries: List[int],
    event_name: str,
) -> Dict:
    """
    Measure the alpha of an event-triggered strategy.

    Computes:
    - Event frequency (per day)
    - Forward return distribution conditioned on event direction
    - IC of confidence score vs forward return
    - Win rate, profit factor, avg gain/loss
    - MFE (maximum favorable excursion)
    """
    direction = event_result['direction']
    n_days = len(day_boundaries) - 1

    # Get confidence arrays (different naming convention per event)
    conf_keys = [k for k in event_result if k.startswith('confidence')]
    conf_combined = np.zeros(len(direction), dtype=np.float32)
    for k in conf_keys:
        conf_combined = np.maximum(conf_combined, event_result[k])

    # Active bars: where we have a directional signal
    active = direction != 0
    n_active = int(active.sum())

    if n_active == 0:
        return {'event': event_name, 'n_events': 0, 'skip': True}

    result = {
        'event': event_name,
        'n_events': n_active,
        'events_per_day': n_active / max(n_days, 1),
        'pct_active': n_active / len(direction) * 100,
    }

    # Per-horizon analysis
    for hz_name, fwd_ret in fwd_returns.items():
        valid = active & np.isfinite(fwd_ret)
        n_valid = int(valid.sum())
        if n_valid < 50:
            result[f'{hz_name}_skip'] = True
            continue

        # Signed return in predicted direction
        signed_ret = fwd_ret[valid] * direction[valid]

        # Basic stats
        mean_ret = float(np.mean(signed_ret))
        median_ret = float(np.median(signed_ret))
        std_ret = float(np.std(signed_ret))

        # Win rate (direction correct = positive signed return)
        winners = signed_ret > 0
        win_rate = float(winners.mean())

        # Average gain / average loss
        avg_gain = float(np.mean(signed_ret[winners])) if winners.any() else 0.0
        avg_loss = float(np.mean(np.abs(signed_ret[~winners]))) if (~winners).any() else 0.0
        profit_factor = avg_gain / max(avg_loss, 1e-8) if avg_loss > 0 else float('inf')

        # IC: confidence vs signed return
        conf_valid = conf_combined[valid]
        if np.std(conf_valid) > 1e-8 and np.std(signed_ret) > 1e-8:
            ic, _ = spearmanr(conf_valid, signed_ret)
            ic = float(ic) if np.isfinite(ic) else 0.0
        else:
            ic = 0.0

        # Unsigned IC: confidence vs absolute return (predicts magnitude?)
        if np.std(np.abs(fwd_ret[valid])) > 1e-8:
            mag_ic, _ = spearmanr(conf_valid, np.abs(fwd_ret[valid]))
            mag_ic = float(mag_ic) if np.isfinite(mag_ic) else 0.0
        else:
            mag_ic = 0.0

        # Cost-adjusted PnL
        cost_ticks_mkt = 0.376  # HC #231(A): commission only — no spread crossing cost
        cost_ticks_lmt = 0.376  # commission only ($4.70 RT, HC #52)   # Limit in + limit out
        pnl_mkt = mean_ret - cost_ticks_mkt
        pnl_lmt = mean_ret - cost_ticks_lmt

        # Per-day PnL
        events_per_day = n_valid / max(n_days, 1)
        daily_pnl_mkt = pnl_mkt * events_per_day * TICK_VALUE
        daily_pnl_lmt = pnl_lmt * events_per_day * TICK_VALUE

        result[hz_name] = {
            'n_valid': n_valid,
            'mean_ret_ticks': round(mean_ret, 4),
            'median_ret_ticks': round(median_ret, 4),
            'std_ret_ticks': round(std_ret, 4),
            'win_rate': round(win_rate, 4),
            'avg_gain': round(avg_gain, 4),
            'avg_loss': round(avg_loss, 4),
            'profit_factor': round(profit_factor, 3),
            'direction_ic': round(ic, 4),
            'magnitude_ic': round(mag_ic, 4),
            'pnl_mkt_ticks': round(pnl_mkt, 4),
            'pnl_lmt_ticks': round(pnl_lmt, 4),
            'daily_pnl_mkt': round(daily_pnl_mkt, 2),
            'daily_pnl_lmt': round(daily_pnl_lmt, 2),
        }

    # Per-day stability
    daily_counts = []
    for d in range(n_days):
        start = day_boundaries[d]
        end = day_boundaries[d + 1]
        daily_counts.append(int(active[start:end].sum()))

    result['daily_event_counts'] = {
        'mean': round(np.mean(daily_counts), 1),
        'std': round(np.std(daily_counts), 1),
        'min': int(np.min(daily_counts)),
        'max': int(np.max(daily_counts)),
        'zero_days': int(sum(1 for c in daily_counts if c == 0)),
    }

    return result


# ===========================================================================
# COMPOSITE EVENTS — MULTIPLE EVENTS FIRING TOGETHER
# ===========================================================================

def build_composite_events(
    event_results: Dict[str, Dict],
    n_bars: int,
) -> Dict[str, Dict]:
    """
    Build composite event signals from individual event detections.

    Composite strategies:
    1. ANY_EVENT: Any single event fires (broadest, most events)
    2. DUAL_CONFIRM: 2+ events agree on direction (stronger signal)
    3. ABSORPTION_PLUS: Absorption + (wall OR iceberg) — strongest mean-reversion
    4. MOMENTUM_COMBO: Toxicity + institutional + sweep_continuation — strongest momentum
    """
    composites = {}

    # Collect all directions
    all_dirs = {}
    for name, res in event_results.items():
        all_dirs[name] = res['direction']

    # Stack into matrix (N, n_events)
    dir_matrix = np.column_stack([d for d in all_dirs.values()])

    # 1. ANY_EVENT: any direction != 0, use majority vote for direction
    any_active = np.any(dir_matrix != 0, axis=1)
    vote = np.sum(dir_matrix, axis=1)
    any_dir = np.sign(vote).astype(np.int8)
    any_dir[~any_active] = 0

    composites['ANY_EVENT'] = {
        'direction': any_dir,
        'confidence_combined': np.clip(np.abs(vote) / dir_matrix.shape[1], 0, 1),
    }

    # 2. DUAL_CONFIRM: 2+ events agree on same direction
    n_bullish = np.sum(dir_matrix > 0, axis=1)
    n_bearish = np.sum(dir_matrix < 0, axis=1)
    dual_dir = np.zeros(n_bars, dtype=np.int8)
    dual_dir[n_bullish >= 2] = +1
    dual_dir[n_bearish >= 2] = -1
    # If both >= 2, cancel out (conflicting)
    dual_dir[(n_bullish >= 2) & (n_bearish >= 2)] = 0

    composites['DUAL_CONFIRM'] = {
        'direction': dual_dir,
        'confidence_combined': np.clip(np.maximum(n_bullish, n_bearish) / dir_matrix.shape[1], 0, 1),
    }

    # 3. ABSORPTION_PLUS: absorption + wall_bounce or iceberg agree
    mean_reversion_names = ['absorption', 'wall_bounce', 'iceberg_fade']
    mr_dirs = [all_dirs[n] for n in mean_reversion_names if n in all_dirs]
    if len(mr_dirs) >= 2:
        mr_matrix = np.column_stack(mr_dirs)
        mr_bull = np.sum(mr_matrix > 0, axis=1)
        mr_bear = np.sum(mr_matrix < 0, axis=1)
        absorb_plus_dir = np.zeros(n_bars, dtype=np.int8)
        # Need absorption + at least one other
        absorb_active = all_dirs.get('absorption', np.zeros(n_bars)) != 0
        absorb_plus_dir[(mr_bull >= 2) & absorb_active] = +1
        absorb_plus_dir[(mr_bear >= 2) & absorb_active] = -1

        composites['ABSORPTION_PLUS'] = {
            'direction': absorb_plus_dir,
            'confidence_combined': np.clip(np.maximum(mr_bull, mr_bear) / len(mr_dirs), 0, 1),
        }

    # 4. MOMENTUM_COMBO: toxicity + institutional agree
    mom_names = ['toxicity_burst', 'institutional_pressure']
    mom_dirs = [all_dirs[n] for n in mom_names if n in all_dirs]
    if len(mom_dirs) >= 2:
        mom_matrix = np.column_stack(mom_dirs)
        mom_bull = np.sum(mom_matrix > 0, axis=1)
        mom_bear = np.sum(mom_matrix < 0, axis=1)
        mom_dir = np.zeros(n_bars, dtype=np.int8)
        mom_dir[mom_bull >= 2] = +1
        mom_dir[mom_bear >= 2] = -1

        composites['MOMENTUM_COMBO'] = {
            'direction': mom_dir,
            'confidence_combined': np.clip(np.maximum(mom_bull, mom_bear) / len(mom_dirs), 0, 1),
        }

    return composites


# ===========================================================================
# IS/OOS SPLIT ANALYSIS
# ===========================================================================

def split_analysis(
    result: Dict,
    direction: np.ndarray,
    fwd_returns: Dict[str, np.ndarray],
    day_boundaries: List[int],
    oos_start_day: int,
) -> Dict:
    """Add IS/OOS split statistics to a result dict."""
    n_days = len(day_boundaries) - 1
    oos_bar_start = day_boundaries[min(oos_start_day, n_days)]

    is_mask = np.zeros(len(direction), dtype=bool)
    is_mask[:oos_bar_start] = True
    oos_mask = np.zeros(len(direction), dtype=bool)
    oos_mask[oos_bar_start:] = True

    for hz_name, fwd_ret in fwd_returns.items():
        if f'{hz_name}_skip' in result:
            continue

        for split_name, split_mask in [('IS', is_mask), ('OOS', oos_mask)]:
            active = (direction != 0) & split_mask & np.isfinite(fwd_ret)
            n = int(active.sum())
            if n < 30:
                result[f'{hz_name}_{split_name}'] = {'n': n, 'skip': True}
                continue

            signed = fwd_ret[active] * direction[active]
            winners = signed > 0
            avg_g = float(np.mean(signed[winners])) if winners.any() else 0.0
            avg_l = float(np.mean(np.abs(signed[~winners]))) if (~winners).any() else 0.0
            pf = avg_g / max(avg_l, 1e-8)

            split_days = oos_start_day if split_name == 'IS' else (n_days - oos_start_day)

            result[f'{hz_name}_{split_name}'] = {
                'n': n,
                'mean_ret': round(float(np.mean(signed)), 4),
                'win_rate': round(float(winners.mean()), 4),
                'profit_factor': round(pf, 3),
                'per_day': round(n / max(split_days, 1), 1),
                'pnl_lmt': round(float(np.mean(signed)) - 0.248, 4),
            }

    return result


# ===========================================================================
# MAIN ANALYSIS
# ===========================================================================

def run_analysis(args):
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("EVENT-TRIGGERED ALPHA ANALYSIS")
    logger.info(f"  n_days:     {args.n_days}")
    logger.info(f"  threshold:  {args.threshold}")
    logger.info(f"  cache_dir:  {args.feature_cache}")
    logger.info(f"  quick mode: {args.quick}")
    logger.info("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    logger.info("\n[1/5] Loading features and mid prices...")
    features, mid_prices, day_boundaries, dates = load_features_and_prices(
        feature_cache_dir=args.feature_cache,
        n_days=args.n_days,
    )
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    logger.info(f"  Loaded {n_days} days, {N:,} bars, {features.shape[1]} features")

    # -----------------------------------------------------------------------
    # 2. Compute forward returns at multiple horizons
    # -----------------------------------------------------------------------
    logger.info("\n[2/5] Computing forward returns...")
    horizons = HORIZONS if not args.quick else {
        '5s': 50, '10s': 100, '20s': 200,
    }

    fwd_returns = {}
    for hz_name, hz_bars in horizons.items():
        t1 = time.time()
        fwd_returns[hz_name] = compute_forward_returns(mid_prices, day_boundaries, hz_bars)
        valid = np.isfinite(fwd_returns[hz_name]).sum()
        logger.info(f"  {hz_name} ({hz_bars} bars): {valid:,} valid bars ({time.time()-t1:.1f}s)")

    # -----------------------------------------------------------------------
    # 3. Detect events
    # -----------------------------------------------------------------------
    logger.info("\n[3/5] Detecting microstructure events...")
    detector = EnhancedEventDetector(features, day_boundaries)

    events = {}
    event_methods = {
        'absorption': detector.detect_absorption,
        'wall_bounce': detector.detect_wall_bounce,
        'book_thinning': detector.detect_book_thinning_breakout,
        'toxicity_burst': detector.detect_toxicity_burst,
        'sweep_reversal': detector.detect_sweep_reversal,
        'iceberg_fade': detector.detect_iceberg_fade,
        'institutional_pressure': detector.detect_institutional_pressure,
    }

    for name, method in event_methods.items():
        t1 = time.time()
        events[name] = method(threshold=args.threshold)
        n_active = int((events[name]['direction'] != 0).sum())
        n_bull = int((events[name]['direction'] > 0).sum())
        n_bear = int((events[name]['direction'] < 0).sum())

        # Diagnostic: confidence distribution
        conf_keys = [k for k in events[name] if k.startswith('confidence')]
        conf_max = np.zeros(N, dtype=np.float32)
        for k in conf_keys:
            conf_max = np.maximum(conf_max, events[name][k])
        valid_conf = conf_max[conf_max > 0]
        if len(valid_conf) > 0:
            pcts = np.percentile(valid_conf, [50, 90, 95, 99, 99.9])
            logger.info(f"  {name:25s}: {n_active:>8,} events ({n_active/n_days:>6.1f}/day) "
                       f"[+{n_bull:,} / -{n_bear:,}] ({time.time()-t1:.1f}s)")
            logger.info(f"    conf>0: {len(valid_conf):,}  "
                       f"P50={pcts[0]:.4f} P90={pcts[1]:.4f} P95={pcts[2]:.4f} "
                       f"P99={pcts[3]:.4f} P99.9={pcts[4]:.4f}")
        else:
            logger.info(f"  {name:25s}: {n_active:>8,} events -- ALL CONF ZERO ({time.time()-t1:.1f}s)")

    # Build composites
    composites = build_composite_events(events, N)
    for name, comp in composites.items():
        n_active = int((comp['direction'] != 0).sum())
        n_bull = int((comp['direction'] > 0).sum())
        n_bear = int((comp['direction'] < 0).sum())
        logger.info(f"  {name:25s}: {n_active:>8,} events ({n_active/n_days:>6.1f}/day) "
                    f"[+{n_bull:,} / -{n_bear:,}]")

    # -----------------------------------------------------------------------
    # 4. Measure alpha for each event type
    # -----------------------------------------------------------------------
    logger.info("\n[4/5] Measuring event alpha...")
    all_results = {}

    # Individual events
    for name, event_res in events.items():
        result = measure_event_alpha(event_res, fwd_returns, mid_prices, day_boundaries, name)

        # Add IS/OOS split (70/30)
        oos_day = int(n_days * 0.7)
        if not result.get('skip'):
            split_analysis(result, event_res['direction'], fwd_returns, day_boundaries, oos_day)

        all_results[name] = result

        # Log summary
        if result.get('skip'):
            logger.info(f"\n  {name}: NO EVENTS")
            continue

        logger.info(f"\n  === {name.upper()} ===")
        logger.info(f"  Events: {result['n_events']:,} ({result['events_per_day']:.1f}/day, "
                    f"{result['pct_active']:.2f}% of bars)")

        for hz_name in horizons:
            hz_data = result.get(hz_name)
            if not hz_data or hz_data.get('skip'):
                continue
            logger.info(f"    {hz_name:>5s}: mean={hz_data['mean_ret_ticks']:+.4f}t  "
                       f"WR={hz_data['win_rate']:.1%}  PF={hz_data['profit_factor']:.2f}  "
                       f"IC={hz_data['direction_ic']:+.4f}  "
                       f"MKT=${hz_data['daily_pnl_mkt']:+.0f}/d  "
                       f"LMT=${hz_data['daily_pnl_lmt']:+.0f}/d")

            # IS/OOS if available
            for split in ['IS', 'OOS']:
                split_data = result.get(f'{hz_name}_{split}')
                if split_data and not split_data.get('skip'):
                    logger.info(f"      {split:>3s}: mean={split_data['mean_ret']:+.4f}t  "
                               f"WR={split_data['win_rate']:.1%}  "
                               f"PF={split_data['profit_factor']:.2f}  "
                               f"n={split_data['n']:,}  "
                               f"({split_data['per_day']:.1f}/day)")

    # Composite events
    for name, comp in composites.items():
        result = measure_event_alpha(
            comp, fwd_returns, mid_prices, day_boundaries, name,
        )
        oos_day = int(n_days * 0.7)
        if not result.get('skip'):
            split_analysis(result, comp['direction'], fwd_returns, day_boundaries, oos_day)
        all_results[name] = result

        if result.get('skip'):
            logger.info(f"\n  {name}: NO EVENTS")
            continue

        logger.info(f"\n  === COMPOSITE: {name} ===")
        logger.info(f"  Events: {result['n_events']:,} ({result['events_per_day']:.1f}/day)")

        for hz_name in horizons:
            hz_data = result.get(hz_name)
            if not hz_data or hz_data.get('skip'):
                continue
            logger.info(f"    {hz_name:>5s}: mean={hz_data['mean_ret_ticks']:+.4f}t  "
                       f"WR={hz_data['win_rate']:.1%}  PF={hz_data['profit_factor']:.2f}  "
                       f"IC={hz_data['direction_ic']:+.4f}  "
                       f"MKT=${hz_data['daily_pnl_mkt']:+.0f}/d  "
                       f"LMT=${hz_data['daily_pnl_lmt']:+.0f}/d")

            for split in ['IS', 'OOS']:
                split_data = result.get(f'{hz_name}_{split}')
                if split_data and not split_data.get('skip'):
                    logger.info(f"      {split:>3s}: mean={split_data['mean_ret']:+.4f}t  "
                               f"WR={split_data['win_rate']:.1%}  "
                               f"PF={split_data['profit_factor']:.2f}  "
                               f"n={split_data['n']:,}")

    # -----------------------------------------------------------------------
    # 5. Summary comparison table
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("[5/5] SUMMARY COMPARISON — 10s Horizon")
    logger.info("=" * 70)

    hz_key = '10s'
    header = f"{'Event':30s} {'Events/d':>8s} {'Mean(t)':>8s} {'WR':>6s} {'PF':>6s} {'IC':>7s} {'MKT$/d':>8s} {'LMT$/d':>8s}"
    logger.info(header)
    logger.info("-" * len(header))

    for name, result in all_results.items():
        if result.get('skip'):
            continue
        hz_data = result.get(hz_key)
        if not hz_data or hz_data.get('skip'):
            continue
        logger.info(
            f"{name:30s} {result['events_per_day']:>8.1f} "
            f"{hz_data['mean_ret_ticks']:>+8.4f} "
            f"{hz_data['win_rate']:>6.1%} "
            f"{hz_data['profit_factor']:>6.2f} "
            f"{hz_data['direction_ic']:>+7.4f} "
            f"{hz_data['daily_pnl_mkt']:>+8.0f} "
            f"{hz_data['daily_pnl_lmt']:>+8.0f}"
        )

    # OOS-specific table
    logger.info("\n" + "=" * 70)
    logger.info("OOS ONLY — 10s Horizon")
    logger.info("=" * 70)
    header_oos = f"{'Event':30s} {'N':>7s} {'Mean(t)':>8s} {'WR':>6s} {'PF':>6s} {'/day':>6s} {'LMT(t)':>8s}"
    logger.info(header_oos)
    logger.info("-" * len(header_oos))

    for name, result in all_results.items():
        if result.get('skip'):
            continue
        oos_data = result.get(f'{hz_key}_OOS')
        if not oos_data or oos_data.get('skip'):
            continue
        logger.info(
            f"{name:30s} {oos_data['n']:>7,} "
            f"{oos_data['mean_ret']:>+8.4f} "
            f"{oos_data['win_rate']:>6.1%} "
            f"{oos_data['profit_factor']:>6.2f} "
            f"{oos_data['per_day']:>6.1f} "
            f"{oos_data['pnl_lmt']:>+8.4f}"
        )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    result_file = RESULTS_DIR / f"event_alpha_{_ts}.json"

    # Clean for JSON serialization
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(result_file, 'w') as f:
        json.dump(clean(all_results), f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"\n{'='*70}")
    logger.info(f"Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"Log: {_log_file}")
    logger.info(f"Results: {result_file}")
    logger.info(f"{'='*70}")

    return all_results


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Event-Triggered Alpha Analysis')
    parser.add_argument('--feature-cache', type=str,
                       default=str(LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'),
                       help='Feature cache directory')
    parser.add_argument('--n-days', type=int, default=70,
                       help='Number of days to analyze')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Event detection confidence threshold (0-1)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer horizons')
    parser.add_argument('--load-predictions', type=str, default=None,
                       help='Load existing predictions NPZ for comparison')

    args = parser.parse_args()
    run_analysis(args)


if __name__ == '__main__':
    main()
