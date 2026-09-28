"""
Multi-Channel Alpha Scanner — Discover independent alpha sources from MBO data.

The core problem: a single LightGBM model collapses onto L1 order book imbalance
(a simple linear signal) and never explores other alpha sources. Ridge regression
captures 97% of the IC, meaning tree-based nonlinear patterns are not being found.

Solution: Train independent models on DISJOINT feature subsets ("channels"),
where each channel captures a different alpha source. No channel sees the
dominant L1 features, forcing the model to discover weaker but independent signals.

Channels:
- Ch1 (L1 Imbalance): L1 dynamics + quote stability (~18 features)
- Ch2 (Event-Driven): Discrete events + trade/sweep + distribution (~35 features)
- Ch3 (Toxicity): VPIN, cancel asym, HFT, repositioning, order sizing (~30 features)
- Ch4 (Deep Book): L2-L5 depth + book shape (spatial structure) (~40 features)
- Ch5 (Regime/Vol): Volatility regime, realized vol, trade timing (~23 features)
- Ch_res (Residual): ALL non-L1 features trained on L1-residualized target

Key mechanism: L1 CANNOT dominate because Ch2-Ch5 NEVER see depth_ratio_l1 or OFI.
The residual channel explicitly asks "what alpha exists AFTER removing L1?"
"""

import gc
import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, ttest_1samp

logger = logging.getLogger("multi_channel_alpha")


# ============================================================================
# CHANNEL FEATURE ASSIGNMENTS (STRICTLY DISJOINT)
# ============================================================================

# Ch1: L1 Imbalance — the known dominant linear signal
CH1_L1_IMBALANCE = [
    'depth_ratio_l1',
    'vol_imbalance',
    'microprice',
    'pressure_imbalance',
    'ofi_5', 'ofi_20', 'ofi_50',
    'trade_imbalance',
    'aggressive_imbalance',
    'bid_L1_orders', 'ask_L1_orders',
    'bid_L1_conc', 'ask_L1_conc',
    # Phase 3: L1 quote dynamics (stability of the top of book)
    'l1_bid_change_rate', 'l1_ask_change_rate',
    'l1_change_asymmetry',
    'l1_depletion_log', 'depletion_asymmetry',
    # Group N rolling: L1 instability persistence
    'l1_instability_5', 'l1_instability_20',
    'depletion_momentum_5', 'depletion_dir_20',
    # Group O z-scores: L1 flow anomalies
    'ofi_5_zscore_50', 'ofi_5_zscore_500', 'ofi_5_diverge',
    'aggr_imb_zscore_50', 'aggr_imb_zscore_500', 'aggr_imb_diverge',
    'depth_ratio_l1_zscore_50', 'depth_ratio_l1_zscore_500', 'depth_ratio_l1_diverge',
    # Group P magnitude: L1-related magnitude features
    'book_thinning', 'flow_acceleration', 'cross_flow_agreement',
    # Group Q: Order flow sequences (L1-related)
    'buy_run_max_5', 'sell_run_max_5',       # Run-length of buy/sell dominance
    'run_direction_20',                       # Net flow direction
    'flow_persistence_20',                    # Flow autocorrelation
    'momentum_exhaustion',                    # Slowing flow in trend direction
    # Group R: L1 staleness
    'bars_since_l1_change',                   # Time since L1 price moved
    'price_level_duration',                   # Consecutive bars at same mid
    # Group S: L1 interactions
    'imb_spread_interaction',                 # Imbalance × spread width
]

# Ch2: Event-Driven — discrete events + trade/sweep structure
# 14 event detector features + 5 existing + 13 Phase 2 event features
CH2_EVENT_DRIVEN = [
    # 14 event detector outputs (from event_detector.py)
    'sweep_event_buy', 'sweep_event_sell',
    'iceberg_event_buy', 'iceberg_event_sell',
    'cancel_storm_bid', 'cancel_storm_ask',
    'absorption_event_bid', 'absorption_event_ask',
    'spoof_event_bid', 'spoof_event_ask',
    'regime_break_up', 'regime_break_down',
    'event_intensity', 'event_diversity',
    # Related existing features
    'large_trade_rev',
    'sweep_count',
    'aggressive_burst',
    'momentum_ignition',
    'book_flip',
    # Phase 2: Trade size distribution (unusual trade patterns)
    'large_trade_ratio', 'small_trade_ratio',
    'trade_size_p75_log', 'trade_heterogeneity',
    # Phase 2: Sweep signatures (consecutive trade streaks)
    'consec_buy_log', 'consec_sell_log',
    'sweep_asymmetry', 'sweep_intensity',
    # Phase 2: Multi-level trades (level-sweeping events)
    'multi_level_ratio', 'multi_level_log',
    # Phase 2: Event × context interactions
    'large_sweep_combo', 'cancel_l1_trade_press', 'modify_chain_sweep',
    # Phase 3: Trade distribution (event character)
    'trade_at_bid_frac',
    'trade_size_cv', 'trade_size_skew_raw',
    # Group N rolling: Selling pressure persistence
    'sell_pressure_5', 'sell_pressure_20',
    # Group O z-scores: Trade flow anomalies
    'trade_imb_zscore_50', 'trade_imb_zscore_500', 'trade_imb_diverge',
    # Group P magnitude: Event-driven magnitude
    'trade_rate_spike', 'institutional_signal',
    # Group Q: Order flow sequences (event-related)
    'flow_reversal_5',                        # Buy↔sell transitions (regime changes)
    'aggressive_streak_5',                     # Consecutive aggressive trade bars
    'buy_intensity_ratio', 'sell_intensity_ratio',  # Short-term volume surges
    'flow_regime_zscore',                      # Unusual reversal rate
    # Group R: Event staleness
    'bars_since_trade',                        # Time since last trade
    'bars_since_big_trade',                    # Time since last big trade
    'activity_halflife',                       # Exponential decay of activity
]

# Ch3: Toxicity / Informed Flow
CH3_TOXICITY = [
    'vpin_20', 'vpin_50', 'vpin_100',
    'cancel_asym_5', 'cancel_asym_20',
    'fleeting_ratio_5', 'fleeting_ratio_20',
    'kyle_lambda_20', 'kyle_lambda_50',
    'price_impact', 'adverse_sel', 'toxicity_score',
    'modify_rate_5', 'modify_rate_20',
    'lifetime_mean_5', 'lifetime_mean_20',
    'mean_order_lifetime', 'fleeting_ratio',
    'cancel_side_imbalance',
    'cancel_asym_vol',
    # Phase 2: Cancel L1 structure (quote pulling = informed flow signal)
    'cancel_l1_ratio', 'cancel_l1_vol_frac',
    'add_l1_ratio', 'l1_net_activity',
    # Phase 2: Modify chains (HFT price discovery = informed participant)
    'modify_chain_ratio', 'max_chain_log', 'chain_per_order',
    # Phase 3: Repositioning + order size (informed flow character)
    'reposition_rate',
    'mean_add_size_log', 'max_add_size_norm',
    # Group N rolling: Informed flow persistence
    'reposition_intensity_5',
    'size_dispersion_20',
    'add_size_trend_20',
    'institutional_flow_5',
    # Group O z-scores: Toxicity anomalies
    'vpin_zscore_50', 'vpin_zscore_500', 'vpin_diverge',
    'cancel_asym_zscore_50', 'cancel_asym_zscore_500', 'cancel_asym_diverge',
    # Group P magnitude: Toxicity spike
    'toxicity_spike',
    # Group S: Toxicity interactions
    'toxicity_imb_combo',                      # VPIN × imbalance composite
    'flow_vol_interaction',                    # Abnormal flow × abnormal vol
]

# Ch4: Deep Book Structure (L2-L5)
CH4_DEEP_BOOK = [
    'depth_ratio_l3', 'depth_ratio_l5',
    'bid_slope', 'ask_slope',
    'depth_concentration',
    'bid_L2_orders', 'bid_L2_conc',
    'bid_L3_orders', 'bid_L3_conc',
    'bid_L4_orders', 'bid_L4_conc',
    'bid_L5_orders', 'bid_L5_conc',
    'ask_L2_orders', 'ask_L2_conc',
    'ask_L3_orders', 'ask_L3_conc',
    'ask_L4_orders', 'ask_L4_conc',
    'ask_L5_orders', 'ask_L5_conc',
    'weighted_book_imb',
    'book_refresh',
    'hidden_liq',
    # Phase 2: Deep cancel volume (cancel activity away from L1)
    'cancel_deep_vol_frac',
    # Phase 3: Book shape (spatial structure of deep book)
    'bid_depth_skew', 'ask_depth_skew',
    'bid_gap_count', 'ask_gap_count',
    'max_bid_wall_frac', 'max_ask_wall_frac',
    'bid_depth_cog', 'ask_depth_cog',
    'bid_size_entropy', 'ask_size_entropy',
    'book_symmetry', 'total_depth_log',
    'l1_l3_bid_ratio', 'l1_l3_ask_ratio',
    'order_frag_asym',
    # Group M rolling: Book shape dynamics
    'entropy_change_5', 'entropy_change_20',
    'cog_asym_5', 'cog_asym_20',
    'depth_accel',
    'symmetry_mean_20',
    'wall_bid_persist_20', 'wall_ask_persist_20',
    'gap_trend_5',
    'l1_ratio_shift_5',
    # Group O z-scores: Book structure anomalies
    'book_imb_zscore_50', 'book_imb_zscore_500', 'book_imb_diverge',
    'entropy_zscore_50', 'entropy_zscore_500', 'entropy_diverge',
    # Group P magnitude: Book-structure magnitude
    'stacked_depth_asym', 'spread_expansion', 'depletion_rate_5',
    # Group R: Depth-weighted OFI (multi-level flow — deep book exclusive)
    'dwfi_5', 'dwfi_20',                      # Depth-weighted flow imbalance
    'ofi_l3_5',                                # OFI at levels 1-3
    'ofi_deep_5',                              # OFI at levels 4-10
    'ofi_deep_vs_l1',                          # Deep vs L1 flow disagreement
    # Group S: Book structure interactions
    'spread_vol_interaction',                  # Spread expansion × vol regime
    'hurst_proxy_20',                          # Trend vs mean-reversion proxy
]

# Ch5: Regime / Volatility
CH5_REGIME_VOL = [
    'vol_regime', 'vol_accel',
    'flow_during_vol', 'aggr_momentum',
    'rvol_10', 'rvol_20', 'rvol_50',
    'vov_10', 'vov_20', 'vov_50',
    'spread_change', 'spread_zscore',
    'event_density', 'time_since_rth',
    'ret_5', 'ret_10', 'ret_20', 'ret_50', 'ret_100',
    # Phase 2: Trade timing (activity regime features)
    'mean_inter_trade_log', 'min_inter_trade_log',
    'trade_burstiness', 'trade_rate',
    # Group O z-scores: Vol regime anomalies
    'rvol_zscore_50', 'rvol_zscore_500', 'rvol_diverge',
    'event_density_zscore_50', 'event_density_zscore_500', 'event_density_diverge',
    # Group P magnitude: Vol regime magnitude
    'volatility_compression',
    # Group S: Regime interactions
    'return_autocorr_10', 'return_autocorr_50',  # Trend vs mean-reversion regime
    'vol_price_corr_20',                          # Volume-price correlation
    'return_skew_20',                              # Return asymmetry
    'multi_signal_strength',                       # Count of features > 2σ
]

# Ch6: Queue Dynamics — bid/ask queue depletion and refill patterns
CH6_QUEUE_DYNAMICS = [
    'queue_depletion_rate_bid_5', 'queue_depletion_rate_ask_5',
    'queue_depletion_rate_bid_20', 'queue_depletion_rate_ask_20',
    'queue_refill_speed_bid_5', 'queue_refill_speed_ask_5',
    'queue_refill_speed_bid_20', 'queue_refill_speed_ask_20',
    'queue_depletion_asymmetry_5', 'queue_depletion_asymmetry_20',
]

# Ch7: Trade Clustering/Herding — sequential trade patterns and size concentration
CH7_TRADE_CLUSTERING = [
    'burst_same_dir_5', 'burst_same_dir_20',
    'trade_persistence_5', 'trade_persistence_20',
    'herding_intensity_5', 'herding_intensity_20',
    'flow_concentration_5', 'flow_concentration_20',
    'trade_arrival_accel_5', 'trade_arrival_accel_20',
]

# Ch8: Cross-Timescale Momentum — autocorrelation and momentum alignment
CH8_CROSS_TIMESCALE = [
    'autocorr_1bar', 'autocorr_10bar', 'autocorr_100bar',
    'momentum_alignment_short', 'momentum_alignment_long',
    'momentum_divergence',
    'ret_skew_50', 'ret_skew_200',
    'momentum_reversal_50', 'momentum_reversal_200',
]

# Ch9: Spread Dynamics — spread change patterns and regime
CH9_SPREAD_DYNAMICS = [
    'spread_change_velocity_5', 'spread_change_velocity_20',
    'spread_widening_count_50', 'spread_tightening_count_50',
    'spread_regime_50', 'spread_regime_200',
    'spread_vol_50', 'spread_vol_200',
    'spread_return_corr_50', 'spread_return_corr_200',
]

# Ch10: Size Classification — institutional vs retail flow inference
CH10_SIZE_CLASS = [
    'avg_order_size_ratio',
    'large_order_frac_bid_5', 'large_order_frac_ask_5',
    'size_imbalance_5', 'size_imbalance_20',
    'institutional_flow_proxy_5', 'institutional_flow_proxy_20',
    'order_size_divergence',
    'size_concentration_bid', 'size_concentration_ask',
]

CHANNEL_DEFINITIONS = {
    'ch1_l1_imbalance': CH1_L1_IMBALANCE,
    'ch2_event_driven': CH2_EVENT_DRIVEN,
    'ch3_toxicity': CH3_TOXICITY,
    'ch4_deep_book': CH4_DEEP_BOOK,
    'ch5_regime_vol': CH5_REGIME_VOL,
    'ch6_queue_dynamics': CH6_QUEUE_DYNAMICS,
    'ch7_trade_clustering': CH7_TRADE_CLUSTERING,
    'ch8_cross_timescale': CH8_CROSS_TIMESCALE,
    'ch9_spread_dynamics': CH9_SPREAD_DYNAMICS,
    'ch10_size_class': CH10_SIZE_CLASS,
}


def validate_channel_disjointness(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Validate that channel feature assignments are strictly disjoint.
    Returns dict of any overlap issues found.
    """
    issues = {}
    channel_names = list(CHANNEL_DEFINITIONS.keys())

    for i in range(len(channel_names)):
        for j in range(i + 1, len(channel_names)):
            ch_a = channel_names[i]
            ch_b = channel_names[j]
            overlap = set(CHANNEL_DEFINITIONS[ch_a]) & set(CHANNEL_DEFINITIONS[ch_b])
            if overlap:
                issues[f"{ch_a} ∩ {ch_b}"] = list(overlap)

    # Check for features not in any channel
    all_assigned = set()
    for feats in CHANNEL_DEFINITIONS.values():
        all_assigned.update(feats)

    unassigned = [f for f in feature_names if f not in all_assigned]
    if unassigned:
        issues['unassigned'] = unassigned

    # Check for channel features not in the feature matrix
    for ch_name, ch_feats in CHANNEL_DEFINITIONS.items():
        missing = [f for f in ch_feats if f not in feature_names]
        if missing:
            issues[f"{ch_name}_missing"] = missing

    return issues


# ============================================================================
# ALPHA CHANNEL — Trains one LightGBM model on a disjoint feature subset
# ============================================================================

class AlphaChannel:
    """
    One independent alpha channel.

    Trains a LightGBM model on a DISJOINT subset of features,
    preventing cross-channel signal leakage.
    """

    def __init__(
        self,
        name: str,
        feature_subset: List[str],
        lgb_params: Optional[dict] = None,
    ):
        self.name = name
        self.feature_subset = feature_subset
        self.lgb_params = lgb_params or self._default_params()

        # Results storage
        self.fold_predictions = []
        self.fold_actuals = []
        self.fold_ics = []
        self.feature_importance = None

    @staticmethod
    def _default_params() -> dict:
        return {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 100,
            'verbose': -1,
            'n_jobs': -1,
            'device': 'gpu',
            'objective': 'regression',
            'metric': 'rmse',
        }

    def get_feature_columns(self, all_feature_names: List[str]) -> np.ndarray:
        """Get column indices for this channel's features."""
        name_to_idx = {n: i for i, n in enumerate(all_feature_names)}
        indices = []
        for f in self.feature_subset:
            if f in name_to_idx:
                indices.append(name_to_idx[f])
        return np.array(indices, dtype=int)

    def get_available_features(self, all_feature_names: List[str]) -> List[str]:
        """Return channel features that actually exist in the feature matrix."""
        available = set(all_feature_names)
        return [f for f in self.feature_subset if f in available]

    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        col_indices: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Train on one walk-forward fold using only this channel's features.

        Args:
            X_train: (N_train, F_total) full feature matrix for training
            y_train: (N_train,) target
            X_test: (N_test, F_total) full feature matrix for testing
            y_test: (N_test,) target
            col_indices: column indices for this channel's features

        Returns:
            predictions array or None if training failed
        """
        import lightgbm as lgb

        # Extract this channel's features only
        X_tr = X_train[:, col_indices]
        X_te = X_test[:, col_indices]

        # Remove NaN targets
        train_valid = np.isfinite(y_train)
        test_valid = np.isfinite(y_test)

        if train_valid.sum() < 500 or test_valid.sum() < 100:
            return None

        X_tr_v = X_tr[train_valid]
        y_tr_v = y_train[train_valid]
        X_te_v = X_te[test_valid]
        y_te_v = y_test[test_valid]

        # 80/20 split for early stopping
        split = int(len(X_tr_v) * 0.8)
        if split < 200:
            return None

        try:
            model = lgb.LGBMRegressor(**self.lgb_params)
            model.fit(
                X_tr_v[:split], y_tr_v[:split],
                eval_set=[(X_tr_v[split:], y_tr_v[split:])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        except Exception as e:
            logger.warning(f"  [{self.name}] Training failed: {e}")
            return None

        preds = model.predict(X_te_v)

        # Track importance
        if self.feature_importance is None:
            self.feature_importance = np.zeros(len(col_indices))
        if hasattr(model, 'feature_importances_'):
            self.feature_importance += model.feature_importances_

        # Track fold metrics
        self.fold_predictions.append(preds)
        self.fold_actuals.append(y_te_v)

        if len(preds) > 10:
            try:
                ic = float(spearmanr(preds, y_te_v)[0])
                if np.isfinite(ic):
                    self.fold_ics.append(ic)
            except Exception:
                pass

        del model
        gc.collect()

        return preds

    def get_results(self, available_features: List[str]) -> dict:
        """Compile channel results."""
        if not self.fold_ics:
            return {
                'name': self.name,
                'n_features': len(self.feature_subset),
                'error': 'no valid folds',
            }

        ics = np.array(self.fold_ics)
        ic_mean = float(ics.mean())
        ic_std = float(ics.std())
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        tstat = ic_mean / ic_std * np.sqrt(len(ics)) if ic_std > 0 else 0.0

        try:
            _, pvalue = ttest_1samp(ics, 0)
            pvalue = float(pvalue)
        except Exception:
            pvalue = 1.0

        # Concatenate all predictions and actuals
        all_preds = np.concatenate(self.fold_predictions)
        all_actuals = np.concatenate(self.fold_actuals)
        valid = np.isfinite(all_preds) & np.isfinite(all_actuals)
        p, a = all_preds[valid], all_actuals[valid]

        hr = float((np.sign(p) == np.sign(a)).mean()) if len(p) > 0 else 0.5
        winners = np.abs(a[np.sign(p) == np.sign(a)]).sum()
        losers = np.abs(a[np.sign(p) != np.sign(a)]).sum()
        pf = float(winners / losers) if losers > 0 else 0.0

        # Top features
        top_features = []
        if self.feature_importance is not None and len(available_features) > 0:
            for idx in np.argsort(self.feature_importance)[::-1][:10]:
                if idx < len(available_features) and self.feature_importance[idx] > 0:
                    top_features.append(
                        (available_features[idx], float(self.feature_importance[idx]))
                    )

        return {
            'name': self.name,
            'n_features': len(available_features),
            'feature_names': available_features,
            'ic': float(spearmanr(p, a)[0]) if len(p) > 50 else ic_mean,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'tstat': tstat,
            'pvalue': pvalue,
            'hit_rate': hr,
            'profit_factor': pf,
            'n_folds': len(self.fold_ics),
            'n_predictions': len(p),
            'fold_ics': [float(x) for x in self.fold_ics],
            'top_features': top_features,
        }

    def reset(self):
        """Clear state for re-use."""
        self.fold_predictions = []
        self.fold_actuals = []
        self.fold_ics = []
        self.feature_importance = None


# ============================================================================
# L1 RESIDUAL EXTRACTION
# ============================================================================

def extract_l1_residual(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: List[str],
    day_boundaries: List[int],
    min_train_days: int = 5,
) -> np.ndarray:
    """
    Extract the target component NOT explained by L1 imbalance features.

    For each walk-forward fold:
    1. Fit Ridge regression on Ch1 features → predict L1 component
    2. Residual = target - L1_prediction

    This explicitly asks: "what alpha exists AFTER removing the L1 linear signal?"

    Args:
        features: (N, F) full feature matrix
        target: (N,) original target
        feature_names: F feature names
        day_boundaries: day boundary indices
        min_train_days: minimum training days

    Returns:
        (N,) residualized target (NaN where not computed)
    """
    from sklearn.linear_model import Ridge

    N = len(target)
    n_days = len(day_boundaries) - 1
    residual = np.full(N, np.nan, dtype=np.float32)

    # Get Ch1 column indices
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    ch1_cols = [name_to_idx[f] for f in CH1_L1_IMBALANCE if f in name_to_idx]

    if not ch1_cols:
        logger.warning("No L1 features found — returning original target")
        return target.copy()

    ch1_idx = np.array(ch1_cols, dtype=int)

    # Pre-extract Ch1 columns ONCE (avoids 2.7GB copy per fold from fancy indexing)
    ch1_data = features[:, ch1_idx].copy()  # (N, 42) float32 ~2.7GB
    logger.info(f"  L1 residual: pre-extracted {len(ch1_cols)} Ch1 columns "
                f"({ch1_data.nbytes / (1024**3):.1f} GB)")

    for test_day in range(min_train_days, n_days):
        # Training: all days up to test_day-2 (1-day purge gap)
        train_end_day = test_day - 1
        train_start = day_boundaries[0]
        train_end = day_boundaries[train_end_day + 1]

        test_start = day_boundaries[test_day]
        test_end = day_boundaries[test_day + 1]

        X_tr = ch1_data[train_start:train_end]  # VIEW, no copy
        y_tr = target[train_start:train_end]
        X_te = ch1_data[test_start:test_end]     # VIEW, no copy
        y_te = target[test_start:test_end]

        # Only valid rows
        tr_valid = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
        te_valid = np.isfinite(y_te) & np.all(np.isfinite(X_te), axis=1)

        if tr_valid.sum() < 500 or te_valid.sum() < 50:
            continue

        # Fit Ridge on L1 features
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr[tr_valid], y_tr[tr_valid])

        # Predict L1 component on test set
        l1_pred = ridge.predict(X_te[te_valid])

        # Residual = actual - L1_component
        res_values = y_te[te_valid] - l1_pred.astype(np.float32)
        te_indices = np.arange(test_start, test_end)[te_valid]
        residual[te_indices] = res_values

    del ch1_data  # Free pre-extracted columns (~2.7GB)

    n_valid = np.isfinite(residual).sum()
    logger.info(f"L1 residual: {n_valid:,} valid values, "
                f"residual std={np.nanstd(residual):.6f} "
                f"(original std={np.nanstd(target):.6f})")

    return residual


# ============================================================================
# MULTI-CHANNEL ALPHA SCANNER
# ============================================================================

class MultiChannelAlphaScanner:
    """
    Orchestrates multiple independent alpha channels with walk-forward evaluation.

    Key design:
    - Each channel trains on DISJOINT feature subsets
    - Walk-forward is run independently per channel
    - L1 residual channel trains on residualized targets
    - Smart ensemble combines channels with IC-weighting + decorrelation bonus
    """

    def __init__(
        self,
        features: np.ndarray,
        feature_names: List[str],
        mid_prices: np.ndarray,
        day_boundaries: List[int],
        min_train_days: int = 5,
        lgb_params: Optional[dict] = None,
    ):
        self.features = features
        self.feature_names = feature_names
        self.mid_prices = mid_prices
        self.day_boundaries = day_boundaries
        self.min_train_days = min_train_days

        # Build channels
        self.channels: Dict[str, AlphaChannel] = {}
        for ch_name, ch_feats in CHANNEL_DEFINITIONS.items():
            self.channels[ch_name] = AlphaChannel(
                name=ch_name,
                feature_subset=ch_feats,
                lgb_params=lgb_params,
            )

        # Residual channel uses ALL non-L1 features, capped at 100 to avoid OOM
        # (262 features × 23M rows exceeds 32GB RAM during LightGBM training)
        residual_feats = [f for f in feature_names if f not in set(CH1_L1_IMBALANCE)]
        MAX_RESIDUAL_FEATURES = 100
        if len(residual_feats) > MAX_RESIDUAL_FEATURES:
            # Deterministic subsample: pick evenly spaced features to maintain diversity
            import numpy as _np
            _rng = _np.random.RandomState(42)
            indices = _rng.choice(len(residual_feats), MAX_RESIDUAL_FEATURES, replace=False)
            indices.sort()
            residual_feats = [residual_feats[i] for i in indices]
            logger.info(f"  ch_residual: capped at {MAX_RESIDUAL_FEATURES} features "
                        f"(from {len([f for f in feature_names if f not in set(CH1_L1_IMBALANCE)])})")
        self.channels['ch_residual'] = AlphaChannel(
            name='ch_residual',
            feature_subset=residual_feats,
            lgb_params=lgb_params,
        )

    def run_all_channels(
        self,
        target: np.ndarray,
        target_name: str = 'return',
        horizon_name: str = 'ret_3s',
        residual_target: Optional[np.ndarray] = None,
    ) -> Dict[str, dict]:
        """
        Run walk-forward evaluation for all channels independently.

        Args:
            target: (N,) target values
            target_name: name for logging
            horizon_name: horizon for logging
            residual_target: (N,) L1-residualized target for ch_residual

        Returns:
            Dict of channel_name → results dict
        """
        n_days = len(self.day_boundaries) - 1
        results = {}

        for ch_name, channel in self.channels.items():
            t0 = time.time()
            logger.info(f"\n--- Channel: {ch_name} ---")

            # Get feature indices
            available = channel.get_available_features(self.feature_names)
            col_indices = channel.get_feature_columns(self.feature_names)

            if len(col_indices) == 0:
                logger.warning(f"  [{ch_name}] No features available, skipping")
                results[ch_name] = {'name': ch_name, 'error': 'no features'}
                continue

            logger.info(f"  [{ch_name}] {len(col_indices)} features: "
                         f"{', '.join(available[:5])}{'...' if len(available) > 5 else ''}")

            # Use residual target for residual channel
            ch_target = residual_target if (ch_name == 'ch_residual' and residual_target is not None) else target

            # Walk-forward: expanding window, 1-day purge gap
            channel.reset()
            for test_day in range(self.min_train_days, n_days):
                train_end_day = test_day - 1
                train_start = self.day_boundaries[0]
                train_end = self.day_boundaries[train_end_day + 1]

                test_start = self.day_boundaries[test_day]
                test_end = self.day_boundaries[test_day + 1]

                X_train = self.features[train_start:train_end]
                y_train = ch_target[train_start:train_end]
                X_test = self.features[test_start:test_end]
                y_test = ch_target[test_start:test_end]

                channel.train_fold(X_train, y_train, X_test, y_test, col_indices)

            elapsed = time.time() - t0
            result = channel.get_results(available)
            result['elapsed_sec'] = elapsed
            result['horizon'] = horizon_name
            result['target'] = target_name

            results[ch_name] = result

            if 'error' not in result:
                logger.info(
                    f"  [{ch_name}] IC={result['ic']:.4f} ICIR={result['icir']:.2f} "
                    f"t={result['tstat']:.2f} HR={result['hit_rate']:.1%} "
                    f"PF={result['profit_factor']:.2f} [{elapsed:.0f}s]"
                )
                if result['top_features']:
                    top3 = result['top_features'][:3]
                    logger.info(f"  [{ch_name}] Top: {', '.join(f'{n}' for n, _ in top3)}")
            else:
                logger.info(f"  [{ch_name}] ERROR: {result.get('error')}")

        return results


# ============================================================================
# SMART ENSEMBLE — IC-weighted + decorrelation bonus
# ============================================================================

class SmartEnsemble:
    """
    Combines predictions from multiple channels using IC-weighted averaging
    with a decorrelation bonus: channels whose predictions are uncorrelated
    with other channels get higher weight.

    This rewards independent alpha sources.
    """

    def __init__(self, min_ic: float = 0.0, decorr_weight: float = 0.5):
        """
        Args:
            min_ic: minimum IC for a channel to receive non-zero weight
            decorr_weight: how much to weight decorrelation (0=pure IC, 1=pure decorr)
        """
        self.min_ic = min_ic
        self.decorr_weight = decorr_weight

    def combine(
        self,
        channel_results: Dict[str, dict],
        channels: Dict[str, AlphaChannel],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Combine channel predictions into ensemble.

        Returns:
            (ensemble_predictions, ensemble_actuals, channel_weights)
        """
        # Collect channels that have predictions
        valid_channels = {}
        for ch_name, result in channel_results.items():
            ch = channels[ch_name]
            if (
                'error' not in result and
                len(ch.fold_predictions) > 0 and
                result.get('ic_mean', 0) >= self.min_ic
            ):
                valid_channels[ch_name] = ch

        if not valid_channels:
            return np.array([]), np.array([]), {}

        # Find common fold count (minimum across channels)
        min_folds = min(len(ch.fold_predictions) for ch in valid_channels.values())

        if min_folds == 0:
            return np.array([]), np.array([]), {}

        # Compute IC-based weights
        ic_weights = {}
        for ch_name, ch in valid_channels.items():
            ic = np.mean(ch.fold_ics[:min_folds]) if ch.fold_ics else 0.0
            ic_weights[ch_name] = max(0, ic)  # Zero out negative IC

        # Compute decorrelation bonus
        # For each channel pair, check correlation of predictions
        decorr_bonus = {ch: 1.0 for ch in valid_channels}
        ch_names = list(valid_channels.keys())

        if len(ch_names) > 1:
            # Compute pairwise prediction correlations per fold
            for i, ch_a in enumerate(ch_names):
                avg_corr = 0.0
                n_pairs = 0
                for j, ch_b in enumerate(ch_names):
                    if i == j:
                        continue
                    # Compute correlation over common folds
                    for fold in range(min_folds):
                        p_a = valid_channels[ch_a].fold_predictions[fold]
                        p_b = valid_channels[ch_b].fold_predictions[fold]
                        min_len = min(len(p_a), len(p_b))
                        if min_len > 10:
                            corr = abs(float(np.corrcoef(p_a[:min_len], p_b[:min_len])[0, 1]))
                            avg_corr += corr
                            n_pairs += 1

                if n_pairs > 0:
                    avg_corr /= n_pairs
                    # Low correlation → high bonus
                    decorr_bonus[ch_a] = 1.0 + (1.0 - avg_corr)

        # Final weights: IC * (1 + decorr_weight * decorr_bonus)
        final_weights = {}
        for ch_name in valid_channels:
            w_ic = ic_weights.get(ch_name, 0)
            w_decorr = decorr_bonus.get(ch_name, 1.0)
            final_weights[ch_name] = w_ic * (1.0 + self.decorr_weight * (w_decorr - 1.0))

        total_w = sum(final_weights.values())
        if total_w <= 0:
            # Equal weight fallback
            for ch_name in final_weights:
                final_weights[ch_name] = 1.0 / len(final_weights)
        else:
            for ch_name in final_weights:
                final_weights[ch_name] /= total_w

        # Build ensemble predictions per fold, then concatenate
        ensemble_preds = []
        ensemble_actuals = []

        for fold in range(min_folds):
            # Find common length for this fold
            fold_lens = [
                len(valid_channels[ch].fold_predictions[fold])
                for ch in valid_channels
            ]
            min_len = min(fold_lens)
            if min_len == 0:
                continue

            fold_pred = np.zeros(min_len, dtype=np.float32)
            for ch_name, ch in valid_channels.items():
                w = final_weights[ch_name]
                fold_pred += w * ch.fold_predictions[fold][:min_len]

            ensemble_preds.append(fold_pred)

            # Use first channel's actuals (they should all be the same target)
            first_ch = next(iter(valid_channels.values()))
            ensemble_actuals.append(first_ch.fold_actuals[fold][:min_len])

        if not ensemble_preds:
            return np.array([]), np.array([]), final_weights

        all_preds = np.concatenate(ensemble_preds)
        all_actuals = np.concatenate(ensemble_actuals)

        return all_preds, all_actuals, final_weights

    @staticmethod
    def compute_ensemble_metrics(
        predictions: np.ndarray,
        actuals: np.ndarray,
        label: str = 'ensemble',
    ) -> dict:
        """Compute standard metrics for ensemble predictions."""
        valid = np.isfinite(predictions) & np.isfinite(actuals)
        p, a = predictions[valid], actuals[valid]

        if len(p) < 50:
            return {'label': label, 'error': f'too few predictions: {len(p)}'}

        ic = float(spearmanr(p, a)[0])
        hr = float((np.sign(p) == np.sign(a)).mean())
        winners = np.abs(a[np.sign(p) == np.sign(a)]).sum()
        losers = np.abs(a[np.sign(p) != np.sign(a)]).sum()
        pf = float(winners / losers) if losers > 0 else 0.0

        return {
            'label': label,
            'ic': ic,
            'hit_rate': hr,
            'profit_factor': pf,
            'n_predictions': len(p),
        }


# ============================================================================
# STACKING META-MODEL — Learns conditional channel combinations
# ============================================================================

class StackingMetaModel:
    """
    Replaces naive IC-weighted averaging with a LightGBM meta-model.

    Instead of: ensemble = Σ(w_i * pred_i),
    this learns: ensemble = f(pred_ch1, pred_ch2, ..., pred_ch5, pred_mag, regime_features)

    This captures CONDITIONAL relationships:
    - "Trust Ch3 toxicity when Ch1 is weak"
    - "Weight Ch4 deep book higher when spread is wide"
    - "Only trade when magnitude prediction > threshold AND direction channels agree"

    Uses walk-forward: trains on fold 0..k-1, predicts fold k (no lookahead).
    """

    def __init__(self, magnitude_threshold: float = 1.5):
        """
        Args:
            magnitude_threshold: minimum predicted magnitude (ticks) for trading.
                Default 1.5 ticks = $18.75, above 1.24 tick cost.
        """
        self.magnitude_threshold = magnitude_threshold

    def build_meta_features(
        self,
        channel_results: Dict[str, dict],
        channels: Dict[str, 'AlphaChannel'],
        fold_idx: int,
    ) -> Optional[np.ndarray]:
        """Build meta-feature matrix for one fold from channel predictions.

        For fold `fold_idx`, each channel has a prediction array. We stack them
        plus derived features (agreement, magnitude, dispersion).

        Returns: (n_samples, n_meta_features) array, or None if data insufficient.
        """
        valid_channels = {
            name: ch for name, ch in channels.items()
            if name in channel_results
            and 'error' not in channel_results[name]
            and len(ch.fold_predictions) > fold_idx
        }

        if len(valid_channels) < 2:
            return None

        # Get common length for this fold
        fold_lens = [len(ch.fold_predictions[fold_idx]) for ch in valid_channels.values()]
        min_len = min(fold_lens)
        if min_len < 50:
            return None

        # Raw channel predictions
        preds = {}
        for ch_name, ch in valid_channels.items():
            preds[ch_name] = ch.fold_predictions[fold_idx][:min_len]

        pred_matrix = np.column_stack(list(preds.values()))  # (N, n_channels)

        # Derived meta-features
        n = min_len
        meta_feats = [pred_matrix]

        # 1. Agreement: sign agreement across channels (how many agree on direction)
        signs = np.sign(pred_matrix)
        mean_sign = np.mean(signs, axis=1, keepdims=True)  # (N, 1)
        agreement = np.abs(mean_sign)  # 1.0 = all agree, 0.0 = split
        meta_feats.append(agreement)

        # 2. Dispersion: std of predictions across channels
        pred_std = np.std(pred_matrix, axis=1, keepdims=True)
        meta_feats.append(pred_std)

        # 3. Max prediction (strongest signal regardless of channel)
        max_abs_pred = np.max(np.abs(pred_matrix), axis=1, keepdims=True)
        meta_feats.append(max_abs_pred)

        # 4. Weighted direction: sign-weighted average (channels that predict bigger moves count more)
        if pred_matrix.shape[1] >= 2:
            abs_preds = np.abs(pred_matrix)
            abs_sum = abs_preds.sum(axis=1, keepdims=True)
            abs_sum = np.maximum(abs_sum, 1e-10)
            weighted_dir = (pred_matrix * abs_preds).sum(axis=1, keepdims=True) / abs_sum
            meta_feats.append(weighted_dir)

        return np.hstack(meta_feats).astype(np.float32)

    def combine(
        self,
        channel_results: Dict[str, dict],
        channels: Dict[str, 'AlphaChannel'],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Walk-forward stacking: for each fold k, train meta-model on folds 0..k-1,
        predict fold k. This ensures no lookahead.

        Returns: (ensemble_predictions, ensemble_actuals, info_dict)
        """
        import lightgbm as lgb

        valid_channels = {
            name: ch for name, ch in channels.items()
            if name in channel_results
            and 'error' not in channel_results[name]
            and len(ch.fold_predictions) > 0
        }

        if len(valid_channels) < 2:
            return np.array([]), np.array([]), {'method': 'stacking', 'error': 'too_few_channels'}

        n_folds = min(len(ch.fold_predictions) for ch in valid_channels.values())
        if n_folds < 3:
            # Not enough folds for walk-forward stacking, fall back to simple average
            logger.warning("Stacking needs 3+ folds, falling back to SmartEnsemble")
            fallback = SmartEnsemble()
            return fallback.combine(channel_results, channels)

        # Walk-forward stacking
        stacking_preds = []
        stacking_actuals = []
        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'min_child_samples': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': -1,
        }

        for fold_k in range(2, n_folds):  # Need at least 2 folds for training
            # Build training data from folds 0..fold_k-1
            train_X, train_y = [], []
            for prev_fold in range(fold_k):
                meta_X = self.build_meta_features(channel_results, channels, prev_fold)
                if meta_X is None:
                    continue
                first_ch = next(iter(valid_channels.values()))
                meta_y = first_ch.fold_actuals[prev_fold][:len(meta_X)]
                valid = np.isfinite(meta_y) & np.all(np.isfinite(meta_X), axis=1)
                train_X.append(meta_X[valid])
                train_y.append(meta_y[valid])

            if not train_X or sum(len(x) for x in train_X) < 500:
                continue

            X_train = np.vstack(train_X)
            y_train = np.concatenate(train_y)

            # Build test data from fold_k
            meta_test = self.build_meta_features(channel_results, channels, fold_k)
            if meta_test is None:
                continue

            first_ch = next(iter(valid_channels.values()))
            y_test = first_ch.fold_actuals[fold_k][:len(meta_test)]

            # Train meta-model
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_train, y_train)

            # Predict
            fold_pred = model.predict(meta_test).astype(np.float32)

            valid_test = np.isfinite(y_test) & np.isfinite(fold_pred)
            stacking_preds.append(fold_pred[valid_test])
            stacking_actuals.append(y_test[valid_test])

        if not stacking_preds:
            return np.array([]), np.array([]), {'method': 'stacking', 'error': 'no_valid_folds'}

        all_preds = np.concatenate(stacking_preds)
        all_actuals = np.concatenate(stacking_actuals)

        info = {
            'method': 'stacking',
            'n_folds_used': len(stacking_preds),
            'n_channels': len(valid_channels),
            'channel_names': list(valid_channels.keys()),
        }

        return all_preds, all_actuals, info


# ============================================================================
# COMPARISON REPORT
# ============================================================================

def format_multi_channel_report(
    channel_results: Dict[str, dict],
    ensemble_result: dict,
    single_model_result: Optional[dict] = None,
    weights: Optional[Dict[str, float]] = None,
) -> str:
    """Format comprehensive comparison report."""
    lines = [
        "",
        "=" * 90,
        "MULTI-CHANNEL ALPHA SCANNER — COMPARISON REPORT",
        "=" * 90,
    ]

    # Individual channels
    lines.append(f"\n{'Channel':<25s} {'IC':>7s} {'ICIR':>6s} {'t':>6s} "
                  f"{'HR':>6s} {'PF':>6s} {'Feats':>5s} {'Folds':>5s}")
    lines.append("-" * 75)

    sorted_channels = sorted(
        channel_results.items(),
        key=lambda x: abs(x[1].get('ic', 0)),
        reverse=True,
    )

    for ch_name, result in sorted_channels:
        if 'error' in result:
            lines.append(f"{ch_name:<25s}  ERROR: {result['error']}")
            continue

        weight_str = f" (w={weights[ch_name]:.2f})" if weights and ch_name in weights else ""
        lines.append(
            f"{ch_name:<25s} {result['ic']:>7.4f} {result['icir']:>6.2f} "
            f"{result['tstat']:>6.2f} {result['hit_rate']:>6.1%} "
            f"{result['profit_factor']:>6.2f} {result['n_features']:>5d} "
            f"{result['n_folds']:>5d}{weight_str}"
        )

    # Ensemble
    lines.append("-" * 75)
    if 'error' not in ensemble_result:
        lines.append(
            f"{'ENSEMBLE':<25s} {ensemble_result['ic']:>7.4f} "
            f"{'---':>6s} {'---':>6s} "
            f"{ensemble_result['hit_rate']:>6.1%} "
            f"{ensemble_result['profit_factor']:>6.2f} "
            f"{'---':>5s} {'---':>5s}"
        )
    else:
        lines.append(f"{'ENSEMBLE':<25s}  ERROR: {ensemble_result.get('error')}")

    # Single model comparison
    if single_model_result and 'error' not in single_model_result:
        lines.append(
            f"{'SINGLE MODEL':<25s} {single_model_result['ic']:>7.4f} "
            f"{single_model_result.get('icir', 0):>6.2f} "
            f"{single_model_result.get('tstat', 0):>6.2f} "
            f"{single_model_result['hit_rate']:>6.1%} "
            f"{single_model_result['profit_factor']:>6.2f}"
        )
    lines.append("=" * 75)

    # Channel weights
    if weights:
        lines.append("\nENSEMBLE WEIGHTS:")
        for ch_name, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(w * 40)
            lines.append(f"  {ch_name:<25s}: {w:.3f} {bar}")

    # Top features per channel
    lines.append("\nTOP FEATURES PER CHANNEL:")
    lines.append("-" * 75)
    for ch_name, result in sorted_channels:
        if 'error' in result or not result.get('top_features'):
            continue
        top5 = result['top_features'][:5]
        feat_str = ", ".join(f"{n}" for n, _ in top5)
        lines.append(f"  {ch_name}: {feat_str}")

    # Fold IC evolution per channel
    lines.append("\nFOLD IC EVOLUTION:")
    lines.append("-" * 75)
    for ch_name, result in sorted_channels:
        if 'error' in result or not result.get('fold_ics'):
            continue
        ics_str = " ".join(f"{x:+.3f}" for x in result['fold_ics'])
        lines.append(f"  {ch_name}: [{ics_str}]")

    # Key findings
    lines.append("\nKEY FINDINGS:")
    lines.append("-" * 75)

    best_ch = sorted_channels[0] if sorted_channels else None
    if best_ch and 'error' not in best_ch[1]:
        lines.append(f"  Best channel: {best_ch[0]} (IC={best_ch[1]['ic']:.4f})")

    if 'error' not in ensemble_result and best_ch and 'error' not in best_ch[1]:
        delta = ensemble_result['ic'] - best_ch[1]['ic']
        if delta > 0:
            lines.append(f"  Ensemble beats best channel by {delta:+.4f} IC")
        else:
            lines.append(f"  Ensemble underperforms best channel by {delta:+.4f} IC")

    # Check which channels have independent signal
    positive_channels = [
        (name, r) for name, r in sorted_channels
        if 'error' not in r and r.get('ic', 0) > 0.005
    ]
    if len(positive_channels) > 1:
        lines.append(f"  {len(positive_channels)} channels with positive IC — "
                      f"evidence of multiple independent alpha sources!")
    elif len(positive_channels) == 1:
        lines.append(f"  Only 1 channel has positive IC — alpha is concentrated in {positive_channels[0][0]}")
    else:
        lines.append(f"  No channels with meaningful positive IC")

    return "\n".join(lines)
