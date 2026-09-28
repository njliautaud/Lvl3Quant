"""
MBO Feature Engine v5 — 340 features from Level 3 order book data.

Covers 13 independent strategy families:
1. L1 Imbalance (Ch1): Order flow, quote dynamics, book thinning
2. Event-Driven (Ch2): Sweeps, icebergs, trade bursts, cancel storms
3. Toxicity (Ch3): VPIN, adverse selection, HFT fingerprints, repositioning
4. Deep Book (Ch4): Multi-level OFI, book shape, spatial structure
5. Regime/Vol (Ch5): Vol regime, return dynamics, activity patterns
6. Order Flow Sequences (Q): Run-length, momentum exhaustion, flow persistence
7. Deep Book Flow (R): Depth-weighted OFI, staleness, price level duration
8. Cross-Signal Interactions (S): Feature combinations, Hurst proxy, multi-signal strength
9. Queue Dynamics (T/Ch6): Queue depletion, refill speed, bid/ask asymmetry
10. Trade Clustering/Herding (U/Ch7): Burst direction, persistence, herding intensity
11. Cross-Timescale Momentum (V/Ch8): Multi-lag autocorr, momentum alignment, divergence
12. Spread Dynamics (W/Ch9): Spread velocity, regime, volatility, return correlation
13. Size Classification (X/Ch10): Order size ratio, institutional flow, concentration

All features are STRICTLY CAUSAL (only past data used).

Feature Groups:
A. Static snapshot (96) — from enhanced engineering.py (45 base + 25 Phase 2 + 15 book shape + 11 quote dynamics)
B. Rolling order flow (15) — OFI, trade imbalance, cancel ratios
C. VPIN (3) — informed trading probability
D. Price momentum (10) — multi-scale returns
E. Realized volatility (6) — vol and vol-of-vol
F. Book shape dynamics (10) — depth changes, spread dynamics
G. Toxicity/adverse selection (5) — Kyle's lambda, price impact
H. Microstructure strategies (10) — icebergs, absorption, sweeps, spoofing
I. MBO-Enhanced rolling (20) — modify rate, order lifetime, aggressive flow, cancel asymmetry
J. Spatial per-level (20) — flattened order counts + concentration per level
K. Vol-Direction interaction (5) — two-stage features
M. Rolling book shape dynamics (10) — entropy change, COG shift, wall persistence
N. Rolling quote dynamics (10) — L1 instability, depletion, selling pressure
O. Cross-timeframe z-scores (30) — 10 key features × (50-bar z, 500-bar z, divergence)
P. Magnitude predictors (10) — book thinning, flow accel, toxicity spike, vol compression
Q. Order flow sequences (10) — run-length, persistence, momentum exhaustion
R. Depth-weighted OFI + Staleness (10) — multi-level flow, bars-since-event, price duration
S. Cross-signal interactions (10) — return autocorr, Hurst proxy, multi-signal composite
T. Queue Dynamics (10) — depletion/refill rates, bid/ask asymmetry (Ch6)
U. Trade Clustering/Herding (10) — burst direction, persistence, herding intensity (Ch7)
V. Cross-Timescale Momentum (10) — multi-lag autocorr, momentum alignment (Ch8)
W. Spread Dynamics (10) — velocity, regime, volatility, return correlation (Ch9)
X. Size Classification (10) — order size ratio, institutional flow, concentration (Ch10)
"""

import warnings
import numpy as np
from typing import List

# Suppress harmless divide-by-zero in np.where (guarded by condition)
warnings.filterwarnings('ignore', message='invalid value encountered in divide')


# ============================================================================
# FEATURE NAMES
# ============================================================================

def get_feature_names() -> List[str]:
    """Return ordered list of all feature names."""
    names = []

    # A: Static snapshot (96) — 45 base + 25 Phase 2 + 15 book shape + 11 quote dynamics
    # These map 1:1 to global_features_raw columns 0-95 in the output matrix.
    names += [
        # Base (10): cols 0-9
        'mid', 'spread', 'vol_imbalance', 'microprice',
        'total_bid_vol', 'total_ask_vol', 'mean_bid_size', 'mean_ask_size',
        'best_bid', 'best_ask',
        # Order flow (8): cols 10-17
        'trade_imbalance', 'buy_volume', 'sell_volume',
        'add_count', 'cancel_count', 'trade_count',
        'cancel_to_add', 'trade_to_add',
        # Microstructure (8): cols 18-25
        'bid_pressure', 'ask_pressure', 'pressure_imbalance',
        'depth_concentration', 'bid_slope', 'ask_slope',
        'spread_ticks', 'depth_ratio',
        # Temporal (5): cols 26-30
        'hour_norm', 'minute_norm', 'time_since_rth', 'time_to_close', 'event_density',
        # MBO Enhanced (14): cols 31-44
        'modify_count', 'modify_to_add',
        'mean_order_lifetime', 'fleeting_ratio',
        'aggressive_buy_count', 'aggressive_sell_count', 'aggressive_imbalance',
        'max_trade_size',
        'cancel_bid_vol', 'cancel_ask_vol', 'cancel_side_imbalance',
        'tick_count', 'sequence_gaps', 'n_orders_completed',
        # Phase 2 deep MBO (25): cols 45-69
        # Trade size distribution (4)
        'large_trade_ratio', 'small_trade_ratio',
        'trade_size_p75_log', 'trade_heterogeneity',
        # Cancel level structure (5)
        'cancel_l1_ratio', 'cancel_l1_vol_frac', 'cancel_deep_vol_frac',
        'add_l1_ratio', 'l1_net_activity',
        # Sweep signature (4)
        'consec_buy_log', 'consec_sell_log',
        'sweep_asymmetry', 'sweep_intensity',
        # Trade timing (4)
        'mean_inter_trade_log', 'min_inter_trade_log',
        'trade_burstiness', 'trade_rate',
        # Modify chains (3)
        'modify_chain_ratio', 'max_chain_log', 'chain_per_order',
        # Multi-level trades (2)
        'multi_level_ratio', 'multi_level_log',
        # Event × context interactions (3)
        'large_sweep_combo', 'cancel_l1_trade_press', 'modify_chain_sweep',
        # Phase 3: Book Shape (15): cols 70-84
        'bid_depth_skew', 'ask_depth_skew',
        'bid_gap_count', 'ask_gap_count',
        'max_bid_wall_frac', 'max_ask_wall_frac',
        'bid_depth_cog', 'ask_depth_cog',
        'bid_size_entropy', 'ask_size_entropy',
        'book_symmetry', 'total_depth_log',
        'l1_l3_bid_ratio', 'l1_l3_ask_ratio',
        'order_frag_asym',
        # Phase 3: Quote Dynamics (11): cols 85-95
        'l1_bid_change_rate', 'l1_ask_change_rate',
        'l1_change_asymmetry',
        'l1_depletion_log', 'depletion_asymmetry',
        'trade_at_bid_frac',
        'trade_size_cv', 'trade_size_skew_raw',
        'reposition_rate',
        'mean_add_size_log', 'max_add_size_norm',
    ]

    # B: Rolling order flow (15)
    for w in [5, 20, 50]:
        names += [f'ofi_{w}', f'trade_imb_{w}', f'cancel_trade_{w}',
                  f'net_flow_{w}', f'event_int_{w}']

    # C: VPIN (3)
    names += ['vpin_20', 'vpin_50', 'vpin_100']

    # D: Price momentum (10)
    for w in [5, 10, 20, 50, 100]:
        names += [f'ret_{w}', f'ret_vel_{w}']

    # E: Realized volatility (6)
    for w in [10, 20, 50]:
        names += [f'rvol_{w}', f'vov_{w}']

    # F: Book shape dynamics (10)
    names += ['depth_ratio_l1', 'depth_ratio_l3', 'depth_ratio_l5',
              'weighted_book_imb', 'spread_change', 'spread_zscore',
              'mid_ret_1', 'mid_ret_5', 'mid_accel', 'book_refresh']

    # G: Toxicity (5)
    names += ['kyle_lambda_20', 'kyle_lambda_50', 'price_impact',
              'adverse_sel', 'toxicity_score']

    # H: Microstructure strategies (10)
    names += ['iceberg_score', 'absorption_bid', 'absorption_ask',
              'large_trade_rev', 'spoof_score', 'aggressive_burst',
              'sweep_count', 'momentum_ignition', 'book_flip', 'hidden_liq']

    # I: MBO-Enhanced rolling features (20) — NEW
    for w in [5, 20]:
        names += [
            f'modify_rate_{w}',         # Rolling modify rate
            f'fleeting_ratio_{w}',       # Rolling fleeting order ratio
            f'aggr_imb_{w}',            # Rolling aggressive buy/sell imbalance
            f'cancel_asym_{w}',          # Rolling cancel side asymmetry
            f'lifetime_mean_{w}',        # Rolling mean order lifetime
            f'max_trade_{w}',            # Rolling max trade size
            f'tick_density_{w}',         # Rolling tick count (market activity)
            f'orders_completed_{w}',     # Rolling order completions
            f'aggr_buy_rate_{w}',        # Rolling aggressive buy rate
            f'cancel_bid_rate_{w}',      # Rolling cancel bid rate
        ]

    # J: Spatial per-level features (20) — NEW
    # Flattened: top 5 bid levels + top 5 ask levels × (order_count, concentration)
    for side in ['bid', 'ask']:
        for lvl in range(5):
            names += [f'{side}_L{lvl+1}_orders', f'{side}_L{lvl+1}_conc']

    # K: Vol-Direction interaction features (5) — NEW
    names += [
        'vol_regime',               # Current vol state (high/low/normal)
        'vol_accel',                # Vol acceleration (rising/falling)
        'flow_during_vol',          # Order flow imbalance when vol is high
        'cancel_asym_vol',          # Cancel asymmetry during vol expansion
        'aggr_momentum',            # Aggressive trade momentum
    ]

    # M: Rolling book shape dynamics (10) — Phase 3 rolling
    # Captures HOW the book's spatial structure is changing over time.
    # Static entropy/COG/walls are per-snapshot; these track their DYNAMICS.
    names += [
        'entropy_change_5',         # Change in total entropy over 5 bars (concentrating?)
        'entropy_change_20',        # Change in total entropy over 20 bars
        'cog_asym_5',              # Rolling mean of (bid_cog - ask_cog) over 5 bars
        'cog_asym_20',             # Same over 20 bars (persistent depth shift?)
        'depth_accel',             # Acceleration of total_depth_log
        'symmetry_mean_20',        # Rolling mean of book_symmetry (persistent asymmetry)
        'wall_bid_persist_20',     # Rolling mean of max_bid_wall_frac (persistent wall?)
        'wall_ask_persist_20',     # Rolling mean of max_ask_wall_frac
        'gap_trend_5',             # Change in (bid_gap + ask_gap) over 5 bars (thinning?)
        'l1_ratio_shift_5',        # Change in (l1_l3_bid - l1_l3_ask) over 5 bars
    ]

    # N: Rolling quote dynamics (10) — Phase 3 rolling
    # Captures persistence of quote stability/instability patterns.
    names += [
        'l1_instability_5',        # Rolling mean of total change rate over 5 bars
        'l1_instability_20',       # Same over 20 bars (sustained instability)
        'depletion_momentum_5',    # Rolling sum of depletions over 5 bars
        'depletion_dir_20',        # Rolling mean of depletion asymmetry over 20 bars
        'sell_pressure_5',         # Rolling mean of trade_at_bid_frac over 5 bars
        'sell_pressure_20',        # Same over 20 bars (persistent selling)
        'size_dispersion_20',      # Rolling mean of trade_size_cv over 20 bars
        'reposition_intensity_5',  # Rolling mean of reposition_rate over 5 bars
        'add_size_trend_20',       # Rolling mean of mean_add_size_log over 20 bars
        'institutional_flow_5',    # Rolling mean of max_add_size_norm over 5 bars
    ]

    # O: Cross-timeframe z-scores (30) — key missing representation
    # Every feature becomes more informative when you know if its current
    # value is ABNORMAL relative to its own recent history. A depth_ratio of
    # 0.6 means different things if it's been 0.6 all day (normal) vs if it
    # was 0.3 an hour ago (sudden shift). Z-score = (x - rolling_mean) / rolling_std.
    # We select the most informative features to z-score (not all 96).
    #
    # Short-term anomaly (50-bar z-score): "Is this unusual right now?"
    # Long-term anomaly (500-bar z-score): "Is this unusual for today?"
    # Divergence (short - long z-score): "Is short-term deviating from the session trend?"
    _zscore_features = [
        # Flow signals — z-scores reveal sudden flow regime changes
        ('ofi_5', 'ofi_5_zscore_50', 'ofi_5_zscore_500', 'ofi_5_diverge'),
        ('trade_imb_5', 'trade_imb_zscore_50', 'trade_imb_zscore_500', 'trade_imb_diverge'),
        ('aggressive_imbalance', 'aggr_imb_zscore_50', 'aggr_imb_zscore_500', 'aggr_imb_diverge'),
        # Book structure — z-scores reveal structural regime changes
        ('depth_ratio_l1', 'depth_ratio_l1_zscore_50', 'depth_ratio_l1_zscore_500', 'depth_ratio_l1_diverge'),
        ('weighted_book_imb', 'book_imb_zscore_50', 'book_imb_zscore_500', 'book_imb_diverge'),
        # Toxicity — z-scores reveal informed flow bursts
        ('vpin_20', 'vpin_zscore_50', 'vpin_zscore_500', 'vpin_diverge'),
        ('cancel_asym_5', 'cancel_asym_zscore_50', 'cancel_asym_zscore_500', 'cancel_asym_diverge'),
        # Volatility — z-scores reveal vol regime transitions
        ('rvol_10', 'rvol_zscore_50', 'rvol_zscore_500', 'rvol_diverge'),
        # Activity — z-scores reveal unusual activity bursts
        ('event_density', 'event_density_zscore_50', 'event_density_zscore_500', 'event_density_diverge'),
        # Book shape — z-scores reveal structural abnormalities
        ('bid_size_entropy', 'entropy_zscore_50', 'entropy_zscore_500', 'entropy_diverge'),
    ]
    for _, short_name, long_name, div_name in _zscore_features:
        names += [short_name, long_name, div_name]

    # P: Magnitude predictors (10) — features specifically targeting BIG moves
    # These don't predict direction; they predict whether a large move is imminent.
    # Combined with direction prediction, this enables selective trading on high-magnitude signals.
    names += [
        'book_thinning',           # (bid_L1_size + ask_L1_size) / rolling_mean — thin book = big move potential
        'stacked_depth_asym',      # (sum bid L1-L3) / (sum ask L1-L3) — one-sided stacking
        'flow_acceleration',       # d(OFI)/dt — accelerating flow = momentum building
        'spread_expansion',        # spread / rolling_mean(spread) — widening = uncertainty
        'cross_flow_agreement',    # sign(ofi_5) == sign(ofi_50) — multi-scale flow alignment
        'toxicity_spike',          # vpin_20 / rolling_mean(vpin_20) — sudden informed flow
        'trade_rate_spike',        # trade_count / rolling_mean(trade_count) — activity burst
        'depletion_rate_5',        # rolling_sum(depletions, 5) — L1 getting wiped repeatedly
        'institutional_signal',    # large_trade_ratio * aggressive_imbalance — big directional orders
        'volatility_compression',  # rvol_5 / rvol_50 — low vol = breakout pending
    ]

    # Q: Order flow sequences (10) — captures sequential flow patterns
    # Current features are all aggregated (sums/means). These capture the
    # SEQUENTIAL structure of order flow: run-length, persistence, transitions.
    # Key strategies: momentum exhaustion, flow regime shifts, institutional stepping.
    names += [
        'buy_run_max_5',           # Max consecutive buy-dominated bars in last 5
        'sell_run_max_5',          # Max consecutive sell-dominated bars in last 5
        'run_direction_20',        # Net run direction over 20 bars (+buy, -sell)
        'flow_reversal_5',         # Buy↔sell transitions in last 5 bars
        'flow_persistence_20',     # Autocorrelation of signed volume over 20 bars
        'aggressive_streak_5',     # Max consecutive aggressive-trade bars in 5
        'buy_intensity_ratio',     # buy_vol_5 / buy_vol_50 (short-term buy surge)
        'sell_intensity_ratio',    # sell_vol_5 / sell_vol_50 (short-term sell surge)
        'momentum_exhaustion',     # Price accel * (-flow_persistence) — weakening flow in trend
        'flow_regime_zscore',      # z(flow_reversal_5) — unusual regime change
    ]

    # R: Depth-weighted OFI + Staleness (10) — multi-level flow + time signals
    # Standard OFI uses only L1. Deep book OFI captures informed traders who
    # post deeper. Staleness features capture when "nothing happening" IS the signal.
    names += [
        'dwfi_5',                  # Depth-weighted flow imbalance (Σ Δbid_i-Δask_i / (i+1), 5 bars)
        'dwfi_20',                 # Same over 20 bars
        'ofi_l3_5',               # OFI at levels 1-3 combined over 5 bars
        'ofi_deep_5',             # OFI at levels 4-10 (deep book only) over 5 bars
        'ofi_deep_vs_l1',         # deep_ofi / l1_ofi — deep book disagreement
        'bars_since_trade',        # Bars since last bar with trade_count > 0
        'bars_since_big_trade',    # Bars since max_trade_size > 2*median
        'bars_since_l1_change',    # Bars since L1 mid price changed
        'price_level_duration',    # Consecutive bars at same mid price (mean-reversion signal)
        'activity_halflife',       # Exponential decay of trade intensity
    ]

    # S: Cross-signal interactions (10) — feature combinations models struggle to learn
    # Tree models can learn interactions but only discover them if they appear in
    # leaf splits. Explicit interactions help: if imbalance is high AND vol is high
    # AND spread is wide, that's a much stronger signal than any one alone.
    names += [
        'return_autocorr_10',      # Rolling autocorr of 1-bar returns (10 bars)
        'return_autocorr_50',      # Same over 50 bars (trend vs mean-reversion regime)
        'vol_price_corr_20',       # Rolling corr(volume, |returns|) — activity predicting moves
        'spread_vol_interaction',   # spread_expansion * vol_ratio — uncertainty + vol = move
        'imb_spread_interaction',   # depth_ratio_l1 * spread_zscore — imbalance + wide spread
        'flow_vol_interaction',     # ofi_5_zscore * rvol_zscore — abnormal flow + abnormal vol
        'toxicity_imb_combo',       # vpin_zscore * depth_ratio_l1 — informed flow + directional book
        'hurst_proxy_20',          # Rolling var(ret_1) / var(ret_5) — trend vs MR proxy
        'return_skew_20',          # Rolling skewness of returns (asymmetric move potential)
        'multi_signal_strength',   # Composite: count of features > 2σ (how many signals firing)
    ]

    # T: Queue Dynamics (Ch6) — 10 features
    names += [
        'queue_depletion_rate_bid_5',    # Rolling mean negative bid queue changes (5 bars)
        'queue_depletion_rate_ask_5',    # Rolling mean negative ask queue changes (5 bars)
        'queue_depletion_rate_bid_20',   # Rolling mean negative bid queue changes (20 bars)
        'queue_depletion_rate_ask_20',   # Rolling mean negative ask queue changes (20 bars)
        'queue_refill_speed_bid_5',      # Rolling mean positive bid queue changes (5 bars)
        'queue_refill_speed_ask_5',      # Rolling mean positive ask queue changes (5 bars)
        'queue_refill_speed_bid_20',     # Rolling mean positive bid queue changes (20 bars)
        'queue_refill_speed_ask_20',     # Rolling mean positive ask queue changes (20 bars)
        'queue_depletion_asymmetry_5',   # Bid vs ask depletion ratio over 5 bars
        'queue_depletion_asymmetry_20',  # Bid vs ask depletion ratio over 20 bars
    ]

    # U: Trade Clustering/Herding (Ch7) — 10 features
    names += [
        'burst_same_dir_5',              # Consecutive same-direction trade count (5 bars)
        'burst_same_dir_20',             # Consecutive same-direction trade count (20 bars)
        'trade_persistence_5',           # Autocorrelation of trade_imbalance (5 bars)
        'trade_persistence_20',          # Autocorrelation of trade_imbalance (20 bars)
        'herding_intensity_5',           # Max trade size / mean trade size ratio (5 bars)
        'herding_intensity_20',          # Max trade size / mean trade size ratio (20 bars)
        'flow_concentration_5',          # Aggressive volume fraction of total (5 bars)
        'flow_concentration_20',         # Aggressive volume fraction of total (20 bars)
        'trade_arrival_accel_5',         # Change in trade count rate (5 bars)
        'trade_arrival_accel_20',        # Change in trade count rate (20 bars)
    ]

    # V: Cross-Timescale Momentum (Ch8) — 10 features
    names += [
        'autocorr_1bar',                 # Return autocorrelation lag-1 (50 bars)
        'autocorr_10bar',                # Return autocorrelation lag-10 (100 bars)
        'autocorr_100bar',               # Return autocorrelation lag-100 (500 bars)
        'momentum_alignment_short',      # Corr between 5-bar and 50-bar return (100 bars)
        'momentum_alignment_long',       # Corr between 50-bar and 500-bar return (500 bars)
        'momentum_divergence',           # 5-bar minus 50-bar momentum (vol-normalized)
        'ret_skew_50',                   # Rolling skewness of returns (50 bars)
        'ret_skew_200',                  # Rolling skewness of returns (200 bars)
        'momentum_reversal_50',          # Sign changes in returns per 50-bar window
        'momentum_reversal_200',         # Sign changes in returns per 200-bar window
    ]

    # W: Spread Dynamics (Ch9) — 10 features
    names += [
        'spread_change_velocity_5',      # Rolling mean of spread changes (5 bars)
        'spread_change_velocity_20',     # Rolling mean of spread changes (20 bars)
        'spread_widening_count_50',      # Bars where spread increased in last 50 bars
        'spread_tightening_count_50',    # Bars where spread decreased in last 50 bars
        'spread_regime_50',              # Spread / rolling mean spread (50 bars)
        'spread_regime_200',             # Spread / rolling mean spread (200 bars)
        'spread_vol_50',                 # Rolling std of spread changes (50 bars)
        'spread_vol_200',                # Rolling std of spread changes (200 bars)
        'spread_return_corr_50',         # Corr between spread changes and returns (50 bars)
        'spread_return_corr_200',        # Corr between spread changes and returns (200 bars)
    ]

    # X: Size Classification (Ch10) — 10 features
    names += [
        'avg_order_size_ratio',          # Avg bid order size / avg ask order size
        'large_order_frac_bid_5',        # Large bid order fraction proxy (5 bars)
        'large_order_frac_ask_5',        # Large ask order fraction proxy (5 bars)
        'size_imbalance_5',              # (large_bid - large_ask) / total (5 bars)
        'size_imbalance_20',             # (large_bid - large_ask) / total (20 bars)
        'institutional_flow_proxy_5',    # Max trade size * trade imbalance (5 bars)
        'institutional_flow_proxy_20',   # Max trade size * trade imbalance (20 bars)
        'order_size_divergence',         # Mean minus median order size at best levels
        'size_concentration_bid',        # Order count / total volume at best bid
        'size_concentration_ask',        # Order count / total volume at best ask
    ]

    # NOTE: Event detector features (14 features) are computed by
    # alpha_discovery/event_detector.py and appended to the feature matrix
    # OUTSIDE of this function. They are NOT included in TOTAL_FEATURES
    # or the base feature names. The multi-channel alpha scanner handles
    # the augmented feature matrix with names from get_event_feature_names().

    return names


TOTAL_FEATURES = len(get_feature_names())


# ============================================================================
# COLUMN INDICES for the 45 global features
# ============================================================================

# Base (0-9)
COL_MID = 0
COL_SPREAD = 1
COL_VOL_IMBALANCE = 2
COL_MICROPRICE = 3
COL_TOTAL_BID_VOL = 4
COL_TOTAL_ASK_VOL = 5
COL_MEAN_BID_SIZE = 6
COL_MEAN_ASK_SIZE = 7
COL_BEST_BID = 8
COL_BEST_ASK = 9

# Order flow (10-17)
COL_TRADE_IMBALANCE = 10
COL_BUY_VOL = 11
COL_SELL_VOL = 12
COL_ADD_COUNT = 13
COL_CANCEL_COUNT = 14
COL_TRADE_COUNT = 15
COL_CANCEL_TO_ADD = 16
COL_TRADE_TO_ADD = 17

# Microstructure (18-25)
COL_BID_PRESSURE = 18
COL_ASK_PRESSURE = 19
COL_PRESSURE_IMBALANCE = 20
COL_DEPTH_CONCENTRATION = 21
COL_BID_SLOPE = 22
COL_ASK_SLOPE = 23
COL_SPREAD_TICKS = 24
COL_DEPTH_RATIO = 25

# Temporal (26-30)
COL_HOUR_NORM = 26
COL_MINUTE_NORM = 27
COL_TIME_SINCE_RTH = 28
COL_TIME_TO_CLOSE = 29
COL_EVENT_DENSITY = 30

# MBO Enhanced (31-44) — NEW
COL_MODIFY_COUNT = 31
COL_MODIFY_TO_ADD = 32
COL_MEAN_LIFETIME = 33
COL_FLEETING_RATIO = 34
COL_AGGR_BUY_COUNT = 35
COL_AGGR_SELL_COUNT = 36
COL_AGGR_IMBALANCE = 37
COL_MAX_TRADE_SIZE = 38
COL_CANCEL_BID_VOL = 39
COL_CANCEL_ASK_VOL = 40
COL_CANCEL_SIDE_IMB = 41
COL_TICK_COUNT = 42
COL_SEQ_GAPS = 43
COL_N_COMPLETED = 44

# Phase 2: Deep MBO event features (45-69)
COL_LARGE_TRADE_RATIO = 45
COL_SMALL_TRADE_RATIO = 46
COL_TRADE_SIZE_P75_LOG = 47
COL_TRADE_HETEROGENEITY = 48
COL_CANCEL_L1_RATIO = 49
COL_CANCEL_L1_VOL_FRAC = 50
COL_CANCEL_DEEP_VOL_FRAC = 51
COL_ADD_L1_RATIO = 52
COL_L1_NET_ACTIVITY = 53
COL_CONSEC_BUY_LOG = 54
COL_CONSEC_SELL_LOG = 55
COL_SWEEP_ASYMMETRY = 56
COL_SWEEP_INTENSITY = 57
COL_MEAN_INTER_TRADE_LOG = 58
COL_MIN_INTER_TRADE_LOG = 59
COL_TRADE_BURSTINESS = 60
COL_TRADE_RATE = 61
COL_MODIFY_CHAIN_RATIO = 62
COL_MAX_CHAIN_LOG = 63
COL_CHAIN_PER_ORDER = 64
COL_MULTI_LEVEL_RATIO = 65
COL_MULTI_LEVEL_LOG = 66
COL_LARGE_SWEEP_COMBO = 67
COL_CANCEL_L1_TRADE_PRESS = 68
COL_MODIFY_CHAIN_SWEEP = 69

# Phase 3: Book Shape features (70-84)
COL_BID_DEPTH_SKEW = 70
COL_ASK_DEPTH_SKEW = 71
COL_BID_GAP_COUNT = 72
COL_ASK_GAP_COUNT = 73
COL_MAX_BID_WALL_FRAC = 74
COL_MAX_ASK_WALL_FRAC = 75
COL_BID_DEPTH_COG = 76
COL_ASK_DEPTH_COG = 77
COL_BID_SIZE_ENTROPY = 78
COL_ASK_SIZE_ENTROPY = 79
COL_BOOK_SYMMETRY = 80
COL_TOTAL_DEPTH_LOG = 81
COL_L1_L3_BID_RATIO = 82
COL_L1_L3_ASK_RATIO = 83
COL_ORDER_FRAG_ASYM = 84

# Phase 3: Quote Dynamics features (85-95)
COL_L1_BID_CHANGE_RATE = 85
COL_L1_ASK_CHANGE_RATE = 86
COL_L1_CHANGE_ASYMMETRY = 87
COL_L1_DEPLETION_LOG = 88
COL_DEPLETION_ASYMMETRY = 89
COL_TRADE_AT_BID_FRAC = 90
COL_TRADE_SIZE_CV = 91
COL_TRADE_SIZE_SKEW_RAW = 92
COL_REPOSITION_RATE = 93
COL_MEAN_ADD_SIZE_LOG = 94
COL_MAX_ADD_SIZE_NORM = 95

N_GLOBAL_FEATURES = 96

# Node feature indices
NODE_SIZE_COL = 2          # Raw size at level
NODE_ORDER_COUNT_COL = 6   # Order count at level
NODE_AVG_SIZE_COL = 7      # Average order size
NODE_CONCENTRATION_COL = 8  # Level concentration


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

def compute_mbo_features(
    mid_prices: np.ndarray,
    global_features_raw: np.ndarray,
    node_features_raw: np.ndarray,
    tick_size: float = 0.25,
    depth_levels: int = 10,
    day_boundaries: list = None,
) -> np.ndarray:
    """
    Compute all ~200 features from pre-extracted snapshot arrays.

    Args:
        mid_prices: (N,) mid prices
        global_features_raw: (N, 96) from enhanced compute_features() v3
        node_features_raw: (N, 2*depth_levels, 9) node features
        tick_size: ES futures tick size (0.25)
        depth_levels: number of price levels per side
        day_boundaries: list of indices where each trading day starts.
            Used to NaN-fill the first max_window bars of each day
            to prevent rolling features from looking across overnight gaps.
            Format: [0, n_day1, n_day1+n_day2, ...] (cumulative).
            If None, no day-boundary masking is applied.

    Returns: (N, TOTAL_FEATURES) float32 feature matrix.
    """
    N = len(mid_prices)
    features = np.full((N, TOTAL_FEATURES), np.nan, dtype=np.float32)
    col = 0

    # FIX: Build day-boundary mask for rolling features.
    # The largest rolling window used anywhere is 100 bars.
    # At each day start, the first 100 bars have rolling windows that
    # would look back into the previous day's data (across overnight gap).
    # We NaN-fill those bars for ALL rolling features (Groups B-K).
    # LightGBM handles NaN natively — it learns to split around missing values.
    MAX_ROLLING_WINDOW = 100
    day_boundary_mask = None
    if day_boundaries is not None and len(day_boundaries) > 1:
        day_boundary_mask = np.zeros(N, dtype=bool)
        for i in range(len(day_boundaries) - 1):
            day_start = day_boundaries[i]
            warmup_end = min(day_start + MAX_ROLLING_WINDOW,
                           day_boundaries[i + 1] if i + 1 < len(day_boundaries) else N)
            day_boundary_mask[day_start:warmup_end] = True

    # === A: Static snapshot features (96) — 45 base + 25 P2 + 15 book shape + 11 quote dynamics ===
    n_global = global_features_raw.shape[1]
    features[:, col:col + n_global] = global_features_raw
    col += n_global  # Dynamically handle both v2 (45) and v3 (70) caches

    # Extract commonly used arrays
    buy_volumes = global_features_raw[:, COL_BUY_VOL]
    sell_volumes = global_features_raw[:, COL_SELL_VOL]
    add_counts = global_features_raw[:, COL_ADD_COUNT]
    cancel_counts = global_features_raw[:, COL_CANCEL_COUNT]
    trade_counts = global_features_raw[:, COL_TRADE_COUNT]
    total_bid_vols = global_features_raw[:, COL_TOTAL_BID_VOL]
    total_ask_vols = global_features_raw[:, COL_TOTAL_ASK_VOL]
    bid_pressures = global_features_raw[:, COL_BID_PRESSURE]
    ask_pressures = global_features_raw[:, COL_ASK_PRESSURE]
    spreads = global_features_raw[:, COL_SPREAD]

    # New MBO columns
    modify_counts = global_features_raw[:, COL_MODIFY_COUNT] if n_global > COL_MODIFY_COUNT else np.zeros(N)
    fleeting_ratios = global_features_raw[:, COL_FLEETING_RATIO] if n_global > COL_FLEETING_RATIO else np.zeros(N)
    aggr_buy_counts = global_features_raw[:, COL_AGGR_BUY_COUNT] if n_global > COL_AGGR_BUY_COUNT else np.zeros(N)
    aggr_sell_counts = global_features_raw[:, COL_AGGR_SELL_COUNT] if n_global > COL_AGGR_SELL_COUNT else np.zeros(N)
    aggr_imbalances = global_features_raw[:, COL_AGGR_IMBALANCE] if n_global > COL_AGGR_IMBALANCE else np.zeros(N)
    max_trade_sizes = global_features_raw[:, COL_MAX_TRADE_SIZE] if n_global > COL_MAX_TRADE_SIZE else np.zeros(N)
    cancel_bid_vols = global_features_raw[:, COL_CANCEL_BID_VOL] if n_global > COL_CANCEL_BID_VOL else np.zeros(N)
    cancel_ask_vols = global_features_raw[:, COL_CANCEL_ASK_VOL] if n_global > COL_CANCEL_ASK_VOL else np.zeros(N)
    cancel_side_imbs = global_features_raw[:, COL_CANCEL_SIDE_IMB] if n_global > COL_CANCEL_SIDE_IMB else np.zeros(N)
    tick_counts = global_features_raw[:, COL_TICK_COUNT] if n_global > COL_TICK_COUNT else np.zeros(N)
    n_completed = global_features_raw[:, COL_N_COMPLETED] if n_global > COL_N_COMPLETED else np.zeros(N)
    mean_lifetimes = global_features_raw[:, COL_MEAN_LIFETIME] if n_global > COL_MEAN_LIFETIME else np.zeros(N)

    # Extract node-level data for spatial features
    # node_features_raw shape: (N, 2*depth_levels, 9)
    has_nodes = node_features_raw is not None and len(node_features_raw.shape) == 3

    # Get L1/L3/L5 sizes from node features
    if has_nodes:
        bid_sizes_l1 = node_features_raw[:, 0, NODE_SIZE_COL]
        ask_sizes_l1 = node_features_raw[:, depth_levels, NODE_SIZE_COL]
        bid_sizes_l3 = node_features_raw[:, min(2, depth_levels - 1), NODE_SIZE_COL] if depth_levels > 2 else bid_sizes_l1
        ask_sizes_l3 = node_features_raw[:, depth_levels + min(2, depth_levels - 1), NODE_SIZE_COL] if depth_levels > 2 else ask_sizes_l1
        bid_sizes_l5 = node_features_raw[:, min(4, depth_levels - 1), NODE_SIZE_COL] if depth_levels > 4 else bid_sizes_l1
        ask_sizes_l5 = node_features_raw[:, depth_levels + min(4, depth_levels - 1), NODE_SIZE_COL] if depth_levels > 4 else ask_sizes_l1
    else:
        bid_sizes_l1 = total_bid_vols
        ask_sizes_l1 = total_ask_vols
        bid_sizes_l3 = bid_sizes_l5 = bid_sizes_l1
        ask_sizes_l3 = ask_sizes_l5 = ask_sizes_l1

    # Pre-compute common arrays
    delta_bid = np.diff(bid_sizes_l1, prepend=bid_sizes_l1[0])
    delta_ask = np.diff(ask_sizes_l1, prepend=ask_sizes_l1[0])
    ofi_raw = delta_bid - delta_ask

    total_trade_vol = buy_volumes + sell_volumes
    safe_ttv = np.maximum(1.0, total_trade_vol)
    trade_imb_raw = (buy_volumes - sell_volumes) / safe_ttv

    signed_vol = buy_volumes - sell_volumes
    log_mid = np.log(np.maximum(mid_prices, 1.0))
    log_ret_1 = np.diff(log_mid, prepend=log_mid[0])
    price_change = np.diff(mid_prices, prepend=mid_prices[0])

    # === B: Rolling order flow (15) ===
    for w in [5, 20, 50]:
        features[:, col] = _rolling_sum(ofi_raw, w)
        features[:, col + 1] = _rolling_mean(trade_imb_raw, w)
        tc_sum = _rolling_sum(trade_counts, w)
        features[:, col + 2] = np.where(
            tc_sum > 0,
            _rolling_sum(cancel_counts, w) / np.maximum(1, tc_sum),
            0.0
        )
        features[:, col + 3] = _rolling_sum(add_counts - cancel_counts, w)
        features[:, col + 4] = _rolling_mean(add_counts + cancel_counts + trade_counts, w)
        col += 5

    # === C: VPIN (3) ===
    abs_imb = np.abs(buy_volumes - sell_volumes)
    for w in [20, 50, 100]:
        features[:, col] = _rolling_mean(abs_imb / safe_ttv, w)
        col += 1

    # === D: Price momentum (10) ===
    for w in [5, 10, 20, 50, 100]:
        ret = log_mid - _shift(log_mid, w)
        features[:, col] = ret
        features[:, col + 1] = ret / max(w, 1)
        col += 2

    # === E: Realized volatility (6) ===
    for w in [10, 20, 50]:
        rvol = _rolling_std(log_ret_1, w)
        features[:, col] = rvol
        features[:, col + 1] = _rolling_std(rvol, w)
        col += 2

    # === F: Book shape dynamics (10) ===
    for bs, asz in [(bid_sizes_l1, ask_sizes_l1),
                    (bid_sizes_l3, ask_sizes_l3),
                    (bid_sizes_l5, ask_sizes_l5)]:
        total = bs + asz
        features[:, col] = np.where(total > 0, bs / np.maximum(1, total), 0.5)
        col += 1

    total_p = bid_pressures + ask_pressures
    features[:, col] = np.where(total_p > 0, (bid_pressures - ask_pressures) / np.maximum(1, total_p), 0.0)
    col += 1

    spread_chg = np.diff(spreads, prepend=spreads[0])
    features[:, col] = spread_chg / tick_size
    col += 1

    sp_mean = _rolling_mean(spreads, 50)
    sp_std = _rolling_std(spreads, 50)
    features[:, col] = np.where(sp_std > 1e-8, (spreads - sp_mean) / sp_std, 0.0)
    col += 1

    features[:, col] = log_mid - _shift(log_mid, 1)
    features[:, col + 1] = log_mid - _shift(log_mid, 5)
    col += 2

    ret1 = log_mid - _shift(log_mid, 1)
    features[:, col] = ret1 - _shift(ret1, 1)  # acceleration
    col += 1

    book_total = total_bid_vols + total_ask_vols
    features[:, col] = np.where(
        book_total > 0,
        (add_counts + cancel_counts) / np.maximum(1.0, book_total),
        0.0
    )
    col += 1

    # === G: Toxicity / adverse selection (5) ===
    for w in [20, 50]:
        features[:, col] = _rolling_regression_slope(price_change, signed_vol, w)
        col += 1

    features[:, col] = np.where(
        _rolling_mean(total_trade_vol, 20) > 0,
        _rolling_mean(np.abs(price_change), 20) / np.maximum(1e-8, _rolling_mean(total_trade_vol, 20)),
        0.0
    )
    col += 1

    trade_dir = np.sign(signed_vol)
    past_ret = _shift(price_change, 1)
    features[:, col] = _rolling_corr(trade_dir, past_ret, 50)
    col += 1

    # FIX: Compute VPIN column index dynamically instead of hardcoding.
    # vpin_20 is the first VPIN feature (Group C), after static + 15 rolling.
    # Use n_global (runtime) to handle both v2 (45) and v3 (70) caches.
    vpin_20_col = n_global + 15
    features[:, col] = features[:, vpin_20_col] * np.maximum(0, -spread_chg / tick_size)
    col += 1

    # === H: Microstructure strategy features (10) ===
    no_move = (np.abs(price_change) < tick_size * 0.5).astype(np.float32)
    features[:, col] = _rolling_sum(trade_counts * no_move, 10)
    col += 1

    bid_absorb = sell_volumes * (price_change >= 0).astype(np.float32)
    features[:, col] = _rolling_sum(bid_absorb, 10)
    col += 1

    ask_absorb = buy_volumes * (price_change <= 0).astype(np.float32)
    features[:, col] = _rolling_sum(ask_absorb, 10)
    col += 1

    avg_tv = _rolling_mean(total_trade_vol, 50)
    large_mask = (total_trade_vol > 2.0 * np.maximum(1, avg_tv)).astype(np.float32)
    lagged_large = _shift(large_mask, 5)
    lagged_sign = _shift(np.sign(signed_vol), 5)
    ret_since = log_mid - _shift(log_mid, 5)
    features[:, col] = np.where(lagged_large > 0.5, -ret_since * lagged_sign, 0.0)
    col += 1

    add_5 = _rolling_sum(add_counts, 5)
    cancel_5 = _rolling_sum(cancel_counts, 5)
    features[:, col] = np.where(
        add_5 > 0,
        (cancel_5 / np.maximum(1, add_5)) * (cancel_5 > add_5).astype(np.float32),
        0.0
    )
    col += 1

    features[:, col] = _rolling_max(np.abs(signed_vol), 5)
    col += 1

    features[:, col] = np.where(trade_counts > 0, np.abs(price_change) / tick_size, 0.0)
    col += 1

    trade_burst = _rolling_sum(trade_counts, 3)
    mom_3 = log_mid - _shift(log_mid, 3)
    features[:, col] = trade_burst * np.abs(mom_3)
    col += 1

    imb_sign = np.sign(total_bid_vols - total_ask_vols)
    sign_chg = np.abs(np.diff(imb_sign, prepend=imb_sign[0]))
    features[:, col] = _rolling_mean(sign_chg, 20)
    col += 1

    visible = bid_sizes_l1 + ask_sizes_l1
    features[:, col] = np.where(visible > 0, total_trade_vol / np.maximum(1, visible), 0.0)
    col += 1

    # === I: MBO-Enhanced rolling features (20) — NEW ===
    for w in [5, 20]:
        # Rolling modify rate (modifies per add)
        mod_sum = _rolling_sum(modify_counts, w)
        add_sum = _rolling_sum(add_counts, w)
        features[:, col] = np.where(add_sum > 0, mod_sum / np.maximum(1, add_sum), 0.0)
        col += 1

        # Rolling fleeting order ratio
        features[:, col] = _rolling_mean(fleeting_ratios, w)
        col += 1

        # Rolling aggressive imbalance
        features[:, col] = _rolling_mean(aggr_imbalances, w)
        col += 1

        # Rolling cancel side asymmetry
        features[:, col] = _rolling_mean(cancel_side_imbs, w)
        col += 1

        # Rolling mean order lifetime
        features[:, col] = _rolling_mean(mean_lifetimes, w)
        col += 1

        # Rolling max trade size
        features[:, col] = _rolling_max(max_trade_sizes, w)
        col += 1

        # Rolling tick density (market activity)
        features[:, col] = _rolling_mean(tick_counts, w)
        col += 1

        # Rolling orders completed
        features[:, col] = _rolling_mean(n_completed, w)
        col += 1

        # Rolling aggressive buy rate
        total_aggr = aggr_buy_counts + aggr_sell_counts
        safe_aggr = np.maximum(1.0, total_aggr)
        features[:, col] = _rolling_mean(aggr_buy_counts / safe_aggr, w)
        col += 1

        # Rolling cancel bid rate (directional cancel)
        total_cancel = cancel_bid_vols + cancel_ask_vols
        safe_cancel = np.maximum(1.0, total_cancel)
        features[:, col] = _rolling_mean(cancel_bid_vols / safe_cancel, w)
        col += 1

    # === J: Spatial per-level features (20) — NEW ===
    # Flatten top 5 bid + top 5 ask levels' order counts and concentrations
    if has_nodes and node_features_raw.shape[2] > NODE_CONCENTRATION_COL:
        for lvl in range(min(5, depth_levels)):
            # Bid levels
            features[:, col] = node_features_raw[:, lvl, NODE_ORDER_COUNT_COL]
            features[:, col + 1] = node_features_raw[:, lvl, NODE_CONCENTRATION_COL]
            col += 2

        for lvl in range(min(5, depth_levels)):
            # Ask levels
            features[:, col] = node_features_raw[:, depth_levels + lvl, NODE_ORDER_COUNT_COL]
            features[:, col + 1] = node_features_raw[:, depth_levels + lvl, NODE_CONCENTRATION_COL]
            col += 2
    else:
        col += 20  # Skip if node features not available

    # === K: Vol-Direction interaction features (5) — NEW ===
    # These enable the two-stage vol→direction strategy

    # Vol regime: current vol relative to rolling average
    rvol_20 = _rolling_std(log_ret_1, 20)
    rvol_100 = _rolling_std(log_ret_1, 100)
    features[:, col] = np.where(rvol_100 > 1e-10, rvol_20 / rvol_100, 1.0)
    col += 1

    # Vol acceleration: is vol rising or falling?
    rvol_5 = _rolling_std(log_ret_1, 5)
    features[:, col] = rvol_5 - _shift(rvol_5, 5)
    col += 1

    # FIX: Order flow imbalance weighted by vol regime (continuous, not binary).
    # Old code used binary vol_expanding flag which zeroed out flow in low-vol.
    # New code uses vol_ratio as continuous weight: high vol amplifies the signal,
    # low vol dampens it but doesn't zero it out.
    vol_ratio = features[:, col - 2]  # vol_regime = rvol_20/rvol_100 (computed 2 cols ago)
    flow_imb_5 = _rolling_mean(trade_imb_raw, 5)
    features[:, col] = flow_imb_5 * vol_ratio
    col += 1

    # FIX: Cancel asymmetry weighted by vol regime (continuous, not binary).
    features[:, col] = _rolling_mean(cancel_side_imbs, 5) * vol_ratio
    col += 1

    # Aggressive trade momentum (direction of aggressive flow over last 5 bars)
    aggr_momentum = _rolling_mean(aggr_imbalances, 5)
    features[:, col] = aggr_momentum
    col += 1

    # === M: Rolling book shape dynamics (10) ===
    # These capture HOW spatial book structure changes over time.
    # The Phase 3 static features (cols 70-95) are per-snapshot; these track dynamics.
    if n_global > COL_BID_SIZE_ENTROPY:
        bid_entropy = global_features_raw[:, COL_BID_SIZE_ENTROPY]
        ask_entropy = global_features_raw[:, COL_ASK_SIZE_ENTROPY]
        total_entropy = bid_entropy + ask_entropy
        bid_cog = global_features_raw[:, COL_BID_DEPTH_COG]
        ask_cog = global_features_raw[:, COL_ASK_DEPTH_COG]
        book_sym = global_features_raw[:, COL_BOOK_SYMMETRY]
        wall_bid = global_features_raw[:, COL_MAX_BID_WALL_FRAC]
        wall_ask = global_features_raw[:, COL_MAX_ASK_WALL_FRAC]
        bid_gaps = global_features_raw[:, COL_BID_GAP_COUNT]
        ask_gaps = global_features_raw[:, COL_ASK_GAP_COUNT]
        total_depth = global_features_raw[:, COL_TOTAL_DEPTH_LOG]
        l1_l3_bid = global_features_raw[:, COL_L1_L3_BID_RATIO]
        l1_l3_ask = global_features_raw[:, COL_L1_L3_ASK_RATIO]

        # Entropy change (book concentrating or dispersing?)
        features[:, col] = total_entropy - _shift(total_entropy, 5)
        features[:, col + 1] = total_entropy - _shift(total_entropy, 20)
        col += 2

        # COG asymmetry rolling (persistent depth shift toward one side?)
        cog_asym = bid_cog - ask_cog
        features[:, col] = _rolling_mean(cog_asym, 5)
        features[:, col + 1] = _rolling_mean(cog_asym, 20)
        col += 2

        # Depth acceleration (book growing or shrinking?)
        depth_vel = total_depth - _shift(total_depth, 1)
        features[:, col] = depth_vel - _shift(depth_vel, 1)
        col += 1

        # Symmetry persistence
        features[:, col] = _rolling_mean(book_sym, 20)
        col += 1

        # Wall persistence (are walls sustained or flickering?)
        features[:, col] = _rolling_mean(wall_bid, 20)
        features[:, col + 1] = _rolling_mean(wall_ask, 20)
        col += 2

        # Gap trend (book thinning out?)
        total_gaps = bid_gaps + ask_gaps
        features[:, col] = total_gaps - _shift(total_gaps, 5)
        col += 1

        # L1 ratio shift (top-heaviness changing sides?)
        l1_asym = l1_l3_bid - l1_l3_ask
        features[:, col] = l1_asym - _shift(l1_asym, 5)
        col += 1
    else:
        col += 10  # Skip if Phase 3 features not in cache

    # === N: Rolling quote dynamics (10) ===
    if n_global > COL_REPOSITION_RATE:
        l1_bid_chg = global_features_raw[:, COL_L1_BID_CHANGE_RATE]
        l1_ask_chg = global_features_raw[:, COL_L1_ASK_CHANGE_RATE]
        depletion = global_features_raw[:, COL_L1_DEPLETION_LOG]
        depletion_asym = global_features_raw[:, COL_DEPLETION_ASYMMETRY]
        trade_bid_frac = global_features_raw[:, COL_TRADE_AT_BID_FRAC]
        trade_cv = global_features_raw[:, COL_TRADE_SIZE_CV]
        repos_rate = global_features_raw[:, COL_REPOSITION_RATE]
        add_size = global_features_raw[:, COL_MEAN_ADD_SIZE_LOG]
        max_add = global_features_raw[:, COL_MAX_ADD_SIZE_NORM]

        # L1 instability (sustained quote flickering?)
        total_chg_rate = l1_bid_chg + l1_ask_chg
        features[:, col] = _rolling_mean(total_chg_rate, 5)
        features[:, col + 1] = _rolling_mean(total_chg_rate, 20)
        col += 2

        # Depletion momentum (one side getting wiped repeatedly?)
        features[:, col] = _rolling_sum(depletion, 5)
        col += 1

        # Depletion direction persistence
        features[:, col] = _rolling_mean(depletion_asym, 20)
        col += 1

        # Selling pressure (persistent trades at bid?)
        features[:, col] = _rolling_mean(trade_bid_frac, 5)
        features[:, col + 1] = _rolling_mean(trade_bid_frac, 20)
        col += 2

        # Trade size dispersion trend
        features[:, col] = _rolling_mean(trade_cv, 20)
        col += 1

        # Repositioning intensity
        features[:, col] = _rolling_mean(repos_rate, 5)
        col += 1

        # Order size trends
        features[:, col] = _rolling_mean(add_size, 20)
        col += 1

        # Institutional flow detection (big orders arriving?)
        features[:, col] = _rolling_mean(max_add, 5)
        col += 1
    else:
        col += 10  # Skip if Phase 3 features not in cache

    # === O: Cross-timeframe z-scores (30) ===
    # For each selected feature: compute 50-bar z-score (short), 500-bar z-score (long),
    # and divergence (short - long). This asks "is this value abnormal?" at two timescales.
    # Z-score = (x - rolling_mean(x, w)) / rolling_std(x, w)
    _zscore_source_cols = {
        'ofi_5': n_global + 0,          # First rolling OFI (Group B)
        'trade_imb_5': n_global + 1,    # First rolling trade imbalance
        'aggressive_imbalance': COL_AGGR_IMBALANCE if n_global > COL_AGGR_IMBALANCE else None,
        'depth_ratio_l1': None,          # Computed below from Group F
        'weighted_book_imb': None,       # Computed below from Group F
        'vpin_20': n_global + 15,        # First VPIN (Group C)
        'cancel_asym_5': None,           # From Group I
        'rvol_10': None,                 # From Group E
        'event_density': COL_EVENT_DENSITY if n_global > COL_EVENT_DENSITY else None,
        'bid_size_entropy': COL_BID_SIZE_ENTROPY if n_global > COL_BID_SIZE_ENTROPY else None,
    }

    # Pre-locate the feature columns we need for z-scoring.
    # Some are in Group A (static), others in computed rolling groups.
    # For rolling features, we reference their already-computed column in the output matrix.
    zscore_arrays = {}

    # Static features (from global_features_raw)
    if n_global > COL_AGGR_IMBALANCE:
        zscore_arrays['aggressive_imbalance'] = global_features_raw[:, COL_AGGR_IMBALANCE]
    if n_global > COL_EVENT_DENSITY:
        zscore_arrays['event_density'] = global_features_raw[:, COL_EVENT_DENSITY]
    if n_global > COL_BID_SIZE_ENTROPY:
        zscore_arrays['bid_size_entropy'] = global_features_raw[:, COL_BID_SIZE_ENTROPY]

    # Rolling features (already computed in this function)
    zscore_arrays['ofi_5'] = features[:, n_global + 0]        # Group B: ofi_5
    zscore_arrays['trade_imb_5'] = features[:, n_global + 1]  # Group B: trade_imb_5
    zscore_arrays['vpin_20'] = features[:, n_global + 15]     # Group C: vpin_20

    # Group F features (computed earlier at known offsets)
    f_start = n_global + 15 + 3 + 10 + 6  # After B(15) + C(3) + D(10) + E(6)
    zscore_arrays['depth_ratio_l1'] = features[:, f_start]         # First in Group F
    zscore_arrays['weighted_book_imb'] = features[:, f_start + 3]  # 4th in Group F

    # Group E: rvol_10 (first in Group E)
    e_start = n_global + 15 + 3 + 10  # After B(15) + C(3) + D(10)
    zscore_arrays['rvol_10'] = features[:, e_start]

    # Group I: cancel_asym_5 (4th feature in first window of Group I)
    i_start = n_global + 15 + 3 + 10 + 6 + 10 + 5 + 10  # After B+C+D+E+F+G+H
    zscore_arrays['cancel_asym_5'] = features[:, i_start + 3]  # 4th in I(w=5)

    # Now compute z-scores for each
    zscore_order = [
        'ofi_5', 'trade_imb_5', 'aggressive_imbalance',
        'depth_ratio_l1', 'weighted_book_imb',
        'vpin_20', 'cancel_asym_5',
        'rvol_10', 'event_density', 'bid_size_entropy',
    ]
    for feat_name in zscore_order:
        arr = zscore_arrays.get(feat_name)
        if arr is not None and len(arr) == N:
            # Short-term z-score (50 bars = 5 seconds)
            mean_50 = _rolling_mean(arr, 50)
            std_50 = _rolling_std(arr, 50)
            z_50 = np.where(std_50 > 1e-8, (arr - mean_50) / std_50, 0.0)
            features[:, col] = np.clip(z_50, -5.0, 5.0)
            col += 1

            # Long-term z-score (500 bars = 50 seconds)
            mean_500 = _rolling_mean(arr, 500)
            std_500 = _rolling_std(arr, 500)
            z_500 = np.where(std_500 > 1e-8, (arr - mean_500) / std_500, 0.0)
            features[:, col] = np.clip(z_500, -5.0, 5.0)
            col += 1

            # Divergence: short-term vs long-term (regime change detection)
            features[:, col] = np.clip(z_50 - z_500, -10.0, 10.0).astype(np.float32)
            col += 1
        else:
            # Feature not available, fill with 0
            col += 3

    # === P: Magnitude predictors (10) ===
    # Features specifically designed to predict WHETHER a big move is coming,
    # regardless of direction. These enable selective trading on high-magnitude signals.

    # Book thinning: thin L1 = easier to move through
    l1_size = bid_sizes_l1 + ask_sizes_l1
    l1_mean = _rolling_mean(l1_size, 100)
    features[:, col] = np.where(l1_mean > 1e-3, l1_size / l1_mean, 1.0)
    col += 1

    # Stacked depth asymmetry: one side much deeper = directional pressure
    if has_nodes and depth_levels >= 3:
        bid_l1_l3 = sum(node_features_raw[:, i, NODE_SIZE_COL] for i in range(min(3, depth_levels)))
        ask_l1_l3 = sum(node_features_raw[:, depth_levels + i, NODE_SIZE_COL] for i in range(min(3, depth_levels)))
        total_l1_l3 = bid_l1_l3 + ask_l1_l3
        features[:, col] = np.where(total_l1_l3 > 0, bid_l1_l3 / np.maximum(1.0, total_l1_l3), 0.5)
    else:
        features[:, col] = 0.5
    col += 1

    # Flow acceleration: d(OFI)/dt — is order flow speeding up?
    ofi_5_arr = features[:, n_global + 0]  # OFI_5 from Group B
    ofi_vel = ofi_5_arr - _shift(ofi_5_arr, 3)
    features[:, col] = ofi_vel - _shift(ofi_vel, 3)
    col += 1

    # Spread expansion: widening spread signals uncertainty / imminent move
    spread_mean = _rolling_mean(spreads, 100)
    features[:, col] = np.where(spread_mean > 1e-6, spreads / spread_mean, 1.0)
    col += 1

    # Cross-flow agreement: when short and long OFI agree on direction
    ofi_50_arr = features[:, n_global + 10]  # OFI_50 from Group B (3rd window, col 10)
    features[:, col] = (np.sign(ofi_5_arr) * np.sign(ofi_50_arr)).astype(np.float32)
    col += 1

    # Toxicity spike: sudden jump in informed trading
    vpin_20_arr = features[:, n_global + 15]  # VPIN_20 from Group C
    vpin_mean = _rolling_mean(vpin_20_arr, 100)
    features[:, col] = np.where(vpin_mean > 1e-6, vpin_20_arr / vpin_mean, 1.0)
    col += 1

    # Trade rate spike: activity burst = something happening
    tc_mean = _rolling_mean(trade_counts, 100)
    features[:, col] = np.where(tc_mean > 0.1, trade_counts / np.maximum(0.1, tc_mean), 1.0)
    col += 1

    # Depletion rate: L1 getting wiped repeatedly = directional pressure
    if n_global > COL_L1_DEPLETION_LOG:
        depl = global_features_raw[:, COL_L1_DEPLETION_LOG]
        features[:, col] = _rolling_sum(depl, 5)
    else:
        features[:, col] = 0.0
    col += 1

    # Institutional signal: large trades + directional aggression = institutional flow
    if n_global > COL_LARGE_TRADE_RATIO:
        lt_ratio = global_features_raw[:, COL_LARGE_TRADE_RATIO]
        features[:, col] = lt_ratio * aggr_imbalances
    else:
        features[:, col] = 0.0
    col += 1

    # Volatility compression: low recent vol relative to session = breakout pending
    features[:, col] = np.where(rvol_100 > 1e-10, rvol_5 / rvol_100, 1.0)
    col += 1

    # === Q: Order flow sequences (10) ===
    # These capture the SEQUENTIAL structure of order flow that aggregated features miss.
    # Run-length, persistence, transitions, momentum exhaustion.

    # Signed flow direction per bar: +1 = buy dominated, -1 = sell dominated
    flow_sign = np.sign(signed_vol)

    # Buy/sell run max: max consecutive same-direction bars in window
    # Efficient: track run lengths, then rolling max
    buy_dom = (flow_sign > 0).astype(np.float32)
    sell_dom = (flow_sign < 0).astype(np.float32)

    # Compute run-lengths using cumsum trick
    buy_run_len = _run_length(buy_dom)
    sell_run_len = _run_length(sell_dom)

    features[:, col] = _rolling_max(buy_run_len, 5)
    features[:, col + 1] = _rolling_max(sell_run_len, 5)
    col += 2

    # Run direction: net signed runs over 20 bars
    features[:, col] = _rolling_sum(flow_sign, 20)
    col += 1

    # Flow reversals: count sign changes in 5 bars
    sign_changes = (np.abs(np.diff(flow_sign, prepend=flow_sign[0])) > 0.5).astype(np.float32)
    features[:, col] = _rolling_sum(sign_changes, 5)
    col += 1

    # Flow persistence: autocorrelation of signed volume over 20 bars
    features[:, col] = _rolling_corr(signed_vol, _shift(signed_vol, 1), 20)
    col += 1

    # Aggressive streak: max consecutive aggressive-dominated bars
    aggr_sign = np.sign(aggr_buy_counts.astype(np.float64) - aggr_sell_counts.astype(np.float64))
    aggr_dom = (np.abs(aggr_sign) > 0).astype(np.float32)
    aggr_run_len = _run_length(aggr_dom)
    features[:, col] = _rolling_max(aggr_run_len, 5)
    col += 1

    # Buy/sell intensity ratio: short-term volume surge vs baseline
    buy_5 = _rolling_sum(buy_volumes, 5)
    buy_50 = _rolling_sum(buy_volumes, 50)
    sell_5 = _rolling_sum(sell_volumes, 5)
    sell_50 = _rolling_sum(sell_volumes, 50)
    features[:, col] = np.where(buy_50 > 0.1, buy_5 / np.maximum(0.1, buy_50) * 10.0, 1.0)
    features[:, col + 1] = np.where(sell_50 > 0.1, sell_5 / np.maximum(0.1, sell_50) * 10.0, 1.0)
    col += 2

    # Momentum exhaustion: slowing flow in the direction of price movement
    price_accel = log_ret_1 - _shift(log_ret_1, 1)
    flow_persist = features[:, col - 5]  # flow_persistence_20 (computed above)
    features[:, col] = price_accel * np.clip(-flow_persist, -1.0, 1.0)
    col += 1

    # Flow regime z-score: is the reversal rate unusual?
    reversal_mean = _rolling_mean(sign_changes, 100)
    reversal_std = _rolling_std(sign_changes, 100)
    features[:, col] = np.where(reversal_std > 1e-6,
                                (features[:, col - 6] - reversal_mean * 5) / np.maximum(1e-6, reversal_std * np.sqrt(5.0)),
                                0.0)
    features[:, col] = np.clip(features[:, col], -5.0, 5.0)
    col += 1

    # === R: Depth-weighted OFI + Staleness (10) ===
    # Multi-level OFI captures deeper book dynamics. Staleness captures information
    # in the ABSENCE of events.

    if has_nodes and depth_levels >= 3:
        # Depth-weighted flow imbalance: changes at each level, weighted by 1/(level+1)
        # Deeper levels get lower weight but still contribute
        dwfi_raw = np.zeros(N, dtype=np.float64)
        for lvl in range(min(5, depth_levels)):
            bid_lvl = node_features_raw[:, lvl, NODE_SIZE_COL]
            ask_lvl = node_features_raw[:, depth_levels + lvl, NODE_SIZE_COL]
            delta_b = np.diff(bid_lvl, prepend=bid_lvl[0])
            delta_a = np.diff(ask_lvl, prepend=ask_lvl[0])
            weight = 1.0 / (lvl + 1)
            dwfi_raw += (delta_b - delta_a) * weight

        features[:, col] = _rolling_sum(dwfi_raw.astype(np.float32), 5)
        features[:, col + 1] = _rolling_sum(dwfi_raw.astype(np.float32), 20)
        col += 2

        # OFI at levels 1-3 combined
        ofi_l3 = np.zeros(N, dtype=np.float64)
        for lvl in range(min(3, depth_levels)):
            bid_lvl = node_features_raw[:, lvl, NODE_SIZE_COL]
            ask_lvl = node_features_raw[:, depth_levels + lvl, NODE_SIZE_COL]
            ofi_l3 += np.diff(bid_lvl, prepend=bid_lvl[0]) - np.diff(ask_lvl, prepend=ask_lvl[0])
        features[:, col] = _rolling_sum(ofi_l3.astype(np.float32), 5)
        col += 1

        # OFI at levels 4-10 (deep book only)
        ofi_deep = np.zeros(N, dtype=np.float64)
        for lvl in range(3, min(depth_levels, 10)):
            bid_lvl = node_features_raw[:, lvl, NODE_SIZE_COL]
            ask_lvl = node_features_raw[:, depth_levels + lvl, NODE_SIZE_COL]
            ofi_deep += np.diff(bid_lvl, prepend=bid_lvl[0]) - np.diff(ask_lvl, prepend=ask_lvl[0])
        features[:, col] = _rolling_sum(ofi_deep.astype(np.float32), 5)
        col += 1

        # Deep vs L1 disagreement: when deep book says opposite of L1
        ofi_l1_5 = features[:, n_global + 0]  # OFI_5 from Group B
        ofi_deep_5 = features[:, col - 1]     # Just computed above
        safe_l1 = np.maximum(np.abs(ofi_l1_5), 0.01)
        features[:, col] = np.clip(ofi_deep_5 / safe_l1, -10.0, 10.0)
        col += 1
    else:
        col += 5  # Skip depth-weighted features if no node data

    # Staleness features: how long since last event?
    # bars_since_trade
    has_trade = (trade_counts > 0).astype(np.float32)
    features[:, col] = _bars_since(has_trade)
    col += 1

    # bars_since_big_trade
    median_trade = _rolling_mean(max_trade_sizes, 50)
    has_big = (max_trade_sizes > 2.0 * np.maximum(0.1, median_trade)).astype(np.float32)
    features[:, col] = _bars_since(has_big)
    col += 1

    # bars_since_l1_change
    mid_changed = (np.abs(np.diff(mid_prices, prepend=mid_prices[0])) > 1e-6).astype(np.float32)
    features[:, col] = _bars_since(mid_changed)
    col += 1

    # Price level duration: consecutive bars at same mid
    same_price = (np.abs(np.diff(mid_prices, prepend=mid_prices[0])) < 1e-6).astype(np.float32)
    features[:, col] = _run_length(same_price).astype(np.float32)
    col += 1

    # Activity halflife: exponential decay of trade intensity
    features[:, col] = _ema(trade_counts.astype(np.float32), 0.1)
    col += 1

    # === S: Cross-signal interactions (10) ===
    # Explicit feature interactions that tree models CAN learn but take many
    # splits to discover. Providing them directly helps the model find multi-signal
    # regimes faster.

    # Return autocorrelation (trend vs mean-reversion regime)
    features[:, col] = _rolling_corr(log_ret_1, _shift(log_ret_1, 1), 10)
    features[:, col + 1] = _rolling_corr(log_ret_1, _shift(log_ret_1, 1), 50)
    col += 2

    # Volume-price correlation
    features[:, col] = _rolling_corr(total_trade_vol, np.abs(log_ret_1), 20)
    col += 1

    # Spread × vol interaction: wide spread + high vol = imminent big move
    s_spread_exp = np.where(spread_mean > 1e-6, spreads / spread_mean, 1.0)
    s_vol_ratio = np.where(rvol_100 > 1e-10, rvol_20 / rvol_100, 1.0)
    features[:, col] = s_spread_exp * s_vol_ratio
    col += 1

    # Imbalance × spread interaction
    # depth_ratio_l1 is at Group F start, spread_zscore is 4 positions later
    f_group_start = n_global + 15 + 3 + 10 + 6  # After B(15) + C(3) + D(10) + E(6)
    s_dr_l1 = features[:, f_group_start]          # depth_ratio_l1
    s_sp_z = features[:, f_group_start + 5]        # spread_zscore (6th in F)
    features[:, col] = (s_dr_l1 - 0.5) * s_sp_z
    col += 1

    # Flow × vol interaction: abnormal flow + abnormal vol (recompute z-scores directly)
    s_ofi_5 = features[:, n_global + 0]  # OFI_5 from Group B
    s_ofi_mean = _rolling_mean(s_ofi_5, 50)
    s_ofi_std = _rolling_std(s_ofi_5, 50)
    s_ofi_z = np.where(s_ofi_std > 1e-8, (s_ofi_5 - s_ofi_mean) / s_ofi_std, 0.0)
    s_rvol_mean = _rolling_mean(rvol_5, 50)
    s_rvol_std = _rolling_std(rvol_5, 50)
    s_rvol_z = np.where(s_rvol_std > 1e-10, (rvol_5 - s_rvol_mean) / s_rvol_std, 0.0)
    features[:, col] = np.clip(s_ofi_z * s_rvol_z, -25.0, 25.0).astype(np.float32)
    col += 1

    # Toxicity × imbalance: informed flow + directional book
    s_vpin = features[:, n_global + 15]  # VPIN_20 from Group C
    s_vpin_mean = _rolling_mean(s_vpin, 100)
    s_vpin_z = np.where(s_vpin_mean > 1e-6, s_vpin / s_vpin_mean, 1.0)
    features[:, col] = ((s_vpin_z - 1.0) * (s_dr_l1 - 0.5) * 4.0).astype(np.float32)
    col += 1

    # Hurst proxy: var(ret_1, w) / var(ret_5, w) — trend vs mean-reversion indicator
    var_ret_1 = _rolling_std(log_ret_1, 20) ** 2
    ret_5_arr = log_mid - _shift(log_mid, 5)
    var_ret_5 = _rolling_std(ret_5_arr, 20) ** 2
    features[:, col] = np.where(var_ret_5 > 1e-16, var_ret_1 * 5.0 / np.maximum(1e-16, var_ret_5), 1.0)
    features[:, col] = np.clip(features[:, col], 0.0, 5.0)
    col += 1

    # Return skewness: asymmetric move potential
    features[:, col] = _rolling_skew(log_ret_1, 20)
    col += 1

    # Multi-signal strength: how many key features are > 2σ from their mean?
    # When many signals fire simultaneously, the move is more likely to be real.
    n_signals = np.zeros(N, dtype=np.float32)
    for key_feat in ['ofi_5', 'aggressive_imbalance', 'depth_ratio_l1',
                      'vpin_20', 'cancel_asym_5', 'rvol_10']:
        arr = zscore_arrays.get(key_feat)
        if arr is not None:
            z_std = _rolling_std(arr, 50)
            z_val = np.where(z_std > 1e-8,
                        (arr - _rolling_mean(arr, 50)) / np.maximum(1e-8, z_std),
                        0.0)
            n_signals += (np.abs(z_val) > 2.0).astype(np.float32)
    features[:, col] = n_signals
    col += 1

    # === T: Queue Dynamics (Ch6) — 10 features ===
    # Tracks bid/ask queue depth changes to detect depletion vs refill patterns.
    if has_nodes:
        bid_queue = node_features_raw[:, 0, NODE_SIZE_COL].astype(np.float32)           # best bid queue
        ask_queue = node_features_raw[:, depth_levels, NODE_SIZE_COL].astype(np.float32) # best ask queue
        bid_queue_diff = np.diff(bid_queue.astype(np.float64), prepend=bid_queue[0]).astype(np.float32)
        ask_queue_diff = np.diff(ask_queue.astype(np.float64), prepend=ask_queue[0]).astype(np.float32)

        # Depletion: negative changes only (queue shrinking)
        bid_depletion = np.minimum(bid_queue_diff, 0.0)
        ask_depletion = np.minimum(ask_queue_diff, 0.0)
        # Refill: positive changes only (queue growing)
        bid_refill = np.maximum(bid_queue_diff, 0.0)
        ask_refill = np.maximum(ask_queue_diff, 0.0)

        features[:, col] = _rolling_mean(bid_depletion, 5)
        col += 1
        features[:, col] = _rolling_mean(ask_depletion, 5)
        col += 1
        features[:, col] = _rolling_mean(bid_depletion, 20)
        col += 1
        features[:, col] = _rolling_mean(ask_depletion, 20)
        col += 1
        features[:, col] = _rolling_mean(bid_refill, 5)
        col += 1
        features[:, col] = _rolling_mean(ask_refill, 5)
        col += 1
        features[:, col] = _rolling_mean(bid_refill, 20)
        col += 1
        features[:, col] = _rolling_mean(ask_refill, 20)
        col += 1

        # Depletion asymmetry: bid vs ask depletion ratio
        bid_dep_5 = _rolling_mean(np.abs(bid_depletion), 5)
        ask_dep_5 = _rolling_mean(np.abs(ask_depletion), 5)
        total_dep_5 = bid_dep_5 + ask_dep_5
        features[:, col] = np.where(total_dep_5 > 1e-6, (bid_dep_5 - ask_dep_5) / total_dep_5, 0.0)
        col += 1
        bid_dep_20 = _rolling_mean(np.abs(bid_depletion), 20)
        ask_dep_20 = _rolling_mean(np.abs(ask_depletion), 20)
        total_dep_20 = bid_dep_20 + ask_dep_20
        features[:, col] = np.where(total_dep_20 > 1e-6, (bid_dep_20 - ask_dep_20) / total_dep_20, 0.0)
        col += 1
    else:
        col += 10

    # === U: Trade Clustering/Herding (Ch7) — 10 features ===
    # Detects sequential clustering of trade direction and size concentration.
    trade_imb_sign = np.sign(trade_imb_raw)

    # Burst same direction: run-length of same-sign trade imbalance
    sign_match = (trade_imb_sign == _shift(trade_imb_sign, 1)).astype(np.float32)
    same_dir_run = _run_length(sign_match)
    features[:, col] = _rolling_mean(same_dir_run, 5)
    col += 1
    features[:, col] = _rolling_mean(same_dir_run, 20)
    col += 1

    # Trade persistence: autocorrelation of trade_imbalance
    features[:, col] = _rolling_corr(trade_imb_raw, _shift(trade_imb_raw, 1), 5)
    col += 1
    features[:, col] = _rolling_corr(trade_imb_raw, _shift(trade_imb_raw, 1), 20)
    col += 1

    # Herding intensity: max_trade_size / mean trade size rolling
    mean_trade_size_5 = _rolling_mean(total_trade_vol / np.maximum(trade_counts, 1.0), 5)
    max_trade_5 = _rolling_mean(max_trade_sizes, 5)
    features[:, col] = np.where(mean_trade_size_5 > 1e-6, max_trade_5 / np.maximum(mean_trade_size_5, 1e-6), 1.0)
    col += 1
    mean_trade_size_20 = _rolling_mean(total_trade_vol / np.maximum(trade_counts, 1.0), 20)
    max_trade_20 = _rolling_mean(max_trade_sizes, 20)
    features[:, col] = np.where(mean_trade_size_20 > 1e-6, max_trade_20 / np.maximum(mean_trade_size_20, 1e-6), 1.0)
    col += 1

    # Flow concentration: |imbalance| / total volume (aggressive fraction)
    abs_imb_vol = np.abs(buy_volumes - sell_volumes)
    features[:, col] = _rolling_mean(abs_imb_vol / safe_ttv, 5)
    col += 1
    features[:, col] = _rolling_mean(abs_imb_vol / safe_ttv, 20)
    col += 1

    # Trade arrival acceleration: change in trade count rate
    tc_mean_5 = _rolling_mean(trade_counts, 5)
    features[:, col] = tc_mean_5 - _shift(tc_mean_5, 5)
    col += 1
    tc_mean_20 = _rolling_mean(trade_counts, 20)
    features[:, col] = tc_mean_20 - _shift(tc_mean_20, 20)
    col += 1

    # === V: Cross-Timescale Momentum (Ch8) — 10 features ===
    # Captures autocorrelation and momentum alignment across timescales.
    ret = np.diff(mid_prices.astype(np.float64), prepend=mid_prices[0]) / np.maximum(mid_prices, 0.01)
    ret = ret.astype(np.float32)

    # Autocorrelations at different lags
    features[:, col] = _rolling_corr(ret, _shift(ret, 1), 50)
    col += 1
    features[:, col] = _rolling_corr(ret, _shift(ret, 10), 100)
    col += 1
    features[:, col] = _rolling_corr(ret, _shift(ret, 100), 500)
    col += 1

    # Momentum alignment: corr between short and long return
    ret_5bar = log_mid - _shift(log_mid, 5)
    ret_50bar = log_mid - _shift(log_mid, 50)
    ret_500bar = log_mid - _shift(log_mid, 500)
    features[:, col] = _rolling_corr(ret_5bar, ret_50bar, 100)
    col += 1
    features[:, col] = _rolling_corr(ret_50bar, ret_500bar, 500)
    col += 1

    # Momentum divergence: 5-bar vs 50-bar momentum, normalized by rolling vol
    ret_5_norm = ret_5bar / np.maximum(_rolling_std(ret, 50), 1e-8)
    ret_50_norm = ret_50bar / np.maximum(_rolling_std(ret, 200), 1e-8)
    features[:, col] = np.clip(ret_5_norm - ret_50_norm, -10.0, 10.0)
    col += 1

    # Rolling skewness of returns
    features[:, col] = _rolling_skew(ret, 50)
    col += 1
    features[:, col] = _rolling_skew(ret, 200)
    col += 1

    # Momentum reversal: sign changes in returns (mean-reversion indicator)
    ret_sign = np.sign(ret).astype(np.float32)
    ret_sign_diff = np.abs(np.diff(ret_sign, prepend=ret_sign[0])).astype(np.float32)
    features[:, col] = _rolling_sum(ret_sign_diff, 50)
    col += 1
    features[:, col] = _rolling_sum(ret_sign_diff, 200)
    col += 1

    # === W: Spread Dynamics (Ch9) — 10 features ===
    # Tracks how the bid-ask spread changes over time.
    spread_diff = np.diff(spreads.astype(np.float64), prepend=spreads[0]).astype(np.float32)

    # Spread change velocity
    features[:, col] = _rolling_mean(spread_diff, 5)
    col += 1
    features[:, col] = _rolling_mean(spread_diff, 20)
    col += 1

    # Spread widening and tightening counts
    widening = (spread_diff > 0).astype(np.float32)
    tightening = (spread_diff < 0).astype(np.float32)
    features[:, col] = _rolling_sum(widening, 50)
    col += 1
    features[:, col] = _rolling_sum(tightening, 50)
    col += 1

    # Spread regime: current spread relative to rolling mean
    spread_mean_50 = _rolling_mean(spreads, 50)
    spread_mean_200 = _rolling_mean(spreads, 200)
    features[:, col] = np.where(spread_mean_50 > 1e-8, spreads / np.maximum(spread_mean_50, 1e-8), 1.0)
    col += 1
    features[:, col] = np.where(spread_mean_200 > 1e-8, spreads / np.maximum(spread_mean_200, 1e-8), 1.0)
    col += 1

    # Spread volatility (std of spread changes)
    features[:, col] = _rolling_std(spread_diff, 50)
    col += 1
    features[:, col] = _rolling_std(spread_diff, 200)
    col += 1

    # Spread-return correlation
    features[:, col] = _rolling_corr(spread_diff, log_ret_1, 50)
    col += 1
    features[:, col] = _rolling_corr(spread_diff, log_ret_1, 200)
    col += 1

    # === X: Size Classification (Ch10) — 10 features ===
    # Classifies order sizes to detect institutional vs retail flow.
    if has_nodes and node_features_raw.shape[2] > NODE_AVG_SIZE_COL:
        avg_bid_size = node_features_raw[:, 0, NODE_AVG_SIZE_COL].astype(np.float32)
        avg_ask_size = node_features_raw[:, depth_levels, NODE_AVG_SIZE_COL].astype(np.float32)

        # Avg order size ratio: bid vs ask
        features[:, col] = np.where(avg_ask_size > 1e-6, avg_bid_size / np.maximum(avg_ask_size, 1e-6), 1.0)
        col += 1
    else:
        features[:, col] = 1.0
        col += 1

    # Large order fraction proxy: use max_bid_wall_frac (col 74) and max_ask_wall_frac (col 75)
    if n_global > COL_MAX_ASK_WALL_FRAC:
        wall_bid_frac = global_features_raw[:, COL_MAX_BID_WALL_FRAC]
        wall_ask_frac = global_features_raw[:, COL_MAX_ASK_WALL_FRAC]
        features[:, col] = _rolling_mean(wall_bid_frac, 5)
        col += 1
        features[:, col] = _rolling_mean(wall_ask_frac, 5)
        col += 1

        # Size imbalance: (large_bid - large_ask) / total
        lbid_5 = _rolling_mean(wall_bid_frac, 5)
        lask_5 = _rolling_mean(wall_ask_frac, 5)
        features[:, col] = (lbid_5 - lask_5) / np.maximum(lbid_5 + lask_5 + 1e-6, 1e-6)
        col += 1
        lbid_20 = _rolling_mean(wall_bid_frac, 20)
        lask_20 = _rolling_mean(wall_ask_frac, 20)
        features[:, col] = (lbid_20 - lask_20) / np.maximum(lbid_20 + lask_20 + 1e-6, 1e-6)
        col += 1
    else:
        col += 4

    # Institutional flow proxy: max_trade_size * trade_imbalance (directional big trades)
    features[:, col] = _rolling_mean(max_trade_sizes * trade_imb_raw, 5)
    col += 1
    features[:, col] = _rolling_mean(max_trade_sizes * trade_imb_raw, 20)
    col += 1

    # Order size divergence: mean vs median approximation at best levels
    if has_nodes and node_features_raw.shape[2] > NODE_AVG_SIZE_COL:
        # Use avg_size at L1 vs avg_size at L2 as mean-vs-spread proxy
        bid_avg_l1 = node_features_raw[:, 0, NODE_AVG_SIZE_COL].astype(np.float32)
        bid_size_l1 = node_features_raw[:, 0, NODE_SIZE_COL].astype(np.float32)
        bid_orders_l1 = node_features_raw[:, 0, NODE_ORDER_COUNT_COL].astype(np.float32)
        # Divergence: avg_size - (total_size / order_count) captures skew
        reconstructed_mean = np.where(bid_orders_l1 > 0, bid_size_l1 / np.maximum(bid_orders_l1, 1.0), bid_avg_l1)
        features[:, col] = bid_avg_l1 - reconstructed_mean
        col += 1
    else:
        features[:, col] = 0.0
        col += 1

    # Size concentration: order_count / total_volume (more orders per unit = more fragmentation)
    if has_nodes:
        bid_orders = node_features_raw[:, 0, NODE_ORDER_COUNT_COL].astype(np.float32)
        ask_orders = node_features_raw[:, depth_levels, NODE_ORDER_COUNT_COL].astype(np.float32)
        bid_vol_l1 = node_features_raw[:, 0, NODE_SIZE_COL].astype(np.float32)
        ask_vol_l1 = node_features_raw[:, depth_levels, NODE_SIZE_COL].astype(np.float32)
        features[:, col] = np.where(bid_vol_l1 > 1e-6, bid_orders / np.maximum(bid_vol_l1, 1e-6), 0.0)
        col += 1
        features[:, col] = np.where(ask_vol_l1 > 1e-6, ask_orders / np.maximum(ask_vol_l1, 1e-6), 0.0)
        col += 1
    else:
        col += 2

    # FIX: Apply day-boundary NaN mask to all rolling features (columns after static).
    # Static snapshot features are from individual snapshots and are fine.
    # Rolling features look back N bars and cross overnight gaps without this.
    # LightGBM handles NaN natively — it learns optimal splits around missing values.
    N_STATIC = n_global  # Group A static features that don't need masking
    if day_boundary_mask is not None:
        features[day_boundary_mask, N_STATIC:] = np.nan

    # Sanitize: replace inf but PRESERVE NaN for LightGBM.
    # LightGBM handles NaN natively — converting to 0 would be wrong
    # because it would treat day-boundary warmup bars as real data.
    # In-place operations to avoid allocating a 25GB temporary copy (OOM fix for 100-day runs).
    inf_mask = np.isinf(features)
    if inf_mask.any():
        features[inf_mask] = 0.0
    del inf_mask
    np.clip(features, -1e6, 1e6, out=features)

    return features


# ============================================================================
# VECTORIZED HELPERS
# ============================================================================

def _shift(arr: np.ndarray, n: int) -> np.ndarray:
    """Shift array by n positions. Positive = look back."""
    result = np.empty_like(arr)
    if n > 0:
        result[:n] = arr[0]
        result[n:] = arr[:-n]
    elif n < 0:
        result[n:] = arr[-1]
        result[:n] = arr[-n:]
    else:
        result[:] = arr
    return result


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling sum using cumsum trick."""
    cs = np.cumsum(arr.astype(np.float64))
    result = np.empty(len(arr), dtype=np.float64)
    result[:window] = cs[:window]
    result[window:] = cs[window:] - cs[:-window]
    return result.astype(np.float32)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean."""
    cs = np.cumsum(arr.astype(np.float64))
    counts = np.minimum(np.arange(1, len(arr) + 1, dtype=np.float64), window)
    result = np.empty(len(arr), dtype=np.float64)
    result[:window] = cs[:window] / counts[:window]
    result[window:] = (cs[window:] - cs[:-window]) / window
    return result.astype(np.float32)


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling standard deviation."""
    mean = _rolling_mean(arr, window)
    sq_mean = _rolling_mean(arr.astype(np.float64) ** 2, window)
    var = sq_mean - mean.astype(np.float64) ** 2
    var = np.maximum(var, 0.0)
    return np.sqrt(var).astype(np.float32)


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling max."""
    N = len(arr)
    result = np.empty(N, dtype=arr.dtype)
    for i in range(min(window, N)):
        result[i] = np.max(arr[:i + 1])
    if N > window:
        from numpy.lib.stride_tricks import sliding_window_view
        wins = sliding_window_view(arr, window)
        result[window - 1:window - 1 + len(wins)] = np.max(wins, axis=1)
    return result


def _rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling Pearson correlation."""
    mx = _rolling_mean(x, window)
    my = _rolling_mean(y, window)
    mxy = _rolling_mean(x * y, window)
    sx = _rolling_std(x, window)
    sy = _rolling_std(y, window)
    denom = sx * sy
    result = np.where(denom > 1e-10, (mxy - mx * my) / denom, 0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def _rolling_regression_slope(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling OLS slope: y = a + b*x, returns b."""
    mx = _rolling_mean(x, window)
    my = _rolling_mean(y, window)
    mxy = _rolling_mean(x * y, window)
    mx2 = _rolling_mean(x.astype(np.float64) ** 2, window)
    denom = mx2 - mx.astype(np.float64) ** 2
    result = np.where(np.abs(denom) > 1e-10, (mxy - mx * my) / denom, 0.0)
    return np.clip(result, -1e4, 1e4).astype(np.float32)


def _run_length(mask: np.ndarray) -> np.ndarray:
    """Compute current run-length of True/1.0 values at each position.

    If mask[i] is True, run_length[i] = how many consecutive True values
    ending at position i. If mask[i] is False, run_length[i] = 0.
    """
    N = len(mask)
    result = np.zeros(N, dtype=np.float32)
    current_run = 0.0
    for i in range(N):
        if mask[i] > 0.5:
            current_run += 1.0
        else:
            current_run = 0.0
        result[i] = current_run
    return result


def _bars_since(event: np.ndarray) -> np.ndarray:
    """Count bars since last event (where event[i] > 0.5).

    Returns array where result[i] = number of bars since last event.
    Capped at 500 to avoid extreme values. Returns 500 if no event yet seen.
    """
    N = len(event)
    result = np.empty(N, dtype=np.float32)
    last_event = -500  # Start as if no event seen
    for i in range(N):
        if event[i] > 0.5:
            last_event = i
        result[i] = min(i - last_event, 500)
    return result


def _ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average. Causal, starts from first value.

    EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}
    """
    N = len(arr)
    result = np.empty(N, dtype=np.float32)
    result[0] = arr[0]
    decay = 1.0 - alpha
    for i in range(1, N):
        result[i] = alpha * arr[i] + decay * result[i - 1]
    return result


def _rolling_skew(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling skewness."""
    mean = _rolling_mean(arr, window)
    std = _rolling_std(arr, window)
    # Third central moment
    diff = arr.astype(np.float64) - mean.astype(np.float64)
    m3 = _rolling_mean(diff ** 3, window)
    safe_std = np.maximum(std.astype(np.float64), 1e-10)
    skew = m3 / (safe_std ** 3)
    return np.clip(skew, -10.0, 10.0).astype(np.float32)
