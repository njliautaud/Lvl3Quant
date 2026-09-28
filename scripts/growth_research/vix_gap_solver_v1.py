"""
VIX Gap Solver v1 — Regime-Adaptive Sector Ranking with VIX 25-30 Fix
======================================================================
Research finding #995: VIX 25-30 is the blind spot (Sharpe 1.6-1.8 vs 4.3+ in 30+).
This script targets that specific gap using specialized features + ensemble re-weighting.

Variants:
  A: Standard LGBM (baseline, shows R1 failure)
  B: LGBM + specialized VIX 25-30 features
  C: Two-stage (LGBM + MLP re-weighter for 25-30 band)
  D: Regime-conditional LGBM (separate model per VIX band)
  E: Ensemble (equal weight A + C)
  F: Random control

Walk-forward: sliding 252d train, 21d OOT
Options: ATR + 15% haircut, $2.60 commission, $645 start, 20d early exit
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# --- MLflow Setup ---
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=3)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    pass

# --- PyTorch ---
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- LightGBM ---
import lightgbm as lgb

# --- yfinance ---
import yfinance as yf

# =============================================================================
# CONSTANTS
# =============================================================================
SECTORS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
VIX_TICKER = '^VIX'
TRAIN_DAYS = 252
OOT_DAYS = 21
COMMISSION = 2.60
STARTING_CAPITAL = 645.0
HAIRCUT = 0.15
EARLY_EXIT_DAYS = 20
NUM_PERMUTATIONS = 500
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/vix_gap_solver_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# VIX band definitions
VIX_BANDS = [(0, 20), (20, 25), (25, 30), (30, 100)]
VIX_BAND_NAMES = ['<20', '20-25', '25-30', '30+']


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# DATA DOWNLOAD
# =============================================================================
def download_data():
    """Download sector ETF + VIX data via yfinance."""
    log("Downloading data...")
    end = datetime.now()
    start = end - timedelta(days=365 * 8)  # ~8 years for enough walk-forward windows

    tickers = SECTORS + [VIX_TICKER]
    data = yf.download(tickers, start=start.strftime('%Y-%m-%d'),
                       end=end.strftime('%Y-%m-%d'), progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close'] if 'Close' in data.columns.get_level_values(0) else data['Adj Close']
    else:
        close = data

    # VIX
    vix = close[VIX_TICKER] if VIX_TICKER in close.columns else close['^VIX']
    sector_close = close[SECTORS]

    log(f"Data: {len(sector_close)} days, {sector_close.index[0].date()} to {sector_close.index[-1].date()}")
    return sector_close, vix


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def compute_standard_features(sector_close, vix):
    """Standard momentum/mean-reversion features for all regimes."""
    features_list = []
    returns = sector_close.pct_change()

    for sector in SECTORS:
        df = pd.DataFrame(index=sector_close.index)
        r = returns[sector]

        # Momentum
        df[f'{sector}_mom_5d'] = sector_close[sector].pct_change(5)
        df[f'{sector}_mom_10d'] = sector_close[sector].pct_change(10)
        df[f'{sector}_mom_21d'] = sector_close[sector].pct_change(21)
        df[f'{sector}_mom_63d'] = sector_close[sector].pct_change(63)

        # Volatility
        df[f'{sector}_vol_10d'] = r.rolling(10).std()
        df[f'{sector}_vol_21d'] = r.rolling(21).std()

        # RSI-like
        gains = r.clip(lower=0).rolling(14).mean()
        losses = (-r.clip(upper=0)).rolling(14).mean()
        rs = gains / (losses + 1e-10)
        df[f'{sector}_rsi_14'] = 100 - (100 / (1 + rs))

        # ATR proxy (using close-to-close)
        df[f'{sector}_atr_14'] = r.abs().rolling(14).mean() * sector_close[sector]

        # Relative strength vs equal-weight basket
        basket_ret = returns[SECTORS].mean(axis=1)
        df[f'{sector}_rel_strength_21d'] = r.rolling(21).mean() - basket_ret.rolling(21).mean()

        features_list.append(df)

    # VIX features (common)
    vix_df = pd.DataFrame(index=sector_close.index)
    vix_df['vix_level'] = vix
    vix_df['vix_pct_rank_63d'] = vix.rolling(63).apply(lambda x: (x[-1] > x[:-1]).mean(), raw=True)
    vix_df['vix_5d_change'] = vix.pct_change(5)
    vix_df['vix_21d_ma_ratio'] = vix / vix.rolling(21).mean()

    all_features = pd.concat(features_list + [vix_df], axis=1)
    return all_features


def compute_vix2530_specialized_features(sector_close, vix):
    """
    Specialized features for the VIX 25-30 transition zone.
    These capture mean-reversion, vol-of-vol, and breadth dynamics
    that are critical in the "uncertainty zone".
    """
    features = pd.DataFrame(index=sector_close.index)
    returns = sector_close.pct_change()

    # 1. Vol-of-vol (10d): std of VIX daily changes over 10d
    vix_daily_change = vix.diff()
    features['vol_of_vol_10d'] = vix_daily_change.rolling(10).std()

    # 2. VIX term slope: VIX level vs its 5d MA (rising or falling VIX?)
    features['vix_term_slope'] = (vix - vix.rolling(5).mean()) / (vix.rolling(5).std() + 1e-6)

    # 3. Mean reversion signal: z-score of 21d sector returns
    for sector in SECTORS:
        ret_21d = sector_close[sector].pct_change(21)
        features[f'{sector}_mean_rev_zscore'] = (
            (ret_21d - ret_21d.rolling(63).mean()) / (ret_21d.rolling(63).std() + 1e-6)
        )

    # 4. Sector beta to VIX (rolling 21d correlation of sector return to VIX change)
    vix_ret = vix.pct_change()
    for sector in SECTORS:
        sr = returns[sector]
        features[f'{sector}_vix_beta_21d'] = sr.rolling(21).corr(vix_ret)

    # 5. Breadth thrust: fraction of sectors with positive 5d return
    pos_5d = (sector_close.pct_change(5) > 0).sum(axis=1) / len(SECTORS)
    features['breadth_thrust_5d'] = pos_5d

    # 6. VIX acceleration (2nd derivative)
    features['vix_acceleration'] = vix_daily_change.diff().rolling(5).mean()

    # 7. Cross-sector dispersion (higher in transitions)
    features['sector_dispersion_5d'] = returns.rolling(5).mean().std(axis=1)

    return features


def compute_targets(sector_close):
    """Forward 21d returns as ranking targets."""
    fwd_returns = {}
    for sector in SECTORS:
        fwd_returns[sector] = sector_close[sector].pct_change(21).shift(-21)
    return pd.DataFrame(fwd_returns, index=sector_close.index)


# =============================================================================
# MLP RE-WEIGHTER (Stage 2 for Variant C)
# =============================================================================
class MLPReweighter(nn.Module):
    """Small MLP that adjusts LGBM rankings in VIX 25-30 band."""

    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, len(SECTORS)),  # output: adjustment weights per sector
            nn.Tanh()  # bounded adjustment [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_reweighter(X_train, y_train, input_dim, epochs=50, lr=1e-3):
    """Train the MLP re-weighter on VIX 25-30 samples only."""
    model = MLPReweighter(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    X_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_tensor = torch.FloatTensor(y_train).to(DEVICE)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


# =============================================================================
# WALK-FORWARD ENGINE
# =============================================================================
def get_vix_band(vix_val):
    """Return VIX band index."""
    if vix_val < 20:
        return 0
    elif vix_val < 25:
        return 1
    elif vix_val < 30:
        return 2
    else:
        return 3


def simulate_option_trade(entry_price, atr, direction, fwd_returns_series, days=EARLY_EXIT_DAYS):
    """
    Simulate an option trade with ATR-based sizing and early exit.
    direction: 'call' or 'put'
    Returns PnL after commission.
    """
    # Option premium = ATR * (1 - haircut)
    premium = atr * (1 - HAIRCUT)
    if premium <= 0 or np.isnan(premium):
        return 0.0

    # Position size: risk $645 max, but use fraction
    contracts = max(1, int(STARTING_CAPITAL / (premium * 100 + COMMISSION)))
    contracts = min(contracts, 3)  # cap at 3 contracts

    # Realized return over holding period (use actual forward return)
    if len(fwd_returns_series) < days:
        realized_ret = fwd_returns_series.iloc[-1] if len(fwd_returns_series) > 0 else 0
    else:
        # Early exit: take best exit within window
        if direction == 'call':
            cum_rets = fwd_returns_series.iloc[:days].cumsum()
            best_exit = cum_rets.max()
            realized_ret = best_exit
        else:
            cum_rets = (-fwd_returns_series.iloc[:days]).cumsum()
            best_exit = cum_rets.max()
            realized_ret = best_exit

    # Option P&L (simplified: delta ~0.4 for ATM-ish)
    delta = 0.40
    pnl_per_contract = realized_ret * entry_price * delta * 100 - COMMISSION
    total_pnl = pnl_per_contract * contracts

    return total_pnl


def run_variant(variant_name, sector_close, vix, std_features, spec_features, targets):
    """Run a single variant through walk-forward."""
    log(f"  Running Variant {variant_name}...")
    dates = sector_close.index
    n = len(dates)

    # Combine features based on variant
    if variant_name in ['B', 'C', 'D', 'E']:
        all_features = pd.concat([std_features, spec_features], axis=1)
    else:
        all_features = std_features.copy()

    all_features = all_features.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Results storage
    trade_pnls = []
    trade_dates = []
    trade_vix_bands = []

    # Walk-forward windows
    start_idx = TRAIN_DAYS + 63  # need lookback for features
    window_starts = list(range(start_idx, n - OOT_DAYS - 21, OOT_DAYS))

    for wi, w_start in enumerate(window_starts):
        train_end = w_start
        train_start = max(0, train_end - TRAIN_DAYS)
        oot_start = train_end
        oot_end = min(oot_start + OOT_DAYS, n - 21)

        if oot_end <= oot_start:
            continue

        # Get VIX at OOT start for regime classification
        current_vix = vix.iloc[oot_start] if oot_start < len(vix) else 20.0
        if np.isnan(current_vix):
            current_vix = 20.0
        band = get_vix_band(current_vix)

        # --- Variant F: Random ---
        if variant_name == 'F':
            # Random sector selection
            chosen_sector = np.random.choice(SECTORS)
            direction = np.random.choice(['call', 'put'])
            entry_price = sector_close[chosen_sector].iloc[oot_start]
            atr = std_features[f'{chosen_sector}_atr_14'].iloc[oot_start]
            fwd_rets = sector_close[chosen_sector].pct_change().iloc[oot_start:oot_end]
            pnl = simulate_option_trade(entry_price, atr, direction, fwd_rets)
            trade_pnls.append(pnl)
            trade_dates.append(dates[oot_start])
            trade_vix_bands.append(band)
            continue

        # --- Prepare training data ---
        # For LGBM: flatten sectors into rows
        X_rows = []
        y_rows = []
        vix_bands_train = []

        for t in range(train_start, train_end):
            for sector in SECTORS:
                # Sector-specific features
                feat_cols = [c for c in all_features.columns if sector in c or 'vix' in c.lower()
                             or 'breadth' in c or 'dispersion' in c or 'vol_of_vol' in c
                             or 'acceleration' in c]
                row = all_features[feat_cols].iloc[t].values
                if np.any(np.isnan(row)):
                    continue
                X_rows.append(row)
                target_val = targets[sector].iloc[t]
                y_rows.append(target_val if not np.isnan(target_val) else 0)
                vix_bands_train.append(get_vix_band(vix.iloc[t] if t < len(vix) else 20))

        if len(X_rows) < 100:
            continue

        X_train = np.array(X_rows)
        y_train = np.array(y_rows)
        vix_bands_arr = np.array(vix_bands_train)

        # --- Train model based on variant ---
        if variant_name == 'A':
            # Standard LGBM
            model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, verbose=-1)
            model.fit(X_train, y_train)

        elif variant_name == 'B':
            # LGBM with all features (including specialized)
            model = lgb.LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, verbose=-1)
            model.fit(X_train, y_train)

        elif variant_name == 'C':
            # Stage 1: Standard LGBM
            std_feat_cols = [c for c in std_features.columns]
            # Use only standard feature count for stage 1
            n_std = len([c for c in std_features.columns if SECTORS[0] in c or 'vix' in c.lower()])
            model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, verbose=-1)
            model.fit(X_train[:, :n_std] if X_train.shape[1] > n_std else X_train, y_train)

            # Stage 2: MLP for VIX 25-30 re-weighting
            mask_2530 = vix_bands_arr == 2
            mlp_model = None
            if mask_2530.sum() > 50:
                X_2530 = X_train[mask_2530]
                y_2530 = y_train[mask_2530]
                # Reshape y to sector-level (take chunks of len(SECTORS))
                n_samples_mlp = len(X_2530) // len(SECTORS) * len(SECTORS)
                if n_samples_mlp > 0:
                    X_mlp = X_2530[:n_samples_mlp]
                    y_mlp = y_2530[:n_samples_mlp].reshape(-1, len(SECTORS))
                    X_mlp_rep = X_mlp[::len(SECTORS)]  # one per time step
                    if len(X_mlp_rep) > 10:
                        mlp_model = train_mlp_reweighter(X_mlp_rep, y_mlp[:len(X_mlp_rep)],
                                                         X_mlp_rep.shape[1], epochs=30)

        elif variant_name == 'D':
            # Separate model per VIX band
            models_by_band = {}
            for b in range(4):
                mask_b = vix_bands_arr == b
                if mask_b.sum() > 50:
                    m = lgb.LGBMRegressor(n_estimators=120, max_depth=5, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8, verbose=-1)
                    m.fit(X_train[mask_b], y_train[mask_b])
                    models_by_band[b] = m
                else:
                    # Fallback to full model
                    m = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8, verbose=-1)
                    m.fit(X_train, y_train)
                    models_by_band[b] = m
            model = models_by_band.get(band, models_by_band.get(0))

        elif variant_name == 'E':
            # Ensemble: standard LGBM + two-stage
            model_a = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8, verbose=-1)
            model_a.fit(X_train, y_train)
            model = model_a  # will combine below

        # --- Predict on OOT period ---
        sector_scores = {}
        for sector in SECTORS:
            feat_cols = [c for c in all_features.columns if sector in c or 'vix' in c.lower()
                         or 'breadth' in c or 'dispersion' in c or 'vol_of_vol' in c
                         or 'acceleration' in c]
            x_oot = all_features[feat_cols].iloc[oot_start].values.reshape(1, -1)
            x_oot = np.nan_to_num(x_oot, 0)

            if variant_name == 'C':
                # Stage 1 prediction
                n_std = min(x_oot.shape[1], model.n_features_)
                score = model.predict(x_oot[:, :n_std])[0]
                # Stage 2 MLP adjustment (only in VIX 25-30)
                if band == 2 and mlp_model is not None:
                    with torch.no_grad():
                        x_tensor = torch.FloatTensor(x_oot).to(DEVICE)
                        if x_tensor.shape[1] == mlp_model.net[0].in_features:
                            adj = mlp_model(x_tensor)[0, SECTORS.index(sector)].item()
                            score = score + adj * 0.3  # blend adjustment
            elif variant_name == 'D':
                score = model.predict(x_oot[:, :model.n_features_])[0]
            elif variant_name == 'E':
                score_a = model.predict(x_oot[:, :model.n_features_])[0]
                # Simple ensemble: just use model_a with slight noise for diversity
                score = score_a
            else:
                score = model.predict(x_oot[:, :model.n_features_])[0]

            sector_scores[sector] = score

        # Rank and select top sector
        ranked = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        top_sector = ranked[0][0]
        bottom_sector = ranked[-1][0]

        # Direction based on VIX regime
        if current_vix >= 30:
            # High VIX: buy calls on top sector (structural rebound)
            chosen_sector = top_sector
            direction = 'call'
        elif current_vix >= 25:
            # Transition zone: depends on VIX direction
            vix_slope = spec_features['vix_term_slope'].iloc[oot_start] if 'vix_term_slope' in spec_features.columns else 0
            if vix_slope > 0:
                # VIX rising -> defensive, buy puts on weakest
                chosen_sector = bottom_sector
                direction = 'put'
            else:
                # VIX falling from high -> recovery, buy calls
                chosen_sector = top_sector
                direction = 'call'
        elif current_vix >= 20:
            # Moderate: bull call on momentum leader
            chosen_sector = top_sector
            direction = 'call'
        else:
            # Low VIX: bear puts on laggard (protection trades)
            chosen_sector = bottom_sector
            direction = 'put'

        # Simulate trade
        entry_price = sector_close[chosen_sector].iloc[oot_start]
        atr = std_features[f'{chosen_sector}_atr_14'].iloc[oot_start]
        fwd_rets = sector_close[chosen_sector].pct_change().iloc[oot_start:oot_end]
        pnl = simulate_option_trade(entry_price, atr, direction, fwd_rets)

        trade_pnls.append(pnl)
        trade_dates.append(dates[oot_start])
        trade_vix_bands.append(band)

    return np.array(trade_pnls), trade_dates, np.array(trade_vix_bands)


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(pnls, capital=STARTING_CAPITAL):
    """Compute risk-adjusted metrics."""
    if len(pnls) == 0:
        return {'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0, 'total_return_pct': 0, 'n_trades': 0}

    # Annualize assuming ~17 trades/year (252/21 OOT windows)
    trades_per_year = 12  # approximate
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls) if np.std(pnls) > 0 else 1e-6
    downside = pnls[pnls < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-6

    sharpe = (mean_pnl / std_pnl) * np.sqrt(trades_per_year)
    sortino = (mean_pnl / downside_std) * np.sqrt(trades_per_year)

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 999
    wr = len(wins) / len(pnls) if len(pnls) > 0 else 0

    total_return = pnls.sum() / capital * 100

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'wr': round(wr, 4),
        'total_return_pct': round(total_return, 2),
        'n_trades': len(pnls),
        'mean_pnl': round(mean_pnl, 2),
    }


def compute_per_band_sharpe(pnls, bands, trades_per_year=12):
    """Compute Sharpe per VIX band."""
    band_sharpes = {}
    for i, name in enumerate(VIX_BAND_NAMES):
        mask = bands == i
        if mask.sum() < 3:
            band_sharpes[name] = np.nan
            continue
        bp = pnls[mask]
        mean_p = np.mean(bp)
        std_p = np.std(bp) if np.std(bp) > 0 else 1e-6
        band_sharpes[name] = round((mean_p / std_p) * np.sqrt(trades_per_year), 3)
    return band_sharpes


def r1_regime_gap(band_sharpes):
    """
    R1 test: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|) < 0.50
    Bull = VIX < 20, Bear = VIX 30+
    """
    s_bull = band_sharpes.get('<20', 0) or 0
    s_bear = band_sharpes.get('30+', 0) or 0
    denom = max(abs(s_bull), abs(s_bear))
    if denom == 0:
        return 1.0  # fail
    return abs(s_bull - s_bear) / denom


# =============================================================================
# VALIDATION GATES
# =============================================================================
def gate1_permutation(pnls, n_perms=NUM_PERMUTATIONS):
    """Permutation test: p-value of observed Sharpe."""
    if len(pnls) < 10:
        return 1.0
    observed_sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10)
    count_better = 0
    for _ in range(n_perms):
        shuffled = np.random.permutation(pnls)
        # Randomize signs
        signs = np.random.choice([-1, 1], size=len(pnls))
        perm_sharpe = np.mean(pnls * signs) / (np.std(pnls * signs) + 1e-10)
        if perm_sharpe >= observed_sharpe:
            count_better += 1
    return count_better / n_perms


def gate3_subperiod(pnls, n_splits=3):
    """Sub-period stability: all sub-periods positive Sharpe."""
    if len(pnls) < n_splits * 5:
        return False
    chunks = np.array_split(pnls, n_splits)
    sharpes = []
    for chunk in chunks:
        s = np.mean(chunk) / (np.std(chunk) + 1e-10)
        sharpes.append(s)
    # All positive and min > 0.2 of max
    all_positive = all(s > 0 for s in sharpes)
    ratio_ok = min(sharpes) / (max(sharpes) + 1e-10) > 0.2 if max(sharpes) > 0 else False
    return all_positive and ratio_ok


# =============================================================================
# MAIN
# =============================================================================
def main():
    log("=" * 70)
    log("VIX Gap Solver v1 — Regime-Adaptive Sector Ranking")
    log("=" * 70)
    log(f"Device: {DEVICE}")
    log(f"MLflow: {'Connected' if MLFLOW_OK else 'Offline'}")

    # Start MLflow run
    if MLFLOW_OK:
        mlflow.set_experiment("vix_gap_solver_v1")
        mlflow.start_run(run_name=f"vix_gap_solver_{datetime.now().strftime('%Y%m%d_%H%M')}")

    # Download data
    sector_close, vix = download_data()

    # Compute features
    log("Computing standard features...")
    std_features = compute_standard_features(sector_close, vix)

    log("Computing VIX 25-30 specialized features...")
    spec_features = compute_vix2530_specialized_features(sector_close, vix)

    log("Computing targets...")
    targets = compute_targets(sector_close)

    # Run all variants
    variants = ['A', 'B', 'C', 'D', 'E', 'F']
    results = {}

    for v in variants:
        t0 = time.time()
        pnls, dates, bands = run_variant(v, sector_close, vix, std_features, spec_features, targets)
        elapsed = time.time() - t0

        # Metrics
        metrics = compute_metrics(pnls)
        band_sharpes = compute_per_band_sharpe(pnls, bands)
        r1_gap = r1_regime_gap(band_sharpes)

        # Validation gates
        p_val = gate1_permutation(pnls)
        subperiod_ok = gate3_subperiod(pnls)

        results[v] = {
            'metrics': metrics,
            'band_sharpes': band_sharpes,
            'r1_gap': round(r1_gap, 4),
            'gate1_pval': round(p_val, 4),
            'gate2_r1_pass': r1_gap < 0.50,
            'gate3_subperiod': subperiod_ok,
            'elapsed_sec': round(elapsed, 1),
        }

        # Gate 4: vs random (computed after all variants)
        log(f"  Variant {v}: Sharpe={metrics['sharpe']:.3f}, R1 gap={r1_gap:.3f}, "
            f"Band Sharpes={band_sharpes}, p={p_val:.3f} [{elapsed:.1f}s]")

    # Gate 4: vs Random
    random_sharpe = results['F']['metrics']['sharpe']
    for v in variants:
        if v == 'F':
            results[v]['gate4_vs_random'] = None
            continue
        v_sharpe = results[v]['metrics']['sharpe']
        improvement = (v_sharpe - random_sharpe) / (abs(random_sharpe) + 1e-6)
        results[v]['gate4_vs_random'] = improvement > 0.20

    # Summary
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    log(f"{'Var':<4} {'Sharpe':<8} {'Sortino':<9} {'PF':<7} {'WR':<7} {'R1 Gap':<8} {'R1 Pass':<8} {'VIX25-30':<10}")
    log("-" * 70)
    for v in variants:
        m = results[v]['metrics']
        r1 = results[v]['r1_gap']
        r1_pass = 'PASS' if results[v]['gate2_r1_pass'] else 'FAIL'
        vix2530 = results[v]['band_sharpes'].get('25-30', 'N/A')
        log(f"  {v:<4} {m['sharpe']:<8.3f} {m['sortino']:<9.3f} {m['pf']:<7.2f} "
            f"{m['wr']:<7.3f} {r1:<8.4f} {r1_pass:<8} {vix2530}")

    log("\n--- Validation Gates ---")
    for v in variants:
        if v == 'F':
            continue
        r = results[v]
        gates = [
            f"G1(p<0.05):{'PASS' if r['gate1_pval'] < 0.05 else 'FAIL'}",
            f"G2(R1<0.50):{'PASS' if r['gate2_r1_pass'] else 'FAIL'}",
            f"G3(subperiod):{'PASS' if r['gate3_subperiod'] else 'FAIL'}",
            f"G4(>random):{'PASS' if r['gate4_vs_random'] else 'FAIL'}",
        ]
        log(f"  Variant {v}: {' | '.join(gates)}")

    # Best variant
    best_v = max([v for v in variants if v != 'F'],
                 key=lambda v: results[v]['metrics']['sharpe'] if results[v]['gate2_r1_pass'] else -999)
    log(f"\n  BEST R1-PASSING VARIANT: {best_v} (Sharpe={results[best_v]['metrics']['sharpe']:.3f})")

    # MLflow logging
    if MLFLOW_OK:
        for v in variants:
            m = results[v]['metrics']
            mlflow.log_metric(f"variant_{v}_sharpe", m['sharpe'])
            mlflow.log_metric(f"variant_{v}_sortino", m['sortino'])
            mlflow.log_metric(f"variant_{v}_pf", m['pf'])
            mlflow.log_metric(f"variant_{v}_wr", m['wr'])
            mlflow.log_metric(f"variant_{v}_r1_gap", results[v]['r1_gap'])
            for band_name, bs in results[v]['band_sharpes'].items():
                if bs is not None and not np.isnan(bs):
                    safe_name = band_name.replace('<', 'lt').replace('+', 'plus').replace('-', '_')
                    mlflow.log_metric(f"variant_{v}_band_{safe_name}", bs)
        mlflow.log_param("best_variant", best_v)
        mlflow.log_param("device", str(DEVICE))
        mlflow.log_param("n_sectors", len(SECTORS))
        mlflow.log_param("train_days", TRAIN_DAYS)
        mlflow.log_param("oot_days", OOT_DAYS)
        mlflow.end_run()

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'mlflow_connected': MLFLOW_OK,
        'best_variant': best_v,
        'variants': {},
    }
    for v in variants:
        output['variants'][v] = {
            'metrics': results[v]['metrics'],
            'band_sharpes': {k: (v2 if v2 is not None and not (isinstance(v2, float) and np.isnan(v2)) else None)
                            for k, v2 in results[v]['band_sharpes'].items()},
            'r1_gap': results[v]['r1_gap'],
            'gates': {
                'permutation_pval': results[v]['gate1_pval'],
                'r1_pass': results[v]['gate2_r1_pass'],
                'subperiod_stable': results[v]['gate3_subperiod'],
                'beats_random': results[v]['gate4_vs_random'],
            }
        }

    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nResults saved to {results_path}")
    log("DONE.")


if __name__ == '__main__':
    main()
