#!/usr/bin/env python3
"""
Momentum-Flow Neural Ranker v1
===============================
MLP-based stock ranker for S&P 500 monthly rebalancing.

ARCHITECTURE: Simple 3-layer MLP (NOT attention — DL Stock Ranker v1 already
uses attention). ListMLE + pairwise ranking loss.

FEATURES:
  - Price momentum (1m, 3m, 6m, 12m, 12-1m classic)
  - Flow features (OBV trend, MFI, volume ratio, A/D line)
  - Quality features (volatility, max drawdown, Sharpe ratio)
  - Value proxy (earnings yield from price-level changes)

STRATEGY: Long top-10 stocks, equal weight, monthly rebalance.
Commission-free (HC #694), 5bps BA spread.

VALIDATION:
  - Walk-forward: 504d train, 21d test, SLIDING (HC #0)
  - Permutation test: 200 shuffles (random rankings must NOT be profitable)
  - R1 regime test: |Sharpe_green - Sharpe_red| / max > 0.50 = FAIL
  - Sub-period stability: CV of Sharpe across 4 sub-periods

GPU: Neptune RTX 3090 (24GB)
"""

import os
import sys
import json
import time
import logging
import warnings
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─── Config ───
CONFIG = {
    'experiment_name': 'momentum_flow_ranker_v1',
    'start_date': '2010-01-01',
    'end_date': '2026-07-23',
    'train_days': 504,        # ~2 years
    'test_days': 21,          # 1 month
    'top_n': 10,              # Long top 10 stocks
    'cost_bps': 5,            # 0.05% BA spread (HC #694: commission-free)
    'rebalance_freq': 21,     # Monthly
    # MLP architecture
    'hidden_dims': [256, 128, 64],
    'dropout': 0.20,
    'lr': 3e-4,
    'weight_decay': 1e-3,
    'epochs': 120,
    'patience': 20,
    'batch_size': 8,
    # Validation
    'n_permutations': 200,
    'regime_gap_max': 0.50,
    'subperiod_cv_max': 0.70,
    # Output
    'output_dir': '/home/nick/Lvl3Quant/output/momentum_flow_ranker_v1',
    'mlflow_tracking_uri': 'http://jupiter:5000',
    'mlflow_experiment': 'momentum_flow_ranker_v1',
    'num_workers': 8,
    'pin_memory': True,
}

OUTPUT_DIR = CONFIG['output_dir']
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'training.log')),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

log.info("=" * 70)
log.info("MOMENTUM-FLOW NEURAL RANKER v1")
log.info("MLP-based stock ranker on Neptune RTX 3090")
log.info(f"Config: {json.dumps({k: v for k, v in CONFIG.items() if not k.startswith('mlflow')}, indent=2)}")
log.info("=" * 70)

# ─── GPU check ───
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    log.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    log.warning("No GPU detected — will be slow")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: DATA DOWNLOAD & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

log.info("\n" + "=" * 70)
log.info("PHASE 0: DATA DOWNLOAD")
log.info("=" * 70)

import yfinance as yf

CACHE_FILE = os.path.join(OUTPUT_DIR, 'price_data_cache.pkl')


def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        sectors = dict(zip(
            df['Symbol'].str.replace('.', '-', regex=False),
            df['GICS Sector']
        ))
        log.info(f"  Got {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers, sectors
    except Exception as e:
        log.warning(f"  Wikipedia failed: {e}, using fallback list")
        tickers = [
            'AAPL','MSFT','AMZN','NVDA','GOOGL','META','TSLA','AVGO','BRK-B','JPM',
            'V','MA','UNH','JNJ','LLY','XOM','PG','HD','COST','ABBV',
            'MRK','PFE','KO','PEP','WMT','BAC','TMO','CRM','CSCO','ABT',
            'MCD','ACN','ADBE','TXN','AMD','DHR','CMCSA','NFLX','WFC','PM',
            'INTC','VZ','T','NEE','DIS','ORCL','IBM','QCOM','INTU','GE',
            'CAT','BA','HON','LOW','UNP','RTX','AMAT','ISRG','GS','SYK',
            'MS','BLK','LMT','TJX','ADI','BKNG','SBUX','AXP','MDLZ','GILD',
            'CVS','CI','SO','DUK','REGN','VRTX','CME','SPGI','PGR','MMC',
            'DE','EMR','ITW','PH','ETN','ADP','LRCX','KLAC','MU','SNPS',
            'CDNS','NOW','PLTR','CRWD','PANW','MRVL','FTNT','SQ','PYPL','UBER',
            'NKE','TGT','ROST','CMG','YUM','MNST','STZ','EL','CL','CLX',
            'GIS','CPB','HRL','DLTR','DG','RCL','HLT','MAR','MGM','LVS',
            'FDX','CSX','NSC','WM','RSG','URI','PWR','GD','NOC','LHX',
            'CARR','TT','AME','DOV','ROK','SWK','XYL','IEX','MMM','SHW',
            'APD','ECL','PPG','DD','LIN','FCX','NEM','FSLR','ENPH','OXY',
            'SLB','EOG','DVN','COP','CVX','PSX','VLO','MPC','HES','HAL',
            'PXD','CTRA','APA','FANG','BKR','XEL','AEP','D','EXC','SRE',
            'ED','WEC','ES','ATO','CMS','CNP','NI','EVRG','AWK','PLD',
            'AMT','CCI','EQIX','PSA','O','SPG','DLR','WELL','VTR','ARE',
            'EQR','AVB','MAA','ESS','UDR','CPT','INVH','PEAK','KIM','REG',
        ]
        return tickers, {}


def download_data(tickers, start, end):
    """Download OHLCV data for all tickers with caching."""
    if os.path.exists(CACHE_FILE):
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if (datetime.now() - cache_mtime).total_seconds() < 86400:
            log.info("  Loading cached price data (< 24h old)")
            return pd.read_pickle(CACHE_FILE)

    log.info(f"  Downloading {len(tickers)} tickers from {start} to {end}...")
    all_data = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            df = yf.download(batch, start=start, end=end, group_by='ticker',
                           auto_adjust=True, threads=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                for t in batch:
                    try:
                        sub = df[t].dropna(subset=['Close'])
                        if len(sub) > 252:
                            all_data[t] = sub[['Open', 'High', 'Low', 'Close', 'Volume']]
                    except:
                        pass
            log.info(f"    Batch {i//batch_size + 1}: {len(batch)} tickers, {len(all_data)} usable")
        except Exception as e:
            log.warning(f"    Batch {i//batch_size + 1} failed: {e}")
        time.sleep(0.5)

    panel = {}
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        panel[col] = pd.DataFrame({t: all_data[t][col] for t in all_data})

    result = {
        'close': panel['Close'],
        'open': panel['Open'],
        'high': panel['High'],
        'low': panel['Low'],
        'volume': panel['Volume'],
    }

    pd.to_pickle(result, CACHE_FILE)
    log.info(f"  Cached {len(all_data)} tickers, {len(panel['Close'])} trading days")
    return result


def compute_features(data):
    """
    Compute cross-sectional features each month-end.
    Returns DataFrame with (date, ticker) rows and feature columns.

    Feature groups:
    MOMENTUM (5):
      1. mom_1m: 1-month return
      2. mom_3m: 3-month return
      3. mom_6m: 6-month return
      4. mom_12m_skip1: 12-1m classic Jegadeesh-Titman momentum
      5. mom_acceleration: change in 3m momentum vs prior month

    FLOW (6):
      6. obv_trend: OBV 20d slope normalized
      7. mfi_14: Money Flow Index 14d
      8. vol_ratio_20_60: 20d/60d average volume
      9. ad_line_trend: A/D line 20d slope normalized
      10. vol_price_corr: 20d volume-price correlation
      11. up_volume_ratio: fraction of volume on up days (20d)

    QUALITY (5):
      12. volatility_21d: 21d annualized vol
      13. max_dd_63d: 63d max drawdown
      14. sharpe_63d: 63d rolling Sharpe ratio
      15. skewness_21d: 21d return skewness
      16. range_ratio: avg (H-L)/C over 21d

    VALUE PROXY (3):
      17. dist_from_52w_high: proximity to 52w high
      18. dist_from_52w_low: proximity to 52w low
      19. earnings_yield_proxy: 12m return / volatility (high = cheap relative to risk)
    """
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    open_px = data['open']

    monthly_dates = close.resample('ME').last().index
    monthly_dates = [d for d in monthly_dates if d in close.index]

    all_features = []

    for i, date in enumerate(monthly_dates):
        if i < 12:
            continue
        if i >= len(monthly_dates) - 1:
            continue

        date_idx = close.index.get_loc(date)
        if date_idx < 252:
            continue

        # Forward 21-day return
        next_date = monthly_dates[i + 1]
        fwd_ret = (close.loc[next_date] / close.loc[date] - 1)

        c = close.iloc[:date_idx + 1]
        h = high.iloc[:date_idx + 1]
        l = low.iloc[:date_idx + 1]
        v = volume.iloc[:date_idx + 1]
        o = open_px.iloc[:date_idx + 1]

        feats = pd.DataFrame(index=close.columns)

        # ── MOMENTUM ──
        feats['mom_1m'] = c.iloc[-1] / c.iloc[-21] - 1
        feats['mom_3m'] = c.iloc[-1] / c.iloc[-63] - 1
        feats['mom_6m'] = c.iloc[-1] / c.iloc[-126] - 1
        feats['mom_12m_skip1'] = c.iloc[-21] / c.iloc[-252] - 1
        mom3m_now = c.iloc[-1] / c.iloc[-63] - 1
        mom3m_prev = c.iloc[-21] / c.iloc[-84] - 1
        feats['mom_acceleration'] = mom3m_now - mom3m_prev

        # ── FLOW ──
        v20 = v.iloc[-20:].mean()
        v60 = v.iloc[-60:].mean()
        feats['vol_ratio_20_60'] = v20 / v60.replace(0, np.nan)

        # OBV trend
        daily_ret_sign = np.sign(c.iloc[-20:].pct_change())
        obv_changes = daily_ret_sign * v.iloc[-20:]
        obv = obv_changes.cumsum()
        x = np.arange(len(obv))
        obv_vals = obv.values
        slopes = []
        for col_idx in range(obv_vals.shape[1]):
            col_data = obv_vals[:, col_idx]
            valid = ~np.isnan(col_data)
            if valid.sum() >= 5:
                slope = np.polyfit(x[valid], col_data[valid], 1)[0]
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        feats['obv_trend'] = slopes
        feats['obv_trend'] = feats['obv_trend'] / v60.replace(0, np.nan).values

        # MFI 14d
        typical_price = (c.iloc[-14:] + h.iloc[-14:] + l.iloc[-14:]) / 3
        raw_money_flow = typical_price * v.iloc[-14:]
        tp_change = typical_price.diff()
        pos_flow = raw_money_flow.where(tp_change > 0, 0).sum()
        neg_flow = raw_money_flow.where(tp_change < 0, 0).sum()
        mfr = pos_flow / neg_flow.replace(0, np.nan)
        feats['mfi_14'] = 100 - (100 / (1 + mfr))

        # A/D line trend
        clv = ((c.iloc[-20:] - l.iloc[-20:]) - (h.iloc[-20:] - c.iloc[-20:])) / \
              (h.iloc[-20:] - l.iloc[-20:]).replace(0, np.nan)
        ad = (clv * v.iloc[-20:]).cumsum()
        ad_slopes = []
        ad_vals = ad.values
        for col_idx in range(ad_vals.shape[1]):
            col_data = ad_vals[:, col_idx]
            valid = ~np.isnan(col_data)
            if valid.sum() >= 5:
                slope = np.polyfit(x[valid][:valid.sum()], col_data[valid], 1)[0]
                ad_slopes.append(slope)
            else:
                ad_slopes.append(np.nan)
        feats['ad_line_trend'] = ad_slopes
        feats['ad_line_trend'] = feats['ad_line_trend'] / v60.replace(0, np.nan).values

        # Volume-price correlation
        vol_chg = v.iloc[-20:].pct_change()
        px_chg = c.iloc[-20:].pct_change()
        feats['vol_price_corr'] = vol_chg.corrwith(px_chg)

        # Up-volume ratio
        up_days = (c.iloc[-20:].pct_change() > 0)
        up_vol = v.iloc[-20:].where(up_days, 0).sum()
        total_vol = v.iloc[-20:].sum()
        feats['up_volume_ratio'] = up_vol / total_vol.replace(0, np.nan)

        # ── QUALITY ──
        rets_21 = c.iloc[-21:].pct_change().dropna()
        feats['volatility_21d'] = rets_21.std() * np.sqrt(252)

        # Max drawdown 63d
        rolling_max = c.iloc[-63:].cummax()
        dd = c.iloc[-63:] / rolling_max - 1
        feats['max_dd_63d'] = dd.min()

        # Sharpe 63d (rolling)
        rets_63 = c.iloc[-63:].pct_change().dropna()
        feats['sharpe_63d'] = (rets_63.mean() / rets_63.std()) * np.sqrt(252)

        # Skewness
        feats['skewness_21d'] = rets_21.skew()

        # Range ratio
        feats['range_ratio'] = ((h.iloc[-21:] - l.iloc[-21:]) / c.iloc[-21:]).mean()

        # ── VALUE PROXY ──
        high_52w = c.iloc[-252:].max()
        low_52w = c.iloc[-252:].min()
        feats['dist_from_52w_high'] = c.iloc[-1] / high_52w - 1
        feats['dist_from_52w_low'] = c.iloc[-1] / low_52w - 1

        # Earnings yield proxy: 12m return / volatility
        ret_12m = c.iloc[-1] / c.iloc[-252] - 1
        vol_12m = c.iloc[-252:].pct_change().dropna().std() * np.sqrt(252)
        feats['earnings_yield_proxy'] = ret_12m / vol_12m.replace(0, np.nan)

        feats['date'] = date
        feats['fwd_ret_21d'] = fwd_ret
        feats['ticker'] = feats.index

        all_features.append(feats.reset_index(drop=True))

    panel = pd.concat(all_features, ignore_index=True)
    feature_cols = [c for c in panel.columns if c not in ['date', 'ticker', 'fwd_ret_21d']]
    log.info(f"  Feature panel: {len(panel)} stock-months, {len(feature_cols)} features")
    log.info(f"  Date range: {panel['date'].min()} to {panel['date'].max()}")
    log.info(f"  Features: {feature_cols}")

    return panel, feature_cols


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: MLP RANKER MODEL
# ═══════════════════════════════════════════════════════════════════════

class MLPRanker(nn.Module):
    """
    Simple 3-layer MLP that scores each stock independently.
    No cross-stock attention — relies on rank-normalized features
    to capture relative positioning.
    """
    def __init__(self, n_features, hidden_dims=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))  # Single score output
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        """
        x: (B, S, F) — batch of cross-sectional snapshots
        mask: (B, S) — True = invalid/padded
        Returns: (B, S) scores
        """
        B, S, F = x.shape
        # Reshape for BatchNorm: (B*S, F)
        x_flat = x.reshape(B * S, F)
        scores_flat = self.net(x_flat)  # (B*S, 1)
        scores = scores_flat.reshape(B, S)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        return scores


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: RANKING LOSSES
# ═══════════════════════════════════════════════════════════════════════

def listMLE_loss(scores, relevance, mask=None):
    """
    ListMLE: likelihood of the permutation defined by relevance labels.
    scores: (B, S) — model predictions
    relevance: (B, S) — true rankings (higher = better)
    mask: (B, S) — True = invalid
    """
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
        relevance = relevance.masked_fill(mask, float('-inf'))

    _, sorted_idx = relevance.sort(dim=1, descending=True)
    sorted_scores = scores.gather(1, sorted_idx)

    B, S = sorted_scores.shape
    loss = torch.zeros(B, device=scores.device)

    for i in range(S - 1):
        remaining = sorted_scores[:, i:]
        log_softmax = remaining[:, 0] - torch.logsumexp(remaining, dim=1)
        if mask is not None:
            valid = ~mask.gather(1, sorted_idx)[:, i]
            loss = loss - log_softmax * valid.float()
        else:
            loss = loss - log_softmax

    return loss.mean()


def pairwise_ranking_loss(scores, relevance, mask=None, margin=0.1):
    """Pairwise margin ranking loss with sampling for efficiency."""
    B, S = scores.shape
    n_pairs = min(S * 2, 200)

    total_loss = torch.zeros(1, device=scores.device)
    count = 0

    for b in range(B):
        valid_mask = ~mask[b] if mask is not None else torch.ones(S, dtype=torch.bool, device=scores.device)
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        if len(valid_idx) < 2:
            continue

        n_valid = len(valid_idx)
        idx_i = valid_idx[torch.randint(n_valid, (n_pairs,), device=scores.device)]
        idx_j = valid_idx[torch.randint(n_valid, (n_pairs,), device=scores.device)]

        rel_diff = relevance[b, idx_i] - relevance[b, idx_j]
        keep = rel_diff > 0

        if keep.sum() == 0:
            continue

        score_diff = scores[b, idx_i[keep]] - scores[b, idx_j[keep]]
        pair_loss = torch.clamp(margin - score_diff, min=0)
        total_loss = total_loss + pair_loss.mean()
        count += 1

    return total_loss / max(count, 1)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: WALK-FORWARD TRAINING
# ═══════════════════════════════════════════════════════════════════════

def prepare_cross_sectional_batch(panel, feature_cols, dates, max_stocks=300):
    """
    Prepare a batch of cross-sectional snapshots.
    Each snapshot: (n_stocks, n_features) at a given date.
    """
    batches_x = []
    batches_y = []
    batches_mask = []
    batches_tickers = []

    for date in dates:
        df = panel[panel['date'] == date].copy()
        df = df.dropna(subset=feature_cols + ['fwd_ret_21d'])

        if len(df) < 20:
            continue

        if len(df) > max_stocks:
            df = df.iloc[:max_stocks]

        # Cross-sectional rank-normalize (robust to outliers)
        X = df[feature_cols].values.copy()
        for j in range(X.shape[1]):
            col = X[:, j]
            valid = ~np.isnan(col)
            if valid.sum() > 5:
                ranks = np.zeros_like(col)
                ranks[valid] = pd.Series(col[valid]).rank(pct=True).values
                ranks[~valid] = 0.5
                X[:, j] = ranks
            else:
                X[:, j] = 0.5

        y = df['fwd_ret_21d'].values
        y_rank = pd.Series(y).rank(pct=True).values

        n = len(df)
        X_padded = np.zeros((max_stocks, len(feature_cols)), dtype=np.float32)
        y_padded = np.zeros(max_stocks, dtype=np.float32)
        mask = np.ones(max_stocks, dtype=bool)

        X_padded[:n] = X
        y_padded[:n] = y_rank
        mask[:n] = False

        batches_x.append(X_padded)
        batches_y.append(y_padded)
        batches_mask.append(mask)
        batches_tickers.append(df['ticker'].tolist())

    if not batches_x:
        return None, None, None, None

    X_tensor = torch.FloatTensor(np.array(batches_x))
    y_tensor = torch.FloatTensor(np.array(batches_y))
    mask_tensor = torch.BoolTensor(np.array(batches_mask))

    return X_tensor, y_tensor, mask_tensor, batches_tickers


def train_fold(model, optimizer, scheduler, X_train, y_train, mask_train,
               X_val, y_val, mask_val, epochs, patience, device):
    """Train one walk-forward fold with early stopping."""
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        idx = torch.randperm(len(X_train))
        batch_size = CONFIG['batch_size']

        for start in range(0, len(X_train), batch_size):
            batch_idx = idx[start:start + batch_size]
            x_b = X_train[batch_idx].to(device)
            y_b = y_train[batch_idx].to(device)
            m_b = mask_train[batch_idx].to(device)

            optimizer.zero_grad()
            scores = model(x_b, m_b)

            loss_list = listMLE_loss(scores, y_b, m_b)
            loss_pair = pairwise_ranking_loss(scores, y_b, m_b)
            loss = loss_list + 0.5 * loss_pair

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        with torch.no_grad():
            x_v = X_val.to(device)
            y_v = y_val.to(device)
            m_v = mask_val.to(device)
            val_scores = model(x_v, m_v)
            val_loss = listMLE_loss(val_scores, y_v, m_v).item()

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss


def run_walk_forward(panel, feature_cols):
    """
    Walk-forward with SLIDING window (HC #0).
    504d train, 21d test.
    """
    dates = sorted(panel['date'].unique())
    n_dates = len(dates)

    train_periods = CONFIG['train_days'] // CONFIG['test_days']  # ~24 months
    test_periods = 1

    log.info(f"\nWalk-forward: {n_dates} monthly periods, {train_periods} train, {test_periods} test")

    all_predictions = []
    n_features = len(feature_cols)

    for fold_start in range(train_periods, n_dates - test_periods + 1):
        fold_num = fold_start - train_periods
        train_dates = dates[fold_start - train_periods:fold_start]
        test_dates = dates[fold_start:fold_start + test_periods]

        X_train, y_train, mask_train, _ = prepare_cross_sectional_batch(
            panel, feature_cols, train_dates, max_stocks=300
        )
        X_test, y_test, mask_test, test_tickers = prepare_cross_sectional_batch(
            panel, feature_cols, test_dates, max_stocks=300
        )

        if X_train is None or X_test is None:
            continue

        # Train/val split
        val_size = max(2, len(X_train) // 5)
        if val_size >= len(X_train) - 2:
            val_size = max(1, len(X_train) // 3)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        mask_val = mask_train[-val_size:]
        X_tr = X_train[:-val_size]
        y_tr = y_train[:-val_size]
        mask_tr = mask_train[:-val_size]

        if len(X_tr) < 2:
            continue

        # Fresh model each fold
        model = MLPRanker(
            n_features=n_features,
            hidden_dims=CONFIG['hidden_dims'],
            dropout=CONFIG['dropout'],
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG['lr'],
            weight_decay=CONFIG['weight_decay'],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
        )

        val_loss = train_fold(
            model, optimizer, scheduler,
            X_tr, y_tr, mask_tr,
            X_val, y_val, mask_val,
            CONFIG['epochs'], CONFIG['patience'], device,
        )

        # Predict on test
        model.eval()
        with torch.no_grad():
            x_t = X_test.to(device)
            m_t = mask_test.to(device)
            test_scores = model(x_t, m_t).cpu()

        for batch_i in range(len(X_test)):
            test_date = test_dates[batch_i] if batch_i < len(test_dates) else test_dates[-1]
            tickers = test_tickers[batch_i]
            n_valid = len(tickers)

            scores = test_scores[batch_i, :n_valid].numpy()
            true_rets = panel[panel['date'] == test_date].set_index('ticker').loc[tickers, 'fwd_ret_21d'].values

            pred_df = pd.DataFrame({
                'date': test_date,
                'ticker': tickers,
                'score': scores,
                'fwd_ret_21d': true_rets,
            })
            all_predictions.append(pred_df)

        if fold_num % 10 == 0:
            log.info(f"  Fold {fold_num}: train={train_dates[0].strftime('%Y-%m-%d')} to "
                     f"{train_dates[-1].strftime('%Y-%m-%d')}, "
                     f"test={test_dates[0].strftime('%Y-%m-%d')}, val_loss={val_loss:.4f}")

    predictions = pd.concat(all_predictions, ignore_index=True)
    log.info(f"  Total predictions: {len(predictions)} stock-months across {predictions['date'].nunique()} periods")

    return predictions


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: PORTFOLIO CONSTRUCTION & METRICS
# ═══════════════════════════════════════════════════════════════════════

def build_portfolio_top10(predictions):
    """
    Build monthly-rebalanced portfolio: long top-10 stocks, equal weight.
    Benchmark: equal-weight all stocks (market proxy).
    """
    cost_bps = CONFIG['cost_bps']
    top_n = CONFIG['top_n']

    dates = sorted(predictions['date'].unique())
    port_rets = []
    prev_holdings = set()

    for date in dates:
        df = predictions[predictions['date'] == date].copy()
        df = df.dropna(subset=['score', 'fwd_ret_21d'])

        if len(df) < 20:
            continue

        # Top N by score
        df_sorted = df.nlargest(top_n, 'score')
        holdings = set(df_sorted['ticker'].tolist())

        # Portfolio return (equal weight)
        port_ret = df_sorted['fwd_ret_21d'].mean()

        # Benchmark: equal-weight all stocks
        bench_ret = df['fwd_ret_21d'].mean()

        # Turnover
        if prev_holdings:
            turnover = len(holdings - prev_holdings) / top_n
        else:
            turnover = 1.0

        # Cost: turnover * cost_bps * 2 (buy + sell)
        cost = turnover * cost_bps / 10000 * 2
        net_ret = port_ret - cost

        port_rets.append({
            'date': date,
            'gross_ret': port_ret,
            'net_ret': net_ret,
            'bench_ret': bench_ret,
            'excess_ret': net_ret - bench_ret,
            'turnover': turnover,
            'holdings': list(holdings),
            'n_total': len(df),
        })

        prev_holdings = holdings

    return pd.DataFrame(port_rets)


def build_portfolio_ls(predictions):
    """Long-short portfolio for validation: long top quintile, short bottom quintile."""
    cost_bps = CONFIG['cost_bps']
    top_pct = 0.20
    bot_pct = 0.20

    dates = sorted(predictions['date'].unique())
    port_rets = []
    prev_longs = set()
    prev_shorts = set()

    for date in dates:
        df = predictions[predictions['date'] == date].copy()
        df = df.dropna(subset=['score', 'fwd_ret_21d'])
        if len(df) < 20:
            continue

        df['rank'] = df['score'].rank(pct=True)
        long_mask = df['rank'] >= (1 - top_pct)
        short_mask = df['rank'] <= bot_pct

        longs = set(df[long_mask]['ticker'].tolist())
        shorts = set(df[short_mask]['ticker'].tolist())

        long_turnover = len(longs - prev_longs) / max(len(longs), 1)
        short_turnover = len(shorts - prev_shorts) / max(len(shorts), 1)
        avg_turnover = (long_turnover + short_turnover) / 2

        long_ret = df[long_mask]['fwd_ret_21d'].mean()
        short_ret = df[short_mask]['fwd_ret_21d'].mean()
        gross_ret = (long_ret - short_ret) / 2

        cost = avg_turnover * cost_bps / 10000 * 2
        net_ret = gross_ret - cost

        port_rets.append({
            'date': date,
            'gross_ret': gross_ret,
            'net_ret': net_ret,
            'long_ret': long_ret,
            'short_ret': short_ret,
            'turnover': avg_turnover,
            'n_longs': long_mask.sum(),
            'n_shorts': short_mask.sum(),
            'n_total': len(df),
        })

        prev_longs = longs
        prev_shorts = shorts

    return pd.DataFrame(port_rets)


def compute_metrics(port_df, ret_col='net_ret'):
    """Compute risk-adjusted performance metrics."""
    rets = port_df[ret_col].values
    n = len(rets)

    if n < 5:
        return {}

    ann_factor = 12
    mean_ret = rets.mean()
    std_ret = rets.std()

    sharpe = mean_ret / std_ret * np.sqrt(ann_factor) if std_ret > 0 else 0

    downside = rets[rets < 0]
    downside_std = downside.std() if len(downside) > 0 else std_ret
    sortino = mean_ret / downside_std * np.sqrt(ann_factor) if downside_std > 0 else 0

    cum_ret = (1 + rets).cumprod()
    max_dd = (cum_ret / np.maximum.accumulate(cum_ret) - 1).min()
    cagr = cum_ret[-1] ** (ann_factor / n) - 1
    win_rate = (rets > 0).mean()

    pos_sum = rets[rets > 0].sum()
    neg_sum = abs(rets[rets < 0].sum())
    pf = pos_sum / neg_sum if neg_sum > 0 else float('inf')

    vol = std_ret * np.sqrt(ann_factor)

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'wr': round(win_rate, 4),
        'cagr': round(cagr, 4),
        'max_dd': round(max_dd, 4),
        'vol': round(vol, 4),
        'n_periods': n,
        'mean_monthly_ret': round(mean_ret, 4),
        'total_ret': round(cum_ret[-1] - 1, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: VALIDATION GATES
# ═══════════════════════════════════════════════════════════════════════

def permutation_test(predictions, n_perms=200):
    """
    Permutation test: shuffle model SCORES across stocks each period.
    If model has no edge, shuffled scores yield same Sharpe.
    """
    log.info(f"\nRunning {n_perms}-shuffle permutation test...")

    port = build_portfolio_top10(predictions)
    observed_sharpe = compute_metrics(port)['sharpe']

    null_sharpes = []
    for perm_i in range(n_perms):
        shuffled = predictions.copy()
        for date in shuffled['date'].unique():
            mask = shuffled['date'] == date
            scores = shuffled.loc[mask, 'score'].values.copy()
            np.random.shuffle(scores)
            shuffled.loc[mask, 'score'] = scores

        perm_port = build_portfolio_top10(shuffled)
        perm_metrics = compute_metrics(perm_port)
        if perm_metrics:
            null_sharpes.append(perm_metrics['sharpe'])

        if (perm_i + 1) % 50 == 0:
            log.info(f"  Permutation {perm_i + 1}/{n_perms}...")

    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= observed_sharpe).mean()

    result = {
        'observed_sharpe': observed_sharpe,
        'null_mean': round(float(null_sharpes.mean()), 4),
        'null_std': round(float(null_sharpes.std()), 4),
        'p_value': round(float(p_value), 4),
        'pass': p_value < 0.05,
    }

    log.info(f"  Permutation test: p={p_value:.3f}, observed={observed_sharpe:.3f}, "
             f"null={null_sharpes.mean():.3f} +/- {null_sharpes.std():.3f}")
    log.info(f"  PASS: {result['pass']}")

    return result


def regime_test(port_df):
    """R1: Regime-agnostic validation. Split by SPY monthly return (green/red)."""
    import yfinance as yf

    log.info("\nRunning regime test (R1)...")

    spy = yf.download('SPY', start='2009-01-01', end='2027-01-01', auto_adjust=True, progress=False)
    spy_monthly = spy['Close'].resample('ME').last().pct_change().dropna()

    green_dates = set()
    red_dates = set()

    for d in port_df['date']:
        closest = spy_monthly.index[spy_monthly.index.get_indexer([d], method='nearest')[0]]
        val = spy_monthly.loc[closest]
        if hasattr(val, 'item'):
            val = val.item()
        elif hasattr(val, 'iloc'):
            val = val.iloc[0]
        if val > 0:
            green_dates.add(d)
        else:
            red_dates.add(d)

    green_rets = port_df[port_df['date'].isin(green_dates)]['net_ret'].values
    red_rets = port_df[port_df['date'].isin(red_dates)]['net_ret'].values

    sharpe_green = green_rets.mean() / green_rets.std() * np.sqrt(12) if len(green_rets) > 3 and green_rets.std() > 0 else 0
    sharpe_red = red_rets.mean() / red_rets.std() * np.sqrt(12) if len(red_rets) > 3 and red_rets.std() > 0 else 0

    gap = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red), 0.001)

    result = {
        'sharpe_green': round(sharpe_green, 3),
        'sharpe_red': round(sharpe_red, 3),
        'n_green': len(green_rets),
        'n_red': len(red_rets),
        'regime_gap': round(gap, 4),
        'pass': gap <= CONFIG['regime_gap_max'],
    }

    log.info(f"  Regime: green Sharpe={sharpe_green:.3f} ({len(green_rets)}m), "
             f"red Sharpe={sharpe_red:.3f} ({len(red_rets)}m), gap={gap:.3f}")
    log.info(f"  PASS: {result['pass']}")

    return result


def subperiod_test(port_df):
    """Sub-period stability: split into 4 equal periods, check CV of Sharpe."""
    log.info("\nRunning sub-period stability test...")

    n = len(port_df)
    quarter = n // 4

    sharpes = []
    for q in range(4):
        start = q * quarter
        end = (q + 1) * quarter if q < 3 else n
        rets = port_df.iloc[start:end]['net_ret'].values
        s = rets.mean() / rets.std() * np.sqrt(12) if len(rets) > 3 and rets.std() > 0 else 0
        sharpes.append(round(s, 3))

    mean_s = np.mean(sharpes)
    std_s = np.std(sharpes)
    cv = std_s / abs(mean_s) if abs(mean_s) > 0.001 else float('inf')

    result = {
        'sharpes': sharpes,
        'mean': round(float(mean_s), 3),
        'std': round(float(std_s), 3),
        'cv': round(float(cv), 4),
        'pass': cv <= CONFIG['subperiod_cv_max'],
    }

    log.info(f"  Sub-period Sharpes: {sharpes}, CV={cv:.3f}")
    log.info(f"  PASS: {result['pass']}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6: LGBM BASELINE
# ═══════════════════════════════════════════════════════════════════════

def run_lgbm_baseline(panel, feature_cols):
    """LightGBM baseline for comparison."""
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("LightGBM not available, skipping baseline")
        return None

    log.info("\nRunning LightGBM baseline...")

    dates = sorted(panel['date'].unique())
    n_dates = len(dates)
    train_periods = CONFIG['train_days'] // CONFIG['test_days']

    all_preds = []

    for fold_start in range(train_periods, n_dates - 1 + 1):
        train_dates = dates[fold_start - train_periods:fold_start]
        test_dates = dates[fold_start:fold_start + 1]

        train_df = panel[panel['date'].isin(train_dates)].dropna(subset=feature_cols + ['fwd_ret_21d'])
        test_df = panel[panel['date'].isin(test_dates)].dropna(subset=feature_cols + ['fwd_ret_21d'])

        if len(train_df) < 100 or len(test_df) < 20:
            continue

        X_tr = train_df[feature_cols].values
        y_tr = train_df['fwd_ret_21d'].values
        X_te = test_df[feature_cols].values

        # Handle NaN
        X_tr = np.nan_to_num(X_tr, 0)
        X_te = np.nan_to_num(X_te, 0)

        dtrain = lgb.Dataset(X_tr, label=y_tr)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbosity': -1,
        }

        model = lgb.train(params, dtrain, num_boost_round=200)
        preds = model.predict(X_te)

        pred_df = pd.DataFrame({
            'date': test_df['date'].values,
            'ticker': test_df['ticker'].values,
            'score': preds,
            'fwd_ret_21d': test_df['fwd_ret_21d'].values,
        })
        all_preds.append(pred_df)

    if not all_preds:
        return None

    predictions = pd.concat(all_preds, ignore_index=True)
    log.info(f"  LGBM predictions: {len(predictions)} stock-months across {predictions['date'].nunique()} periods")
    return predictions


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()

    # --- Step 1: Download data ---
    log.info("\n" + "=" * 70)
    log.info("STEP 1: DATA DOWNLOAD")
    log.info("=" * 70)

    tickers, sectors = get_sp500_tickers()
    data = download_data(tickers, CONFIG['start_date'], CONFIG['end_date'])

    log.info(f"  Data: {len(data['close'].columns)} stocks, "
             f"{len(data['close'])} days ({data['close'].index[0]} to {data['close'].index[-1]})")

    # --- Step 2: Feature engineering ---
    log.info("\n" + "=" * 70)
    log.info("STEP 2: FEATURE ENGINEERING")
    log.info("=" * 70)

    panel, feature_cols = compute_features(data)
    panel.to_parquet(os.path.join(OUTPUT_DIR, 'feature_panel.parquet'), index=False)

    # --- Step 3: MLP Walk-Forward ---
    log.info("\n" + "=" * 70)
    log.info("STEP 3: MLP RANKER WALK-FORWARD")
    log.info("=" * 70)

    mlp_predictions = run_walk_forward(panel, feature_cols)
    mlp_predictions.to_csv(os.path.join(OUTPUT_DIR, 'mlp_predictions.csv'), index=False)

    # --- Step 4: LGBM Baseline ---
    log.info("\n" + "=" * 70)
    log.info("STEP 4: LGBM BASELINE")
    log.info("=" * 70)

    lgbm_predictions = run_lgbm_baseline(panel, feature_cols)
    if lgbm_predictions is not None:
        lgbm_predictions.to_csv(os.path.join(OUTPUT_DIR, 'lgbm_predictions.csv'), index=False)

    # --- Step 5: Portfolio construction ---
    log.info("\n" + "=" * 70)
    log.info("STEP 5: PORTFOLIO CONSTRUCTION")
    log.info("=" * 70)

    # Top-10 long-only portfolios (primary strategy)
    mlp_top10 = build_portfolio_top10(mlp_predictions)
    mlp_top10_metrics = compute_metrics(mlp_top10)

    # L/S portfolios (for validation)
    mlp_ls = build_portfolio_ls(mlp_predictions)
    mlp_ls_metrics = compute_metrics(mlp_ls)

    log.info("\n  MLP Top-10 Long-Only:")
    for k, v in mlp_top10_metrics.items():
        log.info(f"    {k}: {v}")

    log.info("\n  MLP L/S (for validation):")
    for k, v in mlp_ls_metrics.items():
        log.info(f"    {k}: {v}")

    lgbm_top10_metrics = {}
    lgbm_ls_metrics = {}
    lgbm_top10 = None
    lgbm_ls = None

    if lgbm_predictions is not None:
        lgbm_top10 = build_portfolio_top10(lgbm_predictions)
        lgbm_top10_metrics = compute_metrics(lgbm_top10)
        lgbm_ls = build_portfolio_ls(lgbm_predictions)
        lgbm_ls_metrics = compute_metrics(lgbm_ls)

        log.info("\n  LGBM Top-10 Long-Only:")
        for k, v in lgbm_top10_metrics.items():
            log.info(f"    {k}: {v}")

        log.info("\n  LGBM L/S:")
        for k, v in lgbm_ls_metrics.items():
            log.info(f"    {k}: {v}")

    # --- Step 6: Validation gates ---
    log.info("\n" + "=" * 70)
    log.info("STEP 6: VALIDATION GATES")
    log.info("=" * 70)

    # Use top-10 portfolio for permutation test, L/S for regime test
    perm_result = permutation_test(mlp_predictions, CONFIG['n_permutations'])
    regime_result = regime_test(mlp_top10)
    subperiod_result = subperiod_test(mlp_top10)

    gates_passed = sum([
        perm_result['pass'],
        regime_result['pass'],
        subperiod_result['pass'],
    ])

    log.info(f"\n  Validation: {gates_passed}/3 gates passed")
    log.info(f"    Permutation: {'PASS' if perm_result['pass'] else 'FAIL'} (p={perm_result['p_value']})")
    log.info(f"    Regime (R1): {'PASS' if regime_result['pass'] else 'FAIL'} (gap={regime_result['regime_gap']})")
    log.info(f"    Stability:   {'PASS' if subperiod_result['pass'] else 'FAIL'} (CV={subperiod_result['cv']})")

    # --- Step 7: Excess return analysis ---
    log.info("\n" + "=" * 70)
    log.info("STEP 7: EXCESS RETURN ANALYSIS")
    log.info("=" * 70)

    if 'excess_ret' in mlp_top10.columns:
        excess_metrics = compute_metrics(mlp_top10, 'excess_ret')
        log.info("  MLP Top-10 vs Equal-Weight Benchmark (excess returns):")
        for k, v in excess_metrics.items():
            log.info(f"    {k}: {v}")

    # --- Step 8: Save results ---
    log.info("\n" + "=" * 70)
    log.info("STEP 8: SAVING RESULTS")
    log.info("=" * 70)

    runtime_min = (time.time() - start_time) / 60

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {k: v for k, v in CONFIG.items() if not k.startswith('mlflow')},
        'metrics': {
            'mlp_top10': mlp_top10_metrics,
            'mlp_long_short': mlp_ls_metrics,
            'lgbm_top10': lgbm_top10_metrics,
            'lgbm_long_short': lgbm_ls_metrics,
        },
        'excess_return_metrics': excess_metrics if 'excess_ret' in mlp_top10.columns else {},
        'validation': {
            'permutation_test': perm_result,
            'regime_test': regime_result,
            'subperiod_stability': subperiod_result,
            'gates_passed': f"{gates_passed}/3",
        },
        'runtime_minutes': round(runtime_min, 1),
    }

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    mlp_top10.to_csv(os.path.join(OUTPUT_DIR, 'mlp_top10_portfolio.csv'), index=False)
    mlp_ls.to_csv(os.path.join(OUTPUT_DIR, 'mlp_ls_portfolio.csv'), index=False)
    if lgbm_top10 is not None:
        lgbm_top10.to_csv(os.path.join(OUTPUT_DIR, 'lgbm_top10_portfolio.csv'), index=False)

    log.info(f"  Results saved")
    log.info(f"  Runtime: {runtime_min:.1f} minutes")

    # --- Step 9: MLflow logging ---
    log.info("\n" + "=" * 70)
    log.info("STEP 9: MLFLOW LOGGING")
    log.info("=" * 70)

    try:
        import mlflow
        mlflow.set_tracking_uri(CONFIG['mlflow_tracking_uri'])
        mlflow.set_experiment(CONFIG['mlflow_experiment'])

        with mlflow.start_run(run_name=f"momentum_flow_ranker_v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            for k, v in CONFIG.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(k, v)
                elif isinstance(v, list):
                    mlflow.log_param(k, str(v))

            # MLP top-10 metrics
            for k, v in mlp_top10_metrics.items():
                mlflow.log_metric(f"mlp_top10_{k}", v)
            for k, v in mlp_ls_metrics.items():
                mlflow.log_metric(f"mlp_ls_{k}", v)

            if lgbm_top10_metrics:
                for k, v in lgbm_top10_metrics.items():
                    mlflow.log_metric(f"lgbm_top10_{k}", v)

            mlflow.log_metric("perm_p_value", perm_result['p_value'])
            mlflow.log_metric("regime_gap", regime_result['regime_gap'])
            mlflow.log_metric("subperiod_cv", subperiod_result['cv'])
            mlflow.log_metric("gates_passed", gates_passed)

            mlflow.log_artifact(os.path.join(OUTPUT_DIR, 'results.json'))

            log.info("  MLflow run logged successfully")
    except Exception as e:
        log.warning(f"  MLflow logging failed: {e}")

    # --- Summary ---
    log.info("\n" + "=" * 70)
    log.info("SUMMARY — MOMENTUM-FLOW NEURAL RANKER v1")
    log.info("=" * 70)
    log.info(f"  Architecture: MLP {CONFIG['hidden_dims']}")
    log.info(f"  Strategy: Long top-{CONFIG['top_n']} stocks, equal weight, monthly rebalance")
    log.info(f"  Top-10 Sharpe: {mlp_top10_metrics.get('sharpe', 'N/A')}")
    log.info(f"  Top-10 Sortino: {mlp_top10_metrics.get('sortino', 'N/A')}")
    log.info(f"  Top-10 CAGR: {mlp_top10_metrics.get('cagr', 'N/A')}")
    log.info(f"  Top-10 MaxDD: {mlp_top10_metrics.get('max_dd', 'N/A')}")
    log.info(f"  Top-10 WR: {mlp_top10_metrics.get('wr', 'N/A')}")
    log.info(f"  Top-10 PF: {mlp_top10_metrics.get('pf', 'N/A')}")
    log.info(f"  L/S Sharpe: {mlp_ls_metrics.get('sharpe', 'N/A')}")
    log.info(f"  Validation: {gates_passed}/3 gates")
    log.info(f"  Perm p-value: {perm_result.get('p_value', 'N/A')}")
    log.info(f"  Regime gap: {regime_result.get('regime_gap', 'N/A')}")
    log.info(f"  Runtime: {runtime_min:.1f} minutes")
    log.info("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(f"FATAL: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)
