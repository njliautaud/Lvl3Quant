#!/usr/bin/env python3
"""
DL Momentum-Flow Fusion v1
===========================
CONTEXT: DL Stock Ranker v1 FAILED permutation test (p=1.0, null_std=0).
The permutation was broken — shuffled features gave identical Sharpe.
This v2 approach fixes the critical issues:

1. PROPER permutation test: shuffle stock RANKINGS cross-sectionally per period
   (not features), ensuring the null distribution actually varies.
2. MARKET-HEDGED L/S: Long top quintile, short bottom quintile — removes market beta.
3. MOMENTUM + FLOW FUSION: Transformer attention over momentum signals fused with
   volume/flow features (OBV trend, volume ratio, money flow, accumulation/distribution).
4. Walk-forward 504d/21d SLIDING (HC #0). Commission-free stocks (HC #694) + 0.05% BA spread.

ARCHITECTURE:
  - Input: N_stocks × F_features cross-sectional snapshot each month
  - StockEncoder: per-stock MLP to embed features
  - CrossStockAttention: multi-head attention to learn relative relationships
  - RankHead: predicts next-month relative return rank
  - Loss: ListMLE (learning-to-rank) + pairwise margin ranking loss

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
    'experiment_name': 'dl_momentum_flow_v1',
    'start_date': '2010-01-01',
    'end_date': '2026-07-18',
    'train_days': 504,        # ~2 years
    'test_days': 21,          # 1 month
    'top_pct': 0.20,          # Long top 20%
    'bot_pct': 0.20,          # Short bottom 20%
    'cost_bps': 5,            # 0.05% BA spread (HC #694: commission-free)
    'rebalance_freq': 21,     # Monthly
    # Model architecture
    'd_model': 128,
    'n_heads': 4,
    'n_encoder_layers': 2,
    'ff_dim': 256,
    'dropout': 0.15,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'epochs': 100,
    'patience': 15,
    'batch_size': 4,  # Smaller batch for small-sample regime
    # Validation
    'n_permutations': 200,
    'regime_gap_max': 0.50,
    'subperiod_cv_max': 0.70,
    # Output
    'output_dir': '/home/nick/Lvl3Quant/output/dl_momentum_flow_v1',
    'mlflow_tracking_uri': 'http://jupiter:5000',
    'mlflow_experiment': 'dl_momentum_flow_v1',
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
log.info("DL MOMENTUM-FLOW FUSION v1")
log.info("GPU-accelerated PyTorch Transformer on Neptune RTX 3090")
log.info(f"Config: {json.dumps({k: v for k, v in CONFIG.items() if not k.startswith('mlflow')}, indent=2)}")
log.info("=" * 70)

# ─── GPU check ───
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    log.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
        # Fallback: top 200 liquid large caps
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
    """Download OHLCV data for all tickers."""
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
                        if len(sub) > 252:  # Need at least 1 year
                            all_data[t] = sub[['Open', 'High', 'Low', 'Close', 'Volume']]
                    except:
                        pass
            log.info(f"    Batch {i//batch_size + 1}: {len(batch)} tickers, {len(all_data)} usable")
        except Exception as e:
            log.warning(f"    Batch {i//batch_size + 1} failed: {e}")
        time.sleep(0.5)

    # Combine into panel
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
    Compute cross-sectional features for each month-end.
    Returns DataFrame with (date, ticker) multi-index and feature columns.

    Features (momentum + flow/volume):
    MOMENTUM GROUP:
      1. mom_1m: 1-month return
      2. mom_3m: 3-month return
      3. mom_6m: 6-month return
      4. mom_12m_skip1: 12-month return skipping last month (classic Jegadeesh-Titman)
      5. mom_reversal: 5-day reversal (short-term mean reversion)
      6. mom_acceleration: change in 3m momentum
      7. relative_strength: stock return / SPY return (3m)

    VOLUME/FLOW GROUP:
      8. vol_ratio_20_60: 20d avg volume / 60d avg volume (volume surge)
      9. obv_trend: OBV 20d slope normalized
      10. mfi_14: Money Flow Index (14d) — combines price + volume
      11. ad_line_trend: Accumulation/Distribution line 20d slope
      12. vol_price_corr: 20d correlation of volume change and price change
      13. relative_volume: volume / 60d avg volume (current)
      14. up_volume_ratio: fraction of volume on up days (20d)

    QUALITY/VALUE GROUP:
      15. volatility_21d: 21d realized vol (annualized)
      16. volatility_ratio: 21d vol / 63d vol
      17. max_dd_63d: Maximum drawdown in last 63 days
      18. skewness_21d: 21d return skewness
      19. gap_frequency: fraction of days with >1% overnight gap (21d)
      20. range_ratio: Average (H-L)/C over 21d (intraday range)
      21. dist_from_52w_high: distance from 52-week high
      22. dist_from_52w_low: distance from 52-week low
    """
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    open_px = data['open']

    # Monthly rebalance dates (month-end business days)
    monthly_dates = close.resample('ME').last().index
    # Filter to dates that exist in our data
    monthly_dates = [d for d in monthly_dates if d in close.index]

    all_features = []
    all_returns = []

    for i, date in enumerate(monthly_dates):
        if i < 12:  # Need 12 months history minimum
            continue
        if i >= len(monthly_dates) - 1:  # Need forward return
            continue

        date_idx = close.index.get_loc(date)
        if date_idx < 252:
            continue

        # Forward 21-day return (what we're predicting)
        next_date = monthly_dates[i + 1]
        fwd_ret = (close.loc[next_date] / close.loc[date] - 1)

        # Historical slices
        c = close.iloc[:date_idx + 1]
        h = high.iloc[:date_idx + 1]
        l = low.iloc[:date_idx + 1]
        v = volume.iloc[:date_idx + 1]
        o = open_px.iloc[:date_idx + 1]

        feats = pd.DataFrame(index=close.columns)

        # --- MOMENTUM ---
        feats['mom_1m'] = c.iloc[-1] / c.iloc[-21] - 1
        feats['mom_3m'] = c.iloc[-1] / c.iloc[-63] - 1
        feats['mom_6m'] = c.iloc[-1] / c.iloc[-126] - 1
        # 12m skip last month (classic cross-sectional momentum)
        feats['mom_12m_skip1'] = c.iloc[-21] / c.iloc[-252] - 1
        # Short-term reversal (5d)
        feats['mom_reversal'] = c.iloc[-1] / c.iloc[-5] - 1
        # Momentum acceleration
        mom3m_now = c.iloc[-1] / c.iloc[-63] - 1
        mom3m_prev = c.iloc[-21] / c.iloc[-84] - 1
        feats['mom_acceleration'] = mom3m_now - mom3m_prev
        # Relative strength vs equal-weight market
        mkt_ret_3m = (c.iloc[-1] / c.iloc[-63] - 1).mean()
        feats['relative_strength'] = mom3m_now - mkt_ret_3m

        # --- VOLUME / FLOW ---
        v20 = v.iloc[-20:].mean()
        v60 = v.iloc[-60:].mean()
        feats['vol_ratio_20_60'] = v20 / v60.replace(0, np.nan)

        # OBV trend (20d slope of OBV, normalized)
        daily_ret_sign = np.sign(c.iloc[-20:].pct_change())
        obv_changes = daily_ret_sign * v.iloc[-20:]
        obv = obv_changes.cumsum()
        if len(obv) >= 2:
            x = np.arange(len(obv))
            obv_vals = obv.values
            # Vectorized slope per stock
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
            # Normalize by average volume
            feats['obv_trend'] = feats['obv_trend'] / v60.replace(0, np.nan).values

        # Money Flow Index (14d)
        typical_price = (c.iloc[-14:] + h.iloc[-14:] + l.iloc[-14:]) / 3
        raw_money_flow = typical_price * v.iloc[-14:]
        tp_change = typical_price.diff()
        pos_flow = raw_money_flow.where(tp_change > 0, 0).sum()
        neg_flow = raw_money_flow.where(tp_change < 0, 0).sum()
        mfr = pos_flow / neg_flow.replace(0, np.nan)
        feats['mfi_14'] = 100 - (100 / (1 + mfr))

        # A/D line trend (20d slope)
        clv = ((c.iloc[-20:] - l.iloc[-20:]) - (h.iloc[-20:] - c.iloc[-20:])) / \
              (h.iloc[-20:] - l.iloc[-20:]).replace(0, np.nan)
        ad = (clv * v.iloc[-20:]).cumsum()
        if len(ad) >= 2:
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

        # Volume-price correlation (20d)
        vol_chg = v.iloc[-20:].pct_change()
        px_chg = c.iloc[-20:].pct_change()
        feats['vol_price_corr'] = vol_chg.corrwith(px_chg)

        # Relative volume (latest day vs 60d avg)
        feats['relative_volume'] = v.iloc[-1] / v60.replace(0, np.nan)

        # Up-volume ratio (20d)
        up_days = (c.iloc[-20:].pct_change() > 0)
        up_vol = v.iloc[-20:].where(up_days, 0).sum()
        total_vol = v.iloc[-20:].sum()
        feats['up_volume_ratio'] = up_vol / total_vol.replace(0, np.nan)

        # --- QUALITY / RISK ---
        rets_21 = c.iloc[-21:].pct_change().dropna()
        feats['volatility_21d'] = rets_21.std() * np.sqrt(252)
        rets_63 = c.iloc[-63:].pct_change().dropna()
        vol_63 = rets_63.std() * np.sqrt(252)
        feats['volatility_ratio'] = (rets_21.std() * np.sqrt(252)) / vol_63.replace(0, np.nan)

        # Max drawdown 63d
        rolling_max = c.iloc[-63:].cummax()
        dd = c.iloc[-63:] / rolling_max - 1
        feats['max_dd_63d'] = dd.min()

        # Skewness
        feats['skewness_21d'] = rets_21.skew()

        # Gap frequency (overnight gaps > 1%)
        gaps = (o.iloc[-21:] / c.iloc[-22:-1].values - 1).abs()
        feats['gap_frequency'] = (gaps > 0.01).mean()

        # Intraday range ratio
        feats['range_ratio'] = ((h.iloc[-21:] - l.iloc[-21:]) / c.iloc[-21:]).mean()

        # Distance from 52w high/low
        high_52w = c.iloc[-252:].max()
        low_52w = c.iloc[-252:].min()
        feats['dist_from_52w_high'] = c.iloc[-1] / high_52w - 1
        feats['dist_from_52w_low'] = c.iloc[-1] / low_52w - 1

        # Add date and forward return
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
# PHASE 1: MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════

class StockEncoder(nn.Module):
    """Per-stock feature encoder MLP."""
    def __init__(self, n_features, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (batch, n_stocks, n_features)
        return self.net(x)  # (batch, n_stocks, d_model)


class CrossStockAttention(nn.Module):
    """Multi-head attention across stocks to learn relative relationships."""
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1, n_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, mask=None):
        # x: (batch, n_stocks, d_model)
        # mask: (batch, n_stocks) boolean — True = pad/invalid
        src_key_padding_mask = mask if mask is not None else None
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


class MomentumFlowRanker(nn.Module):
    """
    Full model: encode features → cross-stock attention → rank prediction.
    Outputs a score per stock; higher = predicted better relative performance.
    """
    def __init__(self, n_features, d_model=128, n_heads=4, ff_dim=256,
                 dropout=0.15, n_layers=2):
        super().__init__()
        self.encoder = StockEncoder(n_features, d_model, dropout)
        self.attention = CrossStockAttention(d_model, n_heads, ff_dim, dropout, n_layers)
        self.rank_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        """
        x: (batch, n_stocks, n_features)
        mask: (batch, n_stocks) boolean — True = pad/invalid
        Returns: (batch, n_stocks) — ranking scores
        """
        h = self.encoder(x)           # (B, S, D)
        h = self.attention(h, mask)    # (B, S, D)
        scores = self.rank_head(h).squeeze(-1)  # (B, S)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        return scores


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

    # Sort by true relevance (descending)
    _, sorted_idx = relevance.sort(dim=1, descending=True)
    sorted_scores = scores.gather(1, sorted_idx)

    # ListMLE: sum of log(softmax of remaining items at each position)
    B, S = sorted_scores.shape
    loss = torch.zeros(B, device=scores.device)

    for i in range(S - 1):
        remaining = sorted_scores[:, i:]
        log_softmax = remaining[:, 0] - torch.logsumexp(remaining, dim=1)
        if mask is not None:
            # Only count valid positions
            valid = ~mask.gather(1, sorted_idx)[:, i]
            loss = loss - log_softmax * valid.float()
        else:
            loss = loss - log_softmax

    return loss.mean()


def pairwise_ranking_loss(scores, relevance, mask=None, margin=0.1):
    """
    Pairwise margin ranking loss: for each pair (i, j) where rel_i > rel_j,
    enforce score_i > score_j + margin.
    Uses sampling for efficiency.
    """
    B, S = scores.shape
    n_pairs = min(S * 2, 200)  # Sample pairs for efficiency

    total_loss = torch.zeros(1, device=scores.device)
    count = 0

    for b in range(B):
        valid_mask = ~mask[b] if mask is not None else torch.ones(S, dtype=torch.bool, device=scores.device)
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        if len(valid_idx) < 2:
            continue

        # Sample random pairs
        n_valid = len(valid_idx)
        idx_i = valid_idx[torch.randint(n_valid, (n_pairs,), device=scores.device)]
        idx_j = valid_idx[torch.randint(n_valid, (n_pairs,), device=scores.device)]

        # Only keep pairs where i has higher relevance than j
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
# PHASE 2: WALK-FORWARD TRAINING
# ═══════════════════════════════════════════════════════════════════════

def prepare_cross_sectional_batch(panel, feature_cols, dates, max_stocks=200):
    """
    Prepare a batch of cross-sectional snapshots.
    Each snapshot: (n_stocks, n_features) at a given date.
    Returns padded tensor + mask + forward returns.
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

        # Limit to max_stocks (sorted by feature completeness)
        if len(df) > max_stocks:
            df = df.iloc[:max_stocks]

        # Cross-sectional rank-normalize features (robust to outliers)
        X = df[feature_cols].values.copy()
        for j in range(X.shape[1]):
            col = X[:, j]
            valid = ~np.isnan(col)
            if valid.sum() > 5:
                # Rank-normalize within cross-section
                ranks = np.zeros_like(col)
                ranks[valid] = pd.Series(col[valid]).rank(pct=True).values
                ranks[~valid] = 0.5  # Neutral for missing
                X[:, j] = ranks
            else:
                X[:, j] = 0.5

        y = df['fwd_ret_21d'].values
        # Convert to cross-sectional ranks for ranking loss
        y_rank = pd.Series(y).rank(pct=True).values

        # Pad to max_stocks
        n = len(df)
        X_padded = np.zeros((max_stocks, len(feature_cols)), dtype=np.float32)
        y_padded = np.zeros(max_stocks, dtype=np.float32)
        mask = np.ones(max_stocks, dtype=bool)  # True = masked/invalid

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


def train_fold(model, optimizer, X_train, y_train, mask_train,
               X_val, y_val, mask_val, epochs, patience, device):
    """Train one walk-forward fold with early stopping."""
    model.train()
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss = 0
        n_batches = 0

        # Shuffle training data
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

        avg_train_loss = total_loss / max(n_batches, 1)

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            x_v = X_val.to(device)
            y_v = y_val.to(device)
            m_v = mask_val.to(device)
            val_scores = model(x_v, m_v)
            val_loss = listMLE_loss(val_scores, y_v, m_v).item()

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
    fold_metrics = []

    n_features = len(feature_cols)

    for fold_start in range(train_periods, n_dates - test_periods + 1):
        fold_num = fold_start - train_periods
        train_dates = dates[fold_start - train_periods:fold_start]
        test_dates = dates[fold_start:fold_start + test_periods]

        # Prepare data
        X_train, y_train, mask_train, _ = prepare_cross_sectional_batch(
            panel, feature_cols, train_dates, max_stocks=200
        )
        X_test, y_test, mask_test, test_tickers = prepare_cross_sectional_batch(
            panel, feature_cols, test_dates, max_stocks=200
        )

        if X_train is None or X_test is None:
            continue

        # Split train into train/val (last 20% as val, min 2)
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

        # Create fresh model each fold (no information leakage)
        model = MomentumFlowRanker(
            n_features=n_features,
            d_model=CONFIG['d_model'],
            n_heads=CONFIG['n_heads'],
            ff_dim=CONFIG['ff_dim'],
            dropout=CONFIG['dropout'],
            n_layers=CONFIG['n_encoder_layers'],
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG['lr'],
            weight_decay=CONFIG['weight_decay'],
        )

        # Train
        val_loss = train_fold(
            model, optimizer,
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

        # Extract predictions for each test date
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
# PHASE 3: PORTFOLIO CONSTRUCTION & METRICS
# ═══════════════════════════════════════════════════════════════════════

def build_portfolio(predictions, mode='long_short'):
    """
    Build monthly-rebalanced portfolio from model predictions.

    Modes:
    - 'long_short': Long top quintile, short bottom quintile (market-neutral)
    - 'long_only': Long top decile only

    Returns monthly portfolio returns.
    """
    cost_bps = CONFIG['cost_bps']
    top_pct = CONFIG['top_pct']
    bot_pct = CONFIG['bot_pct']

    dates = sorted(predictions['date'].unique())
    port_rets = []
    holdings_history = []

    prev_longs = set()
    prev_shorts = set()

    for date in dates:
        df = predictions[predictions['date'] == date].copy()
        df = df.dropna(subset=['score', 'fwd_ret_21d'])

        if len(df) < 20:
            continue

        # Rank stocks by score
        df['rank'] = df['score'].rank(pct=True)
        n = len(df)

        # Select quintiles
        long_mask = df['rank'] >= (1 - top_pct)
        short_mask = df['rank'] <= bot_pct

        longs = set(df[long_mask]['ticker'].tolist())
        shorts = set(df[short_mask]['ticker'].tolist())

        # Calculate turnover
        long_turnover = len(longs - prev_longs) / max(len(longs), 1)
        short_turnover = len(shorts - prev_shorts) / max(len(shorts), 1)
        avg_turnover = (long_turnover + short_turnover) / 2

        # Portfolio return
        long_ret = df[long_mask]['fwd_ret_21d'].mean()
        short_ret = df[short_mask]['fwd_ret_21d'].mean()

        if mode == 'long_short':
            gross_ret = (long_ret - short_ret) / 2  # Dollar-neutral
        else:
            gross_ret = long_ret

        # Transaction costs (turnover × cost_bps × 2 for round-trip)
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
            'n_total': n,
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

    ann_factor = 12  # Monthly → annual

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
# PHASE 4: VALIDATION GATES
# ═══════════════════════════════════════════════════════════════════════

def permutation_test(predictions, n_perms=200):
    """
    PROPER permutation test: for each period, shuffle the model SCORES
    across stocks (breaking the score-stock mapping). Then rebuild portfolio.
    This preserves the cross-sectional structure but randomizes which stocks
    are selected. If the model has no edge, shuffled scores = same Sharpe.
    """
    log.info(f"\nRunning {n_perms}-shuffle permutation test...")

    # Observed performance
    port = build_portfolio(predictions, mode='long_short')
    observed_sharpe = compute_metrics(port)['sharpe']

    null_sharpes = []
    for perm_i in range(n_perms):
        shuffled = predictions.copy()
        # For each date, shuffle scores across stocks
        for date in shuffled['date'].unique():
            mask = shuffled['date'] == date
            scores = shuffled.loc[mask, 'score'].values.copy()
            np.random.shuffle(scores)
            shuffled.loc[mask, 'score'] = scores

        perm_port = build_portfolio(shuffled, mode='long_short')
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
             f"null={null_sharpes.mean():.3f} ± {null_sharpes.std():.3f}")
    log.info(f"  PASS: {result['pass']}")

    return result


def regime_test(port_df):
    """R1: Regime-agnostic validation. Split by SPY monthly return."""
    import yfinance as yf

    log.info("\nRunning regime test...")

    # Get SPY monthly returns
    spy = yf.download('SPY', start='2009-01-01', end='2027-01-01', auto_adjust=True, progress=False)
    spy_monthly = spy['Close'].resample('ME').last().pct_change().dropna()

    bull_dates = set()
    bear_dates = set()

    for d in port_df['date']:
        # Find closest month-end in SPY data
        closest = spy_monthly.index[spy_monthly.index.get_indexer([d], method='nearest')[0]]
        val = spy_monthly.loc[closest]
        if hasattr(val, 'item'):
            val = val.item()
        elif hasattr(val, 'iloc'):
            val = val.iloc[0]
        if val > 0:
            bull_dates.add(d)
        else:
            bear_dates.add(d)

    bull_rets = port_df[port_df['date'].isin(bull_dates)]['net_ret'].values
    bear_rets = port_df[port_df['date'].isin(bear_dates)]['net_ret'].values

    sharpe_bull = bull_rets.mean() / bull_rets.std() * np.sqrt(12) if len(bull_rets) > 3 and bull_rets.std() > 0 else 0
    sharpe_bear = bear_rets.mean() / bear_rets.std() * np.sqrt(12) if len(bear_rets) > 3 and bear_rets.std() > 0 else 0

    gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 0.001)

    result = {
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'n_bull': len(bull_rets),
        'n_bear': len(bear_rets),
        'regime_gap': round(gap, 4),
        'pass': gap <= CONFIG['regime_gap_max'],
    }

    log.info(f"  Regime: bull Sharpe={sharpe_bull:.3f} ({len(bull_rets)}m), "
             f"bear Sharpe={sharpe_bear:.3f} ({len(bear_rets)}m), gap={gap:.3f}")
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
# PHASE 5: LIGHTGBM BASELINE
# ═══════════════════════════════════════════════════════════════════════

def run_lgbm_baseline(panel, feature_cols):
    """LightGBM cross-sectional ranking as baseline comparison."""
    import lightgbm as lgb

    log.info("\n" + "=" * 70)
    log.info("LGBM BASELINE")
    log.info("=" * 70)

    dates = sorted(panel['date'].unique())
    train_periods = CONFIG['train_days'] // CONFIG['test_days']

    all_preds = []

    for fold_start in range(train_periods, len(dates) - 1):
        fold_num = fold_start - train_periods
        train_dates = dates[fold_start - train_periods:fold_start]
        test_date = dates[fold_start]

        # Prepare data
        train_df = panel[panel['date'].isin(train_dates)].dropna(subset=feature_cols + ['fwd_ret_21d'])
        test_df = panel[panel['date'] == test_date].dropna(subset=feature_cols + ['fwd_ret_21d'])

        if len(train_df) < 100 or len(test_df) < 20:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df['fwd_ret_21d'].values
        X_test = test_df[feature_cols].values

        # Replace inf/nan
        X_train = np.nan_to_num(X_train, nan=0, posinf=1, neginf=-1)
        X_test = np.nan_to_num(X_test, nan=0, posinf=1, neginf=-1)

        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            device='gpu',
            gpu_use_dp=False,
            verbose=-1,
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        pred_df = pd.DataFrame({
            'date': test_date,
            'ticker': test_df['ticker'].values,
            'score': preds,
            'fwd_ret_21d': test_df['fwd_ret_21d'].values,
        })
        all_preds.append(pred_df)

        if fold_num % 10 == 0:
            log.info(f"  [LGBM] Fold {fold_num}: test={test_date.strftime('%Y-%m-%d')}")

    predictions = pd.concat(all_preds, ignore_index=True)
    return predictions


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
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

    # Save feature panel
    panel.to_parquet(os.path.join(OUTPUT_DIR, 'feature_panel.parquet'), index=False)
    log.info(f"  Saved feature panel")

    # --- Step 3: DL Walk-Forward ---
    log.info("\n" + "=" * 70)
    log.info("STEP 3: TRANSFORMER WALK-FORWARD")
    log.info("=" * 70)

    dl_predictions = run_walk_forward(panel, feature_cols)
    dl_predictions.to_csv(os.path.join(OUTPUT_DIR, 'dl_predictions.csv'), index=False)

    # --- Step 4: LGBM Baseline ---
    log.info("\n" + "=" * 70)
    log.info("STEP 4: LGBM BASELINE")
    log.info("=" * 70)

    lgbm_predictions = run_lgbm_baseline(panel, feature_cols)
    lgbm_predictions.to_csv(os.path.join(OUTPUT_DIR, 'lgbm_predictions.csv'), index=False)

    # --- Step 5: Portfolio construction ---
    log.info("\n" + "=" * 70)
    log.info("STEP 5: PORTFOLIO CONSTRUCTION")
    log.info("=" * 70)

    # L/S portfolios
    dl_ls = build_portfolio(dl_predictions, mode='long_short')
    lgbm_ls = build_portfolio(lgbm_predictions, mode='long_short')

    # Long-only portfolios
    dl_lo = build_portfolio(dl_predictions, mode='long_only')
    lgbm_lo = build_portfolio(lgbm_predictions, mode='long_only')

    dl_ls_metrics = compute_metrics(dl_ls)
    lgbm_ls_metrics = compute_metrics(lgbm_ls)
    dl_lo_metrics = compute_metrics(dl_lo)
    lgbm_lo_metrics = compute_metrics(lgbm_lo)

    log.info("\n  DL Transformer L/S:")
    for k, v in dl_ls_metrics.items():
        log.info(f"    {k}: {v}")

    log.info("\n  LGBM Baseline L/S:")
    for k, v in lgbm_ls_metrics.items():
        log.info(f"    {k}: {v}")

    log.info("\n  DL Transformer Long-Only:")
    for k, v in dl_lo_metrics.items():
        log.info(f"    {k}: {v}")

    log.info("\n  LGBM Baseline Long-Only:")
    for k, v in lgbm_lo_metrics.items():
        log.info(f"    {k}: {v}")

    # --- Step 6: Validation gates (on best model) ---
    log.info("\n" + "=" * 70)
    log.info("STEP 6: VALIDATION GATES")
    log.info("=" * 70)

    # Use L/S for validation (market-hedged)
    best_preds = dl_predictions if dl_ls_metrics.get('sharpe', 0) >= lgbm_ls_metrics.get('sharpe', 0) else lgbm_predictions
    best_name = 'DL Transformer' if dl_ls_metrics.get('sharpe', 0) >= lgbm_ls_metrics.get('sharpe', 0) else 'LGBM'
    best_port = dl_ls if best_name == 'DL Transformer' else lgbm_ls
    best_metrics = dl_ls_metrics if best_name == 'DL Transformer' else lgbm_ls_metrics

    log.info(f"\n  Best model: {best_name}")

    perm_result = permutation_test(best_preds, CONFIG['n_permutations'])
    regime_result = regime_test(best_port)
    subperiod_result = subperiod_test(best_port)

    gates_passed = sum([
        perm_result['pass'],
        regime_result['pass'],
        subperiod_result['pass'],
    ])

    log.info(f"\n  Validation: {gates_passed}/3 gates passed")
    log.info(f"    Permutation: {'PASS' if perm_result['pass'] else 'FAIL'} (p={perm_result['p_value']})")
    log.info(f"    Regime:      {'PASS' if regime_result['pass'] else 'FAIL'} (gap={regime_result['regime_gap']})")
    log.info(f"    Stability:   {'PASS' if subperiod_result['pass'] else 'FAIL'} (CV={subperiod_result['cv']})")

    # --- Step 7: Save results ---
    log.info("\n" + "=" * 70)
    log.info("STEP 7: SAVING RESULTS")
    log.info("=" * 70)

    runtime_min = (time.time() - start_time) / 60

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'metrics': {
            'dl_long_short': dl_ls_metrics,
            'lgbm_long_short': lgbm_ls_metrics,
            'dl_long_only': dl_lo_metrics,
            'lgbm_long_only': lgbm_lo_metrics,
        },
        'best_model': best_name,
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

    dl_ls.to_csv(os.path.join(OUTPUT_DIR, 'dl_ls_portfolio.csv'), index=False)
    lgbm_ls.to_csv(os.path.join(OUTPUT_DIR, 'lgbm_ls_portfolio.csv'), index=False)

    log.info(f"  Results saved to {OUTPUT_DIR}")
    log.info(f"  Runtime: {runtime_min:.1f} minutes")

    # --- Step 8: MLflow logging ---
    log.info("\n" + "=" * 70)
    log.info("STEP 8: MLFLOW LOGGING")
    log.info("=" * 70)

    try:
        import mlflow
        mlflow.set_tracking_uri(CONFIG['mlflow_tracking_uri'])
        mlflow.set_experiment(CONFIG['mlflow_experiment'])

        with mlflow.start_run(run_name=f"dl_momentum_flow_v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Log config
            for k, v in CONFIG.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(k, v)

            # Log best model metrics
            for k, v in best_metrics.items():
                mlflow.log_metric(f"best_{k}", v)

            # Log DL vs LGBM
            for k, v in dl_ls_metrics.items():
                mlflow.log_metric(f"dl_ls_{k}", v)
            for k, v in lgbm_ls_metrics.items():
                mlflow.log_metric(f"lgbm_ls_{k}", v)

            # Log validation
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
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"  Best model: {best_name}")
    log.info(f"  L/S Sharpe: {best_metrics.get('sharpe', 'N/A')}")
    log.info(f"  L/S CAGR: {best_metrics.get('cagr', 'N/A')}")
    log.info(f"  L/S MaxDD: {best_metrics.get('max_dd', 'N/A')}")
    log.info(f"  Validation: {gates_passed}/3 gates")
    log.info(f"  Runtime: {runtime_min:.1f} minutes")
    log.info("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(f"FATAL: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)
