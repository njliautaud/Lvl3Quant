#!/usr/bin/env python3
"""
Spread Outcome Predictor v1 — Deep Learning on Spread WIN/LOSS Targets
======================================================================
HYPOTHESIS: Training on spread outcomes (WIN/LOSS) instead of equity returns
captures nonlinear option payoff structure better than LGBM-on-equity-returns.

Finding #221: equity-level sector momentum Sharpe 0.20 — too weak
Finding #217: ~90% of edge is structural (options), ~10% from ML ranking
Question: can a neural net trained directly on SPREAD OUTCOME data extract more?

VARIANTS:
  A: LGBM on spread outcomes (baseline)
  B: Small MLP (64-32) on spread outcomes, GPU
  C: GRU (32 hidden, 10 lookback) on spread outcomes, GPU
  D: LGBM on equity returns (control — proves target matters)

Walk-forward: 250d train, 63d test, sliding window.
ATR-based BS pricing: iv_mult=1.2, haircut=15%, comm=$2.60/spread
DTE=14, OTM=2%, spread_width=3%
"""

import sys, os, json, warnings, time, traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── PATHS ──────────────────────────────────────────────────────
if sys.platform == "win32":
    ROOT = Path(r"C:\Users\claude\Lvl3Quant")
elif os.path.exists("/home/nick/Lvl3Quant"):
    ROOT = Path("/home/nick/Lvl3Quant")
else:
    ROOT = Path("/home/jupiter/Lvl3Quant")

OUTPUT = ROOT / "output" / "growth_research" / "spread_outcome_predictor_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fprint(*a, **kw):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]", *a, **kw, flush=True)

fprint(f"Device: {DEVICE}")
fprint(f"Output: {OUTPUT}")

# ── MLFLOW ─────────────────────────────────────────────────────
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen("http://jupiter:5000/", timeout=3)
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    MLFLOW_OK = True
    fprint("MLflow connected")
except Exception:
    fprint("MLflow unavailable — will log locally only")

# ── CONSTANTS ──────────────────────────────────────────────────
SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
EXTRA_TICKERS = ["SPY", "^VIX", "TLT", "GLD"]

CAP = 645.0
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # $2.60 RT (4 legs)
HAIRCUT = 0.15  # 15% bid-ask haircut
IV_MULT = 1.2   # IV multiplier for BS pricing
OTM_PCT = 0.02  # 2% OTM for long strike
SPREAD_WIDTH = 0.03  # 3% wide
DTE = 14         # 14-day DTE
TOP_K = 3        # top/bottom K sectors

# Walk-forward params
TRAIN_DAYS = 250
TEST_DAYS = 63

# ── DATA DOWNLOAD ──────────────────────────────────────────────

def download_data():
    """Download sector ETF + macro data."""
    import yfinance as yf

    cache = OUTPUT / "price_cache.pkl"
    if cache.exists():
        fprint("Loading cached data...")
        data = pd.read_pickle(cache)
        fprint(f"  Loaded {len(data['close'])} days, {len(data['close'].columns)} tickers")
        return data

    fprint("Downloading data from yfinance...")
    tickers = SECTORS + EXTRA_TICKERS
    raw = yf.download(tickers, start="2010-01-01", end="2026-07-27", progress=False)

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw
    high = raw["High"] if mi else raw
    low = raw["Low"] if mi else raw
    volume = raw["Volume"] if mi else raw

    # Standardize VIX column name
    vix_col = "^VIX" if "^VIX" in close.columns else "VIX"
    if vix_col != "VIX":
        close = close.rename(columns={vix_col: "VIX"})
        high = high.rename(columns={vix_col: "VIX"})
        low = low.rename(columns={vix_col: "VIX"})
        volume = volume.rename(columns={vix_col: "VIX"})

    # Common index
    all_cols = [c for c in SECTORS + ["SPY", "VIX", "TLT", "GLD"] if c in close.columns]
    ix = close[all_cols].dropna(how="any").index
    fprint(f"  Data: {len(ix)} days, {len(all_cols)} tickers")

    data = {
        "close": close.loc[ix, all_cols],
        "high": high.loc[ix, [c for c in all_cols if c in high.columns]],
        "low": low.loc[ix, [c for c in all_cols if c in low.columns]],
        "volume": volume.loc[ix, [c for c in all_cols if c in volume.columns]],
    }
    data["close"].to_pickle(cache.with_name("close_cache.pkl"))
    pd.to_pickle(data, cache)
    return data


# ── ATR + BS PRICING ──────────────────────────────────────────

def compute_atr(high, low, close, period=14):
    """Compute ATR."""
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": abs(high - close.shift(1)),
        "lc": abs(low - close.shift(1)),
    }).max(axis=1)
    return tr.rolling(period).mean()


def bs_call(S, K, T, sigma, r=0.04):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, sigma, r=0.04):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def price_bull_call_spread(S, atr_val, vix_val):
    """Price bull call spread: buy OTM call, sell further OTM call.
    Returns (entry_cost, max_profit, max_loss) per spread (1 contract = 100 shares).
    """
    T = DTE / 252.0
    K1 = S * (1 + OTM_PCT)       # long call strike
    K2 = S * (1 + OTM_PCT + SPREAD_WIDTH)  # short call strike

    # Implied vol from ATR with multiplier
    sigma = (atr_val / S) * np.sqrt(252) * IV_MULT
    sigma = max(sigma, 0.10)  # floor at 10%

    # BS prices with haircut
    long_prem = bs_call(S, K1, T, sigma) * (1 + HAIRCUT)   # pay more to buy
    short_prem = bs_call(S, K2, T, sigma) * (1 - HAIRCUT)  # receive less to sell

    debit = long_prem - short_prem  # per share
    entry_cost = debit * 100 + SPREAD_COMM
    width = (K2 - K1) * 100
    max_profit = width - entry_cost
    max_loss = entry_cost

    return entry_cost, max_profit, max_loss, K1, K2


def price_bear_put_spread(S, atr_val, vix_val):
    """Price bear put spread: buy OTM put, sell further OTM put.
    Returns (entry_cost, max_profit, max_loss) per spread.
    """
    T = DTE / 252.0
    K1 = S * (1 - OTM_PCT)       # long put strike (closer)
    K2 = S * (1 - OTM_PCT - SPREAD_WIDTH)  # short put strike (further OTM)

    sigma = (atr_val / S) * np.sqrt(252) * IV_MULT
    sigma = max(sigma, 0.10)

    long_prem = bs_put(S, K1, T, sigma) * (1 + HAIRCUT)
    short_prem = bs_put(S, K2, T, sigma) * (1 - HAIRCUT)

    debit = long_prem - short_prem
    entry_cost = debit * 100 + SPREAD_COMM
    width = (K1 - K2) * 100
    max_profit = width - entry_cost
    max_loss = entry_cost

    return entry_cost, max_profit, max_loss, K1, K2


def compute_spread_outcome(S_entry, S_expiry, K1, K2, entry_cost, spread_type):
    """Compute intrinsic value at expiry, return (pnl, win_flag)."""
    if spread_type == "bull":
        intrinsic = (max(0, S_expiry - K1) - max(0, S_expiry - K2)) * 100
    else:  # bear
        intrinsic = (max(0, K1 - S_expiry) - max(0, K2 - S_expiry)) * 100
    pnl = intrinsic - entry_cost
    return pnl, int(pnl > 0)


# ── FEATURE ENGINEERING ───────────────────────────────────────

FEATURE_NAMES = [
    "ret_5d", "ret_21d", "ret_63d",
    "rs_5d", "rs_21d", "rs_63d",
    "vix_level", "vix_5d_chg", "vix_21d_chg",
    "atr_ratio",
    "vol_21d", "vol_63d",
    "dispersion_21d",
    "pct_52w_high", "pct_52w_low",
    "vol_trend_5d", "vol_trend_21d",
    "tlt_ret_21d", "gld_ret_21d",
    "mom_accel",
]


def build_features(data, date_idx, ticker):
    """Build feature vector for a sector ETF on a given date index.
    Returns dict of features or None if insufficient data.
    """
    close = data["close"]
    volume = data["volume"]
    high = data["high"]
    low = data["low"]

    if date_idx < 260:
        return None

    px = close[ticker].iloc[:date_idx + 1].dropna()
    if len(px) < 260:
        return None

    spy = close["SPY"].iloc[:date_idx + 1].dropna()
    vix = close["VIX"].iloc[:date_idx + 1].dropna()

    f = {}

    # Momentum
    for lb, nm in [(5, "ret_5d"), (21, "ret_21d"), (63, "ret_63d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    # Relative strength vs SPY
    for lb, nm in [(5, "rs_5d"), (21, "rs_21d"), (63, "rs_63d")]:
        if len(px) > lb and len(spy) > lb:
            f[nm] = float((px.iloc[-1] / px.iloc[-lb]) / (spy.iloc[-1] / spy.iloc[-lb]) - 1)
        else:
            f[nm] = 0.0

    # VIX
    f["vix_level"] = float(vix.iloc[-1]) if len(vix) > 0 else 20.0
    f["vix_5d_chg"] = float(vix.iloc[-1] / vix.iloc[-5] - 1) if len(vix) > 5 else 0.0
    f["vix_21d_chg"] = float(vix.iloc[-1] / vix.iloc[-21] - 1) if len(vix) > 21 else 0.0

    # ATR / vol ratio
    if ticker in high.columns and ticker in low.columns:
        atr = compute_atr(high[ticker].iloc[:date_idx + 1],
                          low[ticker].iloc[:date_idx + 1],
                          px)
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else float(px.iloc[-1]) * 0.015
    else:
        atr_val = float(px.iloc[-1]) * 0.015
    f["atr_ratio"] = atr_val / float(px.iloc[-1])

    # Realized vol
    rets = px.pct_change().dropna()
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2

    # Cross-sector dispersion (std of sector returns)
    sector_cols = [c for c in SECTORS if c in close.columns]
    sector_rets_21d = []
    for sc in sector_cols:
        spx = close[sc].iloc[:date_idx + 1].dropna()
        if len(spx) > 21:
            sector_rets_21d.append(float(spx.iloc[-1] / spx.iloc[-21] - 1))
    f["dispersion_21d"] = float(np.std(sector_rets_21d)) if len(sector_rets_21d) > 3 else 0.0

    # Distance from 52w high/low
    high_52w = float(px.iloc[-252:].max())
    low_52w = float(px.iloc[-252:].min())
    f["pct_52w_high"] = float(px.iloc[-1]) / high_52w if high_52w > 0 else 1.0
    f["pct_52w_low"] = float(px.iloc[-1]) / low_52w if low_52w > 0 else 1.0

    # Volume trends
    if ticker in volume.columns:
        vol_series = volume[ticker].iloc[:date_idx + 1].dropna()
        if len(vol_series) > 21:
            f["vol_trend_5d"] = float(vol_series.iloc[-5:].mean() / vol_series.iloc[-21:].mean()) - 1.0
            f["vol_trend_21d"] = float(vol_series.iloc[-21:].mean() / vol_series.iloc[-63:].mean()) - 1.0 if len(vol_series) > 63 else 0.0
        else:
            f["vol_trend_5d"] = 0.0
            f["vol_trend_21d"] = 0.0
    else:
        f["vol_trend_5d"] = 0.0
        f["vol_trend_21d"] = 0.0

    # TLT and GLD momentum
    for macro, nm in [("TLT", "tlt_ret_21d"), ("GLD", "gld_ret_21d")]:
        if macro in close.columns:
            mpx = close[macro].iloc[:date_idx + 1].dropna()
            f[nm] = float(mpx.iloc[-1] / mpx.iloc[-21] - 1) if len(mpx) > 21 else 0.0
        else:
            f[nm] = 0.0

    # Momentum acceleration
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3.0

    return f, atr_val


# ── BUILD DATASET ─────────────────────────────────────────────

def build_dataset(data):
    """Build full dataset with features, spread outcomes, and equity return labels."""
    close = data["close"]
    high = data["high"]
    low = data["low"]
    vix = close["VIX"]

    dates = close.index
    records = []

    fprint("Building dataset with spread outcome labels...")
    sector_cols = [c for c in SECTORS if c in close.columns]

    for i in range(260, len(dates) - DTE - 1):
        dt = dates[i]
        vix_val = float(vix.iloc[i]) if not pd.isna(vix.iloc[i]) else 20.0

        for tk in sector_cols:
            result = build_features(data, i, tk)
            if result is None:
                continue
            feats, atr_val = result

            S = float(close[tk].iloc[i])
            S_expiry = float(close[tk].iloc[i + DTE])

            # Bull call spread outcome
            try:
                bc_cost, bc_profit, bc_loss, bc_K1, bc_K2 = price_bull_call_spread(S, atr_val, vix_val)
                bc_pnl, bc_win = compute_spread_outcome(S, S_expiry, bc_K1, bc_K2, bc_cost, "bull")
            except Exception:
                bc_win, bc_pnl = 0, 0.0

            # Bear put spread outcome
            try:
                bp_cost, bp_profit, bp_loss, bp_K1, bp_K2 = price_bear_put_spread(S, atr_val, vix_val)
                bp_pnl, bp_win = compute_spread_outcome(S, S_expiry, bp_K1, bp_K2, bp_cost, "bear")
            except Exception:
                bp_win, bp_pnl = 0, 0.0

            # Equity forward return (for control variant D)
            fwd_ret = float(S_expiry / S - 1)

            record = {
                "date": dt,
                "ticker": tk,
                "bull_win": bc_win,
                "bear_win": bp_win,
                "bull_pnl": bc_pnl,
                "bear_pnl": bp_pnl,
                "fwd_ret": fwd_ret,
                "spot": S,
                "atr": atr_val,
                "vix": vix_val,
            }
            record.update(feats)
            records.append(record)

        if (i - 260) % 500 == 0:
            fprint(f"  Processed {i - 260}/{len(dates) - 260 - DTE} days ({len(records)} records)")

    df = pd.DataFrame(records)
    fprint(f"  Dataset: {len(df)} records, {df['date'].nunique()} dates, "
           f"bull WR: {df['bull_win'].mean():.1%}, bear WR: {df['bear_win'].mean():.1%}")
    return df


# ── MODELS ─────────────────────────────────────────────────────

class MLP(nn.Module):
    """Small MLP for binary classification (64-32)."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GRUPredictor(nn.Module):
    """GRU with lookback sequence for spread outcome prediction."""
    def __init__(self, input_dim, hidden_dim=32, n_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.1 if n_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        _, h = self.gru(x)  # h: (n_layers, batch, hidden)
        out = self.head(h[-1])
        return out.squeeze(-1)


# ── WALK-FORWARD ENGINE ───────────────────────────────────────

def train_lgbm(X_train, y_train, X_test):
    """Train LGBM and return predictions."""
    import lightgbm as lgb
    try:
        m = lgb.LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
            verbose=-1,
        )
        m.fit(X_train, y_train)
        return m.predict_proba(X_test)[:, 1]
    except Exception:
        # Fallback: regression
        m = lgb.LGBMRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
            verbose=-1,
        )
        m.fit(X_train, y_train.astype(float))
        return m.predict(X_test)


def train_mlp(X_train, y_train, X_test, epochs=50):
    """Train MLP on GPU and return predictions."""
    model = MLP(X_train.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_e = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(X_e)).cpu().numpy()
    return scores


def build_gru_sequences(df, feature_cols, lookback=10):
    """Build sequence data for GRU: group by ticker, create lookback windows."""
    sequences = []
    labels = []
    meta = []  # (date, ticker) for each sequence

    for tk in df["ticker"].unique():
        tk_df = df[df["ticker"] == tk].sort_values("date").reset_index(drop=True)
        X = tk_df[feature_cols].values.astype(np.float32)
        y = tk_df["label"].values.astype(np.float32)
        dates = tk_df["date"].values

        for i in range(lookback, len(tk_df)):
            sequences.append(X[i - lookback:i])
            labels.append(y[i])
            meta.append((dates[i], tk))

    return np.array(sequences), np.array(labels), meta


def train_gru(seq_train, y_train, seq_test, hidden=32, epochs=50):
    """Train GRU on GPU and return predictions."""
    input_dim = seq_train.shape[2]
    model = GRUPredictor(input_dim, hidden).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(seq_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_e = torch.tensor(seq_test, dtype=torch.float32).to(DEVICE)

    model.train()
    for ep in range(epochs):
        # Mini-batch training for memory efficiency
        perm = torch.randperm(len(X_t))
        batch_size = 256
        for start in range(0, len(X_t), batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            pred = model(X_t[idx])
            loss = criterion(pred, y_t[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        # Batch inference
        scores = []
        for start in range(0, len(X_e), 512):
            batch = X_e[start:start + 512]
            s = torch.sigmoid(model(batch)).cpu().numpy()
            scores.append(s)
    return np.concatenate(scores)


def walkforward_variant(df, variant_name, feature_cols):
    """Run sliding walk-forward for a given variant.
    Returns dict of {date: {ticker: score}} rankings.
    """
    dates = sorted(df["date"].unique())
    rankings = {}
    n_windows = 0

    fprint(f"\n{'='*60}")
    fprint(f"VARIANT {variant_name}")
    fprint(f"{'='*60}")

    is_gru = "GRU" in variant_name
    lookback = 10

    for test_start_idx in range(TRAIN_DAYS, len(dates) - TEST_DAYS, TEST_DAYS):
        train_dates = dates[test_start_idx - TRAIN_DAYS:test_start_idx]
        test_dates = dates[test_start_idx:test_start_idx + TEST_DAYS]

        train_df = df[df["date"].isin(train_dates)].copy()
        test_df = df[df["date"].isin(test_dates)].copy()

        if len(train_df) < 100 or len(test_df) < 20:
            continue

        n_windows += 1

        if is_gru:
            # GRU needs sequence building
            train_df = train_df.sort_values(["ticker", "date"])
            test_df = test_df.sort_values(["ticker", "date"])

            # Build combined df for sequence creation (need lookback context)
            # Get a few extra days before test start for context
            context_start = max(0, test_start_idx - lookback)
            context_dates = dates[context_start:test_start_idx]
            context_df = df[df["date"].isin(context_dates)].copy()

            # Train sequences
            seq_train, y_train, meta_train = build_gru_sequences(train_df, feature_cols, lookback)

            # Test sequences: prepend context
            combined_test = pd.concat([context_df, test_df]).drop_duplicates(subset=["date", "ticker"])
            combined_test = combined_test.sort_values(["ticker", "date"])
            seq_test, _, meta_test = build_gru_sequences(combined_test, feature_cols, lookback)

            if len(seq_train) < 50 or len(seq_test) < 10:
                continue

            # Filter test meta to only include actual test dates
            test_date_set = set(test_dates)
            test_mask = [m[0] in test_date_set for m in meta_test]
            seq_test_filtered = seq_test[test_mask]
            meta_test_filtered = [m for m, mask in zip(meta_test, test_mask) if mask]

            if len(seq_test_filtered) < 5:
                continue

            scores = train_gru(seq_train, y_train, seq_test_filtered)

            for (dt, tk), score in zip(meta_test_filtered, scores):
                if dt not in rankings:
                    rankings[dt] = {}
                rankings[dt][tk] = float(score)

        else:
            # LGBM or MLP — flat features
            X_train = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
            y_train = train_df["label"].values.astype(np.float32)
            X_test = np.nan_to_num(test_df[feature_cols].values.astype(np.float32))

            # Standardize
            mu = X_train.mean(axis=0)
            std = X_train.std(axis=0) + 1e-8
            X_train = (X_train - mu) / std
            X_test = (X_test - mu) / std

            if "LGBM" in variant_name:
                scores = train_lgbm(X_train, y_train, X_test)
            else:  # MLP
                scores = train_mlp(X_train, y_train, X_test)

            for idx_row, (_, row) in enumerate(test_df.iterrows()):
                dt = row["date"]
                tk = row["ticker"]
                if dt not in rankings:
                    rankings[dt] = {}
                rankings[dt][tk] = float(scores[idx_row])

        if n_windows % 5 == 0:
            fprint(f"  Window {n_windows}: {len(rankings)} dates ranked so far")

    fprint(f"  {variant_name}: {n_windows} windows, {len(rankings)} dates with rankings")
    return rankings


# ── PORTFOLIO SIMULATION ──────────────────────────────────────

def simulate_portfolio(rankings, data, variant_name):
    """Simulate weekly rebalanced portfolio with bull/bear spreads.
    Top-K sectors → bull call spreads, Bottom-K → bear put spreads.
    Fixed allocation per trade (no compounding death spiral).
    """
    close = data["close"]
    high = data["high"]
    low = data["low"]
    vix = close["VIX"]

    sector_cols = [c for c in SECTORS if c in close.columns]
    atr_cache = {}
    for tk in sector_cols:
        if tk in high.columns and tk in low.columns:
            atr_cache[tk] = compute_atr(high[tk], low[tk], close[tk])

    equity = CAP
    trades = []
    eq_curve = [CAP]
    weekly_pnls = []

    sorted_dates = sorted(rankings.keys())

    # Convert ranking dates to pandas Timestamps for matching
    ranking_ts_map = {}
    for dt in sorted_dates:
        ts = pd.Timestamp(dt) if isinstance(dt, np.datetime64) else dt
        ranking_ts_map[ts] = rankings[dt]

    # Rebalance every DTE days (biweekly for DTE=14)
    rebal_dates = sorted(ranking_ts_map.keys())[::max(DTE // 2, 5)]

    for dt_ts in rebal_dates:
        if dt_ts not in close.index:
            # Find nearest date in close index
            idx_arr = close.index.get_indexer([dt_ts], method="nearest")
            if idx_arr[0] < 0:
                continue
            dt_ts = close.index[idx_arr[0]]

        if dt_ts not in vix.index:
            continue

        # Get scores - try exact match first, then nearest
        scores = ranking_ts_map.get(dt_ts)
        if scores is None:
            # Find nearest ranking date
            nearest = min(ranking_ts_map.keys(), key=lambda x: abs((x - dt_ts).total_seconds()))
            if abs((nearest - dt_ts).days) <= 3:
                scores = ranking_ts_map[nearest]
            else:
                continue

        if len(scores) < 2 * TOP_K:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        bull_picks = [t for t, _ in ranked[:TOP_K]]
        bear_picks = [t for t, _ in ranked[-TOP_K:]]

        di = close.index.get_loc(dt_ts)
        ei = min(di + DTE, len(close) - 1)
        vix_val = float(vix.iloc[di]) if not pd.isna(vix.iloc[di]) else 20.0

        # Fixed allocation: risk $CAP/6 per trade regardless of current equity
        # This prevents death spiral and measures TRUE signal quality
        max_per_trade = CAP / 3  # ~$215 max per spread
        if equity < CAP * 0.05:  # Stop only if catastrophic
            eq_curve.append(equity)
            continue

        week_pnl = 0.0

        # Bull call spreads on top-K
        for tk in bull_picks:
            if tk not in close.columns or tk not in atr_cache:
                continue
            S = float(close[tk].iloc[di])
            atr_val = float(atr_cache[tk].iloc[di]) if not pd.isna(atr_cache[tk].iloc[di]) else S * 0.015
            S_exp = float(close[tk].iloc[ei])

            try:
                cost, max_prof, max_loss, K1, K2 = price_bull_call_spread(S, atr_val, vix_val)
            except Exception:
                continue
            if cost <= 0 or cost > max_per_trade:
                continue

            pnl, win = compute_spread_outcome(S, S_exp, K1, K2, cost, "bull")
            equity += pnl
            week_pnl += pnl
            trades.append({
                "date": str(dt_ts.date()) if hasattr(dt_ts, "date") else str(dt_ts)[:10],
                "ticker": tk,
                "type": "bull_call",
                "pnl": round(pnl, 2),
                "win": win,
                "spot": round(S, 2),
                "spot_exp": round(S_exp, 2),
                "cost": round(cost, 2),
            })

        # Bear put spreads on bottom-K
        for tk in bear_picks:
            if tk not in close.columns or tk not in atr_cache:
                continue
            S = float(close[tk].iloc[di])
            atr_val = float(atr_cache[tk].iloc[di]) if not pd.isna(atr_cache[tk].iloc[di]) else S * 0.015
            S_exp = float(close[tk].iloc[ei])

            try:
                cost, max_prof, max_loss, K1, K2 = price_bear_put_spread(S, atr_val, vix_val)
            except Exception:
                continue
            if cost <= 0 or cost > max_per_trade:
                continue

            pnl, win = compute_spread_outcome(S, S_exp, K1, K2, cost, "bear")
            equity += pnl
            week_pnl += pnl
            trades.append({
                "date": str(dt_ts.date()) if hasattr(dt_ts, "date") else str(dt_ts)[:10],
                "ticker": tk,
                "type": "bear_put",
                "pnl": round(pnl, 2),
                "win": win,
                "spot": round(S, 2),
                "spot_exp": round(S_exp, 2),
                "cost": round(cost, 2),
            })

        weekly_pnls.append(week_pnl)
        eq_curve.append(equity)

    return trades, equity, eq_curve, weekly_pnls


# ── METRICS ───────────────────────────────────────────────────

def compute_metrics(trades, final_eq, eq_curve, weekly_pnls, name):
    """Compute risk-adjusted metrics."""
    if not trades:
        fprint(f"  {name}: No trades")
        return None

    n = len(trades)
    wins = sum(1 for t in trades if t["win"])
    wr = wins / n * 100
    pnls = [t["pnl"] for t in trades]
    total = sum(pnls)

    # Monthly-ish returns (use weekly_pnls)
    wp = np.array(weekly_pnls) / CAP if weekly_pnls else np.array([0.0])
    n_years = max(len(wp) / 52, 0.5)

    sharpe = (wp.mean() * 52) / (wp.std() * np.sqrt(52) + 1e-10) if len(wp) > 3 else 0
    dn_wp = wp[wp < 0]
    sortino = (wp.mean() * 52) / (dn_wp.std() * np.sqrt(52) + 1e-10) if len(dn_wp) > 1 else 0
    cagr = (final_eq / CAP) ** (1 / n_years) - 1

    eq = np.array(eq_curve)
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / (pk + 1e-10)).min())

    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gp / (gl + 1e-10)
    avg_pnl = np.mean(pnls)
    avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
    avg_loss = np.mean([p for p in pnls if p <= 0]) if any(p <= 0 for p in pnls) else 0

    # Bull vs bear split
    bull_trades = [t for t in trades if t["type"] == "bull_call"]
    bear_trades = [t for t in trades if t["type"] == "bear_put"]
    bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
    bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100

    r = {
        "name": name,
        "n_trades": n,
        "win_rate": round(wr, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "cagr_pct": round(cagr * 100, 1),
        "maxdd_pct": round(mdd * 100, 1),
        "profit_factor": round(pf, 2),
        "avg_pnl": round(avg_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_pnl": round(total, 2),
        "final_equity": round(final_eq, 2),
        "n_bull": len(bull_trades),
        "n_bear": len(bear_trades),
        "bull_wr": round(bull_wr, 1),
        "bear_wr": round(bear_wr, 1),
        "n_years": round(n_years, 1),
    }

    fprint(f"  {name}: {n} trades | WR {wr:.1f}% | Sharpe {sharpe:.2f} | Sort {sortino:.2f} | "
           f"PF {pf:.2f} | CAGR {cagr*100:.1f}% | MaxDD {mdd*100:.1f}% | "
           f"${CAP:.0f} -> ${final_eq:.0f}")
    fprint(f"    Bull: {len(bull_trades)} trades WR {bull_wr:.1f}% | "
           f"Bear: {len(bear_trades)} trades WR {bear_wr:.1f}%")

    return r


# ── ADVERSARIAL VALIDATION (5-GATE) ──────────────────────────

def adversarial_validation(df, rankings, feature_cols, variant_name):
    """Run 5-gate adversarial validation."""
    import lightgbm as lgb

    fprint(f"\n  Adversarial validation for {variant_name}:")
    gates = {}

    dates = sorted(df["date"].unique())
    mid = len(dates) // 2
    first_half = set(dates[:mid])
    second_half = set(dates[mid:])

    df1 = df[df["date"].isin(first_half)].copy()
    df2 = df[df["date"].isin(second_half)].copy()

    X = np.nan_to_num(np.vstack([
        df1[feature_cols].values,
        df2[feature_cols].values,
    ]).astype(np.float32))
    y = np.concatenate([np.zeros(len(df1)), np.ones(len(df2))])

    # Gate 1: Temporal distinguishability (AUC < 0.65)
    try:
        m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, verbose=-1)
        m.fit(X[:len(X)//2], y[:len(X)//2])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y[len(X)//2:], m.predict_proba(X[len(X)//2:])[:, 1])
        gates["G1_temporal_auc"] = round(auc, 3)
        gates["G1_pass"] = auc < 0.65
        fprint(f"    G1 Temporal AUC: {auc:.3f} ({'PASS' if auc < 0.65 else 'FAIL'})")
    except Exception as e:
        gates["G1_temporal_auc"] = None
        gates["G1_pass"] = False
        fprint(f"    G1 error: {e}")

    # Gate 2: Feature stability (correlation between halves > 0.7)
    try:
        corrs = []
        for col in feature_cols:
            c1 = df1[col].mean()
            c2 = df2[col].mean()
            corrs.append(abs(c1 - c2) / (abs(c1) + abs(c2) + 1e-10))
        drift = np.mean(corrs)
        gates["G2_feature_drift"] = round(drift, 3)
        gates["G2_pass"] = drift < 0.3
        fprint(f"    G2 Feature drift: {drift:.3f} ({'PASS' if drift < 0.3 else 'FAIL'})")
    except Exception as e:
        gates["G2_feature_drift"] = None
        gates["G2_pass"] = False

    # Gate 3: Win rate stability (first vs second half within 10pp)
    try:
        ranked_dates = sorted(rankings.keys())
        mid_r = len(ranked_dates) // 2
        first_r = ranked_dates[:mid_r]
        second_r = ranked_dates[mid_r:]

        df_r = df.copy()
        df_r["in_first"] = df_r["date"].isin(first_r)
        wr1 = df_r[df_r["in_first"]]["label"].mean() if len(df_r[df_r["in_first"]]) > 0 else 0.5
        wr2 = df_r[~df_r["in_first"]]["label"].mean() if len(df_r[~df_r["in_first"]]) > 0 else 0.5
        wr_gap = abs(wr1 - wr2)
        gates["G3_wr_gap"] = round(wr_gap, 3)
        gates["G3_pass"] = wr_gap < 0.10
        fprint(f"    G3 WR gap: {wr_gap:.3f} ({'PASS' if wr_gap < 0.10 else 'FAIL'})")
    except Exception as e:
        gates["G3_wr_gap"] = None
        gates["G3_pass"] = False

    # Gate 4: Permutation test (shuffled labels should degrade)
    try:
        # Compare actual vs shuffled predictions
        # Use a small sample for speed
        sample_dates = dates[TRAIN_DAYS:TRAIN_DAYS + TEST_DAYS * 2]
        sample = df[df["date"].isin(sample_dates)].copy()
        if len(sample) > 50:
            X_s = np.nan_to_num(sample[feature_cols].values.astype(np.float32))
            y_s = sample["label"].values

            real_acc = 0
            shuf_acc = 0
            for _ in range(3):
                split = len(X_s) // 2
                m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, verbose=-1)
                m.fit(X_s[:split], y_s[:split])
                real_acc += (m.predict(X_s[split:]) == y_s[split:]).mean()

                y_shuf = np.random.permutation(y_s[:split])
                m2 = lgb.LGBMClassifier(n_estimators=50, max_depth=3, verbose=-1)
                m2.fit(X_s[:split], y_shuf)
                shuf_acc += (m2.predict(X_s[split:]) == y_s[split:]).mean()

            real_acc /= 3
            shuf_acc /= 3
            lift = real_acc - shuf_acc
            gates["G4_perm_lift"] = round(lift, 3)
            gates["G4_pass"] = lift > 0.02
            fprint(f"    G4 Perm lift: {lift:.3f} ({'PASS' if lift > 0.02 else 'FAIL'})")
        else:
            gates["G4_perm_lift"] = None
            gates["G4_pass"] = False
    except Exception as e:
        gates["G4_perm_lift"] = None
        gates["G4_pass"] = False

    # Gate 5: Overfit check (train acc vs test acc gap < 0.15)
    try:
        sample_dates2 = dates[TRAIN_DAYS:TRAIN_DAYS + TRAIN_DAYS + TEST_DAYS]
        split_date = dates[TRAIN_DAYS + TRAIN_DAYS]
        tr = df[df["date"].isin(sample_dates2[:TRAIN_DAYS])].copy()
        te = df[df["date"].isin(sample_dates2[TRAIN_DAYS:])].copy()
        if len(tr) > 50 and len(te) > 20:
            X_tr = np.nan_to_num(tr[feature_cols].values.astype(np.float32))
            y_tr = tr["label"].values
            X_te = np.nan_to_num(te[feature_cols].values.astype(np.float32))
            y_te = te["label"].values
            m = lgb.LGBMClassifier(n_estimators=150, max_depth=4, verbose=-1)
            m.fit(X_tr, y_tr)
            train_acc = (m.predict(X_tr) == y_tr).mean()
            test_acc = (m.predict(X_te) == y_te).mean()
            gap = train_acc - test_acc
            gates["G5_overfit_gap"] = round(gap, 3)
            gates["G5_pass"] = gap < 0.15
            fprint(f"    G5 Overfit gap: {gap:.3f} ({'PASS' if gap < 0.15 else 'FAIL'})")
        else:
            gates["G5_overfit_gap"] = None
            gates["G5_pass"] = False
    except Exception as e:
        gates["G5_overfit_gap"] = None
        gates["G5_pass"] = False

    n_pass = sum(1 for k, v in gates.items() if k.endswith("_pass") and v)
    gates["total_pass"] = n_pass
    gates["total_gates"] = 5
    fprint(f"    TOTAL: {n_pass}/5 gates passed")

    return gates


# ── MAIN ──────────────────────────────────────────────────────

def main():
    t0 = time.time()
    fprint("="*70)
    fprint("SPREAD OUTCOME PREDICTOR v1")
    fprint("Hypothesis: training on spread WIN/LOSS > equity returns")
    fprint("="*70)

    # Download data
    data = download_data()

    # Build dataset
    df = build_dataset(data)
    df.to_parquet(OUTPUT / "dataset.parquet")

    feature_cols = FEATURE_NAMES
    results = {}

    # ── VARIANT A: LGBM on spread outcomes (bull) ──
    fprint("\n" + "="*70)
    fprint("VARIANT A: LGBM on SPREAD OUTCOMES (bull_win)")
    df_a = df.copy()
    df_a["label"] = df_a["bull_win"]
    rankings_a = walkforward_variant(df_a, "A: LGBM-SpreadOutcome", feature_cols)
    trades_a, eq_a, curve_a, wpnl_a = simulate_portfolio(rankings_a, data, "A")
    results["A_lgbm_spread"] = compute_metrics(trades_a, eq_a, curve_a, wpnl_a, "A: LGBM-SpreadOutcome")
    gates_a = adversarial_validation(df_a, rankings_a, feature_cols, "A")
    if results["A_lgbm_spread"]:
        results["A_lgbm_spread"]["adversarial"] = gates_a

    # ── VARIANT B: MLP on spread outcomes ──
    fprint("\n" + "="*70)
    fprint("VARIANT B: MLP (64-32) on SPREAD OUTCOMES (bull_win)")
    df_b = df.copy()
    df_b["label"] = df_b["bull_win"]
    rankings_b = walkforward_variant(df_b, "B: MLP-SpreadOutcome", feature_cols)
    trades_b, eq_b, curve_b, wpnl_b = simulate_portfolio(rankings_b, data, "B")
    results["B_mlp_spread"] = compute_metrics(trades_b, eq_b, curve_b, wpnl_b, "B: MLP-SpreadOutcome")
    gates_b = adversarial_validation(df_b, rankings_b, feature_cols, "B")
    if results["B_mlp_spread"]:
        results["B_mlp_spread"]["adversarial"] = gates_b

    # ── VARIANT C: GRU on spread outcomes ──
    fprint("\n" + "="*70)
    fprint("VARIANT C: GRU (32h, 10 lookback) on SPREAD OUTCOMES (bull_win)")
    df_c = df.copy()
    df_c["label"] = df_c["bull_win"]
    rankings_c = walkforward_variant(df_c, "C: GRU-SpreadOutcome", feature_cols)
    trades_c, eq_c, curve_c, wpnl_c = simulate_portfolio(rankings_c, data, "C")
    results["C_gru_spread"] = compute_metrics(trades_c, eq_c, curve_c, wpnl_c, "C: GRU-SpreadOutcome")
    gates_c = adversarial_validation(df_c, rankings_c, feature_cols, "C")
    if results["C_gru_spread"]:
        results["C_gru_spread"]["adversarial"] = gates_c

    # ── VARIANT D: LGBM on equity returns (control) ──
    fprint("\n" + "="*70)
    fprint("VARIANT D: LGBM on EQUITY RETURNS (control)")
    df_d = df.copy()
    # Binary: did equity go up over DTE period?
    df_d["label"] = (df_d["fwd_ret"] > 0).astype(int)
    rankings_d = walkforward_variant(df_d, "D: LGBM-EquityReturns", feature_cols)
    trades_d, eq_d, curve_d, wpnl_d = simulate_portfolio(rankings_d, data, "D")
    results["D_lgbm_equity"] = compute_metrics(trades_d, eq_d, curve_d, wpnl_d, "D: LGBM-EquityReturns")
    gates_d = adversarial_validation(df_d, rankings_d, feature_cols, "D")
    if results["D_lgbm_equity"]:
        results["D_lgbm_equity"]["adversarial"] = gates_d

    # ── COMPARISON ──
    fprint("\n" + "="*70)
    fprint("COMPARISON SUMMARY")
    fprint("="*70)
    fprint(f"{'Variant':<30} {'Trades':>6} {'WR%':>6} {'Sharpe':>7} {'Sort':>7} {'PF':>6} {'CAGR%':>7} {'MaxDD%':>7} {'Final$':>8}")
    fprint("-" * 95)
    for key in ["A_lgbm_spread", "B_mlp_spread", "C_gru_spread", "D_lgbm_equity"]:
        r = results.get(key)
        if r:
            fprint(f"{r['name']:<30} {r['n_trades']:>6} {r['win_rate']:>6.1f} {r['sharpe']:>7.2f} "
                   f"{r['sortino']:>7.2f} {r['profit_factor']:>6.2f} {r['cagr_pct']:>7.1f} "
                   f"{r['maxdd_pct']:>7.1f} {r['final_equity']:>8.0f}")

    # Key comparison: A vs D (same model, different target)
    a_res = results.get("A_lgbm_spread")
    d_res = results.get("D_lgbm_equity")
    if a_res and d_res:
        fprint(f"\nKEY FINDING — Same model (LGBM), different target:")
        fprint(f"  Spread outcome target: Sharpe {a_res['sharpe']:.2f}, WR {a_res['win_rate']:.1f}%")
        fprint(f"  Equity return target:  Sharpe {d_res['sharpe']:.2f}, WR {d_res['win_rate']:.1f}%")
        delta = a_res['sharpe'] - d_res['sharpe']
        fprint(f"  Delta Sharpe: {delta:+.2f} ({'spread target wins' if delta > 0 else 'equity target wins'})")

    # Best neural net vs best LGBM
    best_nn = None
    for key in ["B_mlp_spread", "C_gru_spread"]:
        r = results.get(key)
        if r and (best_nn is None or r["sharpe"] > best_nn["sharpe"]):
            best_nn = r
    if best_nn and a_res:
        fprint(f"\nNeural net vs LGBM (both on spread target):")
        fprint(f"  Best NN ({best_nn['name']}): Sharpe {best_nn['sharpe']:.2f}")
        fprint(f"  LGBM baseline:              Sharpe {a_res['sharpe']:.2f}")
        delta2 = best_nn["sharpe"] - a_res["sharpe"]
        fprint(f"  Delta: {delta2:+.2f} ({'NN wins' if delta2 > 0 else 'LGBM wins'})")

    # ── SAVE RESULTS ──
    elapsed = time.time() - t0
    output = {
        "experiment": "spread_outcome_predictor_v1",
        "hypothesis": "Training on spread WIN/LOSS > equity returns for sector options",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed),
        "device": str(DEVICE),
        "params": {
            "dte": DTE,
            "otm_pct": OTM_PCT,
            "spread_width": SPREAD_WIDTH,
            "iv_mult": IV_MULT,
            "haircut": HAIRCUT,
            "commission": SPREAD_COMM,
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "top_k": TOP_K,
            "capital": CAP,
        },
        "results": {},
    }
    for key, r in results.items():
        if r:
            output["results"][key] = r

    with open(OUTPUT / "results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Save equity curves
    curves = {}
    for name, curve in [("A", curve_a), ("B", curve_b), ("C", curve_c), ("D", curve_d)]:
        curves[name] = [round(v, 2) for v in curve]
    with open(OUTPUT / "equity_curves.json", "w") as f:
        json.dump(curves, f)

    # Save trade logs
    for name, trades in [("A", trades_a), ("B", trades_b), ("C", trades_c), ("D", trades_d)]:
        with open(OUTPUT / f"trades_{name}.json", "w") as f:
            json.dump(trades, f, indent=2, default=str)

    fprint(f"\nResults saved to {OUTPUT}")
    fprint(f"Total runtime: {elapsed/60:.1f} minutes")

    # ── MLFLOW LOGGING ──
    if MLFLOW_OK:
        try:
            exp = mlflow.set_experiment("spread_outcome_predictor_v1")
            with mlflow.start_run(run_name=f"v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({
                    "dte": DTE, "otm_pct": OTM_PCT, "spread_width": SPREAD_WIDTH,
                    "iv_mult": IV_MULT, "haircut": HAIRCUT, "commission": SPREAD_COMM,
                    "train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "top_k": TOP_K,
                    "device": str(DEVICE),
                })
                for key, r in results.items():
                    if r:
                        pfx = key.split("_")[0]
                        mlflow.log_metrics({
                            f"{pfx}_sharpe": r["sharpe"],
                            f"{pfx}_sortino": r["sortino"],
                            f"{pfx}_wr": r["win_rate"],
                            f"{pfx}_pf": r["profit_factor"],
                            f"{pfx}_cagr": r["cagr_pct"],
                            f"{pfx}_maxdd": r["maxdd_pct"],
                            f"{pfx}_trades": r["n_trades"],
                        })
                mlflow.log_artifact(str(OUTPUT / "results.json"))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint("\nDONE.")
    return output


if __name__ == "__main__":
    main()
