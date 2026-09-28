#!/usr/bin/env python3
"""
ML Volatility Breakout — PAPER ENGINE
======================================
Runs daily at 16:30 ET (PM2 cron). Predicts which of 30 mega-cap stocks will
move >8% in the next 21 trading days. When confident, simulates buying ATM
straddles. Tracks paper PnL of open positions.

Based on validated research (Sharpe 1.11, Sortino 2.45, MaxDD -10.2%, 4/4
adversarial gates PASS).

Output: /home/jupiter/Lvl3Quant/output/ml_vol_breakout_paper/
  - positions_log.json   (daily position snapshots)
  - signals_history.json (every signal generated)
  - daily_pnl.csv        (daily mark-to-market PnL)
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_vol_breakout_paper"
OUTPUT.mkdir(parents=True, exist_ok=True)

POSITIONS_LOG = OUTPUT / "positions_log.json"
SIGNALS_HISTORY = OUTPUT / "signals_history.json"
DAILY_PNL = OUTPUT / "daily_pnl.csv"

# ---------------------------------------------------------------------------
# Constants (from validated research)
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 100_000
RISK_PER_TRADE = 0.02       # 2% risk per trade
MAX_CONCURRENT = 5           # max concurrent positions
IV_MARKUP = 1.15             # IV = realized vol * 1.15
BID_ASK_HAIRCUT = 0.15       # 15% haircut on straddle premium
MOVE_THRESHOLD = 0.08        # 8% move threshold

TRAIN_DAYS = 252             # 1 year sliding window
TARGET_HORIZON = 21          # predict 21-day max move
LOOKBACK_MAX = 63            # max feature lookback
ML_THRESHOLD = 0.50          # default prediction threshold

# Universe: 30 liquid mega-cap stocks (exact copy from research)
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "MA",
    "JNJ", "UNH", "HD", "PG", "BAC", "XOM", "CVX", "COST", "CRM", "NFLX",
    "AMD", "ORCL", "ADBE", "LLY", "MRK", "PEP", "KO", "WMT", "DIS", "GS",
]

# Cross-asset tickers for features (exact copy from research)
CROSS_ASSET = ["SPY", "GLD", "TLT", "HYG"]
VIX_TICKER = "^VIX"

# Sector mapping for dummy features (exact copy from research)
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "ConsDisc",
    "NVDA": "Tech", "META": "Tech", "TSLA": "ConsDisc", "JPM": "Financials",
    "V": "Financials", "MA": "Financials", "JNJ": "Healthcare", "UNH": "Healthcare",
    "HD": "ConsDisc", "PG": "ConsStaples", "BAC": "Financials", "XOM": "Energy",
    "CVX": "Energy", "COST": "ConsStaples", "CRM": "Tech", "NFLX": "Tech",
    "AMD": "Tech", "ORCL": "Tech", "ADBE": "Tech", "LLY": "Healthcare",
    "MRK": "Healthcare", "PEP": "ConsStaples", "KO": "ConsStaples",
    "WMT": "ConsStaples", "DIS": "Tech", "GS": "Financials",
}
SECTORS = sorted(set(SECTOR_MAP.values()))


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def get_data(lookback_days=700):
    """Download recent OHLCV for universe + cross-asset tickers."""
    all_tickers = UNIVERSE + CROSS_ASSET + [VIX_TICKER]
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    print(f"Downloading {len(all_tickers)} tickers from {start}...")
    data = {}
    failed = []

    for tkr in all_tickers:
        try:
            df = yf.download(tkr, start=start, auto_adjust=True, progress=False)
            if df is not None and len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[tkr] = df
            else:
                failed.append(tkr)
        except Exception as e:
            failed.append(tkr)
            print(f"  WARNING: {tkr} download failed: {e}")

    if failed:
        print(f"  Failed tickers: {failed}")

    print(f"  Got {len(data)} tickers, date range: "
          f"{min(d.index[0] for d in data.values()).date()} to "
          f"{max(d.index[-1] for d in data.values()).date()}")
    return data


# ============================================================================
# FEATURE ENGINEERING (exact copy from research script)
# ============================================================================

def compute_features(stock_df: pd.DataFrame, ticker: str,
                     spy_df: pd.DataFrame, vix_df: pd.DataFrame,
                     gld_df: pd.DataFrame, tlt_df: pd.DataFrame,
                     hyg_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for a single stock."""
    df = stock_df[["Close", "Volume"]].copy()
    df.columns = ["close", "volume"]
    df = df.dropna()

    if len(df) < LOOKBACK_MAX + 10:
        return pd.DataFrame()

    ret = df["close"].pct_change()
    log_ret = np.log(df["close"] / df["close"].shift(1))

    feats = pd.DataFrame(index=df.index)

    # --- Historical realized vol ---
    for w in [5, 10, 21, 63]:
        feats[f"rvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    # --- Vol-of-vol ---
    feats["vol_of_vol_21d"] = feats["rvol_21d"].rolling(21).std()
    feats["vol_of_vol_63d"] = feats["rvol_63d"].rolling(21).std()

    # --- Recent max move ---
    feats["max_abs_ret_10d"] = ret.abs().rolling(10).max()
    feats["max_abs_ret_21d"] = ret.abs().rolling(21).max()

    # --- Bollinger bandwidth (vol compression -> breakout) ---
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    feats["bband_width"] = (2 * std20) / sma20
    feats["bband_pctb"] = (df["close"] - (sma20 - 2 * std20)) / (4 * std20)

    # --- Volume spike ---
    vol_ma20 = df["volume"].rolling(20).mean()
    feats["volume_spike"] = df["volume"] / vol_ma20.replace(0, np.nan)
    feats["volume_spike_max5d"] = feats["volume_spike"].rolling(5).max()

    # --- RSI ---
    delta = ret.copy()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    feats["rsi_14"] = rsi
    feats["rsi_extreme"] = ((rsi > 70) | (rsi < 30)).astype(float)

    # --- Days since last big move ---
    big_move_mask = ret.abs() > 0.05
    feats["days_since_5pct_move"] = big_move_mask.groupby(
        big_move_mask.cumsum()
    ).cumcount()
    feats["days_since_5pct_move"] = feats["days_since_5pct_move"].clip(upper=252)

    # --- IV rank proxy (vol percentile over 252d) ---
    feats["vol_percentile_252d"] = feats["rvol_21d"].rolling(252).rank(pct=True)

    # --- Vol ratio (short/long) -- compression indicator ---
    feats["vol_ratio_5_63"] = feats["rvol_5d"] / feats["rvol_63d"].replace(0, np.nan)
    feats["vol_ratio_10_63"] = feats["rvol_10d"] / feats["rvol_63d"].replace(0, np.nan)

    # --- Momentum features ---
    feats["ret_5d"] = ret.rolling(5).sum()
    feats["ret_21d"] = ret.rolling(21).sum()
    feats["abs_ret_5d"] = ret.abs().rolling(5).sum()

    # --- Cross-asset features ---
    def _safe_align(ext_df, col="Close"):
        if ext_df is None or ext_df.empty:
            return pd.Series(np.nan, index=df.index)
        s = ext_df[col].reindex(df.index).ffill()
        return s

    # SPY vol
    spy_close = _safe_align(spy_df)
    spy_ret = spy_close.pct_change()
    feats["spy_rvol_21d"] = np.log(spy_close / spy_close.shift(1)).rolling(21).std() * np.sqrt(252)
    feats["spy_ret_21d"] = spy_ret.rolling(21).sum()

    # VIX
    vix_close = _safe_align(vix_df)
    feats["vix_level"] = vix_close
    feats["vix_pctile_252d"] = vix_close.rolling(252).rank(pct=True)
    feats["vix_change_5d"] = vix_close.pct_change(5)

    # GLD vol
    gld_close = _safe_align(gld_df)
    feats["gld_rvol_21d"] = np.log(gld_close / gld_close.shift(1)).rolling(21).std() * np.sqrt(252)

    # TLT vol
    tlt_close = _safe_align(tlt_df)
    feats["tlt_rvol_21d"] = np.log(tlt_close / tlt_close.shift(1)).rolling(21).std() * np.sqrt(252)

    # Credit spread proxy: HYG - TLT return spread
    hyg_close = _safe_align(hyg_df)
    feats["credit_spread_21d"] = hyg_close.pct_change(21) - tlt_close.pct_change(21)

    # --- Sector dummies ---
    sector = SECTOR_MAP.get(ticker, "Unknown")
    for s in SECTORS:
        feats[f"sector_{s}"] = 1.0 if s == sector else 0.0

    # --- Beta to SPY ---
    cov_60 = ret.rolling(60).cov(spy_ret)
    var_60 = spy_ret.rolling(60).var()
    feats["beta_60d"] = cov_60 / var_60.replace(0, np.nan)

    return feats


def compute_target(stock_df: pd.DataFrame) -> pd.Series:
    """Binary target: will stock move >8% (either direction) in next 21 trading days?"""
    close = stock_df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    fwd_returns = pd.DataFrame(index=close.index)
    for d in range(1, TARGET_HORIZON + 1):
        fwd_returns[f"fwd_{d}"] = close.shift(-d) / close - 1

    max_abs_fwd = fwd_returns.abs().max(axis=1)
    target = (max_abs_fwd >= MOVE_THRESHOLD).astype(int)
    return target


# ============================================================================
# BLACK-SCHOLES STRADDLE PRICING (from research script)
# ============================================================================

def bs_call(S, K, T, sigma, r=0.04):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, sigma, r=0.04):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def straddle_price(S, sigma_annual, T_days=21, r=0.04):
    """ATM straddle price (call + put) using BS with IV markup + bid-ask haircut."""
    T = T_days / 252.0
    iv = sigma_annual * IV_MARKUP
    K = S  # ATM
    call = bs_call(S, K, T, iv, r)
    put = bs_put(S, K, T, iv, r)
    premium = call + put
    premium *= (1 + BID_ASK_HAIRCUT)
    return premium, iv


def straddle_mtm(S_current, K, T_remaining_days, sigma_annual, r=0.04):
    """Mark-to-market value of an open straddle position."""
    if T_remaining_days <= 0:
        # At expiry: intrinsic value only
        return abs(S_current - K)
    T = T_remaining_days / 252.0
    iv = sigma_annual * IV_MARKUP
    call = bs_call(S_current, K, T, iv, r)
    put = bs_put(S_current, K, T, iv, r)
    return call + put


# ============================================================================
# PANEL BUILDER (for training)
# ============================================================================

def build_panel(data: dict) -> pd.DataFrame:
    """Build feature panel across all stocks and dates."""
    spy_df = data.get("SPY")
    vix_df = data.get("^VIX")
    gld_df = data.get("GLD")
    tlt_df = data.get("TLT")
    hyg_df = data.get("HYG")

    all_rows = []
    for ticker in UNIVERSE:
        if ticker not in data:
            continue

        stock_df = data[ticker]
        feats = compute_features(stock_df, ticker, spy_df, vix_df, gld_df, tlt_df, hyg_df)
        target = compute_target(stock_df)

        if feats.empty:
            continue

        common_idx = feats.index.intersection(target.index)
        feats = feats.loc[common_idx]
        target = target.loc[common_idx]

        feats["target"] = target
        feats["ticker"] = ticker
        feats["close"] = stock_df["Close"].reindex(common_idx)
        if isinstance(feats["close"], pd.DataFrame):
            feats["close"] = feats["close"].iloc[:, 0]

        all_rows.append(feats)

    if not all_rows:
        return pd.DataFrame()

    panel = pd.concat(all_rows, axis=0).sort_index()
    return panel


# ============================================================================
# PAPER ENGINE: DAILY RUN
# ============================================================================

def load_open_positions():
    """Load currently open straddle positions from positions log."""
    if not POSITIONS_LOG.exists():
        return []
    with open(POSITIONS_LOG, "r") as f:
        history = json.load(f)
    # Find all positions that haven't expired (entered within last 30 calendar days)
    open_pos = []
    today = datetime.now()
    for entry in history:
        if "open_positions" not in entry:
            continue
        for pos in entry["open_positions"]:
            entry_date = datetime.fromisoformat(pos["entry_date"])
            days_held = (today - entry_date).days
            if days_held <= 30 and pos.get("status", "open") == "open":
                open_pos.append(pos)
    return open_pos


def generate_daily_signal():
    """Main: download data, train ML, generate signals, track positions."""
    print(f"\n{'='*80}")
    print(f"ML VOL BREAKOUT PAPER ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")

    # ── 1. Download data ──────────────────────────────────────────────────
    data = get_data(lookback_days=700)
    if len(data) < 15:
        print("ERROR: Too few tickers downloaded. Aborting.")
        return {}

    # ── 2. Build feature panel ────────────────────────────────────────────
    panel = build_panel(data)
    if panel.empty:
        print("ERROR: Empty feature panel. Aborting.")
        return {}

    feature_cols = [c for c in panel.columns if c not in ["target", "ticker", "close"]]
    dates = panel.index.unique().sort_values()
    today_date = dates[-1]
    today_str = today_date.strftime("%Y-%m-%d")

    print(f"Panel: {len(panel)} rows, {len(dates)} dates, {len(feature_cols)} features")
    print(f"Latest date in data: {today_str}")

    # ── 3. Train GBM on most recent TRAIN_DAYS ───────────────────────────
    # Use all data EXCEPT the last day for training (last day = prediction target)
    # Training target requires forward-looking data, so we train on rows where
    # target is known (i.e., at least TARGET_HORIZON days before today)
    cutoff_date = dates[-1] - pd.Timedelta(days=TARGET_HORIZON + 5)
    train_start = cutoff_date - pd.Timedelta(days=int(TRAIN_DAYS * 2.0))

    train_mask = (panel.index >= train_start) & (panel.index <= cutoff_date)
    train_panel = panel[train_mask].copy()

    # Drop rows with NaN features or target
    valid = train_panel[feature_cols].notna().all(axis=1) & train_panel["target"].notna()
    train_panel = train_panel[valid]

    if len(train_panel) < 200:
        print(f"ERROR: Too few training samples ({len(train_panel)}). Aborting.")
        return {}

    X_train = train_panel[feature_cols].values
    y_train = train_panel["target"].values.astype(int)

    pos_rate = y_train.mean()
    print(f"Training: {len(train_panel)} samples, positive rate: {pos_rate:.1%}")

    # Compute sample weight to handle class imbalance
    pos_weight = max(1.0, (1 - pos_rate) / max(pos_rate, 0.01))

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        max_features=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    # Apply class weight via sample_weight
    sample_weights = np.where(y_train == 1, pos_weight, 1.0)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    train_acc = model.score(X_train, y_train, sample_weight=sample_weights)
    print(f"Train accuracy: {train_acc:.1%}")

    # Feature importance (top 10)
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nTop 10 features:")
    for feat, imp in importance.head(10).items():
        print(f"  {feat}: {imp:.4f}")

    # ── 4. Generate predictions for today ─────────────────────────────────
    today_panel = panel[panel.index == today_date].copy()
    valid_today = today_panel[feature_cols].notna().all(axis=1)
    today_panel = today_panel[valid_today]

    if today_panel.empty:
        print("No valid feature rows for today. Aborting.")
        return {}

    X_today = today_panel[feature_cols].values
    probs = model.predict_proba(X_today)[:, 1]
    today_panel = today_panel.copy()
    today_panel["pred_prob"] = probs

    print(f"\nPredictions for {today_str}: {len(today_panel)} stocks scored")
    print(f"Prob distribution: min={probs.min():.3f}, median={np.median(probs):.3f}, "
          f"max={probs.max():.3f}, >threshold={(probs >= ML_THRESHOLD).sum()}")

    # ── 5. Load existing state ────────────────────────────────────────────
    if POSITIONS_LOG.exists():
        with open(POSITIONS_LOG, "r") as f:
            positions_history = json.load(f)
    else:
        positions_history = []

    if SIGNALS_HISTORY.exists():
        with open(SIGNALS_HISTORY, "r") as f:
            signals_history = json.load(f)
    else:
        signals_history = []

    # Get capital from last entry or use initial
    if positions_history:
        capital = positions_history[-1].get("capital", INITIAL_CAPITAL)
        open_positions = positions_history[-1].get("open_positions", [])
    else:
        capital = INITIAL_CAPITAL
        open_positions = []

    # ── 6. Mark-to-market open positions ──────────────────────────────────
    mtm_pnl = 0.0
    updated_positions = []
    closed_today = []

    for pos in open_positions:
        ticker = pos["ticker"]
        entry_date = datetime.fromisoformat(pos["entry_date"])
        days_held = (datetime.now() - entry_date).days
        trading_days_remaining = max(0, TARGET_HORIZON - int(days_held * 5 / 7))

        if ticker in data:
            close_series = data[ticker]["Close"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            current_price = float(close_series.iloc[-1])
        else:
            current_price = pos["entry_price"]

        if days_held > 30 or trading_days_remaining <= 0:
            # Position expired -- compute final PnL
            intrinsic = abs(current_price - pos["strike"]) * 100 * pos["n_contracts"]
            final_pnl = intrinsic - pos["total_premium"]
            mtm_pnl += final_pnl
            pos_copy = dict(pos)
            pos_copy["status"] = "closed"
            pos_copy["exit_date"] = today_str
            pos_copy["exit_price"] = current_price
            pos_copy["realized_pnl"] = round(final_pnl, 2)
            closed_today.append(pos_copy)
            print(f"  CLOSED: {ticker} straddle @ ${pos['strike']:.2f} -> "
                  f"${current_price:.2f}, PnL=${final_pnl:,.0f}")
        else:
            # Still open -- mark to market
            rvol = pos.get("rvol_21d", 0.3)
            mtm_value = straddle_mtm(current_price, pos["strike"],
                                     trading_days_remaining, rvol) * 100 * pos["n_contracts"]
            unrealized = mtm_value - pos["total_premium"]
            pos_copy = dict(pos)
            pos_copy["current_price"] = current_price
            pos_copy["mtm_value"] = round(mtm_value, 2)
            pos_copy["unrealized_pnl"] = round(unrealized, 2)
            pos_copy["days_held"] = days_held
            pos_copy["status"] = "open"
            updated_positions.append(pos_copy)

    capital += mtm_pnl

    # ── 7. Open new positions based on signals ────────────────────────────
    active_count = len(updated_positions)
    new_signals = today_panel[today_panel["pred_prob"] >= ML_THRESHOLD].sort_values(
        "pred_prob", ascending=False
    )

    new_positions = []
    all_signals_today = []

    for _, row in today_panel.iterrows():
        signal_entry = {
            "date": today_str,
            "ticker": row["ticker"],
            "pred_prob": round(float(row["pred_prob"]), 4),
            "close": round(float(row["close"]), 2),
            "rvol_21d": round(float(row.get("rvol_21d", 0.3)), 4),
            "vix_level": round(float(row.get("vix_level", 0)), 2),
            "triggered": bool(row["pred_prob"] >= ML_THRESHOLD),
        }
        all_signals_today.append(signal_entry)

    for _, row in new_signals.iterrows():
        if active_count >= MAX_CONCURRENT:
            break

        ticker = row["ticker"]
        S = float(row["close"])
        prob = float(row["pred_prob"])

        if pd.isna(S) or S <= 0:
            continue

        # Skip if already have a position in this ticker
        existing_tickers = [p["ticker"] for p in updated_positions + new_positions]
        if ticker in existing_tickers:
            continue

        # Compute realized vol for straddle pricing
        rvol = float(row.get("rvol_21d", 0.3))
        if pd.isna(rvol) or rvol <= 0:
            rvol = 0.3

        # Price the straddle
        premium_per_share, iv_used = straddle_price(S, rvol)

        # Position size: risk RISK_PER_TRADE of capital
        risk_amount = capital * RISK_PER_TRADE
        n_contracts = max(1, int(risk_amount / (premium_per_share * 100)))
        total_premium = premium_per_share * 100 * n_contracts

        # Cap at 10% of capital per trade
        if total_premium > capital * 0.10:
            n_contracts = max(1, int(capital * 0.10 / (premium_per_share * 100)))
            total_premium = premium_per_share * 100 * n_contracts

        new_pos = {
            "ticker": ticker,
            "entry_date": today_str,
            "entry_price": round(S, 2),
            "strike": round(S, 2),  # ATM
            "rvol_21d": round(rvol, 4),
            "iv_used": round(iv_used, 4),
            "premium_per_share": round(premium_per_share, 2),
            "n_contracts": n_contracts,
            "total_premium": round(total_premium, 2),
            "pred_prob": round(prob, 4),
            "status": "open",
            "current_price": round(S, 2),
            "mtm_value": round(total_premium, 2),
            "unrealized_pnl": 0.0,
            "days_held": 0,
        }
        new_positions.append(new_pos)
        active_count += 1

        print(f"  NEW: {ticker} straddle @ ${S:.2f}, prob={prob:.3f}, "
              f"premium=${premium_per_share:.2f}/sh, {n_contracts} contracts, "
              f"cost=${total_premium:,.0f}")

    all_open = updated_positions + new_positions

    # ── 8. Save state ─────────────────────────────────────────────────────
    total_unrealized = sum(p.get("unrealized_pnl", 0) for p in all_open)
    total_premium_at_risk = sum(p.get("total_premium", 0) for p in all_open)

    log_entry = {
        "date": today_str,
        "timestamp": datetime.now().isoformat(),
        "capital": round(capital, 2),
        "n_open": len(all_open),
        "n_new": len(new_positions),
        "n_closed": len(closed_today),
        "total_unrealized_pnl": round(total_unrealized, 2),
        "total_premium_at_risk": round(total_premium_at_risk, 2),
        "realized_pnl_today": round(mtm_pnl, 2),
        "open_positions": all_open,
        "closed_today": closed_today,
        "vix": round(float(today_panel["vix_level"].iloc[0]), 2) if "vix_level" in today_panel.columns else None,
    }

    positions_history.append(log_entry)
    with open(POSITIONS_LOG, "w") as f:
        json.dump(positions_history, f, indent=2, default=str)

    signals_history.extend(all_signals_today)
    with open(SIGNALS_HISTORY, "w") as f:
        json.dump(signals_history, f, indent=2, default=str)

    # ── 9. Update daily PnL CSV ───────────────────────────────────────────
    pnl_row = pd.DataFrame([{
        "date": today_str,
        "capital": round(capital, 2),
        "n_positions": len(all_open),
        "realized_pnl": round(mtm_pnl, 2),
        "unrealized_pnl": round(total_unrealized, 2),
        "total_pnl": round(capital - INITIAL_CAPITAL, 2),
        "return_pct": round((capital / INITIAL_CAPITAL - 1) * 100, 4),
    }])

    if DAILY_PNL.exists():
        existing = pd.read_csv(DAILY_PNL)
        # Don't double-append same date
        if today_str not in existing["date"].values:
            pnl_df = pd.concat([existing, pnl_row], ignore_index=True)
        else:
            existing.loc[existing["date"] == today_str] = pnl_row.iloc[0].values
            pnl_df = existing
    else:
        pnl_df = pnl_row

    pnl_df.to_csv(DAILY_PNL, index=False)

    # ── 10. Print summary ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"PORTFOLIO SUMMARY ({today_str})")
    print(f"{'─'*60}")
    print(f"  Capital:      ${capital:>12,.2f}")
    print(f"  Total Return: {(capital / INITIAL_CAPITAL - 1) * 100:>+10.2f}%")
    print(f"  Open:         {len(all_open)} positions")
    print(f"  New today:    {len(new_positions)}")
    print(f"  Closed today: {len(closed_today)}")
    print(f"  Unrealized:   ${total_unrealized:>+12,.2f}")
    print(f"  Premium risk: ${total_premium_at_risk:>12,.2f}")

    if all_open:
        print(f"\n  {'Ticker':6s} {'Strike':>8s} {'Curr':>8s} {'Prob':>6s} "
              f"{'Days':>5s} {'Unreal PnL':>12s}")
        print(f"  {'─'*50}")
        for p in all_open:
            print(f"  {p['ticker']:6s} ${p['strike']:>7.2f} ${p['current_price']:>7.2f} "
                  f"{p['pred_prob']:>5.3f} {p['days_held']:>4d}d "
                  f"${p.get('unrealized_pnl', 0):>+10,.2f}")

    if len(pnl_df) > 5:
        rets = pnl_df["return_pct"].astype(float).diff().dropna() / 100
        if len(rets) > 2 and rets.std() > 0:
            sharpe = rets.mean() / rets.std() * np.sqrt(252)
            print(f"\n  Running Sharpe: {sharpe:.2f} ({len(pnl_df)} days tracked)")

    # Log high-confidence signals for review
    triggered = [s for s in all_signals_today if s["triggered"]]
    if triggered:
        print(f"\n  Triggered signals ({len(triggered)}):")
        for s in sorted(triggered, key=lambda x: -x["pred_prob"]):
            print(f"    {s['ticker']:6s} prob={s['pred_prob']:.3f} close=${s['close']:.2f}")

    print(f"\nDone. Output: {OUTPUT}")
    return {p["ticker"]: p for p in all_open}


if __name__ == "__main__":
    generate_daily_signal()
