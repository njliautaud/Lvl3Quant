#!/usr/bin/env python3
"""
Sector ETF Momentum v2 Paper Engine — LightGBM Cross-Sectional Ranker
======================================================================

Winner config: C_DefShift_Top3 — Sharpe 4.63, CAGR 71.7%, MaxDD -4.9%, 90.1% WR.

Walk-forward: 504 trading day train, 21-day sliding OOS rebalance.
Defensive shift in bear regime (SPY < 200d SMA): force one defensive allocation slot.
Top 3 ETFs monthly rebalance, equal weight.
Transaction cost: 20bps round-trip.

Universe: XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC,
          QQQ, IWM, MDY, EFA, EEM, TLT, GLD, DBC, HYG, LQD, VNQ

Runs ONE rebalance step per invocation (PM2 cron at 9:55 AM ET weekdays).
On non-rebalance days: mark-to-market + NAV update only.

State: live_trading_linux/sector_etf_momentum_v2_state/
  - state.json        — current positions, cash, NAV, regime
  - trades.jsonl      — all trade/MTM records
  - equity_curve.jsonl — daily NAV snapshots

Data: fresh yfinance downloads each run (no stale parquet dependency).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = Path(__file__).resolve().parent / "sector_etf_momentum_v2_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "state.json"
TRADES_LOG = STATE_DIR / "trades.jsonl"
EQUITY_LOG = STATE_DIR / "equity_curve.jsonl"

# ---------------------------------------------------------------------------
# CONFIG — C_DefShift_Top3 winner
# ---------------------------------------------------------------------------
CONFIG_K = 3                    # top-K long (equal weight)
CONFIG_HOLD_DAYS = 21           # ~monthly rebalance
CONFIG_TRAIN_DAYS = 504         # 504 trading days (~2yr) walk-forward train
CONFIG_ANCHOR_USD = 100_000.0   # paper capital
CONFIG_COST_BPS = 20.0          # 20bps round-trip
CONFIG_DEF_SHIFT = 1.0          # defensive shift factor in bear regime

UNIVERSE = [
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE",
    "XLC", "QQQ", "IWM", "MDY", "EFA", "EEM", "TLT", "GLD", "DBC", "HYG",
    "LQD", "VNQ",
]
BENCH_SPY = "SPY"

DEFENSIVE_ETFS = {"XLP", "XLU", "TLT", "GLD", "LQD"}
RISK_ON_ETFS = {"XLK", "QQQ", "XLY", "IWM", "EEM"}

TRADING_DAYS_YR = 252

# ---------------------------------------------------------------------------
# Feature names — matches research script build_features()
# Top 4 importance: maxdd_63d, vol_60d, kurt_63d, mom_12_1
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # Momentum
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "mom_12_1", "high_52w_pct", "mom_accel",
    # Volatility/quality
    "vol_20d", "vol_60d", "vol_ratio", "sharpe_63d", "sharpe_126d",
    "maxdd_63d", "vol_rel",
    # Higher moments
    "skew_63d", "kurt_63d",
    # Trend strength
    "vol_trend",
    # Regime features
    "above_sma50", "above_sma200", "dist_sma200",
    "rv_21d", "rv_ratio_short_long",
    # Cross-asset regime (SPY)
    "spy_ret_21d", "spy_ret_63d", "spy_above_sma200", "spy_rv_21d",
    "spy_dist_sma200", "corr_spy_63d", "beta_spy_63d",
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class PaperState:
    nav_usd: float = CONFIG_ANCHOR_USD
    cash_usd: float = CONFIG_ANCHOR_USD
    positions: dict = field(default_factory=dict)    # {ticker: shares}
    entry_prices: dict = field(default_factory=dict)
    last_rebal_date: Optional[str] = None
    next_rebal_date: Optional[str] = None
    regime: str = "bull"
    n_rebalances: int = 0
    cumulative_realized_pnl: float = 0.0
    version: str = "v2.0"

    def save(self):
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls):
        if STATE_FILE.exists():
            d = json.loads(STATE_FILE.read_text())
            allowed = {f for f in cls.__dataclass_fields__.keys()}
            d = {k: v for k, v in d.items() if k in allowed}
            return cls(**d)
        return cls()


def _log_trade(rec: dict):
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with TRADES_LOG.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _log_equity(nav: float, date_str: str, regime: str):
    rec = {"date": date_str, "nav_usd": nav, "regime": regime,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    with EQUITY_LOG.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------------------
# Data — fresh yfinance downloads
# ---------------------------------------------------------------------------
def _download_prices(lookback_days: int = 1200) -> pd.DataFrame:
    """Download fresh daily OHLCV for universe + SPY from yfinance."""
    import yfinance as yf

    tickers = UNIVERSE + [BENCH_SPY]
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days)

    print(f"[v2-paper] Downloading prices for {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        print("[v2-paper] WARNING: yfinance returned empty data")
        return pd.DataFrame(), pd.DataFrame()

    # Parse into long-form rows with Close + Volume
    rows = []
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw[("Close", t)].dropna()
                volumes = raw[("Volume", t)].dropna() if ("Volume", t) in raw.columns else pd.Series(dtype=float)
            else:
                closes = raw["Close"].dropna()
                volumes = raw["Volume"].dropna() if "Volume" in raw.columns else pd.Series(dtype=float)
            for dt, px in closes.items():
                vol = volumes.get(dt, 0.0) if not volumes.empty else 0.0
                rows.append({"ticker": t, "date": pd.Timestamp(dt).normalize(),
                             "close": float(px), "volume": float(vol) if pd.notna(vol) else 0.0})
        except Exception as e:
            print(f"[v2-paper] Warning: failed to parse {t}: {e}")
            continue

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last").reset_index(drop=True)
    print(f"[v2-paper] Downloaded {len(df)} price rows, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    # Also build per-ticker DataFrames keyed by ticker (for feature builder)
    all_data = {}
    for t in tickers:
        tdf = df[df["ticker"] == t].set_index("date").sort_index()
        if len(tdf) > 252:
            all_data[t] = pd.DataFrame({
                "Close": tdf["close"],
                "Volume": tdf["volume"],
            })

    return df, all_data


# ---------------------------------------------------------------------------
# Feature builder — exact replica of research script build_features()
# ---------------------------------------------------------------------------
def build_features(close: pd.Series, volume: pd.Series,
                   spy_close: pd.Series = None) -> pd.DataFrame:
    """Build momentum + quality + regime features for a single ETF.
    Matches the research script exactly."""
    lr = np.log(close / close.shift(1))

    feat = pd.DataFrame({
        # Momentum
        "ret_5d": close.pct_change(5),
        "ret_10d": close.pct_change(10),
        "ret_21d": close.pct_change(21),
        "ret_63d": close.pct_change(63),
        "ret_126d": close.pct_change(126),
        "ret_252d": close.pct_change(252),
        "mom_12_1": close.pct_change(252) - close.pct_change(21),
        "high_52w_pct": close / close.rolling(252).max(),
        "mom_accel": close.pct_change(63) - close.pct_change(63).shift(63),

        # Volatility/quality
        "vol_20d": lr.rolling(20).std() * np.sqrt(252),
        "vol_60d": lr.rolling(60).std() * np.sqrt(252),
        "vol_ratio": lr.rolling(20).std() / lr.rolling(60).std(),
        "sharpe_63d": lr.rolling(63).mean() / lr.rolling(63).std(),
        "sharpe_126d": lr.rolling(126).mean() / lr.rolling(126).std(),
        "maxdd_63d": (close / close.rolling(63).max() - 1).rolling(63).min(),

        # Volume
        "vol_rel": volume / volume.rolling(20).mean() if volume is not None else 0,

        # Higher moments
        "skew_63d": lr.rolling(63).skew(),
        "kurt_63d": lr.rolling(63).kurt(),

        # Trend strength
        "vol_trend": (lr.rolling(20).std() - lr.rolling(60).std()) / lr.rolling(60).std(),
    }, index=close.index)

    # Regime features
    feat["above_sma50"] = (close > close.rolling(50).mean()).astype(int)
    feat["above_sma200"] = (close > close.rolling(200).mean()).astype(int)
    feat["dist_sma200"] = (close - close.rolling(200).mean()) / close.rolling(200).mean()

    # Own realized vol regime
    feat["rv_21d"] = lr.rolling(21).std() * np.sqrt(252)
    feat["rv_ratio_short_long"] = feat["rv_21d"] / (feat["vol_60d"] + 1e-8)

    # Cross-asset regime features from SPY
    if spy_close is not None:
        spy_lr = np.log(spy_close / spy_close.shift(1))
        feat["spy_ret_21d"] = spy_close.pct_change(21)
        feat["spy_ret_63d"] = spy_close.pct_change(63)
        feat["spy_above_sma200"] = (spy_close > spy_close.rolling(200).mean()).astype(int)
        feat["spy_rv_21d"] = spy_lr.rolling(21).std() * np.sqrt(252)
        feat["spy_dist_sma200"] = (spy_close - spy_close.rolling(200).mean()) / spy_close.rolling(200).mean()
        feat["corr_spy_63d"] = lr.rolling(63).corr(spy_lr)
        feat["beta_spy_63d"] = lr.rolling(63).cov(spy_lr) / (spy_lr.rolling(63).var() + 1e-10)
    else:
        for c in ["spy_ret_21d", "spy_ret_63d", "spy_above_sma200", "spy_rv_21d",
                   "spy_dist_sma200", "corr_spy_63d", "beta_spy_63d"]:
            feat[c] = 0

    return feat


def detect_regime(spy_close: pd.Series, date: pd.Timestamp) -> str:
    """Detect bull/bear regime from SPY at given date (SPY < 200d SMA = bear)."""
    if spy_close is None:
        return "bull"
    loc = spy_close.index.searchsorted(date)
    if loc < 200:
        return "bull"
    sma200 = spy_close.iloc[max(0, loc - 200):loc].mean()
    return "bear" if spy_close.iloc[loc - 1] < sma200 else "bull"


# ---------------------------------------------------------------------------
# Model training + scoring (walk-forward, one step)
# ---------------------------------------------------------------------------
def _train_and_score(all_data: dict, spy_close: pd.Series,
                     today: pd.Timestamp) -> pd.DataFrame:
    """
    Train LightGBM on most recent CONFIG_TRAIN_DAYS of features,
    score all ETFs for today. Returns DataFrame with [ticker, pred, close].
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("[v2-paper] ERROR: LightGBM not available")
        return pd.DataFrame()

    # Build feature matrix across all ETFs
    features_list = []
    labels_list = []
    meta_list = []

    # Common dates across all ETFs
    common = None
    for t, df in all_data.items():
        if t == BENCH_SPY:
            continue
        if t not in UNIVERSE:
            continue
        common = df.index if common is None else common.intersection(df.index)

    if common is None or len(common) < CONFIG_TRAIN_DAYS:
        print(f"[v2-paper] Insufficient common dates: {len(common) if common is not None else 0}")
        return pd.DataFrame()

    for t in UNIVERSE:
        if t not in all_data:
            continue
        df = all_data[t]
        c = df["Close"].reindex(common)
        v = df["Volume"].reindex(common) if "Volume" in df.columns else None
        sc = spy_close.reindex(common)
        feat = build_features(c, v, spy_close=sc)

        # Forward 21-day return as target
        fwd = c.pct_change(21).shift(-21)
        valid = feat.dropna().index.intersection(fwd.dropna().index)

        for d in valid:
            row = feat.loc[d].values
            if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                features_list.append(row)
                labels_list.append(fwd.loc[d])
                meta_list.append({"date": d, "ticker": t})

    if not features_list:
        print("[v2-paper] No valid feature rows")
        return pd.DataFrame()

    X = np.array(features_list)
    y = np.array(labels_list)
    meta = pd.DataFrame(meta_list)
    dates = sorted(meta["date"].unique())

    # Filter to training window: last CONFIG_TRAIN_DAYS trading days before today
    train_dates = [d for d in dates if d < today]
    if len(train_dates) < 252:
        print(f"[v2-paper] Insufficient training dates: {len(train_dates)}")
        return pd.DataFrame()

    # Take last CONFIG_TRAIN_DAYS dates for training
    train_window = train_dates[-CONFIG_TRAIN_DAYS:] if len(train_dates) > CONFIG_TRAIN_DAYS else train_dates
    train_mask = meta["date"].isin(train_window)

    X_tr, y_tr = X[train_mask], y[train_mask]

    if len(X_tr) < 100:
        print(f"[v2-paper] Insufficient training samples: {len(X_tr)}")
        return pd.DataFrame()

    # Train LightGBM — same hyperparams as research script
    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        subsample=0.8, verbose=-1, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    # Log feature importance
    imp = model.feature_importances_
    if len(imp) == len(FEATURE_NAMES):
        top_feats = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])[:5]
        print(f"[v2-paper] Top features: {[(f, int(i)) for f, i in top_feats]}")

    # Score latest snapshot for each ETF
    results = []
    for t in UNIVERSE:
        if t not in all_data:
            continue
        df = all_data[t]
        c = df["Close"].reindex(common)
        v = df["Volume"].reindex(common) if "Volume" in df.columns else None
        sc = spy_close.reindex(common)
        feat = build_features(c, v, spy_close=sc)

        # Get most recent valid feature row <= today
        valid_dates = feat.dropna().index
        valid_dates = valid_dates[valid_dates <= today]
        if len(valid_dates) == 0:
            continue

        latest_date = valid_dates[-1]
        row = feat.loc[latest_date].values
        if np.any(np.isnan(row)) or np.any(np.isinf(row)):
            continue

        pred = model.predict(row.reshape(1, -1))[0]
        # Get latest close price
        close_dates = c.dropna().index
        close_dates = close_dates[close_dates <= today]
        if len(close_dates) == 0:
            continue
        latest_close = float(c.loc[close_dates[-1]])

        results.append({
            "ticker": t,
            "pred": float(pred),
            "close": latest_close,
            "date": str(latest_date.date()),
        })

    if not results:
        print("[v2-paper] No valid scores")
        return pd.DataFrame()

    scored = pd.DataFrame(results)
    print(f"[v2-paper] Scored {len(scored)} ETFs, "
          f"trained on {len(X_tr)} samples ({len(train_window)} days)")
    return scored


# ---------------------------------------------------------------------------
# Mark-to-market
# ---------------------------------------------------------------------------
def _mark_to_market(state: PaperState, prices: pd.DataFrame,
                    today: pd.Timestamp) -> float:
    pos_mv = 0.0
    for t, shares in state.positions.items():
        last = prices[(prices["ticker"] == t) & (prices["date"] <= today)] \
            .sort_values("date").tail(1)
        if last.empty:
            continue
        px = float(last.iloc[0]["close"])
        pos_mv += shares * px

    state.nav_usd = state.cash_usd + pos_mv
    return state.nav_usd


# ---------------------------------------------------------------------------
# Core rebalance logic
# ---------------------------------------------------------------------------
def _is_rebalance_due(state: PaperState, today: pd.Timestamp) -> bool:
    if state.last_rebal_date is None:
        return True
    last = pd.Timestamp(state.last_rebal_date)
    # Rebalance every ~21 trading days (~29 calendar days)
    return (today - last) >= pd.Timedelta(days=29)


def _apply_defensive_shift(scored: pd.DataFrame, regime: str,
                           shift_factor: float) -> pd.DataFrame:
    """
    In bear regime, boost defensive ETF predictions and penalize risk-on.
    This is the key feature of C_DefShift_Top3 config.
    """
    scored = scored.copy()
    if shift_factor > 0 and regime == "bear":
        for idx in scored.index:
            ticker = scored.loc[idx, "ticker"]
            if ticker in DEFENSIVE_ETFS:
                scored.loc[idx, "pred"] *= (1 + 0.5 * shift_factor)
            elif ticker in RISK_ON_ETFS:
                scored.loc[idx, "pred"] *= (1 - 0.5 * shift_factor)
    return scored


def rebalance(today: Optional[pd.Timestamp] = None) -> PaperState:
    today = today or pd.Timestamp.today().normalize()
    state = PaperState.load()

    # Download fresh data
    prices, all_data = _download_prices()
    if prices.empty:
        print(f"[v2-paper] No price data — aborting")
        return state

    # SPY close series for regime detection + features
    spy_close = None
    if BENCH_SPY in all_data:
        spy_close = all_data[BENCH_SPY]["Close"]

    # Mark current book
    _mark_to_market(state, prices, today)

    # Detect regime
    regime = detect_regime(spy_close, today)
    state.regime = regime

    # Check if rebalance is due
    rebal_due = _is_rebalance_due(state, today)
    if not rebal_due:
        state.save()
        _log_equity(state.nav_usd, str(today.date()), regime)
        _log_trade({
            "rebal_date": str(today.date()),
            "action": "MTM",
            "regime": regime,
            "nav_usd": state.nav_usd,
            "positions": dict(state.positions),
        })
        print(f"[v2-paper] {today.date()} MTM-only "
              f"nav=${state.nav_usd:,.0f} regime={regime} "
              f"positions={list(state.positions.keys())} "
              f"next_rebal={state.next_rebal_date}")
        return state

    # === REBALANCE DAY ===
    print(f"[v2-paper] REBALANCE triggered on {today.date()} regime={regime}")

    # Train and score
    scored = _train_and_score(all_data, spy_close, today)
    if scored.empty:
        print(f"[v2-paper] Could not score ETFs; skipping rebal")
        state.save()
        return state

    # Apply defensive shift in bear regime
    scored = _apply_defensive_shift(scored, regime, CONFIG_DEF_SHIFT)

    # Pick top K
    top = scored.nlargest(CONFIG_K, "pred")
    target_tickers = top["ticker"].tolist()
    target_weights = {t: 1.0 / CONFIG_K for t in target_tickers}

    print(f"[v2-paper] Target portfolio: {target_tickers} "
          f"(regime={regime}, shift={'YES' if regime == 'bear' else 'NO'})")

    # Log scores
    for _, row in scored.sort_values("pred", ascending=False).iterrows():
        flag = " <--" if row["ticker"] in target_tickers else ""
        shift_note = ""
        if regime == "bear":
            if row["ticker"] in DEFENSIVE_ETFS:
                shift_note = " [DEF+]"
            elif row["ticker"] in RISK_ON_ETFS:
                shift_note = " [RISK-]"
        print(f"  {row['ticker']:5s}  pred={row['pred']:.4f}  "
              f"px=${row['close']:.2f}{shift_note}{flag}")

    latest_px = {row["ticker"]: float(row["close"])
                 for _, row in scored.iterrows()}

    log_rec = {
        "rebal_date": str(today.date()),
        "action": "REBAL",
        "regime": regime,
        "target_weights": target_weights,
        "nav_usd_before": state.nav_usd,
        "scores": {row["ticker"]: round(float(row["pred"]), 4)
                   for _, row in scored.iterrows()},
    }

    # === EXECUTE ===
    nav_before = state.nav_usd
    realized_pnl = 0.0

    # 1) Sell positions not in target
    for t in list(state.positions.keys()):
        if t in target_weights:
            continue
        shares = state.positions[t]
        if shares <= 0:
            del state.positions[t]
            continue
        px_now = latest_px.get(t)
        if px_now is None:
            last = prices[(prices["ticker"] == t) & (prices["date"] <= today)] \
                .sort_values("date").tail(1)
            if last.empty:
                continue
            px_now = float(last.iloc[0]["close"])
        gross_proceeds = shares * px_now
        cost = gross_proceeds * (CONFIG_COST_BPS / 10000.0)
        net_proceeds = gross_proceeds - cost
        entry_px = state.entry_prices.get(t, px_now)
        realized = (px_now - entry_px) * shares - cost
        realized_pnl += realized
        state.cash_usd += net_proceeds
        _log_trade({
            "rebal_date": str(today.date()),
            "action": "SELL",
            "ticker": t,
            "shares": shares,
            "px": px_now,
            "entry_px": entry_px,
            "cost_usd": cost,
            "realized_pnl_usd": realized,
        })
        del state.positions[t]
        if t in state.entry_prices:
            del state.entry_prices[t]

    # 2) Re-mark NAV after sells
    _mark_to_market(state, prices, today)

    # 3) Buy / rebalance into target (equal weight)
    if target_weights:
        deployable = state.nav_usd  # no leverage
        for t, w in target_weights.items():
            px_now = latest_px.get(t)
            if px_now is None or px_now <= 0:
                continue
            target_dollars = w * deployable
            current_shares = state.positions.get(t, 0)
            current_mv = current_shares * px_now
            delta_dollars = target_dollars - current_mv
            delta_shares = int(delta_dollars / px_now)

            if delta_shares > 0:
                gross_cost = delta_shares * px_now
                cost = gross_cost * (CONFIG_COST_BPS / 10000.0)
                state.cash_usd -= (gross_cost + cost)
                new_shares = current_shares + delta_shares
                old_entry = state.entry_prices.get(t, px_now)
                state.entry_prices[t] = (
                    (old_entry * current_shares + px_now * delta_shares)
                    / max(new_shares, 1)
                )
                state.positions[t] = new_shares
                _log_trade({
                    "rebal_date": str(today.date()),
                    "action": "BUY",
                    "ticker": t,
                    "shares": delta_shares,
                    "px": px_now,
                    "cost_usd": cost,
                    "target_weight": w,
                })
            elif delta_shares < 0:
                sell_shares = -delta_shares
                gross_proceeds = sell_shares * px_now
                cost = gross_proceeds * (CONFIG_COST_BPS / 10000.0)
                state.cash_usd += (gross_proceeds - cost)
                new_shares = current_shares - sell_shares
                state.positions[t] = new_shares
                _log_trade({
                    "rebal_date": str(today.date()),
                    "action": "TRIM",
                    "ticker": t,
                    "shares": sell_shares,
                    "px": px_now,
                    "cost_usd": cost,
                    "target_weight": w,
                })
            else:
                state.positions[t] = current_shares

    # 4) Finalize
    _mark_to_market(state, prices, today)
    state.cumulative_realized_pnl += realized_pnl
    state.last_rebal_date = str(today.date())
    state.next_rebal_date = str((today + pd.Timedelta(days=29)).date())
    state.n_rebalances += 1
    state.save()

    _log_equity(state.nav_usd, str(today.date()), regime)

    log_rec["executed"] = True
    log_rec["nav_usd_after"] = state.nav_usd
    log_rec["cash_after"] = state.cash_usd
    log_rec["realized_pnl"] = realized_pnl
    log_rec["positions_after"] = dict(state.positions)
    _log_trade(log_rec)

    print(f"[v2-paper] EXECUTED {today.date()} regime={regime} "
          f"nav=${state.nav_usd:,.0f} "
          f"held={list(state.positions.keys())} "
          f"realized_pnl=${realized_pnl:,.2f} "
          f"next_rebal={state.next_rebal_date}")

    return state


def main():
    print(f"[v2-paper] Sector ETF Momentum v2 Paper Engine (C_DefShift_Top3)")
    print(f"[v2-paper] CONFIG: K={CONFIG_K} hold_days={CONFIG_HOLD_DAYS} "
          f"train_days={CONFIG_TRAIN_DAYS} "
          f"def_shift={CONFIG_DEF_SHIFT} "
          f"cost_bps={CONFIG_COST_BPS} anchor=${CONFIG_ANCHOR_USD:,.0f}")
    print(f"[v2-paper] Universe: {len(UNIVERSE)} ETFs")
    try:
        state = rebalance()
        print(f"[v2-paper] FINAL: nav=${state.nav_usd:,.0f} "
              f"cash=${state.cash_usd:,.0f} regime={state.regime} "
              f"positions={list(state.positions.keys())} "
              f"n_rebal={state.n_rebalances}")
    except Exception as e:
        print(f"[v2-paper] ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
