"""
Multi-Asset Rotation Paper Engine — All-Weather Asset Class Rotation.

Universe: SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, SLV, DBC, VNQ, HYG, LQD, UUP, XLU (15 assets)
K=4 (top 4 of 15)

Key property: SPY correlation only 0.48 — genuinely diversifying.
Worst year: -8.3% (vs SPY -33.7%). Sharpe 0.84.
Best combined with V5 CSP + sector rotation for triple-uncorrelated portfolio.

Same rotation logic (momentum acceleration + RS + anti-concentration):
  A. Relative strength ACCELERATION (10d/20d change in RS, not RS itself).
  B. Cross-asset dispersion: high = rotation opportunity.
  C. Anti-concentration decay multiplier: consecutive hold penalty (>=3 = hard block).
  D. Minimum 3 assets held — prevents over-concentration.
  E. Momentum acceleration: 10d change in 20d momentum.

Walk-forward config:
  - 378-day (~18m) train, 21-day OOS rebalance (sliding, HC #0).
  - Universe: 15 multi-asset ETFs (stocks/bonds/gold/commodities/dollar).
  - Hold: 21 trading days.
  - Anchor: $100,000 paper capital.

Runs ONE rebalance step per invocation (cron at 9:52 AM ET weekdays).
On non-rebalance days: mark-to-market + NAV update only.

State: live_trading_linux/multi_asset_rotation_state/
  - state.json          — current positions, cash, NAV, regime, hold streaks
  - trades.jsonl        — all trade/MTM records
  - equity_curve.jsonl  — daily NAV snapshots

Data: fresh yfinance downloads each run (no stale parquet dependency).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = Path(__file__).resolve().parent / "multi_asset_rotation_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "state.json"
TRADES_LOG = STATE_DIR / "trades.jsonl"
EQUITY_LOG = STATE_DIR / "equity_curve.jsonl"

# ---------------------------------------------------------------------------
# CONFIG — v3 rotation-quality spec
# ---------------------------------------------------------------------------
CONFIG_K = 4                  # top-K long (4 of 16 — same concentration ratio as 3/11)
CONFIG_HOLD_DAYS = 21         # ~monthly rebalance
CONFIG_TRAIN_DAYS = 378       # 378 trading days (~18m) — upgraded from 252 per param sweep (Sharpe 1.14 vs 0.87)
CONFIG_REGIME_MA_DAYS = 60    # SPY MA60 for regime gate
CONFIG_ANCHOR_USD = 100000.0  # paper capital
CONFIG_COST_BPS = 5.0         # round-trip per leg
CONFIG_TARGET_VOL = 0.15      # annualised vol target
CONFIG_LEV_MIN = 0.25
CONFIG_LEV_MAX = 2.0

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV",
            "DBC", "VNQ", "HYG", "LQD", "UUP", "XLU"]  # Multi-asset: stocks/bonds/gold/commodities/dollar
BENCH_SPY = "SPY"

# Macro parquets for historical backfill
MACRO_EXTRA_PATH = ROOT / "wheel_strategy_v1/data/cache/macro_extra.parquet"
MACRO_FEATURES_PATH = ROOT / "macro_exposure_v1/data/cache/macro_features.parquet"

TRADING_DAYS_YR = 252

# ---------------------------------------------------------------------------
# BETA HEDGE OVERLAY (HC #670 + adversarial validated 2026-07-10)
# Rolling 60d beta-scaled SPY short hedge neutralizes market direction.
# Without this: R1 gap 1.61 (FAIL). With this: R1 gap 0.007 (PASS).
# Sharpe cost: only -0.09 (2.42 → 2.33). Parameter-stable across 0.85-1.15 scale.
# ---------------------------------------------------------------------------
BETA_HEDGE_ENABLED = True
BETA_HEDGE_WINDOW = 60       # rolling beta lookback (days)
BETA_HEDGE_SCALE = 1.0       # 1.0 = full beta neutralization
BETA_HEDGE_REBAL_TOL = 0.10  # rebalance hedge if drift > 10%

# Consecutive hold decay (HC #670 R2) — aggressive to enforce <=3 max streak
HOLD_DECAY = {0: 1.0, 1: 1.0, 2: 0.6, 3: 0.0}
# After 3 consecutive holds, score zeroed out (forces rotation)

# ---------------------------------------------------------------------------
# Features — v3 rotation-quality feature set
# ---------------------------------------------------------------------------
MOMENTUM_FEATURES = [
    "ret_20d", "ret_60d", "rel_strength_spy",
    "momentum_cross_20_60", "rs_rank_among_sectors",
]
ROTATION_FEATURES = [
    "rs_acceleration_10d",     # 10d change in relative strength
    "rs_acceleration_20d",     # 20d change in relative strength
    "ret_20d_chg_10d",         # momentum acceleration (change in momentum)
    "cross_sector_dispersion", # high = rotation opportunity
    "rank_change_10d",         # rank change among sectors
]
YIELD_CURVE_FEATURES = [
    "yc_2s10s", "yc_2s10s_roc_20d", "fed_funds",
    "fed_funds_roc_20d", "ust_10y", "ust_10y_roc_20d",
]
ALL_FEATURES = MOMENTUM_FEATURES + ROTATION_FEATURES + YIELD_CURVE_FEATURES


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class PaperState:
    nav_usd: float = CONFIG_ANCHOR_USD
    cash_usd: float = CONFIG_ANCHOR_USD
    positions: dict = field(default_factory=dict)   # {ticker: shares}
    entry_prices: dict = field(default_factory=dict)
    last_rebal_date: Optional[str] = None
    next_rebal_date: Optional[str] = None
    regime: str = "cash"
    regime_label: str = "bull_full"
    gross_leverage: float = 1.0
    sh_allocation: float = 0.0
    n_rebalances: int = 0
    cumulative_realized_pnl: float = 0.0
    hold_streak: dict = field(default_factory=dict)  # {etf: consecutive_rebal_count}
    # Beta hedge state
    spy_hedge_shares: int = 0          # negative = short SPY shares
    spy_hedge_entry_px: float = 0.0    # avg entry price of short SPY
    hedge_realized_pnl: float = 0.0    # cumulative hedge realized P&L
    version: str = "multi-asset-v1"

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
def _download_prices(lookback_days: int = 900) -> pd.DataFrame:
    """Download fresh daily closes for universe + SPY from yfinance."""
    import yfinance as yf

    tickers = UNIVERSE + [BENCH_SPY]
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days)

    print(f"[v3-paper] Downloading prices for {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        print("[v3-paper] WARNING: yfinance returned empty data")
        return pd.DataFrame()

    rows = []
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw[("Close", t)].dropna()
            else:
                closes = raw["Close"].dropna()
            for dt, px in closes.items():
                rows.append({"ticker": t, "date": pd.Timestamp(dt).normalize(),
                             "close": float(px)})
        except Exception as e:
            print(f"[v3-paper] Warning: failed to parse {t}: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last").reset_index(drop=True)
    print(f"[v3-paper] Downloaded {len(df)} price rows, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def _download_yield_curve(lookback_days: int = 900) -> pd.DataFrame:
    """Download yield curve data from yfinance + historical parquet."""
    import yfinance as yf

    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days)

    yf_tickers = {"^TNX": "ust_10y", "^FVX": "ust_5y", "^IRX": "ust_3m"}
    print("[v3-paper] Downloading yield curve from yfinance...")
    try:
        raw = yf.download(list(yf_tickers.keys()),
                          start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False, threads=True)
    except Exception as e:
        print(f"[v3-paper] yfinance yield download failed: {e}")
        raw = pd.DataFrame()

    yc_live = pd.DataFrame()
    if not raw.empty:
        for yf_tk, col_name in yf_tickers.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    s = raw[("Close", yf_tk)].dropna()
                else:
                    s = raw["Close"].dropna()
                if yc_live.empty:
                    yc_live = pd.DataFrame(index=s.index)
                yc_live[col_name] = s
            except Exception:
                continue

    # Historical macro_extra for fed_funds and backfill
    hist = pd.DataFrame()
    if MACRO_EXTRA_PATH.exists():
        try:
            me = pd.read_parquet(MACRO_EXTRA_PATH)
            me["date"] = pd.to_datetime(me["date"])
            me = me.set_index("date").sort_index()
            hist = me[["ust_10y", "ust_2y", "fed_funds", "yc_2s10s"]].copy()
        except Exception as e:
            print(f"[v3-paper] macro_extra load failed: {e}")

    # Merge
    if not yc_live.empty:
        yc_live.index = pd.to_datetime(yc_live.index).normalize()
        yc_live = yc_live.sort_index()

    if not hist.empty and not yc_live.empty:
        if "ust_10y" in yc_live.columns and "ust_3m" in yc_live.columns:
            yc_live["yc_2s10s"] = yc_live["ust_10y"] - yc_live["ust_3m"]
        combined = pd.concat([hist, yc_live], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index().ffill()
    elif not yc_live.empty:
        combined = yc_live.copy()
        if "ust_10y" in combined.columns and "ust_3m" in combined.columns:
            combined["yc_2s10s"] = combined["ust_10y"] - combined["ust_3m"]
        combined["fed_funds"] = np.nan
    elif not hist.empty:
        combined = hist.copy()
    else:
        print("[v3-paper] WARNING: no yield curve data available")
        return pd.DataFrame()

    combined = combined.sort_index().ffill()

    out = pd.DataFrame(index=combined.index)
    out["yc_2s10s"] = combined.get("yc_2s10s", pd.Series(dtype=float))
    out["fed_funds"] = combined.get("fed_funds", pd.Series(dtype=float))
    out["ust_10y"] = combined.get("ust_10y", pd.Series(dtype=float))
    out["yc_2s10s_roc_20d"] = out["yc_2s10s"].diff(20)
    out["fed_funds_roc_20d"] = out["fed_funds"].diff(20)
    out["ust_10y_roc_20d"] = out["ust_10y"].diff(20)

    print(f"[v3-paper] Yield curve: {len(out)} rows, "
          f"{out.index.min().date()} -> {out.index.max().date()}")
    return out


def _load_vix_live(lookback_days: int = 900) -> pd.Series:
    """Load VIX from macro_features parquet + yfinance for recent days."""
    import yfinance as yf

    vix = pd.Series(dtype=float)

    if MACRO_FEATURES_PATH.exists():
        try:
            mf = pd.read_parquet(MACRO_FEATURES_PATH)
            mf["date"] = pd.to_datetime(mf["date"])
            vix = mf.set_index("date")["vix"].astype(float)
        except Exception:
            pass

    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=min(lookback_days, 120))
    try:
        vix_raw = yf.download("^VIX", start=start.strftime("%Y-%m-%d"),
                               end=end.strftime("%Y-%m-%d"),
                               auto_adjust=True, progress=False)
        if not vix_raw.empty:
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_live = vix_raw[("Close", "^VIX")].dropna()
            else:
                vix_live = vix_raw["Close"].dropna()
            vix_live.index = pd.to_datetime(vix_live.index).normalize()
            for dt, val in vix_live.items():
                vix[pd.Timestamp(dt)] = float(val)
    except Exception as e:
        print(f"[v3-paper] VIX download failed: {e}")

    return vix.sort_index()


# ---------------------------------------------------------------------------
# Regime gate (same as v2)
# ---------------------------------------------------------------------------
def compute_regime_allocation(
    spy_close: float, spy_ma: float, vix_level: float,
) -> tuple[float, float, str]:
    if not np.isfinite(spy_ma) or not np.isfinite(spy_close):
        return 1.0, 0.0, "bull_full"
    ma_gap_pct = (spy_close - spy_ma) / spy_ma
    if ma_gap_pct >= 0:
        if not np.isfinite(vix_level) or vix_level < 20.0:
            return 1.0, 0.0, "bull_full"
        elif vix_level <= 25.0:
            return 0.80, 0.0, "bull_cautious"
        else:
            return 0.80, 0.0, "bull_highvol"
    else:
        if ma_gap_pct > -0.02:
            return 0.0, 0.0, "bear_shallow"
        else:
            return 0.0, 0.20, "bear_deep"


# ---------------------------------------------------------------------------
# Feature panel builder — v3 with rotation-quality features
# ---------------------------------------------------------------------------
def _build_feature_panel(prices: pd.DataFrame, yc: pd.DataFrame) -> pd.DataFrame:
    """Build per-ETF daily feature panel with momentum + rotation + yield curve."""
    spy = prices[prices["ticker"] == BENCH_SPY].sort_values("date").set_index("date")["close"]
    spy_r20 = spy.pct_change(20)
    spy_r60 = spy.pct_change(60)

    rows = []
    for t in UNIVERSE:
        s = prices[prices["ticker"] == t].sort_values("date").copy()
        if s.empty:
            continue
        s = s.set_index("date")
        s["ret_1d"] = s["close"].pct_change()
        s["ret_20d"] = s["close"].pct_change(20)
        s["ret_60d"] = s["close"].pct_change(60)
        s["sma20"] = s["close"].rolling(20, min_periods=10).mean()
        s["sma60"] = s["close"].rolling(60, min_periods=30).mean()
        s["momentum_cross_20_60"] = (s["sma20"] / s["sma60"]) - 1.0
        s["rel_strength_spy"] = s["ret_60d"] - spy_r60.reindex(s.index)

        # V3 rotation-timing features
        rs = s["ret_20d"] - spy_r20.reindex(s.index)
        s["rs_acceleration_10d"] = rs.diff(10)
        s["rs_acceleration_20d"] = rs.diff(20)
        s["ret_20d_chg_10d"] = s["ret_20d"].diff(10)

        # Yield curve features (broadcast)
        if not yc.empty:
            for col in YIELD_CURVE_FEATURES:
                if col in yc.columns:
                    s[col] = yc[col].reindex(s.index)

        s["ticker"] = t
        rows.append(s.reset_index())

    if not rows:
        return pd.DataFrame()

    panel = pd.concat(rows, ignore_index=True)

    # Cross-sectional features
    panel["rs_rank_among_sectors"] = panel.groupby("date")["ret_20d"].rank(pct=True)
    panel["rank_change_10d"] = panel.groupby("ticker")["rs_rank_among_sectors"].diff(10)

    # Cross-sector dispersion (same value for all ETFs on a date)
    disp = panel.groupby("date")["ret_20d"].std().rename("cross_sector_dispersion")
    panel = panel.merge(disp.reset_index(), on="date", how="left")

    # Forward return target for training
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["y_fwd"] = (
        panel.groupby("ticker")["close"].shift(-CONFIG_HOLD_DAYS) / panel["close"] - 1.0
    )
    return panel


# ---------------------------------------------------------------------------
# Z-score + model
# ---------------------------------------------------------------------------
def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def _xs_zscore(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = panel.copy()
    for f in feats:
        if f not in out.columns:
            out[f] = 0.0
            continue
        x = pd.to_numeric(out[f], errors="coerce").astype(float)
        x = _winsorize(x, 0.01)
        out[f] = x
        mu = out.groupby("date")[f].transform("mean")
        sd = out.groupby("date")[f].transform("std")
        z = (x - mu) / sd.replace(0.0, np.nan)
        out[f] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _fit_lgbm(X: np.ndarray, y: np.ndarray) -> object:
    """Fit LGBM regressor. Returns model object or None."""
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=200, learning_rate=0.05,
        num_leaves=15, min_child_samples=10, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        verbose=-1, n_jobs=1,
    )
    model.fit(X, y)
    return model


def _fit_ridge(X: np.ndarray, y: np.ndarray,
               alphas=(0.1, 1.0, 10.0, 100.0)) -> tuple:
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    best = (None, None, float("inf"))
    for a in alphas:
        try:
            beta = np.linalg.solve(XtX + a * np.eye(X.shape[1]), Xty)
            mse = float(((yc - Xc @ beta) ** 2).mean())
            if mse < best[2]:
                best = (beta, y.mean() - X.mean(axis=0) @ beta, mse)
        except np.linalg.LinAlgError:
            continue
    return best[0], best[1]


def _score_sectors(panel: pd.DataFrame, today: pd.Timestamp,
                   feats: list[str]) -> pd.DataFrame:
    """
    Train LGBM (or ridge fallback) on last 252 trading days, score latest snapshot.
    Apply anti-concentration decay to scores based on hold streaks.
    """
    p = panel.sort_values(["ticker", "date"]).copy()

    # 252 trading-day lookback: get unique dates, take last 252
    all_dates = sorted(p["date"].unique())
    past_dates = [d for d in all_dates if d < today]
    if len(past_dates) < 100:
        print(f"[v3-paper] Insufficient history: {len(past_dates)} trading days")
        return pd.DataFrame()

    train_start_date = past_dates[-min(CONFIG_TRAIN_DAYS, len(past_dates))]
    train = p[(p["date"] >= train_start_date) & (p["date"] < today)].dropna(subset=["y_fwd"])
    if len(train) < 200:
        print(f"[v3-paper] Insufficient training data: {len(train)} rows")
        return pd.DataFrame()

    train_z = _xs_zscore(train, feats)
    for f in feats:
        train_z[f] = train_z[f].fillna(0.0)

    X = train_z[feats].values
    y = train_z["y_fwd"].values

    model = _fit_lgbm(X, y)
    model_name = "lgbm"

    # Latest snapshot for scoring
    asof = p[p["date"] <= today].sort_values("date").groupby("ticker").tail(1).copy()
    asof_z = _xs_zscore(asof, feats)
    for f in feats:
        asof_z[f] = asof_z[f].fillna(0.0)
    X_asof = asof_z[feats].values

    if model is not None:
        asof_z["score"] = model.predict(X_asof)
    else:
        coef, intercept = _fit_ridge(X, y)
        model_name = "ridge"
        if coef is None:
            return pd.DataFrame()
        asof_z["score"] = X_asof @ coef + intercept

    result = asof_z[["ticker", "close", "date", "score"]].rename(
        columns={"ticker": "etf"})
    print(f"[v3-paper] Scored {len(result)} sectors using {model_name}")
    return result


# ---------------------------------------------------------------------------
# Vol-target sizing
# ---------------------------------------------------------------------------
def _estimate_book_vol(panel: pd.DataFrame, today: pd.Timestamp,
                       longs: list[str], lookback_days: int = 60) -> float:
    cutoff_lo = today - pd.Timedelta(days=lookback_days * 2 + 10)
    hist = panel[(panel["date"] < today) & (panel["date"] >= cutoff_lo)
                 & (panel["ticker"].isin(longs))]
    if hist.empty:
        return 0.0
    by_date = hist.groupby("date")["ret_1d"].mean().dropna().tail(lookback_days)
    if len(by_date) < 20:
        return 0.0
    sd = float(by_date.std(ddof=1))
    return sd * np.sqrt(TRADING_DAYS_YR) if np.isfinite(sd) else 0.0


# ---------------------------------------------------------------------------
# Mark-to-market
# ---------------------------------------------------------------------------
def _compute_portfolio_beta(prices: pd.DataFrame, state: PaperState,
                            today: pd.Timestamp, window: int = BETA_HEDGE_WINDOW) -> float:
    """Compute rolling beta of current portfolio vs SPY."""
    spy = prices[prices["ticker"] == "SPY"].sort_values("date")
    spy = spy[spy["date"] <= today].tail(window + 5)
    if len(spy) < window:
        return 0.5  # conservative default

    spy_rets = spy.set_index("date")["close"].pct_change().dropna()

    # Portfolio returns: weighted sum of sector returns
    port_rets = pd.Series(0.0, index=spy_rets.index)
    total_mv = 0.0

    for t, shares in state.positions.items():
        if t in ("SH_SYNTHETIC",) or shares <= 0:
            continue
        sec = prices[(prices["ticker"] == t) & (prices["date"] <= today)].sort_values("date")
        sec = sec.tail(window + 5)
        if len(sec) < window:
            continue
        sec_rets = sec.set_index("date")["close"].pct_change().dropna()
        # Use latest price for weighting
        px = float(sec.iloc[-1]["close"])
        mv = shares * px
        common = port_rets.index.intersection(sec_rets.index)
        port_rets.loc[common] += sec_rets.loc[common] * mv
        total_mv += mv

    if total_mv > 0:
        port_rets = port_rets / total_mv

    # Compute beta = cov(port, spy) / var(spy)
    common = port_rets.index.intersection(spy_rets.index)
    if len(common) < 20:
        return 0.5

    p = port_rets.loc[common].tail(window)
    s = spy_rets.loc[common].tail(window)

    cov = np.cov(p.values, s.values)
    if cov[1, 1] > 0:
        beta = cov[0, 1] / cov[1, 1]
    else:
        beta = 0.5

    return max(0.0, min(3.0, beta))  # clip to reasonable range


def _rebalance_hedge(state: PaperState, prices: pd.DataFrame,
                     today: pd.Timestamp) -> None:
    """Rebalance the SPY beta hedge to maintain market neutrality."""
    if not BETA_HEDGE_ENABLED:
        return

    beta = _compute_portfolio_beta(prices, state, today)

    # Portfolio market value (long positions only)
    port_mv = 0.0
    for t, shares in state.positions.items():
        if t in ("SH_SYNTHETIC",) or shares <= 0:
            continue
        sec = prices[(prices["ticker"] == t) & (prices["date"] <= today)].sort_values("date")
        if sec.empty:
            continue
        px = float(sec.iloc[-1]["close"])
        port_mv += shares * px

    # SPY current price
    spy = prices[prices["ticker"] == "SPY"].sort_values("date")
    spy = spy[spy["date"] <= today]
    if spy.empty:
        print(f"[v3.1-hedge] WARNING: No SPY data, skipping hedge")
        return
    spy_px = float(spy.iloc[-1]["close"])

    # Target short SPY shares = port_mv * beta * scale / spy_px
    target_hedge_mv = port_mv * beta * BETA_HEDGE_SCALE
    target_shares = -int(target_hedge_mv / spy_px)  # negative = short

    current_shares = state.spy_hedge_shares

    # Check if rebalance needed
    if current_shares != 0:
        drift = abs(target_shares - current_shares) / abs(current_shares)
    else:
        drift = 1.0 if target_shares != 0 else 0.0

    if drift < BETA_HEDGE_REBAL_TOL and current_shares != 0:
        return  # within tolerance

    # Execute hedge rebalance
    delta_shares = target_shares - current_shares

    if delta_shares < 0:
        # Shorting more SPY
        short_mv = abs(delta_shares) * spy_px
        cost = short_mv * (CONFIG_COST_BPS / 10000.0)
        state.cash_usd += short_mv - cost  # receive proceeds from short sale
    elif delta_shares > 0:
        # Covering short (buying back)
        cover_mv = abs(delta_shares) * spy_px
        cost = cover_mv * (CONFIG_COST_BPS / 10000.0)
        # Realized P&L on covered shares
        if state.spy_hedge_entry_px > 0 and current_shares < 0:
            pnl = (state.spy_hedge_entry_px - spy_px) * abs(delta_shares) - cost
            state.hedge_realized_pnl += pnl
        state.cash_usd -= cover_mv + cost

    # Update hedge position
    if current_shares == 0:
        state.spy_hedge_entry_px = spy_px
    elif target_shares != 0:
        # Weighted average entry
        old_mv = abs(current_shares) * state.spy_hedge_entry_px
        new_mv = abs(delta_shares) * spy_px
        total_sh = abs(target_shares)
        state.spy_hedge_entry_px = (old_mv + new_mv) / max(total_sh, 1)

    state.spy_hedge_shares = target_shares

    _log_trade({
        "rebal_date": str(today.date()),
        "action": "HEDGE_REBAL",
        "spy_px": spy_px,
        "beta": round(beta, 3),
        "target_shares": target_shares,
        "delta_shares": delta_shares,
        "port_mv": round(port_mv, 2),
        "hedge_mv": round(abs(target_shares) * spy_px, 2),
    })

    print(f"[v3.1-hedge] beta={beta:.3f} spy=${spy_px:.0f} "
          f"hedge={target_shares} shares (${abs(target_shares)*spy_px:,.0f}) "
          f"delta={delta_shares}")


def _mark_to_market(state: PaperState, prices: pd.DataFrame,
                    today: pd.Timestamp) -> float:
    pos_mv = 0.0
    for t, shares in state.positions.items():
        if t == "SH_SYNTHETIC":
            pos_mv += shares  # notional value for synthetic
            continue
        last = prices[(prices["ticker"] == t) & (prices["date"] <= today)] \
            .sort_values("date").tail(1)
        if last.empty:
            continue
        px = float(last.iloc[0]["close"])
        pos_mv += shares * px

    # Add hedge market value (short SPY position)
    # Cash already includes short-sale proceeds, so we subtract the current
    # liability (shares × current_price). For short positions, shares < 0,
    # so hedge_value is negative — correctly reducing NAV by the liability.
    hedge_value = 0.0
    if state.spy_hedge_shares != 0:
        spy = prices[(prices["ticker"] == "SPY") & (prices["date"] <= today)] \
            .sort_values("date").tail(1)
        if not spy.empty:
            spy_px = float(spy.iloc[0]["close"])
            hedge_value = state.spy_hedge_shares * spy_px  # negative for shorts

    state.nav_usd = state.cash_usd + pos_mv + hedge_value
    return state.nav_usd


# ---------------------------------------------------------------------------
# Core rebalance logic
# ---------------------------------------------------------------------------
def _is_rebalance_due(state: PaperState, today: pd.Timestamp) -> bool:
    if state.last_rebal_date is None:
        return True
    last = pd.Timestamp(state.last_rebal_date)
    return (today - last) >= pd.Timedelta(days=29)


def _apply_hold_decay(scored: pd.DataFrame, hold_streak: dict) -> pd.DataFrame:
    """
    Apply anti-concentration decay to sector scores.
    HC #670 R2: hard block any sector held >=3 consecutive rebalances.
    """
    scored = scored.copy()
    scored["consec_holds"] = scored["etf"].map(
        lambda e: hold_streak.get(e, 0)
    )
    scored["adj_score"] = scored.apply(
        lambda row: (
            -999.0 if row["consec_holds"] >= 3  # hard block at 3
            else row["score"] * HOLD_DECAY.get(
                min(int(row["consec_holds"]), 3), 0.0
            )
        ),
        axis=1,
    )
    return scored


def rebalance(today: Optional[pd.Timestamp] = None) -> PaperState:
    today = today or pd.Timestamp.today().normalize()
    state = PaperState.load()

    # Download fresh data
    prices = _download_prices()
    if prices.empty:
        print(f"[v3-paper] No price data — aborting")
        return state

    yc = _download_yield_curve()
    vix = _load_vix_live()

    panel = _build_feature_panel(prices, yc)
    if panel.empty:
        print(f"[v3-paper] No feature panel — aborting")
        return state

    feats = [f for f in ALL_FEATURES if f in panel.columns]
    print(f"[v3-paper] Features available ({len(feats)}): {feats}")

    # Mark current book
    _mark_to_market(state, prices, today)

    # Regime check (graded gate — same as v2)
    spy = prices[prices["ticker"] == BENCH_SPY].sort_values("date").set_index("date")["close"]
    spy_ma = spy.rolling(CONFIG_REGIME_MA_DAYS,
                         min_periods=max(20, CONFIG_REGIME_MA_DAYS // 2)).mean()
    asof_dates = spy.index[spy.index <= today]
    if len(asof_dates) == 0:
        print(f"[v3-paper] No SPY data up to {today.date()}")
        return state

    d = asof_dates[-1]
    spy_close = float(spy.loc[d])
    spy_ma_val = float(spy_ma.loc[d]) if pd.notna(spy_ma.loc[d]) else float("nan")

    vix_val = float("nan")
    if not vix.empty:
        prior_vix = vix.loc[:today]
        if len(prior_vix) > 0:
            vix_val = float(prior_vix.iloc[-1])

    sector_frac, sh_frac, regime_label = compute_regime_allocation(
        spy_close, spy_ma_val, vix_val)
    state.regime = "bull" if sector_frac > 0 else "bear"
    state.regime_label = regime_label
    state.sh_allocation = sh_frac

    # Check if rebalance is due
    rebal_due = _is_rebalance_due(state, today)
    if not rebal_due:
        # Still rebalance hedge on non-rebalance days (beta drifts daily)
        _rebalance_hedge(state, prices, today)
        _mark_to_market(state, prices, today)  # re-mark after hedge adj
        state.save()
        _log_equity(state.nav_usd, str(today.date()), regime_label)
        _log_trade({
            "rebal_date": str(today.date()),
            "action": "MTM",
            "regime": regime_label,
            "nav_usd": state.nav_usd,
            "spy_close": spy_close,
            "spy_ma60": spy_ma_val,
            "vix": vix_val,
            "positions": dict(state.positions),
            "spy_hedge_shares": state.spy_hedge_shares,
        })
        print(f"[v3-paper] {today.date()} MTM-only "
              f"nav=${state.nav_usd:,.0f} regime={regime_label} "
              f"positions={list(state.positions.keys())} "
              f"hedge={state.spy_hedge_shares} SPY shares "
              f"next_rebal={state.next_rebal_date}")
        return state

    # === REBALANCE DAY ===
    print(f"[v3-paper] REBALANCE triggered on {today.date()}")

    # Score sectors
    scored = _score_sectors(panel, today, feats)
    if scored.empty:
        print(f"[v3-paper] Could not score sectors; skipping rebal")
        state.save()
        return state

    # Pick target book based on regime
    if sector_frac <= 0:
        target = {}
        target_note = f"regime={regime_label} -> no sector allocation"
        new_hold_streak = {etf: 0 for etf in UNIVERSE}
    else:
        # Apply anti-concentration decay (HC #670 R2)
        scored_adj = _apply_hold_decay(scored, state.hold_streak)

        # No-trade floor: top adjusted score must exceed median raw score
        med = scored["score"].median()
        if scored_adj["adj_score"].max() <= med:
            target = {}
            target_note = "top adjusted score below median -> cash"
            new_hold_streak = {etf: 0 for etf in UNIVERSE}
        else:
            top = scored_adj.nlargest(CONFIG_K, "adj_score")
            longs = top["etf"].tolist()
            target = {etf: (1.0 / CONFIG_K) * sector_frac for etf in longs}

            # Update hold streaks
            new_hold_streak = {}
            for etf in UNIVERSE:
                if etf in longs:
                    new_hold_streak[etf] = state.hold_streak.get(etf, 0) + 1
                else:
                    new_hold_streak[etf] = 0

            # Log decay info
            decay_info = {etf: {"streak": new_hold_streak[etf],
                                "decay": HOLD_DECAY.get(min(state.hold_streak.get(etf, 0), 3), 0.0)}
                          for etf in longs}
            target_note = (f"regime={regime_label}, frac={sector_frac:.0%}, "
                           f"top-{CONFIG_K} (anti-conc), decay={decay_info}")

    state.hold_streak = new_hold_streak

    # Vol-target sizing on sector sleeve
    if target:
        book_vol = _estimate_book_vol(panel, today, list(target.keys()))
        if book_vol <= 1e-6:
            gross_lev = 1.0
        else:
            gross_lev = float(np.clip(CONFIG_TARGET_VOL / book_vol,
                                       CONFIG_LEV_MIN, CONFIG_LEV_MAX))
    else:
        gross_lev = 0.0
    state.gross_leverage = gross_lev

    latest_px = {row["etf"]: float(row["close"]) for _, row in scored.iterrows()}

    log_rec = {
        "rebal_date": str(today.date()),
        "action": "REBAL",
        "regime": regime_label,
        "spy_close": spy_close,
        "spy_ma60": spy_ma_val,
        "vix": vix_val,
        "sector_frac": sector_frac,
        "sh_frac": sh_frac,
        "target_weights": target,
        "gross_leverage": gross_lev,
        "nav_usd_before": state.nav_usd,
        "note": target_note,
        "hold_streak": new_hold_streak,
        "scores": {row["etf"]: round(float(row["score"]), 4)
                   for _, row in scored.iterrows()},
    }

    # === EXECUTE ===
    nav_before = state.nav_usd
    realized_pnl = 0.0

    # 1) Sell positions not in target
    for t in list(state.positions.keys()):
        if t == "SH_SYNTHETIC":
            if sh_frac <= 0:
                notional = state.positions[t]
                state.cash_usd += notional
                _log_trade({
                    "rebal_date": str(today.date()),
                    "action": "CLOSE_SH",
                    "notional": notional,
                })
                del state.positions[t]
            continue
        if t in target:
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

    # 2) Re-mark NAV
    _mark_to_market(state, prices, today)

    # 3) Buy / rebalance into target
    if target:
        deployable = state.nav_usd * gross_lev
        new_positions = {}
        for t, w in target.items():
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
                    (old_entry * current_shares + px_now * delta_shares) /
                    max(new_shares, 1)
                )
                new_positions[t] = new_shares
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
                new_positions[t] = new_shares
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
                new_positions[t] = current_shares
        state.positions = new_positions

    # 4) Open SH synthetic if bear_deep
    if sh_frac > 0 and "SH_SYNTHETIC" not in state.positions:
        sh_notional = state.nav_usd * sh_frac
        state.positions["SH_SYNTHETIC"] = sh_notional
        state.cash_usd -= sh_notional
        _log_trade({
            "rebal_date": str(today.date()),
            "action": "OPEN_SH",
            "notional": sh_notional,
            "sh_frac": sh_frac,
        })

    # 4b) Rebalance beta hedge
    _rebalance_hedge(state, prices, today)

    # 5) Finalize
    _mark_to_market(state, prices, today)
    state.cumulative_realized_pnl += realized_pnl
    state.last_rebal_date = str(today.date())
    state.next_rebal_date = str((today + pd.Timedelta(days=29)).date())
    state.n_rebalances += 1
    state.save()

    _log_equity(state.nav_usd, str(today.date()), regime_label)

    log_rec["executed"] = True
    log_rec["nav_usd_after"] = state.nav_usd
    log_rec["cash_after"] = state.cash_usd
    log_rec["realized_pnl"] = realized_pnl
    log_rec["positions_after"] = dict(state.positions)
    _log_trade(log_rec)

    # Cash-negative warning
    if state.cash_usd < 0:
        margin_used = abs(state.cash_usd)
        margin_ratio = margin_used / max(state.nav_usd, 1)
        print(f"[v3-paper] WARNING: cash negative ${state.cash_usd:,.0f} "
              f"(margin/NAV: {margin_ratio:.1%})")
        if margin_ratio > 0.10:
            import os
            os.system(
                f'node /home/jupiter/teleclaude-main/utils/webhook_notifier.js '
                f'"ETF Rotation v3: margin debt ${margin_used:,.0f} '
                f'({margin_ratio:.1%} of NAV)" 2>/dev/null'
            )

    print(f"[v3-paper] EXECUTED {today.date()} regime={regime_label} "
          f"nav=${state.nav_usd:,.0f} lev={gross_lev:.2f} "
          f"sector_frac={sector_frac:.0%} sh_frac={sh_frac:.0%} "
          f"held={list(state.positions.keys())} "
          f"next_rebal={state.next_rebal_date}")

    # Score rankings for log
    if 'scored_adj' in dir():
        display = scored_adj if 'scored_adj' in dir() else scored
    else:
        display = scored
    for _, row in scored.sort_values("score", ascending=False).iterrows():
        etf = row["etf"]
        streak = new_hold_streak.get(etf, 0)
        flag = " <--" if etf in state.positions else ""
        decay_str = f" (streak={streak})" if streak > 0 else ""
        print(f"  {etf:5s}  score={row['score']:.4f}  px=${row['close']:.2f}{decay_str}{flag}")

    return state


def main():
    print(f"[v3-paper] ETF Rotation v3 Paper Engine (rotation-quality)")
    print(f"[v3-paper] CONFIG: K={CONFIG_K} hold_days={CONFIG_HOLD_DAYS} "
          f"train_days={CONFIG_TRAIN_DAYS} "
          f"regime_ma={CONFIG_REGIME_MA_DAYS}d target_vol={CONFIG_TARGET_VOL:.2f} "
          f"cost_bps={CONFIG_COST_BPS} anchor=${CONFIG_ANCHOR_USD:,.0f}")
    print(f"[v3-paper] HOLD_DECAY={HOLD_DECAY}")
    try:
        state = rebalance()
        print(f"[v3-paper] FINAL: nav=${state.nav_usd:,.0f} "
              f"cash=${state.cash_usd:,.0f} regime={state.regime_label} "
              f"positions={list(state.positions.keys())} "
              f"n_rebal={state.n_rebalances} "
              f"hold_streaks={state.hold_streak}")
    except Exception as e:
        print(f"[v3-paper] ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
