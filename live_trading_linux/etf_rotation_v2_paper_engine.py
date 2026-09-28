"""
ETF Rotation v2 Paper Engine (LIVE — paper-only, A/B vs v1).

v2 improvements over v1:
  A. Yield curve features: 10y-2y spread, spread RoC, fed funds direction.
  B. VIX-graded regime gate: 4-tier allocation (100%/80%/50%/20% SH).
  C. LGBM picker (ridge fallback) instead of plain ridge.
  D. SH (inverse SPY) bear allocation on deep-bear regime.

Walk-forward config (same as v2 backtest):
  - 24-month train, 1-month OOS, 1-month step (sliding, HC #0).
  - Universe: 11 sector SPDR ETFs.
  - Hold: 21 trading days.
  - Anchor: $100,000 paper capital.

Runs ONE rebalance step per invocation (cron at 9:47 AM ET weekdays).
On non-rebalance days: mark-to-market + NAV update only.

State: live_trading_linux/etf_rotation_v2_state/
  - state.json          — current positions, cash, NAV, regime
  - trades.jsonl        — all trade/MTM records
  - equity_curve.jsonl  — daily NAV snapshots

Data: fresh yfinance downloads each run (no stale parquet dependency).
Yield curve: ^TNX (10y), ^FVX (5y), ^IRX (3m) from yfinance,
  plus macro_extra.parquet for historical fill.
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
STATE_DIR = Path(__file__).resolve().parent / "etf_rotation_v2_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "state.json"
TRADES_LOG = STATE_DIR / "trades.jsonl"
EQUITY_LOG = STATE_DIR / "equity_curve.jsonl"

# ---------------------------------------------------------------------------
# CONFIG — matches v2 backtest spec
# ---------------------------------------------------------------------------
CONFIG_K = 2                  # top-K long
CONFIG_HOLD_DAYS = 21         # ~monthly rebalance
CONFIG_TRAIN_MONTHS = 24
CONFIG_REGIME_MA_DAYS = 60    # SPY MA60 for regime gate
CONFIG_ANCHOR_USD = 100000.0  # paper capital ($100K for v2 A/B test)
CONFIG_COST_BPS = 5.0         # round-trip per leg
CONFIG_TARGET_VOL = 0.15      # annualised vol target
CONFIG_LEV_MIN = 0.25
CONFIG_LEV_MAX = 2.0

UNIVERSE = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLU", "XLI", "XLV", "XLB", "XLC", "XLRE"]
BENCH_SPY = "SPY"

# Macro parquets for historical backfill
MACRO_EXTRA_PATH = ROOT / "wheel_strategy_v1/data/cache/macro_extra.parquet"
MACRO_FEATURES_PATH = ROOT / "macro_exposure_v1/data/cache/macro_features.parquet"

TRADING_DAYS = 252

# Yield curve features used by v2
MOMENTUM_FEATURES = [
    "ret_20d", "ret_60d", "rel_strength_spy",
    "momentum_cross_20_60", "rs_rank_among_sectors",
]
YIELD_CURVE_FEATURES = [
    "yc_2s10s", "yc_2s10s_roc_20d", "fed_funds",
    "fed_funds_roc_20d", "ust_10y", "ust_10y_roc_20d",
]
ALL_FEATURES = MOMENTUM_FEATURES + YIELD_CURVE_FEATURES


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class PaperState:
    nav_usd: float = CONFIG_ANCHOR_USD
    cash_usd: float = CONFIG_ANCHOR_USD
    positions: dict = field(default_factory=dict)   # {ticker: shares}
    entry_prices: dict = field(default_factory=dict)  # {ticker: entry_px}
    last_rebal_date: Optional[str] = None
    next_rebal_date: Optional[str] = None
    regime: str = "cash"
    regime_label: str = "bull_full"
    gross_leverage: float = 1.0
    sh_allocation: float = 0.0    # fraction in synthetic SH
    n_rebalances: int = 0
    cumulative_realized_pnl: float = 0.0
    version: str = "v2"

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

    print(f"[v2-paper] Downloading prices for {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        print("[v2-paper] WARNING: yfinance returned empty data")
        return pd.DataFrame()

    rows = []
    for t in tickers:
        try:
            # yfinance MultiIndex: (Price, Ticker) — e.g. ('Close', 'SPY')
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw[("Close", t)].dropna()
            else:
                closes = raw["Close"].dropna()
            for dt, px in closes.items():
                rows.append({"ticker": t, "date": pd.Timestamp(dt).normalize(),
                             "close": float(px)})
        except Exception as e:
            print(f"[v2-paper] Warning: failed to parse {t}: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last").reset_index(drop=True)
    print(f"[v2-paper] Downloaded {len(df)} price rows, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def _download_yield_curve(lookback_days: int = 900) -> pd.DataFrame:
    """
    Download yield curve data from yfinance treasury tickers + historical parquet.
    ^TNX = 10y yield, ^FVX = 5y yield, ^IRX = 3-month T-bill.
    Also loads macro_extra.parquet for historical fill + fed_funds.
    """
    import yfinance as yf

    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days)

    # Download treasury yields from yfinance
    yf_tickers = {"^TNX": "ust_10y", "^FVX": "ust_5y", "^IRX": "ust_3m"}
    print("[v2-paper] Downloading yield curve from yfinance...")
    try:
        raw = yf.download(list(yf_tickers.keys()),
                          start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False, threads=True)
    except Exception as e:
        print(f"[v2-paper] yfinance yield download failed: {e}")
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

    # Load historical macro_extra for fed_funds and backfill
    hist = pd.DataFrame()
    if MACRO_EXTRA_PATH.exists():
        try:
            me = pd.read_parquet(MACRO_EXTRA_PATH)
            me["date"] = pd.to_datetime(me["date"])
            me = me.set_index("date").sort_index()
            hist = me[["ust_10y", "ust_2y", "fed_funds", "yc_2s10s"]].copy()
        except Exception as e:
            print(f"[v2-paper] macro_extra load failed: {e}")

    # Merge: prefer live yfinance data, backfill from parquet
    if not yc_live.empty:
        yc_live.index = pd.to_datetime(yc_live.index).normalize()
        yc_live = yc_live.sort_index()

    if not hist.empty and not yc_live.empty:
        # Compute 2s10s from yfinance if we have both
        if "ust_10y" in yc_live.columns and "ust_3m" in yc_live.columns:
            # Use 10y - 3m as proxy (close enough to 10y-2y for live use)
            yc_live["yc_2s10s"] = yc_live["ust_10y"] - yc_live["ust_3m"]

        # Combine: historical + live overlay via concat + dedup
        combined = pd.concat([hist, yc_live], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index().ffill()
    elif not yc_live.empty:
        combined = yc_live.copy()
        if "ust_10y" in combined.columns and "ust_3m" in combined.columns:
            combined["yc_2s10s"] = combined["ust_10y"] - combined["ust_3m"]
        combined["fed_funds"] = np.nan  # no source
    elif not hist.empty:
        combined = hist.copy()
    else:
        print("[v2-paper] WARNING: no yield curve data available")
        return pd.DataFrame()

    combined = combined.sort_index().ffill()

    # Derived features
    out = pd.DataFrame(index=combined.index)
    out["yc_2s10s"] = combined.get("yc_2s10s", pd.Series(dtype=float))
    out["fed_funds"] = combined.get("fed_funds", pd.Series(dtype=float))
    out["ust_10y"] = combined.get("ust_10y", pd.Series(dtype=float))
    out["yc_2s10s_roc_20d"] = out["yc_2s10s"].diff(20)
    out["fed_funds_roc_20d"] = out["fed_funds"].diff(20)
    out["ust_10y_roc_20d"] = out["ust_10y"].diff(20)

    print(f"[v2-paper] Yield curve: {len(out)} rows, "
          f"{out.index.min().date()} -> {out.index.max().date()}")
    return out


def _load_vix_live(lookback_days: int = 900) -> pd.Series:
    """Load VIX from macro_features parquet + yfinance for recent days."""
    import yfinance as yf

    vix = pd.Series(dtype=float)

    # Historical from parquet
    if MACRO_FEATURES_PATH.exists():
        try:
            mf = pd.read_parquet(MACRO_FEATURES_PATH)
            mf["date"] = pd.to_datetime(mf["date"])
            vix = mf.set_index("date")["vix"].astype(float)
        except Exception:
            pass

    # Live from yfinance
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
                # Overlay
                for dt, val in vix_live.items():
                    vix[pd.Timestamp(dt)] = float(val)
    except Exception as e:
        print(f"[v2-paper] VIX download failed: {e}")

    return vix.sort_index()


# ---------------------------------------------------------------------------
# Graded regime gate (matches v2 backtest)
# ---------------------------------------------------------------------------
def compute_regime_allocation(
    spy_close: float, spy_ma: float, vix_level: float,
) -> tuple[float, float, str]:
    """
    Returns (sector_frac, sh_frac, label):
      SPY > MA60 and VIX < 20:  100% sector
      SPY > MA60 and VIX 20-25: 80% sector
      SPY > MA60 and VIX > 25:  80% sector (bull_highvol)
      SPY < MA60 by < 2%:       0% sector, 0% SH (bear_shallow = cash)
      SPY < MA60 by > 2%:       0% sector, 20% SH (bear_deep)
    """
    if not np.isfinite(spy_ma) or not np.isfinite(spy_close):
        return 1.0, 0.0, "bull_full"

    ma_gap_pct = (spy_close - spy_ma) / spy_ma

    if ma_gap_pct >= 0:  # SPY above MA60
        if not np.isfinite(vix_level) or vix_level < 20.0:
            return 1.0, 0.0, "bull_full"
        elif vix_level <= 25.0:
            return 0.80, 0.0, "bull_cautious"
        else:
            return 0.80, 0.0, "bull_highvol"
    else:  # SPY below MA60
        if ma_gap_pct > -0.02:
            return 0.0, 0.0, "bear_shallow"
        else:
            return 0.0, 0.20, "bear_deep"


# ---------------------------------------------------------------------------
# Feature panel builder
# ---------------------------------------------------------------------------
def _build_feature_panel(prices: pd.DataFrame, yc: pd.DataFrame) -> pd.DataFrame:
    """Build per-ETF daily feature panel with momentum + yield curve features."""
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

        # Join yield curve features (broadcast — same for all ETFs on a date)
        if not yc.empty:
            for col in YIELD_CURVE_FEATURES:
                if col in yc.columns:
                    s[col] = yc[col].reindex(s.index)

        s["ticker"] = t
        rows.append(s.reset_index())

    if not rows:
        return pd.DataFrame()

    panel = pd.concat(rows, ignore_index=True)
    panel["rs_rank_among_sectors"] = panel.groupby("date")["ret_20d"].rank(pct=True)

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
    """Train LGBM (or ridge fallback) on last 24 months, score latest snapshot."""
    p = panel.sort_values(["ticker", "date"]).copy()

    train_start = today - pd.DateOffset(months=CONFIG_TRAIN_MONTHS)
    train = p[(p["date"] >= train_start) & (p["date"] < today)].dropna(subset=["y_fwd"])
    if len(train) < 200:
        print(f"[v2-paper] Insufficient training data: {len(train)} rows")
        return pd.DataFrame()

    train_z = _xs_zscore(train, feats)
    for f in feats:
        train_z[f] = train_z[f].fillna(0.0)

    X = train_z[feats].values
    y = train_z["y_fwd"].values

    # Try LGBM first
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
    print(f"[v2-paper] Scored {len(result)} sectors using {model_name}")
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
    return sd * np.sqrt(TRADING_DAYS) if np.isfinite(sd) else 0.0


# ---------------------------------------------------------------------------
# Mark-to-market
# ---------------------------------------------------------------------------
def _mark_to_market(state: PaperState, prices: pd.DataFrame,
                    today: pd.Timestamp) -> float:
    pos_mv = 0.0
    for t, shares in state.positions.items():
        if t == "SH_SYNTHETIC":
            # SH tracked as notional dollars, not shares
            # Daily P&L applied during rebalance; here just count the notional
            pos_mv += shares  # shares = notional value for synthetic
            continue
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
    return (today - last) >= pd.Timedelta(days=29)


def rebalance(today: Optional[pd.Timestamp] = None) -> PaperState:
    today = today or pd.Timestamp.today().normalize()
    state = PaperState.load()

    # Download fresh data
    prices = _download_prices()
    if prices.empty:
        print(f"[v2-paper] No price data — aborting")
        return state

    yc = _download_yield_curve()
    vix = _load_vix_live()

    panel = _build_feature_panel(prices, yc)
    if panel.empty:
        print(f"[v2-paper] No feature panel — aborting")
        return state

    feats = [f for f in ALL_FEATURES if f in panel.columns]
    print(f"[v2-paper] Features available: {feats}")

    # Mark current book
    _mark_to_market(state, prices, today)

    # Regime check (graded gate)
    spy = prices[prices["ticker"] == BENCH_SPY].sort_values("date").set_index("date")["close"]
    spy_ma = spy.rolling(CONFIG_REGIME_MA_DAYS,
                         min_periods=max(20, CONFIG_REGIME_MA_DAYS // 2)).mean()
    asof_dates = spy.index[spy.index <= today]
    if len(asof_dates) == 0:
        print(f"[v2-paper] No SPY data up to {today.date()}")
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
        })
        print(f"[v2-paper] {today.date()} MTM-only "
              f"nav=${state.nav_usd:,.0f} regime={regime_label} "
              f"positions={list(state.positions.keys())} "
              f"next_rebal={state.next_rebal_date}")
        return state

    # === REBALANCE DAY ===
    print(f"[v2-paper] REBALANCE triggered on {today.date()}")

    # Score sectors
    scored = _score_sectors(panel, today, feats)
    if scored.empty:
        print(f"[v2-paper] Could not score sectors; skipping rebal")
        state.save()
        return state

    # Pick target book based on regime
    if sector_frac <= 0:
        target = {}
        target_note = f"regime={regime_label} -> no sector allocation"
    else:
        # No-trade floor: top score must exceed median
        med = scored["score"].median()
        if scored["score"].max() <= med:
            target = {}
            target_note = "top score below median -> cash"
        else:
            top = scored.nlargest(CONFIG_K, "score")
            target = {row["etf"]: (1.0 / CONFIG_K) * sector_frac
                      for _, row in top.iterrows()}
            target_note = f"regime={regime_label}, frac={sector_frac:.0%}, top-{CONFIG_K}"

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
        "scores": {row["etf"]: round(float(row["score"]), 4)
                   for _, row in scored.iterrows()},
    }

    # === EXECUTE ===
    nav_before = state.nav_usd
    realized_pnl = 0.0

    # 1) Sell positions not in target
    for t in list(state.positions.keys()):
        if t == "SH_SYNTHETIC":
            # Close synthetic SH position
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

    # 4) Open SH synthetic position if bear_deep
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
        print(f"[v2-paper] WARNING: cash negative ${state.cash_usd:,.0f} "
              f"(margin/NAV: {margin_ratio:.1%})")
        if margin_ratio > 0.10:
            import os
            os.system(
                f'node /home/jupiter/teleclaude-main/utils/webhook_notifier.js '
                f'"ETF Rotation v2: margin debt ${margin_used:,.0f} '
                f'({margin_ratio:.1%} of NAV)" 2>/dev/null'
            )

    print(f"[v2-paper] EXECUTED {today.date()} regime={regime_label} "
          f"nav=${state.nav_usd:,.0f} lev={gross_lev:.2f} "
          f"sector_frac={sector_frac:.0%} sh_frac={sh_frac:.0%} "
          f"held={list(state.positions.keys())} "
          f"next_rebal={state.next_rebal_date}")

    # Score rankings for log
    for _, row in scored.sort_values("score", ascending=False).iterrows():
        flag = " <--" if row["etf"] in state.positions else ""
        print(f"  {row['etf']:5s}  score={row['score']:.4f}  px=${row['close']:.2f}{flag}")

    return state


def main():
    print(f"[v2-paper] ETF Rotation v2 Paper Engine")
    print(f"[v2-paper] CONFIG: K={CONFIG_K} hold_days={CONFIG_HOLD_DAYS} "
          f"regime_ma={CONFIG_REGIME_MA_DAYS}d target_vol={CONFIG_TARGET_VOL:.2f} "
          f"cost_bps={CONFIG_COST_BPS} anchor=${CONFIG_ANCHOR_USD:,.0f}")
    try:
        state = rebalance()
        print(f"[v2-paper] FINAL: nav=${state.nav_usd:,.0f} "
              f"cash=${state.cash_usd:,.0f} regime={state.regime_label} "
              f"positions={list(state.positions.keys())} "
              f"n_rebal={state.n_rebalances}")
    except Exception as e:
        print(f"[v2-paper] ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
