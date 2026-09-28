"""
ETF Sector Rotation Paper Engine (LIVE — paper-only).

Wraps the deploy-grade ETF sector rotation strategy (HC #589, A1):
  - Universe: 11 sector SPDR ETFs (XLK XLF XLE XLY XLP XLU XLI XLV XLB XLC XLRE).
  - Picker: cross-sectional ridge over momentum features (ret_20d, ret_60d,
    rel_strength_spy, momentum_cross_20_60, rs_rank_among_sectors). Fit on
    rolling 24-month window of forward-21d returns.
  - Portfolio: long top-2 (long-only, KEEP_FLAT verdict per HC #575 R2).
  - Hold / rebalance: 21 trading days.
  - Regime gate: SPY > MA60 = "bull" -> enter/hold; "bear" -> sit in cash.
  - Costs: 5 bps round-trip per leg (commission + slippage proxy).
  - Anchor: $20K (matches megacap K=6 sizing so books are comparable).

Pooled OOT (backtest, etf_rotation_regime_20260608_164726_hold21_longonly):
  Sharpe 1.92, Sortino 2.77, Calmar 3.24, CAGR 26.5%, MaxDD -8.2%,
  419 trades, 8/10 folds Calmar>=1.0.

This engine runs ONE rebalance step per invocation (cron-driven, not a daemon).
On non-rebalance days the engine simply marks the book to market and updates NAV.

Outputs:
  - live_trading_linux/data/etf_rotation_paper_state.json
  - live_trading_linux/data/etf_rotation_paper_trades.jsonl

Strategy spec source:
  strategy/macro_picker/etf_rotation_v1.py
  output/macro_picker/etf_rotation_regime_20260608_164726_hold21_longonly/
"""
from __future__ import annotations
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "strategy" / "macro_picker"))

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRADES_LOG = DATA_DIR / "etf_rotation_paper_trades.jsonl"
STATE_FILE = DATA_DIR / "etf_rotation_paper_state.json"

# ---------------------------------------------------------------------------
# CONFIG — locked to deploy-grade backtest spec (HC #589 A1)
# ---------------------------------------------------------------------------
ENTRIES_PAUSED = False        # Paper-only. KEEP_FLAT regime verdict applies (no risk dial).
CONFIG_K = 2                  # top-K long
CONFIG_HOLD_DAYS = 21         # ~monthly rebalance
CONFIG_TRAIN_MONTHS = 24
CONFIG_REGIME_MA_DAYS = 60    # SPY > MA60 gates bull/bear
CONFIG_ANCHOR_USD = 20000.0
CONFIG_COST_BPS = 5.0         # round-trip per leg, applied on rebal days
CONFIG_TARGET_VOL = 0.15      # annualised, vol-target sizing
CONFIG_LEV_MIN = 0.25
CONFIG_LEV_MAX = 2.0

UNIVERSE = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLU", "XLI", "XLV", "XLB", "XLC", "XLRE"]
BENCH_SPY = "SPY"

PRICE_PATH = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"
TRADING_DAYS = 252


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
    gross_leverage: float = 1.0
    entries_paused: bool = ENTRIES_PAUSED
    n_rebalances: int = 0
    cumulative_realized_pnl: float = 0.0

    def save(self):
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls):
        if STATE_FILE.exists():
            d = json.loads(STATE_FILE.read_text())
            # be tolerant of older schemas
            allowed = {f for f in cls.__dataclass_fields__.keys()}
            d = {k: v for k, v in d.items() if k in allowed}
            return cls(**d)
        return cls()


def _log_trade(rec: dict):
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with TRADES_LOG.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _refresh_price_cache():
    """Refresh prices_v2.parquet with latest data from yfinance.

    Reads the existing parquet, finds the last date across all tickers,
    downloads new daily data from yfinance, recomputes derived columns
    (ret, log_ret, rv_20, rv_60, rv_252), and overwrites the parquet.
    Skips refresh if data is already current (last trading day).
    """
    import yfinance as yf

    if not PRICE_PATH.exists():
        print("[etf-rotation-paper] price cache missing, cannot refresh")
        return

    px = pd.read_parquet(PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    last_date = px["date"].max()
    today = pd.Timestamp.today().normalize()

    # Skip if data is from today or yesterday (weekends/holidays handled by gap check)
    gap_days = (today - last_date).days
    if gap_days <= 1:
        print(f"[etf-rotation-paper] price cache is current (last={last_date.date()}), skipping refresh")
        return

    print(f"[etf-rotation-paper] price cache stale (last={last_date.date()}, gap={gap_days}d), refreshing via yfinance...")
    # Only refresh tickers we actually need (avoids errors on delisted tickers in shared parquet)
    needed = set(UNIVERSE + [BENCH_SPY])
    all_tickers = sorted(t for t in px["ticker"].unique().tolist() if t in needed)

    # Download from day after last_date to today
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        raw = yf.download(all_tickers, start=start, end=end, group_by="ticker",
                          auto_adjust=True, progress=False, threads=True)
    except Exception as e:
        print(f"[etf-rotation-paper] yfinance download failed: {e}")
        return

    if raw.empty:
        print("[etf-rotation-paper] no new data from yfinance (market closed?)")
        return

    # Parse multi-ticker download into long format matching parquet schema
    new_rows = []
    for ticker in all_tickers:
        try:
            if len(all_tickers) > 1:
                tk_data = raw[ticker].dropna(subset=["Close"])
            else:
                tk_data = raw.dropna(subset=["Close"])
        except (KeyError, TypeError):
            continue
        if tk_data.empty:
            continue
        for dt, row in tk_data.iterrows():
            new_rows.append({
                "ticker": ticker,
                "date": pd.Timestamp(dt).normalize(),
                "open": float(row.get("Open", row.get("Close", 0))),
                "high": float(row.get("High", row.get("Close", 0))),
                "low": float(row.get("Low", row.get("Close", 0))),
                "close": float(row["Close"]),
                "volume": int(row.get("Volume", 0)),
            })

    if not new_rows:
        print("[etf-rotation-paper] no parseable new rows from yfinance")
        return

    new_df = pd.DataFrame(new_rows)
    # Avoid duplicates
    new_df = new_df[~new_df.set_index(["ticker", "date"]).index.isin(
        px.set_index(["ticker", "date"]).index)]
    if new_df.empty:
        print("[etf-rotation-paper] all downloaded data already in cache")
        return

    # Concat and recompute derived columns
    combined = pd.concat([px, new_df], ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    combined["ret"] = combined.groupby("ticker")["close"].pct_change()
    combined["log_ret"] = np.log1p(combined["ret"])
    combined["rv_20"] = combined.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(20, min_periods=10).std() * np.sqrt(252))
    combined["rv_60"] = combined.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(60, min_periods=30).std() * np.sqrt(252))
    combined["rv_252"] = combined.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(252, min_periods=120).std() * np.sqrt(252))

    combined.to_parquet(PRICE_PATH, index=False)
    new_max = combined["date"].max()
    print(f"[etf-rotation-paper] price cache updated: {len(new_df)} new rows, "
          f"now through {new_max.date()} ({len(combined)} total rows)")


def _load_prices(today: pd.Timestamp) -> pd.DataFrame:
    """Load the last ~3y of daily closes for the universe + SPY benchmark."""
    px = pd.read_parquet(PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    cutoff = today - pd.Timedelta(days=900)
    sub = px[(px["ticker"].isin(UNIVERSE + [BENCH_SPY]))
             & (px["date"] >= cutoff)
             & (px["date"] <= today)].copy()
    sub["close"] = sub["close"].astype(float)
    return sub.sort_values(["ticker", "date"]).reset_index(drop=True)


def _check_regime(prices: pd.DataFrame, today: pd.Timestamp) -> tuple[str, float, float]:
    """Bull if SPY close > SPY MA60, else bear. Returns (regime, spy_close, spy_ma)."""
    spy = prices[prices["ticker"] == BENCH_SPY].sort_values("date").set_index("date")["close"]
    if spy.empty:
        return "cash", float("nan"), float("nan")
    ma = spy.rolling(CONFIG_REGIME_MA_DAYS,
                     min_periods=max(20, CONFIG_REGIME_MA_DAYS // 2)).mean()
    asof = spy.index[spy.index <= today]
    if len(asof) == 0:
        return "cash", float("nan"), float("nan")
    d = asof[-1]
    s_close = float(spy.loc[d])
    s_ma = float(ma.loc[d]) if pd.notna(ma.loc[d]) else float("nan")
    if not np.isfinite(s_ma):
        return "cash", s_close, s_ma
    return ("bull" if s_close > s_ma else "bear"), s_close, s_ma


# ---------------------------------------------------------------------------
# Picker — cross-sectional ridge on momentum features (matches research spec)
# ---------------------------------------------------------------------------
FEATS = ["ret_20d", "ret_60d", "rel_strength_spy",
         "momentum_cross_20_60", "rs_rank_among_sectors"]


def _build_feature_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Build per-ETF daily feature panel from raw closes."""
    spy = prices[prices["ticker"] == BENCH_SPY].sort_values("date").set_index("date")["close"]
    spy_r20 = spy.pct_change(20)
    spy_r60 = spy.pct_change(60)

    rows = []
    for t in UNIVERSE:
        s = prices[prices["ticker"] == t].sort_values("date").copy()
        if s.empty:
            continue
        s["ret_1d"] = s["close"].pct_change()
        s["ret_20d"] = s["close"].pct_change(20)
        s["ret_60d"] = s["close"].pct_change(60)
        s["sma20"] = s["close"].rolling(20).mean()
        s["sma60"] = s["close"].rolling(60).mean()
        s["momentum_cross_20_60"] = (s["sma20"] / s["sma60"]) - 1.0
        spy20 = spy_r20.reindex(s["date"].values).values
        spy60 = spy_r60.reindex(s["date"].values).values
        # use 60d RS vs SPY to match etf_rotation_v1's rel_strength_spy
        s["rel_strength_spy"] = s["ret_60d"].values - spy60
        rows.append(s)
    panel = pd.concat(rows, ignore_index=True)

    # rs_rank_among_sectors: cross-sectional rank of ret_20d (0..1)
    panel["rs_rank_among_sectors"] = panel.groupby("date")["ret_20d"].rank(pct=True)
    return panel


def _xs_zscore(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = panel.copy()
    for f in feats:
        if f not in out.columns:
            out[f] = 0.0
            continue
        x = pd.to_numeric(out[f], errors="coerce").astype(float)
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        x = x.clip(lower=lo, upper=hi)
        out[f] = x
        mu = out.groupby("date")[f].transform("mean")
        sd = out.groupby("date")[f].transform("std")
        z = (x - mu) / sd.replace(0.0, np.nan)
        out[f] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _fit_ridge(X: np.ndarray, y: np.ndarray,
               alphas=(0.1, 1.0, 10.0, 100.0)) -> tuple:
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    best = (None, None, None, float("inf"))
    for a in alphas:
        try:
            beta = np.linalg.solve(XtX + a * np.eye(X.shape[1]), Xty)
            resid = yc - Xc @ beta
            mse = float((resid ** 2).mean())
            if mse < best[3]:
                best = (beta, y.mean() - X.mean(axis=0) @ beta, a, mse)
        except np.linalg.LinAlgError:
            continue
    return best[0], best[1], best[2]


def _score_sectors(panel: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Train ridge on last 24 months excluding today, score latest snap."""
    p = panel.sort_values(["ticker", "date"]).copy()
    p["y_fwd"] = p.groupby("ticker")["close"].shift(-CONFIG_HOLD_DAYS) / p["close"] - 1.0

    train_start = today - pd.DateOffset(months=CONFIG_TRAIN_MONTHS)
    train = p[(p["date"] >= train_start) & (p["date"] < today)].dropna(subset=["y_fwd"])
    if len(train) < 200:
        return pd.DataFrame()

    train_z = _xs_zscore(train, FEATS)
    X = train_z[FEATS].values
    y = train_z["y_fwd"].values
    coef, intercept, _alpha = _fit_ridge(X, y)
    if coef is None:
        return pd.DataFrame()

    asof = p[p["date"] <= today].sort_values("date").groupby("ticker").tail(1).copy()
    asof_z = _xs_zscore(asof, FEATS)
    asof_z["score"] = asof_z[FEATS].values @ coef + intercept
    return asof_z[["ticker", "close", "date", "score"]].rename(columns={"ticker": "etf"})


# ---------------------------------------------------------------------------
# Vol-target sizing
# ---------------------------------------------------------------------------
def _estimate_book_vol(panel: pd.DataFrame, today: pd.Timestamp,
                       longs: list[str], lookback_days: int = 60) -> float:
    """Equal-weight long-only book daily-vol over prior `lookback_days`."""
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
# Mark-to-market (non-rebalance days)
# ---------------------------------------------------------------------------
def _mark_to_market(state: PaperState, prices: pd.DataFrame, today: pd.Timestamp) -> float:
    """Update NAV from current cash + position MV. Returns total NAV."""
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
# Rebalance
# ---------------------------------------------------------------------------
def _is_rebalance_due(state: PaperState, today: pd.Timestamp) -> bool:
    """First call ever or 21+ calendar days since last rebal."""
    if state.last_rebal_date is None:
        return True
    last = pd.Timestamp(state.last_rebal_date)
    # 21 trading days ~ 30 calendar days; gate on calendar buffer
    return (today - last) >= pd.Timedelta(days=29)


def rebalance(today: Optional[pd.Timestamp] = None) -> PaperState:
    today = today or pd.Timestamp.today().normalize()
    state = PaperState.load()
    state.entries_paused = ENTRIES_PAUSED

    prices = _load_prices(today)
    if prices.empty:
        print(f"[etf-rotation-paper] no price data for {today.date()}")
        return state

    panel = _build_feature_panel(prices)
    if panel.empty:
        print(f"[etf-rotation-paper] no feature panel for {today.date()}")
        return state

    # always mark current book to market first (NAV up-to-date for logs)
    _mark_to_market(state, prices, today)

    # regime check
    regime, spy_close, spy_ma = _check_regime(prices, today)
    state.regime = regime

    # decide if a rebalance is due today
    rebal_due = _is_rebalance_due(state, today)
    if not rebal_due:
        state.save()
        print(f"[etf-rotation-paper] {today.date()} MTM-only "
              f"nav=${state.nav_usd:,.0f} regime={regime} "
              f"positions={list(state.positions.keys())} "
              f"next_rebal={state.next_rebal_date}")
        _log_trade({
            "rebal_date": str(today.date()),
            "action": "MTM",
            "regime": regime,
            "nav_usd": state.nav_usd,
            "positions": dict(state.positions),
        })
        return state

    # score sectors
    scored = _score_sectors(panel, today)
    if scored.empty:
        print(f"[etf-rotation-paper] could not score; skipping rebal")
        state.save()
        return state

    # pick target book
    if regime != "bull":
        target = {}
        target_note = "regime=bear -> cash"
    else:
        # No-trade floor: top score must exceed median
        med = scored["score"].median()
        if scored["score"].max() <= med:
            target = {}
            target_note = "top score below median -> cash"
        else:
            top = scored.nlargest(CONFIG_K, "score")
            target = {row["etf"]: 1.0 / CONFIG_K for _, row in top.iterrows()}
            target_note = "bull regime, equal-weight top-K"

    # vol-target sizing on the picked book
    if target:
        gross_lev = _estimate_book_vol(panel, today, list(target.keys()))
        if gross_lev <= 1e-6:
            gross_lev = 1.0
        else:
            gross_lev = float(np.clip(CONFIG_TARGET_VOL / gross_lev, CONFIG_LEV_MIN, CONFIG_LEV_MAX))
    else:
        gross_lev = 0.0
    state.gross_leverage = gross_lev

    # latest close lookup for execution price
    latest_px = {row["etf"]: float(row["close"]) for _, row in scored.iterrows()}

    log_rec = {
        "rebal_date": str(today.date()),
        "action": "REBAL",
        "regime": regime,
        "spy_close": spy_close,
        "spy_ma60": spy_ma,
        "target_weights": target,
        "gross_leverage": gross_lev,
        "nav_usd_before": state.nav_usd,
        "note": target_note,
        "entries_paused": ENTRIES_PAUSED,
    }

    if ENTRIES_PAUSED:
        log_rec["executed"] = False
        log_rec["reason"] = "ENTRIES_PAUSED"
        _log_trade(log_rec)
        state.save()
        print(f"[etf-rotation-paper] {today.date()} PAUSED regime={regime} "
              f"would-pick={list(target.keys())} lev={gross_lev:.2f}")
        return state

    # === EXECUTE ===
    # 1) sell current positions not in target (or sell all if going to cash)
    nav_before = state.nav_usd
    realized_pnl_this_rebal = 0.0
    for t in list(state.positions.keys()):
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
        realized_pnl_this_rebal += realized
        state.cash_usd += net_proceeds
        _log_trade({
            "rebal_date": str(today.date()),
            "action": "SELL",
            "ticker": t,
            "shares": shares,
            "px": px_now,
            "entry_px": entry_px,
            "cost_usd": cost,
            "gross_proceeds_usd": gross_proceeds,
            "realized_pnl_usd": realized,
        })
        del state.positions[t]
        if t in state.entry_prices:
            del state.entry_prices[t]

    # 2) re-mark NAV (cash + remaining MV)
    _mark_to_market(state, prices, today)

    # 3) buy / rebalance into target
    if target:
        deployable = state.nav_usd * gross_lev
        new_positions = {}
        for t, w in target.items():
            px_now = latest_px.get(t)
            if px_now is None or px_now <= 0:
                continue
            target_dollars = w * deployable
            # if we already hold it, the existing position counts; trade to target
            current_shares = state.positions.get(t, 0)
            current_mv = current_shares * px_now
            delta_dollars = target_dollars - current_mv
            # round to whole shares
            delta_shares = int(delta_dollars / px_now)
            if delta_shares > 0:
                gross_cost = delta_shares * px_now
                cost = gross_cost * (CONFIG_COST_BPS / 10000.0)
                state.cash_usd -= (gross_cost + cost)
                new_shares = current_shares + delta_shares
                # blended entry px (volume-weighted)
                old_entry = state.entry_prices.get(t, px_now)
                state.entry_prices[t] = ((old_entry * current_shares) + (px_now * delta_shares)) / max(new_shares, 1)
                new_positions[t] = new_shares
                _log_trade({
                    "rebal_date": str(today.date()),
                    "action": "BUY",
                    "ticker": t,
                    "shares": delta_shares,
                    "px": px_now,
                    "cost_usd": cost,
                    "gross_cost_usd": gross_cost,
                    "target_weight": w,
                })
            elif delta_shares < 0:
                # trim
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
                    "gross_proceeds_usd": gross_proceeds,
                    "target_weight": w,
                })
            else:
                new_positions[t] = current_shares
        state.positions = new_positions

    # 4) finalize NAV / book-keeping
    _mark_to_market(state, prices, today)
    state.cumulative_realized_pnl += realized_pnl_this_rebal
    state.last_rebal_date = str(today.date())
    state.next_rebal_date = str((today + pd.Timedelta(days=29)).date())
    state.n_rebalances += 1
    state.save()

    log_rec["executed"] = True
    log_rec["nav_usd_after"] = state.nav_usd
    log_rec["cash_after"] = state.cash_usd
    log_rec["realized_pnl_this_rebal"] = realized_pnl_this_rebal
    _log_trade(log_rec)

    # P3-2: Cash-negative alert
    if state.cash_usd < 0:
        margin_used = abs(state.cash_usd)
        print(f"[etf-rotation-paper] ⚠️ CASH NEGATIVE: ${state.cash_usd:,.0f} "
              f"(margin debt: ${margin_used:,.0f}, NAV: ${state.nav_usd:,.0f}, "
              f"margin/NAV: {margin_used/max(state.nav_usd,1):.1%})")
        if margin_used / max(state.nav_usd, 1) > 0.10:
            import os
            os.system(
                f'node /home/jupiter/teleclaude-main/utils/webhook_notifier.js '
                f'"ETF Rotation: margin debt ${margin_used:,.0f} '
                f'({margin_used/max(state.nav_usd,1):.1%} of NAV)" 2>/dev/null'
            )

    print(f"[etf-rotation-paper] EXECUTED {today.date()} regime={regime} "
          f"nav=${state.nav_usd:,.0f} lev={gross_lev:.2f} "
          f"held={list(state.positions.keys())} "
          f"next_rebal={state.next_rebal_date}")
    return state


def main():
    print(f"[etf-rotation-paper] ENTRIES_PAUSED={ENTRIES_PAUSED}")
    print(f"[etf-rotation-paper] CONFIG: K={CONFIG_K} hold_days={CONFIG_HOLD_DAYS} "
          f"regime_ma={CONFIG_REGIME_MA_DAYS}d target_vol={CONFIG_TARGET_VOL:.2f} "
          f"cost_bps={CONFIG_COST_BPS}")
    _refresh_price_cache()
    state = rebalance()
    print(f"[etf-rotation-paper] state: nav=${state.nav_usd:,.0f} "
          f"cash=${state.cash_usd:,.0f} regime={state.regime} "
          f"positions={list(state.positions.keys())} "
          f"n_rebal={state.n_rebalances} paused={state.entries_paused}")


if __name__ == "__main__":
    main()
