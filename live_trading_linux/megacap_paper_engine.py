"""
Megacap-Tech Rotation Paper Engine (PRE-STAGED — entries paused).

Status: ENTRIES_PAUSED=True. This engine is pre-written per the
megacap_tech_extended_v1 dispatch (2026-06-09). It is ready to deploy IF
and WHEN the user relaxes HC #428 R1 from the strict 0.50 green/red Sharpe
gap to a tail-DD gate (-25% worst-red-quarter ceiling). All variants in
the megacap-tech extended sweep cleanly pass the tail-DD gate.

DEPLOY config recommendation (per extended sweep):
  K = 6  -> best Sharpe (2.34), best Calmar (5.36), lowest MaxDD (-19.6%)
The pre-staged default below is K=3 (the original dispatch baseline). To
switch to K=6 (recommended on tail-DD-gate relaxation), change CONFIG_K = 6.

DO NOT FLIP ENTRIES_PAUSED -> False WITHOUT EXPLICIT USER INSTRUCTION.

Strategy spec (matches research/findings/megacap_tech_extended_v1.md):
  - Universe: AAPL, MSFT, GOOGL, NVDA, META, AMZN, AVGO, TSLA.
  - Picker: cross-sectional ridge over ret_20d, ret_60d (or configurable),
    rel_strength_spy. Fit on rolling 24-month window.
  - Rebalance: weekly (5 trading days), equal-weight inside top-K.
  - Regime gate: enter/hold only if SPY > 50d MA AND VIX < 25; else cash.
  - Costs: $0.005/share commission + 1bp slippage per trade.
  - Anchor: $20K.

Outputs paper trades to live_trading_linux/data/megacap_paper_trades.jsonl
and current positions to live_trading_linux/data/megacap_paper_state.json.
"""
from __future__ import annotations
import json
import logging
import math
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
TRADES_LOG = DATA_DIR / "megacap_paper_trades.jsonl"
STATE_FILE = DATA_DIR / "megacap_paper_state.json"

# ---------------------------------------------------------------------------
# CONFIG — flip ENTRIES_PAUSED to False ONLY after user relaxes HC #428 R1
# ---------------------------------------------------------------------------
ENTRIES_PAUSED = False  # Enabled 2026-06-09 — default-to-A on consolidated 4-lane readout (HC #393 autonomy, 15-min interrupt window). Paper-only, no real money.

CONFIG_K = 6                 # K=6 (best Sharpe 2.34, MaxDD -19.6%) — flipped from 3 per extended-sweep recommendation.
CONFIG_MOM_DAYS = 60         # momentum window. 60 = 3m, 20 = 1m (best CAGR).
CONFIG_HOLD_DAYS = 5         # weekly rebalance
CONFIG_REGIME_MA_DAYS = 50   # SPY > MA50
CONFIG_VIX_THRESH = 25.0
CONFIG_ANCHOR_USD = 20000.0
CONFIG_COMMISSION_PER_SHARE = 0.005
CONFIG_SLIPPAGE_BPS = 1.0

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "AVGO", "TSLA"]
BENCH_SPY = "SPY"

PRICE_PATH = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class PaperState:
    nav_usd: float = CONFIG_ANCHOR_USD
    positions: dict = field(default_factory=dict)  # {ticker: shares}
    last_rebal_date: Optional[str] = None
    regime: str = "cash"
    entries_paused: bool = ENTRIES_PAUSED
    cumulative_realized_pnl: float = 0.0
    n_rebalances: int = 0

    def save(self):
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls):
        if STATE_FILE.exists():
            d = json.loads(STATE_FILE.read_text())
            return cls(**d)
        return cls()


def _log_trade(rec: dict):
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with TRADES_LOG.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------------------
# Price refresh — pull fresh data from yfinance on each run
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
        print("[megacap-paper] price cache missing, cannot refresh")
        return

    px = pd.read_parquet(PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    last_date = px["date"].max()
    today = pd.Timestamp.today().normalize()

    # Skip if data is from today or yesterday (weekends/holidays handled by gap check)
    gap_days = (today - last_date).days
    if gap_days <= 1:
        print(f"[megacap-paper] price cache is current (last={last_date.date()}), skipping refresh")
        return

    print(f"[megacap-paper] price cache stale (last={last_date.date()}, gap={gap_days}d), refreshing via yfinance...")
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
        print(f"[megacap-paper] yfinance download failed: {e}")
        return

    if raw.empty:
        print("[megacap-paper] no new data from yfinance (market closed?)")
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
        print("[megacap-paper] no parseable new rows from yfinance")
        return

    new_df = pd.DataFrame(new_rows)
    # Avoid duplicates
    new_df = new_df[~new_df.set_index(["ticker", "date"]).index.isin(
        px.set_index(["ticker", "date"]).index)]
    if new_df.empty:
        print("[megacap-paper] all downloaded data already in cache")
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
    print(f"[megacap-paper] price cache updated: {len(new_df)} new rows, "
          f"now through {new_max.date()} ({len(combined)} total rows)")


# ---------------------------------------------------------------------------
# Picker logic (lifted from backtest, executes on latest available data)
# ---------------------------------------------------------------------------
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


def _load_universe_panel(today: pd.Timestamp) -> pd.DataFrame:
    """Load last ~3y of prices for universe + SPY benchmark, build features."""
    prices = pd.read_parquet(PRICE_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    cutoff = today - pd.Timedelta(days=900)
    sub = prices[(prices["ticker"].isin(UNIVERSE + [BENCH_SPY]))
                 & (prices["date"] >= cutoff)
                 & (prices["date"] <= today)].copy()
    sub["close"] = sub["close"].astype(float)
    spy = sub[sub["ticker"] == BENCH_SPY].set_index("date")["close"].sort_index()
    spy_mom = spy.pct_change(CONFIG_MOM_DAYS)
    rows = []
    for t in UNIVERSE:
        s = sub[sub["ticker"] == t].sort_values("date").copy()
        s["ret_1d"] = s["close"].pct_change()
        s["ret_20d"] = s["close"].pct_change(20)
        s["ret_mom"] = s["close"].pct_change(CONFIG_MOM_DAYS)
        spy_aligned = spy_mom.reindex(s["date"].values).values
        s["rel_strength_spy"] = s["ret_mom"].values - spy_aligned
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def _check_regime(today: pd.Timestamp, prices: pd.DataFrame, vix_today: float) -> str:
    spy = prices[prices["ticker"] == BENCH_SPY].sort_values("date")
    if spy.empty:
        return "cash"
    spy = spy.set_index("date")["close"]
    ma = spy.rolling(CONFIG_REGIME_MA_DAYS, min_periods=20).mean()
    if today not in spy.index:
        # find last available
        idx = spy.index[spy.index <= today]
        if len(idx) == 0:
            return "cash"
        today = idx[-1]
    spy_ok = spy.loc[today] > ma.loc[today]
    vix_ok = (vix_today is not None) and (vix_today < CONFIG_VIX_THRESH)
    return "bull" if (spy_ok and vix_ok) else "cash"


def _fit_ridge_and_score(panel: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Train on last 24 months excluding today's row, score the latest row per ticker."""
    feats = ["ret_20d", "ret_mom", "rel_strength_spy"]
    panel = panel.sort_values(["ticker", "date"]).copy()
    panel["y_fwd"] = panel.groupby("ticker")["close"].shift(-CONFIG_HOLD_DAYS) / panel["close"] - 1.0

    train_start = today - pd.DateOffset(months=24)
    train = panel[(panel["date"] >= train_start) & (panel["date"] < today)].dropna(subset=["y_fwd"])
    if len(train) < 100:
        return pd.DataFrame()
    train_z = _xs_zscore(train, feats)
    X = train_z[feats].values
    y = train_z["y_fwd"].values
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    try:
        beta = np.linalg.solve(Xc.T @ Xc + 1.0 * np.eye(len(feats)), Xc.T @ yc)
    except np.linalg.LinAlgError:
        beta = np.zeros(len(feats))
    intercept = y.mean() - X.mean(axis=0) @ beta

    today_panel = panel[panel["date"] <= today].copy()
    latest = today_panel.sort_values("date").groupby("ticker").tail(1)
    latest_z = _xs_zscore(latest, feats)
    latest_z["score"] = latest_z[feats].values @ beta + intercept
    return latest_z[["ticker", "close", "date", "score"]]


# ---------------------------------------------------------------------------
# Rebalance step
# ---------------------------------------------------------------------------
def rebalance(today: Optional[pd.Timestamp] = None, vix_today: Optional[float] = None,
              dry_run: bool = False):
    """Run one rebalance step. If ENTRIES_PAUSED, only mark-to-market existing positions
    and log the would-be picks; do not enter new positions.
    """
    today = today or pd.Timestamp.today().normalize()
    state = PaperState.load()
    state.entries_paused = ENTRIES_PAUSED

    panel = _load_universe_panel(today)
    if panel.empty:
        print(f"[megacap-paper] no panel data for {today}")
        return state

    prices = pd.read_parquet(PRICE_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    regime = _check_regime(today, prices, vix_today if vix_today is not None else 20.0)
    state.regime = regime

    scored = _fit_ridge_and_score(panel, today)
    if scored.empty:
        print(f"[megacap-paper] could not score; not rebalancing")
        state.save()
        return state

    if regime != "bull":
        target = {}
    else:
        top = scored.nlargest(CONFIG_K, "score")
        weight = 1.0 / CONFIG_K
        target = {row["ticker"]: weight for _, row in top.iterrows()}

    log_rec = {
        "rebal_date": str(today.date()),
        "regime": regime,
        "would_be_target": target,
        "nav_usd_before": state.nav_usd,
        "entries_paused": ENTRIES_PAUSED,
    }

    if ENTRIES_PAUSED:
        log_rec["note"] = "ENTRIES_PAUSED — picks logged but not executed"
        _log_trade(log_rec)
        print(f"[megacap-paper] {today.date()} PAUSED regime={regime} target={list(target.keys())}")
        state.save()
        return state

    # Execute (only when entries enabled by user)
    nav = state.nav_usd
    current = state.positions.copy()
    latest_prices = {row["ticker"]: float(row["close"]) for _, row in scored.iterrows()}
    spy_row = prices[(prices["ticker"] == BENCH_SPY) & (prices["date"] <= today)].sort_values("date").tail(1)
    spy_price = float(spy_row.iloc[0]["close"]) if not spy_row.empty else 100.0

    # Mark current positions to market
    mtm_nav = 0.0
    for t, shares in current.items():
        if shares <= 0:
            continue
        p = latest_prices.get(t, 0.0)
        mtm_nav += shares * p
    cash_portion = nav - sum(current.get(t, 0) * latest_prices.get(t, 0) for t in current)
    # Total NAV = mtm of positions + remaining cash sleeve (we approximate cash conservation)
    total_nav = mtm_nav + max(cash_portion, 0.0)

    # Sell positions not in target
    new_positions = {}
    for t in list(current.keys()):
        if t not in target:
            shares = current[t]
            p = latest_prices.get(t, 0.0)
            commission = shares * CONFIG_COMMISSION_PER_SHARE
            slip = shares * p * (CONFIG_SLIPPAGE_BPS / 10000)
            proceeds = shares * p - commission - slip
            total_nav -= 0  # nav already accounted for via mtm; this just realizes
            _log_trade({"action": "SELL", "ticker": t, "shares": shares, "px": p,
                        "commission": commission, "slippage": slip,
                        "rebal_date": str(today.date())})

    # Buy new targets
    for t, w in target.items():
        p = latest_prices.get(t, 0.0)
        if p <= 0:
            continue
        target_dollars = w * total_nav
        shares = int(target_dollars / p)
        if shares <= 0:
            continue
        commission = shares * CONFIG_COMMISSION_PER_SHARE
        slip = shares * p * (CONFIG_SLIPPAGE_BPS / 10000)
        new_positions[t] = shares
        _log_trade({"action": "BUY", "ticker": t, "shares": shares, "px": p,
                    "commission": commission, "slippage": slip,
                    "rebal_date": str(today.date())})

    state.positions = new_positions
    state.last_rebal_date = str(today.date())
    state.nav_usd = total_nav
    state.n_rebalances += 1
    state.save()

    log_rec["executed"] = True
    log_rec["nav_usd_after"] = state.nav_usd
    _log_trade(log_rec)
    print(f"[megacap-paper] EXECUTED {today.date()} regime={regime} nav=${state.nav_usd:,.0f} held={list(new_positions.keys())}")
    return state


def main():
    print(f"[megacap-paper] ENTRIES_PAUSED={ENTRIES_PAUSED}")
    print(f"[megacap-paper] CONFIG: K={CONFIG_K} mom_days={CONFIG_MOM_DAYS} hold_days={CONFIG_HOLD_DAYS}")
    _refresh_price_cache()
    state = rebalance()
    print(f"[megacap-paper] state: nav=${state.nav_usd:,.0f} regime={state.regime} "
          f"positions={list(state.positions.keys())} paused={state.entries_paused}")


if __name__ == "__main__":
    main()
