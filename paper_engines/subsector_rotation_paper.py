#!/usr/bin/env python3
"""
Sub-Sector Rotation Paper Trading Engine (Validated Pairs)
============================================================
Trades 5 adversarial-validated sub-sector rotation pairs using a pre-trained
LGBM model. Runs daily at 4:25 PM ET after market close.

Validated pairs:
  1. VNQ vs XLRE (REITs vs Real Estate) — 5d, Sharpe 4.6
  2. GDX vs XME (Gold Miners vs Metals) — 5d, Sharpe 2.6
  3. KRE vs XLF (Regional Banks vs Financials) — 10d, Sharpe 2.1
  4. XLY vs XLP (Consumer Disc vs Staples) — 5d, Sharpe 2.1
  5. KBE vs KIE (Banks vs Insurance) — 5d, Sharpe 2.0

Strategy:
  - Compute rotation features (63d relative return, 21d vol ratio, etc.)
  - Run LGBM model for mean-reversion probability
  - If prob > 60%: buy the lagging sub-sector ETF
  - Track paper trades with entry/exit, P&L, hold period
  - Idempotent: safe to run multiple times per day

State: /home/jupiter/Lvl3Quant/state/subsector_rotation_state.json
Logs:  paper_engines/logs/subsector_rotation_paper.log

Usage:
    python3 paper_engines/subsector_rotation_paper.py
"""

import json
import logging
import pickle
import warnings
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
LOG_DIR = BASE_DIR / "paper_engines" / "logs"
LOG_DIR.mkdir(exist_ok=True)
STATE_DIR = BASE_DIR / "state"
STATE_DIR.mkdir(exist_ok=True)

STATE_PATH = STATE_DIR / "subsector_rotation_validated_pairs_state.json"
MODEL_PATH = BASE_DIR / "models" / "subsector_rotation_lgbm.pkl"
TRADE_LOG_PATH = STATE_DIR / "subsector_rotation_paper_trades.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SubSecRotation] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "subsector_rotation_paper.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validated Pairs
# ---------------------------------------------------------------------------
VALIDATED_PAIRS = {
    "VNQ_vs_XLRE": {
        "etf_a": "VNQ", "etf_b": "XLRE", "horizon": 5,
        "label_a": "REITs", "label_b": "Real Estate",
        "backtest_sharpe": 4.6,
    },
    "GDX_vs_XME": {
        "etf_a": "GDX", "etf_b": "XME", "horizon": 5,
        "label_a": "Gold Miners", "label_b": "Metals/Mining",
        "backtest_sharpe": 2.6,
    },
    "KRE_vs_XLF": {
        "etf_a": "KRE", "etf_b": "XLF", "horizon": 10,
        "label_a": "Regional Banks", "label_b": "Financials",
        "backtest_sharpe": 2.1,
    },
    "XLY_vs_XLP": {
        "etf_a": "XLY", "etf_b": "XLP", "horizon": 5,
        "label_a": "Consumer Disc", "label_b": "Consumer Staples",
        "backtest_sharpe": 2.1,
    },
    "KBE_vs_KIE": {
        "etf_a": "KBE", "etf_b": "KIE", "horizon": 5,
        "label_a": "Banks", "label_b": "Insurance",
        "backtest_sharpe": 2.0,
    },
}

# Strategy constants
CONFIDENCE_THRESHOLD = 0.60
CAPITAL_PER_PAIR = 500.0  # Paper capital allocated per pair
STOP_LOSS_PCT = 0.05       # 5% stop loss
TAKE_PROFIT_PCT = 0.10     # 10% take profit


# ---------------------------------------------------------------------------
# Feature computation (must match training exactly)
# ---------------------------------------------------------------------------
def compute_features(close_a: pd.Series, close_b: pd.Series, spy_close: pd.Series) -> pd.DataFrame:
    """Compute rotation features for a sub-sector pair. Must match training script."""
    ret_a = close_a.pct_change()
    ret_b = close_b.pct_change()
    ratio = close_a / close_b

    features = pd.DataFrame(index=close_a.index)

    for w in [5, 10, 21, 63]:
        features[f"rel_ret_{w}d"] = (close_a / close_a.shift(w)) / (close_b / close_b.shift(w)) - 1

    delta = ratio.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features["ratio_rsi_14"] = 100 - (100 / (1 + rs))

    ratio_ret = ratio.pct_change()
    for w in [21, 63]:
        roll_mean = ratio_ret.rolling(w).mean()
        roll_std = ratio_ret.rolling(w).std()
        features[f"rel_zscore_{w}d"] = (ratio_ret - roll_mean) / roll_std.replace(0, np.nan)

    for w in [21, 63]:
        features[f"corr_{w}d"] = ret_a.rolling(w).corr(ret_b)

    features["corr_change_21d"] = features["corr_21d"] - features["corr_21d"].shift(21)

    vol_a = ret_a.rolling(21).std()
    vol_b = ret_b.rolling(21).std()
    features["vol_ratio_21d"] = vol_a / vol_b.replace(0, np.nan)

    mom_a = close_a / close_a.shift(21) - 1
    mom_b = close_b / close_b.shift(21) - 1
    features["mom_divergence_21d"] = mom_a - mom_b

    ratio_sma = ratio.rolling(63).mean()
    features["ratio_dist_sma63"] = (ratio / ratio_sma) - 1

    spy_sma200 = spy_close.rolling(200).mean()
    features["spy_regime"] = (spy_close > spy_sma200).astype(int)
    features["spy_ret_21d"] = spy_close.pct_change(21)

    return features


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_pair_data(etf_a: str, etf_b: str, lookback_days: int = 300) -> tuple:
    """Download recent price data for a pair + SPY."""
    if not HAS_YF:
        return None, None, None

    tickers = sorted(set([etf_a, etf_b, "SPY"]))
    try:
        data = yf.download(tickers, period=f"{lookback_days}d", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data

        if etf_a not in close.columns or etf_b not in close.columns:
            return None, None, None

        common = close.dropna()
        return common[etf_a], common[etf_b], common.get("SPY", common[etf_a] * 0 + 1)
    except Exception as e:
        log.error(f"Data download failed: {e}")
        return None, None, None


def get_price(ticker: str) -> float:
    """Get current price for a single ticker."""
    if not HAS_YF:
        return 0.0
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model() -> dict | None:
    """Load the pre-trained LGBM model."""
    if not MODEL_PATH.exists():
        log.error(f"Model not found at {MODEL_PATH}. Run train_subsector_rotation_model.py first.")
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        return None


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def load_state() -> dict:
    """Load or initialize paper trading state."""
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "signals": [],
        "positions": [],
        "closed_trades": [],
        "trade_count": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "last_run": None,
        "last_run_date": None,
        "created_at": datetime.now().isoformat(),
    }


def save_state(state: dict):
    """Save state to JSON."""
    state["updated_at"] = datetime.now().isoformat()
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(trade: dict):
    """Append trade to JSONL log."""
    with open(TRADE_LOG_PATH, "a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def check_exits(state: dict) -> list:
    """Check open positions for exit conditions."""
    exits = []
    today = date.today()

    for pos in state["positions"]:
        ticker = pos["ticker"]
        entry_price = pos["entry_price"]
        current_price = get_price(ticker)

        if current_price <= 0:
            continue

        pnl_pct = (current_price - entry_price) / entry_price
        pos["current_price"] = round(current_price, 2)
        pos["unrealized_pnl_pct"] = round(pnl_pct * 100, 2)

        # Check hold period expiry
        entry_date = datetime.fromisoformat(pos["entry_date"]).date()
        hold_days = (today - entry_date).days
        horizon = pos.get("horizon", 5)

        exit_reason = None

        if pnl_pct >= TAKE_PROFIT_PCT:
            exit_reason = f"TAKE_PROFIT ({pnl_pct*100:.1f}%)"
        elif pnl_pct <= -STOP_LOSS_PCT:
            exit_reason = f"STOP_LOSS ({pnl_pct*100:.1f}%)"
        elif hold_days >= horizon * 2:
            # Hold for up to 2x the prediction horizon, then exit
            exit_reason = f"HORIZON_EXPIRY ({hold_days}d held, {horizon}d horizon)"

        if exit_reason:
            pnl_dollars = (current_price - entry_price) * pos["shares"]
            exits.append({
                "pair": pos["pair"],
                "ticker": ticker,
                "direction": pos["direction"],
                "entry_price": entry_price,
                "exit_price": round(current_price, 2),
                "shares": pos["shares"],
                "pnl_pct": round(pnl_pct * 100, 2),
                "pnl_dollars": round(pnl_dollars, 2),
                "exit_reason": exit_reason,
                "entry_date": pos["entry_date"],
                "exit_date": datetime.now().isoformat(),
                "hold_days": hold_days,
            })

    return exits


def process_exits(state: dict, exits: list):
    """Process exits and update state."""
    for ex in exits:
        ticker = ex["ticker"]
        pair = ex["pair"]

        state["positions"] = [
            p for p in state["positions"]
            if not (p["ticker"] == ticker and p["pair"] == pair)
        ]

        state["closed_trades"].append(ex)
        state["trade_count"] += 1
        state["total_pnl"] += ex["pnl_dollars"]

        if ex["pnl_dollars"] > 0:
            state["wins"] += 1
        else:
            state["losses"] += 1

        log_trade({"action": "EXIT", **ex})
        log.info(f"  EXIT {ticker} ({pair}): {ex['pnl_pct']:+.1f}% (${ex['pnl_dollars']:+.2f}) — {ex['exit_reason']}")


def generate_signals(model_data: dict) -> list:
    """Generate rotation signals for all validated pairs."""
    signals = []
    models = model_data.get("models", {})

    for pair_name, pair_info in VALIDATED_PAIRS.items():
        etf_a = pair_info["etf_a"]
        etf_b = pair_info["etf_b"]
        horizon = pair_info["horizon"]

        if pair_name not in models:
            log.warning(f"  No model for {pair_name}, skipping")
            continue

        pair_model = models[pair_name]
        lgbm_model = pair_model["model"]
        feature_cols = pair_model["feature_cols"]

        # Download data
        close_a, close_b, spy_close = download_pair_data(etf_a, etf_b)
        if close_a is None:
            log.warning(f"  Could not get data for {pair_name}")
            continue

        # Compute features
        features = compute_features(close_a, close_b, spy_close)
        features = features.dropna()

        if len(features) == 0:
            log.warning(f"  No valid features for {pair_name}")
            continue

        # Get latest feature row
        latest = features.iloc[-1:]
        X = latest[feature_cols].values

        # Predict
        proba = lgbm_model.predict_proba(X)[:, 1]
        reversal_prob = float(proba[0])

        # Determine which ETF is lagging (the one to buy on reversal)
        rel_ret_63d = float(latest["rel_ret_63d"].iloc[0]) if "rel_ret_63d" in latest.columns else 0
        rel_ret_21d = float(latest["rel_ret_21d"].iloc[0]) if "rel_ret_21d" in latest.columns else 0

        # If A has been lagging B (rel_ret < 0), buy A on reversal signal
        # If B has been lagging A (rel_ret > 0), buy B on reversal signal
        if rel_ret_21d < 0:
            lagging_etf = etf_a
            lagging_label = pair_info["label_a"]
            leading_etf = etf_b
        else:
            lagging_etf = etf_b
            lagging_label = pair_info["label_b"]
            leading_etf = etf_a

        signal = {
            "pair": pair_name,
            "etf_a": etf_a,
            "etf_b": etf_b,
            "horizon": horizon,
            "reversal_prob": round(reversal_prob, 4),
            "rel_ret_21d": round(rel_ret_21d, 4),
            "rel_ret_63d": round(rel_ret_63d, 4),
            "lagging_etf": lagging_etf,
            "lagging_label": lagging_label,
            "leading_etf": leading_etf,
            "vol_ratio": round(float(latest["vol_ratio_21d"].iloc[0]), 4) if "vol_ratio_21d" in latest.columns else None,
            "corr_21d": round(float(latest["corr_21d"].iloc[0]), 4) if "corr_21d" in latest.columns else None,
            "ratio_dist_sma63": round(float(latest["ratio_dist_sma63"].iloc[0]), 4) if "ratio_dist_sma63" in latest.columns else None,
            "fires": reversal_prob >= CONFIDENCE_THRESHOLD,
            "timestamp": datetime.now().isoformat(),
            "backtest_sharpe": pair_info["backtest_sharpe"],
        }

        signals.append(signal)

        status = "FIRE" if signal["fires"] else "no signal"
        log.info(
            f"  {pair_name}: prob={reversal_prob:.3f} ({status}) — "
            f"lagging={lagging_etf} ({rel_ret_21d:+.2%}), "
            f"corr={signal['corr_21d']}"
        )

    return signals


def process_entries(state: dict, signals: list):
    """Open new positions for fired signals."""
    # Get currently held pairs
    held_pairs = set(p["pair"] for p in state["positions"])

    for sig in signals:
        if not sig["fires"]:
            continue

        pair = sig["pair"]
        if pair in held_pairs:
            log.info(f"  Already holding {pair}, skipping entry")
            continue

        ticker = sig["lagging_etf"]
        price = get_price(ticker)
        if price <= 0:
            log.warning(f"  Could not get price for {ticker}")
            continue

        shares = round(CAPITAL_PER_PAIR / price, 4)
        if shares <= 0.0001:
            continue

        cost = round(shares * price, 2)

        entry = {
            "pair": pair,
            "ticker": ticker,
            "direction": "LONG",
            "entry_price": round(price, 2),
            "shares": shares,
            "cost": cost,
            "entry_date": datetime.now().isoformat(),
            "horizon": sig["horizon"],
            "reversal_prob": sig["reversal_prob"],
            "lagging_label": sig["lagging_label"],
            "rel_ret_21d": sig["rel_ret_21d"],
        }

        state["positions"].append(entry)
        held_pairs.add(pair)

        log_trade({"action": "ENTRY", **entry})
        log.info(
            f"  ENTRY {ticker} ({sig['lagging_label']}): "
            f"{shares} sh @ ${price:.2f} = ${cost:.2f} | "
            f"prob={sig['reversal_prob']:.3f}, horizon={sig['horizon']}d"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    """Main paper trading loop. Idempotent — safe to run multiple times/day."""
    log.info("=" * 60)
    log.info(f"Sub-Sector Rotation Paper Engine — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Weekend check
    today = date.today()
    if today.weekday() >= 5:
        log.info("Weekend — skipping")
        return

    state = load_state()

    # Idempotency: skip if already ran today
    if state.get("last_run_date") == str(today):
        log.info(f"Already ran today ({today}). Updating position prices only.")
        # Still update position prices
        for pos in state["positions"]:
            price = get_price(pos["ticker"])
            if price > 0:
                pos["current_price"] = round(price, 2)
                pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
                pos["unrealized_pnl_pct"] = round(pnl_pct * 100, 2)
        save_state(state)
        return

    log.info(f"  Positions: {len(state['positions'])}, "
             f"Closed: {len(state['closed_trades'])}, "
             f"Total P&L: ${state['total_pnl']:+.2f}")

    # 1. Check exits
    if state["positions"]:
        exits = check_exits(state)
        if exits:
            process_exits(state, exits)
        else:
            log.info("  No exit signals triggered")

    # 2. Load model and generate signals
    model_data = load_model()
    if model_data is None:
        log.error("  Cannot generate signals without model. Exiting.")
        save_state(state)
        return

    signals = generate_signals(model_data)
    state["signals"] = signals

    # 3. Process new entries
    firing = [s for s in signals if s["fires"]]
    if firing:
        log.info(f"  {len(firing)} signal(s) firing: {[s['pair'] for s in firing]}")
        process_entries(state, signals)
    else:
        log.info("  No signals above threshold")

    # 4. Update state
    state["last_run"] = datetime.now().isoformat()
    state["last_run_date"] = str(today)

    # Performance summary
    total_trades = state["wins"] + state["losses"]
    wr = state["wins"] / total_trades * 100 if total_trades > 0 else 0

    log.info(f"\n  Summary: {len(state['positions'])} open, "
             f"{total_trades} closed trades, WR: {wr:.0f}%, "
             f"P&L: ${state['total_pnl']:+.2f}")

    if state["positions"]:
        log.info("  Open positions:")
        for pos in state["positions"]:
            pnl = pos.get("unrealized_pnl_pct", 0)
            log.info(f"    {pos['ticker']} ({pos['pair']}): "
                     f"{pos['shares']} sh @ ${pos['entry_price']:.2f} | "
                     f"P&L: {pnl:+.1f}%")

    save_state(state)
    log.info("  State saved.")

    return state


if __name__ == "__main__":
    run()
