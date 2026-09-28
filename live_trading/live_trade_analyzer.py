#!/usr/bin/env python3
"""
Live Paper Trade Analyzer
Polls Razer paper trader logs via SSH every 60s during market hours,
computes rolling metrics, and saves periodic JSON summaries.

Usage:
    python live_trade_analyzer.py              # Normal mode
    python live_trade_analyzer.py --dry-run    # Generate synthetic data for testing
"""

import argparse
import csv
import io
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376
PASSIVE_COST_TICKS = 0.376
MARKET_COST_TICKS = 1.376

RAZER_HOST = "razer"
RAZER_USER = "claude"
RAZER_PASS = os.environ.get("CLUSTER_SSH_PASSWORD", "")
RAZER_SSH_PORT = 22

POLL_INTERVAL_S = 60
SAVE_INTERVAL_S = 300  # 5 minutes

ET = pytz.timezone("US/Eastern")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/live_analysis")

LOG_FMT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger("live_analyzer")

# CSV columns expected from the paper trader
COLUMNS = [
    "timestamp", "side", "entry_price", "exit_price",
    "hold_seconds", "pnl_ticks", "signal_score", "meta_score", "fill_type",
]

# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def _get_ssh_client(retries: int = 3, backoff: float = 5.0):
    """Return a connected paramiko SSHClient, with retry + backoff."""
    import paramiko

    for attempt in range(1, retries + 1):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                RAZER_HOST,
                port=RAZER_SSH_PORT,
                username=RAZER_USER,
                password=RAZER_PASS,
                timeout=15,
                banner_timeout=15,
                auth_timeout=15,
            )
            return client
        except Exception as exc:
            wait = backoff * attempt
            log.warning("SSH attempt %d/%d failed: %s — retrying in %.0fs",
                        attempt, retries, exc, wait)
            if attempt < retries:
                time.sleep(wait)
    log.error("SSH connection failed after %d attempts", retries)
    return None


def fetch_trades_ssh(date_str: str) -> str | None:
    """
    SSH to Razer and cat the paper trader CSV for *date_str* (YYYY-MM-DD).
    Returns the raw CSV text, or None on failure.
    """
    # Razer uses Windows paths.  The paper trader writes into a folder
    # named by date in YYYY-MM-DD format.
    remote_path = (
        f"C:\\Users\\claude\\Lvl3Quant\\output\\paper_trading\\{date_str}\\trades.csv"
    )
    client = _get_ssh_client()
    if client is None:
        return None
    try:
        cmd = f'type "{remote_path}"'
        _, stdout, stderr = client.exec_command(cmd, timeout=15)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if "cannot find" in err.lower() or "not recognized" in err.lower():
            # File doesn't exist yet (market just opened, no trades yet)
            log.info("No trades file yet on Razer for %s", date_str)
            return ""
        return out
    except Exception as exc:
        log.warning("SSH command failed: %s", exc)
        return None
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Synthetic data (--dry-run)
# ---------------------------------------------------------------------------

def generate_synthetic_csv(n_trades: int = 0) -> str:
    """Return a CSV string with some random paper trades for testing."""
    now = datetime.now(ET)
    lines = []
    # Between 0 and n_trades additional rows each call
    new = random.randint(1, max(3, n_trades))
    base_price = 5500.0 + random.uniform(-20, 20)
    for i in range(new):
        ts = (now - timedelta(seconds=random.randint(0, 3600))).strftime("%Y-%m-%d %H:%M:%S")
        side = random.choice(["long", "short"])
        entry = round(base_price + random.uniform(-5, 5), 2)
        pnl_ticks = round(random.gauss(0.3, 1.5), 3)
        exit_p = round(entry + (pnl_ticks * 0.25 * (1 if side == "long" else -1)), 2)
        hold = round(random.uniform(1, 45), 1)
        sig = round(random.uniform(0.5, 1.0), 4)
        meta = round(random.uniform(0.4, 0.95), 4)
        fill = random.choice(["passive", "market"])
        lines.append(f"{ts},{side},{entry},{exit_p},{hold},{pnl_ticks},{sig},{meta},{fill}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_trades(csv_text: str) -> list[dict]:
    """Parse CSV text into list of trade dicts. Skips malformed rows."""
    trades = []
    if not csv_text or not csv_text.strip():
        return trades

    reader = csv.reader(io.StringIO(csv_text.strip()))
    for row_num, row in enumerate(reader, 1):
        # Skip header if present
        if row_num == 1 and row and row[0].strip().lower() in ("timestamp", "ts", "time"):
            continue
        if len(row) < len(COLUMNS):
            log.debug("Skipping malformed row %d (only %d cols)", row_num, len(row))
            continue
        try:
            trade = {
                "timestamp": row[0].strip(),
                "side": row[1].strip().lower(),
                "entry_price": float(row[2]),
                "exit_price": float(row[3]),
                "hold_seconds": float(row[4]),
                "pnl_ticks": float(row[5]),
                "signal_score": float(row[6]),
                "meta_score": float(row[7]),
                "fill_type": row[8].strip().lower(),
            }
            trades.append(trade)
        except (ValueError, IndexError) as exc:
            log.debug("Skipping malformed row %d: %s", row_num, exc)
    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[dict]) -> dict:
    """Compute rolling performance metrics from a list of trade dicts."""
    n = len(trades)
    if n == 0:
        return {
            "total_trades": 0,
            "win_rate": None,
            "profit_factor": None,
            "sharpe": None,
            "sortino": None,
            "avg_pnl_ticks": None,
            "fill_rate_passive_pct": None,
            "fill_rate_market_pct": None,
            "avg_hold_seconds": None,
            "total_pnl_ticks": None,
            "total_pnl_usd": None,
            "long_trades": 0,
            "short_trades": 0,
            "adverse_selection_rate": None,
        }

    pnls = [t["pnl_ticks"] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    gross_profit = sum(winners) if winners else 0.0
    gross_loss = abs(sum(losers)) if losers else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    avg_pnl = sum(pnls) / n
    total_pnl = sum(pnls)

    # Net pnl adjusted for execution costs
    net_pnls = []
    for t in trades:
        cost = PASSIVE_COST_TICKS if t["fill_type"] == "passive" else MARKET_COST_TICKS
        net_pnls.append(t["pnl_ticks"] - cost)

    # Sharpe: annualised from per-trade returns
    # Assume ~1000 trades/day, 252 trading days
    if n >= 2:
        mean_r = sum(net_pnls) / n
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in net_pnls) / (n - 1))
        if std_r > 0:
            trades_per_day = max(n, 1)  # use actual count as daily proxy
            sharpe = (mean_r / std_r) * math.sqrt(trades_per_day * 252)
        else:
            sharpe = float("inf") if mean_r > 0 else 0.0
    else:
        sharpe = None

    # Sortino: only downside deviation
    if n >= 2:
        mean_r = sum(net_pnls) / n
        downside = [min(0, r - 0) ** 2 for r in net_pnls]  # target = 0
        downside_dev = math.sqrt(sum(downside) / (n - 1))
        if downside_dev > 0:
            trades_per_day = max(n, 1)
            sortino = (mean_r / downside_dev) * math.sqrt(trades_per_day * 252)
        else:
            sortino = float("inf") if mean_r > 0 else 0.0
    else:
        sortino = None

    # Fill type breakdown
    passive_count = sum(1 for t in trades if t["fill_type"] == "passive")
    market_count = n - passive_count

    # Adverse selection: trades where price moved >1 tick against within 5s
    # We approximate this from pnl_ticks + hold_seconds: if hold <= 5 and pnl < -1, adverse
    adverse = sum(
        1 for t in trades
        if t["hold_seconds"] <= 5.0 and t["pnl_ticks"] < -1.0
    )
    adverse_rate = adverse / n if n > 0 else 0.0

    longs = sum(1 for t in trades if t["side"] == "long")
    shorts = n - longs

    return {
        "total_trades": n,
        "win_rate": round(len(winners) / n * 100, 1) if n > 0 else None,
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "sharpe": round(sharpe, 2) if sharpe is not None and sharpe != float("inf") else sharpe,
        "sortino": round(sortino, 2) if sortino is not None and sortino != float("inf") else sortino,
        "avg_pnl_ticks": round(avg_pnl, 3),
        "avg_pnl_net_ticks": round(sum(net_pnls) / n, 3) if n > 0 else None,
        "total_pnl_ticks": round(total_pnl, 2),
        "total_pnl_usd": round(total_pnl * ES_TICK_VALUE, 2),
        "fill_rate_passive_pct": round(passive_count / n * 100, 1) if n > 0 else None,
        "fill_rate_market_pct": round(market_count / n * 100, 1) if n > 0 else None,
        "avg_hold_seconds": round(sum(t["hold_seconds"] for t in trades) / n, 1),
        "long_trades": longs,
        "short_trades": shorts,
        "adverse_selection_rate": round(adverse_rate * 100, 1),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_summary(metrics: dict, date_str: str):
    """Save rolling metrics JSON to output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = OUTPUT_DIR / f"{date_str.replace('-', '')}_rolling.json"
    payload = {
        "updated_at": datetime.now(ET).isoformat(),
        "date": date_str,
        "metrics": metrics,
    }
    with open(fname, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.debug("Saved summary to %s", fname)


def print_status_line(metrics: dict):
    """Print a compact one-line status to stdout."""
    m = metrics
    if m["total_trades"] == 0:
        ts = datetime.now(ET).strftime("%H:%M:%S")
        print(f"[{ts} ET] No trades yet", flush=True)
        return

    ts = datetime.now(ET).strftime("%H:%M:%S")
    wr = m["win_rate"] if m["win_rate"] is not None else "-"
    pf = m["profit_factor"] if m["profit_factor"] is not None else "-"
    sh = m["sharpe"] if m["sharpe"] is not None else "-"
    so = m["sortino"] if m["sortino"] is not None else "-"
    avg = m["avg_pnl_ticks"] if m["avg_pnl_ticks"] is not None else "-"
    tot = m["total_pnl_ticks"]
    usd = m["total_pnl_usd"]
    n = m["total_trades"]
    adv = m["adverse_selection_rate"]
    pas = m["fill_rate_passive_pct"]

    print(
        f"[{ts} ET] "
        f"Trades={n} WR={wr}% PF={pf} Sharpe={sh} Sortino={so} "
        f"AvgPnL={avg}t Total={tot}t (${usd}) "
        f"Passive={pas}% AdvSel={adv}%",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Market hours check
# ---------------------------------------------------------------------------

def is_market_hours() -> bool:
    """Return True if current time is within 9:30-16:00 ET on a weekday."""
    now = datetime.now(ET)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live Paper Trade Analyzer")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use synthetic data instead of SSH")
    parser.add_argument("--force", action="store_true",
                        help="Run even outside market hours")
    args = parser.parse_args()

    log.info("Live Trade Analyzer started (dry_run=%s, force=%s)", args.dry_run, args.force)

    # Accumulate synthetic trades across polls in dry-run mode
    synthetic_csv_buffer = ""
    last_save_time = 0.0

    while True:
        try:
            now_et = datetime.now(ET)
            date_str = now_et.strftime("%Y-%m-%d")

            if not args.force and not is_market_hours():
                # Outside market hours — sleep and check again
                log.info("Outside market hours (%s ET). Sleeping 60s...",
                         now_et.strftime("%H:%M"))
                time.sleep(60)
                continue

            # --- Fetch trades ---
            if args.dry_run:
                synthetic_csv_buffer += generate_synthetic_csv()
                csv_text = synthetic_csv_buffer
            else:
                csv_text = fetch_trades_ssh(date_str)
                if csv_text is None:
                    log.warning("Failed to fetch trades. Will retry in %ds.", POLL_INTERVAL_S)
                    time.sleep(POLL_INTERVAL_S)
                    continue

            # --- Parse and compute ---
            trades = parse_trades(csv_text)
            metrics = compute_metrics(trades)

            # --- Print one-line status ---
            print_status_line(metrics)

            # --- Periodic JSON save (every 5 min) ---
            now_ts = time.time()
            if now_ts - last_save_time >= SAVE_INTERVAL_S:
                save_summary(metrics, date_str)
                last_save_time = now_ts
                log.info("Saved rolling summary (%d trades)", metrics["total_trades"])

            time.sleep(POLL_INTERVAL_S)

        except KeyboardInterrupt:
            log.info("Interrupted — saving final summary and exiting.")
            if trades:
                save_summary(metrics, date_str)
            break
        except Exception as exc:
            log.error("Unexpected error: %s", exc, exc_info=True)
            time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
