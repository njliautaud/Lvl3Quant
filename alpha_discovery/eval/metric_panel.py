"""
metric_panel.py — Standard 15-metric risk-adjusted evaluation panel.

Per HC #256: every model / sweep / head must be evaluated with this full panel
before any deploy/no-deploy call. P&L is one row, NOT the headline. Sortino is
the headline.

Per HC #254: passes_hc254() enforces:
  - pct_dates_positive >= 0.60
  - single_date_concentration <= 0.40
  - sortino > 0

Per HC #69: Sortino must always be reported.
Per HC #130/#231: commission only (0.376 ticks RT). NO spread-crossing cost.
Per HC #225: winner/loser breakouts mandatory.
Per HC #4: concat IC primary signal-side metric.

Inputs: per-trade DataFrame with columns:
  date, entry_ts_ns, exit_ts_ns, side ("L"/"S"), entry_price, exit_price,
  pnl_ticks (post-commission), pnl_dollars,
  optional: mfe_ticks, mae_ticks, fill_status, adverse_5s_sign, latency_ns

Author: claude / lvl3quant. Pure numpy + pandas + scipy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from scipy.stats import spearmanr

ANNUALIZATION_TRADING_DAYS = 252
EPS = 1e-9


# ---------------------------------------------------------------------------
# core metric helpers
# ---------------------------------------------------------------------------
def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if x.size > 1 else float("nan")


def _downside_std(returns: np.ndarray) -> float:
    """Pure semi-deviation: std of negative returns only (no mean subtraction)."""
    neg = returns[returns < 0]
    if neg.size == 0:
        return EPS
    return float(np.sqrt(np.mean(neg ** 2)))


def sortino_ratio(returns: np.ndarray, periods_per_year: int = ANNUALIZATION_TRADING_DAYS) -> float:
    if returns.size == 0:
        return float("nan")
    mu = np.mean(returns)
    dsd = _downside_std(returns)
    if dsd <= EPS:
        return float("inf") if mu > 0 else float("nan")
    return float(np.sqrt(periods_per_year) * mu / dsd)


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = ANNUALIZATION_TRADING_DAYS) -> float:
    if returns.size < 2:
        return float("nan")
    sd = _safe_std(returns)
    if sd <= EPS:
        return float("nan")
    return float(np.sqrt(periods_per_year) * np.mean(returns) / sd)


def profit_factor(pnl: np.ndarray) -> float:
    pos = pnl[pnl > 0].sum()
    neg = pnl[pnl < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / abs(neg))


# ---------------------------------------------------------------------------
# main panel
# ---------------------------------------------------------------------------
def compute_metric_panel(
    trades_df: pd.DataFrame,
    *,
    predicted_edge: Optional[np.ndarray] = None,
    ic_per_fold: Optional[List[float]] = None,
    periods_per_year: int = ANNUALIZATION_TRADING_DAYS,
) -> Dict:
    """Compute full 15-metric panel from per-trade DataFrame.

    Returns dict keyed by metric name. NaN for unavailable metrics
    (e.g. mfe/mae missing → mfe_capture_ratio = NaN).
    """
    if trades_df is None or len(trades_df) == 0:
        return _empty_panel()

    df = trades_df.copy()
    pnl_d = df["pnl_dollars"].to_numpy(dtype=float)
    pnl_t = df["pnl_ticks"].to_numpy(dtype=float)
    side = df["side"].to_numpy() if "side" in df.columns else np.full(len(df), "L")

    # per-day P&L for Sortino/Sharpe (returns = daily $ P&L)
    if "date" in df.columns:
        per_date = df.groupby("date")["pnl_dollars"].sum().to_numpy(dtype=float)
        unique_dates = df["date"].nunique()
    else:
        per_date = pnl_d.copy()
        unique_dates = len(pnl_d)

    # --- 1-3: risk-adjusted ---
    sortino = sortino_ratio(per_date, periods_per_year)
    sharpe = sharpe_ratio(per_date, periods_per_year)
    pf = profit_factor(pnl_d)

    # --- 4: WR overall + per-side ---
    is_win = pnl_t > 0
    wr_overall = float(np.mean(is_win)) if is_win.size else float("nan")
    long_mask = side == "L"
    short_mask = side == "S"
    wr_long = float(np.mean(is_win[long_mask])) if long_mask.any() else float("nan")
    wr_short = float(np.mean(is_win[short_mask])) if short_mask.any() else float("nan")

    # --- 5: MFE capture ratio ---
    if "mfe_ticks" in df.columns:
        mfe = df["mfe_ticks"].to_numpy(dtype=float)
        capture = pnl_t / np.maximum(np.abs(mfe), EPS)
        capture = np.clip(capture, -2.0, 2.0)
        mfe_capture_ratio = float(np.mean(capture))
    else:
        mfe_capture_ratio = float("nan")

    # --- 6: MAE drawdown distribution ---
    if "mae_ticks" in df.columns:
        mae = df["mae_ticks"].to_numpy(dtype=float)
        mae_p50 = float(np.percentile(mae, 50))
        mae_p95 = float(np.percentile(mae, 5))   # 5% worst (most negative)
        mae_p99 = float(np.percentile(mae, 1))
    else:
        mae_p50 = mae_p95 = mae_p99 = float("nan")

    # --- 7-8: per-date consistency + concentration (HC #254) ---
    if unique_dates > 0:
        pct_dates_positive = float(np.mean(per_date > 0))
        total_pnl = per_date.sum()
        if total_pnl > 0:
            single_date_concentration = float(per_date.max() / total_pnl)
        else:
            denom = np.abs(per_date).sum()
            single_date_concentration = (
                float(np.abs(per_date.min()) / denom) if denom > 0 else float("nan")
            )
    else:
        pct_dates_positive = single_date_concentration = float("nan")

    # --- 9: winner/loser dollar stats ---
    winners = pnl_d[pnl_d > 0]
    losers = pnl_d[pnl_d < 0]
    avg_winner_dollars = _safe_mean(winners)
    avg_loser_dollars = _safe_mean(losers)
    max_winner_dollars = float(winners.max()) if winners.size else float("nan")
    max_loser_dollars = float(losers.min()) if losers.size else float("nan")

    # --- 10: hold time winners vs losers ---
    if "entry_ts_ns" in df.columns and "exit_ts_ns" in df.columns:
        hold_sec = (df["exit_ts_ns"].to_numpy() - df["entry_ts_ns"].to_numpy()) / 1e9
        hold_winners = hold_sec[is_win]
        hold_losers = hold_sec[~is_win]
        hold_time_winners_sec = _safe_mean(hold_winners)
        hold_time_losers_sec = _safe_mean(hold_losers)
    else:
        hold_time_winners_sec = hold_time_losers_sec = float("nan")

    # --- 11: trade frequency ---
    n_trades = int(len(df))
    trades_per_date = float(n_trades / unique_dates) if unique_dates > 0 else float("nan")

    # --- 12: adverse selection rate ---
    if "adverse_5s_sign" in df.columns:
        adv = df["adverse_5s_sign"].to_numpy(dtype=float)
        # adverse if mid moved AGAINST the trade direction:
        # for L (long), adverse_sign < 0 hurts; for S (short), adverse_sign > 0 hurts
        favorable_dir = np.where(side == "L", 1, -1)
        adverse_mask = (adv * favorable_dir) < 0
        adverse_selection_rate = float(np.mean(adverse_mask))
    else:
        adverse_selection_rate = float("nan")

    # --- 13: fill rate ---
    if "fill_status" in df.columns:
        statuses = df["fill_status"].to_numpy()
        n_filled = int(np.sum(statuses == "filled"))
        n_placed = int(len(statuses))
        fill_rate = float(n_filled / n_placed) if n_placed > 0 else float("nan")
    else:
        fill_rate = float("nan")

    # --- 14: latency p50/p95/p99 (production only) ---
    if "latency_ns" in df.columns:
        lat_ms = df["latency_ns"].to_numpy(dtype=float) / 1e6
        latency_p50_ms = float(np.percentile(lat_ms, 50))
        latency_p95_ms = float(np.percentile(lat_ms, 95))
        latency_p99_ms = float(np.percentile(lat_ms, 99))
    else:
        latency_p50_ms = latency_p95_ms = latency_p99_ms = float("nan")

    # --- 15: concat IC + IC IR ---
    if predicted_edge is not None and len(predicted_edge) == len(pnl_t):
        try:
            ic_concat = float(spearmanr(predicted_edge, pnl_t).correlation)
        except Exception:
            ic_concat = float("nan")
    else:
        ic_concat = float("nan")
    if ic_per_fold is not None and len(ic_per_fold) > 1:
        ic_arr = np.array(ic_per_fold, dtype=float)
        ic_mean = float(np.nanmean(ic_arr))
        ic_std = float(np.nanstd(ic_arr, ddof=1))
        ic_ir = ic_mean / ic_std if ic_std > EPS else float("nan")
    else:
        ic_ir = float("nan")

    return {
        "sortino": sortino,
        "sharpe": sharpe,
        "profit_factor": pf,
        "wr_overall": wr_overall,
        "wr_long": wr_long,
        "wr_short": wr_short,
        "mfe_capture_ratio": mfe_capture_ratio,
        "mae_p50": mae_p50,
        "mae_p95": mae_p95,
        "mae_p99": mae_p99,
        "pct_dates_positive": pct_dates_positive,
        "single_date_concentration": single_date_concentration,
        "avg_winner_dollars": avg_winner_dollars,
        "avg_loser_dollars": avg_loser_dollars,
        "max_winner_dollars": max_winner_dollars,
        "max_loser_dollars": max_loser_dollars,
        "hold_time_winners_sec": hold_time_winners_sec,
        "hold_time_losers_sec": hold_time_losers_sec,
        "n_trades": n_trades,
        "trades_per_date": trades_per_date,
        "n_unique_dates": int(unique_dates),
        "total_pnl_dollars": float(pnl_d.sum()),
        "adverse_selection_rate": adverse_selection_rate,
        "fill_rate": fill_rate,
        "latency_p50_ms": latency_p50_ms,
        "latency_p95_ms": latency_p95_ms,
        "latency_p99_ms": latency_p99_ms,
        "concat_ic": ic_concat,
        "ic_ir": ic_ir,
    }


def _empty_panel() -> Dict:
    keys = [
        "sortino", "sharpe", "profit_factor",
        "wr_overall", "wr_long", "wr_short",
        "mfe_capture_ratio", "mae_p50", "mae_p95", "mae_p99",
        "pct_dates_positive", "single_date_concentration",
        "avg_winner_dollars", "avg_loser_dollars",
        "max_winner_dollars", "max_loser_dollars",
        "hold_time_winners_sec", "hold_time_losers_sec",
        "n_trades", "trades_per_date", "n_unique_dates", "total_pnl_dollars",
        "adverse_selection_rate", "fill_rate",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "concat_ic", "ic_ir",
    ]
    panel = {k: float("nan") for k in keys}
    panel["n_trades"] = 0
    panel["n_unique_dates"] = 0
    panel["total_pnl_dollars"] = 0.0
    return panel


# ---------------------------------------------------------------------------
# HC #254 gate
# ---------------------------------------------------------------------------
def passes_hc254(panel: Dict) -> Tuple[bool, List[str]]:
    """HC #254: pct_dates_positive >= 0.60, single_date_concentration <= 0.40, sortino > 0."""
    violations: List[str] = []
    pdp = panel.get("pct_dates_positive", float("nan"))
    sdc = panel.get("single_date_concentration", float("nan"))
    sor = panel.get("sortino", float("nan"))
    if not (pdp >= 0.60):
        violations.append(f"pct_dates_positive={pdp:.3f} < 0.60")
    if not (sdc <= 0.40):
        violations.append(f"single_date_concentration={sdc:.3f} > 0.40")
    if not (sor > 0):
        violations.append(f"sortino={sor:.3f} <= 0")
    return (len(violations) == 0), violations


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def _fmt(v, spec: str = ".4f") -> str:
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if np.isinf(v):
        return "∞" if v > 0 else "-∞"
    return format(v, spec)


def format_panel_table(panel: Dict, head_label: str = "") -> str:
    """Markdown table, Sortino at top. HC #69/#256 compliant."""
    pass_ok, violations = passes_hc254(panel)
    pass_str = "✅ PASS" if pass_ok else "❌ FAIL"
    title = f"### Metric Panel — {head_label}" if head_label else "### Metric Panel"
    lines = [
        title,
        f"**HC #254 gate:** {pass_str}" + ("" if pass_ok else f" ({'; '.join(violations)})"),
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| **SORTINO (headline)** | **{_fmt(panel['sortino'])}** |",
        f"| Sharpe | {_fmt(panel['sharpe'])} |",
        f"| Profit Factor | {_fmt(panel['profit_factor'])} |",
        f"| WR overall | {_fmt(panel['wr_overall'])} |",
        f"| WR long / short | {_fmt(panel['wr_long'])} / {_fmt(panel['wr_short'])} |",
        f"| MFE capture ratio | {_fmt(panel['mfe_capture_ratio'])} |",
        f"| MAE p50 / p95 / p99 (ticks) | {_fmt(panel['mae_p50'])} / {_fmt(panel['mae_p95'])} / {_fmt(panel['mae_p99'])} |",
        f"| % dates positive (≥0.60) | {_fmt(panel['pct_dates_positive'])} |",
        f"| Single-date concentration (≤0.40) | {_fmt(panel['single_date_concentration'])} |",
        f"| Avg winner $ / Avg loser $ | {_fmt(panel['avg_winner_dollars'], '.2f')} / {_fmt(panel['avg_loser_dollars'], '.2f')} |",
        f"| Max winner $ / Max loser $ | {_fmt(panel['max_winner_dollars'], '.2f')} / {_fmt(panel['max_loser_dollars'], '.2f')} |",
        f"| Hold winners / losers (s) | {_fmt(panel['hold_time_winners_sec'], '.2f')} / {_fmt(panel['hold_time_losers_sec'], '.2f')} |",
        f"| n trades | {_fmt(panel['n_trades'])} |",
        f"| Trades / date | {_fmt(panel['trades_per_date'], '.2f')} |",
        f"| n unique dates | {_fmt(panel['n_unique_dates'])} |",
        f"| Total P&L $ | {_fmt(panel['total_pnl_dollars'], '.2f')} |",
        f"| Adverse selection rate | {_fmt(panel['adverse_selection_rate'])} |",
        f"| Fill rate | {_fmt(panel['fill_rate'])} |",
        f"| Latency p50/p95/p99 (ms) | {_fmt(panel['latency_p50_ms'], '.2f')} / {_fmt(panel['latency_p95_ms'], '.2f')} / {_fmt(panel['latency_p99_ms'], '.2f')} |",
        f"| Concat IC | {_fmt(panel['concat_ic'])} |",
        f"| IC IR | {_fmt(panel['ic_ir'])} |",
    ]
    return "\n".join(lines)


def compare_panels(panels: Dict[str, Dict]) -> str:
    """Side-by-side comparison table for multiple heads."""
    if not panels:
        return ""
    heads = list(panels.keys())
    rows = [
        ("**SORTINO**", "sortino", ".4f"),
        ("Sharpe", "sharpe", ".4f"),
        ("Profit Factor", "profit_factor", ".4f"),
        ("WR overall", "wr_overall", ".4f"),
        ("WR long", "wr_long", ".4f"),
        ("WR short", "wr_short", ".4f"),
        ("MFE capture", "mfe_capture_ratio", ".4f"),
        ("% dates positive", "pct_dates_positive", ".4f"),
        ("Single-date concentration", "single_date_concentration", ".4f"),
        ("n trades", "n_trades", "d"),
        ("n dates", "n_unique_dates", "d"),
        ("Total P&L $", "total_pnl_dollars", ".2f"),
        ("Concat IC", "concat_ic", ".4f"),
    ]
    header = "| Metric | " + " | ".join(heads) + " |"
    sep = "|---|" + "|".join(["---:"] * len(heads)) + "|"
    lines = ["### Head Comparison (HC #256)", header, sep]
    for label, key, spec in rows:
        cells = []
        for h in heads:
            v = panels[h].get(key, float("nan"))
            cells.append(_fmt(v, spec) if spec != "d" else _fmt(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    # gate row
    gate_cells = []
    for h in heads:
        ok, _ = passes_hc254(panels[h])
        gate_cells.append("✅" if ok else "❌")
    lines.append("| HC #254 gate | " + " | ".join(gate_cells) + " |")
    return "\n".join(lines)
