#!/usr/bin/env python3
"""
Optuna-Based Parameter Optimization for CNN WF Trading Strategy
================================================================
Loads all sweep results (local + remote Jupiter/Saturn), builds a high-density
Gaussian-process-like surrogate via Optuna TPE, and finds optimal entry/exit
thresholds, vol filter, TP/SL to maximize Sharpe and profit factor.

Modes:
    grid-search   - Use existing aggregated sweep data as the objective
                    (fast, no fill_sim needed, works on 170K+ results)
    live-sim      - Run actual fill_sim_cli.exe for each trial (slow, accurate)

Usage:
    python alpha_discovery/deep_models/optuna_param_optimizer.py --mode grid-search
    python alpha_discovery/deep_models/optuna_param_optimizer.py --mode grid-search --trials 2000
    python alpha_discovery/deep_models/optuna_param_optimizer.py --mode live-sim --trials 500
    python alpha_discovery/deep_models/optuna_param_optimizer.py --mode grid-search --source all
    python alpha_discovery/deep_models/optuna_param_optimizer.py --show-importance

Imports:
    from alpha_discovery.deep_models.optuna_param_optimizer import (
        load_all_sweep_data, run_grid_search_study, get_pareto_front
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent

# ── SSH utility (teleclaude-main) ──────────────────────────────────────────────
# Insert the teleclaude-main directory onto sys.path so we can import run_on.
_TELECLAUDE_ROOT = LVL3_ROOT.parent / "teleclaude-main"
if str(_TELECLAUDE_ROOT) not in sys.path:
    sys.path.insert(0, str(_TELECLAUDE_ROOT))
try:
    from utils.ssh_exec import run_on as _ssh_run_on  # noqa: E402
    _SSH_AVAILABLE = True
except ImportError:
    _ssh_run_on = None
    _SSH_AVAILABLE = False

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "deep_models" / "results"
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli.exe"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR_WF = LVL3_ROOT / "data" / "processed" / "cnn_wf_exec_sweep_predictions"
PRED_DIR_STACKED = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
SIM_OUT_DIR = LVL3_ROOT / "data" / "processed" / "optuna_sim_results"

# Remote server paths
JUPITER_ROOT = "/home/jupiter/Lvl3Quant"
SATURN_ROOT = "/home/saturn/Lvl3Quant"
JUPITER_V2_RESULTS = f"{JUPITER_ROOT}/data/processed/cnn_wf_stacked_v2_results"
SATURN_V2_RESULTS = f"{SATURN_ROOT}/data/processed/cnn_wf_stacked_v2_results"

TICK_VALUE = 12.50  # $12.50 per tick for ES

# ── Logging ───────────────────────────────────────────────────────────────────
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log = logging.getLogger("optuna_optimizer")
log.setLevel(logging.INFO)
_fh = logging.FileHandler(
    str(RESULTS_DIR / f"optuna_optimization_{_ts}.log"),
    mode="w",
    encoding="utf-8",
)
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
log.addHandler(_ch)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ==============================================================================
# DATA LOADING — All local aggregated sweep results
# ==============================================================================


def _normalize_record(rec: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    """
    Normalize a record from any sweep source into a canonical dict.

    Canonical fields:
        config          - unique string key
        entry_method    - str (smooth, book, ema, mom, combo1..5, idea1..5, etc.)
        exit_method     - str (bookExit, smoothExit, hold, emaExit, predstdExit, etc.)
        conv_threshold  - float (conviction multiplier, 0.3–3.0)
        exit_threshold  - float (exit signal threshold, 0.0–1.0)
        vol_percentile  - int (vol filter, 0/50/70/80/90)
        tp_ticks        - int or None
        sl_ticks        - int or None
        hold_min        - float (hold timeout in minutes, 5–60)
        chase           - bool
        n_days          - int
        sharpe          - float (annualized daily Sharpe)
        total_pnl       - float ($)
        annualized_pnl  - float ($)
        n_trades        - int
        win_rate        - float
        fill_rate       - float
        avg_trade_pnl   - float ($)
        max_dd          - float ($)
        profit_factor   - float
        source          - str
    """
    try:
        out: Dict[str, Any] = {"source": source}

        # ── Stacked Exit (v1 aggregated) ──
        if source == "stacked_exit":
            config = rec.get("config", "")
            entry = rec.get("entry", "smooth")
            exit_method = rec.get("exit_method", "bookExit")
            conv_str = rec.get("conv", "conv2.5").replace("conv", "")
            vol_str = rec.get("vol", "vol70").replace("vol", "")
            tp_str = rec.get("tp", "tpN").replace("tp", "")
            sl_str = rec.get("sl", "slN").replace("sl", "")

            conv = float(conv_str) if conv_str else 2.5
            vol = int(vol_str) if vol_str.isdigit() else 70
            tp = int(tp_str) if tp_str.isdigit() else None
            sl = int(sl_str) if sl_str.isdigit() else None

            # Infer exit threshold from exit_method name
            exit_thr = 0.0
            if "predstdExit" in exit_method:
                exit_thr = 0.1
            elif "bookExit" in exit_method:
                exit_thr = 0.0

            out.update(
                config=config,
                entry_method=entry,
                exit_method=exit_method,
                conv_threshold=conv,
                exit_threshold=exit_thr,
                vol_percentile=vol,
                tp_ticks=tp,
                sl_ticks=sl,
                hold_min=60.0,  # stacked exit uses 60-min safety timeout
                chase=True,
                n_days=rec.get("n_days", 0),
                sharpe=rec.get("sharpe", 0.0),
                total_pnl=rec.get("total_pnl", 0.0),
                annualized_pnl=rec.get("annualized_pnl", rec.get("annualized", 0.0)),
                n_trades=rec.get("total_trades", rec.get("n_trades", 0)),
                win_rate=rec.get("win_rate", 0.5),
                fill_rate=rec.get("fill_rate", 0.0),
                avg_trade_pnl=rec.get("avg_trade_pnl", 0.0),
                max_dd=rec.get("max_dd", 0.0),
            )

        # ── Hold Time Sweep ──
        elif source == "hold_sweep":
            label = rec.get("label", "")
            # Format: hold{X}m_conv{Y}[_tp{Z}]_[passive|chase]
            # Parse hold minutes
            m = re.search(r"hold(\d+)m", label)
            hold_min = float(m.group(1)) if m else 30.0
            # Parse conv threshold
            m = re.search(r"conv(\d+)", label)
            conv = float(m.group(1)) / 10.0 if m else 2.5
            # Parse TP
            m = re.search(r"tp(\d+)", label)
            tp = int(m.group(1)) if m else None
            # Chase?
            chase = "chase" in label
            # Vol gate
            vg = rec.get("vg", 0)
            vol = int(str(vg).replace("vol", "")) if str(vg).replace("vol", "").isdigit() else 0

            out.update(
                config=f"{label}_vg{vg}",
                entry_method="smooth",
                exit_method="hold_timeout",
                conv_threshold=conv,
                exit_threshold=0.0,
                vol_percentile=vol,
                tp_ticks=tp,
                sl_ticks=None,
                hold_min=hold_min,
                chase=chase,
                n_days=rec.get("n_days", 0),
                sharpe=rec.get("sharpe", 0.0),
                total_pnl=rec.get("pnl", 0.0),
                annualized_pnl=rec.get("annual", 0.0),
                n_trades=rec.get("trades", 0),
                win_rate=rec.get("wr", 0.5),
                fill_rate=rec.get("fill", 0.0),
                avg_trade_pnl=(rec.get("pnl", 0) / rec.get("trades", 1)) if rec.get("trades", 0) > 0 else 0.0,
                max_dd=0.0,
            )

        # ── Entry/Exit Matrix ──
        elif source == "entry_exit_matrix":
            config = rec.get("config", "")
            entry_method = rec.get("entry_method", "smooth")
            entry_param = rec.get("entry_param", "2.0")
            exit_method = rec.get("exit_method", "smoothExit")
            exit_param = rec.get("exit_param", "0.0")
            vol = int(rec.get("vol_gate", 70))
            sim_type = rec.get("sim_type", "")
            # TP from sim_type
            m = re.search(r"tp(\d+)", sim_type)
            tp = int(m.group(1)) if m else None
            m = re.search(r"sl(\d+)", sim_type)
            sl = int(m.group(1)) if m else None

            try:
                conv = float(entry_param)
            except (ValueError, TypeError):
                conv = 2.0
            try:
                exit_thr = float(exit_param)
            except (ValueError, TypeError):
                exit_thr = 0.0

            out.update(
                config=config,
                entry_method=entry_method,
                exit_method=exit_method,
                conv_threshold=conv,
                exit_threshold=exit_thr,
                vol_percentile=vol,
                tp_ticks=tp,
                sl_ticks=sl,
                hold_min=30.0,
                chase=True,
                n_days=rec.get("n_days", 0),
                sharpe=rec.get("sharpe", 0.0),
                total_pnl=rec.get("total_pnl", 0.0),
                annualized_pnl=rec.get("annualized_pnl", 0.0),
                n_trades=rec.get("total_trades", rec.get("n_trades", 0)),
                win_rate=rec.get("win_rate", 0.5),
                fill_rate=rec.get("fill_rate", 0.0),
                avg_trade_pnl=rec.get("mean_daily_pnl", 0.0) / max(rec.get("trades_per_day", 1), 0.1),
                max_dd=0.0,
            )

        # ── Novel Ideas ──
        elif source == "novel_ideas":
            config = rec.get("config", "")
            # Parse from config string: idea{N}_{type}_{params}
            m_conv = re.search(r"conv([0-9.]+)", config)
            m_vol = re.search(r"vol(\d+)", config)
            m_tp = re.search(r"tp(\d+)", config)
            m_hold = re.search(r"hold(\d+)m", config)
            m_imb = re.search(r"imb([0-9.]+)", config)
            m_thr = re.search(r"thr([0-9.]+)", config)
            m_max = re.search(r"max([0-9.]+)", config)

            conv = float(m_conv.group(1)) if m_conv else 2.0
            vol = int(m_vol.group(1)) if m_vol else 70
            tp = int(m_tp.group(1)) if m_tp else None
            hold_min = float(m_hold.group(1)) if m_hold else 30.0
            chase = "chase" in config

            # Determine idea type and entry/exit method
            idea_type = rec.get("idea_type", "unknown")
            if "idea1" in config:
                entry_method = "momentum"
                exit_method = "hold_timeout"
            elif "idea2" in config:
                entry_method = "book_imbalance"
                exit_method = "hold_timeout"
            elif "idea3" in config:
                entry_method = "predstd_filter"
                exit_method = "hold_timeout"
            elif "idea4" in config:
                entry_method = "ema_cross"
                exit_method = "hold_timeout"
            elif "idea5" in config:
                entry_method = "momentum_reversal"
                exit_method = "hold_timeout"
            else:
                entry_method = idea_type
                exit_method = "hold_timeout"

            out.update(
                config=config,
                entry_method=entry_method,
                exit_method=exit_method,
                conv_threshold=conv,
                exit_threshold=float(m_max.group(1)) if m_max else (float(m_thr.group(1)) if m_thr else 0.0),
                vol_percentile=vol,
                tp_ticks=tp,
                sl_ticks=None,
                hold_min=hold_min,
                chase=chase,
                n_days=rec.get("n_days", 0),
                sharpe=rec.get("daily_sharpe", 0.0),
                total_pnl=rec.get("total_pnl", 0.0),
                annualized_pnl=rec.get("annualized_pnl", 0.0),
                n_trades=rec.get("total_trades", 0),
                win_rate=rec.get("win_rate", 0.5),
                fill_rate=rec.get("fill_rate", 0.0),
                avg_trade_pnl=rec.get("mean_daily_pnl", 0.0) / max(rec.get("trades_per_day", 1), 0.1),
                max_dd=0.0,
            )

        # ── Combo Sweep ──
        elif source == "combo_sweep":
            config = rec.get("config", "")
            m_conv = re.search(r"conv([0-9.]+)", config)
            m_vol = re.search(r"vol(\d+)", config)
            m_tp = re.search(r"tp(\d+)", config)
            m_hold = re.search(r"hold(\d+)m", config)
            m_athr = re.search(r"athr([0-9.]+)", config)
            combo = rec.get("combo", "combo1")

            conv = float(m_conv.group(1)) if m_conv else 2.5
            vol = int(m_vol.group(1)) if m_vol else 70
            tp = int(m_tp.group(1)) if m_tp else None
            hold_min = float(m_hold.group(1)) if m_hold else 30.0
            alpha_thr = float(m_athr.group(1)) if m_athr else 0.0

            out.update(
                config=config,
                entry_method=combo,
                exit_method="hold_timeout_tp",
                conv_threshold=conv,
                exit_threshold=alpha_thr,
                vol_percentile=vol,
                tp_ticks=tp,
                sl_ticks=None,
                hold_min=hold_min,
                chase=True,
                n_days=rec.get("n_days", 0),
                sharpe=rec.get("sharpe", 0.0),
                total_pnl=rec.get("total_pnl", 0.0),
                annualized_pnl=rec.get("annualized", 0.0),
                n_trades=rec.get("n_trades", 0),
                win_rate=rec.get("win_rate", 0.5),
                fill_rate=rec.get("fill_rate", 0.0),
                avg_trade_pnl=rec.get("avg_trade_pnl", 0.0),
                max_dd=rec.get("max_dd", 0.0),
            )

        else:
            return None

        # ── Compute profit_factor ──
        wins = out.get("win_rate", 0.5) * out.get("n_trades", 1)
        losses = out.get("n_trades", 1) - wins
        avg_pnl = out.get("avg_trade_pnl", 0.0)
        if losses > 0 and avg_pnl > 0:
            # Approximate: assume win_pnl and loss_pnl symmetric around avg
            gross_profit = wins * avg_pnl * 2  # rough upper bound
            gross_loss = losses * avg_pnl * 0.5
            out["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else 1.0
        else:
            out["profit_factor"] = 1.0

        # Filter: must have enough data
        if out.get("n_days", 0) < 15 or out.get("n_trades", 0) < 5:
            return None

        return out

    except Exception as exc:
        log.debug(f"Error normalizing record from {source}: {exc}")
        return None


def load_all_sweep_data(source: str = "all") -> List[Dict[str, Any]]:
    """
    Load and normalize all available sweep results from local aggregated files.

    Args:
        source: One of "all", "stacked_exit", "hold_sweep", "entry_exit_matrix",
                "novel_ideas", "combo_sweep"

    Returns:
        List of normalized dicts with canonical fields.
    """
    records: List[Dict[str, Any]] = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    source_map = {
        "stacked_exit": _load_stacked_exit,
        "hold_sweep": _load_hold_sweep,
        "entry_exit_matrix": _load_entry_exit_matrix,
        "novel_ideas": _load_novel_ideas,
        "combo_sweep": _load_combo_sweep,
    }

    sources_to_load = list(source_map.keys()) if source == "all" else [source]

    for src_name in sources_to_load:
        if src_name not in source_map:
            log.warning(f"Unknown source: {src_name}")
            continue
        raw_recs = source_map[src_name]()
        normalized = 0
        for rec in raw_recs:
            norm = _normalize_record(rec, src_name)
            if norm is not None:
                records.append(norm)
                normalized += 1
        log.info(f"Loaded {normalized}/{len(raw_recs)} records from {src_name}")

    log.info(f"Total normalized records: {len(records)}")
    return records


def _load_stacked_exit() -> List[Dict[str, Any]]:
    path = RESULTS_DIR / "stacked_exit_aggregated.json"
    if not path.exists():
        log.warning(f"Not found: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("all_results", [])


def _load_hold_sweep() -> List[Dict[str, Any]]:
    path = RESULTS_DIR / "hold_sweep_aggregated.json"
    if not path.exists():
        log.warning(f"Not found: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("summaries", [])


def _load_entry_exit_matrix() -> List[Dict[str, Any]]:
    path = RESULTS_DIR / "entry_exit_matrix_aggregated.json"
    if not path.exists():
        log.warning(f"Not found: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("all_configs_sorted", [])


def _load_novel_ideas() -> List[Dict[str, Any]]:
    path = RESULTS_DIR / "novel_ideas_aggregated.json"
    if not path.exists():
        log.warning(f"Not found: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("all_configs", [])


def _load_combo_sweep() -> List[Dict[str, Any]]:
    # Find latest combo sweep results file
    combo_files = sorted(RESULTS_DIR.glob("combo_sweep_results_*.json"), reverse=True)
    if not combo_files:
        log.warning("No combo_sweep_results_*.json found")
        return []
    with open(combo_files[0]) as f:
        data = json.load(f)
    return data.get("all_summaries", [])


# ==============================================================================
# SSH AGGREGATION — Pull results from Jupiter/Saturn
# ==============================================================================


def aggregate_from_remote(result_dir_remote: str, server_name: str) -> List[Dict[str, Any]]:
    """
    Pull a sample of result files from remote server and aggregate them.
    Returns list of per-day raw result dicts.

    This reads the individual fill_sim output JSONs remotely and returns
    minimal aggregated data to avoid slow full transfers.

    Uses the pooled SSH utility (utils.ssh_exec.run_on) instead of raw paramiko.
    """
    if not _SSH_AVAILABLE:
        log.error("SSH utility not available — cannot load remote results. "
                  "Ensure teleclaude-main is at the expected path.")
        return []

    log.info(f"Aggregating from {server_name}:{result_dir_remote} ...")

    # Count files on remote
    count_result = _ssh_run_on(
        server_name.lower(),
        f'ls {result_dir_remote}/*.json 2>/dev/null | wc -l',
        timeout=30,
    )
    if not count_result.get("success"):
        log.error(f"{server_name}: file count failed: {count_result.get('error', count_result.get('stderr', ''))}")
        return []

    n_files_str = count_result["stdout"].strip()
    n_files = int(n_files_str) if n_files_str.isdigit() else 0
    log.info(f"{server_name}: {n_files} result files found")

    if n_files == 0:
        return []

    # Run a remote aggregation script inline to avoid downloading millions of files.
    # Encode as base64 to avoid shell quoting issues.
    import base64

    remote_agg_script = r"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

result_dir = Path("{result_dir}")
files = sorted(result_dir.glob("*.json"))
print(f"Found {{len(files)}} files", file=sys.stderr, flush=True)

configs = defaultdict(lambda: {{"daily_pnl": [], "trades": [], "all_trades": []}})
DATE_RE = re.compile(r'_(\d{{4}}-\d{{2}}-\d{{2}})\.json$')

for fpath in files:
    m = DATE_RE.search(fpath.name)
    if not m:
        continue
    config = fpath.name[:m.start()]
    try:
        with open(fpath) as f:
            d = json.load(f)
        configs[config]["daily_pnl"].append(d.get("total_pnl_dollars", 0))
        configs[config]["trades"].append(d.get("total_trades", 0))
        for t in d.get("trades", []):
            configs[config]["all_trades"].append(t.get("pnl_dollars", 0))
    except Exception:
        pass

results = []
for cfg, cd in configs.items():
    n_days = len(cd["daily_pnl"])
    if n_days < 15:
        continue
    total_pnl = sum(cd["daily_pnl"])
    total_trades = sum(cd["trades"])
    if total_trades == 0:
        continue
    mean_d = total_pnl / n_days
    std_d = (sum((p - mean_d) ** 2 for p in cd["daily_pnl"]) / (n_days - 1)) ** 0.5 if n_days > 1 else 1e-9
    sharpe = (mean_d / max(std_d, 1e-9)) * (252 ** 0.5)
    wins = sum(1 for p in cd["all_trades"] if p > 0)
    wr = wins / len(cd["all_trades"]) if cd["all_trades"] else 0
    avg_pnl = sum(cd["all_trades"]) / len(cd["all_trades"]) if cd["all_trades"] else 0
    results.append({{
        "config": cfg,
        "n_days": n_days,
        "sharpe": round(sharpe, 4),
        "total_pnl": round(total_pnl, 2),
        "annualized_pnl": round(total_pnl / n_days * 252, 2),
        "n_trades": total_trades,
        "win_rate": round(wr, 4),
        "avg_trade_pnl": round(avg_pnl, 2),
    }})

print(json.dumps(results))
""".format(result_dir=result_dir_remote)

    script_b64 = base64.b64encode(remote_agg_script.encode()).decode()
    cmd = f'echo {script_b64} | base64 -d | python3 -'

    agg_result = _ssh_run_on(server_name.lower(), cmd, timeout=300)
    output = agg_result.get("stdout", "")
    err = agg_result.get("stderr", "")

    if not agg_result.get("success"):
        log.warning(f"{server_name}: aggregation command returned non-zero exit. stderr: {err[:300]}")

    if err:
        log.debug(f"{server_name} stderr: {err[:500]}")

    # Find the JSON line (last line that starts with [)
    lines = output.strip().split("\n")
    json_line = None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("["):
            json_line = line
            break

    if json_line is None:
        log.warning(f"{server_name}: no JSON output found. Output: {output[:300]}")
        return []

    try:
        remote_results = json.loads(json_line)
        log.info(f"{server_name}: aggregated {len(remote_results)} configs")
        return remote_results
    except json.JSONDecodeError as e:
        log.error(f"{server_name}: JSON parse error: {e}")
        return []


def load_remote_results() -> List[Dict[str, Any]]:
    """
    SSH to Jupiter and Saturn, aggregate their fill_sim results,
    and return normalized records. Falls back gracefully if unreachable.

    Uses the pooled SSH utility (utils.ssh_exec.run_on) — no raw paramiko.
    """
    if not _SSH_AVAILABLE:
        log.error("SSH utility not available — cannot load remote results.")
        return []

    records = []

    def _parse_remote_config(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a remotely-aggregated config summary into normalized form."""
        config = rec.get("config", "")
        # Config format from stacked_v2: {combo_label}_{tp_str}_{sl_str}
        # combo_label: {entry}_{exit}_conv{X}_vol{Y}
        # e.g.: smooth_smoothExit_conv2.5_vol70_tpN_slN
        m_conv = re.search(r"conv([0-9.]+)", config)
        m_vol = re.search(r"vol(\d+)", config)
        m_tp = re.search(r"tp([0-9N]+)", config)
        m_sl = re.search(r"sl([0-9N]+)", config)

        # Entry/exit pair
        parts = config.split("_")
        entry = parts[0] if parts else "smooth"
        exit_method = parts[1] if len(parts) > 1 else "bookExit"

        conv = float(m_conv.group(1)) if m_conv else 2.5
        vol = int(m_vol.group(1)) if m_vol else 70
        tp_s = m_tp.group(1) if m_tp else "N"
        sl_s = m_sl.group(1) if m_sl else "N"
        tp = int(tp_s) if tp_s.isdigit() else None
        sl = int(sl_s) if sl_s.isdigit() else None

        result = {
            "source": "remote_stacked_v2",
            "config": config,
            "entry_method": entry,
            "exit_method": exit_method,
            "conv_threshold": conv,
            "exit_threshold": 0.0,
            "vol_percentile": vol,
            "tp_ticks": tp,
            "sl_ticks": sl,
            "hold_min": 60.0,
            "chase": True,
            "n_days": rec.get("n_days", 0),
            "sharpe": rec.get("sharpe", 0.0),
            "total_pnl": rec.get("total_pnl", 0.0),
            "annualized_pnl": rec.get("annualized_pnl", 0.0),
            "n_trades": rec.get("n_trades", 0),
            "win_rate": rec.get("win_rate", 0.5),
            "fill_rate": 0.08,  # typical WF fill rate
            "avg_trade_pnl": rec.get("avg_trade_pnl", 0.0),
            "max_dd": 0.0,
            "profit_factor": 1.0,
        }
        if result["n_days"] < 15 or result["n_trades"] < 5:
            return None
        return result

    # Iterate over servers using run_on — no SSH client objects needed.
    for server_name, remote_dir in [
        ("jupiter", JUPITER_V2_RESULTS),
        ("saturn", SATURN_V2_RESULTS),
    ]:
        raw = aggregate_from_remote(remote_dir, server_name)
        for rec in raw:
            norm = _parse_remote_config(rec)
            if norm is not None:
                records.append(norm)

    log.info(f"Remote results: {len(records)} normalized records")
    return records


# ==============================================================================
# OBJECTIVE SCORING
# ==============================================================================


def compute_score(record: Dict[str, Any], weights: Dict[str, float] = None) -> float:
    """
    Compute a combined optimization score for a sweep result.

    Default: 0.6 * sharpe + 0.4 * log1p(profit_factor)
    Penalizes:
      - low trade count (< 20 trades gets penalty)
      - high drawdown relative to P&L
      - very low fill rate (< 1% = degenerate)
    """
    if weights is None:
        weights = {"sharpe": 0.6, "log_pf": 0.4}

    sharpe = record.get("sharpe", 0.0)
    pf = max(record.get("profit_factor", 1.0), 0.01)
    n_trades = record.get("n_trades", 0)
    n_days = record.get("n_days", 0)
    fill_rate = record.get("fill_rate", 0.0)
    total_pnl = record.get("total_pnl", 0.0)
    max_dd = record.get("max_dd", 0.0)

    # Base score
    score = weights.get("sharpe", 0.6) * sharpe + weights.get("log_pf", 0.4) * math.log1p(pf)

    # Penalty: low trade count (need at least 30 trades for statistical significance)
    if n_trades < 10:
        score -= 2.0
    elif n_trades < 30:
        score -= 0.5 * (1.0 - n_trades / 30.0)

    # Penalty: very low fill rate suggests degenerate config
    if fill_rate > 0 and fill_rate < 0.01:
        score -= 1.0

    # Penalty: max drawdown > 2x total_pnl (bad risk/reward)
    if max_dd > 0 and total_pnl > 0 and max_dd > 2.0 * total_pnl:
        score -= 0.5

    # Reward: more days = more reliable
    if n_days >= 40:
        score += 0.1
    elif n_days < 20:
        score -= 0.2

    return score


def compute_objectives(record: Dict[str, Any]) -> Tuple[float, float]:
    """
    Return (sharpe, profit_factor) tuple for multi-objective optimization.
    Both should be maximized.
    """
    sharpe = record.get("sharpe", 0.0)
    pf = record.get("profit_factor", 1.0)
    return sharpe, pf


# ==============================================================================
# GRID SEARCH MODE — Surrogate from existing sweep data
# ==============================================================================

# Categorical encodings for entry/exit methods
ENTRY_METHODS = [
    "smooth", "book", "ema", "mom",
    "combo1", "combo2", "combo3", "combo4", "combo5",
    "momentum", "book_imbalance", "predstd_filter", "ema_cross", "momentum_reversal",
    "idea1", "idea2", "idea3", "idea4", "idea5",
]
EXIT_METHODS = [
    "bookExit", "smoothExit", "hold_timeout", "hold_timeout_tp",
    "emaExit", "predstdExit", "ema_bookExit", "mom_emaExit",
]


def _encode_params(record: Dict[str, Any]) -> Dict[str, float]:
    """
    Encode a normalized record's parameters into numeric features for distance search.
    Returns a feature dict for interpolation.
    """
    entry = record.get("entry_method", "smooth")
    exit_ = record.get("exit_method", "bookExit")
    # Simple ordinal encoding: index in list, normalized to [0, 1]
    entry_idx = ENTRY_METHODS.index(entry) / len(ENTRY_METHODS) if entry in ENTRY_METHODS else 0.5
    exit_idx = EXIT_METHODS.index(exit_) / len(EXIT_METHODS) if exit_ in EXIT_METHODS else 0.5

    return {
        "conv_threshold": record.get("conv_threshold", 2.5) / 3.0,  # normalize to ~[0,1]
        "exit_threshold": record.get("exit_threshold", 0.0),
        "vol_percentile": record.get("vol_percentile", 70) / 90.0,
        "tp_ticks": (record.get("tp_ticks") or 0) / 30.0,
        "sl_ticks": (record.get("sl_ticks") or 0) / 50.0,
        "hold_min": record.get("hold_min", 30.0) / 60.0,
        "chase": float(record.get("chase", True)),
        "entry_method": entry_idx,
        "exit_method": exit_idx,
    }


def _interpolate_objective(
    trial_params: Dict[str, float],
    data_features: np.ndarray,
    data_scores: np.ndarray,
    k: int = 5,
    distance_scale: float = 0.1,
) -> float:
    """
    Estimate objective for a new point by inverse-distance-weighted
    interpolation from the k nearest grid neighbors.

    Args:
        trial_params: dict of normalized param values (output of _encode_params-like logic)
        data_features: (N, D) array of encoded feature vectors from sweep data
        data_scores: (N,) array of combined scores
        k: number of nearest neighbors
        distance_scale: sigma for Gaussian kernel weighting

    Returns:
        Estimated score (float)
    """
    trial_vec = np.array(list(trial_params.values()), dtype=float)

    # Euclidean distance to all grid points
    dists = np.linalg.norm(data_features - trial_vec[np.newaxis, :], axis=1)

    # k-nearest neighbors
    knn_idx = np.argsort(dists)[:k]
    knn_dists = dists[knn_idx]
    knn_scores = data_scores[knn_idx]

    # Exact match — return directly
    if knn_dists[0] < 1e-8:
        return float(knn_scores[0])

    # Gaussian kernel weights
    weights = np.exp(-(knn_dists**2) / (2 * distance_scale**2))
    weights_sum = weights.sum()

    if weights_sum < 1e-12:
        return float(knn_scores[0])

    return float(np.dot(weights, knn_scores) / weights_sum)


def build_data_arrays(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode all records into feature matrix and score/objective arrays.

    Returns:
        features: (N, D) float array
        scores: (N,) combined score array
        objectives: (N, 2) [sharpe, profit_factor]
    """
    features = []
    scores = []
    objectives = []

    for rec in records:
        feat = _encode_params(rec)
        features.append(list(feat.values()))
        scores.append(compute_score(rec))
        objectives.append(list(compute_objectives(rec)))

    return (
        np.array(features, dtype=float),
        np.array(scores, dtype=float),
        np.array(objectives, dtype=float),
    )


def run_grid_search_study(
    records: List[Dict[str, Any]],
    n_trials: int = 1000,
    study_name: str = "cnn_param_opt",
    n_objectives: int = 1,
) -> Any:
    """
    Run Optuna study using existing sweep data as the objective.

    For each trial, Optuna suggests continuous parameters. We find the
    nearest grid point(s) and use IDW interpolation to estimate the score.

    Args:
        records: Normalized sweep data from load_all_sweep_data()
        n_trials: Number of Optuna trials
        study_name: Optuna study name (for storage)
        n_objectives: 1 = single-objective (combined score), 2 = multi-objective (Sharpe + PF)

    Returns:
        optuna.Study object
    """
    try:
        import optuna
    except ImportError:
        log.error("optuna not installed. Run: pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if len(records) == 0:
        log.error("No records to optimize over. Load data first.")
        return None

    # Precompute data arrays
    features, scores, objectives = build_data_arrays(records)
    log.info(f"Built data arrays: {features.shape[0]} records, {features.shape[1]} features")
    log.info(f"Score range: {scores.min():.3f} to {scores.max():.3f}")
    log.info(f"Sharpe range: {objectives[:,0].min():.2f} to {objectives[:,0].max():.2f}")

    # Get unique categorical values from data for valid suggestions
    unique_entries = sorted(set(r.get("entry_method", "smooth") for r in records))
    unique_exits = sorted(set(r.get("exit_method", "bookExit") for r in records))
    # Sort with None first for TP/SL (None = no limit/stop)
    unique_tp = sorted(set(r.get("tp_ticks") for r in records), key=lambda x: (x is not None, x or 0))
    unique_sl = sorted(set(r.get("sl_ticks") for r in records), key=lambda x: (x is not None, x or 0))
    unique_hold = sorted(set(r.get("hold_min", 30) for r in records))

    log.info(f"Entry methods in data: {unique_entries}")
    log.info(f"Exit methods in data: {unique_exits}")
    log.info(f"TP values: {unique_tp}")
    log.info(f"SL values: {unique_sl}")
    log.info(f"Hold times (min): {unique_hold}")

    def objective_single(trial):
        """Single-objective: maximize combined score."""
        params = _suggest_params(trial, unique_entries, unique_exits, unique_tp, unique_sl, unique_hold)
        trial_feat = _encode_trial_params(params)
        score = _interpolate_objective(trial_feat, features, scores)
        return score

    def objective_multi(trial):
        """Multi-objective: maximize [sharpe, profit_factor]."""
        params = _suggest_params(trial, unique_entries, unique_exits, unique_tp, unique_sl, unique_hold)
        trial_feat = _encode_trial_params(params)
        sh = _interpolate_objective(trial_feat, features, objectives[:, 0])
        pf = _interpolate_objective(trial_feat, features, objectives[:, 1])
        return sh, pf

    # ── Create study ──
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=max(20, n_trials // 10),
        seed=42,
    )

    if n_objectives == 2:
        study = optuna.create_study(
            study_name=study_name,
            directions=["maximize", "maximize"],
            sampler=sampler,
        )
        obj_fn = objective_multi
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
        )
        obj_fn = objective_single

    log.info(f"Starting Optuna study: {n_trials} trials, {n_objectives}-objective")
    t0 = time.time()

    study.optimize(
        obj_fn,
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[_progress_callback(n_trials, t0)],
    )

    elapsed = time.time() - t0
    log.info(f"Study complete in {elapsed:.1f}s ({n_trials/elapsed:.1f} trials/s)")

    return study


def _suggest_params(
    trial,
    unique_entries: List,
    unique_exits: List,
    unique_tp: List,
    unique_sl: List,
    unique_hold: List,
) -> Dict[str, Any]:
    """Suggest parameters for a trial using Optuna."""
    # Continuous parameters
    conv_threshold = trial.suggest_float("conv_threshold", 0.3, 3.0)
    exit_threshold = trial.suggest_float("exit_threshold", 0.0, 1.0)
    vol_percentile = trial.suggest_float("vol_percentile", 0.0, 90.0)
    hold_min = trial.suggest_float("hold_min", 5.0, 60.0)

    # Categorical parameters
    entry_method = trial.suggest_categorical("entry_method", unique_entries)
    exit_method = trial.suggest_categorical("exit_method", unique_exits)
    chase = trial.suggest_categorical("chase", [True, False])

    # TP/SL — treat None as 0 for continuous suggestion, then round to nearest grid
    tp_raw = trial.suggest_float("tp_raw", 0.0, 30.0)  # 0 = no TP
    sl_raw = trial.suggest_float("sl_raw", 0.0, 50.0)  # 0 = no SL
    tp_ticks = None if tp_raw < 2.5 else int(round(tp_raw))
    sl_ticks = None if sl_raw < 5.0 else int(round(sl_raw))

    return {
        "conv_threshold": conv_threshold,
        "exit_threshold": exit_threshold,
        "vol_percentile": vol_percentile,
        "hold_min": hold_min,
        "entry_method": entry_method,
        "exit_method": exit_method,
        "chase": chase,
        "tp_ticks": tp_ticks,
        "sl_ticks": sl_ticks,
    }


def _encode_trial_params(params: Dict[str, Any]) -> Dict[str, float]:
    """Encode trial params the same way as _encode_params."""
    entry = params.get("entry_method", "smooth")
    exit_ = params.get("exit_method", "bookExit")
    entry_idx = ENTRY_METHODS.index(entry) / len(ENTRY_METHODS) if entry in ENTRY_METHODS else 0.5
    exit_idx = EXIT_METHODS.index(exit_) / len(EXIT_METHODS) if exit_ in EXIT_METHODS else 0.5

    return {
        "conv_threshold": params.get("conv_threshold", 2.5) / 3.0,
        "exit_threshold": params.get("exit_threshold", 0.0),
        "vol_percentile": params.get("vol_percentile", 70) / 90.0,
        "tp_ticks": (params.get("tp_ticks") or 0) / 30.0,
        "sl_ticks": (params.get("sl_ticks") or 0) / 50.0,
        "hold_min": params.get("hold_min", 30.0) / 60.0,
        "chase": float(params.get("chase", True)),
        "entry_method": entry_idx,
        "exit_method": exit_idx,
    }


def _progress_callback(total_trials: int, t0: float):
    """Optuna callback for progress logging."""
    def callback(study, trial):
        if trial.number % max(1, total_trials // 20) == 0:
            elapsed = time.time() - t0
            pct = trial.number / total_trials * 100
            try:
                # Single-objective
                best = study.best_value
            except (RuntimeError, AttributeError):
                # Multi-objective: show Pareto front size
                n_pareto = len(study.best_trials)
                best = f"Pareto front size={n_pareto}"
            log.info(
                f"  Trial {trial.number}/{total_trials} ({pct:.0f}%) | "
                f"Best: {best} | Elapsed: {elapsed:.0f}s"
            )
    return callback


# ==============================================================================
# LIVE SIM MODE — Run actual fill_sim_cli.exe
# ==============================================================================


def run_fill_sim(
    pred_file: Path,
    mbo_file: Path,
    output_file: Path,
    conv_threshold: float,
    hold_ms: int,
    tp_ticks: Optional[int],
    sl_ticks: Optional[int],
    chase: bool = True,
    chase_max_ticks: int = 1,
    chase_max_reprices: int = 3,
    signal_threshold: float = 0.1,
    latency_ms: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Run fill_sim_cli.exe for a single date's prediction file with arbitrary parameters.

    This is the core function for live-sim mode — it allows continuous parameter
    space exploration beyond the fixed grid from the sweep.

    Args:
        pred_file: Path to .npz prediction file
        mbo_file: Path to .mbo.dbn.zst MBO data file
        output_file: Where to write the JSON result
        conv_threshold: Conviction multiplier (signal > conv_threshold triggers entry)
        hold_ms: Max hold duration in milliseconds
        tp_ticks: Take profit in ticks (None = no TP)
        sl_ticks: Trailing stop in ticks (None = no SL)
        chase: Use chase-entry (limit order repricing)
        chase_max_ticks: Max ticks to chase
        chase_max_reprices: Max reprices per entry
        signal_threshold: Base signal threshold (default 0.1)
        latency_ms: Simulated latency in ms

    Returns:
        Parsed JSON result dict or None on failure
    """
    if not BINARY.exists():
        log.error(f"fill_sim_cli.exe not found at {BINARY}")
        return None

    cmd = [
        str(BINARY),
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(output_file),
        "--hold-ms", str(hold_ms),
        "--signal-threshold", str(signal_threshold),
        "--latency-ms", str(latency_ms),
        "--quiet",
    ]

    # Conviction filter (signal threshold scaled by conv)
    # NOTE: fill_sim uses signal_threshold directly; conv is encoded in predictions.
    # For live-sim with conv as a free param, we scale the threshold inversely.
    # Higher conv = more selective entry = higher signal threshold.
    effective_threshold = signal_threshold * conv_threshold
    cmd[cmd.index(str(signal_threshold))] = str(effective_threshold)

    if chase:
        cmd += [
            "--chase-entry",
            "--chase-max-ticks", str(chase_max_ticks),
            "--chase-max-reprices", str(chase_max_reprices),
        ]

    if tp_ticks is not None:
        cmd += ["--take-profit-ticks", str(tp_ticks)]

    if sl_ticks is not None:
        cmd += ["--trailing-ticks", str(sl_ticks)]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and output_file.exists():
            with open(output_file) as f:
                return json.load(f)
        else:
            log.debug(f"fill_sim failed: rc={result.returncode} stderr={result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        log.debug("fill_sim timeout")
    except Exception as e:
        log.debug(f"fill_sim error: {e}")

    return None


def run_live_sim_trial(params: Dict[str, Any], max_dates: int = 30) -> Optional[Dict[str, Any]]:
    """
    Run fill_sim across multiple dates for a given parameter set and aggregate.

    Args:
        params: Dict with conv_threshold, hold_min, tp_ticks, sl_ticks, chase, etc.
        max_dates: Max number of prediction files to run (for speed)

    Returns:
        Aggregated result dict (same format as normalized records) or None
    """
    # Find available prediction + MBO file pairs
    pred_files = sorted(PRED_DIR_WF.glob("*.npz"))[:max_dates]
    if not pred_files:
        pred_files = sorted(PRED_DIR_STACKED.glob("*.npz"))[:max_dates]

    if not pred_files:
        log.error("No prediction files found for live-sim")
        return None

    conv = params.get("conv_threshold", 2.5)
    hold_ms = int(params.get("hold_min", 30.0) * 60 * 1000)
    tp = params.get("tp_ticks")
    sl = params.get("sl_ticks")
    chase = params.get("chase", True)

    trial_id = f"optuna_{int(time.time()*1000) % 1000000}"
    SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_pnls = []
    all_trades = []
    n_signals_total = 0
    n_filled_total = 0

    for pf in pred_files:
        date = pf.stem[:10]
        nodash = date.replace("-", "")
        mbo_file = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn.zst"
        if not mbo_file.exists():
            mbo_file = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn"
        if not mbo_file.exists():
            continue

        out_file = SIM_OUT_DIR / f"{trial_id}_{date}.json"
        result = run_fill_sim(
            pred_file=pf,
            mbo_file=mbo_file,
            output_file=out_file,
            conv_threshold=conv,
            hold_ms=hold_ms,
            tp_ticks=tp,
            sl_ticks=sl,
            chase=chase,
        )

        if result is None:
            continue

        daily_pnls.append(result.get("total_pnl_dollars", 0))
        n_signals_total += result.get("total_signals", 0)
        n_filled_total += result.get("total_filled", 0)

        for trade in result.get("trades", []):
            all_trades.append(trade.get("pnl_dollars", 0))

        # Clean up temp file
        try:
            out_file.unlink()
        except Exception:
            pass

    if len(daily_pnls) < 5 or not all_trades:
        return None

    n_days = len(daily_pnls)
    total_pnl = sum(daily_pnls)
    mean_d = total_pnl / n_days
    std_d = (sum((p - mean_d)**2 for p in daily_pnls) / (n_days - 1))**0.5 if n_days > 1 else 1e-9
    sharpe = (mean_d / max(std_d, 1e-9)) * (252**0.5)
    wins = sum(1 for p in all_trades if p > 0)
    losses = len(all_trades) - wins
    gross_profit = sum(p for p in all_trades if p > 0)
    gross_loss = abs(sum(p for p in all_trades if p < 0))
    pf = gross_profit / max(gross_loss, 1e-9)
    wr = wins / len(all_trades)
    avg_pnl = total_pnl / len(all_trades)
    fill_rate = n_filled_total / n_signals_total if n_signals_total > 0 else 0

    return {
        "config": trial_id,
        "entry_method": params.get("entry_method", "smooth"),
        "exit_method": params.get("exit_method", "hold_timeout"),
        "conv_threshold": conv,
        "exit_threshold": params.get("exit_threshold", 0.0),
        "vol_percentile": params.get("vol_percentile", 70),
        "tp_ticks": tp,
        "sl_ticks": sl,
        "hold_min": params.get("hold_min", 30.0),
        "chase": chase,
        "n_days": n_days,
        "sharpe": round(sharpe, 4),
        "total_pnl": round(total_pnl, 2),
        "annualized_pnl": round(total_pnl / n_days * 252, 2),
        "n_trades": len(all_trades),
        "win_rate": round(wr, 4),
        "fill_rate": round(fill_rate, 4),
        "avg_trade_pnl": round(avg_pnl, 2),
        "max_dd": 0.0,  # not tracked per-trial for speed
        "profit_factor": round(pf, 4),
        "source": "live_sim",
    }


def run_live_sim_study(
    n_trials: int = 200,
    study_name: str = "cnn_live_sim_opt",
    max_dates_per_trial: int = 20,
) -> Any:
    """
    Run Optuna study using actual fill_sim_cli.exe calls for each trial.
    Each trial runs fill_sim on up to max_dates_per_trial prediction files.

    This is slower but explores continuous parameter space directly.
    Recommended: start with grid-search to identify promising regions,
    then use live-sim to refine.
    """
    try:
        import optuna
    except ImportError:
        log.error("optuna not installed. Run: pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if not BINARY.exists():
        log.error(f"fill_sim_cli.exe not found at {BINARY}. Cannot run live-sim mode.")
        sys.exit(1)

    def objective(trial):
        params = {
            "conv_threshold": trial.suggest_float("conv_threshold", 0.5, 3.5),
            "exit_threshold": trial.suggest_float("exit_threshold", 0.0, 1.0),
            "vol_percentile": trial.suggest_float("vol_percentile", 0, 90),
            "hold_min": trial.suggest_float("hold_min", 5.0, 60.0),
            "entry_method": trial.suggest_categorical(
                "entry_method", ["smooth", "book", "ema", "mom"]
            ),
            "exit_method": trial.suggest_categorical(
                "exit_method", ["bookExit", "smoothExit", "hold_timeout"]
            ),
            "chase": trial.suggest_categorical("chase", [True, False]),
            "tp_ticks": trial.suggest_categorical("tp_ticks", [None, 5, 8, 10, 15, 20]),
            "sl_ticks": trial.suggest_categorical("sl_ticks", [None, 10, 15, 20, 25]),
        }

        result = run_live_sim_trial(params, max_dates=max_dates_per_trial)
        if result is None:
            return -10.0

        score = compute_score(result)
        # Store result for later analysis
        trial.set_user_attr("sharpe", result["sharpe"])
        trial.set_user_attr("total_pnl", result["total_pnl"])
        trial.set_user_attr("n_trades", result["n_trades"])
        trial.set_user_attr("win_rate", result["win_rate"])
        trial.set_user_attr("profit_factor", result["profit_factor"])
        return score

    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=max(20, n_trials // 5),
        seed=42,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    log.info(f"Starting live-sim study: {n_trials} trials, {max_dates_per_trial} dates/trial")
    t0 = time.time()

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[_progress_callback(n_trials, t0)],
    )

    elapsed = time.time() - t0
    log.info(f"Live-sim study complete in {elapsed:.1f}s")
    return study


# ==============================================================================
# ANALYSIS & OUTPUT
# ==============================================================================


def get_pareto_front(study) -> List:
    """
    Get Pareto-optimal trials from a multi-objective study.

    Returns:
        List of optuna.Trial objects on the Pareto front
    """
    try:
        import optuna
        return study.best_trials  # Optuna returns Pareto front for multi-objective
    except Exception as e:
        log.error(f"Failed to get Pareto front: {e}")
        return []


def get_top_configs(study, records: List[Dict[str, Any]], n: int = 20) -> List[Dict[str, Any]]:
    """
    Get the top N configs from the Optuna study, matched back to actual sweep data.

    For multi-objective studies, returns Pareto front trials sorted by combined score.
    For single-objective, returns top N by best_value.

    Returns:
        List of dicts with trial params + estimated/actual performance
    """
    try:
        import optuna
    except ImportError:
        return []

    results = []

    # Get trials sorted by value
    if study.directions and len(study.directions) > 1:
        # Multi-objective: use Pareto front
        trials = study.best_trials
        trial_list = [
            (
                0.6 * t.values[0] + 0.4 * math.log1p(max(t.values[1], 0.01)),
                t,
            )
            for t in trials
        ]
    else:
        # Single-objective
        trial_list = [
            (t.value if t.value is not None else -999, t)
            for t in study.trials
            if t.state.name == "COMPLETE"
        ]

    trial_list.sort(key=lambda x: x[0], reverse=True)
    top_trials = trial_list[:n]

    for combined_score, trial in top_trials:
        params = trial.params
        tp_raw = params.get("tp_raw", 0)
        sl_raw = params.get("sl_raw", 0)
        tp_ticks = None if tp_raw < 2.5 else int(round(tp_raw))
        sl_ticks = None if sl_raw < 5.0 else int(round(sl_raw))

        # Find nearest actual data point for reference
        feat = _encode_trial_params({
            "entry_method": params.get("entry_method", "smooth"),
            "exit_method": params.get("exit_method", "bookExit"),
            "conv_threshold": params.get("conv_threshold", 2.5),
            "exit_threshold": params.get("exit_threshold", 0.0),
            "vol_percentile": params.get("vol_percentile", 70),
            "tp_ticks": tp_ticks,
            "sl_ticks": sl_ticks,
            "hold_min": params.get("hold_min", 30.0),
            "chase": params.get("chase", True),
        })

        # Find closest actual record
        feat_vec = np.array(list(feat.values()))
        feats_all, scores_all, objs_all = build_data_arrays(records)
        dists = np.linalg.norm(feats_all - feat_vec[np.newaxis, :], axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_rec = records[nearest_idx]
        nearest_dist = float(dists[nearest_idx])

        result = {
            "rank": len(results) + 1,
            "trial_number": trial.number,
            "combined_score": round(combined_score, 4),
            "suggested_params": {
                "entry_method": params.get("entry_method"),
                "exit_method": params.get("exit_method"),
                "conv_threshold": round(params.get("conv_threshold", 2.5), 3),
                "exit_threshold": round(params.get("exit_threshold", 0.0), 3),
                "vol_percentile": round(params.get("vol_percentile", 70), 1),
                "hold_min": round(params.get("hold_min", 30.0), 1),
                "tp_ticks": tp_ticks,
                "sl_ticks": sl_ticks,
                "chase": params.get("chase", True),
            },
            "nearest_actual": {
                "config": nearest_rec.get("config", ""),
                "distance": round(nearest_dist, 4),
                "sharpe": nearest_rec.get("sharpe", 0.0),
                "total_pnl": nearest_rec.get("total_pnl", 0.0),
                "annualized_pnl": nearest_rec.get("annualized_pnl", 0.0),
                "n_trades": nearest_rec.get("n_trades", 0),
                "win_rate": nearest_rec.get("win_rate", 0.0),
                "fill_rate": nearest_rec.get("fill_rate", 0.0),
                "source": nearest_rec.get("source", ""),
            },
        }

        # Add Pareto values if multi-objective
        if hasattr(trial, "values") and trial.values is not None and len(trial.values) == 2:
            result["estimated_sharpe"] = round(trial.values[0], 4)
            result["estimated_pf"] = round(trial.values[1], 4)
        elif hasattr(trial, "value") and trial.value is not None:
            result["estimated_combined_score"] = round(trial.value, 4)

        results.append(result)

    return results


def show_param_importance(study) -> Dict[str, float]:
    """
    Compute and display parameter importance using Optuna's built-in analysis.

    Returns:
        Dict mapping parameter name -> importance score
    """
    try:
        import optuna
    except ImportError:
        log.error("optuna not installed")
        return {}

    try:
        # Only works for single-objective studies
        if len(study.directions) > 1:
            log.info("Parameter importance not available for multi-objective studies.")
            log.info("Run with n_objectives=1 or --mode grid-search for importance analysis.")
            return {}

        importance = optuna.importance.get_param_importances(study)

        log.info("\n" + "="*50)
        log.info("PARAMETER IMPORTANCE (FAnova)")
        log.info("="*50)
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(imp * 40)
            log.info(f"  {param:<25} {imp:.4f}  {bar}")
        log.info("="*50)

        return dict(importance)

    except Exception as e:
        log.error(f"Failed to compute parameter importance: {e}")
        return {}


def print_top_results(top_configs: List[Dict[str, Any]]) -> None:
    """Pretty-print the top configurations."""
    print("\n" + "="*80)
    print("TOP PARAMETER CONFIGURATIONS (Optuna Optimization)")
    print("="*80)

    for r in top_configs:
        p = r["suggested_params"]
        n = r["nearest_actual"]
        print(f"\nRank #{r['rank']} | Trial #{r['trial_number']} | Score: {r['combined_score']:.4f}")
        print(f"  Suggested: entry={p['entry_method']} exit={p['exit_method']}")
        print(f"             conv={p['conv_threshold']:.3f}  exit_thr={p['exit_threshold']:.3f}")
        print(f"             vol%={p['vol_percentile']:.0f}  hold={p['hold_min']:.0f}min")
        print(f"             TP={p['tp_ticks']}  SL={p['sl_ticks']}  chase={p['chase']}")
        if "estimated_sharpe" in r:
            print(f"  Estimated: Sharpe={r['estimated_sharpe']:.2f}  PF={r['estimated_pf']:.2f}")
        elif "estimated_combined_score" in r:
            print(f"  Estimated score: {r['estimated_combined_score']:.4f}")
        print(f"  Nearest actual [{n['source']}] dist={n['distance']:.4f}:")
        print(f"    Config: {n['config']}")
        print(f"    Sharpe={n['sharpe']:.2f}  PnL=${n['total_pnl']:,.0f}  "
              f"Annual=${n['annualized_pnl']:,.0f}  Trades={n['n_trades']}")
        print(f"    WR={n['win_rate']:.1%}  Fill={n['fill_rate']:.1%}")


def _safe_study_best(study) -> Dict[str, Any]:
    """Get best trial info safely for both single- and multi-objective studies."""
    try:
        # Single-objective
        return {"value": study.best_value, "params": study.best_params}
    except RuntimeError:
        # Multi-objective: return Pareto front summary
        pareto = study.best_trials
        return {
            "pareto_front_size": len(pareto),
            "pareto_values": [
                {"trial": t.number, "values": t.values, "params": t.params}
                for t in pareto[:10]  # Top 10 Pareto-optimal
            ],
        }


def save_results(
    study,
    top_configs: List[Dict[str, Any]],
    importance: Dict[str, float],
    records: List[Dict[str, Any]],
    mode: str,
) -> Path:
    """Save all results to JSON."""
    output_file = RESULTS_DIR / f"optuna_optimization_results.json"

    # Top actual records by combined score (from the raw data, not Optuna estimates)
    top_actual = sorted(records, key=lambda r: compute_score(r), reverse=True)[:50]

    output = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "n_trials": len(study.trials),
        "n_records_used": len(records),
        "param_importance": importance,
        "top_optuna_configs": top_configs,
        "top_actual_records": [
            {
                "config": r.get("config"),
                "source": r.get("source"),
                "combined_score": round(compute_score(r), 4),
                "sharpe": r.get("sharpe"),
                "total_pnl": r.get("total_pnl"),
                "annualized_pnl": r.get("annualized_pnl"),
                "n_trades": r.get("n_trades"),
                "win_rate": r.get("win_rate"),
                "fill_rate": r.get("fill_rate"),
                "entry_method": r.get("entry_method"),
                "exit_method": r.get("exit_method"),
                "conv_threshold": r.get("conv_threshold"),
                "exit_threshold": r.get("exit_threshold"),
                "vol_percentile": r.get("vol_percentile"),
                "tp_ticks": r.get("tp_ticks"),
                "sl_ticks": r.get("sl_ticks"),
                "hold_min": r.get("hold_min"),
                "chase": r.get("chase"),
                "n_days": r.get("n_days"),
            }
            for r in top_actual
        ],
        "study_best": _safe_study_best(study),
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log.info(f"Results saved to: {output_file}")
    return output_file


def print_data_summary(records: List[Dict[str, Any]]) -> None:
    """Print a summary of loaded sweep data."""
    if not records:
        print("No records loaded.")
        return

    sharpes = [r.get("sharpe", 0) for r in records]
    pnls = [r.get("total_pnl", 0) for r in records]
    scores = [compute_score(r) for r in records]

    pos_sharpe = [r for r in records if r.get("sharpe", 0) > 0]
    profitable = [r for r in records if r.get("total_pnl", 0) > 0]

    print("\n" + "="*60)
    print("SWEEP DATA SUMMARY")
    print("="*60)
    print(f"Total records:       {len(records)}")
    print(f"Positive Sharpe:     {len(pos_sharpe)} ({len(pos_sharpe)/len(records):.1%})")
    print(f"Profitable configs:  {len(profitable)} ({len(profitable)/len(records):.1%})")
    print(f"Sharpe range:        {min(sharpes):.2f} to {max(sharpes):.2f}")
    print(f"P&L range:           ${min(pnls):,.0f} to ${max(pnls):,.0f}")
    print(f"Score range:         {min(scores):.3f} to {max(scores):.3f}")

    # Sources breakdown
    sources = defaultdict(int)
    for r in records:
        sources[r.get("source", "unknown")] += 1
    print("\nBy source:")
    for src, cnt in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {src:<30} {cnt:>5} records")

    # Entry methods
    entries = defaultdict(int)
    for r in records:
        entries[r.get("entry_method", "?")] += 1
    print("\nBy entry method:")
    for m, cnt in sorted(entries.items(), key=lambda x: x[1], reverse=True):
        print(f"  {m:<30} {cnt:>5}")
    print("="*60)


# ==============================================================================
# CLI
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Optuna-based parameter optimization for CNN WF trading strategy"
    )
    parser.add_argument(
        "--mode",
        choices=["grid-search", "live-sim"],
        default="grid-search",
        help=(
            "grid-search: use existing sweep data as surrogate objective (fast). "
            "live-sim: run actual fill_sim_cli.exe for each trial (slow, accurate)."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="Number of Optuna trials (default: 1000 for grid-search, 200 for live-sim)",
    )
    parser.add_argument(
        "--source",
        default="all",
        choices=["all", "stacked_exit", "hold_sweep", "entry_exit_matrix", "novel_ideas", "combo_sweep"],
        help="Which sweep data source(s) to use (default: all)",
    )
    parser.add_argument(
        "--n-objectives",
        type=int,
        default=1,
        choices=[1, 2],
        help="1=single-objective (combined score), 2=multi-objective (Sharpe + PF Pareto)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top configs to show (default: 20)",
    )
    parser.add_argument(
        "--show-importance",
        action="store_true",
        help="Show parameter importance analysis (only for single-objective)",
    )
    parser.add_argument(
        "--include-remote",
        action="store_true",
        help="Also pull results from Jupiter/Saturn via SSH (requires network access)",
    )
    parser.add_argument(
        "--dates-per-trial",
        type=int,
        default=20,
        help="Number of dates per trial in live-sim mode (default: 20)",
    )
    parser.add_argument(
        "--study-name",
        default="cnn_param_opt",
        help="Optuna study name (default: cnn_param_opt)",
    )
    parser.add_argument(
        "--data-summary",
        action="store_true",
        help="Only show data summary, do not run optimization",
    )

    args = parser.parse_args()

    log.info("="*60)
    log.info("Optuna Parameter Optimizer — CNN WF Trading Strategy")
    log.info("="*60)
    log.info(f"Mode: {args.mode} | Trials: {args.trials} | Source: {args.source}")
    log.info(f"Objectives: {args.n_objectives} | Include remote: {args.include_remote}")

    # ── Load data ──
    records = load_all_sweep_data(source=args.source)

    if args.include_remote:
        log.info("Loading remote results from Jupiter/Saturn...")
        remote_records = load_remote_results()
        records.extend(remote_records)
        log.info(f"Total records after remote: {len(records)}")

    print_data_summary(records)

    if args.data_summary:
        log.info("--data-summary flag set, exiting without optimization.")
        return

    if len(records) == 0:
        log.error("No data loaded. Check that aggregated JSON files exist in:")
        log.error(f"  {RESULTS_DIR}")
        sys.exit(1)

    # ── Run optimization ──
    if args.mode == "grid-search":
        study = run_grid_search_study(
            records=records,
            n_trials=args.trials,
            study_name=args.study_name,
            n_objectives=args.n_objectives,
        )
    else:  # live-sim
        log.info(f"Live-sim mode: {args.dates_per_trial} dates per trial")
        study = run_live_sim_study(
            n_trials=args.trials,
            study_name=args.study_name,
            max_dates_per_trial=args.dates_per_trial,
        )

    if study is None:
        log.error("Study failed to complete.")
        sys.exit(1)

    # ── Analysis ──
    top_configs = get_top_configs(study, records, n=args.top_n)
    print_top_results(top_configs)

    importance = {}
    if args.show_importance or args.n_objectives == 1:
        importance = show_param_importance(study)

    # ── Save ──
    output_file = save_results(study, top_configs, importance, records, mode=args.mode)
    log.info(f"\nDone. Results: {output_file}")

    # ── Print best actual configs from raw data ──
    print("\n" + "="*60)
    print("TOP ACTUAL SWEEP RECORDS (by combined score)")
    print("="*60)
    top_actual = sorted(records, key=lambda r: compute_score(r), reverse=True)[:args.top_n]
    for i, r in enumerate(top_actual, 1):
        print(
            f"#{i:02d} [{r.get('source','?'):15}] "
            f"Sharpe={r.get('sharpe', 0):.2f}  "
            f"PnL=${r.get('total_pnl', 0):8,.0f}  "
            f"Score={compute_score(r):.3f}  "
            f"Trades={r.get('n_trades', 0):4d}  "
            f"WR={r.get('win_rate', 0):.1%}  "
            f"conf={r.get('conv_threshold', 0):.1f}  "
            f"vol={r.get('vol_percentile', 0):.0f}  "
            f"TP={r.get('tp_ticks')}  "
            f"SL={r.get('sl_ticks')}  "
            f"{r.get('config', '')[:50]}"
        )


if __name__ == "__main__":
    main()
