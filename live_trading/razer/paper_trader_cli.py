"""Razer Paper Trader CLI Wrapper (HC #351).

A pure config-marshalling wrapper around the existing paper trading daemon
(`paper_trading_mamba_v2.py` / `paper_trading_mamba_v2_patched_FROM_RAZER.py`).
This wrapper does NOT duplicate any trading logic — it resolves operational
parameters from a layered config (hard defaults < JSON file < env vars < CLI flags),
validates the resolved values, dumps a timestamped snapshot, and then either
(a) exits in --dry-run mode, or (b) invokes the existing daemon as a subprocess
with the resolved parameters mapped to its native CLI flags.

Why a wrapper:
  - The daemon lives on a Windows host (Razer, RTX 3070). Editing source there
    is operationally painful. With this wrapper the user tweaks JSON or passes
    one flag and dispatches via WMI Win32_Process Create (per HC #308).
  - Every run captures the exact resolved config so we can reproduce results.

Priority order (highest wins):
    1. CLI flag
    2. Environment variable (RAZER_PAPER_<UPPER_NAME>)
    3. JSON config file (--config)
    4. Hard-coded default in CONFIG_SPEC below

Param defaults are sourced from:
  - CLAUDE.md cost constants (ES_RT_COMMISSION_TICKS=0.376, market crossing=1.376)
  - HC #46 / #226 / #230 / #231C (risk management defaults)
  - HC #41 (TOD blocks: 09:25–09:40 ET)
  - HC #324 (quality over quantity — tighter top-band cutoffs)
  - HC #340–341 (cost-aware execution sizing)
  - Current production paper trader values
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Spec: every operationally-relevant param.
# Each entry: (name, type, default, group, help, validator)
# ─────────────────────────────────────────────────────────────────────────────

def _frac(v: float) -> bool:
    return 0.0 <= v <= 1.0


def _pos(v: float) -> bool:
    return v > 0.0


def _nonneg(v: float) -> bool:
    return v >= 0.0


def _hhmm(v: str) -> bool:
    return bool(re.match(r"^\d{2}:\d{2}$", v))


@dataclass
class ParamSpec:
    name: str
    type: type
    default: Any
    group: str
    help: str
    validator: Optional[Callable[[Any], bool]] = None
    choices: Optional[list] = None


# Defaults cite production state and HC references in source comments above.
CONFIG_SPEC: list[ParamSpec] = [
    # ── Confluence / veto gates ────────────────────────────────────────────
    ParamSpec("cnn_threshold", float, 0.60, "gates",
              "CNN-Mamba v2 confidence floor (0-1). Lower=more trades.", _frac),
    ParamSpec("patchtst_threshold", float, 0.55, "gates",
              "PatchTST confluence/veto confidence floor (0-1).", _frac),
    ParamSpec("require_agreement", bool, True, "gates",
              "Require CNN-Mamba and PatchTST to agree on direction."),
    ParamSpec("patchtst_veto_enabled", bool, True, "gates",
              "Enable PatchTST veto on 5s/10s horizons (framework_config.json)."),
    ParamSpec("side_bias", str, "short", "gates",
              "Allowed sides: long | short | both. (HC: short edge stronger.)",
              choices=["long", "short", "both"]),

    # ── Confidence band cutoffs per horizon (percentile, 0–100) ────────────
    ParamSpec("top_band_pct_1s", float, 99.5, "bands",
              "Top-percentile cutoff for 1s horizon entries.",
              lambda v: 0.0 < v <= 100.0),
    ParamSpec("top_band_pct_5s", float, 99.0, "bands",
              "Top-percentile cutoff for 5s horizon entries.",
              lambda v: 0.0 < v <= 100.0),
    ParamSpec("top_band_pct_10s", float, 99.0, "bands",
              "Top-percentile cutoff for 10s horizon entries.",
              lambda v: 0.0 < v <= 100.0),
    ParamSpec("top_band_pct_30s", float, 98.0, "bands",
              "Top-percentile cutoff for 30s horizon entries.",
              lambda v: 0.0 < v <= 100.0),

    # ── Sizing ──────────────────────────────────────────────────────────────
    ParamSpec("max_position", int, 1, "sizing",
              "Max simultaneous contracts. HC #324: stay small.", _pos),
    ParamSpec("per_trade_size", int, 1, "sizing",
              "Contracts per entry.", _pos),
    ParamSpec("vol_scale_enabled", bool, False, "sizing",
              "Scale size by realized vol vs target."),
    ParamSpec("vol_target_ticks", float, 2.0, "sizing",
              "Target per-trade vol (ticks) when vol scaling is on.", _pos),

    # ── Entry/exit timing ───────────────────────────────────────────────────
    ParamSpec("cancel_eval_window", int, 40, "timing",
              "Cancels passive order after N evaluations (250ms each). "
              "Default 40 = ~10s, per CLAUDE.md optimal 9-12s.", _pos),
    ParamSpec("max_hold_seconds", float, 120.0, "timing",
              "Max position hold (seconds). HC #226: alpha exhausts ~30-60s.", _pos),
    ParamSpec("passive_offset_ticks", float, 0.0, "timing",
              "Offset from best bid/ask for passive post (0=join).", _nonneg),
    ParamSpec("aggressive_cross_enabled", bool, False, "timing",
              "Allow IOC market crossing after cancel window expires."),

    # ── Risk ────────────────────────────────────────────────────────────────
    ParamSpec("hard_stop_ticks", float, 2.0, "risk",
              "Hard stop (ticks). HC #46 optimal SL=2 ticks (Sortino=7.41).", _pos),
    ParamSpec("trailing_mfe_trigger_ticks", float, 4.0, "risk",
              "MFE that activates trailing stop. HC #46: mean MFE ≈ 4.37t.", _pos),
    ParamSpec("trailing_lock_ticks", float, 1.0, "risk",
              "Ticks above entry locked after trailing activation.", _nonneg),
    ParamSpec("signal_decay_threshold", float, 0.5, "risk",
              "Exit if model conviction drops below this (0-1).", _frac),
    ParamSpec("max_daily_loss", float, -500.0, "risk",
              "Daily loss circuit breaker ($). Negative number.",
              lambda v: v <= 0.0),
    ParamSpec("max_consecutive_losses", int, 3, "risk",
              "Pause trading after N consecutive losers.", _pos),
    ParamSpec("loss_cooldown_seconds", float, 300.0, "risk",
              "Cooldown duration after consecutive-loss pause.", _nonneg),

    # ── Time-of-day blocks (repeatable HH:MM ET) ───────────────────────────
    ParamSpec("tod_block_start", str, "09:25", "tod",
              "Block-start HH:MM ET (HC #41). Repeatable via JSON list.", _hhmm),
    ParamSpec("tod_block_end", str, "09:40", "tod",
              "Block-end HH:MM ET (HC #41). Repeatable via JSON list.", _hhmm),

    # ── Feature warmup / buffers ───────────────────────────────────────────
    ParamSpec("feature_warmup_count", int, 5000, "warmup",
              "Events before model goes hot (current Razer prod: 5000).", _pos),
    ParamSpec("mbo_buffer_size", int, 100000, "warmup",
              "MBO ring buffer size (events).", _pos),

    # ── Model paths ─────────────────────────────────────────────────────────
    ParamSpec("cnn_mamba_weights", str,
              r"C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_10_best.pt",
              "models", "CNN-Mamba v2 .pt weights (Windows path)."),
    ParamSpec("cnn_mamba_stats", str,
              r"C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_09_feature_stats.npz",
              "models", "CNN-Mamba v2 feature stats .npz."),
    ParamSpec("patchtst_weights", str, "", "models",
              "PatchTST .pt weights path (empty disables PatchTST)."),
    ParamSpec("patchtst_stats", str, "", "models",
              "PatchTST feature stats .npz."),

    # ── Output / logging ────────────────────────────────────────────────────
    ParamSpec("log_dir", str, r"C:\Users\claude\Lvl3Quant\live_trading\logs", "output",
              "Log directory (Windows path on Razer)."),
    ParamSpec("log_verbosity", str, "INFO", "output",
              "Log level.", choices=["DEBUG", "INFO", "WARN", "WARNING", "ERROR"]),
    ParamSpec("mlflow_experiment", str, "razer_paper_live", "output",
              "MLflow experiment name (CLAUDE.md mandates MLflow tracking)."),

    # ── Entrypoint / misc ──────────────────────────────────────────────────
    ParamSpec("daemon_script", str,
              r"C:\Users\claude\Lvl3Quant\live_trading_linux\paper_trading_mamba_v2.py",
              "entrypoint",
              "Path to the existing daemon script the wrapper will invoke."),
    ParamSpec("python_exe", str, "python", "entrypoint",
              "Python executable to use when launching the daemon."),
    ParamSpec("symbol", str, "ESM6", "entrypoint", "Instrument symbol."),
    ParamSpec("exchange", str, "CME", "entrypoint", "Exchange code."),
    ParamSpec("device", str, "cuda", "entrypoint",
              "Inference device.", choices=["cuda", "cpu"]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Resolved config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResolvedConfig:
    """Snapshot of all resolved params plus provenance metadata."""
    values: dict
    sources: dict  # param_name -> "cli" | "env" | "json" | "default"
    json_path: Optional[str]
    resolved_at_utc: str

    def to_dict(self) -> dict:
        return {
            "values": self.values,
            "sources": self.sources,
            "json_path": self.json_path,
            "resolved_at_utc": self.resolved_at_utc,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────────

def _coerce(spec: ParamSpec, raw: Any) -> Any:
    """Best-effort coerce a raw value (string from env/CLI) to spec.type."""
    if raw is None:
        return None
    if spec.type is bool:
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
        raise ValueError(f"{spec.name}: cannot parse bool from {raw!r}")
    if spec.type is int:
        return int(raw)
    if spec.type is float:
        return float(raw)
    return str(raw)


def _env_name(param: str) -> str:
    return f"RAZER_PAPER_{param.upper()}"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper_trader_cli.py",
        description="Razer paper-trader config wrapper (HC #351).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to JSON config file (defaults override hard-coded).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved config and exit without launching daemon.")
    p.add_argument("--no-snapshot", action="store_true",
                   help="Do not write the resolved-config snapshot to log dir.")

    # Group specs by .group attribute for nicer --help output.
    groups: dict[str, argparse._ArgumentGroup] = {}
    for spec in CONFIG_SPEC:
        if spec.group not in groups:
            groups[spec.group] = p.add_argument_group(spec.group)
        flag = "--" + spec.name.replace("_", "-")
        kwargs: dict = {"help": spec.help, "default": None}
        if spec.type is bool:
            # Tri-state: use a string flag so "not provided" stays None.
            # Bind spec via default arg to avoid late-binding closure bug.
            kwargs["type"] = (lambda s, _sp=spec: _coerce(_sp, s))
            kwargs["metavar"] = "BOOL"
        elif spec.choices:
            kwargs["choices"] = spec.choices
            kwargs["type"] = spec.type
        else:
            kwargs["type"] = spec.type
        groups[spec.group].add_argument(flag, **kwargs)
    return p


def _load_json(path: Optional[str]) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"--config file not found: {path}")
    with open(p, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"--config root must be a JSON object, got {type(data).__name__}")
    return data


def resolve_config(argv: Optional[list[str]] = None) -> tuple[ResolvedConfig, argparse.Namespace]:
    parser = build_argparser()
    ns = parser.parse_args(argv)

    json_data = _load_json(ns.config)

    values: dict = {}
    sources: dict = {}

    errors: list[str] = []

    for spec in CONFIG_SPEC:
        cli_val = getattr(ns, spec.name, None)
        env_raw = os.environ.get(_env_name(spec.name))
        json_val = json_data.get(spec.name, None)

        chosen: Any
        source: str
        try:
            if cli_val is not None:
                chosen, source = _coerce(spec, cli_val), "cli"
            elif env_raw is not None:
                chosen, source = _coerce(spec, env_raw), "env"
            elif json_val is not None:
                chosen, source = _coerce(spec, json_val), "json"
            else:
                chosen, source = spec.default, "default"
        except (ValueError, TypeError) as exc:
            errors.append(f"{spec.name}: {exc}")
            continue

        # Choices check
        if spec.choices and chosen not in spec.choices:
            errors.append(f"{spec.name}: {chosen!r} not in {spec.choices}")
            continue

        # Validator
        if spec.validator is not None:
            try:
                ok = spec.validator(chosen)
            except Exception as exc:
                errors.append(f"{spec.name}: validator raised {exc}")
                continue
            if not ok:
                errors.append(f"{spec.name}={chosen!r} failed validation")
                continue

        values[spec.name] = chosen
        sources[spec.name] = source

    # Cross-field validation
    if "tod_block_start" in values and "tod_block_end" in values:
        if values["tod_block_start"] >= values["tod_block_end"]:
            errors.append(
                f"tod_block_start ({values['tod_block_start']}) must be < "
                f"tod_block_end ({values['tod_block_end']})"
            )
    if "per_trade_size" in values and "max_position" in values:
        if values["per_trade_size"] > values["max_position"]:
            errors.append(
                f"per_trade_size ({values['per_trade_size']}) > "
                f"max_position ({values['max_position']})"
            )

    if errors:
        msg = "Config validation failed:\n  - " + "\n  - ".join(errors)
        raise SystemExit(msg)

    resolved = ResolvedConfig(
        values=values,
        sources=sources,
        json_path=ns.config,
        resolved_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    return resolved, ns


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def print_resolved_table(cfg: ResolvedConfig) -> None:
    print("=" * 78)
    print("RESOLVED CONFIG  (HC #351 wrapper)")
    print("=" * 78)
    print(f"{'param':32} {'value':28} {'source':8}")
    print("-" * 78)
    # Iterate in spec order so groups stay together
    last_group = None
    for spec in CONFIG_SPEC:
        if spec.name not in cfg.values:
            continue
        if spec.group != last_group:
            print(f"[{spec.group}]")
            last_group = spec.group
        v = cfg.values[spec.name]
        s = cfg.sources[spec.name]
        vstr = str(v)
        if len(vstr) > 27:
            vstr = vstr[:24] + "..."
        print(f"  {spec.name:30} {vstr:28} {s:8}")
    print("=" * 78)
    print(f"resolved_at_utc: {cfg.resolved_at_utc}")
    print(f"json_config: {cfg.json_path or '(none)'}")
    print("=" * 78)


def write_snapshot(cfg: ResolvedConfig) -> Optional[Path]:
    log_dir = Path(cfg.values["log_dir"])
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logging.warning("Could not create log_dir %s: %s — skipping snapshot.",
                        log_dir, exc)
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = log_dir / f"resolved_config_{ts}.json"
    try:
        with open(out, "w") as f:
            json.dump(cfg.to_dict(), f, indent=2, default=str)
    except OSError as exc:
        logging.warning("Failed to write snapshot %s: %s", out, exc)
        return None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Daemon dispatch
# ─────────────────────────────────────────────────────────────────────────────

def build_daemon_argv(cfg: ResolvedConfig) -> list[str]:
    """Map resolved params to the daemon's native CLI flags.

    The existing daemon (paper_trading_mamba_v2_patched_FROM_RAZER.py) accepts:
        --weights, --stats, --device, --symbol, --exchange,
        --max-hold-minutes, --max-daily-loss, --trailing-mfe-ticks,
        --trailing-lock-ticks, --consec-loss-limit, --consec-loss-pause-min,
        --patchtst-weights, --patchtst-stats

    Anything not in that surface area is written to the snapshot JSON instead
    (the daemon can be extended later to read the snapshot).
    """
    v = cfg.values
    argv = [
        v["python_exe"],
        v["daemon_script"],
        "--symbol", v["symbol"],
        "--exchange", v["exchange"],
        "--device", v["device"],
        "--weights", v["cnn_mamba_weights"],
        "--stats", v["cnn_mamba_stats"],
        "--max-hold-minutes", f"{v['max_hold_seconds'] / 60.0:.4f}",
        "--max-daily-loss", str(v["max_daily_loss"]),
        "--trailing-mfe-ticks", str(v["trailing_mfe_trigger_ticks"]),
        "--trailing-lock-ticks", str(v["trailing_lock_ticks"]),
        "--consec-loss-limit", str(v["max_consecutive_losses"]),
        "--consec-loss-pause-min", f"{v['loss_cooldown_seconds'] / 60.0:.4f}",
    ]
    if v.get("patchtst_weights"):
        argv += ["--patchtst-weights", v["patchtst_weights"]]
    if v.get("patchtst_stats"):
        argv += ["--patchtst-stats", v["patchtst_stats"]]
    return argv


def launch_daemon(cfg: ResolvedConfig) -> int:
    argv = build_daemon_argv(cfg)
    print("\nLAUNCHING DAEMON:")
    print("  " + " ".join(shlex.quote(a) for a in argv))
    print()
    env = os.environ.copy()
    # Pass the resolved snapshot path so the daemon can read extra params later.
    env["RAZER_PAPER_RESOLVED_CONFIG"] = json.dumps(cfg.values)
    try:
        proc = subprocess.run(argv, env=env, check=False)
        return proc.returncode
    except FileNotFoundError as exc:
        print(f"FATAL: could not invoke daemon — {exc}", file=sys.stderr)
        return 127


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    try:
        cfg, ns = resolve_config(argv)
    except SystemExit as exc:
        # argparse raises SystemExit with int code on --help; with str on error.
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 2
        raise

    print_resolved_table(cfg)

    if not ns.no_snapshot:
        snap = write_snapshot(cfg)
        if snap is not None:
            print(f"snapshot: {snap}")

    if ns.dry_run:
        print("\n--dry-run set: not launching daemon. Exit 0.")
        return 0

    return launch_daemon(cfg)


if __name__ == "__main__":
    sys.exit(main())
