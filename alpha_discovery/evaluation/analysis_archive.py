#!/usr/bin/env python3
"""
analysis_archive.py -- Permanent Model Analysis Archive System
==============================================================
Every model evaluation (Tier 1/2/3) is saved permanently in a structured,
retrievable JSON format tied to the specific model run.

Functions:
  - save_analysis()      : Save a JSON report to model result dir + master index
  - load_analysis()      : Load most recent or all reports for a model
  - list_all_analyses()  : Scan all result dirs and return summary table
  - format_summary()     : Discord-ready comparison across all archived analyses
  - run_and_archive()    : Run tier3 evaluator on a prediction source and archive

Usage (CLI):
    python analysis_archive.py --run-all          # Evaluate all known models
    python analysis_archive.py --list             # List all archived analyses
    python analysis_archive.py --summary          # Discord-ready summary
    python analysis_archive.py --run-model <name> # Run specific model

Usage (importable):
    from evaluation.analysis_archive import save_analysis, load_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Add parent dirs to path for imports
_THIS_DIR = Path(__file__).resolve().parent
_EVAL_DIR = _THIS_DIR
_ALPHA_DIR = _THIS_DIR.parent
_LVL3_DIR = _ALPHA_DIR.parent

if str(_ALPHA_DIR) not in sys.path:
    sys.path.insert(0, str(_ALPHA_DIR))

from evaluation.tier3_profit_eval import (
    evaluate_all_tiers,
    load_predictions,
    load_predictions_dir,
    TICK_VALUE_USD,
    DEFAULT_COST_TICKS,
)

# ---- Constants ---------------------------------------------------------------

INDEX_PATH = _EVAL_DIR / "analysis_index.json"
RESULTS_BASE = _ALPHA_DIR / "deep_models" / "results"


# ---- Model Registry ---------------------------------------------------------
# Each entry: name, path/file, architecture info, metadata
# This is the canonical list of all models we want to track.

MODEL_REGISTRY = [
    {
        "model_name": "EventCNN1D s76 (Champion)",
        "version": "s76",
        "architecture": "EventCNN1D",
        "features": "MBO event features (128-dim)",
        "walk_forward": {"window": "expanding", "folds": 76, "mode": "expanding"},
        "pred_source": str(_LVL3_DIR / "data" / "cnn_s76_concat_oot_predictions.npz"),
        "source_type": "file",
        "notes": "Champion model, 76 expanding WF folds, concat IC_10s=0.132",
    },
    {
        "model_name": "CNN1D 256ch 8L Sliding",
        "version": "v1",
        "architecture": "EventCNN1D (256 channels, 8 layers)",
        "features": "MBO event features (256-dim)",
        "walk_forward": {"mode": "sliding"},
        "pred_source": str(RESULTS_BASE / "cnn1d_256ch_8L_sliding"),
        "source_type": "dir",
        "notes": "Wider CNN1D architecture test",
    },
    {
        "model_name": "CNN1D Neptune Apr18",
        "version": "20260418",
        "architecture": "EventCNN1D",
        "features": "MBO event features (128-dim)",
        "walk_forward": {"mode": "expanding"},
        "pred_source": str(RESULTS_BASE / "cnn1d_neptune_20260418_1033"),
        "source_type": "dir",
        "notes": "CNN1D trained on Neptune, 7 folds",
    },
    {
        "model_name": "Event Mamba v4 CUDA",
        "version": "v4",
        "architecture": "Event Mamba SSM (CUDA)",
        "features": "MBO event features (128-dim)",
        "walk_forward": {"mode": "expanding"},
        "pred_source": str(RESULTS_BASE / "event_mamba_cuda"),
        "source_type": "dir",
        "notes": "Mamba state-space model, CUDA implementation",
    },
    {
        "model_name": "Event Mamba v4 CUDA v2",
        "version": "v4.2",
        "architecture": "Event Mamba SSM (CUDA v2)",
        "features": "MBO event features (192-dim)",
        "walk_forward": {"mode": "expanding"},
        "pred_source": str(RESULTS_BASE / "event_mamba_cuda_v2"),
        "source_type": "dir",
        "notes": "Mamba v2 with wider embeddings (192-dim)",
    },
    {
        "model_name": "Event Mamba Fast",
        "version": "fast",
        "architecture": "Event Mamba SSM (Fast variant)",
        "features": "MBO event features (128-dim)",
        "walk_forward": {"mode": "expanding"},
        "pred_source": str(RESULTS_BASE / "event_mamba_fast"),
        "source_type": "dir",
        "notes": "Fast Mamba variant, 3 folds",
    },
    {
        "model_name": "LGBM smart_v2 DA",
        "version": "v2",
        "architecture": "LightGBM classifier",
        "features": "smart_v2 (22 features)",
        "walk_forward": {"window": 60, "oot_days": 5, "folds": 9, "mode": "sliding"},
        "pred_source": str(RESULTS_BASE / "lgbm_da_smart_v2"),
        "source_type": "dir",
        "is_classifier": True,
        "notes": "Directional accuracy classifier, sliding window WF",
    },
    {
        "model_name": "PatchTST feat15 Sliding60d",
        "version": "v1",
        "architecture": "PatchTST",
        "features": "15 selected features",
        "walk_forward": {"window": 60, "mode": "sliding"},
        "pred_source": str(_LVL3_DIR / "output" / "patchtst_sliding60d_feat15" / "concat_oot_predictions.npz"),
        "source_type": "file",
        "notes": "PatchTST with 15 feature subset",
    },
]


# ---- Core Functions ----------------------------------------------------------

def _make_serializable(obj: Any) -> Any:
    """Convert numpy types and special floats for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if np.isnan(val):
            return "NaN"
        if np.isinf(val):
            return "Inf" if val > 0 else "-Inf"
        return val
    if isinstance(obj, float):
        if np.isnan(obj):
            return "NaN"
        if np.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
        return obj
    return obj


def save_analysis(
    report: dict,
    result_dir: str | Path,
    update_index: bool = True,
) -> Path:
    """
    Save a comprehensive analysis report JSON.

    Args:
        report: Full report dict with model_name, tier1, tier2, tier3, metadata
        result_dir: Directory where the model predictions live
        update_index: Whether to update the master index file

    Returns:
        Path to saved JSON file
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"analysis_report_{timestamp}.json"
    report_path = result_dir / filename

    # Ensure report has required fields
    report.setdefault("evaluated_at", datetime.now(timezone.utc).isoformat())
    report.setdefault("report_path", str(report_path))

    clean_report = _make_serializable(report)

    with open(report_path, "w") as f:
        json.dump(clean_report, f, indent=2, default=str)

    print(f"  Saved: {report_path}")

    if update_index:
        _update_index(report, report_path)

    return report_path


def _update_index(report: dict, report_path: Path) -> None:
    """Update the master analysis index."""
    index = _load_index()

    # Extract key metrics for index entry
    t1_all = report.get("tier1", {}).get("All", {})
    t3_all = report.get("tier3", {}).get("All", {})
    t3_top1 = report.get("tier3", {}).get("1%", {})
    t3_top5 = report.get("tier3", {}).get("5%", {})
    t1_top5 = report.get("tier1", {}).get("5%", {})

    entry = {
        "model_name": report.get("model_name", "unknown"),
        "version": report.get("version", ""),
        "architecture": report.get("architecture", ""),
        "report_path": str(report_path),
        "evaluated_at": report.get("evaluated_at", ""),
        "n_predictions": t1_all.get("n", 0),
        "key_metrics": {
            "overall_ic": t1_all.get("ic", "NaN"),
            "overall_da": t1_all.get("da", "NaN"),
            "top5_ic": t1_top5.get("ic", "NaN") if t1_top5 else "NaN",
            "top5_da": t1_top5.get("da", "NaN") if t1_top5 else "NaN",
            "top1_net_pnl_ticks": t3_top1.get("total_pnl_ticks", "NaN") if t3_top1 else "NaN",
            "top5_net_pnl_ticks": t3_top5.get("total_pnl_ticks", "NaN") if t3_top5 else "NaN",
            "all_net_expectancy": t3_all.get("net_expectancy", "NaN"),
            "all_sortino": t3_all.get("sortino", "NaN"),
            "top5_sortino": t3_top5.get("sortino", "NaN") if t3_top5 else "NaN",
        },
        "verdict": report.get("verdict", ""),
    }

    # Replace existing entry for same model_name or append
    found = False
    for i, existing in enumerate(index.get("analyses", [])):
        if existing.get("model_name") == entry["model_name"]:
            index["analyses"][i] = entry
            found = True
            break
    if not found:
        index.setdefault("analyses", []).append(entry)

    index["last_updated"] = datetime.now(timezone.utc).isoformat()
    index["total_analyses"] = len(index["analyses"])

    clean_index = _make_serializable(index)
    with open(INDEX_PATH, "w") as f:
        json.dump(clean_index, f, indent=2, default=str)


def _load_index() -> dict:
    """Load the master index, creating if needed."""
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            return json.load(f)
    return {"analyses": [], "created_at": datetime.now(timezone.utc).isoformat()}


def load_analysis(
    model_dir: str | Path,
    latest_only: bool = True,
) -> list[dict] | dict | None:
    """
    Load analysis reports for a given model directory.

    Args:
        model_dir: Path to model result directory
        latest_only: If True, return only the most recent report

    Returns:
        Single report dict (if latest_only) or list of all reports
    """
    model_dir = Path(model_dir)
    report_files = sorted(model_dir.glob("analysis_report_*.json"))

    if not report_files:
        return None if latest_only else []

    if latest_only:
        with open(report_files[-1]) as f:
            return json.load(f)

    reports = []
    for rp in report_files:
        with open(rp) as f:
            reports.append(json.load(f))
    return reports


def list_all_analyses() -> list[dict]:
    """
    Load the master index and return a summary of all analyses.
    Also scans result directories for any un-indexed reports.
    """
    index = _load_index()
    return index.get("analyses", [])


def format_summary(analyses: list[dict] | None = None) -> str:
    """
    Produce a Discord-ready comparison across all archived analyses.
    Returns a markdown-formatted string.
    """
    if analyses is None:
        analyses = list_all_analyses()

    if not analyses:
        return "No analyses archived yet."

    lines = [
        "**Model Analysis Archive -- Comparison**",
        f"_{len(analyses)} models evaluated_",
        "",
        "```",
        f"{'Model':<30} {'N':>8} {'IC':>6} {'DA':>5} | {'Top5%DA':>7} {'Top5Srt':>7} {'Top1PnL':>9} | {'Verdict'}",
        "-" * 105,
    ]

    # Sort by overall IC descending
    def _get_ic(a):
        v = a.get("key_metrics", {}).get("overall_ic", "NaN")
        try:
            return float(v) if v != "NaN" else -999
        except (ValueError, TypeError):
            return -999

    sorted_analyses = sorted(analyses, key=_get_ic, reverse=True)

    for a in sorted_analyses:
        name = a.get("model_name", "?")[:30]
        n = a.get("n_predictions", 0)
        km = a.get("key_metrics", {})
        ic = _safe_fmt(km.get("overall_ic"), 3)
        da = _safe_fmt(km.get("overall_da"), 3)
        t5da = _safe_fmt(km.get("top5_da"), 3)
        t5sort = _safe_fmt(km.get("top5_sortino"), 1)
        t1pnl = _safe_fmt(km.get("top1_net_pnl_ticks"), 0)
        verdict = a.get("verdict", "")[:25]

        lines.append(
            f"{name:<30} {n:>8,} {ic:>6} {da:>5} | {t5da:>7} {t5sort:>7} {t1pnl:>9} | {verdict}"
        )

    lines.append("```")

    # Find best model
    if sorted_analyses:
        best = sorted_analyses[0]
        best_name = best.get("model_name", "?")
        best_ic = _safe_fmt(best.get("key_metrics", {}).get("overall_ic"), 3)
        lines.append(f"\nBest overall IC: **{best_name}** (IC={best_ic})")

    return "\n".join(lines)


def _safe_fmt(val, decimals: int) -> str:
    """Safely format a value that might be NaN string or actual number."""
    if val is None or val == "NaN" or val == "Inf" or val == "-Inf":
        return "N/A"
    try:
        v = float(val)
        if not np.isfinite(v):
            return "N/A"
        if decimals == 0:
            return f"{v:,.0f}"
        return f"{v:.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


# ---- Run and Archive ---------------------------------------------------------

def _determine_verdict(tier3: dict) -> str:
    """Determine profitability verdict from tier3 results."""
    profitable_tiers = []
    for tier_name in ["All", "50%", "25%", "10%", "5%", "1%"]:
        d = tier3.get(tier_name, {})
        exp = d.get("net_expectancy")
        if exp is not None and isinstance(exp, (int, float)) and np.isfinite(exp) and exp > 0:
            profitable_tiers.append(tier_name)

    if "All" in profitable_tiers:
        return "PROFITABLE at All tiers"
    elif profitable_tiers:
        return f"Profitable at {', '.join(profitable_tiers)} only"
    else:
        return "NOT PROFITABLE after costs"


def run_and_archive(
    model_info: dict,
    horizon: str = "10s",
    cost_ticks: float = DEFAULT_COST_TICKS,
) -> Optional[dict]:
    """
    Run tier3 evaluator on a model and archive the results.

    Args:
        model_info: Dict from MODEL_REGISTRY with pred_source, model_name, etc.
        horizon: Prediction horizon
        cost_ticks: Cost assumption in ticks

    Returns:
        Full report dict, or None if loading fails
    """
    name = model_info["model_name"]
    source = model_info["pred_source"]
    source_type = model_info.get("source_type", "file")
    is_classifier = model_info.get("is_classifier", False)

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"  Source: {source}")

    try:
        if source_type == "dir":
            pred_dir = Path(source)
            if not pred_dir.exists():
                print(f"  SKIP: directory not found")
                return None

            if is_classifier:
                # LGBM classifier: load all fold files and concatenate
                # Use 'confidence' as prediction magnitude, 'labels' as binary labels
                # Convert to regression-like format: pred_sign * confidence, binary label -> direction
                fold_files = sorted(pred_dir.glob("fold*_preds.npz"))
                if not fold_files:
                    print(f"  SKIP: no fold prediction files found")
                    return None

                all_preds, all_labels = [], []
                for ff in fold_files:
                    d = np.load(str(ff), allow_pickle=True)
                    # preds=0/1, probs=probability, labels=0/1, confidence=|prob-0.5|
                    preds_binary = d["preds"]  # 0 or 1
                    labels_binary = d["labels"]  # 0 or 1
                    confidence = d["confidence"]  # |prob - 0.5|

                    # Convert to signed prediction: direction * confidence
                    # 1 -> long (+1), 0 -> short (-1)
                    pred_dir_sign = preds_binary * 2 - 1  # maps 0->-1, 1->+1
                    signed_pred = pred_dir_sign.astype(np.float64) * confidence

                    # Labels: 1 -> up (+1), 0 -> down (-1)
                    # Use as proxy for actual move direction
                    signed_label = (labels_binary * 2 - 1).astype(np.float64)

                    all_preds.append(signed_pred)
                    all_labels.append(signed_label)

                predictions = np.concatenate(all_preds)
                labels = np.concatenate(all_labels)
                n_folds = len(fold_files)
            else:
                predictions, labels = load_predictions_dir(source, horizon)
                n_folds = len(list(pred_dir.glob("fold_*_oot*.npz")))
        else:
            pred_file = Path(source)
            if not pred_file.exists():
                print(f"  SKIP: file not found")
                return None
            predictions, labels = load_predictions(source, horizon)
            n_folds = None

        print(f"  Loaded {len(predictions):,} predictions")

    except Exception as e:
        print(f"  ERROR loading: {e}")
        return None

    # Run full evaluation
    try:
        results = evaluate_all_tiers(predictions, labels, cost_ticks)
    except Exception as e:
        print(f"  ERROR evaluating: {e}")
        return None

    # Build comprehensive report
    verdict = _determine_verdict(results["tier3"])

    report = {
        "model_name": model_info["model_name"],
        "version": model_info.get("version", ""),
        "architecture": model_info.get("architecture", ""),
        "features": model_info.get("features", ""),
        "walk_forward": model_info.get("walk_forward", {}),
        "n_folds_loaded": n_folds,
        "n_predictions": len(predictions),
        "horizon": horizon,
        "cost_assumption_ticks": cost_ticks,
        "cost_assumption_usd": cost_ticks * TICK_VALUE_USD,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "pred_source": source,
        "is_classifier": is_classifier,
        "tier1": results["tier1"],
        "tier2": results["tier2"],
        "tier3": results["tier3"],
        "summary": results["summary"],
        "verdict": verdict,
        "notes": model_info.get("notes", ""),
    }

    # Determine save directory
    source_path = Path(source)
    if source_type == "dir":
        save_dir = source_path
    else:
        save_dir = source_path.parent

    report_path = save_analysis(report, save_dir)
    print(f"  Verdict: {verdict}")

    return report


def run_all_models(
    horizon: str = "10s",
    cost_ticks: float = DEFAULT_COST_TICKS,
) -> list[dict]:
    """Run evaluation on all models in the registry and archive results."""
    print(f"Running analysis archive on {len(MODEL_REGISTRY)} models...")
    print(f"Horizon: {horizon}, Cost: {cost_ticks} ticks")

    results = []
    for model_info in MODEL_REGISTRY:
        report = run_and_archive(model_info, horizon, cost_ticks)
        if report:
            results.append(report)

    print(f"\n{'='*60}")
    print(f"Archived {len(results)} / {len(MODEL_REGISTRY)} models")
    print(f"Master index: {INDEX_PATH}")
    print(f"{'='*60}")

    return results


# ---- CLI Entry Point ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Model Analysis Archive System",
    )
    parser.add_argument("--run-all", action="store_true", help="Evaluate all registered models")
    parser.add_argument("--run-model", type=str, help="Run specific model by name (substring match)")
    parser.add_argument("--list", action="store_true", help="List all archived analyses")
    parser.add_argument("--summary", action="store_true", help="Discord-ready summary")
    parser.add_argument("--horizon", type=str, default="10s", help="Prediction horizon")
    parser.add_argument("--cost", type=float, default=DEFAULT_COST_TICKS, help="Cost in ticks")

    args = parser.parse_args()

    if args.run_all:
        reports = run_all_models(args.horizon, args.cost)
        if reports:
            print("\n" + format_summary())

    elif args.run_model:
        matching = [m for m in MODEL_REGISTRY if args.run_model.lower() in m["model_name"].lower()]
        if not matching:
            print(f"No model matching '{args.run_model}' found in registry")
            sys.exit(1)
        for m in matching:
            run_and_archive(m, args.horizon, args.cost)

    elif args.list:
        analyses = list_all_analyses()
        if not analyses:
            print("No analyses archived yet.")
        else:
            for a in analyses:
                km = a.get("key_metrics", {})
                print(f"  {a['model_name']:<35} IC={_safe_fmt(km.get('overall_ic'), 3):>6}  "
                      f"DA={_safe_fmt(km.get('overall_da'), 3):>5}  "
                      f"N={a.get('n_predictions', 0):>10,}  "
                      f"{a.get('verdict', '')}")

    elif args.summary:
        print(format_summary())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
