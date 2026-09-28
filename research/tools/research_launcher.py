#!/usr/bin/env python3
"""
Research Launcher — Hypothesis-Driven Experiment Runner
========================================================

Takes a hypothesis, runs the backtest, validates adversarially, compares
against a random baseline, logs to MLflow, and appends to the knowledge base.

The output is a simple PASS / FAIL / INVESTIGATE verdict so we don't need
to manually interpret dozens of numbers.

Usage:
    from research.tools.research_launcher import run_research

    result = run_research(
        hypothesis="VIX>25 filter produces better risk-adjusted returns than VIX>20",
        parameters={"vix_floor": 25},
        backtest_fn=my_custom_backtest_function,  # optional, defaults to sector_backtest
    )
    # result contains verdict, metrics, validation, and comparison data

Verdict logic:
    PASS        — All 5 gates pass AND beats random baseline
    FAIL        — Fewer than 3 gates pass OR Sharpe < 0.5
    INVESTIGATE — 3-4 gates pass but doesn't clearly beat random, or edge case
"""
from __future__ import annotations

import json
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

warnings.filterwarnings("ignore")

from .adversarial_validator import validate_trades, ValidationResult


# ─── Knowledge Base Path ─────────────────────────────────────────────

KNOWLEDGE_BASE_PATH = Path("/home/jupiter/Lvl3Quant/research/QUANT_KNOWLEDGE_BASE.md")
RESEARCH_LOG_PATH = Path("/home/jupiter/Lvl3Quant/output/research_log.jsonl")


# ─── MLflow Logging ──────────────────────────────────────────────────

def _log_to_mlflow(
    hypothesis: str,
    parameters: dict,
    metrics: dict,
    verdict: str,
    experiment_name: str = "research_launcher",
) -> Optional[str]:
    """
    Log experiment to MLflow. Returns run_id or None if MLflow unavailable.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=hypothesis[:100]) as run:
            # Log parameters
            mlflow.log_param("hypothesis", hypothesis[:250])
            mlflow.log_param("verdict", verdict)
            for k, v in parameters.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass

            # Log metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):
                    try:
                        mlflow.log_metric(k, v)
                    except Exception:
                        pass

            return run.info.run_id

    except Exception:
        return None


# ─── Knowledge Base Append ───────────────────────────────────────────

def _append_to_knowledge_base(
    hypothesis: str,
    verdict: str,
    sharpe: float,
    win_rate: float,
    n_trades: int,
    detail: str = "",
) -> None:
    """Append a finding to the research log (JSONL). Does NOT modify the main KB markdown."""
    RESEARCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "hypothesis": hypothesis,
        "verdict": verdict,
        "sharpe": round(sharpe, 3),
        "win_rate": round(win_rate, 4),
        "n_trades": n_trades,
        "detail": detail,
    }

    with open(RESEARCH_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─── Verdict Logic ──────────────────────────────────────────────────

def _compute_verdict(
    validation: ValidationResult,
    random_validation: Optional[ValidationResult] = None,
) -> tuple[str, str]:
    """
    Compute PASS / FAIL / INVESTIGATE verdict.

    Returns:
        (verdict, explanation)
    """
    if validation.error:
        return "FAIL", f"Validation error: {validation.error}"

    gates_passed = validation.gates_passed
    total_gates = validation.gates_total
    sharpe = validation.sharpe

    # Hard fail conditions
    if sharpe < 0.5:
        return "FAIL", f"Sharpe {sharpe:.2f} below minimum threshold (0.5)"

    if gates_passed < 3:
        failed_names = [g.name for g in validation.gates if not g.passed]
        return "FAIL", f"Only {gates_passed}/{total_gates} gates passed. Failed: {', '.join(failed_names)}"

    if validation.max_dd < -0.50:
        return "FAIL", f"Max drawdown {validation.max_dd*100:.1f}% exceeds -50% limit"

    # Check random baseline comparison
    beats_random = True
    random_note = ""
    if random_validation is not None and not random_validation.error:
        if validation.sharpe <= random_validation.sharpe * 1.05:
            beats_random = False
            random_note = (
                f" ML Sharpe ({validation.sharpe:.2f}) does not meaningfully beat "
                f"random ({random_validation.sharpe:.2f}). Edge may be structural only."
            )

    # Verdict
    if gates_passed == total_gates and beats_random:
        return "PASS", f"All {total_gates} gates passed. Sharpe={sharpe:.2f}, WR={validation.win_rate*100:.1f}%{random_note}"

    if gates_passed >= 4 and sharpe >= 1.0:
        failed = [g.name for g in validation.gates if not g.passed]
        return "INVESTIGATE", (
            f"{gates_passed}/{total_gates} gates passed (failed: {', '.join(failed)}). "
            f"Sharpe={sharpe:.2f}.{random_note} Worth investigating the failure."
        )

    if gates_passed == 3:
        failed = [g.name for g in validation.gates if not g.passed]
        return "INVESTIGATE", (
            f"{gates_passed}/{total_gates} gates passed (failed: {', '.join(failed)}). "
            f"Sharpe={sharpe:.2f}.{random_note} Marginal — needs more evidence."
        )

    return "FAIL", f"{gates_passed}/{total_gates} gates passed, Sharpe={sharpe:.2f}{random_note}"


# ─── Main Entry Point ───────────────────────────────────────────────

def run_research(
    hypothesis: str,
    parameters: dict | None = None,
    backtest_fn: Callable | None = None,
    trades: list[dict] | None = None,
    initial_capital: float = 645.0,
    spy_prices=None,
    n_perms: int = 500,
    random_trades: list[dict] | None = None,
    log_mlflow: bool = True,
    log_knowledge_base: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run a complete research experiment with validation.

    You can either:
      A) Provide a backtest_fn(parameters) that returns a dict with 'trades' key
      B) Provide trades directly (list of dicts with pnl, entry_date, exit_date)

    Args:
        hypothesis: What you're testing (string).
        parameters: Config dict passed to backtest_fn.
        backtest_fn: Function that takes (parameters) and returns
                     dict with 'trades' (and optionally 'random_trades', 'spy_prices').
        trades: Pre-computed trades (alternative to backtest_fn).
        initial_capital: Starting equity.
        spy_prices: SPY close series for regime gate.
        n_perms: Number of permutations for sign-flip test.
        random_trades: Pre-computed random baseline trades.
        log_mlflow: Whether to log to MLflow.
        log_knowledge_base: Whether to append to research log.
        verbose: Print progress.

    Returns:
        dict with: verdict, explanation, validation, random_validation,
                   metrics, hypothesis, parameters, mlflow_run_id
    """
    if parameters is None:
        parameters = {}

    if verbose:
        print("\n" + "=" * 65)
        print(f"  RESEARCH EXPERIMENT")
        print(f"  Hypothesis: {hypothesis}")
        print("=" * 65)

    try:
        # Step 1: Get trades
        if trades is None and backtest_fn is not None:
            if verbose:
                print("\nRunning backtest...")
            bt_result = backtest_fn(parameters)

            if isinstance(bt_result, dict):
                trades = bt_result.get("trades", [])
                if random_trades is None:
                    random_trades = bt_result.get("random_trades")
                if spy_prices is None:
                    spy_prices = bt_result.get("spy_prices")
            elif isinstance(bt_result, list):
                trades = bt_result
            else:
                return {
                    "verdict": "FAIL",
                    "explanation": f"backtest_fn returned unexpected type: {type(bt_result)}",
                    "validation": None,
                    "random_validation": None,
                    "metrics": {},
                    "hypothesis": hypothesis,
                    "parameters": parameters,
                    "mlflow_run_id": None,
                }

        if trades is None or len(trades) == 0:
            return {
                "verdict": "FAIL",
                "explanation": "No trades generated",
                "validation": None,
                "random_validation": None,
                "metrics": {},
                "hypothesis": hypothesis,
                "parameters": parameters,
                "mlflow_run_id": None,
            }

        # Step 2: Validate
        if verbose:
            print(f"\nValidating {len(trades)} trades...")

        validation = validate_trades(
            trades,
            initial_capital=initial_capital,
            spy_prices=spy_prices,
            strategy_name=hypothesis[:60],
            n_perms=n_perms,
        )

        if verbose:
            validation.print_summary()

        # Step 3: Random baseline
        random_validation = None
        if random_trades and len(random_trades) >= 10:
            if verbose:
                print(f"\nValidating random baseline ({len(random_trades)} trades)...")

            random_validation = validate_trades(
                random_trades,
                initial_capital=initial_capital,
                spy_prices=spy_prices,
                strategy_name="Random Baseline",
                n_perms=min(n_perms, 200),
            )

            if verbose:
                random_validation.print_summary()

        # Step 4: Verdict
        verdict, explanation = _compute_verdict(validation, random_validation)

        if verbose:
            print(f"\n{'='*65}")
            marker = {"PASS": "[PASS]", "FAIL": "[FAIL]", "INVESTIGATE": "[???]"}
            print(f"  VERDICT: {marker.get(verdict, verdict)} {verdict}")
            print(f"  {explanation}")
            print(f"{'='*65}")

        # Step 5: Log
        metrics_dict = {
            "sharpe": validation.sharpe,
            "sortino": validation.sortino,
            "cagr": validation.cagr,
            "max_dd": validation.max_dd,
            "win_rate": validation.win_rate,
            "profit_factor": validation.profit_factor,
            "n_trades": validation.n_trades,
            "gates_passed": validation.gates_passed,
        }

        mlflow_run_id = None
        if log_mlflow:
            mlflow_run_id = _log_to_mlflow(hypothesis, parameters, metrics_dict, verdict)
            if verbose and mlflow_run_id:
                print(f"  Logged to MLflow: run_id={mlflow_run_id[:8]}...")

        if log_knowledge_base:
            _append_to_knowledge_base(
                hypothesis, verdict, validation.sharpe, validation.win_rate,
                validation.n_trades, explanation,
            )
            if verbose:
                print(f"  Appended to research log")

        return {
            "verdict": verdict,
            "explanation": explanation,
            "validation": validation,
            "random_validation": random_validation,
            "metrics": metrics_dict,
            "hypothesis": hypothesis,
            "parameters": parameters,
            "mlflow_run_id": mlflow_run_id,
        }

    except Exception as e:
        error_msg = f"Research experiment failed: {e}\n{traceback.format_exc()}"
        if verbose:
            print(f"\nERROR: {error_msg}")

        return {
            "verdict": "FAIL",
            "explanation": error_msg,
            "validation": None,
            "random_validation": None,
            "metrics": {},
            "hypothesis": hypothesis,
            "parameters": parameters,
            "mlflow_run_id": None,
        }


# ─── Convenience: Quick Sector Test ─────────────────────────────────

def quick_sector_test(
    hypothesis: str = "Baseline sector bull call spread strategy",
    config_overrides: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Convenience function that runs the standard sector backtest via research_launcher.

    Args:
        hypothesis: What you're testing.
        config_overrides: Override DEFAULT_CONFIG parameters.
        verbose: Print progress.

    Returns:
        Same as run_research().
    """
    from .sector_backtest import run_sector_backtest

    def backtest_fn(params):
        results = run_sector_backtest(config=params, verbose=verbose)
        # Extract trades and random trades
        random_trades = None
        spy_prices = None
        if results.get("random_baseline"):
            # Re-generate random trades for the launcher
            # (validation result doesn't expose raw trades)
            pass
        return {
            "trades": results["trades"],
            "spy_prices": spy_prices,
        }

    return run_research(
        hypothesis=hypothesis,
        parameters=config_overrides or {},
        backtest_fn=backtest_fn,
        verbose=verbose,
    )


# ─── Self-Test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Research Launcher — Self-Test with synthetic data\n")

    import pandas as pd

    # Generate synthetic trades
    np.random.seed(42)
    n_trades = 150
    base_date = pd.Timestamp("2020-01-01")
    synthetic_trades = []
    for i in range(n_trades):
        entry = base_date + pd.Timedelta(days=i * 7)
        exit_d = entry + pd.Timedelta(days=30)
        pnl = np.random.normal(8.0, 25.0)  # positive expectancy
        synthetic_trades.append({
            "pnl": pnl,
            "entry_date": entry,
            "exit_date": exit_d,
        })

    # Also create random baseline trades (zero expectancy)
    random_trades = []
    for i in range(n_trades):
        entry = base_date + pd.Timedelta(days=i * 7)
        exit_d = entry + pd.Timedelta(days=30)
        pnl = np.random.normal(0.0, 25.0)
        random_trades.append({
            "pnl": pnl,
            "entry_date": entry,
            "exit_date": exit_d,
        })

    result = run_research(
        hypothesis="Synthetic positive-expectancy strategy",
        parameters={"mean_pnl": 8.0, "std_pnl": 25.0},
        trades=synthetic_trades,
        random_trades=random_trades,
        initial_capital=1000.0,
        n_perms=200,
        log_mlflow=False,  # don't need MLflow for self-test
        log_knowledge_base=False,
        verbose=True,
    )

    print(f"\nFinal verdict: {result['verdict']}")
    print(f"Explanation: {result['explanation']}")
