#!/usr/bin/env python3
"""
Adversarial Strategy Auditor — HC #753
========================================

Reusable audit script that checks ANY strategy research script for:
  1. Label leakage (forward-looking data in features/labels)
  2. Walk-forward contamination (train/test overlap, scaler leakage)
  3. Pricing errors (BS assumptions, commission math)
  4. Statistical validity (sample size, Sharpe method, outlier influence, regime concentration)
  5. Code pattern scanning (dangerous AST patterns)

Usage (CLI):
    python adversarial_strategy_auditor.py path/to/strategy_script.py
    python adversarial_strategy_auditor.py path/to/strategy_script.py --results path/to/results.json

Usage (module):
    from adversarial_strategy_auditor import run_audit
    report = run_audit("path/to/script.py", results_json="path/to/results.json")
    print(report)

Self-contained: stdlib + numpy + pandas only.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ─── Result Types ──────────────────────────────────────────────────────

@dataclass
class Finding:
    """A single audit finding."""
    category: str       # e.g. "LABEL_LEAKAGE", "WF_CONTAMINATION", etc.
    severity: str       # "CRITICAL", "WARNING", "INFO"
    message: str
    location: str = ""  # file:line or description
    evidence: str = ""  # code snippet or data

    def __str__(self) -> str:
        loc = f" @ {self.location}" if self.location else ""
        ev = f"\n      Evidence: {self.evidence}" if self.evidence else ""
        return f"  [{self.severity}] {self.category}{loc}: {self.message}{ev}"


@dataclass
class AuditReport:
    """Complete audit report."""
    script_path: str
    findings: list[Finding] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    passed: bool = True

    def add(self, f: Finding):
        self.findings.append(f)
        if f.severity == "CRITICAL":
            self.passed = False

    @property
    def verdict(self) -> str:
        crits = sum(1 for f in self.findings if f.severity == "CRITICAL")
        warns = sum(1 for f in self.findings if f.severity == "WARNING")
        if crits > 0:
            return f"FAIL — {crits} critical finding(s), {warns} warning(s)"
        if warns > 0:
            return f"PASS WITH WARNINGS — {warns} warning(s)"
        return "PASS — no issues found"

    def __str__(self) -> str:
        lines = [
            "=" * 70,
            "ADVERSARIAL STRATEGY AUDIT REPORT",
            "=" * 70,
            f"Script: {self.script_path}",
            f"Checks: {len(self.checks_run)}",
            f"Findings: {len(self.findings)}",
            f"Verdict: {self.verdict}",
            "-" * 70,
        ]

        # Group by category
        cats = {}
        for f in self.findings:
            cats.setdefault(f.category, []).append(f)

        for cat, findings in cats.items():
            lines.append(f"\n[{cat}]")
            for f in findings:
                lines.append(str(f))

        if not self.findings:
            lines.append("\n  No issues found.")

        lines.append("\n" + "=" * 70)
        lines.append(f"FINAL VERDICT: {self.verdict}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "script_path": self.script_path,
            "passed": self.passed,
            "verdict": self.verdict,
            "checks_run": self.checks_run,
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity,
                    "message": f.message,
                    "location": f.location,
                    "evidence": f.evidence,
                }
                for f in self.findings
            ],
        }


# ─── AST Helpers ───────────────────────────────────────────────────────

def _get_source_line(source_lines: list[str], lineno: int) -> str:
    """Get source line safely (1-indexed)."""
    if 1 <= lineno <= len(source_lines):
        return source_lines[lineno - 1].rstrip()
    return ""


def _node_to_str(node: ast.AST) -> str:
    """Best-effort conversion of AST node to source string."""
    try:
        return ast.unparse(node)
    except Exception:
        return repr(node)


class _ShiftDetector(ast.NodeVisitor):
    """Detect .shift(-N) calls — potential forward-looking labels/features."""

    def __init__(self, source_lines: list[str]):
        self.source_lines = source_lines
        self.negative_shifts: list[tuple[int, str, str]] = []  # (lineno, context, code)
        self._current_func: str = "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef):
        old = self._current_func
        self._current_func = node.name
        self.generic_visit(node)
        self._current_func = old

    def visit_Call(self, node: ast.Call):
        # Look for .shift(-N)
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr == "shift"
                and node.args):
            arg = node.args[0]
            is_negative = False
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                is_negative = True
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                if arg.value < 0:
                    is_negative = True

            if is_negative:
                line = _get_source_line(self.source_lines, node.lineno)
                self.negative_shifts.append(
                    (node.lineno, self._current_func, line)
                )
        self.generic_visit(node)


class _FitTransformDetector(ast.NodeVisitor):
    """Detect fit_transform on potentially full dataset (not train-only)."""

    def __init__(self, source_lines: list[str]):
        self.source_lines = source_lines
        self.suspicious_fits: list[tuple[int, str, str]] = []
        self._current_func: str = "<module>"
        self._in_wf_loop = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        old = self._current_func
        self._current_func = node.name
        self.generic_visit(node)
        self._current_func = old

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method in ("fit_transform", "fit"):
                line = _get_source_line(self.source_lines, node.lineno)
                # Check if the argument looks like it includes test data
                # Flag if not obviously train-only
                arg_str = ""
                if node.args:
                    arg_str = _node_to_str(node.args[0])

                # Heuristic: if the argument name doesn't contain "train" or "tr",
                # it might be fitting on the full dataset
                train_indicators = ["train", "_tr", "x_tr", "xt", "x_train"]
                arg_lower = arg_str.lower()
                looks_train_only = any(t in arg_lower for t in train_indicators)
                if not looks_train_only:
                    self.suspicious_fits.append(
                        (node.lineno, self._current_func, line)
                    )
        self.generic_visit(node)


class _ExpandingWindowDetector(ast.NodeVisitor):
    """Detect expanding window usage (banned per HC #0)."""

    def __init__(self, source_lines: list[str]):
        self.source_lines = source_lines
        self.expanding_uses: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "expanding":
                line = _get_source_line(self.source_lines, node.lineno)
                self.expanding_uses.append((node.lineno, line))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr == "expanding":
            line = _get_source_line(self.source_lines, node.lineno)
            self.expanding_uses.append((node.lineno, line))
        self.generic_visit(node)


class _FutureIndexDetector(ast.NodeVisitor):
    """Detect iloc[-N] patterns that access future data from a full dataframe.

    NOTE: iloc[-1] on a windowed/sliced series (e.g., px.iloc[-1] where px is a
    lookback slice) is NORMAL for feature computation. We only flag patterns where
    the negative index is applied to what looks like a full dataframe column (e.g.,
    df.iloc[-1] or df["col"].iloc[-1]) rather than a local variable that's been
    sliced to a lookback window.
    """

    def __init__(self, source_lines: list[str]):
        self.source_lines = source_lines
        self.suspicious_iloc: list[tuple[int, str, str]] = []
        self._current_func: str = "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef):
        old = self._current_func
        self._current_func = node.name
        self.generic_visit(node)
        self._current_func = old

    def visit_Subscript(self, node: ast.Subscript):
        # Look for .iloc[negative_index] patterns on full dataframes
        if (isinstance(node.value, ast.Attribute)
                and node.value.attr == "iloc"):
            slc = node.slice
            if isinstance(slc, ast.UnaryOp) and isinstance(slc.op, ast.USub):
                line = _get_source_line(self.source_lines, node.lineno)
                func_name = self._current_func.lower()

                # Only flag if the object looks like a full dataframe column
                # (e.g., df["col"].iloc[-1] or df.col.iloc[-1])
                # Skip if it looks like a local variable (px, rets, r63, etc.)
                # which are typically windowed slices
                obj = node.value.value  # the thing before .iloc
                obj_str = _node_to_str(obj).lower()

                # Heuristic: flag if object is df["something"].iloc[-N] or
                # data["something"].iloc[-N] — these are full-column accesses
                looks_full_df = any(kw in obj_str for kw in
                                    ["df[", "data[", "dataframe", "full_", "all_"])
                # Skip common local variable patterns (windowed lookback slices)
                # and patterns like t_data[cols].iloc[-1] which get latest row in WF
                looks_windowed = any(kw in obj_str for kw in
                                     ["px", "rets", "ret", "close", "high", "low",
                                      "r21", "r63", "monthly", "daily", "rolling",
                                      "gld", "spy", "shy", "hyg", "tlt", "vix",
                                      "t_data", "train", "test", "values",
                                      "feature_col", "feature_", "_cols"])

                if looks_full_df and not looks_windowed:
                    self.suspicious_iloc.append(
                        (node.lineno, self._current_func, line)
                    )
        self.generic_visit(node)


# ─── Check 1: Label Leakage (AST-based) ───────────────────────────────

def check_label_leakage(source: str, source_lines: list[str], tree: ast.Module,
                        report: AuditReport):
    """Check for forward-looking data usage in labels/features."""
    report.checks_run.append("LABEL_LEAKAGE")

    # 1a: .shift(-N) detection
    detector = _ShiftDetector(source_lines)
    detector.visit(tree)

    for lineno, func, code in detector.negative_shifts:
        # Classify: is this a label creation or feature creation?
        code_lower = code.lower()
        is_label = any(w in code_lower for w in
                       ["fwd_ret", "fwd_return", "forward", "label", "target",
                        "rank_label", "future"])
        is_feature = any(w in code_lower for w in
                         ["feature", "feat", "signal", "indicator"])

        if is_label:
            # Negative shift for label creation is EXPECTED — this is how you
            # create forward returns. But we should verify it's not used
            # in the feature set.
            report.add(Finding(
                category="LABEL_LEAKAGE",
                severity="INFO",
                message=f"Forward shift used for label creation in '{func}' — verify this label is NOT included in features",
                location=f"line {lineno}",
                evidence=code.strip(),
            ))
        elif is_feature:
            # Negative shift in feature computation is CRITICAL — features
            # should never look forward
            report.add(Finding(
                category="LABEL_LEAKAGE",
                severity="CRITICAL",
                message=f"Forward shift (.shift(-N)) found in feature computation '{func}' — this is future information leakage",
                location=f"line {lineno}",
                evidence=code.strip(),
            ))
        else:
            # Unknown context — flag as warning
            report.add(Finding(
                category="LABEL_LEAKAGE",
                severity="WARNING",
                message=f"Forward shift (.shift(-N)) in '{func}' — verify this is label creation, NOT feature computation",
                location=f"line {lineno}",
                evidence=code.strip(),
            ))

    # 1b: Check if any feature lists include "fwd_ret" or similar forward-looking names
    # Match explicit list literals like FEATURES = ["a", "b", "fwd_ret"]
    feature_list_pattern = re.compile(
        r'(?:FEATURES|feature_cols|feature_list|BASE_FEATURES|FLOW_FEATURES)\s*=\s*\[([^\]]+)\]',
        re.DOTALL
    )
    for m in feature_list_pattern.finditer(source):
        full_match = m.group(0)
        feature_text = m.group(1).lower()
        suspicious_names = ["fwd_ret", "forward_ret", "future", "label", "target"]

        # Skip exclusion patterns like: [c for c in df.columns if c not in ['fwd_ret']]
        is_exclusion = ("not in" in full_match.lower() or
                        "!=" in full_match or
                        "if c not" in full_match.lower() or
                        "exclude" in full_match.lower())
        if is_exclusion:
            continue

        for name in suspicious_names:
            if name in feature_text:
                start = source[:m.start()].count("\n") + 1
                report.add(Finding(
                    category="LABEL_LEAKAGE",
                    severity="CRITICAL",
                    message=f"Feature list contains forward-looking name '{name}' — this is almost certainly label leakage",
                    location=f"~line {start}",
                    evidence=full_match[:200],
                ))


# ─── Check 2: Walk-Forward Contamination ──────────────────────────────

def check_wf_contamination(source: str, source_lines: list[str], tree: ast.Module,
                           report: AuditReport):
    """Check for train/test contamination in walk-forward."""
    report.checks_run.append("WF_CONTAMINATION")

    # 2a: fit_transform / fit on non-train data
    detector = _FitTransformDetector(source_lines)
    detector.visit(tree)

    for lineno, func, code in detector.suspicious_fits:
        report.add(Finding(
            category="WF_CONTAMINATION",
            severity="WARNING",
            message=f"fit/fit_transform in '{func}' — argument doesn't clearly indicate train-only data. "
                    f"Fitting a scaler/model on train+test leaks test distribution info.",
            location=f"line {lineno}",
            evidence=code.strip(),
        ))

    # 2b: Expanding window usage (banned per HC #0)
    ew_detector = _ExpandingWindowDetector(source_lines)
    ew_detector.visit(tree)
    for lineno, code in ew_detector.expanding_uses:
        report.add(Finding(
            category="WF_CONTAMINATION",
            severity="CRITICAL",
            message="Expanding window detected — BANNED per HC #0. Must use SLIDING windows only.",
            location=f"line {lineno}",
            evidence=code.strip(),
        ))

    # 2c: Look for train/test date overlap patterns
    # Check for patterns where train_end >= test_start or similar
    overlap_patterns = [
        (r'train_df\s*=\s*df\b', "Check train_df filtering uses strict date boundary"),
        (r'(?:test|val)_df\s*=\s*df\b', "Check test_df filtering uses strict date boundary"),
    ]
    for pat, msg in overlap_patterns:
        for m in re.finditer(pat, source):
            lineno = source[:m.start()].count("\n") + 1
            line = _get_source_line(source_lines, lineno)
            # Only flag if it doesn't have a date filter
            if "[" not in line and "loc" not in line and "query" not in line:
                report.add(Finding(
                    category="WF_CONTAMINATION",
                    severity="WARNING",
                    message=f"Possible full-dataset assignment: {msg}",
                    location=f"line {lineno}",
                    evidence=line.strip(),
                ))

    # 2d: Check for TimeSeriesSplit or proper WF structure
    has_wf = bool(re.search(r'walk.?forward|sliding.*window|WF_TRAIN', source, re.IGNORECASE))
    has_ts_split = "TimeSeriesSplit" in source
    has_manual_wf = bool(re.search(r'train_dates|train_end|train_start', source))

    if not (has_wf or has_ts_split or has_manual_wf):
        # Check if there's any ML model being used
        has_ml = any(kw in source for kw in [
            "LGBMRegressor", "LGBMClassifier", "RandomForest", "GradientBoosting",
            "XGBRegressor", "LinearRegression", "LogisticRegression",
            "lgb.train", "xgb.train", "model.fit",
        ])
        if has_ml:
            report.add(Finding(
                category="WF_CONTAMINATION",
                severity="CRITICAL",
                message="ML model found but no walk-forward/sliding window structure detected. "
                        "Training on full dataset then testing on same data = information leakage.",
                location="entire file",
            ))


# ─── Check 3: Pricing Errors ──────────────────────────────────────────

def check_pricing_errors(source: str, source_lines: list[str], report: AuditReport):
    """Check for BS pricing assumption errors and commission math."""
    report.checks_run.append("PRICING_ERRORS")

    # 3a: Commission validation
    # Expected: $0.65/leg x 4 legs = $2.60 per spread RT
    commission_patterns = [
        (r'COMMISSION.*?=\s*([\d.]+)', "commission constant"),
        (r'SPREAD_COMM.*?=\s*([\d.]+)', "spread commission"),
        (r'commission.*?=\s*([\d.]+)', "commission variable"),
        (r'COMM.*?RT.*?=\s*([\d.]+)', "RT commission"),
    ]

    for pat, desc in commission_patterns:
        for m in re.finditer(pat, source, re.IGNORECASE):
            val = float(m.group(1))
            lineno = source[:m.start()].count("\n") + 1
            line = _get_source_line(source_lines, lineno)

            # For spread strategies: $2.60 RT (4 legs x $0.65)
            # Allow $2.60 or $0.65 per leg
            if val not in (0.65, 2.60, 1.30):
                report.add(Finding(
                    category="PRICING_ERRORS",
                    severity="WARNING",
                    message=f"Commission ${val:.2f} doesn't match expected $2.60/spread RT ($0.65/leg x 4). "
                            f"Verify this is correct for your instrument.",
                    location=f"line {lineno}",
                    evidence=line.strip(),
                ))

    # 3b: Haircut validation
    haircut_pat = re.compile(r'HAIRCUT\s*=\s*([\d.]+)')
    for m in haircut_pat.finditer(source):
        val = float(m.group(1))
        lineno = source[:m.start()].count("\n") + 1
        if val < 0.10:
            report.add(Finding(
                category="PRICING_ERRORS",
                severity="WARNING",
                message=f"Haircut {val:.0%} is below 10% — may underestimate bid-ask spread cost.",
                location=f"line {lineno}",
            ))
        elif val > 0.25:
            report.add(Finding(
                category="PRICING_ERRORS",
                severity="INFO",
                message=f"Haircut {val:.0%} is conservative (>25%). Results may be pessimistic.",
                location=f"line {lineno}",
            ))

    # 3c: Entry cost calculation pattern
    # Expected: entry_cost = debit_per_share * 100 + commission
    # Or: entry_cost_ps = fair * (1 + haircut)  [per-share, then * 100 for per-contract]
    if "entry_cost" in source.lower() or "debit" in source.lower():
        # Check for missing * 100 multiplier
        entry_patterns = re.findall(
            r'entry_cost\s*=\s*([^\n;]+)',
            source, re.IGNORECASE
        )
        for ep in entry_patterns:
            if "100" not in ep and "n_contracts" not in ep.lower() and "multiplier" not in ep.lower():
                # Might be per-share already — not necessarily wrong
                pass

    # 3d: P&L calculation check
    # Expected: pnl = (intrinsic_value - entry_cost_ps) * 100 - commission
    # Or: pnl = (exit_value - entry_cost) * contracts * 100 - commission
    # Only flag actual P&L computation lines, not aggregations or permutation shuffles
    pnl_assignment_re = re.compile(r'^\s*(?:\w+\[?"?pnl"?\]?\s*=|pnl\s*=)\s*(.+)', re.IGNORECASE | re.MULTILINE)
    for m in pnl_assignment_re.finditer(source):
        pp = m.group(1)
        pp_lower = pp.lower()
        # Skip aggregation/groupby/shuffle patterns — these aren't P&L computations
        skip_patterns = ["groupby", "sum()", "mean()", "float(s)", "dict(", "shuffle",
                         "append", "extend", ".pnl", "for ", "zip(",
                         "np.array", "np.sum", "pd.series", "list(", "copy()"]
        if any(sp in pp_lower for sp in skip_patterns):
            continue
        # Check commission is subtracted in actual P&L calculations
        if ("commiss" not in pp_lower and "comm" not in pp_lower
                and "cost" not in pp_lower and "fee" not in pp_lower
                and len(pp.strip()) > 10):
            lineno = source[:m.start()].count("\n") + 1
            report.add(Finding(
                category="PRICING_ERRORS",
                severity="WARNING",
                message="P&L calculation may not include commission deduction.",
                location=f"line {lineno}",
                evidence=f"pnl = {pp.strip()[:120]}",
            ))

    # 3e: Risk-free rate sanity
    rf_pat = re.compile(r'RISK_FREE_RATE\s*=\s*([\d.]+)')
    for m in rf_pat.finditer(source):
        val = float(m.group(1))
        lineno = source[:m.start()].count("\n") + 1
        if val > 0.08 or val < 0.01:
            report.add(Finding(
                category="PRICING_ERRORS",
                severity="WARNING",
                message=f"Risk-free rate {val:.1%} looks unusual. Current ~4.5%.",
                location=f"line {lineno}",
            ))

    # 3f: IV multiplier check
    iv_pat = re.compile(r'IV_MULTIPLIER\s*=\s*([\d.]+)')
    for m in iv_pat.finditer(source):
        val = float(m.group(1))
        lineno = source[:m.start()].count("\n") + 1
        if val < 1.0:
            report.add(Finding(
                category="PRICING_ERRORS",
                severity="WARNING",
                message=f"IV multiplier {val} < 1.0 means IV below realized vol — unrealistic for options pricing.",
                location=f"line {lineno}",
            ))


# ─── Check 4: Statistical Validity (requires results JSON) ────────────

def check_statistical_validity(results: dict, report: AuditReport):
    """Check statistical rigor of reported results."""
    report.checks_run.append("STATISTICAL_VALIDITY")

    # Try to find trade data and metrics from various result formats
    trades = results.get("trades", [])
    variants = results.get("variants", results.get("results", {}))

    # If results is a flat dict with metrics
    if not trades and not variants:
        _check_single_result(results, "main", report)
        return

    # If results has variant structure
    if isinstance(variants, dict):
        for vname, vdata in variants.items():
            if isinstance(vdata, dict):
                _check_single_result(vdata, vname, report)
    elif isinstance(variants, list):
        for i, vdata in enumerate(variants):
            name = vdata.get("name", f"variant_{i}") if isinstance(vdata, dict) else f"variant_{i}"
            if isinstance(vdata, dict):
                _check_single_result(vdata, name, report)


def _check_single_result(data: dict, name: str, report: AuditReport):
    """Check a single variant/strategy result dict."""
    try:
        import numpy as np
    except ImportError:
        report.add(Finding(
            category="STATISTICAL_VALIDITY",
            severity="INFO",
            message="numpy not available — skipping numerical checks",
        ))
        return

    # 4a: Sample size check
    n_trades = data.get("n_trades", data.get("trade_count", data.get("num_trades", 0)))
    if n_trades > 0:
        if n_trades < 30:
            report.add(Finding(
                category="STATISTICAL_VALIDITY",
                severity="CRITICAL",
                message=f"[{name}] Only {n_trades} trades — far too few for reliable Sharpe/stats. "
                        f"Minimum ~100 for 2-decimal Sharpe precision.",
                location=name,
            ))
        elif n_trades < 100:
            report.add(Finding(
                category="STATISTICAL_VALIDITY",
                severity="WARNING",
                message=f"[{name}] {n_trades} trades — borderline sample size. "
                        f"Sharpe SE ~ 1/sqrt(N) = {1/math.sqrt(n_trades):.2f}.",
                location=name,
            ))

    # 4b: Sharpe method check
    sharpe = data.get("sharpe", data.get("sharpe_ratio", data.get("monthly_sharpe", None)))
    if sharpe is not None:
        if abs(sharpe) > 5.0:
            report.add(Finding(
                category="STATISTICAL_VALIDITY",
                severity="WARNING",
                message=f"[{name}] Sharpe {sharpe:.2f} is extremely high (>5.0). "
                        f"Verify: (a) computed from monthly returns, (b) not overfit, (c) enough samples.",
                location=name,
            ))
        elif abs(sharpe) > 3.0 and n_trades < 200:
            report.add(Finding(
                category="STATISTICAL_VALIDITY",
                severity="WARNING",
                message=f"[{name}] Sharpe {sharpe:.2f} with only {n_trades} trades — high chance of overfit.",
                location=name,
            ))

    # 4c: Win rate sanity
    win_rate = data.get("win_rate", data.get("wr", None))
    if win_rate is not None:
        if win_rate > 0.85:
            report.add(Finding(
                category="STATISTICAL_VALIDITY",
                severity="WARNING",
                message=f"[{name}] Win rate {win_rate:.1%} is suspiciously high. "
                        f"Verify not caused by lookahead bias or trivial trades.",
                location=name,
            ))

    # 4d: Outlier influence — check if top trades dominate
    trades = data.get("trades", data.get("trade_list", []))
    if trades and isinstance(trades, list) and len(trades) >= 10:
        pnls = []
        for t in trades:
            if isinstance(t, dict) and "pnl" in t:
                pnls.append(t["pnl"])
            elif isinstance(t, (int, float)):
                pnls.append(t)
        if pnls:
            pnls = np.array(pnls, dtype=float)
            total_pnl = pnls.sum()
            if total_pnl > 0:
                # What fraction comes from top 3 trades?
                top3 = np.sort(pnls)[-3:].sum()
                frac = top3 / total_pnl if total_pnl > 0 else 0
                if frac > 0.50:
                    report.add(Finding(
                        category="STATISTICAL_VALIDITY",
                        severity="WARNING",
                        message=f"[{name}] Top 3 trades account for {frac:.0%} of total P&L — "
                                f"edge may be driven by outliers, not systematic.",
                        location=name,
                        evidence=f"Top 3 P&L: ${top3:.2f} / Total: ${total_pnl:.2f}",
                    ))

                # Remove top 3: is total still positive?
                remaining = total_pnl - top3
                if remaining <= 0:
                    report.add(Finding(
                        category="STATISTICAL_VALIDITY",
                        severity="CRITICAL",
                        message=f"[{name}] Remove top 3 trades and total P&L goes negative "
                                f"(${remaining:.2f}) — edge is NOT systematic.",
                        location=name,
                    ))

    # 4e: Regime concentration
    gates = data.get("gates", data.get("validation", {}).get("gates", []))
    if isinstance(gates, list):
        for g in gates:
            if isinstance(g, dict) and g.get("name") == "Regime Stability":
                if not g.get("passed", True):
                    gap = g.get("metric_value", 0)
                    report.add(Finding(
                        category="STATISTICAL_VALIDITY",
                        severity="CRITICAL",
                        message=f"[{name}] Regime stability FAILED — Sharpe gap ratio {gap:.2f} > 0.50. "
                                f"Strategy is regime-dependent, not robust.",
                        location=name,
                        evidence=g.get("detail", ""),
                    ))

    # 4f: Maximum drawdown
    mdd = data.get("max_drawdown", data.get("mdd", data.get("max_dd", None)))
    if mdd is not None:
        mdd_val = abs(mdd)
        if mdd_val > 0.30:
            report.add(Finding(
                category="STATISTICAL_VALIDITY",
                severity="WARNING",
                message=f"[{name}] Max drawdown {mdd_val:.1%} exceeds 30% — significant risk.",
                location=name,
            ))

    # 4g: Profit factor
    pf = data.get("profit_factor", data.get("pf", None))
    if pf is not None:
        if pf < 1.0:
            report.add(Finding(
                category="STATISTICAL_VALIDITY",
                severity="CRITICAL",
                message=f"[{name}] Profit factor {pf:.2f} < 1.0 — strategy loses money.",
                location=name,
            ))
        elif pf > 5.0:
            report.add(Finding(
                category="STATISTICAL_VALIDITY",
                severity="WARNING",
                message=f"[{name}] Profit factor {pf:.2f} > 5.0 — suspiciously high, verify no leakage.",
                location=name,
            ))


# ─── Check 5: Code Pattern Scanning ───────────────────────────────────

def check_code_patterns(source: str, source_lines: list[str], tree: ast.Module,
                        report: AuditReport):
    """Scan for dangerous code patterns."""
    report.checks_run.append("CODE_PATTERNS")

    # 5a: Future iloc in feature functions
    iloc_detector = _FutureIndexDetector(source_lines)
    iloc_detector.visit(tree)
    for lineno, func, code in iloc_detector.suspicious_iloc:
        report.add(Finding(
            category="CODE_PATTERNS",
            severity="WARNING",
            message=f"Negative iloc indexing in feature/label function '{func}' — "
                    f"verify this doesn't reference future data.",
            location=f"line {lineno}",
            evidence=code.strip(),
        ))

    # 5b: from __future__ imports (benign, but flag for completeness in time-travel context)
    # Skip — `from __future__ import annotations` is standard Python.

    # 5c: Look for common data leakage patterns in regex
    dangerous_patterns = [
        # Full-data normalization
        (r'\.transform\(\s*(?:X|df|data|features)\s*\)',
         "transform() on potentially full dataset — should be train-only transform"),
        # Lookahead in rolling
        (r'\.rolling\([^)]+\)\.(?:mean|std|sum)\(\)\.shift\s*\(\s*-',
         "Rolling calculation followed by negative shift — time-travel pattern"),
        # Cross-validation instead of walk-forward (sometimes wrong for time series)
        (r'cross_val_score|KFold|StratifiedKFold',
         "K-Fold CV detected — not appropriate for time series (use walk-forward instead)"),
        # train_test_split without temporal ordering
        (r'train_test_split\(',
         "train_test_split may shuffle data — use temporal split for time series"),
    ]

    for pat, msg in dangerous_patterns:
        for m in re.finditer(pat, source):
            lineno = source[:m.start()].count("\n") + 1
            line = _get_source_line(source_lines, lineno)
            report.add(Finding(
                category="CODE_PATTERNS",
                severity="WARNING",
                message=msg,
                location=f"line {lineno}",
                evidence=line.strip(),
            ))

    # 5d: Check for output directory reuse (can cause fold leakage)
    output_dirs = re.findall(r'OUTPUT_DIR\s*=\s*(.+)', source)
    if len(output_dirs) > 1:
        report.add(Finding(
            category="CODE_PATTERNS",
            severity="WARNING",
            message="Multiple OUTPUT_DIR assignments — ensure different runs don't share directories.",
        ))

    # 5e: Check walk-forward has strict temporal ordering
    # Look for train < test date enforcement
    has_strict_boundary = bool(re.search(
        r'(?:train_dates|train_end|train_df).*<.*(?:test_date|test_start|test_df)|'
        r'd\s*<\s*test_date|'
        r'date\s*<\s*test|'
        r'\[df\[.*date.*\]\s*<',
        source
    ))
    has_wf = bool(re.search(r'walk.?forward|sliding.*window|WF_TRAIN', source, re.IGNORECASE))
    has_ml = any(kw in source for kw in [
        "LGBMRegressor", "LGBMClassifier", "lgb.", "xgb.", "model.fit",
        "RandomForest", "GradientBoosting",
    ])

    if has_wf and has_ml and not has_strict_boundary:
        report.add(Finding(
            category="CODE_PATTERNS",
            severity="WARNING",
            message="Walk-forward with ML model but no clear train < test date boundary detected. "
                    "Verify temporal ordering is enforced.",
        ))

    # 5f: Check for the banned cost constants (ES-specific, skip for options)
    if "ES_" in source or "es_futures" in source.lower():
        banned_costs = [
            (r'DEFAULT_COST_TICKS\s*=\s*2\.0', "DEFAULT_COST_TICKS=2.0 is absurdly high"),
            (r'COST_RT\s*=\s*1\.0', "COST_RT=1.0 conflates commission with spread"),
            (r'COST_TICKS\s*=\s*1\.24', "COST_TICKS=1.24 is wrong"),
        ]
        for pat, msg in banned_costs:
            if re.search(pat, source):
                m = re.search(pat, source)
                lineno = source[:m.start()].count("\n") + 1
                report.add(Finding(
                    category="PRICING_ERRORS",
                    severity="CRITICAL",
                    message=f"Banned cost constant: {msg}. See CLAUDE.md cost table.",
                    location=f"line {lineno}",
                ))


# ─── Main Audit Runner ────────────────────────────────────────────────

def run_audit(script_path: str, results_json: str | None = None) -> AuditReport:
    """
    Run full adversarial audit on a strategy research script.

    Args:
        script_path: Path to the Python strategy script
        results_json: Optional path to JSON results file for statistical checks

    Returns:
        AuditReport with all findings
    """
    script_path = str(Path(script_path).resolve())
    report = AuditReport(script_path=script_path)

    # Read and parse the script
    try:
        with open(script_path, "r") as f:
            source = f.read()
    except FileNotFoundError:
        report.add(Finding(
            category="FILE_ERROR",
            severity="CRITICAL",
            message=f"Script not found: {script_path}",
        ))
        return report
    except Exception as e:
        report.add(Finding(
            category="FILE_ERROR",
            severity="CRITICAL",
            message=f"Cannot read script: {e}",
        ))
        return report

    source_lines = source.split("\n")

    # Parse AST
    try:
        tree = ast.parse(source, filename=script_path)
    except SyntaxError as e:
        report.add(Finding(
            category="FILE_ERROR",
            severity="CRITICAL",
            message=f"Syntax error in script: {e}",
            location=f"line {e.lineno}",
        ))
        return report

    # Run all code-based checks
    check_label_leakage(source, source_lines, tree, report)
    check_wf_contamination(source, source_lines, tree, report)
    check_pricing_errors(source, source_lines, report)
    check_code_patterns(source, source_lines, tree, report)

    # Run statistical checks if results available
    if results_json:
        try:
            with open(results_json, "r") as f:
                results = json.load(f)
            check_statistical_validity(results, report)
        except FileNotFoundError:
            report.add(Finding(
                category="FILE_ERROR",
                severity="WARNING",
                message=f"Results JSON not found: {results_json}",
            ))
        except json.JSONDecodeError as e:
            report.add(Finding(
                category="FILE_ERROR",
                severity="WARNING",
                message=f"Invalid JSON in results file: {e}",
            ))
    else:
        # Try to find results JSON automatically
        script_dir = Path(script_path).parent
        script_stem = Path(script_path).stem

        # Common output patterns in this codebase
        candidate_paths = [
            script_dir / f"{script_stem}_results.json",
            script_dir.parent.parent / "output" / "growth_research" / script_stem / "results.json",
            script_dir.parent.parent / "output" / "growth_research" / script_stem / f"{script_stem}_results.json",
        ]

        for cp in candidate_paths:
            if cp.exists():
                try:
                    with open(cp, "r") as f:
                        results = json.load(f)
                    report.add(Finding(
                        category="FILE_ERROR",
                        severity="INFO",
                        message=f"Auto-discovered results JSON: {cp}",
                    ))
                    check_statistical_validity(results, report)
                    break
                except Exception:
                    pass
        else:
            report.checks_run.append("STATISTICAL_VALIDITY (skipped — no results JSON)")

    return report


def run_audit_on_results(results: dict, script_path: str = "<results-only>") -> AuditReport:
    """
    Run statistical validity checks on a results dict directly.
    Useful when calling from within a strategy script.

    Args:
        results: Dict with strategy results (trades, sharpe, etc.)
        script_path: Label for the report

    Returns:
        AuditReport with statistical findings only
    """
    report = AuditReport(script_path=script_path)
    check_statistical_validity(results, report)
    return report


# ─── CLI Entry Point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial Strategy Auditor — HC #753",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s scripts/growth_research/v13_combined_best_v1.py
              %(prog)s scripts/growth_research/v10_real_pricing_v1.py --results output/growth_research/v10_real_pricing_v1/results.json
              %(prog)s scripts/growth_research/*.py  (batch audit)
        """),
    )
    parser.add_argument("scripts", nargs="+", help="Path(s) to strategy script(s)")
    parser.add_argument("--results", "-r", help="Path to results JSON file")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output in JSON format")
    parser.add_argument("--fail-on-warning", action="store_true",
                        help="Exit non-zero on warnings too (not just critical)")

    args = parser.parse_args()

    all_reports = []
    any_failed = False

    for script in args.scripts:
        report = run_audit(script, results_json=args.results)
        all_reports.append(report)

        if not report.passed:
            any_failed = True
        if args.fail_on_warning and any(f.severity == "WARNING" for f in report.findings):
            any_failed = True

    if args.json:
        if len(all_reports) == 1:
            print(json.dumps(all_reports[0].to_dict(), indent=2))
        else:
            print(json.dumps([r.to_dict() for r in all_reports], indent=2))
    else:
        for report in all_reports:
            print(report)
            print()

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
