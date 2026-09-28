"""
ModelTracker - SQLite-based model results tracking for Lvl3Quant
================================================================
Single source of truth for every experiment run.

Usage:
    from alpha_discovery.model_tracker import ModelTracker, auto_record

    tracker = ModelTracker()

    # Record a result
    run_id = tracker.record_run(
        model_type='lightgbm',
        target_type='return',
        horizon='10s',
        ic_mean=0.139,
        sharpe=4.03,
        n_days=70,
        result_file='alpha_discovery/results/clean_scan_xxx.json'
    )

    # Import all existing JSON results
    tracker.scan_and_import()

    # Print leaderboard
    print(tracker.leaderboard())

CLI:
    python alpha_discovery/model_tracker.py --summary
    python alpha_discovery/model_tracker.py --leaderboard
    python alpha_discovery/model_tracker.py --search --model lightgbm --horizon 30s
    python alpha_discovery/model_tracker.py --import-all
    python alpha_discovery/model_tracker.py --invalidate RUN_ID --reason "leakage"
"""

import sqlite3
import json
import os
import re
import uuid
import glob
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent  # Lvl3Quant/
_DEFAULT_DB = _PROJECT_ROOT / "data" / "model_results.db"

RESULTS_DIRS = [
    _PROJECT_ROOT / "alpha_discovery" / "results",
    _PROJECT_ROOT / "results",
    _PROJECT_ROOT / "deep_models" / "results",
    _PROJECT_ROOT / "production" / "results",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    UNIQUE NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- What was tested
    model_type          TEXT    NOT NULL,
    model_name          TEXT,
    target_type         TEXT,
    horizon             TEXT,

    -- Training config
    n_days              INTEGER,
    n_oos_days          INTEGER,
    n_folds             INTEGER,
    features_count      INTEGER,
    config_json         TEXT,

    -- Primary metrics
    ic_mean             REAL,
    ic_ir               REAL,
    sharpe              REAL,
    total_pnl           REAL,
    profit_factor       REAL,
    win_rate            REAL,
    n_trades            INTEGER,
    avg_pnl_per_trade   REAL,
    positive_days_pct   REAL,

    -- Validation
    leakage_status      TEXT    DEFAULT 'unchecked',
    leakage_notes       TEXT,
    oos_valid           INTEGER DEFAULT 1,

    -- Metadata
    machine             TEXT,
    runtime_seconds     REAL,
    result_file         TEXT,
    log_file            TEXT,
    notes               TEXT,
    tags                TEXT,

    -- Status
    status              TEXT    DEFAULT 'completed'
);

CREATE TABLE IF NOT EXISTS conclusions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    category            TEXT    NOT NULL,
    subject             TEXT    NOT NULL,
    conclusion          TEXT    NOT NULL,
    evidence_run_ids    TEXT,
    tags                TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_model_type  ON runs(model_type);
CREATE INDEX IF NOT EXISTS idx_runs_horizon     ON runs(horizon);
CREATE INDEX IF NOT EXISTS idx_runs_target_type ON runs(target_type);
CREATE INDEX IF NOT EXISTS idx_runs_status      ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_sharpe      ON runs(sharpe);
"""

# All valid column names for the runs table (to prevent SQL injection)
_RUN_COLUMNS = {
    "run_id", "created_at", "model_type", "model_name", "target_type", "horizon",
    "n_days", "n_oos_days", "n_folds", "features_count", "config_json",
    "ic_mean", "ic_ir", "sharpe", "total_pnl", "profit_factor", "win_rate",
    "n_trades", "avg_pnl_per_trade", "positive_days_pct",
    "leakage_status", "leakage_notes", "oos_valid",
    "machine", "runtime_seconds", "result_file", "log_file", "notes", "tags", "status",
}


# ---------------------------------------------------------------------------
# Helper: generate a run_id
# ---------------------------------------------------------------------------

def _make_run_id(model_type: str = "") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short = str(uuid.uuid4())[:8]
    prefix = re.sub(r"[^a-z0-9]", "", model_type.lower())[:10]
    return f"{prefix}_{ts}_{short}" if prefix else f"run_{ts}_{short}"


# ---------------------------------------------------------------------------
# ModelTracker
# ---------------------------------------------------------------------------

class ModelTracker:
    """SQLite-backed tracker for every model experiment run."""

    def __init__(self, db_path: Optional[str] = None):
        path = Path(db_path) if db_path else _DEFAULT_DB
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(_DDL)

    def _row_to_dict(self, row) -> dict:
        if row is None:
            return {}
        d = dict(row)
        if d.get("config_json"):
            try:
                d["config"] = json.loads(d["config_json"])
            except Exception:
                pass
        return d

    # ------------------------------------------------------------------
    # Recording results
    # ------------------------------------------------------------------

    def record_run(
        self,
        model_type: str,
        target_type: Optional[str] = None,
        horizon: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Insert a new run record. Returns the run_id.

        All columns in the schema are valid kwargs. Extra kwargs are silently
        dropped (they won't crash the insert, but they won't be stored either).

        If 'config_json' is not provided but 'config' (dict) is, it will be
        serialised automatically.
        """
        run_id = kwargs.pop("run_id", _make_run_id(model_type))

        # Accept a config dict and serialise it
        config = kwargs.pop("config", None)
        if config and "config_json" not in kwargs:
            kwargs["config_json"] = json.dumps(config, default=str)

        # Filter to valid columns only
        fields = {"run_id": run_id, "model_type": model_type}
        if target_type is not None:
            fields["target_type"] = target_type
        if horizon is not None:
            fields["horizon"] = self._normalise_horizon(horizon)

        for k, v in kwargs.items():
            if k in _RUN_COLUMNS:
                fields[k] = v

        cols = ", ".join(fields.keys())
        placeholders = ", ".join("?" * len(fields))
        sql = f"INSERT INTO runs ({cols}) VALUES ({placeholders})"

        with self._connect() as conn:
            conn.execute(sql, list(fields.values()))

        return run_id

    def update_run(self, run_id: str, **kwargs):
        """Update any field(s) of an existing run."""
        # Accept config dict
        config = kwargs.pop("config", None)
        if config:
            kwargs["config_json"] = json.dumps(config, default=str)

        fields = {k: v for k, v in kwargs.items() if k in _RUN_COLUMNS}
        if not fields:
            return

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        sql = f"UPDATE runs SET {set_clause} WHERE run_id = ?"
        with self._connect() as conn:
            conn.execute(sql, list(fields.values()) + [run_id])

    def invalidate_run(self, run_id: str, reason: str):
        """Mark a run as invalidated with a reason stored in leakage_notes."""
        self.update_run(
            run_id,
            status="invalidated",
            leakage_status="leaky",
            leakage_notes=reason,
            oos_valid=0,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> dict:
        """Return a single run as a dict."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._row_to_dict(row)

    def get_best(
        self,
        model_type: Optional[str] = None,
        horizon: Optional[str] = None,
        target_type: Optional[str] = None,
        metric: str = "sharpe",
        min_oos_days: int = 0,
        leakage_ok: bool = False,
        top_n: int = 20,
    ) -> List[dict]:
        """
        Return top runs sorted by metric (desc).

        Only includes 'completed' runs unless leakage_ok=True (which also
        includes runs where leakage_status != 'clean' or 'unchecked').
        """
        if metric not in _RUN_COLUMNS:
            raise ValueError(f"Unknown metric: {metric}. Valid: {sorted(_RUN_COLUMNS)}")

        conditions = ["status = 'completed'"]
        params: list = []

        if model_type:
            conditions.append("model_type = ?")
            params.append(model_type)
        if horizon:
            conditions.append("horizon = ?")
            params.append(self._normalise_horizon(horizon))
        if target_type:
            conditions.append("target_type = ?")
            params.append(target_type)
        if min_oos_days > 0:
            conditions.append("(n_oos_days IS NULL OR n_oos_days >= ?)")
            params.append(min_oos_days)
        if not leakage_ok:
            conditions.append("(leakage_status IS NULL OR leakage_status IN ('clean', 'unchecked'))")

        where = " AND ".join(conditions)
        sql = (
            f"SELECT * FROM runs WHERE {where} AND {metric} IS NOT NULL "
            f"ORDER BY {metric} DESC LIMIT ?"
        )
        params.append(top_n)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def search(self, **filters) -> List[dict]:
        """
        Flexible search. All kwargs are AND-ed together as equality checks,
        except special keys:
            status      - exact match (default: excludes 'invalidated')
            horizon     - normalised before matching
            tags_has    - substring match inside tags column
            notes_has   - substring match inside notes column
            min_sharpe  - sharpe >= value
            min_ic      - ic_mean >= value
            min_pf      - profit_factor >= value
        """
        conditions = []
        params = []

        # Special filters
        status = filters.pop("status", None)
        if status:
            conditions.append("status = ?")
            params.append(status)
        else:
            conditions.append("status != 'invalidated'")

        tags_has = filters.pop("tags_has", None)
        if tags_has:
            conditions.append("tags LIKE ?")
            params.append(f"%{tags_has}%")

        notes_has = filters.pop("notes_has", None)
        if notes_has:
            conditions.append("notes LIKE ?")
            params.append(f"%{notes_has}%")

        min_sharpe = filters.pop("min_sharpe", None)
        if min_sharpe is not None:
            conditions.append("sharpe >= ?")
            params.append(min_sharpe)

        min_ic = filters.pop("min_ic", None)
        if min_ic is not None:
            conditions.append("ic_mean >= ?")
            params.append(min_ic)

        min_pf = filters.pop("min_pf", None)
        if min_pf is not None:
            conditions.append("profit_factor >= ?")
            params.append(min_pf)

        # Normalise horizon if present
        if "horizon" in filters:
            filters["horizon"] = self._normalise_horizon(filters["horizon"])

        # Remaining filters as equality
        for k, v in filters.items():
            if k in _RUN_COLUMNS:
                conditions.append(f"{k} = ?")
                params.append(v)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM runs WHERE {where} ORDER BY created_at DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_history(
        self, model_type: Optional[str] = None, limit: int = 200
    ) -> List[dict]:
        """All runs for a model type sorted by date (newest first)."""
        if model_type:
            sql = "SELECT * FROM runs WHERE model_type = ? ORDER BY created_at DESC LIMIT ?"
            params = [model_type, limit]
        else:
            sql = "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?"
            params = [limit]
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Conclusions
    # ------------------------------------------------------------------

    def add_conclusion(
        self,
        category: str,
        subject: str,
        conclusion: str,
        evidence_run_ids: Optional[List[str]] = None,
        tags: Optional[str] = None,
    ):
        """Record a research conclusion / dead-end / finding."""
        evidence = ",".join(evidence_run_ids) if evidence_run_ids else None
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conclusions (category, subject, conclusion, evidence_run_ids, tags) "
                "VALUES (?, ?, ?, ?, ?)",
                (category, subject, conclusion, evidence, tags),
            )

    def get_conclusions(
        self,
        category: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> List[dict]:
        conditions = []
        params = []
        if category:
            conditions.append("category = ?")
            params.append(category)
        if subject:
            conditions.append("subject LIKE ?")
            params.append(f"%{subject}%")
        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM conclusions WHERE {where} ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Importing existing JSON results
    # ------------------------------------------------------------------

    def import_from_json(
        self,
        json_path: str,
        model_type: Optional[str] = None,
        **defaults,
    ) -> List[str]:
        """
        Parse a result JSON file and record it (or them if it contains multiple runs).
        Returns list of run_ids inserted (skips if result_file already imported).
        """
        json_path = str(json_path)

        # Skip if already imported
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT run_id FROM runs WHERE result_file = ?", (json_path,)
            ).fetchall()
        if existing:
            return [r["run_id"] for r in existing]

        try:
            with open(json_path, encoding="utf-8", errors="replace") as fp:
                data = json.load(fp)
        except Exception as e:
            print(f"[import] Could not read {json_path}: {e}")
            return []

        # Auto-detect model_type from filename if not given
        if not model_type:
            model_type = self._detect_model_type(json_path, data)

        records = self._parse_result_file(json_path, data, model_type, defaults)
        run_ids = []
        for rec in records:
            try:
                run_id = self.record_run(**rec)
                run_ids.append(run_id)
            except sqlite3.IntegrityError:
                pass  # duplicate run_id
        return run_ids

    def scan_and_import(self, results_dirs: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Walk all results directories and import every JSON file found.
        Returns {dir_path: n_imported} summary.
        """
        dirs = results_dirs or [str(d) for d in RESULTS_DIRS]
        summary: Dict[str, int] = {}

        for d in dirs:
            d = str(d)
            if not os.path.isdir(d):
                continue
            json_files = glob.glob(os.path.join(d, "**", "*.json"), recursive=True)
            n_imported = 0
            for jf in json_files:
                run_ids = self.import_from_json(jf)
                n_imported += len(run_ids)
            summary[d] = n_imported

        return summary

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            by_status = conn.execute(
                "SELECT status, COUNT(*) as n FROM runs GROUP BY status"
            ).fetchall()
            by_model = conn.execute(
                "SELECT model_type, COUNT(*) as n, "
                "AVG(sharpe) as avg_sharpe, MAX(sharpe) as best_sharpe "
                "FROM runs WHERE status != 'invalidated' "
                "GROUP BY model_type ORDER BY best_sharpe DESC"
            ).fetchall()
            n_conclusions = conn.execute(
                "SELECT COUNT(*) FROM conclusions"
            ).fetchone()[0]
            leakage_q = conn.execute(
                "SELECT leakage_status, COUNT(*) as n FROM runs "
                "GROUP BY leakage_status"
            ).fetchall()

        lines = [
            "=" * 70,
            "LVLQUANT MODEL TRACKER SUMMARY",
            f"  DB: {self._db_path}",
            f"  Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 70,
            f"\nTotal runs: {total}",
        ]

        lines.append("\nRuns by status:")
        for row in by_status:
            lines.append(f"  {row['status']:20s}  {row['n']:>5}")

        lines.append("\nLeakage status:")
        for row in leakage_q:
            ls = row["leakage_status"] or "NULL"
            lines.append(f"  {ls:20s}  {row['n']:>5}")

        lines.append(f"\nConclusions recorded: {n_conclusions}")

        lines.append("\nBy model type (non-invalidated):")
        lines.append(f"  {'Model':25s} {'Runs':>5} {'AvgSharpe':>10} {'BestSharpe':>10}")
        lines.append("  " + "-" * 54)
        for row in by_model:
            avg_s = f"{row['avg_sharpe']:.3f}" if row["avg_sharpe"] is not None else "N/A"
            best_s = f"{row['best_sharpe']:.3f}" if row["best_sharpe"] is not None else "N/A"
            lines.append(f"  {row['model_type']:25s} {row['n']:>5} {avg_s:>10} {best_s:>10}")

        return "\n".join(lines)

    def leaderboard(self, metric: str = "sharpe", top_n: int = 10) -> str:
        rows = self.get_best(metric=metric, top_n=top_n)

        header = f"TOP {top_n} RUNS BY {metric.upper()}"
        lines = [
            "=" * 90,
            header,
            "=" * 90,
            f"  {'#':>3}  {'run_id':32s}  {'model_type':18s}  {'horizon':6s}  "
            f"{'target':16s}  {metric:>8}  {'sharpe':>8}  {'ic':>7}  {'pf':>6}  days",
            "  " + "-" * 84,
        ]
        for i, r in enumerate(rows, 1):
            val = r.get(metric)
            val_s = f"{val:.3f}" if val is not None else "N/A"
            sharpe_s = f"{r.get('sharpe', ''):.3f}" if r.get("sharpe") is not None else "N/A"
            ic_s = f"{r.get('ic_mean', ''):.4f}" if r.get("ic_mean") is not None else "N/A"
            pf_s = f"{r.get('profit_factor', ''):.3f}" if r.get("profit_factor") is not None else "N/A"
            tgt = (r.get("target_type") or "")[:16]
            hz = (r.get("horizon") or "")[:6]
            mt = (r.get("model_type") or "")[:18]
            rid = (r.get("run_id") or "")[:32]
            n_days = r.get("n_days") or ""
            lines.append(
                f"  {i:>3}  {rid:32s}  {mt:18s}  {hz:6s}  {tgt:16s}  "
                f"{val_s:>8}  {sharpe_s:>8}  {ic_s:>7}  {pf_s:>6}  {n_days}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal: parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_horizon(h: str) -> str:
        """Normalise horizon strings: 'ret_10s' -> '10s', '30S' -> '30s', etc."""
        if not h:
            return h
        h = str(h).lower().strip()
        # Strip 'ret_' prefix common in older code
        h = re.sub(r"^ret_", "", h)
        # Strip 'return_' prefix
        h = re.sub(r"^return_", "", h)
        return h

    @staticmethod
    def _detect_model_type(json_path: str, data: dict) -> str:
        """Guess model_type from filename or data structure."""
        fname = os.path.basename(json_path).lower()

        if "arch_benchmark" in fname or "deep_benchmark" in fname:
            # Could be lgbm, cnn, transformer
            if isinstance(data, dict) and "results" in data:
                r = data["results"]
                if isinstance(r, dict):
                    keys = list(r.keys())
                    if "lgbm" in keys:
                        return "lightgbm"
                    if any(k in keys for k in ("cnn", "spatial_cnn", "book_cnn")):
                        return "book_spatial_cnn"
                    if "transformer" in str(keys).lower():
                        return "event_transformer"
            return "lightgbm"

        if "novel_v2" in fname or "novel_queue" in fname or "novel_target" in fname:
            return "novel_target"
        if "clean_scan" in fname or "mbo_scan" in fname:
            return "lightgbm"
        if "mag_gated" in fname:
            return "lightgbm"
        if "mfe" in fname or "mfe_scan" in fname:
            return "lightgbm"
        if "multi_alpha" in fname or "multi_timeframe" in fname:
            return "lightgbm"
        if "event_alpha" in fname or "event_transformer" in fname:
            return "event_transformer"
        if "lstm" in fname:
            return "lstm"
        if "hybrid" in fname:
            return "hybrid"
        if "gnn" in fname or "multitask_gnn" in fname:
            return "event_transformer"
        if "light_" in fname or "medium_" in fname:
            return "event_transformer"
        if "overnight" in fname:
            return "lightgbm"
        if "continuation" in fname or "v2_pipeline" in fname:
            return "lightgbm"
        if "realistic_sim" in fname or "execution_backtest" in fname:
            return "lightgbm"
        if "assessment_summary" in fname or "checkpoint_channels" in fname:
            return "lightgbm"

        return "lightgbm"

    def _parse_result_file(
        self, json_path: str, data: Any, model_type: str, defaults: dict
    ) -> List[dict]:
        """
        Convert a raw JSON dict/list into a list of run record dicts.
        Each dict is passed directly to record_run(**rec).
        """
        fname = os.path.basename(json_path).lower()
        records = []

        # ---- novel_v2 / novel_targets_v2 --------------------------------
        if "novel_v2" in fname or "novel_queue" in fname:
            records = self._parse_novel_format(json_path, data, model_type)

        # ---- arch_benchmark / deep_benchmark ----------------------------
        elif "arch_benchmark" in fname or "deep_benchmark" in fname:
            records = self._parse_arch_benchmark(json_path, data)

        # ---- clean_scan / mbo_scan / multi_channel ----------------------
        elif any(x in fname for x in ("clean_scan", "mbo_scan", "checkpoint_channels")):
            records = self._parse_scan_format(json_path, data, model_type)

        # ---- mag_gated_sim ----------------------------------------------
        elif "mag_gated" in fname:
            records = self._parse_mag_gated(json_path, data)

        # ---- mfe_path / mfe_scan ----------------------------------------
        elif "mfe" in fname:
            records = self._parse_mfe(json_path, data)

        # ---- multi_alpha / multi_timeframe ------------------------------
        elif "multi_alpha" in fname or "multi_timeframe" in fname:
            records = self._parse_multi_alpha(json_path, data)

        # ---- light_/medium_/multitask (deep neural nets) ----------------
        elif any(fname.startswith(p) for p in ("light_", "medium_", "multitask_")):
            records = self._parse_deep_nn(json_path, data, model_type)

        # ---- assessment_summary -----------------------------------------
        elif "assessment_summary" in fname:
            records = self._parse_assessment(json_path, data)

        # ---- overnight / continuation / v2_pipeline -------------------
        elif any(x in fname for x in ("overnight", "continuation", "v2_pipeline",
                                       "realistic_sim", "execution_backtest")):
            records = self._parse_generic(json_path, data, model_type)

        # ---- generic fallback -------------------------------------------
        else:
            records = self._parse_generic(json_path, data, model_type)

        # Apply defaults and set result_file
        for rec in records:
            rec.setdefault("model_type", model_type)
            rec.setdefault("result_file", json_path)
            for k, v in defaults.items():
                rec.setdefault(k, v)
            # Normalise horizon
            if "horizon" in rec and rec["horizon"]:
                rec["horizon"] = self._normalise_horizon(rec["horizon"])

        return records

    # ------------------------------------------------------------------
    # Per-format parsers
    # ------------------------------------------------------------------

    def _parse_novel_format(self, json_path, data, model_type) -> List[dict]:
        """novel_v2_*.json and novel_queue_*.json"""
        records = []
        cfg = data.get("config", {})
        n_days = cfg.get("n_days")
        horizon = self._normalise_horizon(str(cfg.get("horizon", "")))
        ts = data.get("started") or data.get("timestamp") or _ts_from_filename(json_path)

        # novel_queue has 'tests' list, novel_v2 has 'results' list
        items = data.get("tests") or data.get("results") or []
        if not items:
            return records

        for item in items:
            # novel_queue wraps results inside 'sim_result'
            sim = item.get("sim_result") or item
            if not isinstance(sim, dict):
                continue

            target = sim.get("test_name") or item.get("test") or sim.get("label", "")
            # Extract target from label like "risk_adjusted_hold10s"
            if not target and "label" in sim:
                target = re.sub(r"_hold\d+s?$", "", sim["label"])

            # Skip non-sim items (summary rows, etc.)
            if not sim.get("n_trades") and not sim.get("sharpe"):
                continue

            n_oos = sim.get("total_days") or cfg.get("n_oos_days")
            pos_days = None
            if sim.get("positive_days") and sim.get("total_days"):
                pos_days = 100.0 * sim["positive_days"] / sim["total_days"]

            rec = {
                "model_type": model_type,
                "target_type": target,
                "horizon": horizon,
                "n_days": n_days,
                "n_oos_days": n_oos,
                "ic_mean": sim.get("fold_ic") or sim.get("ic"),
                "sharpe": sim.get("sharpe"),
                "total_pnl": sim.get("total_pnl_dollars"),
                "profit_factor": sim.get("profit_factor"),
                "win_rate": sim.get("win_rate"),
                "n_trades": sim.get("n_trades"),
                "avg_pnl_per_trade": sim.get("avg_pnl_per_trade"),
                "positive_days_pct": pos_days,
                "runtime_seconds": data.get("total_elapsed") or item.get("elapsed"),
                "result_file": json_path,
                "config": cfg,
                "notes": item.get("verdict", ""),
            }
            records.append(rec)
        return records

    def _parse_arch_benchmark(self, json_path, data) -> List[dict]:
        """arch_benchmark_*.json — multiple architectures, one per sub-key."""
        records = []
        cfg = data.get("config", {})
        horizon = self._normalise_horizon(str(cfg.get("horizon", "")))
        n_days = cfg.get("n_days")

        results = data.get("results", {})
        if not isinstance(results, dict):
            return records

        arch_to_model = {
            "lgbm": "lightgbm",
            "lightgbm": "lightgbm",
            "cnn": "book_spatial_cnn",
            "spatial_cnn": "book_spatial_cnn",
            "book_cnn": "book_spatial_cnn",
            "transformer": "event_transformer",
            "lstm": "lstm",
        }

        for arch, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            mt = arch_to_model.get(arch.lower(), arch.lower())
            n_folds = metrics.get("n_folds")

            rec = {
                "model_type": mt,
                "model_name": arch,
                "horizon": horizon,
                "n_days": n_days,
                "n_folds": n_folds,
                "ic_mean": metrics.get("ic"),
                "ic_ir": metrics.get("icir"),
                "profit_factor": metrics.get("profit_factor"),
                "win_rate": metrics.get("hit_rate"),
                "runtime_seconds": metrics.get("wall_time"),
                "config": cfg,
                "result_file": json_path,
            }
            records.append(rec)
        return records

    def _parse_scan_format(self, json_path, data, model_type) -> List[dict]:
        """clean_scan / mbo_scan / checkpoint_channels"""
        cfg = data.get("config", {}) or {}
        stats = data.get("stats", {}) or {}
        n_days = stats.get("n_days") or cfg.get("n_days")
        n_features = stats.get("n_features")

        results = data.get("results", [])
        if not results:
            # Try single-result format from arch_benchmark style
            return []

        records = []
        for item in (results if isinstance(results, list) else []):
            horizon = self._normalise_horizon(str(item.get("horizon", "")))
            target = item.get("target", "return")

            rec = {
                "model_type": model_type,
                "target_type": target,
                "horizon": horizon,
                "n_days": n_days,
                "features_count": n_features,
                "ic_mean": item.get("ic_mean") or item.get("ic"),
                "ic_ir": item.get("icir"),
                "profit_factor": item.get("profit_factor"),
                "runtime_seconds": data.get("elapsed_sec"),
                "config": cfg,
                "result_file": json_path,
            }
            records.append(rec)
        return records

    def _parse_mag_gated(self, json_path, data) -> List[dict]:
        """mag_gated_sim_*.json"""
        cfg_outer = data.get("config", {}) or {}
        horizon = self._normalise_horizon(str(data.get("horizon", "")))
        target = data.get("target_type", "mfe_net")

        top = data.get("top_results", [])
        if not top:
            return []

        # Record only the best (top[0])
        best = top[0]
        sim_cfg = best.get("config", {})
        rec = {
            "model_type": "lightgbm",
            "target_type": target,
            "horizon": horizon,
            "profit_factor": best.get("profit_factor"),
            "sharpe": best.get("sharpe"),
            "total_pnl": best.get("total_pnl_doll") or best.get("total_pnl_dollars"),
            "n_trades": best.get("n_filled"),
            "win_rate": best.get("win_rate"),
            "runtime_seconds": data.get("total_time_sec"),
            "config": {**cfg_outer, **sim_cfg},
            "result_file": json_path,
            "notes": f"best of {data.get('total_configs_tested', '?')} configs",
        }
        return [rec]

    def _parse_mfe(self, json_path, data) -> List[dict]:
        """mfe_path_analysis / mfe_scan"""
        n_days = data.get("n_days")
        strategies = data.get("strategy_results_top10") or data.get("top_results") or []
        if not strategies:
            return []

        # Take the best strategy
        best = strategies[0] if isinstance(strategies, list) and strategies else {}
        if not isinstance(best, dict):
            return []

        rec = {
            "model_type": "lightgbm",
            "target_type": "mfe_net",
            "n_days": n_days,
            "profit_factor": best.get("profit_factor") or best.get("pf"),
            "sharpe": best.get("sharpe"),
            "total_pnl": best.get("total_pnl"),
            "n_trades": best.get("n_trades"),
            "win_rate": best.get("win_rate"),
            "runtime_seconds": data.get("total_time_sec"),
            "result_file": json_path,
        }
        return [rec]

    def _parse_multi_alpha(self, json_path, data) -> List[dict]:
        """multi_alpha_*.json / multi_timeframe_*.json"""
        cfg = data.get("config", {}) or {}
        horizon = self._normalise_horizon(str(data.get("horizon", "")))

        comparison = data.get("comparison", {})
        phases = data.get("phases", {})

        # Extract best from comparison if present
        if comparison and isinstance(comparison, dict):
            best_sharpe = None
            best_pf = None
            for k, v in comparison.items():
                if isinstance(v, dict):
                    s = v.get("sharpe") or v.get("Sharpe")
                    if s and (best_sharpe is None or s > best_sharpe):
                        best_sharpe = s
                        best_pf = v.get("profit_factor") or v.get("PF")
            if best_sharpe is not None:
                return [{
                    "model_type": "lightgbm",
                    "horizon": horizon,
                    "sharpe": best_sharpe,
                    "profit_factor": best_pf,
                    "config": cfg,
                    "result_file": json_path,
                    "notes": "best channel from multi_alpha comparison",
                }]
        return []

    def _parse_deep_nn(self, json_path, data, model_type) -> List[dict]:
        """light_/medium_/multitask_gnn result files."""
        fname = os.path.basename(json_path).lower()

        if "multitask" in fname:
            # Multiple horizons in one file
            cfg = data.get("config", {}) or {}
            records = []
            for key in data:
                if key.startswith("ret_") or key.startswith("mfe_"):
                    horizon = self._normalise_horizon(key)
                    metrics = data[key]
                    if not isinstance(metrics, dict):
                        continue
                    rec = {
                        "model_type": "event_transformer",
                        "target_type": "return",
                        "horizon": horizon,
                        "ic_mean": metrics.get("ic") or metrics.get("corr"),
                        "ic_ir": metrics.get("icir"),
                        "config": cfg,
                        "result_file": json_path,
                    }
                    records.append(rec)
            return records
        else:
            # light/medium transformer
            cfg = data.get("config", {}) or {}
            final = data.get("final_metrics", {}) or {}
            best = data.get("best_metrics", {}) or {}
            # Extract horizon-based metrics
            records = []
            seen_horizons = set()
            for k in list(final.keys()) + list(best.keys()):
                m = re.match(r"^(\d+s|1min|10s|30s|5s|3s)_corr", k)
                if m:
                    hz = m.group(1)
                    if hz not in seen_horizons:
                        seen_horizons.add(hz)
                        metrics = final if final else best
                        rec = {
                            "model_type": "event_transformer",
                            "target_type": "return",
                            "horizon": hz,
                            "ic_mean": metrics.get(f"{hz}_corr"),
                            "config": cfg,
                            "result_file": json_path,
                        }
                        records.append(rec)
            if not records:
                # Fallback: single record
                records.append({
                    "model_type": "event_transformer",
                    "config": cfg,
                    "result_file": json_path,
                })
            return records

    def _parse_assessment(self, json_path, data) -> List[dict]:
        """assessment_summary_*.json"""
        # These are high-level summaries; record as a single run
        ts = data.get("timestamp") or _ts_from_filename(json_path)
        results = data.get("results", {}) or {}

        sharpe = None
        pf = None
        if isinstance(results, dict):
            for k, v in results.items():
                if isinstance(v, dict):
                    sharpe = v.get("sharpe") or sharpe
                    pf = v.get("profit_factor") or pf

        rec = {
            "model_type": "lightgbm",
            "sharpe": sharpe,
            "profit_factor": pf,
            "result_file": json_path,
            "notes": "assessment summary",
        }
        return [rec]

    def _parse_generic(self, json_path, data, model_type) -> List[dict]:
        """Generic fallback parser for any JSON with recognisable fields."""
        if not isinstance(data, dict):
            return []

        cfg = data.get("config", {}) or {}
        horizon = self._normalise_horizon(
            str(data.get("horizon") or cfg.get("horizon") or "")
        )

        # Try to find metrics at top level or inside 'results'
        src = data
        results_val = data.get("results")
        if isinstance(results_val, list) and results_val and isinstance(results_val[0], dict):
            src = results_val[0]
        elif isinstance(results_val, dict):
            # Grab first sub-key
            first_v = next(iter(results_val.values()), None)
            if isinstance(first_v, dict):
                src = first_v

        def _get(*keys):
            for k in keys:
                v = src.get(k) or data.get(k)
                if v is not None:
                    return v
            return None

        n_days = cfg.get("n_days") or data.get("n_days")
        n_oos = cfg.get("n_oos_days") or data.get("n_oos_days")
        n_folds = cfg.get("n_folds") or data.get("n_folds") or _get("n_folds")

        rec = {
            "model_type": model_type,
            "horizon": horizon or None,
            "n_days": n_days,
            "n_oos_days": n_oos,
            "n_folds": n_folds,
            "ic_mean": _get("ic_mean", "ic"),
            "ic_ir": _get("icir", "ic_ir"),
            "sharpe": _get("sharpe"),
            "total_pnl": _get("total_pnl_dollars", "total_pnl"),
            "profit_factor": _get("profit_factor"),
            "win_rate": _get("win_rate", "hit_rate"),
            "n_trades": _get("n_trades"),
            "avg_pnl_per_trade": _get("avg_pnl_per_trade"),
            "runtime_seconds": data.get("total_time_sec") or data.get("elapsed_sec"),
            "config": cfg or None,
            "result_file": json_path,
        }
        return [rec]


# ---------------------------------------------------------------------------
# auto_record helper (for integration into training scripts)
# ---------------------------------------------------------------------------

def auto_record(
    tracker: ModelTracker,
    result_dict: dict,
    model_type: str,
    **extra,
) -> str:
    """
    Parse a standard result dict (as produced by most training scripts) and
    record it in the tracker.  Returns the new run_id.

    Expected keys (all optional, best-effort extraction):
        sharpe, ic, icir, profit_factor, win_rate, n_trades,
        avg_pnl_per_trade, total_pnl_dollars, positive_days,
        total_days, n_folds, fold_ics, horizon, target_type,
        n_days, n_oos_days, config, runtime_seconds, result_file
    """
    d = result_dict

    # Positive-day pct
    pos_pct = None
    if d.get("positive_days") and d.get("total_days"):
        pos_pct = 100.0 * d["positive_days"] / d["total_days"]

    # IC mean from fold_ics if not provided
    ic_mean = d.get("ic") or d.get("ic_mean")
    if ic_mean is None and d.get("fold_ics"):
        fics = d["fold_ics"]
        if fics:
            ic_mean = sum(fics) / len(fics)

    run_id = tracker.record_run(
        model_type=model_type,
        target_type=d.get("target_type") or d.get("test_name") or extra.pop("target_type", None),
        horizon=d.get("horizon") or extra.pop("horizon", None),
        n_days=d.get("n_days") or extra.pop("n_days", None),
        n_oos_days=d.get("n_oos_days") or d.get("total_days") or extra.pop("n_oos_days", None),
        n_folds=d.get("n_folds") or extra.pop("n_folds", None),
        ic_mean=ic_mean,
        ic_ir=d.get("icir") or d.get("ic_ir"),
        sharpe=d.get("sharpe"),
        total_pnl=d.get("total_pnl_dollars") or d.get("total_pnl"),
        profit_factor=d.get("profit_factor"),
        win_rate=d.get("win_rate") or d.get("hit_rate"),
        n_trades=d.get("n_trades"),
        avg_pnl_per_trade=d.get("avg_pnl_per_trade"),
        positive_days_pct=pos_pct,
        config=d.get("config"),
        runtime_seconds=d.get("runtime_seconds") or d.get("elapsed") or d.get("train_time_sec"),
        result_file=d.get("result_file"),
        **extra,
    )
    return run_id


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _ts_from_filename(path: str) -> str:
    """Extract a timestamp string from filename like 'xxx_20260222_230028.json'."""
    m = re.search(r"(\d{8}_\d{6})", os.path.basename(path))
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description="Lvl3Quant ModelTracker — query and manage experiment results"
    )
    parser.add_argument("--db", default=None, help="Override DB path")
    parser.add_argument("--summary", action="store_true", help="Print summary")
    parser.add_argument(
        "--leaderboard", action="store_true", help="Print leaderboard"
    )
    parser.add_argument(
        "--metric",
        default="sharpe",
        help="Metric to rank by in leaderboard (default: sharpe)",
    )
    parser.add_argument("--top", type=int, default=10, help="Top N rows")
    parser.add_argument(
        "--search",
        action="store_true",
        help="Search runs (combine with --model, --horizon, --target, etc.)",
    )
    parser.add_argument("--model", default=None, help="Filter by model_type")
    parser.add_argument("--horizon", default=None, help="Filter by horizon")
    parser.add_argument("--target", default=None, help="Filter by target_type")
    parser.add_argument("--status", default=None, help="Filter by status")
    parser.add_argument(
        "--import-all",
        action="store_true",
        help="Scan all results dirs and import JSONs",
    )
    parser.add_argument(
        "--import-file",
        default=None,
        metavar="PATH",
        help="Import a specific JSON file",
    )
    parser.add_argument(
        "--invalidate",
        default=None,
        metavar="RUN_ID",
        help="Invalidate a run by run_id",
    )
    parser.add_argument(
        "--reason", default="manually invalidated", help="Reason for invalidation"
    )
    parser.add_argument(
        "--add-conclusion",
        nargs=3,
        metavar=("CATEGORY", "SUBJECT", "CONCLUSION"),
        help="Add a conclusion: category subject 'conclusion text'",
    )
    parser.add_argument(
        "--conclusions",
        action="store_true",
        help="List all conclusions",
    )
    parser.add_argument(
        "--run",
        default=None,
        metavar="RUN_ID",
        help="Show details for a specific run",
    )

    args = parser.parse_args()
    tracker = ModelTracker(db_path=args.db)

    if args.import_all:
        print("Scanning and importing all results directories...")
        results = tracker.scan_and_import()
        for d, n in results.items():
            print(f"  {d}: {n} runs imported")
        print("\nDone. Run --summary to see overview.")
        return

    if args.import_file:
        run_ids = tracker.import_from_json(args.import_file)
        print(f"Imported {len(run_ids)} run(s): {run_ids}")
        return

    if args.invalidate:
        tracker.invalidate_run(args.invalidate, args.reason)
        print(f"Invalidated run: {args.invalidate}")
        return

    if args.add_conclusion:
        cat, subj, conc = args.add_conclusion
        tracker.add_conclusion(cat, subj, conc)
        print("Conclusion added.")
        return

    if args.conclusions:
        conclusions = tracker.get_conclusions()
        if not conclusions:
            print("No conclusions recorded yet.")
        else:
            for c in conclusions:
                print(f"\n[{c['category']}] {c['subject']}")
                print(f"  {c['conclusion']}")
                if c.get("evidence_run_ids"):
                    print(f"  Evidence: {c['evidence_run_ids']}")
        return

    if args.run:
        r = tracker.get_run(args.run)
        if not r:
            print(f"Run not found: {args.run}")
        else:
            print(json.dumps(r, indent=2, default=str))
        return

    if args.search:
        filters = {}
        if args.model:
            filters["model_type"] = args.model
        if args.horizon:
            filters["horizon"] = args.horizon
        if args.target:
            filters["target_type"] = args.target
        if args.status:
            filters["status"] = args.status
        rows = tracker.search(**filters)
        if not rows:
            print("No results found.")
        else:
            print(
                f"\n{'run_id':32s}  {'model_type':18s}  {'horizon':6s}  "
                f"{'target':16s}  {'sharpe':>8}  {'ic':>7}  {'pf':>6}  "
                f"{'pnl':>10}  days  status"
            )
            print("-" * 120)
            for r in rows[: args.top]:
                sharpe_s = f"{r.get('sharpe'):.3f}" if r.get("sharpe") is not None else "N/A"
                ic_s = f"{r.get('ic_mean'):.4f}" if r.get("ic_mean") is not None else "N/A"
                pf_s = f"{r.get('profit_factor'):.3f}" if r.get("profit_factor") is not None else "N/A"
                pnl_s = f"{r.get('total_pnl'):>10.0f}" if r.get("total_pnl") is not None else "       N/A"
                print(
                    f"{r.get('run_id','')[:32]:32s}  "
                    f"{(r.get('model_type') or '')[:18]:18s}  "
                    f"{(r.get('horizon') or '')[:6]:6s}  "
                    f"{(r.get('target_type') or '')[:16]:16s}  "
                    f"{sharpe_s:>8}  {ic_s:>7}  {pf_s:>6}  "
                    f"{pnl_s}  {r.get('n_days',''):>4}  {r.get('status','')}"
                )
        return

    if args.leaderboard:
        print(tracker.leaderboard(metric=args.metric, top_n=args.top))
        return

    # Default: summary
    print(tracker.summary())


if __name__ == "__main__":
    _cli()
