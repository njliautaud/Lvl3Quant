"""
ETF rotation sweep — runs hold ∈ {5, 10, 21} × {long-short, long-only}
concurrently and writes a comparison sweep_summary.md.

Output dir: output/macro_picker/etf_rotation_sweep_<TS>/
  <run_name>/   (one subdir per run, contents same as etf_rotation_v1)
  sweep_summary.md
  sweep_summary.json
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "strategy/macro_picker"))
from etf_rotation_v1 import run as run_rotation  # type: ignore  # noqa: E402

HOLDS = [5, 10, 21]
VARIANTS = [
    ("long_short", True),
    ("long_only", False),
]


def _run_one(hold: int, variant_name: str, allow_short: bool,
             base_out: Path) -> tuple[str, dict]:
    name = f"hold{hold}_{variant_name}"
    out_dir = base_out / name
    metrics = run_rotation(
        hold_days=hold,
        n_long=2,
        n_short=2 if allow_short else 0,
        allow_short=allow_short,
        out_dir=out_dir,
        n_jobs=1,  # outer sweep parallelism; inner stays serial to avoid oversubscription
    )
    return name, metrics


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = ROOT / f"output/macro_picker/etf_rotation_sweep_{ts}"
    base_out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for hold in HOLDS:
        for variant_name, allow_short in VARIANTS:
            jobs.append((hold, variant_name, allow_short))

    print(f"[sweep] dispatching {len(jobs)} runs into {base_out}")
    results = Parallel(n_jobs=min(len(jobs), 6), verbose=10)(
        delayed(_run_one)(hold, vname, allow_short, base_out)
        for (hold, vname, allow_short) in jobs
    )

    summary_rows = []
    for name, metrics in results:
        pooled = metrics.get("pooled", {})
        deploy = metrics.get("deploy_gate", {}).get("PASS", False)
        summary_rows.append({
            "run": name,
            "sharpe": pooled.get("sharpe"),
            "calmar": pooled.get("calmar"),
            "cagr": pooled.get("cagr"),
            "max_dd": pooled.get("max_dd"),
            "deploy": "Y" if deploy else "N",
        })

    # JSON
    (base_out / "sweep_summary.json").write_text(
        json.dumps(summary_rows, indent=2, default=str))

    # Markdown
    md = []
    md.append("# ETF rotation sweep")
    md.append("")
    md.append(f"Output: `{base_out}`")
    md.append("")
    md.append("| Run | Sharpe | Calmar | CAGR | MaxDD | Deploy |")
    md.append("|---|---|---|---|---|---|")
    # Sort by Calmar desc, then Sharpe desc.
    summary_rows.sort(key=lambda r: (-(r["calmar"] or -999), -(r["sharpe"] or -999)))
    for r in summary_rows:
        sh = r["sharpe"]
        cm = r["calmar"]
        cg = r["cagr"]
        dd = r["max_dd"]
        md.append(
            f"| {r['run']} | "
            f"{(sh if sh is not None else float('nan')):.2f} | "
            f"{(cm if cm is not None else float('nan')):.2f} | "
            f"{((cg or 0) * 100):.1f}% | "
            f"{((dd or 0) * 100):.1f}% | "
            f"{r['deploy']} |"
        )
    md.append("")
    md.append("Deploy gate: Calmar >= 1.0")
    (base_out / "sweep_summary.md").write_text("\n".join(md))

    print(f"[sweep] wrote {base_out}/sweep_summary.md")
    print("\n".join(md))


if __name__ == "__main__":
    main()
