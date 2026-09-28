"""Render JSON + Markdown summary for the HC #413 sanity gate."""
from __future__ import annotations
import json
import os
from datetime import datetime


def write_reports(results: list[dict], meta: dict, output_dir: str) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.utcnow().isoformat() + "Z"
    overall_pass = all(r.get("passed", False) for r in results)
    payload = {
        "generated_at_utc": ts,
        "meta": meta,
        "overall_pass": overall_pass,
        "results": results,
    }
    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    md_path = os.path.join(output_dir, "report.md")
    lines: list[str] = []
    lines.append(f"# HC #413 Sanity Gate Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {ts}")
    for k, v in meta.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append(f"## OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    lines.append("")
    lines.append("| Check | Result | Failures | Notes |")
    lines.append("|---|---|---|---|")
    for r in results:
        name = r.get("check", "?")
        if r.get("skipped"):
            result_str = "SKIP"
        else:
            result_str = "PASS" if r.get("passed") else "FAIL"
        failures = r.get("failures") or []
        fail_str = "; ".join(failures)[:200] if failures else "-"
        d = r.get("details", {}) or {}
        notes_keys = list(d.keys())[:3]
        notes = ", ".join(notes_keys)
        lines.append(f"| {name} | {result_str} | {fail_str} | {notes} |")
    lines.append("")
    lines.append("## Per-check details")
    for r in results:
        lines.append(f"### {r.get('check')}")
        lines.append("```json")
        lines.append(json.dumps(r, indent=2, default=str))
        lines.append("```")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return json_path, md_path
