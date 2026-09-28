#!/usr/bin/env python3
"""
cluster_status — single-command machine-readable + human-readable view of the cluster.
Pulls Razer + Neptune heartbeats via SSH, shows per-stage state.
Use:
    python3 cluster_status.py            # human summary
    python3 cluster_status.py --json     # raw JSON of both heartbeats
    python3 cluster_status.py --razer    # only razer
    python3 cluster_status.py --neptune  # only neptune
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

RAZER_HB = r"C:\Users\claude\Lvl3Quant\status\razer_heartbeat.json"
NEPTUNE_HB = "/home/nick/Lvl3Quant/output/split_dqn_v1_r22_patched/heartbeat.json"


def fetch_razer() -> dict:
    try:
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "claude@razer",
             f'powershell -NoProfile -Command "Get-Content \'{RAZER_HB}\' -Raw"'],
            capture_output=True, text=True, timeout=15)
        return json.loads(out.stdout)
    except Exception as e:
        return {"node": "razer", "fetch_error": str(e)[:200]}


NEPTUNE_GOV = "/home/nick/Lvl3Quant/output/split_dqn_v1_r22_patched/memory_governor_state.json"


def fetch_neptune() -> dict:
    try:
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "nick@neptune",
             f"cat {NEPTUNE_HB} 2>/dev/null; echo '|||SEP|||'; cat {NEPTUNE_GOV} 2>/dev/null; echo '|||SEP|||'; pgrep -f memory_governor.py | head -1"],
            capture_output=True, text=True, timeout=15)
        parts = out.stdout.split("|||SEP|||")
        hb = json.loads(parts[0]) if parts[0].strip() else {}
        gov = json.loads(parts[1]) if len(parts) > 1 and parts[1].strip() else {}
        gov_pid = parts[2].strip() if len(parts) > 2 else ""
        hb["_governor"] = gov
        hb["_governor_alive"] = bool(gov_pid and gov_pid.isdigit())
        hb["_governor_pid"] = int(gov_pid) if gov_pid.isdigit() else None
        return hb
    except Exception as e:
        return {"node": "neptune", "fetch_error": str(e)[:200]}


def hb_age_s(hb: dict) -> int | None:
    ts = hb.get("ts")
    if not ts:
        return None
    try:
        # parse iso utc
        if ts.endswith("+00:00"):
            t = datetime.fromisoformat(ts)
        else:
            t = datetime.fromisoformat(ts)
        return int(time.time() - t.timestamp())
    except Exception:
        return None


def health_color(h: str) -> str:
    return {
        "ok": "🟢",
        "live": "🟢",
        "live_trading": "🟢",
        "live_idle": "🟡",
        "warmup": "🟡",
        "stalled": "🟠",
        "memory_warn": "🟠",
        "degraded": "🟠",
        "error": "🔴",
        "down": "🔴",
        "emitter_error": "🔴",
    }.get(h, "⚪")


STAGE_ICON = {
    "green": "🟢", "ok": "🟢", "live": "🟢", "live_trading": "🟢",
    "warming": "🟡", "idle": "🟡", "live_idle": "🟡",
    "disabled": "⚫", "not_loaded": "⚫",
    "stale": "🟠", "degraded": "🟠", "memory_warn": "🟠",
    "down": "🔴", "error": "🔴", "emitter_error": "🔴", "unknown": "⚪",
}


def fmt_razer(hb: dict) -> str:
    if "fetch_error" in hb:
        return f"🔴 RAZER  fetch_error: {hb['fetch_error']}"
    age = hb_age_s(hb)
    age_str = f"{age}s ago" if age is not None else "?"
    health = hb.get("health", "?")
    icon = health_color(health)
    lines = [
        f"{icon} RAZER  paper_trader_mamba_v2  health={health}  hb_age={age_str}",
        f"   pid={hb.get('pid')} alive={hb.get('alive')} uptime={hb.get('uptime_s')}s rss={hb.get('rss_mb')}MB  log_age={hb.get('log_age_s')}s",
    ]
    stages = hb.get("stages") or {}
    if stages:
        lines.append("   STAGES (per HC #210):")
        order = ["ingest", "preprocess", "signal_cnn_mamba", "signal_patchtst",
                 "confluence", "exec_rl_entry", "exec_rl_exit",
                 "gates_filters", "order_submission", "risk_monitor"]
        for name in order:
            st = stages.get(name)
            if not st:
                continue
            s = st.get("status", "?")
            ic = STAGE_ICON.get(s, "⚪")
            tp = st.get("throughput") or {}
            tp_str = ", ".join(f"{k}={v}" for k, v in tp.items() if v is not None)
            age_s = st.get("age_s")
            age_part = f"  age={age_s}s" if age_s is not None else ""
            lines.append(f"     {ic} {name:<22} {s:<11}{age_part}  {tp_str}")
    else:
        # legacy schema fallback
        lines.append(f"   rithmic: md={hb.get('rithmic_md')} order={hb.get('rithmic_order')} acct={hb.get('account')} sub={hb.get('subscribed')}")
        lines.append(f"   stage={hb.get('stage')}  warmup={hb.get('warmup_pct')}%  events={hb.get('events_seen')}/{hb.get('warmup_target')}  signals={hb.get('signals_total')}  trades={hb.get('trades_total')}")
    if hb.get("health_reasons"):
        lines.append(f"   reasons: {hb['health_reasons']}")
    return "\n".join(lines)


def fmt_neptune(hb: dict) -> str:
    if "fetch_error" in hb:
        return f"🔴 NEPTUNE  fetch_error: {hb['fetch_error']}"
    age = hb_age_s(hb)
    age_str = f"{age}s ago" if age is not None else "?"
    health = hb.get("health", "?")
    icon = health_color(health)
    sysd = hb.get("system") or {}
    gpu = hb.get("gpu") or {}
    files_done = hb.get("files_done")
    files_total = hb.get("files_total")
    progress = f"{files_done}/{files_total}" if files_done is not None else "?"
    lines = [
        f"{icon} NEPTUNE  split_dqn_v1_r22_patched  health={health}  hb_age={age_str}",
        f"   pid={hb.get('pid')} alive={hb.get('alive')} uptime={hb.get('uptime_s')}s rss={hb.get('rss_mb')}MB workers={hb.get('n_workers')} threads={hb.get('n_threads')}",
        f"   sys: ram={sysd.get('ram_used_gb')}/{sysd.get('ram_total_gb')}GB ({sysd.get('ram_pct')}%, cap=28GB) load1m={sysd.get('load_1m')}",
        f"   gpu: util={gpu.get('util_pct')}% mem={gpu.get('mem_used_mb')}/{gpu.get('mem_total_mb')}MB",
        f"   training: fold={hb.get('fold')} ep={hb.get('epoch')} files={progress} trades={hb.get('trades')} pnl=${hb.get('pnl_usd')}",
        f"            gpu_updates E={hb.get('gpu_updates_e')} C={hb.get('gpu_updates_c')} X={hb.get('gpu_updates_x')}  ε={hb.get('epsilon')}",
        f"            buffers E={hb.get('buffer_e')} C={hb.get('buffer_c')} X={hb.get('buffer_x')}",
        f"   checkpoints: {hb.get('checkpoints_count')} saved (latest={hb.get('latest_checkpoint')})",
        f"   last_progress_ts={hb.get('last_progress_ts')}  log_age={hb.get('log_age_s')}s",
    ]
    gov = hb.get("_governor") or {}
    gov_alive = hb.get("_governor_alive")
    gov_icon = "🟢" if gov_alive else "🔴"
    if gov:
        verdict = gov.get("verdict", "?")
        rss = gov.get("rss_total_gb")
        sysu = gov.get("system_ram_used_gb")
        soft = gov.get("soft_cap_gb")
        hard = gov.get("hard_cap_gb")
        lines.append(f"   {gov_icon} mem_governor: alive={gov_alive} verdict={verdict} train_rss={rss}GB sys={sysu}GB caps[soft={soft} hard={hard} sys=28]GB")
    else:
        lines.append(f"   🔴 mem_governor: NO STATE FILE (HC #211 — should auto-relaunch via cron)")
    if hb.get("health_reasons"):
        lines.append(f"   reasons: {hb['health_reasons']}")
    if hb.get("last_error"):
        lines.append(f"   last_error: {hb['last_error']}")
    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    json_mode = "--json" in args
    only_razer = "--razer" in args
    only_neptune = "--neptune" in args

    razer = fetch_razer() if not only_neptune else None
    neptune = fetch_neptune() if not only_razer else None

    if json_mode:
        print(json.dumps({"razer": razer, "neptune": neptune,
                          "queried_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                         indent=2, default=str))
        return

    print(f"== CLUSTER STATUS  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==\n")
    if razer is not None:
        print(fmt_razer(razer))
        print()
    if neptune is not None:
        print(fmt_neptune(neptune))
        print()


if __name__ == "__main__":
    main()
