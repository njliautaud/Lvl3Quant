#!/usr/bin/env python3
"""
quant_precheck.py — HC #817 SCRIPT-FIRST pre-checks for the Head of Quant bridge.

Replaces the "wake Claude every N minutes" cron injects with a cheap Python check
that computes the same facts and wakes Claude ONLY when something is actionable
or changed. READ-ONLY on broker + Lvl3Quant state (it never writes position files;
its own memory lives in ~/agent/state/quant_precheck/).

Modes (one per cron line):
  positions  — replaces */10 rh_position_check.txt   (wake = CRITICAL)
  spreads    — replaces */30 spread_exit_monitor.txt (wake = CRITICAL)
  scanner    — replaces */30 data_driven_scanner.txt (wake = CRITICAL; it can place entries)
  pulse      — replaces 4h PULSE_CHECK               (wake = LOW)

Fail-safe: if the check itself errors, it WAKES Claude with the original prompt
(for trade paths), so a script bug can never silently drop a trade-safety check.

  --dry-run : print the decision, inject nothing, persist nothing.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, date
from pathlib import Path

BASE = Path("/home/jupiter/Lvl3Quant")
PROMPTS = BASE / "scripts" / "prompts"
INJECT = str(BASE / "scripts" / "autonomy_inject.sh")
STATE_DIR = Path.home() / "agent" / "state" / "quant_precheck"
LOG = Path.home() / "agent" / "logs" / "quant_precheck.log"
INACTIVE = {"closed", "expired", "cancelled", "canceled"}

# Position-check tuning (the ONLY frequency reduction: no-op heartbeats 10 -> 30 min)
HEARTBEAT_MIN = 30          # while positions are open, wake Claude at least this often
PNL_MOVE_PTS = 5.0          # wake if P&L moved this many points since last wake
RECONCILE_SLOTS = ["10:00", "14:00"]  # when locally flat: broker reconcile wakes (catches unrecorded positions)
PULSE_MAX_SILENCE_H = 24    # pulse: wake at least once a day even if nothing changed

CRON_ENV = dict(os.environ, PATH="/home/jupiter/.npm-global/bin:/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", ""))
PULSE_PROMPT = "PULSE_CHECK. Plain English per HC #433. Node status, GPU usage, running jobs, any alerts."


# ─────────────────────────── helpers ───────────────────────────
def jload(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_state(mode):
    return jload(STATE_DIR / f"{mode}.json", {}) or {}


def save_state(mode, st, dry):
    if dry:
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_DIR / f".{mode}.json.tmp"
    tmp.write_text(json.dumps(st, indent=2, default=str))
    tmp.replace(STATE_DIR / f"{mode}.json")


def minutes_since(iso):
    if not iso:
        return 1e9
    try:
        return (datetime.now() - datetime.fromisoformat(iso)).total_seconds() / 60
    except Exception:
        return 1e9


def log_line(rec):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def inject(priority, msg, dry):
    if dry:
        env = dict(os.environ, INJECT_DRY_RUN="1")
    else:
        env = dict(os.environ)
    r = subprocess.run([INJECT, "--priority", priority, msg], env=env,
                       capture_output=True, text=True, timeout=60)
    return (r.stdout or "").strip()[:200]


# ─────────────────────────── positions ───────────────────────────
def _pos_key(p):
    return {k: p.get(k) for k in ("ticker", "symbol", "strike", "type", "option_type", "expiry",
                                   "expiration", "status", "quantity", "shares", "tp_order_id",
                                   "sl_order_id", "sl_type", "entry_debit", "entry_price", "legs")}


def collect_open():
    """Everything the local books say is open. Returns (singles, spreads, others, signature)."""
    ao = jload(BASE / "data" / "active_options.json", {}) or {}
    singles, spreads = [], []
    for p in ao.get("positions", []) or []:
        if str(p.get("status", "")).lower() in INACTIVE:
            continue
        (spreads if (p.get("legs") or p.get("spread_type")) else singles).append(p)
    others = []
    for e in ao.get("equity_positions", []) or []:
        if str(e.get("status", "open")).lower() not in INACTIVE:
            others.append({"src": "active_options.equity", **_pos_key(e)})
    ag = jload(BASE / "state" / "agentic_positions.json", {}) or {}
    for k, p in (ag.get("positions") or {}).items():
        if str(p.get("status", "")).upper() != "CLOSED" and (p.get("quantity") or 0) > 0:
            others.append({"src": "agentic_positions", "id": k, **_pos_key(p)})
    up = jload(BASE / "state" / "unified_portfolio_state.json", {}) or {}
    cp = up.get("current_position") or {}
    if cp.get("symbol") and (cp.get("shares") or 0) > 0:
        others.append({"src": "unified_portfolio", "symbol": cp.get("symbol"), "shares": cp.get("shares")})
    ah = jload(BASE / "state" / "afterhours_equity_position.json", {}) or {}
    if ah.get("ticker") and not ah.get("action_taken") and ah.get("timestamp"):
        try:
            age_d = (datetime.now() - datetime.fromisoformat(str(ah["timestamp"])[:19])).days
        except Exception:
            age_d = 0
        if age_d <= 2:
            others.append({"src": "afterhours_equity", "ticker": ah.get("ticker"), "ts": ah.get("timestamp")})
    sig = sha({"s": [_pos_key(p) for p in singles], "sp": [_pos_key(p) for p in spreads], "o": others})
    return singles, spreads, others, sig


def evaluate_single(p):
    """Same exit engine rh_trigger_watchdog uses (lib/exit_rules) — read-only, no state writes."""
    sys.path.insert(0, str(BASE))
    from lib.exit_rules import evaluate_exits, Position, MarketData
    import yfinance as yf
    ticker = p.get("ticker") or p.get("symbol")
    opt_type = (p.get("option_type") or p.get("type") or "call").lower()
    strike = float(p.get("strike") or 0)
    expiry = p.get("expiration") or p.get("expiry") or ""
    entry = float(p.get("entry_price") or p.get("entry_debit") or 0)
    mark = None
    try:
        ch = yf.Ticker(ticker).option_chain(expiry)
        df = ch.puts if opt_type == "put" else ch.calls
        row = df[abs(df["strike"] - strike) < 0.01]
        if len(row):
            b, a = float(row.iloc[0]["bid"]), float(row.iloc[0]["ask"])
            mark = (b + a) / 2 if b > 0 and a > 0 else float(row.iloc[0]["lastPrice"])
    except Exception:
        mark = None
    und = None
    try:
        h = yf.Ticker(ticker).history(period="1d")
        und = float(h["Close"].iloc[-1]) if len(h) else None
    except Exception:
        pass
    vix = 16.0
    try:
        v = yf.download("^VIX", period="2d", progress=False)
        vix = float(v["Close"].values.flatten()[-1])
    except Exception:
        pass
    regime = (jload(BASE / "data" / "macro" / "macro_summary.json", {}) or {}).get("regime", "RISK_ON")
    out = {"ticker": ticker, "type": opt_type, "strike": strike, "expiry": expiry, "entry": entry,
           "mark": mark, "underlying": und, "vix": round(vix, 2), "regime": regime,
           "has_bracket": bool(p.get("tp_order_id") or p.get("sl_order_id"))}
    if mark is None or entry <= 0:
        out["error"] = "no_price" if mark is None else "no_entry"
        return out
    res = evaluate_exits(
        Position(ticker=ticker, option_type=opt_type, strike=strike, expiration=expiry,
                 entry_price=entry, entry_date=p.get("entry_date", date.today().isoformat()),
                 underlying_entry_price=p.get("underlying_entry_price", 0) or 0,
                 quantity=p.get("quantity", 1) or 1, peak_value=p.get("peak_value", entry) or entry,
                 n_sources=p.get("n_sources", 0) or 0, tp_order_id=p.get("tp_order_id"),
                 sl_order_id=p.get("sl_order_id"), trailing_active=p.get("trailing_active", False)),
        MarketData(current_mark=mark, underlying_price=und or 0, vix=vix, macro_regime=regime,
                   market_open_time=datetime.now().replace(hour=9, minute=30, second=0)))
    out.update(triggered=res.triggered, rule=res.rule_name, reason=res.reason,
               pnl_pct=round(res.pnl_pct, 1),
               proximity=res.details.get("proximity_warnings", []))
    return out


def mode_positions(dry):
    st = load_state("positions")
    singles, spreads, others, sig = collect_open()
    reasons, facts = [], {"n_single": len(singles), "n_spread": len(spreads), "n_other": len(others), "sig": sig}
    now = datetime.now()
    if st.get("sig") is not None and st.get("sig") != sig:
        reasons.append("position records changed since last check (new entry/exit/bracket) — verify at broker")

    if not singles and not spreads and not others:
        # Locally flat. Claude's check would 'skip silently' — except it also sees broker-side
        # positions the books missed. Reconcile at fixed slots.
        today = now.date().isoformat()
        done = st.get("reconciled", {})
        for slot in RECONCILE_SLOTS:
            hh, mm = map(int, slot.split(":"))
            if now >= now.replace(hour=hh, minute=mm, second=0) and done.get(slot) != today:
                reasons.append(f"scheduled broker reconcile ({slot}) — books show flat")
                done[slot] = today
                break
        st["reconciled"] = done
    else:
        evals = []
        for p in singles:
            try:
                e = evaluate_single(p)
            except Exception as ex:
                e = {"ticker": p.get("ticker"), "error": f"eval_failed: {ex}"}
            evals.append(e)
            t = e.get("ticker")
            if e.get("error"):
                reasons.append(f"{t}: cannot verify by script ({e['error']})")
                continue
            if e.get("triggered"):
                reasons.append(f"{t}: EXIT TRIGGER {e['rule']} — {e['reason']}")
            if not e.get("has_bracket"):
                reasons.append(f"{t}: no bracket orders on record (HC #810 — place TP/SL)")
            last_pnl = (st.get("last_pnl") or {}).get(t)
            if last_pnl is not None and abs(e["pnl_pct"] - last_pnl) >= PNL_MOVE_PTS:
                reasons.append(f"{t}: P&L moved {last_pnl:+.1f}% -> {e['pnl_pct']:+.1f}%")
            if e.get("proximity") and sorted(e["proximity"]) != sorted((st.get("last_prox") or {}).get(t, [])):
                reasons.append(f"{t}: near exit ({', '.join(e['proximity'])})")
        facts["evals"] = evals
        if spreads or others:
            facts["unscripted"] = [p.get("ticker") or p.get("symbol") for p in spreads] + \
                                  [o.get("ticker") or o.get("symbol") for o in others]
        if not reasons and minutes_since(st.get("last_wake")) >= HEARTBEAT_MIN:
            reasons.append(f"heartbeat: positions open, {HEARTBEAT_MIN}-min broker check (bracket fills / hygiene)")

    wake = bool(reasons)
    st["sig"] = sig
    st["last_run"] = now.isoformat()
    if wake:
        st["last_wake"] = now.isoformat()
        st["last_pnl"] = {e["ticker"]: e["pnl_pct"] for e in facts.get("evals", []) if "pnl_pct" in e}
        st["last_prox"] = {e["ticker"]: e.get("proximity", []) for e in facts.get("evals", []) if "pnl_pct" in e}
    return wake, "critical", reasons, facts, st, (PROMPTS / "rh_position_check.txt").read_text()


def mode_spreads(dry):
    st = load_state("spreads")
    _, spreads, _, _ = collect_open()
    sig = sha([_pos_key(p) for p in spreads])
    reasons = []
    if spreads:
        # Multi-leg exit math needs live leg quotes -> keep Claude at the original 30-min cadence.
        reasons.append(f"{len(spreads)} open spread(s): " + ", ".join(str(p.get("ticker")) for p in spreads))
    st.update(sig=sig, last_run=datetime.now().isoformat())
    if reasons:
        st["last_wake"] = datetime.now().isoformat()
    return bool(reasons), "critical", reasons, {"n_spread": len(spreads)}, st, \
        (PROMPTS / "spread_exit_monitor.txt").read_text()


def mode_scanner(dry):
    st = load_state("scanner")
    os.chdir(BASE)
    sys.path.insert(0, str(BASE))
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from scripts.options_backtest_scanner import live_scan
        r = live_scan()
    actionable, best = [], []
    for opp in r.get("active_opportunities", []):
        for s in opp.get("active_signals", []):
            best.append((s.get("edge_weight", 0), opp["ticker"], s["signal"], s["direction"]))
            if (s.get("edge_weight") or 0) > 0.01:   # same threshold as data_driven_scanner.txt
                actionable.append(f"{opp['ticker']} {s['signal']} {s['direction']} edge={s['edge_weight']}")
    best.sort(reverse=True)
    n_err = sum(1 for v in r.get("all_tickers", {}).values() if "error" in v)
    n_all = len(r.get("all_tickers", {}))
    reasons = []
    if actionable:
        reasons.append("scanner signal(s) above edge 0.01: " + "; ".join(actionable))
    elif n_all and n_err == n_all:
        reasons.append(f"scanner could not fetch data for any ticker ({n_err}/{n_all} errors) — manual check")
    facts = {"n_actionable": len(actionable), "best": [f"{t} {sg} {e:.4f}" for e, t, sg, d in best[:3]],
             "ticker_errors": f"{n_err}/{n_all}"}
    if not reasons and not dry:
        with open(BASE / "logs" / "scanner_checks.log", "a") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S} ET] Scanner precheck (script, no Claude wake): "
                    f"No actionable signals. Best: {', '.join(facts['best']) or 'none'}.\n")
    st.update(last_run=datetime.now().isoformat())
    if reasons:
        st["last_wake"] = datetime.now().isoformat()
    return bool(reasons), "critical", reasons, facts, st, (PROMPTS / "data_driven_scanner.txt").read_text()


def mode_pulse(dry):
    st = load_state("pulse")
    facts = {}
    try:
        ts = json.loads(subprocess.run(["tailscale", "status", "--json"], capture_output=True,
                                       text=True, timeout=15, env=CRON_ENV).stdout)
        facts["nodes"] = sorted({f"{p['HostName']}={'up' if p.get('Online') else 'down'}"
                                 for p in ts.get("Peer", {}).values()
                                 if any(k in p["HostName"].lower() for k in ("neptune", "razer", "saturn", "uranus"))})
    except Exception as e:
        facts["nodes"] = f"err:{type(e).__name__}"
    try:
        pm = json.loads(subprocess.run(["pm2", "jlist"], capture_output=True, text=True, timeout=20, env=CRON_ENV).stdout)
        facts["pm2_not_online"] = sorted(f"{p['name']}={p['pm2_env']['status']}" for p in pm
                                         if p["pm2_env"]["status"] not in ("online", "stopped", "waiting restart"))
        facts["pm2_online"] = sum(1 for p in pm if p["pm2_env"]["status"] == "online")
    except Exception as e:
        facts["pm2_not_online"] = f"err:{type(e).__name__}"
    try:  # dead-air / high-priority tripwires in the last 4h
        cutoff = time.time() - 4 * 3600
        n = 0
        with open(BASE / "logs" / "accountability.log", errors="ignore") as f:
            for line in f.readlines()[-400:]:
                if "!!!" in line:
                    try:
                        t = datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp()
                    except Exception:
                        continue
                    n += t >= cutoff
        facts["alerts_4h"] = "yes" if n else "no"
    except Exception:
        facts["alerts_4h"] = "unknown"
    h = sha(facts)
    reasons = []
    if st.get("hash") != h:
        reasons.append("infra state changed since last pulse: " + json.dumps(facts, default=str)[:400])
    elif minutes_since(st.get("last_wake")) >= PULSE_MAX_SILENCE_H * 60:
        reasons.append("daily pulse (nothing changed)")
    st.update(last_run=datetime.now().isoformat())
    if reasons:
        st.update(hash=h, last_wake=datetime.now().isoformat())
    return bool(reasons), "low", reasons, facts, st, PULSE_PROMPT


MODES = {"positions": mode_positions, "spreads": mode_spreads, "scanner": mode_scanner, "pulse": mode_pulse}
FALLBACK = {"positions": ("critical", "rh_position_check.txt"), "spreads": ("critical", "spread_exit_monitor.txt"),
            "scanner": ("critical", "data_driven_scanner.txt"), "pulse": ("low", None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=list(MODES))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    import signal
    class PrecheckTimeout(BaseException):  # BaseException so inner `except Exception` can't swallow it
        pass
    def _hung(*_):
        raise PrecheckTimeout("precheck timed out (hung data call)")
    signal.signal(signal.SIGALRM, _hung)
    signal.alarm(int(os.environ.get("PRECHECK_TIMEOUT", "180")))  # a hang must reach the FAIL-SAFE below
    try:
        wake, prio, reasons, facts, st, prompt = MODES[a.mode](a.dry_run)
        save_state(a.mode, st, a.dry_run)
    except (Exception, PrecheckTimeout) as ex:
        # FAIL-SAFE: never silently drop a trade-safety check because the script broke.
        prio, pf = FALLBACK[a.mode]
        wake, facts, reasons = True, {"error": traceback.format_exc()[-600:]}, [f"precheck error ({ex}) — running full check"]
        prompt = (PROMPTS / pf).read_text() if pf else PULSE_PROMPT
    signal.alarm(0)
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "mode": a.mode, "wake": wake, "priority": prio,
           "reasons": reasons, "facts": facts, "dry_run": a.dry_run, "secs": round(time.time() - t0, 1)}
    if wake:
        msg = "[PRECHECK — script found: " + " | ".join(reasons)[:700] + "]\n\n" + prompt
        rec["inject"] = inject(prio, msg, a.dry_run)
    print(json.dumps(rec, default=str))
    log_line(rec)


if __name__ == "__main__":
    main()
