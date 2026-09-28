"""
safety_nets.py — HC #344 + HC #368 production safety nets for v3.3 deploy.

EIGHT watchdog classes. ANY trip → halt new entries (existing positions managed
per `exit_logic` config; never abandoned mid-trade).

Authorized: HC #344 (LIVE-READINESS gates), HC #368 (this package).
Malware-guard: this is NEW operations code under live_trading/, NOT trainer
                modification. HC #307D respected.

Usage:
    from safety_nets import SafetyNetEnsemble
    ensemble = SafetyNetEnsemble.from_config_path("configs/v33_paper_trader_config.json")
    while True:
        verdict = ensemble.evaluate(paper_state)
        if not verdict.allow_new_entry:
            log_halt(verdict.reason)

Self-test:
    python safety_nets.py --self-test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger("v33.safety_nets")


# ============================================================
# Verdict object
# ============================================================
@dataclass
class SafetyVerdict:
    allow_new_entry: bool
    halt_existing: bool = False  # forced exit-all (only for MAX_DD breach)
    reason: str = ""
    trips: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def merge(self, other: "SafetyVerdict") -> "SafetyVerdict":
        return SafetyVerdict(
            allow_new_entry=self.allow_new_entry and other.allow_new_entry,
            halt_existing=self.halt_existing or other.halt_existing,
            reason="; ".join(filter(None, [self.reason, other.reason])),
            trips=self.trips + other.trips,
        )


# ============================================================
# Watchdog base
# ============================================================
class Watchdog:
    name = "base"

    def __init__(self, config: Dict):
        self.config = config

    def evaluate(self, state: Dict[str, Any]) -> SafetyVerdict:
        raise NotImplementedError


# ============================================================
# 1. Max-Drawdown Circuit Breaker (HC #344)
#    Trip: intraday DD exceeds config threshold → HALT + exit-all.
# ============================================================
class MaxDDCircuitBreaker(Watchdog):
    name = "max_dd_circuit_breaker"

    def evaluate(self, state):
        dd_ticks = state.get("intraday_dd_ticks", 0.0)
        loss_dollars = state.get("intraday_realized_loss_dollars", 0.0)
        max_dd_ticks = float(self.config.get("max_intraday_drawdown_ticks", 50))
        max_loss = float(self.config.get("max_intraday_loss_dollars", 625))
        if dd_ticks >= max_dd_ticks:
            return SafetyVerdict(
                allow_new_entry=False, halt_existing=True,
                reason=f"MAX_DD breached: {dd_ticks:.1f}t >= {max_dd_ticks:.1f}t",
                trips=[self.name],
            )
        if loss_dollars >= max_loss:
            return SafetyVerdict(
                allow_new_entry=False, halt_existing=True,
                reason=f"MAX_LOSS_$ breached: ${loss_dollars:.0f} >= ${max_loss:.0f}",
                trips=[self.name],
            )
        return SafetyVerdict(allow_new_entry=True)


# ============================================================
# 2. Per-Trade Stop-Loss
#    NOT a halt — flags when stop is appropriate for OPEN trade.
#    (Returned as advisory; exit_logic in trader consumes it.)
# ============================================================
class PerTradeStopLoss(Watchdog):
    name = "per_trade_stop_loss"

    def evaluate(self, state):
        stop_ticks = float(self.config.get("per_trade_stop_loss_ticks", 3.0))
        open_trades = state.get("open_trades", [])
        forced_exits = []
        for t in open_trades:
            if t.get("unrealized_ticks", 0.0) <= -stop_ticks:
                forced_exits.append(t.get("trade_id"))
        if forced_exits:
            return SafetyVerdict(
                allow_new_entry=True,
                reason=f"STOP_LOSS trips: trades={forced_exits}",
                trips=[self.name],
            )
        return SafetyVerdict(allow_new_entry=True)


# ============================================================
# 3. Per-Day Trade Count Cap
# ============================================================
class TradeCountCap(Watchdog):
    name = "trade_count_cap"

    def evaluate(self, state):
        cap = int(self.config.get("max_trades_per_day", 30))
        count = int(state.get("trades_today_count", 0))
        if count >= cap:
            return SafetyVerdict(
                allow_new_entry=False,
                reason=f"TRADE_CAP reached: {count}/{cap}",
                trips=[self.name],
            )
        return SafetyVerdict(allow_new_entry=True)


# ============================================================
# 4. Model-Stale Watchdog
#    No new prediction in N seconds → halt entries.
# ============================================================
class ModelStaleWatchdog(Watchdog):
    name = "model_stale"

    def evaluate(self, state):
        thresh = float(self.config.get("model_stale_seconds", 30))
        last_pred_ts = state.get("last_prediction_ts", 0.0)
        age = time.time() - last_pred_ts if last_pred_ts > 0 else float("inf")
        if age > thresh:
            return SafetyVerdict(
                allow_new_entry=False,
                reason=f"MODEL_STALE: last pred {age:.1f}s ago (thresh {thresh:.1f}s)",
                trips=[self.name],
            )
        return SafetyVerdict(allow_new_entry=True)


# ============================================================
# 5. MBO Feed Stalled Watchdog
# ============================================================
class MBOFeedWatchdog(Watchdog):
    name = "mbo_feed_stalled"

    def evaluate(self, state):
        thresh = float(self.config.get("mbo_feed_stale_seconds", 10))
        last_evt_ts = state.get("last_mbo_event_ts", 0.0)
        age = time.time() - last_evt_ts if last_evt_ts > 0 else float("inf")
        if age > thresh:
            return SafetyVerdict(
                allow_new_entry=False,
                reason=f"MBO_FEED_STALLED: last event {age:.1f}s ago (thresh {thresh:.1f}s)",
                trips=[self.name],
            )
        return SafetyVerdict(allow_new_entry=True)


# ============================================================
# 6. σ-Spike Halt
#    If ANY head's sigma (uncertainty) jumps >5× its OOT mean → halt.
#    Surge in model uncertainty = stop trusting predictions.
# ============================================================
class SigmaSpikeHalt(Watchdog):
    name = "sigma_spike"

    def __init__(self, config):
        super().__init__(config)
        # Per-head OOT-mean σ baseline (from v3.3 fold_00_sigma.json)
        # Updated from actual v3.3 sigma_json once available; defaults conservative
        self.oot_mean_sigma = config.get("oot_mean_sigma_per_head", {})
        self.multiplier = float(config.get("sigma_spike_multiplier_halt", 5.0))

    def evaluate(self, state):
        current_sigmas = state.get("current_sigma_per_head", {})
        if not current_sigmas or not self.oot_mean_sigma:
            return SafetyVerdict(allow_new_entry=True)
        spikes = []
        for h, sig in current_sigmas.items():
            base = self.oot_mean_sigma.get(h, 0.0)
            if base > 0 and sig > self.multiplier * base:
                spikes.append(f"{h}={sig:.3f}vs{base:.3f}")
        if spikes:
            return SafetyVerdict(
                allow_new_entry=False,
                reason=f"SIGMA_SPIKE: {spikes}",
                trips=[self.name],
            )
        return SafetyVerdict(allow_new_entry=True)


# ============================================================
# 7. Confluence-Disagreement Halt
#    If primary heads disagree on sign at top confidence band → no trade.
# ============================================================
class ConfluenceDisagreementHalt(Watchdog):
    name = "confluence_disagreement"

    def evaluate(self, state):
        require_n = int(self.config.get("min_agreeing_heads", 2))
        primary = self.config.get("agreement_horizon_set", ["log_ret_1s", "log_ret_5s"])
        preds = state.get("current_predictions", {})
        if not preds:
            return SafetyVerdict(allow_new_entry=True)
        signs = []
        for h in primary:
            v = preds.get(h)
            if v is None:
                continue
            signs.append(1 if v > 0 else -1 if v < 0 else 0)
        if len(signs) < require_n:
            return SafetyVerdict(allow_new_entry=True)
        # Count majority sign
        pos = sum(1 for s in signs if s > 0)
        neg = sum(1 for s in signs if s < 0)
        agreeing = max(pos, neg)
        if agreeing < require_n:
            return SafetyVerdict(
                allow_new_entry=False,
                reason=f"CONFLUENCE_DISAGREE: signs={signs}, max_agree={agreeing}<{require_n}",
                trips=[self.name],
            )
        return SafetyVerdict(allow_new_entry=True)


# ============================================================
# 8. Inference Latency Watchdog
#    If forward pass > N ms → flag (advisory; not halting unless persistent).
# ============================================================
class InferenceLatencyWatchdog(Watchdog):
    name = "inference_latency"

    def __init__(self, config):
        super().__init__(config)
        self.consecutive_slow = 0

    def evaluate(self, state):
        thresh_ms = float(self.config.get("inference_latency_max_ms", 100))
        last_ms = float(state.get("last_inference_latency_ms", 0.0))
        if last_ms > thresh_ms:
            self.consecutive_slow += 1
            if self.consecutive_slow >= 5:
                return SafetyVerdict(
                    allow_new_entry=False,
                    reason=f"INFERENCE_LATENCY: {last_ms:.1f}ms > {thresh_ms:.1f}ms for {self.consecutive_slow} preds",
                    trips=[self.name],
                )
        else:
            self.consecutive_slow = 0
        return SafetyVerdict(allow_new_entry=True)


# ============================================================
# 9. Position Reconciliation (extra — broker vs internal state)
# ============================================================
class PositionReconcileWatchdog(Watchdog):
    name = "position_reconcile"

    def evaluate(self, state):
        internal = state.get("internal_position_contracts", 0)
        broker = state.get("broker_position_contracts", None)
        if broker is None:
            return SafetyVerdict(allow_new_entry=True)  # no broker feed = paper mode
        if internal != broker:
            return SafetyVerdict(
                allow_new_entry=False, halt_existing=True,
                reason=f"POS_MISMATCH: internal={internal} broker={broker}",
                trips=[self.name],
            )
        return SafetyVerdict(allow_new_entry=True)


# ============================================================
# Ensemble
# ============================================================
class SafetyNetEnsemble:
    def __init__(self, watchdogs: List[Watchdog]):
        self.watchdogs = watchdogs

    @classmethod
    def from_config(cls, config: Dict) -> "SafetyNetEnsemble":
        sn_cfg = config.get("safety_nets", {})
        # Merge confluence config in for ConfluenceDisagreementHalt
        confluence_cfg = config.get("confluence_gate", {})
        merged_confluence = {**sn_cfg, **confluence_cfg}

        watchdogs = [
            MaxDDCircuitBreaker(sn_cfg),
            PerTradeStopLoss(sn_cfg),
            TradeCountCap({**sn_cfg, **config.get("sizing", {})}),
            ModelStaleWatchdog(sn_cfg),
            MBOFeedWatchdog(sn_cfg),
            SigmaSpikeHalt({**sn_cfg, **config.get("model_health", {})}),
            ConfluenceDisagreementHalt(merged_confluence),
            InferenceLatencyWatchdog(sn_cfg),
            PositionReconcileWatchdog(sn_cfg),
        ]
        return cls(watchdogs)

    @classmethod
    def from_config_path(cls, path: str) -> "SafetyNetEnsemble":
        with open(path) as fh:
            return cls.from_config(json.load(fh))

    def evaluate(self, state: Dict[str, Any]) -> SafetyVerdict:
        verdict = SafetyVerdict(allow_new_entry=True)
        for w in self.watchdogs:
            v = w.evaluate(state)
            verdict = verdict.merge(v)
        return verdict

    def names(self) -> List[str]:
        return [w.name for w in self.watchdogs]


# ============================================================
# Self-test
# ============================================================
def _self_test():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "v33_config_template.json"
    if not cfg_path.exists():
        print(f"[self-test] config not found at {cfg_path}", file=sys.stderr)
        return 1
    ens = SafetyNetEnsemble.from_config_path(str(cfg_path))
    print(f"[self-test] loaded {len(ens.watchdogs)} watchdogs: {ens.names()}")

    # Case 1: healthy state → allow
    healthy = {
        "intraday_dd_ticks": 5.0,
        "intraday_realized_loss_dollars": 0,
        "trades_today_count": 3,
        "last_prediction_ts": time.time() - 1.0,
        "last_mbo_event_ts": time.time() - 0.05,
        "current_sigma_per_head": {"log_ret_1s": 0.05},
        "current_predictions": {"log_ret_1s": -0.001, "log_ret_5s": -0.002},
        "last_inference_latency_ms": 12.0,
        "internal_position_contracts": 1,
        "broker_position_contracts": 1,
        "open_trades": [],
    }
    v = ens.evaluate(healthy)
    print(f"[self-test] HEALTHY: allow={v.allow_new_entry} halt_existing={v.halt_existing}")
    assert v.allow_new_entry and not v.halt_existing, "healthy state should allow"

    # Case 2: max-dd breach → halt + force exit
    bad_dd = dict(healthy, intraday_dd_ticks=60.0)
    v = ens.evaluate(bad_dd)
    print(f"[self-test] MAX_DD: allow={v.allow_new_entry} halt={v.halt_existing} reason={v.reason}")
    assert not v.allow_new_entry and v.halt_existing, "MAX_DD should halt all"

    # Case 3: stale model
    stale = dict(healthy, last_prediction_ts=time.time() - 60)
    v = ens.evaluate(stale)
    print(f"[self-test] MODEL_STALE: allow={v.allow_new_entry} reason={v.reason}")
    assert not v.allow_new_entry, "stale model should halt entries"

    # Case 4: confluence disagree
    disagree = dict(healthy, current_predictions={"log_ret_1s": +0.001, "log_ret_5s": -0.001})
    v = ens.evaluate(disagree)
    print(f"[self-test] CONFLUENCE_DISAGREE: allow={v.allow_new_entry} reason={v.reason}")
    # With min_agreeing_heads=2 from config and 2 disagreeing signs, max_agree=1<2 → halt
    assert not v.allow_new_entry, "confluence disagree should halt"

    # Case 5: trade cap
    capped = dict(healthy, trades_today_count=999)
    v = ens.evaluate(capped)
    print(f"[self-test] TRADE_CAP: allow={v.allow_new_entry} reason={v.reason}")
    assert not v.allow_new_entry, "trade cap should halt"

    # Case 6: position mismatch
    mismatch = dict(healthy, internal_position_contracts=1, broker_position_contracts=3)
    v = ens.evaluate(mismatch)
    print(f"[self-test] POS_MISMATCH: allow={v.allow_new_entry} halt={v.halt_existing} reason={v.reason}")
    assert not v.allow_new_entry and v.halt_existing, "position mismatch should halt all"

    print("[self-test] ALL PASS ✓")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        sys.exit(_self_test())
    print("safety_nets.py — use --self-test to run smoke tests")
