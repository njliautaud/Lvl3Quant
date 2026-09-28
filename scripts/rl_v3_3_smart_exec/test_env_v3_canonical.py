"""
HC #399 #1 binding unit test for CanonicalReplayEnv.

Spec (from build brief):
    "Take ONE deterministic day of signals, feed the SAME (timestep, action)
     sequence to (a) env.step() loop and (b) direct full_market_replay()
     call. The episode-end reward from (a) MUST equal the canonical
     net_ticks from (b) to 4 decimal places. If they don't match, the env
     is wrong — do not move on."

CHOICE OF "DIRECT CANONICAL CALL":
    `full_market_replay()` is a percentile-gated trade-generation pipeline,
    NOT a per-event pricing routine. It takes ONE TradeConfig and walks the
    whole NPZ generating signals + filling them. It cannot accept an
    arbitrary agent action sequence.

    The canonical TRUTH for "agent action sequence → trade PnL" is
    `canonical_reprice()` from ppo_v2_1_canonical_replay_eval.py, which
    walks a list of trades (with entry_idx, side, action_type) through
    the SAME primitives (_queue_position_model, _entry_price_edge_ticks,
    target_log_ret_30s, COMMISSION_RT_TICKS).

    This test therefore compares:
      (a) sum of env step rewards over an episode
      (b) sum of canonical_reprice() net per trade for the SAME (entry_step,
          side, action_type) trade list

    Both MUST agree to 4 decimal places. This is the strongest verification
    available for an action-sequence-driven env — exactly equivalent to what
    HC #397 / HC #399 demand of every reported number.

NOT MALWARE. Test-only script. Read-only on data dirs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))
sys.path.insert(0, str(LVL3 / "scripts" / "rl_v3_3_smart_exec"))

from env_v3_canonical import (  # noqa: E402
    CanonicalReplayEnv,
    A_HOLD, A_BID, A_ASK, A_MKT_BUY, A_MKT_SELL, A_CANCEL, A_EXIT_POS,
    PASSIVE_FILL_WINDOW, MAX_HOLD_STEPS, HC344_PENALTY_PER_VIOLATION_TICKS,
    DAY_CONC_GATE,
)

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    PRICE_UNIT_TO_TICKS,
    ES_RT_COMMISSION_TICKS_DEFAULT,
    ES_SPREAD_TICKS_RTH_DEFAULT,
    _queue_position_model,
    _entry_price_edge_ticks,
    _load_fifo_labels,
)

RT_COMM = ES_RT_COMMISSION_TICKS_DEFAULT  # 0.376
SPREAD = ES_SPREAD_TICKS_RTH_DEFAULT       # 1.0 (used by passive edge_offset = 0)

NPZ = LVL3 / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
LABELS = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"

TOL = 1e-4


def canonical_price_trade_list(trades, npz_path, labels_dir):
    """Mirror of canonical_reprice() — accept a list of dicts with
    {entry_step, side, action_type} and return total net_ticks +
    adv_sel_30s sum."""
    d = np.load(npz_path, allow_pickle=True)
    n_npz = int(d["n_samples"])
    fifo = _load_fifo_labels(labels_dir, [str(x) for x in d["oot_dates"]])
    n_fifo = sum(fifo["_n_per_day"])
    n_total = min(n_npz, n_fifo)
    tgt = d["target_log_ret_30s"][:n_total].astype(np.float64)
    mk = d["mask_log_ret_30s"][:n_total].astype(bool) & np.isfinite(tgt)

    net_total = 0.0
    adv_total = 0.0
    n_filled = 0
    for tr in trades:
        idx = int(tr["entry_step"])
        side = int(tr["side"])
        if idx < 0 or idx >= n_total:
            continue
        if tr["action_type"] == "market":
            if not mk[idx]:
                continue
            lr = float(tgt[idx]) * PRICE_UNIT_TO_TICKS
            edge = 0.0  # HC #392
            net = side * lr + edge - RT_COMM
            net_total += net
            adv_total += min(0.0, side * lr)
            n_filled += 1
        elif tr["action_type"] == "passive":
            side_key = "long" if side == +1 else "short"
            fl = fifo[f"tp4sl3_{side_key}_filled"][idx : idx + 1]
            er = fifo[f"tp4sl3_{side_key}_exit_reason"][idx : idx + 1]
            ht = fifo[f"tp4sl3_{side_key}_hold_time_ns"][idx : idx + 1]
            fm, _, _ = _queue_position_model(
                "passive_at_touch",
                cancel_eval_window=50,
                label_filled=fl,
                label_exit_reason=er,
                label_hold_time_ns=ht,
            )
            if not fm[0] or not mk[idx]:
                continue
            lr = float(tgt[idx]) * PRICE_UNIT_TO_TICKS
            edge = _entry_price_edge_ticks("passive_at_touch", SPREAD)
            net = side * lr + edge - RT_COMM
            net_total += net
            adv_total += min(0.0, side * lr)
            n_filled += 1
    return {
        "net_total": net_total,
        "adv_sel_total": adv_total,
        "n_filled": n_filled,
    }


def hc344_surrogate_penalty(n_filled, net_total):
    """Mirror of env's _episode_end_penalties HC #344 component."""
    if n_filled >= 2 and net_total > 0:
        return -HC344_PENALTY_PER_VIOLATION_TICKS * (1.0 - DAY_CONC_GATE)
    return 0.0


def build_deterministic_action_sequence(env, day_idx):
    """Deterministic mix of market + passive actions on day_idx.

    Strategy: every ~500 steps, place a market trade alternating long/short;
    every ~1000 steps, place a passive bid + cancel after a few steps;
    every ~700 steps, place a passive ask that's allowed to fill.

    This exercises: A_MKT_BUY, A_MKT_SELL, A_BID, A_ASK, A_CANCEL,
    A_EXIT_POS (forced via MAX_HOLD_STEPS auto-exit), and the natural
    end-of-episode close.
    """
    actions = []
    day_len = env._day_starts[day_idx + 1] - env._day_starts[day_idx]
    side_toggle = +1
    for step_in_day in range(day_len):
        a = A_HOLD
        # Market entries every 500 steps (alternating sides)
        if step_in_day % 500 == 100:
            a = A_MKT_BUY if side_toggle == +1 else A_MKT_SELL
            side_toggle *= -1
        # Passive bids every 1000 steps — cancel 10 steps later
        elif step_in_day % 1000 == 300:
            a = A_BID
        elif step_in_day % 1000 == 310:
            a = A_CANCEL
        # Passive asks every 700 steps — let them fill via PASSIVE_FILL_WINDOW
        elif step_in_day % 700 == 555:
            a = A_ASK
        # Explicit EXIT_POS every 2000 steps when in position
        elif step_in_day % 2000 == 1500:
            a = A_EXIT_POS
        actions.append(a)
    return actions


def run_env_episode(env, actions, day_idx):
    """Run env on day_idx with the given action sequence; return total
    reward sum + the list of (entry_step, side, action_type, filled) events."""
    obs, info = env.reset(options={"day_idx": day_idx})
    # ^ NOTE: reset() ignores options; episode day is set via constructor
    # _episode_day_override OR random. We use _episode_day_override below.

    cum = 0.0
    n_steps = 0
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        cum += r
        n_steps += 1
        if term or trunc:
            break
    return cum, env.events, env.get_episode_summary()


def main():
    print("=" * 78)
    print("HC #399 #1 UNIT TEST: env_v3_canonical reward == canonical_reprice")
    print("=" * 78)

    # Pin to day 0 deterministically
    DAY = 0
    env = CanonicalReplayEnv(
        npz_path=NPZ, labels_dir=LABELS, seed=0,
        episode_day_idx=DAY, deterministic_queue=True,
    )
    print(f"Env loaded: N={env.N}, n_days={env._n_days}, "
          f"day{DAY} = {env.oot_dates[DAY]} "
          f"[{env._day_starts[DAY]}, {env._day_starts[DAY+1]})")

    actions = build_deterministic_action_sequence(env, DAY)
    print(f"Action sequence length: {len(actions)}")
    print(f"Action distribution: "
          f"HOLD={actions.count(A_HOLD)}, BID={actions.count(A_BID)}, "
          f"ASK={actions.count(A_ASK)}, MKT_BUY={actions.count(A_MKT_BUY)}, "
          f"MKT_SELL={actions.count(A_MKT_SELL)}, CANCEL={actions.count(A_CANCEL)}, "
          f"EXIT={actions.count(A_EXIT_POS)}")

    cum_env, events, summary = run_env_episode(env, actions, DAY)
    print(f"\n--- ENV RESULT ---")
    print(f"Sum of step rewards: {cum_env:+.6f} ticks")
    print(f"n_events (all attempts): {len(events)}")
    print(f"n_filled: {sum(1 for e in events if e.filled)}")
    print(f"Summary: {summary}")

    # Build the canonical trade list from env events
    trade_list = []
    for e in events:
        trade_list.append({
            "entry_step": e.entry_step,
            "side": e.side,
            "action_type": e.action_type,
        })
    print(f"\n--- CANONICAL DIRECT RESULT ---")
    canon = canonical_price_trade_list(trade_list, NPZ, LABELS)
    print(f"Canonical net_total: {canon['net_total']:+.6f} ticks")
    print(f"Canonical n_filled: {canon['n_filled']}")
    print(f"Canonical adv_sel_total: {canon['adv_sel_total']:+.6f} ticks")

    # Apply the SAME HC #344 surrogate penalty the env applies at end-of-episode
    penalty = hc344_surrogate_penalty(canon["n_filled"], canon["net_total"])
    canon_with_penalty = canon["net_total"] + penalty
    print(f"HC #344 surrogate penalty: {penalty:+.6f} ticks")
    print(f"Canonical + penalty: {canon_with_penalty:+.6f} ticks")

    diff = abs(cum_env - canon_with_penalty)
    print(f"\n--- VERDICT ---")
    print(f"|env_reward - (canon + penalty)| = {diff:.8f}")
    print(f"Tolerance: {TOL}")

    if diff < TOL:
        print(f"\n  ✅ PASS — env reward matches canonical to {TOL} ticks")
        print(f"  HC #399 #1 binding requirement satisfied.")
        return 0
    else:
        print(f"\n  ❌ FAIL — gap of {diff:.6f} ticks")
        print(f"  Env reward signal is NOT canonical. DO NOT proceed to training.")
        print(f"\n  Per-event breakdown for debugging:")
        for i, (e, tr) in enumerate(zip(events, trade_list)):
            canon_single = canonical_price_trade_list([tr], NPZ, LABELS)
            print(f"   ev{i}: entry_step={e.entry_step} side={e.side:+d} "
                  f"type={e.action_type} filled={e.filled} "
                  f"env_net={e.canon_net_ticks:+.4f} "
                  f"canon_net={canon_single['net_total']:+.4f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
