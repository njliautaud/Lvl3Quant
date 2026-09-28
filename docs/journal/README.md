# Research journal

This folder holds the working records of the autonomous research agent that ran the Lvl3Quant program: its directives, experiment ledger, session state and plans. It is published as it was written, not as a cleaned-up narrative. It shows how the research was actually steered, checked and corrected over several months.

The code and the reports show *what* was built and found. The journal shows *how* those decisions were made:
- which hypotheses were tried
- which gates they faced
- which "wins" were later retracted
- how the human direction and the agent's execution fit together

---

## How the program was run

- **The human sets constraints; the agent executes.** The author gave direction as numbered **hard constraints** (`HC #NNN`). Each one records the directive quoted verbatim, binding rules (`R1`, `R2`, ...), what it supersedes, and a change log. The agent reads `DIRECTIVES.md` at the start of every session and treats it as binding. The numbering passed #800 over the program, and the file is periodically consolidated.
- **Every experiment is logged with a verdict.** Runs are appended to the run history with the node, the configuration, the key metrics and a verdict: validated, rejected, dead, or superseded. Negative results are kept on purpose so they are not re-run, and they often explain *why* an idea failed (costs, regime dependence, leakage, too few samples).
- **Continuity across restarts.** The agent's context is finite, so it writes out its session state (cluster status, what is running, what is next) and reloads it after each restart. The node ledger and task queue keep GPUs busy and make idle time visible.
- **Self-correction is recorded, not erased.** When a later check overturns an earlier claim, the correction is added next to the original. Examples include a 5-day out-of-time result that did not hold at 17 days, a proxy-cost win that turned negative under FIFO replay, and an exit model whose gain came from look-ahead. The failure modes of the agent itself are tracked in [`../research/WEAKNESSES.md`](../research/WEAKNESSES.md).

---

## File guide

| File | What it is |
|---|---|
| `DIRECTIVES.md` | Active hard constraints, newest first. The "Merged" sections near the end (validation framework, risk-adjusted metrics) are the most useful summary of the research standards. |
| `DIRECTIVES_ARCHIVE_*.md` | Earlier snapshots of the directive set, from the ES microstructure phase. |
| `RUN_HISTORY.md` | Condensed experiment ledger for the later phase (daily strategies, agentic evolution, lockbox validation). |
| `RUN_HISTORY_archive_*.md`, `RUN_HISTORY_ARCHIVE_*.md` | Full, detailed experiment logs from earlier phases, including the ES deep-learning and execution work. |
| `SESSION_STATE.md` and its archives | The session-by-session record of cluster state, active jobs, decisions and next steps. |
| `NODE_LEDGER.md` | Per-node job ledger: what each GPU/CPU node is running, crash history, and why any node is idle. |
| `TASK_QUEUE.md`, `WEEK_PLAN.md`, `WIP.md`, `HANDOFF.md` | Planning and continuity files. `HANDOFF.md` is deprecated and kept for history. |
| `IMPROVEMENT_BACKLOG.md` | An append-only log of process gaps found in periodic self-checks, and how each was closed. |
| Dated notes and reports (e.g. `*_2026MMDD*.md`) | Point-in-time plans, morning briefs and sub-agent reports. |

---

## Suggested reading order

1. **`DIRECTIVES.md`, "Merged — validation framework".** This is the gate every strategy had to pass: a five-gate backtest, a six-point adversarial check, paper trading, then wiring into the live pipeline and verifying it works.
2. **[`../research/STRATEGY_2026-05-28.md`](../research/STRATEGY_2026-05-28.md).** An honest scorecard after three months of ES order-book research: the signal is real, the edge is smaller than the commission, and the document lays out the resulting pivot decision.
3. **`RUN_HISTORY_archive_pre_aug10.md`.** Search for `VERDICT`, `DEAD` or `KILLED` to see how hypotheses were closed.
4. **`RUN_HISTORY.md`.** The agentic-evolution phase, including a strategy that converged in-sample but failed its lockbox window.
5. **`DIRECTIVES.md`, the newest entries.** These include a post-mortem after a losing streak in a small live options sleeve. Trading was paused until new risk controls were in force.

---

## Reading notes and caveats

- **These are raw working notes.** Times are US Eastern. Entries refer to internal names (nodes such as Jupiter, Neptune, Razer and Saturn; process IDs; MLflow run IDs; `HC` numbers) and to absolute paths on the original cluster.
- **Numbers are point-in-time.** Many interim results were later revised, and the latest verdict on a topic is the one that counts. Metrics are research metrics: walk-forward/out-of-time backtests, replay simulation or paper trading. None of them is an audited performance record.
- **Some tone is operational.** The files mix research notes with operations work (crash recovery, watchdogs, scheduling) because the same agent did both.
- **Sensitive details are redacted.** Credentials, account identifiers, network addresses and personal details were removed or replaced with placeholders (for example `XXXXXXXXX`) before publication.

This is research material, not investment advice. © Nicholas Liautaud — all rights reserved.
