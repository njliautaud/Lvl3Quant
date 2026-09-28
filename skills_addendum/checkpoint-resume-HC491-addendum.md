# HC #491 ADDENDUM — Silent-Resume-Failure — 2026-05-23

This addendum supplements `~/.claude/skills/infra/checkpoint-resume-after-kill/SKILL.md` with
two corrections that emerged from a real failure on Neptune at 00:44 ET 2026-05-23.

## The Failure (don't repeat it)

A prior agent killed a diverged CNN-Mamba v3.4.2 training run and "resumed" it with:

```
cd /home/nick/Lvl3Quant && ... train_cnn_mamba_v3_2.py \
    --output-dir output/cnn_mamba_v3_4_2_hc477fix_v2 \
    --resume-from-intra-ckpt fold_00_intra_ckpt.pt          # RELATIVE PATH
```

`Path("fold_00_intra_ckpt.pt")` resolves against the trainer's cwd `/home/nick/Lvl3Quant`.
The ckpt actually lives at `/home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_hc477fix_v2/fold_00_intra_ckpt.pt`.
The trainer logged ONE line — `Resume ckpt path does not exist: fold_00_intra_ckpt.pt` —
then silently fell through to fresh-start (epoch 0, batch 0). The previous agent thought it was
resuming; ~22 minutes of wasted GPU before the next session caught it.

## CORRECTION 1 — ABSOLUTE PATH MANDATORY

`--resume-from-intra-ckpt` MUST be an absolute path. Reject any value that doesn't start with `/`
(Linux) or `C:\` (Windows). Helper wrappers must `realpath`/`Resolve-Path` before passing to the
trainer.

Pre-launch verify:

```bash
test -f "$CKPT" || { echo "ABORT: ckpt $CKPT missing"; exit 1; }
case "$CKPT" in /*) : ;; *) echo "ABORT: ckpt must be absolute"; exit 1 ;; esac
```

## CORRECTION 2 — VERIFY-TIMING IS WRONG IN PARENT SKILL

Parent SKILL.md step 4 says: "After 60s the helper grep's the new log for `RESUMED from intra_ckpt`."

That's wrong for Neptune CNN-Mamba. The trainer's resume-load code (`train_cnn_mamba_v3_2.py:1721-1753`)
runs inside `train_one_fold()`, which only fires after dataset construction + T1/T2/T3
feature-stats compute. On the smart_v3 60-day dataset that takes 25–35 minutes.

Correct verify protocol:

1. **T+60s**: process alive (`ps -p $PID`), log shows `mamba_ssm CUDA kernels` loaded,
   `Dataset: N days, M samples`, `Computing T1/T2/T3 feature stats`. If log shows
   `argument --resume-from-intra-ckpt: expected one argument` OR `usage: ...` argparse
   error, ABORT — the shell variable was unexpanded.

2. **T+~30min** (after feature stats complete): `grep -E "RESUMED from intra_ckpt|Resume ckpt path does not exist|Resume ckpt load failed"` on the new log. Exactly ONE of those three should appear. Only `RESUMED` is a pass; the other two are fall-through-to-fresh-start.

3. **T+~32min**: confirm the first `Batch N/22137` line shows `N > 0` (resume position) NOT
   `Batch 100/22137` (fresh start at first log interval).

## CORRECTION 3 — PRE-LAUNCH DRY-LOAD (the only verify that fits 60s)

Because the in-trainer verify takes 30 min, the meaningful 60s-verify is pre-launch, not
post-launch:

```bash
python3 -c "import torch; sd = torch.load('$CKPT', map_location='cpu', weights_only=False); \
    print('ckpt keys:', list(sd.keys())[:8]); \
    assert 'model_state' in sd, 'not a v3.2 intra_ckpt'; \
    print('global_step:', sd.get('global_step'), 'epoch:', sd.get('epoch'), 'batch:', sd.get('batch'))"
```

If this dry-load works, the trainer's load_state_dict will work too.

## CORRECTION 4 — INTRA_CKPT GETS OVERWRITTEN BY THE FAILED FRESH-START

If the silent fall-through ran for any duration, the fresh-start has been writing its own
`fold_NN_intra_ckpt.pt` on the standard intra-save cadence. The good pre-divergence ckpt is now
overwritten. Recovery:

1. Look for sibling files with suffix `.deferred_*`, `.diverged_*.bak`, `.stalled_*.bak`.
   The latest of these is the LAST-KNOWN-GOOD pre-fresh-start state.
2. `cp` the good sibling over `fold_NN_intra_ckpt.pt`.
3. Relaunch with the absolute path to the now-restored file.

## Trainer source-of-truth locations

```
alpha_discovery/deep_models/train_cnn_mamba_v3_2.py
  line 1721  train_one_fold entry
  line 1726  if resume_from_intra_ckpt is not None:
  line 1727  rp = Path(resume_from_intra_ckpt)       <- path resolution
  line 1751  Resume ckpt load failed (silent fall-through #1)
  line 1753  Resume ckpt path does not exist (silent fall-through #2)
  line 1883  --resume-from-intra-ckpt arg definition
  line 1920  Path(args.resume_from_intra_ckpt) if ... else None
  line 1938  resume_from_intra_ckpt=resume_path
```

Both fall-throughs are `logger.warning` not `raise`. The trainer was explicitly designed to fall
through on bad ckpt path — which means the caller carries 100% of the verification burden.

Created 2026-05-23 05:13 ET after catching the 00:44 silent-fail during recovery.

---

## ADDENDUM 2 — 2026-05-23 05:18 ET — RAZER SERVICES-SESSION-0 GOTCHA

Different failure mode caught by `RAZER_GPU_IDLE` trigger at 05:18. Recovery check ran:

```cmd
tasklist | findstr python.exe
```

Returned nothing → concluded Razer was idle. WRONG. The actual training process was running under Services session 0 (because it was launched via `schtasks` long ago and persists across the user-session disconnect Razer experienced).

**Correct Razer python-enumeration commands**:

```cmd
tasklist /svc | findstr python                                   # shows services-session entries
wmic process where "Name='python.exe'" get ProcessId,CommandLine # full enumeration
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

The nvidia-smi compute-apps query is the most reliable: it shows PIDs holding CUDA contexts regardless of session.

**Recovery rule**: before declaring Razer idle, run all three. If any returns a python process whose CommandLine matches a known training script, Razer is NOT idle. Add this check to the `recovery` skill workflow.
