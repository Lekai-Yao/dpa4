#!/usr/bin/env bash
# Launch DPA4-Air training on MPtrj inside the dpa4-train cluster image (8x A100).
#
# Schedule: 20 epochs. train.lmdb = 1,420,395 frames -> 157,809 same-nloc batches
# per pass; over 8 ranks that is 19,727 optimizer steps/epoch, so the config pins
# numb_steps = 20*19,727 = 394,540 (pinned, not num_epoch, so all 8 DDP ranks agree
# on the step count and the wsd LR decay horizon).
#
# Prerequisite (one-time): convert the ASE .aselmdb data to deepmd LMDB, e.g.
#   python scripts/aselmdb_to_deepmd_lmdb.py \
#     --src /mnt/afs/share/dataset/periodicSystem/MPtrj/aselmdb/train \
#     --dst /mnt/afs/share/dataset/periodicSystem/MPtrj/deepmd_lmdb/train.lmdb
#   python scripts/aselmdb_to_deepmd_lmdb.py \
#     --src /mnt/afs/share/dataset/periodicSystem/MPtrj/aselmdb/val \
#     --dst /mnt/afs/share/dataset/periodicSystem/MPtrj/deepmd_lmdb/val.lmdb
set -euo pipefail

REPO=/mnt/afs/home/yaolekai/MLIP/DPA4
CFG="$REPO/dpa4/configs/dpa4_mptrj.json"

# The image ships deps only; editable-install the mounted deepmd-kit once
# (training-only path; build isolation transiently pulls the toolchain + TF).
uv pip install --no-deps -e "$REPO/deepmd-kit"

# Real-time wandb is fed by a sidecar that tails lcurve.out (see below); only
# wandb is needed here (its API key / timeout are configured outside this script).
uv pip install -q wandb

# DeePMD writes every output (checkpoints, lcurve.out, stat hdf5) to the CWD and
# has NO --run-dir/--identifier flag, so cd-ing into the run dir IS the run-dir
# mechanism. Mirror equiformer's <run-dir>/<identifier> with a dpa4-air-tagged,
# timestamped dir; ckpt files are further prefixed "dpa4-air.ckpt" via save_ckpt.
RUNBASE=/mnt/afs/share/checkpoint/dpa4/yaolekai
IDENT="${IDENT:-dpa4-air_mptrj_20ep}_$(date +%Y%m%d-%H%M%S)"
RUNDIR="$RUNBASE/$IDENT"
mkdir -p "$RUNDIR"
cd "$RUNDIR"

# Real-time wandb: a sidecar tails lcurve.out and streams each flushed row to the
# `dpa4_mptrj` project live. deepmd writes lcurve.out only on global rank 0 (node
# 0), so run the sidecar on node 0 only. The wandb *display name* is the clean,
# timestamp-free `dpa4-air_mptrj_20ep`; the run *id* is the unique-per-job IDENT
# (so reruns never collide and a restarted sidecar resumes the same run).
SENTINEL="$RUNDIR/.train_done"
SIDE_PID=""
if [ "${SENSECORE_PYTORCH_NODE_RANK:-0}" = "0" ]; then
  python "$REPO/dpa4/scripts/wandb_lcurve.py" \
    --logfile "$RUNDIR/lcurve.out" \
    --project dpa4_mptrj \
    --name dpa4-air_mptrj_20ep \
    --id "$IDENT" \
    --sentinel "$SENTINEL" &
  SIDE_PID=$!
fi

# torchrun env vars are provided by the cluster; defaults cover a single 8-GPU node.
# Don't let a nonzero training exit abort the script before the sidecar is flushed
# (so a crashed run still ships its partial curves). Capture the code and re-exit.
set +e
torchrun \
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}" \
  --nnodes "${SENSECORE_PYTORCH_NNODES:-1}" \
  --node_rank "${SENSECORE_PYTORCH_NODE_RANK:-0}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" \
  --master_port "${MASTER_PORT:-29500}" \
  --no-python \
  dp --pt train "$CFG"
train_rc=$?
set -e

# Signal the sidecar to drain the final rows and finish the wandb run, then wait.
if [ -n "$SIDE_PID" ]; then
  touch "$SENTINEL"
  wait "$SIDE_PID" 2>/dev/null || true
fi

exit "$train_rc"
