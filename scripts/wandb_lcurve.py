#!/usr/bin/env python3
"""Real-time wandb logger: tail deepmd's ``lcurve.out`` and stream rows to wandb.

deepmd appends one whitespace-separated row to ``lcurve.out`` every ``disp_freq``
steps and ``flush()``-es immediately (deepmd/pt/train/training.py:2314), preceded
by a header line such as::

    # step      rmse_val    rmse_trn  rmse_e_val  rmse_e_trn  rmse_f_val ...      lr

This sidecar runs alongside training, follows the file, parses each *completed*
data row, and logs the metrics to a wandb run keyed by the training step. It is
genuinely real-time (lcurve.out is flushed per disp_freq, unlike the TensorBoard
SummaryWriter which buffers ~120 s). It exits once the launcher drops a sentinel
file (written after ``torchrun`` returns) and the final rows have been drained.

Run id is unique per job (so reruns never collide on step ordering and a crashed
sidecar can ``resume`` the same run mid-job); the wandb *display name* is the
clean, timestamp-free identifier.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import wandb


def _parse_header(line: str) -> list[str]:
    # "# step  rmse_val rmse_trn ... lr" -> ["step", "rmse_val", ..., "lr"]
    return line.lstrip("#").split()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logfile", required=True, help="path to deepmd lcurve.out")
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True, help="wandb display name (no timestamp)")
    ap.add_argument("--id", required=True, help="unique-per-job wandb run id")
    ap.add_argument("--sentinel", required=True, help="file whose appearance ends the tail")
    ap.add_argument("--poll", type=float, default=5.0)
    args = ap.parse_args()

    wandb.init(project=args.project, name=args.name, id=args.id, resume="allow")

    cols: list[str] | None = None
    pos = 0
    last_step = -1
    pending = ""

    def drain() -> None:
        nonlocal pos, cols, last_step, pending
        if not os.path.exists(args.logfile):
            return
        with open(args.logfile) as f:
            f.seek(pos)
            pending_chunk = f.read()
            pos = f.tell()
        buf = pending + pending_chunk
        lines = buf.split("\n")
        # keep the last (possibly incomplete) line for the next poll
        pending = lines.pop()
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if cols is None and "step" in s:
                    cols = _parse_header(s)
                continue
            toks = s.split()
            if cols is None or len(toks) != len(cols):
                continue
            try:
                step = int(float(toks[0]))
            except ValueError:
                continue
            if step <= last_step:  # enforce monotonic step for wandb
                continue
            metrics = {}
            for c, t in zip(cols[1:], toks[1:]):
                try:
                    v = float(t)
                except ValueError:
                    continue
                if math.isfinite(v):
                    metrics[c] = v
            if metrics:
                wandb.log(metrics, step=step)
                last_step = step

    try:
        while True:
            drain()
            if os.path.exists(args.sentinel):
                drain()  # final pass to catch the last flushed rows
                break
            time.sleep(args.poll)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
