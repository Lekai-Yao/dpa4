# dpa4 — DPA4 reproduction recipe

This repo is the **experiment recipe** for reproducing DPA4 with
[DeePMD-kit](https://github.com/deepmodeling/deepmd-kit). It version-controls the
training configs, evaluation scripts, and the environment definition — but **not**
the deepmd-kit source code or any model artifacts.

```
DPA4/
├── .venv/        # local dev env (uv venv; not in any git repo)
├── deepmd-kit/   # repo A: fork of upstream, clone-only (read / pull)
└── dpa4/         # repo B: THIS repo — configs, scripts, Dockerfile
```

Code, experiments, and environment are kept **decoupled and peer-level**:
`deepmd-kit/` is treated as an upgradeable software dependency, `dpa4/` is the
version-controlled recipe, `.venv/` is the local environment.

## Reproduction goal

Reproduce the DPA4 baseline training/evaluation on the water dataset, then extend
to the target systems. The training config in `configs/dpa4.json` is copied from
`../deepmd-kit/examples/water/dpa4/input.json` (deepmd-kit commit `99c1ece2`) and
is a starting point — adjust `training_data` / `validation_data` paths and
`numb_steps` before real runs.

## Layout

```
dpa4/
├── README.md            # this file
├── requirements.txt     # runtime deps snapshot (excludes deepmd-kit itself); shared by local + image
├── Dockerfile           # full env image (train + inference), CUDA 12.4 -devel base (has nvcc)
├── Dockerfile.train     # training-only env image, CUDA 12.4 -runtime base (lighter, no nvcc)
├── .dockerignore
├── .gitignore
├── configs/
│   ├── dpa4.json        # DPA4 water training config (provenance noted in its _source field)
│   └── dpa4_mptrj.json  # DPA4-Air on MPtrj (94-element type_map, e+f+v loss, LMDB data)
└── scripts/
    ├── eval.py          # minimal eval skeleton (load model, compute E/F errors)
    ├── aselmdb_to_deepmd_lmdb.py  # MPtrj .aselmdb (ASE schema) -> deepmd LMDB converter
    ├── wandb_lcurve.py  # sidecar: tail lcurve.out -> live wandb (project dpa4_mptrj)
    └── train_mptrj.sh   # editable-install + torchrun launch + wandb sidecar for 8x A100
```

## MPtrj (DPA4-Air) training

`configs/dpa4_mptrj.json` is the DPA4-Air parameter set (same architecture as
`dpa4.json`, which `doc/model/dpa4.md` calls "close to the pretrained DPA4-Air
model") retargeted to MPtrj. Two MPtrj-specific changes: a 94-element Z-ordered
`type_map` (H..Pu) and `descriptor.sel: 200` (denser neighbor shells than water).

> **Data format gotcha.** deepmd-kit **cannot read MPtrj's `.aselmdb` files.**
> Those are the ASE/fairchem LMDB schema (zlib-compressed ASE-db JSON rows);
> deepmd's `is_lmdb()` only accepts `*.lmdb` / `data.mdb` and its own msgpack
> frame layout. The raw `MPtrj_2022.9_full.json` (pymatgen dicts) is not directly
> usable either. **Convert first** with `scripts/aselmdb_to_deepmd_lmdb.py`
> (uses only `lmdb+zlib+json+numpy+msgpack`; verified to round-trip through
> deepmd's `LmdbDataReader`). Virial uses the dpdata convention
> `virial = -stress * volume`; pass `--no-virial` (and zero the `v` loss prefs)
> to skip it.

```bash
# one-time conversion (writes deepmd LMDB next to the source aselmdb)
python scripts/aselmdb_to_deepmd_lmdb.py \
  --src /mnt/afs/share/dataset/periodicSystem/MPtrj/aselmdb/train \
  --dst /mnt/afs/share/dataset/periodicSystem/MPtrj/deepmd_lmdb/train.lmdb
python scripts/aselmdb_to_deepmd_lmdb.py \
  --src /mnt/afs/share/dataset/periodicSystem/MPtrj/aselmdb/val \
  --dst /mnt/afs/share/dataset/periodicSystem/MPtrj/deepmd_lmdb/val.lmdb

# launch on 8x A100 (deepmd uses torchrun + `--no-python dp --pt train`)
bash scripts/train_mptrj.sh
```

### Schedule & batch size (20 epochs on 8x A100)

`train.lmdb` has **1,420,395** frames. At `batch_size: "auto:256"` (each batch
holds >=256 atoms; ~256 atoms/GPU, ~2048 atoms/optimizer-step across 8 ranks) the
dataset is **157,809** same-nloc batches per pass, i.e. **19,727 optimizer
steps/epoch** over 8 ranks. The config therefore pins:

- `numb_steps: 394540` = 20 x 19,727. **Pinned, not `num_epoch`**: 157,809 isn't
  divisible by 8, so under `num_epoch` each rank would compute `num_steps` from
  its *own* dataloader length and disagree by 1 -> DDP desync. A fixed `numb_steps`
  keeps all 8 ranks (and the wsd LR decay horizon) in lockstep.
- `save_freq: 19727` ~= one checkpoint per epoch (`max_ckpt_keep: 5`).

### Run dir & checkpoint naming (mirrors equiformer's `--run-dir`/`--identifier`)

deepmd writes **every** output (checkpoints, `lcurve.out`, stat hdf5) to the CWD
and has no `--run-dir`/`--identifier` flag, so `train_mptrj.sh` `cd`s into a
per-run dir before launching — that `cd` *is* the run-dir mechanism. The config
is passed by **absolute** path (we've left the repo dir). Outputs land at:

```
/mnt/afs/share/checkpoint/dpa4/yaolekai/dpa4-air_mptrj_20ep_<YYYYmmdd-HHMMSS>/
    dpa4-air.ckpt-<step>.pt      # save_ckpt: "dpa4-air.ckpt" -> ckpts self-describe
    lcurve.out                   # learning curve (also the wandb source, below)
    dpa4_mptrj.hdf5              # neighbor-stat cache (stat_file)
```

The run dir keeps a timestamp so reruns never clobber each other (override the
prefix with `IDENT=... bash scripts/train_mptrj.sh`).

### Live wandb logging

deepmd-kit has **no native wandb support**, so `scripts/wandb_lcurve.py` runs as a
sidecar: it tails `lcurve.out` (deepmd `flush()`es one row per `disp_freq=1000`
steps — genuinely live, unlike the TensorBoard writer's ~120 s buffer) and
streams each row to wandb via `wandb.log(..., step=step)`. `train_mptrj.sh`
launches it on **node 0 only** (deepmd writes `lcurve.out` on global rank 0),
runs training, then drops a `.train_done` sentinel so the sidecar drains the final
rows and calls `wandb.finish()`. A training crash still ships partial curves, and
the script exits with training's real return code.

| | value |
|---|---|
| wandb project | `dpa4_mptrj` |
| wandb **display name** | `dpa4-air_mptrj_20ep` (no timestamp) |
| wandb run **id** (internal) | the timestamped `IDENT` — unique per job, so reruns never collide on step ordering and a restarted sidecar `resume`s the same run |
| logged scalars | `rmse_{,e_,f_,v_}{val,trn}`, `lr` (per 1000 steps, ~395 points); `nan` columns and out-of-order steps are dropped |

> wandb's API key / timeout are configured **outside** this script (env); the
> script only `uv pip install -q wandb` and sets the project/name/id. To watch
> progress without wandb: `lcurve.out`, stdout, or `numpy.loadtxt("lcurve.out")`.

## Environment facts (verified on the dev box)

This recipe was built and **smoke-tested** on a node with **8× A100-80GB**, CUDA
**12.4** toolkit (`/usr/local/cuda-12.4`, `nvcc` 12.4), driver 550.90.07.

- **PyTorch**: `torch==2.6.0` **cu124** build — chosen to match the local nvcc
  12.4 / driver 12.4. deepmd-kit's core requirement is only `torch>=2.1.0`; its
  `torch==2.11` pin (cu13 wheels) is *not* required and would mismatch a CUDA 12.4
  toolkit. PyPI's `torch==2.6.0` Linux wheel already pins `nvidia-*-cu12==12.4.*`,
  so plain `requirements.txt` installs the GPU build from the Tsinghua mirror.
- **`e3nn`**: the DPA4/SeZM descriptor imports `e3nn`, but deepmd-kit does **not**
  declare it. It is pinned in `requirements.txt` here; without it the model build
  fails with `ModuleNotFoundError: No module named 'e3nn'`.
- **Training vs inference**:
  - **Training** (`dp --pt train`) runs in pure Python — no compiled ops needed.
  - **Inference** (`dp --pt test`, `DeepEval`, `scripts/eval.py`) requires
    deepmd-kit's compiled custom ops (e.g. `deepmd::tabulate_fusion_*`), so the
    editable install must be built with `DP_ENABLE_PYTORCH=1 DP_VARIANT=cuda`
    (needs `nvcc`). The command below does this, enabling both.

Verified end-to-end: a 3-step `dp --pt train` on the example water data (2.754 M
params, loss decreasing) **and** a single-point `dp --pt test` with the bundled
DPA4 checkpoint (Energy MAE/atom ≈ 1.5e-3 eV, Force MAE ≈ 0.58 eV/Å).

## Local environment (uv + venv + Tsinghua mirror)

The environment lives at the **`DPA4/` level** (one directory up), so it serves
both `deepmd-kit/` and `dpa4/`, and `DPA4/` is not a git repo so nothing is
polluted.

```bash
cd ..   # into DPA4/
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 1. Create the virtual environment
uv venv .venv
source .venv/bin/activate

# 2. Install the runtime deps (torch cu124, e3nn, ...) — the source of truth
uv pip install -r dpa4/requirements.txt

# 3. Editable-install deepmd-kit (code only; deps already installed above).
#    DP_ENABLE_PYTORCH=1 + DP_VARIANT=cuda compiles the PyTorch CUDA ops so that
#    inference works too; build isolation pulls the build toolchain + a matching
#    torch (pinned to 2.6.0 so it matches nvcc 12.4) into a throwaway build env.
export PATH=/usr/local/cuda-12.4/bin:$PATH
export CUDACXX=/usr/local/cuda-12.4/bin/nvcc CUDAToolkit_ROOT=/usr/local/cuda-12.4
export DP_ENABLE_PYTORCH=1 DP_VARIANT=cuda PYTORCH_VERSION=2.6.0
uv pip install --no-deps -e ./deepmd-kit   # ~3 min (compiles ops)
```

> **Training only?** If you do not need inference, you can skip the env vars in
> step 3 and just run `uv pip install --no-deps -e ./deepmd-kit` (pure-Python,
> build isolation transiently pulls TensorFlow to satisfy the C++ `find_package`).
> `dp --pt train` works either way; `dp --pt test` / `eval.py` need the compiled ops.

### Regenerating `requirements.txt` (maintainers)

```bash
# After changing deps; --exclude-editable drops the deepmd-kit editable line
uv pip freeze --exclude-editable > dpa4/requirements.txt
```

> The local `.venv` cannot be baked into the image (a venv is bound to its host
> path). The image rebuilds the **same** dependency set from `requirements.txt`,
> so local and image install from one source of truth.

## Docker (image supplies env, mounted disk supplies code)

Two images are provided; both contain **deps only** (no source — code is mounted
at runtime):

| File | Base | Scope | nvcc | Size |
|------|------|-------|------|------|
| `Dockerfile.train` | `cuda:12.4.1-cudnn-runtime` | training only | no | ~15 GB |
| `Dockerfile`       | `cuda:12.4.1-cudnn-devel`   | training + inference | yes | larger |

### Build (training-only image — verified)

```bash
cd dpa4
docker build -f Dockerfile.train -t dpa4-train:latest .
# docker tag dpa4-train:latest <registry>/dpa4-train:latest && docker push ...
```

> **If the base image pull fails** (`not found` / DNS error): this host's docker
> daemon has an unreachable registry mirror configured in `/etc/docker/daemon.json`
> (`mirror.ccs.tencentyun.com`). Without root to fix it, pull the base from a
> reachable mirror and retag so `FROM` uses the local cache:
> ```bash
> B=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
> docker pull docker.m.daocloud.io/$B && docker tag docker.m.daocloud.io/$B $B
> ```

### Run training (pin to a free GPU — do not grab all of them)

The box is shared. Pick an idle GPU first, then expose only that one:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader  # find idle
docker run --rm --gpus '"device=4"' \
  -v /mnt/afs/home/yaolekai/MLIP/DPA4:/workspace/DPA4 \
  dpa4-train:latest \
  bash -c '
    uv pip install --no-deps -e /workspace/DPA4/deepmd-kit && \
    cd /workspace/DPA4/dpa4 && \
    dp --pt train configs/dpa4.json'
```

Verified 2026-06-08 on GPU 4: a 3-step `dp --pt train` ran inside the container
(2.754 M params, loss decreasing, checkpoint saved). The container saw only the
one pinned GPU. The runner may be docker / enroot / singularity — same idea:
image = deps, code = mounted disk; editing code needs no image rebuild.

> **Per-run cost (training image, ~13 min).** The runtime editable install uses
> build isolation, which each time pulls the build toolchain **and TensorFlow**
> into a throwaway build env and compiles deepmd-kit's TF Python ops — wasted work
> for a PyTorch-only run (deepmd-kit's C++ `find_package(tensorflow)` is
> unconditional and `DP_ENABLE_TENSORFLOW` defaults to 1). It still works; to
> avoid it, persist `/opt/venv` across runs (install once) or, for a hermetic
> image, bake the build toolchain + `tensorflow` in and use `--no-build-isolation`.

### Run training + inference (full image)

Same as above but with the `Dockerfile` image and the ops-compile env vars, so
`dp --pt test` / `eval.py` also work (needs the `-devel` base's `nvcc`):

```bash
docker run --rm --gpus '"device=4"' \
  -v /mnt/afs/home/yaolekai/MLIP/DPA4:/workspace/DPA4 \
  dpa4-full:latest \
  bash -c '
    export DP_ENABLE_PYTORCH=1 DP_VARIANT=cuda PYTORCH_VERSION=2.6.0
    uv pip install --no-deps -e /workspace/DPA4/deepmd-kit && \
    cd /workspace/DPA4/dpa4 && dp --pt train configs/dpa4.json'
```

### Evaluation

```bash
# Built-in tester (energy/force MAE/RMSE) on a PyTorch checkpoint:
dp --pt test -m model.ckpt.pt -s /path/to/test/system -n 100

# Or the minimal script in this repo:
python scripts/eval.py --model frozen_model.pth --system /path/to/test/system
```

## Constraints

- Never commit into `deepmd-kit/`; never touch its git state. No `data/` dir. No conda.
- The venv lives at `DPA4/.venv` (parent level), not inside `dpa4/`.
- `requirements.txt` must be generated with `--exclude-editable` (no editable
  deepmd-kit line), otherwise the image build fails.
- The Dockerfile installs deps only — no source copied; code comes from the mount.
- Weights, logs, `*.egg-info`, and other artifacts never enter git.
