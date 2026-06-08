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
├── Dockerfile           # rebuilds the same deps in the image (env only, no source)
├── .dockerignore
├── .gitignore
├── configs/
│   └── dpa4.json        # DPA4 training config (provenance noted in its _source field)
└── scripts/
    └── eval.py          # minimal eval skeleton (load model, compute E/F errors)
```

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

## Cluster run (Docker: image supplies env, mounted disk supplies code)

Build and push the env image (deps only, no source):

```bash
cd dpa4
docker build -t <registry>/dpa4-env:latest .
docker push <registry>/dpa4-env:latest
```

On the cluster, mount the dev disk, editable-install the mounted deepmd-kit
(`--no-deps`, since deps are already in the image), then train/test. The
`-devel` base image provides the `nvcc` the ops compile needs:

```bash
docker run --gpus all \
  -v /mnt/devdisk/DPA4:/workspace/DPA4 \
  <registry>/dpa4-env:latest \
  bash -c '
    export DP_ENABLE_PYTORCH=1 DP_VARIANT=cuda PYTORCH_VERSION=2.6.0
    uv pip install --no-deps -e /workspace/DPA4/deepmd-kit && \
    cd /workspace/DPA4/dpa4 && \
    dp --pt train configs/dpa4.json'
```

The runner may be docker / enroot / singularity — the idea is the same: image =
deps, code = mounted disk. Editing code needs no image rebuild, just re-run.

> **Note on per-run cost**: the editable install recompiles the ops (~3 min) and
> build isolation pulls TensorFlow into a throwaway build env each time. If the
> venv is persisted (or the container is long-lived), do this install once. A
> fully hermetic alternative is to add the build toolchain + `tensorflow` to the
> image and use `--no-build-isolation`.

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
