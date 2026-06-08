# DPA4 environment image — provides dependencies only, no source code.
# deepmd-kit and this repo's code are mounted from the cluster disk at runtime.
#
# Base = CUDA 12.4 *devel* image: it ships nvcc, which is required because
# deepmd-kit's PyTorch C++/CUDA custom ops are compiled at runtime when the
# mounted source is editable-installed with DP_ENABLE_PYTORCH=1 (see README).
# 12.4 matches the torch cu124 build pinned in requirements.txt and the dev box
# toolkit/driver.
# TODO(confirm): verify 12.4 matches the cluster's driver (`nvidia-smi`). If the
#   cluster CUDA differs, bump both this tag and the torch cu1XX build together.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    CUDACXX=/usr/local/cuda/bin/nvcc

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# Build the venv and install the runtime deps (incl. the cu124 torch and e3nn).
# deepmd-kit itself is NOT installed here; it is mounted and editable-installed at
# runtime. That runtime install uses build isolation, which transiently pulls the
# build toolchain (scikit-build-core/cmake/ninja) and TensorFlow into a throwaway
# build env — hence `git` + `build-essential` + nvcc above. See README.
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv && uv pip install -r /tmp/requirements.txt
