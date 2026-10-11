# S04_swiglu — reproducible container (GB200 / sm_100 profile)
# Run/scoring target: single NVIDIA GB200-class Blackwell GPU (compute capability 10.0 / sm_100).
# Base image is pinned by digest; the task declares no vendor-library anchor.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev build-essential git tmux ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# Match the GB200 calibration profile: torch 2.9.0 + CUDA 12.8 wheels.
RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple

WORKDIR /app
COPY reference_model.py solve.py run_local.py run_task.py _score_core.py kernel_static_checker.py /app/

# Freeze the naive oracle + local harness + scorer adapter + shared scoring core + anti-hack
# checker; agent may only edit solve.py.
RUN mkdir -p /orig /logs/verifier && \
    sha256sum /app/reference_model.py /app/run_local.py /app/run_task.py /app/_score_core.py /app/kernel_static_checker.py > /orig/protected.sha256

RUN python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)"
CMD ["bash"]
