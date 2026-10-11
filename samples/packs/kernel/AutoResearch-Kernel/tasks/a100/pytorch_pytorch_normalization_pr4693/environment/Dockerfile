# pytorch_pytorch_normalization_pr4693 — reproducible container (a100_pcie_80gb profile)
# A100 80GB PCIe target: driver 580.142, Python 3.12.13, CUDA 12.4, torch 2.6.0+cu124, Triton 3.2.0; calibration is Ampere sm_80-specific.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git tmux ca-certificates curl xz-utils libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev \
    libgdbm-dev libncursesw5-dev libnss3-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL 'https://www.python.org/ftp/python/3.12.13/Python-3.12.13.tar.xz' -o /tmp/Python-3.12.13.tar.xz && \
    echo 'c08bc65a81971c1dd5783182826503369466c7e67374d1646519adf05207b684  /tmp/Python-3.12.13.tar.xz' | sha256sum -c - && \
    tar -xJf /tmp/Python-3.12.13.tar.xz -C /tmp && \
    cd /tmp/Python-3.12.13 && \
    ./configure --prefix=/usr/local --with-ensurepip=install && \
    make -j2 && make install && \
    test "$(/usr/local/bin/python3.12 -c 'import platform; print(platform.python_version())')" = '3.12.13' && \
    ln -sf /usr/local/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.12 /usr/local/bin/python && \
    rm -rf /tmp/Python-3.12.13.tar.xz /tmp/Python-3.12.13

# --- strong-baseline library stack (pinned to the calibration host) ---
RUN /usr/local/bin/python3.12 -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple

WORKDIR /app
COPY reference_model.py solve.py run_local.py run_task.py _score_core.py kernel_static_checker.py /app/

# freeze the naive oracle + local harness + scorer adapter + shared scoring core + anti-hack
# checker; agent may only edit solve.py
RUN mkdir -p /orig /logs/verifier && \
    sha256sum /app/reference_model.py /app/run_local.py /app/run_task.py /app/_score_core.py /app/kernel_static_checker.py > /orig/protected.sha256

RUN python -c "import os, platform, sys, torch, triton; assert platform.python_version() == '3.12.13'; assert os.path.realpath(sys.executable) == '/usr/local/bin/python3.12'; assert torch.__version__ == '2.6.0+cu124'; assert torch.version.cuda == '12.4'; assert triton.__version__ == '3.2.0'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)"
CMD ["bash"]
