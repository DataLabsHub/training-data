# h200_fp8_variable_m_gemm — reproducible container (h200_nv2505 profile)
# H200 Hopper sm_90 target: driver 590.48.01, Python 3.12.3, CUDA 12.9, torch 2.8.0a0+5228986c39.nv25.05, Triton 3.1.0, and liger-kernel 0.8.1.
FROM nvcr.io/nvidia/pytorch:25.05-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git tmux ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- strong-baseline library stack (pinned to the calibration host) ---
RUN printf '%s\n' 'liger-kernel==0.8.1 --hash=sha256:90836c1e9de22a1c57a640775fb79151ea704748cf9c3f6658dddaa06fff1989' > /tmp/liger-requirements.txt && python -m pip install --no-cache-dir --require-hashes --no-deps -r /tmp/liger-requirements.txt && rm -f /tmp/liger-requirements.txt

WORKDIR /app
COPY reference_model.py solve.py run_local.py run_task.py _score_core.py kernel_static_checker.py /app/

# freeze the naive oracle + local harness + scorer adapter + shared scoring core + anti-hack
# checker; agent may only edit solve.py
RUN mkdir -p /orig /logs/verifier && \
    sha256sum /app/reference_model.py /app/run_local.py /app/run_task.py /app/_score_core.py /app/kernel_static_checker.py > /orig/protected.sha256

RUN python -c "import importlib.metadata as md, platform, torch, triton, liger_kernel; assert platform.python_version() == '3.12.3'; assert torch.__version__ == '2.8.0a0+5228986c39.nv25.05'; assert torch.version.cuda == '12.9'; assert triton.__version__ == '3.1.0'; assert md.version('liger-kernel') == '0.8.1'; print('H200 NGC 25.05 profile OK')"
CMD ["bash"]
