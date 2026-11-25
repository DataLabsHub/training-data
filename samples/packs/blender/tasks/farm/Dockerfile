# Immutable Python base + frozen Debian snapshot + hash-locked official bpy runtime.
FROM python:3.13.7-slim-bookworm@sha256:781449467ffb6f04218f09b1ecdcdc7d22b289ee5da9ec498b024e24ad7a6db7
ENV DEBIAN_FRONTEND=noninteractive PYTHONUTF8=1 LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    printf '%s\n' 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20260901T000000Z bookworm main' > /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libegl1 libsm6 libice6 libxkbcommon0 libgomp1 libfontconfig1 libxext6 && \
    rm -rf /var/lib/apt/lists/*
COPY runner/requirements.lock /opt/runtime/requirements.lock
RUN python3 -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /opt/runtime/requirements.lock
COPY runner/blender_cli.py /usr/local/bin/blender
RUN chmod 755 /usr/local/bin/blender && blender --version
COPY task.md /task/task.md
COPY assets/ /task/assets/
COPY references/ /task/references/
COPY starter/ /task/starter/
COPY runner/ /task/runner/
RUN mkdir -p /task/output/renders /workspace /logs/verifier /opt/public
RUN cp /task/references/reference.png /workspace/reference.png
WORKDIR /workspace
CMD ["sleep", "infinity"]
