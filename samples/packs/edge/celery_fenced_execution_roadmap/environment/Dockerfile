FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY repo/ /app/

RUN python -m pip install --no-cache-dir \
      -e /app/components/vine \
      -e /app/components/kombu \
      -e /app/components/billiard \
      -e /app \
      pytest==8.4.1 \
    && python -m compileall -q \
      /app/celery \
      /app/components/kombu/kombu \
      /app/components/billiard/billiard \
      /app/components/vine/vine

CMD ["bash"]
