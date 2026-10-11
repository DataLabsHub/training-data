FROM node:22.14.0-bookworm-slim AS codex-runtime
RUN npm install --global @openai/codex@0.147.0

FROM python:3.12.11-slim-bookworm
COPY --from=codex-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=codex-runtime /usr/local/bin/codex /usr/local/bin/codex
COPY --from=codex-runtime /usr/local/lib/node_modules /usr/local/lib/node_modules

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        fonts-noto-cjk \
        libreoffice \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir \
        openpyxl==3.1.5 \
        Pillow==11.3.0 \
        pypdf==6.0.0 \
        python-docx==1.2.0

RUN mkdir -p /app/deliverables /logs/verifier
COPY . /tests
RUN chmod 0755 /tests/test.sh /tests/grade.py
WORKDIR /app
