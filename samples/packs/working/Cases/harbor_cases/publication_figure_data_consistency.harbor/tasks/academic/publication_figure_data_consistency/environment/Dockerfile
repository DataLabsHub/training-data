FROM python:3.12.11-slim-bookworm

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        fonts-noto-cjk \
        libreoffice \
        poppler-utils \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir \
        openpyxl==3.1.5 \
        Pillow==11.3.0 \
        pypdf==6.0.0 \
        python-docx==1.2.0

WORKDIR /app
COPY input_files/ /app/
RUN mkdir -p /app/deliverables
