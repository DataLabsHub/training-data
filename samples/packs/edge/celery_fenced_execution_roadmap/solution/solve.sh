#!/bin/bash
set -euo pipefail

cd /app
git apply --check /solution/changes.patch
git apply /solution/changes.patch
python -m compileall -q celery components/kombu/kombu components/billiard/billiard components/vine/vine
