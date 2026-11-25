#!/usr/bin/env bash
set -euo pipefail
blender --background --factory-startup --disable-autoexec --python-exit-code 3 \
  --python /solution/oracle.py -- /workspace/initial.blend /workspace/submission.blend
