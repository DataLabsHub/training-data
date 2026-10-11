#!/usr/bin/env bash
set -euo pipefail
cd /testbed
git apply --whitespace=nowarn /solution/code.patch
