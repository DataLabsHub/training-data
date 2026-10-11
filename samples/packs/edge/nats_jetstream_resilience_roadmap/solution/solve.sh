#!/bin/bash
set -euo pipefail

cd /app
git apply --check /solution/changes.patch
git apply /solution/changes.patch
go test ./... -run '^$'
