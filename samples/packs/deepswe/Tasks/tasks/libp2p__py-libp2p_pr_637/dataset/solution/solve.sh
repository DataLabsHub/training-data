#!/usr/bin/env bash
set -euo pipefail

cd /testbed
SOLUTION_DIR="${1:-/solution}"

if [ -d "$SOLUTION_DIR/aux" ]; then
  tar -cf - -C "$SOLUTION_DIR/aux" . | tar -xf - -C /testbed
fi

if [ -s "$SOLUTION_DIR/code.patch" ]; then
  git -C /testbed -c user.email=harbor@bot -c user.name=harbor \
    apply --whitespace=nowarn "$SOLUTION_DIR/code.patch"
fi
