#!/usr/bin/env bash
set -u

export PATH="/usr/local/go/bin:$PATH"
export GO111MODULE=off
export GOPATH="${GOPATH:-/root/go}"
export GOCACHE="${GOCACHE:-/testbed/.cache/go-build}"
mkdir -p "$GOCACHE" "$GOPATH/src/github.com/wmnsk" /tmp/run

if [ ! -e "$GOPATH/src/github.com/wmnsk/gopcua" ]; then
  ln -s /testbed "$GOPATH/src/github.com/wmnsk/gopcua"
fi

cd "$GOPATH/src/github.com/wmnsk/gopcua" || exit 2

if [ "$#" -eq 0 ]; then
  echo "OMNIGRIL_EXIT_CODE=2"
  exit 2
fi

tmp_json=$(mktemp /tmp/run/omnigril_go_test_output.XXXXXX.jsonl)
tmp_results=$(mktemp /tmp/run/omnigril_go_test_results.XXXXXX.txt)
tmp_filter=$(mktemp /tmp/run/omnigril_go_test_filter.XXXXXX.txt)
trap 'rm -f "$tmp_json" "$tmp_results" "$tmp_filter"' EXIT

declare -A package_to_tests=()
declare -A package_unfiltered=()
declare -a packages=()

add_package_once() {
  local pkg="$1"
  local seen
  for seen in "${packages[@]}"; do
    if [ "$seen" = "$pkg" ]; then
      return
    fi
  done
  packages+=("$pkg")
}

normalize_package() {
  local target="$1"
  if [[ "$target" == *_test.go ]]; then
    local dir
    dir=$(dirname "$target")
    if [ "$dir" = "." ]; then
      printf 'github.com/wmnsk/gopcua'
    elif [[ "$dir" == ./* ]]; then
      printf 'github.com/wmnsk/gopcua/%s' "${dir#./}"
    else
      printf 'github.com/wmnsk/gopcua/%s' "$dir"
    fi
    return
  fi
  if [[ "$target" == ./* ]]; then
    if [ "$target" = "./..." ]; then
      printf './...'
    elif [ "$target" = "./." ]; then
      printf 'github.com/wmnsk/gopcua'
    else
      printf 'github.com/wmnsk/gopcua/%s' "${target#./}"
    fi
    return
  fi
  local first="${target%%/*}"
  if [[ "$first" == *.* && "$first" != "$target" ]]; then
    printf '%s' "$target"
    return
  fi
  printf 'github.com/wmnsk/gopcua/%s' "$target"
}

escape_go_run_regex() {
  printf '%s' "$1" | sed -e 's/[][(){}.^$*+?|\\]/\\&/g'
}

for target in "$@"; do
  test_name=""
  base="$target"
  if [[ "$target" == *"::"* ]]; then
    base="${target%%::*}"
    test_name="${target#*::}"
  fi
  pkg=$(normalize_package "$base")
  add_package_once "$pkg"
  if [ -z "$test_name" ] && [[ "$base" == *_test.go ]] && [ -f "$base" ]; then
    package_unfiltered[$pkg]=1
    file_tests=$(sed -n 's/^func \(Test[A-Za-z0-9_]*\)(.*$/\1/p' "$base" | paste -sd '|' -)
    if [ -n "$file_tests" ]; then
      if [ -n "${package_to_tests[$pkg]:-}" ]; then
        package_to_tests[$pkg]="${package_to_tests[$pkg]}|${file_tests}"
      else
        package_to_tests[$pkg]="$file_tests"
      fi
    fi
  elif [ -n "$test_name" ]; then
    printf '%s::%s\n' "$pkg" "$test_name" >> "$tmp_filter"
    top_name="${test_name%%/*}"
    escaped=$(escape_go_run_regex "$top_name")
    if [ -n "${package_to_tests[$pkg]:-}" ]; then
      package_to_tests[$pkg]="${package_to_tests[$pkg]}|${escaped}"
    else
      package_to_tests[$pkg]="$escaped"
    fi
  else
    package_unfiltered[$pkg]=1
  fi
done

for pkg in "${!package_unfiltered[@]}"; do
  sed -i "\#^${pkg}::#d" "$tmp_filter"
done

overall_rc=0
for pkg in "${packages[@]}"; do
  run_regex="${package_to_tests[$pkg]:-}"
  cmd=(go test -vet=off -count=1 -json "$pkg")
  if [ -n "$run_regex" ]; then
    cmd+=(-run "^(${run_regex})(/.*)?$")
  fi
  set +e
  "${cmd[@]}" >> "$tmp_json" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    overall_rc="$rc"
  fi
done

python3 - "$tmp_json" "$tmp_filter" > "$tmp_results" <<'PY'
import json
import sys
from collections import OrderedDict

json_path, filter_path = sys.argv[1:3]
results = OrderedDict()
priority = {"PASSED": 1, "SKIPPED": 2, "FAILED": 3, "ERROR": 4}
status_map = {"pass": "PASSED", "fail": "FAILED", "skip": "SKIPPED"}
filters = {}

with open(filter_path, encoding="utf-8", errors="replace") as handle:
    for raw in handle:
        item = raw.rstrip("\n")
        if "::" not in item:
            continue
        package, test_name = item.split("::", 1)
        filters.setdefault(package, set()).add(test_name)

def record(test_id, status):
    previous = results.get(test_id)
    if previous is None or priority[status] >= priority[previous]:
        results[test_id] = status

with open(json_path, encoding="utf-8", errors="replace") as handle:
    for line in handle:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        test_name = event.get("Test")
        if not test_name:
            continue
        status = status_map.get(event.get("Action"))
        if not status:
            continue
        package = event.get("Package", "")
        if package in filters and test_name not in filters[package]:
            continue
        record(f"{package}::{test_name}" if package else test_name, status)

for test_id, status in results.items():
    print(f"RESULT::{test_id}::{status}")
PY

cat "$tmp_results"
echo "OMNIGRIL_EXIT_CODE=$overall_rc"
exit "$overall_rc"
