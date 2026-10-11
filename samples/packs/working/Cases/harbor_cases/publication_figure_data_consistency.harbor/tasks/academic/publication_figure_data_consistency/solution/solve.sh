#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_root=${SCF_REFERENCE_ROOT:-$script_dir/reference_outputs}
candidate_root=${SCF_CANDIDATE_ROOT:-/app}

mkdir -p "$candidate_root/deliverables"
cp -R "$source_root/deliverables/." "$candidate_root/deliverables/"
