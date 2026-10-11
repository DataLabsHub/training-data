#!/usr/bin/env bash
# tests/test.sh - AutoResearch/Harbor verifier for a autoresearch-kernel task.
# DO NOT MODIFY. Mounted by the harness AFTER the agent's /app is frozen.
#
# Placeholders substituted at generation time:
#   S02_layernorm  1.1874  python
set -euo pipefail

REWARD_FILE="/logs/verifier/reward.json"
REWARD_TXT="/logs/verifier/reward.txt"
mkdir -p /logs/verifier
PY="python"          # python interpreter that holds the strong baseline library

write_zero() {
    # Harbor parses reward.json as a flat dict of numbers only; keep it numeric.
    printf '{"reward": 0.0, "correctness": 0}\n' > "$REWARD_FILE"
    echo "0.0" > "$REWARD_TXT"
    printf '{"reward": 0.0, "correctness": false, "reason": "%s"}\n' "$1" > /logs/verifier/reward_detail.json
    echo "$1 -- scoring 0" >&2
    exit 0
}

# -- Step 0: protected-file integrity (naive oracle + harness must be untouched) --
echo "=== Integrity ===" >&2
if ! sha256sum -c /orig/protected.sha256 --quiet 2>/dev/null; then
    sha256sum -c /orig/protected.sha256 >&2 || true
    write_zero "protected_file_modified"
fi

# -- Step 0a: scorer/provenance digest closure --
echo "=== Scorer digest closure ===" >&2
EXPECT_SCORE_CORE_DIGEST="c45328f570e3eaa34741c6582f904715647c20ece0292f3292c0856fc311833b"
EXPECT_RUN_TASK_DIGEST="d3652f5016b9b80294d37e1fa0cc83a49b0f824311e966b51be5514b197c9c88"
EXPECT_STATIC_CHECKER_DIGEST="bbce361d706a9984b241f590d7be06384f013ab49e6a6c018fa992b65f5045ec"
check_digest() {
    local path="$1"
    local expected="$2"
    local label="$3"
    if [ -f "$path" ] && [ -n "$expected" ]; then
        local got
        got=$(sha256sum "$path" 2>/dev/null | cut -d' ' -f1)
        if [ "$got" != "$expected" ]; then
            echo "$label digest drift: got $got expected $expected ($path)" >&2
            write_zero "${label}_digest_drift"
        fi
    fi
}
check_digest /app/_score_core.py "$EXPECT_SCORE_CORE_DIGEST" score_core
check_digest /tests/_score_core.py "$EXPECT_SCORE_CORE_DIGEST" score_core_tests
check_digest /app/run_task.py "$EXPECT_RUN_TASK_DIGEST" run_task
check_digest /tests/run_task.py "$EXPECT_RUN_TASK_DIGEST" run_task_tests
check_digest /app/kernel_static_checker.py "$EXPECT_STATIC_CHECKER_DIGEST" static_checker
check_digest /tests/kernel_static_checker.py "$EXPECT_STATIC_CHECKER_DIGEST" static_checker_tests

# -- Step 0b: strict /app whitelist --
echo "=== /app whitelist ===" >&2
EXPECTED=$(cat <<'EOF'
/app
/app/reference_model.py
/app/solve.py
/app/run_local.py
/app/run_task.py
/app/_score_core.py
/app/kernel_static_checker.py
EOF
)
ACTUAL=$(find /app -mindepth 0 \( -type f -o -type d -o -type l \) ! -name '__pycache__' ! -path '*/__pycache__/*' ! -name '*.pyc' 2>/dev/null | sort)
UNEXPECTED=$(comm -23 <(echo "$ACTUAL") <(echo "$EXPECTED" | sort) || true)
if [ -n "$UNEXPECTED" ]; then
    echo "Unexpected files under /app:" >&2; echo "$UNEXPECTED" >&2
    write_zero "app_tree_modified"
fi

# -- Step 1: constraint checks on solve.py --
echo "=== Constraints ===" >&2
SOLVE=/app/solve.py
# The agent must not import the strong baseline library directly (that trivializes
# the task) nor read the private reference / hidden anchor.
if grep -qiE '^\s*(import|from)\s+(flashinfer|gsplat|cuvs|fbgemm_gpu|liger_kernel)' "$SOLVE" 2>/dev/null; then
    write_zero "banned_baseline_library_import"
fi
if grep -qE '(private_baseline|solution/|/orig/|reward\.json|HEADROOM|os\.environ|subprocess|socket|/proc/|/tests|sys\.argv)' "$SOLVE" 2>/dev/null; then
    write_zero "banned_io_or_probe"
fi
# No network egress: the scorer needs none, and a candidate that phones home could
# fetch a precomputed answer or leak the hidden inputs. Ban the common egress modules.
if grep -qiE '^\s*(import|from)\s+(urllib|requests|http|httpx|aiohttp|ftplib|telnetlib|smtplib)\b|torch\.hub|urllib\.request|http\.client|xmlrpc' "$SOLVE" 2>/dev/null; then
    write_zero "banned_network_egress"
fi
if grep -qE 'torch\.nn\.attention\.flex_attention' "$SOLVE" 2>/dev/null; then
    # FlexAttention IS the strong baseline for A4 tasks -> disallow calling it directly.
    if [ "0" != "1" ]; then write_zero "banned_flex_attention_import"; fi
fi
if grep -qE '_grouped_mm' "$SOLVE" 2>/dev/null; then
    # torch._grouped_mm IS the strong baseline for the MoE task -> disallow calling it
    # directly (the agent must implement the grouped GEMM itself). No other task needs it.
    write_zero "banned_grouped_mm_call"
fi

# -- Step 2: syntax --
echo "=== Syntax ===" >&2
if ! "$PY" -m py_compile "$SOLVE" 2>&1; then
    write_zero "syntax_failed"
fi

# -- Step 2b: static anti-hack gate (frozen-time defense) --
# The scorer (run_task.py) also runs this gate, but we re-run it here at verify
# time so a tampered solve.py cannot slip a fused-op/no-kernel solution past a
# stale scorer. Rejects: no real Triton/CUDA kernel, banned torch computation ops
# (F.rms_norm, torch.softmax, torch.matmul, ...), timing/precision manipulation.
echo "=== Static anti-hack ===" >&2
if ! "$PY" - "$SOLVE" <<'PYEOF' >&2
import sys, ast
sys.path.insert(0, "/app")
from kernel_static_checker import validate_kernel_static
code = open(sys.argv[1]).read()
if "@triton.jit" in code or "@triton.autotune" in code or "import triton" in code:
    backend = "triton"
elif "__global__" in code or "load_inline" in code or "cpp_extension" in code:
    backend = "cuda"
else:
    backend = "triton"
# read the task's ALLOW_TORCH_OPS glue-list from the public reference_model without
# importing it (cheap AST scan; avoids importing torch just for a constant).
allow_ops = []
try:
    tree = ast.parse(open("/app/reference_model.py").read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "ALLOW_TORCH_OPS":
                    allow_ops = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
except Exception:
    pass
forbidden = ["code_bypass","timing_event_patch","thread_injection","lazy_eval","torch_computation_ops","stream_injection"]
valid, errors, _warns = validate_kernel_static(code, backend=backend, precision="bf16", forbidden=forbidden, allow_ops=allow_ops)
if not valid:
    print("static_check_failed: " + "; ".join(errors[:4]))
    sys.exit(1)
sys.exit(0)
PYEOF
then
    write_zero "static_check_failed"
fi

# -- Step 3: public correctness smoke (small shape, no scoring) --
echo "=== Public smoke ===" >&2
if ! "$PY" /app/run_local.py --smoke 2>&1; then
    write_zero "public_smoke_failed"
fi

# -- Step 4: scored benchmark (correctness gate + geomean speedup over naive) --
# reward == geomean(t_naive/t_cand ... composite) over correct shapes (KernelBench gmsr).
# --headroom is the measured GB200 frontier from calibration, passed for diagnostics only.
# Hidden seeds are generated HERE, at verify time, so the candidate (frozen in /app,
# forbidden from reading env/argv) cannot have cached or hard-coded these inputs.
echo "=== Scored benchmark ===" >&2
HSEEDS="$(( (RANDOM % 900) + 100 )),$(( (RANDOM % 900) + 1000 )),$(( (RANDOM % 900) + 2000 ))"
HTIME="$(( (RANDOM % 900) + 5000 ))"
# Wall-clock watchdog: a candidate with an infinite loop / runaway kernel must not
# stall the verifier forever. `timeout` sends SIGTERM then SIGKILL; a killed scorer
# yields exit!=0 -> scored 0. 1200s covers JIT warmup + all workload shapes.
SCORE_TIMEOUT="${KERNEL_SCORE_TIMEOUT:-1200}"
# Separate frontier components (P0 §3.1); this route has no calibrated library
# anchor, so only pass the compile frontier. Never pass the string "None".
EXTRA_HR=()
[ -n "1.1874" ] && EXTRA_HR+=(--compile-headroom "1.1874")
if ! timeout --signal=TERM --kill-after=30 "$SCORE_TIMEOUT" "$PY" /tests/run_task.py \
        --task-dir /app \
        --candidate /app/solve.py \
        --headroom 1.1874 \
        "${EXTRA_HR[@]}" \
        --reward-anchor naive \
        --seeds "$HSEEDS" \
        --time-seed "$HTIME" \
        --reward-file "$REWARD_FILE" \
        --reward-txt "$REWARD_TXT" 2>&1; then
    write_zero "scorer_crashed_or_timeout"
fi

echo "Results written to $REWARD_FILE" >&2
cat "$REWARD_FILE" >&2 || true
