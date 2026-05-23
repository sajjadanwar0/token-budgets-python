#!/usr/bin/env bash
set -u

cd "$(dirname "$0")"

PY=/home/neo/RustroverProjects/token-budgets-python/.venv/bin/python3

if [[ ! -x "$PY" ]]; then
    echo "ERROR: $PY not found or not executable."
    exit 1
fi

PASS=0
FAIL=0

check_fail() {
    local fixture=$1
    local pattern=$2
    local out
    out=$("$PY" -m mypy --config-file=mypy.ini --no-error-summary \
                 --hide-error-context "$fixture" 2>&1)
    if echo "$out" | grep -q "INTERNAL ERROR"; then
        echo "[FAIL] $fixture (mypy internal error)"
        echo "$out" | sed 's/^/         /'
        FAIL=$((FAIL + 1))
    elif echo "$out" | grep -q -- "$pattern"; then
        echo "[PASS] $fixture (correctly flagged: $pattern)"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $fixture (did not flag: $pattern)"
        echo "$out" | sed 's/^/         /'
        FAIL=$((FAIL + 1))
    fi
}

check_pass() {
    local fixture=$1
    local out
    out=$("$PY" -m mypy --config-file=mypy.ini --no-error-summary \
                 --hide-error-context "$fixture" 2>&1)
    if echo "$out" | grep -q "INTERNAL ERROR"; then
        echo "[FAIL] $fixture (mypy internal error)"
        echo "$out" | sed 's/^/         /'
        FAIL=$((FAIL + 1))
    elif echo "$out" | grep -q "tb-double-use"; then
        echo "[FAIL] $fixture (false positive: legitimate code flagged)"
        echo "$out" | sed 's/^/         /'
        FAIL=$((FAIL + 1))
    else
        echo "[PASS] $fixture (cleanly accepted)"
        PASS=$((PASS + 1))
    fi
}

echo "=== token-budgets-mypy plugin: test suite (v0.2 final) ==="
echo "    using: $PY"
echo
check_fail  test_double_spend.py        "already consumed by spend"
check_fail  test_use_after_split.py     "already consumed by split"
check_fail  test_use_after_merge.py     "already consumed by merge_with"
check_fail  test_module_level.py        "GLOBAL_BUDGET.*already consumed"
check_pass  test_legitimate_single.py
check_pass  test_legitimate_iter.py

echo
echo "Summary: $PASS PASS, $FAIL FAIL"
exit $FAIL
