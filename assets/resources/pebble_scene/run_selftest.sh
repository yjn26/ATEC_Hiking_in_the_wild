#!/usr/bin/env bash
# run_selftest.sh — 一键自检（可换机器直接用）
#
# 用法:
#   PY=/path/to/python bash run_selftest.sh          # 单 env (静态 + 动态)
#   PY=/path/to/python bash run_selftest.sh 2        # 2 个 env
#
# 判据: 输出里出现 "SELFTEST OK"（Isaac 会把异常吞成 exit 0，不能只看返回码）
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

PY=${PY:-python}
BUNDLE=${BUNDLE:-bundle}
OUT=${OUT:-bench_out}
ENVS=${1:-1}
mkdir -p "$OUT"
LOG="$OUT/run_selftest.log"
: > "$LOG"
fail=0

one() {
    local mode="$1"
    local out="$OUT/selftest_${mode}_envs${ENVS}.out"
    echo "--- $mode (envs=$ENVS) start $(date +%H:%M:%S)" | tee -a "$LOG"
    PYTHONPATH=. timeout 900 "$PY" loader_selftest.py \
        --bundle "$BUNDLE" --mode "$mode" --envs "$ENVS" > "$out" 2>&1
    if grep -q "SELFTEST OK" "$out"; then
        echo "[$mode] OK   (log: $out)" | tee -a "$LOG"
    else
        echo "[$mode] FAIL (log: $out)" | tee -a "$LOG"
        grep -E "FAIL|Error|Traceback" "$out" | tail -5 | tee -a "$LOG"
        fail=1
    fi
}

one static
one dynamic

if [ $fail -eq 0 ]; then
    echo "SELFTEST OK (all)" | tee -a "$LOG"
    echo "SELFTEST_OK" > "$OUT/SELFTEST_DONE"
else
    echo "SELFTEST FAILED" | tee -a "$LOG"
    echo "SELFTEST_FAIL" > "$OUT/SELFTEST_DONE"
fi
exit $fail
