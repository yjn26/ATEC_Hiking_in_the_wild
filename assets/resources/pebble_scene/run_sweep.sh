#!/usr/bin/env bash
# run_sweep.sh — 刚体数量扫描（开发者用，不是交接必需）
# 产出 "ms/步 ~ 刚体数" 曲线 + nvidia-smi 显存峰值
#
# 用法: PY=/path/to/python bash run_sweep.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PY=${PY:-python}
BUNDLE=${BUNDLE:-bundle}
OUT=${OUT:-bench_out}
mkdir -p "$OUT"
LOG="$OUT/run_sweep.log"
: > "$LOG"
echo "=== sweep start $(date)" | tee -a "$LOG"

case_run() {
    local name="$1"; shift
    local json="$OUT/${name}.json"
    echo "--- $name start $(date +%H:%M:%S)" | tee -a "$LOG"
    timeout 900 "$PY" isaac_bench.py --bundle "$BUNDLE" --out "$json" --vram-sample "$@" \
        > "$OUT/${name}.out" 2>&1
    echo "[$name] rc=$? json=$( [ -f "$json" ] && echo yes || echo no )" | tee -a "$LOG"
}

case_run sweep_s_4467 --mode static  --no-drop --warmup 200 --bench-steps 500
case_run sweep_d_0500 --mode dynamic --no-drop --warmup 200 --bench-steps 500 --limit 500
case_run sweep_d_1000 --mode dynamic --no-drop --warmup 200 --bench-steps 500 --limit 1000
case_run sweep_d_2000 --mode dynamic --no-drop --warmup 200 --bench-steps 500 --limit 2000
case_run sweep_d_4467 --mode dynamic --no-drop --warmup 200 --bench-steps 500

echo "=== sweep end $(date)" | tee -a "$LOG"
