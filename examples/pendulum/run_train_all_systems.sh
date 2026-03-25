#!/usr/bin/env bash
# Train all 5 systems with SE(3) Hamiltonian NODE on 2 GPUs in parallel.
#
# GPU assignment:
#   GPU 0: pendulum, double_pendulum, three_body
#   GPU 1: mass_spring, two_body
#
# Usage:
#   bash run_train_all_systems.sh                # default 2000 steps, 64 samples
#   bash run_train_all_systems.sh --total_steps 500 --samples 32   # override
#
# Logs are saved to examples/pendulum/data/logs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="hgan"
TRAIN_SCRIPT="train_multisystem_SE3.py"
LOG_DIR="./data/logs"
mkdir -p "$LOG_DIR"

# Default training arguments (can be overridden via CLI)
TOTAL_STEPS="${TOTAL_STEPS:-50000}"
SAMPLES="${SAMPLES:-1000}"
NUM_POINTS="${NUM_POINTS:-5}"
SOLVER="${SOLVER:-rk4}"
EXTRA_ARGS="$*"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "Multi-system SE(3) Hamiltonian NODE training"
echo "  Steps: $TOTAL_STEPS | Samples: $SAMPLES | Solver: $SOLVER"
echo "  Extra args: $EXTRA_ARGS"
echo "  Logs: $LOG_DIR"
echo "============================================================"

SYSTEMS_GPU0=( "double_pendulum" "three_body")
SYSTEMS_GPU1=("mass_spring""pendulum" "two_body")

PIDS=()
declare -A PID_SYSTEM

launch() {
    local gpu=$1
    local system=$2
    local logfile="${LOG_DIR}/${system}_${TIMESTAMP}.log"

    echo "Launching $system on GPU $gpu -> $logfile"
    conda run -n "$CONDA_ENV" --no-capture-output \
        python "$TRAIN_SCRIPT" \
        --system "$system" \
        --gpu "$gpu" \
        --total_steps "$TOTAL_STEPS" \
        --samples "$SAMPLES" \
        --num_points "$NUM_POINTS" \
        --solver "$SOLVER" \
        --name "multisystem" \
        --print_every 100 \
        --verbose \
        $EXTRA_ARGS \
        > "$logfile" 2>&1 &

    local pid=$!
    PIDS+=("$pid")
    PID_SYSTEM[$pid]="$system"
}

# Launch all systems
for sys in "${SYSTEMS_GPU0[@]}"; do
    launch 0 "$sys"
done
for sys in "${SYSTEMS_GPU1[@]}"; do
    launch 1 "$sys"
done

echo ""
echo "All ${#PIDS[@]} jobs launched. Waiting for completion..."
echo "  PIDs: ${PIDS[*]}"
echo ""

# Wait and collect results
FAILED=()
PASSED=()

for pid in "${PIDS[@]}"; do
    sys="${PID_SYSTEM[$pid]}"
    if wait "$pid"; then
        echo "  DONE: $sys (pid $pid) - SUCCESS"
        PASSED+=("$sys")
    else
        echo "  DONE: $sys (pid $pid) - FAILED (exit $?)"
        FAILED+=("$sys")
    fi
done

echo ""
echo "============================================================"
echo "Results: ${#PASSED[@]}/${#PIDS[@]} succeeded"
if [ ${#PASSED[@]} -gt 0 ]; then
    echo "  Passed: ${PASSED[*]}"
fi
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Failed: ${FAILED[*]}"
    echo ""
    echo "Check logs in $LOG_DIR for details."
    exit 1
fi
echo "============================================================"
