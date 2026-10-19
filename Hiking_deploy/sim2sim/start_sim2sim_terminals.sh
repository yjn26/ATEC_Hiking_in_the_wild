#!/usr/bin/env bash
# Launch the complete loopback-only sim2sim stack in three GNOME terminals.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZMQ_PORT=5556
# Machine-local overrides live in env.local.sh (gitignored).  Use it to point
# the launcher at another checkout's environment, e.g.
#     SIM_ENV_DEPLOY_DIR=/path/to/Hiking_deploy
# Without it the launcher uses this checkout's own sim2sim/env_sim.sh.
if [[ -f "$DEPLOY_DIR/sim2sim/env.local.sh" ]]; then
    source "$DEPLOY_DIR/sim2sim/env.local.sh"
fi
SIM_ENV_DEPLOY_DIR="${SIM_ENV_DEPLOY_DIR:-$DEPLOY_DIR}"
SIM_ENV_SCRIPT="${SIM_ENV_DEPLOY_DIR}/sim2sim/env_sim.sh"
STATE_DIR="${XDG_RUNTIME_DIR:-/tmp}/g1-sim2sim-launch"
BRIDGE_LOG="${STATE_DIR}/bridge.log"
DEPLOY_LOG="${STATE_DIR}/deploy.log"
SELECTOR_SCENE="sim2sim/assets/scene/scene_29dof_terrain_selector.xml"

if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X1 ]]; then
    export DISPLAY=:1
fi
if [[ -z "${XDG_RUNTIME_DIR:-}" && -d "/run/user/$(id -u)" ]]; then
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi
if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" && -S "${XDG_RUNTIME_DIR}/bus" ]]; then
    export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"
fi
if [[ -z "${XAUTHORITY:-}" && -r "/run/user/$(id -u)/gdm/Xauthority" ]]; then
    export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
fi

if ! command -v gnome-terminal >/dev/null; then
    echo "gnome-terminal is required to open the three sim2sim windows." >&2
    exit 1
fi
if ! command -v xdotool >/dev/null; then
    echo "xdotool is required to prepare the MuJoCo keyboard state." >&2
    exit 1
fi

if [[ ! -f "$SIM_ENV_SCRIPT" ]]; then
    echo "Missing sim2sim environment script: $SIM_ENV_SCRIPT" >&2
    exit 1
fi

if ss -ltn | grep -qE ":${ZMQ_PORT}[[:space:]]"; then
    echo "ZMQ port ${ZMQ_PORT} is already in use. Stop the existing bridge first." >&2
    exit 1
fi

mkdir -p "$STATE_DIR"
: > "$BRIDGE_LOG"
: > "$DEPLOY_LOG"

open_terminal() {
    local title="$1"
    local command="$2"
    local shell_command

    printf -v shell_command 'cd %q; source %q; %s; status=$?; printf "\\n[%s] exited with status %%s\\n" "$status"; exec bash' \
        "$DEPLOY_DIR" "$SIM_ENV_SCRIPT" "$command" "$title"
    gnome-terminal --title="$title" -- bash -lc "$shell_command"
}

wait_for_log() {
    local pattern="$1"
    local log_file="$2"
    local description="$3"
    local timeout="$4"

    for _ in $(seq 1 "$timeout"); do
        if grep -q "$pattern" "$log_file"; then
            return 0
        fi
        sleep 1
    done
    echo "Timed out waiting for ${description}; inspect ${log_file}." >&2
    return 1
}

find_mujoco_window() {
    local window_id
    for _ in $(seq 1 30); do
        window_id="$(xdotool search --onlyvisible --name '^MuJoCo' 2>/dev/null | tail -n 1 || true)"
        if [[ -n "$window_id" ]]; then
            printf '%s\n' "$window_id"
            return 0
        fi
        sleep 1
    done
    echo "MuJoCo window did not appear." >&2
    return 1
}

send_mujoco_key() {
    local window_id="$1"
    local key="$2"
    xdotool windowactivate --sync "$window_id"
    xdotool key --window "$window_id" "$key"
}

open_terminal "G1 Sim2Sim Bridge" "python -u sim2sim/g1_mujoco_bridge.py --scene ${SELECTOR_SCENE} --terrain-mode 1 --forward-speed 1.0 2>&1 | tee $(printf '%q' "$BRIDGE_LOG")"

for _ in {1..30}; do
    if ss -ltn | grep -qE ":${ZMQ_PORT}[[:space:]]"; then
        break
    fi
    sleep 1
done

if ! ss -ltn | grep -qE ":${ZMQ_PORT}[[:space:]]"; then
    echo "Bridge did not bind ZMQ port ${ZMQ_PORT} within 30 seconds." >&2
    exit 1
fi

open_terminal "G1 Sim2Sim Deploy" \
    "python -u instinct_onboard/scripts/g1_parkour_laptop.py --zmq_addr tcp://127.0.0.1:${ZMQ_PORT} --lin_vel_range 1.0 1.0 --nodryrun 2>&1 | tee $(printf '%q' "$DEPLOY_LOG")"
open_terminal "G1 Sim2Sim Depth" \
    "real_time_img --source zmq --zmq-addr tcp://127.0.0.1:${ZMQ_PORT} --stages"

wait_for_log "G1ParkourLaptopNode is ready to run" "$DEPLOY_LOG" "deploy readiness" 60
MUJOCO_WINDOW="$(find_mujoco_window)"

# Prepare the requested state machine while the virtual gantry is still on.
send_mujoco_key "$MUJOCO_WINDOW" Return
wait_for_log "ColdStartAgent done" "$DEPLOY_LOG" "cold-start completion" 60
send_mujoco_key "$MUJOCO_WINDOW" 7
wait_for_log "R1 button pressed, switching to stand agent" "$DEPLOY_LOG" "stand mode" 20
# R1/L1 are 1.2-second virtual button pulses.  Do not overlap them or the
# deploy state machine can alternate between stand and parkour.
sleep 2
send_mujoco_key "$MUJOCO_WINDOW" 6
wait_for_log "L1 button pressed, switching to parkour agent" "$DEPLOY_LOG" "parkour mode" 20
sleep 2
send_mujoco_key "$MUJOCO_WINDOW" 8
send_mujoco_key "$MUJOCO_WINDOW" Down

echo "Opened all three terminals on the centered four-direction course: Enter -> 7 -> 6 -> 8."
echo "The robot is in parkour mode with zero speed; Up adds 0.2 m/s (max 1.0), Down stops."
