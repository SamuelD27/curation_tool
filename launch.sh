#!/usr/bin/env bash
# -------------------------------------------------------------------
#  launch.sh  --  Start ComfyUI backend + Curation Tool GUI frontend
#
#  Usage:
#    ./launch.sh                  # defaults
#    ./launch.sh --port 8080      # custom GUI port
#    ./launch.sh --no-backend     # GUI only (ComfyUI already running)
#    ./launch.sh --backend-only   # ComfyUI only, no GUI
# -------------------------------------------------------------------
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────
COMFYUI_DIR="$HOME/ComfyUI"
COMFYUI_VENV="$COMFYUI_DIR/.venv/bin/activate"
CURATION_CONDA_ENV="curation-tool-env"
CURATION_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$CURATION_DIR/logs"
SESSION_ID="$(date +%Y%m%d_%H%M%S)_$$"

# ── Defaults ───────────────────────────────────────────────────────
COMFYUI_PORT=8188
COMFYUI_LISTEN="127.0.0.1"
GUI_PORT=7860
GUI_HOST="0.0.0.0"
NO_BACKEND=false
BACKEND_ONLY=false
SHARE=false
VERBOSE=false
NO_SYNC=false

# ── Remote sync config ────────────────────────────────────────────
SYNC_REMOTE="samueldukmedjian@192.168.1.2"
SYNC_DEST="/Users/samueldukmedjian/OF/ComfyUI_output"
SYNC_INTERVAL=10

# ── Colors ─────────────────────────────────────────────────────────
RST="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
RED="\033[31m"; GREEN="\033[32m"; YELLOW="\033[33m"
BLUE="\033[34m"; MAGENTA="\033[35m"; CYAN="\033[36m"
WHITE="\033[97m"; BG_BLUE="\033[44m"; BG_GREEN="\033[42m"
TAG_BACK="${BOLD}${MAGENTA}[BACKEND]${RST}"
TAG_FRONT="${BOLD}${CYAN}[FRONTEND]${RST}"
TAG_SYS="${BOLD}${YELLOW}[SYSTEM]${RST}"
TAG_OK="${BOLD}${GREEN}[  OK  ]${RST}"
TAG_FAIL="${BOLD}${RED}[ FAIL ]${RST}"
TAG_INFO="${BOLD}${BLUE}[ INFO ]${RST}"
TAG_SYNC="${BOLD}${GREEN}[  SYNC ]${RST}"

# ── Parse args ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)          GUI_PORT="$2"; shift 2 ;;
        --comfyui-port)  COMFYUI_PORT="$2"; shift 2 ;;
        --no-backend)    NO_BACKEND=true; shift ;;
        --backend-only)  BACKEND_ONLY=true; shift ;;
        --share)         SHARE=true; shift ;;
        --no-sync)       NO_SYNC=true; shift ;;
        --sync-dest)     SYNC_DEST="$2"; shift 2 ;;
        --sync-remote)   SYNC_REMOTE="$2"; shift 2 ;;
        --verbose|-v)    VERBOSE=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "  --port PORT / --comfyui-port PORT / --no-backend / --backend-only"
            echo "  --share / --no-sync / --sync-dest PATH / --sync-remote USER@IP"
            echo "  --verbose / --help"
            exit 0 ;;
        *) echo -e "${TAG_FAIL} Unknown option: $1"; exit 1 ;;
    esac
done

COMFYUI_URL="http://${COMFYUI_LISTEN}:${COMFYUI_PORT}"

# ── Console helpers ────────────────────────────────────────────────
timestamp() { date "+%H:%M:%S"; }
log_sys()   { echo -e "${DIM}$(timestamp)${RST}  ${TAG_SYS}  $*"; }
log_ok()    { echo -e "${DIM}$(timestamp)${RST}  ${TAG_OK}  $*"; }
log_fail()  { echo -e "${DIM}$(timestamp)${RST}  ${TAG_FAIL}  $*"; }
log_info()  { echo -e "${DIM}$(timestamp)${RST}  ${TAG_INFO}  $*"; }
log_back()  { echo -e "${DIM}$(timestamp)${RST}  ${TAG_BACK}  $*"; }
log_front() { echo -e "${DIM}$(timestamp)${RST}  ${TAG_FRONT}  $*"; }
separator() { echo -e "${DIM}$(printf '%.0s─' {1..60})${RST}"; }
banner()    { echo -e "\n${BOLD}${BG_BLUE}${WHITE}       CURATION TOOL  --  Launch Script          ${RST}\n"; }

colorize_backend() {
    while IFS= read -r line; do
        case "$line" in
            *ERROR*|*error*|*Error*|*CRITICAL*)
                echo -e "${DIM}$(timestamp)${RST}  ${TAG_BACK}  ${RED}${line}${RST}" ;;
            *WARNING*|*warning*|*Warning*)
                echo -e "${DIM}$(timestamp)${RST}  ${TAG_BACK}  ${YELLOW}${line}${RST}" ;;
            *"Prompt executed"*|*"100%"*|*"done"*|*"loaded"*|*"Loading"*)
                echo -e "${DIM}$(timestamp)${RST}  ${TAG_BACK}  ${GREEN}${line}${RST}" ;;
            *)
                echo -e "${DIM}$(timestamp)${RST}  ${TAG_BACK}  ${DIM}${line}${RST}" ;;
        esac
    done
}

colorize_frontend() {
    while IFS= read -r line; do
        case "$line" in
            *ERROR*|*error*|*Error*|*CRITICAL*)
                echo -e "${DIM}$(timestamp)${RST}  ${TAG_FRONT}  ${RED}${line}${RST}" ;;
            *WARNING*|*warning*|*Warning*)
                echo -e "${DIM}$(timestamp)${RST}  ${TAG_FRONT}  ${YELLOW}${line}${RST}" ;;
            *"Running on"*|*"Generated"*|*"Connected"*|*"exported"*)
                echo -e "${DIM}$(timestamp)${RST}  ${TAG_FRONT}  ${GREEN}${line}${RST}" ;;
            *)
                echo -e "${DIM}$(timestamp)${RST}  ${TAG_FRONT}  ${DIM}${line}${RST}" ;;
        esac
    done
}

# ── Cleanup on exit ────────────────────────────────────────────────
PIDS=()
MONITOR_PID=""

cleanup() {
    echo ""
    separator
    log_sys "Shutting down..."

    # Stop monitor first (let it capture final state)
    if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        sleep 2  # give monitor time to detect exits and dump diagnostics
        kill "$MONITOR_PID" 2>/dev/null
        wait "$MONITOR_PID" 2>/dev/null || true
    fi

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log_sys "Stopping PID $pid"
            kill "$pid" 2>/dev/null
            local waited=0
            while kill -0 "$pid" 2>/dev/null && [[ $waited -lt 5 ]]; do
                sleep 1
                ((waited++))
            done
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null
            fi
            wait "$pid" 2>/dev/null || true
        fi
    done
    rm -f /tmp/comfyui_sync_marker.* 2>/dev/null
    log_ok "All processes stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# ── Pre-flight checks ─────────────────────────────────────────────
mkdir -p "$LOG_DIR"

banner
separator
log_sys "Running pre-flight checks..."

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "")
    if [[ -n "$GPU_INFO" ]]; then
        IFS=',' read -r GPU_NAME VRAM_USED VRAM_TOTAL GPU_UTIL <<< "$GPU_INFO"
        log_ok "GPU: ${GREEN}$(echo "$GPU_NAME"|xargs)${RST}  VRAM: $(echo "$VRAM_USED"|xargs)/$(echo "$VRAM_TOTAL"|xargs) MiB  Util: $(echo "$GPU_UTIL"|xargs)%"
    fi
fi

log_info "Memory: $(free -h | awk 'NR==2{printf "Total: %s  Used: %s  Avail: %s", $2, $3, $7}')"
log_info "Disk: $(df -h "$HOME" | awk 'NR==2{print $4}') available"
log_info "Logs: ${CYAN}${LOG_DIR}/${RST}"

separator

# ── Collect monitor args (built up as services start) ──────────────
MONITOR_ARGS=(--session "$SESSION_ID" --log-dir "$LOG_DIR")

# ── Start ComfyUI backend ─────────────────────────────────────────
COMFYUI_PID=""

if [[ "$NO_BACKEND" == false ]]; then
    log_back "Starting ComfyUI on port ${BOLD}${COMFYUI_PORT}${RST}..."

    if ! [[ -f "$COMFYUI_VENV" ]]; then
        log_fail "ComfyUI venv not found at ${COMFYUI_VENV}"
        exit 1
    fi

    COMFYUI_LOG="$LOG_DIR/comfyui_${SESSION_ID}.log"

    (
        source "$COMFYUI_VENV"
        cd "$COMFYUI_DIR"
        python main.py \
            --listen "$COMFYUI_LISTEN" \
            --port "$COMFYUI_PORT" \
            --preview-method auto \
            2>&1 | tee -a "$COMFYUI_LOG" | colorize_backend
    ) &
    COMFYUI_PID=$!
    PIDS+=("$COMFYUI_PID")
    MONITOR_ARGS+=(--pid "$COMFYUI_PID:comfyui" --log "$COMFYUI_PID:$COMFYUI_LOG")

    log_back "PID: ${BOLD}${COMFYUI_PID}${RST}  Log: ${CYAN}${COMFYUI_LOG}${RST}"

    # Wait for ComfyUI to be ready
    log_back "Waiting for ComfyUI to be ready..."
    READY=false
    for i in $(seq 1 60); do
        if ! kill -0 "$COMFYUI_PID" 2>/dev/null; then
            log_fail "ComfyUI died during startup -- check logs"
            break
        fi
        if curl -sf "${COMFYUI_URL}/system_stats" >/dev/null 2>&1; then
            READY=true
            break
        fi
        sleep 2
    done

    if [[ "$READY" == true ]]; then
        log_ok "ComfyUI is ${GREEN}${BOLD}ready${RST} at ${CYAN}${COMFYUI_URL}${RST}"
    elif kill -0 "$COMFYUI_PID" 2>/dev/null; then
        log_fail "ComfyUI did not respond within 120s -- starting GUI anyway"
    fi
else
    log_sys "Skipping ComfyUI backend (--no-backend)"
fi

separator

# ── Start Curation Tool GUI ───────────────────────────────────────
GUI_PID=""

if [[ "$BACKEND_ONLY" == false ]]; then
    log_front "Starting Gradio GUI on port ${BOLD}${GUI_PORT}${RST}..."

    GUI_LOG="$LOG_DIR/gui_${SESSION_ID}.log"

    CURATE_ARGS=(gui --host "$GUI_HOST" --port "$GUI_PORT")
    [[ "$SHARE" == true ]] && CURATE_ARGS+=(--share)
    VERBOSE_FLAG=""
    [[ "$VERBOSE" == true ]] && VERBOSE_FLAG="-v"

    (
        eval "$(conda shell.bash hook 2>/dev/null)"
        conda activate "$CURATION_CONDA_ENV"
        cd "$CURATION_DIR"
        curate $VERBOSE_FLAG --comfyui-url "$COMFYUI_URL" "${CURATE_ARGS[@]}" \
            2>&1 | tee -a "$GUI_LOG" | colorize_frontend
    ) &
    GUI_PID=$!
    PIDS+=("$GUI_PID")
    MONITOR_ARGS+=(--pid "$GUI_PID:gui" --log "$GUI_PID:$GUI_LOG")

    log_front "PID: ${BOLD}${GUI_PID}${RST}  Log: ${CYAN}${GUI_LOG}${RST}"
    sleep 3
    log_ok "GUI should be available at ${CYAN}${BOLD}http://${GUI_HOST}:${GUI_PORT}${RST}"
else
    log_sys "Skipping GUI frontend (--backend-only)"
fi

# ── Auto-sync ComfyUI output to Mac ───────────────────────────────
SYNC_PID=""

if [[ "$NO_SYNC" == false ]]; then
    if ssh -o BatchMode=yes -o ConnectTimeout=3 "$SYNC_REMOTE" "true" 2>/dev/null; then
        log_info "Starting output auto-sync to ${CYAN}${SYNC_REMOTE}:${SYNC_DEST}${RST}"

        COMFYUI_OUTPUT="$COMFYUI_DIR/output"
        SYNC_LOG="$LOG_DIR/sync_${SESSION_ID}.log"
        SYNC_MARKER=$(mktemp /tmp/comfyui_sync_marker.XXXXXX)
        touch "$SYNC_MARKER"

        (
            while true; do
                sleep "$SYNC_INTERVAL"
                FILELIST=$(find "$COMFYUI_OUTPUT" -maxdepth 1 -type f -newer "$SYNC_MARKER" -printf '%f\n' 2>/dev/null)
                if [[ -n "$FILELIST" ]]; then
                    TMPLIST=$(mktemp)
                    echo "$FILELIST" > "$TMPLIST"
                    SYNCED=$(rsync -az --itemize-changes --files-from="$TMPLIST" "$COMFYUI_OUTPUT/" "$SYNC_REMOTE:$SYNC_DEST/" 2>> "$SYNC_LOG")
                    COUNT=$(echo "$SYNCED" | grep -c '^>f' || true)
                    if [[ "$COUNT" -gt 0 ]]; then
                        echo -e "${DIM}$(date +%H:%M:%S)${RST}  ${TAG_SYNC}  ${GREEN}${COUNT} new file(s) synced to Mac${RST}"
                    fi
                    touch "$SYNC_MARKER"
                    rm -f "$TMPLIST"
                fi
            done
        ) &
        SYNC_PID=$!
        PIDS+=("$SYNC_PID")
        log_ok "Auto-sync active (every ${SYNC_INTERVAL}s)  PID: ${BOLD}${SYNC_PID}${RST}"
    else
        log_fail "Cannot reach ${SYNC_REMOTE} -- auto-sync disabled"
    fi
else
    log_sys "Skipping auto-sync (--no-sync)"
fi

separator

# ── Start background monitor ──────────────────────────────────────
if [[ ${#MONITOR_ARGS[@]} -gt 4 ]]; then
    # Only start if there are processes to watch (more than just --session and --log-dir)
    bash "$CURATION_DIR/monitor.sh" "${MONITOR_ARGS[@]}" &
    MONITOR_PID=$!
    log_info "Monitor PID: ${BOLD}${MONITOR_PID}${RST}  (metrics + events + diagnostics)"
fi

separator

# ── Summary ────────────────────────────────────────────────────────
echo ""
echo -e "  ${BOLD}${BG_GREEN}${WHITE}  Services Running  ${RST}"
echo ""
[[ -n "$COMFYUI_PID" ]] && \
    echo -e "  ${MAGENTA}Backend${RST}   ${COMFYUI_URL}        PID ${COMFYUI_PID}"
[[ -n "$GUI_PID" ]] && \
    echo -e "  ${CYAN}Frontend${RST}  http://${GUI_HOST}:${GUI_PORT}   PID ${GUI_PID}"
[[ -n "$SYNC_PID" ]] && \
    echo -e "  ${GREEN}Sync${RST}      -> ${SYNC_REMOTE}:${SYNC_DEST}   PID ${SYNC_PID}"
echo ""
echo -e "  ${DIM}Logs: ${LOG_DIR}/${RST}"
echo -e "  ${DIM}Press ${BOLD}Ctrl+C${RST}${DIM} to stop all services${RST}"
echo ""
separator

# ── Wait for processes ─────────────────────────────────────────────
wait
