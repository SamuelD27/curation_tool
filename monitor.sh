#!/usr/bin/env bash
# -------------------------------------------------------------------
#  monitor.sh  --  Background system & process monitor
#
#  Writes detailed machine-readable snapshots to logs/.
#  Launched automatically by launch.sh, or run standalone:
#
#    ./monitor.sh --pid 12345:comfyui --pid 67890:gui
#    ./monitor.sh --pid 12345:comfyui --interval 15
# -------------------------------------------------------------------
set -uo pipefail

LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
SESSION_ID="${MONITOR_SESSION_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
INTERVAL=30
WATCHED_PIDS=()   # "pid:name" pairs
WATCHED_LOGS=()   # "pid:logfile" pairs for exit diagnostics

# ── Parse args ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid)       WATCHED_PIDS+=("$2"); shift 2 ;;
        --log)       WATCHED_LOGS+=("$2"); shift 2 ;;
        --interval)  INTERVAL="$2"; shift 2 ;;
        --log-dir)   LOG_DIR="$2"; shift 2 ;;
        --session)   SESSION_ID="$2"; shift 2 ;;
        *)           shift ;;
    esac
done

mkdir -p "$LOG_DIR"

# Output files
METRICS_LOG="$LOG_DIR/metrics_${SESSION_ID}.jsonl"
EVENTS_LOG="$LOG_DIR/events_${SESSION_ID}.jsonl"
DIAG_LOG="$LOG_DIR/diagnostics_${SESSION_ID}.txt"

# ═══════════════════════════════════════════════════════════════════
#  DATA COLLECTORS
# ═══════════════════════════════════════════════════════════════════

collect_memory() {
    # Full /proc/meminfo as JSON
    local -A m
    while IFS=':' read -r key val; do
        key=$(echo "$key" | xargs)
        val=$(echo "$val" | tr -d ' kB' | xargs)
        [[ -n "$val" ]] && m["$key"]="$val"
    done < /proc/meminfo

    printf '{"total_kb":%s,"free_kb":%s,"avail_kb":%s,"buffers_kb":%s,"cached_kb":%s,"slab_kb":%s,"active_kb":%s,"inactive_kb":%s,"dirty_kb":%s,"writeback_kb":%s,"mapped_kb":%s,"shmem_kb":%s,"anon_pages_kb":%s,"swap_total_kb":%s,"swap_free_kb":%s,"swap_cached_kb":%s,"commit_limit_kb":%s,"committed_kb":%s,"hugepages_total":%s,"hugepages_free":%s,"page_tables_kb":%s}' \
        "${m[MemTotal]:-0}" "${m[MemFree]:-0}" "${m[MemAvailable]:-0}" \
        "${m[Buffers]:-0}" "${m[Cached]:-0}" "${m[Slab]:-0}" \
        "${m[Active]:-0}" "${m[Inactive]:-0}" "${m[Dirty]:-0}" \
        "${m[Writeback]:-0}" "${m[Mapped]:-0}" "${m[Shmem]:-0}" \
        "${m[AnonPages]:-0}" "${m[SwapTotal]:-0}" "${m[SwapFree]:-0}" \
        "${m[SwapCached]:-0}" "${m[CommitLimit]:-0}" "${m[Committed_AS]:-0}" \
        "${m[HugePages_Total]:-0}" "${m[HugePages_Free]:-0}" "${m[PageTables]:-0}"
}

collect_cpu() {
    local load_1 load_5 load_15 procs running total
    read -r load_1 load_5 load_15 procs _ < /proc/loadavg
    running="${procs%/*}"
    total="${procs#*/}"

    # Per-CPU aggregate from /proc/stat (first line = total)
    local cpu_user cpu_nice cpu_sys cpu_idle cpu_iowait cpu_irq cpu_softirq cpu_steal
    read -r _ cpu_user cpu_nice cpu_sys cpu_idle cpu_iowait cpu_irq cpu_softirq cpu_steal _ < <(head -1 /proc/stat)

    printf '{"load_1m":%s,"load_5m":%s,"load_15m":%s,"running_procs":%s,"total_procs":%s,"user":%s,"nice":%s,"system":%s,"idle":%s,"iowait":%s,"irq":%s,"softirq":%s,"steal":%s}' \
        "$load_1" "$load_5" "$load_15" "$running" "$total" \
        "$cpu_user" "$cpu_nice" "$cpu_sys" "$cpu_idle" "$cpu_iowait" "$cpu_irq" "$cpu_softirq" "${cpu_steal:-0}"
}

collect_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        printf 'null'
        return
    fi

    local csv
    csv=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory,pstate,fan.speed,enforced.power.limit \
        --format=csv,noheader,nounits 2>/dev/null || echo "")

    if [[ -z "$csv" ]]; then
        printf 'null'
        return
    fi

    IFS=',' read -r g_util g_mem_util g_temp g_power g_power_limit g_sm g_mem_clk g_sm_max g_mem_max g_pstate g_fan g_enforced_power <<< "$csv"

    printf '{"util_pct":%s,"mem_util_pct":%s,"temp_c":%s,"power_w":%s,"power_limit_w":%s,"sm_clock_mhz":%s,"mem_clock_mhz":%s,"sm_max_mhz":%s,"mem_max_mhz":%s,"pstate":"%s","fan_pct":%s,"enforced_power_w":%s}' \
        "$(echo "$g_util"|xargs)" "$(echo "$g_mem_util"|xargs)" "$(echo "$g_temp"|xargs)" \
        "$(echo "$g_power"|xargs)" "$(echo "$g_power_limit"|xargs)" \
        "$(echo "$g_sm"|xargs)" "$(echo "$g_mem_clk"|xargs)" \
        "$(echo "$g_sm_max"|xargs)" "$(echo "$g_mem_max"|xargs)" \
        "$(echo "$g_pstate"|xargs)" "$(echo "${g_fan:-0}"|xargs)" "$(echo "${g_enforced_power:-0}"|xargs)"
}

collect_gpu_processes() {
    if ! command -v nvidia-smi &>/dev/null; then
        printf '[]'
        return
    fi

    local lines
    lines=$(nvidia-smi --query-compute-apps=pid,used_gpu_memory,name --format=csv,noheader,nounits 2>/dev/null || echo "")

    if [[ -z "$lines" ]]; then
        printf '[]'
        return
    fi

    local result="[" first=true
    while IFS=',' read -r gp_pid gp_mem gp_name; do
        [[ -z "$gp_pid" ]] && continue
        $first || result+=","
        first=false
        result+=$(printf '{"pid":%s,"gpu_mem_mb":%s,"name":"%s"}' \
            "$(echo "$gp_pid"|xargs)" "$(echo "$gp_mem"|xargs)" "$(echo "$gp_name"|xargs)")
    done <<< "$lines"
    printf '%s]' "$result"
}

collect_psi() {
    # Pressure Stall Information -- shows system resource contention
    local mem_some=0 mem_full=0 cpu_some=0 io_some=0 io_full=0
    local mem_some_60=0 mem_full_60=0 cpu_some_60=0 io_some_60=0 io_full_60=0
    local mem_some_300=0 mem_full_300=0 cpu_some_300=0 io_some_300=0 io_full_300=0
    local mem_some_total=0 mem_full_total=0 cpu_some_total=0 io_some_total=0 io_full_total=0

    if [[ -f /proc/pressure/memory ]]; then
        eval "$(awk '/^some/{split($2,a,"="); split($3,b,"="); split($4,c,"="); split($5,d,"="); printf "mem_some=%s mem_some_60=%s mem_some_300=%s mem_some_total=%s", a[2], b[2], c[2], d[2]}' /proc/pressure/memory)"
        eval "$(awk '/^full/{split($2,a,"="); split($3,b,"="); split($4,c,"="); split($5,d,"="); printf "mem_full=%s mem_full_60=%s mem_full_300=%s mem_full_total=%s", a[2], b[2], c[2], d[2]}' /proc/pressure/memory)"
    fi
    if [[ -f /proc/pressure/cpu ]]; then
        eval "$(awk '/^some/{split($2,a,"="); split($3,b,"="); split($4,c,"="); split($5,d,"="); printf "cpu_some=%s cpu_some_60=%s cpu_some_300=%s cpu_some_total=%s", a[2], b[2], c[2], d[2]}' /proc/pressure/cpu)"
    fi
    if [[ -f /proc/pressure/io ]]; then
        eval "$(awk '/^some/{split($2,a,"="); split($3,b,"="); split($4,c,"="); split($5,d,"="); printf "io_some=%s io_some_60=%s io_some_300=%s io_some_total=%s", a[2], b[2], c[2], d[2]}' /proc/pressure/io)"
        eval "$(awk '/^full/{split($2,a,"="); split($3,b,"="); split($4,c,"="); split($5,d,"="); printf "io_full=%s io_full_60=%s io_full_300=%s io_full_total=%s", a[2], b[2], c[2], d[2]}' /proc/pressure/io)"
    fi

    printf '{"mem_some_10":%s,"mem_some_60":%s,"mem_some_300":%s,"mem_some_total_us":%s,"mem_full_10":%s,"mem_full_60":%s,"mem_full_300":%s,"mem_full_total_us":%s,"cpu_some_10":%s,"cpu_some_60":%s,"cpu_some_300":%s,"cpu_some_total_us":%s,"io_some_10":%s,"io_some_60":%s,"io_some_300":%s,"io_some_total_us":%s,"io_full_10":%s,"io_full_60":%s,"io_full_300":%s,"io_full_total_us":%s}' \
        "$mem_some" "$mem_some_60" "$mem_some_300" "$mem_some_total" \
        "$mem_full" "$mem_full_60" "$mem_full_300" "$mem_full_total" \
        "$cpu_some" "$cpu_some_60" "$cpu_some_300" "$cpu_some_total" \
        "$io_some" "$io_some_60" "$io_some_300" "$io_some_total" \
        "$io_full" "$io_full_60" "$io_full_300" "$io_full_total"
}

collect_disk() {
    local total used avail use_pct
    read -r total used avail use_pct < <(df -k "$HOME" | awk 'NR==2{gsub(/%/,"",$5); print $2, $3, $4, $5}')
    # Disk I/O from /proc/diskstats (all disks combined reads/writes)
    local reads writes read_sectors write_sectors io_ms
    read -r reads writes read_sectors write_sectors io_ms < <(
        awk '$3 ~ /^(nvme|sd|vd)/{r+=$4; w+=$8; rs+=$6; ws+=$10; io+=$13} END{print r, w, rs, ws, io}' /proc/diskstats 2>/dev/null || echo "0 0 0 0 0"
    )
    printf '{"home_total_kb":%s,"home_used_kb":%s,"home_avail_kb":%s,"home_use_pct":%s,"io_reads":%s,"io_writes":%s,"io_read_sectors":%s,"io_write_sectors":%s,"io_ms":%s}' \
        "${total:-0}" "${used:-0}" "${avail:-0}" "${use_pct:-0}" \
        "$reads" "$writes" "$read_sectors" "$write_sectors" "$io_ms"
}

collect_network() {
    # TCP connections to ComfyUI port + overall socket counts
    local comfyui_conns established time_wait close_wait
    comfyui_conns=$(ss -tn 2>/dev/null | grep -c ":8188 " || echo "0")
    established=$(ss -tn state established 2>/dev/null | tail -n +2 | wc -l || echo "0")
    time_wait=$(ss -tn state time-wait 2>/dev/null | tail -n +2 | wc -l || echo "0")
    close_wait=$(ss -tn state close-wait 2>/dev/null | tail -n +2 | wc -l || echo "0")
    printf '{"comfyui_port_conns":%s,"tcp_established":%s,"tcp_time_wait":%s,"tcp_close_wait":%s}' \
        "$comfyui_conns" "$established" "$time_wait" "$close_wait"
}

collect_process() {
    local pid="$1"
    if [[ ! -d "/proc/$pid" ]]; then
        printf 'null'
        return
    fi

    # Main process stats
    local comm state ppid pgrp vsz_pages rss_pages threads utime stime starttime
    read -r _ comm state ppid pgrp _ _ _ _ _ _ _ _ utime stime _ _ _ _ threads _ starttime vsz_pages rss_pages _ < <(cat "/proc/$pid/stat" 2>/dev/null || echo "0 (?) ? 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")
    comm=$(echo "$comm" | tr -d '()')
    local rss_kb=$((rss_pages * 4))
    local vsz_kb=$((vsz_pages / 1024))

    # OOM info
    local oom_score oom_adj
    oom_score=$(cat "/proc/$pid/oom_score" 2>/dev/null || echo "-1")
    oom_adj=$(cat "/proc/$pid/oom_score_adj" 2>/dev/null || echo "0")

    # Open FDs
    local num_fds
    num_fds=$(ls -1 "/proc/$pid/fd" 2>/dev/null | wc -l || echo "0")

    # Voluntary/involuntary context switches
    local vol_cs invol_cs
    vol_cs=$(awk '/^voluntary_ctxt_switches:/{print $2}' "/proc/$pid/status" 2>/dev/null || echo "0")
    invol_cs=$(awk '/^nonvoluntary_ctxt_switches:/{print $2}' "/proc/$pid/status" 2>/dev/null || echo "0")

    # VmPeak, VmHWM (high water marks)
    local vm_peak vm_hwm vm_swap
    vm_peak=$(awk '/^VmPeak:/{print $2}' "/proc/$pid/status" 2>/dev/null || echo "0")
    vm_hwm=$(awk '/^VmHWM:/{print $2}' "/proc/$pid/status" 2>/dev/null || echo "0")
    vm_swap=$(awk '/^VmSwap:/{print $2}' "/proc/$pid/status" 2>/dev/null || echo "0")

    # Child processes
    local children="["
    local first=true
    for cpid in $(pgrep -P "$pid" 2>/dev/null || true); do
        $first || children+=","
        first=false
        local ccomm crss cvsz cthreads cutime cstime
        ccomm=$(cat "/proc/$cpid/comm" 2>/dev/null || echo "?")
        read -r _ _ _ _ _ _ _ _ _ _ _ _ _ cutime cstime _ _ _ _ cthreads _ _ cvsz_p crss_p _ < <(cat "/proc/$cpid/stat" 2>/dev/null || echo "0 (?) ? 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")
        crss=$((crss_p * 4))
        cvsz=$((cvsz_p / 1024))
        local c_oom=$(cat "/proc/$cpid/oom_score" 2>/dev/null || echo "-1")
        local c_fds=$(ls -1 "/proc/$cpid/fd" 2>/dev/null | wc -l || echo "0")
        local c_peak=$(awk '/^VmPeak:/{print $2}' "/proc/$cpid/status" 2>/dev/null || echo "0")
        local c_hwm=$(awk '/^VmHWM:/{print $2}' "/proc/$cpid/status" 2>/dev/null || echo "0")
        children+=$(printf '{"pid":%s,"comm":"%s","rss_kb":%s,"vsz_kb":%s,"threads":%s,"utime":%s,"stime":%s,"oom_score":%s,"open_fds":%s,"vm_peak_kb":%s,"vm_hwm_kb":%s}' \
            "$cpid" "$ccomm" "$crss" "$cvsz" "$cthreads" "$cutime" "$cstime" "$c_oom" "$c_fds" "$c_peak" "$c_hwm")
    done
    children+="]"

    printf '{"pid":%s,"comm":"%s","state":"%s","ppid":%s,"rss_kb":%s,"vsz_kb":%s,"threads":%s,"utime":%s,"stime":%s,"oom_score":%s,"oom_score_adj":%s,"open_fds":%s,"vol_cs":%s,"invol_cs":%s,"vm_peak_kb":%s,"vm_hwm_kb":%s,"vm_swap_kb":%s,"children":%s}' \
        "$pid" "$comm" "$state" "$ppid" "$rss_kb" "$vsz_kb" "$threads" "$utime" "$stime" \
        "$oom_score" "$oom_adj" "$num_fds" "$vol_cs" "$invol_cs" \
        "${vm_peak:-0}" "${vm_hwm:-0}" "${vm_swap:-0}" "$children"
}

# ═══════════════════════════════════════════════════════════════════
#  WRITERS
# ═══════════════════════════════════════════════════════════════════

write_metrics() {
    # Full system snapshot + per-watched-process stats
    local ts
    ts=$(date -Iseconds)

    local procs_json="{"
    local first=true
    for entry in "${WATCHED_PIDS[@]}"; do
        local wpid="${entry%%:*}"
        local wname="${entry#*:}"
        $first || procs_json+=","
        first=false
        procs_json+="\"$wname\":$(collect_process "$wpid")"
    done
    procs_json+="}"

    printf '{"ts":"%s","session":"%s","mem":%s,"cpu":%s,"gpu":%s,"gpu_procs":%s,"psi":%s,"disk":%s,"net":%s,"procs":%s}\n' \
        "$ts" "$SESSION_ID" \
        "$(collect_memory)" "$(collect_cpu)" "$(collect_gpu)" "$(collect_gpu_processes)" \
        "$(collect_psi)" "$(collect_disk)" "$(collect_network)" "$procs_json" \
        >> "$METRICS_LOG"
}

write_event() {
    local event="$1" service="$2" pid="${3:-0}"
    shift 3
    local extra=""
    [[ $# -gt 0 ]] && extra=",$*"
    printf '{"ts":"%s","session":"%s","event":"%s","service":"%s","pid":%s%s}\n' \
        "$(date -Iseconds)" "$SESSION_ID" "$event" "$service" "$pid" "$extra" \
        >> "$EVENTS_LOG"
}

write_diagnostics() {
    # Full crash/exit diagnostic dump to text file
    local service="$1" pid="$2" exit_code="$3" log_file="${4:-}"
    {
        echo "================================================================"
        echo "DIAGNOSTIC DUMP: $service (PID $pid) exit_code=$exit_code"
        echo "Time: $(date -Iseconds)"
        echo "================================================================"
        echo ""
        echo "--- exit code analysis ---"
        if [[ $exit_code -eq 0 ]]; then
            echo "Clean exit"
        elif [[ $exit_code -le 128 ]]; then
            echo "Error exit (code $exit_code)"
        else
            local sig_num=$((exit_code - 128))
            local sig_name
            sig_name=$(kill -l "$sig_num" 2>/dev/null || echo "SIG$sig_num")
            echo "Killed by signal $sig_name ($sig_num)"
            echo "  SIGKILL(9)=OOM/external  SIGTERM(15)=graceful  SIGSEGV(11)=segfault  SIGABRT(6)=abort"
        fi
        echo ""
        echo "--- /proc/$pid/status ---"
        cat "/proc/$pid/status" 2>/dev/null || echo "(process already gone)"
        echo ""
        echo "--- /proc/$pid/oom_score ---"
        cat "/proc/$pid/oom_score" 2>/dev/null || echo "(process already gone)"
        echo ""
        echo "--- /proc/$pid/cgroup ---"
        cat "/proc/$pid/cgroup" 2>/dev/null || echo "(process already gone)"
        echo ""
        echo "--- /proc/$pid/maps summary (top 10 by size) ---"
        awk '{split($1,a,"-"); size=strtonum("0x"a[2])-strtonum("0x"a[1]); if(size>1048576) printf "%8.1f MB  %s %s\n", size/1048576, $2, $6}' \
            "/proc/$pid/maps" 2>/dev/null | sort -rn | head -10 || echo "(process already gone)"
        echo ""
        echo "--- free -m ---"
        free -m
        echo ""
        echo "--- nvidia-smi ---"
        nvidia-smi 2>/dev/null || echo "(unavailable)"
        echo ""
        echo "--- dmesg tail (last 30 lines, OOM/kill/memory filtered) ---"
        dmesg --time-format iso 2>/dev/null | tail -30 || echo "(unavailable)"
        echo ""
        echo "--- dmesg OOM/kill grep ---"
        dmesg --time-format iso 2>/dev/null | grep -iE "oom|kill|memory|out of" | tail -10 || echo "(none found)"
        echo ""
        echo "--- top processes by RSS ---"
        ps aux --sort=-rss 2>/dev/null | head -15 || echo "(unavailable)"
        echo ""
        if [[ -n "$log_file" && -f "$log_file" ]]; then
            echo "--- last 50 lines of $log_file ---"
            tail -50 "$log_file"
            echo ""
        fi
        echo "================================================================"
        echo "END DIAGNOSTIC DUMP"
        echo "================================================================"
        echo ""
    } >> "$DIAG_LOG" 2>&1
}

# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════

# Build a pid->name and pid->logfile lookup
declare -A PID_TO_NAME
declare -A PID_TO_LOG
declare -A PID_ALIVE

for entry in "${WATCHED_PIDS[@]}"; do
    _pid="${entry%%:*}"
    _name="${entry#*:}"
    PID_TO_NAME["$_pid"]="$_name"
    PID_ALIVE["$_pid"]=1
done

for entry in "${WATCHED_LOGS[@]}"; do
    _pid="${entry%%:*}"
    _log="${entry#*:}"
    PID_TO_LOG["$_pid"]="$_log"
done

# Log startup
write_event "monitor_start" "monitor" "$$" "\"metrics_log\":\"$METRICS_LOG\",\"events_log\":\"$EVENTS_LOG\",\"diag_log\":\"$DIAG_LOG\",\"interval\":$INTERVAL,\"watched\":[$(printf '"%s",' "${WATCHED_PIDS[@]}" | sed 's/,$//')]"

# Initial snapshot
write_metrics

# Monitor loop
while true; do
    sleep "$INTERVAL"

    # System metrics snapshot
    write_metrics

    # Check each watched process
    any_alive=false
    for entry in "${WATCHED_PIDS[@]}"; do
        _pid="${entry%%:*}"
        _name="${entry#*:}"

        if [[ "${PID_ALIVE[$_pid]:-0}" != "1" ]]; then
            continue
        fi

        if ! kill -0 "$_pid" 2>/dev/null; then
            # Process died -- capture exit code
            wait "$_pid" 2>/dev/null || true
            exit_code=$?
            signal_extra=""
            if [[ $exit_code -gt 128 ]]; then
                sig_num=$((exit_code - 128))
                sig_name=$(kill -l "$sig_num" 2>/dev/null || echo "SIG$sig_num")
                signal_extra=",\"signal\":\"$sig_name\",\"signal_num\":$sig_num"
            fi

            write_event "process_exit" "$_name" "$_pid" \
                "\"exit_code\":$exit_code$signal_extra"

            # Full diagnostic dump
            write_diagnostics "$_name" "$_pid" "$exit_code" "${PID_TO_LOG[$_pid]:-}"

            # Final metrics snapshot right at death
            write_metrics

            PID_ALIVE["$_pid"]=0
        else
            any_alive=true
        fi
    done

    if [[ "$any_alive" == false ]]; then
        write_event "monitor_stop" "monitor" "$$" "\"reason\":\"all_watched_exited\""
        break
    fi
done
