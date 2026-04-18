#!/bin/bash
# Monitor kai0_mixed_1 training on both gf0 and gf1
# Usage:
#   bash monitor_kai0_mixed_1.sh              # loop, refresh every 300s
#   bash monitor_kai0_mixed_1.sh 60           # loop, refresh every 60s
#   bash monitor_kai0_mixed_1.sh --once       # print once and exit

ONCE=false
if [[ "$1" == "--once" ]]; then
    ONCE=true
    REFRESH=0
else
    REFRESH=${1:-300}  # default 5 min
fi
GF0_HOST="root@192.168.0.144"
GF0_SSH_PORT=2222
WORK_DIR="/home/tim/workspace/deepdive_kai0"
KAI0_DIR="$WORK_DIR/kai0"
CKPT_DIR="$KAI0_DIR/checkpoints"

# Expected splits per machine
GF1_SPLITS=(0 2)
GF0_SPLITS=(1 3)

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

get_split_status() {
    # Args: split_id log_file
    local split=$1
    local log=$2

    # Check if completed
    if grep -q "=== END split_$split" "$log" 2>/dev/null; then
        if grep -q "split_$split FAILED" "$log" 2>/dev/null; then
            echo "FAILED"
        else
            echo "DONE"
        fi
        return
    fi

    # Check if started
    if grep -q "=== START split_$split" "$log" 2>/dev/null; then
        # Get latest step
        local step=$(grep -oE "Progress on: [0-9.]+[kKit]+/[0-9.]+[kKit]+" "$log" 2>/dev/null | tail -1 | sed 's/Progress on: //')
        local loss=$(grep "^Step " "$log" 2>/dev/null | tail -1 | sed 's/.*loss=//;s/,.*//')
        echo "RUNNING ${step:-?} loss=${loss:-?}"
        return
    fi

    echo "PENDING"
}

get_checkpoint_steps() {
    # Args: split_id
    local split=$1
    local dir="$CKPT_DIR/kai0_mixed_1_split_$split/split_${split}_v1"
    if [[ -d "$dir" ]]; then
        ls "$dir" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' '
    fi
}

get_gpu_summary() {
    # Args: "local" or "remote"
    local mode=$1
    if [[ "$mode" == "local" ]]; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null
    else
        ssh -p $GF0_SSH_PORT -o ConnectTimeout=5 $GF0_HOST \
            "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits" 2>/dev/null
    fi
}

format_gpu_summary() {
    # Pretty format GPU output
    local output="$1"
    local util_avg=0
    local mem_total=0
    local count=0
    while IFS=, read -r util mem; do
        util=$(echo "$util" | tr -d ' ')
        mem=$(echo "$mem" | tr -d ' ')
        util_avg=$((util_avg + util))
        mem_total=$((mem_total + mem))
        count=$((count + 1))
    done <<< "$output"
    if [[ $count -gt 0 ]]; then
        util_avg=$((util_avg / count))
        mem_total=$((mem_total / 1024))  # GB
        echo "$count GPUs, avg util ${util_avg}%, total mem ${mem_total}GB"
    else
        echo "unreachable"
    fi
}

check_process() {
    # Args: "local" or "remote"
    local mode=$1
    if [[ "$mode" == "local" ]]; then
        ps aux | grep -E "train.py kai0_mixed_1" | grep -v grep | awk '{print $2}' | wc -l
    else
        ssh -p $GF0_SSH_PORT -o ConnectTimeout=5 $GF0_HOST \
            "ps aux | grep -E 'train.py kai0_mixed_1' | grep -v grep | awk '{print \$2}' | wc -l" 2>/dev/null || echo "0"
    fi
}

print_status() {
    clear
    local now=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${CYAN}======================================================================${NC}"
    echo -e "${CYAN}  kai0_mixed_1 training monitor — $now${NC}"
    echo -e "${CYAN}  (refresh every ${REFRESH}s, press Ctrl+C to exit)${NC}"
    echo -e "${CYAN}======================================================================${NC}"

    # Process status
    echo ""
    echo -e "${BLUE}### Process Status ###${NC}"
    local gf1_proc=$(check_process local)
    local gf0_proc=$(check_process remote)

    local gf1_state
    if [[ "$gf1_proc" -gt 0 ]]; then gf1_state="${GREEN}✓ RUNNING ($gf1_proc procs)${NC}"
    else gf1_state="${YELLOW}⚠ NOT RUNNING${NC}"; fi

    local gf0_state
    if [[ "$gf0_proc" -gt 0 ]]; then gf0_state="${GREEN}✓ RUNNING ($gf0_proc procs)${NC}"
    else gf0_state="${YELLOW}⚠ NOT RUNNING${NC}"; fi

    printf "  %-8s %-50s\n" "gf1:" "$(echo -e $gf1_state)"
    printf "  %-8s %-50s\n" "gf0:" "$(echo -e $gf0_state)"

    # GPU utilization
    echo ""
    echo -e "${BLUE}### GPU Utilization ###${NC}"
    local gf1_gpu=$(get_gpu_summary local)
    local gf0_gpu=$(get_gpu_summary remote)
    printf "  %-8s %s\n" "gf1:" "$(format_gpu_summary "$gf1_gpu")"
    printf "  %-8s %s\n" "gf0:" "$(format_gpu_summary "$gf0_gpu")"

    # Training progress per split
    echo ""
    echo -e "${BLUE}### Split Training Progress ###${NC}"
    printf "  %-10s %-8s %-45s %s\n" "Split" "Machine" "Status" "Checkpoints"
    printf "  %-10s %-8s %-45s %s\n" "-----" "-------" "------" "-----------"

    for i in 0 1 2 3; do
        local machine
        local log
        if [[ " ${GF1_SPLITS[@]} " =~ " $i " ]]; then
            machine="gf1"
            log="$WORK_DIR/kai0_mixed_1_gf1.log"
        else
            machine="gf0"
            log="$WORK_DIR/kai0_mixed_1_gf0.log"
        fi

        local status=$(get_split_status "$i" "$log")
        local ckpts=$(get_checkpoint_steps "$i")

        # Color by status
        local color="$NC"
        case "$status" in
            DONE*) color="$GREEN" ;;
            RUNNING*) color="$CYAN" ;;
            FAILED*) color="$RED" ;;
            PENDING*) color="$YELLOW" ;;
        esac

        printf "  %-10s %-8s ${color}%-45s${NC} %s\n" "split_$i" "$machine" "$status" "${ckpts:-none}"
    done

    # Latest loss values
    echo ""
    echo -e "${BLUE}### Latest Loss (last 3 per active split) ###${NC}"
    for i in 0 1 2 3; do
        local log
        if [[ " ${GF1_SPLITS[@]} " =~ " $i " ]]; then
            log="$WORK_DIR/kai0_mixed_1_gf1.log"
        else
            log="$WORK_DIR/kai0_mixed_1_gf0.log"
        fi

        # Only show if currently running or last done split
        if grep -q "START split_$i" "$log" 2>/dev/null && ! grep -q "END split_$i" "$log" 2>/dev/null; then
            local losses=$(grep "^Step " "$log" 2>/dev/null | tail -3)
            if [[ -n "$losses" ]]; then
                echo -e "  ${CYAN}split_$i:${NC}"
                echo "$losses" | sed 's/^/    /'
            fi
        fi
    done

    # ETA estimate
    echo ""
    echo -e "${BLUE}### ETA Estimate ###${NC}"
    # Count completed + find any running progress
    local done_count=0
    local running_progress=""
    for i in 0 1 2 3; do
        local log
        if [[ " ${GF1_SPLITS[@]} " =~ " $i " ]]; then
            log="$WORK_DIR/kai0_mixed_1_gf1.log"
        else
            log="$WORK_DIR/kai0_mixed_1_gf0.log"
        fi
        if grep -q "END split_$i" "$log" 2>/dev/null; then
            done_count=$((done_count + 1))
        fi
    done
    # Get latest Progress line with remaining time
    local latest_progress=$(cat "$WORK_DIR"/kai0_mixed_1_gf{0,1}.log 2>/dev/null | \
                            grep -oE "remaining:[0-9:]+" | tail -1 | sed 's/remaining://')
    echo "  Completed splits: $done_count / 4"
    if [[ -n "$latest_progress" ]]; then
        echo "  Current split remaining: $latest_progress"
    fi

    # Errors
    echo ""
    local errors=$(grep -h -iE "FAILED|Traceback|RESOURCE_EXHAUSTED|CUDA error|AttributeError" "$WORK_DIR"/kai0_mixed_1_gf{0,1}.log 2>/dev/null | \
                   grep -v "tpu\|orbax\|pynvml\|deprecat\|torchcodec\|Failed to get flag" | tail -3)
    if [[ -n "$errors" ]]; then
        echo -e "${RED}### Recent Errors ###${NC}"
        echo "$errors" | sed 's/^/  /'
    else
        echo -e "${GREEN}### No errors detected ###${NC}"
    fi

    echo ""
    echo -e "${CYAN}----------------------------------------------------------------------${NC}"
    echo "Next refresh in ${REFRESH}s at $(date -d "+${REFRESH} seconds" '+%H:%M:%S' 2>/dev/null || date)"
}

# One-shot mode
if [[ "$ONCE" == "true" ]]; then
    print_status
    exit 0
fi

# Trap Ctrl+C
trap 'echo -e "\n${YELLOW}Monitor stopped.${NC}"; exit 0' INT

# Main loop
while true; do
    print_status
    sleep $REFRESH
done
