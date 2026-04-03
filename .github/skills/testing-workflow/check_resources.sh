#!/bin/bash
# =============================================================================
# 资源检查脚本 - Testing Workflow Skill
# 用途：测试前检查 GPU/CPU/内存/磁盘资源是否充足
# 用法：bash .github/skills/testing-workflow/check_resources.sh
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

WARNINGS=0
ERRORS=0

echo "============================================"
echo "  资源检查 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

# --- GPU 检查 ---
echo "--- GPU 状态 ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu \
        --format=csv,noheader | while IFS=',' read -r idx name used total free util; do
        used_mb=$(echo "$used" | grep -oP '\d+')
        total_mb=$(echo "$total" | grep -oP '\d+')
        free_mb=$(echo "$free" | grep -oP '\d+')

        if [ "$free_mb" -lt 4000 ]; then
            echo -e "  GPU $idx ($name):${RED} 可用 ${free_mb}MB / ${total_mb}MB — 显存不足！${NC}"
            ((ERRORS++)) || true
        elif [ "$free_mb" -lt 8000 ]; then
            echo -e "  GPU $idx ($name):${YELLOW} 可用 ${free_mb}MB / ${total_mb}MB — 显存偏低${NC}"
            ((WARNINGS++)) || true
        else
            echo -e "  GPU $idx ($name):${GREEN} 可用 ${free_mb}MB / ${total_mb}MB${NC}"
        fi
    done

    # 检查是否有其他 GPU 进程
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_PROCS" ]; then
        echo -e "  ${YELLOW}检测到正在运行的 GPU 进程：${NC}"
        echo "$GPU_PROCS" | while read -r line; do
            echo "    $line"
        done
        ((WARNINGS++)) || true
    fi
else
    echo -e "  ${RED}nvidia-smi 不可用${NC}"
    ((ERRORS++)) || true
fi
echo ""

# --- 系统内存检查 ---
echo "--- 系统内存 ---"
AVAIL_MB=$(free -m | awk '/^Mem:/ {print $7}')
TOTAL_MB=$(free -m | awk '/^Mem:/ {print $2}')
if [ "$AVAIL_MB" -lt 4000 ]; then
    echo -e "  ${RED}可用内存: ${AVAIL_MB}MB / ${TOTAL_MB}MB — 不足！${NC}"
    ((ERRORS++)) || true
elif [ "$AVAIL_MB" -lt 8000 ]; then
    echo -e "  ${YELLOW}可用内存: ${AVAIL_MB}MB / ${TOTAL_MB}MB — 偏低${NC}"
    ((WARNINGS++)) || true
else
    echo -e "  ${GREEN}可用内存: ${AVAIL_MB}MB / ${TOTAL_MB}MB${NC}"
fi
echo ""

# --- CPU 负载检查 ---
echo "--- CPU 负载 ---"
LOAD_1MIN=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | xargs)
CPU_CORES=$(nproc)
LOAD_INT=$(echo "$LOAD_1MIN" | awk '{printf "%d", $1}')
if [ "$LOAD_INT" -gt "$CPU_CORES" ]; then
    echo -e "  ${YELLOW}1min 负载: ${LOAD_1MIN} (${CPU_CORES} 核) — 负载偏高${NC}"
    ((WARNINGS++)) || true
else
    echo -e "  ${GREEN}1min 负载: ${LOAD_1MIN} (${CPU_CORES} 核)${NC}"
fi
echo ""

# --- 磁盘空间检查 ---
echo "--- 磁盘空间 ---"
for mount_point in /data / ; do
    if df "$mount_point" &>/dev/null; then
        AVAIL_GB=$(df -BG "$mount_point" | awk 'NR==2 {gsub("G",""); print $4}')
        if [ "$AVAIL_GB" -lt 10 ]; then
            echo -e "  ${RED}${mount_point}: 可用 ${AVAIL_GB}GB — 空间不足！${NC}"
            ((ERRORS++)) || true
        elif [ "$AVAIL_GB" -lt 30 ]; then
            echo -e "  ${YELLOW}${mount_point}: 可用 ${AVAIL_GB}GB — 空间偏低${NC}"
            ((WARNINGS++)) || true
        else
            echo -e "  ${GREEN}${mount_point}: 可用 ${AVAIL_GB}GB${NC}"
        fi
    fi
done
echo ""

# --- 汇总 ---
echo "============================================"
if [ "$ERRORS" -gt 0 ]; then
    echo -e "  ${RED}✗ 发现 ${ERRORS} 个严重问题，${WARNINGS} 个警告${NC}"
    echo -e "  ${RED}建议：解决严重问题后再开始测试${NC}"
    exit 1
elif [ "$WARNINGS" -gt 0 ]; then
    echo -e "  ${YELLOW}⚠ 发现 ${WARNINGS} 个警告，无严重问题${NC}"
    echo -e "  ${YELLOW}建议：注意资源状态，谨慎进行大规模测试${NC}"
    exit 0
else
    echo -e "  ${GREEN}✓ 资源充足，可以开始测试${NC}"
    exit 0
fi
