#!/bin/bash
# CAN 接口完整配置流程
#
# 用法:
#   bash can_tools/setup_can.sh           # 完整流程 (扫描→激活→校准→重命名→校验)
#   bash can_tools/setup_can.sh --quick   # 跳过校准, 使用已有映射直接激活
#
# 流程:
#   Step 1: 扫描所有 CAN 适配器端口
#   Step 2: 激活所有接口 (临时名 canX)
#   Step 3: 交互式校准映射关系 (晃动臂检测)
#   Step 4: 按校准结果重命名为符号名并激活
#   Step 5: 交互式校验映射 (verify_can_mapping.py)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BITRATE=1000000

if [[ $(id -u) -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
    # 预先获取 sudo 权限
    echo "需要 sudo 权限操作 CAN 接口..."
    sudo -v || { echo "[FAIL] 无法获取 sudo 权限"; exit 1; }
fi

QUICK=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=true
fi

echo "============================================"
echo "  CAN 接口配置"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  模式: $(if $QUICK; then echo '快速 (使用已有映射)'; else echo '完整 (含校准)'; fi)"
echo "============================================"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: 扫描所有 CAN 适配器端口
# ═══════════════════════════════════════════════════════════════════════════════
echo "━━━ Step 1: 扫描 CAN 端口 ━━━"
echo ""

CAN_COUNT=$(ip -br link show type can 2>/dev/null | wc -l)
if [[ "$CAN_COUNT" -eq 0 ]]; then
    echo "[FAIL] 未检测到 CAN 接口, 请检查 USB-CAN 适配器连接"
    exit 1
fi

echo "检测到 $CAN_COUNT 个 CAN 接口:"
for iface in $(ip -br link show type can | awk '{print $1}'); do
    bus=$($SUDO ethtool -i "$iface" 2>/dev/null | grep "bus-info" | sed 's/.*bus-info: *//')
    state=$(ip -br link show "$iface" | awk '{print $2}')
    echo "  $iface  bus=$bus  state=$state"
done
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: 激活所有接口 (统一为 canX 临时名)
# ═══════════════════════════════════════════════════════════════════════════════
echo "━━━ Step 2: 激活所有接口 ━━━"
echo ""

# 先全部 down
for iface in $(ip -br link show type can | awk '{print $1}'); do
    $SUDO ip link set "$iface" down 2>/dev/null || true
done

# 先全部改临时名 (避免 canX 互相冲突)
TMP_IDX=0
declare -a TMP_NAMES=()
for iface in $(ip -br link show type can | awk '{print $1}'); do
    tmp="can_setup_${TMP_IDX}"
    $SUDO ip link set "$iface" name "$tmp" 2>/dev/null || true
    TMP_NAMES+=("$tmp")
    TMP_IDX=$((TMP_IDX + 1))
done

# 再从临时名改为 canX 并激活
IDX=0
for tmp in "${TMP_NAMES[@]}"; do
    target="can${IDX}"
    $SUDO ip link set "$tmp" name "$target" 2>/dev/null || true
    $SUDO ip link set "$target" type can bitrate "$BITRATE"
    $SUDO ip link set "$target" up
    IDX=$((IDX + 1))
done

sleep 1

echo "激活后状态:"
for iface in $(ip -br link show type can | sort | awk '{print $1}'); do
    bus=$($SUDO ethtool -i "$iface" 2>/dev/null | grep "bus-info" | sed 's/.*bus-info: *//')
    result=$(timeout 1 candump "$iface" -n 1 2>&1 || true)
    if [[ -n "$result" ]]; then
        rx="有数据 ✓"
    else
        rx="无数据 ✗"
    fi
    echo "  $iface  bus=$bus  $rx"
done
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: 校准映射关系
# ═══════════════════════════════════════════════════════════════════════════════
if $QUICK; then
    echo "━━━ Step 3: 跳过校准 (--quick) ━━━"
    echo ""
    echo "使用已有映射, 直接执行 Step 4..."
    echo ""

    # 直接用 activate_can.sh 的已有配置重命名
    # 先全部 down
    for iface in $(ip -br link show type can | awk '{print $1}'); do
        $SUDO ip link set "$iface" down 2>/dev/null || true
    done

    # 调用 activate_can.sh
    bash "$SCRIPT_DIR/activate_can.sh"
    exit 0
fi

echo "━━━ Step 3: 交互式校准 ━━━"
echo ""
echo "接下来依次校准每个臂的映射关系。"
echo "请在提示时晃动对应的臂, 脚本会自动检测。"
echo ""

# 调用 calibrate_can_mapping.py (它会做校准 + 保存配置 + 最终重命名)
python3 "$SCRIPT_DIR/calibrate_can_mapping.py"
CALIBRATE_EXIT=$?

if [[ $CALIBRATE_EXIT -ne 0 ]]; then
    echo "[FAIL] 校准失败"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: 按校准结果重命名并激活
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━ Step 4: 按校准结果重命名并激活 ━━━"
echo ""

# calibrate_can_mapping.py 已更新 activate_can.sh 的配置
# 先全部 down
for iface in $(ip -br link show type can | awk '{print $1}'); do
    $SUDO ip link set "$iface" down 2>/dev/null || true
done

# 用更新后的配置重命名
bash "$SCRIPT_DIR/activate_can.sh"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: 校验映射 (verify_can_mapping.py)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━ Step 5: 校验映射 ━━━"
echo ""
echo "接下来运行 verify_can_mapping.py, 依次晃动每个臂,"
echo "核对哪个接口标记 '<<< MOVING' 是否与预期符号名一致。"
echo "完成后按 Ctrl+C 退出校验。"
echo ""
read -r -p "按 Enter 开始校验, 或输入 s 跳过: " _ans
if [[ "${_ans:-}" != "s" && "${_ans:-}" != "S" ]]; then
    python3 "$SCRIPT_DIR/verify_can_mapping.py" || true
fi

echo ""
echo "============================================"
echo "  配置完成!"
echo ""
echo "  后续使用:"
echo "    bash can_tools/setup_can.sh --quick   # 快速激活 (已有映射)"
echo "    bash scripts/start_teleop.sh          # 启动遥操"
echo "============================================"
