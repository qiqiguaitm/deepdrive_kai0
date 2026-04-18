#!/bin/bash
# CAN 接口完整配置流程
#
# 用法:
#   bash can_tools/setup_can.sh                 # 默认: 仅做映射校准 (不写角色)
#   bash can_tools/setup_can.sh --setup-roles   # 先 HITL 写角色, 再校准/保存
#   bash can_tools/setup_can.sh --roles-only    # 只写角色, 不做映射校准
#   bash can_tools/setup_can.sh --quick         # 跳过所有交互, 直接用已有映射激活
#   bash can_tools/setup_can.sh --no-roles      # 历史别名, 等价于默认 (兼容旧脚本)
#
# 默认流程 (无参数, 不改固件):
#   Step 1: 扫描所有 CAN 适配器端口
#   Step 2: 激活所有接口 (临时名 canX)
#   Step 3: 映射校准 Phase 1-3 (分类 slave/master + 区分左右)
#   Step 4: 按校准结果重命名为符号名并激活
#   Step 5: 交互式校验映射 (verify_can_mapping.py)
#
# 常见场景:
#   - 日常 (USB 重插/重启后): 默认 (只校准, 不动固件)
#   - 已确认映射正确, 只需激活: --quick
#   - 新臂/换臂/reset 后重配角色: --setup-roles (HITL 4 步按 master/slave/左/右 提示)
#   - 只想改角色不重校准: --roles-only
#
# 设计原则: 默认流程不写入臂固件 (role writes 只在显式 --setup-roles / --roles-only 时做).

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BITRATE=1000000

QUICK=false
SETUP_ROLES=false      # 默认不写角色; --setup-roles 或 --roles-only 才开启
ROLES_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --quick)
            QUICK=true
            ;;
        --setup-roles)
            SETUP_ROLES=true
            ;;
        --no-roles)
            # 历史别名, 现在是默认行为; 保留兼容, 显式指定不报错
            SETUP_ROLES=false
            ;;
        --roles-only)
            ROLES_ONLY=true
            SETUP_ROLES=true
            ;;
        -h|--help)
            sed -n '1,27p' "$0"
            exit 0
            ;;
        *)
            echo "[FAIL] 未知参数: $arg"
            echo "  支持: --setup-roles, --roles-only, --quick, --no-roles, -h/--help"
            exit 1
            ;;
    esac
done

if $QUICK && $SETUP_ROLES; then
    echo "[FAIL] --quick 与 --setup-roles/--roles-only 不能同时使用"
    exit 1
fi

if [[ $(id -u) -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
    # 预先获取 sudo 权限
    echo "需要 sudo 权限操作 CAN 接口..."
    sudo -v || { echo "[FAIL] 无法获取 sudo 权限"; exit 1; }
fi

if $ROLES_ONLY; then
    MODE_DESC="仅角色写入 (跳过映射校准)"
elif $QUICK; then
    MODE_DESC="快速 (使用已有映射, 不交互)"
elif $SETUP_ROLES; then
    MODE_DESC="HITL 角色写入 + 映射校准"
else
    MODE_DESC="默认: 仅映射校准 (不动固件)"
fi

echo "============================================"
echo "  CAN 接口配置"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  模式: $MODE_DESC"
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
if $SETUP_ROLES; then
    echo "先写入 master/slave 角色, 再进行映射校准。"
else
    echo "接下来依次校准每个臂的映射关系。"
    echo "请在提示时晃动对应的臂, 脚本会自动检测。"
fi
echo ""

# 构建 calibrate_can_mapping.py 的参数
CAL_ARGS=()
if $ROLES_ONLY; then
    CAL_ARGS+=("--roles-only")
elif $SETUP_ROLES; then
    CAL_ARGS+=("--setup-roles")
fi

# 调用 calibrate_can_mapping.py (它会做校准 + 保存配置 + 最终重命名)
python3 "$SCRIPT_DIR/calibrate_can_mapping.py" "${CAL_ARGS[@]}"
CALIBRATE_EXIT=$?

if [[ $CALIBRATE_EXIT -ne 0 ]]; then
    echo "[FAIL] 校准失败"
    exit 1
fi

# --roles-only 不做映射, 直接退出
if $ROLES_ONLY; then
    echo ""
    echo "============================================"
    echo "  角色写入完成!"
    echo ""
    echo "  后续: 给切换过角色的臂断电重启, 然后运行:"
    echo "    bash can_tools/setup_can.sh           # 正常映射校准"
    echo "============================================"
    exit 0
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
echo "    bash can_tools/setup_can.sh                 # 日常: 仅校准映射 (默认, 不动固件)"
echo "    bash can_tools/setup_can.sh --quick         # 最快: 用已有映射直接激活"
echo "    bash can_tools/setup_can.sh --setup-roles   # 重配 master/slave 角色 + 校准"
echo "    bash can_tools/setup_can.sh --roles-only    # 仅改角色, 不校准"
echo "    bash start_scripts/start_teleop.sh          # 启动遥操"
echo "============================================"
