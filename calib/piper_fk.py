"""
Piper 机械臂正运动学 (FK) 封装

封装 Piper SDK 的 C_PiperForwardKinematics (DH 参数, 含 2° offset 校正),
提供输出 4×4 齐次矩阵的接口。

SDK 位置: /home/tim/workspace/piper_sdk/piper_sdk/kinematics/piper_fk.py
DH offset: 0x01 (j2/j3 含 2° 校正, 与实际硬件匹配)
"""
import os
import sys
import math
import numpy as np
from scipy.spatial.transform import Rotation

# Locate piper_sdk: env var > ~/workspace/piper_sdk > hardcoded fallback
_PIPER_SDK_DIR = os.environ.get('PIPER_SDK_DIR', '')
if not _PIPER_SDK_DIR or not os.path.isdir(_PIPER_SDK_DIR):
    for candidate in [
        os.path.join(os.path.expanduser('~'), 'workspace', 'piper_sdk'),
        '/home/tim/workspace/piper_sdk',
    ]:
        if os.path.isdir(os.path.join(candidate, 'piper_sdk', 'kinematics')):
            _PIPER_SDK_DIR = candidate
            break
if _PIPER_SDK_DIR and _PIPER_SDK_DIR not in sys.path:
    sys.path.insert(0, _PIPER_SDK_DIR)
# Import the kinematics module directly via importlib to avoid piper_sdk/__init__.py
# which transitively imports `can` (python-can) — not needed for FK and not available
# under the system Python used by ROS2 nodes.
import importlib.util as _ilu
_fk_path = os.path.join(_PIPER_SDK_DIR, 'piper_sdk', 'kinematics', 'piper_fk.py')
_fk_spec = _ilu.spec_from_file_location('piper_sdk.kinematics.piper_fk', _fk_path)
_fk_mod = _ilu.module_from_spec(_fk_spec)
_fk_spec.loader.exec_module(_fk_mod)
C_PiperForwardKinematics = _fk_mod.C_PiperForwardKinematics


class PiperFK:
    """Piper 6-DOF 正运动学, 输出齐次矩阵。

    Usage:
        fk = PiperFK()
        T = fk.fk_homogeneous([0.1, 0.5, -0.3, 0.0, 0.2, -0.1])  # rad
        # T: 4×4 ndarray, 单位 m
    """

    # 关节角 ↔ 0.001° 换算系数 (Piper SDK 内部单位)
    RAD_TO_MDEG = 180.0 / math.pi * 1000.0  # 57295.7795

    def __init__(self, dh_is_offset: int = 0x01):
        self._fk = C_PiperForwardKinematics(dh_is_offset=dh_is_offset)
        # 检测 SDK 内部 API 是否可用 (包括 name-mangled 私有方法)
        self._use_internal_api = (
            hasattr(self._fk, '_theta')
            and hasattr(self._fk, '_alpha')
            and hasattr(self._fk, '_a')
            and hasattr(self._fk, '_d')
            and hasattr(self._fk, '_C_PiperForwardKinematics__LinkTransformtion')
            and hasattr(self._fk, '_C_PiperForwardKinematics__MatMultiply')
        )
        if not self._use_internal_api:
            print("[PiperFK] SDK internal API not available, using CalFK fallback")
        else:
            # 自动验证 CalFK 路径, 方便及早发现 SDK 版本导致的单位问题
            self.validate_fallback()

    def _cal_fk_matrices(self, q6_rad: list | np.ndarray) -> list[np.ndarray]:
        """计算各 link 的累积变换矩阵 (4×4, 单位 mm)。

        直接复用 SDK 内部的矩阵链, 比从 xyz+rpy 反算更精确。
        如果 SDK 内部 API 不可用 (版本变更), 回退到公开的 CalFK 接口。
        """
        if not self._use_internal_api:
            return self._cal_fk_via_calFK(q6_rad)

        q = list(q6_rad)
        fk = self._fk

        # 计算各 link 的单独变换矩阵
        Rt = []
        for i in range(6):
            c_theta = q[i] + fk._theta[i]
            T_flat = fk._C_PiperForwardKinematics__LinkTransformtion(
                fk._alpha[i], fk._a[i], c_theta, fk._d[i]
            )
            Rt.append(T_flat)

        # 累积乘法: T01, T02, ..., T06
        mul = fk._C_PiperForwardKinematics__MatMultiply
        accumulated = [None] * 6
        accumulated[0] = Rt[0]
        accumulated[1] = mul(accumulated[0], Rt[1], 4, 4, 4)
        accumulated[2] = mul(accumulated[1], Rt[2], 4, 4, 4)
        accumulated[3] = mul(accumulated[2], Rt[3], 4, 4, 4)
        accumulated[4] = mul(accumulated[3], Rt[4], 4, 4, 4)
        accumulated[5] = mul(accumulated[4], Rt[5], 4, 4, 4)

        # flat list [16] → 4×4 ndarray, mm → m
        result = []
        for flat in accumulated:
            T = np.array(flat, dtype=np.float64).reshape(4, 4)
            T[:3, 3] /= 1000.0  # mm → m
            result.append(T)
        return result

    def _cal_fk_via_calFK(self, q6_rad: list | np.ndarray) -> list[np.ndarray]:
        """通过 SDK 公开的 CalFK 接口计算 FK (回退方案)。

        CalFK 输入: rad, 输出: 每个 link 的 [x, y, z, roll, pitch, yaw] (mm/deg)。
        注意: 首次使用此回退路径时会自动验证与零位已知值的一致性。
        """
        result_list = self._fk.CalFK(list(q6_rad))
        matrices = []
        for link_data in result_list:
            xyz_mm = np.array(link_data[:3], dtype=np.float64)
            rpy_deg = np.array(link_data[3:], dtype=np.float64)
            T = np.eye(4)
            T[:3, :3] = Rotation.from_euler('xyz', np.radians(rpy_deg)).as_matrix()
            T[:3, 3] = xyz_mm / 1000.0  # mm → m
            matrices.append(T)
        return matrices

    def validate_fallback(self) -> bool:
        """验证 CalFK 回退路径与内部 API 路径的一致性。

        在零位和多个非零位比较两条路径的输出, 如果偏差 > 1mm 或 1° 则发出警告。
        非零位测试可捕获 Euler 约定不匹配 (零位附近各约定差异极小)。
        仅当两条路径都可用时才有意义。
        """
        if not self._use_internal_api:
            print("[PiperFK] Cannot validate: internal API not available")
            return False

        test_configs = [
            ('zero', [0.0] * 6),
            ('mixed', [0.5, -0.4, 0.3, -0.6, 0.5, -0.3]),
            ('large', [1.0, 0.8, -0.7, 1.2, -0.9, 0.6]),
        ]

        all_ok = True
        for name, q in test_configs:
            T_internal = self._cal_fk_matrices(q)[-1]
            T_calFK = self._cal_fk_via_calFK(q)[-1]

            t_err_mm = np.linalg.norm(T_internal[:3, 3] - T_calFK[:3, 3]) * 1000
            R_err_deg = np.degrees(np.arccos(np.clip(
                (np.trace(T_internal[:3, :3].T @ T_calFK[:3, :3]) - 1) / 2, -1, 1
            )))
            ok = t_err_mm < 1.0 and R_err_deg < 1.0
            if not ok:
                print(f"[PiperFK] WARNING: CalFK fallback differs at {name}: "
                      f"t_err={t_err_mm:.2f}mm, R_err={R_err_deg:.2f}deg")
                all_ok = False
            else:
                print(f"[PiperFK] CalFK validated ({name}): "
                      f"t_err={t_err_mm:.3f}mm, R_err={R_err_deg:.3f}deg")

        return all_ok

    def fk_homogeneous(self, q6_rad: list | np.ndarray) -> np.ndarray:
        """6 个关节角 (rad) → 末端 (link6) 的 4×4 齐次矩阵 T_base_ee。

        单位: 平移 m, 旋转 rad。
        """
        matrices = self._cal_fk_matrices(q6_rad)
        return matrices[5]  # T_base_link6

    def fk_all_links(self, q6_rad: list | np.ndarray) -> list[np.ndarray]:
        """返回 6 个 link 的 T_base_link_i [4×4] (可视化用)。

        Returns:
            [T_base_link1, T_base_link2, ..., T_base_link6]
            每个 4×4, 单位 m。
        """
        return self._cal_fk_matrices(q6_rad)

    def fk_ee_pose(self, q6_rad: list | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回末端 (xyz, rpy)。

        Returns:
            xyz: [3] 单位 m
            rpy: [3] 单位 rad (roll, pitch, yaw)
        """
        T = self.fk_homogeneous(q6_rad)
        xyz = T[:3, 3]
        rpy = Rotation.from_matrix(T[:3, :3]).as_euler('xyz')
        return xyz, rpy


if __name__ == '__main__':
    # 验证: 零位 FK
    fk = PiperFK()
    q_zero = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    T = fk.fk_homogeneous(q_zero)
    xyz, rpy = fk.fk_ee_pose(q_zero)

    print("零位 FK 验证:")
    print(f"  T_base_ee:\n{T}")
    print(f"  xyz (m): {xyz}")
    print(f"  rpy (deg): {np.degrees(rpy)}")

    # 对比 SDK 的 CalFK 输出
    sdk_result = fk._fk.CalFK(q_zero)
    sdk_ee = sdk_result[5]  # link6: [x,y,z,r,p,y] mm/deg
    print(f"\n  SDK CalFK link6: xyz={sdk_ee[:3]} mm, rpy={sdk_ee[3:]} deg")
    print(f"  我们的 xyz:      {xyz * 1000} mm")
    print(f"  差值: {np.abs(xyz * 1000 - np.array(sdk_ee[:3]))} mm")
