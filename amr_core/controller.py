"""
Bộ điều khiển chuyển động Robot AMR và mô hình động học 2 bánh vi sai.
"""
import math
from typing import Tuple


def normalize_angle(angle: float) -> float:
    a = (angle + math.pi) % (2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


class AMRController:
    """
    Bộ điều khiển tốc độ bám mốc tọa độ và bẻ lái theo góc hướng mong muốn.
    """

    def __init__(
        self,
        max_v: float = 0.40,
        max_w: float = 1.50,
        wheel_radius: float = 0.05,
        wheel_base: float = 0.25,
    ) -> None:
        self.max_v = float(max_v)
        self.max_w = float(max_w)
        self.r = float(wheel_radius)
        self.L = float(wheel_base)

    def compute_velocity(
        self,
        robot_pose: Tuple[float, float, float],
        target_point: Tuple[float, float],
        desired_heading: float | None = None,
    ) -> Tuple[float, float]:
        """
        Tính toán vận tốc dài (v) và vận tốc góc (w).
        """
        rx, ry, rtheta = robot_pose
        tx, ty = target_point

        dist = math.hypot(tx - rx, ty - ry)

        if desired_heading is None:
            desired_heading = math.atan2(ty - ry, tx - rx)

        alpha = normalize_angle(desired_heading - rtheta)

        # Giảm tốc độ tiến thẳng nếu góc lệch còn lớn
        alignment = max(0.0, math.cos(alpha))
        v = min(self.max_v * alignment, max(0.05, dist * 0.8))

        # Điều khiển góc
        w = max(-self.max_w, min(self.max_w, 2.5 * alpha))

        return v, w

    def update_pose(
        self,
        robot_pose: Tuple[float, float, float],
        v: float,
        w: float,
        dt: float,
    ) -> Tuple[float, float, float]:
        """
        Tích phân động học vi sai cập nhật vị trí mới.
        """
        rx, ry, rtheta = robot_pose
        new_x = rx + v * math.cos(rtheta) * dt
        new_y = ry + v * math.sin(rtheta) * dt
        new_theta = normalize_angle(rtheta + w * dt)
        return new_x, new_y, new_theta
