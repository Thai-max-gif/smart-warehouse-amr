"""
Thuật toán né vật cản thời gian thực Lực thế nhân tạo (Artificial Potential Field - APF).
Tự động tạo lực đẩy khi LiDAR phát hiện có người hoặc vật cản cắt ngang đường đi.
"""
import math
from typing import List, Tuple


class PotentialFieldAvoidance:
    """
    Bộ phản xạ né vật cản cục bộ (Local Reactive Planner) dựa trên trường thế năng.
    """

    def __init__(self, k_att: float = 1.0, k_rep: float = 1.2, d0: float = 0.9) -> None:
        self.k_att = float(k_att)
        self.k_rep = float(k_rep)
        self.d0 = float(d0)  # Bán kính vùng cảnh báo nguy hiểm (m)

    def compute_steering(
        self,
        robot_pose: Tuple[float, float, float],
        target_waypoint: Tuple[float, float],
        lidar_points: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """
        Tính toán góc hướng mong muốn (theta_desired) và độ lớn lực tổng hợp.

        Trả về:
            (theta_desired, force_magnitude)
        """
        rx, ry, rtheta = robot_pose
        tx, ty = target_waypoint

        # 1. Lực hút kéo về điểm mốc mục tiêu
        dx = tx - rx
        dy = ty - ry
        dist_target = math.hypot(dx, dy)

        if dist_target > 1e-4:
            f_att_x = self.k_att * (dx / dist_target)
            f_att_y = self.k_att * (dy / dist_target)
        else:
            f_att_x = 0.0
            f_att_y = 0.0

        # 2. Lực đẩy từ các điểm va chạm của chùm tia LiDAR
        f_rep_x = 0.0
        f_rep_y = 0.0

        for ox, oy in lidar_points:
            d_obs = math.hypot(rx - ox, ry - oy)
            if 0.05 < d_obs < self.d0:
                # Lực đẩy tỷ lệ nghịch với bình phương khoảng cách
                rep_mag = self.k_rep * (1.0 / d_obs - 1.0 / self.d0) * (1.0 / (d_obs * d_obs))
                # Hướng từ vật cản đẩy ra xa phía robot
                rep_dir_x = (rx - ox) / d_obs
                rep_dir_y = (ry - oy) / d_obs

                f_rep_x += rep_mag * rep_dir_x
                f_rep_y += rep_mag * rep_dir_y

        # 3. Lực tổng hợp
        f_total_x = f_att_x + f_rep_x
        f_total_y = f_att_y + f_rep_y

        f_mag = math.hypot(f_total_x, f_total_y)
        theta_desired = math.atan2(f_total_y, f_total_x)

        return theta_desired, f_mag
