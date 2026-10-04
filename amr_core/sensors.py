"""
Hệ thống cảm biến ảo (Virtual Sensors):
- VirtualLidar: Mô phỏng LiDAR 2D 360 độ (Raycasting).
- VirtualCamera: Mô phỏng Camera FPV RGB (FOV 60 độ, độ phân giải 640x480).
"""
import math
from typing import List, Tuple
import cv2
import numpy as np
from amr_core.environment import WarehouseMap


def normalize_angle(angle: float) -> float:
    """Chuẩn hóa góc về (-pi, pi]."""
    a = (angle + math.pi) % (2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


class VirtualLidar:
    """
    Mô phỏng cảm biến LiDAR 2D quét xung quanh robot.
    """

    def __init__(self, num_beams: int = 72, max_range: float = 4.0) -> None:
        self.num_beams = int(num_beams)
        self.max_range = float(max_range)
        self.angles = np.linspace(-math.pi, math.pi, num_beams, endpoint=False)

    def scan(
        self,
        robot_pose: Tuple[float, float, float],
        env: WarehouseMap,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Bắn các tia laser để đo khoảng cách tới kệ hàng, tường và vật cản động.
        """
        rx, ry, rtheta = robot_pose
        ranges = np.full(self.num_beams, self.max_range, dtype=np.float32)
        hit_points: List[Tuple[float, float]] = []

        # Tọa độ vật cản di động
        ox, oy, o_rad = env.get_dynamic_obstacle_pose()

        step_size = 0.05
        max_steps = int(self.max_range / step_size)

        for i, beam_angle in enumerate(self.angles):
            global_angle = normalize_angle(rtheta + beam_angle)
            cos_a = math.cos(global_angle)
            sin_a = math.sin(global_angle)

            hit_dist = self.max_range
            for step in range(1, max_steps + 1):
                dist = step * step_size
                px = rx + dist * cos_a
                py = ry + dist * sin_a

                # 1. Kiểm tra va chạm với tường và kệ hàng
                if env.is_occupied(px, py, margin=0.0):
                    hit_dist = dist
                    break

                # 2. Kiểm tra va chạm với vật cản di động (hình tròn)
                if math.hypot(px - ox, py - oy) <= o_rad:
                    hit_dist = dist
                    break

            ranges[i] = hit_dist
            if hit_dist < self.max_range:
                hx = rx + hit_dist * cos_a
                hy = ry + hit_dist * sin_a
                hit_points.append((hx, hy))

        return ranges, hit_points


class VirtualCamera:
    """
    Mô phỏng Camera FPV gắn trước đầu xe AMR nhìn ra thế giới.
    """

    def __init__(self, width: int = 640, height: int = 480, fov: float = 60.0) -> None:
        self.width = width
        self.height = height
        self.fov_rad = math.radians(fov)
        self.focal_length = (width / 2.0) / math.tan(self.fov_rad / 2.0)
        self.cx = width / 2.0
        self.cy = height / 2.0

        # Tạo mẫu ảnh thẻ ArUco Dict 4x4
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.marker_img = cv2.aruco.generateImageMarker(self.aruco_dict, id=0, sidePixels=120)
        except Exception:
            # Fallback nếu phiên bản OpenCV cũ
            self.marker_img = np.zeros((120, 120), dtype=np.uint8)
            cv2.rectangle(self.marker_img, (20, 20), (100, 100), 255, -1)

    def render(
        self,
        robot_pose: Tuple[float, float, float],
        env: WarehouseMap,
    ) -> np.ndarray:
        """
        Render khung ảnh FPV RGB thời gian thực từ vị trí của robot.
        """
        rx, ry, rtheta = robot_pose

        # Nền ảnh: Sàn nhà kho màu xám nhạt và tường màu tối
        frame = np.full((self.height, self.width, 3), 220, dtype=np.uint8)
        # Nửa trên là tường, nửa dưới là sàn
        cv2.rectangle(frame, (0, 0), (self.width, int(self.height * 0.45)), (140, 150, 160), -1)
        # Kẻ vạch sàn nhà kho
        cv2.line(frame, (0, int(self.height * 0.45)), (self.width, int(self.height * 0.45)), (80, 90, 100), 2)

        # 1. Kiểm tra Dock Station có trong tầm nhìn không
        dock_x, dock_y, dock_theta = env.dock_station
        dx = dock_x - rx
        dy = dock_y - ry
        dist_dock = math.hypot(dx, dy)
        angle_to_dock = math.atan2(dy, dx)
        rel_angle = normalize_angle(angle_to_dock - rtheta)

        # Nếu dock nằm trong góc nhìn FOV và khoảng cách < 4m
        if abs(rel_angle) < (self.fov_rad / 2.0) and 0.2 < dist_dock < 4.5:
            # Chiếu tọa độ ngang lên màn hình camera: góc dương (bên trái) chiếu sang x < cx
            screen_x = int(self.cx - math.tan(rel_angle) * self.focal_length)


            # Tính kích thước điểm ảnh theo đúng mô hình camera lỗ kim: W_px = (f * W_real) / Z
            marker_size_real = env.dock_marker_size  # 0.20m
            marker_px = int((self.focal_length * marker_size_real) / max(0.15, dist_dock))
            marker_px = max(16, min(int(self.height * 0.8), marker_px))
            scale = marker_px / 120.0

            marker_w = marker_px
            marker_h = marker_px
            dock_box_w = int(marker_px * 1.8)
            dock_box_h = int(marker_px * 1.5)

            # Căn chỉnh vị trí trạm sạc trên ảnh để không bị tràn viền
            base_y = int(self.height * 0.52)

            # Vẽ thân trạm sạc Docking Station màu vàng công nghiệp
            x1 = max(0, screen_x - dock_box_w // 2)
            y1 = max(0, base_y - dock_box_h // 2)
            x2 = min(self.width, screen_x + dock_box_w // 2)
            y2 = min(self.height, base_y + dock_box_h // 2)

            if x1 < x2 and y1 < y2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 180, 240), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), 2)

                # Dán thẻ ArUco lên trạm sạc với viền trắng an toàn (Quiet Zone)
                resized_marker = cv2.resize(self.marker_img, (marker_w, marker_h), interpolation=cv2.INTER_NEAREST)
                pad = max(4, int(marker_w * 0.12))
                padded_marker = cv2.copyMakeBorder(resized_marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
                marker_bgr = cv2.cvtColor(padded_marker, cv2.COLOR_GRAY2BGR)

                full_w = marker_w + 2 * pad
                full_h = marker_h + 2 * pad

                mx1 = screen_x - full_w // 2
                my1 = base_y - full_h // 2
                mx2 = mx1 + full_w
                my2 = my1 + full_h

                # Dán với kiểm tra biên an toàn (slicing)
                src_x1 = max(0, -mx1)
                src_y1 = max(0, -my1)
                src_x2 = full_w - max(0, mx2 - self.width)
                src_y2 = full_h - max(0, my2 - self.height)

                dst_x1 = max(0, mx1)
                dst_y1 = max(0, my1)
                dst_x2 = min(self.width, mx2)
                dst_y2 = min(self.height, my2)

                if dst_x1 < dst_x2 and dst_y1 < dst_y2 and src_x1 < src_x2 and src_y1 < src_y2:
                    frame[dst_y1:dst_y2, dst_x1:dst_x2] = marker_bgr[src_y1:src_y2, src_x1:src_x2]



                # Nhãn dock sạc
                cv2.putText(
                    frame,
                    "CHARGING DOCK #0",
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale,
                    (20, 20, 20),
                    2,
                )

        # 2. Kiểm tra Vật cản động (Công nhân kho áo cam phản quang)
        ox, oy, _ = env.get_dynamic_obstacle_pose()
        odx = ox - rx
        ody = oy - ry
        dist_obs = math.hypot(odx, ody)
        rel_obs_angle = normalize_angle(math.atan2(ody, odx) - rtheta)

        if abs(rel_obs_angle) < (self.fov_rad / 2.0) and 0.3 < dist_obs < 4.0:
            obs_screen_x = int(self.cx - math.tan(rel_obs_angle) * self.focal_length)
            obs_scale = max(0.2, min(1.5, 1.0 / dist_obs))

            pw = int(70 * obs_scale)
            ph = int(180 * obs_scale)
            py = int(self.height * 0.45 + 80 * obs_scale)

            px1 = obs_screen_x - pw // 2
            py1 = py - ph
            px2 = obs_screen_x + pw // 2
            py2 = py

            if 0 <= px1 < self.width or 0 <= px2 < self.width:
                # Vẽ hình dáng người công nhân mặc áo phản quang màu cam neon
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 140, 255), -1)  # Áo cam
                # Vẽ dải phản quang vàng
                cv2.rectangle(frame, (px1, py1 + int(ph * 0.4)), (px2, py1 + int(ph * 0.55)), (50, 230, 230), -1)
                # Vẽ đầu nón bảo hộ trắng
                head_r = int(pw * 0.35)
                cv2.circle(frame, (obs_screen_x, py1 - head_r), head_r, (255, 255, 255), -1)

        return frame
