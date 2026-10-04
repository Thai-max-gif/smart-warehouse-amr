"""
Môi trường Bản đồ Nhà kho Thông minh (Smart Warehouse Environment).
Bao gồm các dãy kệ hàng (Racks), Trạm sạc (Charging Dock) và Vật cản động (Dynamic Obstacle).
"""
import math
from typing import List, Tuple


class WarehouseMap:
    """
    Bản đồ nhà kho 2D hệ tọa độ Descartes (mét).
    Gốc (0, 0) ở góc dưới bên trái, (width, height) ở góc trên bên phải.
    """

    def __init__(self, width: float = 12.0, height: float = 12.0, resolution: float = 0.1) -> None:
        self.width = float(width)
        self.height = float(height)
        self.resolution = float(resolution)

        # Danh sách các kệ hàng lưu trữ: (x, y, width, height)
        # Các dãy kệ song song tạo nên các hành lang di chuyển tiêu chuẩn
        self.racks: List[Tuple[float, float, float, float]] = [
            # Cụm kệ hàng phía dưới
            (2.0, 2.0, 1.2, 3.2),
            (4.8, 2.0, 1.2, 3.2),
            (7.6, 2.0, 1.2, 3.2),
            # Cụm kệ hàng phía trên
            (2.0, 6.8, 1.2, 3.2),
            (4.8, 6.8, 1.2, 3.2),
            (7.6, 6.8, 1.2, 3.2),
        ]

        # Trạm sạc tự động (Auto-docking station): (x, y, theta_facing)
        # Nằm ở góc trên bên phải (10.5m, 10.5m), mặt dock hướng về phía Tây-Nam (-3pi/4 hoặc -pi)
        self.dock_station: Tuple[float, float, float] = (10.5, 10.5, -math.pi * 0.75)
        self.dock_marker_id: int = 0
        self.dock_marker_size: float = 0.20  # 20cm

        # Vật cản di động (Người đi bộ / xe nâng): x, y, radius, time
        self.obs_center_x = 6.2
        self.obs_center_y = 5.8
        self.obs_radius = 0.25
        self.obs_time = 0.0
        self.obs_speed = 0.6  # rad/s cho quỹ đạo tuần tra

    def get_dynamic_obstacle_pose(self) -> Tuple[float, float, float]:
        """Trả về (x, y, radius) của vật cản di động tại thời điểm hiện tại."""
        # Di chuyển tuần tra qua lại hành lang ngang chính giữa
        x = self.obs_center_x + 1.8 * math.sin(self.obs_speed * self.obs_time)
        y = self.obs_center_y + 0.3 * math.cos(self.obs_speed * self.obs_time)
        return x, y, self.obs_radius

    def update_dynamic_obstacle(self, dt: float) -> Tuple[float, float, float]:
        """Cập nhật vị trí vật cản di động sau bước thời gian dt."""
        self.obs_time += dt
        return self.get_dynamic_obstacle_pose()

    def is_occupied(self, x: float, y: float, margin: float = 0.0) -> bool:
        """
        Kiểm tra điểm (x, y) có va chạm với tường biên, kệ hàng hoặc vùng an toàn (margin) hay không.
        """
        # Kiểm tra vượt biên nhà kho
        if x < margin or x > (self.width - margin) or y < margin or y > (self.height - margin):
            return True

        # Kiểm tra va chạm với từng kệ hàng
        for rx, ry, rw, rh in self.racks:
            if (rx - margin) <= x <= (rx + rw + margin) and (ry - margin) <= y <= (ry + rh + margin):
                return True

        return False
