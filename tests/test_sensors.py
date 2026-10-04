import math
import numpy as np
import pytest
from amr_core.environment import WarehouseMap
from amr_core.sensors import VirtualLidar, VirtualCamera


def test_warehouse_map_occupancy():
    env = WarehouseMap(width=12.0, height=12.0)
    # Tâm của một kệ hàng mẫu chắc chắn bị chiếm dụng (Occupied)
    rack = env.racks[0]
    rx, ry, rw, rh = rack
    center_x = rx + rw / 2.0
    center_y = ry + rh / 2.0
    assert env.is_occupied(center_x, center_y) is True

    # Điểm giữa hành lang thoáng không bị chiếm dụng
    assert env.is_occupied(1.0, 1.0) is False

    # Điểm ngoài biên bản đồ phải tính là occupied
    assert env.is_occupied(-0.5, 5.0) is True
    assert env.is_occupied(12.5, 5.0) is True


def test_dynamic_obstacle_updates():
    env = WarehouseMap(width=12.0, height=12.0)
    initial_pos = env.get_dynamic_obstacle_pose()
    # Cập nhật thời gian 1.0 giây
    env.update_dynamic_obstacle(dt=1.0)
    new_pos = env.get_dynamic_obstacle_pose()
    # Tọa độ phải có sự di chuyển
    assert initial_pos != new_pos


def test_virtual_lidar():
    env = WarehouseMap(width=12.0, height=12.0)
    lidar = VirtualLidar(num_beams=72, max_range=4.0)

    # Robot đứng tại vị trí thoáng (1.0, 1.0), góc 0 rad
    ranges, hit_points = lidar.scan(robot_pose=(1.0, 1.0, 0.0), env=env)

    assert len(ranges) == 72
    assert isinstance(hit_points, list)
    # Mọi giá trị khoảng cách phải nằm trong [0, max_range]
    assert np.all(ranges >= 0.0)
    assert np.all(ranges <= 4.0 + 1e-4)


def test_virtual_camera_rendering():
    env = WarehouseMap(width=12.0, height=12.0)
    cam = VirtualCamera(width=640, height=480, fov=60.0)

    img = cam.render(robot_pose=(10.0, 10.0, math.pi / 4), env=env)

    assert isinstance(img, np.ndarray)
    assert img.shape == (480, 640, 3)
    assert img.dtype == np.uint8
