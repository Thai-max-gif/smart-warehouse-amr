import math
import pytest
from amr_core.environment import WarehouseMap
from amr_core.planning import AStarPlanner
from amr_core.avoidance import PotentialFieldAvoidance
from amr_core.controller import AMRController


def test_astar_finds_valid_path():
    env = WarehouseMap(width=12.0, height=12.0)
    planner = AStarPlanner(env=env, safety_margin=0.25)

    start = (1.0, 1.0)
    goal = (10.0, 10.0)

    path = planner.plan(start, goal)

    assert isinstance(path, list)
    assert len(path) > 1

    # Điểm đầu gần start, điểm cuối gần goal
    assert math.hypot(path[0][0] - start[0], path[0][1] - start[1]) < 0.5
    assert math.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]) < 0.5

    # Đảm bảo không có điểm nào trong path va chạm vào kệ hàng
    for pt in path:
        assert env.is_occupied(pt[0], pt[1], margin=0.1) is False


def test_astar_unreachable_goal():
    env = WarehouseMap(width=12.0, height=12.0)
    planner = AStarPlanner(env=env, safety_margin=0.25)

    start = (1.0, 1.0)
    # Tâm của một kệ hàng chắc chắn bị chặn
    rack = env.racks[0]
    unreachable_goal = (rack[0] + rack[2] / 2.0, rack[1] + rack[3] / 2.0)

    path = planner.plan(start, unreachable_goal)
    assert path == []


def test_apf_avoidance():
    apf = PotentialFieldAvoidance(k_att=1.0, k_rep=1.5, d0=1.0)

    robot_pose = (0.0, 0.0, 0.0)
    target_wp = (3.0, 0.0)  # Đích nằm thẳng phía trước (trục X)

    # Không có vật cản: lực chỉ kéo thẳng về phía trước (góc 0 rad)
    steer_clear, force_mag = apf.compute_steering(robot_pose, target_wp, lidar_points=[])
    assert math.isclose(steer_clear, 0.0, abs_tol=1e-2)

    # Có vật cản nằm ngay phía trước bên phải (0.8, -0.2) nằm trong d0 = 1.0
    # Lực đẩy sẽ bẻ lái robot hướng sang trái (góc dương)
    lidar_points = [(0.8, -0.2)]
    steer_avoid, _ = apf.compute_steering(robot_pose, target_wp, lidar_points=lidar_points)
    assert steer_avoid > 0.05  # Bẻ lái sang bên ngược lại



def test_amr_controller():
    controller = AMRController(max_v=0.4, max_w=1.5)

    # Đích phía trước
    v, w = controller.compute_velocity(robot_pose=(0.0, 0.0, 0.0), target_point=(2.0, 0.0))
    assert 0.0 < v <= 0.4
    assert math.isclose(w, 0.0, abs_tol=1e-2)

    # Đích ở bên trái (góc pi/2)
    v, w = controller.compute_velocity(robot_pose=(0.0, 0.0, 0.0), target_point=(0.0, 2.0))
    assert w > 0.5
