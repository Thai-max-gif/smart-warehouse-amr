import cv2
import numpy as np
import pytest
from vision_ai.aruco_estimator import ArUcoPoseEstimator
from vision_ai.obstacle_detector import ObstacleDetector


def test_aruco_pose_estimation():
    estimator = ArUcoPoseEstimator(marker_size=0.20)

    # 1. Kiểm tra ảnh trắng không có thẻ
    blank_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    t_vec, r_vec, detected, vis_frame = estimator.estimate_pose(blank_frame)
    assert detected is False
    assert t_vec is None
    assert r_vec is None

    # 2. Tạo ảnh chứa thẻ ArUco Dict 4x4 ID=0
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(aruco_dict, id=0, sidePixels=160)
    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    test_frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    # Dán marker vào giữa ảnh
    test_frame[160:320, 240:400] = marker_bgr

    t_vec, r_vec, detected, vis_frame = estimator.estimate_pose(test_frame)
    assert detected is True
    assert t_vec is not None
    assert r_vec is not None
    assert len(t_vec) == 3
    # Khoảng cách phía trước t_z phải là số dương
    assert t_vec[2] > 0.0


def test_docking_velocity_control():
    estimator = ArUcoPoseEstimator(marker_size=0.20)

    # Robot ở cách 1.5m, hơi lệch sang phải 0.1m
    t_vec = np.array([0.1, 0.0, 1.5])
    r_vec = np.array([0.0, 0.05, 0.0])

    v, w, is_docked = estimator.compute_docking_velocity(t_vec, r_vec)
    assert is_docked is False
    assert v > 0.0
    # Cần bẻ lái để bù trừ độ lệch
    assert abs(w) > 0.0

    # Robot đã vào khít sát dock sạc (t_z < 0.15m, t_x < 0.03m)
    t_docked = np.array([0.01, 0.0, 0.12])
    v_d, w_d, is_docked_flag = estimator.compute_docking_velocity(t_docked, np.zeros(3))
    assert is_docked_flag is True
    assert v_d == 0.0
    assert w_d == 0.0


def test_obstacle_detector():
    detector = ObstacleDetector()

    # Tạo ảnh có hình người áo cam
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    # Vẽ dải màu cam phản quang của áo công nhân
    cv2.rectangle(frame, (280, 200), (360, 380), (0, 140, 255), -1)

    detections, latency_ms, annotated_frame = detector.detect(frame)

    assert isinstance(detections, list)
    assert latency_ms > 0.0
    assert annotated_frame.shape == (480, 640, 3)
    assert len(detections) >= 1
    assert detections[0]["label"] == "person"
