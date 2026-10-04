"""
Module ước lượng tư thế 3D ArUco 6-DoF (ArUco 6-DoF Pose Estimator).
Sử dụng giải thuật Perspective-n-Point (solvePnP) và điều khiển Visual Servoing Auto-Docking.
"""
import math
from typing import Tuple
import cv2
import numpy as np


class ArUcoPoseEstimator:
    """
    Ước lượng vị trí 3D (X, Y, Z) và góc xoay của Trạm sạc dựa trên mã ArUco.
    """

    def __init__(self, marker_size: float = 0.20, width: int = 640, height: int = 480, fov: float = 60.0) -> None:
        self.marker_size = float(marker_size)
        self.width = width
        self.height = height

        # Ma trận nội suy camera K
        fov_rad = math.radians(fov)
        f = (width / 2.0) / math.tan(fov_rad / 2.0)
        self.camera_matrix = np.array([
            [f, 0, width / 2.0],
            [0, f, height / 2.0],
            [0, 0, 1.0],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # Định nghĩa 4 đỉnh 3D của thẻ ArUco trong hệ quy chiếu Marker (gốc ở tâm)
        half_s = marker_size / 2.0
        self.obj_points = np.array([
            [-half_s, half_s, 0],
            [half_s, half_s, 0],
            [half_s, -half_s, 0],
            [-half_s, -half_s, 0],
        ], dtype=np.float64)

        # Khởi tạo ArUco Dictionary và Detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.detector_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            self._use_new_api = True
        except AttributeError:
            self.detector_params = cv2.aruco.DetectorParameters_create()
            self._use_new_api = False

    def estimate_pose(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray | None, np.ndarray | None, bool, np.ndarray]:
        """
        Phát hiện thẻ ArUco và giải phương trình PnP để tính t_vec và r_vec.

        Trả về:
            (t_vec, r_vec, detected_flag, annotated_image)
        """
        vis_img = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Phát hiện góc marker
        if self._use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.detector_params)

        if ids is None or len(ids) == 0:
            return None, None, False, vis_img

        # Vẽ viền thẻ ArUco
        cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)

        # Giải bài toán solvePnP cho marker đầu tiên (ID=0)
        img_points = corners[0].reshape((4, 2))
        success, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            img_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not success:
            return None, None, False, vis_img

        t_vec = tvec.flatten()
        r_vec = rvec.flatten()

        # Vẽ 3 trục tọa độ không gian 3D (X-đỏ, Y-lục, Z-lam) lên marker
        axis_length = self.marker_size * 1.2
        cv2.drawFrameAxes(vis_img, self.camera_matrix, self.dist_coeffs, rvec, tvec, axis_length, 2)

        # Hiển thị thông số 3D lên góc ảnh
        info_text = f"DOCK 3D: X={t_vec[0]:+5.2f}m, Y={t_vec[1]:+5.2f}m, Z={t_vec[2]:5.2f}m"
        cv2.putText(vis_img, info_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        return t_vec, r_vec, True, vis_img

    def compute_docking_velocity(
        self,
        t_vec: np.ndarray,
        r_vec: np.ndarray,
    ) -> Tuple[float, float, bool]:
        """
        Bộ điều khiển bám thị giác (Visual Servoing) căn chỉnh robot vào dock sạc.

        Trả về:
            (v, w, is_docked)
        """
        tx, ty, tz = float(t_vec[0]), float(t_vec[1]), float(t_vec[2])

        # Tiêu chuẩn cập bến thành công: camera cách dock < 45cm và lệch ngang < 10cm (tiếp xúc cắm sạc)
        if tz < 0.45 and abs(tx) < 0.10:
            return 0.0, 0.0, True



        # Vận tốc tiến tỷ lệ thuận với khoảng cách Z (giảm dần khi tới gần)
        v = min(0.20, max(0.04, 0.25 * tz))

        # Vận tốc góc bẻ lái để triệt tiêu độ lệch ngang tx
        w = max(-0.8, min(0.8, -1.8 * tx))

        return v, w, False

