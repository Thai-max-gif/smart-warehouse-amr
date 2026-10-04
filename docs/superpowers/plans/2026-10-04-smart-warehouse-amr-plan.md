# AI-Powered Smart Warehouse AMR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng hệ thống Robot AMR Nhà kho Thông minh hoàn chỉnh tích hợp AI thị giác (A* Global Planning, LiDAR & APF Reactive Avoidance, ArUco 6-DoF Precision Docking, YOLOv8 Obstacle Detection, Dual-view Dashboard 2D) cùng Báo cáo Kỹ thuật nộp Lab nghiên cứu.

**Architecture:** Kiến trúc phân tầng 5 lớp độc lập (Sensor -> Perception -> Planning -> Control -> Visualization): `amr_core` quản lý bản đồ môi trường, cảm biến ảo, quy hoạch đường đi A* và né vật cản APF; `vision_ai` quản lý các mô hình thị giác ArUco solvePnP và YOLOv8; `tests` đảm bảo chất lượng thuật toán qua Pytest; `main_simulation.py` điều phối giao diện 2 màn hình đồng bộ; `BAO_CAO_XIN_VAO_LAB.md` đóng vai trò hồ sơ nghiên cứu gửi giáo sư.

**Tech Stack:** Python 3.10+, NumPy, OpenCV (cv2), Matplotlib, Pytest, Torch / Ultralytics (tùy chọn GPU CUDA).

**Spec:** `D:\smart-warehouse-amr\docs\superpowers\specs\2026-10-04-smart-warehouse-amr-design.md`

## Global Constraints

- Hoạt động mượt mà 100% trên Windows 11 (không yêu cầu Linux/ROS 2).
- Nếu máy có GPU NVIDIA (như RTX 5050) và CUDA, tự động tận dụng CUDA; nếu chưa cài CUDA, tự động fallback sang CPU mà không gây crash chương trình.
- Cấu trúc module sạch (Clean Architecture), độ dài hàm < 50 dòng, hàm toán học có chú thích công thức rõ ràng.
- Toàn bộ bài test trong `tests/` phải đạt 100% pass với độ bao phủ cao.

## Review Focus

- Bản đồ lưới và thuật toán A*: Điểm đích nằm sát mép chướng ngại vật hoặc không có đường đi (unreachable goal) phải trả về danh sách rỗng hoặc thông báo lỗi tường minh, không rơi vào vòng lặp vô tận.
- Thuật toán APF né vật cản: Khi vật cản nằm ngay trước mặt cách < 0.3m, lực đẩy cực đại không làm bão hòa gây tràn số hoặc lật góc đột ngột.
- Ước lượng tư thế ArUco: Trường hợp không tìm thấy Marker trên ảnh phải trả về `None, None, False` an toàn, không ném ngoại lệ làm gián đoạn vòng lặp mô phỏng.
- Đồng bộ giao diện Matplotlib & OpenCV: Không bị rò rỉ bộ nhớ (Memory Leak) khi cập nhật vòng lặp đồ họa hàng trăm khung hình liên tục.
- File `.gitignore` phải chặn toàn bộ file trọng số mô hình lớn (`*.pt`, `*.onnx`), cache Python và môi trường ảo.

---

### Task 1: Khởi tạo Cấu trúc Dự án & Cấu hình Môi trường

**Files:**
- Create: `D:\smart-warehouse-amr\.gitignore`
- Create: `D:\smart-warehouse-amr\requirements.txt`
- Create: `D:\smart-warehouse-amr\amr_core/__init__.py`
- Create: `D:\smart-warehouse-amr\vision_ai/__init__.py`
- Create: `D:\smart-warehouse-amr\tests/__init__.py`

**Interfaces:**
- Produces: Môi trường cấu hình chuẩn với `numpy`, `matplotlib`, `opencv-python`, `pytest`.

- [ ] **Step 1: Tạo file `.gitignore`**
- [ ] **Step 2: Tạo file `requirements.txt`**
- [ ] **Step 3: Tạo các file package marker `__init__.py`**

---

### Task 2: Xây dựng Bản đồ Nhà kho & Cảm biến Ảo (`amr_core/environment.py`, `amr_core/sensors.py`)

**Files:**
- Create: `D:\smart-warehouse-amr\amr_core/environment.py`
- Create: `D:\smart-warehouse-amr\amr_core/sensors.py`
- Test: `D:\smart-warehouse-amr\tests/test_sensors.py`

**Interfaces:**
- Produces:
  - `class WarehouseMap(width: float = 12.0, height: float = 12.0, resolution: float = 0.1)`
    - `racks: list[tuple[float, float, float, float]]` (x, y, w, h)
    - `dock_station: tuple[float, float, float]` (x, y, theta)
    - `is_occupied(x: float, y: float, margin: float = 0.0) -> bool`
    - `update_dynamic_obstacle(dt: float) -> tuple[float, float]`
  - `class VirtualLidar(num_beams: int = 72, max_range: float = 4.0)`
    - `scan(robot_pose: tuple[float, float, float], env: WarehouseMap) -> tuple[np.ndarray, list[tuple[float, float]]]`
  - `class VirtualCamera(width: int = 640, height: int = 480, fov: float = 60.0)`
    - `render(robot_pose: tuple[float, float, float], env: WarehouseMap) -> np.ndarray`

- [ ] **Step 1: Viết bài test `test_sensors.py`**
- [ ] **Step 2: Viết mã nguồn `amr_core/environment.py`**
- [ ] **Step 3: Viết mã nguồn `amr_core/sensors.py`**
- [ ] **Step 4: Chạy kiểm thử xác nhận pass**

---

### Task 3: Xây dựng Thuật toán Tìm đường A* & Né vật cản APF (`amr_core/planning.py`, `amr_core/avoidance.py`)

**Files:**
- Create: `D:\smart-warehouse-amr\amr_core/planning.py`
- Create: `D:\smart-warehouse-amr\amr_core/avoidance.py`
- Create: `D:\smart-warehouse-amr\amr_core/controller.py`
- Test: `D:\smart-warehouse-amr\tests/test_planning.py`

**Interfaces:**
- Consumes: `WarehouseMap` từ `amr_core.environment`
- Produces:
  - `class AStarPlanner(env: WarehouseMap, safety_margin: float = 0.35)`
    - `plan(start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]`
  - `class PotentialFieldAvoidance(k_att: float = 1.0, k_rep: float = 0.8, d0: float = 0.9)`
    - `compute_steering(robot_pose: tuple[float, float, float], target_waypoint: tuple[float, float], lidar_points: list[tuple[float, float]]) -> tuple[float, float]`
  - `class AMRController(max_v: float = 0.4, max_w: float = 1.5)`
    - `compute_velocity(robot_pose, target_waypoint, avoidance_active: bool) -> tuple[float, float]`

- [ ] **Step 1: Viết test `test_planning.py` kiểm tra A* và APF**
- [ ] **Step 2: Viết mã nguồn `amr_core/planning.py`**
- [ ] **Step 3: Viết mã nguồn `amr_core/avoidance.py`**
- [ ] **Step 4: Viết mã nguồn `amr_core/controller.py`**
- [ ] **Step 5: Chạy test xác nhận qua 100%**

---

### Task 4: Xây dựng Module Thị giác AI & Auto-Docking (`vision_ai/aruco_estimator.py`, `vision_ai/obstacle_detector.py`)

**Files:**
- Create: `D:\smart-warehouse-amr\vision_ai/aruco_estimator.py`
- Create: `D:\smart-warehouse-amr\vision_ai/obstacle_detector.py`
- Test: `D:\smart-warehouse-amr\tests/test_vision.py`

**Interfaces:**
- Produces:
  - `class ArUcoPoseEstimator(marker_size: float = 0.20)`
    - `estimate_pose(image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, bool, np.ndarray]`
    - `compute_docking_velocity(t_vec, r_vec) -> tuple[float, float, bool]`
  - `class ObstacleDetector(use_gpu: bool = True)`
    - `detect(image: np.ndarray) -> tuple[list[dict], float, np.ndarray]`

- [ ] **Step 1: Viết test `test_vision.py`**
- [ ] **Step 2: Viết mã nguồn `vision_ai/aruco_estimator.py`**
- [ ] **Step 3: Viết mã nguồn `vision_ai/obstacle_detector.py`**
- [ ] **Step 4: Chạy test xác nhận qua 100%**

---

### Task 5: Xây dựng Giao diện Mô phỏng Kép Dashboard (`main_simulation.py`)

**Files:**
- Create: `D:\smart-warehouse-amr\main_simulation.py`

**Interfaces:**
- Tích hợp toàn bộ các module: Môi trường -> Lidar -> Camera -> A* -> APF né người -> ArUco Auto-docking.
- Hiển thị 2 cửa sổ / màn hình chia đôi thời gian thực.
- Hỗ trợ cờ `--headless` để chạy tự động kiểm thử.

- [ ] **Step 1: Viết mã nguồn `main_simulation.py`**
- [ ] **Step 2: Chạy thử ở chế độ headless xác nhận chạy hết kịch bản từ A* đến Docking an toàn**

---

### Task 6: Soạn thảo Bản Báo Cáo Kỹ Thuật Nộp Thầy (`BAO_CAO_XIN_VAO_LAB.md`) & `README.md`

**Files:**
- Create: `D:\smart-warehouse-amr\BAO_CAO_XIN_VAO_LAB.md`
- Create: `D:\smart-warehouse-amr\README.md`

**Nội dung:**
- `BAO_CAO_XIN_VAO_LAB.md`: Bản báo cáo nghiên cứu khoa học 5 phần chuẩn mực (Tổng quan, Kiến trúc, Cơ sở toán học, Kết quả kiểm thử, Thư ngỏ xin gia nhập Lab).
- `README.md`: Hồ sơ mã nguồn mở chuyên nghiệp trên GitHub.
