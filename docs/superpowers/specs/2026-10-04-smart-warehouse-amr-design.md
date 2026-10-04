# Thiết Kế Hệ Thống: Robot AMR Nhà Kho Thông Minh Tích Hợp Thị Giác AI & Điều Hướng Thích Nghi

**Tên dự án:** AI-Powered Smart Warehouse Autonomous Mobile Robot (AMR)  
**Ngày lập:** 2026-10-04  
**Trạng thái:** Bản thiết kế đã được phê duyệt (Spec Approved)  
**Tác giả:** Nhóm Phát Triển Robot Tự Hành & Thị Giác AI  
**Mục tiêu ứng dụng:** Dự án nghiên cứu thực chứng (Proof-of-Concept & Portfolio) xin gia nhập Phòng Thí Nghiệm Robot & Điều Khiển Tự Động.

---

## 1. Tổng Quan & Mục Tiêu Dự Án

### 1.1 Bối cảnh
Trong các trung tâm logistics và nhà máy sản xuất hiện đại (Smart Warehouses / Industry 4.0), xe tự hành AMR (Autonomous Mobile Robot) đóng vai trò xương sống trong việc vận chuyển pallet hàng hóa. Tuy nhiên, các hệ thống AMR truyền thống gặp hai bài toán hóc búa:
1. **Va chạm với vật cản động:** Công nhân hoặc xe nâng di chuyển bất ngờ qua các giao lộ hành lang hẹp khiến robot dễ bị kẹt hoặc dừng đột ngột (Deadlock).
2. **Căn chỉnh sạc và lấy hàng chính xác (Precision Docking):** Sai số tích lũy của bánh xe (Odometry Drift) khiến robot không thể tự căn chỉnh khít vào trạm sạc tự động hoặc gầm pallet nếu chỉ dựa vào bản đồ laser thô.

### 1.2 Giải pháp Đề xuất
Xây dựng một hệ thống mô phỏng AMR hoàn chỉnh tích hợp công nghệ AI và Thị giác máy tính:
- **Tầng Lập kế hoạch (Planning):** Kết hợp thuật toán **A\*** (Global Planning) tìm đường đi ngắn nhất không va chạm kệ hàng với thuật toán **Lực thế nhân tạo APF** (Local Reactive Avoidance) để lách qua người đi bộ thời gian thực dựa trên dữ liệu chùm tia LiDAR 2D ảo.
- **Tầng Thị giác AI (Perception & Auto-Docking):** Sử dụng mô hình Camera ảo gắn trên đầu xe, tận dụng card đồ họa **NVIDIA RTX 5050 (Tensor Cores / CUDA)** để:
  - Nhận diện người và xe nâng bằng mô hình học sâu **YOLOv8**.
  - Ước lượng tư thế 6-DOF $(X, Y, Z, \text{Yaw})$ của trạm sạc qua thẻ **ArUco Marker** bằng giải thuật hình học thị giác **solvePnP** để tự động căn chỉnh lùi chuồng chính xác dưới $2\,cm$.
- **Giao diện Giám sát (Dashboard 2-in-1):** Màn hình kép hiển thị song song bản đồ toàn cảnh nhà kho (Top-down Warehouse) và góc nhìn thực tế của camera AI (First-Person View - FPV).

---

## 2. Kiến Trúc Phân Tầng Hệ Thống (System Architecture)

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   LỚP CẢM BIẾN (SENSOR SIMULATION)                     │
│  - 2D LiDAR Simulator (72 tia laser, tầm quét 4.0m, góc quét 360 độ)   │
│  - Virtual Pinhole RGB Camera (FOV 60 độ, độ phân giải 640x480)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             LỚP NHẬN THỨC & THỊ GIÁC AI (PERCEPTION LAYER)             │
│  - YOLOv8 Object Detector (Chạy tăng tốc trên GPU NVIDIA RTX 5050)     │
│  - ArUco 6-DoF Pose Estimator (solvePnP: tính t_x, t_y, t_z, Delta_yaw)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              LỚP LẬP KẾ HOẠCH & ĐIỀU HƯỚNG (PLANNING LAYER)            │
│  - Global Path Planner: Thuật toán A* trên ma trận lưới Occupancy Grid │
│  - Safety Margin Inflation: Thổi phồng vùng an toàn quanh kệ hàng      │
│  - Local Reactive Planner: Lực thế nhân tạo APF né vật cản động        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             LỚP ĐIỀU KHIỂN & ĐỘNG HỌC (CONTROL & KINEMATICS)           │
│  - Differential Drive Forward & Inverse Kinematics                     │
│  - Waypoint Pure Pursuit / Go-to-Goal PID Heading Controller           │
│  - Vision-Guided Docking Closed-Loop Controller                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                GIAO DIỆN TRỰC QUAN HÓA (VISUALIZATION)                 │
│  - Split-Screen Dashboard: Warehouse Map 2D + FPV AI Camera View       │
│  - Bảng HUD thời gian thực: Tọa độ, Vận tốc, Sai số Docking, FPS       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Thiết Kế Chi Tiết Từng Module

### 3.1 Module Môi trường & Bản đồ Nhà kho (`amr_core/environment.py`)
- Kích thước không gian: $12.0\,m \times 12.0\,m$.
- Lưới ma trận độ phân giải $0.1\,m/cell$ ($120 \times 120$ cells).
- Kệ hàng (Storage Racks): Gồm các khối hình chữ nhật đặt song song, mỗi kệ có kích thước $1.2\,m \times 3.5\,m$, khoảng cách giữa các dãy kệ là lối đi rộng $1.8\,m$.
- Vị trí Trạm sạc (Charging Dock): Đặt tại tọa độ $(10.5\,m, 10.5\,m)$ với hướng nhìn ngược ra hành lang.
- Vật cản động (Dynamic Obstacle): Mô phỏng người đi bộ di chuyển cắt ngang hành lang với phương trình chuyển động điều hòa hoặc tuần tra khép kín:
  $$x_{obs}(t) = x_{center} + A \sin(\omega t)$$

### 3.2 Module Cảm biến Ảo (`amr_core/sensors.py`)
- **LiDAR 2D:**
  - Phát $N = 72$ chùm tia quét đa hướng.
  - Sử dụng giải thuật Raycasting kiểm tra va chạm giữa đoạn thẳng tia laser và các cạnh của kệ hàng cũng như vòng tròn bao quanh vật cản động.
  - Trả về mảng `ranges` và danh sách tọa độ điểm phát hiện $(x_i, y_i)$.
- **Camera RGB Pinhole:**
  - Ma trận nội suy camera (Camera Intrinsic Matrix):
    $$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
  - Chiếu hình chiếu phối cảnh của trạm sạc và thẻ ArUco lên mặt phẳng ảnh $640 \times 480$.

### 3.3 Module Lập kế hoạch Điều hướng (`amr_core/planning.py` & `amr_core/avoidance.py`)
1. **Thuật toán A\*:**
   - Tìm kiếm đường đi ngắn nhất trên đồ thị lưới 8 hướng di chuyển.
   - Hàm chi phí: $f(n) = g(n) + h(n)$, với $h(n) = \sqrt{(x - x_{goal})^2 + (y - y_{goal})^2}$.
   - Tự động gán chi phí vô cùng (Obstacle) cho các ô lưới nằm trong bán kính an toàn $R_{safe} = 0.35\,m$ quanh chân kệ.
2. **Thuật toán Né vật cản thời gian thực (APF):**
   - Lực hút hướng tới điểm mốc A\*:
     $$\vec{F}_{att} = K_{att} \cdot (P_{waypoint} - P_{robot})$$
   - Lực đẩy từ các điểm phản xạ LiDAR có khoảng cách $d < d_0 = 0.9\,m$:
     $$\vec{F}_{rep} = \sum_{i} K_{rep} \left( \frac{1}{d_i} - \frac{1}{d_0} \right) \frac{1}{d_i^2} \frac{P_{robot} - P_{obs, i}}{d_i}$$
   - Lực tổng hợp $\vec{F}_{total} = \vec{F}_{att} + \vec{F}_{rep}$ quyết định góc lái tức thời $\theta_{desired} = \text{atan2}(F_y, F_x)$.

### 3.4 Module Thị giác AI & Auto-Docking (`vision_ai/aruco_estimator.py` & `vision_ai/obstacle_detector.py`)
1. **Ước lượng tư thế ArUco (Pose Estimation):**
   - Thư viện: `cv2.aruco`.
   - Từ các điểm góc của Marker nhận diện được trên ảnh, giải phương trình PnP tính vector tịnh tiến $T = [t_x, t_y, t_z]$ và ma trận quay $R$.
   - Tính toán góc lệch hướng: $\Delta \theta = \text{atan2}(R_{0,2}, R_{2,2})$.
   - Khi ở chế độ Docking:
     $$v = \min(K_v \cdot t_z, v_{max, dock})$$
     $$\omega = K_\omega \cdot (K_x \cdot t_x + K_\theta \cdot \Delta \theta)$$
2. **Nhận diện vật cản AI YOLOv8:**
   - Sử dụng thư viện `ultralytics` chạy trên backend PyTorch CUDA.
   - Tự động nhận diện lớp `person` và `forklift` với độ tin cậy $> 0.5$.
   - Trích xuất nhãn và đo đạc thời gian suy luận (Inference time) tính bằng miligiây.

---

## 4. Kế Hoạch Bàn Giao & Tài Liệu Báo Cáo

1. **Toàn bộ mã nguồn chạy được:** Lưu tại `D:\smart-warehouse-amr`.
2. **Bộ kiểm thử tự động (Unit Tests):** Đảm bảo $100\%$ các hàm toán học A\*, cảm biến, và giải thuật ArUco đạt Pass.
3. **Báo cáo Kỹ thuật Chuyên sâu (`BAO_CAO_XIN_VAO_LAB.md`):** Tài liệu phân tích đầy đủ lý thuyết, kiến trúc, kết quả thực nghiệm và thư ngỏ xin vào Lab gửi Thầy.
4. **Hồ sơ GitHub (`README.md`):** Trình bày chuẩn mực bài báo khoa học quốc tế.
