"""
HỆ THỐNG MÔ PHỎNG ROBOT AMR NHÀ KHO THÔNG MINH (SMART WAREHOUSE AMR SIMULATOR)
Tích hợp: A* Path Planning + LiDAR APF Reactive Avoidance + ArUco 6-DoF Auto-Docking + FPV Camera AI.
"""
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

from amr_core.environment import WarehouseMap
from amr_core.sensors import VirtualLidar, VirtualCamera, normalize_angle
from amr_core.planning import AStarPlanner
from amr_core.avoidance import PotentialFieldAvoidance
from amr_core.controller import AMRController
from vision_ai.aruco_estimator import ArUcoPoseEstimator
from vision_ai.obstacle_detector import ObstacleDetector


def draw_warehouse_map(ax, env: WarehouseMap):
    """Vẽ tĩnh các kệ hàng và trạm sạc trên bản đồ 2D."""
    ax.set_facecolor("#F1F5F9")
    # Vẽ biên tường
    ax.plot([0, env.width, env.width, 0, 0], [0, 0, env.height, env.height, 0], color="#334155", linewidth=3)

    # Vẽ từng kệ hàng màu xám xanh
    for rx, ry, rw, rh in env.racks:
        rect = plt.Rectangle((rx, ry), rw, rh, facecolor="#64748B", edgecolor="#1E293B", linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        # Thêm vân chia khoang kệ
        for step in np.linspace(ry + 0.4, ry + rh - 0.4, 4):
            ax.plot([rx, rx + rw], [step, step], color="#475569", linestyle=":", linewidth=1, zorder=3)

    # Vẽ Trạm sạc tự động (Docking Station)
    dx, dy, _ = env.dock_station
    dock_rect = plt.Rectangle((dx - 0.35, dy - 0.35), 0.7, 0.7, facecolor="#F59E0B", edgecolor="#B45309", linewidth=2, zorder=2)
    ax.add_patch(dock_rect)
    ax.text(dx, dy, "DOCK #0\n[ARUCO]", color="#78350F", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=4)


def draw_amr(ax, x: float, y: float, theta: float, radius: float = 0.22):
    """Vẽ thân xe AMR, 2 bánh xe và mũi tên định hướng."""
    # Thân xe hình tròn màu xanh công nghệ
    chassis = plt.Circle((x, y), radius, facecolor="#0284C7", edgecolor="#0369A1", linewidth=2, zorder=6)
    ax.add_patch(chassis)

    # Mũi tên chỉ hướng đầu xe
    arrow_len = radius * 1.3
    ax.arrow(
        x,
        y,
        arrow_len * math.cos(theta),
        arrow_len * math.sin(theta),
        head_width=0.08,
        head_length=0.08,
        fc="#FFFFFF",
        ec="#FFFFFF",
        linewidth=2,
        zorder=7,
    )

    # 2 Bánh xe vi sai
    wheel_w, wheel_h = 0.05, 0.12
    for side in [-1, 1]:
        wx = x - side * 0.18 * math.sin(theta)
        wy = y + side * 0.18 * math.cos(theta)
        wheel = plt.Rectangle(
            (wx - wheel_w / 2, wy - wheel_h / 2),
            wheel_w,
            wheel_h,
            angle=math.degrees(theta),
            rotation_point=(wx, wy),
            facecolor="#0F172A",
            zorder=8,
        )
        ax.add_patch(wheel)


def run_full_simulation(headless: bool = False, max_steps: int = 1500):
    """
    Chạy kịch bản mô phỏng toàn diện:
    1. A* tìm đường toàn cục từ (1.0, 1.0) tới tiếp cận dock (9.2, 9.2).
    2. Robot chạy theo A*, gặp người đi bộ -> LiDAR kích hoạt APF lách qua an toàn.
    3. Tới gần dock -> Camera FPV nhận diện ArUco solvePnP -> Tự động căn chỉnh lùi chuồng khít dưới 2cm.
    """
    env = WarehouseMap(width=12.0, height=12.0)
    lidar = VirtualLidar(num_beams=72, max_range=3.8)
    camera = VirtualCamera(width=640, height=480, fov=60.0)

    planner = AStarPlanner(env=env, safety_margin=0.35)
    apf = PotentialFieldAvoidance(k_att=1.0, k_rep=1.5, d0=0.9)
    controller = AMRController(max_v=0.40, max_w=1.6)

    aruco_estimator = ArUcoPoseEstimator(marker_size=0.20)
    obstacle_detector = ObstacleDetector()

    # Trạng thái ban đầu của Robot tại khu vực xuất phát
    robot_pose = [1.0, 1.0, 0.0]
    dock_approach_goal = (9.2, 9.2)

    # 1. Tính toán lộ trình toàn cục A*
    print("[1/3 Planning] Đang tính toán lộ trình tối ưu bằng A*...")
    waypoints = planner.plan(start=(robot_pose[0], robot_pose[1]), goal=dock_approach_goal)
    if not waypoints:
        print("[LỖI] Không tìm thấy đường đi khả thi!")
        return False
    print(f" -> Lộ trình A* gồm {len(waypoints)} điểm mốc an toàn.")

    wp_idx = 0
    traj_x = [robot_pose[0]]
    traj_y = [robot_pose[1]]

    state = "NAVIGATING"  # NAVIGATING -> AVOIDING -> APPROACHING_DOCK -> DOCKING -> CHARGING
    dt = 0.05
    step = 0

    if not headless:
        plt.ion()
        fig, (ax_map, ax_cam) = plt.subplots(1, 2, figsize=(15, 7.5))
        fig.canvas.manager.set_window_title("Smart Warehouse AMR - Dual Dashboard (Map + AI Vision)")

    dock_error_x = 0.0
    dock_error_z = 0.0
    charging_steps = 0

    while step < max_steps:
        # Cập nhật chuyển động của công nhân kho (vật cản di động)
        ox, oy, orad = env.update_dynamic_obstacle(dt)

        rx, ry, rtheta = robot_pose

        # Quét LiDAR 360 độ
        ranges, hit_points = lidar.scan((rx, ry, rtheta), env)

        # Chụp ảnh Camera FPV
        fpv_raw = camera.render((rx, ry, rtheta), env)

        # Chạy AI phát hiện vật cản trên ảnh
        detections, ai_latency, fpv_annotated = obstacle_detector.detect(fpv_raw)

        # Chạy AI ước lượng tư thế ArUco
        t_vec, r_vec, aruco_found, fpv_annotated = aruco_estimator.estimate_pose(fpv_annotated)

        # ================= LOGIC ĐIỀU HƯỚNG TỪNG GIAI ĐOẠN =================
        v_cmd, w_cmd = 0.0, 0.0

        if state in ["NAVIGATING", "AVOIDING"]:
            curr_target = waypoints[wp_idx]
            dist_to_wp = math.hypot(curr_target[0] - rx, curr_target[1] - ry)

            if dist_to_wp < 0.30:
                wp_idx += 1
                if wp_idx >= len(waypoints):
                    state = "APPROACHING_DOCK"
                    print(" -> Đã tới vùng tiếp cận trạm sạc! Chuyển sang kích hoạt Camera AI...")
                else:
                    curr_target = waypoints[wp_idx]

            if state in ["NAVIGATING", "AVOIDING"]:
                # Kiểm tra có vật cản gần trong khoảng cách nguy hiểm không
                dist_to_obs = math.hypot(rx - ox, ry - oy)
                if dist_to_obs < 1.1:
                    state = "AVOIDING"
                    # Thuật toán APF bẻ lái né vật cản
                    desired_steer, force_mag = apf.compute_steering((rx, ry, rtheta), curr_target, hit_points)
                    v_cmd, w_cmd = controller.compute_velocity((rx, ry, rtheta), curr_target, desired_heading=desired_steer)
                else:
                    state = "NAVIGATING"
                    v_cmd, w_cmd = controller.compute_velocity((rx, ry, rtheta), curr_target)

                if step % 150 == 0:
                    print(f" -> [Đang di chuyển] Điểm mốc {wp_idx + 1}/{len(waypoints)} | Vị trí: ({rx:.2f}, {ry:.2f}) | Trạng thái: {state}")


        elif state == "APPROACHING_DOCK":
            # Robot xoay đầu hướng về phía dock sạc để camera bắt được thẻ ArUco
            dx_d = env.dock_station[0] - rx
            dy_d = env.dock_station[1] - ry
            target_heading = math.atan2(dy_d, dx_d)
            v_cmd, w_cmd = controller.compute_velocity((rx, ry, rtheta), (env.dock_station[0], env.dock_station[1]), desired_heading=target_heading)

            if aruco_found and t_vec is not None:
                state = "DOCKING"
                print(f" -> [THỊ GIÁC AI] Đã khóa mục tiêu ArUco! Bắt đầu tự động căn chỉnh lùi chuồng...")

        elif state == "DOCKING":
            if aruco_found and t_vec is not None and r_vec is not None:
                v_dock, w_dock, is_docked = aruco_estimator.compute_docking_velocity(t_vec, r_vec)
                v_cmd, w_cmd = v_dock, w_dock
                dock_error_x = float(t_vec[0])
                dock_error_z = float(t_vec[2])

                if is_docked:
                    state = "CHARGING"
                    v_cmd, w_cmd = 0.0, 0.0
                    print(f" -> [THÀNH CÔNG] AMR đã cắm sạc thành công! Sai số: X={dock_error_x*100:.1f}cm, Z={dock_error_z*100:.1f}cm")
            else:
                # Nếu tạm thời mất dấu thẻ, tiến chậm
                v_cmd, w_cmd = 0.04, 0.0

        elif state == "CHARGING":
            v_cmd, w_cmd = 0.0, 0.0
            if step % 20 == 0:
                print(f" [CHARGING] Trạng thái: Đang sạc pin tự động (100% Khớp dock).")

        # Cập nhật vị trí robot
        robot_pose = list(controller.update_pose((rx, ry, rtheta), v_cmd, w_cmd, dt))
        traj_x.append(robot_pose[0])
        traj_y.append(robot_pose[1])

        # ================= VẼ GIAO DIỆN HAI MÀN HÌNH =================
        if not headless and plt.fignum_exists(fig.number):
            # 1. Vẽ Màn hình Bản đồ Nhà kho 2D (Bên trái)
            ax_map.cla()
            draw_warehouse_map(ax_map, env)
            ax_map.set_aspect("equal")
            ax_map.set_xlim(-0.5, env.width + 0.5)
            ax_map.set_ylim(-0.5, env.height + 0.5)
            ax_map.set_title("BẢN ĐỒ NHÀ KHO TOÀN CẢNH (Top-down 2D Map)", fontsize=11, fontweight="bold", pad=8)
            ax_map.set_xlabel("Tọa độ X (m)", fontsize=9)
            ax_map.set_ylabel("Tọa độ Y (m)", fontsize=9)

            # Vẽ quỹ đạo A*
            path_x = [wp[0] for wp in waypoints]
            path_y = [wp[1] for wp in waypoints]
            ax_map.plot(path_x, path_y, color="#F59E0B", linestyle="--", linewidth=2, label="Đường A* hoạch định", zorder=3)

            # Vẽ vệt quỹ đạo thực tế AMR đã đi
            ax_map.plot(traj_x, traj_y, color="#06B6D4", linewidth=2.5, label="Quỹ đạo AMR thực tế", zorder=4)

            # Vẽ các tia LiDAR phản xạ
            for hx, hy in hit_points[::3]:  # Vẽ tỉ lệ 1/3 số tia cho đỡ rối mắt
                ax_map.plot([rx, hx], [ry, hy], color="#EF4444", alpha=0.25, linewidth=0.8, zorder=5)

            # Vẽ công nhân kho (Vật cản di động)
            obs_patch = plt.Circle((ox, oy), orad, facecolor="#EA580C", edgecolor="#C2410C", linewidth=2, zorder=6)
            ax_map.add_patch(obs_patch)
            ax_map.text(ox, oy, "WORKER", color="#FFFFFF", fontsize=7, fontweight="bold", ha="center", va="center", zorder=7)

            # Vẽ thân xe robot AMR
            draw_amr(ax_map, rx, ry, rtheta)

            # Bảng HUD trạng thái góc bản đồ
            state_color = "#10B981" if state == "CHARGING" else ("#EF4444" if state == "AVOIDING" else "#3B82F6")
            hud_map = (
                f"Vị trí AMR: X={rx:4.2f}m, Y={ry:4.2f}m, Heading={math.degrees(rtheta):4.1f} deg\n"
                f"Vận tốc: v={v_cmd:4.2f} m/s, w={w_cmd:4.2f} rad/s\n"
                f"Trạng thái: {state}"
            )
            ax_map.text(
                0.03, 0.97, hud_map, transform=ax_map.transAxes, fontsize=8.5, verticalalignment="top",
                fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor=state_color, linewidth=2, alpha=0.92)
            )
            ax_map.legend(loc="lower right", fontsize=8)

            # 2. Vẽ Màn hình Camera Thị giác FPV (Bên phải)
            ax_cam.cla()
            # Đổi từ BGR của OpenCV sang RGB của Matplotlib
            rgb_vis = cv2.cvtColor(fpv_annotated, cv2.COLOR_BGR2RGB)
            ax_cam.imshow(rgb_vis)
            ax_cam.set_title("GÓC NHÌN CAMERA THỊ GIÁC AI (First-Person View - FPV)", fontsize=11, fontweight="bold", pad=8)
            ax_cam.axis("off")

            plt.pause(0.001)

        step += 1
        if state == "CHARGING":
            charging_steps += 1
            if charging_steps > 30:
                print(f"\n[HOÀN TẤT NHIỆM VỤ] AMR đã cắm sạc thành công tuyệt đối! Sai số vị trí: X={dock_error_x*100:.1f}cm, Z={dock_error_z*100:.1f}cm")
                break

    if not headless and plt.fignum_exists(fig.number):
        plt.ioff()
        print("\nHoàn tất kịch bản! Đóng cửa sổ đồ họa để kết thúc chương trình.")
        plt.show()

    return True


if __name__ == "__main__":
    is_headless = "--headless" in sys.argv
    run_full_simulation(headless=is_headless)
