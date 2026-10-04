"""
Thuật toán tìm đường tối ưu toàn cục A* (A-Star Global Path Planner).
Quy hoạch đường đi ngắn nhất tránh các kệ hàng với bán kính an toàn (Safety Margin).
"""
import heapq
import math
from typing import List, Tuple
from amr_core.environment import WarehouseMap


class AStarPlanner:
    """
    Bộ quy hoạch đường đi A* trên bản đồ lưới Occupancy Grid Map.
    """

    def __init__(self, env: WarehouseMap, safety_margin: float = 0.35, grid_res: float = 0.25) -> None:
        self.env = env
        self.safety_margin = float(safety_margin)
        self.res = float(grid_res)
        self.nx = int(math.ceil(env.width / self.res))
        self.ny = int(math.ceil(env.height / self.res))

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(round(x / self.res))
        gy = int(round(y / self.res))
        return max(0, min(self.nx - 1, gx)), max(0, min(self.ny - 1, gy))

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        return gx * self.res, gy * self.res

    def is_cell_free(self, gx: int, gy: int) -> bool:
        wx, wy = self.grid_to_world(gx, gy)
        return not self.env.is_occupied(wx, wy, margin=self.safety_margin)

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Tìm kiếm đường đi tối ưu từ start (x, y) tới goal (x, y).
        Trả về danh sách các điểm mốc (Waypoints) theo đơn vị mét.
        """
        sx, sy = self.world_to_grid(start[0], start[1])
        gx, gy = self.world_to_grid(goal[0], goal[1])

        # Kiểm tra nếu đích nằm trong chướng ngại vật
        if not self.is_cell_free(gx, gy):
            return []

        # 8 hướng di chuyển: (dx, dy, cost)
        directions = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, 1.414),
            (-1, 1, 1.414),
            (1, -1, 1.414),
            (-1, -1, 1.414),
        ]

        open_set: List[Tuple[float, int, int]] = []
        heapq.heappush(open_set, (0.0, sx, sy))

        came_from = {}
        g_cost = {(sx, sy): 0.0}

        def heuristic(x1: int, y1: int, x2: int, y2: int) -> float:
            return math.hypot(x1 - x2, y1 - y2)

        found = False
        while open_set:
            _, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (gx, gy):
                found = True
                break

            for dx, dy, step_cost in directions:
                nx, ny = cx + dx, cy + dy

                if 0 <= nx < self.nx and 0 <= ny < self.ny:
                    if not self.is_cell_free(nx, ny):
                        continue

                    new_g = g_cost[(cx, cy)] + step_cost
                    if (nx, ny) not in g_cost or new_g < g_cost[(nx, ny)]:
                        g_cost[(nx, ny)] = new_g
                        f_cost = new_g + heuristic(nx, ny, gx, gy)
                        heapq.heappush(open_set, (f_cost, nx, ny))
                        came_from[(nx, ny)] = (cx, cy)

        if not found:
            return []

        # Tái tạo đường đi từ goal về start
        path_grid = []
        curr = (gx, gy)
        while curr in came_from:
            path_grid.append(curr)
            curr = came_from[curr]
        path_grid.append((sx, sy))
        path_grid.reverse()

        # Làm mịn đường đi (Subsampling giảm số điểm mốc thừa)
        smoothed_path = []
        for i, (cell_x, cell_y) in enumerate(path_grid):
            # Lấy các điểm cách nhau 3 ô hoặc điểm đầu/cuối
            if i == 0 or i == len(path_grid) - 1 or i % 3 == 0:
                wx, wy = self.grid_to_world(cell_x, cell_y)
                smoothed_path.append((round(wx, 2), round(wy, 2)))

        return smoothed_path
