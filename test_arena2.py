"""
2D Arena Skeleton with Sensors, Light Shadowing, Custom Robot Spawn,
and Option B (Front-Cone Bucket Camera).
"""

import math
import numpy as np
import Box2D
from Box2D.b2 import world, polygonShape, circleShape, rayCastCallback


class ClosestRayCallback(rayCastCallback):
    """Box2D raycast helper that captures the closest hit fixture and fraction."""

    def __init__(self, ignore_body=None, walls_only: bool = False):
        super().__init__()
        self.ignore_body = ignore_body
        self.walls_only = walls_only
        self.hit = False
        self.fraction = 1.0
        self.hit_body = None
        self.point = None

    def ReportFixture(self, fixture, point, normal, fraction):
        if fixture.body == self.ignore_body:
            return -1.0  # Ignore self
        if self.walls_only and fixture.body.userData.get("type") != "wall":
            return -1.0  # Ignore buckets/robot when checking wall shadows
        self.hit = True
        self.fraction = fraction
        self.hit_body = fixture.body
        self.point = point
        return fraction


class BucketArena:
    """Box2D environment with outer walls, 4 gaps, T-wall, shadowed light, and random buckets."""

    def __init__(self, config: dict, seed: int = None, robot_pose: tuple = None):
        self.cfg = config
        self.rng = np.random.default_rng(seed)

        self.W = self.cfg["arena"]["width"]
        self.H = self.cfg["arena"]["height"]
        self.gap = self.cfg["arena"]["gap_width"]
        self.wall_t = self.cfg["arena"]["wall_thickness"]
        self.light_pos = np.array(self.cfg["arena"]["light_pos"], dtype=float)

        self.dt = self.cfg["simulation"]["dt"]
        self.max_time = self.cfg["simulation"]["max_time"]

        self.world = None
        self.robot_body = None
        self.bucket_bodies = []
        self.initial_bucket_count = 0
        self.buckets_collected = 0
        self.time_elapsed = 0.0
        self.robot_left_arena = False

        self.reset(seed=seed, robot_pose=robot_pose)

    def reset(self, seed: int = None, robot_pose: tuple = None):
        """
        Reset arena.
        - seed: optional int for reproducible layouts.
        - robot_pose: optional (x, y, theta_rad) to fix robot spawn; None = random.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.world = world(gravity=(0, 0), doSleep=True)
        self.bucket_bodies = []
        self.buckets_collected = 0
        self.time_elapsed = 0.0
        self.robot_left_arena = False

        self._build_outer_walls_with_gaps()
        self._build_t_wall()
        self._spawn_entities(robot_pose=robot_pose)

    def _add_wall_box(self, cx: float, cy: float, half_w: float, half_h: float, tag: str = "wall"):
        body = self.world.CreateStaticBody(position=(cx, cy))
        body.CreateFixture(shape=polygonShape(box=(half_w, half_h)), friction=0.3)
        body.userData = {"type": "wall", "subtype": tag}
        return body

    def _build_outer_walls_with_gaps(self):
        seg_x = (self.W - self.gap) / 2.0
        seg_y = (self.H - self.gap) / 2.0
        ht = self.wall_t / 2.0

        # Bottom & Top
        self._add_wall_box(seg_x / 2.0, 0.0, seg_x / 2.0, ht, "outer")
        self._add_wall_box(self.W - seg_x / 2.0, 0.0, seg_x / 2.0, ht, "outer")
        self._add_wall_box(seg_x / 2.0, self.H, seg_x / 2.0, ht, "outer")
        self._add_wall_box(self.W - seg_x / 2.0, self.H, seg_x / 2.0, ht, "outer")
        # Left & Right
        self._add_wall_box(0.0, seg_y / 2.0, ht, seg_y / 2.0, "outer")
        self._add_wall_box(0.0, self.H - seg_y / 2.0, ht, seg_y / 2.0, "outer")
        self._add_wall_box(self.W, seg_y / 2.0, ht, seg_y / 2.0, "outer")
        self._add_wall_box(self.W, self.H - seg_y / 2.0, ht, seg_y / 2.0, "outer")

    def _build_t_wall(self):
        bar_len = self.cfg["arena"]["t_wall"]["bar_length"]
        stem_len = self.cfg["arena"]["t_wall"]["stem_length"]
        t = self.cfg["arena"]["t_wall"]["thickness"]

        cx, cy = self.W / 2.0, self.H / 2.0
        bar_y = cy + stem_len / 2.0
        self._add_wall_box(cx, bar_y, bar_len / 2.0, t / 2.0, "t_wall")
        self._add_wall_box(cx, cy, t / 2.0, stem_len / 2.0, "t_wall")

    def _is_pose_valid(self, x: float, y: float, radius: float, placed: list) -> bool:
        eff_r = radius + 0.04
        if x - eff_r < self.wall_t or x + eff_r > self.W - self.wall_t:
            return False
        if y - eff_r < self.wall_t or y + eff_r > self.H - self.wall_t:
            return False

        bar_len = self.cfg["arena"]["t_wall"]["bar_length"]
        stem_len = self.cfg["arena"]["t_wall"]["stem_length"]
        t = self.cfg["arena"]["t_wall"]["thickness"]
        cx, cy = self.W / 2.0, self.H / 2.0
        bar_y = cy + stem_len / 2.0

        if abs(x - cx) < (bar_len / 2.0 + eff_r) and abs(y - bar_y) < (t / 2.0 + eff_r):
            return False
        if abs(x - cx) < (t / 2.0 + eff_r) and abs(y - cy) < (stem_len / 2.0 + eff_r):
            return False

        for px, py, pr in placed:
            if math.hypot(x - px, y - py) < (eff_r + pr):
                return False
        return True

    def _spawn_entities(self, robot_pose: tuple = None):
        placed = []
        r_rad = self.cfg["robot"]["radius"]

        # 1. Spawn Robot (user-selected pose or random)
        if robot_pose is not None:
            rx, ry, r_theta = robot_pose
            if not self._is_pose_valid(rx, ry, r_rad, placed):
                raise ValueError(f"Requested robot_pose {robot_pose} overlaps a wall!")
            placed.append((rx, ry, r_rad))
        else:
            while True:
                rx = self.rng.uniform(0.2, self.W - 0.2)
                ry = self.rng.uniform(0.2, self.H - 0.2)
                if self._is_pose_valid(rx, ry, r_rad, placed):
                    r_theta = self.rng.uniform(-math.pi, math.pi)
                    placed.append((rx, ry, r_rad))
                    break

        self.robot_body = self.world.CreateDynamicBody(
            position=(rx, ry), angle=r_theta, linearDamping=2.0, angularDamping=2.0
        )
        self.robot_body.CreateFixture(
            shape=circleShape(radius=r_rad),
            density=self.cfg["robot"]["density"],
            friction=self.cfg["robot"]["friction"],
        )
        self.robot_body.userData = {"type": "robot"}

        # 2. Spawn N random buckets avoiding walls and robot
        b_min, b_max = self.cfg["buckets"]["count_range"]
        self.initial_bucket_count = int(self.rng.integers(b_min, b_max + 1))
        b_rad = self.cfg["buckets"]["radius"]

        for i in range(self.initial_bucket_count):
            while True:
                bx = self.rng.uniform(0.15, self.W - 0.15)
                by = self.rng.uniform(0.15, self.H - 0.15)
                if self._is_pose_valid(bx, by, b_rad, placed):
                    placed.append((bx, by, b_rad))
                    break

            b_body = self.world.CreateDynamicBody(
                position=(bx, by),
                linearDamping=self.cfg["buckets"]["linear_damping"],
                angularDamping=2.0,
            )
            b_body.CreateFixture(
                shape=circleShape(radius=b_rad),
                density=self.cfg["buckets"]["density"],
                friction=self.cfg["buckets"]["friction"],
            )
            b_body.userData = {"type": "bucket", "id": i}
            self.bucket_bodies.append(b_body)

    def light_intensity_at(self, x: float, y: float) -> float:
        """
        Compute light intensity in [0, 1] at (x, y) with distance falloff
        and raycast shadow occlusion from the T-wall.
        """
        lx, ly = self.light_pos
        dist = math.hypot(x - lx, y - ly)
        # Smooth inverse-quadratic falloff across the 2m x 2m arena
        intensity = 1.0 / (1.0 + 1.2 * (dist ** 2))

        if dist > 1e-3:
            cb = ClosestRayCallback(ignore_body=self.robot_body, walls_only=True)
            self.world.RayCast(cb, (x, y), (lx, ly))
            if cb.hit and cb.hit_body.userData.get("type") == "wall":
                # Occluded by T-wall (or wall corner): sharply reduce brightness
                intensity *= self.cfg["sensors"]["shadow_factor"]

        return float(np.clip(intensity, 0.0, 1.0))

    def step(self, v_left: float, v_right: float) -> bool:
        v_max = self.cfg["robot"]["v_max"]
        noise_std = self.cfg["robot"]["motor_noise_std"]

        vl = np.clip(v_left + self.rng.normal(0.0, noise_std), -v_max, v_max)
        vr = np.clip(v_right + self.rng.normal(0.0, noise_std), -v_max, v_max)

        wheelbase = 2.0 * self.cfg["robot"]["radius"]
        v_linear = (vr + vl) / 2.0
        omega = (vr - vl) / wheelbase

        angle = self.robot_body.angle
        self.robot_body.linearVelocity = (v_linear * math.cos(angle), v_linear * math.sin(angle))
        self.robot_body.angularVelocity = omega

        self.world.Step(self.dt, 6, 2)
        self.time_elapsed += self.dt

        remaining = []
        for b in self.bucket_bodies:
            bx, by = b.position
            if bx < 0.0 or bx > self.W or by < 0.0 or by > self.H:
                self.buckets_collected += 1
                self.world.DestroyBody(b)
            else:
                remaining.append(b)
        self.bucket_bodies = remaining

        rx, ry = self.robot_body.position
        if rx < 0.0 or rx > self.W or ry < 0.0 or ry > self.H:
            self.robot_left_arena = True

        return self.time_elapsed >= self.max_time or len(self.bucket_bodies) == 0 or self.robot_left_arena


class RobotSensors:
    """
    Computes the 12-value normalized sensor vector:
      [0..7]  : 8 noisy IR proximity sensors (walls + buckets) in [0, 1]
      [8..9]  : 2 noisy front light sensors (front-left, front-right) in [0, 1]
      [10..11]: Option B front-cone bucket camera (angle in [-1, 1], proximity in [0, 1])
    """

    def __init__(self, arena: BucketArena):
        self.arena = arena
        scfg = arena.cfg["sensors"]
        self.ir_range = scfg["ir_range"]
        self.ir_noise = scfg["ir_noise_std"]
        self.light_noise = scfg["light_noise_std"]
        self.cam_fov = math.radians(scfg["camera_fov_deg"])
        self.cam_range = scfg["camera_range"]
        self.cam_noise = scfg["camera_noise_std"]

        # Store last debug data for Pygame visualization
        self.last_ir_rays = []
        self.last_detected_bucket = None

    def read(self) -> np.ndarray:
        robot = self.arena.robot_body
        rx, ry = robot.position
        theta = robot.angle
        r_rad = self.arena.cfg["robot"]["radius"]
        rng = self.arena.rng

        # 1. Eight IR proximity sensors at 45-degree intervals
        ir_vals = np.zeros(8, dtype=float)
        self.last_ir_rays = []
        for i in range(8):
            ray_angle = theta + i * (math.pi / 4.0)
            p1 = (rx + r_rad * math.cos(ray_angle), ry + r_rad * math.sin(ray_angle))
            p2 = (
                rx + (r_rad + self.ir_range) * math.cos(ray_angle),
                ry + (r_rad + self.ir_range) * math.sin(ray_angle),
            )
            cb = ClosestRayCallback(ignore_body=robot)
            self.arena.world.RayCast(cb, p1, p2)

            if cb.hit:
                val = 1.0 - cb.fraction  # Closer obstacle -> higher value
                end_pt = cb.point
            else:
                val = 0.0
                end_pt = p2

            ir_vals[i] = np.clip(val + rng.normal(0.0, self.ir_noise), 0.0, 1.0)
            self.last_ir_rays.append((p1, end_pt, cb.hit))

        # 2. Two light sensors (front-left at +45 deg, front-right at -45 deg)
        light_vals = np.zeros(2, dtype=float)
        for idx, offset in enumerate([math.pi / 4.0, -math.pi / 4.0]):
            sx = rx + r_rad * math.cos(theta + offset)
            sy = ry + r_rad * math.sin(theta + offset)
            raw_light = self.arena.light_intensity_at(sx, sy)
            light_vals[idx] = np.clip(raw_light + rng.normal(0.0, self.light_noise), 0.0, 1.0)

        # 3. Option B: Front-cone camera distinguishing buckets from walls
        cam_angle = 0.0
        cam_prox = 0.0
        best_dist = float("inf")
        self.last_detected_bucket = None

        for b in self.arena.bucket_bodies:
            bx, by = b.position
            dist = math.hypot(bx - rx, by - ry)
            if dist > self.cam_range or dist >= best_dist:
                continue

            # Relative angle wrapped to [-pi, pi]
            rel_ang = math.atan2(by - ry, bx - rx) - theta
            rel_ang = (rel_ang + math.pi) % (2.0 * math.pi) - math.pi

            if abs(rel_ang) <= self.cam_fov / 2.0:
                # Line-of-sight check: ignore bucket if blocked by a wall
                cb = ClosestRayCallback(ignore_body=robot)
                self.arena.world.RayCast(cb, (rx, ry), (bx, by))
                if cb.hit and cb.hit_body == b:
                    best_dist = dist
                    cam_angle = rel_ang / (self.cam_fov / 2.0)  # [-1, 1]
                    cam_prox = 1.0 - (dist / self.cam_range)    # [0, 1]
                    self.last_detected_bucket = b

        if self.last_detected_bucket is not None:
            cam_angle = float(np.clip(cam_angle + rng.normal(0.0, self.cam_noise), -1.0, 1.0))
            cam_prox = float(np.clip(cam_prox + rng.normal(0.0, self.cam_noise), 0.0, 1.0))

        return np.concatenate([ir_vals, light_vals, [cam_angle, cam_prox]])


class PygameRenderer:
    """Visualizes the arena, light/shadow map, IR rays, and Option B bucket camera cone."""

    def __init__(self, arena: BucketArena, sensors: RobotSensors):
        import pygame
        self.pygame = pygame
        self.arena = arena
        self.sensors = sensors
        self.ppm = arena.cfg["simulation"]["ppm"]
        self.margin_px = 60

        self.arena_px_w = int(arena.W * self.ppm)
        self.arena_px_h = int(arena.H * self.ppm)
        self.screen_w = self.arena_px_w + 2 * self.margin_px
        self.screen_h = self.arena_px_h + 2 * self.margin_px

        self.pygame.init()
        self.screen = self.pygame.display.set_mode((self.screen_w, self.screen_h))
        self.pygame.display.set_caption("EvoRob - Arena with Shadowed Light & Sensors")
        self.clock = self.pygame.time.Clock()
        self.font = self.pygame.font.SysFont("monospace", 15)

        # Precompute static light & T-wall shadow map once
        self.light_surface = self._build_light_map()

    def _build_light_map(self):
        """Sample light_intensity_at on a 60x60 grid and smoothscale for fast rendering."""
        res = 60
        surf = self.pygame.Surface((res, res))
        for gx in range(res):
            for gy in range(res):
                wx = (gx + 0.5) / res * self.arena.W
                wy = (1.0 - (gy + 0.5) / res) * self.arena.H
                val = self.arena.light_intensity_at(wx, wy)
                # Map intensity [0, 1] to dark shadow gray -> warm bright floor
                r = int(95 + 160 * val)
                g = int(95 + 155 * val)
                b = int(100 + 115 * val)
                surf.set_at((gx, gy), (r, g, b))
        return self.pygame.transform.smoothscale(surf, (self.arena_px_w, self.arena_px_h))

    def _to_screen(self, x: float, y: float) -> tuple[int, int]:
        px = int(self.margin_px + x * self.ppm)
        py = int(self.screen_h - (self.margin_px + y * self.ppm))
        return px, py

    def render(self, sensor_vec: np.ndarray) -> bool:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False
            elif event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_r:
                self.arena.reset(robot_pose=None)
                sensor_vec = self.sensors.read()  # Refresh rays & HUD immediately for the new spawn

        self.screen.fill((40, 40, 45))
        # Draw precomputed light & shadow floor
        self.screen.blit(self.light_surface, (self.margin_px, self.margin_px))

        # Draw light emitter icon
        lx, ly = self._to_screen(*self.arena.light_pos)
        self.pygame.draw.circle(self.screen, (255, 220, 70), (lx, ly), 12)

        # Draw Option B front camera FOV cone
        rx, ry = self.arena.robot_body.position
        theta = self.arena.robot_body.angle
        half_fov = self.sensors.cam_fov / 2.0
        cam_r = self.sensors.cam_range
        p_center = self._to_screen(rx, ry)
        p_left = self._to_screen(rx + cam_r * math.cos(theta + half_fov), ry + cam_r * math.sin(theta + half_fov))
        p_right = self._to_screen(rx + cam_r * math.cos(theta - half_fov), ry + cam_r * math.sin(theta - half_fov))
        self.pygame.draw.line(self.screen, (70, 180, 90), p_center, p_left, 1)
        self.pygame.draw.line(self.screen, (70, 180, 90), p_center, p_right, 1)

        # Draw 8 IR rays
        for p1, p2, hit in self.sensors.last_ir_rays:
            color = (220, 60, 60) if hit else (160, 160, 160)
            self.pygame.draw.line(self.screen, color, self._to_screen(*p1), self._to_screen(*p2), 1)

        # Draw Walls
        for body in self.arena.world.bodies:
            if body.userData and body.userData.get("type") == "wall":
                for fixture in body.fixtures:
                    verts = [self._to_screen(*(body.transform * v)) for v in fixture.shape.vertices]
                    self.pygame.draw.polygon(self.screen, (90, 90, 95), verts)

        # Draw Buckets (highlight the one currently seen by the Option B camera)
        b_rad_px = int(self.arena.cfg["buckets"]["radius"] * self.ppm)
        for b in self.arena.bucket_bodies:
            bx, by = self._to_screen(*b.position)
            is_seen = b == self.sensors.last_detected_bucket
            fill = (120, 235, 140) if is_seen else (245, 190, 175)
            border = (30, 150, 50) if is_seen else (210, 95, 65)
            self.pygame.draw.circle(self.screen, fill, (bx, by), b_rad_px)
            self.pygame.draw.circle(self.screen, border, (bx, by), b_rad_px, 2)

        # Draw Robot
        r_rad_px = int(self.arena.cfg["robot"]["radius"] * self.ppm)
        self.pygame.draw.circle(self.screen, (195, 220, 245), p_center, r_rad_px)
        self.pygame.draw.circle(self.screen, (45, 100, 165), p_center, r_rad_px, 2)
        hx = rx + self.arena.cfg["robot"]["radius"] * math.cos(theta)
        hy = ry + self.arena.cfg["robot"]["radius"] * math.sin(theta)
        self.pygame.draw.line(self.screen, (45, 100, 165), p_center, self._to_screen(hx, hy), 3)

        # HUD showing Light sensors & Camera output live
        hud = (
            f"T:{self.arena.time_elapsed:4.1f}s | Out:{self.arena.buckets_collected}/{self.arena.initial_bucket_count} | "
            f"Light(L,R):({sensor_vec[8]:.2f},{sensor_vec[9]:.2f}) | "
            f"Cam(ang,prox):({sensor_vec[10]:+.2f},{sensor_vec[11]:.2f})"
        )
        self.screen.blit(self.font.render(hud, True, (240, 240, 240)), (15, 15))

        self.pygame.display.flip()
        self.clock.tick(int(1.0 / self.arena.dt))
        return True

    def close(self):
        self.pygame.quit()


if __name__ == "__main__":
    cfg = {
        "arena": {
            "width": 2.0, "height": 2.0, "gap_width": 0.30, "wall_thickness": 0.05,
            "light_pos": [0.08, 1.92],  # Inside top-left corner so outer walls don't self-shadow
            "t_wall": {"bar_length": 0.80, "stem_length": 0.50, "thickness": 0.05},
        },
        "buckets": {"count_range": [3, 7], "radius": 0.05, "density": 0.5, "friction": 0.3, "linear_damping": 5.0},
        "robot": {"radius": 0.08, "v_max": 0.5, "density": 2.0, "friction": 0.3, "motor_noise_std": 0.02},
        "sensors": {
            "ir_range": 0.50,
            "ir_noise_std": 0.02,
            "light_noise_std": 0.01,
            "shadow_factor": 0.15,   # Brightness multiplier behind/inside the T-wall
            "camera_fov_deg": 60.0,  # Option B front cone width
            "camera_range": 1.20,
            "camera_noise_std": 0.02,
        },
        "simulation": {"dt": 0.02, "max_time": 60.0, "ppm": 300},
    }

    # Example: fixed robot spawn pose (x=0.5, y=0.5, facing 45 deg). Set to None for random!
    # custom_spawn = (0.5, 0.5, math.radians(45))

    # seed=None -> new random buckets & robot pose on every run
    # robot_pose=None -> random start position and heading inside the arena
    arena = BucketArena(cfg, seed=None, robot_pose=None)
    sensors = RobotSensors(arena)
    renderer = PygameRenderer(arena, sensors)

    running = True
    while running:
        obs = sensors.read()
        done = arena.step(v_left=0.35, v_right=0.40)
        running = renderer.render(obs)
        if done:
            arena.reset(robot_pose=None)  # Subsequent resets use random pose

    renderer.close()
