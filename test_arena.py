"""
2D Arena Skeleton using Box2D (physics) and Pygame (optional visualization).
Implements the 2m x 2m arena, 4 gaps, central T-wall, light source,
pushable buckets, and a differential-drive circular robot.
"""

import math
import numpy as np
import Box2D
from Box2D.b2 import world, polygonShape, circleShape, staticBody, dynamicBody


class BucketArena:
    """
    Box2D simulation environment for the Bucket-Clearing Robot.
    Coordinate system: (0,0) is bottom-left corner, (2.0, 2.0) is top-right corner.
    Top-left light source is at (0.0, 2.0).
    """

    def __init__(self, config: dict, seed: int = 42):
        """Initialize arena parameters and RNG from a fixed seed."""
        self.cfg = config
        self.rng = np.random.default_rng(seed)

        # Arena dimensions
        self.W = self.cfg["arena"]["width"]
        self.H = self.cfg["arena"]["height"]
        self.gap = self.cfg["arena"]["gap_width"]
        self.wall_t = self.cfg["arena"]["wall_thickness"]
        self.light_pos = np.array(self.cfg["arena"]["light_pos"], dtype=float)

        # Physics parameters
        self.dt = self.cfg["simulation"]["dt"]
        self.max_time = self.cfg["simulation"]["max_time"]

        # State variables initialized in reset()
        self.world = None
        self.robot_body = None
        self.bucket_bodies = []
        self.initial_bucket_count = 0
        self.buckets_collected = 0
        self.time_elapsed = 0.0
        self.robot_left_arena = False

        self.reset()

    def reset(self, seed: int = None):
        """
        Build a fresh Box2D world, static walls, and randomly spawn
        buckets and the robot without overlaps.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Zero gravity because we are looking top-down on a 2D horizontal plane
        self.world = world(gravity=(0, 0), doSleep=True)
        self.bucket_bodies = []
        self.buckets_collected = 0
        self.time_elapsed = 0.0
        self.robot_left_arena = False

        self._build_outer_walls_with_gaps()
        self._build_t_wall()
        self._spawn_entities()

    def _add_wall_box(self, cx: float, cy: float, half_w: float, half_h: float):
        """Helper to create a static rectangular wall segment centered at (cx, cy)."""
        body = self.world.CreateStaticBody(position=(cx, cy))
        box = polygonShape(box=(half_w, half_h))
        body.CreateFixture(shape=box, friction=0.3)
        body.userData = {"type": "wall"}
        return body

    def _build_outer_walls_with_gaps(self):
        """
        Create the 2m x 2m outer perimeter with one 0.30m gap centered on each side.
        Each side of length W (2.0m) is split into two segments of length (W - gap) / 2 = 0.85m.
        """
        seg_len_x = (self.W - self.gap) / 2.0
        seg_len_y = (self.H - self.gap) / 2.0
        ht = self.wall_t / 2.0

        # Bottom wall (y = 0): left segment and right segment
        self._add_wall_box(seg_len_x / 2.0, 0.0, seg_len_x / 2.0, ht)
        self._add_wall_box(self.W - seg_len_x / 2.0, 0.0, seg_len_x / 2.0, ht)

        # Top wall (y = H): left segment and right segment
        self._add_wall_box(seg_len_x / 2.0, self.H, seg_len_x / 2.0, ht)
        self._add_wall_box(self.W - seg_len_x / 2.0, self.H, seg_len_x / 2.0, ht)

        # Left wall (x = 0): bottom segment and top segment
        self._add_wall_box(0.0, seg_len_y / 2.0, ht, seg_len_y / 2.0)
        self._add_wall_box(0.0, self.H - seg_len_y / 2.0, ht, seg_len_y / 2.0)

        # Right wall (x = W): bottom segment and top segment
        self._add_wall_box(self.W, seg_len_y / 2.0, ht, seg_len_y / 2.0)
        self._add_wall_box(self.W, self.H - seg_len_y / 2.0, ht, seg_len_y / 2.0)

    def _build_t_wall(self):
        """
        Create the central T-shaped obstacle in the center of the arena (1.0, 1.0).
        Default: horizontal bar = 0.8m, vertical stem = 0.5m, thickness = 0.05m.
        """
        bar_len = self.cfg["arena"]["t_wall"]["bar_length"]
        stem_len = self.cfg["arena"]["t_wall"]["stem_length"]
        t = self.cfg["arena"]["t_wall"]["thickness"]

        cx, cy = self.W / 2.0, self.H / 2.0
        # Place the horizontal bar slightly above center and stem extending down
        bar_y = cy + stem_len / 2.0
        stem_y = cy

        # Horizontal bar
        self._add_wall_box(cx, bar_y, bar_len / 2.0, t / 2.0)
        # Vertical stem
        self._add_wall_box(cx, stem_y, t / 2.0, stem_len / 2.0)

    def _is_pose_valid(self, x: float, y: float, radius: float, placed_circles: list) -> bool:
        """
        Verify that a candidate circle at (x, y) does not overlap with outer walls,
        the central T-wall bounding region, or previously spawned entities.
        """
        margin = 0.04  # Extra clearance buffer
        eff_r = radius + margin

        # 1. Check outer walls
        if x - eff_r < self.wall_t or x + eff_r > self.W - self.wall_t:
            return False
        if y - eff_r < self.wall_t or y + eff_r > self.H - self.wall_t:
            return False

        # 2. Check T-wall bounding boxes (bar + stem)
        bar_len = self.cfg["arena"]["t_wall"]["bar_length"]
        stem_len = self.cfg["arena"]["t_wall"]["stem_length"]
        t = self.cfg["arena"]["t_wall"]["thickness"]
        cx, cy = self.W / 2.0, self.H / 2.0
        bar_y = cy + stem_len / 2.0

        # Check overlap with horizontal bar
        if (abs(x - cx) < (bar_len / 2.0 + eff_r)) and (abs(y - bar_y) < (t / 2.0 + eff_r)):
            return False
        # Check overlap with vertical stem
        if (abs(x - cx) < (t / 2.0 + eff_r)) and (abs(y - cy) < (stem_len / 2.0 + eff_r)):
            return False

        # 3. Check overlap with already placed circles (robot and other buckets)
        for px, py, pr in placed_circles:
            if math.hypot(x - px, y - py) < (eff_r + pr):
                return False

        return True

    def _spawn_entities(self):
        """Spawn the robot with a random pose and N random buckets without overlap."""
        placed = []

        # 1. Spawn Robot
        r_rad = self.cfg["robot"]["radius"]
        while True:
            rx = self.rng.uniform(0.2, self.W - 0.2)
            ry = self.rng.uniform(0.2, self.H - 0.2)
            if self._is_pose_valid(rx, ry, r_rad, placed):
                r_theta = self.rng.uniform(-math.pi, math.pi)
                placed.append((rx, ry, r_rad))
                break

        self.robot_body = self.world.CreateDynamicBody(
            position=(rx, ry),
            angle=r_theta,
            linearDamping=2.0,
            angularDamping=2.0,
        )
        self.robot_body.CreateFixture(
            shape=circleShape(radius=r_rad),
            density=self.cfg["robot"]["density"],
            friction=self.cfg["robot"]["friction"],
        )
        self.robot_body.userData = {"type": "robot"}

        # 2. Spawn N Buckets (N in [3, 7])
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

    def step(self, v_left: float, v_right: float) -> bool:
        """
        Advance the simulation by one timestep (dt).
        Applies differential drive kinematics with motor noise, steps Box2D,
        and checks if buckets have exited through the gaps.
        Returns True if the episode is done.
        """
        v_max = self.cfg["robot"]["v_max"]
        noise_std = self.cfg["robot"]["motor_noise_std"]

        # Clip wheel speeds to [-v_max, +v_max] and inject Gaussian motor noise
        vl = np.clip(v_left + self.rng.normal(0.0, noise_std), -v_max, v_max)
        vr = np.clip(v_right + self.rng.normal(0.0, noise_std), -v_max, v_max)

        # Differential drive forward kinematics (wheelbase = 2 * radius)
        wheelbase = 2.0 * self.cfg["robot"]["radius"]
        v_linear = (vr + vl) / 2.0
        omega = (vr - vl) / wheelbase

        angle = self.robot_body.angle
        vx = v_linear * math.cos(angle)
        vy = v_linear * math.sin(angle)

        self.robot_body.linearVelocity = (vx, vy)
        self.robot_body.angularVelocity = omega

        # Step physics engine (velocityIterations=6, positionIterations=2 are standard for Box2D)
        self.world.Step(self.dt, 6, 2)
        self.time_elapsed += self.dt

        # Check if any bucket center has left the 2m x 2m arena through a gap
        remaining_buckets = []
        for b in self.bucket_bodies:
            bx, by = b.position
            if bx < 0.0 or bx > self.W or by < 0.0 or by > self.H:
                self.buckets_collected += 1
                self.world.DestroyBody(b)  # Remove collected bucket from physics world
            else:
                remaining_buckets.append(b)
        self.bucket_bodies = remaining_buckets

        # Check if robot itself illegally left the arena
        rx, ry = self.robot_body.position
        if rx < 0.0 or rx > self.W or ry < 0.0 or ry > self.H:
            self.robot_left_arena = True

        # Termination condition: time limit reached, all buckets out, or robot left arena
        done = (
            self.time_elapsed >= self.max_time
            or len(self.bucket_bodies) == 0
            or self.robot_left_arena
        )
        return done


class PygameRenderer:
    """
    Visualizes the BucketArena state using Pygame.
    Completely decoupled from Box2D physics so headless runs don't import or open Pygame.
    """

    def __init__(self, arena: BucketArena):
        import pygame
        self.pygame = pygame
        self.arena = arena
        self.ppm = arena.cfg["simulation"]["ppm"]  # Pixels per meter
        self.margin_px = 60                        # Border margin around the 2m x 2m wall

        self.screen_w = int(arena.W * self.ppm) + 2 * self.margin_px
        self.screen_h = int(arena.H * self.ppm) + 2 * self.margin_px

        self.pygame.init()
        self.screen = self.pygame.display.set_mode((self.screen_w, self.screen_h))
        self.pygame.display.set_caption("EvoRob - Bucket Clearing Arena (Box2D + Pygame)")
        self.clock = self.pygame.time.Clock()
        self.font = self.pygame.font.SysFont("monospace", 16)

    def _to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Convert Box2D meters (bottom-left origin) to Pygame pixels (top-left origin)."""
        px = int(self.margin_px + x * self.ppm)
        py = int(self.screen_h - (self.margin_px + y * self.ppm))
        return px, py

    def render(self) -> bool:
        """Draw current frame. Returns False if user closes the window."""
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False

        self.screen.fill((245, 245, 245))

        # 1. Draw light source glow in top-left corner
        lx, ly = self._to_screen(*self.arena.light_pos)
        self.pygame.draw.circle(self.screen, (255, 235, 130), (lx, ly), 28)
        self.pygame.draw.circle(self.screen, (245, 170, 20), (lx, ly), 12)

        # 2. Draw all static wall polygons (outer walls + T-wall)
        for body in self.arena.world.bodies:
            if body.userData and body.userData.get("type") == "wall":
                for fixture in body.fixtures:
                    shape = fixture.shape
                    vertices = [body.transform * v for v in shape.vertices]
                    screen_verts = [self._to_screen(vx, vy) for vx, vy in vertices]
                    self.pygame.draw.polygon(self.screen, (140, 140, 140), screen_verts)

        # 3. Draw Buckets
        b_rad_px = int(self.arena.cfg["buckets"]["radius"] * self.ppm)
        for b in self.arena.bucket_bodies:
            bx, by = self._to_screen(*b.position)
            self.pygame.draw.circle(self.screen, (245, 190, 175), (bx, by), b_rad_px)
            self.pygame.draw.circle(self.screen, (210, 95, 65), (bx, by), b_rad_px, 2)

        # 4. Draw Robot and heading line
        r_rad_px = int(self.arena.cfg["robot"]["radius"] * self.ppm)
        rx, ry = self.arena.robot_body.position
        angle = self.arena.robot_body.angle
        s_rx, s_ry = self._to_screen(rx, ry)

        self.pygame.draw.circle(self.screen, (195, 220, 245), (s_rx, s_ry), r_rad_px)
        self.pygame.draw.circle(self.screen, (45, 100, 165), (s_rx, s_ry), r_rad_px, 2)

        # Heading indicator
        hx = rx + self.arena.cfg["robot"]["radius"] * math.cos(angle)
        hy = ry + self.arena.cfg["robot"]["radius"] * math.sin(angle)
        self.pygame.draw.line(self.screen, (45, 100, 165), (s_rx, s_ry), self._to_screen(hx, hy), 3)

        # 5. HUD status text
        hud = (
            f"Time: {self.arena.time_elapsed:5.1f}s / {self.arena.max_time:.0f}s | "
            f"Buckets Out: {self.arena.buckets_collected}/{self.arena.initial_bucket_count}"
        )
        text_surf = self.font.render(hud, True, (30, 30, 30))
        self.screen.blit(text_surf, (20, 15))

        self.pygame.display.flip()
        self.clock.tick(int(1.0 / self.arena.dt))
        return True

    def close(self):
        self.pygame.quit()


if __name__ == "__main__":
    import yaml

    # Load from config.yaml (or fallback dictionary for standalone testing)
    default_cfg = {
        "arena": {
            "width": 2.0, "height": 2.0, "gap_width": 0.30, "wall_thickness": 0.05,
            "light_pos": [0.0, 2.0],
            "t_wall": {"bar_length": 0.80, "stem_length": 0.50, "thickness": 0.05},
        },
        "buckets": {"count_range": [3, 7], "radius": 0.05, "density": 0.5, "friction": 0.3, "linear_damping": 5.0},
        "robot": {"radius": 0.08, "v_max": 0.5, "density": 2.0, "friction": 0.3, "motor_noise_std": 0.02},
        "simulation": {"dt": 0.02, "max_time": 60.0, "ppm": 300},
    }

    arena = BucketArena(default_cfg, seed=101)
    renderer = PygameRenderer(arena)

    # Simple test loop: drive robot in a gentle curve to test collisions & rendering
    running = True
    while running:
        done = arena.step(v_left=0.35, v_right=0.42)
        running = renderer.render()
        if done:
            arena.reset()  # Respawn a new random episode automatically

    renderer.close()
