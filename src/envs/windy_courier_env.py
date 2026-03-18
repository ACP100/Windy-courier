from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


@dataclass
class WindZone:
    x0: float
    y0: float
    x1: float
    y1: float
    vector: tuple[float, float]
    label: str

    def contains(self, point: np.ndarray) -> bool:
        return self.x0 <= point[0] <= self.x1 and self.y0 <= point[1] <= self.y1


class WindyCourierEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 180,
        seed: int | None = None,
        grid_size: tuple[int, int] = (12, 12),
        terminate_on_delivery: bool = True,
        step_penalty: float = -0.04,
        pickup_reward: float = 8.0,
        delivery_reward: float = 30.0,
        collision_penalty: float = -18.0,
        out_of_bounds_penalty: float = -20.0,
        distance_reward_scale: float = 1.1,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.width, self.height = grid_size
        self.max_steps = max_steps
        self.terminate_on_delivery = terminate_on_delivery
        self.step_penalty = step_penalty
        self.pickup_reward = pickup_reward
        self.delivery_reward = delivery_reward
        self.collision_penalty = collision_penalty
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.distance_reward_scale = distance_reward_scale
        self.window_size = 720
        self.cell_size = self.window_size // max(self.width, self.height)
        self.clock: pygame.time.Clock | None = None
        self.window: pygame.Surface | None = None
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.action_to_vector = {
            0: np.array([0.0, 0.0], dtype=np.float32),
            1: np.array([0.0, -1.0], dtype=np.float32),
            2: np.array([0.0, 1.0], dtype=np.float32),
            3: np.array([-1.0, 0.0], dtype=np.float32),
            4: np.array([1.0, 0.0], dtype=np.float32),
        }
        self.action_space = spaces.Discrete(len(self.action_to_vector))

        low = np.array(
            [0.0, 0.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0],
            dtype=np.float32,
        )
        high = np.array(
            [
                float(self.width - 1),
                float(self.height - 1),
                2.0,
                2.0,
                float(self.width - 1),
                float(self.height - 1),
                1.0,
                float(self.width - 1),
                float(self.height - 1),
                float(self.width - 1),
                float(self.height - 1),
                2.0,
                2.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.wind_zones = [
            WindZone(2, 1, 4, 10, (0.0, -0.65), "North gust"),
            WindZone(6, 0, 8, 5, (0.75, 0.0), "East tunnel"),
            WindZone(7, 7, 10, 10, (-0.55, 0.45), "Cross-current"),
        ]
        self.package_spawn = np.array([1.5, 9.5], dtype=np.float32)
        self.goal_spawn = np.array([10.5, 1.5], dtype=np.float32)
        self.obstacle_base = np.array([6.0, 6.0], dtype=np.float32)
        self.obstacle_direction = 1.0
        self.obstacle_position = self.obstacle_base.copy()

        self.agent_position = np.zeros(2, dtype=np.float32)
        self.agent_velocity = np.zeros(2, dtype=np.float32)
        self.package_picked = False
        self.step_count = 0
        self.last_distance = 0.0
        self.delivery_count = 0
        self.total_score = 0.0

    def _sample_start(self) -> np.ndarray:
        min_x, max_x = 1.25, self.width - 1.25
        min_y, max_y = 1.25, self.height - 1.25
        min_distance = 1.6

        for _ in range(64):
            candidate = np.array(
                [
                    float(self.np_random.uniform(min_x, max_x)),
                    float(self.np_random.uniform(min_y, max_y)),
                ],
                dtype=np.float32,
            )
            if (
                np.linalg.norm(candidate - self.package_spawn) >= min_distance
                and np.linalg.norm(candidate - self.goal_spawn) >= min_distance
                and np.linalg.norm(candidate - self.obstacle_base) >= min_distance
            ):
                return candidate

        return np.array([1.5, 1.5], dtype=np.float32)

    def _get_wind_vector(self, point: np.ndarray) -> np.ndarray:
        for zone in self.wind_zones:
            if zone.contains(point):
                return np.array(zone.vector, dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _current_target(self) -> np.ndarray:
        return self.goal_spawn if self.package_picked else self.package_spawn

    def _distance_to_target(self) -> float:
        return float(np.linalg.norm(self._current_target() - self.agent_position))

    def _get_obs(self) -> np.ndarray:
        wind = self._get_wind_vector(self.agent_position)
        return np.array(
            [
                self.agent_position[0],
                self.agent_position[1],
                self.agent_velocity[0],
                self.agent_velocity[1],
                self.goal_spawn[0],
                self.goal_spawn[1],
                float(self.package_picked),
                self.package_spawn[0],
                self.package_spawn[1],
                self.obstacle_position[0],
                self.obstacle_position[1],
                wind[0],
                wind[1],
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "picked": self.package_picked,
            "distance_to_target": self._distance_to_target(),
            "obstacle_position": self.obstacle_position.copy(),
            "deliveries": self.delivery_count,
            "score": self.total_score,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.agent_position = self._sample_start()
        self.agent_velocity = np.zeros(2, dtype=np.float32)
        self.package_picked = False
        self.step_count = 0
        self.obstacle_direction = 1.0
        self.obstacle_position = self.obstacle_base.copy()
        self.delivery_count = 0
        self.total_score = 0.0
        self.last_distance = self._distance_to_target()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        thrust = self.action_to_vector[int(action)] * 0.85
        wind = self._get_wind_vector(self.agent_position)
        self.agent_velocity = np.clip(0.45 * self.agent_velocity + thrust + wind, -1.5, 1.5)
        self.agent_position = self.agent_position + self.agent_velocity * 0.55

        self.obstacle_position[1] += 0.35 * self.obstacle_direction
        if self.obstacle_position[1] >= self.height - 1.5:
            self.obstacle_direction = -1.0
        elif self.obstacle_position[1] <= 1.5:
            self.obstacle_direction = 1.0

        terminated = False
        truncated = self.step_count >= self.max_steps
        reward = self.step_penalty

        out_of_bounds = not (
            0.0 <= self.agent_position[0] <= self.width - 0.01
            and 0.0 <= self.agent_position[1] <= self.height - 0.01
        )
        if out_of_bounds:
            reward += self.out_of_bounds_penalty
            terminated = True

        collision = np.linalg.norm(self.agent_position - self.obstacle_position) < 0.75
        if collision and not terminated:
            reward += self.collision_penalty
            terminated = True

        if not self.package_picked and np.linalg.norm(self.agent_position - self.package_spawn) < 0.85:
            self.package_picked = True
            reward += self.pickup_reward

        current_distance = self._distance_to_target()
        reward += self.distance_reward_scale * (self.last_distance - current_distance)
        self.last_distance = current_distance

        delivered = self.package_picked and np.linalg.norm(self.agent_position - self.goal_spawn) < 0.9
        if delivered and not terminated:
            reward += self.delivery_reward
            self.delivery_count += 1
            if self.terminate_on_delivery:
                terminated = True
            else:
                self.package_picked = False
                self.last_distance = self._distance_to_target()

        self.total_score += float(reward)

        info = self._get_info()
        info.update(
            {
                "is_success": bool(delivered),
                "collision": bool(collision),
                "out_of_bounds": bool(out_of_bounds),
                "steps": self.step_count,
            }
        )

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        return self._render_frame()

    def _render_frame(self) -> np.ndarray | None:
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            else:
                self.window = pygame.Surface((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((244, 246, 249))

        zone_colors = [(201, 231, 255), (255, 238, 194), (222, 246, 217)]
        for zone, color in zip(self.wind_zones, zone_colors):
            rect = pygame.Rect(
                int(zone.x0 * self.cell_size),
                int(zone.y0 * self.cell_size),
                int((zone.x1 - zone.x0 + 1) * self.cell_size),
                int((zone.y1 - zone.y0 + 1) * self.cell_size),
            )
            pygame.draw.rect(canvas, color, rect, border_radius=10)
            self._draw_vector(canvas, zone)

        for x in range(self.width + 1):
            pygame.draw.line(canvas, (210, 214, 220), (x * self.cell_size, 0), (x * self.cell_size, self.height * self.cell_size), 1)
        for y in range(self.height + 1):
            pygame.draw.line(canvas, (210, 214, 220), (0, y * self.cell_size), (self.width * self.cell_size, y * self.cell_size), 1)

        self._draw_square(canvas, self.goal_spawn, (62, 163, 107), "Goal")
        if not self.package_picked:
            self._draw_square(canvas, self.package_spawn, (227, 165, 52), "Pkg")

        obstacle_px = self._to_pixels(self.obstacle_position)
        pygame.draw.circle(canvas, (214, 76, 76), obstacle_px, int(self.cell_size * 0.35))
        agent_px = self._to_pixels(self.agent_position)
        pygame.draw.circle(canvas, (63, 81, 181), agent_px, int(self.cell_size * 0.28))
        pygame.draw.circle(canvas, (255, 255, 255), agent_px, int(self.cell_size * 0.12))

        font = pygame.font.SysFont("arial", 18)
        legend = [
            ("Drone", (63, 81, 181)),
            ("Package", (227, 165, 52)),
            ("Delivery", (62, 163, 107)),
            ("Obstacle", (214, 76, 76)),
        ]
        for idx, (label, color) in enumerate(legend):
            y = 10 + idx * 24
            pygame.draw.rect(canvas, color, pygame.Rect(10, y, 18, 18), border_radius=4)
            canvas.blit(font.render(label, True, (40, 44, 52)), (34, y - 1))

        target_label = "Deliver" if self.package_picked else "Pickup"
        status = f"Step {self.step_count}/{self.max_steps} | {target_label} | Score {self.total_score:.1f} | Drops {self.delivery_count}"
        canvas.blit(font.render(status, True, (40, 44, 52)), (10, self.window_size - 28))

        if self.render_mode == "human" and self.window is not None:
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))

    def _draw_square(self, canvas: pygame.Surface, point: np.ndarray, color: tuple[int, int, int], label: str) -> None:
        px = self._to_pixels(point)
        size = int(self.cell_size * 0.52)
        rect = pygame.Rect(px[0] - size // 2, px[1] - size // 2, size, size)
        pygame.draw.rect(canvas, color, rect, border_radius=8)
        font = pygame.font.SysFont("arial", 14)
        canvas.blit(font.render(label, True, (33, 37, 43)), (rect.x, rect.y - 18))

    def _draw_vector(self, canvas: pygame.Surface, zone: WindZone) -> None:
        center = np.array([(zone.x0 + zone.x1 + 1) / 2, (zone.y0 + zone.y1 + 1) / 2], dtype=np.float32)
        start = self._to_pixels(center)
        delta = np.array(zone.vector) * self.cell_size * 1.5
        end = (int(start[0] + delta[0]), int(start[1] + delta[1]))
        pygame.draw.line(canvas, (64, 90, 126), start, end, 4)
        pygame.draw.circle(canvas, (64, 90, 126), end, 6)

    def _to_pixels(self, point: np.ndarray) -> tuple[int, int]:
        return (int(point[0] * self.cell_size), int(point[1] * self.cell_size))

    def close(self) -> None:
        if self.window is not None and self.render_mode == "human":
            pygame.display.quit()
            pygame.quit()
        self.window = None
        self.clock = None
