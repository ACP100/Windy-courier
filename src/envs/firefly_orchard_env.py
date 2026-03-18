from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class FireflyOrchardEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 360,
        seed: int | None = None,
        world_size: tuple[float, float] = (100.0, 100.0),
        n_fireflies: int = 18,
        fireflies_per_tree: int = 6,
        step_penalty: float = -0.01,
        delivery_reward: float = 2.0,
        tree_completion_reward: float = 8.0,
        success_reward: float = 30.0,
        progress_reward_scale: float = 0.15,
        cluster_reward_scale: float = 0.035,
        aura_penalty: float = 0.02,
        fog_penalty: float = -0.05,
        boundary_penalty: float = -0.18,
        bat_penalty: float = -0.35,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.world_width, self.world_height = world_size
        self.world_size = np.array(world_size, dtype=np.float32)
        self.max_steps = max_steps
        self.n_fireflies = n_fireflies
        self.fireflies_per_tree = fireflies_per_tree
        self.tree_count = 3

        self.step_penalty = step_penalty
        self.delivery_reward = delivery_reward
        self.tree_completion_reward = tree_completion_reward
        self.success_reward = success_reward
        self.progress_reward_scale = progress_reward_scale
        self.cluster_reward_scale = cluster_reward_scale
        self.aura_penalty = aura_penalty
        self.fog_penalty = fog_penalty
        self.boundary_penalty = boundary_penalty
        self.bat_penalty = bat_penalty

        self.agent_acceleration = 1.2
        self.agent_drag = 0.82
        self.agent_max_speed = 3.1
        self.firefly_max_speed = 1.7
        self.base_aura_radius = 17.0
        self.tree_capture_radius = 4.4
        self.bat_scare_radius = 12.0
        self.max_no_progress_steps = 120

        self.window_size = 720
        self.clock: pygame.time.Clock | None = None
        self.window: pygame.Surface | None = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(47,), dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.tree_positions = np.array(
            [
                [20.0, 20.0],
                [50.0, 15.0],
                [80.0, 22.0],
            ],
            dtype=np.float32,
        )
        self.fog_base_centers = np.array([[28.0, 42.0], [72.0, 38.0]], dtype=np.float32)
        self.fog_radii = np.array([12.0, 11.0], dtype=np.float32)
        self.fog_phases = self.np_random.uniform(0.0, 2.0 * np.pi, size=2).astype(np.float32)
        self.stars = self.np_random.uniform(0.0, 1.0, size=(90, 2)).astype(np.float32)

        self.agent_position = np.zeros(2, dtype=np.float32)
        self.agent_velocity = np.zeros(2, dtype=np.float32)
        self.agent_energy = 1.0
        self.last_aura = 0.0
        self.step_count = 0
        self.total_score = 0.0
        self.no_progress_steps = 0
        self.fog_steps = 0
        self.boundary_hits = 0
        self.bat_hits = 0
        self.total_deliveries = 0

        self.ambient_drift = np.zeros(2, dtype=np.float32)
        self.drift_phase = 0.0
        self.fog_centers = self.fog_base_centers.copy()

        self.bat_phase = 0.0
        self.bat_position = np.array([50.0, 26.0], dtype=np.float32)
        self.bat_velocity = np.zeros(2, dtype=np.float32)

        self.firefly_positions = np.zeros((self.n_fireflies, 2), dtype=np.float32)
        self.previous_firefly_positions = np.zeros((self.n_fireflies, 2), dtype=np.float32)
        self.firefly_velocities = np.zeros((self.n_fireflies, 2), dtype=np.float32)
        self.firefly_active = np.ones(self.n_fireflies, dtype=bool)
        self.tree_fill = np.zeros(self.tree_count, dtype=np.int32)
        self.completed_trees = np.zeros(self.tree_count, dtype=bool)

    def _spawn_fireflies(self) -> None:
        clusters = np.array(
            [
                [24.0, 62.0],
                [50.0, 70.0],
                [76.0, 58.0],
            ],
            dtype=np.float32,
        )
        cluster_indices = self.np_random.integers(0, len(clusters), size=self.n_fireflies)
        jitter = self.np_random.normal(0.0, 5.0, size=(self.n_fireflies, 2)).astype(np.float32)
        self.firefly_positions = clusters[cluster_indices] + jitter
        self.firefly_positions = np.clip(self.firefly_positions, 4.0, 96.0)
        self.previous_firefly_positions = self.firefly_positions.copy()
        self.firefly_velocities = self.np_random.normal(0.0, 0.35, size=(self.n_fireflies, 2)).astype(np.float32)
        self.firefly_active[:] = True

    def _normalize_xy(self, value: np.ndarray) -> np.ndarray:
        return np.clip((value / self.world_size) * 2.0 - 1.0, -1.0, 1.0)

    def _normalize_rel(self, rel: np.ndarray) -> np.ndarray:
        return np.clip(rel / self.world_size, -1.0, 1.0)

    def _clip_speed(self, values: np.ndarray, max_speed: float) -> np.ndarray:
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        scale = np.where(norms > max_speed, max_speed / np.maximum(norms, 1e-6), 1.0)
        return values * scale

    def _update_fog(self) -> None:
        offsets = np.stack(
            [
                5.0 * np.sin(self.step_count * 0.03 + self.fog_phases),
                3.5 * np.cos(self.step_count * 0.025 + self.fog_phases),
            ],
            axis=1,
        ).astype(np.float32)
        self.fog_centers = self.fog_base_centers + offsets

    def _update_bat(self) -> None:
        previous = self.bat_position.copy()
        self.bat_phase += 0.06
        self.bat_position = np.array(
            [
                50.0 + 34.0 * np.sin(self.bat_phase),
                24.0 + 8.0 * np.sin(self.bat_phase * 1.7),
            ],
            dtype=np.float32,
        )
        self.bat_velocity = self.bat_position - previous

    def _update_drift(self) -> None:
        self.drift_phase += 0.015
        self.ambient_drift = np.array(
            [
                0.18 * np.cos(self.drift_phase),
                0.12 * np.sin(self.drift_phase * 0.7),
            ],
            dtype=np.float32,
        )

    def _active_positions(self) -> np.ndarray:
        return self.firefly_positions[self.firefly_active]

    def _active_tree_positions(self) -> np.ndarray:
        return self.tree_positions[~self.completed_trees]

    def _mean_distance_to_goal_tree(self) -> float:
        active_positions = self._active_positions()
        target_positions = self._active_tree_positions()
        if len(active_positions) == 0 or len(target_positions) == 0:
            return 0.0
        deltas = target_positions[None, :, :] - active_positions[:, None, :]
        distances = np.linalg.norm(deltas, axis=2)
        return float(distances.min(axis=1).mean())

    def _cluster_count(self) -> int:
        active_positions = self._active_positions()
        if len(active_positions) == 0:
            return 0
        distances = np.linalg.norm(active_positions - self.agent_position, axis=1)
        radius = self.base_aura_radius + 8.0 * abs(self.last_aura)
        return int(np.sum(distances < radius))

    def _get_obs(self) -> np.ndarray:
        obs: list[float] = []
        obs.extend(self._normalize_xy(self.agent_position).tolist())
        obs.extend(np.clip(self.agent_velocity / self.agent_max_speed, -1.0, 1.0).tolist())
        obs.append(float(np.clip(self.agent_energy, 0.0, 1.0)))
        obs.append(float(np.clip((self.last_aura + 1.0) * 0.5, 0.0, 1.0)))
        obs.append(float(np.clip(1.0 - (self.step_count / self.max_steps), 0.0, 1.0)))

        for tree_position, fill in zip(self.tree_positions, self.tree_fill):
            obs.extend(self._normalize_rel(tree_position - self.agent_position).tolist())
            obs.append(float(fill / self.fireflies_per_tree))

        for center, radius in zip(self.fog_centers, self.fog_radii):
            obs.extend(self._normalize_rel(center - self.agent_position).tolist())
            obs.append(float(radius / max(self.world_width, self.world_height)))

        obs.extend(self._normalize_rel(self.bat_position - self.agent_position).tolist())
        obs.extend(np.clip(self.bat_velocity / 4.0, -1.0, 1.0).tolist())

        active_positions = self._active_positions()
        if len(active_positions) > 0:
            centroid = active_positions.mean(axis=0)
            obs.extend(self._normalize_rel(centroid - self.agent_position).tolist())
            obs.append(float(len(active_positions) / self.n_fireflies))
        else:
            obs.extend([0.0, 0.0, 0.0])

        bin_counts = np.zeros(8, dtype=np.float32)
        bin_distances = np.zeros(8, dtype=np.float32)
        if len(active_positions) > 0:
            rel = active_positions - self.agent_position
            distances = np.linalg.norm(rel, axis=1)
            angles = (np.arctan2(rel[:, 1], rel[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
            bins = np.floor((angles / (2.0 * np.pi)) * 8).astype(int) % 8
            for idx in range(8):
                mask = bins == idx
                if np.any(mask):
                    bin_counts[idx] = np.sum(mask) / self.n_fireflies
                    bin_distances[idx] = distances[mask].mean() / np.linalg.norm(self.world_size)
        for count, distance in zip(bin_counts, bin_distances):
            obs.append(float(np.clip(count, 0.0, 1.0)))
            obs.append(float(np.clip(distance, 0.0, 1.0)))

        obs.extend(np.clip(self.ambient_drift / 0.2, -1.0, 1.0).tolist())
        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        return {
            "deliveries": int(self.total_deliveries),
            "trees_lit": int(self.completed_trees.sum()),
            "active_fireflies": int(self.firefly_active.sum()),
            "fog_steps": int(self.fog_steps),
            "bat_hit": False,
            "boundary_hit": False,
            "energy": float(self.agent_energy),
            "score": float(self.total_score),
            "collision": False,
            "out_of_bounds": False,
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
            self.fog_phases = self.np_random.uniform(0.0, 2.0 * np.pi, size=2).astype(np.float32)

        self.step_count = 0
        self.total_score = 0.0
        self.no_progress_steps = 0
        self.fog_steps = 0
        self.boundary_hits = 0
        self.bat_hits = 0
        self.total_deliveries = 0
        self.agent_energy = 1.0
        self.last_aura = 0.0
        self.agent_position = np.array([50.0, 84.0], dtype=np.float32)
        self.agent_velocity = np.zeros(2, dtype=np.float32)
        self.tree_fill = np.zeros(self.tree_count, dtype=np.int32)
        self.completed_trees = np.zeros(self.tree_count, dtype=bool)
        self.drift_phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        self.bat_phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        self._update_drift()
        self._update_fog()
        self._update_bat()
        self._spawn_fireflies()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, -1.0, 1.0)

        self.step_count += 1
        self.previous_firefly_positions = self.firefly_positions.copy()

        self._update_drift()
        self._update_fog()
        self._update_bat()

        progress_before = self._mean_distance_to_goal_tree()
        cluster_before = self._cluster_count()

        thrust = action[:2] * self.agent_acceleration
        self.last_aura = float(action[2])
        aura_strength = abs(self.last_aura)

        fog_distances = np.linalg.norm(self.fog_centers - self.agent_position, axis=1)
        agent_in_fog = bool(np.any(fog_distances <= self.fog_radii))
        if agent_in_fog:
            self.fog_steps += 1

        drag = self.agent_drag * (0.72 if agent_in_fog else 1.0)
        self.agent_velocity = self.agent_velocity * drag + thrust + self.ambient_drift * 0.4
        self.agent_velocity = self._clip_speed(self.agent_velocity[np.newaxis, :], self.agent_max_speed)[0]
        self.agent_position = self.agent_position + self.agent_velocity

        boundary_hit = False
        for axis, limit in enumerate((self.world_width, self.world_height)):
            if self.agent_position[axis] < 2.0:
                self.agent_position[axis] = 2.0
                self.agent_velocity[axis] *= -0.3
                boundary_hit = True
            elif self.agent_position[axis] > limit - 2.0:
                self.agent_position[axis] = limit - 2.0
                self.agent_velocity[axis] *= -0.3
                boundary_hit = True
        if boundary_hit:
            self.boundary_hits += 1

        active_idx = np.flatnonzero(self.firefly_active)
        if len(active_idx) > 0:
            positions = self.firefly_positions[active_idx]
            velocities = self.firefly_velocities[active_idx]
            centroid = positions.mean(axis=0, keepdims=True)

            rel_to_agent = self.agent_position - positions
            agent_distances = np.linalg.norm(rel_to_agent, axis=1, keepdims=True)
            agent_distances = np.maximum(agent_distances, 1e-6)
            agent_directions = rel_to_agent / agent_distances
            aura_radius = self.base_aura_radius + 8.0 * aura_strength
            aura_mask = (agent_distances[:, 0] < aura_radius).astype(np.float32)[:, None]
            aura_falloff = np.clip(1.0 - (agent_distances / max(aura_radius, 1e-6)), 0.0, 1.0)
            aura_force = np.sign(self.last_aura) * agent_directions * aura_mask * aura_falloff * (0.5 + aura_strength)

            cohesion_force = (centroid - positions) * 0.015
            wander_force = self.np_random.normal(0.0, 0.09, size=positions.shape).astype(np.float32)

            tree_force = np.zeros_like(positions)
            active_trees = self._active_tree_positions()
            if len(active_trees) > 0:
                tree_deltas = active_trees[None, :, :] - positions[:, None, :]
                tree_distances = np.linalg.norm(tree_deltas, axis=2)
                nearest_tree_idx = np.argmin(tree_distances, axis=1)
                nearest_tree_delta = tree_deltas[np.arange(len(positions)), nearest_tree_idx]
                nearest_tree_distance = np.maximum(tree_distances[np.arange(len(positions)), nearest_tree_idx], 1e-6)
                tree_force = (
                    nearest_tree_delta
                    / nearest_tree_distance[:, None]
                    * np.clip(1.0 - nearest_tree_distance[:, None] / 18.0, 0.0, 1.0)
                    * 0.12
                )

            bat_delta = positions - self.bat_position
            bat_distance = np.linalg.norm(bat_delta, axis=1, keepdims=True)
            bat_distance = np.maximum(bat_distance, 1e-6)
            bat_force = (
                bat_delta
                / bat_distance
                * np.clip(1.0 - bat_distance / self.bat_scare_radius, 0.0, 1.0)
                * 1.45
            )

            fog_drag = np.ones((len(positions), 1), dtype=np.float32)
            fog_noise = np.zeros_like(positions)
            for center, radius in zip(self.fog_centers, self.fog_radii):
                fog_mask = np.linalg.norm(positions - center, axis=1, keepdims=True) < radius
                fog_drag *= np.where(fog_mask, 0.78, 1.0).astype(np.float32)
                fog_noise += np.where(
                    fog_mask,
                    self.np_random.normal(0.0, 0.16, size=positions.shape),
                    0.0,
                ).astype(np.float32)

            velocities = velocities * 0.86 * fog_drag + cohesion_force + wander_force + aura_force + tree_force + bat_force
            velocities = velocities + fog_noise + self.ambient_drift * 0.1
            velocities = self._clip_speed(velocities, self.firefly_max_speed)
            positions = positions + velocities

            for axis, limit in enumerate((self.world_width, self.world_height)):
                low_mask = positions[:, axis] < 2.0
                high_mask = positions[:, axis] > limit - 2.0
                if np.any(low_mask):
                    positions[low_mask, axis] = 2.0
                    velocities[low_mask, axis] *= -0.45
                if np.any(high_mask):
                    positions[high_mask, axis] = limit - 2.0
                    velocities[high_mask, axis] *= -0.45

            self.firefly_positions[active_idx] = positions
            self.firefly_velocities[active_idx] = velocities

        deliveries_this_step = 0
        completed_before = self.completed_trees.copy()
        for tree_idx, tree_position in enumerate(self.tree_positions):
            remaining = self.fireflies_per_tree - self.tree_fill[tree_idx]
            if remaining <= 0:
                continue
            active_idx = np.flatnonzero(self.firefly_active)
            if len(active_idx) == 0:
                break
            tree_distances = np.linalg.norm(self.firefly_positions[active_idx] - tree_position, axis=1)
            sorted_local = np.argsort(tree_distances)
            captured_local = sorted_local[tree_distances[sorted_local] < self.tree_capture_radius][:remaining]
            if len(captured_local) == 0:
                continue
            captured_indices = active_idx[captured_local]
            self.firefly_active[captured_indices] = False
            deliveries = int(len(captured_indices))
            deliveries_this_step += deliveries
            self.total_deliveries += deliveries
            self.tree_fill[tree_idx] += deliveries

        self.completed_trees = self.tree_fill >= self.fireflies_per_tree
        newly_completed = int(np.sum(self.completed_trees & ~completed_before))

        bat_distance_to_agent = float(np.linalg.norm(self.agent_position - self.bat_position))
        bat_hit = bat_distance_to_agent < 7.5
        if bat_hit:
            self.bat_hits += 1
            push = self.agent_position - self.bat_position
            push_norm = np.linalg.norm(push)
            if push_norm > 1e-6:
                self.agent_velocity += (push / push_norm) * 1.25

        self.agent_energy -= 0.002 + 0.012 * aura_strength
        if agent_in_fog:
            self.agent_energy -= 0.004
        if bat_hit:
            self.agent_energy -= 0.05
        self.agent_energy = float(max(self.agent_energy, 0.0))

        progress_after = self._mean_distance_to_goal_tree()
        progress_delta = float(np.clip(progress_before - progress_after, -4.0, 4.0))
        cluster_after = self._cluster_count()
        cluster_delta = cluster_after - cluster_before

        reward = self.step_penalty
        reward += self.progress_reward_scale * progress_delta
        reward += self.cluster_reward_scale * cluster_delta
        reward += self.delivery_reward * deliveries_this_step
        reward += self.tree_completion_reward * newly_completed
        reward -= self.aura_penalty * aura_strength
        reward -= 0.004 * float(np.linalg.norm(action[:2]))
        if agent_in_fog:
            reward += self.fog_penalty
        if boundary_hit:
            reward += self.boundary_penalty
        if bat_hit:
            reward += self.bat_penalty

        if deliveries_this_step == 0 and newly_completed == 0 and progress_delta < 0.03:
            self.no_progress_steps += 1
            reward -= 0.01
        else:
            self.no_progress_steps = 0

        success = bool(np.all(self.completed_trees))
        if success:
            reward += self.success_reward

        terminated = success or self.agent_energy <= 0.0
        truncated = self.step_count >= self.max_steps or self.no_progress_steps >= self.max_no_progress_steps
        self.total_score += float(reward)

        info = self._get_info()
        info.update(
            {
                "is_success": success,
                "bat_hit": bool(bat_hit),
                "boundary_hit": bool(boundary_hit),
                "collision": bool(bat_hit),
                "out_of_bounds": bool(boundary_hit),
                "steps": int(self.step_count),
                "trees_lit": int(self.completed_trees.sum()),
                "active_fireflies": int(self.firefly_active.sum()),
                "energy": float(self.agent_energy),
            }
        )

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        return self._render_frame()

    def _world_to_pixels(self, point: np.ndarray) -> tuple[int, int]:
        scale = self.window_size / max(self.world_width, self.world_height)
        return int(point[0] * scale), int(point[1] * scale)

    def _draw_glow(
        self,
        surface: pygame.Surface,
        position: tuple[int, int],
        color: tuple[int, int, int],
        radius: int,
        alpha: int,
    ) -> None:
        glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*color, alpha), (radius, radius), radius)
        surface.blit(glow, (position[0] - radius, position[1] - radius))

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
        canvas.fill((8, 10, 28))

        overlay = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        pygame.draw.circle(overlay, (36, 72, 108, 50), (100, 90), 160)
        pygame.draw.circle(overlay, (214, 126, 65, 28), (610, 100), 180)
        pygame.draw.circle(overlay, (38, 120, 93, 38), (620, 600), 240)
        canvas.blit(overlay, (0, 0))

        for star_x, star_y in self.stars:
            px = int(star_x * self.window_size)
            py = int(star_y * self.window_size * 0.82)
            brightness = 110 + int(120 * star_y)
            pygame.draw.circle(canvas, (brightness, brightness, 180), (px, py), 1)

        for trunk_x in (90, 180, 520, 640):
            pygame.draw.rect(canvas, (28, 19, 24), pygame.Rect(trunk_x, 420, 24, 240), border_radius=12)
            pygame.draw.circle(canvas, (19, 28, 32), (trunk_x + 12, 390), 80)

        fog_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        for center, radius in zip(self.fog_centers, self.fog_radii):
            fog_pos = self._world_to_pixels(center)
            self._draw_glow(fog_surface, fog_pos, (80, 138, 132), int(radius * 10), 42)
            self._draw_glow(fog_surface, fog_pos, (120, 188, 176), int(radius * 6), 28)
        canvas.blit(fog_surface, (0, 0))

        for tree_index, tree_position in enumerate(self.tree_positions):
            tree_px = self._world_to_pixels(tree_position)
            fill_ratio = float(self.tree_fill[tree_index] / self.fireflies_per_tree)
            self._draw_glow(canvas, tree_px, (242, 177, 88), 46 + int(14 * fill_ratio), 60 + int(80 * fill_ratio))
            pygame.draw.circle(canvas, (71, 50, 34), tree_px, 22)
            pygame.draw.circle(canvas, (180, 112, 56), tree_px, 16)
            pygame.draw.circle(canvas, (247, 221, 126), tree_px, 9 + int(6 * fill_ratio))
            for orbit_idx in range(self.tree_fill[tree_index]):
                angle = (orbit_idx / max(self.fireflies_per_tree, 1)) * (2.0 * np.pi) + self.step_count * 0.03
                orbit_point = (
                    int(tree_px[0] + np.cos(angle) * 22),
                    int(tree_px[1] + np.sin(angle) * 16),
                )
                self._draw_glow(canvas, orbit_point, (255, 232, 124), 6, 140)
                pygame.draw.circle(canvas, (255, 240, 165), orbit_point, 2)

        bat_px = self._world_to_pixels(self.bat_position)
        bat_prev = self._world_to_pixels(self.bat_position - self.bat_velocity * 2.0)
        pygame.draw.line(canvas, (150, 84, 122), bat_prev, bat_px, 4)
        pygame.draw.polygon(
            canvas,
            (42, 26, 41),
            [
                (bat_px[0] - 18, bat_px[1]),
                (bat_px[0], bat_px[1] - 12),
                (bat_px[0] + 18, bat_px[1]),
                (bat_px[0], bat_px[1] + 8),
            ],
        )
        pygame.draw.circle(canvas, (230, 128, 155), bat_px, 2)

        for idx in np.flatnonzero(self.firefly_active):
            prev_px = self._world_to_pixels(self.previous_firefly_positions[idx])
            firefly_px = self._world_to_pixels(self.firefly_positions[idx])
            pygame.draw.line(canvas, (150, 180, 84), prev_px, firefly_px, 2)
            self._draw_glow(canvas, firefly_px, (190, 255, 126), 10, 120)
            pygame.draw.circle(canvas, (250, 255, 182), firefly_px, 3)

        agent_px = self._world_to_pixels(self.agent_position)
        aura_color = (255, 196, 89) if self.last_aura >= 0 else (124, 200, 255)
        aura_radius = int((self.base_aura_radius + 8.0 * abs(self.last_aura)) * (self.window_size / self.world_width))
        pygame.draw.circle(canvas, aura_color, agent_px, max(aura_radius, 18), 2)
        self._draw_glow(canvas, agent_px, aura_color, max(18, int(22 + 10 * abs(self.last_aura))), 95)
        pygame.draw.circle(canvas, (146, 255, 240), agent_px, 10)
        pygame.draw.circle(canvas, (245, 255, 255), agent_px, 4)

        font = pygame.font.SysFont("georgia", 22)
        small_font = pygame.font.SysFont("georgia", 18)

        title = font.render("Firefly Orchard", True, (244, 236, 216))
        hud = small_font.render(
            f"Step {self.step_count}/{self.max_steps}   Trees {int(self.completed_trees.sum())}/3   "
            f"Delivered {self.total_deliveries}/{self.n_fireflies}   Energy {self.agent_energy:.2f}",
            True,
            (196, 211, 216),
        )
        lore = small_font.render(
            "Guide the swarm into the lantern trees before dawn. Gold attracts, blue repels.",
            True,
            (146, 178, 176),
        )
        canvas.blit(title, (20, 18))
        canvas.blit(hud, (20, 52))
        canvas.blit(lore, (20, self.window_size - 34))

        sunrise = pygame.Surface((self.window_size, 90), pygame.SRCALPHA)
        dawn_alpha = int(120 * (self.step_count / self.max_steps))
        sunrise.fill((255, 136, 88, dawn_alpha))
        canvas.blit(sunrise, (0, self.window_size - 90))

        if self.render_mode == "human" and self.window is not None:
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))

    def close(self) -> None:
        if self.window is not None and self.render_mode == "human":
            pygame.display.quit()
            pygame.quit()
        self.window = None
        self.clock = None
