"""Gymnasium environment for active localization of a hidden rectangle."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.axes import Axes


def candidate_uncertainty(
    rectangles: np.ndarray, grid_size: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a grid and its binary entropy under candidate rectangles."""
    axis = np.linspace(0.0, 1.0, grid_size)
    grid_x, grid_y = np.meshgrid(axis, axis)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    rectangles = np.asarray(rectangles, dtype=float)

    inside = (
        (points[:, 0, None] >= rectangles[None, :, 0])
        & (points[:, 0, None] <= rectangles[None, :, 2])
        & (points[:, 1, None] >= rectangles[None, :, 1])
        & (points[:, 1, None] <= rectangles[None, :, 3])
    )
    probability = inside.mean(axis=1)
    entropy = np.zeros_like(probability)
    uncertain = (probability > 0.0) & (probability < 1.0)
    p = probability[uncertain]
    entropy[uncertain] = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
    return grid_x, grid_y, entropy.reshape(grid_x.shape)


def action_toward(
    position: np.ndarray, goal: np.ndarray, max_velocity: float
) -> np.ndarray:
    """Return a bounded velocity pointing from ``position`` to ``goal``."""
    direction = np.asarray(goal, dtype=float) - np.asarray(position, dtype=float)
    distance = np.linalg.norm(direction)
    if distance < 1e-12:
        return np.zeros(2, dtype=np.float32)
    return (direction / distance * max_velocity).astype(np.float32)


class BoxGym(gym.Env):
    """Collect binary samples to localize a hidden axis-aligned rectangle.

    The action is a two-dimensional velocity. Candidate rectangles are sampled
    from the set consistent with all retained binary measurements.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        dt: float = 0.1,
        sensor_box_size: float = 0.2,
        num_sensor_samples: int = 10,
        max_velocity: float = 0.2,
        inference_num: int = 100,
        max_samples: int = 1000,
        uncertainty_threshold: float = 1e-5,
    ) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if not 0.0 < sensor_box_size <= 1.0:
            raise ValueError("sensor_box_size must be in (0, 1]")
        if dt <= 0 or max_velocity <= 0:
            raise ValueError("dt and max_velocity must be positive")
        if num_sensor_samples < 1 or inference_num < 1 or max_samples < 1:
            raise ValueError("sample counts must be positive")

        self.render_mode = render_mode
        self.dt = float(dt)
        self.sensor_box_size = float(sensor_box_size)
        self.num_sensor_samples = int(num_sensor_samples)
        self.max_velocity = float(max_velocity)
        self.inference_num = int(inference_num)
        self.max_samples = int(max_samples)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.metadata = {**self.metadata, "render_fps": round(1.0 / self.dt)}

        self.action_space = spaces.Box(
            low=-self.max_velocity,
            high=self.max_velocity,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "sensor_pos": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
                "current_samples": spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.num_sensor_samples, 3),
                    dtype=np.float32,
                ),
                "history": spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.max_samples, 3),
                    dtype=np.float32,
                ),
                "history_count": spaces.Discrete(self.max_samples + 1),
                "pred_boxes": spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.inference_num, 4),
                    dtype=np.float32,
                ),
            }
        )

        self.sensor_pos = np.zeros(2, dtype=float)
        self.rect = np.zeros(4, dtype=float)
        self.samples = np.empty((0, 3), dtype=float)
        self.pred_boxes = np.empty((self.inference_num, 4), dtype=float)
        self.timestep = 0
        self._figure = None
        self._axes = None

    @property
    def positive_samples(self) -> np.ndarray:
        return self.samples[self.samples[:, 2] > 0.5, :2]

    @property
    def negative_samples(self) -> np.ndarray:
        return self.samples[self.samples[:, 2] <= 0.5, :2]

    @property
    def uncertainty(self) -> float:
        return float(np.var(self.pred_boxes, axis=0).max())

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self.sensor_pos = self.np_random.uniform(0.1, 0.9, size=2)

        if "rectangle" in options:
            rectangle = np.asarray(options["rectangle"], dtype=float)
            if rectangle.shape != (4,):
                raise ValueError("options['rectangle'] must be [left, bottom, right, top]")
            left, bottom, right, top = rectangle
            if not (0 <= left < right <= 1 and 0 <= bottom < top <= 1):
                raise ValueError("rectangle must lie inside the unit square")
            self.rect = rectangle.copy()
        else:
            self.rect = self._sample_rectangle()

        current = self._sample_sensor()
        self.samples = current.copy()
        self.pred_boxes = self._infer_rectangles(
            self.negative_samples, self.positive_samples
        )
        self.timestep = 0
        return self._observation(current), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        velocity = np.asarray(action, dtype=float)
        if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
            raise ValueError("action must be a finite two-dimensional velocity")
        speed = np.linalg.norm(velocity)
        if speed > self.max_velocity:
            velocity = velocity / speed * self.max_velocity

        self.sensor_pos = np.clip(self.sensor_pos + velocity * self.dt, 0.0, 1.0)
        current = self._sample_sensor()
        self.samples = np.concatenate((self.samples, current), axis=0)[-self.max_samples :]
        self.pred_boxes = self._infer_rectangles(
            self.negative_samples, self.positive_samples
        )
        self.timestep += 1

        terminated = self.uncertainty < self.uncertainty_threshold
        reward = -self.uncertainty
        return self._observation(current), reward, terminated, False, self._info()

    def _sample_rectangle(self) -> np.ndarray:
        center = self.np_random.uniform(0.2, 0.8, size=2)
        maximum = 2.0 * np.minimum(center, 1.0 - center)
        size = self.np_random.uniform(0.1 * maximum, maximum)
        lower = center - size / 2.0
        return np.array([lower[0], lower[1], lower[0] + size[0], lower[1] + size[1]])

    def _sample_sensor(self) -> np.ndarray:
        half = self.sensor_box_size / 2.0
        lower = np.clip(self.sensor_pos - half, 0.0, 1.0)
        upper = np.clip(self.sensor_pos + half, 0.0, 1.0)
        points = self.np_random.uniform(lower, upper, size=(self.num_sensor_samples, 2))
        left, bottom, right, top = self.rect
        labels = (
            (points[:, 0] >= left)
            & (points[:, 0] <= right)
            & (points[:, 1] >= bottom)
            & (points[:, 1] <= top)
        )
        return np.column_stack((points, labels.astype(float)))

    def _infer_rectangles(
        self, negative: np.ndarray, positive: np.ndarray
    ) -> np.ndarray:
        count = self.inference_num
        batch_size = max(10 * count, 100)
        accepted: list[np.ndarray] = []

        if positive.size:
            pos_min = positive.min(axis=0)
            pos_max = positive.max(axis=0)
            left_min, bottom_min = 0.0, 0.0
            right_max, top_max = 1.0, 1.0

            vertical_band = (negative[:, 1] > pos_min[1]) & (negative[:, 1] < pos_max[1])
            horizontal_band = (negative[:, 0] > pos_min[0]) & (negative[:, 0] < pos_max[0])
            candidates = negative[vertical_band & (negative[:, 0] < pos_min[0]), 0]
            if candidates.size:
                left_min = candidates.max()
            candidates = negative[vertical_band & (negative[:, 0] > pos_max[0]), 0]
            if candidates.size:
                right_max = candidates.min()
            candidates = negative[horizontal_band & (negative[:, 1] < pos_min[1]), 1]
            if candidates.size:
                bottom_min = candidates.max()
            candidates = negative[horizontal_band & (negative[:, 1] > pos_max[1]), 1]
            if candidates.size:
                top_max = candidates.min()

        for _ in range(1000):
            if positive.size:
                boxes = np.column_stack(
                    (
                        self.np_random.uniform(left_min, pos_min[0], batch_size),
                        self.np_random.uniform(bottom_min, pos_min[1], batch_size),
                        self.np_random.uniform(pos_max[0], right_max, batch_size),
                        self.np_random.uniform(pos_max[1], top_max, batch_size),
                    )
                )
            else:
                xs = np.sort(self.np_random.uniform(0.0, 1.0, (batch_size, 2)), axis=1)
                ys = np.sort(self.np_random.uniform(0.0, 1.0, (batch_size, 2)), axis=1)
                boxes = np.column_stack((xs[:, 0], ys[:, 0], xs[:, 1], ys[:, 1]))

            if negative.size:
                inside_negative = (
                    (negative[:, 0, None] > boxes[None, :, 0])
                    & (negative[:, 0, None] < boxes[None, :, 2])
                    & (negative[:, 1, None] > boxes[None, :, 1])
                    & (negative[:, 1, None] < boxes[None, :, 3])
                )
                boxes = boxes[~inside_negative.any(axis=0)]

            accepted.extend(boxes)
            if len(accepted) >= count:
                return np.asarray(accepted[:count], dtype=float)

        raise RuntimeError("Could not sample enough rectangles consistent with the data")

    def _observation(self, current: np.ndarray) -> dict[str, np.ndarray]:
        history = np.zeros((self.max_samples, 3), dtype=np.float32)
        history[: len(self.samples)] = self.samples
        return {
            "sensor_pos": self.sensor_pos.astype(np.float32),
            "current_samples": current.astype(np.float32),
            "history": history,
            "history_count": np.int64(len(self.samples)),
            "pred_boxes": self.pred_boxes.astype(np.float32),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "ground_truth_rectangle": self.rect.copy(),
            "uncertainty": self.uncertainty,
            "timestep": self.timestep,
        }

    def plot(
        self,
        ax: Axes | None = None,
        *,
        show_ground_truth: bool = True,
        uncertainty_map: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        goal: np.ndarray | None = None,
        trajectory: np.ndarray | None = None,
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.clear()

        if uncertainty_map is not None:
            grid_x, grid_y, entropy = uncertainty_map
            ax.contourf(grid_x, grid_y, entropy, levels=12, cmap="Blues", alpha=0.8)
        if show_ground_truth:
            left, bottom, right, top = self.rect
            ax.add_patch(
                plt.Rectangle(
                    (left, bottom), right - left, top - bottom, color="0.45", alpha=0.4
                )
            )
        for left, bottom, right, top in self.pred_boxes:
            ax.add_patch(
                plt.Rectangle(
                    (left, bottom),
                    right - left,
                    top - bottom,
                    fill=False,
                    edgecolor="red",
                    alpha=0.08,
                    linewidth=1,
                )
            )

        positive = self.positive_samples
        negative = self.negative_samples
        if len(positive):
            ax.scatter(positive[:, 0], positive[:, 1], s=12, c="black", zorder=4)
        if len(negative):
            ax.scatter(
                negative[:, 0], negative[:, 1], s=12, facecolors="white", edgecolors="black", zorder=4
            )
        if trajectory is not None and len(trajectory):
            trajectory = np.asarray(trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:purple", linewidth=1.5)
        if goal is not None:
            ax.scatter(*goal, marker="*", s=180, color="tab:purple", zorder=6)

        half = self.sensor_box_size / 2.0
        ax.add_patch(
            plt.Rectangle(
                self.sensor_pos - half,
                self.sensor_box_size,
                self.sensor_box_size,
                fill=False,
                edgecolor="black",
                linewidth=2,
                zorder=5,
            )
        )
        ax.scatter(*self.sensor_pos, marker="+", s=80, color="black", zorder=6)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title=f"Step {self.timestep}")
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            raise RuntimeError("Set render_mode='human' or 'rgb_array' when creating BoxGym")
        if self._figure is None:
            self._figure, self._axes = plt.subplots(figsize=(6, 6), dpi=100)
        self.plot(self._axes)
        self._figure.canvas.draw()
        if self.render_mode == "human":
            plt.pause(1.0 / self.metadata["render_fps"])
            return None
        rgba = np.asarray(self._figure.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    def close(self) -> None:
        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
            self._axes = None
