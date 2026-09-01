"""Gymnasium environment for active localization of a hidden rectangle."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


class BoxGym(gym.Env):
    """Collect binary samples to localize a hidden axis-aligned rectangle."""

    def __init__(
        self,
        dt: float = 0.1,
        max_velocity: float = 0.25,
        sensor_size: float = 0.1,
        samples_per_step: int = 1,
        max_history: int = 1000,
        candidate_count: int = 100,
        uncertainty_grid_size: int = 100,
        uncertainty_threshold: float = 1e-5,
        render_size: int = 480,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.max_velocity = max_velocity
        self.sensor_size = sensor_size
        self.samples_per_step = samples_per_step
        self.max_history = max_history
        self.candidate_count = candidate_count
        self.uncertainty_grid_size = uncertainty_grid_size
        self.uncertainty_threshold = uncertainty_threshold
        self.render_size = render_size

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
                    shape=(self.samples_per_step, 3),
                    dtype=np.float32,
                ),
                "history": spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.max_history, 3),
                    dtype=np.float32,
                ),
                "history_count": spaces.Discrete(self.max_history + 1),
                "pred_boxes": spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.candidate_count, 4),
                    dtype=np.float32,
                ),
                "uncertainty": spaces.Box(
                    0.0,
                    np.log(2.0),
                    shape=(self.uncertainty_grid_size, self.uncertainty_grid_size),
                    dtype=np.float32,
                ),
            }
        )

        axis = np.linspace(0.0, 1.0, self.uncertainty_grid_size)
        self._grid_x, self._grid_y = np.meshgrid(axis, axis)
        self._grid_points = np.column_stack(
            (self._grid_x.ravel(), self._grid_y.ravel())
        )

        self.sensor_pos = np.zeros(2, dtype=float)
        self.rect = np.zeros(4, dtype=float)
        self.samples = np.empty((0, 3), dtype=float)
        self.pred_boxes = np.empty((self.candidate_count, 4), dtype=float)
        self.uncertainty = np.zeros(
            (self.uncertainty_grid_size, self.uncertainty_grid_size), dtype=float
        )
        self.trajectory = np.empty((0, 2), dtype=float)
        self.timestep = 0

    @property
    def positive_samples(self) -> np.ndarray:
        return self.samples[self.samples[:, 2] > 0.5, :2]

    @property
    def negative_samples(self) -> np.ndarray:
        return self.samples[self.samples[:, 2] <= 0.5, :2]

    @property
    def uncertainty_score(self) -> float:
        return float(np.var(self.pred_boxes, axis=0).max())

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.sensor_pos = self.np_random.uniform(0.1, 0.9, size=2)
        self.rect = self._sample_rectangle()

        current = self._sample_sensor()
        self.samples = current.copy()
        self.pred_boxes = self._infer_rectangles(
            self.negative_samples, self.positive_samples
        )
        self._update_uncertainty()
        self.trajectory = self.sensor_pos[None, :].copy()
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
        self.samples = np.concatenate((self.samples, current), axis=0)[
            -self.max_history :
        ]
        self.pred_boxes = self._infer_rectangles(
            self.negative_samples, self.positive_samples
        )
        self._update_uncertainty()
        self.trajectory = np.concatenate(
            (self.trajectory, self.sensor_pos[None, :]), axis=0
        )
        self.timestep += 1

        terminated = self.uncertainty_score < self.uncertainty_threshold
        reward = -self.uncertainty_score
        return self._observation(current), reward, terminated, False, self._info()

    def _sample_rectangle(self) -> np.ndarray:
        center = self.np_random.uniform(0.2, 0.8, size=2)
        maximum = 2.0 * np.minimum(center, 1.0 - center)
        size = self.np_random.uniform(0.1 * maximum, maximum)
        lower = center - size / 2.0
        return np.array(
            [lower[0], lower[1], lower[0] + size[0], lower[1] + size[1]]
        )

    def _sample_sensor(self) -> np.ndarray:
        half = self.sensor_size / 2.0
        lower = np.clip(self.sensor_pos - half, 0.0, 1.0)
        upper = np.clip(self.sensor_pos + half, 0.0, 1.0)
        points = self.np_random.uniform(lower, upper, size=(self.samples_per_step, 2))
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
        count = self.candidate_count
        batch_size = max(10 * count, 100)
        accepted: list[np.ndarray] = []

        if positive.size:
            pos_min = positive.min(axis=0)
            pos_max = positive.max(axis=0)
            left_min, bottom_min = 0.0, 0.0
            right_max, top_max = 1.0, 1.0

            vertical_band = (negative[:, 1] > pos_min[1]) & (
                negative[:, 1] < pos_max[1]
            )
            horizontal_band = (negative[:, 0] > pos_min[0]) & (
                negative[:, 0] < pos_max[0]
            )
            candidates = negative[
                vertical_band & (negative[:, 0] < pos_min[0]), 0
            ]
            if candidates.size:
                left_min = candidates.max()
            candidates = negative[
                vertical_band & (negative[:, 0] > pos_max[0]), 0
            ]
            if candidates.size:
                right_max = candidates.min()
            candidates = negative[
                horizontal_band & (negative[:, 1] < pos_min[1]), 1
            ]
            if candidates.size:
                bottom_min = candidates.max()
            candidates = negative[
                horizontal_band & (negative[:, 1] > pos_max[1]), 1
            ]
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
                xs = np.sort(
                    self.np_random.uniform(0.0, 1.0, (batch_size, 2)), axis=1
                )
                ys = np.sort(
                    self.np_random.uniform(0.0, 1.0, (batch_size, 2)), axis=1
                )
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

    def _update_uncertainty(self) -> None:
        points = self._grid_points
        rectangles = self.pred_boxes
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
        self.uncertainty = entropy.reshape(self._grid_x.shape)

    def _observation(self, current: np.ndarray) -> dict[str, np.ndarray]:
        history = np.zeros((self.max_history, 3), dtype=np.float32)
        history[: len(self.samples)] = self.samples
        return {
            "sensor_pos": self.sensor_pos.astype(np.float32),
            "current_samples": current.astype(np.float32),
            "history": history,
            "history_count": np.int64(len(self.samples)),
            "pred_boxes": self.pred_boxes.astype(np.float32),
            "uncertainty": self.uncertainty.astype(np.float32),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "ground_truth_rectangle": self.rect.copy(),
            "uncertainty_score": self.uncertainty_score,
            "timestep": self.timestep,
        }

    def render(self, diagnostics: bool = False) -> np.ndarray:
        """Return the fixed visualization as an RGB array."""
        dpi = 100
        side = self.render_size / dpi
        figure = Figure(figsize=(side, side), dpi=dpi)
        figure.patch.set_facecolor("white")
        canvas = FigureCanvasAgg(figure)
        axes = figure.subplots()
        axes.set_facecolor("white")

        if diagnostics:
            axes.contourf(
                self._grid_x,
                self._grid_y,
                self.uncertainty,
                levels=10,
                cmap="Blues",
                zorder=0,
            )
            left, bottom, right, top = self.rect
            axes.add_patch(
                Rectangle(
                    (left, bottom),
                    right - left,
                    top - bottom,
                    color="#808080",
                    alpha=0.55,
                    zorder=1,
                )
            )

        display_indices = np.linspace(
            0, len(self.pred_boxes) - 1, min(10, len(self.pred_boxes)), dtype=int
        )
        for left, bottom, right, top in self.pred_boxes[display_indices]:
            axes.add_patch(
                Rectangle(
                    (left, bottom),
                    right - left,
                    top - bottom,
                    fill=False,
                    edgecolor="#ff1f1f",
                    alpha=0.9,
                    linewidth=1.6,
                    zorder=2,
                )
            )

        positive = self.positive_samples
        negative = self.negative_samples
        if len(positive):
            axes.scatter(
                positive[:, 0],
                positive[:, 1],
                s=20,
                c="black",
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )
        if len(negative):
            axes.scatter(
                negative[:, 0],
                negative[:, 1],
                s=20,
                facecolors="white",
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )
        half = self.sensor_size / 2.0
        axes.add_patch(
            Rectangle(
                self.sensor_pos - half,
                self.sensor_size,
                self.sensor_size,
                fill=False,
                edgecolor="black",
                linewidth=2.5,
                zorder=5,
            )
        )
        axes.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            aspect="equal",
        )
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_color("black")
            spine.set_linewidth(2.5)
        figure.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985)

        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    def close(self) -> None:
        pass
