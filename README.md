# BoxGPT

BoxGPT is a small Gymnasium environment for active data collection. A mobile binary sensor must localize a hidden axis-aligned rectangle by choosing where to sample next. The environment maintains a generative set of rectangles consistent with the observations collected so far.

The live [interactive web demo](https://murpheylab.github.io/boxgpt/) presents the same idea in the browser.

## Install

Clone the repository and install the environment with its notebook dependencies:

```bash
git clone https://github.com/MurpheyLab/boxgpt.git
cd boxgpt
python -m pip install -e ".[notebooks]"
```

## Quick start

```python
from box_gym import BoxGym

env = BoxGym(render_mode="rgb_array")
observation, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    if terminated or truncated:
        break

env.close()
```

Actions are two-dimensional velocities. The environment clips their Euclidean norm to `max_velocity` and integrates them using `sensor_position += action * dt`.

The observation dictionary contains:

| Key | Meaning |
| --- | --- |
| `sensor_pos` | Current sensor center in the unit square |
| `current_samples` | Current `(x, y, binary_label)` measurements |
| `history` | Zero-padded measurement history |
| `history_count` | Number of valid rows in `history` |
| `pred_boxes` | Candidate rectangles as `(left, bottom, right, top)` |

Ground truth is omitted from the observation. It is available as `info["ground_truth_rectangle"]` for evaluation and visualization. The scalar `info["uncertainty"]` is the largest coordinate variance across the candidate rectangles.

`reset(seed=...)` seeds rectangle generation, sensing, and candidate inference, so complete runs are reproducible when supplied the same actions.

## Notebooks

- [Uniform goals](notebooks/01_uniform_goals.ipynb) introduces the environment and repeatedly visits uniformly sampled waypoints.
- [Highest uncertainty](notebooks/02_highest_uncertainty.ipynb) greedily visits the point where candidate rectangles disagree most.
- [Ergodic controller](notebooks/03_ergodic_controller.ipynb) distributes the trajectory according to the full spatial uncertainty distribution.

Start Jupyter from the repository root so the notebooks can import `box_gym`:

```bash
jupyter lab
```

## Environment parameters

```python
BoxGym(
    dt=0.1,
    sensor_box_size=0.2,
    num_sensor_samples=10,
    max_velocity=0.2,
    inference_num=100,
    max_samples=1000,
)
```

The environment follows the current Gymnasium API: `reset` returns `(observation, info)`, while `step` returns `(observation, reward, terminated, truncated, info)`.

## License

BoxGPT is available under the GNU General Public License v3.0. See [LICENSE](LICENSE).
