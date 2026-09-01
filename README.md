# BoxGPT

BoxGPT is a Gymnasium-style environment for embodied decision-making of sequential data collection. It is distributed as single Python file: [`box_gym.py`](box_gym.py). To use it in your project, download the file and import it directly.

*This README file focuses on the code documentation of the environment, for detailed description of the task and the mathematical formulation, please visit the [**project homepage**](https://murpheylab.github.io/boxgpt/).*

## Task description

The task invovles a mobile binary sensor moving across a planar space, with the purpose of locating a hidden rectangle in the environment. The sensor registers a positive signal if the signal is sampled from within the hidden rectangle, and negative otherwise. The agent has a first-order point-mass dynamics, in other words, is controlled by velocity. 

The environment has a built-in learning model that infers parameters of the hidden rectangle given the accumulated signals. Further, the environment also provides quantified predictive uncertainty of the current model, represented as a spatial distribution over the task space. 

You will need implement a feedback control policy given the accumulated signals and inferred parameters of teh hidden rectangle fro the data. chooses where to sample, while the environment maintains candidate rectangles consistent with the measurements and computes the resulting spatial uncertainty.

*For an interactive demonstration and more details, visit the [BoxGPT project homepage](https://murpheylab.github.io/boxgpt/).*

## Using BoxGPT

### Download

```bash
wget https://raw.githubusercontent.com/MurpheyLab/boxgpt/main/box_gym.py
```

Dependencies: NumPy, Gymnasium, and Matplotlib.

### Basic use 

```python
from box_gym import BoxGym

env = BoxGym()
observation, info = env.reset(seed=42)
frames = [env.render(diagnostics=True)]

for _ in range(300):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render(diagnostics=True))

env.close()
```

Actions are two-dimensional velocities. The environment clips their Euclidean norm to `max_velocity` and updates the sensor position using `sensor_position += action * dt`.

The observation contains the sensor position, current and historical binary measurements, candidate rectangles, and a spatial uncertainty grid. Ground truth is excluded from the observation and provided through `info["ground_truth_rectangle"]` for evaluation. The scalar `info["uncertainty_score"]` reports the largest coordinate variance across the candidate rectangles.

`render()` returns an RGB frame. `render(diagnostics=True)` adds the uncertainty distribution and ground-truth rectangle. Rendering does not open a window or write a file.

The environment sets `terminated` when uncertainty falls below `uncertainty_threshold`. It does not impose a time limit; the calling script controls the number of steps and may use or ignore `terminated`.

### Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `dt` | `0.1` | Duration of one environment step |
| `max_velocity` | `0.25` | Maximum magnitude of the velocity action |
| `sensor_size` | `0.1` | Width and height of the square binary sensor footprint |
| `samples_per_step` | `1` | Sensor measurements collected per step |
| `max_history` | `1000` | Maximum number of measurements retained in the observation |
| `candidate_count` | `100` | Number of consistent candidate rectangles maintained |
| `uncertainty_grid_size` | `100` | Width and height of the spatial uncertainty grid |
| `uncertainty_threshold` | `1e-5` | Termination threshold for the uncertainty score |
| `render_size` | `480` | Width and height of each rendered RGB frame in pixels |

For example:

```python
env = BoxGym(
    dt=0.05,
    max_velocity=0.5,
    candidate_count=200,
)
```

## Example policies

Each notebook downloads the current `box_gym.py` directly from GitHub and implements its own controller.

- [Random walk](https://colab.research.google.com/github/MurpheyLab/boxgpt/blob/main/notebooks/random_walk.ipynb)
- [Greedy information maximization](https://colab.research.google.com/github/MurpheyLab/boxgpt/blob/main/notebooks/greedy_infomax.ipynb)
- [Ergodic search](https://colab.research.google.com/github/MurpheyLab/boxgpt/blob/main/notebooks/ergodic_search.ipynb)

## License

BoxGPT is available under the GNU General Public License v3.0. See [LICENSE](LICENSE).
