BoxGPT is a simplification of the general problem formulation as follows.

We are learning a generative model, denoted as $p(y|x;\theta)$, where $x$ is the state of the robot, $y$ is the variable of interest and the variable of measurement (e.g., images in vision models), and $\theta$ is the parameter of the model.

The robot moves across the space to collect data. Each data point is denoted as $d_i = (x_i, y_i)$, with $i$ being the index of the data point. The dataset is denoted as $D = {d_i}$.

The learning process (model regression) infers a distribution of the model parameters, denoted as $q(\theta)$, from the dataset. Note that maximum likelihood estimation can be considered a special case where $q(\theta)$ is a Dirac delta function.

Given an inferred distribution $q(\theta)$ of model parameters, we can quantify the expected predictive uncertainty at a particular robot state $x$, denoted as $h(x)$ and referred to as the information function, as follows:

$$
h(x) = \mathbb{E}_{\theta\sim q(\theta)} \Big[ \mathbb{H}_{y} (p(y|x;\theta)) \Big],
$$

where $\mathbb{E}$ denotes expectation and $\mathbb{H}$ denotes entropy.

The information function quantifies how uncertain the model is at a particular robot state $x$ and, consequently, indicates how much information a measurement at that state can provide to the learning process. The problem, however, is how to leverage the information function to generate robot actions in order to acquire new data. A robot that collects data faces unique limitations: the data is inherently collected sequentially, and the robot's motion is constrained by its dynamics. BoxGPT is designed to highlight and examine these challenges of embodied data collection and serve as a testbed for prototyping data collection policies for embodied agents.

BoxGPT simplifies the above general problem formulation to make **model regression** tractable, so much so that model regression is analytical, leaving the challenge entirely to the data collection policy itself rather than the capability of the model. Modern AI paradigms tend to overemphasize the importance of model architecture and overlook the impact of data quality and, more importantly, the data collection process itself on learning performance. BoxGPT is designed to expose the importance of decision-making in data collection, especially for embodied agents.

More specifically, in the case of BoxGPT, the measurement $y\in\{\text{Positive},\text{Negative}\}$ is binary, and the predictive model is represented as a Bernoulli distribution:

$$
p(y | x; \theta) =
\begin{cases}
g(x; \theta), & \text{if $y$ is Positive;} \\
1 - g(x;\theta), & \text{if $y$ is Negative.}
\end{cases}
$$

The parameter $\theta$ fully characterizes a rectangle (width, height, and coordinates of the center), and the function $g(x;\theta)$ indicates whether the robot state falls within the rectangle characterized by $\theta$:

$$
g(x; \theta) =
\begin{cases}
1, & \text{if $x$ is within $Rectangle(\theta)$;} \\
0, & \text{otherwise.}
\end{cases}
$$

Following this simplification, given a dataset $D={(x_i, y_i)}$, the inferred distribution $q(\theta)$ is simply a uniform distribution over all parameters $\theta$ that lead to rectangles that include all the positive signals and none of the negative signals, making model regression analytical.

Furthermore, following this simplification, the information function has an intuitive explanation: a state $x$ has the highest uncertainty when exactly half of the inferred rectangles include it while the other half do not.
