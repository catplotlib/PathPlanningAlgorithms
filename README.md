# Path Planning Algorithms Comparison

This README provides an overview of four path planning algorithms: A*, Dijkstra's, Potential Fields, and Probabilistic Roadmap (PRM). The algorithms were tested on a grid-based environment, and their performance was compared in terms of path length and computation time.

## Algorithms

### A*
A* is a heuristic-based search algorithm that finds the optimal path between a start and goal node. It uses a cost function that combines the actual cost from the start to the current node (g-cost) and an estimated cost from the current node to the goal (h-cost). A* expands nodes with the lowest total cost (f-cost = g-cost + h-cost) first, making it efficient and guaranteeing the optimal path if the heuristic is admissible.

Mathematically, the cost function for A* can be represented as:
$$f(n) = g(n) + h(n)$$
where $f(n)$ is the total cost, $g(n)$ is the actual cost from the start to node $n$, and $h(n)$ is the estimated cost from node $n$ to the goal.

### Dijkstra's Algorithm
Dijkstra's algorithm is a classic graph search algorithm that finds the shortest path between nodes in a weighted graph. It starts from the start node and expands nodes in order of their distance from the start. Dijkstra's algorithm is guaranteed to find the optimal path but explores all possible paths, making it less efficient than A* in large environments.

The core of Dijkstra's algorithm lies in the update of the tentative distances. For each node $v$, the algorithm maintains the shortest distance $d(v)$ from the start node. Initially, $d(v) = \infty$ for all nodes except the start node, which has $d(start) = 0$. The algorithm then iteratively selects the node with the minimum tentative distance and updates the distances of its neighbors using the following equation:
$$d(v) = \min(d(v), d(u) + w(u, v))$$
where $u$ is the current node, $v$ is a neighbor of $u$, and $w(u, v)$ is the weight of the edge between $u$ and $v$.

### Potential Fields
The Potential Fields algorithm represents the environment as a field of attractive and repulsive forces. The goal exerts an attractive force, while obstacles exert repulsive forces. The algorithm guides the robot along the gradient of the potential field towards the goal while avoiding obstacles. Potential Fields is computationally efficient but may suffer from local minima and may not always find the optimal path.

The total potential field $U(q)$ is the sum of the attractive potential $U_{att}(q)$ and the repulsive potential $U_{rep}(q)$:
$$U(q) = U_{att}(q) + U_{rep}(q)$$
where $q$ represents the robot's configuration (position and orientation).

The attractive potential is typically defined as a quadratic function:
$$U_{att}(q) = \frac{1}{2}k_{att}d^2(q, q_{goal})$$
where $k_{att}$ is a scaling factor and $d(q, q_{goal})$ is the distance between the robot's current configuration $q$ and the goal configuration $q_{goal}$.

The repulsive potential is usually defined as an inverse quadratic function:
$$U_{rep}(q) = \begin{cases}
\frac{1}{2}k_{rep}(\frac{1}{d(q, q_{obs})} - \frac{1}{d_0})^2 & \text{if } d(q, q_{obs}) \leq d_0 \\
0 & \text{if } d(q, q_{obs}) > d_0
\end{cases}$$
where $k_{rep}$ is a scaling factor, $d(q, q_{obs})$ is the distance between the robot's current configuration $q$ and the nearest obstacle, and $d_0$ is a threshold distance beyond which the repulsive potential is zero.

### Probabilistic Roadmap (PRM)
PRM is a sampling-based algorithm that constructs a roadmap of the environment by randomly sampling points and connecting them with collision-free paths. The start and goal positions are then connected to the roadmap, and a graph search algorithm (like Dijkstra's or A*) is used to find the optimal path. PRM is suitable for complex environments and can efficiently handle high-dimensional spaces.

The PRM algorithm consists of two main phases:

1. Sampling phase: Random configurations $q_{rand}$ are generated in the free space of the environment. The number of samples is a user-defined parameter.

2. Connection phase: Each sampled configuration $q_{rand}$ is connected to its $k$ nearest neighbors $q_{near}$ within a specified radius $r$. The connections are made using a local planner that checks for collision-free paths between the configurations.

After the roadmap is constructed, the start and goal configurations are connected to the roadmap, and a graph search algorithm is used to find the optimal path.

## Results

The algorithms were tested on a grid-based environment, and the following results were obtained:

| Algorithm           | Path Length (meters) | Computation Time (seconds) |
|--------------------|--------------------|--------------------------|
| A*                 | 11.90              | 1.9076                   |
| Dijkstra's         | 11.90              | 26.2737                  |
| Potential Fields   | 11.90              | 0.0207                   |
| Probabilistic Roadmap | 11.58              | 70.1667                  |

## Analysis

Based on the results, we can observe the following:

1. A* and Dijkstra's algorithms found the same optimal path length of 11.90 meters, but A* was significantly faster than Dijkstra's. This is because A* uses a heuristic to guide the search towards the goal, while Dijkstra's explores all possible paths.

2. The Potential Fields algorithm also found a path of 11.90 meters but had the fastest computation time of 0.0207 seconds. However, it's important to note that Potential Fields may not always find the optimal path and can get stuck in local minima.

3. The Probabilistic Roadmap (PRM) algorithm found a slightly shorter path of 11.58 meters but had the longest computation time of 70.1667 seconds. This is because PRM spends time constructing the roadmap by sampling points and connecting them before running the graph search.

In conclusion, the choice of algorithm depends on the specific requirements of the application. If finding the optimal path is crucial and computation time is not a constraint, A* or Dijkstra's algorithm can be used. If a fast solution is needed and optimality is not a strict requirement, Potential Fields can be a good choice. For complex environments or high-dimensional spaces, PRM can be effective but may require more computation time.

The mathematical foundations of each algorithm provide insights into their behavior and performance. A* uses a cost function that combines actual and estimated costs, Dijkstra's algorithm updates tentative distances, Potential Fields relies on attractive and repulsive potentials, and PRM constructs a roadmap through sampling and connection phases. Understanding these mathematical concepts can help in selecting and implementing the most suitable algorithm for a given path planning problem.