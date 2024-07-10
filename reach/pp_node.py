import math
import time
import rclpy
from rclpy.node import Node as RosNode
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import pickle
import functools

import jax
from jax import grad, jit
import jax.numpy as jnp
import jax_verify

def relu_nn(params, inputs, v_max=55, omega_max=np.pi):
    for W, b in params[:-1]:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.maximum(outputs, 0)
    Wout, bout = params[-1]
    outputs = jnp.dot(inputs, Wout) + bout

    # Apply limits to the control outputs
    v = jnp.clip(outputs[0], -v_max, v_max)
    omega = jnp.clip(outputs[1], -omega_max, omega_max)

    return jnp.array([v, omega])

def init_network_params(layer_sizes, rng_key):
    params = []
    for i in range(1, len(layer_sizes)):
        in_dim = layer_sizes[i - 1]
        out_dim = layer_sizes[i]
        key, rng_key = jax.random.split(rng_key)
        bound = jnp.sqrt(6.0 / (in_dim + out_dim))
        weights = jax.random.uniform(key, (in_dim, out_dim), minval=-bound, maxval=bound)
        biases = jnp.zeros(out_dim)
        params.append((weights, biases))
    return params

layer_sizes = [3, 16, 32, 16, 2]
rng_key = jax.random.PRNGKey(0)
params = init_network_params(layer_sizes, rng_key)
controller = functools.partial(relu_nn, params)

class UnicycleModel:
    def __init__(self):
        self.delta_t = 0.1  # Sampling time

    def dynamics_step(self, xs, ut):
        x, y, theta = xs
        v, omega = ut

        theta_new = theta + omega * self.delta_t
        x_new = x + v * jnp.cos(theta) * self.delta_t
        y_new = y + v * jnp.sin(theta) * self.delta_t

        return jnp.array([x_new, y_new, theta_new])

dyn = UnicycleModel()

class PathPlanningNode(RosNode):
    def __init__(self):
        super().__init__('path_planning_node')
        self.subscription = self.create_subscription(
            OccupancyGrid, 'costmap', self.costmap_callback, 10)
        self.start_pub = self.create_publisher(PoseStamped, 'start', 10)
        self.goal_pub = self.create_publisher(PoseStamped, 'goal', 10)
        self.path_pub = self.create_publisher(Path, 'path', 10)
        self.costmap = None
        self.start = None
        self.goal = None
        self.trained_params = self.load_trained_params('trained_params3.pkl')
        self.initial_state_range = jnp.array([[-4.5, -3.5], [4.5, 3.5], [0, np.pi/6]])

    def load_trained_params(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                params = pickle.load(f)
            self.get_logger().info(f'Loaded trained parameters from {file_path}')
            return params
        except Exception as e:
            self.get_logger().error(f'Failed to load trained parameters: {str(e)}')
            return None

    def costmap_callback(self, msg):
        self.get_logger().info('Costmap received')
        self.costmap = msg
        self.initialize_start_and_goal()
        self.run_path_planning()
    
    @staticmethod
    def euler_to_quaternion(yaw, pitch=0, roll=0):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

    def initialize_start_and_goal(self):
        start_x = np.random.uniform(self.initial_state_range[0][0], self.initial_state_range[0][1])
        start_y = np.random.uniform(self.initial_state_range[1][0], self.initial_state_range[1][1])
        start_theta = np.random.uniform(self.initial_state_range[2][0], self.initial_state_range[2][1])

        start_msg = PoseStamped()
        start_msg.header.stamp = self.get_clock().now().to_msg()
        start_msg.header.frame_id = self.costmap.header.frame_id
        start_msg.pose.position.x = start_x
        start_msg.pose.position.y = start_y
        start_msg.pose.orientation.w = np.cos(start_theta / 2)
        start_msg.pose.orientation.z = np.sin(start_theta / 2)
        self.start = (start_x, start_y, start_theta)  
        self.start_pub.publish(start_msg)
        

        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = self.costmap.header.frame_id
        goal_msg.pose.position.x = 4.0
        goal_msg.pose.position.y = -4.0
        goal_msg.pose.orientation.w = 1.0
        self.goal = (goal_msg.pose.position.x, goal_msg.pose.position.y, 0.0)
        self.goal_pub.publish(goal_msg)

        self.get_logger().info(f'Start position: {self.start}')
        self.get_logger().info(f'Goal position: {self.goal}')

    def step_1(self, xt, params):
        ut = relu_nn(params, xt)
        return dyn.dynamics_step(xt, ut)
    
    def check_collision(self, path):
        for point in path:
            x, y, _ = point
            index_x, index_y, _ = self.coord_to_index(point, self.costmap.info)
            cost = self.costmap.data[index_y * self.costmap.info.width + index_x]
            if cost == 100:
                return True
        return False

    def run_path_planning(self):
        if self.costmap and self.start and self.goal:
            start_index = self.coord_to_index(self.start, self.costmap.info)
            goal_index = self.coord_to_index(self.goal, self.costmap.info)
            self.get_logger().info(f'Start index: {start_index}')
            self.get_logger().info(f'Goal index: {goal_index}')

            start_time = time.time()
            path = self.generate_path_with_trained_model(start_index, goal_index)
            end_time = time.time()
            computation_time = end_time - start_time

            if path:
                if not self.check_collision(path):
                    path_length = self.calculate_path_length(path)
                    self.publish_path(path)
                    self.get_logger().info(f'Path found! Length: {path_length:.2f} meters, Computation time: {computation_time:.4f} seconds')
                else:
                    self.get_logger().info('Path collides with an obstacle. Path not published.')
            else:
                self.get_logger().info('No path found')

    def generate_path_with_trained_model(self, start_index, goal_index):
        start_state = np.array(self.index_to_coord(start_index, self.costmap.info))
        goal_state = np.array(self.index_to_coord(goal_index, self.costmap.info))

        current_state = start_state
        path = [(float(current_state[0]), float(current_state[1]), float(current_state[2]))]  # Start the path with the initial state
        max_iterations = 1000
        iteration = 0
        goal_reached = False

        while not goal_reached and iteration < max_iterations:
            input_state = jnp.array(current_state)
            ut = relu_nn(self.trained_params, input_state)
            next_state = dyn.dynamics_step(current_state, ut)
            
            # Convert next_state to a tuple of floats before appending
            path.append((float(next_state[0]), float(next_state[1]), float(next_state[2])))
            
            current_state = next_state
            self.get_logger().info(f'Iteration {iteration}: Current state = {current_state}, Control = {ut}')
            
            goal_reached = self.is_goal_reached(current_state, goal_state)
            iteration += 1

        if goal_reached:
            self.get_logger().info(f'Goal reached in {iteration} iterations')
        elif iteration == max_iterations:
            self.get_logger().warn('Path generation reached maximum iterations without finding goal')

        self.get_logger().info(f'Path length: {len(path)}')
        return path  # Return the full path including orientations

    def is_goal_reached(self, current_state, goal_state, threshold=1.0):
        distance = np.linalg.norm(current_state[:2] - goal_state[:2])
        self.get_logger().info(f'Current state: {current_state}')
        self.get_logger().info(f'Goal state: {goal_state}')
        self.get_logger().info(f'Distance: {distance}')
        return distance < threshold

    def calculate_path_length(self, path):
        length = 0.0
        for i in range(1, len(path)):
            length += math.sqrt((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2)
        return length

    def coord_to_index(self, coord, info):
        x, y, theta = coord
        index_x = int((x - info.origin.position.x) / info.resolution)
        index_y = int((y - info.origin.position.y) / info.resolution)
        return index_x, index_y, theta

    def index_to_coord(self, index, info):
        index_x, index_y, theta = index
        x = index_x * info.resolution + info.origin.position.x
        y = index_y * info.resolution + info.origin.position.y
        return x, y, theta
    
    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.costmap.header.frame_id
        for point in path:
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = path_msg.header.frame_id
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            qx, qy, qz, qw = self.euler_to_quaternion(point[2])  
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
        self.get_logger().info('Path published')

def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()