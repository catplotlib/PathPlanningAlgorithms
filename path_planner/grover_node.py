import time
import random
from qiskit_algorithms import AmplificationProblem
from qiskit_algorithms import Grover
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit
from math import log2, ceil
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

class GroverPathPlanner(Node):
    def __init__(self):
        super().__init__('grover_path_planner')
        self.costmap = None
        self.start_state = None
        self.goal_state = None
        self.costmap_subscriber = self.create_subscription(OccupancyGrid, 'costmap', self.costmap_callback, 1)

    def costmap_callback(self, msg):
        self.get_logger().info('Costmap received')
        self.costmap = msg
        self.initialize_start_and_goal()
        self.run_grover()

    def initialize_start_and_goal(self):
        # Set the start and goal states based on your requirements
        self.start_state = 0
        self.goal_state = self.costmap.info.width - 1

    def is_valid_state(self, state):
        x = state // self.costmap.info.width
        y = state % self.costmap.info.width
        index = x * self.costmap.info.width + y
        return self.costmap.data[index] == 0

    def find_path_grover(self, start_state, goal_state):
        valid_states = [state for state in range(self.costmap.info.width * self.costmap.info.height) if self.is_valid_state(state)]
        num_qubits = ceil(log2(len(valid_states)))

        oracle = QuantumCircuit(num_qubits)
        goal_state_index = valid_states.index(goal_state)
        goal_state_binary = format(goal_state_index, f"0{num_qubits}b")
        oracle.z(range(num_qubits))
        for i in range(num_qubits):
            if goal_state_binary[i] == '0':
                oracle.x(i)
        oracle.h(num_qubits - 1)
        oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        oracle.h(num_qubits - 1)
        for i in range(num_qubits):
            if goal_state_binary[i] == '0':
                oracle.x(i)

        def is_good_state(measurement):
            state_index = int(measurement, 2)
            if state_index < len(valid_states):
                state = valid_states[state_index]
                return state == goal_state
            return False

        state_preparation = QuantumCircuit(num_qubits)
        state_preparation.h(range(num_qubits))

        problem = AmplificationProblem(oracle, state_preparation=state_preparation, is_good_state=is_good_state)
        grover = Grover(sampler=Sampler())
        result = grover.amplify(problem)

        if result.oracle_evaluation:
            state_index = int(result.top_measurement, 2)
            return valid_states[state_index]
        else:
            return None

    def run_grover(self):
        if self.costmap is None or self.start_state is None or self.goal_state is None:
            self.get_logger().warning('Costmap or start/goal state not initialized')
            return

        path = self.find_path_grover(self.start_state, self.goal_state)
        if path is not None:
            self.get_logger().info(f"Most likely path: {path}")
            path_x = path // self.costmap.info.width
            path_y = path % self.costmap.info.width
            self.get_logger().info(f"Path coordinates: ({path_x}, {path_y})")
        else:
            self.get_logger().info("No valid path found.")

def main(args=None):
    rclpy.init(args=args)
    grover_path_planner = GroverPathPlanner()
    rclpy.spin(grover_path_planner)
    grover_path_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()