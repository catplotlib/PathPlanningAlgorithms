import math
import random
import time
import rclpy
from rclpy.node import Node as RosNode
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped

class PRMNode(RosNode):
    def __init__(self):
        super().__init__('prm_node')
        self.subscription = self.create_subscription(
            OccupancyGrid, 'costmap', self.costmap_callback, 10)
        self.start_pub = self.create_publisher(PoseStamped, 'start', 10)
        self.goal_pub = self.create_publisher(PoseStamped, 'goal', 10)
        self.path_pub = self.create_publisher(Path, 'path', 10)
        self.costmap = None
        self.start = None
        self.goal = None

    def costmap_callback(self, msg):
        self.get_logger().info('Costmap received')
        self.costmap = msg
        self.initialize_start_and_goal()
        self.run_prm()

    def initialize_start_and_goal(self):
        # Initialize start
        start_msg = PoseStamped()
        start_msg.header.stamp = self.get_clock().now().to_msg()
        start_msg.header.frame_id = self.costmap.header.frame_id
        start_msg.pose.position.x = -4.0  # Example start position
        start_msg.pose.position.y = -4.0
        start_msg.pose.orientation.w = 1.0
        self.start = (int((start_msg.pose.position.x - self.costmap.info.origin.position.x) / self.costmap.info.resolution),
                      int((start_msg.pose.position.y - self.costmap.info.origin.position.y) / self.costmap.info.resolution))
        self.start_pub.publish(start_msg)

        # Initialize goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = self.costmap.header.frame_id
        goal_msg.pose.position.x = 4.0  # Example goal position
        goal_msg.pose.position.y = 4.0
        goal_msg.pose.orientation.w = 1.0
        self.goal = (int((goal_msg.pose.position.x - self.costmap.info.origin.position.x) / self.costmap.info.resolution),
                     int((goal_msg.pose.position.y - self.costmap.info.origin.position.y) / self.costmap.info.resolution))
        self.goal_pub.publish(goal_msg)

        self.get_logger().info(f'Start position: {self.start}')
        self.get_logger().info(f'Goal position: {self.goal}')

    def run_prm(self):
        if self.costmap and self.start and self.goal:
            start_time = time.time()
            path = prm(self.start, self.goal, self.costmap.data, self.costmap.info.width, self.costmap.info.height, 1000, 10)
            end_time = time.time()
            computation_time = end_time - start_time

            if path:
                world_path = [self.index_to_coord(index, self.costmap.info) for index in path]
                path_length = self.calculate_path_length(world_path)
                self.publish_path(world_path)
                self.get_logger().info(f'Path found! Length: {path_length:.2f} meters, Computation time: {computation_time:.4f} seconds')
            else:
                self.get_logger().info('No path found')

    def calculate_path_length(self, path):
        length = 0.0
        for i in range(1, len(path)):
            length += math.sqrt((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2)
        return length

    def index_to_coord(self, index, info):
        x = index[0] * info.resolution + info.origin.position.x
        y = index[1] * info.resolution + info.origin.position.y
        return x, y

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
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
        self.get_logger().info('Path published')

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.edges = []
        self.parent = None

def prm(start, goal, costmap, width, height, num_samples, connection_radius):
    nodes = [Node(start[0], start[1]), Node(goal[0], goal[1])]

    # Sampling phase
    while len(nodes) < num_samples:
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        if costmap[y * width + x] != 100:
            nodes.append(Node(x, y))

    # Connection phase
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if distance(nodes[i], nodes[j]) <= connection_radius and is_collision_free(nodes[i], nodes[j], costmap, width, height):
                nodes[i].edges.append(nodes[j])
                nodes[j].edges.append(nodes[i])

    # Find the shortest path using A*
    start_node = next((n for n in nodes if n.x == start[0] and n.y == start[1]), None)
    goal_node = next((n for n in nodes if n.x == goal[0] and n.y == goal[1]), None)

    if start_node and goal_node:
        open_list = [(0, start_node)]
        closed_list = []

        while open_list:
            current_cost, current_node = min(open_list, key=lambda x: x[0])
            if current_node == goal_node:
                path = []
                while current_node:
                    path.append((current_node.x, current_node.y))
                    current_node = current_node.parent
                return path[::-1]

            open_list.remove((current_cost, current_node))
            closed_list.append(current_node)

            for neighbor in current_node.edges:
                if neighbor in closed_list:
                    continue

                tentative_cost = current_cost + distance(current_node, neighbor)
                if not any(n == neighbor for _, n in open_list) or tentative_cost < next(c for c, n in open_list if n == neighbor):
                    neighbor.parent = current_node
                    open_list.append((tentative_cost, neighbor))

    return None

def distance(node1, node2):
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

def is_collision_free(node1, node2, costmap, width, height):
    x1, y1 = node1.x, node1.y
    x2, y2 = node2.x, node2.y
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while x1 != x2 or y1 != y2:
        if costmap[y1 * width + x1] == 100:
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return True

def main(args=None):
    rclpy.init(args=args)
    node = PRMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()