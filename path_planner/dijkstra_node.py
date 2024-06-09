import math
import time
import rclpy
from rclpy.node import Node as RosNode
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped

class DijkstraNode(RosNode):
    def __init__(self):
        super().__init__('astar_node')
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
        self.run_dijkstra()

    def initialize_start_and_goal(self):
        # Initialize start
        start_msg = PoseStamped()
        start_msg.header.stamp = self.get_clock().now().to_msg()
        start_msg.header.frame_id = self.costmap.header.frame_id
        start_msg.pose.position.x = -4.0  # Example start position
        start_msg.pose.position.y = -4.0
        start_msg.pose.orientation.w = 1.0
        self.start = (start_msg.pose.position.x, start_msg.pose.position.y)
        self.start_pub.publish(start_msg)

        # Initialize goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = self.costmap.header.frame_id
        goal_msg.pose.position.x = 4.0  # Example goal position
        goal_msg.pose.position.y = 4.0
        goal_msg.pose.orientation.w = 1.0
        self.goal = (goal_msg.pose.position.x, goal_msg.pose.position.y)
        self.goal_pub.publish(goal_msg)

        self.get_logger().info(f'Start position: {self.start}')
        self.get_logger().info(f'Goal position: {self.goal}')

    def run_dijkstra(self):
        if self.costmap and self.start and self.goal:
            start_index = self.coord_to_index(self.start, self.costmap.info)
            goal_index = self.coord_to_index(self.goal, self.costmap.info)
            self.get_logger().info(f'Start index: {start_index}')
            self.get_logger().info(f'Goal index: {goal_index}')

            start_time = time.time()
            path = dijkstra(start_index, goal_index, self.costmap.data, self.costmap.info.width, self.costmap.info.height)
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

    def coord_to_index(self, coord, info):
        x, y = coord
        index_x = int((x - info.origin.position.x) / info.resolution)
        index_y = int((y - info.origin.position.y) / info.resolution)
        return index_x, index_y

    def index_to_coord(self, index, info):
        index_x, index_y = index
        x = index_x * info.resolution + info.origin.position.x
        y = index_y * info.resolution + info.origin.position.y
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
    def __init__(self, x, y, cost, parent):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

def dijkstra(start, goal, costmap, width, height):
    open_list = []
    closed_list = []
    start_node = Node(start[0], start[1], 0, None)
    goal_node = Node(goal[0], goal[1], 0, None)
    open_list.append(start_node)

    while open_list:
        current_node = min(open_list, key=lambda node: node.cost)
        if current_node.x == goal_node.x and current_node.y == goal_node.y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        open_list.remove(current_node)
        closed_list.append(current_node)

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            x = current_node.x + dx
            y = current_node.y + dy
            if x < 0 or y < 0 or x >= width or y >= height or costmap[y * width + x] == 100:
                continue
            movement_cost = math.sqrt(dx * dx + dy * dy)
            neighbor = Node(x, y, current_node.cost + movement_cost, current_node)
            if any(n.x == neighbor.x and n.y == neighbor.y for n in closed_list):
                continue

            existing_node = next((n for n in open_list if n.x == neighbor.x and n.y == neighbor.y), None)
            if existing_node and existing_node.cost > neighbor.cost:
                open_list.remove(existing_node)
            if not existing_node or existing_node.cost > neighbor.cost:
                open_list.append(neighbor)

    return None  # If no path is found

def main(args=None):
    rclpy.init(args=args)
    node = DijkstraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()