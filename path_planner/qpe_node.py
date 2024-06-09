import math
import time
import rclpy
from rclpy.node import Node as RosNode
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped

class PotentialFieldsNode(RosNode):
    def __init__(self):
        super().__init__('potential_fields_node')
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
        self.run_potential_fields()

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

    def run_potential_fields(self):
        if self.costmap and self.start and self.goal:
            start_index = self.coord_to_index(self.start, self.costmap.info)
            goal_index = self.coord_to_index(self.goal, self.costmap.info)
            self.get_logger().info(f'Start index: {start_index}')
            self.get_logger().info(f'Goal index: {goal_index}')

            start_time = time.time()
            path = potential_fields(start_index, goal_index, self.costmap.data, self.costmap.info.width, self.costmap.info.height)
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

def potential_fields(start, goal, costmap, width, height):
    current_pos = start
    path = [current_pos]

    while current_pos != goal:
        min_potential = float('inf')
        next_pos = current_pos

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            x = current_pos[0] + dx
            y = current_pos[1] + dy
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            potential = attractive_potential(x, y, goal) + repulsive_potential(x, y, costmap, width, height)
            if potential < min_potential:
                min_potential = potential
                next_pos = (x, y)

        if next_pos == current_pos:
            break
        current_pos = next_pos
        path.append(current_pos)

    if current_pos != goal:
        return None  # If no path is found
    return path

def attractive_potential(x, y, goal):
    return math.sqrt((x - goal[0]) ** 2 + (y - goal[1]) ** 2)

def repulsive_potential(x, y, costmap, width, height):
    potential = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            obstacle_x = x + dx
            obstacle_y = y + dy
            if 0 <= obstacle_x < width and 0 <= obstacle_y < height and costmap[obstacle_y * width + obstacle_x] == 100:
                distance = math.sqrt(dx ** 2 + dy ** 2)
                if distance < 1:
                    if distance == 0:
                        potential += float('inf')  # Assign a large value when distance is zero
                    else:
                        potential += 1.0 / distance
    return potential

def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()