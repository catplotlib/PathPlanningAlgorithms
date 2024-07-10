import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import numpy as np
 
class CostMapNode(Node):
    def __init__(self):
        super().__init__('cost_map_node')
        self.publisher = self.create_publisher(OccupancyGrid, 'costmap', 10)
        self.timer = self.create_timer(2.0, self.update_costmap)
        self.costmap_msg = OccupancyGrid()
        self.initialize_costmap()
 
    def initialize_costmap(self):
        self.costmap_msg.header = Header()
        self.costmap_msg.header.frame_id = 'map'
        self.costmap_msg.info.resolution = 1.0  # meters per cell
        self.costmap_msg.info.width = 10  # number of cells in X direction
        self.costmap_msg.info.height = 10  # number of cells in Y direction
        self.costmap_msg.info.origin.position.x = -5.0
        self.costmap_msg.info.origin.position.y = -5.0
        self.costmap_msg.info.origin.orientation.w = 1.0
        self.costmap_msg.data = [0] * (self.costmap_msg.info.width * self.costmap_msg.info.height)
        self.add_obstacles()
 
    def add_obstacles(self):
        # Create a 10x10 grid
        grid_size = 10
        cost_map = np.ones((grid_size, grid_size), dtype=np.int8)
 
        # Define danger zones (high cost areas)
        danger_zones = [(2, 3), (6, 8), (7, 2), (8, 8)]
        for zone in danger_zones:
            cost_map[zone] = 100  # High cost (100) for danger zones
 
        # Define goal zone (top right corner)
        goal_zone = (4, -4)
        cost_map[goal_zone] = 0  # Low cost (0) for goal
 
        # Update the costmap message data
        self.costmap_msg.data = cost_map.flatten().tolist()
 
    def update_costmap(self):
        self.costmap_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.costmap_msg)
        self.get_logger().info('Updated costmap published...')
 
def main(args=None):
    rclpy.init(args=args)
    node = CostMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()