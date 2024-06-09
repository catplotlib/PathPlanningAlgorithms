import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

class OccupancyGridNode(Node):
    def __init__(self):
        super().__init__('occupancy_grid_node')
        self.publisher = self.create_publisher(OccupancyGrid, 'costmap', 10)
        self.timer = self.create_timer(2.0, self.update_costmap)
        self.costmap_msg = OccupancyGrid()
        self.initialize_costmap()

    def initialize_costmap(self):
        self.costmap_msg.header = Header()
        self.costmap_msg.header.frame_id = 'map'
        self.costmap_msg.info.resolution = 0.1  # meters per cell
        self.costmap_msg.info.width = 100  # number of cells in X direction
        self.costmap_msg.info.height = 100  # number of cells in Y direction
        self.costmap_msg.info.origin.position.x = -5.0
        self.costmap_msg.info.origin.position.y = -5.0
        self.costmap_msg.info.origin.position.z = 0.0
        self.costmap_msg.info.origin.orientation.w = 1.0
        self.costmap_msg.data = [0] * (self.costmap_msg.info.width * self.costmap_msg.info.height)
        self.add_obstacles()

    def add_obstacles(self):
        # Example: add a block of obstacles in the center
        center_x, center_y = self.costmap_msg.info.width // 2, self.costmap_msg.info.height // 2
        size = 10
        for i in range(center_x - size // 2, center_x + size // 2):
            for j in range(center_y - size // 2, center_y + size // 2):
                index = i + j * self.costmap_msg.info.width
                self.costmap_msg.data[index] = 100  # Mark as occupied

        # Add a horizontal bar of obstacles
        bar_y = center_y - 15
        bar_length = 20
        for i in range(center_x - bar_length // 2, center_x + bar_length // 2):
            index = i + bar_y * self.costmap_msg.info.width
            self.costmap_msg.data[index] = 100  # Mark as occupied

        # Add a vertical bar of obstacles
        bar_x = center_x + 15
        for j in range(center_y - bar_length // 2, center_y + bar_length // 2):
            index = bar_x + j * self.costmap_msg.info.width
            self.costmap_msg.data[index] = 100  # Mark as occupied

        # Add random obstacles
        import random
        num_random_obstacles = 50
        for _ in range(num_random_obstacles):
            rand_x = random.randint(0, self.costmap_msg.info.width - 1)
            rand_y = random.randint(0, self.costmap_msg.info.height - 1)
            index = rand_x + rand_y * self.costmap_msg.info.width
            self.costmap_msg.data[index] = 100  # Mark as occupied


    def update_costmap(self):
        self.costmap_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.costmap_msg)
        self.get_logger().info('Updated costmap published.')

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
