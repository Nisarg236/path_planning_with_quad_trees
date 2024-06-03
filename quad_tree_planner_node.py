import rospy
import numpy as np
from itertools import count
from nav_msgs.srv import GetPlan, GetPlanResponse
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from grid_to_graph import *

class qtreePlanner:
    def __init__(self):
        rospy.init_node('custom_global_planner')

        # Parameters
        self.json_file = rospy.get_param('~graph_json_file', 'graph.json')
        self.grid = None
        self.graph = load_graph_from_json(self.json_file)

        self.map_resolution = 0.05  # 1 pixel = 0.05 meters

        # Subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)

        # Publishers
        self.path_pub = rospy.Publisher('/custom_global_planner/path', Path, queue_size=10)

        # Services
        self.make_plan_srv = rospy.Service('/custom_global_planner/make_plan', GetPlan, self.make_plan_callback)
        self.current_pose = None

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def map_callback(self, msg):
        self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution

    def pixel_to_world(self, pixel):
        """Converts pixel coordinates to world coordinates."""
        return (pixel[0] * self.map_resolution, pixel[1] * self.map_resolution)

    def world_to_pixel(self, world):
        """Converts world coordinates to pixel coordinates."""
        return (int(world[0] / self.map_resolution), int(world[1] / self.map_resolution))

    def make_plan_callback(self, req):
        start_world = (req.start.pose.position.x, req.start.pose.position.y)
        goal_world = (req.goal.pose.position.x, req.goal.pose.position.y)

        start_pixel = self.world_to_pixel(start_world)
        goal_pixel = self.world_to_pixel(goal_world)

        # Find the high-level path
        high_level_path = find_high_level_path(self.graph, start_pixel, goal_pixel)

        # Add the start and end pixels to the high-level path
        high_level_path = [start_pixel] + high_level_path + [goal_pixel]

        # Convert the high-level path to a detailed pixel-level path
        pixel_path = high_level_path_to_pixels(high_level_path, self.graph)

        # Convert the pixel path to a ROS Path message
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"

        for (x, y) in pixel_path:
            world_x, world_y = self.pixel_to_world((x, y))
            pose = PoseStamped()
            pose.pose.position = Point(world_x, world_y, 0)
            pose.pose.orientation = Quaternion(0, 0, 0, 1)
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        return GetPlanResponse(path=path_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    planner = qtreePlanner()
    planner.run()