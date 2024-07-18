import rospy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import cv2
from cv_bridge import CvBridge
import numpy as np

class ColorTieDetect:
   def __init__(self):
       rospy.init_node('color_tie_detect_node')
       self.bridge = CvBridge()

       # Subscribers for color images and point cloud
       rospy.Subscriber("/camera/color/image_raw", Image, self.color_image_callback)
       rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.point_cloud_callback)

       self.color_image = None
       self.point_cloud = None

       self.coordinates = []  # List to store coordinates

       rate = rospy.Rate(30)
       while not rospy.is_shutdown():
           if self.color_image is not None and self.point_cloud is not None:
               self.detect_color_tie(self.color_image, self.point_cloud)
           rate.sleep()

   def color_image_callback(self, msg):
       self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

   def point_cloud_callback(self, msg):
       self.point_cloud = msg

   def detect_color_tie(self, color_img, point_cloud):
       hue_image = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
       low_range = np.array([0, 123, 100])
       high_range = np.array([5, 255, 255])
       th = cv2.inRange(hue_image, low_range, high_range)
       dilated = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
       dilated = cv2.erode(dilated, None, iterations=2)
       dilated = cv2.dilate(dilated, None, iterations=2)

       (cnts, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       closed_cs = sorted(cnts, key=cv2.contourArea, reverse=True)

       centers = []
       for c in closed_cs[:2]:  # Limit to the first 2 largest contours
           rect = cv2.minAreaRect(c)
           box = np.int0(cv2.boxPoints(rect))
           center = tuple(np.mean(np.array(box), axis=0).reshape(-1).astype(np.uint32))
           cv2.circle(color_img, center, 5, (0, 255, 0), 2)

           z = self.get_depth_at_pixel(point_cloud, center)
           centers.append((*center, z))

           # Store the detected coordinates
           if len(self.coordinates) < 2:  # Limit to two coordinates
               self.coordinates.append((*center, z))

       # Print the detected coordinates if we have exactly two
       if len(self.coordinates) == 2:
           print("Start Coordinate: ", self.coordinates[0])
           print("Goal Coordinate: ", self.coordinates[1])

       cv2.imshow('original_image', color_img)
       cv2.waitKey(1)

       cv2.imshow('color_tie_detect', dilated)
       cv2.waitKey(1)

   def get_depth_at_pixel(self, point_cloud, pixel):
       x, y = pixel
       # Convert pixel coordinates to point cloud index
       pc_gen = pc2.read_points(point_cloud, skip_nans=True, field_names=("x", "y", "z"))
       closest_point = None
       min_dist = float('inf')
       for point in pc_gen:
           u, v = self.project_point_to_pixel(point, point_cloud.width, point_cloud.height)
           dist = np.sqrt((u - x) ** 2 + (v - y) ** 2)
           if dist < min_dist:
               min_dist = dist
               closest_point = point
       return closest_point[2] if closest_point is not None else 0.0

   def project_point_to_pixel(self, point, width, height):
       fx = fy = 525.0  # Example values, replace with your camera's intrinsics
       cx = width / 2
       cy = height / 2
       x = int((point[0] * fx) / point[2] + cx)
       y = int((point[1] * fy) / point[2] + cy)
       return x, y

if __name__ == '__main__':
   cd = ColorTieDetect()
   rospy.spin()



