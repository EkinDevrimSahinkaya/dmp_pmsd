import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

import numpy as np


class ColorTieDetect:
  def __init__(self):
      rospy.init_node('color_tie_detect_node')
      rospy.Subscriber("/camera/color/image_raw", Image, self.cimg_cb, queue_size=20)

      self.wait_update_img = True
      self.bridgeC = CvBridge()
      self.coordinates = []  # List to store coordinates
      rospy.wait_for_message("/camera/color/image_raw", Image)

      rate = rospy.Rate(30)
      while not rospy.is_shutdown():
          rate.sleep()
          if not self.wait_update_img:
              self.color_tie_detection(self.cimg)
              self.wait_update_img = True

  def color_tie_detection(self, img):
      hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      low_range = np.array([0, 123, 100])
      high_range = np.array([5, 255, 255])
      th = cv2.inRange(hue_image, low_range, high_range)
      # dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
      dilated = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
      dilated = cv2.erode(dilated, None, iterations=2)
      dilated = cv2.dilate(dilated, None, iterations=2)

      (cnts, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      closed_cs = sorted(cnts, key=cv2.contourArea, reverse=True)
      # print('number of closed contours:', len(closed_cs))

      centers = list()
      for c in closed_cs[:2]:  # Limit to the first 2 largest contours
          rect = cv2.minAreaRect(c)
          box = np.int0(cv2.boxPoints(rect))
          # cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
          center = tuple(np.mean(np.array(box), axis=0).reshape(-1).astype(np.uint32))
          cv2.circle(img, center, 5, (0, 255, 0), 2)

          centers.append(center)

          # Store the detected coordinates
          if len(self.coordinates) < 2:  # Limit to two coordinates
              self.coordinates.append(center)

      # Print the detected coordinates if we have exactly two
      if len(self.coordinates) == 2:
          print("Start Coordinate: ", self.coordinates[0])
          print("Goal Coordinate: ", self.coordinates[1])

      cv2.imshow('original_image', img)
      cv2.waitKey(1)

      cv2.imshow('color_tie_detect', dilated)
      cv2.waitKey(1)

  def cimg_cb(self, msg):
      if self.wait_update_img:
          self.wait_update_img = False
          self.cimg = self.bridgeC.imgmsg_to_cv2(msg, "bgr8")


if __name__ == '__main__':
  cd = ColorTieDetect()



