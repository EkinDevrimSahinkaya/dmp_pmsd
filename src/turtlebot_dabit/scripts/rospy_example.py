import os
import argparse

import cv2
import numpy as np
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
   
    bag_file = "/home/pmsd/Downloads/demo1.bag";
    output_dir = "/home/pmsd/Downloads";
    image_topic = "/camera/color/image_raw"

 
    print("Extract images from %s on topic %s into %s" % (bag_file,
                                                          image_topic, output_dir))

    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)
        print ("Wrote image %i" % count)

        count += 1

    bag.close()
    # Load image


    return

if __name__ == '__main__':
    main()
    im = cv2.imread('/home/pmsd/Downloads/image.png')

# Define the blue colour we want to find - remember OpenCV uses BGR ordering
Sblue = [202,19,32]
Fblue = [173,9,15]



# Get X and Y coordinates of all blue pixels
Ys, Xs = np.where(np.all(im==Sblue,axis=2))
Yf, Xf = np.where(np.all(im==Fblue,axis=2))

print("start coordinates")
print(Xs,Ys)
print("finished coordinates")
print(Xf,Yf)
