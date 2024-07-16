'''
Author: Dianye Huang
Date: 2024-05-31 11:06:29
LastEditors: Dianye Huang
LastEditTime: 2024-05-31 19:47:41
Description:
'''

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dmp import DMPs_cartesian

# Step 1. Extract trajectory as numpy.array
trans_list = list()
bag_file = '/home/pmsd/Downloads/demo_traj2.bag'

with rosbag.Bag(bag_file, 'r') as bag:
  topic_name = '/franka_state_controller/cartesian_pose'
  traj_data = bag.read_messages(topic_name)
  for _, msg, t in traj_data:
      if msg is not None:
          translation = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
          trans_list.append(translation)
trans_arr = np.array(trans_list)
print("Trajectory shape:", trans_arr.shape)

# Step 2. DMP fitting
x_des = trans_arr

# Create DMPs in Cartesian space
dmps_cartesian = DMPs_cartesian(n_dmps=3, n_bfs=100, dt=0.01, x_0=x_des[0], x_goal=x_des[-1], T=1.0, K=1050, D=None, w=None)

# Fit DMP to the extracted trajectory
dmps_cartesian.imitate_path(x_des)

# Generate new trajectory with original start and goal
gen_traj, _, _, _ = dmps_cartesian.rollout()

# Step 3. Plot to show the trajectory before and after fitting
# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_des[:, 0], x_des[:, 1], x_des[:, 2], label='Original Robot trajectory')
ax.plot(gen_traj[:, 0], gen_traj[:, 1], gen_traj[:, 2], label='Generated trajectory (Original)')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()

# 2D plot for each axis
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(x_des[:, 0], label='Original x-axis')
ax1.plot(gen_traj[:, 0], label='Generated x-axis (Original)')

ax1.set_xlabel('x-axis')
ax1.legend()
ax1.grid()

ax2.plot(x_des[:, 1], label='Original y-axis')
ax2.plot(gen_traj[:, 1], label='Generated y-axis (Original)')

ax2.set_xlabel('y-axis')
ax2.legend()
ax2.grid()

ax3.plot(x_des[:, 2], label='Original z-axis')
ax3.plot(gen_traj[:, 2], label='Generated z-axis (Original)')

ax3.set_xlabel('z-axis')
ax3.legend()
ax3.grid()

plt.show()



