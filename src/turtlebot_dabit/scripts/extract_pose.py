import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rosbag

# Define the DMP class as provided
class DMP(object):
    def __init__(self, pastor_mod=False):
        self.pastor_mod = pastor_mod
        # Transformation system
        self.alpha = 25.0
        self.beta = self.alpha / 4.0
        # Canonical system
        self.alpha_t = self.alpha / 3.0
        # Obstacle avoidance
        self.gamma_o = 1000.0
        self.beta_o = 20.0 / np.pi

    def phase(self, n_steps, t=None):
        phases = np.exp(-self.alpha_t * np.linspace(0, 1, n_steps))
        if t is None:
            return phases
        else:
            return phases[t]

    def spring_damper(self, x0, g, tau, s, X, Xd):
        if self.pastor_mod:
            mod = -self.beta * (g - x0) * s
        else:
            mod = 0.0
        return self.alpha * (self.beta * (g - X) - tau * Xd + mod) / tau ** 2

    def forcing_term(self, x0, g, tau, w, s, X, scale=False):
        n_features = w.shape[1]
        f = np.dot(w, self._features(tau, n_features, s))
        if scale:
            f *= g - x0

        if X.ndim == 3:
            F = np.empty_like(X)
            F[:, :] = f
            return F
        else:
            return f

    def _features(self, tau, n_features, s):
        if n_features == 0:
            return np.array([])
        elif n_features == 1:
            return np.array([1.0])
        c = self.phase(n_features)
        h = np.diff(c)
        h = np.hstack((h, [h[-1]]))
        phi = np.exp(-h * (s - c) ** 2)
        return s * phi / phi.sum()

    def obstacle(self, o, X, Xd):
        if X.ndim == 1:
          X = X[np.newaxis, np.newaxis, :]
        if Xd.ndim == 1:
          Xd = Xd[np.newaxis, np.newaxis, :]

        C = np.zeros_like(X)
        R = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                      [np.sin(np.pi / 2.0),  np.cos(np.pi / 2.0)]])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                obstacle_diff = o - X[i, j]
                theta = (np.arccos(obstacle_diff.dot(Xd[i, j]) /
                                   (np.linalg.norm(obstacle_diff) *
                                    np.linalg.norm(Xd[i, j]) + 1e-10)))
                C[i, j] = (self.gamma_o * R.dot(Xd[i, j]) * theta *
                           np.exp(-self.beta_o * theta))

        return np.squeeze(C)

    def imitate(self, X, tau, n_features):
        n_steps, n_dims = X.shape
        dt = tau / float(n_steps - 1)
        g = X[:, -1]

        Xd = np.vstack((np.zeros((1, n_dims)), np.diff(X, axis=0) / dt))
        Xdd = np.vstack((np.zeros((1, n_dims)), np.diff(Xd, axis=0) / dt))

        F = tau * tau * Xdd - self.alpha * (self.beta * (g[:, np.newaxis] - X)
                                            - tau * Xd)

        design = np.array([self._features(tau, n_features, s)
                           for s in self.phase(n_steps)])
        from sklearn.linear_model import Ridge
        lr = Ridge(alpha=1.0, fit_intercept=False)
        lr.fit(design, F)
        w = lr.coef_

        return w


def trajectory(dmp, w, x0, g, tau, dt, o=None, shape=True, avoidance=False,
               verbose=0):
    if verbose >= 1:
        print("Trajectory with x0 = %s, g = %s, tau=%.2f, dt=%.3f"
              % (x0, g, tau, dt))

    x = x0.copy()
    xd = np.zeros_like(x, dtype=np.float64)
    xdd = np.zeros_like(x, dtype=np.float64)
    X = [x0.copy()]
    Xd = [xd.copy()]
    Xdd = [xdd.copy()]

    internal_dt = min(0.001, dt)
    n_internal_steps = int(tau / internal_dt)
    steps_between_measurement = int(dt / internal_dt)

    t = 0.5 * internal_dt
    ti = 0
    S = dmp.phase(n_internal_steps + 1)
    while t < tau:
        t += internal_dt
        ti += 1
        s = S[ti]

        x += internal_dt * xd
        xd += internal_dt * xdd

        sd = dmp.spring_damper(x0, g, tau, s, x, xd)
        f = dmp.forcing_term(x0, g, tau, w, s, x) if shape else 0.0
        C = dmp.obstacle(o, x, xd) if avoidance else 0.0
        xdd = sd + f + C

        if ti % steps_between_measurement == 0:
            X.append(x.copy())
            Xd.append(xd.copy())
            Xdd.append(xdd.copy())

    return np.array(X), np.array(Xd), np.array(Xdd)


def extract_xyz_from_rosbag(bag_file, topic_name):
    positions = []
    bag = rosbag.Bag(bag_file)
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if topic == topic_name:
            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z
            positions.append([x, y, z])
    bag.close()
    return np.array(positions).T

def plot_trajectories(original, generated):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(original[0, :], original[1, :], original[2, :], label='Original', color='b')
    ax.plot(generated[0, :], generated[1, :], generated[2, :], label='Generated DMP', color='r', linestyle='dashed')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def plot_original_trajectory(original):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(original[0, :], original[1, :], original[2, :], label='Original', color='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    bag_file = '/home/pmsd/Downloads/demo_traj2.bag'
    topic_name = '/franka_state_controller/cartesian_pose'

    # Extract positions from the bag
    positions = extract_xyz_from_rosbag(bag_file, topic_name)
    
    # Plot the original trajectory for verification
    plot_original_trajectory(positions)
    
    # Apply DMP to the extracted positions
    n_dmps, n_bfs = positions.shape[0], 100
    tau = 1.0  # Assuming tau to be 1.0 for simplicity, adjust as necessary
    dt = 0.01  # Assuming a time step of 0.01, adjust as necessary
    
    # Initialize DMP
    dmp = DMP(pastor_mod=True)
    
    # Imitate the path
    w = dmp.imitate(positions.T, tau, n_bfs)
    
    # Generate a trajectory using the learned weights
    x0 = positions[:, 0]
    g = positions[:, -1]
    y_track, _, _ = trajectory(dmp, w, x0, g, tau, dt)
    
    # Transpose y_track to match the original format
    y_track = y_track.T
    
    # Plot the results
    plot_trajectories(positions, y_track)

