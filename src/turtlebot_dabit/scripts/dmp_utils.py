import numpy as np
from math import sqrt, exp
import matplotlib.pyplot as plt

# Weighted truncated approximators
class TruncGaussApprox:
   def __init__(self, n_kfs: float, th=2, k=2):
       # Meta parameters
       if n_kfs % 2 != 0:  # Make the number of kernel functions odd
           self.n_kfs = n_kfs + 1
       else:
           self.n_kfs = n_kfs
       delta_c = 1 / (self.n_kfs - 1)
       self.centers = np.array(range(self.n_kfs)) * delta_c  # Evenly spaced
       self.width = th / delta_c ** 2
       self.thresh = k / sqrt(self.width)
       # Model parameters
       self.weights = list()
       self.biases = list()

   def truncated_gaussian(self, x, center, width, thresh):
       if abs(x - center) <= thresh:
           return exp(-width / 2 * (x - center) ** 2)
       return 0

   def forward(self, x):
       sum_psi = 0
       weighted_sum_psi = 0
       for c, w, b in zip(self.centers, self.weights, self.biases):
           psi = self.truncated_gaussian(x, c, self.width, self.thresh)
           As = x
           sum_psi += psi
           weighted_sum_psi += psi * (w * x + As * b)
       return weighted_sum_psi / sum_psi + self.f0

   def learn_params(self, x_input, y_output):
       self.weights = list()
       self.biases = list()

       self.f0 = y_output[0]
       y_output = y_output - self.f0

       for c in self.centers:
           Sx2, Sxy = 0, 0
           Spsi = 0
           for x, f in zip(x_input, y_output):
               psi = self.truncated_gaussian(x, c, self.width, self.thresh)
               Sx2 += psi * x * x
               Sxy += psi * x * f
               Spsi += psi

           if Sx2 > -1e-8 and Sx2 < 1e-8:  # Exit
               return False
           w = Sxy / Sx2

           self.weights.append(w)
           self.biases.append(0)
       return True

   def rollout(self, x_vec):
       y_list = list()
       for x in x_vec:
           y_list.append(self.forward(x))
       return y_list

class OneDMP(object):
   def __init__(self, n_kfns=31, th=2, k=2, alpha=25):
       self.approx = TruncGaussApprox(n_kfs=n_kfns, th=th, k=k)
       self.x0 = None
       self.g = None

       self.alpha = alpha
       self.beta = self.alpha / 4.0
       self.tau = 1.0

   def fit(self, traj):
       self.x0 = traj[0]
       self.xg = traj[-1]
       self.num_data = len(traj)
       self.dt = 1.0 / float(self.num_data)

       Xd = np.concatenate((np.zeros((1)), np.diff(traj, axis=0) / self.dt))
       Xdd = np.concatenate((np.zeros((1)), np.diff(Xd, axis=0) / self.dt))
       force = self.tau * self.tau * Xdd - self.alpha * (self.beta * (self.xg - self.x0) - self.tau * Xd)

       input = np.array(range(self.num_data)) / float(self.num_data)
       self.approx.learn_params(x_input=input, y_output=force)

   def rollout(self, dt=1.0 / 50.0, tau=1.0, x0=None, xg=None):
       if xg is None:
           xg = self.xg
       if x0 is None:
           x0 = self.x0

       x = x0
       xd = 0.0
       xdd = 0.0
       xarr = list()

       t = 0
       while t <= 1.0:
           t += dt
           force = self.approx.forward(t)
           xdd = (force + self.alpha * (self.beta * (xg - x0) - self.tau * xd)) / tau ** 2
           xd += dt * xdd
           x += dt * xd
           xarr.append(x)

       return np.array(xarr)



