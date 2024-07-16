import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.linalg
import copy

from cs import CanonicalSystem
from exponential_integration import exp_eul_step
from exponential_integration import phi1
from derivative_matrices import compute_D1, compute_D2

class DMPs_cartesian(object):
   def __init__(self, n_dmps=3, n_bfs=50, dt=0.01, x_0=None, x_goal=None, T=1.0, K=1050, D=None, w=None, tol=0.01, alpha_s=4.0, basis='gaussian', **kwargs):
       self.tol = tol
       self.n_dmps = n_dmps
       self.n_bfs = n_bfs

       self.K = K
       if D is None:
           D = 2 * np.sqrt(self.K)
       self.D = D

       self.cs = CanonicalSystem(dt=dt, run_time=T, alpha_s=alpha_s)
       self.compute_linear_part()

       if x_0 is None:
           x_0 = np.zeros(self.n_dmps)
       if x_goal is None:
           x_goal = np.ones(self.n_dmps)
       self.x_0 = x_0
       self.x_goal = x_goal
       self.basis = basis
       self.reset_state()
       self.gen_centers()
       self.gen_width()

       if w is None:
           w = np.zeros([self.n_dmps, self.n_bfs + 1])
       self.w = w

   def compute_linear_part(self):
       self.linear_part = np.zeros([2 * self.n_dmps, 2 * self.n_dmps])
       self.linear_part[range(0, 2 * self.n_dmps, 2), range(0, 2 * self.n_dmps, 2)] = -self.D
       self.linear_part[range(0, 2 * self.n_dmps, 2), range(1, 2 * self.n_dmps, 2)] = -self.K
       self.linear_part[range(1, 2 * self.n_dmps, 2), range(0, 2 * self.n_dmps, 2)] = 1.

   def gen_centers(self):
       self.c = np.exp(-self.cs.alpha_s * self.cs.run_time * ((np.cumsum(np.ones([1, self.n_bfs + 1])) - 1) / self.n_bfs))

   def gen_psi(self, s):
       c = np.reshape(self.c, [self.n_bfs + 1, 1])
       w = np.reshape(self.width, [self.n_bfs + 1, 1])
       if self.basis == 'gaussian':
           xi = w * (s - c) ** 2
           psi_set = np.exp(-xi)
       else:
           xi = np.abs(w * (s - c))
           if self.basis == 'mollifier':
               psi_set = (np.exp(-1.0 / (1.0 - xi ** 2))) * (xi < 1.0)
           elif self.basis == 'wendland2':
               psi_set = ((1.0 - xi) ** 2) * (xi < 1.0)
           elif self.basis == 'wendland3':
               psi_set = ((1.0 - xi) ** 3) * (xi < 1.0)
           elif self.basis == 'wendland4':
               psi_set = ((1.0 - xi) ** 4 * (4.0 * xi + 1.0)) * (xi < 1.0)
           elif self.basis == 'wendland5':
               psi_set = ((1.0 - xi) ** 5 * (5.0 * xi + 1)) * (xi < 1.0)
           elif self.basis == 'wendland6':
               psi_set = ((1.0 - xi) ** 6 * (35.0 * xi ** 2 + 18.0 * xi + 3.0)) * (xi < 1.0)
           elif self.basis == 'wendland7':
               psi_set = ((1.0 - xi) ** 7 * (16.0 * xi ** 2 + 7.0 * xi + 1.0)) * (xi < 1.0)
           elif self.basis == 'wendland8':
               psi_set = ((1.0 - xi) ** 8 * (32.0 * xi ** 3 + 25.0 * xi ** 2 + 8.0 * xi + 1.0)) * (xi < 1.0)
       psi_set = np.nan_to_num(psi_set)
       return psi_set

   def gen_width(self):
       if self.basis == 'gaussian':
           self.width = 1.0 / np.diff(self.c) / np.diff(self.c)
           self.width = np.append(self.width, self.width[-1])
       else:
           self.width = 1.0 / np.diff(self.c)
           self.width = np.append(self.width[0], self.width)

   def imitate_path(self, x_des, dx_des=None, ddx_des=None, t_des=None, g_w=True, **kwargs):
       self.x_0 = x_des[0].copy()
       self.x_goal = x_des[-1].copy()
       print(f"Imitate Path - Start Point: {self.x_0}")
       print(f"Imitate Path - Goal Point: {self.x_goal}")

       if t_des is None:
           t_des = np.linspace(0, self.cs.run_time, x_des.shape[0])
       else:
           t_des -= t_des[0]
           t_des /= t_des[-1]
           t_des *= self.cs.run_time
       time = np.linspace(0., self.cs.run_time, self.cs.timesteps)

       path_gen = scipy.interpolate.interp1d(t_des, x_des.transpose())
       path = path_gen(time)
       x_des = path.transpose()

       if dx_des is None:
           D1 = compute_D1(self.cs.timesteps, self.cs.dt)
           dx_des = np.dot(D1, x_des)
       else:
           dpath = np.zeros([self.cs.timesteps, self.n_dmps])
           dpath_gen = scipy.interpolate.interp1d(t_des, dx_des)
           dpath = dpath_gen(time)
           dx_des = dpath.transpose()
       if ddx_des is None:
           D2 = compute_D2(self.cs.timesteps, self.cs.dt)
           ddx_des = np.dot(D2, x_des)
       else:
           ddpath = np.zeros([self.cs.timesteps, self.n_dmps])
           ddpath_gen = scipy.interpolate.interp1d(t_des, ddx_des)
           ddpath = ddpath_gen(time)
           ddx_des = ddpath.transpose()

       s_track = self.cs.rollout()
       f_target = ((ddx_des / self.K - (self.x_goal - x_des) + self.D / self.K * dx_des).transpose() + np.reshape((self.x_goal - self.x_0), [self.n_dmps, 1]) * s_track)
       if g_w:
           self.gen_weights(f_target)
           self.reset_state()
           self.learned_position = self.x_goal - self.x_0
       return f_target

   def gen_weights(self, f_target):
       s_track = self.cs.rollout()
       psi_track = self.gen_psi(s_track)
       sum_psi = np.sum(psi_track, 0)
       P = psi_track / sum_psi * s_track
       self.w = np.nan_to_num(f_target @ np.linalg.pinv(P))

   def reset_state(self, v0=None, **kwargs):
       self.x = self.x_0.copy()
       if v0 is None:
           v0 = 0.0 * self.x_0
       self.dx = v0
       self.ddx = np.zeros(self.n_dmps)
       self.cs.reset_state()

   def rollout(self, tau=1.0, v0=None, **kwargs):
       if v0 is None:
           v0 = 0.0 * self.x_0
       self.reset_state(v0=v0)
       x_track = np.array([self.x_0])
       dx_track = np.array([v0])
       t_track = np.array([0])
       state = np.zeros(2 * self.n_dmps)
       state[range(0, 2 * self.n_dmps, 2)] = copy.deepcopy(v0)
       state[range(1, 2 * self.n_dmps + 1, 2)] = copy.deepcopy(self.x_0)
       psi = self.gen_psi(self.cs.s)
       f0 = (np.dot(self.w, psi[:, 0])) / (np.sum(psi[:, 0])) * self.cs.s
       f0 = np.nan_to_num(f0)
       ddx_track = np.array([-self.D * v0 + self.K * f0])
       err = np.linalg.norm(state[range(1, 2 * self.n_dmps + 1, 2)] - self.x_goal)
       P = phi1(self.cs.dt * self.linear_part / tau)
       while err > self.tol:
           psi = self.gen_psi(self.cs.s)
           f = (np.dot(self.w, psi[:, 0])) / (np.sum(psi[:, 0])) * self.cs.s
           f = np.nan_to_num(f)
           beta = np.zeros(2 * self.n_dmps)
           beta[range(0, 2 * self.n_dmps, 2)] = self.K * (self.x_goal * (1.0 - self.cs.s) + self.x_0 * self.cs.s + f) / tau
           vect_field = np.dot(self.linear_part / tau, state) + beta
           state += self.cs.dt * np.dot(P, vect_field)
           x_track = np.append(x_track, np.array([state[range(1, 2 * self.n_dmps + 1, 2)]]), axis=0)
           dx_track = np.append(dx_track, np.array([state[range(0, 2 * self.n_dmps, 2)]]), axis=0)
           t_track = np.append(t_track, t_track[-1] + self.cs.dt)
           err = np.linalg.norm(state[range(1, 2 * self.n_dmps + 1, 2)] - self.x_goal)
           self.cs.step(tau=tau)
           ddx_track = np.append(ddx_track, np.array([self.K * (self.x_goal - x_track[-1]) - self.D * dx_track[-1] - self.K * (self.x_goal - self.x_0) * self.cs.s + self.K * f]), axis=0)
       print(f"Rollout - Final Error: {err}")
       print(f"Rollout - Final State: {state}")
       return x_track, dx_track, ddx_track, t_track

   def step(self, tau=1.0, error=0.0, external_force=None, adapt=False, tols=None, **kwargs):
       if tols is None:
           tols = [1e-03, 1e-06]
       error_coupling = 1.0 + error
       alpha_tilde = -self.cs.alpha_s / tau / error_coupling
       state = np.zeros(2 * self.n_dmps)
       state[0::2] = self.dx
       state[1::2] = self.x
       A_m = self.linear_part / tau

       def beta_s(s, x, v):
           psi = self.gen_psi(s)
           f = (np.dot(self.w, psi[:, 0])) / (np.sum(psi[:, 0])) * self.cs.s
           f = np.nan_to_num(f)
           out = np.zeros(2 * self.n_dmps)
           out[0::2] = self.K * (self.x_goal * (1.0 - s) + self.x_0 * s + f)
           if external_force is not None:
               out[0::2] += external_force(x, v)
           return out / tau

       flag_tol = False
       while not flag_tol:
           s1 = copy.deepcopy(self.cs.s)
           s2 = s1 * np.exp(-alpha_tilde * self.cs.dt * 1.0 / 2.0)
           s3 = s1 * np.exp(-alpha_tilde * self.cs.dt * 3.0 / 4.0)
           s4 = s1 * np.exp(-alpha_tilde * self.cs.dt)
           xi1 = np.dot(A_m, state) + beta_s(s1, state[1::2], state[0::2])
           xi2 = np.dot(A_m, state + self.cs.dt * xi1 * 1.0 / 2.0) + beta_s(s2, state[1::2] + self.cs.dt * xi1[1::2] * 1.0 / 2.0, state[0::2] + self.cs.dt * xi1[0::2] * 1.0 / 2.0)
           xi3 = np.dot(A_m, state + self.cs.dt * xi2 * 3.0 / 4.0) + beta_s(s3, state[1::2] + self.cs.dt * xi2[1::2] * 3.0 / 4.0, state[0::2] + self.cs.dt * xi2[0::2] * 3.0 / 4.0)
           xi4 = np.dot(A_m, state + self.cs.dt * (2.0 * xi1 + 3.0 * xi2 + 4.0 * xi3) / 9.0) + beta_s(s4, state[1::2] + self.cs.dt * (2.0 * xi1[1::2] + 3.0 * xi2[1::2] + 4.0 * xi3[1::2]) / 9.0, state[0::2] + self.cs.dt * (2.0 * xi1[0::2] + 3.0 * xi2[0::2] + 4.0 * xi3[0::2]) / 9.0)
           y_ord2 = state + self.cs.dt * (2.0 * xi1 + 3.0 * xi2 + 4.0 * xi3) / 9.0
           y_ord3 = state + self.cs.dt * (7.0 * xi1 + 6.0 * xi2 + 8.0 * xi3 + 3.0 * xi4) / 24.0
           if (np.linalg.norm(y_ord2 - y_ord3) < tols[0] * np.linalg.norm(state) + tols[1]) or (not adapt):
               flag_tol = True
               state = copy.deepcopy(y_ord3)
           else:
               self.cs.dt /= 1.1
       self.cs.step(tau=tau, error_coupling=error_coupling)
       self.x = copy.deepcopy(state[1::2])
       self.dx = copy.deepcopy(state[0::2])
       psi = self.gen_psi(self.cs.s)
       f = (np.dot(self.w, psi[:, 0])) / (np.sum(psi[:, 0])) * self.cs.s
       f = np.nan_to_num(f)
       self.ddx = (self.K * (self.x_goal - self.x) - self.D * self.dx - self.K * (self.x_goal - self.x_0) * self.cs.s + self.K * f) / tau
       if external_force is not None:
           self.ddx += external_force(self.x, self.dx) / tau
       return self.x, self.dx, self.ddx



