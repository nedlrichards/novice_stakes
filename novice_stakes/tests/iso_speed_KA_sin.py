import numpy as np
import numexpr as ne
from math import pi
from scipy.optimize import newton
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from novice_stakes import p_sca, initialize_nuttall
from novice_stakes.refraction import IsoSpeedFan, p_sca_KA_fan, initialize_axes

plt.ion()

z_src = -105
z_rcr = -15
x_rcr = 460

# Sinusoid paramters
H = 2.
L = 40.
phi = 0.

c = 1500
fc = 1e3
fs = 2.25e3 * 2
tau_lim = 25e-3

# compute source and receiver ray fans
dz_iso = 2
num_rays = 500
theta_max = 0.1 * (pi / 180)

ray_src = IsoSpeedFan(c, z_src + dz_iso, num_rays, theta_max)
ray_rcr = IsoSpeedFan(c, z_rcr + dz_iso, num_rays, theta_max)

# compute time/frequency domain parameters
faxis, dx, sig_FT = initialize_nuttall(fc, fs, c, tau_lim)

# setup xaxis
xaxis, yaxis, tau_img = initialize_axes(ray_src, ray_rcr, tau_lim, x_rcr, dx, dz_iso=dz_iso)

# 1 and 2D surfaces
K = 2 * pi / L
eta = (H / 2) * np.cos(K * xaxis + phi)
eta_dx = -(H * K / 2) * np.sin(K * xaxis + phi)

# 2-D calculations
eta_2D = np.broadcast_to(eta[:, None], (xaxis.size, yaxis.size))
eta_dx_2D = np.broadcast_to(eta_dx[:, None], (xaxis.size, yaxis.size))
eta_p = np.array([-eta_dx_2D, np.zeros_like(eta_dx_2D)])

# stationary phase results
p_sta_fan, t_rcr_sta, p_ref = p_sca_KA_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                eta, eta_dx, tau_img, tau_lim, faxis, sig_FT,
                                dz_iso=dz_iso)
# line source result
kc = 2 * pi * fc / c
p_ls_fan, t_rcr_1D, p_ref = p_sca_KA_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                      eta, eta_dx,
                                      tau_img, tau_lim, faxis, sig_FT,
                                      kc=kc, dz_iso=dz_iso)

# 2-D result
p_2D_fan, t_rcr_2D, p_ref = p_sca_KA_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                      eta_2D, eta_p,
                                      tau_img, tau_lim, faxis, sig_FT,
                                      yaxis=yaxis, dz_iso=dz_iso)

# 1-D calculations
# compute full source vector for projection
r_src = np.array([xaxis, np.full(xaxis.shape, eta - z_src)])
d_src = np.linalg.norm(r_src, axis=0)

n = np.array([-eta_dx, np.ones_like(xaxis)])
proj_src = np.einsum('ik,ik->k', n, r_src) / d_src

# greens function from source
dpdn_g_as_point = -faxis[:, None] * proj_src / (2 *c * d_src) \
                * np.exp(-2j * pi * faxis[:, None] * d_src / c)

dpdn_g_as_line = -1j * pi * faxis[:, None] * proj_src / (2 * c) \
               * np.sqrt(2 * c / (pi * 2 * pi * faxis[:, None] * d_src)) \
               * np.exp(-2j * pi * faxis[:, None] * d_src / c + 3j * pi / 4)

# receiver vector
d_rcr = np.sqrt((x_rcr - xaxis) ** 2 + (z_rcr - eta) ** 2)

g_ra_point = np.exp(-2j * pi * faxis[:, None] * d_rcr / c) / (4 * pi * d_rcr)

g_ra_line = (1j / 4) * np.sqrt(2 * c / (pi * 2 * pi * faxis[:, None] * d_rcr)) \
          * np.exp(-2j * pi * faxis[:, None] * d_rcr / c + 1j * pi / 4)

# compute full 2D source vector for projection
r_src = np.array([*np.meshgrid(xaxis, yaxis, indexing='ij'),
                 eta_2D - z_src])
d_src_2D = np.linalg.norm(r_src, axis=0)

n = np.array([-eta_dx_2D, np.zeros_like(d_src_2D), np.ones_like(d_src_2D)])
proj_src_2D = np.einsum('ijk,ijk->jk', n, r_src) / d_src_2D

r_rcr = np.array([*np.meshgrid(x_rcr - xaxis, yaxis, indexing='ij'),
                 np.full((xaxis.size, yaxis.size), (z_rcr - eta_2D))])
d_rcr_2D = np.linalg.norm(r_rcr, axis=0)

# greens function from source
f_ = faxis[:, None, None]
ds_ = d_src_2D[None, :, :]
dr_ = d_rcr_2D[None, :, :]

ne_str = '-f_ * proj_src_2D / (2 * c * ds_) * exp(-2j * pi * f_ * ds_ / c)'
dpdn_g_as_2D = ne.evaluate(ne_str)


ne_str = 'exp(-2j * pi * f_ * dr_ / c) / (4 * pi * dr_)'
g_ra_2D = ne.evaluate(ne_str)

# surface integral for pressure at receiver

# 1-D geometry
kc = 2 * pi * fc / c
p_rcr_1D, taxis_1D = p_sca(2 * dpdn_g_as_line,
                           g_ra_line,
                           dx,
                           sig_FT,
                           faxis,
                           (d_src + d_rcr) / c,
                           tau_img,
                           tau_lim,
                           spreading=kc)

# stationary phase

# compute spreading factor for stationary phase approximation
# second derivative of (d_src + d_rcr) / c wrt y
d2d = (1 / d_src + 1 / d_rcr) / c

p_rcr_sta, taxis_sta = p_sca(2 * dpdn_g_as_point,
                             g_ra_point,
                             dx,
                             sig_FT,
                             faxis,
                             (d_src + d_rcr) / c,
                             tau_img,
                             tau_lim,
                             spreading=d2d)

# 2D integration
p_rcr_2D, taxis_2D = p_sca(2 * dpdn_g_as_2D,
                           g_ra_2D,
                           dx,
                           sig_FT,
                           faxis,
                           (d_src_2D + d_rcr_2D) / c,
                           tau_img,
                           tau_lim)

fig, ax = plt.subplots()
ax.plot(t_rcr_sta, 20 * np.log10(np.abs(hilbert(p_sta_fan))))
ax.plot(taxis_sta, 20 * np.log10(np.abs(hilbert(p_rcr_sta))))

fig, ax = plt.subplots()
ax.plot(t_rcr_2D, 20 * np.log10(np.abs(hilbert(p_2D_fan))))
ax.plot(taxis_2D, 20 * np.log10(np.abs(hilbert(p_rcr_2D))))

num_tail = 35
error_ls = np.max(np.abs(np.abs(hilbert(p_ls_fan[:-num_tail])) -
                  np.abs(hilbert(p_rcr_1D[:-num_tail])))) \
         / np.max(np.abs(hilbert(p_rcr_1D[:-num_tail])))

error_sta = np.max(np.abs(np.abs(hilbert(p_sta_fan[:-num_tail])) -
                  np.abs(hilbert(p_rcr_sta[:-num_tail])))) \
         / np.max(np.abs(hilbert(p_rcr_sta[:-num_tail])))

error_2D = np.max(np.abs(np.abs(hilbert(p_2D_fan[:-num_tail])) -
                  np.abs(hilbert(p_rcr_2D[:-num_tail])))) \
         / np.max(np.abs(hilbert(p_rcr_2D[:-num_tail])))

