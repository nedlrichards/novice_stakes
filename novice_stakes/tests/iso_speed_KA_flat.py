import numpy as np
import numexpr as ne
from math import pi
from scipy.optimize import newton
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from novice_stakes import p_sca, initialize_nuttall, initialize_axes

plt.ion()

z_src = -105
z_rcr = -15
x_rcr = 460

c = 1500.
fc = 1e3
fs = 2.25e3 * 2
tau_lim = 10e-3

# compute time/frequency domain parameters
faxis, dx, sig_FT = initialize_nuttall(fc, fs, c, tau_lim)

# setup xaxis
tau_src_ier = lambda rho: np.sqrt(rho ** 2 + z_src ** 2) / c
tau_rcr_ier = lambda rho: np.sqrt(rho ** 2 + z_rcr ** 2) / c
xaxis, yaxis, tau_img = initialize_axes(tau_src_ier, tau_rcr_ier, tau_lim, x_rcr, dx)

numx = xaxis.size
numy = yaxis.size

# 1-D calculations
# compute full source vector for projection
r_src = np.array([xaxis, np.full(xaxis.shape, -z_src)])
d_src = np.linalg.norm(r_src, axis=0)

g_dx = np.zeros_like(xaxis)
g_dz = np.ones_like(xaxis)

n = np.array([g_dx, g_dz])
proj_src = np.einsum('ik,ik->k', n, r_src) / d_src

# greens function from source
dpdn_g_as_point = -faxis[:, None] * proj_src / (2 *c * d_src) \
                * np.exp(-2j * pi * faxis[:, None] * d_src / c)

dpdn_g_as_line = -1j * pi * faxis[:, None] * proj_src / (2 * c) \
               * np.sqrt(2 * c / (pi * 2 * pi * faxis[:, None] * d_src)) \
               * np.exp(-2j * pi * faxis[:, None] * d_src / c + 3j * pi / 4)

# receiver vector
d_rcr = np.sqrt((x_rcr - xaxis) ** 2 + z_rcr ** 2)

g_ra_point = np.exp(-2j * pi * faxis[:, None] * d_rcr / c) / (4 * pi * d_rcr)

g_ra_line = (1j / 4) * np.sqrt(2 * c / (pi * 2 * pi * faxis[:, None] * d_rcr)) \
          * np.exp(-2j * pi * faxis[:, None] * d_rcr / c + 1j * pi / 4)

# 2-D calculations
# compute full 2D source vector for projection
r_src = np.array([*np.meshgrid(xaxis, yaxis, indexing='ij'),
                 np.full((numx, numy), -z_src)])
d_src_2D = np.linalg.norm(r_src, axis=0)
n = np.array([np.zeros_like(d_src_2D), np.zeros_like(d_src_2D), np.ones_like(d_src_2D)])
proj_src_2D = np.einsum('ijk,ijk->jk', n, r_src) / d_src_2D

r_rcr = np.array([*np.meshgrid(x_rcr - xaxis, yaxis, indexing='ij'),
                 np.full((numx, numy), -z_rcr)])
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
# second derivative of (d_src + d_rcr) wrt y
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

# compute reference amplitudes
p_ref_1D = np.sqrt(2 / (pi * kc * tau_img * c)) / 4
p_ref_2D = 1 / (4 * pi * tau_img * c)

p_sca_dB_1D = 20 * np.log10(np.abs(hilbert(p_rcr_1D))) - 20 * np.log10(p_ref_1D)
p_sca_dB_sta = 20 * np.log10(np.abs(hilbert(p_rcr_sta))) - 20 * np.log10(p_ref_2D)
p_sca_dB_2D = 20 * np.log10(np.abs(hilbert(p_rcr_2D))) - 20 * np.log10(p_ref_2D)

fig, ax = plt.subplots()
ax.plot((taxis_1D - tau_img) * 1e3, p_sca_dB_1D)
ax.plot((taxis_sta - tau_img) * 1e3, p_sca_dB_sta)
ax.plot((taxis_2D - tau_img) * 1e3, p_sca_dB_2D)
