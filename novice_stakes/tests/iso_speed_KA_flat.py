import numpy as np
import numexpr as ne
from math import pi
from scipy.optimize import newton
from scipy.signal import hilbert
from scipy.special import hankel2
import matplotlib.pyplot as plt

from novice_stakes import p_sca, initialize_nuttall
from novice_stakes.refraction import IsoSpeedFan, p_sca_KA_fan, initialize_axes

plt.ion()

z_src = -105
z_rcr = -15
x_rcr = 460

c = 1500.
fc = 1e3
fs = 2.25e3 * 2
tau_lim = 15e-3

# compute time/frequency domain parameters
faxis, dx, sig_FT = initialize_nuttall(fc, fs, c, tau_lim)

# compute source and receiver ray fans
num_rays = 2000
theta_max = 0.1 * (pi / 180)
# check extrapolation formulation
dz_iso = 5

ray_src = IsoSpeedFan(c, z_src, num_rays, theta_max)
ray_rcr = IsoSpeedFan(c, z_rcr, num_rays, theta_max)

# compute time/frequency domain parameters
faxis, dx, sig_FT = initialize_nuttall(fc, fs, c, tau_lim)

# setup xaxis
xaxis, yaxis, tau_img = initialize_axes(ray_src, ray_rcr, tau_lim, x_rcr, dx)

# ray fan calculations
# flat surfaces
eta = np.zeros_like(xaxis)
eta_p = np.zeros_like(xaxis)

# compute full 2D coordinates for 2D eta calculations
eta_2D = np.zeros((xaxis.size, yaxis.size))
eta_p_2D = np.zeros((2, xaxis.size, yaxis.size))

# stationary phase results
p_sta_fan, t_rcr_sta, p_ref = p_sca_KA_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                eta, eta_p, tau_img, tau_lim, faxis, sig_FT)

# line source result
kc = 2 * pi * fc / c
p_ls_fan, t_rcr_1D, p_ref = p_sca_KA_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                      eta, eta_p,
                                      tau_img, tau_lim, faxis, sig_FT,
                                      kc=kc)

# 2-D result
p_2D_fan, t_rcr_2D, p_ref = p_sca_KA_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                      eta_2D, eta_p_2D,
                                      tau_img, tau_lim, faxis, sig_FT,
                                      yaxis=yaxis)

# Analytical expressions for isospeed case
# 1-D calculations
# compute full source vector for projection
r_src = np.array([xaxis, np.full(xaxis.shape, -z_src)])
d_src = np.linalg.norm(r_src, axis=0)

g_dx = np.zeros_like(xaxis)
g_dz = np.ones_like(xaxis)

n = np.array([g_dx, g_dz])
proj_src = np.einsum('ik,ik->k', n, r_src) / d_src

# greens function from source
kaxis = 2 * pi * faxis[:, None] / c

dpdn_g_as_point = -1j * kaxis * proj_src / (4 * pi * d_src) \
                * np.exp(-1j * kaxis * d_src)

dpdn_g_as_line = -(1j / 4) * kaxis * proj_src * hankel2(1, kaxis * d_src)

# receiver vector
d_rcr = np.sqrt((x_rcr - xaxis) ** 2 + z_rcr ** 2)

g_ra_point = np.exp(-1j * kaxis * d_rcr) / (4 * pi * d_rcr)

g_ra_line = (1j / 4) * hankel2(0, kaxis * d_rcr)
# compute reference time series from image source
r_img = np.sqrt(x_rcr ** 2 + (z_src + z_rcr) ** 2)

point_FT = -sig_FT / (4 * pi * r_img)
p_ref_point = np.fft.irfft(point_FT)

line_FT = -sig_FT * (1j / 4) * hankel2(0, kaxis[:, 0] * r_img)
line_FT[np.isnan(line_FT)] = 0. + 0.j
p_ref_line = np.fft.irfft(line_FT * np.exp(1j * kaxis[:, 0] * r_img))

# 2-D calculations
# compute full 2D source vector for projection
r_src = np.array([*np.meshgrid(xaxis, yaxis, indexing='ij'),
                 np.full((xaxis.size, yaxis.size), -z_src)])
d_src_2D = np.linalg.norm(r_src, axis=0)
n = np.array([np.zeros_like(d_src_2D), np.zeros_like(d_src_2D), np.ones_like(d_src_2D)])
proj_src_2D = np.einsum('ijk,ijk->jk', n, r_src) / d_src_2D

r_rcr = np.array([*np.meshgrid(x_rcr - xaxis, yaxis, indexing='ij'),
                 np.full((xaxis.size, yaxis.size), -z_rcr)])
d_rcr_2D = np.linalg.norm(r_rcr, axis=0)

# greens function from source
k_ = 2 * pi * faxis[:, None, None] / c
ds_ = d_src_2D[None, :, :]
dr_ = d_rcr_2D[None, :, :]

ne_str = '-1j * k_ * proj_src_2D / (4 * pi * ds_) * exp(-1j * k_ * ds_)'
dpdn_g_as_2D = ne.evaluate(ne_str)


ne_str = 'exp(-1j * k_ * dr_) / (4 * pi * dr_)'
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

# for ray exptrapolation tests
ray_src_dz = IsoSpeedFan(c, z_src + dz_iso, num_rays, theta_max)
ray_rcr_dz = IsoSpeedFan(c, z_rcr + dz_iso, num_rays, theta_max)

# setup xaxis
xaxis_dz, yaxis_dz, tau_img = initialize_axes(ray_src_dz, ray_rcr_dz, tau_lim, x_rcr, dx, dz_iso=dz_iso)

# flat surfaces
eta_dz = np.zeros_like(xaxis_dz)
eta_p_dz = np.zeros_like(xaxis_dz)

# compute full 2D coordinates for 2D eta calculations
eta_2D_dz = np.zeros((xaxis_dz.size, yaxis_dz.size))
eta_p_2D_dz = np.zeros((2, xaxis_dz.size, yaxis_dz.size))

# stationary phase results
p_sta_ext, _, _ = p_sca_KA_fan(ray_src_dz, ray_rcr_dz, xaxis_dz, x_rcr,
                                eta_dz, eta_p_dz, tau_img, tau_lim, faxis, sig_FT,
                                dz_iso=dz_iso)

# line source result
p_ls_ext, _, _ = p_sca_KA_fan(ray_src_dz, ray_rcr_dz, xaxis_dz, x_rcr, eta_dz, eta_p_dz,
                              tau_img, tau_lim, faxis, sig_FT,
                              kc=kc, dz_iso=dz_iso)

# 2-D result
p_2D_ext, _, _ = p_sca_KA_fan(ray_src_dz, ray_rcr_dz, xaxis_dz, x_rcr, eta_2D_dz, eta_p_2D_dz,
                              tau_img, tau_lim, faxis, sig_FT,
                               yaxis=yaxis_dz, dz_iso=dz_iso)

num_tail = 35
def ts_error(ts, ts_ref, num_tail):
    """Error metric used to compare time series"""
    error_norm = np.max(np.abs(ts_ref[:-num_tail]))
    error = np.max(np.abs(ts[:-num_tail] - ts_ref[:-num_tail]))
    return error / error_norm

# max allowed relative error
max_error = 0.001
assert(ts_error(p_rcr_1D, p_ref_line, num_tail) < max_error)
assert(ts_error(p_ls_fan, p_ref_line, num_tail) < max_error)
assert(ts_error(p_ls_ext, p_ref_line, num_tail) < max_error)

assert(ts_error(p_rcr_sta, p_ref_point, num_tail) < max_error)
assert(ts_error(p_sta_fan, p_ref_point, num_tail) < max_error)
assert(ts_error(p_sta_ext, p_ref_point, num_tail) < max_error)

assert(ts_error(p_rcr_2D, p_ref_point, num_tail) < max_error)
assert(ts_error(p_2D_fan, p_ref_point, num_tail) < max_error)
assert(ts_error(p_2D_ext, p_ref_point, num_tail) < max_error)

print("All flat surface tests passed")
