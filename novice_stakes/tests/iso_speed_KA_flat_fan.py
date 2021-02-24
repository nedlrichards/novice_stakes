import numpy as np
import numexpr as ne
from math import pi
from scipy.optimize import newton
from scipy.signal import hilbert
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from novice_stakes import p_sca_fan, initialize_nuttall, initialize_axes
from novice_stakes.refraction import IsoSpeedFan

plt.ion()

z_src = -105
z_rcr = -15
x_rcr = 460

fc = 1e3
fs = 2.25e3 * 2
tau_lim = 10e-3

# compute source and receiver ray fans
c0 = 1500
num_rays = 500
theta_max = 0.1 * (pi / 180)

ray_src = IsoSpeedFan(c0, z_src, num_rays, theta_max)
ray_rcr = IsoSpeedFan(c0, z_rcr, num_rays, theta_max)

# compute time/frequency domain parameters
faxis, dx, sig_FT = initialize_nuttall(fc, fs, c0, tau_lim)

# setup xaxis
tau_src_ier = interp1d(ray_src.rho, ray_src.travel_time, fill_value="extrapolate")
tau_rcr_ier = interp1d(ray_rcr.rho, ray_rcr.travel_time, fill_value="extrapolate")
xaxis, yaxis, tau_img = initialize_axes(tau_src_ier, tau_rcr_ier, tau_lim, x_rcr, dx)

# flat surfaces
eta = np.zeros_like(xaxis)
eta_p = np.zeros_like(xaxis)

# compute full 2D coordinates for 2D eta calculations
eta_2D = np.zeros((xaxis.size, yaxis.size))
eta_p_2D = np.zeros((2, xaxis.size, yaxis.size))

# stationary phase results
p_rcr, t_rcr_sta, p_ref = p_sca_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                eta, eta_p,
                                tau_img, tau_lim, faxis, sig_FT, None)
p_dB_sta = 20 * np.log10(np.abs(hilbert(p_rcr))) - 20 * np.log10(p_ref)

# line source result
kc = 2 * pi * fc / c0
p_rcr, t_rcr_1D, p_ref = p_sca_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                eta, eta_p,
                                tau_img, tau_lim, faxis, sig_FT, kc)
p_dB_1D = 20 * np.log10(np.abs(hilbert(p_rcr))) - 20 * np.log10(p_ref)

# 2-D result
p_rcr, t_rcr_2D, p_ref = p_sca_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                eta_2D, eta_p_2D,
                                tau_img, tau_lim, faxis, sig_FT, yaxis)
p_dB_2D = 20 * np.log10(np.abs(hilbert(p_rcr))) - 20 * np.log10(p_ref)

fig, ax = plt.subplots()

ax.plot((t_rcr_sta - tau_img) * 1e3, p_dB_sta)
ax.plot((t_rcr_1D - tau_img) * 1e3, p_dB_1D)
ax.plot((t_rcr_2D - tau_img) * 1e3, p_dB_2D)

