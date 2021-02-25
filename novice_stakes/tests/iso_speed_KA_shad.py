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

z_src = -30
z_rcr = -30
dz_iso = 2
x_rcr = 460

fc = 1e3
fs = 2.25e3 * 2
tau_lim = 30e-3

# sinusoid parameters
H = 2
L = 40
Phi = 0
K = 2 * pi / L

# compute source and receiver ray fans
c0 = 1500
num_rays = 500
theta_max = 0.1 * (pi / 180)

ray_src = IsoSpeedFan(c0, z_src + dz_iso, num_rays, theta_max)
ray_rcr = IsoSpeedFan(c0, z_rcr + dz_iso, num_rays, theta_max)

# compute time/frequency domain parameters
faxis, dx, sig_FT = initialize_nuttall(fc, fs, c0, tau_lim)

# setup xaxis
tau_src_ier = interp1d(ray_src.rho, ray_src.travel_time, fill_value="extrapolate")
tau_rcr_ier = interp1d(ray_rcr.rho, ray_rcr.travel_time, fill_value="extrapolate")
xaxis, yaxis, tau_img = initialize_axes(tau_src_ier, tau_rcr_ier, tau_lim, x_rcr, dx)

# wave profile
eta = (H / 2) * np.cos(K * xaxis + Phi)
eta_p = -(H * K / 2) * np.sin(K * xaxis + Phi)

# stationary phase results
p_rcr, t_rcr_sta, p_ref = p_sca_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                eta, eta_p,
                                tau_img, tau_lim, faxis, sig_FT, None, dz_iso=dz_iso)
p_dB_sta = 20 * np.log10(np.abs(hilbert(p_rcr))) - 20 * np.log10(p_ref)

# shadowed result
p_rcr, t_rcr_sha, p_ref = p_sca_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                eta, eta_p,
                                tau_img, tau_lim, faxis, sig_FT, None,
                                dz_iso=dz_iso, shadow=True)
p_dB_sha = 20 * np.log10(np.abs(hilbert(p_rcr))) - 20 * np.log10(p_ref)

fig, ax = plt.subplots()
ax.plot((t_rcr_sta - tau_img) * 1e3, p_dB_sta)
ax.plot((t_rcr_sha - tau_img) * 1e3, p_dB_sha)
