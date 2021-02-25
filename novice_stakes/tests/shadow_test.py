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
tau_lim = 25e-3

# sinusoid parameters
H = 5
L = 40
Phi = 0
K = 2 * pi / L

# compute source and receiver ray fans
c0 = 1500
num_rays = 500
theta_max = 0.1 * (pi / 180)

ray_src = IsoSpeedFan(c0, z_src + dz_iso, num_rays, theta_max)
ray_rcr = IsoSpeedFan(c0, z_rcr + dz_iso, num_rays, theta_max)

# setup xaxis
dx = fc / (8 * c0)
tau_src_ier = interp1d(ray_src.rho, ray_src.travel_time, fill_value="extrapolate")
tau_rcr_ier = interp1d(ray_rcr.rho, ray_rcr.travel_time, fill_value="extrapolate")
xaxis, yaxis, tau_img = initialize_axes(tau_src_ier, tau_rcr_ier, tau_lim, x_rcr, dx)

# wave profile
eta = (H / 2) * np.cos(K * xaxis + Phi)
eta_p = -(H * K / 2) * np.sin(K * xaxis + Phi)

def shadow(rho, d_rho):
    # check for an axis where rho moves past x_src =0
    grad_sign = np.sign(np.diff(rho))
    grad_sign = np.hstack([grad_sign[0], grad_sign])
    neg_grad_i = grad_sign < 0
    pos_grad_i = grad_sign >= 0
    neg_grad_start = np.argmax(neg_grad_i)
    pos_grad_start = np.argmax(pos_grad_i)

    # return non-shadowed points expecting a monotonic decrease in rho
    if np.any(neg_grad_i):
        neg_values = (rho + d_rho)[neg_grad_i]
        mono_values = np.maximum.accumulate(neg_values[::-1])
        _, no_shad_neg = np.unique(mono_values, return_index=True)
        #undo index flip
        no_shad_neg = neg_values.size - 1 - no_shad_neg
        no_shad_neg += neg_grad_start
    if np.any(pos_grad_i):
        mono_values = np.maximum.accumulate((rho + d_rho)[pos_grad_i])
        _, no_shad_pos = np.unique(mono_values, return_index=True)
        no_shad_pos += pos_grad_start

    if np.any(neg_grad_i) and np.any(pos_grad_i):
        all_i = np.hstack([no_shad_pos, no_shad_neg])
    elif np.any(neg_grad_i):
        all_i = no_shad_neg
    else:
        all_i = no_shad_pos
    return all_i

# relate surface position to incident angle
ray_fan = ray_src
rho_src = np.abs(xaxis)
px_ier = interp1d(ray_fan.rho, ray_fan.px, kind=3,
                    bounds_error=False, fill_value=np.nan)
px_n = px_ier(rho_src)
cos_n = px_n * ray_fan.c0
sin_n = np.sqrt(1 - cos_n ** 2)
d_rho_src = -eta * cos_n / sin_n

props = np.array([ray_fan.travel_time, ray_fan.q])
ray_ier = interp1d(ray_fan.rho, props, kind=3,
                    bounds_error=False, fill_value=np.nan)
src_no_shad = shadow(rho_src, d_rho_src)

# relate surface position to incident angle
ray_fan = ray_rcr
rho_rcr = np.abs(x_rcr - xaxis)
px_ier = interp1d(ray_fan.rho, ray_fan.px, kind=3,
                    bounds_error=False, fill_value=np.nan)
px_n = px_ier(rho_rcr)
cos_n = px_n * ray_fan.c0
sin_n = np.sqrt(1 - cos_n ** 2)
d_rho_rcr = -eta * cos_n / sin_n

props = np.array([ray_fan.travel_time, ray_fan.q])
ray_ier = interp1d(ray_fan.rho, props, kind=3,
                    bounds_error=False, fill_value=np.nan)
rcr_no_shad = shadow(rho_rcr, d_rho_rcr)

all_shadow = src_no_shad[np.isin(src_no_shad, rcr_no_shad)]

fig, ax = plt.subplots()
ax.plot(xaxis, rho_src + d_rho_src)
ax.plot(xaxis[src_no_shad], (rho_src + d_rho_src)[src_no_shad], '.', color='C1')

ax.plot(xaxis, rho_rcr + d_rho_rcr, color='C0')
ax.plot(xaxis[rcr_no_shad], (rho_rcr + d_rho_rcr)[rcr_no_shad], '.', color='C1')

1/0
fig, ax = plt.subplots()
ax.plot(xaxis, eta)
ax.plot(xaxis[all_shadow], eta[all_shadow], '.')



1/0
src_amp, src_tt = rays_to_surface(ray_src, xaxis, dz_iso + eta, eta_p=eta_p, kc=kc, shadow=True)
rcr_amp, rcr_tt = rays_to_surface(ray_rcr, np.abs(x_rcr - xaxis), dx_iso+eta, kc=kc, shadow=True)

src_amp, src_tt, prho = rays_to_surface(ray_src, xaxis, dz_iso + eta, eta_p=eta_p, kc=kc, shadow=True)
rcr_amp, rcr_tt, prho_rcr = rays_to_surface(ray_rcr, np.abs(x_rcr - xaxis), dx_iso+eta, kc=kc, shadow=True)

