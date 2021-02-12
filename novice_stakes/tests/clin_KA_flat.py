import numpy as np
import numexpr as ne
from math import pi
from scipy.optimize import newton
from scipy.signal import hilbert
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from novice_stakes import p_sca, nuttall_pulse
from novice_stakes.refraction import CLinear, rays_to_surface

plt.ion()

z_src = -105
z_rcr = -15
x_rcr = 460

c = 1500
fc = 1e3
fs = 2.25e3 * 2
kc = 2 * pi * fc / c

decimation = 8
dx = c / (decimation * fc)

# compute source and receiver ray fans
c0 = 1500
cm = -0.017
num_rays = 500
theta_max = -1.5 * (pi / 180)

ray_src = CLinear(c0, cm, z_src, num_rays, theta_max)
ray_rcr = CLinear(c0, cm, z_rcr, num_rays, theta_max)

# compute time/frequency domain parameters
tau_lim = 10e-3

# transmitted signal
sig_y, sig_t = nuttall_pulse(fc, fs)

# compute t and f axes
num_t = int(np.ceil(tau_lim * fs + sig_y.size))
if num_t % 2: num_t += 1

# flat surface specifications
# compute FT of transmitted signal
faxis = np.arange(num_t // 2 + 1) * fs / num_t
sig_FT = np.fft.rfft(sig_y, num_t)

# setup xaxis
tau_src_ier = interp1d(ray_src.rho, ray_src.travel_time, fill_value="extrapolate")
tau_rcr_ier = interp1d(ray_rcr.rho, ray_rcr.travel_time, fill_value="extrapolate")

x_test = np.arange(np.ceil(x_rcr * 1.2 / dx)) * dx
x_test += ray_src.rho[0]

dflat = lambda x: tau_src_ier(np.abs(x)) + tau_rcr_ier(np.abs(x_rcr - x))

dflat_iso = lambda x: (np.sqrt(x ** 2 + z_src ** 2)
                      + np.sqrt((x_rcr - x) ** 2 + z_rcr ** 2)) / c0

fudgef = 5  # necassary because we don't know surface profile

tau_total = dflat(x_test)
# find image ray delay and position at z=0
i_img = np.argmin(tau_total)
x_img = x_test[i_img]
tau_img = tau_total[i_img]

rooter = lambda x: dflat(x) - tau_img - tau_lim
xbounds = (newton(rooter, 0) - fudgef, newton(rooter, x_rcr) + fudgef)

numx = int(np.ceil((xbounds[1] - xbounds[0]) / dx)) + 1
if numx % 2: numx += 1
xaxis = np.arange(numx) * dx + xbounds[0]

# iterative process to compute yaxis
# x_ref is best guess for x position of travel time minimum at y_max
x_ref = x_img

for i in range(10):
    # setup yaxis
    rho_src = lambda y: np.sqrt(x_ref ** 2 + y ** 2)
    rho_rcr = lambda y: np.sqrt((x_rcr - x_ref) ** 2 + y ** 2)
    dflat = lambda y: tau_rcr_ier(rho_rcr(y)) + tau_src_ier(rho_src(y))
    rooter = lambda y: dflat(y) - tau_img - tau_lim
    ymax = newton(rooter, tau_lim * c) + fudgef
    # compute x-postion of travel time minimum at y_max
    d_ymax = np.sqrt(xaxis ** 2 + ymax ** 2 + z_src ** 2) \
        + np.sqrt((x_rcr - xaxis) ** 2 + ymax ** 2 + z_rcr ** 2)
    x_nxt = xaxis[np.argmin(d_ymax)]
    if x_ref - x_nxt == 0:
        break
    x_ref = x_nxt

numy = int(np.ceil((2 * ymax / dx))) + 1
if numy % 2: numy += 1
yaxis = np.arange(numy) * dx - ymax

la_src = rays_to_surface(ray_src, xaxis, np.zeros_like(xaxis))
1/0

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
tau_ref = d_img / c

# 1-D geometry
p_rcr_1D, taxis_1D = p_sca(2 * dpdn_g_as_line,
                           g_ra_line,
                           dx,
                           sig_FT,
                           faxis,
                           (d_src + d_rcr) / c,
                           tau_ref,
                           tau_lim,
                           spreading=kc,
                           c=c)

# stationary phase

# compute spreading factor for stationary phase approximation
# second derivative of (d_src + d_rcr) wrt y
d2d = 1 / d_src + 1 / d_rcr

p_rcr_sta, taxis_sta = p_sca(2 * dpdn_g_as_point,
                             g_ra_point,
                             dx,
                             sig_FT,
                             faxis,
                             (d_src + d_rcr) / c,
                             tau_ref,
                             tau_lim,
                             spreading=d2d,
                             c=c)

# 2D integration
p_rcr_2D, taxis_2D = p_sca(2 * dpdn_g_as_2D,
                           g_ra_2D,
                           dx,
                           sig_FT,
                           faxis,
                           (d_src_2D + d_rcr_2D) / c,
                           tau_ref,
                           tau_lim,
                           c=c)

# compute reference amplitudes
p_ref_1D = np.sqrt(2 / (pi * kc * d_img)) / 4
p_ref_2D = 1 / (4 * pi * d_img)

p_sca_dB_1D = 20 * np.log10(np.abs(hilbert(p_rcr_1D))) - 20 * np.log10(p_ref_1D)
p_sca_dB_sta = 20 * np.log10(np.abs(hilbert(p_rcr_sta))) - 20 * np.log10(p_ref_2D)
p_sca_dB_2D = 20 * np.log10(np.abs(hilbert(p_rcr_2D))) - 20 * np.log10(p_ref_2D)

fig, ax = plt.subplots()
ax.plot((taxis_1D - tau_ref) * 1e3, p_sca_dB_1D)
ax.plot((taxis_sta - tau_ref) * 1e3, p_sca_dB_sta)
ax.plot((taxis_2D - tau_ref) * 1e3, p_sca_dB_2D)
