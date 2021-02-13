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

src_amp, src_tt, src_d2d = rays_to_surface(ray_src, xaxis,
                                           np.zeros_like(xaxis),
                                           c0 + cm * z_src,
                                           eta_p=np.zeros_like(xaxis),
                                           return_d2tau=True)

rcr_amp, rcr_tt, rcr_d2d = rays_to_surface(ray_rcr,
                                           (x_rcr - xaxis),
                                            np.zeros_like(xaxis),
                                            c0 + cm * z_rcr,
                                            return_d2tau=True)

omega = 2 * pi * faxis[:, None]
omega_c = 2 * pi * fc

# greens function from source
dpdn_g_as_point = -omega * src_amp * np.exp(-1j * omega * src_tt)

dpdn_g_as_line = -1j * omega * np.sqrt(2 / (pi * omega_c * src_tt)) \
               * np.exp(-1j * omega * src_tt + 3j * pi / 4)

# receiver vector
g_ra_point = rcr_amp * np.exp(-1j * omega * rcr_tt)

g_ra_line = (1j / 4) * np.sqrt(2 / (pi * omega_c * rcr_tt)) \
          * np.exp(-1j * omega * rcr_tt + 1j * pi / 4)

# 2-D calculations
# compute full 2D source vector for projection
axes_src = np.array(np.meshgrid(xaxis, yaxis, indexing='ij'))

src_amp_2D, src_tt_2D = rays_to_surface(ray_src, axes_src,
                                  np.zeros_like(xaxis[0]), c0 + cm * z_src,
                                  eta_p=np.zeros_like(axes_src))

axes_rcr = np.array(np.meshgrid(x_rcr - xaxis, yaxis, indexing='ij'))
rcr_amp_2D, rcr_tt_2D = rays_to_surface(ray_rcr, axes_rcr,
                                  np.zeros_like(xaxis), c0 + cm * z_rcr)


# greens function from source
omega_ = 2 * pi * faxis[:, None, None]
aas_ = src_amp_2D[None, :, :]
ara_ = rcr_amp_2D[None, :, :]
ttas_ = src_tt_2D[None, :, :]
ttra_ = rcr_tt_2D[None, :, :]


ne_str = '-1j * omega_ * aas_ * exp(-1j * omega_ * ttas_)'
dpdn_g_as_2D = ne.evaluate(ne_str)


ne_str = 'ara_ * exp(-1j * omega_ * ttra_)'
g_ra_2D = ne.evaluate(ne_str)

# surface integral for pressure at receiver

# 1-D geometry
p_rcr_1D, taxis_1D = p_sca(2 * dpdn_g_as_line,
                           g_ra_line,
                           dx,
                           sig_FT,
                           faxis,
                           src_tt + rcr_tt,
                           tau_img,
                           tau_lim,
                           spreading=kc,
                           c=c)

# stationary phase

# compute spreading factor for stationary phase approximation
# second derivative of (d_src + d_rcr) wrt y
d2d = src_d2d + rcr_d2d

p_rcr_sta, taxis_sta = p_sca(2 * dpdn_g_as_point,
                             g_ra_point,
                             dx,
                             sig_FT,
                             faxis,
                             src_tt + rcr_tt,
                             tau_img,
                             tau_lim,
                             spreading=d2d,
                             c=c)

# 2D integration
p_rcr_2D, taxis_2D = p_sca(2 * dpdn_g_as_2D,
                           g_ra_2D,
                           dx,
                           sig_FT,
                           faxis,
                           rcr_tt_2D + src_tt_2D,
                           tau_img,
                           tau_lim,
                           c=c)

# compute reference amplitudes
p_ref_1D = np.sqrt(2 / (pi * omega_c * tau_img)) / 4
p_ref_2D = 1 / (4 * pi * tau_img / c0)

p_sca_dB_1D = 20 * np.log10(np.abs(hilbert(p_rcr_1D))) - 20 * np.log10(p_ref_1D)
p_sca_dB_sta = 20 * np.log10(np.abs(hilbert(p_rcr_sta))) - 20 * np.log10(p_ref_2D)
p_sca_dB_2D = 20 * np.log10(np.abs(hilbert(p_rcr_2D))) - 20 * np.log10(p_ref_2D)

fig, ax = plt.subplots()
ax.plot((taxis_1D - tau_img) * 1e3, p_sca_dB_1D)
ax.plot((taxis_sta - tau_img) * 1e3, p_sca_dB_sta)
ax.plot((taxis_2D - tau_img) * 1e3, p_sca_dB_2D)
