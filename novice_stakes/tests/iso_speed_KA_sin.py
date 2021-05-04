import numpy as np
import numexpr as ne
from math import pi
from scipy.optimize import newton
from scipy.signal import hilbert
from scipy.special import hankel2
import matplotlib.pyplot as plt

from novice_stakes import p_sca, initialize_nuttall
from novice_stakes.refraction import IsoSpeedFan, p_sca_KA_fan, initialize_axes
from novice_stakes.periodic_scatter import CosineRs, make_theta_axis

plt.ion()

z_src = -105
z_rcr = -15
x_rcr = 460

# Sinusoid paramters
H = 2.
L = 40.

c = 1500
fc = 1e3
fs = 2.25e3 * 2
tau_lim = 40e-3

# compute time/frequency domain parameters
faxis, dx, sig_FT = initialize_nuttall(fc, fs, c, tau_lim)

# compute source and receiver ray fans
num_rays = 2000
theta_max = 0.1 * (pi / 180)
dz_iso = 5

ray_src = IsoSpeedFan(c, z_src + dz_iso, num_rays, theta_max)
ray_rcr = IsoSpeedFan(c, z_rcr + dz_iso, num_rays, theta_max)

# setup xaxis
xaxis, yaxis, tau_img = initialize_axes(ray_src, ray_rcr, tau_lim, x_rcr, dx, dz_iso=dz_iso)

# 1 and 2D surfaces
K = 2 * pi / L
eta = (H / 2) * np.cos(K * xaxis)
eta_dx = -(H * K / 2) * np.sin(K * xaxis)

# 2-D calculations
eta_2D = np.broadcast_to(eta[:, None], (xaxis.size, yaxis.size))
eta_dx_2D = np.broadcast_to(eta_dx[:, None], (xaxis.size, yaxis.size))
eta_p = np.array([eta_dx_2D, np.zeros_like(eta_dx_2D)])

# stationary phase results
p_sta_fan, t_rcr_sta, _ = p_sca_KA_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                eta, eta_dx, tau_img, tau_lim, faxis, sig_FT,
                                dz_iso=dz_iso)
# line source result
kc = 2 * pi * fc / c
p_ls_fan, t_rcr_1D, p_ref = p_sca_KA_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                      eta, eta_dx,
                                      tau_img, tau_lim, faxis, sig_FT,
                                      kc=kc, dz_iso=dz_iso)

# 2-D result
p_2D_fan, t_rcr_2D, _ = p_sca_KA_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                      eta_2D, eta_p,
                                      tau_img, tau_lim, faxis, sig_FT,
                                      yaxis=yaxis, dz_iso=dz_iso)

# Analytical expressions for isospeed case
# 1-D calculations
# compute full source vector for projection
r_src = np.array([xaxis, eta - z_src])
d_src = np.linalg.norm(r_src, axis=0)

n = np.array([-eta_dx, np.ones_like(xaxis)])
proj_src = np.einsum('ik,ik->k', n, r_src) / d_src

# greens function from source
kaxis = 2 * pi * faxis[:, None] / c

dpdn_g_as_point = -1j * kaxis * proj_src / (4 * pi * d_src) \
                * np.exp(-1j * kaxis * d_src)

dpdn_g_as_line = -(1j / 4) * kaxis * proj_src * hankel2(1, kaxis * d_src)

# receiver vector
d_rcr = np.sqrt((x_rcr - xaxis) ** 2 + (z_rcr - eta) ** 2)

g_ra_point = np.exp(-1j * kaxis * d_rcr) / (4 * pi * d_rcr)

g_ra_line = (1j / 4) * hankel2(0, kaxis * d_rcr)

# 2-D calculations
# compute full 2D source vector for projection
r_src = np.array([*np.meshgrid(xaxis, yaxis, indexing='ij'), eta_2D - z_src])
d_src_2D = np.linalg.norm(r_src, axis=0)

n = np.array([-eta_dx_2D, np.zeros_like(eta_dx_2D), np.ones_like(eta_2D)])
proj_src_2D = np.einsum('ijk,ijk->jk', n, r_src) / d_src_2D

r_rcr = np.array([*np.meshgrid(x_rcr - xaxis, yaxis, indexing='ij'), z_rcr - eta_2D])
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

# Wave number synthesis for pressure at receiver
eva_range = 0.1
num_eva = 10
numquad = 50000

# vector element formulation used in periodic solution
rsrc = np.array([0., z_src])
rrcr = np.array([x_rcr, z_rcr])

def p_KA(facous, rANA, sol_type='KA'):
    # periodic scatter solution
    tcoarse = make_theta_axis(2000, eva_range)

    if sol_type == 'KA':
        func = rANA.hka_coefficents
    else:
        func = rANA.rfm_1st


    r0, q0 = func(tcoarse[0], facous, num_eva)
    rn1, qn1 = func(tcoarse[-1], facous, num_eva)

    all_qs = np.unique(np.hstack([q0, qn1]))
    one_freq = np.zeros((tcoarse.size, all_qs.size), dtype=np.complex_)
    one_freq[0, np.isin(all_qs, q0)] = r0
    one_freq[-1, np.isin(all_qs, qn1)] = rn1

    for i, t in enumerate(tcoarse[1: -1]):
        r, q = func(t, facous, num_eva)
        one_freq[i + 1, np.isin(all_qs, q)] = r

    print('computing freq {}'.format(facous))

    p_sca = rANA.bragg.quad(all_qs, tcoarse, one_freq, numquad, eva_range, rsrc,
                            rrcr, facous)

    return p_sca

rANA = CosineRs(H, L, attn=1e-9)

start_phase = -1j * 2 * np.pi * faxis * taxis_1D[0]
p_FT_KA = np.zeros(faxis.size, dtype=np.complex)

fci = faxis > 1
p_FT_KA[fci] = np.squeeze(np.array([p_KA(f, rANA, sol_type='RFM') for f in faxis[fci]]))

channel_FD = p_FT_KA * np.conj(sig_FT)
p_wn_KA = np.fft.irfft(np.conj(np.exp(start_phase) * channel_FD), axis=0)

fig, ax = plt.subplots()
p_dB = 20 * (np.log10(np.abs(hilbert(p_wn_KA)) + np.spacing(1))
             - np.log10(np.abs(p_ref)))
plt.plot(taxis_1D - tau_img, p_dB)
p_dB = 20 * (np.log10(np.abs(hilbert(p_rcr_1D)) + np.spacing(1))
             - np.log10(np.abs(p_ref)))
plt.plot(taxis_1D - tau_img, p_dB)

num_tail = 35
def ts_error(ts, ts_ref, num_tail):
    """Error metric used to compare time series"""
    error_norm = np.max(np.abs(ts_ref[:-num_tail]))
    error = np.max(np.abs(ts[:-num_tail] - ts_ref[:-num_tail]))
    return error / error_norm

# max allowed relative error
max_error = 0.001
# check line source soutions
assert(ts_error(p_ls_fan, p_rcr_1D, num_tail) < max_error)

# check point source solutions
assert(ts_error(p_rcr_sta, p_rcr_sta, num_tail) < max_error)
assert(ts_error(p_rcr_2D, p_rcr_sta, num_tail) < max_error)
assert(ts_error(p_2D_fan, p_rcr_sta, num_tail) < max_error)
print('All iso-speed sinusoidal surface tests passed')
