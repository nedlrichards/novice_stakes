import numpy as np
import numexpr as ne
from math import pi
from scipy.optimize import newton
from scipy.signal import hilbert
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from novice_stakes.surfaces import Surface, spectrum
from novice_stakes import initialize_nuttall
from novice_stakes.refraction import p_sca_KA_fan, IsoSpeedFan, initialize_axes

plt.ion()

z_src = -3800.
z_rcr = -200.
x_rcr = 3800.

# Lorentzian surface parameters
rms_height = 0.3
corr_length = 20

# compute time/frequency domain parameters
c = 1500
fc = 4e3
fs = 2.1 * fc * 2
tau_lim = 30e-3

faxis, dx, sig_FT = initialize_nuttall(fc, fs, c, tau_lim)
fi = np.argmin(np.abs(faxis - fc))
kmax = 2 * pi / dx

# compute source and receiver ray fans
dz_iso = 2
num_rays = 2000
theta_max = 0.1 * (pi / 180)

ray_src = IsoSpeedFan(c, z_src + dz_iso, num_rays, theta_max)
ray_rcr = IsoSpeedFan(c, z_rcr + dz_iso, num_rays, theta_max)

# setup xaxis
xaxis, yaxis, tau_img = initialize_axes(ray_src, ray_rcr, tau_lim, x_rcr, dx, dz_iso=dz_iso)

kx = np.arange(xaxis.size // 2 + 1) * kmax / xaxis.size
ky = (np.arange(yaxis.size) - (yaxis.size // 2 - 1)) * kmax / yaxis.size

# stationary phase results
def realize_surface(rms_height, corr_length, num_realizations=100):
    """Compute a set of pressure time series"""
    Pxx = spectrum.lorentzian(kx, rms_height, corr_length)
    surf_1D = Surface(kmax, Pxx)

    p_ts_all = []
    for _ in range(num_realizations):

        realization_1D = surf_1D.realization()
        eta = surf_1D.surface_synthesis(realization_1D)
        eta_p = surf_1D.surface_synthesis(realization_1D, derivative='x')

        p_rcr, taxis, p_ref = p_sca_KA_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                           eta, eta_p,
                                           tau_img, tau_lim, faxis, sig_FT,
                                           dz_iso=dz_iso, shadow=False)

        p_ts_all.append(p_rcr / p_ref)

    intensity = np.abs(hilbert(np.array(p_ts_all))) ** 2

    all_corr = []
    for i_t in intensity:
        acorr = np.correlate(i_t, i_t, mode='same')
        all_corr.append(acorr[acorr.size // 2: ])
    all_corr = np.array(all_corr)

    mean_corr = np.mean(all_corr, axis=0)
    t_corr = np.arange(mean_corr.size) / fs
    return t_corr, mean_corr




t_corr, corr_0_0 = realize_surface(0.0, corr_length, num_realizations=200)
t_corr, corr_0_1 = realize_surface(0.1, corr_length, num_realizations=200)
t_corr, corr_0_2 = realize_surface(0.2, corr_length, num_realizations=200)
t_corr, corr_0_3 = realize_surface(0.3, corr_length, num_realizations=200)
t_corr, corr_0_4 = realize_surface(0.4, corr_length, num_realizations=200)
t_corr, corr_0_5 = realize_surface(0.5, corr_length, num_realizations=200)

fig, axes = plt.subplots(1, 2, sharey=True)
axes[0].semilogy(t_corr * 1e3, corr_0_0 / corr_0_0.max(), label='    0.1')
axes[0].semilogy(t_corr * 1e3, corr_0_1 / corr_0_1.max(), label='    0.1')
axes[0].semilogy(t_corr * 1e3, corr_0_2 / corr_0_2.max(), label='    0.2')
axes[0].semilogy(t_corr * 1e3, corr_0_3 / corr_0_3.max(), label='    0.3')
axes[0].semilogy(t_corr * 1e3, corr_0_4 / corr_0_4.max(), label='    0.4')
axes[0].semilogy(t_corr * 1e3, corr_0_5 / corr_0_5.max(), label='    0.5')

axes[1].semilogy(t_corr * 1e3, corr_0_0 / corr_0_0.max(), label='rms=0.')
axes[1].semilogy(t_corr * 1e3, corr_0_1 / corr_0_1.max(), label='    0.1')
axes[1].semilogy(t_corr * 1e3, corr_0_2 / corr_0_2.max(), label='    0.2')
axes[1].semilogy(t_corr * 1e3, corr_0_3 / corr_0_3.max(), label='    0.3')
axes[1].semilogy(t_corr * 1e3, corr_0_4 / corr_0_4.max(), label='    0.4')
axes[1].semilogy(t_corr * 1e3, corr_0_5 / corr_0_5.max(), label='    0.5')

axes[0].set_xlim(0, 1)
axes[1].set_xlim(0, 10)
axes[0].set_ylim(5e-2, 2)
axes[1].legend(fontsize=8)
axes[0].set_xlabel('Lag (ms)')
axes[0].set_ylabel('Magnitude')

pos = axes[0].get_position()
pos.x0 += 0.04
pos.x1 += 0.04
pos.y0 += 0.04
pos.y1 += 0.04
axes[0].set_position(pos)

pos = axes[1].get_position()
pos.x0 += 0.04
pos.x1 += 0.04
pos.y0 += 0.04
pos.y1 += 0.04
axes[1].set_position(pos)


1/0


p_dB_rms = []
rms = []
for h in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:

    p_ts_all = []
    # 1D spectrum
    Pxx = spectrum.lorentzian(kx, h, corr_length)
    surf_1D = Surface(kmax, Pxx)

    for i in range(100):

        realization_1D = surf_1D.realization()
        eta = surf_1D.surface_synthesis(realization_1D)
        eta_p = surf_1D.surface_synthesis(realization_1D, derivative='x')

        p_rcr, t_rcr_sta, p_ref = p_sca_KA_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                               eta, eta_p,
                                               tau_img, tau_lim, faxis, sig_FT,
                                               dz_iso=dz_iso, shadow=False)

        p_ts_all.append(np.abs(p_rcr) ** 2)
        rms.append(np.sqrt(np.sum(eta ** 2) / eta.size))

    p_ts_all = np.array(p_ts_all)
    p_dB_rms.append(10 * np.log10(np.mean(p_ts_all, axis=0)) - 20 * np.log10(p_ref))

p_dB_rms = np.array(p_dB_rms)
# shadowed source result
#p_rcr, t_rcr_sha, p_ref = p_sca_fan(ray_src, ray_rcr, xaxis, x_rcr,
                                #eta, eta_p, tau_img, tau_lim, faxis, sig_FT,
                                #None, dz_iso=dz_iso, shadow=True)
#p_dB_sha = 20 * np.log10(np.abs(hilbert(p_rcr))) - 20 * np.log10(p_ref)


fig, axes = plt.subplots(1, 2, sharey=True)
#ax.plot((t_rcr_sta - tau_img) * 1e3, p_dB_sta)
#ax.plot((t_rcr_sha - tau_img) * 1e3, p_dB_sha)
axes[0].plot((t_rcr_sta - tau_img) * 1e3, p_dB_rms.T, linewidth=1)
axes[1].plot((t_rcr_sta - tau_img) * 1e3, p_dB_rms.T, linewidth=1)

axes[1].text(-25, -35, 'Delay from image arrival (ms)', clip_on=False)
axes[0].set_ylabel('Magnitude (dB re. image arrival)')
axes[0].set_ylim(-30, 0.5)
axes[0].set_xlim(-0.5, 2)
axes[1].set_xlim(-0.5, 30)

pos = axes[0].get_position()
pos.x0 += 0.04
pos.x1 += 0.04
pos.y0 += 0.04
pos.y1 += 0.04
axes[0].set_position(pos)

pos = axes[1].get_position()
pos.x0 += 0.04
pos.x1 += 0.04
pos.y0 += 0.04
pos.y1 += 0.04
axes[1].set_position(pos)
