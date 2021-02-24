import numpy as np
from math import pi
from novice_stakes.surfaces import spectrum, Surface
import matplotlib.pyplot as plt
from scipy.signal import correlate, welch

plt.ion()

xbounds = (0, 1600)
dx = 0.1

numx = int(np.ceil((xbounds[1] - xbounds[0]) / dx)) + 1
if numx % 2: numx += 1
xaxis = np.arange(numx) * dx + xbounds[0]

# compute surface realization
rms_height = 0.5
corr_length = 20

kmax = 2 * pi / dx
kx = np.arange(numx // 2 + 1) * kmax / numx

Pxx_non = spectrum.lorentzian(kx, rms_height, corr_length)
dk = (kx[-1] - kx[0]) / (kx.size - 1)
ms_height = 2 * np.sum(Pxx_non) * dk

surf_1D = Surface(kmax, Pxx_non)
realization_1D = surf_1D.realization()
eta_1D = surf_1D.surface_synthesis(realization_1D)

# compute Fourier transform of lorentzian
Pxx_FT = np.fft.irfft(Pxx_non) * kmax
x_FT = xaxis - xaxis[numx // 2]

# compute the power spectrum of the surface
k_est, Pxx_est = welch(eta_1D, fs=kmax, nperseg=2**9, detrend=False)
Rxx = np.fft.irfft(Pxx_est) * k_est[-1] * 2
Rxx = Rxx[:Rxx.size // 2 + 1]
Rxx -= Rxx[-1]
xaxisR = np.arange(Rxx.size) * dx

auto_corr = correlate(eta_1D, eta_1D, mode='full')
auto_corr = 2 * auto_corr[auto_corr.size // 2: ] / numx
xcorr = np.arange(auto_corr.size) * dx

fig, ax = plt.subplots()
ax.plot(x_FT, rms_height ** 2 * np.exp(-(1 / corr_length) * np.abs(x_FT)))
ax.plot(xaxisR, Rxx)
ax.plot(xcorr, auto_corr)
ax.plot(x_FT, rms_height ** 2 * np.full_like(x_FT, np.exp(-1)), linewidth=0.5, color='0.6')
ax.plot([corr_length, corr_length], [-1e3, 1e3], linewidth=0.5, color='0.6')
ax.set_xlim(0, 50)
ax.set_ylim(-rms_height ** 2 / 5, rms_height ** 2)
1/0

fig, ax = plt.subplots()
ax.plot(kx, Pxx_non)
ax.plot(k_est, Pxx_est)
#ax.set_xlim(0, 50)
#ax.set_ylim(0, rms_height ** 2)

1 / 0

fig, ax = plt.subplots()
ax.plot(x_FT, rms_height ** 2 * np.exp(-(1 / corr_length) * np.abs(x_FT)))
ax.plot(x_FT, np.fft.fftshift(Pxx_FT), ':')
ax.plot(x_FT, rms_height ** 2 * np.full_like(x_FT, np.exp(-1)), linewidth=0.5, color='0.6')
ax.plot([corr_length, corr_length], [0, 1e3], linewidth=0.5, color='0.6')
ax.set_xlim(0, 50)
ax.set_ylim(0, rms_height ** 2)



auto_corr = correlate(eta_1D, eta_1D, mode='full')
auto_corr = auto_corr[auto_corr.size // 2: ]
xcorr = np.arange(auto_corr.size) * dx

fig, ax = plt.subplots()
#ax.plot(xaxis, eta_1D)
ax.plot(xcorr, auto_corr)
1/0

ky = (np.arange(numx) - numx // 2) * kmax / numx

kmag = np.sqrt(kx[:, None] ** 2 + ky ** 2)
kphi = np.arctan2(ky, kx[:, None])

Pxx_non = spectrum.lorentzian(rms_height, k_length, kmag)
delta = np.zeros_like(kmag)
Pxx = spectrum.directional_spectrum(delta, kphi, kmag, Pxx_non)

surf_2D = Surface(kmax, Pxx)
realization_2D = surf_2D.realization()
eta_2D = surf_2D.surface_synthesis(realization_2D)

fig, ax = plt.subplots()
ax.pcolormesh(xaxis, xaxis, eta_2D)
