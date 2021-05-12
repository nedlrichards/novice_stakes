import numpy as np
from numexpr import evaluate
from math import pi
from scipy.linalg import solve
from scipy.special import jv

# basic periodic scatter information
from novice_stakes.periodic_scatter import Bragg

# complete reflection coefficent calculation modules to check results
from novice_stakes.periodic_scatter import CosineRs, QuadRs
from greens import G_spec_naive, G_spec

import matplotlib.pyplot as plt
plt.ion()

# incident plane wave parameters
theta_inc = 15. * pi / 180.
c = 1500.
fc = 500.

# Sinusoidal paramters
H = 2.
L = 40.
K = 2 * pi / L

# discretize surface
decimation = 8  # integration lengths per acoustic wavelength
dx = c / (8 * fc)
numx = int(np.ceil(L / dx))
dx = L / numx
xper = np.arange(numx) * dx
z_wave = (H / 2) * np.cos(K * xper)
zp_wave = -(H * K / 2) * np.sin(K * xper)

# general considerations for periodic scatter calculations
num_eva = 10
bragg = Bragg(L)
qs = bragg.qvec(theta_inc, num_eva, fc)
a0, an, b0, bn = bragg.bragg_angles(theta_inc, qs, fc)

# source and receiver specifications
xsrc = 0
zsrc = -10
xrcr = 200
zrcr = -20

# Analytic solution of reflection coefficents specific to sinusoidal surface
a_inc = 1j ** qs * jv(qs, -b0 * H / 2)

qdiff = qs[None, :] - qs[:, None]
a_sca = 1j ** qdiff * jv(qdiff, bn[None, :] * H / 2)

# solve system of equation for reflection coefficents
rs_ana = solve(-a_sca, a_inc)
p_rfm_ana = bragg.p_sca(theta_inc, qs, fc, rs_ana, xsrc, zsrc, xrcr, zrcr)

# compaire RFM to other solutions
r_cos = QuadRs(xper, z_wave, zp_wave, c=c)

# compute normal derivative of Green's function
kcL = 2 * pi * fc / c
alpha_0L = kcL * np.cos(theta_inc)

rs_L = np.array([xper, z_wave])
n_L = np.array([-zp_wave, np.ones_like(xper)]) / L

G_norm = G_spec(kcL, alpha_0L, rs_L, 1000, n_L)

projection = np.dot(np.array([a0, b0]),
                    np.array([-zp_wave, np.ones_like(xper)]))
dpinc = 2j * projection * np.exp(1j * (a0 * xper - b0 * z_wave))

dp_tot = solve(np.identity(xper.size) - 2 * G_norm * dx, dpinc)

rs_ka = r_cos._r_from_dpdn(dpinc * np.exp(-1j * a0 * xper), theta_inc, qs, fc)
rs_hie = r_cos._r_from_dpdn(dp_tot * np.exp(-1j * a0 * xper), theta_inc, qs, fc)

print("RFM energy: {:.6f}".format(bragg.r_energy(theta_inc, qs, fc, rs_ana)))
print("KA energy: {:.6f}".format(bragg.r_energy(theta_inc, qs, fc, rs_ka)))
print("HIE energy: {:.6f}".format(bragg.r_energy(theta_inc, qs, fc, rs_hie)))
