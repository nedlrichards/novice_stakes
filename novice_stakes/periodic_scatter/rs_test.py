import numpy as np
from math import pi
from scipy.linalg import solve
from scipy.special import jv

# basic periodic scatter information
from novice_stakes.periodic_scatter import Bragg

# complete relfection coefficent calculation modules to check results
from novice_stakes.periodic_scatter import CosineRs, QuadRs

# incident plane wave parameters
theta_inc = 15.
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
num_eva = 3
bragg = Bragg(L)
qs = bragg.qvec(theta_inc, num_eva, fc)
a0, an, b0, bn = bragg.bragg_angles(theta_inc, qs, fc)

# Analytic solution of reflection coefficents specific to sinusoidal surface
a_inc = 1j ** qs * jv(qs, -b0 * H / 2)
qdiff = qs[None, :] - qs[:, None]
a_sca = 1j ** qdiff * jv(qdiff, bn[None, :] * H / 2)

# solve system of equation for reflection coefficents
rs_ana = solve(-a_sca, a_inc)

# check naive notebook implimentation against module results
r_cos = CosineRs(H, L, c=c)
r1_ana, _ = r_cos.rfm_1st(theta_inc, qs, fc)
r2_ana, _ = r_cos.rfm_2nd(theta_inc, qs, fc)

r_quad = QuadRs(xper, z_wave, zp_wave, c=c)
r1_quad, _ = r_quad.rfm_1st(theta_inc, num_eva, fc)
r2_quad, _ = r_quad.rfm_2nd(theta_inc, num_eva, fc)
