import numpy as np
from math import pi
from scipy.special import hankel1
import matplotlib.pyplot as plt
from spectral import G_spec, G_spec_Kummar, G_dz0

plt.ion()

# acoustic parameters
theta_inc = 50. * pi / 180
fc = 2500.  # monofrequency source
c = 1500.  # sound speed, m/s
kc = 2 * pi * fc / c

L = 70.

rsrc = np.array([0.3, 0.])
#rrcr = np.array([0.3, .01])
rrcr = np.array([0.3, .01])

dx = rrcr[0] - rsrc[0]
dz = rrcr[1] - rsrc[1]

alpha_0 = kc * np.cos(theta_inc)

ref = G_spec_Kummar(kc * L, alpha_0 * L, dx / L, dz / L, 10000)

# test of dz0
dx_tmp = 7.
test1_dz0 = G_dz0(kc * L, alpha_0 * L, dx_tmp / L, 10000, 20)
test2_dz0 = G_spec_Kummar(kc * L, alpha_0 * L, dx_tmp / L, 0, 10000)

n_test = [10., 25., 50., 100., 250., 500., 750., 1000., 2000., 3000., 4000., 5000.]
G_1 = []
G_2 = []
for n in n_test:
    G_1.append(G_dz0(kc * L, alpha_0 * L, dx_tmp / L, n, 10))
    G_2.append(G_spec_Kummar(kc * L, alpha_0 * L, dx_tmp / L, 0, n))

fig, ax = plt.subplots()
ax.semilogy(n_test, np.abs(G_1 - test2_dz0))
ax.semilogy(n_test, np.abs(G_2 - test2_dz0))

1/0

n_test = [10., 25., 50., 100., 250., 500., 750., 1000., 2000., 3000., 4000., 5000.]
G_1 = []
G_2 = []
for n in n_test:
    G_1.append(G_spec(kc * L, alpha_0 * L, dx / L, dz / L, n))
    G_2.append(G_spec_Kummar(kc * L, alpha_0 * L, dx / L, dz / L, n))

fig, ax = plt.subplots()
ax.semilogy(n_test, np.abs(G_1 - ref))
ax.semilogy(n_test, np.abs(G_2 - ref))
