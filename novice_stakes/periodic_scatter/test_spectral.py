import numpy as np
from math import pi
from scipy.special import hankel1
import matplotlib.pyplot as plt
from spectral import G_spec, G_spec_Kummar

plt.ion()

# acoustic parameters
theta_inc = 50. * pi / 180
fc = 2500.  # monofrequency source
c = 1500.  # sound speed, m/s
kc = 2 * pi * fc / c

L = 70.

dx = 0.1
x_axis = np.arange(10) * dx
z = np.sin(x_axis * 2 * pi)
r_axis = np.array([x_axis, z])

alpha_0 = kc * np.cos(theta_inc)
ref = G_spec_Kummar(kc * L, alpha_0 * L, r_axis / L, 10000)
test = G_spec(kc * L, alpha_0 * L, r_axis / L, 10000)
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
