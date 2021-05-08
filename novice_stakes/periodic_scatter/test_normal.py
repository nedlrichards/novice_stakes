import numpy as np
from math import pi
from scipy.special import hankel1
from novice_stakes.periodic_scatter import G_spec, G_spec_naive

# acoustic parameters
theta_inc = 35. * pi / 180
fc = 500.  # monofrequency source
c = 1500.  # sound speed, m/s
kc = 2 * pi * fc / c

# Sinusiodal parameters
H = 2.
L = 70.

xtest = 4.  # position on first wave length
numL = 6000

K = 2 * pi / L

# setup an x-axis
dx = c / (8 * fc)
numx = np.ceil(L / dx)
xaxis = np.arange(numx) * L / numx
dx = (xaxis[-1] - xaxis[0]) / (xaxis.size - 1)
z = H * np.cos(K * xaxis) / 2

# choose two specific points from the xaxis
i1 = np.argmin(np.abs(3. - xaxis))
i2 = np.argmin(np.abs(5. - xaxis))
x1 = xaxis[i1]
x2 = xaxis[i2]

z1 = z[i1]
z2 = z[i2]

# sum of Hankel1 functions
# make number of wavelengths odd

ns = np.arange(-numL, numL + 1)
xs = ns * L + x2

a0 = kc * np.cos(theta_inc)

# spectral formulation
a_q = a0 + ns * K
g_q = -1j * np.sqrt(kc ** 2 - a_q ** 2 + 0j)

dx = x1 - xs
dz = z1 - z2
rho = np.sqrt(dx ** 2 + dz ** 2)

# Normal derivative of Periodic Greens function as a sum of Hankel functions
n_vec = np.array([H * K * np.sin(K * x1) / 2, 1])
g_grad = -1j * kc * np.array([dx, np.full_like(dx, dz)]) * np.exp(1j * a0 * ns * L) * hankel1(1, kc * rho) / (4 * rho)
g_per = np.einsum('i,ij->', n_vec, g_grad)
print(g_per)

# spectral formulation of normal derivative
g_spec_grad = np.array([1j * a_q, -g_q * np.sign(dz)]) * np.exp(-g_q * np.abs(dz) + 1j * a_q * (x1 - x2)) / g_q
g_spec = np.einsum('i,ij->', n_vec, g_spec_grad) / (2 * L)
print(g_spec)

# Use canned routine to calculate normal derivative of periodic greens function
rs = np.array([xaxis, z])
ns = np.array([H * K * np.sin(K * xaxis) / 2, np.ones_like(xaxis)])
G_mat = G_spec_naive(kc * L, a0 * L, rs / L, 500, n_L=ns / L)
print(G_mat[i2, i1])

rs_L = rs / L
kcL = kc * L
alpha_0L = a0 * L
n_L = ns / L
num_eva = 5000

# make accelerated sum match naive formulation
from greens import compute_differences, define_ms
from numexpr import evaluate

dx, dz = compute_differences(rs_L)
adz = np.abs(dz)

# treat m=0 as a special case
gamma_0L = -1j * np.sqrt(kcL ** 2 - alpha_0L ** 2 + 0j)
ms, alpha_mL, gamma_mL = define_ms(kcL, alpha_0L, num_eva, include0=False)
ms = ms.squeeze()
alpha_mL = alpha_mL.squeeze()
gamma_mL = gamma_mL.squeeze()
sign_ms = np.sign(ms)

dx = (x1 - x2)/L
dz = (z1 - z2)/L
adz = np.abs(dz)

G0th = np.exp(-gamma_0L * adz + 1j * alpha_0L * dx) / gamma_0L
u_m = evaluate("""(exp(-(2 * pi * abs(ms) + sign_ms * alpha_0L) * adz + 1j * alpha_mL * dx)
                    / (2 * pi * abs(ms)))""").sum()

G_n = evaluate("exp(-gamma_mL * adz + 1j * alpha_mL * dx) / gamma_mL").sum()
G_rem = G_n - u_m

argZ = evaluate("exp(-2 * pi * (adz + 1j * dx))") + np.spacing(1)
argZc = np.conj(argZ)
S1 = evaluate("exp(-alpha_0L * adz) * log(1 - argZc) / (2 * pi)")
S2 = evaluate("exp(alpha_0L * adz) * log(1 - argZ) / (2 * pi)")
S = (S1 + S2) * np.exp(1j * alpha_0L * dx)

G = (G0th - S + G_rem) / 2

s1_g = np.array([np.full_like(argZc, -1j), np.sign(dz)]) \
     * argZc * np.exp(-alpha_0L * adz) / (1 - argZc)
s1_g[1] -= np.sign(dz) * alpha_0L * S1

s2_g = np.array([np.full_like(argZ, 1j), np.sign(dz)]) \
     * argZ * np.exp(alpha_0L * adz) / (1 - argZ)
s2_g[1] += np.sign(dz) * alpha_0L * S2

s_phase = evaluate("exp(1j * alpha_0L * dx)")

s1_g[0] = 1j * alpha_0L * s_phase * S1 + s_phase * s1_g[0]
s2_g[0] = 1j * alpha_0L * s_phase * S2 + s_phase * s2_g[0]
s1_g[1] *= s_phase
s2_g[1] *= s_phase

delta = 0.000001
dx = dx
dz = dz + delta
adz = np.abs(dz)

argZ = evaluate("exp(-2 * pi * (adz + 1j * dx))") + np.spacing(1)
argZc = np.conj(argZ)
S1_d = evaluate("exp(-alpha_0L * adz) * log(1 - argZc) / (2 * pi)")
S2_d = evaluate("exp(alpha_0L * adz) * log(1 - argZ) / (2 * pi)")
S_d = (S1_d + S2_d) * np.exp(1j * alpha_0L * dx)

#diff = (S1_d - S1) / delta
diff = (S_d - S) / delta

print(
