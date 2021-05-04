import numpy as np
from math import pi
from mpmath import mp
from scipy.special import factorial, erfc

def G_spec(kcL, alpha_0L, dx_L, dz_L, sum_num):
    """naive implimentation of spectral representation"""
    adz = np.abs(dz_L)
    ms = np.arange(-sum_num, sum_num + 1)
    alpha_m = alpha_0L + 2 * pi * ms
    gamma_m = -1j * np.sqrt(kcL ** 2 - alpha_m ** 2 + 0j)
    G = np.exp(-gamma_m * adz + 1j * alpha_m * dx_L) / gamma_m

    return -G.sum() / 2

def G_spec_Kummar(kcL, alpha_0L, dx_L, dz_L, sum_num):
    """kummar implimentation of spectral representation"""
    adz = np.abs(dz_L)
    ms = np.arange(-sum_num, sum_num + 1)
    ms = ms[ms != 0]
    alpha_mL = alpha_0L + 2 * pi * ms
    gamma_0L = -1j * np.sqrt(kcL ** 2 - alpha_0L ** 2 + 0j)
    gamma_mL = -1j * np.sqrt(kcL ** 2 - alpha_mL ** 2 + 0j)

    u_m = np.exp(-(2 * pi * np.abs(ms) + np.sign(ms) * alpha_0L) * adz)
    u_m /= 2 * pi * np.abs(ms)
    u_m *= 1 - alpha_0L / (2 * pi * ms) + kcL ** 2 * adz / (4 * pi * np.abs(ms))

    Z = adz + 1j * dx_L
    Zc = np.conj(Z)

    S = np.exp(-alpha_0L * adz) \
      * (mp.polylog(1, np.exp(-2 * pi * Zc)) / (2 * pi)
         - (2 * alpha_0L - kcL ** 2 * adz) * mp.polylog(2, np.exp(-2 * pi * Zc)) \
            / (8 * pi ** 2))
    S += np.exp(alpha_0L * adz) \
      * (mp.polylog(1, np.exp(-2 * pi * Z)) / (2 * pi)
         + (2 * alpha_0L + kcL ** 2 * adz) * mp.polylog(2, np.exp(-2 * pi * Z)) \
            / (8 * pi ** 2))

    G = (np.exp(-gamma_mL * adz) / gamma_mL - u_m) * np.exp(2j * pi * ms * dx_L)
    G = np.exp(-gamma_0L * adz) / gamma_0L + complex(S) + G.sum()
    G *= -np.exp(1j * alpha_0L * dx_L) / 2
    return G

def G_Ewald(L, kc, theta_inc, dx, dz, sum_num, n_max, a=2):
    """Accelerate periodic greens function estimate with Ewald sum"""
    K = 2 * pi / L
    adz = np.abs(dz)
    alpha_0 = kc * np.cos(theta_inc)
    ms = np.arange(-sum_num, sum_num + 1)
    ns = np.arange(n_max + 1)
    alpha_m = alpha_0 + ms * K
    gamma_m = -1j * np.sqrt(kc ** 2 - alpha_m ** 2 + 0j)
    r_m = np.sqrt((dx + ms * L) ** 2 + dz ** 2)

    gm = gamma_m.copy()
    gm[np.abs(np.imag(gm)) > np.spacing(1)] *= 1j

    #G1 = np.exp(gamma_m * adz) * erfc(gamma_m * L / (2 * a) + a * adz / L) \
       #+ np.exp(-gamma_m * adz) * erfc(gamma_m * L / (2 * a) - a * adz / L)
    G1 = np.exp(gm * adz) * erfc(gm * L / (2 * a) + a * adz / L) \
       + np.exp(-gm * adz) * erfc(gm * L / (2 * a) - a * adz / L)

    #G1 *= np.exp(1j * alpha_m * dx) / gamma_m
    G1 *= np.exp(1j * alpha_m * dx) / gm
    G1 = -G1.sum() / (4 * L)

    v_exp = np.vectorize(lambda *a: float(mp.expint(*a)))

    G2 = (kc * L / (2 * a)) ** (2 * ns) / factorial(ns, exact=True) \
       * v_exp(ns + 1, (a * r_m[:, None] / L) ** 2)
    G2 = G2.sum(axis=-1) * np.exp(1j * ms * alpha_0 * L)
    G2 = -G2.sum() / (4 * pi)

    return G1 + G2

if __name__ == '__main__':
    from scipy.special import hankel1

    # acoustic parameters
    theta_inc = 35. * pi / 180
    fc = 500.  # monofrequency source
    c = 1500.  # sound speed, m/s
    kc = 2 * pi * fc / c
    beta0 = np.cos(theta_inc) * kc

    L = 70.
    l_max = 19
    qmax = 500

    rsrc = np.array([0.3, 0.])
    rrcr = np.array([0.3, .01])

    # estimate g with sum
    qmax = 100000
    qs = np.arange(-qmax, qmax + 1)

    dx = rrcr[0] - rsrc[0]
    dz = rrcr[1] - rsrc[1]

    rq = np.sqrt((dx + qs * L) ** 2 + dz ** 2)
    g_sum = (-1j / 4) * (hankel1(0, kc * rq) * np.exp(1j * qs * beta0 * L)).sum()
    #print(g_sum)

    beta_q = beta0 + qs * 2 * pi / L
    gamma_q = -1j * np.sqrt(kc ** 2 - beta_q ** 2 + 0j)
    g_spec = (np.exp(-gamma_q * np.abs(dz) + 1j * beta_q * dx) / gamma_q).sum()
    g_spec *= -(1 / (2 * L))
    #print(g_spec)

    alpha_0 = kc * np.cos(theta_inc)

    print(G_spec(kc * L, alpha_0 * L, dx / L, dz / L, qmax))
    print(G_spec_Kummar(kc * L, alpha_0 * L, dx / L, dz / L, 5000))

    1/0
    dx = 0.01
    dz = 0.
    kc = 2
    theta_inc = np.arccos(np.sqrt(2) / kc)
    print(G_spec_Kummar(1, kc, theta_inc, dx, dz, 5000))
    print(G_Ewald(1, kc, theta_inc, dx, dz, 4, 5, a=10))
