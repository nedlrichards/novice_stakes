import numpy as np
from scipy.special import zeta, factorial, hankel1, jv
from math import pi
from mpmath import mp
import matplotlib.pyplot as plt

plt.ion()

def G_pseudo(beta0, kc, L, rsrc, rrcr, l_max, qmax):
    """Compute the psuedo-periodic greens function from rsrc to rrcr"""
    dx = rrcr[0] - rsrc[0]
    dz = rrcr[1] - rsrc[1]

    kr = kc * np.sqrt(dx ** 2 + dz ** 2)
    theta = np.arctan2(-np.abs(dx), -np.abs(dz))

    # compute lattice sum coefficents
    s_arr = [S_Kummer(beta0, kc, L, 0, qmax)]
    for l in range(1, l_max + 1):
        Seven, Sodd = S_Kummer(beta0, kc, L, l, qmax)
        s_arr.append(Seven)
        s_arr.append(Sodd)
    s_arr = np.array(s_arr)

    eps = np.full(s_arr.size, 2)
    eps[0] = 1
    orders = np.arange(l_max * 2 + 1)
    jterms = jv(orders, kr)
    costerms = np.cos(orders * (pi / 2 - theta))

    g = hankel1(0, kr) + (eps * s_arr * jterms * costerms).sum()
    g *= -1j / 4

    return g

def S_Kummer(beta0, kc, L, order, qmax):
    """Compute even and odd coefficents at an order number"""
    K = 2 * pi / L
    beta_n = lambda n: beta0 + n * K
    gamma_n = lambda n: -1j * np.sqrt(kc ** 2 - beta_n(n) ** 2 + 0j)
    theta_n = lambda n: np.arcsin(beta_n(n) / kc - 1j * np.sign(n) * np.spacing(1))

    if order == 0:
        qs = np.hstack([np.arange(-qmax, 0), np.arange(1, qmax + 1)])

        S0sum = 1 / gamma_n(qs) - 1 / (K * np.abs(qs)) \
            - (kc ** 2 + 2 * beta0 ** 2) / (2 * K ** 3 * np.abs(qs) ** 3)

        S0sum = np.sum(S0sum)

        S0 = -1 - (2j / pi) * (np.euler_gamma + np.log(kc / (2 * K))) \
        - 2j / (gamma_n(0) * L) \
        - 2j * (kc ** 2 + 2 * beta0 ** 2) * zeta(3) / (K ** 3 * L) \
        - (2j / L) * S0sum

        return S0

    qs = np.arange(1, qmax + 1)

    oeven = 2 * order
    oodd = 2 * order - 1
    # Even sum with wavenumber terms, large terms excluded
    SEsum0 = np.exp(-1j * oeven * theta_n(qs)) / gamma_n(qs) \
        + np.exp(1j * oeven * theta_n(-qs)) / gamma_n(-qs)
    SEsum0 = SEsum0.sum()
    SEsum0 += np.exp(-1j * oeven * theta_n(0)) / gamma_n(0)
    SEsum0 *= -2j / L

    # Odd sum with wavenumber terms, large terms excluded
    SOsum0 = np.exp(-1j * oodd * theta_n(qs)) / gamma_n(qs) \
        - np.exp(1j * oodd * theta_n(-qs)) / gamma_n(-qs)
    SOsum0 = SOsum0.sum()
    SOsum0 += np.exp(-1j * oodd * theta_n(0)) / gamma_n(0)
    SOsum0 *= 2j / L

    # Even sum with factorial terms
    ms = np.arange(1., order + 1)
    b = np.array([float(mp.bernpoly(2 * m, beta0 / K)) for m in ms])

    SEsumF = (-1) ** ms * 2 ** (2 * ms) * factorial(order + ms - 1) \
        * (K / kc) ** (2 * ms) * b \
        / (factorial(2 * ms) * factorial(order - ms))
    SEsumF = np.sum(SEsumF)
    SEsumF *= 1j / pi

    # Odd sum with factorial terms
    ms = np.arange(0, order)
    b = np.array([float(mp.bernpoly(2 * m + 1, beta0 / K)) for m in ms])

    SOsumF = (-1) ** ms * 2 ** (2 * ms) * factorial(order + ms - 1) \
        * (K / kc) ** (2 * ms + 1) * b \
        / (factorial(2 * ms + 1) * factorial(order - ms - 1))
    SOsumF = np.sum(SOsumF)
    SOsumF *= -2 / pi

    # extended precision calculations for large sum terms
    t1 = (1 / pi) * (kc / (2 * K)) ** oeven
    t2 = (beta0 * L * order / pi ** 2) * (kc / (2 * K)) ** oodd

    # assume we need ~15 digits of precision at a magnitude of 1
    dps = np.max([int(np.ceil(np.log10(np.abs(t1)))) + 15,
                 int(np.ceil(np.log10(np.abs(t2)))) + 15,
                 15])
    mp.dps = dps

    arg_ = mp.mpf(kc / (2 * K))
    SEinf = -(-1) ** order * arg_ ** (2 * order) \
          * mp.zeta(2 * order + 1) / mp.pi
    SOinf = (-1) ** order * beta0 * L * order * arg_ ** (2 * order - 1) \
          * mp.zeta(2 * order + 1) / mp.pi ** 2

    for i, m in enumerate(mp.arange(1, qmax + 1)):

        even_term = ((-1) ** order / (m * mp.pi)) * (arg_ / m) ** (2 * order)
        odd_term = ((-1) ** order * beta0 * L * order / (m ** 2 * mp.pi ** 2)) \
                * (arg_ / m) ** (2 * order - 1)
        SEinf += even_term
        SOinf -= odd_term

        # break condition, where we should be OK moving back to double precision
        if mp.fabs(even_term) < 1 and mp.fabs(odd_term) < 1:
            break

    mp.dps = 15
    SEinf = complex(SEinf)
    SOinf = complex(SOinf)

    if i + 1 < qmax:
        ms = np.arange(i + 2, qmax)
        even_terms = ((-1) ** order / (ms * pi)) \
                   * (kc / (2 * K * ms)) ** (2 * order)
        odd_terms = ((-1) ** order * beta0 * L * order / (ms ** 2 * pi ** 2)) \
                  * (kc / (2 * K * ms)) ** (2 * order - 1)

    SEinf += even_terms.sum()
    SEinf *= 2j
    SOinf -= odd_terms.sum()
    SOinf *= 2

    SEven = SEsum0 + SEinf + 1j / (pi * order) + SEsumF
    SOdd = SOsum0 + SOinf + SOsumF

    return SOdd, SEven

def S_naive(beta0, kc, L, order, qmax):
    """Compute even and odd coefficents at an order number"""
    # naive sum
    ms = np.arange(1, qmax + 1)
    oeven = 2 * order
    oodd = 2 * order - 1

    phase = 1j * ms * beta0 * L

    even_terms = hankel1(oeven, ms * kc * L) \
            * (np.exp(phase) + (-1) ** (oeven) * np.exp(-phase))

    odd_terms = hankel1(oodd, ms * kc * L) \
            * (np.exp(phase) + (-1) ** oodd * np.exp(-phase))

    NEven = np.sum(even_terms)
    NOdd = np.sum(odd_terms)
    return NOdd, NEven

def S_Twersky(beta0, kc, L, order, qmax):
    """Twersky formulation of the lattice sum coefficents"""

    K = 2 * pi / L
    beta_n = lambda n: beta0 + n * K
    gamma_n = lambda n: -1j * np.sqrt(kc ** 2 - beta_n(n) ** 2 + 0j)
    theta_n = lambda n: np.arcsin(beta_n(n) / kc - 1j * np.sign(n) * np.spacing(1))


    # Twersky sums
    m_pos = np.arange(qmax // 2 + 1)
    m_neg = np.arange(-qmax // 2, 0)
    oeven = 2 * order
    oodd = 2 * order - 1

    # Even sum with factorial terms
    ms = np.arange(1, order + 1)
    b = np.array([float(mp.bernpoly(2 * m, beta0 / K)) for m in ms])

    SEsumF = (-1) ** ms * 2 ** (2 * ms) * factorial(order + ms - 1) \
        * (K / kc) ** (2 * ms) * b \
        / (factorial(2 * ms) * factorial(order - ms))
    SEsumF = np.sum(SEsumF)
    SEsumF *= 1j / pi

    # Odd sum with factorial terms
    ms = np.arange(0, order)
    b = np.array([float(mp.bernpoly(2 * m + 1, beta0 / K)) for m in ms])

    SOsumF = (-1) ** ms * 2 ** (2 * ms) * factorial(order + ms - 1) \
        * (K / kc) ** (2 * ms + 1) * b \
        / (factorial(2 * ms + 1) * factorial(order - ms - 1))
    SOsumF = np.sum(SOsumF)
    SOsumF *= -2 / pi


    SEven_tw = (-2j / L) * (
            np.sum(np.exp(-1j * oeven * theta_n(m_pos)) / gamma_n(m_pos))
            + np.sum(np.exp(1j * oeven * theta_n(m_neg)) / gamma_n(m_neg)))

    SOdd_tw = (2j / L) * (
            np.sum(np.exp(-1j * oodd * theta_n(m_pos)) / gamma_n(m_pos))
            - np.sum(np.exp(1j * oodd * theta_n(m_neg)) / gamma_n(m_neg)))

    TEven = SEven_tw + 1j / (pi * order) + SEsumF
    TOdd = SOdd_tw + SOsumF
    return TOdd, TEven

if __name__ == '__main__':

    # acoustic parameters
    theta_inc = 50. * pi / 180
    fc = 500.  # monofrequency source
    c = 1500.  # sound speed, m/s
    kc = 2 * pi * fc / c
    beta0 = np.cos(theta_inc) * kc

    L = 70.
    l_max = 15
    qmax = 400

    rsrc = np.array([0., 0.])
    rrcr = np.array([50., 0])

    g_test = G_pseudo(beta0, kc, L, rsrc, rrcr, l_max, qmax)
    print(g_test)

    # estimate g with sum
    qmax = 100000
    qs = np.arange(-qmax, qmax + 1)

    dx = rrcr[0] - rsrc[0]
    dz = rrcr[1] - rsrc[1]

    rq = np.sqrt((dx + qs * L) ** 2 + dz ** 2)
    g_sum = (-1j / 4) * (hankel1(0, kc * rq) * np.exp(1j * qs * beta0 * L)).sum()
    print(g_sum)

    beta_q = beta0 + qs * 2 * pi / L
    gamma_q = -1j * np.sqrt(kc ** 2 - beta_q ** 2 + 0j)
    g_spec = (np.exp(-gamma_q * np.abs(dz) + 1j * beta_q * dx) / gamma_q).sum()
    g_spec *= -(1 / (2 * L))
    print(g_spec)
