import numpy as np
from math import pi
from mpmath import mp, fp
from numexpr import evaluate
from scipy.special import factorial, erfc, kn

def define_ms(kcL, alpha_0L, num_eva, include0=True):
    """calculate a spectral sum vector based on number of evanescent terms"""
    kcL = np.real(kcL)
    num_p = np.fix((kcL - alpha_0L) / (2 * pi)) + num_eva
    num_n = np.fix(-(kcL + alpha_0L) / (2 * pi)) - num_eva
    ms = np.arange(num_n, num_p + 1)

    # compute wavenumbers
    alpha_mL = alpha_0L + 2 * pi * ms
    gamma_mL = -1j * np.sqrt(kcL ** 2 - alpha_mL ** 2 + 0j)

    if not include0:
        # remove m==0
        mi = ms != 0
        ms = ms[mi]
        alpha_mL = alpha_mL[mi]
        gamma_mL = gamma_mL[mi]

    # Assume last axis for summation
    ms = ms[None, None, :]
    alpha_mL = alpha_mL[None, None, :]
    gamma_mL = gamma_mL[None, None, :]

    return ms, alpha_mL, gamma_mL

def compute_differences(rs_L):
    """compute distances between points"""
    if rs_L.shape[0] != 2:
        raise(ValueError('First dimension of rs should have dimension 2'))
    rL_diff = rs_L[:, None, :] - rs_L[:, :, None]
    # append an axis for summation
    rL_diff = rL_diff[:, :, :, None]
    return rL_diff

def G_spec_naive(kcL, alpha_0L, rs_L, num_eva, n_L=None):
    """naive implimentation of spectral representation"""
    dx, dz = compute_differences(rs_L)
    adz = np.abs(dz)
    ms, alpha_mL, gamma_mL = define_ms(kcL, alpha_0L, num_eva)
    #G = np.exp(-gamma_mL * adz + 1j * alpha_mL * rL_diff[0]) / gamma_mL
    G = evaluate("exp(-gamma_mL * adz + 1j * alpha_mL * dx) / gamma_mL")

    # take termwise normal derivative if a normal derivative vector is
    # specified
    if n_L is not None:
        g_vec = np.array([np.full(dz.shape, 1j) * alpha_mL, -np.sign(dz) * gamma_mL])
        G = np.einsum('ik,ijkl,jkl->jk', n_L, g_vec, G) / 2
    else:
        G = G.sum(axis=-1) / 2

    # compute spectral sum, approximate main diagonal as 0
    G[np.diag_indices_from(G)] = 0. + 0.j
    return G

def G_spec(kcL, alpha_0L, rs_L, num_eva, n_L=None):
    """kummar implimentation of spectral representation"""
    dx, dz = compute_differences(rs_L)
    adz = np.abs(dz)

    # treat m=0 as a special case
    gamma_0L = -1j * np.sqrt(kcL ** 2 - alpha_0L ** 2 + 0j)
    ms, alpha_mL, gamma_mL = define_ms(kcL, alpha_0L, num_eva, include0=False)
    sign_ms = np.sign(ms)

    # spectral sum with aymptotics removed
    # formulation requiring polylog of order 2
    #u_m = evaluate("""(exp(-(2 * pi * abs(ms) + sign_ms * alpha_0L) * adz)
                      #/ (2 * pi * abs(ms))) \
                      #* (1 - alpha_0L / (2 * pi * ms) + kcL ** 2 * adz / (4 * pi * abs(ms)))""")

    G_0 = np.exp(-gamma_0L * adz + 1j * alpha_0L * dx) / gamma_0L

    u_m = evaluate("""(exp(-(2 * pi * abs(ms) + sign_ms * alpha_0L) * adz)
                      / (2 * pi * abs(ms)))""")

    G_n = evaluate("exp(-gamma_mL * adz) * exp(2j * pi * ms * dx) / gamma_mL")

    # asymptotics of spectral sum
    dx = dx[:, :, 0]
    adz = adz[:, :, 0]

    Z = adz + 1j * dx
    Zc = np.conj(Z)
    argZ = evaluate("exp(-2 * pi * Z)") + np.spacing(1)
    argZc = np.conj(argZ)

    if n_L is not None:
        g_vec = np.array([1j * alpha_0L, -np.sign(dz) * gamma_0L])
        G_0 = np.einsum('ik,ijk,jk->jk', n_L, g_vec, G_0)
        g_vec = np.array([np.full(dz.shape, 1j) * alpha_mL, -np.sign(dz) * gamma_mL])
        G_n = np.einsum('ik,ijkl,jkl->jk', n_L, g_vec, G_n)
        g_vec[1] = -(2 * pi * np.abs(ms) + np.sign(ms) * alpha_0L)
        u_m = np.einsum('ik,ijkl,jkl->jk', n_L, g_vec, u_m)

        # Compute normal derivative
        Sd1 = evaluate("-argZc * exp(-alpha_0L * adz) / (1 - argZc)")
        Sd1y = evaluate("alpha_0L * exp(-alpha_0L * adz) * log(1 - argZc) / (2 * pi)")

        Sd2 = evaluate("-argZ * exp(alpha_0L * adz) / (1 - argZ)")
        Sd2y = evaluate("-alpha_0L * exp(alpha_0L * adz) * log(1 - argZ) / (2 * pi)")

        g_vec = np.array([Sd1 + Sd2, -1j * Sd1 + Sd1y + 1j * Sd2 + Sd2y])
        S = np.einsum('ik,ijkl,jkl->jk', n_L, g_vec)

    else:
        G1 = evaluate("(G_n - u_m) * exp(1j * alpha_mL * dx)").sum(axis=-1)

        # formulation requiring polylog of order 2
        #v_poly = np.vectorize(lambda *a: complex(fp.polylog(*a)))
        #S = evaluate("exp(-alpha_0L * adz)") * (v_poly(1, argZc) / (2 * pi)
            #- evaluate("2 * alpha_0L - kcL ** 2 * adz") \
            #* v_poly(2, argZc) / (8 * pi ** 2))
        #S += evaluate("exp(alpha_0L * adz)") * (v_poly(1, argZ) / (2 * pi)
            #+ evaluate("2 * alpha_0L + kcL ** 2 * adz") \
            #* v_poly(2, argZ) / (8 * pi ** 2))

        S = evaluate("-exp(-alpha_0L * adz) * log(1 - argZc) / (2 * pi) \
                    - exp(alpha_0L * adz) * log(1 - argZ) / (2 * pi)")

    G = (G0 + S * evaluate("exp(1j * alpha_0L * dx) / 2") + G1) / 2
    G[np.diag_indices_from(G)] = 0. + 0.j
    return G

def G0(kc, alpha_0, L, dx, qmax=500.):
    """treat singularity when dx=0"""
    # compute 0th lattice sum coefficent
    G1 = 1j * S_Kummer(alpha0, kc, L, 0, qmax) / 4
    # singularity contribution
    G2 = -dx * (np.log(dx / 2) - 1) / (2 * pi)
    return G1 + G2

def G_lattice(kcL, alpha_0L, rs_L, l_max, qmax):
    """Compute the psuedo-periodic greens function from rsrc to rrcr"""
    dx, dz = compute_differences(rs_L)

    kr = kcL * np.sqrt(dx ** 2 + dz ** 2)
    theta = np.arctan2(-np.abs(dx), -np.abs(dz))
    h0 = hankel1(0, kr)

    # compute lattice sum coefficents
    s_arr = [S_Kummer(beta0, kc, L, 0, qmax)]
    for l in range(1, l_max + 1):
        Seven, Sodd = S_Kummer(beta0, kc, L, l, qmax)
        s_arr.append(Seven)
        s_arr.append(Sodd)
    s_arr = np.array(s_arr)

    eps = np.full(s_arr.size, 2.)
    eps[0] = 1.
    orders = np.arange(l_max * 2 + 1)
    jterms = jv(orders, kr)
    costerms = np.cos(orders * (pi / 2 - theta))

    g = h0 + (eps * s_arr * jterms * costerms).sum()
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
