import numpy as np
from math import pi
from mpmath import mp
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
    dx = rL_diff[0]
    adz = np.abs(rL_diff[1])
    return dx, adz

def G_spec(kcL, alpha_0L, rs_L, num_eva, n=None):
    """naive implimentation of spectral representation"""
    dx, adz = compute_differences(rs_L)
    ms, alpha_mL, gamma_mL = define_ms(kcL, alpha_0L, num_eva)
    #G = np.exp(-gamma_mL * adz + 1j * alpha_mL * rL_diff[0]) / gamma_mL
    G = evaluate("exp(-gamma_mL * adz + 1j * alpha_mL * dx) / gamma_mL")

    # take termwise normal derivative if a normal derivative vector is
    # specified
    if n is not None:
        g_vec = np.array([-np.sign(dz_L) * gamma_mL, 1j * alpha_mL])

    # compute spectral sum, approximate main diagonal as 0
    # TODO: properly treat singularity at 0
    G = -G.sum(axis=-1) / 2
    G[np.diag_indices_from(G)] = 0. + 0.j

    return G

def G_spec_Kummar(kcL, alpha_0L, rs_L, num_eva):
    """kummar implimentation of spectral representation"""
    dx, adz = compute_differences(rs_L)

    # treat m=0 as a special case
    gamma_0L = -1j * np.sqrt(kcL ** 2 - alpha_0L ** 2 + 0j)
    ms, alpha_mL, gamma_mL = define_ms(kcL, alpha_0L, num_eva, include0=False)
    sign_ms = np.sign(ms)

    # spectral sum with aymptotics removed
    u_m = evaluate("""(exp(-(2 * pi * abs(ms) + sign_ms * alpha_0L) * adz)
                      / (2 * pi * abs(ms))) \
                      * (1 - alpha_0L / (2 * pi * ms) + kcL ** 2 * adz / (4 * pi * abs(ms)))""")
    G1 = evaluate("(exp(-gamma_mL * adz) / gamma_mL - u_m) * exp(2j * pi * ms * dx)")
    G1 = G1.sum(axis=-1)

    # asymptotics of spectral sum
    dx = dx[:, :, 0]
    adz = adz[:, :, 0]

    Z = adz + 1j * dx
    Zc = np.conj(Z)
    argZ = evaluate("exp(-2 * pi * Z)") + np.spacing(1)
    argZc = np.conj(argZ)

    v_poly = np.vectorize(lambda *a: complex(mp.polylog(*a)))

    S = evaluate("exp(-alpha_0L * adz)") * (v_poly(1, argZc) / (2 * pi)
          - evaluate("2 * alpha_0L - kcL ** 2 * adz") \
          * v_poly(2, argZc) / (8 * pi ** 2))
    S += evaluate("exp(alpha_0L * adz)") * (v_poly(1, argZ) / (2 * pi)
         + evaluate("2 * alpha_0L + kcL ** 2 * adz") \
         * v_poly(2, argZ) / (8 * pi ** 2))

    G = (evaluate("exp(-gamma_0L * adz) / gamma_0L") + S) + G1
    G *= evaluate("-exp(1j * alpha_0L * dx) / 2")
    G[np.diag_indices_from(G)] = 0. + 0.j
    return G
