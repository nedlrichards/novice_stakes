import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.special import hankel2
from novice_stakes.periodic_scatter import QuadRs, CosineRs, make_theta_axis
from novice_stakes import initialize_nuttall

plt.ion()

NFFT = 2 ** 10  # number of x points in periodic length (if using quadrature)

z_src = -105
z_rcr = -15
x_rcr = 460

# Sinusoid paramters
H = 2.
L = 40.
phi = 0.

c = 1500
fc = 1e3
fs = 2.25e3 * 2
tau_lim = 30e-3

# vector element formulation used in periodic solution
rsrc = np.array([0., z_src])
rrcr = np.array([x_rcr, z_rcr])

# compute time/frequency domain parameters
faxis, _, sig_FT = initialize_nuttall(fc, fs, c, tau_lim)
tau_img = np.sqrt(x_rcr ** 2 + (z_src + z_rcr) ** 2) / c
p_img = hankel2(0, 2 * np.pi * fc * np.sqrt(x_rcr ** 2 + (z_src + z_rcr) ** 2) / c)
taxis = np.arange((faxis.size - 1) * 2) / (2 * faxis[-1]) + tau_img

# specifications for wn synthesis
eva_range = 0.1
num_eva = 4
numquad = 50000

def p_KA(facous, rANA):
    # periodic KA solution
    tcoarse = make_theta_axis(2000, eva_range, is_half=False)

    r0, q0 = rANA.rfm_1st(tcoarse[0], facous, num_eva)
    rn1, qn1 = rANA.rfm_1st(tcoarse[-1], facous, num_eva)

    all_qs = np.unique(np.hstack([q0, qn1]))
    one_freq = np.zeros((tcoarse.size, all_qs.size), dtype=np.complex_)
    one_freq[0, np.isin(all_qs, q0)] = r0
    one_freq[-1, np.isin(all_qs, qn1)] = rn1

    for i, t in enumerate(tcoarse[1: -1]):
        r, q = rANA.hka_coefficents(t, facous, num_eva)
        one_freq[i + 1, np.isin(all_qs, q)] = r

    print('computing freq {}'.format(facous))

    p_sca = rANA.bragg.quad(all_qs, tcoarse, one_freq, numquad, eva_range, rsrc,
                            rrcr, facous, is_symetric=False)

    return p_sca


def p_rfm(facous, rDFT):
    # RFM integration spcifications
    tcoarse = make_theta_axis(2000, eva_range, is_half=False)

    r0, q0 = rDFT.rfm_1st(tcoarse[0], facous, num_eva)
    rn1, qn1 = rDFT.rfm_1st(tcoarse[-1], facous, num_eva)

    all_qs = np.unique(np.hstack([q0, qn1]))
    one_freq = np.zeros((tcoarse.size, all_qs.size), dtype=np.complex_)
    one_freq[0, np.isin(all_qs, q0)] = r0
    one_freq[-1, np.isin(all_qs, qn1)] = rn1

    for i, t in enumerate(tcoarse[1: -1]):
        r, q = rDFT.rfm_1st(t, facous, num_eva)
        one_freq[i + 1, np.isin(all_qs, q)] = r

    print('computing freq {}'.format(facous))

    p_sca = rDFT.bragg.quad(all_qs, tcoarse, one_freq, numquad, eva_range, rsrc,
                            rrcr, facous, is_symetric=False)

    return p_sca

xwave = np.arange(NFFT) * L / NFFT
K = 2 * np.pi / L
zwave = (H / 2) * np.cos(K * xwave)
zpwave = -(H * K / 2) * np.sin(K * xwave)

rANA = CosineRs(H, L, attn=1e-6)
rDFT = QuadRs(xwave, zwave, zpwave)

start_phase = -1j * 2 * np.pi * faxis * taxis[0]
p_FT_RFM = np.zeros(faxis.size, dtype=np.complex)
p_FT_KA = np.zeros(faxis.size, dtype=np.complex)

fci = faxis > 100
p_FT_RFM[fci] = np.squeeze(np.array([p_rfm(f, rDFT) for f in faxis[fci]]))
p_FT_KA[fci] = np.squeeze(np.array([p_KA(f, rANA) for f in faxis[fci]]))

# compute RFM timeseries
channel_FD = p_FT_RFM * np.conj(sig_FT)
p_t_RFM = np.fft.irfft(np.conj(np.exp(start_phase) * channel_FD), axis=0)

# compute KA timeseries
channel_FD = p_FT_KA * np.conj(sig_FT)
p_t_KA = np.fft.irfft(np.conj(np.exp(start_phase) * channel_FD), axis=0)

fig, ax = plt.subplots()
p_dB = 20 * (np.log10(np.abs(hilbert(p_t_RFM)) + np.spacing(1))
             - np.log10(np.abs(p_img)))
plt.plot(taxis - tau_img, p_dB)
p_dB = 20 * (np.log10(np.abs(hilbert(p_t_KA)) + np.spacing(1))
             - np.log10(np.abs(p_img)))
plt.plot(taxis - tau_img, p_dB)
