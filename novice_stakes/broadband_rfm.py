import numpy as np
import os

from sheltie import QuadRs
from sheltie.periodic_pw_scatter import make_theta_axis

from spaniel import BroadbandScatter, kaiser, RoughSurface

NFFT = 2**10  # for RFM DFT calculation
Lper = 70  # this seems like a standard

load_dir = '../computed_results/Cos_long_70_7_d30'
#load_dir = 'computed_results/Cos_standard_70_3'
#load_dir = 'computed_results/Cos_standard_70_3_short'
#load_dir = 'computed_results/Cos_standard_70'
fn = load_dir.split('/')[-1]
#suffix = 't_009_99.npz'
suffix = 't_010_00.npz'

fsplit = '_'.join(fn.split('_')[:2])
diem_file = '_'.join([fsplit, '1st', suffix])
diem = np.load(os.path.join(load_dir, diem_file))

solution_duration = diem['taxis'][-1] - diem['taxis'][0]

# standard geometry
zsrc = diem['zsrc']
xrcr = diem['xrcr']
zrcr = diem['zrcr']
fc = diem['fc']
rsrc = np.array([0., zsrc])
rrcr = np.array([xrcr, zrcr])


def p_rfm(facous, rDFT):
    # RFM integration spcifications
    numquad = 50000
    eva_range = 0.1
    num_eva = 4

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

xwave = np.arange(NFFT) * Lper / NFFT

# offset by lper if xaxis[0] is not less than 0
noff = 1 if diem['xaxis'][0] > 0 else 0

zwave = np.interp(xwave + Lper * noff, diem['xaxis'], diem['eta'])
zpwave = np.interp(xwave + Lper * noff, diem['xaxis'], diem['eta_p'])

save_name = diem_file.split('_')
save_name[save_name.index('1st')] = 'RFM'
save_name = '_'.join(save_name)

#rDFT = QuadRs(xwave, zwave, zpwave, attn=0.1)
rDFT = QuadRs(xwave, zwave, zpwave)

faxis = diem['faxis']
fci = diem['fci']

xmitt_FD = np.fft.rfft(diem['xmitt'], n=diem['taxis'].size)
start_phase = -1j * 2 * np.pi * faxis * diem['taxis'][0]
p_FT = np.zeros(faxis.size, dtype=np.complex)
p_FT[fci] = np.squeeze(np.array([p_rfm(f, rDFT) for f in faxis[fci]]))

channel_FD = p_FT * np.conj(xmitt_FD)
p_t = np.fft.irfft(np.conj(np.exp(start_phase) * channel_FD), axis=0)

print('save file: ' + os.path.join(load_dir, save_name))

np.savez(os.path.join(load_dir, save_name),
         p_FT=p_FT, p_t=p_t, zsrc=zsrc, zrcr=zrcr,
         xrcr=xrcr, faxis=faxis, zwave=zwave, xwave=xwave,
         fci=fci, taxis=diem['taxis'])
