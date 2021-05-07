        g_vec = np.array([np.full(dz.shape, 1j) * alpha_mL, -np.sign(dz) * gamma_mL])
        G = np.einsum('ij,ijkl,jkl->jk', n_L, g_vec, G) / 2

