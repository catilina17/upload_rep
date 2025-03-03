import numpy as np
import numba as nb

@nb.guvectorize([(nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :, :],
                  nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :, :])],
                '(n,p,r), (n,p,r), (n,r,q), (n,p,r),(n,p,r), (n,r,r), (r,p,r)->(n,p,q)',
                target="parallel", nopython=True, fastmath=False)
def calculate_amortization_3D(credit_depart, rate, current_month, mois_depart_amor, duree, prof_ech_const,
                                          mois_reneg, rem_cap):
    n = current_month.shape[0]
    t = current_month.shape[2]
    t_l = credit_depart.shape[1]
    relative_month = np.zeros((n, t_l, t))
    is_credit = np.zeros((n, t_l, t))
    const = np.zeros((n, t_l, 1))
    rate_mod = np.zeros((n, t_l, 1))

    for i in range(n):
        for k in range(t_l):
            rate_mod[i,k,0] = 0 if not prof_ech_const[i, 0, 0] else rate[i, k, 0]
            const[i,k,0] = 1 / (duree[i,k,0] + rate[i,k,0] * duree[i,k,0] * (duree[i,k,0] - 1) / 2) if duree[i,k,0] != 0 else 0
            for j in range(t):
                is_credit[i,k,j] = 1 if current_month[i,0,j] - mois_reneg[0,k,0] >= 0 else 0
                relative_month[i, k, j] = max(0, current_month[i, 0, j] - mois_depart_amor[i, k, 0] + 1)

    for i in range(n):
        for k in range(t_l):
            for j in range(t):
                rem_cap[i, k, j] = 0.5 * const[i,k,0] * relative_month[i,k,j] * (2 - 2 * rate_mod[i,k,0]  + rate_mod[i,k,0]  * (relative_month[i,k,j] + 1))
                rem_cap[i, k, j] = max(0, credit_depart[i, k, 0] * (is_credit[i,k,j] - rem_cap[i, k, j]))
