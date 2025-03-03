import numpy as np


def adjust_interest_rate_to_accruals(sc_rates, interests_periods, accrued_interests, begin_capital,
                                                 year_nb_days, nb_days_m0, n):
    rate_mtx = sc_rates.copy()

    rate_mtx_m0 = (accrued_interests) \
                  / (interests_periods[:, 0] / year_nb_days.reshape(n) * begin_capital[:, 0])
    rate_mtx_m0 = np.where(interests_periods[:, 0] == 0, 0, rate_mtx_m0)
    rate_mtx[:, 0] = np.where(np.isnan(accrued_interests), rate_mtx[:, 0], rate_mtx_m0)
    rate_mtx = np.nan_to_num(rate_mtx)
    return rate_mtx

"""@nb.guvectorize([(nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :, :],nb.float64[:, :, :],
                  nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :, :], nb.float64[:, :, :],nb.float64[:, :, :])],
                '(n,r,r), (n,p,r), (r,r,p),(n,r,r), (r,p,p),(n,r,p), (n,p),(n,r,p),(n,r,p) ->(n,p,p)',
                target="parallel", nopython=True, fastmath=True)
def calculate_capital_with_capi(init_strat_ec, stratif_coeffs, curr_month, capi_freq, ind_strat_sup_t, ind_is_cap,
                                sum_coeff, monthly_interest, sc_rates, capital):

    t = stratif_coeffs.shape[1]
    n = stratif_coeffs.shape[0]
    capital[:, :, 0:1] = stratif_coeffs * init_strat_ec
    ones = np.ones((n, 1))

    for h in range(1, t):
        is_in_bucket = np.where((curr_month <= h) & (curr_month > h - capi_freq), 1, 0)
        int_coeff = ind_is_cap[:, 0, h:h + 1] * np.sum((sc_rates * monthly_interest * is_in_bucket), axis=2)
        strat_coeff = ind_is_cap[:, 0, h:h + 1] * stratif_coeffs[:, :, 0] / np.where(sum_coeff[:, h:h + 1] == 0, 1,
                                                                                     sum_coeff[:, h:h + 1])
        int_lag = np.sum(np.sum(capital[:, :h] * is_in_bucket * monthly_interest * sc_rates, axis=1), axis=1)
        int_lag = np.transpose((int_lag * ones)[0:1])

        core_capital = capital[:, :, h - 1] * ind_strat_sup_t[:, :, h]
        mni_strate = capital[:, :, h - 1] * ind_strat_sup_t[:, :, h] * int_coeff
        mni_other_strates = ind_strat_sup_t[:, :, h] * (ind_is_cap[:, :, h] * (strat_coeff * int_lag))
        capital[:, :, h] = core_capital + mni_strate + np.where(np.isnan(mni_other_strates), 0, mni_other_strates)"""