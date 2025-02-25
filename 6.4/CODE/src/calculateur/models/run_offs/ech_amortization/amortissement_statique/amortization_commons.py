import numexpr as ne
import numpy as np
from calculateur.models.utils import utils as ut
import bottleneck as bn
from dateutil.relativedelta import relativedelta
import numba as nb
from calculateur.models.data_manager.data_format_manager.class_fields_manager import Data_Fields

nan = np.nan


class AmorCommons():

    cls_fields = Data_Fields()

    @staticmethod
    def adjust_interest_rate_to_profile_and_accruals(ech_rate, sc_rates, prof_ech_const, interests_periods,
                                                     accrued_interests, begin_capital, year_nb_days, nb_days_m0,
                                                     suspend_or_capitalize, n, t, is_palier=False, palier_nb=0):
        rate_mtx = sc_rates.copy()
        real_rate_mtx = sc_rates.copy()

        if (not is_palier) or (is_palier and palier_nb == 1):
            rate_mtx_m0 = (accrued_interests + begin_capital[:, 0] * rate_mtx[:, 0] * nb_days_m0
                           / year_nb_days[:, 0].reshape(n)) \
                          / (interests_periods[:, 0] / year_nb_days[:, 0].reshape(n) * begin_capital[:, 0])
            rate_mtx_m0 = np.where(interests_periods[:, 0] == 0, 0, rate_mtx_m0)
            rate_mtx[:, 0] = np.where(np.isnan(accrued_interests), rate_mtx[:, 0], rate_mtx_m0)
            real_rate_mtx[:, 0] = np.where(np.isnan(accrued_interests), real_rate_mtx[:, 0], rate_mtx_m0)
            rate_mtx = np.nan_to_num(rate_mtx)
            real_rate_mtx = np.nan_to_num(real_rate_mtx)

        rate_mtx[~prof_ech_const.reshape(n) | (suspend_or_capitalize.reshape(n))] = 0

        return rate_mtx, real_rate_mtx

    @staticmethod
    def get_plage_application_orig_ech(prof_ech_const, freq_amor, orig_ech_value, begin_month, calendar_amor_adj,
                                       current_month, n, t, is_fixed, val=0):
        noech_const_orig = (np.isnan(orig_ech_value) if val == 0 else orig_ech_value == val)
        plage_idx = np.where(
            (current_month >= begin_month) & (current_month < begin_month + freq_amor) & (calendar_amor_adj == 0))
        s = plage_idx[0].shape[0]
        months = plage_idx[1].reshape(s, 1) + 1
        plage = np.full((n, t), False)
        plage[plage_idx[0]] = np.where((current_month[plage_idx[0]] >= months) &
                                       (current_month[plage_idx[0]] < months + freq_amor[plage_idx[0]]), True, False)
        plage = np.where(prof_ech_const & (~noech_const_orig) & plage & (~is_fixed), -1, 1)
        return np.nan_to_num(orig_ech_value), plage

    @staticmethod
    def get_ech_const(ech_value, rate, begin_capital, duree, prof_ech_const, prof_linear, prof_linear_ech,
                      freq_amor, val=0):
        no_ech_const = np.isnan(ech_value) if val == 0 else ech_value == val
        ech_value = np.where(prof_linear_ech & no_ech_const, 0.0, ech_value)
        no_ech_const = np.where(prof_linear_ech & no_ech_const, False, no_ech_const).astype(bool)
        ech_const = ech_value.copy().astype(np.float64)
        n_p = duree[no_ech_const].shape[0]
        ech_const[no_ech_const] = AmorCommons.calculate_ech_const(rate[no_ech_const], begin_capital[no_ech_const], \
                                                      duree[no_ech_const], freq_amor[no_ech_const]).reshape(n_p)

        cases_profil = [(prof_linear) & (no_ech_const), (~prof_linear) & (~prof_ech_const) & (no_ech_const)]
        duree_p = np.where(np.mod(duree, freq_amor) != 0, duree // freq_amor + 1, duree // freq_amor)
        ech_const = np.select(cases_profil, [begin_capital / duree_p, begin_capital], default=ech_const)

        return np.nan_to_num(ech_const)

    @staticmethod
    def get_deferment_ech(capital_depart, data_ldp, dar, interest_periods, n):
        mat_date_real = np.array(data_ldp[AmorCommons.cls_fields.NC_LDP_MATUR_DATE_REAL]).reshape(n, 1).astype("datetime64[D]")
        val_date_real = np.array(data_ldp[AmorCommons.cls_fields.NC_LDP_VALUE_DATE_REAL]).reshape(n, 1).astype("datetime64[D]")
        nd_days_left = (mat_date_real - np.maximum(val_date_real, np.array(dar + relativedelta(days=1)).astype("datetime64[D]")))/np.timedelta64(1, "D")
        coeff_ajust = np.where(data_ldp[AmorCommons.cls_fields.NC_LDP_ACCRUAL_BASIS].isin(["30/360", "30E/360", "30A/360", "ACT/360"]), 360/365, 1)
        nd_days_left_ajust = nd_days_left * coeff_ajust.reshape(n, 1)
        ech_lin = interest_periods / nd_days_left_ajust * capital_depart
        return ech_lin

    @staticmethod
    def calculate_remaining_capital(rate_adj_nominal, rate_adj_ech, ec_avt_amor, current_month, interests_schedule,
                                    mois_fin_amor, linear_ech_prof, n, t):
        remaining_capital = np.zeros((n, t + 1))
        remaining_capital[:, 1:] = ne.evaluate("rate_adj_nominal - rate_adj_ech")
        # A AMELIORER POUR TF puisque ça s'applique aussi au linear_ech simple
        # ERREUR_RCO
        rate_adj_ech = np.where((remaining_capital[:, 1:] <= 0) & (linear_ech_prof), np.nan, rate_adj_ech)
        rate_adj_ech = np.nan_to_num(ut.np_ffill(rate_adj_ech), 0)
        remaining_capital[:, 1:] = ne.evaluate("rate_adj_nominal - rate_adj_ech")
        remaining_cap_proj = remaining_capital[:, 1:]
        ec_avt_amor_proj = ec_avt_amor[:, 1:]
        remaining_capital[:, 1:] = ne.evaluate("where(interests_schedule <= 0, ec_avt_amor_proj, remaining_cap_proj)")
        remaining_capital[:, 1:] = ne.evaluate("where(current_month >= mois_fin_amor, 0, remaining_cap_proj)")
        remaining_capital[:, 1:] = ne.evaluate("where(remaining_cap_proj < 0, 0, remaining_cap_proj)")
        remaining_capital[:, 0] = ec_avt_amor[:, 0]

        return remaining_capital

    @staticmethod
    def calculate_ech_const(rate, principal, duree, per):
        n = principal.shape[0]
        rate = rate.reshape(n, 1)
        duree = duree.reshape(n, 1)
        per = per.reshape(n, 1)
        duree_p = np.where(np.mod(duree, per) != 0, duree // per + 1, duree // per)
        rate_p = rate * per
        principal = principal.reshape(n, 1)
        ech_const = np.where(rate_p != 0, (principal * rate_p) / (1 - (1 + rate_p) ** (-duree_p)), principal / duree_p)
        return ech_const


    @staticmethod
    def calc_rate_factor_with_capi(rate_factor, capitalize, interests_periods, year_nb_days, real_tx, cap_rate,
                                   interests_schedule_b1, tombee_schedule, capi_freq, n):
        rate_factor_cap = ne.evaluate("interests_periods / year_nb_days * real_tx * cap_rate")
        rate_factor_cap = ne.evaluate("where(interests_schedule_b1 <= 0, 0, rate_factor_cap)")
        rate_factor_cap = AmorCommons.adapt_rate_facto_to_freq(rate_factor_cap, tombee_schedule, capi_freq, n)
        rate_factor = ne.evaluate("where(capitalize, rate_factor_cap, rate_factor)")
        return rate_factor

    @staticmethod
    def get_compounded_rate_factor(tx, interests_schedule_b1, interests_periods, year_nb_days, capi_freq,
                                   capitalize, cap_rate, real_tx, freq_amor, amortizing_schedule,
                                   capi_schedule, capi_interests_schedule, n):
        # Calcul de la partie P(i) = (1+b1r) * (1+b2r) ... * (1+b3r)
        rate_factor = AmorCommons.calc_rate_factor(interests_periods, year_nb_days, interests_schedule_b1, tx)

        # Adaptation des P(i) en cas d'amortissement non mensuel'
        rate_factor = AmorCommons.adapt_rate_facto_to_freq(rate_factor, amortizing_schedule, freq_amor, n)

        # Adaptation des P(i) en cas de capitalisation
        rate_factor = AmorCommons.calc_rate_factor_with_capi(rate_factor, capitalize, interests_periods, year_nb_days,
                                                 real_tx, cap_rate, capi_interests_schedule, capi_schedule, capi_freq, n)
        rate_factor = 1 + rate_factor
        compounded_rate = rate_factor.cumprod(axis=1)

        return compounded_rate

    @staticmethod
    def get_rate_factor(tx, interests_schedule_b1, interests_periods, year_nb_days):
        # Calcul de la partie P(i) = (1+b1r) * (1+b2r) ... * (1+b3r)
        rate_factor = AmorCommons.calc_rate_factor(interests_periods, year_nb_days, interests_schedule_b1, tx)
        return rate_factor

    @staticmethod
    def adapt_rate_facto_to_freq(rate_factor, tombee_schedule, freq, n, interests_schedule_b1=[], interests_schedule_b2=[],
                                 ):
        cond_per = ((freq > 1)).reshape(n)
        if np.any(cond_per, axis=0):
            _amortizing_schedule = tombee_schedule[cond_per]
            _rate_per = rate_factor[cond_per].copy()
            _cum_rate_per = AmorCommons.cum_reset_2d(_rate_per, _amortizing_schedule)
            _cum_rate_per = ne.evaluate('where(_amortizing_schedule == 0, _cum_rate_per, 0)')
            rate_factor[cond_per] = _cum_rate_per

        if interests_schedule_b1 != []:
            rate_factor = np.where((interests_schedule_b1 > 0) & (interests_schedule_b2 < 0), rate_factor, 0)

        return rate_factor

    @staticmethod
    def adapt_rate_facto_to_freq_3D(rate_factor, tombee_schedule, freq, n, t):
        #rate_factor = rate_factor.astype(np.float64)
        #tombee_schedule = tombee_schedule.astype(np.int32)
        rate_factor_cum = AmorCommons.cum_reset_3d_vect(rate_factor, rate_factor, tombee_schedule)
        return rate_factor_cum

    @staticmethod
    def calc_rate_factor(interests_periods, year_nb_days, interests_schedule_b1, tx):
        rate_factor = ne.evaluate("interests_periods / year_nb_days * tx")
        rate_factor = ne.evaluate("where(interests_schedule_b1 <= 0, 0, rate_factor)")
        return rate_factor

    @staticmethod
    def get_compounded_rate_for_ech(compounded_rate, amortizing_schedule, interests_schedule_b1, interests_schedule_b2=[]):
        inv_compounded_rate = np.nan_to_num(ne.evaluate("1 / compounded_rate"), posinf=0, neginf=0)
        inv_compounded_rate = ne.evaluate("where(interests_schedule_b1 <= 0, 0, inv_compounded_rate)")
        if interests_schedule_b2 != []:
            inv_compounded_rate = ne.evaluate("where(interests_schedule_b2 >= 0, 0, inv_compounded_rate)")
        inv_compounded_rate = ne.evaluate('where(amortizing_schedule == 0, inv_compounded_rate, 0)')
        cum_inv_compounded_rate = inv_compounded_rate.cumsum(axis=1)
        adj_rate = ne.evaluate("cum_inv_compounded_rate * compounded_rate")
        return adj_rate

    @staticmethod
    def cum_reset_2d(arr, cond_arr):
        accum = arr.cumsum(axis=1)
        adj = np.where(cond_arr == 1, ut.roll_and_null(accum, 1), nan)
        adj = np.nan_to_num(ut.np_ffill(adj))
        return accum - adj

    @staticmethod
    def cum_reset_3d(arr, cond_arr):
        accum = arr.cumsum(axis=2)
        roll_accum = ut.roll_and_null_axis2(accum, 1)
        adj = ne.evaluate("where(cond_arr == 1, roll_accum, nan)")
        adj = bn.push(adj,axis=2)
        adj_nan = np.isnan(adj)
        adj = ne.evaluate("where(adj_nan,accum,accum - adj)")
        return adj

    @nb.guvectorize([(nb.float64[:,:,:], nb.float64[:,:,:], nb.float64[:,:], nb.float64[:,:,:])],
                    '(n,p,q), (n,p,q), (n,q)->(n,p,q)',
                    target="parallel", nopython=True, fastmath=False)
    def cum_reset_3d_vect(x, y, cond, res):
        for i in range(x.shape[0]):
            for k in range(0, x.shape[1]):
                res[i,k,0] = 0 if cond[i, 0] == 0 else x[i,k,0]
                acc = x[i,k,0] if cond[i, 0] == 0 else 0
                for j in range(0, x.shape[2] - 1):
                    if cond[i, j + 1] == 0:
                        res[i,k,j + 1] = acc + y[i, k, j + 1]
                        acc = 0.0
                    else:
                        acc = acc + y[i, k, j + 1]
                        res[i, k, j + 1] = 0

    @staticmethod
    def get_remaining_periods(amor_begin_month, amor_end_month, capital_amor_shift, amor_freq, n):
        max_end = np.max(amor_end_month)
        current_month = np.repeat(np.arange(1, max_end + 1).reshape(1, max_end), n, axis=0)
        amortization_tombee = ne.evaluate("(current_month - amor_begin_month - capital_amor_shift) %  amor_freq")
        nb_periods_pal = (np.where((current_month >= amor_begin_month) & (amortization_tombee == 0)
                                   & (current_month <= amor_end_month), 1, 0).sum(axis=1)).reshape(n, 1)
        return nb_periods_pal
