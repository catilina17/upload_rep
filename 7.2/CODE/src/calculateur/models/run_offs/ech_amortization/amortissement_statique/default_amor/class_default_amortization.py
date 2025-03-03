import numpy as np
import numexpr as ne
from ..amortization_commons import AmorCommons as ac

class Default_Amortization():
    """
    Formate les données
    """
    def __init__(self, class_init_ecoulement):
        self.cls_init_ec = class_init_ecoulement
        self.cls_proj = class_init_ecoulement.cls_proj
        self.cls_rate = class_init_ecoulement.cls_rate
        self.cls_cal = class_init_ecoulement.cls_cal
        self.cls_fields = class_init_ecoulement.cls_fields
        self.dar_usr = self.cls_proj.cls_hz_params.dar_usr

    ###########@profile
    def calculate_default_amortization(self, data_ldp, cap_before_amor, ech_val, sc_rates, ech_rate, begin_capital,
                                       current_month, amor_freq, capital_amor_shift, begin_month, amor_begin_month,
                                       amor_end_month, mat_date, interests_periods, ech_const_prof, linear_prof,
                                       linear_ech_prof, duree, year_nb_days, capi_freq, suspend_or_capitalize,
                                       capitalize, cap_rate, accrued_interests, nb_days_m0, orig_ech_value,
                                       interest_cap_shift, is_fixed, t):

        # Capital(i) = N * Π(i) - ech_const * Π(i) *  Σ(j=1 à j=i)(1/Π(j)) avec Π(k) = (1+r1)*(1+r2)*...(1+rk)

        n = cap_before_amor.shape[0]
        if n > 0:

            """ CHARGEMENT DU CALENDRIER"""
            suspend_or_capitalize = ne.evaluate("where((~linear_prof) & (~ech_const_prof), True, suspend_or_capitalize)")
            amortization_schedule = ne.evaluate("current_month - amor_begin_month + 1")
            amortization_tombee = ne.evaluate("(current_month - amor_begin_month - capital_amor_shift) %  amor_freq")
            interests_schedule = ne.evaluate("current_month - begin_month + 1")
            capi_schedule = ne.evaluate("(current_month - begin_month - interest_cap_shift) %  capi_freq")

            """ AJUSTEMENT DU TAUX D'INTERET au profil et aux accruals"""
            rate_mtx, real_rate_mtx =\
                ac.adjust_interest_rate_to_profile_and_accruals(ech_rate, sc_rates, ech_const_prof, interests_periods,
                                                             accrued_interests, begin_capital, year_nb_days, nb_days_m0,
                                                             suspend_or_capitalize, n, t)

            """ Calcul de la partie Π(j) """
            compounded_rate =\
                ac.get_compounded_rate_factor(rate_mtx, amortization_schedule, interests_periods, year_nb_days, capi_freq,
                                               capitalize, cap_rate, real_rate_mtx, amor_freq,
                                              amortization_tombee, capi_schedule, interests_schedule, n)

            """ Calcul de la partie N * Π(j) """
            rate_adjusted_nominal = ne.evaluate("compounded_rate * begin_capital")

            """ Calcul de l'échéance constante """
            for nb_pal in range(1, 3):
                if nb_pal == 1:
                    if (~is_fixed).any(axis=0):
                        orig_ech, plage_app_orig_ech = ac.get_plage_application_orig_ech(ech_const_prof, amor_freq, orig_ech_value, begin_month,
                                                                  amortization_tombee, current_month, n, t, is_fixed, val=0)

                        adjusted_rate_for_orig_ech = ac.get_compounded_rate_for_ech(compounded_rate, amortization_tombee,
                                                                                    amortization_schedule,
                                                                                    interests_schedule_b2=plage_app_orig_ech)
                    else:
                        adjusted_rate_for_orig_ech = np.zeros((n, t))
                        orig_ech = 0
                        plage_app_orig_ech = np.ones((n, t))

                else:
                    decalage_amor = np.nan_to_num(np.amax(adjusted_rate_for_orig_ech.astype(int), axis=1))

                    pal_begin_month = amor_begin_month + np.where(decalage_amor.reshape(n, 1) > 0, amor_freq, 0)

                    nb_periods_pal = ac.get_remaining_periods(pal_begin_month, amor_end_month, capital_amor_shift,
                                                              amor_freq, n)

                    temp_capital = (rate_adjusted_nominal - adjusted_rate_for_orig_ech * orig_ech)
                    index_capital = np.maximum(0, np.minimum(pal_begin_month.reshape(n) - 2, t - 1)).astype(int)
                    begin_capital_palier = temp_capital[np.arange(0, n), index_capital].reshape(n, 1)
                    begin_capital_palier = np.where(np.any(plage_app_orig_ech==-1, axis=1).reshape(n,1),
                                                    begin_capital_palier, begin_capital)

                    ech_val = np.where(linear_prof & ~linear_ech_prof, np.nan, ech_val)# valable que pour les taux variables

                    ech_const = ac.get_ech_const(ech_val, ech_rate, begin_capital_palier, nb_periods_pal * amor_freq,
                                                           ech_const_prof, linear_prof, linear_ech_prof,
                                                           amor_freq)

                    plage = np.where(amortization_schedule >= 0, plage_app_orig_ech, 1)

                    adjusted_rate_for_ech = ac.get_compounded_rate_for_ech(compounded_rate, amortization_tombee,
                                                                           amortization_schedule * (plage))

            """ Calcul de la partie : e * Π(j) *  Σ(i=1 à i=j)(1/Π(i)) """
            rate_adjusted_ech = ne.evaluate("ech_const * adjusted_rate_for_ech + orig_ech * adjusted_rate_for_orig_ech")

            rate_adjusted_ech = ne.evaluate("where(suspend_or_capitalize, 0, rate_adjusted_ech)")

            """ Calcul du capital restant """
            self.remaining_capital = ac.calculate_remaining_capital(rate_adjusted_nominal, rate_adjusted_ech, cap_before_amor,
                                                            current_month, interests_schedule, amor_end_month,
                                                            linear_ech_prof, n, t)

        else:
            self.remaining_capital = 0
