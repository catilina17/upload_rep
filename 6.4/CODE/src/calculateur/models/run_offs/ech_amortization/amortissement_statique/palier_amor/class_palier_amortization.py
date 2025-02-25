import numpy as np
import numexpr as ne
from ..amortization_commons import AmorCommons as ac
from calculateur.models.utils import utils as ut


class Palier_Amortization():
    """
    Formate les données
    """

    def __init__(self, cls_format, class_init_ecoulement, tombee_fixing, period_fixing,
                 sc_rates, sc_rates_ftp, sc_rates_lag, sc_rates_ftp_lag, current_month, current_month_max,
                 interests_periods, year_nb_days, t, t_max):
        self.cls_format = cls_format
        self.cls_init_ec = class_init_ecoulement
        self.cls_proj = class_init_ecoulement.cls_proj
        self.cls_rate = class_init_ecoulement.cls_rate
        self.cls_data_rate = class_init_ecoulement.cls_data_rate
        self.cls_cal = class_init_ecoulement.cls_cal
        self.cls_fix_cal = class_init_ecoulement.cls_fix_cal
        self.cls_hz_params = class_init_ecoulement.cls_proj.cls_hz_params
        self.cls_palier = class_init_ecoulement.cls_proj.cls_palier
        self.tombee_fixing = tombee_fixing.copy()
        self.period_fixing = period_fixing.copy()
        self.sc_rates = sc_rates.copy()
        self.sc_rates_ftp = sc_rates_ftp.copy()
        self.sc_rates_lag = sc_rates_lag.copy()
        self.sc_rates_ftp_lag = sc_rates_ftp_lag.copy()
        self.interests_periods = interests_periods
        self.cls_fields = class_init_ecoulement.cls_fields
        self.current_month = current_month
        self.current_month_max = current_month_max
        self.year_nb_days = year_nb_days
        self.t = t
        self.t_max = t_max

    ###########@profile
    def calculate_palier_amortization(self, data_ldp, dic_palier, cap_before_amor, begin_capital,
                                      capi_freq, begin_month, amor_begin_month, amor_end_month, mat_date, dar_mois,
                                      cle_contrat, duree, accrued_interests, nb_day_m0,
                                      capi_rate, tx_params):

        # e(1), e(2),... e(p) les échéances et s(1), t(2), ... s(p) les mois de tombée des échéances. s(p+1) = mois max de proj
        # Capital(i) = N * Π(i) - Π(i) *  Σ(j=1 à p)( e(j) * Σ(k=t(j) à k=min(i,t(j+1)-1))(1/Π(k)) avec Π(h) = ((1+r1)*(1+r2)*...(1+rh))
        n = cap_before_amor.shape[0]

        if n > 0:
            max_palier = dic_palier["max_palier"]
            palier_schedule = dic_palier["palier_schedule"].loc[cle_contrat]
            interests_schedule = self.current_month - begin_month + 1
            self.create_pal_dics()

            for pal_nb in range(1, max_palier + 1):

                palier_schedule = self.load_palier_parameters(data_ldp, palier_schedule, pal_nb, amor_end_month,
                                                              tx_params, dar_mois, max_palier, n)
                if pal_nb == 1:
                    capi_schedule = (self.current_month - begin_month - self.cap_int_shift_pal[pal_nb]) % capi_freq

                """ AJUSTEMENT DU TAUX D'INTERET"""
                self.rate_mtx[pal_nb], self.real_rate_mtx[pal_nb] \
                    = ac.adjust_interest_rate_to_profile_and_accruals(self.rate_ech[pal_nb], self.sc_rates[:, :self.t],
                                                                      self.ech_const_prof[pal_nb],
                                                                      self.interests_periods, accrued_interests,
                                                                      begin_capital,
                                                                      self.year_nb_days, nb_day_m0,
                                                                      self.suspend_or_capitalize[pal_nb],
                                                                      n, self.t, is_palier=True, palier_nb=pal_nb)


            compounded_rate = self.get_compounded_rate(max_palier, self.interests_periods, self.year_nb_days, capi_rate,
                                                       capi_schedule, capi_freq, n)
            rate_adjusted_nominal = begin_capital * compounded_rate
            rate_adjusted_ech_cum = 0

            for pal_nb in range(1, max_palier + 1):
                """ Calcul de la partie: Π(j) *  Σ(k=1 à p)( e(k) * Σ(i=s(k) à k=min(j,s(k+1)-1))(1/P(i)) """

                begin_capital_palier = self.get_begin_capital_palier(begin_capital, rate_adjusted_nominal,
                                                                     rate_adjusted_ech_cum, pal_nb, n, self.t)

                rate_adjusted_ech_cum = \
                    self.get_rate_adjusted_ech_palier(rate_adjusted_ech_cum, compounded_rate, self.palier_schedule,
                                                      self.rate_ech[pal_nb],
                                                      # ou adj_rate_for_nominal_cum
                                                      begin_capital_palier, self.ech_const_prof[pal_nb],
                                                      self.linear_prof[pal_nb],
                                                      self.linear_prof_ech[pal_nb], pal_nb, self.amor_freq[pal_nb],
                                                      self.amortizing_schedule[pal_nb],
                                                      self.amor_schedule_b1[pal_nb],
                                                      self.suspend_or_capitalize[pal_nb],
                                                      self.amor_schedule_b2[pal_nb], self.nb_periods_pal[pal_nb],
                                                      self.mois_amor_pal_i[pal_nb], self.is_fixed[pal_nb],
                                                      rate_adjusted_nominal,
                                                      amor_end_month, self.capital_amor_shift[pal_nb], n, self.t)

            """ CALCUL DU CAPITAL RESTANT"""
            self.remaining_capital = ac.calculate_remaining_capital(rate_adjusted_nominal, rate_adjusted_ech_cum,
                                                                    cap_before_amor, self.current_month,
                                                                    interests_schedule,
                                                                    amor_end_month, self.linear_prof_ech[1], n, self.t)
        else:
            self.remaining_capital = 0

    def get_compounded_rate(self, max_palier, interests_periods, nb_days_an, capi_rate, capi_schedule,
                            capi_freq, n):
        """ Calcul de la partie N * Π(j) (2/2)"""
        rate_factor_cum = 0
        for pal_nb in range(1, max_palier + 1):
            """ Calcul de la matrice (1+b(i)) (1/2)"""
            rate_factor = self.get_rate_factor_for_nominal_palier(self.rate_mtx[pal_nb], self.amor_schedule_b1[pal_nb],
                                                                  self.amor_schedule_b2[pal_nb], interests_periods,
                                                                  nb_days_an)
            rate_factor_cum = rate_factor + rate_factor_cum

        rate_factor_cum_freq = 0
        for pal_nb in range(1, max_palier + 1):
            rate_factor_cum_freq_tmp = ac.adapt_rate_facto_to_freq(rate_factor_cum.copy(),
                                                                   self.amortizing_schedule[pal_nb],
                                                                   self.amor_freq[pal_nb], n,
                                                                   self.amor_schedule_b1[pal_nb],
                                                                   self.amor_schedule_b2[pal_nb])
            rate_factor_cum_freq = rate_factor_cum_freq + rate_factor_cum_freq_tmp

        # Adaptation des P(i) en cas de capitalisation
        rate_factor_cap_cum = 0
        for pal_nb in range(1, max_palier + 1):
            """ Calcul de la matrice (1+b(i)) (1/2)"""
            rate_factor_cap = self.get_rate_factor_for_nominal_palier(self.real_rate_mtx[pal_nb] * capi_rate,
                                                                      self.interests_schedule_b1[pal_nb],
                                                                      self.interests_schedule_b2[pal_nb],
                                                                      interests_periods,
                                                                      nb_days_an)
            rate_factor_cap_cum = rate_factor_cap + rate_factor_cap_cum

        rate_factor_cap_cum_freq = 0
        for pal_nb in range(1, max_palier + 1):
            rate_factor_cap_cum_freq_tmp = ac.adapt_rate_facto_to_freq(rate_factor_cap_cum.copy(), capi_schedule,
                                                                       capi_freq, n, self.interests_schedule_b1[pal_nb],
                                                                       self.interests_schedule_b2[pal_nb])
            rate_factor_cap_cum_freq = rate_factor_cap_cum_freq + rate_factor_cap_cum_freq_tmp

        for pal_nb in range(1, max_palier + 1):
            rate_factor_cum_freq = np.where(self.capitalize[pal_nb] & (self.interests_schedule_b1[pal_nb] > 0)
                                            & (self.interests_schedule_b2[pal_nb] < 0),
                                            rate_factor_cap_cum_freq, rate_factor_cum_freq)

        rate_factor_cum_freq = 1 + rate_factor_cum_freq
        compounded_rate = rate_factor_cum_freq.cumprod(axis=1)

        return compounded_rate

    ##############@profile
    def get_compounded_rate_3D(self, dat_hab, max_palier, interests_periods, nb_days_an, capi_rate,
                               capi_schedule, capi_freq, n, t, is_versement, v):
        # Adaptation des P(i) en cas de capitalisation
        rate_factor_cap_vers = 0
        for pal_nb in range(1, max_palier + 1):
            """ Calcul de la matrice (1+b(i)) (1/2)"""
            rate_factor_cap = self.get_rate_factor_for_nominal_palier(self.real_rate_mtx[pal_nb] * capi_rate,
                                                                      self.interests_schedule_b1[pal_nb],
                                                                      self.interests_schedule_b2[pal_nb],
                                                                      interests_periods,
                                                                      nb_days_an)
            rate_factor_cap = rate_factor_cap.reshape(n, 1, t)
            rate_factor_cap = ne.evaluate("rate_factor_cap * is_versement")
            rate_factor_cap_vers = rate_factor_cap + rate_factor_cap_vers

        rate_factor_cap_cum_freq = ac.adapt_rate_facto_to_freq_3D(rate_factor_cap_vers, capi_schedule,
                                                                  capi_freq, n, t)

        rate_factor_cap_cum_freq = ne.evaluate("1 + rate_factor_cap_cum_freq")
        compounded_rate = rate_factor_cap_cum_freq.cumprod(axis=2)

        return compounded_rate

    ###############@profile
    def load_palier_parameters(self, data_ldp, palier_schedule, pal_nb, amor_end_month, tx_params,
                               dar_mois, max_palier, n):
        t = self.t
        t_max = self.t_max
        current_month = self.current_month
        current_month_max = self.current_month_max
        ech_const_prof = np.array(palier_schedule[self.cls_fields.NC_PROFIL_AMOR + str(pal_nb)] == "ECHCONST").reshape(
            n, 1)
        linear_prof = np.array(palier_schedule[self.cls_fields.NC_PROFIL_AMOR + str(pal_nb)].
                               isin(["LINEAIRE", "LINEAIRE_ECH"])).reshape(n, 1)
        linear_prof_ech = np.array(
            palier_schedule[self.cls_fields.NC_PROFIL_AMOR + str(pal_nb)].isin(["LINEAIRE_ECH"])).reshape(n, 1)

        amor_freq = np.array(palier_schedule[self.cls_fields.NC_FREQ_AMOR + str(pal_nb)]).reshape(n, 1)
        mois_amor_pal_i = np.array(palier_schedule[self.cls_fields.NC_MOIS_PALIER_AMOR + str(pal_nb)]).reshape(n, 1)
        date_pal_i_real = palier_schedule[self.cls_fields.NC_DATE_PALIER_REAL + str(pal_nb)]
        mois_date_pal_i = np.array(palier_schedule[self.cls_fields.NC_MOIS_DATE_PALIER + str(pal_nb)]).reshape(n, 1)
        mois_int_pal_i = np.array(palier_schedule[self.cls_fields.NC_MOIS_PALIER_DEBUT + str(pal_nb)]).reshape(n, 1)

        if self.cls_fields.NC_MOIS_PALIER_AMOR + str(pal_nb + 1) in palier_schedule.columns.tolist():
            mois_amor_pal_iplus1 = np.array(
                palier_schedule[self.cls_fields.NC_MOIS_PALIER_AMOR + str(pal_nb + 1)]).reshape(n, 1)
            mois_int_pal_iplus1 = np.array(
                palier_schedule[self.cls_fields.NC_MOIS_PALIER_DEBUT + str(pal_nb + 1)]).reshape(n, 1)
            mois_date_pal_iplus1 = np.array(
                palier_schedule[self.cls_fields.NC_MOIS_DATE_PALIER + str(pal_nb + 1)]).reshape(n, 1)
            mois_date_pal_iplus1 = np.nan_to_num(mois_date_pal_iplus1, nan=dar_mois + t + 1)
        else:
            mois_amor_pal_iplus1 = (amor_end_month + 1).reshape(n, 1)
            mois_int_pal_iplus1 = np.full((n, 1), t + 1)
            mois_date_pal_iplus1 = dar_mois + t + 2

        interests_schedule_b1 = ne.evaluate("current_month - mois_int_pal_i + 1")
        interests_schedule_b2 = ne.evaluate("current_month - mois_int_pal_iplus1")

        rates_schedule_b1 = ne.evaluate("current_month - mois_date_pal_i + dar_mois + 1")
        rates_schedule_b2 = ne.evaluate("current_month - mois_date_pal_iplus1 + dar_mois")

        is_fixed = palier_schedule[self.cls_fields.NC_RATE_TYPE_PALIER + str(pal_nb)].values.reshape(n, 1) == "FIXED"
        not_null_rate = palier_schedule[self.cls_fields.NC_RATE_TYPE_PALIER + str(pal_nb)].notnull().values
        is_tv = ~is_fixed.reshape(n) & not_null_rate

        date_pal_i_real = np.array(date_pal_i_real.fillna(self.cls_format.default_date)).astype(
            "datetime64[D]").reshape(n, 1)

        pal_sc_rates = self.sc_rates.copy()
        pal_sc_rates_ftp = self.sc_rates_ftp.copy()
        pal_sc_rates_lag = self.sc_rates_lag.copy()
        pal_sc_rates_ftp_lag = self.sc_rates_ftp_lag.copy()
        pal_tombee_fixing = self.tombee_fixing.copy()
        pal_period_fixing = self.period_fixing.copy()
        _n = pal_sc_rates[is_tv].shape[0]

        if pal_nb > 1 :
            date_mat = data_ldp[self.cls_fields.NC_LDP_MATUR_DATE_REAL].values.astype("datetime64[D]").reshape(n, 1)

            val_date_day = ut.dt2cal(date_pal_i_real)[:, :, 2].astype(int)
            mat_date_day = ut.dt2cal(date_mat)[:, :, 2].astype(int)

            period_begin_date, period_end_date, begin_period_was_changed, end_period_was_changed \
                = self.cls_cal.get_base_periods(data_ldp, current_month_max, mois_int_pal_i, t + 1, dar_mois, val_date_day,
                                                mat_date_day, date_pal_i_real, date_mat, mois_date_pal_i, n, t_max)

            col_accrual_basis = self.cls_fields.NC_ACCRUAL_BASIS_PALIER + str(pal_nb)
            col_nb_days_an = self.cls_fields.NB_DAYS_AN + str(pal_nb)
            palier_schedule = self.cls_format.format_base_calc(palier_schedule, col_accrual_basis, col_nb_days_an)


            pal_interests_calc_periods \
                = self.cls_cal.get_interest_payment_periods(palier_schedule, col_accrual_basis, period_end_date,
                                                            period_begin_date, current_month_max, begin_period_was_changed,
                                                            end_period_was_changed, t + 1, n)

            self.update_interests_periods(pal_interests_calc_periods, palier_schedule[col_nb_days_an].values,
                                          rates_schedule_b1, rates_schedule_b2, t, n)

            if len(is_tv[is_tv]) > 0:
                _n_tf = pal_sc_rates[~is_tv].shape[0]
                pal_fixing_cal, palier_schedule\
                    = self.get_fixing_calendar_palier(pal_nb, data_ldp, palier_schedule, date_pal_i_real, mois_int_pal_i,
                                                      period_begin_date, current_month_max, is_fixed, n, t, t_max)

                trade_date = data_ldp[self.cls_fields.NC_LDP_TRADE_DATE_REAL]
                date_mat = data_ldp[self.cls_fields.NC_LDP_MATUR_DATE_REAL]
                fixing_date_col = self.cls_fields.NC_FIXING_NEXT_DATE_PALIER + str(pal_nb)
                col_fixing_nb_days = self.cls_fields.FIXING_NB_DAYS_PAL + str(pal_nb)
                palier_schedule = self.cls_fix_cal.update_fixing_date(palier_schedule, pal_fixing_cal, fixing_date_col,
                                                                      col_fixing_nb_days, trade_date, date_mat, n, t_max)

                (pal_sc_rates, pal_sc_rates_lag, pal_tombee_fixing, palier_schedule, pal_sc_rates_ftp,
                 pal_sc_ftp_rates_lag, pal_period_fixing) \
                    = self.load_rate_scenarios_palier(data_ldp, pal_nb, palier_schedule, pal_sc_rates, pal_sc_rates_ftp,
                                                      pal_sc_rates_lag, pal_sc_rates_ftp_lag, pal_tombee_fixing,
                                                      pal_period_fixing, pal_fixing_cal, is_tv,
                                                      self.current_month_max, tx_params, dar_mois, _n, t, t_max)

                self.update_rates_palier_tv(pal_sc_rates, pal_sc_rates_lag, pal_sc_rates_ftp, pal_sc_rates_ftp_lag,
                                            rates_schedule_b1, rates_schedule_b2, is_tv, t)

                self.update_period_fixing_palier_tv(pal_tombee_fixing, pal_period_fixing, is_tv, rates_schedule_b1,
                                                    rates_schedule_b2, _n, t)

                self.update_tombee_fixing_palier_tv(pal_nb, palier_schedule, pal_tombee_fixing, is_tv,
                                                    rates_schedule_b1, _n, t)

            else:
                pal_sc_rates = self.sc_rates.copy()

        rate_ech = self.get_rate_ech_palier(pal_nb, pal_sc_rates, rates_schedule_b1, rates_schedule_b2, is_fixed,
                                            palier_schedule, amor_end_month, mois_amor_pal_i, mois_amor_pal_iplus1,
                                            self.current_month_max, n, t_max, t)

        self.update_period_fixing_palier(pal_nb, rates_schedule_b1, rates_schedule_b2, date_pal_i_real,
                                         palier_schedule, is_fixed, n, t)

        palier_schedule = self.update_rate_palier(palier_schedule, pal_nb, max_palier, mois_date_pal_iplus1, dar_mois,
                                                  n, t)

        capital_amor_shift_pal = np.array(
            palier_schedule[self.cls_fields.NC_DECALAGE_AMOR_PALIER + str(pal_nb)]).reshape(n, 1)
        cap_int_shift_pal = np.array(palier_schedule[self.cls_fields.NC_DECALAGE_INT_CAP + str(pal_nb)]).reshape(n, 1)
        amortizing_schedule_pal = (current_month - mois_amor_pal_i - capital_amor_shift_pal) % amor_freq

        self.rate_ech[pal_nb] = rate_ech
        self.is_fixed[pal_nb] = is_fixed

        self.nb_periods_pal[pal_nb] = ac.get_remaining_periods(mois_amor_pal_i, amor_end_month, capital_amor_shift_pal,
                                                               amor_freq, n)
        self.mois_amor_pal_i[pal_nb] = mois_amor_pal_i

        self.amor_schedule_b1[pal_nb] = ne.evaluate("current_month - mois_amor_pal_i + 1")
        self.amor_schedule_b2[pal_nb] = ne.evaluate("current_month - mois_amor_pal_iplus1")
        self.palier_schedule = palier_schedule
        self.amortizing_schedule[pal_nb] = amortizing_schedule_pal
        self.interests_schedule_b1[pal_nb] = interests_schedule_b1
        self.interests_schedule_b2[pal_nb] = interests_schedule_b2

        suspend_or_capitalize = np.array(
            palier_schedule[self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS + str(pal_nb)]).astype(
            bool).reshape(n, 1)
        suspend_or_capitalize = ne.evaluate("where((~linear_prof) & (~ech_const_prof), True, suspend_or_capitalize)")
        self.suspend_or_capitalize[pal_nb] = suspend_or_capitalize
        self.capitalize[pal_nb] = np.array(palier_schedule[self.cls_fields.NC_CAPITALIZE + str(pal_nb)]).astype(
            bool).reshape(n, 1)
        self.capital_amor_shift[pal_nb] = capital_amor_shift_pal
        self.cap_int_shift_pal[pal_nb] = cap_int_shift_pal

        self.amor_freq[pal_nb] = amor_freq
        self.ech_const_prof[pal_nb] = ech_const_prof
        self.linear_prof[pal_nb] = linear_prof
        self.linear_prof_ech[pal_nb] = linear_prof_ech

        self.accrual_basis_pal[pal_nb] = palier_schedule[self.cls_fields.NC_ACCRUAL_BASIS_PALIER + str(pal_nb)]

        return palier_schedule


    def update_interests_periods(self, pal_interest_periods, pal_nb_days_an, rates_schedule_b1, rates_schedule_b2, t, n):
        self.interests_periods[:, :t] = np.where((rates_schedule_b1 > 0) & (rates_schedule_b2 < 0),
                                            pal_interest_periods[:, :t], self.interests_periods[:, :t])

        self.year_nb_days = np.where((rates_schedule_b1 > 0) & (rates_schedule_b2 < 0),
                                            pal_nb_days_an.reshape(n, 1), self.year_nb_days)



    def get_fixing_calendar_palier(self, pal_nb, data_ldp, palier_schedule, date_pal_i_real,
                                   mois_int_pal_i, period_begin_date, current_month,
                                   is_fixed, n, t, t_max):

        fixing_date_col = self.cls_fields.NC_FIXING_NEXT_DATE_PALIER + str(pal_nb)

        periodicity = np.array(palier_schedule[self.cls_fields.NC_FREQ_INT_PALIER + str(pal_nb)]).reshape(n, 1)
        periodicity_cap = data_ldp[self.cls_fields.NC_LDP_CAPITALIZATION_PERIOD].fillna("").values.reshape(n, 1)
        periodicity_num = np.where(np.isin(periodicity, ["N", ""]).reshape(n, 1),
                                   data_ldp[self.cls_fields.NC_FREQ_CAP].values.reshape(n, 1),
                                   palier_schedule[self.cls_fields.NC_FREQ_INT_NUM_PALIER + str(pal_nb)].values.reshape(
                                       n, 1))

        decalage_deb_per = np.zeros((n, 1)).astype(int)

        fixing_date = palier_schedule[fixing_date_col].values.reshape(n, 1)
        fixing_date = np.where(is_fixed, date_pal_i_real, fixing_date)
        is_cap_floor = np.array(data_ldp[self.cls_fields.NC_PRODUCT_TYPE] == "CAP_FLOOR").reshape(n, 1)
        fixing_calendar_pal \
            = self.cls_fix_cal.get_fixing_calendar_ech(decalage_deb_per, periodicity, periodicity_cap, periodicity_num,
                                                       period_begin_date, current_month, mois_int_pal_i,
                                                       t + 1, fixing_date, is_cap_floor, n, t_max)

        return fixing_calendar_pal, palier_schedule

    ###############@profile
    def load_rate_scenarios_palier(self, data_ldp, pal_nb, palier_schedule, pal_sc_rates, pal_sc_ftp_rates,
                                   pal_sc_rates_lag, pal_sc_ftp_rates_lag, pal_tombee_fixing, pal_period_fixing,
                                   period_fixing_cal, is_tv, current_month, tx_params, dar_mois, _n, t_proj, t):

        palier_schedule \
            = self.cls_data_rate.get_curve_accruals_col(palier_schedule, tx_params,
                                                        {"CURVE_NAME": self.cls_fields.NC_CURVE_NAME_PALIER + str(
                                                            pal_nb),
                                                         "TENOR": self.cls_fields.NC_TENOR_NUM + str(pal_nb)}, is_tv)

        data_rate_sc_tv \
            = self.cls_data_rate.match_data_with_sc_rates(palier_schedule[is_tv].copy(), tx_params, t,
                                                          [self.cls_fields.NC_CURVE_NAME_PALIER + str(pal_nb),
                                                           self.cls_fields.NC_TENOR_PALIER + str(pal_nb)],
                                                          raise_error=False)

        data_spread_tv = self.get_data_spread_palier(palier_schedule, is_tv, pal_nb, _n, t)

        (pal_sc_rates[is_tv], pal_sc_rates_lag[is_tv], pal_tombee_fixing[is_tv], palier_schedule[is_tv],
         pal_sc_ftp_rates[is_tv],
         pal_sc_ftp_rates_lag[is_tv], pal_period_fixing[is_tv]) \
            = self.cls_rate.get_scenario_rates(palier_schedule[is_tv].copy(), data_rate_sc_tv.copy(),
                                               period_fixing_cal[is_tv].copy(), current_month[is_tv].copy(),
                                               self.cls_fields.NC_FIXING_NEXT_DATE_PALIER + str(pal_nb),
                                               self.cls_fields.NC_RATE_PALIER + str(pal_nb),
                                               self.cls_fields.NC_FIXING_PERIODICITY_NUM_PALIER + str(pal_nb),
                                               self.cls_fields.NC_MULT_FACT_PALIER + str(pal_nb),
                                               self.cls_fields.NC_MKT_SPREAD_PALIER + str(pal_nb),
                                               self.cls_fields.NC_FLOOR_STRIKE_PALIER + str(pal_nb),
                                               self.cls_fields.NC_CAP_STRIKE_PALIER + str(pal_nb),
                                               self.cls_fields.NC_TENOR_NUM + str(pal_nb),
                                               self.cls_fields.NC_TENOR_PALIER + str(pal_nb),
                                               self.cls_fields.NC_ACCRUAL_BASIS_PALIER + str(pal_nb),
                                               self.cls_fields.NC_FIXING_PERIODICITY_PALIER + str(pal_nb),
                                               data_ldp.loc[is_tv, self.cls_fields.NC_LDP_TRADE_DATE],
                                               data_ldp.loc[is_tv, self.cls_fields.NC_LDP_MATUR_DATE],
                                               self.cls_fields.NC_MOIS_DATE_PALIER + str(pal_nb),
                                               data_ldp.loc[is_tv, self.cls_fields.NC_LDP_CONTRACT_TYPE],
                                               self.cls_fields.FIXING_RULE_PAL + str(pal_nb),
                                               self.cls_fields.NC_LDP_TARGET_RATE,
                                               self.cls_fields.NC_LDP_FTP_FUNDING_SPREAD,
                                               _n, t_proj, t, data_spread=data_spread_tv)

        palier_schedule = palier_schedule.drop([self.cls_rate.cls_data_rate.ACCRUAL_METHOD,
                                                self.cls_rate.cls_data_rate.ACCRUAL_CONVERSION,
                                                self.cls_rate.cls_data_rate.STANDALONE_INDEX], axis=1)

        return (pal_sc_rates, pal_sc_rates_lag, pal_tombee_fixing, palier_schedule, pal_sc_ftp_rates,
                pal_sc_ftp_rates_lag, pal_period_fixing)

    def get_data_spread_palier(self, palier_schedule, is_tv, pal_nb, _n, t):
        palier_tv = palier_schedule[is_tv].copy()
        data_spread_tv = self.cls_rate.get_data_spread(palier_tv, _n, t,
                                                       self.cls_fields.NC_RATE_CODE_PALIER + str(pal_nb))
        return data_spread_tv

    def update_rates_palier_tv(self, pal_sc_rates, pal_sc_rates_lag, pal_sc_rates_ftp, pal_sc_rates_ftp_lag,
                               rates_schedule_b1, rates_schedule_b2, is_tv, t):
        self.sc_rates[is_tv, :t] = np.where((rates_schedule_b1[is_tv] > 0) & (rates_schedule_b2[is_tv] < 0),
                                            pal_sc_rates[is_tv, :t], self.sc_rates[is_tv, :t])

        self.sc_rates_ftp[is_tv, :t] = np.where((rates_schedule_b1[is_tv] > 0) & (rates_schedule_b2[is_tv] < 0),
                                                pal_sc_rates_ftp[is_tv, :t], self.sc_rates_ftp[is_tv, :t])

        self.sc_rates_lag[is_tv, :t] = np.where((rates_schedule_b1[is_tv] > 1) & (rates_schedule_b2[is_tv] <= 0),
                                                pal_sc_rates_lag[is_tv, :t], self.sc_rates_lag[is_tv, :t])

        self.sc_rates_ftp_lag[is_tv, :t] = np.where((rates_schedule_b1[is_tv] > 1) & (rates_schedule_b2[is_tv] <= 0),
                                                    pal_sc_rates_ftp_lag[is_tv, :t], self.sc_rates_ftp_lag[is_tv, :t])

    def update_period_fixing_palier_tv(self, pal_tombee_fixing, pal_period_fixing,
                                       is_tv, rates_schedule_b1, rates_schedule_b2, _n, t):
        pal_tombee_fixing[is_tv, :t] = np.where((rates_schedule_b1[is_tv] > 0) & (rates_schedule_b2[is_tv] < 0),
                                                pal_tombee_fixing[is_tv, :t], self.tombee_fixing[is_tv, :t])

        period_fixing_before = np.prod(np.where(rates_schedule_b1[is_tv] > 0, True,
                                                self.period_fixing[is_tv, :t]), axis=1).reshape(_n, 1)
        self.period_fixing[is_tv, :t] = np.where((rates_schedule_b1[is_tv] > 0) & (period_fixing_before == 1),
                                                 pal_period_fixing[is_tv, :t], self.period_fixing[is_tv, :t])

    ###############@profile
    def update_tombee_fixing_palier_tv(self, pal_nb, palier_schedule, pal_tombee_fixing, is_tv,
                                       rates_schedule_b1, _n, t):

        date_fix_month = ut.dt2cal(
            palier_schedule.loc[is_tv, self.cls_fields.NC_FIXING_NEXT_DATE_PALIER + str(pal_nb)].values)[:, 1].astype(
            int).reshape(_n, 1)

        date_fix_year = ut.dt2cal(
            palier_schedule.loc[is_tv, self.cls_fields.NC_FIXING_NEXT_DATE_PALIER + str(pal_nb)].values)[:, 0].astype(
            int).reshape(_n, 1)

        fixing_date_previous_pal = self.get_fixing_date_previous_pal(palier_schedule[is_tv], pal_nb)
        date_fix_month_moins1 = ut.dt2cal(fixing_date_previous_pal)[:, 1].astype(int).reshape(_n, 1)
        date_fix_year_moins1 = ut.dt2cal(fixing_date_previous_pal)[:, 0].astype(int).reshape(_n, 1)

        self.tombee_fixing[is_tv, :t] = np.where(
            (rates_schedule_b1[is_tv] == 1) & (date_fix_month == date_fix_month_moins1)
            & (date_fix_year == date_fix_year_moins1),
            np.minimum(pal_tombee_fixing[is_tv, :t], self.tombee_fixing[is_tv, :t]),
            pal_tombee_fixing[is_tv, :t])

    def update_period_fixing_palier(self, pal_nb, rates_schedule_b1, rates_schedule_b2, date_pal_i_real,
                                    palier_schedule, is_fixed, n, t):
        # ERREUR_RCO
        period_fixing_before = np.prod(np.where(rates_schedule_b1 > 0, True, self.period_fixing[:, :t]),
                                       axis=1).reshape(n, 1)
        # ERREUR_RCO
        is_date_palier_inf_fixing_date = self.compare_date_palier_fixing_date(pal_nb, date_pal_i_real, palier_schedule,
                                                                              n)

        # ERREUR_RCO
        self.period_fixing[:, :t] = np.where((rates_schedule_b1 > 0) & (rates_schedule_b2 < 0) & (is_fixed)
                                             & (period_fixing_before == 1) & is_date_palier_inf_fixing_date, True,
                                             self.period_fixing[:, :t])

    ###############@profile
    def get_rate_ech_palier(self, pal_nb, pal_sc_rates, rates_schedule_b1, rates_schedule_b2, is_fixed,
                            palier_schedule, amor_end_month, mois_amor_pal_i, mois_amor_pal_iplus1, current_month, n,
                            t_max, t):

        duree_palier = np.maximum(0, amor_end_month - mois_amor_pal_i)
        fixed_rate_ech = np.array(palier_schedule[self.cls_fields.NC_RATE_PALIER + str(pal_nb)])
        rate_ech = self.cls_rate.mean_sc_rate(pal_sc_rates, n, duree_palier, mois_amor_pal_iplus1 - 1,
                                              mois_amor_pal_i, current_month, t_max, ~is_fixed.reshape(n),
                                              fixed_rate_ech)
        cond_palier = (rates_schedule_b1 > 0) & (rates_schedule_b2 < 0) & (is_fixed)
        cond_palier_lag = (rates_schedule_b1 > 1) & (rates_schedule_b2 <= 0) & (is_fixed)
        self.sc_rates[:, :t] = np.where(cond_palier, 12 * rate_ech, self.sc_rates[:, :t])
        self.sc_rates_lag[:, :t] = np.where(cond_palier_lag, 12 * rate_ech, self.sc_rates_lag[:, :t])
        return rate_ech

    def update_rate_palier(self, palier_schedule, pal_nb, max_palier, mois_date_pal_iplus1, dar_mois, n, t):
        if pal_nb < max_palier:
            index_rate = np.maximum(0, np.minimum((mois_date_pal_iplus1 - dar_mois).reshape(n) - 2, t - 1)).astype(int)

            not_fixed = ((palier_schedule[self.cls_fields.NC_RATE_TYPE_PALIER + str(pal_nb + 1)] != "FIXED").values
                         | (palier_schedule[self.cls_fields.NC_RATE_PALIER + str(pal_nb + 1)] == -100)).values

            not_fixed_rate = self.sc_rates[np.arange(0, n), index_rate] / 12
            first_palier_rate = palier_schedule[self.cls_fields.NC_RATE_PALIER + str(1)].values
            next_palier_rate = palier_schedule[self.cls_fields.NC_RATE_PALIER + str(pal_nb + 1)]

            palier_schedule[self.cls_fields.NC_RATE_PALIER + str(pal_nb + 1)] \
                = np.where(not_fixed & (index_rate >= 0), not_fixed_rate, np.where(not_fixed & (index_rate < 0),
                                                                                   first_palier_rate, next_palier_rate))

        return palier_schedule

    def get_fixing_date_previous_pal(self, palier_schedule, pal_nb):
        fixing_date = palier_schedule[self.cls_fields.NC_FIXING_NEXT_DATE_PALIER + str(pal_nb - 1)].values
        return fixing_date

    def compare_date_palier_fixing_date(self, pal_nb, date_pal_i_real, palier_schedule, n):
        if pal_nb >= 2:
            has_prev_fxi_date = palier_schedule[
                                    self.cls_fields.NC_RATE_TYPE_PALIER + str(pal_nb - 1)].values == "FLOATING"
            is_date_palier_inf_fixing_date = np.full((n, 1), False)
            _n = has_prev_fxi_date[has_prev_fxi_date].shape[0]
            is_date_palier_inf_fixing_date[has_prev_fxi_date] = (
                        date_pal_i_real[has_prev_fxi_date].astype("datetime64[D]") <=
                        self.get_fixing_date_previous_pal(palier_schedule[has_prev_fxi_date], pal_nb)
                        .reshape(_n, 1).astype("datetime64[D]"))
        else:
            is_date_palier_inf_fixing_date = np.array([True] * n).reshape(n, 1)

        return is_date_palier_inf_fixing_date

    def get_begin_capital_palier(self, begin_capital, rate_adjusted_nominal, rate_adjusted_ech_cum, pal_nb, n, t):
        if pal_nb > 1:
            temp_capital = (rate_adjusted_nominal - rate_adjusted_ech_cum)
            index_capital = np.maximum(0, np.minimum(self.mois_amor_pal_i[pal_nb].reshape(n) - 2, t - 1)).astype(int)
            begin_capital_palier = temp_capital[np.arange(0, n), index_capital].reshape(n, 1)
        else:
            begin_capital_palier = begin_capital.copy()
        return begin_capital_palier

    def get_rate_factor_for_nominal_palier(self, rate_mtx, interests_schedule_b1, interests_schedule_b2,
                                           interests_calc_periods,
                                           nb_days_an):
        rate_factor = ac.get_rate_factor(rate_mtx, interests_schedule_b1, interests_calc_periods, nb_days_an)
        rate_factor = ne.evaluate("where(interests_schedule_b2 >= 0, 0, rate_factor)")
        return rate_factor

    def get_rate_adjusted_ech_palier(self, rate_adjusted_ech_cum, compounded_rate, palier_schedule, rate, begin_capital,
                                     prof_ech_const, prof_linear, prof_linear_ech, pal_nb, freq_amor, calendar_amor_adj,
                                     interests_schedule_b1, suspend_or_capitalize,
                                     interests_schedule_b2, nb_periods, mois_pal_i, is_fixed,
                                     rate_adjusted_nominal, amor_end_month, capital_amor_shift,
                                     n, t):
        """ Calcul de la partie: P(i) *  Σ(j=1 à p)( e(j) * Σ(k=t(j) à k=min(i,t(j+1)-1))(1/P(k)) """
        const_ech = np.array(palier_schedule[self.cls_fields.NC_VAL_PALIER + str(pal_nb)]).reshape(n, 1)
        orig_ech_value = const_ech.copy()
        const_ech = np.where(prof_ech_const & (~is_fixed), -10000, const_ech)
        const_ech = np.where(prof_linear & ~prof_linear_ech & (~is_fixed), -10000, const_ech)

        for nb_pal_sup in range(1, 3):
            if nb_pal_sup == 1:
                if (~is_fixed).any(axis=0):
                    orig_ech, plage \
                        = ac.get_plage_application_orig_ech(prof_ech_const, freq_amor, orig_ech_value, mois_pal_i,
                                                            calendar_amor_adj, self.current_month, n, t, is_fixed,
                                                            val=-10000)

                    plage_app_orig_ech = np.where(interests_schedule_b2 >= 0, 1, plage.copy())

                    adjusted_rate_for_orig_ech = ac.get_compounded_rate_for_ech(compounded_rate, calendar_amor_adj,
                                                                                interests_schedule_b1,
                                                                                interests_schedule_b2=plage_app_orig_ech)
                else:
                    adjusted_rate_for_orig_ech = np.zeros((n, t))
                    orig_ech = 0
                    plage = np.ones((n, t))
                    plage_app_orig_ech = np.ones((n, t))
            else:

                decalage_amor = np.nan_to_num(np.amax(adjusted_rate_for_orig_ech.astype(int), axis=1))

                pal_begin_month = mois_pal_i + np.where(decalage_amor.reshape(n, 1) > 0, freq_amor, 0)

                nb_periods_pal = ac.get_remaining_periods(pal_begin_month, amor_end_month,
                                                          capital_amor_shift, freq_amor, n)

                temp_capital = (rate_adjusted_nominal - rate_adjusted_ech_cum - adjusted_rate_for_orig_ech * orig_ech)
                index_capital = np.maximum(0, np.minimum(pal_begin_month.reshape(n) - 2, t - 1)).astype(int)
                begin_capital_pal_sup = temp_capital[np.arange(0, n), index_capital].reshape(n, 1)
                begin_capital_pal_sup = np.where(np.any(plage_app_orig_ech == -1, axis=1).reshape(n, 1),
                                                 begin_capital_pal_sup, begin_capital)

                const_ech \
                    = ac.get_ech_const(const_ech, rate, begin_capital_pal_sup, nb_periods_pal * freq_amor,
                                       prof_ech_const, prof_linear, prof_linear_ech,
                                       freq_amor, val=-10000)

                adj_rate_for_ech = ac.get_compounded_rate_for_ech(compounded_rate, calendar_amor_adj,
                                                                  interests_schedule_b1 * (plage),
                                                                  interests_schedule_b2=interests_schedule_b2)

        rate_adjusted_ech = ne.evaluate("const_ech * adj_rate_for_ech + orig_ech * adjusted_rate_for_orig_ech")

        rate_adjusted_ech = np.where(suspend_or_capitalize, 0, rate_adjusted_ech)
        rate_adjusted_ech_cum = ne.evaluate("rate_adjusted_ech_cum + rate_adjusted_ech")

        return rate_adjusted_ech_cum

    def create_pal_dics(self):
        self.amor_freq = {}
        self.ech_const_prof = {}
        self.linear_prof = {}
        self.linear_prof_ech = {}
        self.rate_ech = {}
        self.interests_schedule_b1 = {}
        self.interests_schedule_b2 = {}
        self.suspend_or_capitalize = {}
        self.nb_periods_pal = {}
        self.mois_amor_pal_i = {}
        self.is_fixed = {}
        self.amortizing_schedule = {}
        self.begin_capital_palier = {}
        self.rate_mtx = {}
        self.real_rate_mtx = {}
        self.capital_amor_shift = {}
        self.amor_schedule_b1 = {}
        self.amor_schedule_b2 = {}
        self.capitalize = {}
        self.cap_int_shift_pal = {}
        self.accrual_basis_pal = {}
