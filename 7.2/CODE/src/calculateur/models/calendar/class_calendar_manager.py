import numpy as np
import pandas as pd
import numexpr as ne
from dateutil.relativedelta import relativedelta
from calculateur.models.utils import utils as ut
from calculateur.calc_params.model_params import *
import datetime

nan = np.nan


class Calendar_Manager():
    """
    Load les données utilisateurs
    """

    def __init__(self, cls_hz_params, cls_fields, name_product):
        self.cls_hz_params = cls_hz_params
        self.cls_fields = cls_fields
        self.get_absent_dates()
        self.name_product = name_product

    def get_absent_dates(self):
        self.absent_date_str = '01/01/1900'
        self.absent_date_np = np.array([datetime.datetime(1900, 1, 1)]).astype("datetime64[D]")

    def get_calendar_coeff(self):
        self.start = self.cls_hz_params.dar_usr - relativedelta(months=1)
        self.period = self.cls_hz_params.max_proj_ecoul + 1
        date_deb = pd.DataFrame(pd.date_range(start=self.start, periods=self.period, freq='MS'))
        date_fin = pd.DataFrame(pd.date_range(start=self.start + relativedelta(months=1),
                                              periods=self.period, freq='MS'))

        end_month_date_fin = pd.DataFrame(pd.date_range(start=self.start + relativedelta(months=2),
                                                        periods=self.period, freq="ME"))

        end_month_date_deb = pd.DataFrame(pd.date_range(start=self.start + relativedelta(months=1),
                                                        periods=self.period, freq="ME"))

        self.date_deb = (np.array([x.date() for x in date_deb[0]]).astype("datetime64[D]")
                         .reshape(1, self.cls_hz_params.max_proj_ecoul + 1))

        self.date_fin = (np.array([x.date() for x in date_fin[0]]).astype("datetime64[D]")
                         .reshape(1, self.cls_hz_params.max_proj_ecoul + 1))

        self.end_month_date_fin = (np.array([x.date() for x in end_month_date_fin[0]]).astype("datetime64[D]")
                                   .reshape(1, self.cls_hz_params.max_proj_ecoul + 1))

        self.end_month_date_deb = (np.array([x.date() for x in end_month_date_deb[0]]).astype("datetime64[D]")
                                   .reshape(1, self.cls_hz_params.max_proj_ecoul + 1))

        self.delta_days = (self.date_fin - self.date_deb).astype(int)

        self.end_month_date_fin_day = ut.dt2cal(self.end_month_date_fin)[:, :, 2].astype(int)
        self.end_month_date_deb_day = ut.dt2cal(self.end_month_date_deb)[:, :, 2].astype(int)

    def generate_amortization_and_interest_payment_begin_dates(self, data):
        value_date_month = np.array(data[self.cls_fields.NC_LDP_VALUE_DATE])
        maturity_date_month = np.array(data[self.cls_fields.NC_LDP_MATUR_DATE])
        val_date = data[self.cls_fields.NC_LDP_VALUE_DATE_REAL]
        mat_date = data[self.cls_fields.NC_LDP_MATUR_DATE_REAL]
        first_amor_date = np.array(data[self.cls_fields.NC_LDP_FIRST_AMORT_DATE])
        broken_period = np.array(data[self.cls_fields.NC_LDP_BROKEN_PERIOD])
        dar_mois = self.cls_hz_params.dar_mois

        data[self.cls_fields.NC_DATE_DEBUT_AMOR], data[self.cls_fields.NC_DATE_FIN_AMOR] = \
            self.calculate_amortizing_and_interest_payment_begin_month(value_date_month, maturity_date_month, val_date,
                                                                       mat_date, first_amor_date, broken_period, dar_mois)

        data[self.cls_fields.NC_DATE_DEBUT], data[self.cls_fields.NC_DATE_FIN] = \
            self.calculate_amortizing_and_interest_payment_begin_month(value_date_month, maturity_date_month, val_date, mat_date,
                                                 first_amor_date, broken_period, dar_mois, typo="normal")

        data[self.cls_fields.NC_DECALAGE_AMOR_CAP] \
            = self.calculate_month_shift_for_non_monthly_amor(data, self.cls_fields.NC_FREQ_AMOR,
                                                              maturity_date_month,
                                                              data[self.cls_fields.NC_DATE_DEBUT_AMOR].values)

        data[self.cls_fields.NC_DECALAGE_INT_CAP] \
            = self.calculate_month_shift_for_non_monthly_amor(data, self.cls_fields.NC_FREQ_CAP,
                                                              maturity_date_month,
                                                              data[self.cls_fields.NC_DATE_DEBUT].values)

        data[self.cls_fields.NC_FREQ_INT + "_REAL"] = np.where(data[self.cls_fields.NC_LDP_FREQ_INT] == "N",
                                                               maturity_date_month - data[
                                                                   self.cls_fields.NC_DATE_DEBUT].values + 1,
                                                               data[self.cls_fields.NC_FREQ_INT])

        data[self.cls_fields.NC_FREQ_INT + "_REAL"] = np.where(data[self.cls_fields.NC_CAPITALIZE].values,
                                                               maturity_date_month - data[
                                                                   self.cls_fields.NC_DATE_DEBUT].values + 1,
                                                               data[self.cls_fields.NC_FREQ_INT + "_REAL"])

        data[self.cls_fields.NC_DECALAGE_VERS_INT] \
            = self.calculate_month_shift_for_non_monthly_amor(data, self.cls_fields.NC_FREQ_INT + "_REAL",
                                                              maturity_date_month,
                                                              data[self.cls_fields.NC_DATE_DEBUT].values)

        data = self.calculate_month_shift_for_begin_per(data, self.cls_fields.NC_FREQ_INT,
                                                        self.cls_fields.NC_DECALAGE_DEB_PER)
        data = self.calculate_month_shift_for_begin_per(data, self.cls_fields.NC_FREQ_CAP,
                                                        self.cls_fields.NC_DECALAGE_DEB_PER_CAP)

        data = self.calculate_month_shift_for_begin_per(data, self.cls_fields.NC_FREQ_CAP,
                                                        self.cls_fields.NC_DECALAGE_DEB_PER_CAP)

        data[self.cls_fields.NC_DECALAGE_DEB_PER_FIX] = np.where(
            data[self.cls_fields.NC_LDP_FREQ_INT].fillna("").isin(["N", ""]),
            data[self.cls_fields.NC_DECALAGE_DEB_PER_CAP],
            data[self.cls_fields.NC_DECALAGE_DEB_PER])

        data = self.calculate_month_shift_for_begin_per(data, self.cls_fields.NC_FIXING_PERIODICITY_NUM,
                                                        self.cls_fields.NC_DECALAGE_DEB_FIXING_PER)

        return data

    def calculate_amortizing_and_interest_payment_begin_month(self, value_date_month, maturity_date_month, val_date, mat_date,
                                        first_amor_date, broken_period, dar_mois, typo="amor"):

        n = value_date_month.shape[0]
        val_date_pd = pd.to_datetime(val_date)
        day_val_date = np.array(val_date_pd.dt.day).reshape(n, 1)

        mat_date_pd = pd.to_datetime(mat_date)
        day_mat_date = np.array(mat_date_pd.dt.day).reshape(n, 1)

        day_mat_and_day_val_end_of_month = (day_mat_date == (
                mat_date_pd + pd.offsets.MonthEnd(n=0)).dt.day.values.reshape(n, 1)) \
                                           & (day_val_date == (
                val_date_pd + pd.offsets.MonthEnd(n=0)).dt.day.values.reshape(n, 1))
        day_mat_and_day_val_end_of_month = day_mat_and_day_val_end_of_month.reshape(n, 1)

        broken_period = np.array(broken_period).reshape(n, 1)
        value_date_month = np.array(value_date_month).reshape(n, 1)
        first_amor_date = np.array(first_amor_date).reshape(n, 1)
        maturity_date_month = np.array(maturity_date_month).reshape(n, 1)

        if typo != "amor":
            # Cas par défaut : mois de début des intérêts: mois de la value_date + 1
            mois_deb_def = value_date_month + 1
        else:
            # Cas par défaut : mois de début de l'amortissement: max(mois de la value_date + 1, mois de la première date d'amortissement + 1)
            mois_deb_def = np.maximum(first_amor_date, value_date_month + 1)

        # Cas  Start Short + jour de la date de valeur < jour de la date de maturité => mois de début des intérêts et amor anticipé de 1 mois
        mois_deb = np.where((broken_period == 'SS') & (day_val_date < day_mat_date) & ~day_mat_and_day_val_end_of_month,
                            mois_deb_def - 1, mois_deb_def)

        # Cas  Start Long + jour de la date de valeur > jour de la date de maturité => mois de début des intérêts et amor retardé de 1 mois
        mois_deb = np.where((broken_period == 'SL') & (day_val_date > day_mat_date) & day_mat_and_day_val_end_of_month,
                            mois_deb_def + 1, mois_deb)
        mois_deb = np.maximum(dar_mois + 1, mois_deb)
        #mois_deb = np.minimum(mois_deb, dar_mois + t + 2)

        mois_fin = np.where((broken_period == 'EL') & (day_val_date > day_mat_date), maturity_date_month - 1,
                            maturity_date_month)

        return mois_deb.reshape(n), mois_fin.reshape(n)

    def calculate_month_shift_for_begin_per(self, data_ldp_hab, col, col_out):
        freq = data_ldp_hab[col]
        f_duree = (freq != 1)
        dar_mois = self.cls_hz_params.dar_mois
        mois_dar = self.cls_hz_params.dar_usr.month
        decalage = 0
        if f_duree.any(axis=0):
            date_mat = data_ldp_hab[self.cls_fields.NC_LDP_MATUR_DATE]
            val_date = data_ldp_hab[self.cls_fields.NC_LDP_VALUE_DATE]

            mois_fin = date_mat - (mois_dar)
            decalage = mois_fin % freq

            decalage = np.where(val_date <= dar_mois, decalage, 0)
        data_ldp_hab[col_out] = decalage

        return data_ldp_hab

    def calculate_month_shift_for_non_monthly_amor(self, data, col_freq_amor, maturity_month, mois_depart_amor):
        freq_amor = data[col_freq_amor]
        f_duree_amor = (freq_amor != 1)
        decalage = 0
        if f_duree_amor.any(axis=0):
            mois_fin = maturity_month - (mois_depart_amor)
            decalage = mois_fin % freq_amor
        return decalage

    def prepare_calendar_parameters(self, clas_proj):
        data_ldp = clas_proj.data_ldp
        dar_mois = clas_proj.cls_hz_params.dar_mois
        n = clas_proj.data_ldp.shape[0]

        clas_proj.data_ldp = self.generate_amortization_and_interest_payment_begin_dates(data_ldp)
        mois_depart, mois_fin = self.load_interest_payment_calendar_parameters(data_ldp, dar_mois, n)
        mois_depart_amor, mois_fin_amor, drac_amor = self.load_amor_calendar_parameters(data_ldp, dar_mois, n)

        self.mois_depart = mois_depart
        self.mois_fin = mois_fin
        self.mois_depart_amor = mois_depart_amor
        self.mois_fin_amor = mois_fin_amor
        self.drac_amor = drac_amor
        clas_proj.data_ldp = data_ldp

    def get_calendar_periods(self, clas_proj):
        data_ldp = clas_proj.data_ldp
        current_month = clas_proj.current_month
        current_month_max = clas_proj.current_month_max
        dar_mois = clas_proj.cls_hz_params.dar_mois
        n = clas_proj.n
        t_max = clas_proj.t_max
        t = clas_proj.t

        self.get_val_date_mat_date_params(data_ldp, n)

        data_ldp, rarn_periods, interests_calc_periods, period_begin_date, period_end_date \
            = self.get_all_periods(data_ldp, current_month_max, self.mois_depart, self.mois_fin, dar_mois, n, t_max, t)

        self.current_month = current_month
        self.current_month_max = current_month_max
        self.period_begin_date = period_begin_date
        self.period_end_date = period_end_date
        self.rarn_periods = rarn_periods
        self.interests_calc_periods = interests_calc_periods
        clas_proj.data_ldp = data_ldp

    def load_interest_payment_calendar_parameters(self, data_hab, dar_mois, n):
        # calcule les mois de départ des RARN relatifs à la DAR
        begin_month_date = np.array(data_hab[self.cls_fields.NC_DATE_DEBUT]).reshape(n, 1).astype(int)
        mois_depart = ne.evaluate("begin_month_date - dar_mois")
        mois_depart = np.array(mois_depart.astype(int)).reshape(n, 1)

        end_month_date = np.array(data_hab[self.cls_fields.NC_DATE_FIN]).reshape(n, 1).astype(int)
        mois_fin = ne.evaluate("end_month_date - dar_mois")
        mois_fin = np.array(mois_fin.astype(int)).reshape(n, 1)

        return mois_depart, mois_fin

    def load_rarn_calendar_parameters(self, cls_proj):
        mois_depart = self.mois_depart
        current_month = cls_proj.current_month
        data_ldp = cls_proj.data_ldp
        n = data_ldp.shape[0]
        dar_mois = self.cls_hz_params.dar_mois
        maturity_date_month = np.array(data_ldp[self.cls_fields.NC_LDP_MATUR_DATE]).reshape(n, 1)
        self.maturity_date_month = maturity_date_month
        drac_rarn_m1 = ne.evaluate("maturity_date_month - dar_mois")
        drac_rarn_m1 = drac_rarn_m1.reshape(n, 1)
        drac_rarn = ne.evaluate("drac_rarn_m1 - current_month + 1")
        drac_rarn = ne.evaluate("where(current_month < mois_depart, nan, drac_rarn)")
        drac_rarn = ne.evaluate("where(drac_rarn < 0, nan, drac_rarn)")  # durée résiduelle dynamique
        value_date_month = np.array(data_ldp[self.cls_fields.NC_LDP_VALUE_DATE]).reshape(n, 1)
        drac_init = ne.evaluate("maturity_date_month - value_date_month")

        self.drac_init = ne.evaluate("drac_init - 1")  # calibrage RCO
        self.drac_rarn = ne.evaluate("drac_rarn - 1")  # calibrage RCO

    def load_amor_calendar_parameters(self, data_hab, dar_mois, n):
        # calcule les mois de départ des amortissements relatifs à la DAR
        amor_begin_date_month = np.array(data_hab[self.cls_fields.NC_DATE_DEBUT_AMOR]).reshape(n, 1).astype(int)
        amor_end_date_month = np.array(data_hab[self.cls_fields.NC_DATE_FIN_AMOR]).reshape(n, 1).astype(int)

        mois_dep_amor = ne.evaluate("amor_begin_date_month - dar_mois")
        mois_dep_amor = np.array(mois_dep_amor.astype(int)).reshape(n, 1)

        mois_fin_amor = ne.evaluate("amor_end_date_month - dar_mois")
        mois_fin_amor = np.array(mois_fin_amor.astype(int)).reshape(n, 1)

        drac_amor = ne.evaluate("amor_end_date_month - amor_begin_date_month + 1")

        return mois_dep_amor, mois_fin_amor, drac_amor

    def get_all_periods(self, data_ldp, current_month, mois_depart_int, mois_fin_int, dar_mois, n, t_max, t):

        period_begin_date, period_end_date, begin_period_was_changed, end_period_was_changed \
            = self.get_base_periods(data_ldp, current_month, mois_depart_int, mois_fin_int, dar_mois,
                                    self.val_date_day, self.mat_date_day, self.value_date, self.mat_date,
                                    self.value_date_month, n, t_max)

        rarn_periods = self.get_rarn_periods(period_end_date, period_begin_date, current_month, mois_fin_int)

        interests_calc_periods = self.get_interest_payment_periods(data_ldp,self.cls_fields.NC_LDP_ACCRUAL_BASIS,
                                                                   period_end_date, period_begin_date,
                                                                   current_month, begin_period_was_changed,
                                                                   end_period_was_changed, mois_fin_int, n)

        data_ldp = self.get_nb_days_M0(data_ldp, dar_mois, n)

        return (data_ldp, rarn_periods[:, :t], interests_calc_periods[:, :t], period_begin_date, period_end_date)

    def get_val_date_mat_date_params(self, data_ldp, n):
        value_date = data_ldp[self.cls_fields.NC_LDP_VALUE_DATE + "_REAL"]
        mat_date = data_ldp[self.cls_fields.NC_LDP_MATUR_DATE + "_REAL"]

        self.val_date_day = value_date.dt.day
        self.mat_date_day = mat_date.dt.day

        self.value_date = np.array([x.date() for x in value_date]).astype("datetime64[D]").reshape(n, 1)
        self.mat_date = np.array([x.date() for x in mat_date]).astype("datetime64[D]").reshape(n, 1)

        self.value_date_month = np.array(data_ldp[self.cls_fields.NC_LDP_VALUE_DATE]).reshape(n, 1)

    def get_nb_days_M0(self, data_ldp, dar_mois, n):
        val_date_day = np.array(self.val_date_day).reshape(n, 1)
        mat_date_day = np.array(self.mat_date_day).reshape(n, 1)

        data_ldp[self.cls_fields.NB_DAYS_M0] = np.where(self.value_date_month > dar_mois, val_date_day - 1,
                                                        mat_date_day - 1)

        return data_ldp

    def get_interest_payment_periods(self, data_ldp, accrual_basis_col, period_end_date, period_begin_date,
                                     current_month, begin_period_was_changed, end_period_was_changed, mois_fin_int, n):
        interests_calc_periods = (period_end_date - period_begin_date).astype("timedelta64[D]").astype(int)
        base_calc_cond = data_ldp[accrual_basis_col].isin(["30/360", "30E/360", "30A/360"]).values.reshape(n, 1)
        interests_calc_periods = np.where(base_calc_cond & ~(begin_period_was_changed | end_period_was_changed), 30,
                                          interests_calc_periods)
        interests_calc_periods = np.where(current_month <= mois_fin_int, interests_calc_periods, 0)
        return interests_calc_periods

    def get_rarn_periods(self, period_end_date, period_begin_date, current_month, mois_fin_int):
        rarn_periods = (period_end_date - period_begin_date).astype("timedelta64[D]").astype(int)
        rarn_periods = np.where(current_month <= mois_fin_int, rarn_periods, 0)
        return rarn_periods

    def get_base_periods(self, data_ldp, current_month, mois_depart_int, mois_fin_int, dar_mois,
                         val_date_day, mat_date_day, val_date, mat_date, value_date_month, n, t):

        broken_period = np.array(data_ldp[self.cls_fields.NC_LDP_BROKEN_PERIOD])

        period_begin_date, period_end_date = self.get_unadjusted_periods(val_date_day, mat_date_day, broken_period,
                                                                         current_month, mois_depart_int,
                                                                         value_date_month,
                                                                         dar_mois, n, t)

        (period_begin_date, period_end_date, begin_period_was_changed, end_period_was_changed,
         val_date, mat_date) \
            = self.apply_broken_period_adjst_to_periods(value_date_month, val_date, mat_date, current_month,
                                                        mois_depart_int,
                                                        broken_period,
                                                        period_begin_date,
                                                        period_end_date, n, mois_fin_int,
                                                        dar_mois, val_date_day, mat_date_day)

        period_begin_date = self.apply_business_day_rule_to_periods(data_ldp, period_begin_date, [val_date, mat_date],
                                                                    n)
        period_end_date = self.apply_business_day_rule_to_periods(data_ldp, period_end_date, [val_date, mat_date], n)

        return period_begin_date, period_end_date, begin_period_was_changed, end_period_was_changed

    def apply_broken_period_adjst_to_periods(self, value_date_month, value_date, mat_date, current_month, mois_depart,
                                             broken_period, period_begin_date, period_end_date, n, mois_fin, dar_mois,
                                             day_val, day_mat):
        broken_period = broken_period.reshape(n, 1)
        cond_forward = value_date_month >= dar_mois
        day_mat = np.array(day_mat).reshape(n, 1)
        day_val = np.array(day_val).reshape(n, 1)

        # begin_is_first_day = pd.to_datetime(period_end_date[:, 0]).day == 1
        # begin_is_first_day = begin_is_first_day & (pd.to_datetime(period_begin_date[:, 0]).month == self.cls_hz_params.dar_usr.month)
        # begin_is_first_day = begin_is_first_day & (pd.to_datetime(period_begin_date[:, 0]).year == self.cls_hz_params.dar_usr.year)

        day_mat_and_day_val_end_of_month = (day_mat == (
                pd.to_datetime(mat_date.reshape(n)) + pd.offsets.MonthEnd(n=0)).day.values.reshape(n, 1)) \
                                           & (day_val == (
                pd.to_datetime(value_date.reshape(n)) + pd.offsets.MonthEnd(n=0)).day.values.reshape(n, 1))
        day_mat_and_day_val_end_of_month = day_mat_and_day_val_end_of_month.reshape(n, 1)

        period_begin_date_bis = np.where(
            (current_month == mois_depart) & (np.isin(broken_period, ['SS', 'SL'])) & cond_forward
            & (~day_mat_and_day_val_end_of_month),
            value_date, period_begin_date)

        period_end_date_bis = np.where(
            (current_month == mois_fin) & (np.isin(broken_period, ['ES', 'EL'])) & (~day_mat_and_day_val_end_of_month),
            mat_date, period_end_date)

        # (period_begin_date, period_end_date, period_begin_date_bis,
        # period_end_date_bis) = self.correct_dates_if_day_beg_is_first_day_after_dar(begin_is_first_day, period_begin_date,
        #                                                period_end_date, period_begin_date_bis, period_end_date_bis, n)

        begin_period_was_changed = period_begin_date != period_begin_date_bis
        end_period_was_changed = period_end_date != period_end_date_bis

        period_end_date_bis = period_end_date_bis.astype("datetime64[D]")
        period_begin_date_bis = period_begin_date_bis.astype("datetime64[D]")

        return (period_begin_date_bis, period_end_date_bis, begin_period_was_changed,
                end_period_was_changed, value_date, mat_date)

    def correct_dates_if_day_beg_is_first_day_after_dar(self, begin_is_first_day, period_begin_date,
                                                        period_end_date, period_begin_date_bis, period_end_date_bis, n):
        period_begin_date_bis = np.where(begin_is_first_day.reshape(n, 1), ut.roll_and_null(period_begin_date, -1),
                                         period_begin_date_bis)
        period_end_date_bis = np.where(begin_is_first_day.reshape(n, 1), ut.roll_and_null(period_end_date, -1),
                                       period_end_date_bis)
        period_begin_date = np.where(begin_is_first_day.reshape(n, 1), period_begin_date_bis, period_begin_date)
        period_end_date = np.where(begin_is_first_day.reshape(n, 1), period_end_date_bis, period_end_date)

        return period_begin_date, period_end_date, period_begin_date_bis, period_end_date_bis

    def get_unadjusted_periods(self, day_val_o, day_mat_o, broken_period, current_month, mois_depart_int,
                               value_date_month, dar_mois, n, t):
        period_begin_date = self.date_deb[:, :t]
        period_end_date = self.date_fin[:, :t]
        period_begin_date_end_month = self.end_month_date_deb[:, :t]
        period_end_date_end_month = self.end_month_date_fin[:, :t]

        day_mat = day_mat_o - 1
        day_mat = np.array(day_mat.astype("timedelta64[D]")).reshape(n, 1)

        day_val = day_val_o - 1
        day_val = np.array(day_val.astype("timedelta64[D]")).reshape(n, 1)

        is_ss_or_sl = np.isin(broken_period, ["SS", "SL"]).reshape(n, 1)
        cond_forward = value_date_month >= dar_mois
        # day_adj = np.where(np.isin(broken_period, ["SS", "SL"]), day_mat, day_val)
        day_adj = np.where(((current_month < mois_depart_int) & (cond_forward)) | ~(is_ss_or_sl), day_val, day_mat)
        period_begin_date = period_begin_date.reshape(1, t) + np.array(day_adj).reshape(n, t)
        period_begin_date = np.minimum(period_begin_date, period_begin_date_end_month.reshape(1, t))

        day_adj = np.where(((current_month < mois_depart_int) & cond_forward) | ~(is_ss_or_sl), day_val, day_mat)
        period_end_date = period_end_date.reshape(1, t) + np.array(day_adj).reshape(n, t)
        period_end_date = np.minimum(period_end_date, period_end_date_end_month.reshape(1, t))

        return period_begin_date, period_end_date

    def apply_business_day_rule_to_periods(self, data_hab, periods, no_change_date, n):
        is_mod_bday = np.array(
            data_hab[self.cls_fields.NC_LDP_CALC_DAY_CONVENTION].astype(float).astype(int).isin([4, 2])).reshape(n)
        next_bs_periods = np.busday_offset(periods[is_mod_bday], 0, roll='forward')
        former_bs_periods = np.busday_offset(periods[is_mod_bday], 0, roll='backward')
        cond = (periods[is_mod_bday].astype('datetime64[M]').astype(int) % 12 + 1) != \
               (next_bs_periods.astype('datetime64[M]').astype(int) % 12 + 1)
        _n = periods[is_mod_bday].shape[0]
        if _n > 0:
            cond_dates = np.array([False] * _n).reshape(_n, 1)
            for i in range(0, len(no_change_date)):
                cond_dates = cond_dates | (periods[is_mod_bday] == no_change_date[i][is_mod_bday])
            periods[is_mod_bday] = np.where(cond_dates, periods[is_mod_bday],
                                            np.where(cond, former_bs_periods, next_bs_periods))

        is_mod_bday = np.array(
            data_hab[self.cls_fields.NC_LDP_CALC_DAY_CONVENTION].astype(float).astype(int) == 3).reshape(n)
        _n = periods[is_mod_bday].shape[0]
        if _n > 0:
            cond_dates = np.array([False] * _n).reshape(_n, 1)
            for i in range(0, len(no_change_date)):
                cond_dates = cond_dates | (periods[is_mod_bday] == no_change_date[i][is_mod_bday])
            next_bs_periods = np.busday_offset(periods[is_mod_bday], 0, roll='forward')
            former_bs_periods = np.busday_offset(periods[is_mod_bday], 0, roll='backward')
            cond = (periods[is_mod_bday].astype('datetime64[M]').astype(int) % 12 + 1) != \
                   (next_bs_periods.astype('datetime64[M]').astype(int) % 12 + 1)
            periods[is_mod_bday] = np.where(cond_dates, periods[is_mod_bday],
                                            np.where(cond, next_bs_periods, former_bs_periods))
        # periods[is_mod_bday] = np.busday_offset(periods[is_mod_bday], 0, roll='backward')

        is_mod_bday = np.array(
            data_hab[self.cls_fields.NC_LDP_CALC_DAY_CONVENTION].astype(float).astype(int).isin([5, 6, 7])).reshape(n)

        _n = periods[is_mod_bday].shape[0]
        if _n > 0:
            cond_dates = np.array([False] * _n).reshape(_n, 1)
            for i in range(0, len(no_change_date)):
                cond_dates = cond_dates | (periods[is_mod_bday] == no_change_date[i][is_mod_bday])
            periods_end_month = ut.add_months_date2(periods.astype("datetime64[M]").astype("datetime64[D]"), 1)
            periods_end_month = ut.end_of_month_numpy(periods_end_month, 1)
            periods[is_mod_bday] = np.where(cond_dates, periods[is_mod_bday], periods_end_month[is_mod_bday])

        return periods

    def prepare_data_cal_indicators(self, data_ldp, name_product, cls_stat_runoff, n, t):
        mat_date = data_ldp[self.cls_fields.NC_LDP_MATUR_DATE + "_REAL"]
        val_date = data_ldp[self.cls_fields.NC_LDP_VALUE_DATE + "_REAL"]
        self.val_date_num = np.array(data_ldp[self.cls_fields.NC_LDP_VALUE_DATE]).reshape(n, 1)

        self.day_mat_date = np.array(pd.to_datetime(mat_date).dt.day).astype(int)
        self.day_mat_date = self.day_mat_date.reshape(n, 1)
        self.day_val_date = np.array(pd.to_datetime(val_date).dt.day).astype(int)
        self.day_val_date = self.day_val_date.reshape(n, 1)

        if name_product not in (models_nmd_st + models_nmd_pn):
            day_tombee = self.period_end_date[:, :t] - self.period_end_date[:, :t].astype('datetime64[M]') + 1
            self.day_tombee = np.nan_to_num(day_tombee.astype(int))
            self.day_tombee_gptx = self.day_tombee.copy()
        else:
            self.day_tombee = np.nan_to_num(cls_stat_runoff.cls_init_ec.cls_flow_params.day_tombee.astype(int))
            self.day_tombee_gptx = np.nan_to_num(
                cls_stat_runoff.cls_init_ec.cls_flow_params.day_tombee_gptx.astype(int))
