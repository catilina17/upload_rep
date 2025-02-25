import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from calculateur.models.utils import utils as ut
import datetime


class Fixing_Calendar_Manager():

    def __init__(self, cls_hz_params, cls_fields, name_product, cls_cal):
        self.cls_hz_params = cls_hz_params
        self.cls_fields = cls_fields
        self.cls_cal = cls_cal
        self.name_product = name_product
        self.dar_mois = self.cls_hz_params.dar_mois
        self.dar_usr = self.cls_hz_params.dar_usr

    def get_fixing_parameters(self, cls_proj):

        data_ldp = cls_proj.data_ldp
        period_begin_date = self.cls_cal.period_begin_date
        current_month = cls_proj.current_month_max
        mois_depart = self.cls_cal.mois_depart
        mois_fin = self.cls_cal.mois_fin
        n = cls_proj.n
        t_max = cls_proj.t_max
        trade_date = cls_proj.data_ldp[self.cls_fields.NC_LDP_TRADE_DATE_REAL]
        mat_date = cls_proj.data_ldp[self.cls_fields.NC_LDP_MATUR_DATE_REAL]
        col_fix_next_date = self.cls_fields.NC_LDP_FIXING_NEXT_DATE_REAL
        col_fixing_nb_days = self.cls_fields.NC_LDP_FIXING_NB_DAYS

        cls_proj.fixing_calendar \
            = self.get_periodic_calendar_per_type_of_product(data_ldp, col_fix_next_date, period_begin_date,
                                                             current_month, mois_depart, mois_fin, n, t_max)

        if self.name_product not in ["cap_floor"]:
            cls_proj.data_ldp = self.update_fixing_date(data_ldp, cls_proj.fixing_calendar, col_fix_next_date,
                                                        col_fixing_nb_days,  trade_date, mat_date, n, t_max)
        else:
            cls_proj.data_ldp = self.update_fixing_date_cap_floor(data_ldp, period_begin_date, col_fix_next_date, n,
                                                                  t_max)

    def get_periodic_calendar_per_type_of_product(self, data_ldp, col_fix_next_date, period_begin_date, current_month,
                                                  mois_depart_int, mois_fin_int, n, t):
        fixing_date = data_ldp[col_fix_next_date].values.reshape(n, 1)
        if self.name_product not in ["nmd_st", "nmd_pn"]:
            is_cap_floor = np.array(data_ldp[self.cls_fields.NC_PRODUCT_TYPE] == "CAP_FLOOR").reshape(n, 1)
            if not data_ldp[self.cls_fields.NC_LDP_IS_FUTURE_PRODUCT].any():
                periodicity = data_ldp[self.cls_fields.NC_LDP_FREQ_INT].fillna("").values.reshape(n, 1)
                periodicity_cap = data_ldp[self.cls_fields.NC_LDP_CAPITALIZATION_PERIOD].fillna("").values.reshape(n, 1)
                periodicity_num = np.where(np.isin(periodicity, ["N", ""]).reshape(n, 1),
                                           data_ldp[self.cls_fields.NC_FREQ_CAP].values.reshape(n, 1),
                                           data_ldp[self.cls_fields.NC_FREQ_INT].values.reshape(n, 1))
                periodicity_num_cap_floor = data_ldp[self.cls_fields.NC_FREQ_INT].values.reshape(n, 1)
                periodicity_num = np.where(is_cap_floor, periodicity_num_cap_floor, periodicity_num)
            else:
                periodicity = data_ldp[self.cls_fields.NC_LDP_FIXING_PERIODICITY].fillna("").values.reshape(n, 1)
                periodicity_num = data_ldp[self.cls_fields.NC_FIXING_PERIODICITY_NUM].fillna("").values.reshape(n, 1)
                periodicity_cap = data_ldp[self.cls_fields.NC_LDP_CAPITALIZATION_PERIOD].fillna("").values.reshape(n, 1)

            decalage_deb_per = data_ldp[self.cls_fields.NC_DECALAGE_DEB_PER_FIX].values.reshape(n, 1)
            decalage_deb_per_cap_floor = data_ldp[self.cls_fields.NC_DECALAGE_DEB_PER].values.reshape(n, 1)
            decalage_deb_per = np.where(is_cap_floor, decalage_deb_per_cap_floor, decalage_deb_per)
            fixing_calendar = self.get_fixing_calendar_ech(decalage_deb_per, periodicity, periodicity_cap,
                                                           periodicity_num, period_begin_date,
                                                           current_month, mois_depart_int,
                                                           mois_fin_int, fixing_date, is_cap_floor, n, t)
        else:
            fixing_date_month = data_ldp[self.cls_fields.NC_LDP_FIXING_NEXT_DATE].values.reshape(n, 1) - self.dar_mois
            trade_date = data_ldp[self.cls_fields.NC_LDP_TRADE_DATE_REAL].values.reshape(n, 1)
            fixing_date = data_ldp[col_fix_next_date]
            periodicity_num = data_ldp[self.cls_fields.NC_FIXING_PERIODICITY_NUM].fillna("").values.reshape(n, 1)
            fixing_calendar \
                = self.get_fixing_calendar_nmd(periodicity_num, fixing_date_month, fixing_date, trade_date, n, t)

        return fixing_calendar

    def get_fixing_calendar_nmd(self, periodicity_num, fixing_date_month, fixing_date, trade_date, n, t):
        current_month = np.repeat(np.arange(1, t + 1).reshape(1, t), n, axis=0)
        day_fix = np.array(fixing_date.dt.day.astype("timedelta64[D]")).reshape(n, 1)
        periodic_calendar = self.cls_cal.date_fin[:, :t] + np.array(day_fix).reshape(n, 1) - 1
        periodic_calendar = np.minimum(periodic_calendar, self.cls_cal.end_month_date_fin[:, :t])
        periodic_calendar = np.where(current_month < fixing_date_month.reshape(n, 1), np.datetime64("NaT"),
                                     periodic_calendar)

        for freq in np.unique(periodicity_num).astype(int):
            if freq > 1:
                is_freq = (periodicity_num == freq).reshape(n)
                periodic_calendar_orig = periodic_calendar[is_freq].copy()
                _n = periodic_calendar_orig.shape[0]
                periodic_calendar_t = periodic_calendar[is_freq].copy()
                indices = np.repeat(np.arange(0, t).reshape(1, t), _n, axis=0)
                indices_rolled = ut.strided_indexing_roll(indices,
                                                          np.maximum(
                                                              -fixing_date_month[is_freq].astype(int).reshape(_n) + 1,
                                                              -t + 1))
                real_indices_rolled = indices_rolled.copy()
                indices_rolled = np.full(indices_rolled.shape, np.nan)
                indices_rolled[:, ::freq] = real_indices_rolled[:, ::freq]
                indices_rolled = ut.np_ffill(indices_rolled).astype(int)
                indices_rolled = ut.strided_indexing_roll(indices_rolled,
                                                          np.minimum(t - 1,
                                                                     fixing_date_month[is_freq].astype(int).reshape(
                                                                         _n) - 1))
                periodic_calendar_t = periodic_calendar_t[np.arange(0, _n).reshape(_n, 1), indices_rolled.astype(int)]
                curr_month = np.arange(0, t).reshape(1, t)
                periodic_calendar_t = np.where(curr_month < fixing_date_month[is_freq].reshape(_n, 1),
                                               periodic_calendar[is_freq], periodic_calendar_t)
                periodic_calendar[is_freq] = periodic_calendar_t

        #periodic_calendar = np.where(periodic_calendar < trade_date, np.datetime64("NaT"), periodic_calendar)
        return periodic_calendar

    def get_fixing_calendar_ech(self, decalage_interets, periodicity, periodicity_cap, periodicity_num,
                                period_begin_date, current_month, mois_depart, mois_fin_int, fixing_date,
                                is_cap_floor, n, t):

        mois_depart_real = mois_depart + decalage_interets
        # CAP_FLOOR
        # ERREUR_RCO
        mois_depart_real_cap_floor = np.where((periodicity == "N"), mois_fin_int + 1, mois_depart_real)
        mois_depart_real = np.where((np.isin(periodicity_cap, ["N", ""])) & (np.isin(periodicity, ["N", ""]))
                                    & (fixing_date == self.cls_cal.absent_date_np.reshape(1, 1)), mois_fin_int + 1,
                                    mois_depart_real)
        mois_depart_real = np.where(is_cap_floor, mois_depart_real_cap_floor, mois_depart_real)

        periodic_calendar = np.where(current_month < mois_depart_real, np.datetime64("NaT"), period_begin_date)
        for freq in np.unique(periodicity_num).astype(int):
            if freq > 1:
                is_freq = (periodicity_num == freq).reshape(n)
                periodic_calendar_orig = periodic_calendar[is_freq].copy()
                _n = periodic_calendar_orig.shape[0]
                periodic_calendar_t = periodic_calendar[is_freq].copy()
                indices = np.repeat(np.arange(0, t).reshape(1, t), _n, axis=0)
                indices_rolled = ut.strided_indexing_roll(indices,
                                                          np.maximum(
                                                              -mois_depart_real[is_freq].astype(int).reshape(_n) + 1,
                                                              -t + 1))
                real_indices_rolled = indices_rolled.copy()
                indices_rolled = np.full(indices_rolled.shape, np.nan)
                indices_rolled[:, ::freq] = real_indices_rolled[:, ::freq]
                indices_rolled = ut.np_ffill(indices_rolled).astype(int)
                indices_rolled = ut.strided_indexing_roll(indices_rolled,
                                                          np.minimum(t - 1,
                                                                     mois_depart_real[is_freq].astype(int).reshape(
                                                                         _n) - 1))
                periodic_calendar_t = periodic_calendar_t[np.arange(0, _n).reshape(_n, 1), indices_rolled.astype(int)]
                curr_month = np.arange(0, t).reshape(1, t)
                periodic_calendar_t = np.where(curr_month < mois_depart_real[is_freq].reshape(_n, 1),
                                               periodic_calendar[is_freq], periodic_calendar_t)
                periodic_calendar[is_freq] = periodic_calendar_t

        return periodic_calendar

    def update_fixing_date(self, data_ldp, fixing_calendar, col_fix_next_date, col_fixing_nb_days,
                           trade_date, mat_date, n, t_max):
        fixing_date = data_ldp[col_fix_next_date]
        no_fixing_date = (np.array(fixing_date).astype("datetime64[D]") == self.cls_cal.absent_date_np).reshape(n, 1)
        fixing_date = np.array(fixing_date).astype("datetime64[D]").reshape(n, 1)
        fixing_date = ut.add_days_date(fixing_date, data_ldp[col_fixing_nb_days].values.reshape(n, 1).astype(int))
        index_apply_fixing_date = ut.first_sup_val(fixing_calendar, axis=1, val=fixing_date,
                                                   invalid_val=t_max - 1).reshape(n, 1)
        index_apply_fixing_date = np.minimum(index_apply_fixing_date, t_max - 1)
        fixing_date = np.where(no_fixing_date,
                               fixing_calendar[np.arange(0, n).reshape(n, 1), index_apply_fixing_date],
                               fixing_date)

        if self.name_product not in ["nmd_st", "nmd_pn"]:
            index_next_fixing_date = ut.first_sup_strict_zero(fixing_calendar, axis=1, val=fixing_date,
                                                               invalid_val=t_max - 1).reshape(n, 1)
            next_fixing_date = fixing_calendar[np.arange(0, n).reshape(n, 1), index_next_fixing_date]
            fixing_date_dt = pd.Series(fixing_date.reshape(n)).replace(pd.NaT, datetime.datetime(1900, 1, 1))
            fixing_date_dt = np.array([x.date() for x in fixing_date_dt]).astype("datetime64[D]").reshape(n, 1)
            trade_date_np = pd.to_datetime(trade_date)
            trade_date_np = np.array([x.date() for x in trade_date_np]).astype("datetime64[D]").reshape(n, 1)
            cond_sup = (fixing_date_dt >= np.maximum(trade_date_np, (self.dar_usr + relativedelta(days=1)).date()))
            fixing_date = np.where(cond_sup, fixing_date, next_fixing_date.reshape(n, 1))

        mat_date_np = pd.to_datetime(mat_date)
        mat_date_np = np.array([x.date() for x in mat_date_np]).astype("datetime64[D]").reshape(n, 1)
        fixing_date = np.where(np.isnat(fixing_date), mat_date_np.reshape(n, 1), fixing_date)

        data_ldp[col_fix_next_date] = fixing_date

        return data_ldp

    def update_fixing_date_cap_floor(self, data_ldp, period_begin_date_cal, col_fix_next_date, n, t):
        fixing_date = data_ldp[col_fix_next_date].values.reshape(n, 1)
        no_fixing_date = (fixing_date == self.cls_cal.absent_date_np.reshape(1, 1)).reshape(n, 1)
        index_apply_fixing_date = ut.first_sup_val(period_begin_date_cal, axis=1, val=fixing_date,
                                                   invalid_val=t - 1).reshape(n, 1)
        index_apply_fixing_date = np.minimum(index_apply_fixing_date, t - 1)
        fixing_date = np.where(no_fixing_date,
                               period_begin_date_cal[np.arange(0, n).reshape(n, 1), index_apply_fixing_date],
                               fixing_date)
        data_ldp[col_fix_next_date] = fixing_date
        return data_ldp
