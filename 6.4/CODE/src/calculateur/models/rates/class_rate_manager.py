import numpy as np
from calculateur.simul_params import model_params as mod
import pandas as pd
import numexpr as ne
import datetime
import logging
from calculateur.models.utils import utils as ut

nan = np.nan
logger = logging.getLogger(__name__)


class Rate_Manager():
    """
    Formate les données
    """

    def __init__(self, cls_data_rate_manager, cls_fields, cls_hz_params, cls_cal, cls_fix_cal, name_product):
        self.cls_fields = cls_fields
        self.cls_cal = cls_cal
        self.cls_fix_cal = cls_fix_cal
        self.cls_hz_params = cls_hz_params
        self.name_product = name_product
        self.cls_data_rate = cls_data_rate_manager
        self.cls_spread_index = cls_data_rate_manager.cls_spread_index
        self.cls_target_rates = cls_data_rate_manager.cls_target_rates
        self.with_dyn_data = cls_data_rate_manager.with_dyn_data
        self.list_contracts_without_conv = ["A-REGUL", "P-REGUL"]

    ####@profile
    def get_rates(self, cls_proj, cls_model_params):
        data_ldp = cls_proj.data_ldp
        sc_rates = np.zeros((cls_proj.n, cls_proj.t_max))
        sc_rates_lag = np.zeros((cls_proj.n, cls_proj.t_max))
        sc_rates_ftp = np.zeros((cls_proj.n, cls_proj.t_max))
        sc_rates_ftp_lag = np.zeros((cls_proj.n, cls_proj.t_max))
        tombee_fixing = np.full((cls_proj.n, cls_proj.t_max), np.nan)
        period_fixing = np.full((cls_proj.n, cls_proj.t_max), True)
        current_month = cls_proj.current_month_max

        data_spread = self.get_data_spread(data_ldp, cls_proj.n, cls_proj.t_max, self.cls_fields.NC_LDP_RATE_CODE)
        data_target_rates = self.get_data_target_rates(data_ldp, cls_proj.n, cls_proj.t_max)

        is_tv = (data_ldp[self.cls_fields.NC_LDP_RATE_TYPE] == "FLOATING").values
        _n = is_tv[is_tv].shape[0]
        if (cls_proj.name_product != "cap_floor"):
            if _n > 0:
                (sc_rates[is_tv], sc_rates_lag[is_tv], tombee_fixing[is_tv], data_ldp[is_tv], sc_rates_ftp[is_tv],
                 sc_rates_ftp_lag[is_tv], period_fixing[is_tv]) \
                    = self.get_scenario_rates(cls_proj.data_ldp[is_tv].copy(), self.cls_data_rate.data_rate_sc[is_tv],
                                              cls_proj.fixing_calendar[is_tv],
                                              current_month[is_tv],
                                              self.cls_fields.NC_LDP_FIXING_NEXT_DATE_REAL,
                                              self.cls_fields.NC_LDP_RATE, self.cls_fields.NC_FIXING_PERIODICITY_NUM,
                                              self.cls_fields.NC_LDP_MULT_SPREAD, self.cls_fields.NC_LDP_MKT_SPREAD,
                                              self.cls_fields.NC_LDP_FLOOR_STRIKE, self.cls_fields.NC_LDP_CAP_STRIKE,
                                              self.cls_fields.NC_TENOR_NUM, self.cls_fields.NC_LDP_TENOR,
                                              self.cls_fields.NC_LDP_ACCRUAL_BASIS,
                                              self.cls_fields.NC_LDP_FIXING_PERIODICITY,
                                              data_ldp.loc[is_tv, self.cls_fields.NC_LDP_TRADE_DATE],
                                              data_ldp.loc[is_tv, self.cls_fields.NC_LDP_MATUR_DATE],
                                              self.cls_fields.NC_DATE_DEBUT,
                                              data_ldp.loc[is_tv, self.cls_fields.NC_LDP_CONTRACT_TYPE],
                                              self.cls_fields.NC_LDP_FIXING_RULE, self.cls_fields.NC_LDP_TARGET_RATE,
                                              self.cls_fields.NC_LDP_FTP_FUNDING_SPREAD, _n, cls_proj.t, cls_proj.t_max,
                                              data_spread=data_spread[is_tv],
                                              data_target_rates=data_target_rates[is_tv])

            is_tf = (data_ldp[self.cls_fields.NC_LDP_RATE_TYPE] == "FIXED").values
            _n = is_tf[is_tf].shape[0]
            if _n > 0:
                sc_rates, sc_rates_lag, sc_rates_ftp, sc_rates_ftp_lag \
                    = self.get_fixed_rate(cls_proj, cls_model_params, data_spread, data_target_rates, data_ldp,
                                          sc_rates, sc_rates_lag, sc_rates_ftp, sc_rates_ftp_lag, is_tf, _n,
                                          cls_proj.t_max)
        else:
            sc_rates, sc_rates_lag, data_ldp, period_fixing \
                = self.get_scenario_rates_cap_floor(cls_proj.data_ldp, self.cls_data_rate.data_rate_sc,
                                                    cls_proj.fixing_calendar,
                                                    current_month,
                                                    self.cls_fields.NC_LDP_FIXING_NEXT_DATE_REAL,
                                                    self.cls_fields.NC_LDP_CURRENT_RATE,
                                                    self.cls_fields.NC_LDP_MULT_SPREAD,
                                                    self.cls_fields.NC_LDP_MKT_SPREAD,
                                                    self.cls_fields.NC_LDP_FLOOR_STRIKE,
                                                    self.cls_fields.NC_LDP_CAP_STRIKE, self.cls_fields.NC_FREQ_INT,
                                                    self.cls_fields.NC_TENOR_NUM,
                                                    self.cls_fields.NC_LDP_ACCRUAL_BASIS,
                                                    self.cls_fields.NC_DATE_DEBUT,
                                                    self.cls_fields.NC_LDP_FIXING_RULE,
                                                    data_ldp[self.cls_fields.NC_LDP_TRADE_DATE],
                                                    cls_proj.n, cls_proj.t, cls_proj.t_max)

            sc_rates_ftp = np.zeros((cls_proj.n, cls_proj.t_max))
            sc_rates_ftp_lag = np.zeros((cls_proj.n, cls_proj.t_max))

        self.tombee_fixing = tombee_fixing
        self.period_fixing = period_fixing
        self.sc_rates = sc_rates
        self.sc_rates_ftp = sc_rates_ftp
        self.sc_rates_lag = sc_rates_lag
        self.sc_rates_ftp_lag = sc_rates_ftp_lag
        cls_proj.data_ldp = data_ldp

    ####@profile
    def get_fixed_rate(self, cls_proj, cls_model_params, data_spread, data_target_rates, data_ldp, sc_rates,
                       sc_rates_lag, sc_rates_ftp, sc_rates_ftp_lag, is_tf, _n, t):

        rate = np.array(cls_proj.data_ldp.loc[is_tf, self.cls_fields.NC_LDP_RATE]).reshape(_n, 1)
        sc_rates[is_tf] = (rate * 12 * np.ones((_n, t)))
        sc_rates[is_tf] = sc_rates[is_tf] + data_spread[is_tf]
        sc_rates_lag[is_tf] = sc_rates[is_tf].copy()

        ftp_rate = np.array(data_ldp.loc[is_tf, cls_proj.cls_fields.NC_LDP_FTP_RATE]).reshape(_n, 1)
        marge = np.array(data_ldp.loc[is_tf, cls_proj.cls_fields.NC_LDP_MKT_SPREAD]).reshape(_n, 1)
        sc_rates_ftp[is_tf] = (ftp_rate * 12 * np.ones((_n, t)))
        sc_rates_ftp[is_tf] = np.where(np.isnan(ftp_rate),
                                       sc_rates[is_tf] - data_spread[is_tf] - marge, sc_rates_ftp[is_tf])
        sc_rates_ftp_lag[is_tf] = sc_rates_ftp[is_tf].copy()

        target_rate = np.array(cls_proj.data_ldp.loc[is_tf, self.cls_fields.NC_LDP_TARGET_RATE]).reshape(_n, 1)
        does_have_target_rate = (~np.isnan(target_rate))
        sc_rates[is_tf] = np.where(does_have_target_rate, target_rate * 12, sc_rates[is_tf])
        sc_rates_lag[is_tf] = np.where(does_have_target_rate, target_rate * 12, sc_rates_lag[is_tf])

        sc_rates[is_tf] = np.where(np.isnan(data_target_rates[is_tf]), sc_rates[is_tf], data_target_rates[is_tf])
        sc_rates_lag[is_tf] = np.where(np.isnan(data_target_rates[is_tf]), sc_rates_lag[is_tf],
                                       data_target_rates[is_tf])

        if self.name_product not in (mod.models_nmd_pn + mod.models_nmd_st):
            floor = (cls_proj.data_ldp[[self.cls_fields.NC_LDP_CONTRACT_TYPE]]
                     .join(cls_model_params.cls_rates_floor.rates_floors,
                           on=[self.cls_fields.NC_LDP_CONTRACT_TYPE])).iloc[:, 1].values
            floor = floor[is_tf].reshape(_n, 1)
            sc_rates[is_tf] = np.where(~np.isnan(floor) & (sc_rates[is_tf] < floor), floor, sc_rates[is_tf])

        return sc_rates, sc_rates_lag, sc_rates_ftp, sc_rates_ftp_lag

    ######@profile
    def get_data_spread(self, data_ldp, n, t, col_rate_code):
        if self.with_dyn_data:
            data_spread = self.cls_spread_index.get_spread_data(data_ldp, col_rate_code, n, t)
        else:
            data_spread = np.zeros((n, t))
        return data_spread

    ######@profile
    def get_data_target_rates(self, data_ldp, n, t):
        if self.with_dyn_data:
            data_target_rates = self.cls_target_rates.get_target_rates_data(data_ldp, self.cls_fields.NC_LDP_RATE_CODE,
                                                                            n, t)
        else:
            data_target_rates = np.full((n, t), np.nan)
        return data_target_rates

    ####@profile
    def get_scenario_rates(self, data_ldp, data_rate_sc, fixing_calendar, current_month,
                           col_fix_next_date, col_curr_rate, col_fix_per_num, col_mult_fact, col_mkt_spread,
                           col_floor, col_cap, col_tenor_num, col_tenor, col_accrual_basis,
                           col_fix_per, trade_date_month, mat_date_month, col_deb, contract_type,
                           col_fixing_rule, col_target_rate, col_ftp_liq, n, t_proj, t, data_spread=[], data_target_rates=[]):

        if len(data_ldp) > 0:
            if len(data_spread) == 0:
                data_spread = np.zeros((n, t))
            if len(data_target_rates) == 0:
                data_target_rates = np.full((n, t), np.nan)

            dar_mois = self.cls_hz_params.dar_mois
            dar_usr = self.cls_hz_params.dar_usr
            is_pre_fix = (data_ldp[col_fixing_rule] != "A").values
            fixing_date = data_ldp[col_fix_next_date].values.reshape(n, 1)
            fixing_date_index, fixing_date_day = self.get_fixing_date_index(fixing_date, n, t)

            sc_rates_conv_adj \
                = self.get_accrual_convention_adjusted_sc_rates(data_ldp, data_rate_sc.copy(), col_tenor_num,
                                                                col_accrual_basis, contract_type,
                                                                self.cls_cal.date_fin, n, t)

            sc_rates_periodic = self.make_rate_periodic(data_ldp, sc_rates_conv_adj.copy(),
                                                        data_ldp[col_fix_next_date], dar_usr, col_fix_per_num)

            period_fixing = self.get_fixing_period(data_ldp, current_month, fixing_date_index,
                                                   trade_date_month, is_pre_fix, col_fix_per_num, dar_mois,
                                                   fixing_calendar, fixing_date, n, t)

            sc_rates_mkt_spread = self.add_market_spread(data_ldp, sc_rates_periodic, data_spread,
                                                      col_mult_fact, col_mkt_spread, n)

            sc_rates_cap_floor = self.add_cap_and_floor(sc_rates_mkt_spread, data_ldp, col_cap, col_floor, n)

            sc_rates = self.shift_post_fix(data_ldp, sc_rates_cap_floor, is_pre_fix, col_fix_per_num, col_fix_per)

            sc_rates, current_rate = self.add_current_rate(data_ldp, sc_rates, period_fixing, col_curr_rate, col_deb,
                                                           dar_mois, n)

            sc_rates_lag = self.get_rates_mtx_lags(sc_rates, is_pre_fix, current_rate, n)

            sc_rates_ftp, sc_rates_ftp_lag = self.get_ftp_rates(sc_rates, sc_rates_lag, data_spread,
                                                                data_ldp, col_mkt_spread, col_mult_fact, col_ftp_liq, n)

            sc_rates, sc_rates_lag = self.add_target_rates(sc_rates, sc_rates_lag, data_ldp, col_target_rate,
                                                           data_target_rates, n)

            tombee_fixing = self.get_fixing_tombee(data_ldp, fixing_date_day, fixing_date_index,
                                                   data_ldp[col_fix_next_date], current_month, col_fix_per_num,
                                                   col_fix_per, is_pre_fix, dar_usr, n, t)

        else:
            sc_rates = np.zeros((0, t))
            sc_rates_lag = np.zeros((0, t))
            sc_rates_ftp_lag = np.zeros((0, t))
            tombee_fixing = np.zeros((0, t))
            sc_rates_ftp = np.zeros((0, t))
            period_fixing = np.zeros((0, t))

        return sc_rates, sc_rates_lag, tombee_fixing, data_ldp.values, sc_rates_ftp, sc_rates_ftp_lag, period_fixing


    def get_fixing_date_index(self, fixing_date, n, t):
        fixing_date_year = ut.dt2cal(fixing_date.reshape(n))[:, 0].astype(int)
        fixing_date_month = ut.dt2cal(fixing_date.reshape(n))[:, 1].astype(int)
        fixing_date_index = ((fixing_date_month - self.cls_hz_params.dar_usr.month)
                             + (fixing_date_year - self.cls_hz_params.dar_usr.year) * 12)
        fixing_date_index = np.minimum(np.maximum(0, fixing_date_index), t)

        fixing_date_day = ut.dt2cal(fixing_date.reshape(n))[:, 2].astype(int)

        return fixing_date_index, fixing_date_day

    def get_rates_mtx_lags(self, sc_rates, is_pre_fix, current_rate, n):
        lag_rate = np.where(~is_pre_fix.reshape(n, 1), sc_rates[:, 0:1], current_rate)
        sc_rates_lag = ut.roll_and_null(sc_rates, shift=1)
        sc_rates_lag[:, 0] = lag_rate.reshape(n)
        return sc_rates_lag

    def get_ftp_rates(self, sc_rates, sc_rates_lag, data_spread, data_ldp, col_mkt_spread, col_mult_fact, col_ftp_liq, n):
        mult_fact = np.where(data_ldp[col_mult_fact].values.reshape(n, 1) == 0, 1,
                             data_ldp[col_mult_fact].values.reshape(n, 1))
        sc_rates_ftp = (sc_rates - data_ldp[col_mkt_spread].values.reshape(n, 1) - data_spread) / mult_fact
        sc_rates_ftp_lag = (sc_rates_lag - data_ldp[col_mkt_spread].values.reshape(n, 1) - data_spread) / mult_fact

        if self.name_product not in (mod.models_nmd_pn + mod.models_nmd_st):
            sc_rates_ftp = sc_rates_ftp + data_ldp[col_ftp_liq].values.reshape(n, 1)
            sc_rates_ftp_lag = sc_rates_ftp_lag + data_ldp[col_ftp_liq].values.reshape(n, 1)

        return sc_rates_ftp, sc_rates_ftp_lag

    def add_target_rates(self, sc_rates, sc_rates_lag, data_ldp, col_target_rate, data_target_rates, n):
        if data_ldp[col_target_rate].notnull().any():
            filter_target_rate = data_ldp[col_target_rate].notnull().values.reshape(n, 1)
            target_rate = data_ldp[col_target_rate].values.reshape(n, 1) * 12
            sc_rates = np.where(filter_target_rate, target_rate, sc_rates)
            sc_rates_lag = np.where(filter_target_rate, target_rate, sc_rates_lag)

        sc_rates = np.where(np.isnan(data_target_rates), sc_rates, data_target_rates)
        sc_rates_lag = np.where(np.isnan(data_target_rates), sc_rates_lag, data_target_rates)

        return sc_rates, sc_rates_lag

    def add_current_rate(self, data_ldp, sc_rates, period_fixing, col_curr_rate, col_deb, dar_mois, n):
        current_rate = np.array(data_ldp[col_curr_rate]).reshape(n, 1) * 12
        if self.name_product in mod.models_ech_pn:
            current_rate = self.get_current_rate_future_products(data_ldp, col_deb, sc_rates, dar_mois, n)

        sc_rates = np.where(period_fixing, current_rate, sc_rates)
        return sc_rates, current_rate

    def add_market_spread(self, data_ldp, data_rate, data_spread, col_mult_fact, col_mkt_spread, n):
        data_rate_sc_mod = data_rate * data_ldp[col_mult_fact].values.reshape(n, 1) \
                           + data_ldp[col_mkt_spread].values.reshape(n, 1) + data_spread
        return data_rate_sc_mod

    def add_cap_and_floor(self, data_rate_sc_mod, data_ldp, col_cap, col_floor, n):
        sc_rates = np.minimum(np.maximum(data_rate_sc_mod, data_ldp[col_floor].values.reshape(n, 1)),
                              data_ldp[col_cap].values.reshape(n, 1))
        return sc_rates

    def shift_post_fix(self, data_ldp, rate, is_pre_fix, col_fix_per_num, col_fix_per):
        for freq in data_ldp[col_fix_per_num].unique().astype(int):
            is_freq = (data_ldp[col_fix_per_num].values == freq) & (~ data_ldp[col_fix_per].isin(["1D", "1W"]))
            _n = is_freq[is_freq].shape[0]
            rate[is_freq] = np.where(~is_pre_fix[is_freq].reshape(_n, 1), ut.roll_and_null(rate[is_freq], shift=-freq),
                                     rate[is_freq])
        return rate

    ######@profile
    def get_fixing_period(self, data_ldp, current_month, fixing_date_index,
                          trade_date_month, is_pre_fix, col_fix_per_num, dar_mois,
                          fixing_calendar, fixing_date, n, t):
        decalage = np.where(~is_pre_fix.reshape(n, 1), data_ldp[col_fix_per_num].values.reshape(n, 1), 0)
        period_fixing = ((current_month < fixing_date_index.reshape(n, 1) - decalage)
                         & (current_month >= np.maximum(1, trade_date_month.values.reshape(n, 1) - dar_mois)))

        if self.name_product in mod.models_ech_pn:
            index_apply_fixing_date = ut.first_sup_strict_zero(fixing_calendar, axis=1, val=fixing_date,
                                                               invalid_val=t - 1).reshape(n, 1)
            fixing_date_plus_freq = fixing_calendar[np.arange(0, n).reshape(n, 1), index_apply_fixing_date]

            fixing_date_year = ut.dt2cal(fixing_date_plus_freq.reshape(n))[:, 0].astype(int)
            fixing_date_month = ut.dt2cal(fixing_date_plus_freq.reshape(n))[:, 1].astype(int)
            fixing_date_index_plus = ((fixing_date_month - self.cls_hz_params.dar_usr.month)
                                      + (fixing_date_year - self.cls_hz_params.dar_usr.year) * 12)
            period_fixing = (current_month < fixing_date_index_plus.reshape(n, 1))

        return period_fixing

    def get_current_rate_future_products(self, data_ldp, col_deb, sc_rates, dar_mois, n):
        first_date_index = (data_ldp[col_deb] - dar_mois).values - 1
        current_rate = sc_rates[np.arange(0, n), first_date_index - 1]
        return current_rate.reshape(n, 1)

    ######@profile
    def get_accrual_convention_adjusted_sc_rates(self, data_ldp, data_rate_sc_cal, col_tenor_num,
                                                 col_accrual_basis, contract_type, calendar_dates, n, t):

        is_base_calc_30 = (
            data_ldp[col_accrual_basis].str.upper().isin(["30/360", "30E/360", "30A/360"])).values.reshape(n, 1)

        is_curve_calc_30 = (data_ldp[self.cls_data_rate.ACCRUAL_METHOD].str.upper().isin(
            ["30/360", "30E/360", "30A/360"])).values.reshape(n, 1)

        is_not_regulated_contract = (~contract_type.isin(self.list_contracts_without_conv)).values.reshape(n, 1)

        accrual_conversion = (data_ldp[self.cls_data_rate.ACCRUAL_CONVERSION].astype(str) != "F").values.reshape(n, 1)

        cond_existence = (((is_base_calc_30 & ~is_curve_calc_30) | (~is_base_calc_30 & is_curve_calc_30))
                          & accrual_conversion & is_not_regulated_contract)
        cond_existence2 = ((data_ldp[self.cls_data_rate.STANDALONE_INDEX].astype(
            str) == self.cls_data_rate.STANDALONE_INDEX_CONST).values.reshape(n, 1)
                           & accrual_conversion & is_not_regulated_contract)

        if (cond_existence | cond_existence2).any():
            nb_days_per = self.calculate_nb_days_per_tenor_period(data_ldp, col_tenor_num, calendar_dates, n, t)

            coeff_passage_ACT_30 = ut.np_ffill(nb_days_per / (30 * np.array(data_ldp[col_tenor_num])).reshape(n, 1))

            cases = [is_base_calc_30 & ~is_curve_calc_30 & is_not_regulated_contract & accrual_conversion,
                     # & tenor_sup_1M,
                     ~is_base_calc_30 & is_curve_calc_30 & is_not_regulated_contract & accrual_conversion]  # & tenor_sup_1M ]
            values_coeff = [coeff_passage_ACT_30, 1 / coeff_passage_ACT_30]
            coeff_passage = np.select(cases, values_coeff, default=1)
        else:
            coeff_passage = 1

        is_base_calc_360 = (
            data_ldp[col_accrual_basis].str.upper().isin(["30/360", "30E/360", "30A/360", "ACT/360"])).values.reshape(n,
                                                                                                                      1)
        is_curve_calc_360 = (
            data_ldp[self.cls_data_rate.ACCRUAL_METHOD].str.upper().isin(
                ["30/360", "30E/360", "30A/360", "ACT/360"])).values.reshape(n, 1)

        coeff_passage_360_365 = 365 / 360
        cases = [is_base_calc_360 & ~is_curve_calc_360 & is_not_regulated_contract & accrual_conversion,
                 ~is_base_calc_360 & is_curve_calc_360 & is_not_regulated_contract & accrual_conversion]
        values_coeff = [1 / coeff_passage_360_365, coeff_passage_360_365]
        coeff_passage = np.select(cases, [x * coeff_passage for x in values_coeff], default=coeff_passage)

        data_rate_sc_conv = coeff_passage * data_rate_sc_cal

        if cond_existence2.any():
            nb_days_per_contract = np.where(is_base_calc_30, 30 * np.array(data_ldp[col_tenor_num]).reshape(n, 1),
                                            nb_days_per)
            nb_days_per_curve = np.where(is_curve_calc_30, 30 * np.array(data_ldp[col_tenor_num]).reshape(n, 1),
                                         nb_days_per)
            nb_days_an_contract = np.where(is_base_calc_360, 360, 365)
            nb_days_an_curve = np.where(is_curve_calc_360, 360, 365)

            new_rate = ne.evaluate(
                "((1 + data_rate_sc_cal) ** (nb_days_per_curve/nb_days_an_curve) - 1) / (nb_days_per_contract/nb_days_an_contract)")
            data_rate_sc_conv = np.where(cond_existence2, new_rate, data_rate_sc_conv)

        return np.nan_to_num(data_rate_sc_conv)

    ######@profile
    def get_fixing_tombee(self, data_ldp, fixing_date_day, index_fix_date, fixing_date, current_month, col_fix_per_num,
                          col_fix_per, is_ff_fill, dar_usr, n, t):
        decalage = np.where(~is_ff_fill.reshape(n, 1), data_ldp[col_fix_per_num].values.reshape(n, 1), 0)
        tombee_fixing = np.where(current_month >= index_fix_date.reshape(n, 1) - decalage,
                                 fixing_date_day.reshape(n, 1), np.nan)
        tombee_fixing = self.make_rate_periodic(data_ldp, tombee_fixing, fixing_date, dar_usr, col_fix_per_num, na_fill=False)
        tombee_fixing = np.where(current_month >= index_fix_date.reshape(n, 1) - decalage,
                                 tombee_fixing, np.nan)
        cond_per_inf_1M = (data_ldp[col_fix_per].str[-1:].isin(["D", "W"])).values.reshape(n, 1)
        tombee_fixing = np.where((current_month >= index_fix_date.reshape(n, 1) + 1) & (cond_per_inf_1M),
                                 1, tombee_fixing)
        cond_fin_mois = (self.cls_cal.end_month_date_fin_day[:, index_fix_date - 1].reshape(n, 1)
                         == fixing_date_day.reshape(n, 1)) & ~np.isnan(tombee_fixing)
        tombee_fixing = np.where(cond_fin_mois, self.cls_cal.end_month_date_fin_day[:, :t], tombee_fixing)

        return tombee_fixing

    ###############@profile
    def calculate_nb_days_per_tenor_period(self, data_ldp, col_tenor_per, calendar_dates, n, t):
        freq_tenor = np.array(data_ldp[col_tenor_per]).astype(int)
        t_max = np.max(freq_tenor) + t + 1
        t_max = np.maximum(t, t_max)
        date_per = np.concatenate([calendar_dates[:, :t_max]] * n, axis=0)

        roll_index = np.minimum(t_max - 1, freq_tenor)
        nb_jours_per_dec = (date_per - ut.strided_indexing_roll(date_per, roll_index, rep_nan=False,
                                                                val_nan=np.datetime64("NaT"))) / np.timedelta64(1, 'D')
        nb_jours_per = ut.strided_indexing_roll(ut.np_ffill(nb_jours_per_dec), -roll_index)

        nb_jours_per = nb_jours_per[:, :t]

        return nb_jours_per

    def cal_nb_per_cap_floor(self, fixing_date, index_apply_fixing_date, t, data_ldp, col_int_per, n):
        t_max = np.max(np.array(data_ldp[col_int_per].astype(int))) + t + 1
        current_month = np.repeat(np.arange(1, t_max + 1).reshape(1, t_max), n, axis=0)
        nb_jours_per = np.concatenate([fixing_date] * t_max, axis=1)
        nb_jours_per = np.where(current_month < index_apply_fixing_date.reshape(n, 1), np.datetime64("NaT"),
                                nb_jours_per)
        months = (np.maximum(0, np.arange(0, t_max).reshape(1, t_max) - index_apply_fixing_date.reshape(n, 1) + 1)).astype(
            "timedelta64[M]").reshape(n, t_max)
        nb_jours_per = (nb_jours_per.astype("datetime64[M]") + months).astype("datetime64[D]")

        roll_index = np.array(data_ldp[col_int_per].astype(int))
        nb_jours_per = (nb_jours_per - ut.strided_indexing_roll(nb_jours_per, roll_index, rep_nan=False,
                                                                val_nan=np.datetime64("NaT"))) / np.timedelta64(1, 'D')
        nb_jours_per = ut.strided_indexing_roll(ut.np_ffill(nb_jours_per), -roll_index)

        return nb_jours_per[:, :t]

    ######@profile
    def make_rate_periodic(self, data_ldp, rate, fixing_date, dar_usr, col_fix_per, na_fill=True):
        """ CREE DE NOUVELLES COLONNES POUR EVITER DE TOUCHER AUX DONNEES ORIGINALES"""
        fixing_date_dt = pd.to_datetime(fixing_date)
        for freq in data_ldp[col_fix_per].unique().astype(int):
            is_freq = (data_ldp[col_fix_per].values == freq)
            _n = is_freq[is_freq].shape[0]
            if freq > 1:
                fixing_date_dt_freq = fixing_date_dt[is_freq]
                is_freq = (data_ldp[col_fix_per].values == freq)
                rate_t = rate[is_freq].copy()
                roll = (freq - (12 * (fixing_date_dt_freq.dt.year - dar_usr.year) + (fixing_date_dt_freq.dt.month - dar_usr.month - 1)) % freq).values
                rate_t = ut.strided_indexing_roll(ut.np_ffill(rate_t), np.maximum(0, roll))
                real_rate = rate_t.copy()
                rate_t = np.full(rate_t.shape, np.nan)
                rate_t[:, ::freq] = real_rate[:, ::freq]
                if na_fill:
                    rate_t = ut.np_ffill(rate_t)
                rate_t = ut.np_ffill(rate_t)
                rate_t = ut.strided_indexing_roll(rate_t, np.minimum(0, -roll))
                rate[is_freq] = rate_t

        rate[rate == np.inf] = np.nan
        rate[rate == -np.inf] = np.nan

        return rate

    def get_accrual_conv_adjusted_sc_rates_cap_floor(self, data_ldp, data_rate_sc_cal, fixing_date,
                                                     index_apply_fixing_date,
                                                     current_month, col_tenor_num, col_accrual_basis, n, t):

        nb_jours_per = self.cal_nb_per_cap_floor(fixing_date, index_apply_fixing_date, t, data_ldp,
                                                 col_tenor_num, n)

        coeff_passage_360_365 = 365 / 360
        coeff_passage_ACT_30 = ut.np_ffill(nb_jours_per / (30 * np.array(data_ldp[col_tenor_num])).reshape(n, 1))

        accrual_conversion = (data_ldp[self.cls_data_rate.ACCRUAL_CONVERSION].astype(str) != "F").values.reshape(n, 1)

        is_base_calc_30 = (
            data_ldp[col_accrual_basis].str.upper().isin(["30/360", "30E/360", "30A/360"])).values.reshape(n, 1)
        is_base_calc_360 = (
            data_ldp[col_accrual_basis].str.upper().isin(["30/360", "30E/360", "30A/360", "ACT/360"])).values.reshape(n,
                                                                                                                      1)
        is_curve_calc_30 = (data_ldp[self.cls_data_rate.ACCRUAL_METHOD].str.upper().isin(
            ["30/360", "30E/360", "30A/360"])).values.reshape(n, 1)
        is_curve_calc_360 = (data_ldp[self.cls_data_rate.ACCRUAL_METHOD].str.upper().isin(
            ["30/360", "30E/360", "30A/360", "ACT/360"])).values.reshape(n, 1)

        cases = [is_base_calc_30 & ~is_curve_calc_30 & accrual_conversion,
                 ~is_base_calc_30 & is_curve_calc_30 & accrual_conversion]
        values_coeff = [coeff_passage_ACT_30, 1 / coeff_passage_ACT_30]
        coeff_passage = np.select(cases, values_coeff, default=1)

        cases = [is_base_calc_360 & ~is_curve_calc_360 & accrual_conversion,
                 ~is_base_calc_360 & is_curve_calc_360 & accrual_conversion]
        values_coeff = [1 / coeff_passage_360_365, coeff_passage_360_365]
        coeff_passage = np.select(cases, [x * coeff_passage for x in values_coeff], default=coeff_passage)

        data_rate_sc_conv = coeff_passage * data_rate_sc_cal

        return data_rate_sc_conv

    def get_scenario_rates_cap_floor(self, data_ldp, data_rate_sc, periodic_calendar, current_month,
                                     col_fix_next_date, col_curr_rate, col_mult_fact, col_mkt_spread,
                                     col_floor, col_cap, col_int_per, col_tenor_num, col_accrual_basis,
                                     col_deb, col_fixing_rule, trade_date_mois, n, t_proj, t_max):

        dar_mois = self.cls_hz_params.dar_mois
        dar_usr = self.cls_hz_params.dar_usr

        current_rate = np.array(data_ldp[col_curr_rate]).reshape(n, 1)
        fixing_date = data_ldp[col_fix_next_date].values.reshape(n, 1)
        fixing_date_index, fixing_date_day = self.get_fixing_date_index(fixing_date, n, t_proj)

        data_rate_sc_cal = data_rate_sc.copy()
        fixing_date_rate = data_rate_sc[np.arange(0, n), fixing_date_index - 1]

        index_apply_fixing_date, date_deb\
            = self.get_cap_floor_index_from_which_to_apply_fxing_date_rate(data_ldp, periodic_calendar,
                                                                           fixing_date, t_proj, n)

        data_rate_sc_cal[np.arange(0, n), index_apply_fixing_date - 1] = fixing_date_rate

        data_rate_sc_conv\
            = self.get_accrual_conv_adjusted_sc_rates_cap_floor(data_ldp, data_rate_sc_cal, date_deb,
                                                                index_apply_fixing_date, current_month, col_tenor_num,
                                                                col_accrual_basis, n, t_max)

        is_pre_fix = (data_ldp[col_fixing_rule] != "A").values

        data_rate_sc_conv_per \
            = self.make_rate_periodic_cap_floor(data_ldp, data_rate_sc_conv, index_apply_fixing_date, col_int_per, t_max)

        data_rate_sc_mod = self.add_market_spread(data_ldp, data_rate_sc_conv_per, 0, col_mult_fact, col_mkt_spread, n)

        sc_rates = self.add_cap_and_floor(data_rate_sc_mod, data_ldp, col_cap, col_floor, n)

        period_fixing = self.get_fixing_period(data_ldp, current_month, index_apply_fixing_date, trade_date_mois,
                                               is_pre_fix, col_int_per, dar_mois, periodic_calendar, fixing_date, n,
                                               t_max)

        if data_ldp[self.cls_fields.NC_LDP_IS_FUTURE_PRODUCT].values.any():
            current_rate = self.get_current_rate_future_products(data_ldp, col_deb, sc_rates, dar_mois, n)
        sc_rates = np.where(period_fixing, current_rate, sc_rates)

        sc_rates_lag = ut.roll_and_null(sc_rates, 1)
        sc_rates_lag[:, 0] = current_rate.reshape(n)

        return sc_rates, sc_rates_lag, data_ldp, period_fixing


    def make_rate_periodic_cap_floor(self, data_ldp, rate, fixing_date_month, col_fix_per, t):
        """ CREE DE NOUVELLES COLONNES POUR EVITER DE TOUCHER AUX DONNEES ORIGINALES"""
        for freq in data_ldp[col_fix_per].unique().astype(int):
            is_freq = (data_ldp[col_fix_per].values == freq)
            _n = is_freq[is_freq].shape[0]
            if freq > 1:
                is_freq = (data_ldp[col_fix_per].values == freq)
                orig_rate = rate[is_freq].copy()
                rate_t = rate[is_freq].copy()
                rate_t = ut.strided_indexing_roll(rate_t, np.minimum(0, -fixing_date_month[is_freq] + 1))
                real_rate = rate_t.copy()
                rate_t = np.full(rate_t.shape, np.nan)
                rate_t[:, ::freq] = real_rate[:, ::freq]
                rate_t = ut.np_ffill(rate_t)
                rate_t = ut.strided_indexing_roll(rate_t, np.maximum(0, fixing_date_month[is_freq] - 1), rep_nan=False)
                curr_month = np.arange(0, t).reshape(1, t).repeat(_n, axis=0)
                rate_t = np.where(curr_month < fixing_date_month[is_freq].reshape(_n, 1) - 1, orig_rate, rate_t)
                rate[is_freq] = rate_t

        rate[rate == np.inf] = np.nan
        rate[rate == -np.inf] = np.nan

        return rate

    def get_cap_floor_index_from_which_to_apply_fxing_date_rate(self, data_ldp, periodic_calendar,
                                                                fixing_date, t_proj, n):
        index_apply_fixing_date = np.maximum(1, ut.first_sup_val(periodic_calendar, axis=1, val=fixing_date,
                                                                 invalid_val=t_proj - 1))
        freq_int = np.array(data_ldp[self.cls_fields.NC_FREQ_INT])
        date_prec = periodic_calendar[np.arange(0, n).reshape(n, 1), index_apply_fixing_date.reshape(n, 1) - 1]
        date_post = periodic_calendar[np.arange(0, n).reshape(n, 1), index_apply_fixing_date.reshape(n, 1)]
        diff_fix_pre = np.abs(
            pd.DataFrame(fixing_date - date_prec).replace({pd.NaT: str(t_proj) + " days"}).values).reshape(
            n)
        diff_fix_post = pd.DataFrame(date_post - fixing_date).replace({pd.NaT: str(t_proj) + " days"}).values.reshape(n)
        date_prec2 = np.array(
            pd.Series(date_prec.reshape(n)).replace(pd.NaT, datetime.datetime(1900, 1, 1).date())).astype(
            'datetime64[M]')
        cond_precedent = (diff_fix_pre < diff_fix_post) & (date_prec2.reshape(n) > self.cls_hz_params.dar_usr.date())
        real_index_apply_fixing_date = np.maximum(1, np.where(cond_precedent, index_apply_fixing_date - freq_int,
                                                              index_apply_fixing_date))

        date_deb = np.where(cond_precedent.reshape(n, 1), date_prec, date_post)

        return real_index_apply_fixing_date, date_deb

    def mean_sc_rate(self, rate_sc, n, drac_amor, end_month, begin_month, current_month, t, is_float, fixed_rate):
        mean_rate = np.zeros((n))
        mean_rate[~is_float] = fixed_rate[~is_float]
        if is_float.any():
            _n = is_float[is_float].shape[0]
            rate_sc_mod = np.where(
                (current_month[is_float] < begin_month[is_float]) | (current_month[is_float] > end_month[is_float]),
                np.nan, rate_sc[is_float])
            rate_sc_mod = np.ma.masked_array(rate_sc_mod, np.isnan(rate_sc_mod))
            weights = ut.strided_indexing_roll(np.maximum(0, (drac_amor[is_float] - np.arange(0, t))),
                                               np.minimum(begin_month[is_float].reshape(_n) - 1, t))
            rate_average = np.ma.average(rate_sc_mod, axis=1, weights=weights)
            mean_rate[is_float] = (rate_average.filled(np.nan) / 12)

        return mean_rate.reshape(n, 1)
