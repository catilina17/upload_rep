import datetime
import logging
import copy
import numpy as np
import numexpr as ne
import pandas as pd

from ..amortissement_statique.class_init_ecoulement import Init_Ecoulement
from ..amortissement_statique.class_static_amortization import Static_Amortization
from calculateur.models.indicators.class_indicators_calculations import Indicators_Calculations
from calculateur.models.utils import utils as ut
from utils import general_utils as gu
from calculateur.models.rarn.class_rarn_manager import RARN_Manager

logger = logging.getLogger(__name__)


class RENGOCIATED_AMORTIZATION():

    def __init__(self, cls_proj, cls_model_params, cls_cash_flow, cls_format, tx_params, cls_data_palier):
        self.cls_proj = cls_proj
        self.cls_format = cls_format
        self.cls_fields = cls_proj.cls_fields
        self.cls_cal = self.cls_proj.cls_cal
        self.cls_model_params = cls_model_params
        self.cls_ra_rn_params = self.cls_model_params.cls_ra_rn_params
        self.tx_params = tx_params
        self.cls_cash_flow = cls_cash_flow
        self.cls_data_palier = cls_data_palier

    def calculate_renegociated_amortization(self, cls_static_ind, remaining_capital, rarn_effect, cls_rarn,
                                            current_month, mois_depart, douteux, n, t):

        (self.reneg_leg_capital, self.avg_reneg_leg_capital,
         self.reneg_leg_mni, self.reneg_leg_ftp_mni, self.reneg_rate) = (np.zeros((n, t)).copy(),
                                                                         np.zeros((n, t)).copy(),
                                                                         np.zeros((n, t)).copy(),
                                                                         np.zeros((n, t)).copy(),
                                                                         np.zeros((n, t)).copy())

        if (cls_static_ind.cls_proj.name_product not in ['nmd_st', 'nmd_pn']):
            is_reneg = (~np.isnan(cls_rarn.rate_renego_cat)).any(axis=1)
            _n = is_reneg[is_reneg].shape[0]
            if _n > 0:
                amount_renegociated = self.calculate_renegociated_amount(remaining_capital, rarn_effect,
                                                                         cls_rarn.tx_rn, current_month,
                                                                         mois_depart)
                if amount_renegociated.sum() == 0:
                    t_rn = 0
                else:
                    t_rn = int(np.where(amount_renegociated > 0, current_month, 0).max())

                filter_reneg = (amount_renegociated.sum(axis=1) != 0) & (~douteux)
                n_r = filter_reneg[filter_reneg].shape[0]

                if t_rn > 0 and n_r > 0:
                    cls_reneg_ind = self.generate_reneg_amortization(amount_renegociated[filter_reneg, :t_rn],
                                                                     cls_rarn.rate_renego_cat[filter_reneg, :t_rn],
                                                                     filter_reneg, n_r, t_rn)

                    self.reneg_leg_capital[filter_reneg] = self.sum_and_reshape(
                        cls_reneg_ind.static_leg_capital["liq"]["all"], t_rn)
                    self.avg_reneg_leg_capital[filter_reneg] = self.sum_and_reshape(
                        cls_reneg_ind.avg_static_leg_capital["liq"]["all"], t_rn)
                    self.reneg_leg_mni[filter_reneg] = self.sum_and_reshape(cls_reneg_ind.static_mni["liq"]["all"],
                                                                            t_rn)
                    self.reneg_leg_ftp_mni[filter_reneg] = self.sum_and_reshape(
                        cls_reneg_ind.static_ftp_mni["liq"]["all"], t_rn)
                    self.reneg_rate[filter_reneg] = np.nan_to_num(self.reneg_leg_mni[filter_reneg]
                                                                  / self.avg_reneg_leg_capital[filter_reneg]) * 12

    def sum_and_reshape(self, ind_mat, t_rn):
        ind_mat_1 = ind_mat[:, 1:].copy()
        ind_mat_2 = ind_mat_1.reshape(ind_mat_1.shape[0] // t_rn, t_rn, ind_mat_1.shape[1])
        ind_mat_3 = np.sum(ind_mat_2, axis=1)
        return ind_mat_3

    def calculate_renegociated_amount(self, remaining_capital, rarn_effect, tx_rn, current_month, mois_depart):
        rarn_effect_lagged = ut.roll_and_null(rarn_effect, val=1)
        rc = remaining_capital[:, 1:]
        base_capital_rn = ne.evaluate('rc * rarn_effect_lagged')
        base_capital_rn = ne.evaluate('where(current_month < mois_depart, 0, base_capital_rn)')
        amount_renegociated = ne.evaluate('base_capital_rn * tx_rn')
        return amount_renegociated

    def calculate_new_calendar_params(self):
        begin_month = self.cls_cal.mois_depart
        amor_begin_month = self.cls_cal.mois_depart_amor
        amor_end_month = self.cls_cal.mois_fin_amor
        drac_amor = self.cls_cal.drac_amor

    def calculate_mois_depart_amor(self, drac_amor, mois_depart_amor, n, t, t_l):
        mois_reneg = np.arange(1, t + 1).reshape(1, t, 1)
        _drac_amor = (np.maximum(0, drac_amor.reshape(n, 1, 1)
                                 - np.maximum(0, mois_reneg - mois_depart_amor.reshape(n, 1, 1) + 1)))[:, :t_l]
        _mois_depart_amor = np.maximum(mois_reneg + 1, mois_depart_amor.reshape(n, 1, 1))[:, :t_l]

        return _mois_depart_amor

    def reformat_data_ldp(self, cls_proj_rn, filter_reneg, t_rn, n_r):
        cls_proj_rn.data_ldp = cls_proj_rn.data_ldp[filter_reneg].copy()
        cls_proj_rn.data_ldp = cls_proj_rn.data_ldp.loc[cls_proj_rn.data_ldp.index.repeat(t_rn)].reset_index(drop=True)
        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_CLE] \
            = cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_CLE] + "_" + pd.Series(
            np.tile(np.arange(1, t_rn + 1), n_r)).astype(str)

        _n = len(cls_proj_rn.contracts_updated)
        if _n > 0:
            cls_proj_rn.contracts_updated\
                = cls_proj_rn.contracts_updated.loc[cls_proj_rn.contracts_updated.index.repeat(t_rn)]
            cls_proj_rn.contracts_updated.index \
                = cls_proj_rn.contracts_updated.index + "_" + pd.Series(np.tile(np.arange(1, t_rn + 1), _n)).astype(str)

    def assign_new_nominal(self, cls_proj_rn, amount_renegociated):
        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_NOMINAL] = amount_renegociated.reshape(cls_proj_rn.n, 1)
        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_OUTSTANDING] = 0
        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_RELEASING_RULE] = np.nan
        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_RELEASING_DATE] = 0
        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_RELEASING_DATE + "_REAL"] = datetime.datetime(1990, 1, 1)
        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_ECHEANCE_VAL] = np.nan
        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_INTERESTS_ACCRUALS] = np.nan
        cls_proj_rn.data_ldp[self.cls_fields.NC_PROFIL_AMOR] \
            = np.where(cls_proj_rn.data_ldp[self.cls_fields.NC_PROFIL_AMOR] == "LINEAIRE_ECH", "LINEAIRE",
                       cls_proj_rn.data_ldp[self.cls_fields.NC_PROFIL_AMOR])

    def assign_new_value_date(self, cls_proj_rn, filter_reneg, dar_mois, dar_usr, n_r, t_rn):
        n = n_r * t_rn
        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_VALUE_DATE] \
            = (np.full((n_r, 1), dar_mois)
               + np.arange(1, t_rn + 1).reshape(1, t_rn)).reshape(cls_proj_rn.n)

        begin_date = datetime.datetime(dar_usr.year, dar_usr.month, 1)
        new_val_date = (np.array(begin_date).astype("datetime64[M]")
                        + np.concatenate([np.arange(1, t_rn + 1)] * n_r, axis=0).astype("timedelta64[M]")).astype(
            "datetime64[D]").reshape(n_r, t_rn)

        day_mat = cls_proj_rn.cls_cal.mat_date_day[filter_reneg] - 1
        day_mat = np.array(day_mat.astype("timedelta64[D]")).reshape(n_r, 1)

        day_val = cls_proj_rn.cls_cal.val_date_day[filter_reneg] - 1
        day_val = np.array(day_val.astype("timedelta64[D]")).reshape(n_r, 1)

        broken_period = np.array(cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_BROKEN_PERIOD])[::t_rn]
        is_ss_or_sl = np.isin(broken_period, ["SS", "SL"]).reshape(n_r, 1)

        day_adj = np.where(~(is_ss_or_sl), day_val, day_mat)
        new_val_date = new_val_date + day_adj
        new_val_date = np.minimum(new_val_date, cls_proj_rn.cls_cal.end_month_date_deb[:, 1: t_rn + 1])

        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_VALUE_DATE_REAL] = new_val_date.reshape(n)

        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_TRADE_DATE]\
            = cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_VALUE_DATE].copy()
        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_TRADE_DATE_REAL]\
            = cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_VALUE_DATE_REAL].copy()

    def assign_new_rate(self, cls_proj_rn, rate_renego):
        spread = self.cls_ra_rn_params.spread_renego / 10000
        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_RATE] = rate_renego.reshape(cls_proj_rn.n, 1) / 12
        cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_FTP_RATE] = (rate_renego - spread).reshape(cls_proj_rn.n, 1) / 12
        #Commenter ligne précédente si MNI FTP RCO

    def calculate_new_palier_schedule(self, cls_proj_rn):
        cls_proj_rn.cls_palier.prepare_palier_data(cls_proj_rn, self.cls_data_palier,
                                                   self.cls_model_params.cls_ra_rn_params)
        max_palier = cls_proj_rn.cls_palier.dic_palier["max_palier"]
        if max_palier > 1 and "palier_schedule" in cls_proj_rn.cls_palier.dic_palier:
            cols_rate_palier = [self.cls_fields.NC_RATE_PALIER + str(pal_nb) for pal_nb in
                                     range(2, max_palier + 1)]
            cls_proj_rn.cls_palier.dic_palier["palier_schedule"][cols_rate_palier] = np.nan
            cols_rate_palier = [self.cls_fields.NC_RATE_PALIER + str(pal_nb) for pal_nb in
                                     range(1, max_palier + 1)]
            cls_proj_rn.cls_palier.dic_palier["palier_schedule"][cols_rate_palier] = \
                cls_proj_rn.cls_palier.dic_palier["palier_schedule"][cols_rate_palier].ffill(axis=1)

            cols_val_ech_palier = [self.cls_fields.NC_VAL_PALIER + str(pal_nb) for pal_nb in
                                     range(1, max_palier + 1)]
            cls_proj_rn.cls_palier.dic_palier["palier_schedule"][cols_val_ech_palier] = -10000

            _n = cls_proj_rn.cls_palier.dic_palier["palier_schedule"].shape[0]

            cols = [self.cls_fields.NC_DATE_PALIER_REAL + str(pal_nb) for pal_nb in range(2, max_palier + 1)]
            obj = cls_proj_rn.cls_palier.dic_palier["palier_schedule"][cols].values
            compa = cls_proj_rn.cls_palier.dic_palier["palier_schedule"][self.cls_fields.NC_DATE_PALIER_REAL + str(1)].values.reshape(_n, 1)
            compa = np.concatenate([compa] * (max_palier - 1), axis=1)
            cls_proj_rn.cls_palier.dic_palier["palier_schedule"][cols] = np.where(obj < compa, compa, obj)

            all_cols = [self.cls_fields.NC_MOIS_DATE_PALIER, self.cls_fields.NC_MOIS_PALIER_AMOR,
                        self.cls_fields.NC_MOIS_PALIER_DEBUT]

            cols = gu.flatten([[col + str(pal_nb) for pal_nb in range(2, max_palier + 1)] for col in all_cols])
            obj = cls_proj_rn.cls_palier.dic_palier["palier_schedule"][cols].values
            compa = [cls_proj_rn.cls_palier.dic_palier["palier_schedule"][col + str(1)].values.reshape(_n, 1) for col in all_cols]
            compa = np.concatenate(compa, axis=1)
            compa = np.repeat(compa, max_palier - 1,axis=1)
            cls_proj_rn.cls_palier.dic_palier["palier_schedule"][cols] = np.where(obj < compa, compa, obj).astype(int)

    def generate_reneg_amortization(self, amount_renegociated, rate_renego, filter_reneg, n_r, t_rn):
        cls_proj_rn = copy.deepcopy(self.cls_proj)
        dar_mois = cls_proj_rn.cls_hz_params.dar_mois
        dar_usr = cls_proj_rn.cls_hz_params.dar_usr
        self.reformat_data_ldp(cls_proj_rn, filter_reneg, t_rn, n_r)

        cls_proj_rn.n = n_r * t_rn
        cls_proj_rn.current_month, cls_proj_rn.current_month_max = cls_proj_rn.build_current_month()

        self.assign_new_nominal(cls_proj_rn, amount_renegociated)
        self.assign_new_value_date(cls_proj_rn, filter_reneg, dar_mois, dar_usr, n_r, t_rn)
        self.assign_new_rate(cls_proj_rn, rate_renego)

        cls_proj_rn.cls_cal.prepare_calendar_parameters(cls_proj_rn)
        cls_proj_rn.cls_cal.get_calendar_periods(cls_proj_rn)
        cls_proj_rn.cls_fixing_cal.get_fixing_parameters(cls_proj_rn)

        self.calculate_new_palier_schedule(cls_proj_rn)
        self.generate_rates_chronicles(cls_proj_rn, self.cls_model_params, self.tx_params)

        cls_init_ec = Init_Ecoulement(cls_proj_rn, self.cls_model_params)
        cls_init_ec.get_ec_before_amortization()

        """ CALCUL AMORTISSEMENT """
        cls_reneg_amor = Static_Amortization(self.cls_format, cls_init_ec, self.cls_cash_flow)
        cls_reneg_amor.generate_static_amortization(self.tx_params)

        cls_rarn = RARN_Manager(cls_proj_rn, self.cls_model_params)
        cls_rarn.tx_rarn = np.zeros((cls_proj_rn.n, cls_proj_rn.t))
        cls_rarn.tx_ra = np.zeros((cls_proj_rn.n, cls_proj_rn.t))
        cls_rarn.tx_rn = np.zeros((cls_proj_rn.n, cls_proj_rn.t))

        n = cls_proj_rn.n
        t = cls_proj_rn.t
        cls_reneg_ind = Indicators_Calculations(cls_reneg_amor)
        mois_depart = cls_proj_rn.cls_cal.mois_depart
        mois_fin = cls_proj_rn.cls_cal.mois_fin
        mois_fin_amor = cls_proj_rn.cls_cal.mois_fin_amor
        current_month = cls_proj_rn.cls_cal.current_month
        douteux = np.array((cls_proj_rn.data_ldp[self.cls_fields.NC_LDP_PERFORMING] == "T"))
        tombee_fixing = cls_proj_rn.cls_rate.tombee_fixing[:, :t]
        period_fixing = cls_proj_rn.cls_rate.period_fixing[:, :t]

        cls_proj_rn.cls_cal.prepare_data_cal_indicators(cls_proj_rn.data_ldp, cls_proj_rn.name_product,
                                                        cls_reneg_amor, n, t)
        cls_reneg_ind.get_static_indics(cls_reneg_amor, cls_proj_rn.data_ldp, cls_reneg_amor.capital_ec, cls_rarn,
                                        cls_reneg_amor.cls_init_ec.ec_depart, mois_depart, mois_fin,
                                        mois_fin_amor, current_month, dar_mois, douteux, tombee_fixing, n, t,
                                        type_ind="liq", period_fixing=period_fixing,
                                        type_capital="all")
        return cls_reneg_ind

    def generate_rates_chronicles(self, cls_proj, cls_model_params, tx_params):
        cls_proj.cls_data_rate.prepare_curve_rates(cls_proj, cls_model_params, tx_params)
        cls_proj.cls_rate.get_rates(cls_proj, cls_model_params)
