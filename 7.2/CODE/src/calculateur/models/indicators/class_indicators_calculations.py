import numpy as np
import numexpr as ne
import pandas as pd
from calculateur.models.utils import utils as ut


class Indicators_Calculations():
    """
    Load les données utilisateurs
    """

    def __init__(self, cls_stat_runoff):
        self.cls_stat_runoff = cls_stat_runoff
        self.cls_proj = cls_stat_runoff.cls_proj
        self.cls_cal = cls_stat_runoff.cls_cal
        self.cls_hz_params = cls_stat_runoff.cls_hz_params
        self.cls_fields = cls_stat_runoff.cls_fields
        self.cls_palier = cls_stat_runoff.cls_proj.cls_palier
        self.load_dictionaries()

    def load_dictionaries(self):
        self.effet_rarn_cum = {}
        self.static_leg_capital = {}
        self.static_leg_capital_db = {}
        self.avg_static_leg_capital = {}
        self.static_mni = {}
        self.static_ftp_mni = {}
        self.day_val = {}
        for dic in [self.static_leg_capital, self.static_leg_capital_db, self.avg_static_leg_capital,
                    self.static_mni, self.static_ftp_mni, self.day_val]:
                for ind in ["liq", "taux"]:
                    dic[ind] = {}

    ######@profile
    def get_static_indics(self, cls_stat_runoff, data_ldp, remaining_capital, cls_rarn, ec_avt_amor,
                          mois_depart, mois_fin, mois_fin_amor, current_month, dar_mois, douteux,
                          tombee_fixing, n, t, type_ind="liq", period_fixing=[],
                          type_capital="all", rem_cap_gptx= []):

        self.update_former_leg_capital(data_ldp, remaining_capital, cls_rarn.tx_rarn, mois_fin_amor, current_month,
                                          dar_mois, douteux, n, type_ind=type_ind,
                                          period_fixing=period_fixing, type_capital=type_capital,
                                          rem_cap_gptx = rem_cap_gptx)

        self.generate_average_capital(data_ldp, self.static_leg_capital["liq"][type_capital].copy(), ec_avt_amor,
                                         mois_depart,
                                         current_month, dar_mois, douteux, n, t,
                                         cls_stat_runoff.cls_cash_flow.data_cash_flows_tombee,
                                         type_ind=type_ind, period_fixing=period_fixing,
                                         day_fixing=tombee_fixing, type_capital=type_capital,
                                         gptx_capital=self.static_leg_capital["taux"])

        self.calculate_static_leg_mni(data_ldp, self.static_leg_capital["liq"][type_capital].copy(),
                                         self.static_leg_capital_db["liq"][type_capital].copy(),
                                         self.day_val["liq"][type_capital],
                                         cls_stat_runoff.cls_rate, douteux,
                                         current_month, mois_fin, n, t, tombee_fixing, dar_mois,
                                         cls_stat_runoff.cls_cash_flow.data_cash_flows_tombee,
                                         type_ind=type_ind, period_fixing=period_fixing, type_capital=type_capital,
                                         gptx_capital = self.static_leg_capital["taux"],
                                         gptx_capital_db = self.static_leg_capital_db["taux"])



    ######@profile
    def update_former_leg_capital(self, data_ldp, remaining_capital, tx_rarn, mois_fin_amor, current_month, dar_mois,
                                  douteux, n, type_ind="liq", period_fixing=[], type_capital='all',
                                  rem_cap_gptx = []):
        mat_date = np.array(data_ldp[self.cls_fields.NC_LDP_MATUR_DATE]).reshape(n, 1)
        static_leg_capital = remaining_capital.copy()

        if type_ind == "taux":
            is_gptx_model = (data_ldp[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX] != "").values
            if is_gptx_model.any():
                static_leg_capital[is_gptx_model] = rem_cap_gptx[type_capital][is_gptx_model]

        rarn_na = np.isnan(tx_rarn)
        tx_rarn_max = np.minimum(1, tx_rarn)
        rarn_effect = ne.evaluate('where(rarn_na, 1, 1 - tx_rarn_max)')
        rarn_effect_cum = rarn_effect.cumprod(axis=1)

        static_leg_proj = static_leg_capital[:, 1:]
        static_leg_capital[:, 1:] = ne.evaluate('static_leg_proj * rarn_effect_cum')

        static_leg_cap_proj_lag = ut.roll_and_null(static_leg_capital[:, 1:])
        static_leg_cap_proj = static_leg_capital[:, 1:]

        # Lorsqu'on est en end long et que le jour de la date de valeur est supérieure à la date
        # de maturité, il faut ajouter 1 mois intermédiare sans amortissement
        static_leg_capital[:, 1:] = ne.evaluate("where((mois_fin_amor != mat_date - dar_mois) & (current_month == mois_fin_amor),"
                                                "static_leg_cap_proj_lag, static_leg_cap_proj)")
        if type_ind == "taux":
            #ERREUR_RCO
            period_fixing_gap\
                = self.ajust_period_fixing_with_date_gap_sortie(data_ldp, period_fixing, current_month, dar_mois, douteux, n)
            static_leg_cap_proj = static_leg_capital[:, 1:]
            static_leg_capital[:, 1:] = ne.evaluate("where(period_fixing_gap, static_leg_cap_proj, 0)")

        static_leg_capital[douteux] = remaining_capital[douteux]

        self.static_leg_capital[type_ind][type_capital] = static_leg_capital
        if type_ind == "liq":
            self.effet_rarn_cum = rarn_effect_cum

    ######@profile
    def generate_average_capital(self, data_ldp, static_leg_capital, ec_avt_amor, mois_depart,
                                 current_month, dar_mois, douteux, n, t, data_cf_tombee,
                                 type_ind="liq", period_fixing = [], day_fixing = [], type_capital= "all",
                                 gptx_capital = []):

        day_mat_date = self.cls_cal.day_mat_date
        day_val_date = self.cls_cal.day_val_date
        val_date_num = self.cls_cal.val_date_num
        nb_days = self.cls_cal.delta_days[:, 1:t + 1]
        day_tombee = self.cls_cal.day_tombee.copy()

        is_gptx_model = (data_ldp[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX] != "").values
        if is_gptx_model.any() and type_ind == "taux":
            static_leg_capital[is_gptx_model] = gptx_capital[type_capital][is_gptx_model]

        if len(data_cf_tombee) > 0:
            tombee_data_cf = np.array(data_cf_tombee[[x for x in data_cf_tombee.columns if x != self.cls_fields.NC_LDP_CONTRAT]])
            tombee_data_cf = tombee_data_cf[:, :t]
            new_day_tombee = day_tombee[data_cf_tombee.index]
            tombee_cf_exists = ~np.isnan(tombee_data_cf)
            day_tombee[data_cf_tombee.index] = ne.evaluate("where(tombee_cf_exists, tombee_data_cf, new_day_tombee)")

        static_leg_capital_db = np.zeros((static_leg_capital.shape))
        static_leg_capital_db[:, 1:] = static_leg_capital[:, :-1]
        static_leg_capital_db, day_val = self.adj_cap_deb_mois(static_leg_capital_db, ec_avt_amor,
                                                               val_date_num, day_val_date, day_mat_date, dar_mois,
                                                               current_month,mois_depart)

        avg_static_leg_capital = np.zeros(static_leg_capital.shape)
        static_leg_capital_db_proj = static_leg_capital_db[:, 1:]
        static_leg_capital_proj = static_leg_capital[:, 1:]

        if type_ind == "taux":
            period_fixing_gap = self.ajust_period_fixing_with_date_gap_sortie(data_ldp, period_fixing, current_month, dar_mois, douteux, n)
            period_fixing_gap_lag = ut.roll_and_null(period_fixing_gap, 1)
            period_fixing_gap_lag[:, 0] = True
            no_day_fixing = np.isnan(day_fixing) | (day_fixing == 0)
            day_fixing = ne.evaluate("where(no_day_fixing, day_tombee, day_fixing)")
            day_fix_gp = self.ajust_tombee_fixing_with_date_gap_sortie(data_ldp, day_fixing, current_month, dar_mois, n)
            day_fix_gp = np.nan_to_num(day_fix_gp)

            max_val_fix = np.maximum(day_fix_gp, day_val)
            max_mat_fix = np.maximum(day_fix_gp, day_tombee)
            min_mat_fix = np.minimum(day_fix_gp, day_tombee)

            nb_days1 = np.maximum(0, ne.evaluate("min_mat_fix - day_val + 1 - 1"))
            cap1 = ne.evaluate("period_fixing_gap_lag * static_leg_capital_db_proj")

            nb_days2 = np.maximum(0, ne.evaluate("day_tombee - max_val_fix + 1 - 1"))
            cap2 = (ne.evaluate("period_fixing_gap * static_leg_capital_db_proj"))

            nb_days3 = np.maximum(0, ne.evaluate("day_fix_gp - day_tombee + 1 - 1"))
            cap3 = ne.evaluate("static_leg_capital_proj * period_fixing_gap_lag")

            nb_days4 = ne.evaluate("nb_days - max_mat_fix + 1")
            cap4 = ne.evaluate("static_leg_capital_proj * period_fixing_gap")

            avg_static_leg_capital[:, 1:]\
                = ne.evaluate("(nb_days1 * cap1 + nb_days2 * cap2 + nb_days3 * cap3 + nb_days4 * cap4) / nb_days")

        else:
            nb_days1 = (day_tombee - day_val + 1 - 1)
            nb_days2 = (nb_days - day_tombee + 1)
            avg_static_leg_capital[:, 1:] = ne.evaluate("(nb_days1 * static_leg_capital_db_proj +"
                                                        " nb_days2 * static_leg_capital_proj) / nb_days")

        self.avg_static_leg_capital[type_ind][type_capital] = avg_static_leg_capital
        self.static_leg_capital_db[type_ind][type_capital] = static_leg_capital_db
        self.day_val[type_ind][type_capital] = day_val

    def adj_cap_deb_mois(self, static_leg_capital_db, ec_avt_amor, val_date, day_val_date, day_mat_date,
                         dar_mois, current_month, mois_depart):
        is_there_adj = ne.evaluate("where((day_val_date < day_mat_date) & (val_date > dar_mois), True, False)")
        ec_avt_amor_proj = ec_avt_amor[:, 1:]
        static_leg_capital_db_proj = static_leg_capital_db[:, 1:]
        static_leg_capital_db[:, 1:] = ne.evaluate("where(is_there_adj & (current_month == mois_depart),"
                                                   "ec_avt_amor_proj, static_leg_capital_db_proj)")

        day_val = ne.evaluate("where(is_there_adj & (current_month == mois_depart), day_val_date, 1)")

        return static_leg_capital_db, day_val

    ######@profile
    def calculate_static_leg_mni(self, data_ldp, static_leg_capital, static_leg_capital_db, day_val, cls_rate,
                                 douteux, current_month, mois_fin, n, t, day_fixing, dar_mois,
                                 data_cf_tombee, type_ind="taux", period_fixing = [],
                                 type_capital = "all", gptx_capital = [], gptx_capital_db=[]):

        sc_rates = np.nan_to_num(cls_rate.sc_rates[:, :t])
        sc_rates_lag = np.nan_to_num(cls_rate.sc_rates_lag[:, :t])
        sc_rates_ftp = np.nan_to_num(cls_rate.sc_rates_ftp[:, :t])
        sc_rates_ftp_lag = np.nan_to_num(cls_rate.sc_rates_ftp_lag[:, :t])

        nb_days_an = self.cls_stat_runoff.year_nb_days
        nb_days = self.cls_cal.delta_days[:, 1:t + 1]
        base_calc = data_ldp[self.cls_fields.NC_LDP_ACCRUAL_BASIS]
        nb_days_div = np.where(base_calc.isin(["30/360", "30E/360", "30A/360"]).values.reshape(n, 1), 30, nb_days)
        nb_days_div = np.where(data_ldp[self.cls_fields.NC_LDP_CONTRACT_TYPE].isin(["A-REGUL", "P-REGUL"]).values.reshape(n, 1),
                               nb_days, nb_days_div)
        day_tombee = self.cls_cal.day_tombee.copy()

        divider = 1 / nb_days_an * nb_days_div / nb_days

        is_gptx_model = (data_ldp[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX] != "").values
        if is_gptx_model.any() and type_ind=="taux":
            static_leg_capital[is_gptx_model] = gptx_capital[type_capital][is_gptx_model]
            static_leg_capital_db[is_gptx_model] = gptx_capital_db[type_capital][is_gptx_model]

        if len(data_cf_tombee) > 0:
            tombee_data_cf = np.array(data_cf_tombee[[x for x in data_cf_tombee.columns if x != self.cls_fields.NC_LDP_CONTRAT]])
            tombee_data_cf = tombee_data_cf[:, :t]
            new_day_tombee = day_tombee[data_cf_tombee.index]
            tombee_cf_exists = ~np.isnan(tombee_data_cf)
            day_tombee[data_cf_tombee.index] = ne.evaluate("where(tombee_cf_exists, tombee_data_cf, new_day_tombee)")

        no_day_fixing = np.isnan(day_fixing) | (day_fixing == 0)
        day_fixing = ne.evaluate("where(no_day_fixing, day_tombee, day_fixing)")

        if hasattr(self.cls_proj.cls_rate, 'day_fixing_ftp'):
            day_fixing_ftp = self.cls_proj.cls_rate.day_fixing_ftp
        else:
            day_fixing_ftp = day_fixing.copy()

        static_mni = np.zeros(static_leg_capital.shape)
        static_ftp_mni = np.zeros(static_leg_capital.shape)

        # CAP_FLOOR
        is_cap_floor = np.array(data_ldp[self.cls_fields.NC_PRODUCT_TYPE] == "CAP_FLOOR")
        _n = is_cap_floor[is_cap_floor].shape[0]
        if _n > 0:
            sc_rates_cf = sc_rates[is_cap_floor].copy()
            sc_rates_lag_cf = sc_rates_lag[is_cap_floor].copy()
            data_hab_cf = data_ldp[is_cap_floor].copy()
            sc_rates[is_cap_floor], sc_rates_lag[is_cap_floor]\
                = self.get_sc_rates_cap_floor(data_hab_cf, sc_rates_cf, sc_rates_lag_cf, _n)

        static_leg_capital_db_proj = static_leg_capital_db[:, 1:]
        static_leg_capital_proj = static_leg_capital[:, 1:]

        if type_ind == "liq":
            max_val_fix = np.maximum(day_fixing, day_val)
            max_mat_fix = np.maximum(day_fixing, day_tombee)
            min_mat_fix = np.minimum(day_fixing, day_tombee)

            nb_days1 = np.maximum(0, ne.evaluate("min_mat_fix - day_val"))
            nb_days2 = np.maximum(0, ne.evaluate("day_tombee - max_val_fix"))
            nb_days3 = np.maximum(0, ne.evaluate("day_fixing - day_tombee"))
            nb_days4 = ne.evaluate("(nb_days - max_mat_fix + 1)")

            mni1 = ne.evaluate("sc_rates_lag * static_leg_capital_db_proj")
            mni2 = ne.evaluate("sc_rates * static_leg_capital_db_proj")
            mni3 = ne.evaluate("sc_rates_lag * static_leg_capital_proj")
            mni4 = ne.evaluate("sc_rates * static_leg_capital_proj")

            static_mni[:, 1:] = ne.evaluate("(nb_days1 * mni1 + nb_days2 * mni2 +"
                                            " nb_days3 * mni3 + nb_days4 * mni4) * divider")

            max_val_fix = np.maximum(day_fixing_ftp, day_val)
            max_mat_fix = np.maximum(day_fixing_ftp, day_tombee)
            min_mat_fix = np.minimum(day_fixing_ftp, day_tombee)

            nb_days1 = np.maximum(0, ne.evaluate("min_mat_fix - day_val"))
            nb_days2 = np.maximum(0, ne.evaluate("day_tombee - max_val_fix"))
            nb_days3= np.maximum(0, ne.evaluate("day_fixing_ftp - day_tombee"))
            nb_days4 = ne.evaluate("(nb_days - max_mat_fix + 1)")

            mni1 = ne.evaluate("sc_rates_ftp_lag  * static_leg_capital_db_proj")
            mni2 = ne.evaluate("sc_rates_ftp  * static_leg_capital_db_proj")
            mni3 = ne.evaluate("sc_rates_ftp_lag * static_leg_capital_proj")
            mni4 = ne.evaluate("sc_rates_ftp * static_leg_capital_proj")

            static_ftp_mni[:, 1:] = ne.evaluate("(nb_days1 * mni1 + nb_days2 * mni2 +"
                                                " nb_days3 * mni3 + nb_days4 * mni4) * divider")

        else:
            period_fixing_gap = self.ajust_period_fixing_with_date_gap_sortie(data_ldp, period_fixing, current_month, dar_mois,
                                                                douteux, n)
            period_fixing_gap_lag = ut.roll_and_null(period_fixing_gap, 1)
            period_fixing_gap_lag[:, 0] = True

            day_fix_gp = self.ajust_tombee_fixing_with_date_gap_sortie(data_ldp, day_fixing, current_month, dar_mois, n)

            max_val_fix = np.maximum(day_fix_gp, day_val)
            max_mat_fix = np.maximum(day_fix_gp, day_tombee)
            min_mat_fix = np.minimum(day_fix_gp, day_tombee)

            nb_days1 = np.maximum(0, ne.evaluate("min_mat_fix - day_val + 1 - 1"))
            mni1 = ne.evaluate("sc_rates_lag * period_fixing_gap_lag * static_leg_capital_db_proj")

            nb_days2 = np.maximum(0, ne.evaluate("day_tombee - max_val_fix + 1 - 1"))
            mni2 = ne.evaluate("sc_rates * period_fixing_gap * static_leg_capital_db_proj")

            nb_days3 = np.maximum(0, ne.evaluate("day_fix_gp - day_tombee + 1 - 1"))
            mni3 = ne.evaluate("sc_rates_lag * static_leg_capital_proj * period_fixing_gap_lag ")

            nb_days4 = ne.evaluate("(nb_days - max_mat_fix + 1)")
            mni4 = ne.evaluate("sc_rates * period_fixing_gap * static_leg_capital_proj")

            static_mni[:, 1:] = ne.evaluate("(nb_days1 * mni1 + nb_days2 * mni2 + nb_days3 * mni3 + nb_days4 * mni4) * divider")

            mni1 = ne.evaluate("sc_rates_ftp_lag * period_fixing_gap_lag * static_leg_capital_db_proj")
            mni2 = ne.evaluate("sc_rates_ftp * period_fixing_gap * static_leg_capital_db_proj")
            mni3 = ne.evaluate("sc_rates_ftp_lag * static_leg_capital_proj * period_fixing_gap_lag ")
            mni4 = ne.evaluate("sc_rates_ftp * period_fixing_gap * static_leg_capital_proj")

            static_ftp_mni[:, 1:] = ne.evaluate("(nb_days1 * mni1 + nb_days2 * mni2 + nb_days3 * mni3"
                                                " + nb_days4 * mni4) * divider")

        mni_proj = static_mni[:, 1:]
        mni_ftp_proj = static_ftp_mni[:, 1:]
        not_douteux = (~douteux).reshape(n, 1)
        cond_zero = ne.evaluate("(day_tombee == 0) & not_douteux & (current_month > mois_fin)")
        static_mni[:, 1:] = ne.evaluate("where(cond_zero, 0, mni_proj)")
        static_ftp_mni[:, 1:] = ne.evaluate("where(cond_zero, 0, mni_ftp_proj)")

        if self.cls_proj.name_product in ["p-security-tf", "p-security-tv", "a-security-tf", "a-security-tv"]:
            self.adjust_mni_bonds(static_mni, static_leg_capital_proj, static_leg_capital_db_proj)

        is_nan_mni = np.isnan(static_mni)
        is_nan_ftp_mni = np.isnan(static_ftp_mni)
        self.static_mni[type_ind][type_capital] = ne.evaluate("where(is_nan_mni,0, static_mni)")
        self.static_ftp_mni[type_ind][type_capital] = ne.evaluate("where(is_nan_ftp_mni,0, static_ftp_mni)")


    def adjust_mni_bonds(self, static_mni, static_leg_capital_proj, static_leg_capital_db_proj):
        index_surcote_decote = np.arange(static_mni.shape[0])[1::2]
        delta_decote_surcote = (static_leg_capital_proj[index_surcote_decote]
                                - static_leg_capital_db_proj[index_surcote_decote])
        static_mni[index_surcote_decote, 1:] = delta_decote_surcote

        return static_mni

    def get_sc_rates_cap_floor(self, data_hab, sc_rates, sc_rates_lag, n):
        is_cap = np.array(data_hab[self.cls_fields.NC_LDP_IS_CAP_FLOOR] == "C").reshape(n, 1)
        strike_rate = np.array(data_hab[self.cls_fields.NC_LDP_RATE]).reshape(n, 1) * 12
        sc_rates = np.maximum(0, ne.evaluate("where(is_cap, sc_rates - strike_rate, strike_rate - sc_rates)"))
        sc_rates_lag = np.maximum(0, ne.evaluate("where(is_cap, sc_rates_lag - strike_rate, strike_rate - sc_rates_lag)"))
        return sc_rates, sc_rates_lag

    def ajust_tombee_fixing_with_date_gap_sortie(self, data, tombee_fixing, current_month, dar_mois, n):
        month_date_gap = data[self.cls_fields.NC_LDP_DATE_SORTIE_GAP].values.reshape(n, 1)
        is_gptx_model = (data[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX] != "").values.reshape(n, 1)
        date_gap = data[self.cls_fields.NC_LDP_DATE_SORTIE_GAP_REAL]
        day_date_gap = np.array(pd.to_datetime(date_gap).dt.day).astype(int).reshape(n, 1)
        is_date_gap = month_date_gap != 0
        tombee_fixing_gap = ne.evaluate("where(is_date_gap & (current_month == month_date_gap - dar_mois), day_date_gap, tombee_fixing)")

        return tombee_fixing_gap

    def ajust_period_fixing_with_date_gap_sortie(self, data, period_fixing, current_month, dar_mois, douteux, n):
        month_date_gap = data[self.cls_fields.NC_LDP_DATE_SORTIE_GAP].values.reshape(n, 1)
        is_gptx_model = (data[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX] != "").values.reshape(n, 1)
        is_date_gap = (month_date_gap != 0)
        is_date_gap_n = is_date_gap.reshape(n)
        period_fixing_gap = period_fixing.copy()
        period_fixing_gap[is_date_gap_n]\
            = np.where((current_month[is_date_gap_n] >= (month_date_gap - dar_mois)[is_date_gap_n]), False, True)
        period_fixing_gap = np.where(is_gptx_model, True, period_fixing_gap)

        # ERREUR_RCO
        null_compound_per = np.isin(data[self.cls_fields.NC_LDP_CAPITALIZATION_PERIOD].fillna("").astype(str).values.reshape(n, 1), ["N",""])
        is_douteux_nofixing = ((data[self.cls_fields.NC_LDP_FIXING_NEXT_DATE].values.reshape(n, 1)==0) & np.array(douteux).reshape(n, 1)
                               & (~is_date_gap)) & (null_compound_per)
        period_fixing_gap = ne.evaluate("where(is_douteux_nofixing, True, period_fixing_gap)")

        return period_fixing_gap
    ###########@profile


