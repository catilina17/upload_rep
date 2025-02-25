import numpy as np
import pandas as pd
import logging
import numexpr as ne
from datetime import datetime
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)
list_absent_vars_ldp = [];
list_absent_vars_palier = []


class Data_Formater():
    """
    Formate les données
    """

    def __init__(self, cls_fields):
        self.cls_fields = cls_fields
        self.list_absent_vars_ldp = []
        self.list_absent_vars_palier = []
        self.default_date = datetime(1900,1,1)
        self.map_faulty_periodicities = {"1M2W" : "1M", "3M2W" : "3M"}

    #######@profile
    def parse_coupon_date(self, data):
        coupon_date = pd.to_datetime(data[self.cls_fields.NC_LDP_FIRST_COUPON_DATE + "_REAL"])
        coupon_date_mois = data[self.cls_fields.NC_LDP_FIRST_COUPON_DATE]
        cond = coupon_date.dt.day < 15
        coupon_date = np.where(cond,
                               coupon_date.values.astype('datetime64[M]').astype('datetime64[D]'),
                               pd.to_datetime(coupon_date.values.astype('datetime64[M]')) + pd.DateOffset(months=1))
        data[self.cls_fields.NC_LDP_FIRST_COUPON_DATE + "_REAL"] = coupon_date.astype('datetime64[D]')
        data[self.cls_fields.NC_LDP_FIRST_COUPON_DATE] = np.where(cond, coupon_date_mois, coupon_date_mois + 1)

        return data

    def parse_date_vars(self, data_ldp_hab):
        data_ldp_hab[self.cls_fields.NC_LDP_CLE] = ["LDP" + str(x) for x in range(1, len(data_ldp_hab) + 1)]

        first_amor_date = self.cls_fields.NC_LDP_FIRST_AMORT_DATE
        value_date = self.cls_fields.NC_LDP_VALUE_DATE
        matur_date = self.cls_fields.NC_LDP_MATUR_DATE
        trade_date = self.cls_fields.NC_LDP_TRADE_DATE
        next_fixing_date = self.cls_fields.NC_LDP_FIXING_NEXT_DATE
        date_sortie_taux = self.cls_fields.NC_LDP_DATE_SORTIE_GAP
        first_coupon_date = self.cls_fields.NC_LDP_FIRST_COUPON_DATE

        data_ldp_hab = \
            self.parse_date_list(data_ldp_hab,
                                 [first_amor_date, value_date, matur_date, self.cls_fields.NC_LDP_RELEASING_DATE,
                                  trade_date,
                                  next_fixing_date, date_sortie_taux, first_coupon_date])

        cas1 = (data_ldp_hab[first_amor_date] != 0) & (data_ldp_hab[value_date] > data_ldp_hab[first_amor_date])

        cas2 = (data_ldp_hab[matur_date] < data_ldp_hab[first_amor_date])

        cas3 = (data_ldp_hab[value_date] > data_ldp_hab[matur_date]) & (
                data_ldp_hab[matur_date] >= data_ldp_hab[trade_date])

        data_ldp_hab[value_date] = np.select([cas3, cas1], [data_ldp_hab[trade_date], data_ldp_hab[first_amor_date]], \
                                             default=data_ldp_hab[value_date])

        data_ldp_hab[first_amor_date] = np.select([cas2, cas1], [data_ldp_hab[matur_date], 0],
                                                  default=data_ldp_hab[first_amor_date])

        return data_ldp_hab

    ###########@profile
    def parse_date_list(self, data, date_list, in_month=[]):
        if in_month == []:
            in_month = [True] * len(date_list)
        for name_col, in_num in zip(date_list, in_month):
            data[name_col] = self.convert_to_datetime(data, name_col)
            if in_num:
                data[name_col + "_REAL"], data[name_col] = self.convert_to_nb_months(data[name_col])
        return data

    def convert_to_nb_months(self, datum):
        cole_date_real = datum.fillna(self.default_date).values.astype("datetime64[D]")
        cole_date_month = (12 * datum.dt.year + datum.dt.month).fillna(0).astype(int)
        return cole_date_real, cole_date_month

    def convert_to_datetime(self, data, name_col):
        col_date_real = data[name_col].copy()
        cole_date_new = pd.to_datetime(data[name_col], errors='coerce', format="%d/%m/%Y")
        cole_date_new = cole_date_new.mask(col_date_real.astype(str).str.contains("/2") & data[name_col].isnull(),
                                             datetime(2200, 1, 1))
        return cole_date_new

    def format_capi_mode_col(self, data):
        data[self.cls_fields.NC_LDP_CAPI_MODE] = data[self.cls_fields.NC_LDP_CAPI_MODE].fillna("").astype(str)
        return data

    #######@profile
    def format_fixing_date_fixing_rule(self, data_ldp, dar_usr):
        filter_fixing_rule = (data_ldp[self.cls_fields.NC_LDP_FIXING_RULE] == "R").values
        value_date = data_ldp.loc[filter_fixing_rule, self.cls_fields.NC_LDP_VALUE_DATE_REAL].values
        new_fix_date = np.array([dar_usr.replace(day=1) + relativedelta(months=1)] * data_ldp[filter_fixing_rule].shape[0]).astype("datetime64[D]")
        new_date = np.where(value_date > dar_usr, value_date, new_fix_date)
        data_ldp.loc[filter_fixing_rule, self.cls_fields.NC_LDP_FIXING_NEXT_DATE_REAL] = new_date
        return data_ldp

    #######@profile
    def filter_ldp_data(self, data_ldp_hab, dar_usr):
        data_ldp_hab = data_ldp_hab[data_ldp_hab[self.cls_fields.NC_LDP_MATUR_DATE_REAL] >= dar_usr].copy()
        #non_null_amounts = ((data_ldp_hab[self.cls_fields.NC_LDP_NOMINAL] != 0) | (
        #            data_ldp_hab[self.cls_fields.NC_LDP_OUTSTANDING] != 0)).values
        #data_ldp_hab = data_ldp_hab.loc[non_null_amounts].copy()
        return data_ldp_hab

    def create_unvailable_variables(self, data_hab, vars, default_values, type="ldp"):
        list_absent = []
        for var in vars:
            if var not in data_hab.columns.tolist():
                if (not var in list_absent_vars_ldp and type == "ldp") \
                        or (not var in list_absent_vars_palier and type != "ldp"):
                    list_absent.append(var)
                    if type == "ldp":
                        list_absent_vars_ldp.append(var)
                    else:
                        list_absent_vars_palier.append(var)
                if "EXISTING:" in str(default_values[var]):
                    data_hab[var] = data_hab[default_values[var].replace("EXISTING:", "")].values
                elif '__IS_CALC' not in str(default_values[var]):
                    data_hab[var] = default_values[var]
        if list_absent != []:
            logger.debug("    File %s : list of absent variables created : %s" % (type.upper(), list_absent))
        return data_hab

    def format_ldp_num_cols(self, data, num_cols, divider, na_fills, passif_sensitive=[]):
        if passif_sensitive == []:
            passif_sensitive = [False] * len(divider)
        for num_col, div, na_fill, passif in zip(num_cols, divider, na_fills, passif_sensitive):
            if passif:
                is_passif = (data[self.cls_fields.NC_PRODUCT_TYPE] == "PASSIF").values
                div = ne.evaluate("div * where(is_passif, -1, 1)")
            data[num_col] = data[num_col].astype(np.float64).fillna(na_fill).astype(str).str.replace(",", ".") \
                                .replace("-", str(na_fill)).replace("", str(na_fill)).apply(float) / div

        return data

    def format_nb_contracts(self, data_ldp):
        data_ldp[self.cls_fields.NC_LDP_NB_CONTRACTS] = data_ldp[self.cls_fields.NC_LDP_NB_CONTRACTS].fillna(1)
        return data_ldp

    def verify_mult_spread(self, data, col_mult_spread):
        if (data[col_mult_spread] != 1).any():
            logger.debug("Certaines valeurs du MULT SPREAD sont différentes de 1 !")
        data[col_mult_spread] = 1



    def correct_adjst_signs(self, data, col_outstanding, col_nominal, col_nom_multiplier, name_product):
        if name_product in ["nmd_st", "nmd_pn"]:
            cond_nominal = ((data[col_outstanding] < 0).values)
        else:
            cond_nominal = ((data[col_nominal] < 0).values)
        data[col_nom_multiplier] = ne.evaluate("where(cond_nominal, -1, 1)")
        multiplier = data[col_nom_multiplier].values
        nominal = data[col_nominal].values
        outstanding = data[col_outstanding].values
        data[col_nominal] = ne.evaluate("multiplier * nominal")
        data[col_outstanding] = ne.evaluate("multiplier * outstanding")
        return data

    def format_value_passif(self, data, col_ech):
        n = data.shape[0]
        is_passif_actif = (data[self.cls_fields.NC_PRODUCT_TYPE] == "PASSIF").values
        coeff = ne.evaluate("where(is_passif_actif, -1, 1)").reshape(n, 1)
        old_vals = data.loc[:, col_ech].values.reshape(n, len(col_ech))
        data.loc[:, col_ech] = ne.evaluate("coeff * old_vals")
        return data

    def generate_amortissement_profil(self, data, col_type_amor):
        echconst = data[col_type_amor].isin(["A", "", "-", np.nan])
        lineaire = data[col_type_amor].isin(["L"])
        lineaire_ech = data[col_type_amor].isin(["M"])
        infine = (data[col_type_amor].isin(["F"]))
        cashflow = (data[col_type_amor].isin(["O"]))
        data[self.cls_fields.NC_PROFIL_AMOR] = np.select([echconst, lineaire, lineaire_ech, infine, cashflow],
                                                       ["ECHCONST", "LINEAIRE", "LINEAIRE_ECH", "INFINE", "CASHFLOWS"],
                                                       default="ECHCONST")

        return data

    def get_supsension_or_interets_capitalization_var(self, data, col_freq_int, dar_mois, capi_rate_col=[], capi_freq_col=[],
                                                      type_data="ldp"):
        _none = data[col_freq_int].isin(["N"])
        data[self.cls_fields.NC_LDP_FREQ_AMOR] = data[self.cls_fields.NC_LDP_FREQ_AMOR].fillna("").astype(str)
        if type_data == "ldp":
            # RCO ne semble le faire que pour les contrats sans palier
            # N Signifie en attente de  quelquechose avant amortissement
            #forward_amor = data[self.cls_fields.NC_LDP_FIRST_AMORT_DATE].values > dar_mois
            #per_amor_not_none = (data[self.cls_fields.NC_LDP_FREQ_AMOR].str.upper() != "NONE").values \
            #                    & (data[self.cls_fields.NC_LDP_FREQ_AMOR].str.upper().str.strip() != "").values
            data[self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS] = ne.evaluate('where(~_none, False, True)')

            is_sus_or_cap = data[self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS].values
            not_null_capi_rate_col = (data[capi_rate_col] != 0).values
            capi_freq_col_not_N = (data[capi_freq_col] != "N").values
            capi_freq_col_not_null = (data[capi_freq_col].notnull())
            data[self.cls_fields.NC_CAPITALIZE] \
                = ne.evaluate(
                "where(is_sus_or_cap & not_null_capi_rate_col & capi_freq_col_not_N & capi_freq_col_not_null, True, False)")

        else:
            #per_amor_not_none = ((data[self.cls_fields.NC_LDP_FREQ_AMOR].str.upper() != "NONE") \
            #                     & (data[self.cls_fields.NC_LDP_FREQ_AMOR].str.upper().str.strip() != "")).values

            data[self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS] = ne.evaluate('where(~_none, False, True)')
        return data

    def get_capitalization_var(self, data, col_freq_int, capi_rate_col, capi_freq_col):
        capi_not_null = (data[capi_rate_col] != 0) & (data[capi_rate_col].notnull()).values
        data[capi_freq_col] = data[capi_freq_col].mask(capi_not_null, data[col_freq_int])
        data[self.cls_fields.NC_CAPITALIZE] = np.where(capi_not_null, True, False)
        data[self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS] = np.where(capi_not_null, True, False)
        data[capi_freq_col] = data[capi_freq_col].mask(data[capi_freq_col].isnull() & capi_not_null, "E")

        return data
    def generate_freq_int(self, data, col_freq_int):
        mensuel = data[col_freq_int].isin(["M", "N"])
        annuel = data[col_freq_int].isin(["A"])
        trimestriel = data[col_freq_int].isin(["Q"])
        semestriel = data[col_freq_int].isin(["S"])  # PAS SUR

        data[self.cls_fields.NC_FREQ_INT] = \
            np.select([mensuel, annuel, trimestriel, semestriel], [1, 12, 3, 6], 1)

        return data

    def generate_freq_amor(self, data, col_freq_amor, col_freq_int):
        is_none = data[col_freq_amor].isin(["None"])
        mensuel = data[col_freq_amor].isin(["Monthly", "None"])
        annuel = data[col_freq_amor].isin(["Annual"])
        trimestriel = data[col_freq_amor].isin(["Quarterly"])
        semestriel = data[col_freq_amor].isin(["Semi-annual"])  # PAS SUR
        sus_or_cap = data[self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS].values
        data[self.cls_fields.NC_SUSPEND_AMOR_OR_CAPITALIZE_INTERESTS] = ne.evaluate("where(is_none, True, sus_or_cap)")

        data[col_freq_amor] = \
            np.select([is_none, mensuel, annuel, trimestriel, semestriel],
                      ["NONE", "MENSUEL", "ANNUEL", "TRIMESTRIEL", "SEMESTRIEL"])
        data[self.cls_fields.NC_FREQ_AMOR] = \
            np.select([mensuel, annuel, trimestriel, semestriel], [1, 12, 3, 6], 1)

        cond_update = data[col_freq_int].isin(["M", "A", "Q", "S"])
        data[self.cls_fields.NC_FREQ_AMOR] = np.where(cond_update,
                                                    np.maximum(data[self.cls_fields.NC_FREQ_AMOR],
                                                               data[self.cls_fields.NC_FREQ_INT]),
                                                    data[self.cls_fields.NC_FREQ_AMOR])

        return data

    def add_bilan_column(self, data, name_product):
        is_actif = ((data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "A-")
                    | (data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:5] == "HB-NS")
                    | (data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[-2:] == "-A")
                    | (data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:3] == "AHB"))
        is_passif = ((data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "P-")
                    | (data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[-2:] == "-P")
                     | (data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:3] == "PHB"))
        is_cap_floor = data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("HB-CAP|HB-FLOOR").values
        is_actif = is_actif & ~is_cap_floor
        is_passif = is_passif & ~is_cap_floor
        cases = [is_actif, is_passif, is_cap_floor]
        vals = ["ACTIF", "PASSIF", "CAP_FLOOR"]
        data[self.cls_fields.NC_PRODUCT_TYPE] = np.select(cases, vals, default="OTHER")
        if name_product in ["a-swap-tv", "a-swap-tf"]:
            data[self.cls_fields.NC_PRODUCT_TYPE] = "ACTIF"

        if name_product in ["p-swap-tv", "p-swap-tf"]:
            data[self.cls_fields.NC_PRODUCT_TYPE] = "PASSIF"

        if name_product in ["a-change-tv", "a-change-tf"]:
            data[self.cls_fields.NC_PRODUCT_TYPE] = "ACTIF"

        if name_product in ["p-change-tv", "p-change-tf"]:
            data[self.cls_fields.NC_PRODUCT_TYPE] = "PASSIF"

        if (data[self.cls_fields.NC_PRODUCT_TYPE] == "OTHER").any():
            prob_contrats = data[data[self.cls_fields.NC_PRODUCT_TYPE] == "OTHER"][self.cls_fields.NC_LDP_CONTRACT_TYPE].unique().tolist()
            msg_err = "Il y a un problème avec certains contrats dont le type actif/passif nest pas reconnu : %s" % prob_contrats
            logger.error(msg_err)
            raise ValueError(msg_err)
        return data

    def replace_hb_ns_contracts(self, data_ldp):
        cond = ((data_ldp[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:3] == "AHB")
                | (data_ldp[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:3] == "PHB"))

        data_ldp.loc[cond, self.cls_fields.NC_LDP_CONTRACT_TYPE] = data_ldp.loc[cond, self.cls_fields.NC_LDP_CONTRACT_TYPE].str[1:]

        return data_ldp

    def generate_periode_cap(self, data, col_capi):
        mensuel = data[col_capi].isin(["M"])
        annuel = data[col_capi].isin(["A"])
        trimestriel = data[col_capi].isin(["Q"])
        semestriel = data[col_capi].isin(["S"])

        data[self.cls_fields.NC_FREQ_CAP] = \
            np.select([mensuel, annuel, trimestriel, semestriel], [1, 12, 3, 6], 0)

        return data

    def generate_freq_fixing_periodicity(self, data, col_fixing_periodicity, col_periodicity, name_product=""):
        periodicity = np.select([data[col_periodicity] == x for x in ["M", "Q", "N", "A", "S"]],
                                ["1M", "3M", "1M", "1Y", "6M"], default="1M")
        not_null_fixing_per = data[col_fixing_periodicity].isnull().values
        fix_per_vals = data[col_fixing_periodicity].values
        if name_product not in ["nmd_st", "nmd_pn"]:
                data[col_fixing_periodicity] = np.where(not_null_fixing_per, periodicity, fix_per_vals)
        else:
            data[col_fixing_periodicity] = np.where(not_null_fixing_per, "1M", fix_per_vals)

        data[col_fixing_periodicity] = data[col_fixing_periodicity].replace(self.map_faulty_periodicities)
        is_year = (data[col_fixing_periodicity].astype(str).str[-1:] == "Y").values
        not_month = (data[col_fixing_periodicity].astype(str).str[-1:] != "M").values
        fix_per = data[col_fixing_periodicity].astype(str).str[:-1].astype(int).values
        nb_freq = ne.evaluate("where(not_month & (~is_year), 1, fix_per)")
        data[self.cls_fields.NC_FIXING_PERIODICITY_NUM] = ne.evaluate("where(is_year, 12 * nb_freq, nb_freq)")
        return data

    def generate_freq_curve_tenor(self, data, col_freq_tenor, col_fixing_periodicity):
        #data[col_freq_tenor] = data[col_freq_tenor].replace("7D", "1W")
        cond_not_null = data[col_freq_tenor].isnull().values
        fix_per = data[col_fixing_periodicity].values
        freq_ten = data[col_freq_tenor].values
        data[col_freq_tenor] = np.where(cond_not_null, fix_per, freq_ten)
        is_year = (data[col_freq_tenor].str[-1:] == "Y").values
        not_month = (data[col_freq_tenor].str[-1:] != "M").values
        freq_ten_vals = data[col_freq_tenor].str[:-1].fillna(1).astype(int)
        nb_freq = ne.evaluate("where(~(not_month & (~is_year)), freq_ten_vals, 1)")
        data[self.cls_fields.NC_TENOR_NUM] = ne.evaluate("where(is_year, 12 * nb_freq, nb_freq)").astype(int)
        return data

    def generate_freq_curve_name(self, data, col_curve_name, col_rate_category):
        not_null_cond = (data[col_curve_name].isnull()).values
        rate_cat_cond = (data[col_rate_category] == "FLOATING").values
        rate_cat_vals = data[col_rate_category].values
        fixed = 'FIXED'
        data[col_rate_category] = np.where(not_null_cond & rate_cat_cond, fixed, rate_cat_vals)
        return data

    def format_capitalization_cols(self, data_ldp_hab):
        data_ldp_hab[self.cls_fields.NC_LDP_CAPITALIZATION_RATE] = data_ldp_hab[self.cls_fields.NC_LDP_CAPITALIZATION_RATE] \
            .astype(str).str.replace(",", ".").replace("-", "0").astype(np.float64).fillna(0).astype(str).replace("", 0) \
            .astype(np.float64)
        cond_capi = (data_ldp_hab[self.cls_fields.NC_LDP_CAPI_MODE] == "T").values
        data_ldp_hab[self.cls_fields.NC_IS_CAPI_BEFORE_AMOR] = ne.evaluate("where(cond_capi, True, False)")

        return data_ldp_hab

    def format_base_calc(self, data_hab, col_accrual_basis, col_nb_days_an):
        data_hab[col_accrual_basis]\
            = data_hab[col_accrual_basis].fillna("ACT/360").astype(str).str.upper()
        base_calc = data_hab[col_accrual_basis].replace("", "ACT/360")
        cases = [base_calc == val for val in ["ACT/365", "ACT/360", "30E/360", "ACT/ACT", "30/360", "30A/360"]]
        data_hab[col_nb_days_an] = np.select(cases, [365, 360, 360, 365, 360, 360], default=360)
        return data_hab

    def format_broken_period(self, data_hab):
        vals = ["SS", "SL", "ES", "EL"]
        cases = [data_hab[self.cls_fields.NC_LDP_BROKEN_PERIOD].astype(str).str.upper() == cas for cas in
                 ["START SHORT", "START LONG", "END SHORT", "END LONG"]]
        data_hab[self.cls_fields.NC_LDP_BROKEN_PERIOD] = np.select(cases, vals, default="SS")
        return data_hab

    def format_calc_convention(self, data):
        data[self.cls_fields.NC_LDP_CALC_DAY_CONVENTION] = data[self.cls_fields.NC_LDP_CALC_DAY_CONVENTION].astype(str).fillna("1").replace("nan", "1").replace("None", "1").astype(float).astype(int)
        return data

    def nb_months_in_date(self, datum):
        return 12 * datum.year + datum.month

    ###########@profile
    def nb_months_in_date_pd(self, col_date, def_value=22801, format="%d/%m/%Y"):
        col_date = pd.to_datetime(col_date, format=format).dt
        return np.array([x if x != def_value else 0 for x in 12 * col_date.year + col_date.month])

    ###########@profile
    def transform_date_in_num(self, data, col_date):
        data[col_date] = self.nb_months_in_date_pd(data[col_date], format="%d/%m/%Y")
        return data

    def upper_columns_names(self, data):
        data.columns = [str(x).upper().strip() for x in data.columns.tolist()]
        return data

    def rename_cols(self, data):
        data.columns = [x.replace("ï»¿","").replace("ï»¿".upper(),"") for x in data.columns.tolist()]
        data =  data.rename(columns = {"palier": self.cls_fields.NC_LDP_PALIER})
        return data


