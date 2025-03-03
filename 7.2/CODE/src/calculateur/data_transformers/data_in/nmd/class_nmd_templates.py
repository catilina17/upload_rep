import pandas as pd
from utils import excel_openpyxl as ex
from utils import general_utils as ut
import numpy as np

import logging

logger = logging.getLogger(__name__)


class Data_NMD_TEMPLATES():

    ALLOCATION_KEY = "RCO_ALLOCATION_KEY"
    TEMPLATE_WEIGHT_RCO = "TEMPLATE_WEIGHT_RCO"
    TEMPLATE_WEIGHT_REAL = "TEMPLATE_WEIGHT_REAL"

    def __init__(self, source_data, model_wb, bassin, dar, cls_fields, cls_format):
        self.cls_fields = cls_fields
        self.load_allocation_file_columns()
        self.source_data = source_data
        self.model_wb = model_wb
        self.bassin = bassin
        self.dar = dar
        self.cls_format = cls_format
        self.read_allocation_set_file()
        self.cols_key_allocation_set = [self.cls_fields.NC_LDP_CURRENCY, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                                   self.cls_fields.NC_LDP_MARCHE, self.cls_fields.NC_LDP_RATE_TYPE]
        self.default_set = "ACCOUNT"

    def load_allocation_file_columns(self):
        self.NC_ALC_SET_NAME = "SET_NAME"
        self.NC_ALS_RATE_CATEGORY = "RATE_CATEGORY"
        self.NC_ALS_FAMILY = "FAMILY"
        self.NC_ALS_CURRENCY = "CURRENCY"
        self.NC_ALS_CONTRACT_TYPE = "CONTRACT_TYPE"
        self.NG_ALLOCATION_SET = "_ALLOCATION_SET"

        self.NC_ALV_SET_NAME = "SET_NAME"
        self.NC_ALV_PALIER = "COUNTERPARTY_CODE"
        self.NC_ALV_CURRENCY = "CURRENCY"
        self.NC_ALV_ACCRUAL_BASIS = "ACCRUAL_BASIS"
        self.NC_ALV_CALC_DAY_CONVENTION = "CALC_DAY_CONVENTION"
        self.NC_ALV_CAPITALIZATION_RATE = "CAPITALIZATION_RATE"
        self.NC_ALV_FAMILY = "FAMILY"
        self.NC_ALV_FIRST_COUPON_DATE = "FIRST_COUPON_DATE"
        self.NC_ALV_RATE_VALUE = "RATE_VALUE"
        self.NC_ALV_RATE_CATEGORY = "RATE_CATEGORY"
        self.NC_ALV_PERIODICITY = "PERIODICITY"
        self.NC_ALV_RATE_CODE = "RATE_CODE"
        self.NC_ALV_UNITARY_OUTSTANDING = "UNIT_OUTSTANDING_EUR"

        self.NC_ALV_ACCRUAL_BASIS_AG_TYPE = "ACCRUAL_BASIS_"
        self.NC_ALV_CALC_DAY_CONVENTION_AG_TYPE = "CALC_DAY_CONVENTION_"
        self.NC_ALV_CAPITALIZATION_RATE_AG_TYPE = "CAPITALIZATION_RATE_"
        self.NC_ALV_CONTRACT_TYPE_AG_TYPE = "CONTRACT_TYPE_"
        self.NC_ALV_PALIER_AG_TYPE = "COUNTERPARTY_CODE_"
        self.NC_ALV_CURRENCY_AG_TYPE = "CURRENCY_"
        self.NC_ALV_CURVE_NAME_AG_TYPE = "CURVE_NAME_"
        self.NC_ALV_RATE_CODE_AG_TYPE = "RATE_CODE_"
        self.NC_ALV_FAMILY_AG_TYPE = "FAMILY_"
        self.NC_ALV_FIRST_COUPON_DATE_AG_TYPE = "FIRST_COUPON_DATE_"
        self.NC_ALV_FIRST_FIXING_DATE_AG_TYPE = "FIRST_FIXING_DATE_"
        self.NC_ALV_FIXING_NEXT_DATE_AG_TYPE = "FIXING_NEXT_DATE_"
        self.NC_ALV_FIXING_RULE_AG_TYPE = "FIXING_RULE_"
        self.NC_ALVRATE_VALUE_AG_TYPE = "RATE_VALUE_"
        self.NC_ALV_MKT_SPREAD_AG_TYPE = "MKT_SPREAD_"
        self.NC_ALV_MULT_SPREAD_AG_TYPE = "MULT_SPREAD_"
        self.NC_ALV_PERIODICITY_AG_TYPE = "PERIODICITY_"
        self.NC_ALV_RATE_CATEGORY_AG_TYPE = "RATE_CATEGORY_"
        self.NC_ALV_TENOR_AG_TYPE = "TENOR_"
        self.NC_ALV_FLOOR_AG_TYPE = "FLOOR_STRIKE_"
        self.NC_ALV_IRR_POSITION_DATE_AG_TYPE = "IRR_POSITION_DATE_"
        self.NC_ALV_COMPOUND_PERIODICTY_AG_TYPE = "COMPOUND_PERIODICITY_"
        self.NC_ALV_RESET_FREQUENCY_AG_TYPE = "RESET_FREQUENCY_"
        self.NC_ALV_TENOR_BASED_FREQ_AG_TYPE = "tenor_based_frequency_".upper()
        self.NC_ALV_RESET_FREQUENCY_TENOR_AG_TYPE = "reset_freq_tenor_".upper()
        self.NC_ALV_UNITARY_OUTSTANDING_AG_TYPE  = "UNIT_OUTSTANDING_EUR_"
        self.NC_ALV_TRADE_DATE_AG_TYPE  = "TRADE_DATE_"

        self.NG_ALLOCATION_VALS = "_ALLOCATION_VALS"

        self.AG_TYPE_VARS = [y for x,y in self.__dict__.items() if "AG_TYPE" in x]


    def read_allocation_set_file(self):
        logging.debug('   Lecture du fichier NMD ST')
        self.allocation_set = ex.get_dataframe_from_range(self.model_wb, self.NG_ALLOCATION_SET)
        keys = [self.NC_ALS_CURRENCY, self.NC_ALS_CONTRACT_TYPE, self.NC_ALS_FAMILY, self.NC_ALS_RATE_CATEGORY]
        self.allocation_set ["new_key"]  = self.allocation_set [keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        self.allocation_set = self.allocation_set.set_index("new_key")[[self.NC_ALC_SET_NAME]].copy()
        self.allocation_val = ex.get_dataframe_from_range(self.model_wb, self.NG_ALLOCATION_VALS)
        self.parse_allocation_vals()

    def parse_allocation_vals(self):
        allocation_dic = {}
        for set_ in self.allocation_val[self.NC_ALV_SET_NAME].unique():
            allocation_set_val = self.allocation_val[self.allocation_val[self.NC_ALV_SET_NAME] == set_].copy()
            allocation_dic[set_] = {}
            allocation_dic[set_]["GROUP_VARS"] = []
            allocation_dic[set_]["AVERAGE_VARS"] = []
            allocation_dic[set_]["USER_VARS"] = []
            for var in self.AG_TYPE_VARS:
                if allocation_set_val[var].iloc[0] in ["S", "T"]:
                    allocation_dic[set_]["GROUP_VARS"].append(var[:-1])

                elif allocation_set_val[var].iloc[0] in ["A"]:
                    allocation_dic[set_]["AVERAGE_VARS"].append(var[:-1])

                elif allocation_set_val[var].iloc[0] in ["U"]:
                    allocation_dic[set_]["USER_VARS"].append((var[:-1], allocation_set_val[var[:-1]].iloc[0]))

        self.allocation_dic = allocation_dic


    def generate_templates(self, is_stock=True, data_nmd_all = [], report_vars = [], rco_allocation_key=True):
        if is_stock:
            if not "DATA" in self.source_data["LDP"]:
                data_nmd_all = pd.read_csv(self.source_data["LDP"]["CHEMIN"],
                                           delimiter=self.source_data["LDP"]["DELIMITER"],
                                           decimal=self.source_data["LDP"]["DECIMAL"],
                                          engine='python', encoding="ISO-8859-1", chunksize=100000)
            else:
                data_nmd_all = self.source_data["LDP"]["DATA"]
                data_nmd_all = [data_nmd_all.iloc[i:i + 100000].copy() for i in range(0, len(data_nmd_all), 100000)]
        else:
            data_nmd_all[self.cls_fields.NC_LDP_OUTSTANDING] = 1
            data_nmd_all[self.cls_fields.NC_LDP_RATE_CODE] = np.nan
            data_nmd_all[self.cls_fields.NC_LDP_RATE_CODE] \
                = np.where((data_nmd_all[self.cls_fields.NC_LDP_RATE_TYPE] == "FIXED")
                           & data_nmd_all[self.cls_fields.NC_LDP_RATE_CODE].isnull(),
                           "FIXE", data_nmd_all[self.cls_fields.NC_LDP_RATE_CODE])
            data_nmd_all = [data_nmd_all.iloc[i:i + 100000].copy() for i in range(0, len(data_nmd_all), 100000)]

        dic_data_nmd = {}

        for data_nmd in data_nmd_all:
            data_nmd = self.cls_format.upper_columns_names(data_nmd)

            if rco_allocation_key:
                data_nmd = \
                    ut.map_with_combined_key2(data_nmd, self.allocation_set, self.cols_key_allocation_set, symbol_any="*",
                                             filter_comb=True, necessary_cols=1, error=False, drop_key_col=False,
                                             name_key_col = self.ALLOCATION_KEY).copy()

                if data_nmd["SET_NAME"].isnull().any():
                    list_errors = data_nmd.loc[data_nmd["SET_NAME"].isnull(), self.cols_key_allocation_set].drop_duplicates().values.tolist()
                    msg = "Il y a des clés d'allocation manquantes dans les NMDs: %s" % list_errors
                    logger.error(msg)
                    raise ValueError(msg)

                data_nmd[self.ALLOCATION_KEY] = data_nmd[self.cls_fields.NC_LDP_ETAB] + "_" + data_nmd[self.ALLOCATION_KEY]
            else:
                data_nmd[self.ALLOCATION_KEY] = "*"
                data_nmd[self.NC_ALC_SET_NAME] = self.default_set

            for set_ in data_nmd[self.NC_ALC_SET_NAME].unique():
                filter_set = data_nmd[self.NC_ALC_SET_NAME] == set_
                data_nmd_set = data_nmd[filter_set].copy()
                if not set_ in dic_data_nmd:
                    dic_data_nmd[set_] = []
                if True:#not is_stock:
                    for user_var in self.allocation_dic[set_]["USER_VARS"]:
                        if not user_var[0] in data_nmd.columns.tolist():
                            data_nmd_set[user_var[0]] = user_var[1]
                        else:
                            try:
                                if user_var[0] in self.cols_key_allocation_set:
                                    data_nmd_set[user_var[0]] = data_nmd[user_var[0]].fillna(user_var[1]).values
                                else:
                                    data_nmd_set[user_var[0]] = user_var[1]
                            except:
                                data_nmd_set[user_var[0]] = data_nmd_set[user_var[0]].astype(str)
                                if user_var[0] in self.cols_key_allocation_set:
                                    data_nmd_set[user_var[0]] = data_nmd_set[user_var[0]].replace("nan", user_var[1])
                                else:
                                    data_nmd_set[user_var[0]] = user_var[1]

                keep_vars = ([self.ALLOCATION_KEY, self.cls_fields.NC_LDP_ETAB] + self.allocation_dic[set_]["GROUP_VARS"]
                             + self.allocation_dic[set_]["AVERAGE_VARS"] + [self.cls_fields.NC_LDP_OUTSTANDING] + report_vars)
                use_vars = [x[0] for x in self.allocation_dic[set_]["USER_VARS"]] #JUSTE LES NOMS MAIS PAS LES VALEURS
                keep_vars = (keep_vars + use_vars)

                keep_vars = [x for x in keep_vars if x in data_nmd_set.columns]

                data_nmd_set = data_nmd_set[ keep_vars].copy()

                if len(dic_data_nmd[set_]) > 0:
                    dic_data_nmd[set_] =  pd.concat([dic_data_nmd[set_], data_nmd_set])
                else:
                    dic_data_nmd[set_] = data_nmd_set

        data_template = []

        for set_ in dic_data_nmd:
            key_vars = [self.ALLOCATION_KEY, self.cls_fields.NC_LDP_ETAB]
            dic_data_nmd[set_][self.cls_fields.NC_LDP_OUTSTANDING + "_ABS"] = np.absolute(dic_data_nmd[set_][self.cls_fields.NC_LDP_OUTSTANDING].values)
            grouped_vars = self.allocation_dic[set_]["GROUP_VARS"] + report_vars
            use_vars = [x[0] for x in self.allocation_dic[set_]["USER_VARS"]] #JUSTE LES NOMS MAIS PAS LES VALEURS
            grouped_vars = (grouped_vars + use_vars)
            average_on_vars = self.allocation_dic[set_]["AVERAGE_VARS"]
            grouped_vars = [x for x in grouped_vars if x in dic_data_nmd[set_].columns]
            average_on_vars = [x for x in average_on_vars if x in dic_data_nmd[set_].columns]
            sum_on_vars = [self.cls_fields.NC_LDP_OUTSTANDING, self.cls_fields.NC_LDP_OUTSTANDING + "_ABS"]
            #wm = lambda x: np.average(x, weights=dic_data_nmd[set_].loc[x.index, self.cls_fields.NC_LDP_OUTSTANDING])
            def wm(x):
                try:
                    return np.average(x, weights=dic_data_nmd[set_].loc[x.index, self.cls_fields.NC_LDP_OUTSTANDING])
                except ZeroDivisionError:
                    return np.average(x)
            dict_sum = {k: "sum" for k in sum_on_vars}
            dict_avg = {k: wm for k in average_on_vars}
            dict_agg = {**dict_sum, **dict_avg}
            dic_data_nmd[set_] = dic_data_nmd[set_].groupby(key_vars + grouped_vars, dropna=False, as_index=False).agg(dict_agg)
            dic_data_nmd[set_][self.TEMPLATE_WEIGHT_REAL] = dic_data_nmd[set_].groupby(key_vars)[self.cls_fields.NC_LDP_OUTSTANDING].transform(lambda x: x / x.sum())
            dic_data_nmd[set_][self.TEMPLATE_WEIGHT_RCO] = dic_data_nmd[set_].groupby(key_vars)[self.cls_fields.NC_LDP_OUTSTANDING + "_ABS"].transform(lambda x: abs(x) / abs(x).sum())
            data_template.append(dic_data_nmd[set_])

        return pd.concat(data_template)


    def get_templates(self, data_nmd_pn, rco_allocation_key=True):
        logging.debug('   Génération des templates pour les PN NMDs')

        data_template_st = self.generate_templates(report_vars=[self.cls_fields.NC_LDP_DIM6],
                                                   rco_allocation_key= rco_allocation_key)
        if len(data_nmd_pn) > 0:
            data_template_pn = self.get_template_pn(data_nmd_pn, data_template_st, rco_allocation_key)
            if len(data_template_pn) > 0:
                data_template = pd.concat([data_template_st, data_template_pn])
            else:
                data_template = data_template_st.copy()
        else:
            data_template = data_template_st.copy()

        return data_template

    def get_template_pn(self, data_nmd_pn, data_template_st, rco_allocation_key):
        data_nmd_pn = data_nmd_pn[~data_nmd_pn.set_index(self.cols_key_allocation_set)
        .index.isin(data_template_st.set_index(self.cols_key_allocation_set).index.values.tolist())].copy()

        if len(data_nmd_pn) > 0:
            data_template_pn = self.generate_templates(is_stock=False, data_nmd_all=data_nmd_pn,
                                                       report_vars=[self.cls_fields.NC_LDP_DIM6],
                                                       rco_allocation_key = rco_allocation_key)
            data_template_pn\
                = data_template_pn[~data_template_pn[self.ALLOCATION_KEY].isin(data_template_st[self.ALLOCATION_KEY].values.tolist())]
        else:
            data_template_pn = []

        return data_template_pn






