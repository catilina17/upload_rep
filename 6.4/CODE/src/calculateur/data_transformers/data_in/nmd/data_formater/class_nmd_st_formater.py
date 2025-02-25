import numpy as np
import pandas as pd
from calculateur.data_transformers import commons as com
from .class_nmd_model_mapper import Data_NMD_MODEL_MAPPER
from mappings.pass_alm_fields import PASS_ALM_Fields

import logging

logger = logging.getLogger(__name__)


class Data_NMD_ST_Formater():
    """
    Formate les données
    """

    def __init__(self, cls_fields, cls_format, source_data, dar_usr, model_wb, horizon,
                 filters=[]):
        self.dar_usr = dar_usr
        self.cls_fields = cls_fields
        self.cls_format = cls_format
        self.load_columns_names()
        self.load_default_vars()
        self.load_mappings()
        self.source_data = source_data
        self.model_wb = model_wb
        self.horizon = horizon
        self.filters = filters
        self.cls_pa_fields = PASS_ALM_Fields()
        self.model_mapper = Data_NMD_MODEL_MAPPER(cls_fields, cls_format, self.cls_pa_fields, source_data, dar_usr,
                                                  "STOCK", model_wb, horizon)

    def load_mappings(self):
        self.map_reset_periodicity = {"M": "1M", "Q": "3M", "A": "1Y", "S": "6M"}

    def load_columns_names(self):
        self.NC_NMD_ST_RESET_FREQUENCY = "RESET_FREQUENCY"
        self.NC_NMD_ST_UNIT_OUTSTANDING = "unit_outstanding_eur".upper()
        self.NC_NMD_ST_RESET_FREQUENCY_TENOR = "reset_freq_tenor".upper()
        self.NC_NMD_ST_TENOR_BASED_FREQ =  "tenor_based_frequency".upper()
        self.NC_NMD_ST_FIRST_FIXING_DATE = "FIRST_FIXING_DATE"
        self.MATURITY_ESTIMATION = "MATURITY_ESTIMATION"

    def load_default_vars(self):
        self.default_ldp_vars = \
            {self.cls_fields.NC_LDP_CONTRAT: '', self.cls_fields.NC_LDP_VALUE_DATE: '.',
             self.cls_fields.NC_LDP_TRADE_DATE: '.',
             self.cls_fields.NC_LDP_FIRST_AMORT_DATE: ".", self.cls_fields.NC_LDP_MATUR_DATE: "",
             self.cls_fields.NC_LDP_RELEASING_DATE: ".", self.cls_fields.NC_LDP_RELEASING_RULE: np.nan,
             self.cls_fields.NC_LDP_RATE: 0, self.cls_fields.NC_LDP_TYPE_AMOR: "F",
             self.cls_fields.NC_LDP_FREQ_AMOR: "None",
             self.cls_fields.NC_LDP_FREQ_INT: "M", self.cls_fields.NC_LDP_ECHEANCE_VAL: np.nan,
             self.cls_fields.NC_LDP_NOMINAL: 0,
             self.cls_fields.NC_LDP_OUTSTANDING: 0,
             self.cls_fields.NC_LDP_CAPITALIZATION_PERIOD: np.nan, self.cls_fields.NC_LDP_CAPITALIZATION_RATE: np.nan,
             self.cls_fields.NC_LDP_ACCRUAL_BASIS: "30/360", self.cls_fields.NC_LDP_CURRENCY: "EUR",
             self.cls_fields.NC_LDP_BROKEN_PERIOD: "Start Short", self.cls_fields.NC_LDP_ETAB: "",
             self.cls_fields.NC_LDP_INTERESTS_ACCRUALS: 0, self.cls_fields.NC_LDP_CONTRACT_TYPE: "",
             self.cls_fields.NC_LDP_CURVE_NAME: "EURIBOR",
             self.cls_fields.NC_LDP_TENOR: "3M", self.cls_fields.NC_LDP_MKT_SPREAD: 0,
             self.cls_fields.NC_LDP_FIXING_NEXT_DATE: ".",
             self.cls_fields.NC_LDP_CALC_DAY_CONVENTION: 1,
             self.cls_fields.NC_LDP_FIXING_PERIODICITY: "1M", self.cls_fields.NC_LDP_RATE_CODE: "EUREURIB3M",
             self.cls_fields.NC_LDP_CAP_STRIKE: np.nan, self.cls_fields.NC_LDP_FLOOR_STRIKE: np.nan,
             self.cls_fields.NC_LDP_MULT_SPREAD: 1, self.cls_fields.NC_LDP_RATE_TYPE: "FIXED",
             self.cls_fields.NC_LDP_MARCHE: "",
             self.cls_fields.NC_LDP_FTP_RATE: np.nan, self.cls_fields.NC_LDP_CURRENT_RATE: np.nan,
             self.cls_fields.NC_LDP_CAPI_MODE: "P",
             self.cls_fields.NC_LDP_BUY_SELL: "",
             self.cls_fields.NC_LDP_PERFORMING: "F",
             self.cls_fields.NC_LDP_MATUR: "",
             self.cls_fields.NC_LDP_IS_CAP_FLOOR: "", self.cls_fields.NC_LDP_DATE_SORTIE_GAP: ".",
             self.cls_fields.NC_LDP_FIXING_NB_DAYS: 0, self.cls_fields.NC_LDP_IS_FUTURE_PRODUCT: False,
             self.cls_fields.NC_PRICING_CURVE: "", self.cls_fields.NC_LDP_GESTION: "",
             self.cls_fields.NC_LDP_PALIER: "",
             self.cls_fields.NC_LDP_FLOW_MODEL_NMD: "", self.cls_fields.NC_LDP_RM_GROUP: "",
             self.cls_fields.NC_LDP_RM_GROUP_PRCT: 0, self.cls_fields.NC_LDP_FIXING_RULE: "B",
             self.cls_fields.NC_LDP_FIRST_COUPON_DATE: ".", self.cls_fields.NC_LDP_TARGET_RATE:np.nan,
             self.cls_fields.NC_LDP_FTP_FUNDING_SPREAD: 0}

    def load_calculated_columns(self, data_nmd):
        # pas de fixing_periodicity dans les NMDs
        data_nmd[self.NC_NMD_ST_RESET_FREQUENCY] = data_nmd[self.NC_NMD_ST_RESET_FREQUENCY].map(self.map_reset_periodicity)

        cases_freq = [(data_nmd[self.NC_NMD_ST_TENOR_BASED_FREQ].values != "T"),
                      (data_nmd[self.NC_NMD_ST_TENOR_BASED_FREQ].values == "T")]

        vals_freq = [data_nmd[self.NC_NMD_ST_RESET_FREQUENCY], data_nmd[self.NC_NMD_ST_RESET_FREQUENCY_TENOR]]

        data_nmd[self.cls_fields.NC_LDP_FIXING_PERIODICITY] = np.select(cases_freq, vals_freq)

        data_nmd[self.cls_fields.NC_LDP_FIXING_PERIODICITY] =\
            data_nmd[self.cls_fields.NC_LDP_FIXING_PERIODICITY].mask(data_nmd[self.cls_fields.NC_LDP_FIXING_RULE] == "R",
                                                                 "1M")

        data_nmd[self.cls_fields.NC_LDP_FIXING_NEXT_DATE] =\
            data_nmd[self.cls_fields.NC_LDP_FIXING_NEXT_DATE].mask(data_nmd[self.cls_fields.NC_LDP_FIXING_RULE] == "B",
                                                               data_nmd[self.NC_NMD_ST_FIRST_FIXING_DATE])

        data_nmd[self.cls_fields.NC_LDP_INTERESTS_ACCRUALS] = (data_nmd[self.cls_fields.NC_LDP_INTERESTS_ACCRUALS].values
                                                          /data_nmd[self.cls_fields.NC_LDP_NB_CONTRACTS].values)

        return data_nmd

    def force_pel_columns(self, data_nmd):
        is_pel =  (data_nmd[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("P-PEL")).values
        nb_contracts_pel = np.abs(np.nan_to_num(data_nmd[self.cls_fields.NC_LDP_OUTSTANDING].values
                                                         / data_nmd[self.NC_NMD_ST_UNIT_OUTSTANDING].values))

        data_nmd[self.cls_fields.NC_LDP_NB_CONTRACTS] = np.where(is_pel, nb_contracts_pel, 1.0)
        is_floating =  (data_nmd[self.cls_fields.NC_LDP_RATE_TYPE] == "FLOATING").values
        data_nmd.loc[is_pel, self.cls_fields.NC_LDP_OUTSTANDING] = data_nmd.loc[is_pel,self.NC_NMD_ST_UNIT_OUTSTANDING].values
        data_nmd.loc[is_pel, self.cls_fields.NC_LDP_CAPITALIZATION_RATE] = 1.0

        data_nmd.loc[is_pel & is_floating, self.cls_fields.NC_LDP_FIXING_RULE] = "B"

        return data_nmd

    def create_nb_parts_groups(self, data_nmd_rm):
        self.dic_data_nmd = {}
        for nb_part in data_nmd_rm["NB_PARTS"].unique():
            for is_volatile in data_nmd_rm[data_nmd_rm["NB_PARTS"] == nb_part]["HAS_VOLATILE"].unique():
                filter_not_icne = ~data_nmd_rm[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("ICNE")
                filter_part = (data_nmd_rm["NB_PARTS"] == nb_part) & (data_nmd_rm["HAS_VOLATILE"] == is_volatile)
                self.dic_data_nmd[str(int(is_volatile)) + "_" + str(int(nb_part))] = data_nmd_rm[filter_part & filter_not_icne].copy()

        filter_icne = data_nmd_rm[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("ICNE")
        self.dic_data_nmd[str(1) + "_" + str(int(1)) + "_" + "ICNE"] = data_nmd_rm[filter_icne].copy()

    def apply_filters(self, data_nmd_st):
        for filter in self.filters:
            col_filter, val_filter = filter
            data_nmd_st = data_nmd_st[data_nmd_st[col_filter] == val_filter].copy()
        return data_nmd_st

    def create_absent_vars(self, data_nmd):
        if not self.cls_fields.NC_LDP_BASSIN in data_nmd.columns:
            data_nmd[self.cls_fields.NC_LDP_BASSIN] = "*"

        if not self.cls_fields.NC_LDP_INTERESTS_ACCRUALS in data_nmd.columns:
            data_nmd[self.cls_fields.NC_LDP_INTERESTS_ACCRUALS] = 0

        if not self.cls_fields.NC_LDP_CONTRAT in data_nmd.columns:
            data_nmd[self.cls_fields.NC_LDP_CONTRAT] = "C_" + pd.Series(np.arange(1, len(data_nmd) + 1)).astype(str)

        return data_nmd

    def filter_data(self, data_nmd_st):
        #data_nmd_st = data_nmd_st[data_nmd_st[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("P-PEL")].copy()
        #data_nmd_st = data_nmd_st[data_nmd_st[self.cls_fields.NC_LDP_ETAB].isin(["BRED"])].copy()
        #data_nmd_st = data_nmd_st[data_nmd_st[self.cls_fields.NC_LDP_MARCHE].str.contains("PRO")].copy()
        #data_nmd_st = data_nmd_st[data_nmd_st[self.cls_fields.NC_LDP_RATE_TYPE].str.contains("FLOATING")].copy()
        #data_nmd_st = data_nmd_st[data_nmd_st[self.cls_fields.NC_LDP_CONTRACT_TYPE].isin(["P-PEL-2011"])].copy()
        #data_nmd_st = data_nmd_st[data_nmd_st[self.cls_fields.NC_LDP_ETAB].isin(["CEGEE"])].copy()
        """data_nmd_st = data_nmd_st[data_nmd_st[self.cls_fields.NC_LDP_CONTRAT]
        .isin(["A-2024-03-31-10907-P-PEL-2015-T0-SUP-185", "A-2024-03-31-10907-P-PEL-2015-T0-SUP-191",
               "A-2024-03-31-10907-P-PEL-2015-T0-SUP-172"])].copy()"""
        return data_nmd_st
    def read_file_and_standardize_data(self):
        logging.debug('    Lecture du fichier NMD ST')
        data_nmd_st = com.read_file(self.source_data, "LDP")
        data_nmd_st = self.cls_format.upper_columns_names(data_nmd_st)
        data_nmd_st = self.create_absent_vars(data_nmd_st)
        data_nmd_st = self.filter_data(data_nmd_st)
        data_nmd_st = self.apply_filters(data_nmd_st)
        data_nmd_st = self.force_pel_columns(data_nmd_st)
        data_nmd_st = self.load_calculated_columns(data_nmd_st)
        data_nmd_st = self.model_mapper.map_data_with_model_maps(data_nmd_st)
        data_nmd_st = self.cls_format.create_unvailable_variables(data_nmd_st, self.cls_fields.ldp_vars,
                                                                  self.default_ldp_vars)
        self.create_nb_parts_groups(data_nmd_st)
