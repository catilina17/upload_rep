import numpy as np
from calculateur.data_transformers import commons as com
import logging
from dateutil.relativedelta import relativedelta
import pandas as pd
from calculateur.calc_params import model_params as mod

logger = logging.getLogger(__name__)


class Data_ECH_STOCK():
    """
    Formate les données
    """

    def __init__(self, cls_fields, cls_format, cls_pa_fields, source_data, dar_usr, batch_size,
                 name_product, horizon, exec_mode="simul"):
        self.dar_usr = dar_usr
        self.exec_mode = exec_mode
        self.proj_horizon = horizon
        self.cls_fields = cls_fields
        self.cls_format = cls_format
        self.cls_pa_fields = cls_pa_fields
        self.load_columns_names_titres()
        self.source_data = source_data
        self.batch_size = batch_size
        self.name_product = name_product
        self.default_na = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                           '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']
        self.titres = ["a-security-tf", "a-security-tv", "p-security-tf", "p-security-tv"]
        self.swaps_change = ["a-change-tf", "a-change-tv", "p-change-tf", "p-change-tv"]
        self.swaps = ["a-swap-tf", "a-swap-tv", "p-swap-tf", "p-swap-tv"]
        self.action_contract = "A-ACTION"
        self.load_correspondances_titres()
        self.load_columns_names_cap_floor()
        self.load_repos_columns()

    def load_columns_names_titres(self):
        self.NC_UNITY_NOMINAL = "unity_nominal_eur".upper()
        self.NC_UNITY_OUTSTANDING = "unity_outstanding_eur".upper()
        self.NC_NB_SECURITIES = "number_of_securities".upper()
        self.NC_SECURITY_REFERENCE = "security_reference".upper()
        self.CURRENT_PREMIUM = "current_premium".upper()
        self.LONG_SHORT = "long_short".upper()
        self.PALIER = "palier".upper()

    def load_repos_columns(self):
        self.NC_FIXING_PERIODICITY_TENOR = "Fixing_Period_Tenor".upper()
        self.NC_INTEREST_PERIODICITY_TENOR = "Interest_Period_Tenor".upper()
        self.NC_FIXING_NEXT_DATE = "next_fixing_date".upper()
        self.NC_FIXING_RULE = "cash_fixing_rule".upper()

    def load_columns_names_cap_floor(self):
        self.PALIER_CAP_FLOOR = "palier".upper()

    def load_correspondances_titres(self):
        self.correspondances = {}
        self.correspondances[self.NC_NB_SECURITIES] = self.cls_fields.NC_LDP_NB_CONTRACTS
        self.correspondances[self.LONG_SHORT] = self.cls_fields.NC_LDP_BUY_SELL
        self.correspondances[self.PALIER] = self.cls_fields.NC_LDP_PALIER

    def load_cf_data(self):
        if "CF" in self.source_data:
            com.check_file_existence(self.source_data["CF"]["CHEMIN"])
            data_cf = com.read_file(self.source_data, "CF")
        else:
            data_cf = []

        return data_cf

    def load_pal_data(self):
        if "PAL" in self.source_data:
            com.check_file_existence(self.source_data["PAL"]["CHEMIN"])
            data_pal = com.read_file(self.source_data, "PAL")
        else:
            data_pal = []

        return data_pal

    def add_contract_ref_to_pal_file(self, data_pal, data_ldp):
        if data_pal[self.cls_fields.NC_PAL_CONTRAT].isin(data_ldp[self.NC_SECURITY_REFERENCE].unique().tolist()).any():
            data_pal = data_pal.join(data_ldp.set_index(self.NC_SECURITY_REFERENCE)[[self.cls_fields.NC_LDP_CONTRAT]],
                                     on=self.cls_fields.NC_PAL_CONTRAT)

            data_pal = (data_pal.drop([self.cls_fields.NC_PAL_CONTRAT], axis=1)
                        .rename(columns={self.cls_fields.NC_LDP_CONTRAT: self.cls_fields.NC_PAL_CONTRAT}))

        return data_pal

    def load_ldp_data(self):
        com.check_file_existence(self.source_data["LDP"]["CHEMIN"])
        data_ldp = com.read_file(self.source_data, "LDP")
        return data_ldp

    def calculate_necessary_columns_titres(self, data_ldp):

        coeff = np.where(data_ldp[self.LONG_SHORT] == "S", -1, 1)

        data_ldp[self.cls_fields.NC_LDP_NOMINAL] \
            = coeff * abs(data_ldp[self.NC_UNITY_NOMINAL].values)

        data_ldp[self.cls_fields.NC_LDP_OUTSTANDING] \
            = coeff * abs(data_ldp[self.NC_UNITY_OUTSTANDING].values)

        data_ldp[self.NC_NB_SECURITIES] = abs(data_ldp[self.NC_NB_SECURITIES].values)

        data_ldp = self.format_actions(data_ldp)

        return data_ldp

    def format_actions(self, data_ldp):
        is_action = data_ldp[self.cls_fields.NC_LDP_CONTRACT_TYPE] == self.action_contract
        data_ldp.loc[is_action, self.cls_fields.NC_LDP_NOMINAL] = data_ldp.loc[
            is_action, self.cls_fields.NC_LDP_OUTSTANDING]

        data_ldp.loc[is_action, self.cls_fields.NC_LDP_MATUR_DATE] \
            = (self.dar_usr + relativedelta(months=self.proj_horizon + 5)).strftime("%d/%m/%Y")

        data_ldp.loc[is_action, self.cls_fields.NC_LDP_TYPE_AMOR] = "F"

        return data_ldp

    def calculate_necessary_columns_change(self, data_ldp):
        data_ldp[self.cls_fields.NC_LDP_TYPE_AMOR] = "F"
        return data_ldp

    def add_surcote_decote_line(self, data_ldp):
        data_ldp = data_ldp.reset_index(drop=True).loc[np.repeat(data_ldp.index.values, 2)].copy().reset_index(
            drop=True)
        even_index = data_ldp.index.values[1::2]
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_CONTRAT] = data_ldp.loc[
                                                                       even_index, self.cls_fields.NC_LDP_CONTRAT] + "_PREMIUM"
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_TYPE_AMOR] = "L"
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_FIRST_AMORT_DATE] = "."
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_CAPITALIZATION_RATE] = 0
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_FREQ_AMOR] = "Monthly"
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_FREQ_INT] = "M"
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_INTERESTS_ACCRUALS] = 0
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_RATE] = 0
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_MKT_SPREAD] = 0
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_MULT_SPREAD] = 1
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_ECHEANCE_VAL] = np.nan
        # data_ldp.loc[even_index, self.cls_fields.NC_LDP_VALUE_DATE]\
        #    = self.get_value_date(data_ldp, even_index)
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_CURVE_NAME] = "EURFLAT"
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_TENOR] = "1D"
        coeff = np.where(data_ldp.loc[even_index, self.LONG_SHORT] == "S", -1, 1)
        data_ldp[self.CURRENT_PREMIUM] = np.nan_to_num(data_ldp[self.CURRENT_PREMIUM])
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_NOMINAL] \
            = np.where(data_ldp.loc[even_index, self.NC_NB_SECURITIES] != 0,
                       coeff * data_ldp.loc[even_index, self.CURRENT_PREMIUM].values / data_ldp.loc[
                           even_index, self.NC_NB_SECURITIES].values,
                       0)
        data_ldp.loc[even_index, self.cls_fields.NC_LDP_OUTSTANDING] \
            = data_ldp.loc[even_index, self.cls_fields.NC_LDP_NOMINAL].values
        return data_ldp

    def map_buy_sell_columns(self, data_ldp):
        if self.name_product in self.titres:
            map = {"L": "B"}
        elif self.name_product in self.swaps:
            map = {"PAY": "S", "REC": "B"}
        elif self.name_product in self.swaps_change:
            map = {"PURCHASE": "B", "SALE": "S"}

        data_ldp[self.cls_fields.NC_LDP_BUY_SELL] = data_ldp[self.cls_fields.NC_LDP_BUY_SELL].map(map)

        return data_ldp

    def filter_data(self, data_ldp):
        # data_ldp  = data_ldp[data_ldp[self.cls_fields.NC_LDP_ETAB].isin(["CSDN"])].copy()
        # data_ldp  = data_ldp[data_ldp[self.cls_fields.NC_LDP_MARCHE].isin(["ENTREPRISE"])].copy()
        #data_ldp  = data_ldp [data_ldp[self.cls_fields.NC_LDP_CONTRAT].isin(["KTP000000004415189"])].copy()
        #data_ldp  = data_ldp [data_ldp[self.cls_fields.NC_LDP_CONTRACT_TYPE].isin(["A-CR-HAB-STD"])].copy()
        data_ldp = data_ldp.reset_index(drop=True)
        return data_ldp

    def read_file_and_standardize_data(self):
        logging.debug('    Lecture du fichier STOCK ECH')
        data_ldp = self.load_ldp_data()
        data_ldp = self.cls_format.upper_columns_names(data_ldp)
        if self.exec_mode != "simul":
            data_ldp = self.filter_data(data_ldp)
        self.data_pal = self.load_pal_data()
        self.data_cf = self.load_cf_data()
        if self.name_product in self.titres + self.swaps_change + self.swaps:
            if self.name_product in self.titres:
                data_ldp = self.calculate_necessary_columns_titres(data_ldp)
                self.data_pa = self.cls_format.upper_columns_names(self.data_pal)
                # self.data_pal = self.add_contract_ref_to_pal_file(self.data_pal, data_ldp)
                data_ldp = self.add_surcote_decote_line(data_ldp)
                data_ldp = data_ldp.rename(columns=self.correspondances)

            if self.name_product in self.swaps_change:
                data_ldp = self.calculate_necessary_columns_change(data_ldp)

            self.map_buy_sell_columns(data_ldp)

        elif self.name_product in mod.models_cap_floor:
            data_ldp = data_ldp.rename(columns={self.PALIER_CAP_FLOOR: self.cls_fields.NC_LDP_PALIER})

        elif self.name_product in mod.models_repos:
            data_ldp[self.cls_fields.NC_LDP_OUTSTANDING] = data_ldp[self.cls_fields.NC_LDP_NOMINAL].values
            data_ldp[self.cls_fields.NC_LDP_FIXING_PERIODICITY] = data_ldp[self.NC_FIXING_PERIODICITY_TENOR].values
            data_ldp[self.cls_fields.NC_LDP_FREQ_INT] = np.where(data_ldp[self.NC_INTEREST_PERIODICITY_TENOR] == "99Y",
                                                                 "N", "M")
            data_ldp[self.cls_fields.NC_LDP_FIXING_NEXT_DATE] = data_ldp[self.NC_FIXING_NEXT_DATE].values
            data_ldp[self.cls_fields.NC_LDP_FIXING_RULE] = data_ldp[self.NC_FIXING_RULE].values

        data_ldp = com.chunkized_data(data_ldp, self.batch_size)

        self.data_ldp = data_ldp
