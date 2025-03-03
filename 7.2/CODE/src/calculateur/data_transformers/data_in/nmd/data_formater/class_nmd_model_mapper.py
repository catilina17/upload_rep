import numpy as np
import pandas as pd
from utils import general_utils as ut
from utils import excel_openpyxl as ex
from dateutil.relativedelta import relativedelta

import logging

logger = logging.getLogger(__name__)


class Data_NMD_MODEL_MAPPER():

    def __init__(self, cls_fields, cls_format, cls_pa_fields,
                 source_data, dar_usr, type_data, model_wb, horizon, type_rm="NORMAL"):
        self.dar_usr = dar_usr
        self.cls_fields = cls_fields
        self.cls_format = cls_format
        self.load_nmd_model_map_range_names()
        self.load_nmd_model_map_col_names()
        self.source_data = source_data
        self.type_data = type_data
        self.model_wb = model_wb
        self.horizon = horizon
        self.type_rm = type_rm
        self.cls_pa_fields = cls_pa_fields

    def load_nmd_model_map_range_names(self):
        self.NR_MODEL_MAPPING = "_NMD_MODEL_MAP"
        self.NR_MODEL_MAPPING_FORMULA = "_NMD_MODEL_MAP_FORMULA"
        self.NR_MODEL_ICNE = "_MODELE_ICNE"

    def load_nmd_model_map_col_names(self):
        self.NC_FLOW_TYPE = 'RM_RUNOFF'
        self.NC_NETWORK = 'RESEAU'
        self.NC_BASSIN = 'COMPANY_CODE'
        self.NC_MARCHE = 'FAMILY'
        self.NC_CURRENCY = "CURRENCY"
        self.NC_CONTRACT_TYPE = "CONTRACT_TYPE"
        self.NC_RATE_TYPE = "RATE_TYPE"
        self.NC_STOCK_OR_PN = "DEAL_STATUS"
        self.NC_BALANCE = "BALANCE"
        self.NC_PRCT_BREAKDOWN = "PERCENTAGE"
        self.NC_BREAKDOWN = 'COMPONENT_TYPE'
        self.NC_FIXED_RATE_CODE = 'FIXED_RATE_CODE'
        self.NCMETHOD_KEY = 'METHOD_KEY'
        self.NC_FIXED_TENOR_CODE = 'FIXED_TENOR_CODE'
        self.NC_VARIABLE_CURVE_CODE = 'VARIABLE_CURVE_CODE'
        self.NC_VARIABLE_TENOR_CODE = 'VARIABLE_TENOR_CODE'
        self.NC_USE_STOCK = "USE_STOCK"
        self.NC_DECAY_PERIODICITY = "DECAY_PERIODICITY"
        self.NC_DECAY_TRUNCATION_TERM = "DECAY_TRUNCATION_TERM"
        self.NC_MODEL_NAME_SAVINGS = "MODEL_NAME_SAVINGS"
        self.NC_MODEL_NAME_DRAWDOWNS = "MODEL_NAME_DRAWDOWNS"

        self.NC_TCI_FLOW_MODEL_STOCK = 'FLOW_MODEL_STOCK_TCI'
        self.NC_TCI_FLOW_MODEL_PN = 'FLOW_MODEL_PN_TCI'
        self.NC_TCI_FLOW_MODEL_OPPOSITE_STOCK = "FLOW_MODEL_OPPOSITE_STOCK_TCI"
        self.NC_TCI_FLOW_MODEL_OPPOSITE_PN = "FLOW_MODEL_OPPOSITE_PN_TCI"
        self.tci_flow_columns = [self.NC_TCI_FLOW_MODEL_STOCK, self.NC_TCI_FLOW_MODEL_PN,
                                 self.NC_TCI_FLOW_MODEL_OPPOSITE_STOCK, self.NC_TCI_FLOW_MODEL_OPPOSITE_PN]

        self.NC_FLOW_MODEL_STOCK = 'FLOW_MODEL_STOCK'
        self.NC_FLOW_MODEL_PN = 'FLOW_MODEL_PN'
        self.NC_FLOW_MODEL_OPPOSITE_STOCK = "FLOW_MODEL_OPPOSITE_STOCK"
        self.NC_FLOW_MODEL_OPPOSITE_PN = "FLOW_MODEL_OPPOSITE_PN"
        self.flow_columns = [self.NC_FLOW_MODEL_STOCK, self.NC_FLOW_MODEL_PN,
                             self.NC_FLOW_MODEL_OPPOSITE_STOCK, self.NC_FLOW_MODEL_OPPOSITE_PN]

        self.NC_TCI_FIXED_RATE_CODE = 'FIXED_RATE_CODE'
        self.NC_TCI_METHOD_KEY = 'METHOD_KEY'
        self.NC_TCI_FIXED_TENOR_CODE = 'FIXED_TENOR_CODE'
        self.NC_TCI_VARIABLE_CURVE_CODE = 'VARIABLE_CURVE_CODE'
        self.NC_TCI_VARIABLE_TENOR_CODE = 'VARIABLE_TENOR_CODE'

        self.map_keys = [self.NC_CONTRACT_TYPE, self.NC_MARCHE, self.NC_CURRENCY, self.NC_RATE_TYPE,
                         self.NC_STOCK_OR_PN, self.NC_BALANCE, self.NC_BASSIN, self.NC_NETWORK]

    def parse_model_map(self, map, suff=""):
        map[[self.NC_BREAKDOWN + suff, self.NC_PRCT_BREAKDOWN + suff]] = map[
            [self.NC_BREAKDOWN, self.NC_PRCT_BREAKDOWN]].copy()
        map[self.map_keys] = map[self.map_keys].fillna("").astype(str).replace("", "*")
        map[self.NC_PRCT_BREAKDOWN + suff] = (map[self.NC_PRCT_BREAKDOWN + suff].astype("string")
                                              .str.replace(",", ".", regex=True)
                                              .apply(lambda x: float(x) if pd.notna(x) else np.nan))
        map["new_key"] = map[self.map_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        map = map.drop_duplicates(["new_key"] + [self.NC_BREAKDOWN + suff]).copy()
        map[self.NC_PRCT_BREAKDOWN + suff] = map[self.NC_PRCT_BREAKDOWN + suff].astype(np.float64)
        map[self.flow_columns] = map[self.flow_columns].ffill(axis=0)

        map["HAS_VOLATILE" + suff] = np.where(map[self.NC_BREAKDOWN + suff].str.upper() == "VOLATILE", 1, 0)
        rm_map_grouped = map.groupby("new_key")
        map[self.NC_PRCT_BREAKDOWN + suff] = rm_map_grouped[self.NC_PRCT_BREAKDOWN + suff].transform(
            lambda x: x.fillna(100 - x.sum()))
        map[self.NC_PRCT_BREAKDOWN + suff] = map[self.NC_PRCT_BREAKDOWN + suff].fillna(0)
        map["NB_PARTS" + suff] = rm_map_grouped["new_key"].transform(lambda x: x.count())
        map["HAS_VOLATILE" + suff] = (rm_map_grouped["HAS_VOLATILE" + suff]).transform(lambda x: x.sum())
        map = map.set_index("new_key")
        return map
    def load_liq_runoff_profile_mapping(self):
        self.liq_map = ex.get_dataframe_from_range(self.model_wb, self.NR_MODEL_MAPPING)
        self.liq_map_formula = ex.get_dataframe_from_range(self.model_wb, self.NR_MODEL_MAPPING_FORMULA)
        self.liq_map = pd.concat([self.liq_map, self.liq_map_formula])
        self.liq_map = self.liq_map[self.liq_map[self.NC_FLOW_TYPE].isin(["DUAL", "PROF_RM", "DUAL_PROFILE_LIQ"])].copy()
        self.format_liq_map_pel()
        self.liq_map = self.parse_model_map(self.liq_map, suff="_LIQ")
        cols_to_keep = (self.flow_columns + [self.NC_MODEL_NAME_SAVINGS, self.NC_MODEL_NAME_DRAWDOWNS])
        self.liq_map = self.liq_map[self.map_keys + cols_to_keep + [self.NC_FLOW_TYPE]].copy()

    def format_liq_map_pel(self):
        n = self.liq_map.shape[0]
        cond =  (self.liq_map["DECAY_TRUNCATION_TERM"] == "25Y1D") & (self.liq_map[self.NC_CONTRACT_TYPE].str.contains("P-PEL"))
        self.liq_map[self.flow_columns] = np.where(cond.values.reshape(n, 1), "@TS_I30Y", self.liq_map[self.flow_columns].values)
        self.liq_map[self.NC_BALANCE] = np.where(cond & (self.liq_map[self.NC_BALANCE] == "<=0").values,
                                                 "<0", self.liq_map[self.NC_BALANCE].values)

    def load_rm_runoff_profile_mapping(self):
        self.rm_map = ex.get_dataframe_from_range(self.model_wb, self.NR_MODEL_MAPPING)
        self.rm_map = self.rm_map[self.rm_map[self.NC_FLOW_TYPE] == "RM"].copy()
        self.rm_map[[self.NC_MODEL_NAME_SAVINGS, self.NC_MODEL_NAME_DRAWDOWNS]] = ""
        self.rm_map = self.parse_model_map(self.rm_map)
        self.rm_map[self.tci_flow_columns] = self.rm_map[self.flow_columns].copy()
        cols_to_keep = self.tci_flow_columns + [self.NC_PRCT_BREAKDOWN, self.NC_BREAKDOWN, "HAS_VOLATILE", "NB_PARTS",
                                                self.NC_TCI_FIXED_RATE_CODE, self.NC_TCI_METHOD_KEY,
                                                self.NC_TCI_FIXED_TENOR_CODE, self.NC_TCI_VARIABLE_CURVE_CODE,
                                                self.NC_TCI_VARIABLE_TENOR_CODE]
        self.rm_map = self.rm_map[self.map_keys + cols_to_keep].copy()

    def load_default_mapping(self):
        self.dflt_map = ex.get_dataframe_from_range(self.model_wb, self.NR_MODEL_MAPPING)
        self.dflt_map = self.dflt_map[self.dflt_map[self.NC_FLOW_TYPE].isin(["PROF_PROF"])].copy()
        self.dflt_map[[self.NC_MODEL_NAME_SAVINGS, self.NC_MODEL_NAME_DRAWDOWNS]] = ""
        self.dflt_map = self.parse_model_map(self.dflt_map)
        self.dflt_map[self.tci_flow_columns] = self.dflt_map[self.flow_columns].copy()
        cols_to_keep = self.tci_flow_columns + [self.NC_PRCT_BREAKDOWN, self.NC_BREAKDOWN, "HAS_VOLATILE", "NB_PARTS",
                                                self.NC_TCI_FIXED_RATE_CODE, self.NC_TCI_METHOD_KEY,
                                                self.NC_TCI_FIXED_TENOR_CODE, self.NC_TCI_VARIABLE_CURVE_CODE,
                                                self.NC_TCI_VARIABLE_TENOR_CODE]
        self.dflt_map = self.dflt_map[self.map_keys + cols_to_keep].copy()

    def load_rate_runoff_profile_mapping(self):
        self.rate_map = ex.get_dataframe_from_range(self.model_wb, self.NR_MODEL_MAPPING)
        self.rate_map = self.rate_map[
            self.rate_map[self.NC_FLOW_TYPE].isin(["RM_PROF", "DUAL_PROFILE"])].copy()
        self.rate_map = self.parse_model_map(self.rate_map, suff="_TX")
        self.rate_map[self.NC_FLOW_MODEL_STOCK] = self.rate_map[self.NC_FLOW_MODEL_STOCK].fillna("")

        self.rate_map = self.rate_map.rename(
            columns={self.NC_FLOW_MODEL_STOCK: self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX})

        self.rate_map = self.rate_map[self.map_keys + [self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX]].copy()

    def format_bilan(self, data):
        filtres = [data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "A-",
                   data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "P-", \
                   (data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[-2:] == "-A") & (
                           data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "HB"),
                   (data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[-2:] == "-P") & (
                           data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "HB")]
        choices = ["B ACTIF", "B PASSIF", "HB ACTIF", "HB PASSIF"]
        data["BILAN"] = np.select(filtres, choices)

        return data

    def map_data_with_model_maps(self, data_nmd, type_map="STOCK"):
        # PROBLEME quand RM a plus de modèles de RUNOFF
        self.load_rm_runoff_profile_mapping()
        self.load_rate_runoff_profile_mapping()
        self.load_liq_runoff_profile_mapping()
        self.load_default_mapping()

        # le mapping RM détermine le nombre de parts,les autres mapping sont à part unique
        data_nmd = self.format_data_for_model_map(data_nmd, type_map)
        data_nmd_rate = self.map_data_with_rate_runoff_models(data_nmd, type_map)
        data_nmd_liq = self.map_data_with_liq_runoff_models(data_nmd_rate, type_map)
        data_nmd_rm = self.map_data_with_rm_runoff_models(data_nmd_liq, type_map)
        data_nmd_rm = self.fill_liq_runoff_when_absent(data_nmd_rm)
        data_nmd_rm = self.correct_prof_rm_profiles(data_nmd_rm, type_map)
        data_nmd_model = self.final_rm_format(data_nmd_rm, type_map)

        return data_nmd_model

    def fill_liq_runoff_when_absent(self, data_nmd_rm):
        n = data_nmd_rm.shape[0]
        data_nmd_rm[self.cols_model_stock] = np.where(
            data_nmd_rm[self.cols_model_stock[0]].isnull().values.reshape(n, 1),
            data_nmd_rm[self.cols_model_tci].values,
            data_nmd_rm[self.cols_model_stock].values)
        return data_nmd_rm

    def correct_prof_rm_profiles(self, data_nmd_rm, type_map):
        n = data_nmd_rm.shape[0]
        if type_map == "PN":
            cols_to_get = [self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX, self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX]
            cols_to_change = [self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX, self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX + "_OPPOSITE"]
        else:
            cols_to_get = [self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX]
            cols_to_change = [self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX]
        data_nmd_rm[cols_to_change] = np.where((data_nmd_rm[self.NC_FLOW_TYPE] ==  "PROF_RM").values.reshape(n, 1),
            data_nmd_rm[self.cols_model_tci].values, data_nmd_rm[cols_to_get].values)

        return data_nmd_rm

    def map_data_with_liq_runoff_models(self, data_nmd, type_map):
        keys_data = [self.cls_fields.NC_LDP_CONTRACT_TYPE, self.cls_fields.NC_LDP_MARCHE,
                     self.cls_fields.NC_LDP_CURRENCY,
                     self.cls_fields.NC_LDP_RATE_TYPE, self.NC_STOCK_OR_PN, self.NC_BALANCE,
                     self.cls_fields.NC_LDP_BASSIN, self.NC_NETWORK]

        if type_map == "STOCK":
            self.cols_model_stock = [self.NC_FLOW_MODEL_STOCK]
            data_nmd[self.NC_BALANCE] = "*"
            data_nmd[self.NC_STOCK_OR_PN] = "*"
        else:
            self.cols_model_stock = [self.NC_FLOW_MODEL_PN, self.NC_FLOW_MODEL_OPPOSITE_PN]

            is_neg = data_nmd[self.cls_pa_fields.NC_PA_INDEX].str.contains("NEG")
            cases = [is_neg & (data_nmd["BILAN"].str.contains("ACTIF")),
                     is_neg & (data_nmd["BILAN"].str.contains("PASSIF")),
                     ~is_neg & (data_nmd["BILAN"].str.contains("ACTIF")),
                     ~is_neg & (data_nmd["BILAN"].str.contains("PASSIF"))
                     ]
            data_nmd[self.NC_BALANCE] = np.select(cases, ["<0", ">0", ">0", "<0"], "*")
            data_nmd[self.NC_STOCK_OR_PN] = "T"

        cols_to_get = (self.cols_model_stock + [self.NC_MODEL_NAME_SAVINGS, self.NC_MODEL_NAME_DRAWDOWNS]
                       + [self.NC_FLOW_TYPE])

        data_nmd_liq = \
            ut.map_with_combined_key(data_nmd, self.liq_map, keys_data, symbol_any="*",
                                     filter_comb=True, necessary_cols=1, cols_mapp=cols_to_get,
                                     error=False, allow_duplicates=True, name_key_col="KEY_RM_JOIN1").copy()

        return data_nmd_liq

    def map_data_with_rm_runoff_models(self, data_nmd, type_map):
        keys_data = [self.cls_fields.NC_LDP_CONTRACT_TYPE, self.cls_fields.NC_LDP_MARCHE,
                     self.cls_fields.NC_LDP_CURRENCY,
                     self.cls_fields.NC_LDP_RATE_TYPE, self.NC_STOCK_OR_PN, self.NC_BALANCE,
                     self.cls_fields.NC_LDP_BASSIN, self.NC_NETWORK]

        if type_map == "STOCK":
            self.cols_model_tci = [self.NC_TCI_FLOW_MODEL_STOCK]
            data_nmd[self.NC_BALANCE] = "*"
            data_nmd[self.NC_STOCK_OR_PN] = "*"
        else:
            self.cols_model_tci = [self.NC_TCI_FLOW_MODEL_PN, self.NC_TCI_FLOW_MODEL_OPPOSITE_PN]

            is_neg = data_nmd[self.cls_pa_fields.NC_PA_INDEX].str.contains("NEG")
            cases = [is_neg & (data_nmd["BILAN"].str.contains("ACTIF")),
                     is_neg & (data_nmd["BILAN"].str.contains("PASSIF")),
                     ~is_neg & (data_nmd["BILAN"].str.contains("ACTIF")),
                     ~is_neg & (data_nmd["BILAN"].str.contains("PASSIF"))
                     ]
            data_nmd[self.NC_BALANCE] = np.select(cases, ["<0", ">0", ">0", "<0"], "*")
            data_nmd[self.NC_STOCK_OR_PN] = "T"

        cols_to_get = self.cols_model_tci + [
            self.NC_PRCT_BREAKDOWN, self.NC_BREAKDOWN, "HAS_VOLATILE", "NB_PARTS",
            self.NC_TCI_FIXED_RATE_CODE,
            self.NC_TCI_METHOD_KEY, self.NC_TCI_FIXED_TENOR_CODE, self.NC_TCI_VARIABLE_CURVE_CODE,
            self.NC_TCI_VARIABLE_TENOR_CODE]

        data_nmd_rm = \
            ut.map_with_combined_key(data_nmd, self.rm_map, keys_data, symbol_any="*",
                                     filter_comb=True, necessary_cols=1, cols_mapp=cols_to_get,
                                     error=False, allow_duplicates=True, name_key_col="KEY_RM_JOIN1").copy()

        filter_warning = data_nmd_rm[self.NC_BREAKDOWN].isnull()
        if filter_warning.any():
            liste = data_nmd_rm.loc[filter_warning, keys_data].drop_duplicates().values.tolist()
            logger.warning(
                "Certains produits NMD n'ont pas de profil d'écoulement, ils seront écoulés en 1 jour: %s" % liste)
            data_nmd_rm_1D = data_nmd_rm[filter_warning].drop(cols_to_get, axis=1)
            data_nmd_rm_1D = \
                ut.map_with_combined_key(data_nmd_rm_1D, self.dflt_map, keys_data, symbol_any="*",
                                         filter_comb=False, necessary_cols=1, cols_mapp=cols_to_get,
                                         error=False, allow_duplicates=True, drop_key_col=False,
                                         name_key_col="KEY_RM_JOIN2").copy()
            data_nmd_rm.loc[filter_warning, cols_to_get] = data_nmd_rm_1D[cols_to_get]

        return data_nmd_rm

    def map_data_with_rate_runoff_models(self, data_nmd, type_map):
        keys_data = [self.cls_fields.NC_LDP_CONTRACT_TYPE, self.cls_fields.NC_LDP_MARCHE,
                     self.cls_fields.NC_LDP_CURRENCY,
                     self.cls_fields.NC_LDP_RATE_TYPE, self.NC_STOCK_OR_PN, self.NC_BALANCE,
                     self.cls_fields.NC_LDP_BASSIN, self.NC_NETWORK]

        if type_map == "PN":
            is_neg = data_nmd[self.cls_pa_fields.NC_PA_INDEX].str.contains("NEG")
            cases = [is_neg & (data_nmd["BILAN"].str.contains("ACTIF")),
                     is_neg & (data_nmd["BILAN"].str.contains("PASSIF")),
                     ~is_neg & (data_nmd["BILAN"].str.contains("ACTIF")),
                     ~is_neg & (data_nmd["BILAN"].str.contains("PASSIF"))
                     ]
            data_nmd[self.NC_BALANCE] = np.select(cases, ["<0", ">0", ">0", "<0"], "*")
        else:
            cases = [(data_nmd[self.cls_fields.NC_LDP_OUTSTANDING] < 0) & (
                data_nmd["BILAN"].str.contains("ACTIF")),
                     (data_nmd[self.cls_fields.NC_LDP_OUTSTANDING] > 0) & (
                         data_nmd["BILAN"].str.contains("PASSIF"))]
            data_nmd[self.NC_BALANCE] = np.select(cases, ["<0", ">0"], "*")

        data_nmd_rate = \
            ut.map_with_combined_key(data_nmd, self.rate_map, keys_data, symbol_any="*",
                                     filter_comb=False, necessary_cols=1,
                                     cols_mapp=[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX],
                                     error=False, allow_duplicates=True, drop_key_col=False,
                                     name_key_col="KEY_RUNOFF_JOIN", filter_none_comb=True).copy()

        data_nmd_rate[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX] = data_nmd_rate[
            self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX].fillna("")
        return data_nmd_rate

    def format_data_for_model_map(self, data_nmd, type_map):
        cases = [data_nmd[self.cls_fields.NC_LDP_BASSIN].isin(["CFF", "CONSO_CFF"]),
                 data_nmd[self.cls_fields.NC_LDP_BASSIN].isin(["BPCE", "BPCE SA", "GROUPE_BPCE"])]
        data_nmd[self.NC_NETWORK] = np.select(cases, ["CFF", "BPCE SA"], default="*")

        if type_map == "STOCK":
            data_nmd[self.NC_STOCK_OR_PN] = "ST"
        else:
            data_nmd[self.NC_STOCK_OR_PN] = "T"
        data_nmd = self.format_bilan(data_nmd)

        data_nmd[self.cls_fields.NC_LDP_MATUR_DATE] = (self.dar_usr + relativedelta(months=self.horizon + 3)).strftime \
            ("%d/%m/%Y")

        return data_nmd

    def final_rm_format(self, data_nmd_rm, type_map):
        data_nmd_rm[self.cls_fields.NC_LDP_RM_GROUP] = data_nmd_rm[self.NC_BREAKDOWN].values

        if type_map == "STOCK":
            data_nmd_rm[self.cls_fields.NC_LDP_OUTSTANDING] = data_nmd_rm[self.cls_fields.NC_LDP_OUTSTANDING] * \
                                                              data_nmd_rm[
                                                                  self.NC_PRCT_BREAKDOWN] / 100
            data_nmd_rm[self.cls_fields.NC_LDP_INTERESTS_ACCRUALS] = data_nmd_rm[
                                                                         self.cls_fields.NC_LDP_INTERESTS_ACCRUALS] * \
                                                                     data_nmd_rm[self.NC_PRCT_BREAKDOWN] / 100

        data_nmd_rm[self.cls_fields.NC_LDP_RM_GROUP_PRCT] = data_nmd_rm[self.NC_PRCT_BREAKDOWN].values / 100

        if type_map == "PN":
            data_nmd_rm[self.cls_fields.NC_LDP_FLOW_MODEL_NMD] \
                = np.where(~data_nmd_rm[self.cls_pa_fields.NC_PA_INDEX].str.contains("NEG"),
                           data_nmd_rm[self.NC_FLOW_MODEL_PN], data_nmd_rm[self.NC_FLOW_MODEL_OPPOSITE_PN])

            """data_nmd_rm[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX] \
                = np.where(~data_nmd_rm[self.cls_pa_fields.NC_PA_INDEX].str.contains("NEG"),
                           data_nmd_rm[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX],
                           data_nmd_rm[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX + "_OPPOSITE"])"""

            data_nmd_rm[self.cls_fields.NC_LDP_FLOW_MODEL_NMD_TCI] \
                = np.where(~data_nmd_rm[self.cls_pa_fields.NC_PA_INDEX].str.contains("NEG"),
                           data_nmd_rm[self.NC_TCI_FLOW_MODEL_PN], data_nmd_rm[self.NC_TCI_FLOW_MODEL_OPPOSITE_PN])
        else:
            data_nmd_rm = data_nmd_rm.rename(columns={self.NC_FLOW_MODEL_STOCK: self.cls_fields.NC_LDP_FLOW_MODEL_NMD})
            data_nmd_rm = data_nmd_rm.rename(
                columns={self.NC_TCI_FLOW_MODEL_STOCK: self.cls_fields.NC_LDP_FLOW_MODEL_NMD_TCI})

        data_nmd_rm = data_nmd_rm.rename(columns={self.NC_FIXED_RATE_CODE: self.cls_fields.NC_LDP_TCI_FIXED_RATE_CODE})
        data_nmd_rm = data_nmd_rm.rename(columns={self.NC_TCI_METHOD_KEY: self.cls_fields.NC_LDP_TCI_METHOD})
        data_nmd_rm = data_nmd_rm.rename(
            columns={self.NC_TCI_FIXED_TENOR_CODE: self.cls_fields.NC_LDP_TCI_FIXED_TENOR_CODE})
        data_nmd_rm = data_nmd_rm.rename(
            columns={self.NC_TCI_VARIABLE_TENOR_CODE: self.cls_fields.NC_LDP_TCI_VARIABLE_TENOR_CODE})
        data_nmd_rm = data_nmd_rm.rename(
            columns={self.NC_TCI_VARIABLE_CURVE_CODE: self.cls_fields.NC_LDP_TCI_VARIABLE_CURVE_CODE})
        data_nmd_rm = data_nmd_rm.rename(columns={self.NC_MODEL_NAME_SAVINGS: self.cls_fields.NC_LDP_SAVINGS_MODEL})
        data_nmd_rm = data_nmd_rm.rename(
            columns={self.NC_MODEL_NAME_DRAWDOWNS: self.cls_fields.NC_LDP_REDEMPTION_MODEL})

        if type_map == "STOCK":
            data_nmd_rm = data_nmd_rm.sort_values([self.cls_fields.NC_LDP_CONTRAT, self.cls_fields.NC_LDP_RM_GROUP])
        else:
            data_nmd_rm \
                = data_nmd_rm.sort_values(["NB_PARTS", "HAS_VOLATILE",
                                           self.cls_pa_fields.NC_PA_INDEX, self.cls_pa_fields.NC_PA_IND03,
                                           self.cls_fields.NC_LDP_RM_GROUP])
        return data_nmd_rm
