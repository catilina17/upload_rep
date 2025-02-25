import numpy as np
import pandas as pd
from utils import general_utils as gu
from calculateur.models.utils import utils as ut
from calculateur.data_transformers import commons as com
import logging

logger = logging.getLogger(__name__)


class Data_ECH_PN():
    """
    Formate les données
    """
    def __init__(self, cls_fields, cls_format, cls_pa_fields, source_data, dar_usr, tx_params,
                 type_run_off, map_bassins, max_pn, batch_size, type_ech = ""):
        self.max_pn = max_pn
        self.dar_usr = dar_usr
        self.cls_fields = cls_fields
        self.cls_format = cls_format
        self.cls_pa_fields = cls_pa_fields
        self.type_ech = type_ech
        self.load_columns_names()
        self.load_correspondances()
        self.load_default_vars()
        self.load_mappings()
        self.source_data = source_data
        self.tx_params = tx_params
        self.map_bassins = map_bassins
        self.batch_size = batch_size
        self.type_run_off = type_run_off


    def load_columns_names(self):
        self.NC_DEM = self.cls_pa_fields.NC_PA_DEM_CIBLE if self.type_ech == "%" else self.cls_pa_fields.NC_PA_DEM
        self.NC_MARGE_CO = self.cls_pa_fields.NC_PA_MG_CO
        self.NC_TX_CIBLE = self.cls_pa_fields.NC_PA_TX_CIBLE
        self.NC_TX_SPREAD = self.cls_pa_fields.NC_PA_TX_SP
        self.num_cols = ["M%s" % i for i in range(1, self.max_pn + 1)]
        self.qual_cols = [self.cls_pa_fields.NC_PA_CONTRACT_TYPE, self.cls_pa_fields.NC_PA_DEVISE, self.cls_pa_fields.NC_PA_MARCHE,
                          self.cls_pa_fields.NC_PA_ETAB, self.cls_pa_fields.NC_PA_BASSIN, self.cls_pa_fields.NC_PA_MATUR,
                          self.cls_pa_fields.NC_PA_RATE_CODE, self.cls_pa_fields.NC_PA_ACCRUAL_BASIS, self.cls_pa_fields.NC_PA_JR_PN.upper(),
                          self.cls_pa_fields.NC_PA_AMORTIZING_TYPE,
                          self.cls_pa_fields.NC_PA_MATURITY_DURATION, self.cls_pa_fields.NC_PA_AMORTIZING_PERIODICITY,
                          self.cls_pa_fields.NC_PA_PERIODICITY, self.cls_pa_fields.NC_PA_COMPOUND_PERIODICITY,
                          self.cls_pa_fields.NC_PA_IND03, self.cls_pa_fields.NC_PA_BILAN, self.cls_fields.NC_LDP_CURVE_NAME,
                          self.cls_fields.NC_LDP_TENOR, self.cls_fields.NC_PRICING_CURVE, self.cls_pa_fields.NC_PA_FIXING_PERIODICITY,
                          self.cls_pa_fields.NC_PA_RELEASING_RULE]

    def load_default_vars(self):
        self.default_ldp_vars = \
            {self.cls_fields.NC_LDP_CONTRAT: '', self.cls_fields.NC_LDP_VALUE_DATE: '.',
             self.cls_fields.NC_LDP_TRADE_DATE: '.',
             self.cls_fields.NC_LDP_FIRST_AMORT_DATE: ".", self.cls_fields.NC_LDP_MATUR_DATE: ".",
             self.cls_fields.NC_LDP_RELEASING_DATE: ".", self.cls_fields.NC_LDP_RELEASING_RULE: np.nan,
             self.cls_fields.NC_LDP_RATE: 0, self.cls_fields.NC_LDP_TYPE_AMOR: "F",
             self.cls_fields.NC_LDP_FREQ_AMOR: "Monthly",
             self.cls_fields.NC_LDP_FREQ_INT: "M", self.cls_fields.NC_LDP_ECHEANCE_VAL: np.nan, self.cls_fields.NC_LDP_NOMINAL: 0,
             self.cls_fields.NC_LDP_OUTSTANDING: 0,
             self.cls_fields.NC_LDP_CAPITALIZATION_PERIOD: np.nan, self.cls_fields.NC_LDP_CAPITALIZATION_RATE: np.nan,
             self.cls_fields.NC_LDP_ACCRUAL_BASIS: "30/360", self.cls_fields.NC_LDP_CURRENCY: "EUR",
             self.cls_fields.NC_LDP_BROKEN_PERIOD: "Start Short", self.cls_fields.NC_LDP_ETAB: "",
             self.cls_fields.NC_LDP_INTERESTS_ACCRUALS: 0, self.cls_fields.NC_LDP_CONTRACT_TYPE: "",
             self.cls_fields.NC_LDP_CURVE_NAME: "",
             self.cls_fields.NC_LDP_TENOR: "1M", self.cls_fields.NC_LDP_MKT_SPREAD: 0,
             self.cls_fields.NC_LDP_FIXING_NEXT_DATE: ".",
             self.cls_fields.NC_LDP_CALC_DAY_CONVENTION: 1,
             self.cls_fields.NC_LDP_FIXING_PERIODICITY: "1M", self.cls_fields.NC_LDP_RATE_CODE: "",
             self.cls_fields.NC_LDP_CAP_STRIKE: np.nan, self.cls_fields.NC_LDP_FLOOR_STRIKE: np.nan,
             self.cls_fields.NC_LDP_MULT_SPREAD: 1, self.cls_fields.NC_LDP_RATE_TYPE: "FIXED",
             self.cls_fields.NC_LDP_MARCHE: "",
             self.cls_fields.NC_LDP_FTP_RATE: np.nan, self.cls_fields.NC_LDP_CURRENT_RATE: np.nan,
             self.cls_fields.NC_LDP_CAPI_MODE: "P",
             self.cls_fields.NC_LDP_BUY_SELL: "", self.cls_fields.NC_LDP_NB_CONTRACTS: 1,
             self.cls_fields.NC_LDP_PERFORMING: "F",
             self.cls_fields.NC_LDP_MATUR: "",
             self.cls_fields.NC_LDP_IS_CAP_FLOOR: "", self.cls_fields.NC_LDP_DATE_SORTIE_GAP: ".",
             self.cls_fields.NC_LDP_FIXING_NB_DAYS : 0, self.cls_fields.NC_LDP_IS_FUTURE_PRODUCT : True,
             self.cls_fields.NC_PRICING_CURVE : "",
             self.cls_fields.NC_LDP_FLOW_MODEL_NMD: "", self.cls_fields.NC_LDP_RM_GROUP: "",
             self.cls_fields.NC_LDP_RM_GROUP_PRCT: 0, self.cls_fields.NC_LDP_FIXING_RULE: "B",
             self.cls_fields.NC_LDP_FIRST_COUPON_DATE: ".", self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX: "",
             self.cls_fields.NC_LDP_TARGET_RATE:np.nan, self.cls_fields.NC_LDP_DIM6 : "",
             self.cls_fields.NC_LDP_FTP_FUNDING_SPREAD: 0,  self.cls_fields.NC_LDP_FLOW_MODEL_NMD_TCI : "",
             self.cls_fields.NC_LDP_TCI_METHOD : "", self.cls_fields.NC_LDP_TCI_FIXED_RATE_CODE: "",
             self.cls_fields.NC_LDP_TCI_FIXED_TENOR_CODE : "", self.cls_fields.NC_LDP_TCI_VARIABLE_TENOR_CODE : "",
             self.cls_fields.NC_LDP_TCI_VARIABLE_CURVE_CODE : "", self.cls_fields.NC_LDP_SAVINGS_MODEL: "",
             self.cls_fields.NC_LDP_REDEMPTION_MODEL : ""}

    def load_correspondances(self):
        self.correspondances = {}
        self.correspondances[self.cls_pa_fields.NC_PA_DEVISE] = self.cls_fields.NC_LDP_CURRENCY
        self.correspondances[self.cls_pa_fields.NC_PA_MARCHE] = self.cls_fields.NC_LDP_MARCHE
        self.correspondances[self.cls_pa_fields.NC_PA_ETAB] = self.cls_fields.NC_LDP_ETAB
        self.correspondances[self.cls_pa_fields.NC_PA_BASSIN] = self.cls_fields.NC_LDP_BASSIN
        self.correspondances[self.cls_pa_fields.NC_PA_ACCRUAL_BASIS] = self.cls_fields.NC_LDP_ACCRUAL_BASIS
        self.correspondances[self.cls_pa_fields.NC_PA_CONTRACT_TYPE] = self.cls_fields.NC_LDP_CONTRACT_TYPE
        self.correspondances[self.cls_pa_fields.NC_PA_MATUR] = self.cls_fields.NC_LDP_MATUR
        self.correspondances[self.cls_pa_fields.NC_PA_ACCRUAL_BASIS] = self.cls_fields.NC_LDP_ACCRUAL_BASIS
        self.correspondances[self.cls_pa_fields.NC_PA_AMORTIZING_TYPE] = self.cls_fields.NC_LDP_TYPE_AMOR
        self.correspondances[self.cls_pa_fields.NC_PA_AMORTIZING_PERIODICITY] = self.cls_fields.NC_LDP_FREQ_AMOR
        self.correspondances[self.cls_pa_fields.NC_PA_PERIODICITY] = self.cls_fields.NC_LDP_FREQ_INT
        self.correspondances[self.cls_pa_fields.NC_PA_COMPOUND_PERIODICITY] = self.cls_fields.NC_LDP_CAPITALIZATION_PERIOD
        self.correspondances[self.cls_pa_fields.NC_PA_RATE_CODE] = self.cls_fields.NC_LDP_RATE_CODE
        self.correspondances[self.cls_pa_fields.NC_PA_FIXING_PERIODICITY] = self.cls_fields.NC_LDP_FIXING_PERIODICITY
        self.correspondances[self.cls_pa_fields.NC_PA_RELEASING_RULE] = self.cls_fields.NC_LDP_RELEASING_RULE


    def load_mappings(self):
        self.map_amortizing_type = {"INFINE": "F", "ECHCONST":"A", "LINEAIRE":"L"}
        self.map_amortizing_periodicity = {"MENSUEL": "Monthly", "TRIMESTRIEL":"Quarterly",
                                           "SEMESTRIEL":"Semestrial", "ANNUEL":"Annual", "NONE": "None"}
        self.map_periodicity = {"MENSUEL": "M", "TRIMESTRIEL":"Q",
                                           "SEMESTRIEL":"S", "ANNUEL":"A", "NONE":"N"}
        self.map_comp_periodicity = {"NONE": np.nan, "MENSUEL": "M", "TRIMESTRIEL":"Q",
                                           "SEMESTRIEL":"S", "ANNUEL":"A"}

    def add_deblocage_column_if_does_not_exists(self, data_ech_pn):
        if not self.cls_pa_fields.NC_PA_RELEASING_RULE in data_ech_pn.columns:
            data_ech_pn[self.cls_pa_fields.NC_PA_RELEASING_RULE] = np.nan

        return data_ech_pn

    def get_max_duree(self, data_ech_pn):
        self.max_duree = int(data_ech_pn[self.cls_pa_fields.NC_PA_MATURITY_DURATION].max())

    def get_curve_name_and_tenor_columns(self, data_ech_pn):
        data_ech_index_calc = data_ech_pn[[self.cls_pa_fields.NC_PA_DEVISE, self.cls_pa_fields.NC_PA_RATE_CODE]].copy()
        data_ech_index_calc[self.cls_pa_fields.NC_PA_RATE_CODE] = data_ech_index_calc[self.cls_pa_fields.NC_PA_RATE_CODE].str.upper().copy()
        data_ech_index_calc = data_ech_index_calc.join(self.tx_params["map_index_curve_tenor"]["data"],
                                                       on=[self.cls_pa_fields.NC_PA_DEVISE, self.cls_pa_fields.NC_PA_RATE_CODE])

        if len(data_ech_index_calc) > len(data_ech_pn):
            logger.error("Il y a des doublons dans le mapping rate_code-curve")
            raise ValueError("Il y a des doublons dans le mapping rate_code-curve")

        data_ech_index_calc = data_ech_index_calc.rename(columns = {self.tx_params["map_index_curve_tenor"]["col_curve"]: self.cls_fields.NC_LDP_CURVE_NAME,
                                                    self.tx_params["map_index_curve_tenor"]["col_tenor"]: self.cls_fields.NC_LDP_TENOR})

        missing_filter = (data_ech_index_calc[self.cls_fields.NC_LDP_CURVE_NAME].isnull()
                          & ~data_ech_index_calc[self.cls_pa_fields.NC_PA_RATE_CODE].str.contains("FIXE"))
        if missing_filter.any():
            list_missing = data_ech_index_calc.loc[
                missing_filter, [self.cls_pa_fields.NC_PA_DEVISE, self.cls_pa_fields.NC_PA_RATE_CODE]].drop_duplicates().values.tolist()
            msg = "Il y a des index manquants dans le mapping INDEX / NOM COURBE-TENOR : %s" % list_missing
            logger.error(msg)
            raise ValueError(msg)

        return pd.concat([data_ech_pn,data_ech_index_calc[[self.cls_fields.NC_LDP_CURVE_NAME,
                                                           self.cls_fields.NC_LDP_TENOR]].copy()], axis=1)

    def get_pricing_curve(self, data_ech):
        mapping_pn_pricing = self.tx_params["map_pricing_curves"]["data"]

        data_ech["TYPE_TX"] = np.where(data_ech[self.cls_pa_fields.NC_PA_RATE_CODE].str.contains("FIXE"), 'TF', 'TV')
        cols_data = [self.cls_pa_fields.NC_PA_CONTRACT_TYPE, self.cls_pa_fields.NC_PA_DIM2, self.cls_pa_fields.NC_PA_BILAN, "TYPE_TX",
                     self.cls_pa_fields.NC_PA_MARCHE, self.cls_pa_fields.NC_PA_DEVISE,
                     self.cls_pa_fields.NC_PA_RATE_CODE]

        self.col_pricing_curve = self.tx_params["map_pricing_curves"]["col_pricing_curve"]

        data_ech_dem = data_ech[data_ech[self.cls_pa_fields.NC_PA_IND03] == self.NC_DEM].copy().reset_index(drop=True)

        pricing_curve = \
            gu.map_with_combined_key(data_ech_dem[cols_data].copy(), mapping_pn_pricing, cols_data, symbol_any="*",
                                     cols_mapp=[self.col_pricing_curve], error=True)[[self.col_pricing_curve]].copy().replace("", 0).fillna(0)

        data_ech[self.cls_fields.NC_PRICING_CURVE] = np.repeat(pricing_curve.values, 4, axis=0).reshape(data_ech.shape[0])

        return data_ech

    def format_data_for_calculator(self, data):
        data_dem = data[data[self.cls_pa_fields.NC_PA_IND03] == self.NC_DEM].copy()
        n = data_dem.shape[0]

        """filter_mg_co = (data[self.cls_pa_fields.NC_PA_IND03] == self.NC_MARGE_CO).values
        months_shift = np.minimum(self.max_pn - 1, 25 - data.loc[filter_mg_co, self.cls_pa_fields.NC_PA_RELEASING_RULE].fillna(25).values)
        marge_co_cols = data.loc[filter_mg_co, self.num_cols].values
        marge_co_cols = ut.strided_indexing_roll(marge_co_cols, -months_shift.astype(int), rep_val=np.nan)
        data.loc[filter_mg_co, self.num_cols] = marge_co_cols

        data.loc[filter_mg_co, self.num_cols]\
            = data.loc[filter_mg_co, self.num_cols].ffill(axis=1)"""

        date_usr = np.array([self.dar_usr] * self.max_pn).astype("datetime64[M]").astype("datetime64[D]")
        months_add = np.arange(1, self.max_pn + 1)
        value_dates = pd.to_datetime(ut.add_months_date(date_usr, months_add)).strftime("%d/%m/%Y")
        data = data.rename(columns={month: date for month, date in zip(self.num_cols, value_dates)})

        data = pd.melt(data, id_vars=self.qual_cols + [self.cls_pa_fields.NC_PA_INDEX],
                       value_vars=value_dates.values.tolist(), var_name=self.cls_fields.NC_LDP_VALUE_DATE,
                       value_name=self.cls_fields.NC_LDP_NOMINAL)

        data_dem = data[data[self.cls_pa_fields.NC_PA_IND03] == self.NC_DEM].copy().reset_index(drop=True)
        data_marge = data[data[self.cls_pa_fields.NC_PA_IND03] == self.NC_MARGE_CO].copy()
        data_tx_prod_cible = data[data[self.cls_pa_fields.NC_PA_IND03] == self.NC_TX_CIBLE].copy()
        data_sp = data[data[self.cls_pa_fields.NC_PA_IND03] == self.NC_TX_SPREAD].copy()

        data_dem[self.cls_fields.NC_LDP_CONTRAT] =\
            np.char.add(np.char.add(data_dem[self.cls_pa_fields.NC_PA_INDEX].values.astype(str),"*PN"),
                        np.repeat(np.arange(1, self.max_pn + 1), n).astype(str))

        data_dem = self.set_date_columns(data_dem)

        data_dem[self.cls_fields.NC_LDP_NB_CONTRACTS] = 1

        data_dem[self.cls_fields.NC_LDP_GESTION] = ""
        data_dem[self.cls_fields.NC_LDP_PALIER] = ""

        data_dem = self.set_rates_columns(data_dem, data_marge, data_sp, data_tx_prod_cible)

        data_dem[self.cls_fields.NC_LDP_OUTSTANDING] = 0

        is_passif = ((data_dem[self.cls_pa_fields.NC_PA_BILAN].str.contains("PASSIF"))
                     & (~data_dem[self.cls_pa_fields.NC_PA_CONTRACT_TYPE].str.contains("HB-CAP|HB-FLOOR").values))

        if "profile" in self.type_run_off:
            data_dem[self.cls_fields.NC_LDP_NOMINAL] = np.where(is_passif, -1, 1)
        else:
            data_dem[self.cls_fields.NC_LDP_NOMINAL] = np.where(is_passif,
                                                              -1 * data_dem[self.cls_fields.NC_LDP_NOMINAL].values.astype(float),
                                                              data_dem[self.cls_fields.NC_LDP_NOMINAL].values.astype(float))

        data_dem[self.cls_pa_fields.NC_PA_BASSIN] = data_dem[[self.cls_pa_fields.NC_PA_BASSIN]].join(self.map_bassins, on = [self.cls_pa_fields.NC_PA_BASSIN]).iloc[:, 1]

        data_dem[self.cls_pa_fields.NC_PA_BASSIN] = np.select([data_dem[self.cls_pa_fields.NC_PA_BASSIN].values==bassin for bassin in ["BP", "CEP","CONSO_BRED"]],
                                                   ["CONSO_BC_BP", "CONSO_BC_CEP", "CONSO_BC_BP"],
                                                    default=data_dem[self.cls_pa_fields.NC_PA_BASSIN].values)

        data_dem = data_dem.sort_values([self.cls_pa_fields.NC_PA_MATURITY_DURATION, self.cls_fields.NC_LDP_VALUE_DATE], ascending=[True, True])

        months_to_add = 25 - data_dem[self.cls_pa_fields.NC_PA_RELEASING_RULE].fillna(25).values
        data_dem[self.cls_fields.NC_LDP_VALUE_DATE] = ut.add_months_date(data_dem[self.cls_fields.NC_LDP_TRADE_DATE].values, months_to_add)

        data_dem[self.cls_fields.NC_LDP_MATUR_DATE] = pd.to_datetime(data_dem[self.cls_fields.NC_LDP_MATUR_DATE]).dt.strftime("%d/%m/%Y")
        data_dem[self.cls_fields.NC_LDP_VALUE_DATE] = pd.to_datetime(data_dem[self.cls_fields.NC_LDP_VALUE_DATE]).dt.strftime("%d/%m/%Y")
        data_dem[self.cls_fields.NC_LDP_TRADE_DATE] = pd.to_datetime(data_dem[self.cls_fields.NC_LDP_TRADE_DATE]).dt.strftime("%d/%m/%Y")

        data_dem[self.cls_fields.NC_LDP_CAPITALIZATION_RATE]\
            = np.where(data_dem[self.cls_pa_fields.NC_PA_COMPOUND_PERIODICITY] != "NONE", 1, 0)

        data_dem[self.cls_pa_fields.NC_PA_PERIODICITY]\
            = np.where(data_dem[self.cls_pa_fields.NC_PA_COMPOUND_PERIODICITY] != "NONE", "NONE",
                       data_dem[self.cls_pa_fields.NC_PA_PERIODICITY].values)

        data_dem[self.cls_pa_fields.NC_PA_RELEASING_RULE] = data_dem[self.cls_pa_fields.NC_PA_RELEASING_RULE].astype(np.float64)

        return data_dem

    def remap_data(self, data):
        data.replace({self.cls_pa_fields.NC_PA_AMORTIZING_TYPE: self.map_amortizing_type}, inplace=True)
        data.replace({self.cls_pa_fields.NC_PA_AMORTIZING_PERIODICITY: self.map_amortizing_periodicity}, inplace=True)
        data.replace({self.cls_pa_fields.NC_PA_PERIODICITY: self.map_periodicity}, inplace=True)
        data.replace({self.cls_pa_fields.NC_PA_COMPOUND_PERIODICITY: self.map_comp_periodicity}, inplace=True)
        return data

    def set_date_columns(self, data_dem):
        data_dem[self.cls_fields.NC_LDP_VALUE_DATE] = pd.to_datetime(data_dem[self.cls_fields.NC_LDP_VALUE_DATE],format="%d/%m/%Y")
        data_dem[self.cls_fields.NC_LDP_VALUE_DATE]\
            = np.where(data_dem[self.cls_pa_fields.NC_PA_JR_PN.upper()].astype(float).astype(int).astype(str) == "15",
                       ut.add_days_date(data_dem[self.cls_fields.NC_LDP_VALUE_DATE].values, 14),
                       data_dem[self.cls_fields.NC_LDP_VALUE_DATE])
        data_dem[self.cls_fields.NC_LDP_MATUR_DATE] = ut.add_months_date(data_dem[self.cls_fields.NC_LDP_VALUE_DATE].values,
                                                                   data_dem[self.cls_pa_fields.NC_PA_MATURITY_DURATION])

        data_dem[self.cls_fields.NC_LDP_TRADE_DATE] = data_dem[self.cls_fields.NC_LDP_VALUE_DATE].values

        return data_dem

    def set_rates_columns(self, data_dem, data_marge, data_sp, data_tx_prod_cible):

        data_dem[self.cls_fields.NC_LDP_RATE_TYPE] = np.where(data_dem[self.cls_pa_fields.NC_PA_RATE_CODE].str.contains("FIXE"),
                                                              "FIXED", "FLOATING")

        data_dem[self.cls_fields.NC_LDP_MKT_SPREAD] = (data_marge[self.cls_fields.NC_LDP_NOMINAL].fillna(0).values.astype(float)
                                                     + data_sp[self.cls_fields.NC_LDP_NOMINAL].fillna(0).values.astype(float))

        data_dem[self.cls_fields.NC_LDP_TARGET_RATE] = data_tx_prod_cible[self.cls_fields.NC_LDP_NOMINAL].values.astype(float)

        data_dem[self.cls_fields.NC_LDP_FTP_RATE] =  np.nan

        data_dem[self.cls_fields.NC_LDP_RATE]=  np.nan

        return data_dem

    def filter_data(self, data_ech_pn):
        #data_ech_pn = data_ech_pn[data_ech_pn["CONTRAT"] == "A-CR-HAB-STD"]
        #data_ech_pn = data_ech_pn[data_ech_pn["MARCHE"] == "PRO"]
        #data_ech_pn = data_ech_pn[data_ech_pn["ETAB"] == "BCP_PARIS"]
        #data_ech_pn = data_ech_pn[data_ech_pn["INDEX"].isin(["ECH162"])]
        #data_ech_pn = data_ech_pn[data_ech_pn[self.cls_pa_fields.NC_PA_RELEASING_RULE].notnull()].copy()
        return data_ech_pn

    def read_file_and_standardize_data(self):
        logging.debug('Lecture du fichier PN ECH')
        if not 'DATA' in self.source_data["LDP"]:
            data_ech_pn = com.read_file(self.source_data, "LDP")
        else:
            data_ech_pn = self.source_data["LDP"]["DATA"]

        data_ech_pn = self.filter_data(data_ech_pn)

        data_ech_pn = self.cls_format.upper_columns_names(data_ech_pn)
        data_ech_pn = self.add_deblocage_column_if_does_not_exists(data_ech_pn)
        self.get_max_duree(data_ech_pn)
        data_ech_pn = self.get_curve_name_and_tenor_columns(data_ech_pn)
        data_ech_pn = self.get_pricing_curve(data_ech_pn)
        data_ech_pn = self.format_data_for_calculator(data_ech_pn)
        data_ech_pn = self.remap_data(data_ech_pn)
        data_ech_pn = data_ech_pn.rename(columns=self.correspondances)
        data_ech_pn = self.cls_format.create_unvailable_variables(data_ech_pn, self.cls_fields.ldp_vars,
                                                                  self.default_ldp_vars)
        self.data = com.chunkized_data(data_ech_pn, self.batch_size)



