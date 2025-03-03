import datetime
import numpy as np
import os
import pandas as pd
from calculateur.models.utils import utils as ut
from calculateur.data_transformers.data_in.nmd.class_nmd_templates import Data_NMD_TEMPLATES
from .class_nmd_model_mapper import Data_NMD_MODEL_MAPPER
from utils import excel_openpyxl as ex
from calculateur.data_transformers import commons as com
import logging

logger = logging.getLogger(__name__)


class Data_NMD_PN_Formater():
    """
    Formate les données
    """

    def __init__(self, cls_fields, cls_format, cls_pa_fields, source_data, dar_usr,
                 model_wb, horizon, tx_params, etab, cls_nmd_tmp, max_pn=60, type_rm="NORMAL",
                 batch_size=5000, exec_mode = "simul"):
        self.max_pn = min(max_pn, horizon)
        self.apply_filter_date = False
        self.filter_date = datetime.datetime(2024, 1, 1)
        self.dar_usr = dar_usr
        self.cls_fields = cls_fields
        self.cls_format = cls_format
        self.cls_pa_fields = cls_pa_fields
        self.exec_mode = exec_mode
        self.source_data = source_data
        self.model_wb = model_wb
        self.horizon = horizon
        self.tx_params = tx_params
        self.etab = etab
        self.cls_nmd_tmp = cls_nmd_tmp
        self.type_rm = type_rm
        self.batch_size = batch_size
        self.load_template_column_names()
        self.load_columns()
        self.load_default_vars()
        self.load_mappings()
        self.model_mapper = Data_NMD_MODEL_MAPPER(cls_fields, cls_format, cls_pa_fields, source_data, dar_usr,
                                                  "PN", model_wb, horizon)

    def load_columns(self):
        self.MATURITY_ESTIMATION = "MATURITY_ESTIMATION"

        self.FLOW_OR_TARGET = "FLOW_OR_TARGET"

        self.ALLOCATION_KEY_RCO = Data_NMD_TEMPLATES.ALLOCATION_KEY
        self.TEMPLATE_WEIGHT_RCO = Data_NMD_TEMPLATES.TEMPLATE_WEIGHT_RCO
        self.ALLOCATION_KEY_PASS_ALM = [self.cls_pa_fields.NC_PA_ETAB, self.cls_pa_fields.NC_PA_DEVISE,
                                        self.cls_pa_fields.NC_PA_CONTRACT_TYPE,
                                        self.cls_pa_fields.NC_PA_MARCHE, self.cls_pa_fields.NC_PA_RATE_CODE,
                                        self.cls_pa_fields.NC_PA_PALIER]
        self.TEMPLATE_WEIGHT_PASS_ALM = "TEMPLATE_WEIGHT_PASS_ALM"

        self.NC_DEM = self.cls_pa_fields.NC_PA_DEM_CIBLE

        self.num_cols = ["M%s" % i for i in range(1, min(self.max_pn + 1, self.horizon + 1))]
        self.qual_cols = [self.cls_fields.NC_LDP_CURRENCY, self.cls_fields.NC_LDP_MARCHE,
                          self.cls_pa_fields.NC_PA_BASSIN, self.cls_pa_fields.NC_PA_MATUR,
                          self.cls_fields.NC_LDP_PALIER,
                          self.cls_pa_fields.NC_PA_IND03,
                          self.cls_fields.NC_LDP_FIRST_COUPON_DATE, self.cls_fields.NC_LDP_RATE_TYPE,
                          self.cls_fields.NC_LDP_CURVE_NAME, self.cls_fields.NC_LDP_TENOR,
                          self.cls_fields.NC_LDP_MKT_SPREAD, self.cls_fields.NC_LDP_ACCRUAL_BASIS,
                          self.cls_fields.NC_LDP_FIXING_NEXT_DATE, self.cls_fields.NC_LDP_CAPITALIZATION_RATE,
                          self.cls_fields.NC_LDP_MULT_SPREAD, self.cls_fields.NC_LDP_CALC_DAY_CONVENTION,
                          self.cls_fields.NC_LDP_DATE_SORTIE_GAP, self.cls_fields.NC_LDP_RATE,
                          self.cls_pa_fields.NC_PA_BILAN,
                          self.cls_fields.NC_LDP_FIXING_RULE, self.cls_fields.NC_LDP_FREQ_INT,
                          self.NC_NMD_ST_RESET_FREQUENCY, self.NC_NMD_ST_FIRST_FIXING_DATE,
                          self.cls_fields.NC_LDP_CONTRACT_TYPE, self.cls_fields.NC_LDP_RM_GROUP,
                          self.cls_fields.NC_LDP_RM_GROUP_PRCT, self.cls_fields.NC_LDP_FLOW_MODEL_NMD,
                          self.cls_fields.NC_LDP_FLOW_MODEL_NMD_GPTX, "NB_PARTS", "HAS_VOLATILE",
                          self.cls_fields.NC_LDP_NOMINAL, self.cls_fields.NC_LDP_MATUR_DATE,
                          self.cls_fields.NC_LDP_RATE_CODE, self.cls_fields.NC_LDP_FLOOR_STRIKE,
                          self.cls_fields.NC_LDP_ETAB, self.cls_fields.NC_LDP_DIM6,
                          self.NC_NMD_ST_RESET_FREQUENCY_TENOR, self.NC_NMD_ST_TENOR_BASED_FREQ,
                          self.ALLOCATION_KEY_RCO, self.cls_fields.NC_LDP_FLOW_MODEL_NMD_TCI,
                          self.cls_fields.NC_LDP_TCI_METHOD,
                          self.cls_fields.NC_LDP_TCI_FIXED_RATE_CODE, self.cls_fields.NC_LDP_TCI_FIXED_TENOR_CODE,
                          self.cls_fields.NC_LDP_TCI_VARIABLE_TENOR_CODE,
                          self.cls_fields.NC_LDP_TCI_VARIABLE_CURVE_CODE,
                          self.cls_fields.NC_LDP_SAVINGS_MODEL, self.cls_fields.NC_LDP_REDEMPTION_MODEL,
                          self.NC_NMD_ST_UNIT_OUTSTANDING, self.cls_fields.NC_LDP_TRADE_DATE]

    def load_mappings(self):
        self.map_reset_periodicity = {"M": "1M", "Q": "3M", "A": "1Y", "S": "6M"}

    def load_template_column_names(self):
        self.NC_NMD_ST_RESET_FREQUENCY = "RESET_FREQUENCY"
        self.NC_NMD_ST_FIRST_FIXING_DATE = "FIRST_FIXING_DATE"
        self.NC_NMD_ST_RESET_FREQUENCY_TENOR = "reset_freq_tenor".upper()
        self.NC_NMD_ST_TENOR_BASED_FREQ = "tenor_based_frequency".upper()
        self.NC_NMD_ST_UNIT_OUTSTANDING = "unit_outstanding_eur".upper()

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
             self.cls_fields.NC_LDP_CURVE_NAME: "",
             self.cls_fields.NC_LDP_TENOR: "1M", self.cls_fields.NC_LDP_MKT_SPREAD: 0,
             self.cls_fields.NC_LDP_FIXING_NEXT_DATE: ".",
             self.cls_fields.NC_LDP_CALC_DAY_CONVENTION: 1,
             self.cls_fields.NC_LDP_FIXING_PERIODICITY: "1M", self.cls_fields.NC_LDP_RATE_CODE: "",
             self.cls_fields.NC_LDP_CAP_STRIKE: np.nan, self.cls_fields.NC_LDP_FLOOR_STRIKE: np.nan,
             self.cls_fields.NC_LDP_MULT_SPREAD: 1.0, self.cls_fields.NC_LDP_RATE_TYPE: "FIXED",
             self.cls_fields.NC_LDP_MARCHE: "",
             self.cls_fields.NC_LDP_FTP_RATE: np.nan, self.cls_fields.NC_LDP_CURRENT_RATE: np.nan,
             self.cls_fields.NC_LDP_CAPI_MODE: "P",
             self.cls_fields.NC_LDP_BUY_SELL: "", self.cls_fields.NC_LDP_NB_CONTRACTS: 1,
             self.cls_fields.NC_LDP_PERFORMING: "F",
             self.cls_fields.NC_LDP_MATUR: "",
             self.cls_fields.NC_LDP_IS_CAP_FLOOR: "", self.cls_fields.NC_LDP_DATE_SORTIE_GAP: ".",
             self.cls_fields.NC_LDP_FIXING_NB_DAYS: 0, self.cls_fields.NC_LDP_IS_FUTURE_PRODUCT: True,
             self.cls_fields.NC_PRICING_CURVE: "", self.cls_fields.NC_LDP_GESTION: "",
             self.cls_fields.NC_LDP_PALIER: "",
             self.cls_fields.NC_LDP_FLOW_MODEL_NMD: "", self.cls_fields.NC_LDP_RM_GROUP: "",
             self.cls_fields.NC_LDP_RM_GROUP_PRCT: 0, self.cls_fields.NC_LDP_FIXING_RULE: "B",
             self.cls_fields.NC_LDP_FIRST_COUPON_DATE: ".", self.cls_fields.NC_LDP_TARGET_RATE: np.nan,
             self.cls_fields.NC_LDP_DIM6: "", self.cls_fields.NC_LDP_FTP_FUNDING_SPREAD: 0}

    def load_global_options_names(self):
        self.NR_PN_CALCULATION_MODE = "_mode_pn_nmd"
        self.NR_SCOPE_TO_EXCLUDE_FROM_RCO_FLOW = "_scope_to_exclude_rco_flow"
        self.NR_SIMULATION_TYPE = "_type_simul"

    def load_model_global_options(self):
        self.load_global_options_names()
        self.target_mode = ex.get_value_from_named_ranged(self.model_wb, self.NR_PN_CALCULATION_MODE)
        self.type_simul = ex.get_value_from_named_ranged(self.model_wb, self.NR_SIMULATION_TYPE).upper().strip()
        self.get_flux_calage = False if self.target_mode.strip().upper() == "ENCOURS CIBLE" \
            else self.type_simul == "DYNAMIQUE"
        self.is_calage = (self.type_simul == "CALAGE" and self.target_mode != "ENCOURS CIBLE")
        self.excluded_scope_from_flux = ex.get_value_from_named_ranged(self.model_wb,
                                                                       self.NR_SCOPE_TO_EXCLUDE_FROM_RCO_FLOW)
        if self.excluded_scope_from_flux == "":
            self.excluded_scope_from_flux = "[]"
        if self.is_calage:
            self.excluded_scope_from_flux = "[]"
        if self.target_mode.strip().upper() == "ENCOURS CIBLE":
            self.excluded_scope_from_flux = "[]"

    def create_groups_by_nb_parts(self, data_nmd_rm_all):
        data_nmd_rm_all["ICNE"] = np.where(
            data_nmd_rm_all[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("ICNE").values,
            "ICNE", "NOT_ICNE")
        data_nmd_rm_all["NEG/POS"] \
            = np.where(data_nmd_rm_all[self.cls_pa_fields.NC_PA_INDEX].str.contains("NEG"), "NEG", "POS")
        i = 1
        dic_data_nmd = {}
        dic_data_nmd_tmp = dict(tuple(data_nmd_rm_all.groupby(["NB_PARTS", "HAS_VOLATILE", "ICNE"], dropna=False)))
        cur_len = 0
        for key, val in dic_data_nmd_tmp.items():
            sub_dic_nmd = dict(tuple(dic_data_nmd_tmp[key].groupby(["NEG/POS", self.ALLOCATION_KEY_RCO])))
            for key_sub, val in sub_dic_nmd.items():
                neg_pos = key_sub[0]
                if neg_pos == "POS":
                    if sub_dic_nmd[key_sub].shape[0] * 2 + cur_len >= self.batch_size:
                        i = i + 1
                    name_batch = "BATCH_%s" % i
                    if not name_batch in dic_data_nmd:
                        dic_data_nmd[name_batch] = {}
                    for neg_pos_name in ["NEG", "POS"]:
                        key_sub_new = (neg_pos_name, key_sub[1])
                        name_batch_2 = "_".join([str(x) for x in key]) + "_" + neg_pos_name
                        if name_batch_2 in dic_data_nmd[name_batch]:
                            dic_data_nmd[name_batch][name_batch_2] \
                                = pd.concat([dic_data_nmd[name_batch][name_batch_2], sub_dic_nmd[key_sub_new]])
                        else:
                            dic_data_nmd[name_batch][name_batch_2] = sub_dic_nmd[key_sub_new].copy()

                        cur_len = sum([dic_data_nmd[name_batch][x].shape[0] for x in dic_data_nmd[name_batch].keys()])

        self.dic_data_nmd = dic_data_nmd.copy()

    def create_groups_by_nb_parts_no_neg(self, data_nmd_rm_all):
        data_nmd_rm_all["ICNE"] = np.where(
            data_nmd_rm_all[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("ICNE").values,
            "ICNE", "NOT_ICNE")
        i = 1
        dic_data_nmd = {}
        dic_data_nmd_tmp = dict(tuple(data_nmd_rm_all.groupby(["NB_PARTS", "HAS_VOLATILE", "ICNE"], dropna=False)))
        cur_len = 0
        for key, val in dic_data_nmd_tmp.items():
            sub_dic_nmd = dict(tuple(dic_data_nmd_tmp[key].groupby([self.ALLOCATION_KEY_RCO])))
            for key_sub, val in sub_dic_nmd.items():
                # neg_pos  = key_sub[0]
                # if neg_pos == "POS":
                if sub_dic_nmd[key_sub].shape[0] + cur_len > max(self.batch_size, sub_dic_nmd[key_sub].shape[0]):
                    i = i + 1
                name_batch = "BATCH_%s" % i
                if not name_batch in dic_data_nmd:
                    dic_data_nmd[name_batch] = {}
                # for neg_pos_name in ["NEG", "POS"]:
                # key_sub_new = key_sub[1]
                name_batch_2 = "_".join([str(x) for x in key])
                if name_batch_2 in dic_data_nmd[name_batch]:
                    dic_data_nmd[name_batch][name_batch_2] \
                        = pd.concat([dic_data_nmd[name_batch][name_batch_2], sub_dic_nmd[key_sub]])
                else:
                    dic_data_nmd[name_batch][name_batch_2] = sub_dic_nmd[key_sub].copy()

                cur_len = sum([dic_data_nmd[name_batch][x].shape[0] for x in dic_data_nmd[name_batch].keys()])

        self.dic_data_nmd = dic_data_nmd.copy()

    def merge_with_template_file(self, data_nmd_pn):

        data_nmd_pn \
            = data_nmd_pn.drop(
            self.cls_nmd_tmp.data_template_mapped.set_index(self.ALLOCATION_KEY_PASS_ALM).columns.tolist(),
            axis=1, errors="ignore")
        data_nmd_pn_mg = data_nmd_pn.join(self.cls_nmd_tmp.data_template_mapped.set_index(self.ALLOCATION_KEY_PASS_ALM),
                                          self.ALLOCATION_KEY_PASS_ALM, rsuffix='_TEMPLATE')

        cond_rate = np.where(data_nmd_pn_mg[self.cls_fields.NC_LDP_RATE_TYPE] == "FLOATING",
                             (data_nmd_pn_mg[self.cls_fields.NC_LDP_CURVE_NAME].fillna("").astype(str).isin(["", "nan"])).values,
                             (data_nmd_pn_mg[self.cls_fields.NC_LDP_RATE_CODE].fillna("").astype(str).isin(["", "nan"])).values)

        non_projetable = (((data_nmd_pn_mg[self.cls_fields.NC_LDP_CURRENCY].fillna("").astype(str).isin(["", "nan"])).values)
                          | ((data_nmd_pn_mg[self.cls_fields.NC_LDP_CONTRACT_TYPE].fillna("").astype(str).isin(["", "nan"])).values) |
                          (cond_rate))

        non_projetable_list = data_nmd_pn_mg.loc[non_projetable, self.ALLOCATION_KEY_PASS_ALM].values.tolist()
        if len(non_projetable_list) > 0:
            logger.warning(
                "Les clés suivantes sont éliminées car les informations de projection ne sont pas disponibles: %s"
                % non_projetable_list)
            data_nmd_pn_mg = data_nmd_pn_mg[~non_projetable].copy()

        if data_nmd_pn_mg[self.cls_fields.NC_LDP_CURRENCY].isnull().any():
            list_missing = data_nmd_pn_mg.loc[
                data_nmd_pn_mg[
                    self.cls_fields.NC_LDP_CURRENCY].isnull(), self.ALLOCATION_KEY_PASS_ALM].copy().values.tolist()
            if ~(data_nmd_pn_mg.loc[
                data_nmd_pn_mg[
                    self.cls_fields.NC_LDP_CURRENCY].isnull(), self.cls_pa_fields.NC_PA_CONTRACT_TYPE].str.contains(
                "ICNE")).any():
                logger.error("There are contracts missing in the template : %s" % list_missing)
                raise ValueError()
            else:
                logger.warning("There are contracts missing in the template : %s" % list_missing)
                data_nmd_pn_mg = data_nmd_pn_mg[~data_nmd_pn_mg[self.cls_fields.NC_LDP_CURRENCY].isnull()].copy()

        data_nmd_pn_mg[self.cls_pa_fields.NC_PA_INDEX + "_OLD"] = data_nmd_pn_mg[self.cls_pa_fields.NC_PA_INDEX].values
        # Pour les cas, où il y a plusieurs templates par produit
        data_nmd_pn_mg[self.cls_pa_fields.NC_PA_INDEX] \
            = (data_nmd_pn_mg[self.cls_pa_fields.NC_PA_INDEX] + '_'
               + data_nmd_pn_mg.groupby([self.cls_pa_fields.NC_PA_INDEX]).cumcount().astype(str))

        return data_nmd_pn_mg

    def recover_rco_target_levels(self, data_nmd_pn):
        data_template = self.cls_nmd_tmp.data_template_mapped[
            self.ALLOCATION_KEY_PASS_ALM + [self.ALLOCATION_KEY_RCO]].drop_duplicates()
        data_template = data_template.set_index(self.ALLOCATION_KEY_PASS_ALM)
        data_nmd_pn_mg = data_nmd_pn.join(data_template, on=self.ALLOCATION_KEY_PASS_ALM, rsuffix='_TEMPLATE')

        data_ag_rco = data_nmd_pn_mg[[self.ALLOCATION_KEY_RCO] + self.num_cols].copy()
        was_nan = np.isnan(data_ag_rco[self.num_cols].astype(float).values)
        data_ag_rco[self.num_cols] = data_ag_rco.groupby(self.ALLOCATION_KEY_RCO).transform(
            lambda x: x.astype(float).sum())
        data_ag_rco[self.num_cols] = np.where(was_nan, np.nan, data_ag_rco[self.num_cols].values)

        filter_ec_cible = self.get_encours_cible_scope(data_nmd_pn_mg)
        data_nmd_pn_mg.loc[~filter_ec_cible, self.num_cols] = data_ag_rco.loc[~filter_ec_cible, self.num_cols].values

        return data_nmd_pn_mg

    def format_fixing_parameters(self, data_dem):
        # pas de fixing_periodicity dans les NMDs
        data_dem[self.NC_NMD_ST_RESET_FREQUENCY] = data_dem[self.NC_NMD_ST_RESET_FREQUENCY].map(
            self.map_reset_periodicity)

        cases_freq = [(data_dem[self.NC_NMD_ST_TENOR_BASED_FREQ].values != "T"),
                      (data_dem[self.NC_NMD_ST_TENOR_BASED_FREQ].values == "T")]

        vals_freq = [data_dem[self.NC_NMD_ST_RESET_FREQUENCY], data_dem[self.NC_NMD_ST_RESET_FREQUENCY_TENOR]]

        data_dem[self.cls_fields.NC_LDP_FIXING_PERIODICITY] = np.select(cases_freq, vals_freq)

        data_dem[self.cls_fields.NC_LDP_FIXING_PERIODICITY] = \
            data_dem[self.cls_fields.NC_LDP_FIXING_PERIODICITY].mask(
                data_dem[self.cls_fields.NC_LDP_FIXING_RULE] == "R", "1M")

        data_dem[self.cls_fields.NC_LDP_FIXING_PERIODICITY] = data_dem[
            self.cls_fields.NC_LDP_FIXING_PERIODICITY].fillna("1M")

        data_dem[self.cls_fields.NC_LDP_FIXING_NEXT_DATE] = \
            data_dem[self.cls_fields.NC_LDP_FIXING_NEXT_DATE].mask(
                (data_dem[self.cls_fields.NC_LDP_FIXING_RULE] == "B").values,
                data_dem[self.NC_NMD_ST_FIRST_FIXING_DATE].values)

        data_dem[self.cls_fields.NC_LDP_FIXING_NEXT_DATE] = data_dem[self.cls_fields.NC_LDP_FIXING_NEXT_DATE].fillna(
            ".")

        return data_dem

    def format_for_calculator(self, data_dem):
        n = data_dem.shape[0]
        date_usr = np.array([self.dar_usr] * self.max_pn).astype("datetime64[M]").astype("datetime64[D]")
        months_add = np.arange(1, self.max_pn + 1)
        value_dates = pd.to_datetime(ut.add_months_date(date_usr, months_add)).strftime("%d/%m/%Y")
        data_dem = data_dem.rename(columns={month: date for month, date in zip(self.num_cols, value_dates)})
        data_dem = pd.melt(data_dem, id_vars=self.qual_cols + [self.cls_pa_fields.NC_PA_INDEX,
                                                               self.cls_pa_fields.NC_PA_INDEX + "_OLD"],
                           value_vars=value_dates.values.tolist(), var_name=self.cls_fields.NC_LDP_VALUE_DATE,
                           value_name="TARGET")

        if self.apply_filter_date:
            data_dem = data_dem[
                pd.to_datetime(data_dem[self.cls_fields.NC_LDP_VALUE_DATE],
                               format="%d/%m/%Y") < self.filter_date].copy()

        data_dem[self.cls_fields.NC_LDP_CONTRAT] = \
            np.char.add(np.char.add(data_dem[self.cls_pa_fields.NC_PA_INDEX].values.astype(str), "-PN"),
                        np.repeat(np.arange(1, 1 + self.max_pn), n).astype(str))

        data_dem[self.cls_fields.NC_LDP_NB_CONTRACTS] = 1.0
        data_dem[self.cls_fields.NC_LDP_GESTION] = ""
        data_dem[self.cls_fields.NC_LDP_OUTSTANDING] = 0.0

        data_dem = self.format_fixing_parameters(data_dem)

        data_dem = self.adapt_to_pel(data_dem)

        data_dem[self.cls_fields.NC_LDP_DATE_SORTIE_GAP] = data_dem[self.cls_fields.NC_LDP_DATE_SORTIE_GAP].fillna(".")
        data_dem[self.cls_fields.NC_LDP_CALC_DAY_CONVENTION] \
            = data_dem[self.cls_fields.NC_LDP_CALC_DAY_CONVENTION].astype(str).fillna("1").replace("nan", "1").astype(
            float).astype(int)

        data_dem = data_dem.sort_values([self.cls_fields.NC_LDP_CONTRAT, self.cls_fields.NC_LDP_RM_GROUP])

        return data_dem

    def adapt_to_pel(self, data_dem):
        is_pel = (data_dem[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("P-PEL")).values
        is_pel_pn = (data_dem[self.cls_fields.NC_LDP_CONTRACT_TYPE] == "P-PEL-PN").values
        is_pel_pn_c = (data_dem[self.cls_fields.NC_LDP_CONTRACT_TYPE] == "P-PEL-C-PN").values
        is_pel_stock = is_pel & (~(is_pel_pn | is_pel_pn_c))

        data_dem.loc[is_pel, self.cls_fields.NC_LDP_NOMINAL] = data_dem.loc[
            is_pel, self.NC_NMD_ST_UNIT_OUTSTANDING].values

        data_dem.loc[is_pel, self.cls_fields.NC_LDP_NB_CONTRACTS] = abs(
            1 / data_dem.loc[is_pel, self.NC_NMD_ST_UNIT_OUTSTANDING].values)

        is_floating = (data_dem[self.cls_fields.NC_LDP_RATE_TYPE] == "FLOATING").values
        data_dem.loc[is_pel, self.cls_fields.NC_LDP_CAPITALIZATION_RATE] = 1.0
        data_dem.loc[is_pel & is_floating, self.cls_fields.NC_LDP_FIXING_RULE] = "B"
        data_dem.loc[is_pel & is_floating, self.cls_fields.NC_LDP_RATE_CODE] = "TX_LIVPEL"

        self.load_taux_pel()

        val_date = pd.to_datetime(data_dem.loc[is_pel_pn | is_pel_pn_c, self.cls_fields.NC_LDP_VALUE_DATE],
                                  format="%d/%m/%Y").values

        diff_months = (val_date.astype("datetime64[M]")
                       - np.array(self.dar_usr).astype("datetime64[M]")).astype("timedelta64[M]") / np.timedelta64(1,
                                                                                                                   'M')
        diff_months = diff_months.astype(int) - 1

        data_dem.loc[is_pel_pn | is_pel_pn_c, self.cls_fields.NC_LDP_RATE] = self.tx_pel[:, diff_months].reshape(
            diff_months.shape[0]) * 100

        data_dem.loc[is_pel_pn | is_pel_pn_c, self.cls_fields.NC_LDP_FIXING_NEXT_DATE] \
            = pd.to_datetime(ut.add_months_date(val_date.astype("datetime64[D]"), 15 * 12)).strftime("%d/%m/%Y")

        data_dem.loc[is_pel_pn | is_pel_pn_c, self.cls_fields.NC_LDP_DATE_SORTIE_GAP] \
            = data_dem.loc[is_pel_pn | is_pel_pn_c, self.cls_fields.NC_LDP_FIXING_NEXT_DATE]

        data_dem.loc[~is_pel_stock, self.cls_fields.NC_LDP_TRADE_DATE] \
            = data_dem.loc[~is_pel_stock, self.cls_fields.NC_LDP_VALUE_DATE]

        data_dem.loc[is_pel_pn | is_pel_pn_c, self.cls_fields.NC_LDP_FIRST_COUPON_DATE] \
            = pd.to_datetime(
            [datetime.datetime(x + 1, 1, 1) for x in val_date.astype('datetime64[Y]').astype(int) + 1970]).strftime(
            "%d/%m/%Y")

        return data_dem

    def load_taux_pel(self):
        tx_curves = self.tx_params["curves_df"]["data"].copy()
        col_curve = self.tx_params["curves_df"]["curve_code"]
        col_tenor = self.tx_params["curves_df"]["tenor"]
        nums_cols = self.tx_params["curves_df"]["cols"]
        m1 = nums_cols[1]
        filter_tx_pel = ((tx_curves[col_curve] == self.tx_params["curves_df"]["curve_name_taux_pel"])
                         & (tx_curves[col_tenor] == self.tx_params["curves_df"]["tenor_taux_pel"]))
        tx_pel = tx_curves[filter_tx_pel]
        if len(tx_pel) == 0:
            msg = "No index name %s is present in the data" % self.tx_params["curves_df"]["curve_name_taux_pel"]
            logger.error(msg)
            raise ValueError(msg)
        self.tx_pel = np.array(tx_pel.loc[:, m1:])

    def create_negative_profile(self, data_nmd_pn):
        is_passif = data_nmd_pn[self.cls_pa_fields.NC_PA_BILAN].str.contains("PASSIF")
        data_nmd_pn[self.cls_fields.NC_LDP_NOMINAL] = np.where(is_passif, -1.0, 1.0)
        data_nmd_pn_neg = data_nmd_pn.copy()
        data_nmd_pn_neg[self.cls_pa_fields.NC_PA_INDEX] = data_nmd_pn_neg[self.cls_pa_fields.NC_PA_INDEX] + "_NEG"
        data_nmd_pn = pd.concat([data_nmd_pn, data_nmd_pn_neg])
        return data_nmd_pn

    def get_calage_data(self):
        if not "DATA" in self.source_data["CALAGE"]:
            input_calage_path = self.source_data["CALAGE"]["CHEMIN"]
            if not os.path.isfile(input_calage_path):
                logger.error("    Le fichier " + input_calage_path + " n'existe pas")
                raise ImportError("    Le fichier " + input_calage_path + " n'existe pas")

        if not "DATA" in self.source_data["CALAGE"]:
            data_calage = pd.read_csv(self.source_data["CALAGE"]["CHEMIN"],
                                      delimiter=self.source_data["CALAGE"]["DELIMITER"],
                                      decimal=self.source_data["CALAGE"]["DECIMAL"],
                                      engine='python', encoding="ISO-8859-1")
        else:
            data_calage = self.source_data["CALAGE"]["DATA"]

        data_calage = data_calage[[self.ALLOCATION_KEY_RCO] + self.num_cols].copy().set_index(self.ALLOCATION_KEY_RCO)
        data_calage.columns = [str(x) + "_FLUX" for x in self.num_cols]
        return data_calage

    def get_data_orig_with_parts_weights(self, data_nmd_rm):
        data_pn_nmd = data_nmd_rm[~data_nmd_rm[self.cls_pa_fields.NC_PA_INDEX].str.contains("NEG")]
        data_pn_nmd = data_pn_nmd.sort_values([self.cls_pa_fields.NC_PA_INDEX, self.cls_fields.NC_LDP_RM_GROUP])
        data_pn_nmd[self.FLOW_OR_TARGET] = "TARGET"
        if self.get_flux_calage:
            data_calage = self.get_calage_data()
            data_pn_nmd = data_pn_nmd.join(data_calage, on=self.ALLOCATION_KEY_RCO)
            data_pn_nmd[self.FLOW_OR_TARGET] = "FLOW"
            filter_all = self.get_encours_cible_scope(data_pn_nmd)
            data_pn_nmd.loc[filter_all, self.FLOW_OR_TARGET] = "TARGET"

            if data_pn_nmd["M1_FLUX"].isnull().any():
                manquants = data_pn_nmd[data_pn_nmd["M1_FLUX"].isnull()][self.ALLOCATION_KEY_RCO].unique().tolist()
                logger.error("Certains flux sont manquants : %s" % manquants)
                raise ValueError("Certains flux sont manquants  %s" % manquants)
        else:
            flux = pd.DataFrame(np.nan, columns=[str(x) + "_FLUX" for x in self.num_cols], index=data_pn_nmd.index)
            data_pn_nmd = pd.concat([data_pn_nmd, flux], axis=1)

        sum_new_tmp = [x + "_COEFF_MULT" for x in self.num_cols]
        data_pn_nmd[sum_new_tmp] = np.where(data_pn_nmd[self.num_cols].isnull(), 0, 1)

        return data_pn_nmd

    def get_encours_cible_scope(self, data_pn_nmd):
        filter_all = np.full((data_pn_nmd.shape[0]), False)
        scopes = eval(self.excluded_scope_from_flux)
        if len(scopes) > 0:
            for scope in scopes:
                filter = np.full((data_pn_nmd.shape[0]), True)
                for col_name, val in scope.items():
                    filter = filter & data_pn_nmd[col_name].isin(val).values
                filter_all = filter_all | filter
        return filter_all

    def get_non_rco_template_weights(self, data):
        if self.target_mode == "ENCOURS CIBLE" or (
                self.target_mode == "FLUX RCO" and self.excluded_scope_from_flux != []):
            if self.target_mode == "ENCOURS CIBLE":
                filter_apply = np.full(data.shape[0], True)
            else:
                filter_apply = self.get_encours_cible_scope(data)

            data.loc[filter_apply, self.TEMPLATE_WEIGHT_PASS_ALM] = 1.0
            data_sum = (data.loc[filter_apply, [self.cls_pa_fields.NC_PA_INDEX + "_OLD",
                                                self.cls_fields.NC_LDP_OUTSTANDING + "_ABS"]]
                        .groupby([self.cls_pa_fields.NC_PA_INDEX + "_OLD"])[self.cls_fields.NC_LDP_OUTSTANDING + "_ABS"]
                        .transform(lambda x: x.astype(float).sum()))

            data.loc[filter_apply, self.TEMPLATE_WEIGHT_PASS_ALM] \
                = ((data.loc[filter_apply, self.cls_fields.NC_LDP_OUTSTANDING + "_ABS"].values / data_sum)).fillna(1)

        return data

    def filter_unecessary_data(self, data):
        data = data[data[self.cls_pa_fields.NC_PA_IND03] == self.NC_DEM].copy()
        filter = (~data[self.num_cols].isnull().all(1)).values
        data = data[filter].copy()
        return data

    def filter_data(self, data):
        # data = data[data[self.cls_pa_fields.NC_PA_CONTRACT_TYPE].str.contains("A-ESC-VAL")].copy()
        # data = data[data["INDEX AGREG"].str.contains("FIX")].copy()
        #data = data[data[self.cls_fields.NC_LDP_CONTRACT_TYPE].isin(["P-PEL-2011"])].copy()
        #data = data[data[self.cls_fields.NC_LDP_ETAB].isin(["CEGEE"])].copy()
        return data

    # ####@profile
    def generate_templated_pn_data(self):
        logging.debug('    Lecture du fichier PN NMD')

        self.load_model_global_options()

        if not 'DATA' in self.source_data["LDP"]:
            data_nmd_pn = com.read_file(self.source_data, "LDP")
        else:
            data_nmd_pn = self.source_data["LDP"]["DATA"]

        data_nmd_pn = self.cls_format.upper_columns_names(data_nmd_pn)

        data_nmd_pn_filtered = self.filter_unecessary_data(data_nmd_pn)

        if self.exec_mode != "simul":
            data_nmd_pn_filtered = self.filter_data(data_nmd_pn_filtered)

        if self.target_mode == "FLUX RCO":
            data_nmd_pn_filtered = self.recover_rco_target_levels(data_nmd_pn_filtered)

        data_nmd_mg_tmp = self.merge_with_template_file(data_nmd_pn_filtered)

        data_nmd_mg_tmp = self.get_non_rco_template_weights(data_nmd_mg_tmp)

        data_nmd_pn_pos_neg = self.create_negative_profile(data_nmd_mg_tmp)

        data_nmd_rm = self.model_mapper.map_data_with_model_maps(data_nmd_pn_pos_neg, type_map="PN")

        data_nmd_pn_calc = self.format_for_calculator(data_nmd_rm.copy())

        data_nmd_pn_calc = self.cls_format.create_unvailable_variables(data_nmd_pn_calc, self.cls_fields.ldp_vars,
                                                                       self.default_ldp_vars)

        self.data_pn_nmd = self.get_data_orig_with_parts_weights(data_nmd_rm)

        self.create_groups_by_nb_parts_no_neg(data_nmd_pn_calc)
