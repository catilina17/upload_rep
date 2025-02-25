import datetime
import pandas as pd
from params import simul_params as sp
import modules.alim.parameters.general_parameters as gp
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import mappings.mapping_module as mp
from dateutil.relativedelta import relativedelta
import modules.alim.parameters.RZO_params as rzo_p
import modules.alim.formating_service.stock_formating_service.ONEY_SOCFIM.initial_formating as oney_st
import utils.excel_utils as ex
import utils.general_utils as gu
import modules.alim.lcr_nsfr_service.lcr_nsfr_module as lcr_nsfr
from calculateur.services.projection_services.nmd import run_calculator_stock_nmd as nmd_st
from calculateur.services.projection_services.nmd import run_calculator_pn_nmd as nmd_pn
from calculateur.services.projection_services.nmd import run_nmd_spreads as nmd_spread
from calculateur.data_transformers.data_in.nmd.class_nmd_templates import Data_NMD_TEMPLATES as cls_nmd_tmp
import modules.alim.parameters.user_parameters as up
import numpy as np
from mappings.mapping_module import map_data
from mappings import general_mappings as gma
import logging
from calculateur.services.projection_services.nmd import run_nmd_template as run_tmp
from calculateur.models.data_manager.data_format_manager.class_fields_manager import Data_Fields
from modules.scenario.rate_services import tx_referential as tx_ref
from modules.scenario.rate_services import maturities_referential as m_ref
from calculateur.rates_transformer.swap_rates_interpolator import Rate_Interpolator
from modules.alim.rates_service.rates_module import RatesManager
from params import version_params as vp

logger = logging.getLogger(__name__)

class PN_NMD():
    def __init__(self):
        self.load_spread_file_cols()
        self.load_encours_cible_file_cols()
        self.cls_fields = Data_Fields()

    def load_spread_file_cols(self):
        self.CASDEN_CONTRACT = "A/P-CASDEN"
        self.TX_CASDEN = "TX_CASDEN"
        self.TX_CASDEN2 = "TX_CASDEN2"
        self.NC_MARGES_CONTRACT_TYPE = "contract_type"
        self.NC_MARGES_INDEX_CODE = "index_code"
        self.NC_MARGES_ETAB = "etab"
        self.NC_MARGES_DARN_NUM = "darnum"
        self.NC_MARGES_NUMS_COLS = ["M0%s" %i if i<=9 else "M%s" %i for i in range(1,63)]

    def load_encours_cible_file_cols(self):
        self.EC_FILE_ETAB = "ETAB"
        self.EC_FILE_CURRENCY = "CCY_CODE"
        self.EC_FILE_CONTRACT_TYPE = "CONTRACT_TYPE"
        self.EC_FILE_FAMILY = "FAMILY"
        self.EC_FILE_RATE_TYPE = "RATE_TYPE"
        self.EC_FILE_NUMS_COLS = ["M%s" %i if i<=9 else "M%s" %i for i in range(1,63)]

    def change_casden_contract(self, data_marges):
        filtre_casden_cont = data_marges[self.NC_MARGES_CONTRACT_TYPE] == self.CASDEN_CONTRACT
        filtre_casden_tx1 = data_marges[self.NC_MARGES_INDEX_CODE] == self.TX_CASDEN
        filtre_casden_tx2 = data_marges[self.NC_MARGES_INDEX_CODE] == self.TX_CASDEN2
        casden = {}
        for i in range(1, 3):
            casden[i] = data_marges[filtre_casden_cont & eval("filtre_casden_tx" + str(i))].copy().reset_index(drop=True)
            casden_a = casden[i].copy()
            casden_p = casden[i].copy()
            casden_a2 = casden[i].copy()
            casden_p2 = casden[i].copy()
            if casden[i].shape[0] > 0:
                casden_a[self.NC_MARGES_CONTRACT_TYPE] = "A-CASDEN"
                casden_p[self.NC_MARGES_CONTRACT_TYPE] = "P-CASDEN"
                casden_a2[self.NC_MARGES_CONTRACT_TYPE] = "A-CASDEN2"
                casden_p2[self.NC_MARGES_CONTRACT_TYPE] = "P-CASDEN2"
            casden[i] = pd.concat([casden_a, casden_p, casden_a2, casden_p2])

        data_marges = data_marges[~filtre_casden_cont].copy()

        return pd.concat([data_marges, casden[1], casden[2]])


    def replace_rate_code_dav(self, data, col_rate_code):
        data[col_rate_code] = data[col_rate_code].replace("TX_SCDAVCORP", "EUREURIB3Mf").replace("TX_SCDAVPART", "CMS5Yf")
        return data

    def generate_tx_spread_from_file(self, nmd_pn_data):
        index_file = rzo_p.pn_rzo_files_name["PN-NMD-MARGE-INDEX"]
        logger.info("   Lecture de : " + index_file.split("\\")[-1])
        pn_tx_spread = pd.read_csv(index_file, delimiter="\\t", encoding='latin', engine="python", decimal=",")
        pn_tx_spread.drop([self.NC_MARGES_DARN_NUM], axis=1, inplace=True)

        pn_tx_spread = self.replace_rate_code_dav(pn_tx_spread, self.NC_MARGES_INDEX_CODE)

        nmd_pn_tx = nmd_pn_data.drop(self.EC_FILE_NUMS_COLS, axis=1).copy()

        nmd_pn_tx = self.format_contract_pel(nmd_pn_tx)

        nmd_pn_tx[pa.NC_PA_IND03] = pa.NC_PA_TX_SP

        pn_tx_spread = pn_tx_spread.rename(columns={"m" + str(i) if i >= 10 else "m0" + str(i): \
                                                        "M" + str(i) for i in range(1, pa.MAX_MONTHS_PN + 1)})

        tx_sp_cols = ["M" + str(i) for i in range(1, pa.MAX_MONTHS_PN + 1)]

        pn_tx_spread[tx_sp_cols] = pn_tx_spread[tx_sp_cols].fillna(0).astype(np.float64)

        pn_tx_spread = self.change_casden_contract(pn_tx_spread)

        pn_tx_spread = pn_tx_spread[[self.NC_MARGES_ETAB, self.NC_MARGES_CONTRACT_TYPE, self.NC_MARGES_INDEX_CODE] + tx_sp_cols].copy()

        pn_tx_spread = pn_tx_spread.groupby(by=[self.NC_MARGES_CONTRACT_TYPE, self.NC_MARGES_INDEX_CODE, self.NC_MARGES_ETAB],
                                            as_index=False).mean()

        nmd_pn_tx = mp.map_data(nmd_pn_tx, pn_tx_spread,
                                keys_mapping=[self.NC_MARGES_INDEX_CODE, self.NC_MARGES_CONTRACT_TYPE, self.NC_MARGES_ETAB],
                                keys_data=[pa.NC_PA_RATE_CODE + "_NEW", pa.NC_PA_CONTRACT_TYPE + "_NEW", pa.NC_PA_ETAB],
                                cols_mapp=tx_sp_cols, option="", error_mapping=False, no_map_value=0)

        nmd_pn_data = pd.concat([nmd_pn_data, nmd_pn_tx])

        return nmd_pn_data


    def format_contract_pel(self, data_tmp):
        is_pel = data_tmp[pa.NC_PA_CONTRACT_TYPE].str.contains("P-PEL-")
        pel_ancien = ['P-PEL-C-5,25', 'P-PEL-C-4,25', 'P-PEL-C-4', 'P-PEL-C-3,6', 'P-PEL-C-4,5', 'P-PEL-C-3,5',
                      'P-PEL-ANCIEN', 'P-PEL-6', 'P-PEL-5,25', 'P-PEL-4,25', 'P-PEL-4', 'P-PEL-3,60', 'P-PEL-3,6',
                      'P-PEL-4,50', 'P-PEL-4,5', 'P-PEL-3,50', 'P-PEL-3,5']
        is_not_pel_ancien = ~data_tmp[pa.NC_PA_CONTRACT_TYPE].isin(pel_ancien)

        data_tmp[pa.NC_PA_CONTRACT_TYPE + "_NEW"]= np.where(is_pel, "P-PEL-*", data_tmp[pa.NC_PA_CONTRACT_TYPE])

        data_tmp[pa.NC_PA_RATE_CODE + "_NEW"] = np.where(is_pel & is_not_pel_ancien, "TX_LIVPEL", data_tmp[pa.NC_PA_RATE_CODE])

        return data_tmp

    def final_formating(self, nmd_pn_data):
        keys_order = [pa.NC_PA_BILAN, pa.NC_PA_CLE]
        indics_ordered = [pa.NC_PA_DEM_CIBLE, pa.NC_PA_TX_SP, pa.NC_PA_TX_CIBLE]
        nmd_pn_data = gu.add_empty_indics(nmd_pn_data, [pa.NC_PA_TX_CIBLE], pa.NC_PA_IND03, pa.NC_PA_DEM_CIBLE,
                                          pa.NC_PA_COL_SORTIE_NUM_PN, \
                                          order=True, indics_ordered=indics_ordered, keys_order=keys_order)

        INDEXO = np.array(["NMD" + str(i) for i in range(1, int(nmd_pn_data.shape[0] / 3) + 1)])
        INDEXO = np.repeat(INDEXO, 3)
        nmd_pn_data[pa.NC_PA_INDEX] = INDEXO

        return nmd_pn_data[pa.NC_PA_COL_SORTIE_QUAL + pa.NC_PA_COL_SORTIE_NUM_PN].copy()


    def final_formating2(self, nmd_pn_data):
        keys_order = [pa.NC_PA_BILAN, pa.NC_PA_CLE]
        indics_ordered = [pa.NC_PA_DEM_CIBLE, pa.NC_PA_DEM, pa.NC_PA_TX_SP, pa.NC_PA_TX_CIBLE]
        nmd_pn_data = gu.add_empty_indics(nmd_pn_data, [], pa.NC_PA_IND03, pa.NC_PA_DEM_CIBLE,
                                          pa.NC_PA_COL_SORTIE_NUM_PN, \
                                          order=True, indics_ordered=indics_ordered, keys_order=keys_order)

        INDEXO = np.array(["NMD" + str(i) for i in range(1, int(nmd_pn_data.shape[0] / 4) + 1)])
        INDEXO = np.repeat(INDEXO, 4)
        nmd_pn_data[pa.NC_PA_INDEX] = INDEXO

        return nmd_pn_data[pa.NC_PA_COL_SORTIE_QUAL + pa.NC_PA_COL_SORTIE_NUM_PN].copy()


    def read_and_parse_nmd_pn_file(self):
        file_encours = rzo_p.pn_rzo_files_name["PN-NMD-ENCOURS-CIBLE"]
        logger.info("   Lecture de : " + file_encours.split("\\")[-1])
        delimiter = ";" if vp.version_sources == "csv" else "\t"
        decimal = "," if vp.version_sources == "csv" else ","
        thousands = "" if vp.version_sources == "csv" else " "
        data_pn_nmd = pd.read_csv(file_encours, delimiter=delimiter, engine="python", decimal=decimal,
                                     thousands=thousands)
        #data_pn_nmd = data_pn_nmd[data_pn_nmd[self.EC_FILE_CONTRACT_TYPE].str.contains("P-PEL")].copy()
        data_pn_nmd = data_pn_nmd[~data_pn_nmd.loc[:, self.EC_FILE_NUMS_COLS].isnull().all(1)].copy()
        return data_pn_nmd


    def merge_with_templates_data(self, data_pn_nmd, data_template):
        data_pn_nmd["KEY"] = (data_pn_nmd[self.EC_FILE_ETAB].fillna("*")  + "_" + data_pn_nmd[self.EC_FILE_CURRENCY].fillna("*") + "_" + data_pn_nmd[self.EC_FILE_CONTRACT_TYPE].fillna("*")
                              + "_" + data_pn_nmd[self.EC_FILE_FAMILY].fillna("*") + "_" + data_pn_nmd[self.EC_FILE_RATE_TYPE].fillna("*"))
        cols_keep =  ["KEY", self.EC_FILE_ETAB, self.EC_FILE_CURRENCY, self.EC_FILE_CONTRACT_TYPE, self.EC_FILE_FAMILY, self.EC_FILE_RATE_TYPE]
        data_pn_nmd = data_pn_nmd[cols_keep + self.EC_FILE_NUMS_COLS].copy()
        nmd_template_join = data_template.copy()

        filter_err = ~data_pn_nmd["KEY"].isin(nmd_template_join[cls_nmd_tmp.ALLOCATION_KEY].values.tolist())
        if filter_err.any():
            msg = "Certains contrats PN NMDs n'existent pas dans les TEMPLATES: %s" % list(set(data_pn_nmd[filter_err]["KEY"].values.tolist()))
            logger.warning(msg)

        col_keep = [self.cls_fields.NC_LDP_ETAB, self.cls_fields.NC_LDP_CURRENCY, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                    self.cls_fields.NC_LDP_MARCHE, self.cls_fields.NC_LDP_RATE_CODE, self.cls_fields.NC_LDP_PALIER]
        nmd_template_join = nmd_template_join[[cls_nmd_tmp.ALLOCATION_KEY, cls_nmd_tmp.TEMPLATE_WEIGHT_RCO,
                                               cls_nmd_tmp.TEMPLATE_WEIGHT_REAL] + col_keep].copy()

        data_pn_nmd_merged = nmd_template_join.merge(data_pn_nmd, left_on=cls_nmd_tmp.ALLOCATION_KEY, how='left',
                                                     right_on="KEY", suffixes=('', '_y'))

        n = data_pn_nmd_merged.shape[0]
        data_pn_nmd_merged[self.EC_FILE_NUMS_COLS] = (data_pn_nmd_merged[self.EC_FILE_NUMS_COLS].values
                                               * data_pn_nmd_merged[cls_nmd_tmp.TEMPLATE_WEIGHT_REAL].values.reshape(n, 1))

        return data_pn_nmd_merged

    def format_bilan(self, data):
        filtres = [data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "A-",
                   data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "P-", \
                   (data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[-2:] == "-A") & (data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "HB"),
                   (data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[-2:] == "-P") & (data[self.cls_fields.NC_LDP_CONTRACT_TYPE].str[:2] == "HB")]
        choices = ["B ACTIF", "B PASSIF", "HB ACTIF", "HB PASSIF"]
        data[pa.NC_PA_BILAN] = np.select(filtres, choices)

        return data

    def map_and_agregate_nmd_data(self, data_pn_nmd):

        data_pn_nmd = data_pn_nmd.rename(columns={self.cls_fields.NC_LDP_MARCHE: pa.NC_PA_MARCHE,
                                                  self.cls_fields.NC_LDP_PALIER: self.cls_fields.NC_LDP_PALIER + "_DET",
                                                  self.cls_fields.NC_LDP_RATE_CODE: pa.NC_PA_RATE_CODE,
                                                  self.cls_fields.NC_LDP_CURRENCY: pa.NC_PA_DEVISE})

        data_pn_nmd[self.cls_fields.NC_LDP_PALIER + "_DET"] = data_pn_nmd[self.cls_fields.NC_LDP_PALIER + "_DET"].fillna("-")
        data_pn_nmd = gu.force_integer_to_string(data_pn_nmd, self.cls_fields.NC_LDP_PALIER + "_DET")

        cles_data = {1: [pa.NC_PA_BILAN, self.cls_fields.NC_LDP_CONTRACT_TYPE],
                     2: [pa.NC_PA_RATE_CODE],  3: [self.cls_fields.NC_LDP_PALIER + "_DET"]}

        mappings = {1: "CONTRATS",  2: "INDEX_AGREG", 3: "PALIER"}

        for i in range(1, 4):
            data_pn_nmd = map_data(data_pn_nmd, gma.map_pass_alm[mappings[i]], keys_data=cles_data[i],
                                   name_mapping="PN DATA vs.")
        cols_to_drop =  [self.cls_fields.NC_LDP_CONTRACT_TYPE, cls_nmd_tmp.TEMPLATE_WEIGHT_RCO,
                         self.cls_fields.NC_LDP_PALIER + "_DET", cls_nmd_tmp.TEMPLATE_WEIGHT_REAL]
        data_pn_nmd = data_pn_nmd.drop(cols_to_drop, axis=1)
        qual_cols = [x for x in data_pn_nmd.columns if x not in self.EC_FILE_NUMS_COLS]
        data_pn_nmd = data_pn_nmd.sort_values(qual_cols).groupby(qual_cols, as_index=False, dropna=False).sum(numeric_only=True, min_count=1)

        return data_pn_nmd

    def map_lcr_nsfr_coeff(self, pn_encours):
        pn_encours[pa.NC_PA_LCR_TIERS] = "-"
        pn_encours[pa.NC_PA_LCR_TIERS_SHARE] = 100
        if up.map_lcr_nsfr:
            logger.info("MAPPING DES DONNEES DE PN ECH avec RAY")
            pn_encours = lcr_nsfr.map_lcr_tiers_and_share(pn_encours)
        return pn_encours

    def mapping_consolidation_liquidite(self, data):
        keys_liq_BC = [pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_BOOK, pa.NC_PA_LCR_TIERS]
        keys_EM = [pa.NC_PA_BILAN, pa.NC_PA_Regroupement_1, pa.NC_PA_MATUR]

        """ MAPPING LIQUIDITE BILAN CASH """
        cles_a_combiner = keys_liq_BC
        mapping = gma.mapping_liquidite["LIQ_BC"]
        data = gu.gen_combined_key_col(data, mapping["TABLE"], cols_key=cles_a_combiner, symbol_any="-",
                                       name_col_key="CONTRAT_", set_index=False)
        data = map_data(data, mapping, keys_data=["CONTRAT_"], name_mapping="STOCK/PN DATA vs.")
        data = data.drop(["CONTRAT_"], axis=1)

        """ AUTRES mappings LIQUIDITE """
        keys_liq_IG = [pa.NC_PA_BILAN, pa.NC_PA_Bilan_Cash, pa.NC_PA_BASSIN, pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_PALIER]
        keys_liq_CT = [pa.NC_PA_BILAN, pa.NC_PA_Regroupement_1, pa.NC_PA_BASSIN, pa.NC_PA_PALIER]
        keys_liq_FI = [pa.NC_PA_BILAN, pa.NC_PA_Regroupement_1, pa.NC_PA_BASSIN, pa.NC_PA_Bilan_Cash, pa.NC_PA_CONTRACT_TYPE,
                       "IG/HG Social"]
        keys_liq_SC = [pa.NC_PA_Affectation_Social]
        data["IG/HG Social"] = np.where(data[pa.NC_PA_PALIER] == "-", "HG", "IG")

        keys_data = [keys_EM, keys_liq_IG, keys_liq_CT, keys_liq_FI, keys_liq_SC]
        mappings = ["LIQ_EM", "LIQ_IG", "LIQ_CT", "LIQ_FI", "LIQ_SC"]
        for i in range(0, len(mappings)):
            mapping = gma.mapping_liquidite[mappings[i]]
            key_data = keys_data[i]
            override = False if mappings[i] == "LIQ_SC" else True
            error_mapping = True if mappings[i] == "LIQ_SC" else False
            data = map_data(data, mapping, keys_data=key_data, override=override, error_mapping=error_mapping,
                            name_mapping="STOCK/PN DATA vs.")

        """ MAPPING NSFR """
        cles_a_combiner = [pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_LCR_TIERS]
        mapping = gma.mapping_liquidite["NSFR"]
        data = gu.gen_combined_key_col(data, mapping["TABLE"], cols_key=cles_a_combiner, symbol_any="*",
                                       name_col_key="CONTRAT_", set_index=False)
        data = map_data(data, mapping, keys_data=["CONTRAT_"], name_mapping="STOCK/PN DATA vs.")
        data = data.drop(["CONTRAT_", "IG/HG Social"], axis=1)

        return data

    def add_map_cols(self, pn_encours):
        pn_encours[pa.NC_PA_BASSIN] = up.current_etab
        pn_encours[pa.NC_PA_BOOK] = "-"
        pn_encours[pa.NC_PA_MATUR] = "-"
        pn_encours[pa.NC_PA_SCOPE] = gp.MNI_AND_LIQ
        pn_encours[pa.NC_PA_IND03] = pa.NC_PA_DEM_CIBLE
        return pn_encours

    def add_missing_cols(self, pn_nmd_data):
        missing_indics_cles = [x for x in pa.NC_PA_CLE_OUTPUT if not x in pn_nmd_data.columns.tolist()]

        pn_nmd_data = pd.concat([pn_nmd_data, pd.DataFrame([["-"] * len(missing_indics_cles)], \
                                                           index=pn_nmd_data.index,
                                                           columns=missing_indics_cles)], axis=1)

        CLE = pd.DataFrame(pn_nmd_data[pa.NC_PA_CLE_OUTPUT].astype(str).apply(lambda x: "_".join(x), axis=1).copy(), \
                           index=pn_nmd_data.index, columns=[pa.NC_PA_CLE])

        pn_nmd_data = pd.concat([pn_nmd_data, CLE], axis=1)

        pn_nmd_data = pn_nmd_data.sort_values([pa.NC_PA_BILAN, pa.NC_PA_CLE])

        col_sortie = [x for x in pa.NC_PA_COL_SORTIE_QUAL if x != pa.NC_PA_IND03]

        missing_indics = [x for x \
                          in col_sortie \
                          if not x in pn_nmd_data.columns.tolist()]

        pn_nmd_data = pd.concat([pn_nmd_data, pd.DataFrame([["-"] * len(missing_indics)], index=pn_nmd_data.index,
                                                           columns=missing_indics)], axis=1)

        return pn_nmd_data

    def format_num_cols(self, data):
        n = data.shape[0]
        data[self.EC_FILE_NUMS_COLS] = (data[self.EC_FILE_NUMS_COLS].values
                                        * np.where(data[pa.NC_PA_BILAN].str.contains("PASSIF").values.reshape(n, 1), -1, 1))
        return data

    def format_data(self, data_pn_nmd_tmp):
        join_keys = [self.EC_FILE_CURRENCY, self.EC_FILE_CONTRACT_TYPE, self.EC_FILE_FAMILY, self.EC_FILE_RATE_TYPE]
        cols_cible = [self.cls_fields.NC_LDP_CURRENCY, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                      self.cls_fields.NC_LDP_MARCHE, self.cls_fields.NC_LDP_RATE_TYPE]
        data_pn_nmd_tmp = data_pn_nmd_tmp.rename(columns={x: y for x, y in zip(join_keys, cols_cible)})
        data_pn_nmd_tmp = data_pn_nmd_tmp.rename(columns={self.EC_FILE_ETAB: self.cls_fields.NC_LDP_ETAB})
        return data_pn_nmd_tmp

    def generate_templates(self, data_pn_nmd):
        if len(data_pn_nmd) > 0:
            data_pn_nmd_f = self.format_data(data_pn_nmd.copy())
        else:
            data_pn_nmd_f = data_pn_nmd.copy()

        source_data = self.get_nmd_source_data(data_pn_nmd_f)

        rco_allocation_key = True if up.current_etab != "ONEY" else False

        cls_template = run_tmp.run_nmd_template_getter(source_data, up.current_etab, up.dar, format_data=False,
                                                       save=False, rco_allocation_key=rco_allocation_key)

        source_data["MODELS"]["NMD"]["DATA"].Close(False)

        cls_template.data_template_mapped["BASSIN"] = up.current_etab

        return cls_template

    def format_template_data(self, data_template):
        template_data_ag = data_template.copy()

        #template_data_ag = template_data_ag[
        #    template_data_ag[self.cls_fields.NC_LDP_CONTRACT_TYPE].str.contains("P-PEL")].copy()

        cle = [self.cls_fields.NC_LDP_ETAB, self.cls_fields.NC_LDP_CURRENCY, self.cls_fields.NC_LDP_CONTRACT_TYPE,
               self.cls_fields.NC_LDP_MARCHE, self.cls_fields.NC_LDP_RATE_CODE, self.cls_fields.NC_LDP_PALIER]

        template_data_ag = (template_data_ag[[cls_nmd_tmp.ALLOCATION_KEY] + cle +
                                            [cls_nmd_tmp.TEMPLATE_WEIGHT_RCO, cls_nmd_tmp.TEMPLATE_WEIGHT_REAL]]
                            .groupby([cls_nmd_tmp.ALLOCATION_KEY] + cle, as_index=False, dropna=False).sum())
        return template_data_ag

    def get_model_wb(self, model_file_path):
        model_wb = None
        model_wb = ex.try_close_open(model_file_path, read_only=True)
        ex.unfilter_all_sheets(model_wb)
        return model_wb

    def get_nmd_source_data(self, data_pn_nmd):
        source_data = {}
        source_data["STOCK"] = {}
        source_data["MODELS"] = {}

        source_data["STOCK"] ["LDP"] = {}
        if up.current_etab != "ONEY":
            source_data["STOCK"] ["LDP"]["CHEMIN"] = up.nmd_st_files_name["ST-NMD"]
            source_data["STOCK"] ["LDP"]["DELIMITER"] = "\t"
            source_data["STOCK"] ["LDP"]["DECIMAL"] = "."
        else:
            source_data["STOCK"]  = self.get_oney_formated_stock(source_data["STOCK"] )
        source_data["STOCK"]["MARGE-INDEX"] = {}

        source_data["MODELS"]["NMD"] = {}
        source_data["MODELS"]["PEL"] = {}
        source_data["MODELS"]["NMD"]["DATA"] = self.get_model_wb(up.modele_nmd_file_path)
        source_data["MODELS"]["PEL"]["DATA"] = self.get_model_wb(up.modele_pel_file_path)

        source_data["PN"] = {}
        source_data["PN"]["LDP"] = {}
        source_data["PN"]["LDP"]["DATA"] = data_pn_nmd.copy()

        return source_data

    def get_oney_formated_stock(self, source_data_st):
        source_data_st["LDP"]["CHEMIN"] = up.main_files_name
        source_data_st["LDP"]["DELIMITER"] = "\t"
        source_data_st["LDP"]["DECIMAL"] = ","
        data_st_oney = oney_st.data_st_oney
        data_st_oney = map_data(data_st_oney, gma.map_pass_alm["CONTRATS"],
                                keys_data=[pa.NC_PA_BILAN, gp.NC_CONTRACT_TEMP],
                                name_mapping="DATA STOCK ONEY vs.", error_mapping=False)
        data_st_oney_nmd = data_st_oney[data_st_oney[pa.NC_PA_isECH] == "N"].copy()
        data_st_oney_nmd["ETAB"] = "ONEY"
        keys_data = ["ETAB", gp.NC_CONTRACT_TEMP, pa.NC_PA_RATE_CODE, pa.NC_PA_DEVISE, pa.NC_PA_MARCHE]
        data_st_oney_nmd = gu.gen_combined_key_col(data_st_oney_nmd, gma.mapping_PN["mapping_template_nmd"]["TABLE"],
                                                   cols_key=keys_data, symbol_any="*", name_col_key="CLE_",
                                                   set_index=False)
        data_st_oney_nmd = map_data(data_st_oney_nmd, gma.mapping_PN["mapping_template_nmd"],
                                    keys_data=["CLE_"], name_mapping="STOCK DATA vs.")

        for col in data_st_oney_nmd.columns.tolist():
            if col[-5:]=="_CALC" and (data_st_oney_nmd[col].str.replace(" ","") == "[DAR]+1").any():
                data_st_oney_nmd[col[:-5]] = datetime.datetime.strftime(up.dar + relativedelta(days=1), "%d/%m/%Y")

        data_st_oney_nmd[self.cls_fields.NC_LDP_OUTSTANDING] = data_st_oney_nmd["LEF_M0"]
        data_st_oney_nmd[self.cls_fields.NC_LDP_RATE_TYPE] = np.where(
            data_st_oney_nmd[pa.NC_PA_RATE_CODE].str.contains("FIXE"),
            "FIXED", "FLOATING")

        join_keys = [pa.NC_PA_DEVISE, gp.NC_CONTRACT_TEMP, pa.NC_PA_MARCHE,
                     self.cls_fields.NC_LDP_RATE_TYPE, gp.NC_PALIER_TEMP, pa.NC_PA_RATE_CODE]
        cols_cible = [self.cls_fields.NC_LDP_CURRENCY, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                      self.cls_fields.NC_LDP_MARCHE, self.cls_fields.NC_LDP_RATE_TYPE, self.cls_fields.NC_LDP_PALIER,
                      self.cls_fields.NC_LDP_RATE_CODE]
        data_st_oney_nmd = data_st_oney_nmd.rename(columns={x: y for x, y in zip(join_keys, cols_cible)})
        data_st_oney_nmd = data_st_oney_nmd.rename(columns={self.EC_FILE_ETAB: self.cls_fields.NC_LDP_ETAB})


        source_data_st["LDP"]["DATA"] = data_st_oney_nmd
        return source_data_st

    def get_calculator_params(self, sc_curves_df, data_pn_nmd, data_template_mapped):
        max_pn = pa.MAX_MONTHS_PN
        cols_ri = ["M" + str(i) for i in range(0, tx_ref.NB_PROJ_TAUX + 1)]
        sc_curves_df[cols_ri] = sc_curves_df[cols_ri] / 100
        curve_accruals_map = gma.mapping_taux["CURVES_BASIS_CONV"]
        tx_params = {"curves_df": {"data": sc_curves_df, "cols": cols_ri, "max_proj": tx_ref.NB_PROJ_TAUX,
                                   "curve_code": "CODE COURBE", "tenor": "MATURITE",
                                   "maturity_to_days": m_ref.maturity_to_days_360, "curve_name_taux_pel": "TAUX_PEL",
                                   "tenor_taux_pel": "12M1D"},
                     "accrual_map": {'data': curve_accruals_map, "accrual_conv_col": "ACCRUAL_CONVERSION",
                                     "type_courbe_col": "TYPE DE COURBE", "accrual_method_col": "ACCRUAL_METHOD",
                                     "alias": "ALIAS",
                                     "standalone_const": "Standalone index", "curve_name": "CURVE_NAME"},
                     "ZC_DATA": {"data": []},
                     "map_pricing_curves": {"data": [],
                                            "col_pricing_curve": "COURBE PRICING"},
                     "map_index_curve_tenor": {"data": [],
                                               "col_curve": "CURVE_NAME", "col_tenor": "TENOR"},
                     "tci_vals": {"data":[]},
                     "rate_input_map": {"data": [], "map_index_col": "",
                                        "map_key_cols": ['currency'.upper(), "rate_code".upper()]}, }

        rate_int_cls = Rate_Interpolator()
        tx_params["dic_tx_swap"] = rate_int_cls.interpolate_curves(tx_params)

        source_data =  self.get_nmd_source_data(data_pn_nmd)
        cls_nmd_tmp = type('MyClass', (object,), {'content':{}})()
        cls_nmd_tmp.data_template_mapped = data_template_mapped

        etab = up.current_etab
        horizon = pa.MAX_MONTHS_PN

        return source_data, tx_params, max_pn, etab, horizon, cls_nmd_tmp

    def run_calage_simulation(self, data_pn_nmd, data_template_mapped):
        logger.info("   Démarrage de la simulation de calage avec le scénario @BASELINE1")
        sc_curves_df = RatesManager.get_sc_df(up.sc_ref_nmd)

        nmd_source_files, tx_params, max_pn, etab, horizon, cls_temp\
            = self.get_calculator_params(sc_curves_df, data_pn_nmd, data_template_mapped)

        logger.info("     Projection des NMDs STOCK")

        cls_nmd_spreads = nmd_spread.run_nmd_spreads(etab, horizon, nmd_source_files, cls_temp)

        cls_ag_st\
            = nmd_st.run_calculator_nmd_stock(up.dar, horizon, nmd_source_files, "nmd_st",
                                              cls_nmd_spreads=cls_nmd_spreads, tx_params=tx_params,
                                              exit_indicators_type=["GPLIQ"], agregation_level="NMD_TEMPLATE",
                                              with_dyn_data=True, with_pn_data=True, batch_size=10000,
                                              output_mode="dataframe")

        logger.info("     Run de calage sur les PNs NMD")
        pn_to_generate \
            = nmd_pn.run_calculator_pn_nmd(up.dar, horizon, nmd_source_files, "nmd_pn", etab,
                                           cls_temp, tx_params=tx_params, cls_nmd_spreads=cls_nmd_spreads,
                                           exit_indicators_type=["GPLIQ"], agregation_level="NMD_DT",
                                           with_dyn_data=True, compiled_indics_st=cls_ag_st.compiled_indics,
                                           type_rm="NORMAL", batch_size=10000, output_data_type="pn_monhtly_flow",
                                           output_mode = "dataframe")

        nmd_source_files["MODELS"]["NMD"]["DATA"].Close(False)
        nmd_source_files["MODELS"]["PEL"]["DATA"].Close(False)
        pn_to_generate["M0"] = np.nan
        pn_to_generate["IND03"] = pa.NC_PA_DEM_CIBLE
        num_cols = [x for x in pa.NC_PA_COL_SORTIE_NUM_PN if x!="M0"]
        pn_to_generate[num_cols] = pn_to_generate[num_cols].astype(np.float64).fillna(0)
        pn_to_generate[pa.NC_PA_SCENARIO_REF] = "RCO_BASELINE1"
        flux_pn = pn_to_generate[[cls_nmd_tmp.ALLOCATION_KEY, pa.NC_PA_SCENARIO_REF, pa.NC_PA_IND03] + pa.NC_PA_COL_SORTIE_NUM_PN].copy()

        return flux_pn

    def generate_stock_NMD_templates(self):
        logger.info("TRAITEMENT DES TEMPLATES NMD")
        cls_tmp = self.generate_templates([])
        return  cls_tmp.data_template_mapped

    def format_RZO_PN_NMD(self):
        logger.info("TRAITEMENT DES PN NMD")
        data_pn_nmd = self.read_and_parse_nmd_pn_file()
        cls_template = self.generate_templates(data_pn_nmd)
        data_template_ag = self.format_template_data(cls_template.data_template)
        data_pn_nmd = self.merge_with_templates_data(data_pn_nmd, data_template_ag)
        if data_pn_nmd.shape[0] > 0:
            data_pn_nmd = self.format_bilan(data_pn_nmd)
            data_pn_nmd = self.format_num_cols(data_pn_nmd)
            data_pn_nmd = self.map_and_agregate_nmd_data(data_pn_nmd)
            data_pn_nmd = self.map_lcr_nsfr_coeff(data_pn_nmd)
            data_pn_nmd = self.add_map_cols(data_pn_nmd)
            data_pn_nmd = self.mapping_consolidation_liquidite(data_pn_nmd)
            data_pn_nmd = self.add_missing_cols(data_pn_nmd)
            data_pn_nmd = self.generate_tx_spread_from_file(data_pn_nmd)
            data_pn_nmd = self.final_formating(data_pn_nmd)
            if True or not sp.mode.upper() != "debug":
                data_flux_calage = self.run_calage_simulation(data_pn_nmd, cls_template.data_template_mapped)
            else:
                data_flux_calage = []
            return data_pn_nmd, data_flux_calage, cls_template.data_template_mapped
        else:
            return [], [], []
