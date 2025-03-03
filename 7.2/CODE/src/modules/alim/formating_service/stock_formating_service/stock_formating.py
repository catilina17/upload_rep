import mappings.mapping_functions as mp
import importlib

""" POUR PYINSTALLER"""
from modules.alim.formating_service.stock_formating_service.RZO.initial_formating import RZO_Formater
from modules.alim.formating_service.stock_formating_service.NTX_SEF.initial_formating import NTX_SEF_Formater
from modules.alim.formating_service.stock_formating_service.ONEY_SOCFIM.initial_formating import ONEY_Formater
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import utils.general_utils as gu
import modules.alim.parameters.general_parameters as gp
import numpy as np
import pandas as pd
import modules.alim.lcr_nsfr_service.lcr_nsfr_module as lcr_nsfr
import logging
from warnings import simplefilter

global lcr_nsfr_data

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


class StockFormater():
    def __init__(self, cls_user_params):
        self.up = cls_user_params

    def read_and_format_stock_data(self):
        if self.up.category == "NTX_SEF":
            init_f = NTX_SEF_Formater(self.up)
        elif self.up.category == "ONEY_SOCFIM":
            init_f = ONEY_Formater(self.up)
        else:
            init_f = RZO_Formater(self.up)

        STOCK_DATA = init_f.read_and_join_all_files()

        STOCK_DATA = self.compute_LEM_and_TEM(STOCK_DATA)

        STOCK_DATA = self.map_with_ALM_mappings(STOCK_DATA)

        STOCK_DATA = self.agregate_STOCK(STOCK_DATA)

        STOCK_DATA = self.process_starting_points(STOCK_DATA)

        STOCK_DATA = self.change_to_to_row_format(STOCK_DATA)

        STOCK_DATA = self.prepare_output_STOCK(STOCK_DATA)

        return STOCK_DATA

    def change_to_to_row_format(self, data):
        indics_ordered = [pa.NC_PA_LEF, pa.NC_PA_LEM, pa.NC_PA_LMN, pa.NC_PA_LMN_EVE, pa.NC_PA_TEF, \
                          pa.NC_PA_TEM, pa.NC_PA_DEM_RCO, pa.NC_PA_DMN_RCO]
        data = gu.unpivot_data(data, pa.NC_PA_IND03)
        keys_order = [pa.NC_PA_BILAN, pa.NC_PA_CLE]
        data = gu.add_empty_indics(data, [pa.NC_PA_DEM_RCO, pa.NC_PA_DMN_RCO], pa.NC_PA_IND03, "LEF",
                                   pa.NC_PA_COL_SORTIE_NUM_ST, \
                                   order=True, indics_ordered=indics_ordered, keys_order=keys_order)
        data = self.add_missing_num_cols(data)
        return data

    def add_missing_num_cols(self, data):
        cols = [col for col in pa.NC_PA_COL_SORTIE_NUM_ST if col not in data.columns.tolist()]
        data = pd.concat([data, pd.DataFrame([[0] * len(cols)], \
                                             index=data.index, \
                                             columns=cols)], axis=1)
        return data

    def agregate_STOCK(self, STOCK_DATA):
        nums_cols = [x for x in STOCK_DATA.columns if ("LEF_M" in x) or ("TEF_M" in x) \
                     or ("LMN_M" in x) or ("LMN EVE_M" in x) or ("LEM_M" in x) or ("TEM_M" in x) or ("DEM_M" in x) or (
                                 "DMN_M" in x)]
        qual_cols = [x for x in STOCK_DATA.columns if x not in nums_cols]

        STOCK_DATA[nums_cols] = STOCK_DATA[nums_cols].astype(np.float64)

        STOCK_DATA[qual_cols] = STOCK_DATA[qual_cols].fillna("")

        AG_STOCK_DATA = STOCK_DATA.copy().groupby(by=qual_cols, as_index=False).sum()

        AG_STOCK_DATA[pa.NC_PA_CLE] = AG_STOCK_DATA[pa.NC_PA_CLE_OUTPUT].apply(lambda x: "_".join(x), axis=1)
        AG_STOCK_DATA = AG_STOCK_DATA.sort_values([pa.NC_PA_BILAN, pa.NC_PA_CLE])
        AG_STOCK_DATA[pa.NC_PA_INDEX] = ["ST-ECH" + str(i) if typo == "O" else "ST-NMD" + str(i)
                                         for i, typo in
                                         zip(range(1, len(AG_STOCK_DATA) + 1), AG_STOCK_DATA[pa.NC_PA_isECH].values)]

        return AG_STOCK_DATA

    def process_starting_points(self, STOCK_DATA):
        # Valeur en 0 définie par TEM pour les données FERMAT
        for ind in ["TEM_M0", "TEF_M0", "LEM_M0", "LEF_M0", "DEM_M0"]:
            if self.up.current_etab in gp.NON_RZO_ETABS:
                if ind in STOCK_DATA:
                    STOCK_DATA[ind] = STOCK_DATA["LEF_M0"]
            else:
                if ind in STOCK_DATA:
                    STOCK_DATA[ind] = STOCK_DATA["TEM_M0"]

        STOCK_DATA["LMN_M0"] = 0
        STOCK_DATA["LMN EVE_M0"] = 0

        return STOCK_DATA

    def compute_LEM_and_TEM(self, STOCK_DATA):
        no_mean_input = ("LEM_M1" not in STOCK_DATA.columns) & ("TEM_M1" not in STOCK_DATA.columns)
        if (no_mean_input):
            logger.info("Calcul des indicateurs mensuels moyens LEM et TEM")
            for prefix in ["LE", "TE"]:
                EM_cols = [prefix + "M_M" + str(x) for x in range(0, 121)]
                EF_cols = [prefix + "F_M" + str(x) for x in range(0, 121)]

                # Point de départ (pour le scope MNI)
                filter_scope = (STOCK_DATA[pa.NC_PA_SCOPE] != gp.LIQ)
                STOCK_DATA = pd.concat([STOCK_DATA, pd.DataFrame(0.0, index=STOCK_DATA.index, columns=EM_cols[0:1])],
                                       axis=1)
                STOCK_DATA.loc[filter_scope, EM_cols[0]] = STOCK_DATA.loc[filter_scope, EF_cols[0]]

                # EM calculé comme la moyenne des EF des mois courants et suivants
                STOCK_DATA = pd.concat([STOCK_DATA, pd.DataFrame(index=STOCK_DATA.index, columns=EM_cols[1:])], axis=1)
                STOCK_DATA[EM_cols[1:]] = (STOCK_DATA.loc[:, EF_cols[1:]].values + STOCK_DATA.loc[:,
                                                                                   EF_cols[:-1]]).values / 2

                # EM calculé à partir du mois 120 comme l'année courante et suivante'
                EM_cols = [prefix + "M_M" + str(x) for x in range(132, 241, 12)]
                EF_cols = [prefix + "F_M" + str(x) for x in range(120, 241, 12)]
                STOCK_DATA = pd.concat([STOCK_DATA, pd.DataFrame(index=STOCK_DATA.index, columns=EM_cols)], axis=1)
                STOCK_DATA[EM_cols] = (STOCK_DATA.loc[:, EF_cols[1:]].values + STOCK_DATA.loc[:,
                                                                               EF_cols[:-1]]).values / 2

        return STOCK_DATA

    def map_with_ALM_mappings(self, STOCK_DATA):
        logger.info("MAPPING DES DONNEES DU STOCK avec les indicators PASS ALM")

        if self.up.current_etab in gp.NTX_FORMAT:
            logger.info("   MAPPING DES DONNEES DU STOCK avec les indicators ALM spécifiques à NATIXIS/SEF")
            STOCK_DATA = mp.map_data_with_ntx_mappings(STOCK_DATA, self.up.sc_ref_nmd,
                                                       self.up.ra_file_path, self.up.liq_file_path)

        logger.info("   MAPPING DES DONNEES DU STOCK avec les indicators ALM généraux")
        STOCK_DATA = mp.map_data_with_general_mappings(STOCK_DATA, self.up.current_etab)

        logger.info("   MAPPING DES LIGNES INTRA-GROUPE DU STOCK")
        STOCK_DATA = mp.map_intra_groupes(self.up.current_etab, STOCK_DATA)

        if self.up.current_etab == "BPCE":
            STOCK_DATA = mp.map_data_with_bpce_mappings(STOCK_DATA)

        STOCK_DATA = self.map_lcr_nsfr_data(STOCK_DATA)

        logger.info("   MAPPING DES DONNEES DU STOCK avec les indicators LIQUIDITE")
        STOCK_DATA = mp.mapping_consolidation_liquidite(STOCK_DATA)

        return STOCK_DATA

    def map_lcr_nsfr_data(self, STOCK_DATA):
        global lcr_nsfr_data

        if self.up.map_lcr_nsfr:
            logger.info("   MAPPING DES DONNEES DU STOCK avec RAY")
            lcr_nsfr.parse_ray_file(self.up.lcr_nsfr_file)
            STOCK_DATA = lcr_nsfr.map_lcr_tiers_and_share(STOCK_DATA)

        else:
            logger.info("   PAS DE MAPPING RAY")

        return STOCK_DATA

    def prepare_output_STOCK(self, stock):

        cols = [col for col in pa.NC_PA_COL_SORTIE_QUAL if col not in stock.columns.tolist()]
        final_stock = pd.concat([stock, pd.DataFrame([["-"] * len(cols)], \
                                                     index=stock.index, \
                                                     columns=cols)], axis=1)

        final_stock["BOOK"] = "'" + final_stock["BOOK"]

        return final_stock[pa.NC_PA_COL_SORTIE_QUAL + pa.NC_PA_COL_SORTIE_NUM_ST + [pa.NC_PA_isECH]]
