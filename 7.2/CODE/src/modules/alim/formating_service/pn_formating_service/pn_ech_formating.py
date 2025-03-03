import pandas as pd
import modules.alim.lcr_nsfr_service.lcr_nsfr_module as lcr_nsfr
import utils.general_utils as gu
import modules.alim.parameters.general_parameters as gp
from params import version_params as vp
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import mappings.mapping_functions as mp
import mappings as gmp
import modules.alim.parameters.RZO_params as rzo_p
import numpy as np
import logging
from mappings.mapping_functions import map_data
from mappings import general_mappings as gma
from modules.alim.formating_service.stock_formating_service.stock_common_formating import upper_columns_names

logger = logging.getLogger(__name__)


class PN_ECH_Formater():

    NC_ECH_ETAB = "ETAB"
    NC_ECH_ETAB2 = "COMPANY_CODE"
    NC_ECH_CONTREPARTIE = "attribute_6".upper()
    NC_ECH_ATTRIBUTE_1 = "attribute_1".upper()
    NC_ECH_RATE_CAT = "rate_category".upper()
    NC_ECH_FAMILY = "family".upper()
    NC_ECH_CONTRACT_TYPE = "contract_type".upper()
    NC_ECH_CCY_CODE = "currency".upper()
    NC_ECH_DEBUT = "begin_flow".upper()
    NC_ECH_DUREE = "maturity_flow".upper()
    NC_ECH_DARNUM = "DARNUM"
    NC_ECH_PROFIL_EC = "amortizing_type".upper()
    NC_ECH_PERIOD_INTERETS = "periodicity".upper()
    NC_ECH_PERIOD_FIXING = "fixing_periodicity".upper()
    NC_ECH_INSTRUMENT = "instrument".upper()
    NC_ECH_INDEX_REF = "rate".upper()
    NC_ECH_PERIOD_CAPI = "compound_periodicity".upper()
    NC_ECH_PERIOD_AMOR = "amortizing_periodicity".upper()
    NC_ECH_BASE_CAL = "accrual_basis".upper()
    NC_ECH_RELEASING_RULE = "releasing_rule".upper()
    NC_ECH_CLE = "cle".upper()

    if vp.version_sources == "csv":
        cols_ECH = ["M0" + str(i) if i <= 9 else "M" + str(i) for i in range(1, pa.MAX_MONTHS_PN + 1)]
        cols_marge = ["M0" + str(i) + "_MARGE" if i <= 9 else "M" + str(i) + "_MARGE" for i in range(1, pa.MAX_MONTHS_PN + 1)]
    else:
        cols_ECH = ["M0" + str(i) if i <= 9 else "M" + str(i) for i in range(1, pa.MAX_MONTHS_PN + 1)]
        cols_marge = ["M0" + str(i) + "_MARGE" if i <= 9 else "M" + str(i) + "_MARGE" for i in range(1, pa.MAX_MONTHS_PN + 1)]


    def __init__(self, cls_usr):
        self.up = cls_usr

    def read_marge_file(self, ):
        file_marges = rzo_p.pn_rzo_files_name["PN-ECH-MARGES"]
        logger.info("   Lecture de : " + file_marges.split("\\")[-1])
        delimiter = ";" if vp.version_sources == "csv" else "\t"
        decimal = "," if vp.version_sources == "csv" else ","
        pn_marge = pd.read_csv(file_marges, delimiter=delimiter, engine="python",
                               decimal=decimal)  # Lecture du fichier MARGES
        pn_marge = upper_columns_names(pn_marge)
        pn_marge.loc[:, self.cols_ECH] = pn_marge.loc[:, self.cols_ECH].fillna(0).astype(np.float64)
        pn_marge.rename(columns={self.NC_ECH_ETAB2: self.NC_ECH_ETAB}, inplace=True)
        pn_marge = pn_marge[[self.NC_ECH_ETAB, self.NC_ECH_CLE] + self.cols_ECH].copy()

        pn_marge.columns = [str(col) + '_MARGE' if col in self.cols_ECH else str(col) for col in pn_marge.columns]

        return pn_marge


    def read_encours_file(self, ):
        file_encours = rzo_p.pn_rzo_files_name["PN-ECH-ENCOURS"]
        logger.info("   Lecture de : " + file_encours.split("\\")[-1])
        delimiter = ";" if vp.version_sources == "csv" else "\t"
        decimal = "," if vp.version_sources == "csv" else ","
        thousands = "" if vp.version_sources == "csv" else " "
        pn_encours = pd.read_csv(file_encours, delimiter=delimiter, engine="python", decimal=decimal,
                                 thousands=thousands)  # Lecture du fichier ENCOURS

        pn_encours = upper_columns_names(pn_encours)

        """ FILTRE SUR LES PN ECH"""
        # pn_encours = pn_encours[pn_encours[NC_ECH_INSTRUMENT]=="MAT"].copy()

        filter_non_zero = (pn_encours[self.cols_ECH].sum(axis=1) != 0)
        pn_encours = pn_encours[filter_non_zero].copy()

        if sum(filter_non_zero) > 0:
            ht = pn_encours.shape[0]
            pn_encours[self.cols_ECH] = pn_encours[self.cols_ECH].fillna(0).astype(np.float64)
            pn_encours['COEF'] = 0
            filtre1 = (pn_encours[self.NC_ECH_CONTRACT_TYPE].str[0:2] == "A-") | (
                        pn_encours[self.NC_ECH_CONTRACT_TYPE].str[-2:] == "-A") & (
                                  pn_encours[self.NC_ECH_CONTRACT_TYPE].str[:2] == "HB")
            filtre2 = (pn_encours[self.NC_ECH_CONTRACT_TYPE].str[0:2] == "P-") | (
                        pn_encours[self.NC_ECH_CONTRACT_TYPE].str[-2:] == "-P") & (
                                  pn_encours[self.NC_ECH_CONTRACT_TYPE].str[:2] == "HB")
            pn_encours[self.cols_ECH] = np.select([np.array(filtre1).reshape(ht, 1), np.array(filtre2).reshape(ht, 1)], \
                                             [pn_encours[self.cols_ECH], -1 * pn_encours[self.cols_ECH]], 0)

        cols = [x for x in pn_encours.columns if x not in ["M61", "M62"]]

        return pn_encours[cols].copy()


    def rename_num_cols(self, pn_ech_data):
        pn_ech_data = pn_ech_data.rename(
            columns={"M" + str(i) if i >= 10 else "M0" + str(i): \
                         pa.NC_PA_DEM + "_M" + str(i) for i in range(1, pa.MAX_MONTHS_PN + 1)})

        pn_ech_data = pn_ech_data.rename(
            columns={"M" + str(i) + "_MARGE" if i >= 10 else "M0" + str(i) + "_MARGE": \
                         pa.NC_PA_MG_CO + "_M" + str(i) for i in range(1, pa.MAX_MONTHS_PN + 1)})

        return pn_ech_data


    def join_marge_and_encours(self, pn_encours, pn_marge):
        pn_encours_joined = pn_encours.merge(pn_marge, how="left", on=[self.NC_ECH_CLE, self.NC_ECH_ETAB])

        if pn_encours.shape[0] != pn_encours_joined.shape[0]:
            logger.error("La clé encours/marge n'est pas unique. Veuillez vérifier vos fichiers")

        pn_encours_joined[self.cols_marge] = pn_encours_joined[self.cols_marge].fillna(0).astype(np.float64)

        return pn_encours_joined


    def format_bilan(self, data):
        filtres = [data[self.NC_ECH_CONTRACT_TYPE].str[:2] == "A-",
                   data[self.NC_ECH_CONTRACT_TYPE].str[:2] == "P-", \
                   (data[self.NC_ECH_CONTRACT_TYPE].str[-2:] == "-A") & (data[self.NC_ECH_CONTRACT_TYPE].str[:2] == "HB"),
                   (data[self.NC_ECH_CONTRACT_TYPE].str[-2:] == "-P") & (data[self.NC_ECH_CONTRACT_TYPE].str[:2] == "HB")]
        choices = ["B ACTIF", "B PASSIF", "HB ACTIF", "HB PASSIF"]
        data[pa.NC_PA_BILAN] = np.select(filtres, choices)

        return data


    def map_ech_data(self, pn_ech_data):
        pn_ech_data[self.NC_ECH_INDEX_REF] = np.where(pn_ech_data[self.NC_ECH_RATE_CAT] == "FIXED",
                                                      pn_ech_data[self.NC_ECH_RATE_CAT],
                                                      pn_ech_data[self.NC_ECH_INDEX_REF])

        pn_ech_data[self.NC_ECH_CONTREPARTIE] = pn_ech_data[self.NC_ECH_CONTREPARTIE].fillna("-")

        pn_ech_data = gu.force_integer_to_string(pn_ech_data, self.NC_ECH_CONTREPARTIE)

        pn_ech_data[self.NC_ECH_FAMILY] = pn_ech_data[self.NC_ECH_FAMILY].fillna("-")

        if vp.version_sources == "csv":
            pn_ech_data[self.NC_ECH_DEBUT] = pn_ech_data[self.NC_ECH_DEBUT].astype(np.float64)
            # date_deb = pd.to_datetime(pn_ech_data[NC_ECH_DEBUT], format="%d/%m/%Y").copy()
            # pn_ech_data[NC_ECH_DEBUT] = pd.to_datetime(pn_ech_data[NC_ECH_DEBUT], format="%d/%m/%Y").dt.day.astype(np.float64)
        else:
            pn_ech_data[self.NC_ECH_DEBUT] = pn_ech_data[self.NC_ECH_DEBUT].astype(np.float64)

        pn_ech_data[[self.NC_ECH_PERIOD_INTERETS, self.NC_ECH_PERIOD_CAPI, self.NC_ECH_PERIOD_AMOR]] \
            = pn_ech_data[[self.NC_ECH_PERIOD_INTERETS, self.NC_ECH_PERIOD_CAPI, self.NC_ECH_PERIOD_AMOR]].fillna("None")

        cles_data = {1: [pa.NC_PA_BILAN, self.NC_ECH_CONTRACT_TYPE],
                     2: [self.NC_ECH_INDEX_REF], 3: [self.NC_ECH_CONTREPARTIE], 4: [self.NC_ECH_PROFIL_EC],
                     5: [self.NC_ECH_PERIOD_INTERETS], \
                     6: [self.NC_ECH_PERIOD_AMOR], 7: [self.NC_ECH_PERIOD_CAPI]}

        mappings = {1: "CONTRATS", 2: "INDEX_AGREG", \
                    3: "PALIER", 4: "mapping_ECH_AMOR_PROFIL", 5: "mapping_ECH_PER_INTERETS", \
                    6: "mapping_PERIODE_AMOR", 7: "mapping_PER_CAPI"}

        for i in range(1, 4):
            pn_ech_data = map_data(pn_ech_data, gma.map_pass_alm[mappings[i]], keys_data=cles_data[i],
                                   name_mapping="PN DATA vs.")

        for i in range(4, len(mappings) + 1):
            pn_ech_data = map_data(pn_ech_data, gma.mapping_PN[mappings[i]], keys_data=cles_data[i],
                                   name_mapping="PN DATA vs.")

        if vp.version_sources == "csv":
            pn_ech_data[pa.NC_PA_MATURITY_DURATION] = pn_ech_data[self.NC_ECH_DUREE].fillna("0").str.replace("M", "").astype(np.int64)
        else:
            pn_ech_data[pa.NC_PA_MATURITY_DURATION] = pn_ech_data[self.NC_ECH_DUREE].fillna("0").str.replace("M", "").astype(np.int64)

        pn_ech_data[pa.NC_PA_MATUR] = np.where(pn_ech_data[pa.NC_PA_MATURITY_DURATION] < 13, "CT", "MLT")

        pn_ech_data.rename(columns={self.NC_ECH_FAMILY: pa.NC_PA_MARCHE, self.NC_ECH_BASE_CAL: pa.NC_PA_ACCRUAL_BASIS,
                                    self.NC_ECH_INDEX_REF: pa.NC_PA_RATE_CODE, self.NC_ECH_CCY_CODE: pa.NC_PA_DEVISE}, inplace=True)

        pn_ech_data[pa.NC_PA_ACCRUAL_BASIS] = pn_ech_data[pa.NC_PA_ACCRUAL_BASIS].str.upper()

        pn_ech_data[self.NC_ECH_RELEASING_RULE] = pn_ech_data[self.NC_ECH_RELEASING_RULE].fillna(-1).astype(int).astype(str).replace("-1", "")

        pn_ech_data = pn_ech_data.rename(columns={self.NC_ECH_PERIOD_FIXING: pa.NC_PA_FIXING_PERIODICITY,
                                                  self.NC_ECH_DEBUT: pa.NC_PA_JR_PN,
                                                  self.NC_ECH_RELEASING_RULE: pa.NC_PA_RELEASING_RULE})

        pn_ech_data[pa.NC_PA_FIXING_PERIODICITY] = pn_ech_data[pa.NC_PA_FIXING_PERIODICITY].fillna("")

        col_sortie = [x for x in pa.NC_PA_COL_SORTIE_QUAL_ECH if x != self.NC_ECH_CLE]
        keep_cols = [x for x in col_sortie if x in pn_ech_data.columns] + self.cols_ECH + self.cols_marge
        pn_ech_data = pn_ech_data[keep_cols].copy()

        return pn_ech_data


    def add_map_cols(self, pn_encours):
        pn_encours[pa.NC_PA_BASSIN] = self.up.current_etab
        pn_encours[pa.NC_PA_BOOK] = "-"
        pn_encours[pa.NC_PA_SCOPE] = gp.MNI_AND_LIQ
        return pn_encours


    def map_lcr_nsfr_coeff(self, pn_encours):
        pn_encours[pa.NC_PA_LCR_TIERS] = "-"
        pn_encours[pa.NC_PA_LCR_TIERS_SHARE] = 100
        if self.up.map_lcr_nsfr:
            logger.info("MAPPING DES DONNEES DE PN ECH avec RAY")
            pn_encours = lcr_nsfr.map_lcr_tiers_and_share(pn_encours)
        return pn_encours


    def add_contracts_mapp_cols(self, pn_data, type_pn):
        pn_data = mp.map_data(pn_data, gmp.map_pass_alm["CONTRATS"], \
                              keys_data=[pa.NC_PA_BILAN, self.NC_ECH_CONTRACT_TYPE],
                              name_mapping="%s DATA vs." % type_pn.upper(), \
                              cols_mapp=[pa.NC_PA_POSTE, pa.NC_PA_DIM2, \
                                         pa.NC_PA_DIM3, pa.NC_PA_DIM4, pa.NC_PA_DIM5])

        return pn_data


    def add_missing_cols(self, pn_ech_data):
        missing_indics_cles = [x for x in pa.NC_PA_CLE_OUTPUT if not x in pn_ech_data.columns.tolist()]

        pn_ech_data = pd.concat([pn_ech_data, pd.DataFrame([["-"] * len(missing_indics_cles)], \
                                                           index=pn_ech_data.index,
                                                           columns=missing_indics_cles)], axis=1)

        CLE = pd.DataFrame(pn_ech_data[pa.NC_PA_CLE_OUTPUT].astype(str).apply(lambda x: "_".join(x), axis=1).copy(), \
                           index=pn_ech_data.index, columns=[pa.NC_PA_CLE])

        pn_ech_data = pd.concat([pn_ech_data, CLE], axis=1)

        pn_ech_data = pn_ech_data.sort_values([pa.NC_PA_BILAN, pa.NC_PA_CLE] + pa.NC_PA_COL_SPEC_ECH)

        INDEX = pd.DataFrame(["ECH" + str(i) for i in range(1, pn_ech_data.shape[0] + 1)], \
                             index=pn_ech_data.index, columns=[pa.NC_PA_INDEX])

        pn_ech_data = pd.concat([pn_ech_data, INDEX], axis=1)

        col_sortie = [x for x in pa.NC_PA_COL_SORTIE_QUAL_ECH if x != pa.NC_PA_IND03]

        missing_indics = [x for x \
                          in col_sortie \
                          if not x in pn_ech_data.columns.tolist()]

        pn_ech_data = pd.concat([pn_ech_data, pd.DataFrame([["-"] * len(missing_indics)], \
                                                           index=pn_ech_data.index,
                                                           columns=missing_indics)], axis=1)

        return pn_ech_data


    def group_ech_data(self, pn_ech_data):
        all_cols_except_dem = [x for x in pn_ech_data.columns if
                               x not in self.cols_ECH]  # marges incluses dans les colonnes d'agreg
        pn_ech_data = pn_ech_data.groupby(all_cols_except_dem, dropna=False, as_index=False).sum()
        return pn_ech_data


    def add_missing_num_cols(self, data):
        cols = [col for col in pa.NC_PA_COL_SORTIE_NUM_PN if col not in data.columns.tolist()]
        data = pd.concat([data, pd.DataFrame([[""] * len(cols)], \
                                             index=data.index, \
                                             columns=cols)], axis=1)
        return data


    def add_missing_indics(self, pn_ech_data):
        indic_ordered = [pa.NC_PA_DEM, pa.NC_PA_MG_CO, pa.NC_PA_TX_SP, pa.NC_PA_TX_CIBLE]
        keys_order = [pa.NC_PA_BILAN, pa.NC_PA_CLE, pa.NC_PA_INDEX] + pa.NC_PA_COL_SPEC_ECH
        pn_ech_data = gu.add_empty_indics(pn_ech_data, [pa.NC_PA_MG_CO, pa.NC_PA_TX_SP, pa.NC_PA_TX_CIBLE], pa.NC_PA_IND03, \
                                          pa.NC_PA_DEM, pa.NC_PA_COL_SORTIE_NUM_PN, order=True, indics_ordered=indic_ordered, \
                                          keys_order=keys_order)
        return pn_ech_data


    def final_formatting(self, pn_ech_data):
        pn_ech_data = self.add_missing_num_cols(pn_ech_data)
        pn_ech_data = self.add_missing_indics(pn_ech_data)
        return pn_ech_data[pa.NC_PA_COL_SORTIE_QUAL_ECH + pa.NC_PA_COL_SORTIE_NUM_PN]


    def change_to_to_row_format(self, pn_ech_data):
        pn_ech_data = gu.unpivot_data(pn_ech_data, pa.NC_PA_IND03)
        return pn_ech_data


    def format_RZO_PN_ECH(self):
        logger.info("TRAITEMENT DES PN ECH")
        pn_marge = self.read_marge_file()
        pn_encours = self.read_encours_file()

        if len(pn_encours) > 0:

            pn_encours = self.format_bilan(pn_encours)

            pn_ech_data = self.join_marge_and_encours(pn_encours, pn_marge)

            pn_ech_data = self.map_ech_data(pn_ech_data)

            pn_ech_data = self.group_ech_data(pn_ech_data)

            pn_ech_data = self.add_map_cols(pn_ech_data)

            pn_ech_data = self.rename_num_cols(pn_ech_data)

            pn_ech_data = self.map_lcr_nsfr_coeff(pn_ech_data)

            pn_ech_data = mp.mapping_consolidation_liquidite(pn_ech_data)

            pn_ech_data = self.add_missing_cols(pn_ech_data)

            pn_ech_data = self.change_to_to_row_format(pn_ech_data)

            pn_ech_data = self.final_formatting(pn_ech_data)

        else:
            pn_ech_data = []

        return pn_ech_data
