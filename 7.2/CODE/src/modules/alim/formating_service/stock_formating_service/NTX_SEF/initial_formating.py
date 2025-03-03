import pandas as pd
import datetime
import modules.alim.parameters.user_parameters as up
import modules.alim.parameters.NTX_SEF_params as ntx_p
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import modules.alim.parameters.general_parameters as gp
import numpy as np
import modules.alim.formating_service.stock_formating_service.stock_common_formating as init_f_gen
import logging
import utils.excel_utils as ex

logger = logging.getLogger(__name__)

class NTX_SEF_Formater():
    def __init__(self, cls_users_params):
        self.up = cls_users_params

    def rename_columns(self, data, prefix):
        if prefix == "LEF":
            data = data.rename(columns={"FLAG_LIQ_MNI": ntx_p.NC_NTX_DATA_FLAG_MNI})
            data = data.rename(columns={"FLAGLIQMNI": ntx_p.NC_NTX_DATA_FLAG_MNI})
            data = data.rename(columns={"FLAG_BALANCE_SHEET_LIQ": "FLAG_BALANCE_SHEET"})
        else:
            data = data.rename(columns={"FLAGMNI": ntx_p.NC_NTX_DATA_FLAG_MNI})

        data = data.rename(
            columns={ntx_p.NC_NTX_DATA_DIM1: pa.NC_PA_BILAN, ntx_p.NC_NTX_DATA_FAMILY: pa.NC_PA_MARCHE, \
                     ntx_p.NC_NTX_DATA_BALE: pa.NC_PA_LCR_TIERS, ntx_p.NC_NTX_DATA_CONTRAT: gp.NC_CONTRACT_TEMP})

        data = data.rename(
            columns={ntx_p.NC_NTX_DATA_PALIER: gp.NC_PALIER_TEMP, ntx_p.NC_NTX_DATA_GESTION: gp.NC_GESTION_TEMP, \
                     ntx_p.NC_NTX_DATA_CCY: pa.NC_PA_DEVISE, ntx_p.NC_NTX_DATA_RATEC: pa.NC_PA_RATE_CODE, \
                     ntx_p.NC_NTX_DATA_MATUR: gp.NC_MATUR_TEMP,
                     ntx_p.NC_NTX_DATA_FLAG_MNI: pa.NC_PA_TOP_MNI})

        return data

    def filter_data_depart_decale(self, data):
        if str(ntx_p.depart_decale).upper() == "OUI":
            data = data[data[ntx_p.NC_NTX_DATA_DEPARTD] == "N"].copy()
        data.drop([ntx_p.NC_NTX_DATA_DEPARTD], axis=1, inplace=True)
        return data

    def do_preliminary_checks(self, data, file_name):
        if len(data.drop_duplicates()) != len(data):
            logger.warning("Il y a des doublons dans le rapport: " + file_name.split("\\")[-1])
            qual_cols = [x for x in data.columns if not isinstance(x, datetime.date)]
            data = data.groupby(by=qual_cols, as_index=False, dropna=False).sum()

        if datetime.datetime.strptime(data[ntx_p.NC_NTX_DATA_DAR].iloc[0], '%d/%m/%Y').date() != up.dar.date():
            raise ValueError(
                "La DAR du fichier " + file_name.split("\\")[-1] + " est différente de la DAR en paramètre")

        dar_file = data.columns.tolist()[
            data.columns.tolist().index(next(filter(lambda x: isinstance(x, datetime.date), data.columns.tolist())))]

        if dar_file != up.dar.date():
            raise ValueError("Le colonne montant pour la date de DAR %s dans le fichier %s est absente " % (
                up.dar.date(), file_name.split("\\")[-1]))
        return data

    def modify_contracts_for_lcr_tiers(self, data):
        check = (data[pa.NC_PA_LCR_TIERS] == "BC - BC")
        liste_lcr_tiers_A = ntx_p.lcr_tiers_actif_change_contract
        liste_lcr_tiers_P = ntx_p.lcr_tiers_passif_change_contract

        checkA = check & (data[gp.NC_CONTRACT_TEMP].isin(liste_lcr_tiers_A))
        checkP = check & (data[gp.NC_CONTRACT_TEMP].isin(liste_lcr_tiers_P))

        data[gp.NC_CONTRACT_TEMP] = np.select([checkA, checkP],
                                              [ntx_p.contract_change_actif, ntx_p.contract_change_passif], \
                                              default=data[gp.NC_CONTRACT_TEMP])

        return data

    def transform_num_columns_names(self, data):
        num_cols = [x for x in data.columns.tolist() if x.isnumeric()]
        num_cols = [ex.excel_to_python_date(int(x)).date() for x in num_cols]
        data.columns = [x for x in data.columns.tolist() if not x.isnumeric()] + num_cols
        return data

    def determine_scope_and_marche(self, data):
        swap_lines = data[data[ntx_p.NC_NTX_DATA_SWAP_TAG] == 1].copy()
        swap_lines_shape = swap_lines.shape[0]
        data = pd.concat([data, swap_lines]).reset_index(drop=True)
        num_cols_SWAP = [x for x in data.columns if ("SWAP_M" in x)]
        num_cols_LEF = [x for x in data.columns if ("LEF_M" in x)]
        data.loc[data.index[-swap_lines_shape]:, ntx_p.NC_NTX_DATA_SWAP_TAG] = 2
        data.loc[data.index[-swap_lines_shape]:, num_cols_LEF] = \
            data.loc[data.index[-swap_lines_shape]:, num_cols_SWAP].values
        data.loc[data.index[-swap_lines_shape]:, pa.NC_PA_SCOPE] = gp.MNI
        filters = [data[ntx_p.NC_NTX_DATA_SWAP_TAG] == 1, data[ntx_p.NC_NTX_DATA_SWAP_TAG] == 2,
                   data[ntx_p.NC_NTX_DATA_SWAP_TAG] == 0]
        data[pa.NC_PA_SCOPE] = np.select(filters, [gp.LIQ, gp.MNI, gp.MNI_AND_LIQ], default=gp.MNI_AND_LIQ)
        if ntx_p.NC_NTX_DATA_TYPE_EMETTEUR in data.columns.tolist():
            data[pa.NC_PA_MARCHE] = np.where(
                (data[ntx_p.NC_NTX_DATA_SWAP_TAG].isin([1, 2])) & (data[ntx_p.NC_NTX_DATA_TYPE_EMETTEUR] != ""), \
                data[ntx_p.NC_NTX_DATA_TYPE_EMETTEUR], data[pa.NC_PA_MARCHE])
            data.drop([ntx_p.NC_NTX_DATA_TYPE_EMETTEUR], axis=1, inplace=True)
        return data

    def format_id_column(self, data, prefix):
        data[ntx_p.NC_NTX_DATA_ID] = data[ntx_p.NC_NTX_DATA_ID].fillna(0).astype(int)
        null_values = data[data[ntx_p.NC_NTX_DATA_ID] == 0]
        if len(null_values) > 0:
            logger.warning("Il y a des ids nuls dans le fichier %s" % prefix)
            new_ids = prefix + "_" + pd.DataFrame(np.arange(0, len(null_values))).iloc[:, 0].astype(str)
            data.loc[data[ntx_p.NC_NTX_DATA_ID] == 0, ntx_p.NC_NTX_DATA_ID] = new_ids.values
        data[ntx_p.NC_NTX_DATA_ID] = data[ntx_p.NC_NTX_DATA_ID].astype(str)

        if data[ntx_p.NC_NTX_DATA_ID].duplicated().any():
            logger.error("There are non unique ids in the %s file" % prefix)
            logger.error("Non unique ids are: %s" % data[ntx_p.NC_NTX_DATA_ID][
                data[ntx_p.NC_NTX_DATA_ID].duplicated()].values.tolist())
            logger.error("Non unique ids cannot be matched correctly with lines in the SWAP file")
            raise ValueError("There are non unique ids in the %s file" % prefix)

        data = data.set_index([ntx_p.NC_NTX_DATA_ID])
        return data

    def add_missing_vars(self, data):
        data[pa.NC_PA_BASSIN] = up.current_etab
        for col in [pa.NC_PA_ETAB, pa.NC_PA_CUST, pa.NC_PA_PERIMETRE, pa.NC_PA_SOUS_METIER, pa.NC_PA_ZONE_GEO,
                    pa.NC_PA_METIER, pa.NC_PA_SOUS_ZONE_GEO]:
            if not col in data.columns.tolist():
                data[col] = "-"
        data[pa.NC_PA_LCR_TIERS_SHARE] = 100
        return data

    def filter_encours_zero(self, data):
        if str(ntx_p.del_enc_dar_zero).upper() == "OUI":
            nums_cols = ["LEF_M0", "TEF_M0", "LMN_M1", "LMN EVE_M1", "SWAP_M0"]
            filter_non_zero = ((data[nums_cols] ** 2).sum(axis=1)) != 0
            data = data[filter_non_zero].copy()
        return data

    def finalize_formatting(self, data):
        data = init_f_gen.change_sign_passif(data)
        data = self.filter_data_depart_decale(data)
        data = self.filter_encours_zero(data)
        data[ntx_p.NC_NTX_DATA_SWAP_TAG] = data[ntx_p.NC_NTX_DATA_SWAP_TAG].fillna(0)
        if not up.detail_ig:
            data[gp.NC_GESTION_TEMP] = "-"
        return data

    def format_hors_bilan(self, data, num_cols):
        HB = (data[pa.NC_PA_BILAN] == "HORS BILAN")
        data.loc[HB, [pa.NC_PA_BILAN]] = "HB ACTIF"
        test = (data.loc[HB, num_cols] > 0) * (np.ones([sum(HB), 1]) * np.array(range(1, len(num_cols) + 1)))
        test[test == 0] = 10 * len(num_cols)
        is_actif = np.array(test).min(axis=1)  # Premier pas de temps ou le montant est positif

        test = (data.loc[HB, num_cols] < 0) * (np.ones([sum(HB), 1]) * np.array(range(1, len(num_cols) + 1)))
        test[test == 0] = 10 * len(num_cols)
        is_passif = np.array(test).min(axis=1)  # Premier pas de temps ou le montant est negatif

        tmp = data.loc[HB].copy()
        tmp.loc[is_actif < is_passif, [pa.NC_PA_BILAN]] = "HB ACTIF"
        tmp.loc[is_actif > is_passif, [pa.NC_PA_BILAN]] = "HB PASSIF"

        data.loc[HB, [pa.NC_PA_BILAN]] = tmp[pa.NC_PA_BILAN].values

        HB_ACTIF = (HB) & (
                (data[gp.NC_CONTRACT_TEMP].str[-4:] == "A-KN") | (data[gp.NC_CONTRACT_TEMP].str[-2:] == "-A"))
        data.loc[HB_ACTIF, [pa.NC_PA_BILAN]] = "HB ACTIF"
        HB_PASSIF = (HB) & (
                (data[gp.NC_CONTRACT_TEMP].str[-4:] == "P-KN") | (data[gp.NC_CONTRACT_TEMP].str[-2:] == "-P"))
        data.loc[HB_PASSIF, [pa.NC_PA_BILAN]] = "HB PASSIF"

        return data

    def format_type_emetteur(self, data):
        if ntx_p.NC_NTX_DATA_TYPE_EMETTEUR in data.columns.tolist():
            data[ntx_p.NC_NTX_DATA_TYPE_EMETTEUR] = data[ntx_p.NC_NTX_DATA_TYPE_EMETTEUR].fillna("")
        return data

    def read_and_format_ntx_file(self, file_name, prefix):
        data = pd.read_csv(file_name, delimiter=";", engine='python', decimal=",", thousands=' ', encoding='latin1')
        data = init_f_gen.upper_columns_names(data)
        data = self.transform_num_columns_names(data)
        data = self.do_preliminary_checks(data, file_name)
        data = self.format_id_column(data, prefix)
        data, num_cols = init_f_gen.select_num_cols(self.up.current_etab, self.up.dar, data, prefix)

        if prefix == "LEF":
            data = self.rename_columns(data, prefix)
            data = init_f_gen.select_cols_to_keep(data, ntx_p.cols_to_keep, num_cols)
            data = init_f_gen.format_bilan_column(data)
            data = self.modify_contracts_for_lcr_tiers(data)
            data = init_f_gen.format_paliers(data, gp.NC_PALIER_TEMP)
            data = self.format_type_emetteur(data)
            data = init_f_gen.upper_non_num_cols_vals(data, num_cols)
            data = self.format_hors_bilan(data, num_cols)
        else:
            data = data[num_cols].copy()
            if prefix == "SWAP":
                data[ntx_p.NC_NTX_DATA_SWAP_TAG] = 1

        return data

    def agregate_data(self, data, agreg_vars):
        num_cols = [x for x in data.columns if ("LEF_M" in x) or ("TEF_M" in x) \
                    or ("LMN_M" in x) or ("LMN EVE_M" in x) or ("TEM_M" in x) or \
                    ("TEF_M" in x) or ("LEM_M" in x) or ("DEM_M" in x) or ("DMN_M") in x]
        qual_vars = [x for x in data.columns if x not in num_cols and x in agreg_vars]
        data = data.reset_index(drop=True)
        data = data[qual_vars + num_cols].copy()
        data = data.groupby(by=qual_vars, as_index=False).sum()
        return data

    def read_and_join_all_files(self):
        i = 0
        for file in self.up.main_files_name:
            if i == 0 and self.up.prefixes_main_files[i] != "LEF":
                raise ValueError("Le fichier GAP LIQ doit être lu en premier")

            logger.info("   Pour l'indicateur " + self.up.prefixes_main_files[i] + ", lecture de : " + file.split("\\")[-1])
            data_prefix = self.read_and_format_ntx_file(file, self.up.prefixes_main_files[i])
            if i == 0:
                STOCK_DATA = data_prefix
            else:
                STOCK_DATA = init_f_gen.append_data(STOCK_DATA, data_prefix, self.up.prefixes_main_files[i])

            i = i + 1

        STOCK_DATA = self.finalize_formatting(STOCK_DATA)

        STOCK_DATA = self.determine_scope_and_marche(STOCK_DATA)

        STOCK_DATA = self.agregate_data(STOCK_DATA, ntx_p.agreg_vars)

        STOCK_DATA = self.add_missing_vars(STOCK_DATA)

        return STOCK_DATA

