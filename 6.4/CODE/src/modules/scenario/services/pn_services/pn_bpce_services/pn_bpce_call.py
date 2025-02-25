import os
import numpy as np
import pandas as pd
import utils.general_utils as ut
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import modules.scenario.services.pn_services.pn_bpce_services.mapping_service as mapp
import modules.scenario.services.pn_services.pn_bpce_services.referential_bpce as rf_bpce
import modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_common as com
import params.version_params as vp
import logging

logger = logging.getLogger(__name__)

def generate_PN_call(DAR, etab, file_path):
    call_data = pd.DataFrame(columns=["ADZDZDZ"])

    if os.path.exists(file_path):
        logger.info("       GENERATION DES CALLs")

        mapping_BPCE = mapp.mapping["mapping_BPCE"]
        mapping_refi_bpce = mapp.mapping["mapping_BPCE"]["REFI BPCE"]
        mapping_index = mapp.mapping["mapping_global"]

        logger.info("          Lecture de : " + file_path.split("\\")[-1])
        ut.check_version_templates(file_path.split("\\")[-1], version=vp.version_autre, path=file_path, open=True,
                                   warning=True)
        call_data = pd.read_excel(io=file_path, sheet_name="Opé Call", header=0, index_col=None, engine="openpyxl")
        call_data = call_data[call_data[rf_bpce.NC_CALL_EN_VIE].str.upper() == "OUI"]

        former_cols = [x for x in call_data.columns.tolist() if x != pa.NC_PA_BOOK]

        call_data["MONTH_START"] = 1 + call_data[rf_bpce.NC_CALL_DATE].dt.year * 12 + call_data[
            rf_bpce.NC_CALL_DATE].dt.month - (
                                           12 * DAR.year + DAR.month)

        # Si DUREE < 0 => Nous sommes sur un PUT
        call_data["DUREE CALL"] = call_data[rf_bpce.NC_CALL_DATE_ECH].dt.year * 12 + call_data[
            rf_bpce.NC_CALL_DATE_ECH].dt.month - \
                                  (call_data[rf_bpce.NC_CALL_DATE].dt.year * 12 + call_data[
                                      rf_bpce.NC_CALL_DATE].dt.month)

        call_data[pa.NC_PA_MATURITY_DURATION] = np.where(call_data["DUREE CALL"] < 0, call_data["MONTH_START"],
                                               call_data["DUREE CALL"])

        call_data["MONTH_START"] = call_data["MONTH_START"].mask(call_data["DUREE CALL"] < 0, \
                                                                 np.maximum(1,
                                                                            1 + call_data[
                                                                                rf_bpce.NC_CALL_DATE_ECH].dt.year * 12 \
                                                                            + call_data[
                                                                                rf_bpce.NC_CALL_DATE_ECH].dt.month - (
                                                                                    12 * DAR.year + DAR.month)))

        call_data = mapp.map_data(call_data, mapping_BPCE["mapping_profil_BPCE"], keys_data=[rf_bpce.NC_CALL_BASE],
                                  no_map_value="30/360", error_mapping=False)
        call_data = call_data.rename(columns={mapping_BPCE["mapping_profil_BPCE"]["OUT"][0]: pa.NC_PA_ACCRUAL_BASIS})

        call_data = call_data.rename(columns={rf_bpce.NC_CALL_DEV: pa.NC_PA_DEVISE,
                                              rf_bpce.NC_CALL_INDEX: pa.NC_PA_RATE_CODE})

        if len(call_data) > 0:

            call_data[pa.NC_PA_CONTRACT_TYPE] = rf_bpce.CONTRACT_NAME_CALL
            cols_to_mapp = [pa.NC_PA_PERIMETRE]
            call_data = mapp.map_data(call_data, mapping_refi_bpce, cols_mapp=cols_to_mapp, keys_data=[pa.NC_PA_CONTRACT_TYPE],
                                      join_how="inner")

            matrice_mois = np.concatenate(
                [np.arange(1, pa.MAX_MONTHS_PN + 1).reshape(1, pa.MAX_MONTHS_PN)] * call_data.shape[0])
            filtero_mois = (np.array(call_data["MONTH_START"]).reshape(call_data.shape[0], 1) == matrice_mois)

            call_data = call_data.drop(pa.NC_PA_COL_SORTIE_NUM_PN, axis=1, errors="ignore")

            call_data_dem = call_data.assign(**{x: 0 for x in pa.NC_PA_COL_SORTIE_NUM_PN})
            call_data_dem[pa.NC_PA_COL_SORTIE_NUM_PN2] = np.where(filtero_mois, 1, 0) * np.array(
                call_data_dem[rf_bpce.NC_CALL_ENCOURS]).reshape(call_data_dem.shape[0], 1)
            call_data_dem["IND03"] = pa.NC_PA_DEM

            call_data_mg = call_data.copy()
            call_data_mg = call_data_mg.assign(**{x: 0 for x in pa.NC_PA_COL_SORTIE_NUM_PN})
            call_data_mg[pa.NC_PA_COL_SORTIE_NUM_PN2] = np.where(filtero_mois, 1, 0) * np.array(
                call_data_mg[rf_bpce.NC_CALL_MARGE]).reshape(
                call_data_mg.shape[0], 1) * 10000
            call_data_mg["IND03"] = pa.NC_PA_MG_CO

            call_data_txsp = call_data.copy()
            call_data_txsp = call_data_txsp.assign(**{x: 0 for x in pa.NC_PA_COL_SORTIE_NUM_PN})
            filtero_mois = filtero_mois & np.array((call_data_txsp[rf_bpce.NC_CALL_INDEX] == "FIXED")).reshape(
                call_data_txsp.shape[0], 1)
            call_data_txsp[pa.NC_PA_COL_SORTIE_NUM_PN2] = np.where(filtero_mois, 1, 0) * np.array(
                call_data_txsp[rf_bpce.NC_CALL_TX_SP]).reshape(
                call_data_txsp.shape[0], 1) * 10000
            call_data_txsp["IND03"] = pa.NC_PA_TX_SP

            s = call_data_dem.shape[0]
            t = call_data_dem.shape[1]
            list_data = [call_data_dem.values, call_data_mg.values, call_data_txsp.values]
            call_data = np.stack(list_data, axis=1).reshape(s * 3, t)
            call_data = pd.DataFrame(call_data, columns=call_data_dem.columns.tolist())

            call_data = call_data.rename(columns={rf_bpce.NC_CALL_BOOK:pa.NC_PA_MARCHE})

            call_data = call_data.drop(columns=former_cols, axis=1, errors="ignore")

            call_data[pa.NC_PA_JR_PN] = rf_bpce.JR_PN_CALL
            call_data[pa.NC_PA_AMORTIZING_TYPE] = rf_bpce.AMOR_PROFIL_CALL
            call_data[pa.NC_PA_COMPOUND_PERIODICITY] = rf_bpce.CAPI_PERIOD_CALL
            call_data[pa.NC_PA_AMORTIZING_PERIODICITY] = rf_bpce.AMOR_PERIODE_CALL
            call_data[pa.NC_PA_PERIODICITY] = rf_bpce.INTERETS_PERIOD_CALL
            call_data[pa.NC_PA_FIXING_PERIODICITY] = rf_bpce.FIXING_PERIOD_CALL

            call_data[pa.NC_PA_BASSIN] = etab

            call_data[pa.NC_PA_LCR_TIERS] = rf_bpce.STD_LCR_TIERS

            call_data = com.add_missing_key_cols(call_data)

            call_data = com.generate_index(call_data, repeat=3)

            call_data = com.add_col_bilan(call_data)

            call_data = com.add_missing_indics(call_data)

            call_data = call_data.reset_index(drop=True)

            call_data = call_data.reset_index(drop=True)

    else:
        logger.warning("       Le fichier CALL n'a pas été trouvé : " + file_path)
        logger.info("       Pas de PN CALL générée")

    return call_data