import os
import numpy as np
import pandas as pd
import utils.excel_utils as ex
import utils.general_utils as ut
import modules.scenario.services.pn_services.pn_bpce_services.mapping_service as mapp
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import modules.scenario.services.pn_services.pn_bpce_services.referential_bpce as rf_bpce
import params.version_params as vp
import modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_common as com
import logging

logger = logging.getLogger(__name__)


def generate_PN_REFI_MLT(xl, DAR, etab, file_path):
    if os.path.exists(file_path):

        logger.info("       GENERATION DES OPE DE REFI MLT")

        logger.info("          Lecture de : " + file_path.split("\\")[-1])
        wb = xl.Workbooks.Open(file_path, None, True)
        ut.check_version_templates(file_path.split("\\")[-1], version=vp.version_autre, wb=wb, warning=True)

        mapping_conso_ech = mapp.mapping["mapping_PN"]["mapping_CONSO_ECH"]
        mapping_refi_bpce = mapp.mapping["mapping_BPCE"]["REFI BPCE"]

        data_saisonnalite = ex.get_dataframe_from_range(wb, rf_bpce.NG_SAISONNALITE, header=True)

        axe_annee, axe_mois = data_saisonnalite.shape
        data_saisonnalite = np.array(data_saisonnalite)
        data_saisonnalite = np.concatenate([data_saisonnalite] + [data_saisonnalite[rf_bpce.INDEX_A2_6:]] * 4, axis=0) \
            .reshape((1, axe_annee + 4, axe_mois, 1))

        data_refi_mlt = ex.get_dataframe_from_range(wb, rf_bpce.NG_REFI_MLT, header=True)

        maturity_cols = data_refi_mlt.columns.tolist()
        axe_annee, axe_maturite = data_refi_mlt.shape
        data_refi_mlt = np.array(data_refi_mlt).reshape(
            (rf_bpce.NB_CONTRAT_MLT, int(axe_annee / rf_bpce.NB_CONTRAT_MLT), 1, axe_maturite))
        data_refi_mlt = data_refi_mlt * data_saisonnalite
        axe_contrat, axe_annee, axe_mois, axe_maturite = data_refi_mlt.shape

        data_refi_mlt = data_refi_mlt.reshape((axe_contrat, axe_annee * axe_mois, axe_maturite))[:,
                        DAR.month:DAR.month + pa.MAX_MONTHS_PN]

        dic_refi_mlt = {}
        list_contrats = [rf_bpce.NAME_CONTRAT_OBLIG, rf_bpce.NAME_CONTRAT_OBLIG_SFH, rf_bpce.NAME_CONTRAT_SUB]
        index_contrat = [rf_bpce.INDEX_OBLIG, rf_bpce.INDEX_OBLIG_SFH, rf_bpce.INDEX_SUB]
        i = 0
        for c in list_contrats:
            dic_refi_mlt[c] = pd.DataFrame(data_refi_mlt[index_contrat[i]]).transpose()
            dic_refi_mlt[c].columns = pa.NC_PA_COL_SORTIE_NUM_PN2
            dic_refi_mlt[c][pa.NC_PA_MATURITY_DURATION] = maturity_cols
            dic_refi_mlt[c][pa.NC_PA_CONTRACT_TYPE] = c
            dic_refi_mlt[c] = mapp.map_data(dic_refi_mlt[c], mapping_refi_bpce, keys_data=[pa.NC_PA_CONTRACT_TYPE],
                                            join_how="inner")
            dic_refi_mlt[c] = mapp.map_data(dic_refi_mlt[c], mapping_conso_ech, keys_data=[pa.NC_PA_CONTRACT_TYPE],
                                            join_how="inner")
            dic_refi_mlt[c][pa.NC_PA_MATUR] = np.where(dic_refi_mlt[c][pa.NC_PA_MATURITY_DURATION] <= 12, "CT", "MLT")
            dic_refi_mlt[c][pa.NC_PA_LCR_TIERS] = rf_bpce.STD_LCR_TIERS
            dic_refi_mlt[c] = dic_refi_mlt[c][(dic_refi_mlt[c][pa.NC_PA_COL_SORTIE_NUM_PN2].sum(axis=1) != 0)]

            i = i + 1

        data_refi_mlt = pd.concat([dic_refi_mlt[key] for key in list(dic_refi_mlt.keys())])

        data_refi_mlt[pa.NC_PA_BASSIN] = etab

        missing_indics = [x for x \
                          in pa.NC_PA_CLE_OUTPUT \
                          if not x in data_refi_mlt.columns.tolist()]

        data_refi_mlt = pd.concat([data_refi_mlt, pd.DataFrame([["-"] * len(missing_indics)], \
                                                               index=data_refi_mlt.index, columns=missing_indics)],
                                  axis=1)

        data_refi_mlt["IND03"] = pa.NC_PA_DEM

        data_refi_mlt = com.add_col_bilan(data_refi_mlt)
        data_refi_mlt = com.add_missing_key_cols(data_refi_mlt)
        data_refi_mlt[pa.NC_PA_INDEX] = ["ECH" + str(i) for i in
                                              range(1, data_refi_mlt.shape[0] + 1)]

        data_refi_mlt.reset_index(drop=True)

        data_refi_mlt = com.add_missing_indics(data_refi_mlt)

        wb.Close(False)

        return data_refi_mlt.reset_index(drop=True)

    else:
        logger.warning("       Le fichier REFI-MLT n'a pas été trouvé : " + file_path)
        logger.info("       Pas de PN REFI-MLT générée")
        return pd.DataFrame(columns=["ADZDZDZ"])
