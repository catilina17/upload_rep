import os
import numpy as np
import pandas as pd
import utils.excel_utils as ex
import utils.general_utils as ut
import modules.scenario.services.pn_services.pn_bpce_services.mapping_service as mapp
import modules.scenario.services.pn_services.pn_bpce_services.referential_bpce as rf_bpce
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import modules.scenario.referentials.general_parameters as gp
import params.version_params as vp
from utils.excel_utils import try_close_workbook
import logging
import modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_common as com

logger = logging.getLogger(__name__)


def generate_PN_REFI_CT(xl, etab, file_path):
    wb = None
    try:
        if os.path.exists(file_path):

            logger.info("       GENERATION DES OPE DE REFI CT")

            logger.info("          Lecture de : " + file_path.split("\\")[-1])
            wb = xl.Workbooks.Open(file_path, None, True)
            ut.check_version_templates(file_path.split("\\")[-1], version=vp.version_autre, wb=wb, warning=True)

            mapping_conso_ech = mapp.mapping["mapping_PN"]["mapping_CONSO_ECH"]
            mapping_refi_bpce = mapp.mapping["mapping_BPCE"]["REFI BPCE"]
            data_repartition = ex.get_dataframe_from_range(wb, rf_bpce.NG_REFI_REPARTITION, header=True)
            cols_mat = data_repartition.columns.tolist()
            data_repartition = np.array(data_repartition)
            evol_prct = ex.get_value_from_named_ranged(wb, rf_bpce.NG_EVOL_SUP)
            mois_borne = ex.get_value_from_named_ranged(wb, rf_bpce.NG_MOIS_BORNE)

            data_map = ex.get_dataframe_from_range(wb, rf_bpce.NG_REFI_MAP, header=True)
            data_map.index = ["UNIQUE"]
            data_montants = np.array(ex.get_dataframe_from_range(wb, rf_bpce.NG_REFI_MONTANT, header=True))

            dic_montant = {}
            index_lcr = [rf_bpce.INDEX_LCR_60, rf_bpce.INDEX_LCR_100]
            i = 0
            for lcr in ["lcr60", "lcr100"]:
                data_temp = data_repartition * ((data_montants[:, index_lcr[i]]).reshape(data_montants.shape[0], 1))
                data_temp = np.concatenate(
                    [data_temp] + [data_temp * (1 + evol_prct)] * (int(pa.MAX_MONTHS_PN / mois_borne)))[
                            0:pa.MAX_MONTHS_PN]
                dic_montant[lcr] = pd.DataFrame(data_temp).transpose()
                dic_montant[lcr].columns = pa.NC_PA_COL_SORTIE_NUM_PN2
                dic_montant[lcr][pa.NC_PA_MATURITY_DURATION] = cols_mat
                dic_montant[lcr]["type_lcr"] = lcr
                dic_montant[lcr] = dic_montant[lcr][(dic_montant[lcr][pa.NC_PA_COL_SORTIE_NUM_PN2].sum(axis=1) != 0)]
                i = i + 1
            data_refi_ct = pd.concat([dic_montant[key] for key in list(dic_montant.keys())])

            if data_refi_ct.shape[0] != 0:
                data_refi_ct["INDEXO"] = "UNIQUE"
                data_refi_ct = mapp.map_data(data_refi_ct, data_map, keys_data=["INDEXO"], option="", error_mapping=False)
                data_refi_ct[pa.NC_PA_LCR_TIERS].mask(data_refi_ct["type_lcr"] == "lcr60",
                                                        data_refi_ct[rf_bpce.NC_LCR_TIERS_60],
                                                        inplace=True)
                data_refi_ct = mapp.map_data(data_refi_ct, mapping_conso_ech, keys_data=[pa.NC_PA_CONTRACT_TYPE],
                                             join_how="inner")
                data_refi_ct = data_refi_ct.drop([pa.NC_PA_MATUR], axis=1)
                data_refi_ct = mapp.map_data(data_refi_ct, mapping_refi_bpce, keys_data=[pa.NC_PA_CONTRACT_TYPE],
                                             join_how="inner")

                data_refi_ct = data_refi_ct.drop(columns=["INDEXO", "type_lcr"])

                data_refi_ct[pa.NC_PA_BASSIN] = etab

                data_refi_ct = data_refi_ct.reset_index(drop=True)

                data_refi_ct["IND03"] = pa.NC_PA_DEM

                data_refi_ct = com.add_col_bilan(data_refi_ct)
                data_refi_ct = com.add_missing_key_cols(data_refi_ct)
                data_refi_ct[pa.NC_PA_INDEX] = ["ECH" + str(i) for i in
                                                   range(1, data_refi_ct.shape[0] + 1)]

                data_refi_ct = com.add_missing_indics(data_refi_ct)

                wb.Close(False)

        else:
            data_refi_ct = pd.DataFrame(columns=["ADZDZDZ"])
            logger.warning("       Le fichier REFI-CT n'a pas été trouvé : " + file_path)
            logger.info("       Pas de PN REFI-CT générée")

    except Exception as e:
        try_close_workbook(wb, "REFI_CT_FILE", False)
        raise e

    return data_refi_ct