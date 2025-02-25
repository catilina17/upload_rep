import numpy as np
import pandas as pd
import utils.excel_utils as ex
from modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_call import generate_PN_call
from modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_refi_ajust import generate_PN_ajustements
from modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_refi_ct import generate_PN_REFI_CT
from modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_refi_mlt import generate_PN_REFI_MLT
from modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_refi_sc import generate_PN_REFI_SC
import modules.scenario.services.pn_services.pn_bpce_services.pn_bpce_common as com
import modules.scenario.services.pn_services.pn_bpce_services.mapping_service as mapp
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import utils.excel_utils as excel_helper
from modules.scenario.parameters import user_parameters as up
import modules.scenario.services.pn_services.pn_bpce_services.referential_bpce as rf_bpce
from modules.scenario.services.input_output_services.input_files_nomenclature import NomenclatureSaver
import logging

logger = logging.getLogger(__name__)


def get_PN_BPCE_service(pn_bpce_sc_list, scenario_rows, wb, etab, mapping_wb, scenario_name):
    if etab != "BPCE":
        return
    sc_bpce = pn_bpce_sc_list[pn_bpce_sc_list["NOM SCENARIO"] == scenario_rows["PN BPCE"].iloc[0]]
    if sc_bpce.shape[0] > 0:
        logger.info("    Génération des PN ECH pour BPCE")
        generate_PN_BPCE(sc_bpce.iloc[0, 1], sc_bpce.iloc[0, 2], wb, etab,
                         mapping_wb, scenario_name)
    else:
        logger.info("    Pas de PN ECH pour BPCE")


def generate_PN_BPCE(NG_SC_PN_AUTRES_BASSINS, NG_COMPILS_AUTRES_BASSINS, export_wb, etab,
                     mapping_wb, scenario_name):
    mapp.load_general_mappings(mapping_wb)

    xl = excel_helper.excel
    nom = NomenclatureSaver()
    dar = up.dar

    path_call = nom.get_source_files_path(mapping_wb, etab, "CALL", dar)
    path_refi_ct = nom.get_source_files_path(mapping_wb, etab, "REFI-CT", dar)
    path_refi_mlt = nom.get_source_files_path(mapping_wb, etab, "REFI-MLT", dar)

    pn_ech_ajust = generate_PN_ajustements(NG_COMPILS_AUTRES_BASSINS, etab, scenario_name)
    pn_ech_refi_bassins = generate_PN_REFI_SC(xl, NG_SC_PN_AUTRES_BASSINS, etab, scenario_name)
    pn_ech_refi_ct = generate_PN_REFI_CT(xl, etab, path_refi_ct)
    pn_ech_refi_mlt = generate_PN_REFI_MLT(xl, dar, etab, path_refi_mlt)
    pn_calls = generate_PN_call(dar, etab, path_call)

    pn_ech = pd.concat([pn_ech_refi_mlt, pn_ech_refi_ct, pn_calls, pn_ech_ajust, pn_ech_refi_bassins])

    if pn_ech.shape[0] > 0:
        pn_ech = map_pn_bpce(pn_ech)
        pn_ech = final_format(pn_ech)
        export_wb.Sheets("PN ECH").Visible = True
        ex.write_df_to_range_name2(pn_ech, "_pn_ech", export_wb, header=False, offset=2)

    else:
        logger.warning(
            "     Un scénario BPCE a été ajouté mais aucun fichier n'a été trouvé ou les données procurées sont vides")


def final_format(pn_ech):
    col_sortie = pa.NC_PA_COL_SORTIE_QUAL_ECH + pa.NC_PA_COL_SORTIE_NUM_PN

    pn_ech[pa.NC_PA_LCR_TIERS_SHARE] = 100

    missing_indics = [x for x \
                      in col_sortie \
                      if not x in pn_ech.columns.tolist()]

    pn_ech = pd.concat([pn_ech, pd.DataFrame([[""] * len(missing_indics)], \
                                             index=pn_ech.index, columns=missing_indics)], axis=1)

    pn_ech = com.generate_index(pn_ech, repeat=4)

    pn_ech = pn_ech[col_sortie].copy()

    pn_ech[pa.NC_PA_COL_SORTIE_NUM_PN] = pn_ech[pa.NC_PA_COL_SORTIE_NUM_PN].fillna("")

    return pn_ech

def map_pn_bpce(pn_ech):
    filtres = [pn_ech[pa.NC_PA_CONTRACT_TYPE].str[0:2] == "A-", pn_ech[pa.NC_PA_CONTRACT_TYPE].str[0:2] == "P-"]
    choices = ["B ACTIF", "B PASSIF"]
    pn_ech[pa.NC_PA_BILAN] = np.select(filtres, choices)

    pn_ech[pa.NC_PA_SCOPE] = rf_bpce.SCOPE_MNI_LIQUIDITE

    CLES = pa.NC_PA_CLE_OUTPUT
    pn_ech[CLES] = pn_ech[CLES].fillna("-").replace("", "-")

    pn_ech = mapp.mapping_consolidation_liquidite(pn_ech)

    pn_ech = mapp.map_data(pn_ech, mapp.mapping["mapping_global"]["CONTRATS"], \
                           keys_data=[pa.NC_PA_CONTRACT_TYPE], name_mapping="PN ECH DATA vs.")

    return pn_ech
















