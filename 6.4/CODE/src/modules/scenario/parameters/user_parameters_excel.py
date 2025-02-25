from utils import excel_utils as ex
import os
from datetime import datetime
from modules.scenario.referentials import general_parameters as gp
import logging
import utils.general_utils as ut
import numpy as np
import modules.scenario.parameters.user_parameters as up
from params import version_params as vp

logger = logging.getLogger(__name__)

RN_SHOCKED_TX = 'Shocked TX'
RN_SCENARIO_TAUX = 'CHOC TAUX'
RN_ALL_ETAB = '_ALL_ETAB'


def get_ihm_parameters(ref_wb):
    get_dar(ref_wb)
    control_path_existence(ref_wb)
    get_alim_dir(ref_wb)
    get_output_dir(ref_wb)
    get_source_dir(ref_wb)
    get_mapping_wb(ref_wb)
    get_source_type(ref_wb)
    get_scenarios_parameters(ref_wb)
    get_all_etabs(ref_wb)
    get_rates_files_path(ref_wb)
    get_curves_to_bootsrap(ref_wb)
    get_model_file_paths(ref_wb)
    get_pn_a_activer(ref_wb)
    get_reference_scenarios_list(ref_wb)
    get_holidays_dates(ref_wb)
    get_main_sc_eve(ref_wb)

def get_main_sc_eve(wb):
    up.main_sc_eve = ex.get_cell_value_from_range(wb, gp.RN_MAIN_SC_EVE)

def get_holidays_dates(wb):
    holidays_df = ex.get_dataframe_from_range(wb, '_euro_holidays', False)
    holidays_df.drop(0, inplace=True)
    holidays_df[0] = holidays_df[0].astype(str)
    holidays_df[0] = holidays_df[0].str.split(n=1, expand=True)[0]
    up.holidays_list = np.array([datetime.strptime(x, '%Y-%m-%d') for x in holidays_df[0]], dtype='datetime64[D]')

def get_reference_scenarios_list(referential_wb):
    up.st_refs = ex.get_dataframe_from_range(referential_wb, gp.RN_STOCK_REF_TX_SC)

def get_pn_a_activer(referential_wb):
    up.pn_a_activer_df = ex.get_dataframe_from_range(referential_wb, '_PN_A_ACTIVER')

def get_model_file_paths(referential_wb):
    modele = ex.get_value_from_named_ranged(referential_wb, '_MODELE_DAV')
    up.modele_dav_path = os.path.join( up.source_dir, "MODELES", modele)


def get_rates_files_path(ref_wb):
    up.tx_curves_path = get_input_tx_file_path(ref_wb, gp.RN_TX_PATH)
    up.liq_curves_path = get_input_tx_file_path(ref_wb, gp.RN_LIQ_PATH)
    up.tci_curves_path = get_input_tx_file_path(ref_wb, gp.RN_TCI_PATH)
    get_input_zc_file_path(ref_wb, "_ZC_EVE_FILE_PATH")


def get_input_zc_file_path(referential_wb, range_name):
    up.zc_file_path = ex.get_cell_value_from_range(referential_wb, range_name)
    if up.zc_file_path == "":
        return ""
    up.zc_file_path = os.path.join(up.source_dir, gp.RATE_INPUT_FOLDER, up.zc_file_path)


def get_source_dir(referential_wb):
    up.source_dir = ex.get_value_from_named_ranged(referential_wb, '_sources_dir_path')


def get_input_tx_file_path(referential_wb, range_name):
    tx_file_path = ex.get_cell_value_from_range(referential_wb, range_name)
    tx_file_path = os.path.join(up.source_dir, gp.RATE_INPUT_FOLDER, tx_file_path)
    return tx_file_path


def get_curves_to_bootsrap(referential_wb):
    up.curves_to_bootsrapp = ex.get_dataframe_from_range(referential_wb, '_COURBES_A_BOOSTRAPPER')

def get_source_type(ref_wb):
   up.version_sources = ex.get_value_from_named_ranged(ref_wb, gp.RN_TYPE_SOURCES).lower()


def get_dar(wb):
    up.dar = ex.get_cell_value_from_range(wb, gp.RN_DAR)


def get_output_dir(wb):
    output_path = ex.get_cell_value_from_range(wb, gp.RN_OUTPUT_DIR)
    output_path = os.path.join(output_path, '{}' + '_DAR-{:%Y%m%d}'.format(up.dar)
                               + '_EXEC-' + '{:%Y%m%d.%H%M.%S}'.format(datetime.now()))
    up.output_dir = output_path.format('SC')


def get_alim_dir(wb):
   up.alim_dir_path = ex.get_cell_value_from_range(wb, gp.RN_SCENARIO_FILE_PATH)

def get_mapping_wb(ref_wb):
    up.mapping_dir = ex.get_value_from_named_ranged(ref_wb, '_sources_dir_path')
    up.mapping_dile_path = os.path.join(up.mapping_dir, 'MAPPING', 'MAPPING_PASS_ALM.xlsx')
    up.mapping_wb = ex.try_close_open(up.mapping_dile_path, read_only=True)
    ut.check_version_templates('MAPPING_PASS_ALM.xlsx', open=False, wb=up.mapping_wb, version=vp.version_map)
    ex.unfilter_all_sheets(up.mapping_wb)


def get_scenarios_parameters(ref_wb):
    up.pn_bpce_sc_list = get_scenario_pn_bpce_list(ref_wb)
    up.pn_stress_list = get_scenario_stress_pn_list(ref_wb)
    up.pn_ajout_list = get_scenario_ajout_pn_list(ref_wb)
    up.tx_chocs_list = get_scenario_tx_shocks_list(ref_wb)
    up.scenario_list = get_scenarios_list(ref_wb)
    up.stress_dav_list = get_stress_dav_list(ref_wb)
    up.models_list = get_models_list(ref_wb)
    up.scenarii_calc_all, up.scenarii_dav_all = get_calculateur_and_dav_scenarios(ref_wb)

def get_calculateur_and_dav_scenarios(referential_wb):
    up.scenarii_calc_all = ex.get_dataframe_from_range(referential_wb, gp.RN_SCENARIO_RC_ST)
    up.scenarii_dav_all = ex.get_dataframe_from_range(referential_wb, gp.RN_SCENARIO_SRC_DAV)
    return up.scenarii_calc_all, up.scenarii_dav_all


def get_models_list(referential_wb):
    stress_dav_list = ex.get_dataframe_from_range(referential_wb, '_SC_MOD')
    return stress_dav_list

def get_stress_dav_list(referential_wb):
    models_list = ex.get_dataframe_from_range(referential_wb, gp.RN_SCENARIO_SRC_DAV)
    return models_list


def get_scenarios_list(referential_wb):
    up.scenario_list = ex.get_dataframe_from_range(referential_wb, gp.RN_SCENARIO_LIST_CELL)
    up.scenario_list.drop_duplicates(inplace=True)
    up.scenario_list[gp.RN_SC_TAUX_USER] = up.scenario_list[gp.RN_SC_TAUX_USER].str.split(',')
    up.scenario_list = up.scenario_list.explode(gp.RN_SC_TAUX_USER)
    up.scenario_list[RN_SHOCKED_TX] = up.scenario_list[gp.RN_SC_TAUX_USER] + '_' + up.scenario_list[RN_SCENARIO_TAUX]
    up.scenario_list['NOM SCENARIO ORIG'] = up.scenario_list['NOM SCENARIO']
    up.scenario_list['NOM SCENARIO'] = up.scenario_list['NOM SCENARIO'] + '_' + up.scenario_list[RN_SHOCKED_TX]
    up.scenario_list.reset_index(inplace=True, drop=True)
    return up.scenario_list


def get_scenario_tx_shocks_list(referential_wb):
    up.scenario_list = ex.get_dataframe_from_range(referential_wb, gp.RN_SHOCKS_FST_CELL)
    return up.scenario_list


def get_scenario_ajout_pn_list(referential_wb):
    up.stress_pn_list = ex.get_dataframe_from_range(referential_wb, gp.RN_ADD_PN_FST_CELL)
    return up.stress_pn_list


def get_scenario_stress_pn_list(referential_wb):
    up.stress_pn_list = ex.get_dataframe_from_range(referential_wb, gp.RN_STRESS_PN_FST_CELL)
    return up.stress_pn_list


def get_scenario_pn_bpce_list(referential_wb):
    listo = ex.get_dataframe_from_range(referential_wb, gp.RN_SC_PN_BPCE_FST_CELL)
    return listo


def control_path_existence(ref_wb):
    for patho in [gp.RN_SOURCES_DIR, gp.RN_SCENARIO_FILE_PATH, gp.RN_OUTPUT_DIR, gp.RN_TX_PATH]:
        path_dir = ex.get_value_from_named_ranged(ref_wb, patho, alert="")
        if (not os.path.exists(path_dir) and patho != gp.RN_TX_PATH) \
                or (os.path.exists(os.path.join(gp.RN_SOURCES_DIR, "RATE_INPUT", path_dir)) and patho == gp.RN_TX_PATH):
            logger.error("Le chemin suivant n'existe pas : " + path_dir \
                         + "  Veuillez vérifier vos paramètres")
            raise ImportError("Le chemin suivant n'existe pas : " + path_dir \
                              + " Veuillez vérifier vos paramètres")


def get_all_etabs(referential_wb):
    list_all_etabs = ex.get_dataframe_from_range(referential_wb, RN_ALL_ETAB).iloc[:, 0].tolist()
    list_all_etabs = [x.strip(' ') for x in list_all_etabs]
    list_etab = list(set(
        [item for sublist in [x.split(",") for x in up.scenario_list['ETABLISSEMENTS'].values.tolist()] for item in
         sublist]))
    list_etab = [x.strip(' ') for x in list_etab]
    codification = ex.get_dataframe_from_range(referential_wb, '_codification_etab')
    j = 0
    a_virer = [None]
    for x in codification.columns:
        if x in list_etab:
            if x not in list_all_etabs:
                list_etab = list_etab + codification.iloc[:, j].values.tolist()
                a_virer = a_virer + [x]
            else:
                logger.warning(
                    x + " est à la fois une liste d'établissements et un nom d'établissement." \
                    + " Veuillez changer le nom de la liste")
        j = j + 1
    list_etab = [x for x in list_etab if x not in a_virer]
    a_virer = []
    for x in list_etab:
        if x not in list_all_etabs:
            a_virer = a_virer + [x]
            logger.warning(
                "L'établissement " + x + " n'est pas un établissement pris en charge ou une liste d'établissements")
    list_etab = [x for x in list_etab if x not in a_virer]
    list_etab = list(set(list_etab))
    up.all_etabs = check_if_etab_data_are_in_alim_output_dir(referential_wb, list_etab, warning=True)
    logger.info("La liste des établissements traitée sera: " + str(up.all_etabs))

    up.codification_etab = ex.get_dataframe_from_range(referential_wb, '_codification_etab')


def check_if_etab_data_are_in_alim_output_dir(referential_wb, etabs_liste, warning=False):
    output_alim_dir_path = ex.get_cell_value_from_range(referential_wb, gp.RN_SCENARIO_FILE_PATH)
    valide_etab_liste = [x for x in etabs_liste if os.path.exists(os.path.join(output_alim_dir_path, x))]
    not_found_etabs = [x for x in etabs_liste if x not in valide_etab_liste]
    if warning:
        if len(not_found_etabs) >= 1:
            logger.warning("Les établissements suivants ne sont pas disponibles dans le dossier de l'alim : {}".format(
                ','.join(not_found_etabs)))
    return valide_etab_liste
