from utils import excel_utils as ex
from datetime import datetime
from modules.scenario.referentials import general_parameters as gp
from modules.scenario.referentials import general_parameters_json as gp_json
import logging
import utils.general_utils as ut
import pandas as pd
import utils.json_utils as jsu
import json
import modules.scenario.parameters.user_parameters as up
import os
from params import version_params as vp

global json_data, json_param


logger = logging.getLogger(__name__)

RN_SHOCKED_TX = 'Shocked TX'
RN_SCENARIO_TAUX = 'CHOC TAUX'
RN_ALL_ETAB = '_ALL_ETAB'


def get_ihm_parameters_json():
    get_json_data_param()
    get_dar()
    get_alim_dir_json()
    get_output_dir_json()
    get_source_dir_json()
    get_mapping_wb_json()
    get_source_type_json()
    get_scenarios_parameters_json()
    get_all_etabs_json()
    get_rates_files_path_json()
    get_curves_to_bootsrap_json()
    get_model_file_paths_json()
    get_pn_a_activer_json()
    get_reference_scenarios_list_json()
    get_holidays_dates_json()
    get_main_sc_eve()

def get_main_sc_eve():
    up.main_sc_eve = jsu.get_value_from_json(json_data, gp_json.js_mainScenEve_key, "")

def get_json_data_param():
    global json_data, json_param, file_path_data, file_path_param
    current_dir = os.getcwd()
    param_dir = os.path.join(current_dir, "PROGRAMMES")
    #le fichier de parametrage param_file contient en plus des données param, le chemin de la config_json qui a été exporté
    file_path_param = param_dir + "\Param_file.json"
    json_param = jsu.read_json_file(file_path_param)
    config_exports_dir = json_param.get("OUTPUT_PATH_JSON_CONFIG")
    file_path_data = jsu.get_most_recent_json_file(config_exports_dir)
    json_data = jsu.read_json_file(file_path_data)

def get_holidays_dates_json():

    holidays_df = jsu.get_dataframe_from_json(json_param, gp_json.js_eur_holiday_key)
    holidays_df['EUR_Holidays'] = holidays_df['EUR_Holidays'].astype(str)
    holidays_df['EUR_Holidays'] = holidays_df['EUR_Holidays'].str.split(n=1, expand=True)[0]
    up.holidays_list = pd.to_datetime(holidays_df['EUR_Holidays'], format='%Y-%m-%d').values.astype('datetime64[D]')

def get_reference_scenarios_list_json():
   up.st_refs = jsu.get_dataframe_from_json(json_param, gp_json.js_ENTITE_SC_REF_STOCK_key)

def get_pn_a_activer_json():
    up.pn_a_activer_df = jsu.get_dataframe_from_json(json_param, gp_json.js_entite_pn_a_activer)


def get_model_file_paths_json():
    modele = jsu.get_value_from_json(json_data, gp_json.js_model_dav_key, "renseigner le nom du fichier MODEL_DAV.xlsx")
    up.modele_dav_path = os.path.join(up.source_dir, "MODELES", modele)


def get_rates_files_path_json():
    up.tx_curves_path = get_input_tx_file_path(gp_json.js_rate_input_key)
    up.liq_curves_path = get_input_tx_file_path(gp_json.js_liq_inputs_key)
    up.tci_curves_path = get_input_tx_file_path(gp_json.js_tci_nmd_key)
    get_input_zc_file_path(gp_json.js_rep_zc_key)


def get_input_zc_file_path(json_key):
    up.zc_file_path = jsu.get_value_from_json(json_data, json_key, "zc_path")
    if up.zc_file_path == "":
        return ""
    up.zc_file_path = os.path.join(up.source_dir, gp.RATE_INPUT_FOLDER, up.zc_file_path)


def get_source_dir_json():
    up.source_dir = jsu.get_value_from_json(json_data, gp_json.js_rep_data_source_key,
                                         "repertoire de données sources")


def get_input_tx_file_path(key_json):
    tx_file_path = jsu.get_value_from_json(json_data, key_json, f"{key_json}")
    tx_file_path = os.path.join(up.source_dir, gp.RATE_INPUT_FOLDER, tx_file_path)
    ut.check_version_templates(tx_file_path.split("\\")[-1], version=vp.version_autre, path=tx_file_path, open=True)
    return tx_file_path


def get_curves_to_bootsrap_json():
    up.curves_to_bootsrapp = jsu.get_dataframe_from_json(json_param, gp_json.js_courbe_bootstrap_key)


def get_source_type_json():
     up.version_sources = jsu.get_value_from_json(
        json_data=json_param,
        key_path=gp_json.js_type_donnees_sources,
        alert_message="le type de données sources n'est pas renseigné"
    )


def get_dar():
    up.dar = jsu.get_value_from_json(
        json_data=json_data,
        key_path=gp_json.js_dar_key,
        alert_message="la dar n'est pas renseignée"
    )
    up.dar = datetime.strptime(up.dar, '%Y-%m-%d')


def get_output_dir_json():
    # global output_dir, dar, name_config
    up.output_path = jsu.get_value_from_json(
        json_data=json_data,
        key_path=gp_json.js_rep_output_scen_key,
        alert_message="vérifier le répertoire de sortie du module scen"
    )

    up.name_config = jsu.get_value_from_json(
        json_data=json_data,
        key_path=gp_json.js_name_config_key,
        alert_message="le nom de la configuration n'est pas renseigné"
    )


    up.output_path = jsu.create_output_directory(up.output_path, up.name_config + "\\" + "SCN")

    up.output_path = os.path.join(up.output_path, '{}' + '_{:%y%m%d}'.format(up.dar)
                                  + '_' + '{:%Y%m%d.%H%M.%S}'.format(datetime.now()))
    up.output_dir = up.output_path.format('SC')

    # Mettre à jour le JSON avec le nouveau chemin dans "execution" -> "REPERTOIRE_SCENARIO"
    json_data["execution"]["REPERTOIRE_SCENARIO"] = up.output_dir

    # Enregistrer le JSON mis à jour dans le fichier
    with open(file_path_data, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)


def get_alim_dir_json():

    up.alim_dir_path = jsu.get_value_from_json(json_data, gp_json.js_rep_output_alim_key,
                                            alert_message="verif etab dans les sorties de l'alim")


def get_mapping_wb_json():
    up.mapping_dir = jsu.get_value_from_json(json_data, gp_json.js_rep_data_source_key,
                                          "repertoire de données sources pour mapping")
    up.mapping_dile_path = os.path.join(up.mapping_dir, 'MAPPING', 'MAPPING_PASS_ALM.xlsx')
    up.mapping_wb = ex.try_close_open(up.mapping_dile_path, read_only=True)
    ut.check_version_templates('MAPPING_PASS_ALM.xlsx', open=False, wb=up.mapping_wb, version=vp.version_map)


def get_scenarios_parameters_json():
    up.pn_bpce_sc_list = get_scenario_pn_bpce_list()
    up.pn_stress_list = get_scenario_stress_pn_list()
    up.pn_ajout_list = get_scenario_ajout_pn_list()
    up.tx_chocs_list = get_scenario_tx_shocks_list()
    up.scenario_list = get_scenarios_list()
    up.stress_dav_list = get_stress_dav_list()
    up.models_list = get_models_list()
    up.scenarii_calc_all, up.scenarii_dav_all = get_calculateur_and_dav_scenarios()


def get_calculateur_and_dav_scenarios():
    up.scenarii_calc_all = jsu.get_dataframe_from_json(json_data, gp_json.js_calculateur_key)
    up.scenarii_dav_all = jsu.get_dataframe_from_json(json_data, gp_json.js_surcouche_key)
    return up.scenarii_calc_all, up.scenarii_dav_all


def get_models_list():
    up.stress_model_list = jsu.get_dataframe_from_json(json_data, gp_json.js_modeles_key)
    return up.stress_model_list


def get_stress_dav_list():
    up.stress_dav_list = jsu.get_dataframe_from_json(json_data, gp_json.js_surcouche_key)
    return up.stress_dav_list


def get_out_of_recalc_st_etabs():
    df = jsu.get_dataframe_from_json(json_param, gp_json.js_etab_sans_rarn_pel_key)
    return list(df.iloc[:, 0])


def get_scenarios_list():
    up.scenario_list = jsu.get_dataframe_from_json(json_data, gp_json.js_param_scen_list_key)
    up.scenario_list.drop_duplicates(inplace=True)
    up.scenario_list[gp.RN_SC_TAUX_USER] = up.scenario_list[gp.RN_SC_TAUX_USER].str.split(',')
    up.scenario_list = up.scenario_list.explode(gp.RN_SC_TAUX_USER)
    up.scenario_list[RN_SHOCKED_TX] = up.scenario_list[gp.RN_SC_TAUX_USER] + '_' + up.scenario_list[RN_SCENARIO_TAUX]
    up.scenario_list['NOM SCENARIO ORIG'] = up.scenario_list['NOM SCENARIO']
    up.scenario_list['NOM SCENARIO'] = up.scenario_list['NOM SCENARIO'] + '_' + up.scenario_list[RN_SHOCKED_TX]
    # Récupérer les établissements et les ajouter à la liste des scénarios
    list_etab_launch = jsu.get_dataframe_from_json(json_data, gp_json.js_liste_entites_exec).iloc[0, :].tolist()
    etablissements_column = list_etab_launch * len(up.scenario_list)
    up.scenario_list['ETABLISSEMENTS'] = etablissements_column[:len(up.scenario_list)]
    up.scenario_list.reset_index(inplace=True, drop=True)
    return up.scenario_list


def get_scenario_tx_shocks_list():
    scenario_list_agreg = jsu.get_dataframe_from_json(json_data, gp_json.js_shock_ir_key)
    up.ir_shock_grp_curve = jsu.get_dataframe_from_json(json_param, gp_json.js_TYPE_REGROUP_COURBE_key)
    tab_corres = up.ir_shock_grp_curve.rename(columns={"TYPE DE COURBE CHOC TAUX": "COURBE_AGREGEE"})
    tab_expanded = pd.merge(
        scenario_list_agreg, tab_corres, left_on="COURBE", right_on="COURBE_AGREGEE", how="left"
    )
    tab_expanded = tab_expanded.drop(columns=["COURBE_AGREGEE"])
    tab_expanded = tab_expanded.rename(columns={"COURBE_y": "COURBE", "COURBE_x": "COURBE_AGREGEE"})
    final_columns_order = [
        "NOM SCENARIO", "TYPE SCENARIO", "DEVISE", "COURBE", "MATURITE",
        "MOIS DEBUT", "MOIS FIN", "VAL. CHOC", "PAS CHOC",
        "SEUIL PIVOT 1", "SEUIL PIVOT 2", "VAL. CHOC SEUIL 1", "VAL. CHOC SEUIL 2"
    ]
    up.scenario_list = tab_expanded[final_columns_order].drop_duplicates()
    return up.scenario_list


def get_scenario_ajout_pn_list():
    up.stress_pn_list = jsu.get_dataframe_from_json(json_data, gp_json.js_add_pn_key)
    return up.stress_pn_list


def get_scenario_stress_pn_list():
    up.stress_pn_list = jsu.get_dataframe_from_json(json_data, gp_json.js_stress_pn_key)
    return up.stress_pn_list


def get_scenario_pn_bpce_list():
    listo = jsu.get_dataframe_from_json(json_data, gp_json.js_retro_pn_bpce_key)
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


def get_all_etabs_json():
    # global all_etabs, codification_etab, scenario_list
    list_all_etabs = jsu.get_dataframe_from_json(json_param, gp_json.js_list_all_etab_key).iloc[0, :].tolist()
    list_all_etabs = [x.strip(' ') for x in list_all_etabs]

    #MODIF POUR IHM, afin d'exécuté uniquement l'établissiement de l'accueil. -->à terme la colonne etablissement du tableau scenario global sera supprimée<-- "
    list_etab_launch = jsu.get_dataframe_from_json(json_data, gp_json.js_liste_entites_exec).iloc[0, :].tolist()

    # Extrait et nettoie les établissements de list_etab_launch
    list_etab_launch = list(set(
        [item.strip() for sublist in [x.split(",") for x in list_etab_launch] for item in sublist]
    ))

    codification = jsu.get_dataframe_from_json(json_param, gp_json.js_codification_etab_key)
    j = 0
    a_virer = [None]
    for x in codification.columns:
        if x in list_etab_launch:
            if x not in up.list_all_etabs:
                list_etab_launch = list_etab_launch + codification.iloc[:, j].values.tolist()
                a_virer = a_virer + [x]
            else:
                logger.warning(
                    x + " est à la fois une liste d'établissements et un nom d'établissement." \
                    + " Veuillez changer le nom de la liste")
        j = j + 1
    list_etab_launch = [x for x in list_etab_launch if x not in a_virer]
    a_virer = []
    for x in list_etab_launch:
        if x not in list_all_etabs:
            a_virer = a_virer + [x]
            logger.warning(
                "L'établissement " + x + " n'est pas un établissement pris en charge ou une liste d'établissements")


    list_etab_launch = [x for x in list_etab_launch if x not in a_virer]
    list_etab_launch = list(set(list_etab_launch))
    up.all_etabs = check_if_etab_data_are_in_alim_output_dir(list_etab_launch, warning=True)
    logger.info("La liste des établissements traitée sera: " + str(up.all_etabs))

    up.codification_etab = jsu.get_dataframe_from_json(json_param, gp_json.js_codification_etab_key)


def check_if_etab_data_are_in_alim_output_dir(etabs_liste, warning=False):
    output_alim_dir_path = jsu.get_value_from_json(json_data, gp_json.js_rep_output_alim_key,
                                                   alert_message="verif etab dans les sorties de l'alim")
    valide_etab_liste = [x for x in etabs_liste if os.path.exists(os.path.join(output_alim_dir_path, x))]
    not_found_etabs = [x for x in etabs_liste if x not in valide_etab_liste]
    if warning:
        if len(not_found_etabs) >= 1:
            logger.warning("Les établissements suivants ne sont pas disponibles dans le dossier de l'ALIM : {}".format(
                ','.join(not_found_etabs)))
    return valide_etab_liste
