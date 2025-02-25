# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:01:46 2020

CHARGEMENT DES PARAMETRES UTILISATEURS
"""

import modules.moteur.parameters.general_parameters as gp
import modules.moteur.parameters.general_parameters_json as gp_json
import modules.moteur.parameters.authorized_users as au
import getpass
import pandas as pd
import modules.moteur.utils.generic_functions as gf
import dateutil
from pathlib import Path
import datetime
from os import path
import os
import logging
import re
import utils.json_utils as jsu
import modules.moteur.parameters.user_parameters as up

logger = logging.getLogger(__name__)

up.max_month_pn = {}
up.type_simul = {}

global json_data, json_param


def load_users_param_json():
    get_json_data_param()

    check_user_credentials_json()

    get_input_path_json()

    get_source_path_json()

    get_map_path_json()

    load_projection_period_params_json()

    load_ajustements_options_json()

    load_output_indicators_json()

    load_output_axes_json()

    load_conv_ecoulements_options_json()

    load_coeff_gap_gestion_json()

    load_tla_refixing_params_json()

    load_pn_max_month_json()

    load_compil_format_json()

    get_user_list_etabs_json()

    # get_main_sc_eve()


def get_json_data_param():
    global json_data, json_param, file_path_data, file_path_param
    current_dir = os.getcwd()
    param_dir = os.path.join(current_dir, "PROGRAMMES")
    # le fichier de parametrage param_file contient en plus des données param, le chemin de la config_json qui a été exporté
    file_path_param = param_dir + "\Param_file.json"
    json_param = jsu.read_json_file(file_path_param)
    config_exports_dir = json_param.get("OUTPUT_PATH_JSON_CONFIG")
    file_path_data = jsu.get_most_recent_json_file(config_exports_dir)
    json_data = jsu.read_json_file(file_path_data)




def get_user_list_etabs_json():
    up.list_all_etab = jsu.get_value_from_json(json_param, gp_json.js_list_all_etab_key, "")
    up.list_etab_usr = jsu.get_value_from_json(json_data, gp_json.js_list_etab_usr_key, "")


def get_input_path_json():
    up.input_path = jsu.get_value_from_json(json_data, gp_json.js_scen_path_usr_key,
                                            alert_message="Le chemin du dossier scénario n'est pas renseigné")


def get_source_path_json():
    up.source_path = jsu.get_value_from_json(json_data, gp_json.js_path_source_key, "")



def get_map_path_json():
    up.mapping_path = os.path.join(up.source_path, "MAPPING", "MAPPING_PASS_ALM.xlsx")


def load_output_path_json():
    """DAR"""
    up.dar_usr = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_dar_key,
                                         alert_message="la dar n'est pas renseignée")
    up.dar_usr = dateutil.parser.parse(str(up.dar_usr)).replace(tzinfo=None)

    """ Chargement du répertoire de sortie"""
    up.output_path_usr = jsu.get_value_from_json(json_data, gp_json.js_out_path_key,
                                                 alert_message="Le chemin du dossier de sortie n'est pas renseigné")
    name_config = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_name_config_key,
                                          alert_message="la dar n'est pas renseignée")
    if up.output_path_usr:
        up.output_path_usr = jsu.create_output_directory(up.output_path_usr, name_config + "\\" + "MOT")

    if not path.exists(up.output_path_usr):
        raise ValueError("Le chemin du dossier de sortie n'existe pas")

    """ UPDATE OUTPUT_PATH"""
    now = datetime.datetime.now()
    now = now.strftime("%Y%m%d.%H%M.%S")
    new_dir = str(up.dar_usr.strftime('%y')) + str(up.dar_usr.month) + str(up.dar_usr.day) + "_" + str(now)
    up.output_path_usr = os.path.join(up.output_path_usr, "MOT_" + new_dir)
    nb_car_output = len(up.output_path_usr)
    if nb_car_output > 150:
        logger.warning(
            "le nombre de caractères du chemin de sortie SCN est supérieur à 150. La limite Excel risque d'être dépassée")
    Path(up.output_path_usr).mkdir(parents=True, exist_ok=True)


def load_tla_refixing_params_json():
    """ TX REFIXING """
    up.retraitement_tla = jsu.get_value_from_json(json_data, gp_json.js_refixing_tla_key,
                                                  alert_message="retraitement_tla")
    up.retraitement_tla = True if up.retraitement_tla.upper() == "OUI" else False
    up.date_refix_tla = jsu.get_value_from_json(json_data, gp_json.js_date_refix_tla_key,
                                                alert_message="date_refix_tla")
    up.date_refix_tla = dateutil.parser.parse(str(up.date_refix_tla)).replace(tzinfo=None)
    up.mois_refix_tla = up.date_refix_tla.year * 12 + up.date_refix_tla.month - up.dar_usr.year * 12 - up.dar_usr.month
    try:
        up.freq_refix_tla = jsu.get_value_from_json(json_data, gp_json.js_freq_refixing_tla_key,
                                                    alert_message="freq_refix_tla")
    except:
        up.freq_refix_tla = 4
    if up.mois_refix_tla < 1 and up.retraitement_tla:
        logger.error("   La date de refixing TLA est inférieure ou égale à la DAR")
        raise ValueError("   La date de refixing TLA est inférieure ou égale à la DAR")

    return up.retraitement_tla, up.mois_refix_tla, up.freq_refix_tla


def load_output_indicators_json():
    """ DETERMINATION DES INDICATEURS SORTIES"""
    ind_classique = jsu.get_dataframe_from_json(json_data, gp_json.js_ind_classique_key)
    ind_lcr_nsfr = jsu.get_dataframe_from_json(json_data, gp_json.js_ind_lcr_nsfr_key)
    contrib_an = jsu.get_dataframe_from_json(json_data, gp_json.js_ind_contrib_an_key)

    up.data_indic = pd.concat([ind_classique, ind_lcr_nsfr, contrib_an], axis=0).reset_index(drop=True)
    up.data_indic['RESTITUER'] = up.data_indic['RESTITUER'].replace({True: 'OUI', False: 'NON'})
    up.data_indic = up.data_indic.rename(columns={'RESTITUER': 'Restituer'})

    up.type_eve = jsu.get_value_from_json(json_data, gp_json.js_type_eve_key, alert_message="type_eve")

    up.data_indic_eve = jsu.get_dataframe_from_json(json_data, gp_json.js_ind_eve_key)
    up.data_indic_eve['RESTITUER'] = up.data_indic_eve['RESTITUER'].replace({True: 'OUI', False: 'NON'})
    up.data_indic_eve = up.data_indic_eve.rename(columns={'RESTITUER': 'Restituer'})


def load_coeff_gap_gestion_json():
    Table_GAP_gestion1 = jsu.get_dataframe_from_json(json_data, gp_json.js_gap_gestion1_key)
    Table_GAP_gestion2 = jsu.get_dataframe_from_json(json_data, gp_json.js_gap_gestion2_key)

    up.coeff_tf_tla_usr = Table_GAP_gestion1.loc[Table_GAP_gestion1['TYPE'] == "GAP G TF", 'TLA'].values[0]
    up.coeff_tf_tlb_usr = Table_GAP_gestion1.loc[Table_GAP_gestion1['TYPE'] == "GAP G TF", 'TLB'].values[0]
    up.coeff_tf_cel_usr = Table_GAP_gestion1.loc[Table_GAP_gestion1['TYPE'] == "GAP G TF", 'CEL'].values[0]

    up.coeff_inf_tla_usr = Table_GAP_gestion1.loc[Table_GAP_gestion1['TYPE'] == "GAP G INF", 'TLA'].values[0]
    up.coeff_inf_tlb_usr = Table_GAP_gestion1.loc[Table_GAP_gestion1['TYPE'] == "GAP G INF", 'TLB'].values[0]
    up.coeff_inf_cel_usr = Table_GAP_gestion1.loc[Table_GAP_gestion1['TYPE'] == "GAP G INF", 'CEL'].values[0]

    up.coeff_reg_tf_usr = Table_GAP_gestion2.loc[Table_GAP_gestion2['TYPE'] == "GAP REG", 'TF'].values[0]
    up.coeff_reg_inf_usr = Table_GAP_gestion2.loc[Table_GAP_gestion2['TYPE'] == "GAP REG", 'INF'].values[0]


def load_ajustements_options_json():
    """ NE SORTIR QUE LES AJUSTEMENTS"""
    up.ajust_only = jsu.get_value_from_json(json_data, gp_json.js_ajust_only_key, "")
    up.ajust_only = False if up.ajust_only == "NON" else True

    ajustement_bilan = jsu.get_dataframe_from_json(json_data, gp_json.js_ajust_bilan_tab_key)

    """ SPREAD LIQ AJUST """
    up.spread_liq = ajustement_bilan.iloc[:, [0, 1]]  # à voir pour remplacer les "." par les ","
    up.spread_liq = up.spread_liq.rename(columns={'SPREADS DE LIQUIDITE (BPS)': 'TAUX (bps)'}).copy()

    """ COURBES AJUST """
    up.courbes_ajust = ajustement_bilan.iloc[:, [0, 2]]
    up.courbes_ajust = up.courbes_ajust.rename(columns={'COURBE MNI AJUST': 'COURBE'}).copy()
    up.courbes_ajust = up.courbes_ajust.set_index("DEVISE")


def load_output_axes_json():
    up.cols_sortie_excl = jsu.get_dataframe_from_json(json_data, gp_json.js_axe_sortie_key)
    up.cols_sortie_excl['RESTITUER'] = up.cols_sortie_excl['RESTITUER'].replace({True: 'OUI', False: 'NON'})
    up.cols_sortie_excl = up.cols_sortie_excl.rename(columns={'RESTITUER': 'Restituer'})
    up.cols_sortie_excl = up.cols_sortie_excl[up.cols_sortie_excl[gp.nc_col_sort_restituer] == "NON"][
        gp.nc_axe_nom].values.tolist()


def load_pn_max_month_json():
    mois_pn = jsu.get_dataframe_from_json(json_data, gp_json.js_pn_a_utiliser_key)
    up.max_month_pn = dict([(str(i).lower(), int(j)) for i, j in zip(mois_pn["PN"], mois_pn["MOIS"])])


def load_projection_period_params_json():
    up.nb_mois_proj_out = jsu.get_value_from_json(json_data, gp_json.js_h_proj_key, "")
    if up.nb_mois_proj_out == "" or up.nb_mois_proj_out is None:
        up.nb_mois_proj_out = 60
    try:
        float(str(up.nb_mois_proj_out))
        up.nb_mois_proj_out = int(up.nb_mois_proj_out)
    except ValueError:
        up.nb_mois_proj_out = 60

    if up.nb_mois_proj_out > gp.max_months:
        raise ValueError("La projection ne peut excéder 240 mois")

    up.nb_annees_usr = int(up.nb_mois_proj_out / 12)


def load_conv_ecoulements_options_json():
    up.force_gp_liq = jsu.get_value_from_json(json_data, gp_json.js_force_gp_liq_key, "")
    up.force_gp_liq = False if up.force_gp_liq == "NON" else True

    """ OPTION POUR FORCER LES GAPS DES NMDS """
    up.force_gps_nmd = jsu.get_value_from_json(json_data, gp_json.js_force_gps_nmd_key, "")
    up.force_gps_nmd = False if up.force_gps_nmd == "NON" else True


def load_compil_format_json():
    compil_format = str(
        jsu.get_value_from_json(json_param, gp_json.js_compil_format, "erreur compil format")).strip()
    up.compil_decimal = re.findall(r'"([^"]*)"', compil_format)[1]
    up.compil_sep = re.findall(r'"([^"]*)"', compil_format)[0]
    up.merge_compil = jsu.get_value_from_json(json_param, gp_json.js_fusion_compil, "erreur fusion compil")
    up.merge_compil = up.merge_compil[0].strip() == "OUI"


def check_user_credentials_json():
    if True:
        user_win = os.getlogin().upper()
        user_win2 = str(getpass.getuser()).upper()
        if not gf.begin_with_list(au.users_win, user_win) and gf.begin_with_list(au.users_win, user_win2):
            logger.error("YOU ARE NOT AN AUTHORIZED USER for THIS APP")
            raise ValueError("YOU ARE NOT AN AUTHORIZED USER for THIS APP")


