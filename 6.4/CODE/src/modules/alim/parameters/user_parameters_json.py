import datetime
import dateutil
import pandas as pd
import mappings.general_mappings as mp
import modules.alim.parameters.general_parameters as gp
import modules.alim.parameters.general_parameters_json as gp_json
import modules.alim.parameters.user_parameters as up
import logging
from pathlib import Path
import os
import utils.excel_utils as ex
import utils.json_utils as jsu
import utils.general_utils as gu
import modules.alim.parameters.RZO_params as rzo_p
import modules.alim.parameters.NTX_SEF_params as ntx_sef_p
from params import sources_params as sp
from params import version_params as vp

logger = logging.getLogger(__name__)

global json_data, json_param


def get_json_data_param():
    global json_data, json_param, file_path_alim, file_path_param, param_dir
    current_dir = os.getcwd()
    param_dir = os.path.join(current_dir, "PROGRAMMES")
    # le fichier de parametrage param_file contient en plus des données param, le chemin de la config_json qui a été exporté
    file_path_param = param_dir + "\Param_file.json"
    json_param = jsu.read_json_file(file_path_param)
    file_path_alim = param_dir + "\Alim_file.json"
    json_data = jsu.read_json_file(file_path_alim)


def get_user_main_params_json():
    up.output_folder = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_rep_output_alim_key,
                                               alert_message="le répertoire n'est pas renseigné dans l'alim")

    vp.version_sources = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_dar_key,
                                                 alert_message="la dar n'est pas renseignée")
    vp.version_sources = vp.version_sources.replace(" ", "").lower()

    up.path_map_mis_template_file = param_dir + sp.path_map_mis_template_file
    up.path_stock_template_file = param_dir + sp.path_stock_template_file
    up.path_sc_tx_template_file = param_dir + sp.path_sc_tx_template_file
    up.path_sc_vol_template_file = param_dir + sp.path_sc_vol_template_file
    up.path_sc_lcr_nsfr_template_file = param_dir + sp.path_sc_lcr_nsfr_template_file

    gu.check_version_templates("MAPPINGS_MANQUANTS.xlsb", path=up.path_map_mis_template_file, open=True,
                               version=vp.version_AUTRES_TEMP)
    gu.check_version_templates("STOCK_TEMPLATE.xlsb", path=up.path_stock_template_file, open=True,
                               version=vp.version_STOCK_TMP)
    gu.check_version_templates("SC_TAUX_TEMPLATE.xlsb", path=up.path_sc_tx_template_file, open=True,
                               version=vp.version_SC_TX)
    gu.check_version_templates("SC_VOLUME_TEMPLATE.xlsb", path=up.path_sc_vol_template_file, open=True,
                               version=vp.version_SC_VOL)
    gu.check_version_templates("SC_LCR_NSFR_TEMPLATE.xlsb", path=up.path_sc_lcr_nsfr_template_file, open=True,
                               version=vp.version_AUTRES_TEMP)

    dar_usr = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_dar_key,
                                      alert_message="la dar n'est pas renseignée")
    up.dar = dateutil.parser.parse(str(dar_usr)).replace(tzinfo=None)

    nom_simu_usr = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_nom_simu_key,
                                           alert_message="le nom de la simulation n'est pas renseigné")

    up.name_run = nom_simu_usr if nom_simu_usr is not None else ""

    sc_ref_nmd_user = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_scen_ref_nmd_key,
                                              alert_message="le scenario de ref nmd n'est pas renseigné")
    up.sc_ref_nmd = str(sc_ref_nmd_user)

    liste_entites_usr = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_liste_entites_alim,
                                                alert_message="la liste des établissements n'est pas renseignée")
    up.etabs = str(liste_entites_usr).replace(" ", "").split(",")

    up.sources_folder = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_rep_data_source_key,
                                                alert_message="le repertoire des dossiers sources n'est pas renseigné")

    mapper_ray_user = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_mapper_ray_key,
                                              alert_message="le mapper ray n'est pas renseignée")

    up.map_lcr_nsfr_g = True if mapper_ray_user == "OUI" else False

    rate_input_file_name_usr = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_rate_input_key,
                                                       alert_message="le fichier rate input n'est pas renseigné dans l'alim")
    up.rate_file_path = os.path.join(up.sources_folder, "RATE_INPUT", rate_input_file_name_usr)

    gu.check_version_templates(up.rate_file_path.split("\\")[-1], version=vp.version_rate, path=up.rate_file_path,
                               open=True)

    liq_input_file_name_usr = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_liq_inputs_key,
                                                      alert_message="le fichier liq input n'est pas renseigné dans l'alim")

    up.liq_file_path = os.path.join(up.sources_folder, "RATE_INPUT", liq_input_file_name_usr)
    gu.check_version_templates(up.liq_file_path.split("\\")[-1], version=vp.version_rate, path=up.liq_file_path,
                               open=True)

    if len([x for x in up.etabs if x not in gp.NTX_FORMAT]) > 0:
        modele_nmd_alim_usr = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_model_nmd_key,
                                                      alert_message="le modèle nmd n'est pas renseigné dans l'alim")
        up.modele_nmd_file_path = os.path.join(up.sources_folder, "MODELES", modele_nmd_alim_usr)
        gu.check_version_templates(up.modele_nmd_file_path.split("\\")[-1], version=vp.version_modele_nmd,
                                   path=up.modele_nmd_file_path, open=True)

        modele_pel_alim_usr = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_model_pel_key,
                                                      alert_message="le modèle pel n'est pas renseigné dans l'alim")
        up.modele_pel_file_path = os.path.join(up.sources_folder, "MODELES", modele_pel_alim_usr)
        gu.check_version_templates(up.modele_pel_file_path.split("\\")[-1], version=vp.version_modele_pel,
                                   path=up.modele_pel_file_path, open=True)

    int_ig_usr = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_int_gestion_key,
                                         alert_message="le choix de l'intention de gestion n'est pas renseigné dans l'alim")
    up.detail_ig = True if int_ig_usr == "OUI" else False

    up.mapp_file = up.sources_folder + sp.mapp_file
    if not os.path.exists(up.mapp_file):
        raise ValueError("Le fichier de MAPPING est absent du dossier des SOURCES: " + str(up.mapp_file))

    up.type_simul = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_type_simu_key,
                                            alert_message="le type de simulation n'est pas renseigné dans l'alim")
    # up.mois_inter_trim = accueil_ws.Range("_MOIS_INTERTRIMESTRIEL").Value
    # up.chemin_compils = accueil_ws.Range("_CHEMIN_COMPILS").Value
    up.mois_inter_trim = 3
    up.chemin_compils = r"C:\Users\HOSSAYNE\Documents\BPCE_ARCHIVES\RESULTATS\TEST_CALCULATEUR\COMPILS"


def generate_etab_parameters_json(etab):
    if etab in gp.NTX_FORMAT:
        ntx_sef_p.perim_ntx = jsu.get_value_from_json(json_data=json_data, key_path=gp_json.js_perim_natixis_key,
                                                      alert_message="le perim ntx n'est pas renseigné dans l'alim")
        logger.info("Le périmètre choisi est : %s" % ntx_sef_p.perim_ntx)
        ntx_sef_p.depart_decale = jsu.get_value_from_json(json_data=json_data,
                                                          key_path=gp_json.js_filtre_depart_decale_N_key,
                                                          alert_message="le départ decale ntx n'est pas renseigné dans l'alim")
        logger.info("L'option de départ décalé est : %s" % ntx_sef_p.depart_decale)
        ntx_sef_p.del_enc_dar_zero = jsu.get_value_from_json(json_data=json_data,
                                                             key_path=gp_json.js_supp_encours_dar0_key,
                                                             alert_message="le choix supp_encours_dar0 n'est pas renseigné dans l'alim")
        logger.info("Les encours nuls en DAR seront filtrés : %s" % ntx_sef_p.del_enc_dar_zero)
