import datetime
import dateutil
import pandas as pd
import mappings.general_mappings as mp
import modules.alim.parameters.general_parameters as gp
from params import sources_params as sp
from params import version_params as vp
import logging
from pathlib import Path
import os
import utils.excel_utils as ex
import modules.alim.parameters.user_parameters_json as up_json
import modules.alim.parameters.user_parameters_excel as up_excel
import utils.general_utils as gu
import modules.alim.parameters.RZO_params as rzo_p
import modules.alim.parameters.NTX_SEF_params as ntx_sef_p

logger = logging.getLogger(__name__)

path_scenario_template_file = None
mode = None
dar = None
name_run = None
etabs = None
sources_folder = None
mapp_file = None
map_lcr_nsfr = None
map_lcr_nsfr_g = None
output_folder = None
log_path = None
name_stock_output = None
scenario_output_path = None
prefixes_main_files = None
main_files_name = None
nmd_st_files_name = None
current_etab = None
lcr_nsfr_file = None
database_output_path = None
bcpe_files_name = None
missing_map_output_file = None
update_param_lcr_ncr = None
path_map_mis_template_file = None
accueil_ws = None
category = None
lcr_nsfr_files_name = None
stock_output_path = None
path_sc_tx_template_file = None
path_sc_vol_template_file = None
path_sc_prof_template_file = None
path_sc_lcr_nsfr_template_file = None
path_stock_template_file = None
sc_ref_nmd = None
sc_taux_output_path = None
sc_volume_output_path = None
sc_profil_output_path = None
sc_lcr_nsfr_output_path = None
detail_ig = None
type_simul = None
mois_inter_trim = None
chemin_compils = None
nmd_template_output_path = None
liq_file_path = None
rate_file_path = None
modele_nmd_file_path = None
perim_ntx = None
depart_decale = None
del_enc_dar_zero = None
modele_pel_file_path = None




def get_user_main_params(args):
    if args.use_json:
        up_json.get_json_data_param()
        up_json.get_user_main_params_json()
    else :
        up_excel.get_user_main_params()


def create_alim_dir():
    global output_folder, dar

    now = datetime.datetime.now()
    now = now.strftime("%Y%m%d.%H%M.%S")

    output_folder = os.path.join(output_folder, "ALIM_DAR-" + str(
        dar.year) + '{:02d}'.format(dar.month) + str(dar.day) + "_EXEC-" + str(now))

    Path(output_folder).mkdir(parents=True, exist_ok=True)


def get_category():
    global category
    if current_etab in gp.NTX_FORMAT:
        category = "NTX_SEF"
    elif current_etab in ["ONEY"]:
        category = "ONEY_SOCFIM"
    else:
        category = "RZO"


def generate_etab_parameters(args, etab):
    global current_etab, category, output_folder, perim_ntx_up, depart_decale_up, del_enc_dar_zero_up
    current_etab = etab.upper()
    get_category()
    if args.use_json :
        up_json.generate_etab_parameters_json(etab)
    else :
        up_excel.generate_etab_parameters_excel(etab)

    generate_output_files()
    get_input_files_name(etab)


def generate_output_files():
    global output_folder, current_etab, dar, name_run, log_path, stock_output_path, sc_taux_output_path
    global database_output_path, missing_map_output_file, sc_volume_output_path, sc_lcr_nsfr_output_path
    global nmd_template_output_path
    mp.missing_mapping = {}
    output_folder_etab = os.path.join(output_folder, current_etab)

    if not os.path.exists(output_folder_etab):
        os.makedirs(output_folder_etab)

    name_out = current_etab + "_" + str(dar.year) + str(dar.month) \
               + str(dar.day) + ("_" + name_run if name_run != "" else "")
    log_path = output_folder_etab + "\\\\" + "LOG_" + name_out + ".txt"
    stock_output_path = output_folder_etab + "\\STOCK_AG_" + name_out + ".xlsb"
    sc_taux_output_path = output_folder_etab + "\\SC_TAUX_" + name_out + ".xlsb"
    sc_volume_output_path = output_folder_etab + "\\SC_VOLUME_" + name_out + ".xlsb"
    sc_lcr_nsfr_output_path = output_folder_etab + "\\SC_LCR_NSFR_" + name_out + ".xlsb"
    missing_map_output_file = output_folder_etab + "\\MAPPINGS_MANQUANTS_" + name_out + ".xlsb"
    nmd_template_output_path = output_folder_etab + "\\STOCK_NMD_TEMPLATE_" + name_out + ".csv"

    # stock_output_path = output_folder_etab + "\\STOCK_" + name_out + ".csv"
    # database_output_path = output_folder_etab + "\\DATABASE_" + name_out + ".db"
    # db.create_database()


def get_input_files_name_lcr_nsfr():
    global update_param_lcr_ncr, lcr_nsfr_files_name
    nomenclature_lcr_nsfr = mp.nomenclature_lcr_nsfr
    update_param_lcr_ncr = True
    lcr_nsfr_files_name = {}
    for i in range(0, len(nomenclature_lcr_nsfr)):
        file_path = nomenclature_lcr_nsfr.iloc[i]["CHEMIN"]
        file_path = os.path.join(sources_folder, file_path)
        if not os.path.exists(file_path):
            logger.warning(
                "Le fichier '%s' n'existe pas, les LCR et NSFR DAR et les paramètres d'OUTFLOW ne seront pas mis à jour" % file_path)
            update_param_lcr_ncr = False
            return
        else:
            lcr_nsfr_files_name[nomenclature_lcr_nsfr.iloc[i]["NOM"]] = file_path
            gu.check_version_templates(file_path.split("\\")[-1], version=vp.version_input_lcr_nsfr, \
                                       path=file_path, open=True)


def get_input_files_name(etab):
    global sources_folder, prefixes_main_files, main_files_name, nmd_st_files_name, lcr_nsfr_file, bcpe_files_name
    global map_lcr_nsfr, map_lcr_nsfr_g
    prefixes_main_files = []
    main_files_name = []
    nmd_st_files_name = {}
    bcpe_files_name = []
    is_ray = False

    if etab in gp.NON_RZO_ETABS:
        nomenclature = mp.nomenclature_stock_ag[mp.nomenclature_stock_ag["ENTITE"] == current_etab].copy()
        nomenclature["TYPE CONTRAT"] = ""
    else:
        nomenclature = mp.nomenclature_stock_ag[mp.nomenclature_stock_ag["ENTITE"] == "RZO"].copy()
        nomenclature["CHEMIN"] = nomenclature["CHEMIN"].str.replace("RZO", current_etab)

        nomenclature_pn = mp.nomenclature_pn[mp.nomenclature_pn["ENTITE"] == "RZO"].copy()
        nomenclature_pn["CHEMIN"] = nomenclature_pn["CHEMIN"].str.replace("RZO", current_etab)

    nomenclature = nomenclature[nomenclature["MODULE"] == "ALIM"].copy()

    nomenclature["CHEMIN"] = [os.path.join(sources_folder, x) for x in nomenclature["CHEMIN"]]

    if not etab in gp.NON_RZO_ETABS:
        nomenclature_pn["CHEMIN"] = [os.path.join(sources_folder, x) for x in nomenclature_pn["CHEMIN"]]
        nomenclature_pn["TYPE"] = "D"
        nomenclature_pn.rename(columns={"TYPE FICHIER": "NOM INDICATEUR"}, inplace=True)
        nomenclature = pd.concat([nomenclature, nomenclature_pn])

        nomenclature["TYPE CONTRAT"] = nomenclature["TYPE CONTRAT"].fillna("")

    if not etab in gp.NTX_FORMAT + ["ONEY"]:
        nomenclature_nmd = mp.nomenclature_stock_nmd.copy()
        nomenclature_nmd = nomenclature_nmd[nomenclature_nmd["TYPE CONTRAT"] == "ST-NMD"].copy()
        nomenclature_nmd = nomenclature_nmd[nomenclature_nmd["ENTITE"] == "RZO"].copy()
        nomenclature_nmd["CHEMIN"] = nomenclature_nmd["CHEMIN"].str.replace("RZO", current_etab)
        nomenclature_nmd["CHEMIN"] = [os.path.join(sources_folder, x) for x in nomenclature_nmd["CHEMIN"]]
        nomenclature_nmd["TYPE"] = "S"
        nomenclature_nmd.rename(columns={"TYPE FICHIER": "NOM INDICATEUR"}, inplace=True)

        nomenclature = pd.concat([nomenclature, nomenclature_nmd])
        nomenclature["TYPE CONTRAT"] = nomenclature["TYPE CONTRAT"].fillna("")

    nomenclature = nomenclature.sort_values(["ORDRE LECTURE"])
    nomenclature = parse_nomenclature_table(nomenclature)

    for i in range(0, nomenclature.shape[0]):
        if nomenclature["TYPE"].iloc[i] == "F":
            bcpe_files_name[nomenclature["NOM INDICATEUR"].iloc[i]] = nomenclature["CHEMIN"].iloc[i]
        elif nomenclature["NOM INDICATEUR"].iloc[i] == "RAY":
            is_ray = True
            lcr_nsfr_file = nomenclature["CHEMIN"].iloc[i]
        elif "PN-" in nomenclature["TYPE CONTRAT"].iloc[i]:
            key = nomenclature["TYPE CONTRAT"].iloc[i] + "-" + nomenclature["NOM INDICATEUR"].iloc[i]
            rzo_p.pn_rzo_files_name[key] = nomenclature["CHEMIN"].iloc[i]
        elif "ST-NMD" in nomenclature["TYPE CONTRAT"].iloc[i]:
            key = nomenclature["TYPE CONTRAT"].iloc[i]
            nmd_st_files_name[key] = nomenclature["CHEMIN"].iloc[i]
        else:
            prefixes_main_files.append(nomenclature["NOM INDICATEUR"].iloc[i])
            main_files_name.append(nomenclature["CHEMIN"].iloc[i])

    if is_ray and map_lcr_nsfr_g:
        map_lcr_nsfr = True
    else:
        map_lcr_nsfr = False


def parse_nomenclature_table(nomenclature):
    global map_lcr_nsfr_g
    global current_etab
    nomenclature2 = nomenclature.copy()

    already_ignored_dm = False
    already_ignored_bpce = {}
    already_ignored_fx = False
    already_ignored_dyn_rco = False
    if current_etab in gp.NON_RZO_ETABS:
        rzo_p.do_pn = False
    else:
        rzo_p.do_pn = True

    for i in range(0, nomenclature.shape[0]):
        if not os.path.exists(nomenclature["CHEMIN"].iloc[i]):
            if nomenclature["TYPE"].iloc[i] == "D":
                if not already_ignored_dm:
                    nomenclature2 = nomenclature2[nomenclature2["TYPE"] != "D"]
                    nomenclature2 = nomenclature2[nomenclature2["TYPE"] != "DYN_RCO"]
                    logger.warning("le fichier suivant est manquant: " + nomenclature["CHEMIN"].iloc[i])
                    logger.warning("L'alimentation sera faite en statique")
                    rzo_p.do_pn = False
                    already_ignored_dm = True

            elif "FX-" in nomenclature["NOM INDICATEUR"].iloc[i]:
                if not already_ignored_fx:
                    nomenclature2 = nomenclature2[nomenclature2["TYPE"] != "F"]
                    logger.warning("Le fichier suivant est manquant : " + str(nomenclature["CHEMIN"].iloc[i]))
                    logger.warning("Les taux de change ne seront pas inclus")
                    already_ignored_fx = True

            elif "DEM" in nomenclature["NOM INDICATEUR"].iloc[i] or "DMN" in nomenclature["NOM INDICATEUR"].iloc[i]:
                if not already_ignored_dyn_rco:
                    nomenclature2 = nomenclature2[nomenclature2["TYPE"] != "DYN_RCO"]
                    logger.warning("Le fichier suivant est manquant : " + str(nomenclature["CHEMIN"].iloc[i]))
                    logger.warning("Le dynamique de RCO ne sera pas inclus")
                    already_ignored_dyn_rco = True

            elif nomenclature["NOM INDICATEUR"].iloc[i] == "RAY":
                if map_lcr_nsfr_g:
                    logger.error("Le fichier suivant est manquant : " + str(nomenclature["CHEMIN"].iloc[i]))
                    logger.error("Vous avez activé le mappings RAY et l'établissement sélectionné est mappable par ray")
                    raise ValueError("Le fichier suivant est manquant : " + str(nomenclature["CHEMIN"].iloc[i]))
                else:
                    nomenclature2 = nomenclature2[nomenclature2["NOM INDICATEUR"] != "RAY"]
            else:
                logger.error("Le fichier suivant est manquant : " + str(nomenclature["CHEMIN"].iloc[i]))
                raise ValueError("Le fichier suivant est manquant : " + str(nomenclature["CHEMIN"].iloc[i]))

    return nomenclature2.copy()
