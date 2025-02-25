# -*- coding: utf-8 -*-
"""
Module central des paramètres utilisateur
"""
import utils.excel_utils as ex
import modules.moteur.parameters.user_parameters_excel as up_excel
import modules.moteur.parameters.user_parameters_json as up_json
import modules.moteur.parameters.authorized_users as au
import modules.moteur.mappings.dependances_indic as di
import modules.moteur.utils.generic_functions as gf
import pandas as pd
import datetime
import json
import getpass
from modules.moteur.parameters import general_parameters as gp
import os
import logging

logger = logging.getLogger(__name__)

input_path = ""
output_path_usr = ""

nb_mois_proj_out = None
nb_annees_usr = None

coeff_tf_tla_usr = None
coeff_tf_tlb_usr = None
coeff_tf_cel_usr = None
coeff_inf_tla_usr = None
coeff_inf_tlb_usr = None
coeff_inf_cel_usr = None
coeff_reg_tf_usr = None
coeff_reg_inf_usr = None
main_sc_eve = None
bassin_usr = None
suf_file_usr = None
nb_mois_proj_usr = None
max_month_pn = {}
spread_liq = None

indic_sortie = {
    "ST": [],
    "PN": [],
    "PN_CONTRIB": {},
    "PN_OUTFLOW": {},
    "ST_OUTFLOW": {}
}

indic_sortie_eve = {
    "ST": [],
    "PN": []
}
nom_sc = None
nom_etab = None
ajust_only = False
cols_sortie_excl = None

cff_add_on_tx = None
cff_add_on_liq = None
act_eve = None
force_gp_liq = None
force_gps_nmd = None
type_simul = {}

cc_sans_stress = None
cc_tci_fixe = None
cc_tci_var = None
cc_tci_pn = None
activer_tx_spread = None

list_etab_user = None
output_path_etab = None
stock_folder_path = None
stock_file_path = None
mapping_path = None
stock_nmd_template_file_path = None

retraitement_tla = None
mois_refix_tla = None
freq_refix_tla = None
date_refix_tla = None
source_path = None
list_scenarios = {}
scenarios_files = {}
scenarios_params = {}

compil_sep = None
compil_decimal = None
courbes_ajust = None
merge_compil = None

list_etab_usr = None

data_indic = None
type_eve = None
data_indic_eve = None

list_all_etab = None

dar_usr = None


def load_users_param(args):
    """
    Charger les paramètres utilisateur en fonction de l'argument fourni (JSON ou Excel).
    """
    check_user_credentials()

    if args.use_json:
        up_json.load_users_param_json()  # Charger les paramètres JSON
    else:
        up_excel.load_users_param(ex.xl_interface)  # Charger les paramètres Excel

    load_output_indicators()

    display_user_params()


def check_user_credentials():
    if True:
        user_win = os.getlogin().upper()
        user_win2 = str(getpass.getuser()).upper()
        if not gf.begin_with_list(au.users_win, user_win) and gf.begin_with_list(au.users_win, user_win2):
            logger.error("YOU ARE NOT AN AUTHORIZED USER for THIS APP")
            raise ValueError("YOU ARE NOT AN AUTHORIZED USER for THIS APP")


def load_output_path(args):
    """
    Charger le chemin de sortie en fonction du format sélectionné.
    """
    if args.use_json:
        up_json.get_json_data_param()
        up_json.load_output_path_json()  # Charger les paramètres de sortie via JSON
    else:
        up_excel.load_output_path(ex.xl_interface)  # Charger les paramètres de sortie via Excel


def get_stock_path(etab):
    global stock_folder_path, stock_file_path, stock_nmd_template_file_path, input_path
    list_files = [x[2] for x in os.walk(os.path.join(input_path, etab))][0]
    for fname in list_files:
        if gp.stock_tag in fname:
            stock_folder_path = os.path.join(input_path, etab)
            stock_file_path = os.path.join(input_path, etab, fname)
        if gp.stock_nmd_template_tag in fname:
            stock_nmd_template_file_path = os.path.join(input_path, etab, fname)


def get_list_etab(args):
    global list_all_etab, list_etab_usr, list_scenarios, scenarios_files, scenarios_params
    global input_path, main_sc_eve
    list_scenarios = {}
    scenarios_files = {}
    scenarios_params = {}

    if not os.path.isdir(input_path):
        raise ValueError("Le dossier source " + input_path + " n'existe pas")

    if list_etab_usr == "" or list_etab_usr is None:
        raise ValueError("Veuillez préciser au moins 1 entité à simuler")

    list_etab_usr = list_etab_usr.replace(" ", "").split(",")

    list_etab_usr2 = [x for x in list_etab_usr]

    for etab in list_etab_usr2:
        list_scenarios[etab] = []
        scenarios_files[etab] = {}
        scenarios_params[etab] = {}
        ino = True
        if etab not in list_all_etab:
            logger.warning("L'entité " + etab + " n'existe pas dans la liste des établissements pris en charge")
            list_etab_usr.remove(etab)
            ino = False

        if not os.path.isdir(os.path.join(input_path, etab)) and ino:
            logger.warning("Le dossier relatif à l'entité " + etab + " n'existe pas dans le dossier source indiqué")
            list_etab_usr.remove(etab)
            ino = False

        if ino:
            stock_file = [x[2] for x in os.walk(os.path.join(input_path, etab))][0]
            list_scenarios[etab] = [x[1] for x in os.walk(os.path.join(input_path, etab))][0]
            for sc in list_scenarios[etab]:
                list_files = [x for x in os.walk(os.path.join(input_path, etab, sc))][0][2]
                scenarios_files[etab][sc] = {}
                scenarios_files[etab][sc]["SC_PARAMS"] = [x for x in list_files if gp.sc_params_tag in x]

                if len(scenarios_files[etab][sc]["SC_PARAMS"])==0:
                    msg = "Le fichier de paramètres scénario est manquant"
                    logger.error(msg)
                    raise ValueError(msg)
                else:
                    file = os.path.join(input_path, etab, sc, scenarios_files[etab][sc]["SC_PARAMS"][0])
                    with open(file) as f:
                        dic_json = json.load(f)
                        scenarios_params[etab][sc] = pd.DataFrame(dic_json[gp.data_sc_tag])
                        # if not args.use_json:
                        main_sc_eve = dic_json[gp.main_sc_eve_tag]

                    scenarios_files[etab][sc]["MODELE_NMD"] = [x for x in list_files if gp.sc_modele_nmd_tag in x]
                    scenarios_files[etab][sc]["MODELE_PEL"] = [x for x in list_files if gp.sc_modele_pel_tag in x]
                    scenarios_files[etab][sc]["MODELE_ECH"] = [x for x in list_files if gp.sc_modele_ech_tag in x]
                    scenarios_files[etab][sc]["MODELE_DAV"] = [x for x in list_files if gp.sc_modele_dav_tag in x]
                    scenarios_files[etab][sc]["TAUX"] = [x for x in list_files if gp.sc_tx_tag in x]
                    scenarios_files[etab][sc]["VOLUME"] = [x for x in list_files if gp.sc_vol_tag in x]
                    scenarios_files[etab][sc]["LCR_NSFR"] = [x for x in list_files if gp.sc_lcr_nsfr_tag in x]

                    if len(scenarios_files[etab][sc]["TAUX"]) == 0:
                        msg = "Le fichier taux scénario est manquant, le scénario %s ne sera pas traité" % sc
                        logger.error(msg)
                        list_scenarios[etab].remove(sc)
                    if len(scenarios_files[etab][sc]["VOLUME"]) == 0:
                        msg = "Le fichier volume scénario est manquant"
                        logger.error(msg)
                        list_scenarios[etab].remove(sc)
                    if (len(scenarios_files[etab][sc]["MODELE_ECH"]) == 0):
                        msg = "Le fichier de modèle ech scénario  est manquant"
                        logger.error(msg)
                        list_scenarios[etab].remove(sc)
                    if (len(scenarios_files[etab][sc]["MODELE_PEL"]) == 0
                            and len(scenarios_params[etab][sc][
                                        scenarios_params[etab][sc]["TYPE PRODUIT"].str.contains("PEL")].copy()) > 0):
                        msg = "Le fichier de modèle pel scénario  est manquant"
                        logger.error(msg)
                        list_scenarios[etab].remove(sc)
                    if (len(scenarios_files[etab][sc]["MODELE_NMD"]) == 0):
                        msg = "Le fichier de modèle nmd scénario  est manquant"
                        logger.error(msg)
                        list_scenarios[etab].remove(sc)
                    if (len(scenarios_files[etab][sc]["MODELE_DAV"]) == 0
                            and len(scenarios_params[etab][sc][
                                        scenarios_params[etab][sc]["TYPE PRODUIT"].str.contains("DAV")].copy()) > 0):
                        msg = "Le fichier de modèle dav scénario  est manquant"
                        logger.error(msg)
                        list_scenarios[etab].remove(sc)

                    for model in ["MODELE_NMD", "MODELE_PEL", "MODELE_ECH", "MODELE_DAV"]:
                        if scenarios_files[etab][sc][model] != []:
                            scenarios_files[etab][sc][model] = os.path.join(input_path, etab, sc,
                                                                            scenarios_files[etab][sc][model][0])

            if not gf.begin_in_list2(stock_file, gp.stock_tag):
                logger.warning(
                    "Le fichier STOCK lié à l'entité " + etab + " n'existe pas dans le dossier source indiqué. L'entité sera exclue")
                list_etab_usr.remove(etab)


def display_user_params():
    # global date_refix_tla
    """ RAPPEL DES PARAMETRES UTILISATEURS"""
    logger.info(" PARAMETRES UTILISATEUR FONCTIONNELS:")
    logger.info("  - DATE D'ARRÊTE: %s" % str(datetime.datetime.strftime(dar_usr, "%d/%m/%Y")))
    logger.info("  - NB MOIS PROJECTION: %s" % str(nb_mois_proj_out))
    logger.info("  - NB MOIS PROJECTION IMPLICITE (OUTLFOW): %s", str(nb_mois_proj_usr))
    logger.info("  - COEFFS TLA et LEP TF ET INF: (%s,%s)" % (str(coeff_tf_tla_usr), str(coeff_tf_tlb_usr)))
    logger.info("  - COEFFS TLB TF ET INF: (%s,%s)" % (str(coeff_tf_tlb_usr), str(coeff_inf_tlb_usr)))
    logger.info("  - COEFFS CEL TF ET INF: (%s,%s)" % (str(coeff_tf_cel_usr), str(coeff_inf_cel_usr)))
    logger.info("  - COEFFS GP TF REG: (%s,%s)" % (str(coeff_reg_tf_usr), str(coeff_reg_inf_usr)))
    logger.info("  - MAX MOIS PN: %s" % max_month_pn)
    logger.info("  - SPREADS AJUSTEMENTS: %s", str(spread_liq.set_index([gp.nc_devise_spread]).to_dict('index')))

    logger.info("  PARAMETRES UTILISATEUR d'EXECUTION:")
    logger.info("  - REPERTOIRE DE SORTIE: %s" % output_path_usr)
    logger.info("  - AJUSTEMENTS SLT: %s" % ajust_only)
    logger.info("  - APPLIQUER LES CONVS D'ECOULEMENT GAP LIQ: %s" % force_gp_liq)
    logger.info("  - APPLIQUER LES CONVS D'ECOULEMENT GAPS NMD: %s" % force_gps_nmd)

    logger.info("  PARAMETRES UTILISATEUR DE SORTIE:")
    logger.info("  - IND SORTIE STOCK: %s" % str(indic_sortie["ST"] + indic_sortie_eve["ST"]))
    logger.info("  - IND SORTIE PN: %s" % str(
        indic_sortie["PN"] + indic_sortie_eve["PN"] + list(indic_sortie["PN_CONTRIB"].keys())))
    if ajust_only:
        logger.info(
            "  - IND SORTIE AJUST: %s" % str(indic_sortie["AJUST"] + + indic_sortie_eve["AJUST"] + list(
                indic_sortie["AJUST_CONTRIB"].keys())))
    logger.info("  - COLONNES DE SORTIE A EXCLURE: %s" % str(cols_sortie_excl))
    if retraitement_tla:
        logger.info("    UN RETRAITEMENT TLA SERA EFFECTUE A PARTIR DE %s" % date_refix_tla.date())


def load_output_indicators():
    global indic_sortie, type_simul, indic_sortie_eve, nb_mois_proj_out, nb_mois_proj_usr, ajust_only
    global data_indic, type_eve, data_indic_eve

    if "OUI" in data_indic[gp.nc_ind_restituer].values.tolist():
        type_simul["LIQ"] = True
    else:
        type_simul["LIQ"] = False

    if "OUI" in data_indic_eve[gp.nc_ind_restituer].values.tolist() and type_eve != "ICAAP":
        type_simul["EVE"] = True
    else:
        type_simul["EVE"] = False

    if "OUI" in data_indic_eve[gp.nc_ind_restituer].values.tolist() and type_eve == "ICAAP":
        type_simul["EVE_LIQ"] = True
    else:
        type_simul["EVE_LIQ"] = False

    data_indic = pd.concat([data_indic, data_indic_eve])
    data_indic = data_indic[data_indic[gp.nc_ind_restituer] == "OUI"].copy()

    indic_sortie = {}
    indic_sortie["ST"] = []
    indic_sortie["PN"] = []
    indic_sortie["PN_CONTRIB"] = {}
    indic_sortie["PN_OUTFLOW"] = {}
    indic_sortie["ST_OUTFLOW"] = {}

    indic_sortie_eve = {}
    indic_sortie_eve["ST"] = []
    indic_sortie_eve["PN"] = []

    add_on_project = 0

    for i in range(0, len(data_indic)):
        restit = data_indic[gp.nc_ind_restituer].iloc[i]
        step = 1 if data_indic[gp.nc_ind_pas].iloc[i] == "MENSUEL" else (
            3 if data_indic[gp.nc_ind_pas].iloc[i] == "TRIMESTRIEL" else 12)
        m_deb = int(data_indic[gp.nc_ind_deb].iloc[i]) if data_indic[gp.nc_ind_deb].iloc[i] != "-" else 0
        m_fin = int(data_indic[gp.nc_ind_fin].iloc[i]) if data_indic[gp.nc_ind_fin].iloc[i] not in ["-",
                                                                                                    "inf"] else (
            0 if data_indic[gp.nc_ind_fin].iloc[i] == "-" else "inf")
        cat = data_indic[gp.nc_ind_cat].iloc[i]
        cat = cat if cat == "PN" else "ST"
        indic = data_indic[gp.nc_ind_indic].iloc[i]
        typo = data_indic[gp.nc_ind_type].iloc[i]
        """ SEULS LES INDICATEURS A RESTITUER SONT INCLUS"""
        if restit == "OUI":
            if indic not in [gp.outf_sti, gp.outf_pni]:
                """ TRAITEMENT DES INDICS de TYPE NON OUTFLOW """
                if typo != "NORMAL":
                    for mois in range(m_deb, m_fin + 1):
                        if (mois - m_deb) // step == (mois - m_deb) / step:
                            if mois <= nb_mois_proj_out and not (mois == 0 and cat == "PN"):
                                indic_sortie[cat].append(indic + str(mois))
                    if typo == "CONTRIB":
                        indic_sortie["PN_CONTRIB"][indic] = step
                else:
                    indic_sortie[cat].append(indic)
            else:
                """ TRAITEMENT DES INDICS de TYPE OUTFLOW """
                indic_sortie[cat].append(
                    indic + " " + str(m_deb) + "M-" + ((str(m_fin) + "M") if m_fin != "inf" else "inf"))
                indic_sortie[cat + "_OUTFLOW"][
                    indic + " " + str(m_deb) + "M-" + ((str(m_fin) + "M") if m_fin != "inf" else "inf")] = (
                    m_deb, m_fin)

                """ ON CALCULE LE NB DE MOIS SUPP DE PROJ POUR POUR PVR CALCULER LES OUTFLOW A N MOIS """
                if not ajust_only:
                    add_on_project = max(add_on_project, m_deb, m_fin if m_fin != "inf" else 0)
                else:
                    if cat == "PN":
                        add_on_project = max(add_on_project, m_deb, m_fin if m_fin != "inf" else 0)

        """ AJOUT DE CERTAINS OUTFLOWS SI PRESENCE de L'INDIC NSFR"""
    for cat in ["ST", "PN"]:
        delta_rsf_indic = gp.delta_rsf_pni if cat == "PN" else gp.delta_rsf_sti
        rsf_indic = gp.rsf_pni if cat == "PN" else gp.rsf_sti
        delta_asf_indic = gp.delta_asf_pni if cat == "PN" else gp.delta_asf_sti
        asf_indic = gp.asf_pni if cat == "PN" else gp.asf_sti
        if delta_rsf_indic in indic_sortie[cat] or rsf_indic in indic_sortie[cat] \
                or delta_asf_indic in indic_sortie[cat] or asf_indic in indic_sortie[cat]:
            for indic in gp.list_indic_nsfr:
                if not indic in list(indic_sortie[cat + "_" + "OUTFLOW"].keys()):
                    m_deb = int(indic.split(" ")[1].split("-")[0].replace("M", ""))
                    m_fin = indic.split(" ")[1].split("-")[1].replace("M", "")
                    if m_fin != "inf":
                        m_fin = int(m_fin)
                    indic_sortie[cat + "_" + "OUTFLOW"][indic] = (m_deb, m_fin)
                    if not ajust_only:
                        add_on_project = max(add_on_project, m_deb, m_fin if m_fin != "inf" else 0)
                    else:
                        if cat == "PN":
                            add_on_project = max(add_on_project, m_deb, m_fin if m_fin != "inf" else 0)

        """ AJOUT DE CERTAINS OUTFLOWS SI PRESENCE de L'INDIC LCR"""
    for cat in ["ST", "PN"]:
        delta_nco_indic = gp.delta_nco_pni if cat == "PN" else gp.delta_nco_sti
        nco_indic = gp.nco_pni if cat == "PN" else gp.nco_sti
        if delta_nco_indic in indic_sortie[cat] or nco_indic in indic_sortie[cat]:
            for indic in gp.list_indic_lcr:
                if not indic in list(indic_sortie[cat + "_" + "OUTFLOW"].keys()):
                    m_deb = int(indic.split(" ")[1].split("-")[0].replace("M", ""))
                    m_fin = indic.split(" ")[1].split("-")[1].replace("M", "")
                    if m_fin != "inf":
                        m_fin = int(m_fin)
                    indic_sortie[cat + "_" + "OUTFLOW"][indic] = (m_deb, m_fin)
                    if not ajust_only:
                        add_on_project = max(add_on_project, m_deb, m_fin if m_fin != "inf" else 0)
                    else:
                        if cat == "PN":
                            add_on_project = max(add_on_project, m_deb, m_fin if m_fin != "inf" else 0)

    indic_sortie["AJUST"] = indic_sortie["PN"].copy()
    indic_sortie["AJUST_CONTRIB"] = indic_sortie["PN_CONTRIB"].copy()
    indic_sortie["AJUST_OUTFLOW"] = indic_sortie["PN_OUTFLOW"].copy()

    if ajust_only:
        indic_sortie["ST"] = []
        indic_sortie["PN"] = []
        indic_sortie["PN_CONTRIB"] = {}
        indic_sortie["PN_OUTFLOW"] = {}
        indic_sortie["ST_OUTFLOW"] = {}

    """ ON AJOUTE LE NB DE MOIS SUPP DE PROJ POUR POUR PVR CALCULER LES OUTFLOW A N MOIS """
    nb_mois_proj_usr = min(gp.max_months, add_on_project + nb_mois_proj_out)

    indic_sortie_eve["ST"] = [x for x in indic_sortie["ST"] if
                              gf.begin_with_list(data_indic_eve["INDIC"].values.tolist(), x)]
    indic_sortie_eve["PN"] = [x for x in indic_sortie["PN"] if
                              gf.begin_with_list(data_indic_eve["INDIC"].values.tolist(), x)]
    indic_sortie_eve["AJUST"] = [x for x in indic_sortie["AJUST"] if
                                 gf.begin_with_list(data_indic_eve["INDIC"].values.tolist(), x)]

    indic_sortie["ST"] = [x for x in indic_sortie["ST"] if not x in indic_sortie_eve["ST"]]
    indic_sortie["PN"] = [x for x in indic_sortie["PN"] if not x in indic_sortie_eve["PN"]]
    indic_sortie["AJUST"] = [x for x in indic_sortie["AJUST"] if not x in indic_sortie_eve["AJUST"]]

    di.generate_dependencies_indic()
