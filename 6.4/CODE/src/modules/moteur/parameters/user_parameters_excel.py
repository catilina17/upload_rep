# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:01:46 2020

CHARGEMENT DES PARAMETRES UTILISATEURS
"""

import modules.moteur.parameters.general_parameters as gp
import modules.moteur.parameters.authorized_users as au
import getpass
import modules.moteur.utils.generic_functions as gf
import dateutil
from pathlib import Path
import datetime
from os import path
import utils.excel_utils as ex
import os
import logging
import re
import modules.moteur.parameters.user_parameters as up

logger = logging.getLogger(__name__)


def load_users_param(xl_module):
    check_user_credentials()

    get_input_path(xl_module)

    get_source_path(xl_module)

    get_map_path()

    load_projection_period_params(xl_module)

    load_ajustements_options(xl_module)

    load_output_indicators(xl_module)

    load_output_axes(xl_module)

    load_conv_ecoulements_options(xl_module)

    load_coeff_gap_gestion(xl_module)

    load_tla_refixing_params(xl_module)

    load_pn_max_month(xl_module)

    load_compil_format(xl_module)

    get_user_list_etabs()


def get_input_path(xl_module):
    up.input_path = ex.get_value_from_named_ranged(xl_module, gp.ng_input_path)


def get_source_path(xl_module):
    up.source_path = ex.get_value_from_named_ranged(xl_module, gp.ng_source_path)


def get_map_path():
    up.mapping_path = os.path.join(up.source_path, "MAPPING", "MAPPING_PASS_ALM.xlsx")


def get_stock_path(etab):
    list_files = [x[2] for x in os.walk(os.path.join(up.input_path, etab))][0]
    for fname in list_files:
        if gp.stock_tag in fname:
            up.stock_folder_path = os.path.join(up.input_path, etab)
            up.stock_file_path = os.path.join(up.input_path, etab, fname)
        if gp.stock_nmd_template_tag in fname:
            up.stock_nmd_template_file_path = os.path.join(up.input_path, etab, fname)


def get_user_list_etabs():
    up.list_all_etab = ex.get_dataframe_from_range(ex.xl_interface, gp.ng_liste_etab_global).iloc[:, 0].values.tolist()
    up.list_etab_usr = ex.get_value_from_named_ranged(ex.xl_interface, gp.ng_list_etab_user)


def load_output_path(xl_moteur):
    """DAR"""
    up.dar_usr = ex.get_value_from_named_ranged(xl_moteur, gp.ng_DAR)
    up.dar_usr = dateutil.parser.parse(str(up.dar_usr)).replace(tzinfo=None)

    """ Chargement du répertoire de sortie"""
    up.output_path_usr = ex.get_value_from_named_ranged(xl_moteur, gp.ng_output_path,
                                                        alert="Le chemin du dossier de sortie n'est pas renseigné")
    if not path.exists(up.output_path_usr):
        raise ValueError("Le chemin du dossier de sortie n'existe pas")

    """ UPDATE OUTPUT_PATH"""
    now = datetime.datetime.now()
    now = now.strftime("%Y%m%d.%H%M.%S")
    new_dir = str(up.dar_usr.year) + str(up.dar_usr.month) + str(up.dar_usr.day) + "_EXEC-" + str(now)
    up.output_path_usr = os.path.join(up.output_path_usr, "MOTEUR_DAR-" + new_dir)
    Path(up.output_path_usr).mkdir(parents=True, exist_ok=True)


def load_tla_refixing_params(wb):
    """ TX REFIXING """
    up.retraitement_tla = ex.get_value_from_named_ranged(wb, gp.ng_tla_retraitement)
    up.retraitement_tla = True if up.retraitement_tla.upper() == "OUI" else False
    up.date_refix_tla = ex.get_value_from_named_ranged(wb, gp.ng_date_refix)
    up.date_refix_tla = dateutil.parser.parse(str(up.date_refix_tla)).replace(tzinfo=None)
    up.mois_refix_tla = up.date_refix_tla.year * 12 + up.date_refix_tla.month - up.dar_usr.year * 12 - up.dar_usr.month
    try:
        up.freq_refix_tla = int(ex.get_value_from_named_ranged(wb, gp.ng_tla_retraitement))
    except:
        up.freq_refix_tla = 4
    if up.mois_refix_tla < 1 and up.retraitement_tla:
        logger.error("   La date de refixing TLA est inférieure ou égale à la DAR")
        raise ValueError("   La date de refixing TLA est inférieure ou égale à la DAR")

    return up.retraitement_tla, up.mois_refix_tla, up.freq_refix_tla


def load_output_indicators(xl_module):
    """ DETERMINATION DES INDICATEURS SORTIES"""
    up.data_indic = ex.get_dataframe_from_range(xl_module, gp.ng_ind_sortie)
    up.type_eve = ex.get_value_from_named_ranged(ex.xl_interface, gp.nc_type_eve)
    up.data_indic_eve = ex.get_dataframe_from_range(xl_module, gp.ng_ind_sortie_eve)


def load_coeff_gap_gestion(xl_module):
    # global coeff_inf_cel_usr, coeff_reg_tf_usr, coeff_reg_inf_usr, coeff_tf_tlb_usr
    # global coeff_tf_tla_usr, coeff_tf_cel_usr, coeff_inf_tla_usr, coeff_inf_cel_usr, coeff_inf_tlb_usr

    up.coeff_tf_tla_usr = ex.get_value_from_named_ranged(xl_module, gp.ng_tf_tla)
    up.coeff_tf_tlb_usr = ex.get_value_from_named_ranged(xl_module, gp.ng_tf_tlb)
    up.coeff_tf_cel_usr = ex.get_value_from_named_ranged(xl_module, gp.ng_tf_cel)
    up.coeff_inf_tla_usr = ex.get_value_from_named_ranged(xl_module, gp.ng_inf_tla)
    up.coeff_inf_tlb_usr = ex.get_value_from_named_ranged(xl_module, gp.ng_inf_tlb)
    up.coeff_inf_cel_usr = ex.get_value_from_named_ranged(xl_module, gp.ng_inf_cel)
    up.coeff_reg_tf_usr = ex.get_value_from_named_ranged(xl_module, gp.ng_reg_tf)
    up.coeff_reg_inf_usr = ex.get_value_from_named_ranged(xl_module, gp.ng_reg_inf)


def load_ajustements_options(xl_module):
    # global ajust_only, spread_liq, courbes_ajust

    """ NE SORTIR QUE LES AJUSTEMENTS"""
    up.ajust_only = ex.get_value_from_named_ranged(xl_module, gp.ng_ajust_only)
    up.ajust_only = False if up.ajust_only == "NON" else True

    """ SPREAD LIQ AJUST """
    up.spread_liq = ex.get_dataframe_from_range(xl_module, gp.ng_spread_ajust)

    """ COURBES AJUST """
    up.courbes_ajust = ex.get_dataframe_from_range(xl_module, gp.ng_courbes_ajust).set_index("DEVISE")


def load_output_axes(xl_module):
    # global cols_sortie_excl
    up.cols_sortie_excl = ex.get_dataframe_from_range(xl_module, gp.ng_col_sortie)
    up.cols_sortie_excl = up.cols_sortie_excl[up.cols_sortie_excl[gp.nc_col_sort_restituer] == "NON"][
        gp.nc_axe_nom].values.tolist()


def load_pn_max_month(xl_module):
    # global max_month_pn
    mois_pn = ex.get_dataframe_from_range(xl_module, gp.nc_nb_mois_pn)
    up.max_month_pn = dict([(str(i).lower(), int(j)) for i, j in zip(mois_pn["PN"], mois_pn["MOIS"])])


def load_projection_period_params(xl_module):
    # global nb_mois_proj_out, nb_annees_usr

    up.nb_mois_proj_out = ex.get_value_from_named_ranged(xl_module, gp.ng_nb_mois_proj)
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


def load_conv_ecoulements_options(xl_module):
    # global force_gp_liq, force_gps_nmd

    up.force_gp_liq = ex.get_value_from_named_ranged(xl_module, gp.ng_force_gp_liq)
    up.force_gp_liq = False if up.force_gp_liq == "NON" else True

    """ OPTION POUR FORCER LES GAPS DES NMDS """
    up.force_gps_nmd = ex.get_value_from_named_ranged(xl_module, gp.ng_force_gps_nmd)
    up.force_gps_nmd = False if up.force_gps_nmd == "NON" else True


def load_compil_format(xl_module):
    compil_format = str(ex.get_value_from_named_ranged(xl_module, gp.ng_compil_sep)).strip()
    up.compil_decimal = re.findall(r'"([^"]*)"', compil_format)[1]
    up.compil_sep = re.findall(r'"([^"]*)"', compil_format)[0]
    up.merge_compil = ex.get_value_from_named_ranged(xl_module, gp.ng_compil_merge)
    up.merge_compil = up.merge_compil.strip() == "OUI"


def check_user_credentials():
    if True:
        user_win = os.getlogin().upper()
        user_win2 = str(getpass.getuser()).upper()
        if not gf.begin_with_list(au.users_win, user_win) and gf.begin_with_list(au.users_win, user_win2):
            logger.error("YOU ARE NOT AN AUTHORIZED USER for THIS APP")
            raise ValueError("YOU ARE NOT AN AUTHORIZED USER for THIS APP")
