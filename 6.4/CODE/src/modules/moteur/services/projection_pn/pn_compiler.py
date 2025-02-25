import modules.moteur.services.projection_pn.ajustements.ajustements_module as am
import modules.moteur.utils.generic_functions as gf
import modules.moteur.utils.read_write_utils as rw_ut
import modules.moteur.parameters.general_parameters as gp
import modules.moteur.parameters.user_parameters as up
import modules.moteur.services.projection_stock.stock_compiler as st
import modules.moteur.services.indicateurs_taux.eve_module as eve
import logging
import modules.moteur.services.indicateurs_taux.gap_taux_agreg as gp_ag
import modules.moteur.services.indicateurs_taux.gap_taux_mtx as gp_mtx
import modules.moteur.services.indicateurs_liquidite.gap_liquidite_module as gpl
import modules.moteur.services.indicateurs_liquidite.outflow_module as outfl
import modules.moteur.services.indicateurs_taux.mni_module as mni
import modules.moteur.services.indicateurs_liquidite.nsfr_module as nsfr
import modules.moteur.services.indicateurs_liquidite.lcr_module as lcr
import modules.moteur.mappings.main_mappings as mp
import pandas as pd
import numpy as np
import os
import re

logger = logging.getLogger(__name__)

global dic_pn_main_eve

from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
np.seterr(divide='ignore', invalid='ignore')

def compile_pn(dic_pn_sc, type_pn, name_scenario, save_path, dic_stock_scr=[]):
    """ Fonction permettant d'écrire les données de PN dans le fichier CompilPN.csv"""
    global dic_pn_main_eve
    max_month_pn = min(up.max_month_pn[type_pn], up.nb_mois_proj_out)
    cols = ["M" + str(i) for i in range(1, up.nb_mois_proj_usr + 1)]
    cols_num = ["M0" + str(k) if k < 10 else "M" + str(k) for k in range(0, up.nb_mois_proj_usr + 1)]

    if dic_pn_sc != {}:
        dic_pn_sci = {}
        ht = dic_pn_sc["data"].shape[0]
        lg = up.nb_mois_proj_usr
        size = (ht, max_month_pn, lg)
        indic_sortie = up.indic_sortie["PN"] + up.indic_sortie_eve["PN"]

        """ CALCUL DES INDICATEURS"""
        calc_indicators(dic_pn_sci, dic_pn_sc, cols, type_pn, dic_stock_scr, size)

        """ SAUVEGARDE DU SC PRINCIPAL"""
        """ AJUST EVE """
        if up.type_simul["EVE"] or up.type_simul["EVE_LIQ"]:
            dic_pn_sci[gp.em_eve_pni] = dic_pn_sci[gp.em_pni].copy()
            dic_pn_sci[gp.ef_eve_pni] = dic_pn_sci[gp.ef_pni].copy()
            if name_scenario == up.main_sc_eve:
                filter_cc = dic_pn_sc["data"][gp.nc_output_contrat_cle].isin(mp.cc_sans_stress).values
                dic_pn_main_eve = {x:y.set_index(dic_pn_sc["data"][gp.nc_pn_cle_pn].values)[filter_cc] for x,y in dic_pn_sci.items() if re.sub(r'\d', '', x) in gp.indics_eve_tx_pn}
            else:
                for key, val in dic_pn_sci.items():
                    if re.sub(r'\d', '', key) in gp.indics_eve_tx_pn:
                        dic_pn_sci[key] = dic_pn_sci[key].set_index(dic_pn_sc["data"][gp.nc_pn_cle_pn].values)
                        dic_pn_sci[key].update(dic_pn_main_eve[key])
                        dic_pn_sci[key] = dic_pn_sci[key].reset_index(drop=True)

        eve.calcul_eve(dic_pn_sci, dic_pn_sc["data"], "PN", cols=cols)

        indic_sortie, ajust_vars = st.get_indic_sorties(dic_pn_sci, indic_sortie)

        """ CONCATENATION de ts les INDICATEURS"""
        data_pn_num = st.format_num_data(dic_pn_sci, indic_sortie, ajust_vars, ht, cols_num, 'PN')

        """ AJOUT DE VAR DESCRIPTIVES et concaténation des données de var descrp"""
        data_qual_pn = format_qual_data(dic_pn_sc, name_scenario, type_pn)

        """ CONCAT des cols nums et non nums """
        data_compil_pn = st.concat_qual_and_num(data_qual_pn, data_pn_num, ajust_vars, indic_sortie)

        """ AJOUT DES LIGNES D'INTRAGROUPES ET DES INTRA-GROUPES MIROIR """
        data_compil_pn = st.format_ig_lines(data_compil_pn)

        """ Classement des données"""
        data_compil_pn = st.rank_data(data_compil_pn, ajust_vars, indic_sortie,\
                                      order_cols=["CLE4", "CLE2", "CLE3", "CLE", "IND02"])

        """ FORCAGE de la  MNI à 0 pour le SCOPE 'LIQUIDITE' (NATIXIS) """
        data_compil_pn = mni_to_zero_ntx_scope(data_compil_pn, cols_num)

        """ Précalcul des ajustements"""
        am.prepare_adjustements(data_compil_pn.copy())
        if up.type_simul["EVE"] or up.type_simul["EVE_LIQ"]:
            am.prepare_adjustements(data_compil_pn.copy(), type="EVE")

        """ Supression des colonnes superflues """
        data_compil_pn = st.format_before_write(data_compil_pn, ajust_vars)

        """ Ecriture dans le fichier Compil """
        write_pn_compil(data_compil_pn, indic_sortie, save_path,
                        gp.nom_compil_st_pn if up.merge_compil else gp.nom_compil_pn)



def calcul_encours(dic_pn_sci, data_dic, cols):
    """ Fonction permettant de calculer les encours et les contributions périodiques aux encours"""
    index = data_dic["data"].index
    dic_pn_sci[gp.ef_pni] = gf.sum_each2(data_dic["mtx_ef"], cols)
    dic_pn_sci[gp.em_pni] = gf.sum_each2(data_dic["mtx_em"], cols)

    for p in range(1, up.nb_mois_proj_out + 1):
        if gp.ef_pni + str(p) in up.indic_sortie["PN"]:
            dic_pn_sci[gp.ef_pni + str(p)] = gf.sum_each2(data_dic["mtx_ef"], cols, proj=True, per=p,
                                                          interv=up.indic_sortie["PN_CONTRIB"][gp.ef_pni])

        if gp.em_pni + str(p) in up.indic_sortie["PN"]:
            dic_pn_sci[gp.em_pni + str(p)] = gf.sum_each2(data_dic["mtx_em"], cols, proj=True, per=p,
                                                          interv=up.indic_sortie["PN_CONTRIB"][gp.em_pni])


def calcul_autres_indic(dic_pn_sci, data_dic, ht, type_pn):
    """ Fonction permettant de calculer le taux client et le volumne PN"""
    index_calc = data_dic["data"][gp.nc_output_index_calc_cle]
    filtre_fx = np.array(index_calc.astype(str).str.contains(gp.FIX_ind)).reshape(ht, 1)
    """ TX CLI et VOL PN """
    if gp.tx_cli_pni in up.indic_sortie["PN"]:
        df = np.where(filtre_fx, data_dic["tx_cli"], 0)
        dic_pn_sci[gp.tx_cli_pni] = pd.DataFrame(data=df, index=data_dic["data"].index)
        dic_pn_sci[gp.tx_cli_pni].columns = ["M" + str(j) for j in range(1, up.nb_mois_proj_usr + 1)]

    if gp.vol_pn_pni in up.indic_sortie["PN"]:
        df = np.where(filtre_fx, data_dic["new_pn"].copy(), 0)
        dic_pn_sci[gp.vol_pn_pni] = pd.DataFrame(data=df, index=data_dic["data"].index)
        dic_pn_sci[gp.vol_pn_pni].columns = ["M" + str(j) for j in range(1, up.nb_mois_proj_usr + 1)]


def choose_indic_pn_conv(dic_ind1, dic_ind2, is_there_conv):
    for key, val in dic_ind2.items():
        new_val = np.where(is_there_conv, dic_ind2[key], dic_ind1[key])
        dic_ind1[key] = pd.DataFrame(new_val, index=dic_ind1[key].index, columns=dic_ind1[key].columns)


def decomp_mat_gp_fix(data_dic, size, simul):
    for p, val in data_dic["mtx_gp_fix_f_" + simul].items():
        try:
            data_dic["mtx_gp_fix_f_" + simul][p] = gf.decompress(data_dic["mtx_gp_fix_f_" + simul][p],size)
        except:
            pass
    for p, val in data_dic["mtx_gp_fix_m_" + simul].items():
        try:
            data_dic["mtx_gp_fix_m_" + simul][p] = gf.decompress(data_dic["mtx_gp_fix_m_" + simul][p],size)
        except:
            pass


def calc_gaps_and_mni(dic_pn_sci, data_dic, cols, type_pn, dic_stock_scr, size, simul):
    decomp_mat_gp_fix(data_dic, size, simul)

    if not simul == "EVE":
        mni.calculate_mni_pn(dic_pn_sci, data_dic, simul, type_pn, cols)
        gp_ag.calcul_gap_tx_agreg(data_dic["data"], dic_stock_scr, dic_pn_sci, "PN",simul, data_dic=data_dic, type_pn=type_pn)
    else:
        gp_mtx.calcul_mni_gap_tx_pn_mtx(dic_pn_sci, data_dic, cols, type_pn, simul, size)
        if (mp.force_gps_nmd_eve) and "nmd" in type_pn:
            dic_pn_sci2 = {}
            is_there_conv = gp_ag.calcul_gap_tx_agreg(data_dic["data"], dic_stock_scr, dic_pn_sci2, "PN", simul,
                                                      data_dic=data_dic, type_pn=type_pn)
            mni.calculate_mni_gp_rg_nmd(dic_pn_sci2, data_dic)
            choose_indic_pn_conv(dic_pn_sci, dic_pn_sci2, is_there_conv)


def calc_nsfr_lcr_indics(dic_pn_sci, data_dic, type_pn, cols):
    outfl.calculate_outflows(dic_pn_sci, "PN", dic_data=data_dic, mat_ef=data_dic["mtx_ef"].copy(), cols=cols)
    other_data = data_dic["data"]
    nsfr.calculate_nsfr(data_dic["data"], other_data_init=other_data, dic_indic=dic_pn_sci, typo="PN")
    lcr.calculate_lcr(data_dic["data"], other_data_init=other_data, dic_indic=dic_pn_sci, typo="PN")


def format_num_cols(dic_pn_sci, ind, new_cols_num):
    df = dic_pn_sci[ind]
    df = df.astype(np.float64)
    if "M0" not in df.columns:
        df["M0"] = 0
    cols_num = ["M" + str(k) for k in range(0, up.nb_mois_proj_usr + 1)]
    df = df[cols_num]
    df.columns = new_cols_num
    if not ind == gp.tx_cli_pni:
        df = df.round(0)
    df = df.fillna(0).replace((np.inf, -np.inf), (0, 0))
    df[gp.nc_output_ind3] = ind
    df = df[[gp.nc_output_ind3] + new_cols_num]
    dic_pn_sci[ind] = df


def concat_all_indicators(dic_pn_sci, data_dic, indic_sortie, ajust_vars, new_cols_num):
    """ REINDEXATION """
    for key, data in dic_pn_sci.items():
        dic_pn_sci[key].index = data_dic["data"].index

    for ind in indic_sortie + ajust_vars:
        format_num_cols(dic_pn_sci, ind, new_cols_num)

    dic_pn_sci = {key: dic_pn_sci[key] for key in indic_sortie + ajust_vars}
    data_pn_num = pd.concat([df for key, df in dic_pn_sci.items()])

    return data_pn_num


def format_qual_data(data_dic, name_scenario, type_pn):
    data_dic["data"][gp.nc_output_sc] = name_scenario
    data_dic["data"][gp.nc_output_ind1] = "PN"
    data_pn = data_dic["data"].copy()
    data_pn[gp.nc_output_ind2] = data_pn[gp.nc_pn_cle_pn]

    """ CLassement des cols de sortie """
    var_compil = gp.var_compil.copy()
    var_compil.remove(gp.nc_output_ind3)

    gf.clean_dicos([data_dic])

    return data_pn[var_compil]


def calc_indicators(dic_pn_sci, data_dic, cols, type_pn, dic_stock_scr, size):
    """ CALCUL des ENCOURS & CONTRIB ANNUELLES ENCOURS """

    calcul_encours(dic_pn_sci, data_dic, cols)

    for simul in list(up.type_simul.keys()):
        if up.type_simul[simul] == True:
            """ CALCUL du GAP LIQUIDITE """
            gpl.calcul_gap_liq_pn(dic_pn_sci, data_dic, simul, cols)
            gpl.appliquer_conv_ecoulement_gp_liq(dic_pn_sci, "PN", data_dic["data"], data_dic["data"], simul)

            """ CALCUL des MNI, & CONTRIB ANNUELLES MNI, GAP TX FX, GAP TX INF, GAP TX REG """
            calc_gaps_and_mni(dic_pn_sci, data_dic, cols, type_pn, dic_stock_scr, size, simul)

    """ CALCUL LCR & NSFR INDICS """
    calc_nsfr_lcr_indics(dic_pn_sci, data_dic, type_pn, cols)

    """ CALCUL des TX CLIENT, VOLUME PN """
    calcul_autres_indic(dic_pn_sci, data_dic, size[0], type_pn)

def mni_to_zero_ntx_scope(data_compil_pn, cols_num):
    """ FORCAGE de la  MNI à 0 pour le SCOPE 'LIQUIDITE' (NATIXIS) """
    filter_scope = (data_compil_pn[gp.nc_output_scope_cle] == "LIQUIDITE") & (
            data_compil_pn[gp.nc_output_ind3].astype(str)[:2] == "MN")
    data_compil_pn.loc[filter_scope, cols_num] = 0
    return data_compil_pn.copy()


def write_pn_compil(data_compil_pn, indic_sortie, save_path, name_file):
    """ Préparation du formatage spécial du taux client => arrondi à 7 """
    if gp.tx_cli_pni in indic_sortie:
        i_tx_cli = (data_compil_pn[data_compil_pn[gp.nc_output_ind3] == gp.tx_cli_pni]).index[0] + 1
        f_tx_cli = "%.7f"
    else:
        i_tx_cli = ""
        f_tx_cli = ""

    """ INDICES DES COLONNES DESCRIPTIVES NUMERIQUES"""
    idx_desc_col_num = {}
    if gp.nc_output_lcr_share in data_compil_pn.columns.tolist():
        idx_desc_col_num[data_compil_pn.columns.tolist().index(gp.nc_output_lcr_share)] = "%.2f"

    # mode = 'append' if appendo else 'fail'
    save_name = os.path.join(save_path, name_file)
    appendo = True if os.path.exists(save_name) else False
    mode = 'a' if appendo else 'w'
    header = False if appendo else True
    rw_ut.numpy_write(data_compil_pn, save_path + "\\" + name_file,up.nb_mois_proj_out,
                      encoding='cp1252', mode=mode, \
                   header=header, i_tx_cli=i_tx_cli, format_output_tx_cli=f_tx_cli, nb_indic=len(indic_sortie),
                   idx_desc_col_num=idx_desc_col_num, delimiter = up.compil_sep, decimal_symbol=up.compil_decimal)


