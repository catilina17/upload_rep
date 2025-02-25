# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:18:31 2020

@author: TAHIRIH
"""
import modules.moteur.utils.generic_functions as gf
import modules.moteur.utils.read_write_utils as rw_ut
import os
from collections import OrderedDict
import modules.moteur.services.indicateurs_liquidite.outflow_module as outfl
import modules.moteur.parameters.general_parameters as gp
import modules.moteur.services.indicateurs_taux.mni_module as mni
import modules.moteur.parameters.user_parameters as up
import modules.moteur.services.projection_pn.ajustements.ajustements_module as am
import modules.moteur.services.intra_groupes.intra_group_module as ig
import modules.moteur.services.indicateurs_taux.eve_module as eve
import modules.moteur.services.indicateurs_liquidite.nsfr_module as nsfr
import modules.moteur.services.indicateurs_liquidite.lcr_module as lcr
import modules.moteur.mappings.main_mappings as mp
import numpy as np
import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)

global dic_stock_main_eve
dic_stock_main_eve = {}

def compile_stock(dic_stock_sc, dic_stock_sci, name_scenario, save_path):
    """ GENERE LE COMPIL_ST.csv """
    global dic_stock_main_eve
    if dic_stock_sc["stock"].shape[0] > 0:
        ht = dic_stock_sc["stock"].shape[0]
        data_stock = dic_stock_sc["stock"]
        data_stock_other = dic_stock_sc["other_stock"]
        cols_num = ["M0" + str(k) if k < 10 else "M" + str(k) for k in range(0, up.nb_mois_proj_usr + 1)]

        """ AJUST EVE """
        if up.type_simul["EVE"] or up.type_simul["EVE_LIQ"]:
            dic_stock_sci[gp.em_eve_sti] = dic_stock_sci[gp.em_sti].copy()
            dic_stock_sci[gp.ef_eve_sti] = dic_stock_sci[gp.ef_sti].copy()
            if name_scenario == up.main_sc_eve:
                filter_cc = data_stock.index.get_level_values(gp.nc_output_contrat_cle).isin(mp.cc_sans_stress)
                if len(dic_stock_main_eve) == 0:
                    dic_stock_main_eve = {x:y[filter_cc] for x,y in dic_stock_sci.items() if re.sub(r'\d', '', x) in gp.indics_eve_tx_st}
                else:
                    dic_stock_main_eve = {x:pd.concat([dic_stock_main_eve[x], dic_stock_sci[x][filter_cc]])
                                          for x in dic_stock_sci.keys() if re.sub(r'\d', '', x) in gp.indics_eve_tx_st}
            else:
                for key, val in dic_stock_sci.items():
                    if re.sub(r'\d', '', key)  in gp.indics_eve_tx_st:
                        dic_stock_sci[key].update(dic_stock_main_eve[key])

        """ CALCUL DES INDICS DE SORTIE RESTANTS"""
        indic_sortie = up.indic_sortie["ST"] + up.indic_sortie_eve["ST"]
        compute_remaining_indics(data_stock, dic_stock_sci, dic_stock_sc, data_stock_other,
                                 indic_sortie)

        indic_sortie, ajust_vars = get_indic_sorties(dic_stock_sci, indic_sortie)

        """ FORMATAGE DATA QUAL """
        data_stock_qual = format_qual_data(data_stock, name_scenario, data_stock_other)

        """ FORMATAGE DATA NUM"""
        data_stock_num = format_num_data(dic_stock_sci, indic_sortie, ajust_vars, ht, cols_num, "ST")

        """ CONCAT DATA QUAL & NUM"""
        data_compil_st = concat_qual_and_num(data_stock_qual, data_stock_num, ajust_vars, indic_sortie)

        """ AJOUT DES LIGNES D'INTRAGROUPES ET DES INTRA-GROUPES MIROIR """
        data_compil_st = format_ig_lines(data_compil_st)

        """ CLASSEMENT DES DONNEES """
        data_compil_st = rank_data(data_compil_st, ajust_vars, indic_sortie, \
                                   order_cols=["CLE4", "CLE2", "CLE3"])

        """ PRECALCUL DES AJUSTEMENTS """
        am.prepare_adjustements(data_compil_st.copy())
        if up.type_simul["EVE"] or up.type_simul["EVE_LIQ"]:
            am.prepare_adjustements(data_compil_st.copy(), type="EVE")

        """ DERNIER FORMATAGE AVT ECRITURE """
        data_compil_st = format_before_write(data_compil_st, ajust_vars)

        """ WRITE in .csv"""
        write_st_compil(data_compil_st, save_path, gp.nom_compil_st_pn if up.merge_compil else gp.nom_compil_st)

        del data_compil_st


def get_indic_sorties(dic_pn_i, indic_sortie):
    indic_sortie = [x for x in indic_sortie if x in dic_pn_i]
    ajustements_vars = [x for x in gp.dependencies_ajust if x not in indic_sortie and x in dic_pn_i]
    return indic_sortie, ajustements_vars

def format_ig_lines(data_compil):
    data_compil = ig.add_mirror_contract(data_compil)
    data_compil_st = ig.add_counterparty_contract(data_compil)

    return data_compil_st


def compute_remaining_indics(data_stock, dic_stock_sci, dic_stock_sc, data_stock_other,
                             indic_sortie):
    """"""
    """ CALCUL de l'EVE """
    eve.calcul_eve(dic_stock_sci, data_stock, "ST")

    """ CALCUL DE LA MNI TX ET MNI MG"""
    mni.calculate_mni_st(dic_stock_sci, indic_sortie)

    """ CALCULS DES NSFR + LCR """
    outfl.calculate_outflows(dic_stock_sci, "ST", dic_data=dic_stock_sc)
    lcr.calculate_lcr(data_stock, other_data_init=data_stock_other, dic_indic=dic_stock_sci, typo="ST")
    nsfr.calculate_nsfr(data_stock, other_data_init=data_stock_other, dic_indic=dic_stock_sci, typo="ST")


def format_qual_data(data_stock, name_scenario, data_stock_other):
    data_stock[gp.nc_output_sc] = name_scenario
    data_stock[gp.nc_output_ind1] = "ST"
    data_stock[gp.nc_output_ind2] = data_stock_other["INDEX"].values

    data_stock = data_stock.reset_index(level=[gp.nc_output_palier_cle, gp.nc_output_contrat_cle])
    data_stock = pd.concat([data_stock, data_stock_other], axis=1)

    var_compil = gp.var_compil.copy()
    var_compil.remove(gp.nc_output_ind3)
    data_stock_qual = data_stock[var_compil].copy()

    del data_stock

    return data_stock_qual


def generate_ind03_col(data_st_num, dic_stock_sci, ht):
    """ CALCUL DE LA COL IND03 """
    inds = list(dic_stock_sci.keys())
    inds = np.repeat(inds, ht)
    data_st_num[gp.nc_output_ind3] = inds
    return data_st_num

def format_num_cols(df, cols_num):
    if not "M0" in df:
        df.insert(0,"M0",0)
    df_np = np.round(gf.fill_all_na(np.array(df).astype(np.float64)),2)
    df = pd.DataFrame(df_np, columns=cols_num)
    return df


def format_num_data(dic_sci, indic_sortie, ajust_vars, ht, cols_num, typo):
    """ FORMAT NUM DATA"""
    dic_stock_sci_sv = OrderedDict()
    """ ON GARDE LES INDICATEURS NECESSAIRES AU CALCUL DES AJUSTEMENTS MEME SI ABSENT DES IND DE SORTIE"""

    """ CONCATENATION DES INDICS"""
    for key, val in dic_sci.items():
        if key in indic_sortie + ajust_vars:
            dic_stock_sci_sv[key] = val
    data_num = pd.concat([df.reset_index(drop=True) for key, df in dic_stock_sci_sv.items()])

    """ MISE EN FORME DES COLS NUMS DE CHAQUE INDIC """
    data_num = format_num_cols(data_num, cols_num)

    """ CALCUL DE LA COL IND03 """
    data_num = generate_ind03_col(data_num, dic_stock_sci_sv, ht)

    list_to_clean =[dic_stock_sci_sv] if typo=="ST" else [dic_stock_sci_sv,dic_sci]
    gf.clean_dicos(list_to_clean)


    return data_num


def concat_qual_and_num(data_qual, data_num, vars_ajust, indic_sortie):
    mult = len(vars_ajust + indic_sortie)
    data_stock_qual = pd.concat([data_qual] * mult)
    data_compil_st = pd.concat([data_stock_qual.reset_index(drop=True), data_num], axis=1)
    data_compil_st.index = data_stock_qual.index
    gf.clean_df(data_qual); gf.clean_df(data_num)
    return data_compil_st


def rank_data(data_compil, ajust_vars, indic_sortie, order_cols=[]):
    ind03 = np.array(data_compil[gp.nc_output_ind3])
    list_filtres = [(ind03 == ind) for ind in indic_sortie + ajust_vars]
    choices = [i for i in range(1, len(indic_sortie + ajust_vars) + 1)]
    data_compil["ordre"] = np.select(list_filtres, choices)
    data_compil = data_compil.sort_values(order_cols + ["ordre"]).drop(
        ["ordre", "CLE2", "CLE3", "CLE4"], axis=1).reset_index(drop=True).reindex()

    return data_compil.copy().reset_index(drop=True)


def format_before_write(data_compil, ajust_vars):
    num_cols = ["M0" + str(k) if k < 10 else "M" + str(k) for k in range (0, up.nb_mois_proj_out + 1)]
    drop_cols = ["M0" + str(k) if k < 10 else "M" + str(k) for k in
                 range(up.nb_mois_proj_out + 1, up.nb_mois_proj_usr + 1)]
    data_compil = data_compil.drop(drop_cols, axis=1)

    data_compil = data_compil[~data_compil[gp.nc_output_ind3].isin(ajust_vars)]

    data_compil[gp.nc_output_nsfr2] = data_compil[gp.nc_output_nsfr2].str.replace('\n', ' ').str.replace('\r', '')

    var_compil = [x for x in gp.var_compil if x not in up.cols_sortie_excl and x in data_compil.columns]

    return data_compil[var_compil + num_cols].copy()


def write_st_compil(data_compil_st, save_path, name_file):
    """"""
    """ INDICES DES COLONNES QUAL MAIS NUMERIQUES"""
    idx_desc_col_num = {}
    if gp.nc_output_lcr_share in data_compil_st.columns.tolist():
        idx_desc_col_num[data_compil_st.columns.tolist().index(gp.nc_output_lcr_share)] = "%.2f"

    """ ECRITURE DANS LE FICHIER COMPIL """
    save_name = os.path.join(save_path, name_file)
    appendo = True if os.path.exists(save_name) else False
    mode = 'a' if appendo else 'w'
    header = False if appendo else True
    rw_ut.numpy_write(data_compil_st, save_name,up.nb_mois_proj_out, encoding='cp1252', mode=mode, header=header,
                   idx_desc_col_num=idx_desc_col_num, delimiter=up.compil_sep, decimal_symbol=up.compil_decimal)
