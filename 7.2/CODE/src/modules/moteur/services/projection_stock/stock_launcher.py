# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:18:31 2020

@author: TAHIRIH
"""
import modules.moteur.parameters.general_parameters as gp
import pandas as pd
import modules.moteur.utils.generic_functions as gf
from modules.moteur.services.projection_stock.stock_projection import Stock_Projecter

import logging
logger = logging.getLogger(__name__)

def project_loop_stock(st_comp, tx, cal, up, mp, dic_stock_ref, dic_stock_ind_ref, dic_updated, name_scenario, new_dir):
    dic_stock_ind = {};
    dic_stock_ratio = {}
    sc = st_comp
    """ Boucle sur les différents blocs de stock """
    for i in range(0, len(dic_stock_ref)):
        sp = Stock_Projecter(up, mp, cal, tx)
        dic_data_sim, dic_ind_sim, dic_updated_sc, dic_ratio_sim\
            = sp.create_sc_dictionaries(dic_stock_ref[i], dic_stock_ind_ref[i], dic_updated[i])
        sp.project_stock(dic_stock_ref[i], dic_data_sim, dic_ind_sim, dic_ratio_sim, dic_updated_sc)

        sc.compile_stock(dic_data_sim, dic_ind_sim, name_scenario, new_dir)

        concat_results(i, dic_stock_ind, dic_stock_ratio, dic_ind_sim, dic_ratio_sim)

        gf.clean_vars([dic_ind_sim, dic_ratio_sim])

    return dic_stock_ind, dic_stock_ratio


def concat_results(i, dic_stock_sci, dic_stock_scr, dic_st_sci_tp, dic_st_scr_tp):
    """ Concaténation des résultats"""
    if i != 0:
        dic_stock_sci[gp.em_sti] = pd.concat([dic_stock_sci[gp.em_sti], dic_st_sci_tp[gp.em_sti]], axis=0)
        for ind in dic_st_scr_tp:
            dic_stock_scr[ind] = pd.concat(
                [dic_stock_scr[ind], dic_st_scr_tp[ind]], axis=0)
    else:
        dic_stock_sci[gp.em_sti] = dic_st_sci_tp[gp.em_sti]
        for ind in dic_st_scr_tp:
            dic_stock_scr[ind] = dic_st_scr_tp[ind]