# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:18:31 2020

@author: TAHIRIH
"""
import modules.moteur.utils.generic_functions as gf
import modules.moteur.utils.read_write_utils as rw_ut
import os
from collections import OrderedDict
import modules.moteur.parameters.general_parameters as gp
from modules.moteur.services.indicateurs_taux.mni_module import MNI_Calculator
import modules.moteur.services.intra_groupes.intra_group_module as ig
from modules.moteur.services.indicateurs_taux.eve_module import EVE_Calculator
import numpy as np
import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)


class Stock_Compiler():
    def __init__(self, cls_usr, cls_mp, cls_ig):
        self.up = cls_usr
        self.mp = cls_mp
        self.dic_stock_main_eve = {}
        self.ig = cls_ig

    def set_ajustements(self, am):
        self.ajust_cls = am

    def set_tx_cls(self, cls_tx):
        self.tx = cls_tx

    def compile_stock(self, dic_stock_sc, dic_stock_sci, name_scenario, save_path):
        """ GENERE LE COMPIL_ST.csv """
        if dic_stock_sc["stock"].shape[0] > 0:
            ht = dic_stock_sc["stock"].shape[0]
            data_stock = dic_stock_sc["stock"]
            data_stock_other = dic_stock_sc["other_stock"]
            cols_num = ["M0" + str(k) if k < 10 else "M" + str(k) for k in range(0, self.up.nb_mois_proj_usr + 1)]

            """ AJUST EVE """
            if self.up.type_simul["EVE"] or self.up.type_simul["EVE_LIQ"]:
                dic_stock_sci[gp.em_eve_sti] = dic_stock_sci[gp.em_sti].copy()
                dic_stock_sci[gp.ef_eve_sti] = dic_stock_sci[gp.ef_sti].copy()
                if name_scenario == self.up.main_sc_eve:
                    filter_cc = data_stock.index.get_level_values(gp.nc_output_contrat_cle).isin(self.mp.mapping_eve["cc_sans_stress"])
                    if len(self.dic_stock_main_eve) == 0:
                        self.dic_stock_main_eve = {x: y[filter_cc] for x, y in dic_stock_sci.items() if
                                              re.sub(r'\d', '', x) in gp.indics_eve_tx_st}
                    else:
                        self.dic_stock_main_eve = {x: pd.concat([self.dic_stock_main_eve[x], dic_stock_sci[x][filter_cc]])
                                              for x in dic_stock_sci.keys() if
                                              re.sub(r'\d', '', x) in gp.indics_eve_tx_st}
                else:
                    for key, val in dic_stock_sci.items():
                        if re.sub(r'\d', '', key) in gp.indics_eve_tx_st:
                            dic_stock_sci[key].update(self.dic_stock_main_eve[key])

            """ CALCUL DES INDICS DE SORTIE RESTANTS"""
            indic_sortie = self.up.indic_sortie["ST"] + self.up.indic_sortie_eve["ST"]
            self.compute_remaining_indics(data_stock, dic_stock_sci, dic_stock_sc, data_stock_other,
                                     indic_sortie)

            indic_sortie, ajust_vars = self.get_indic_sorties(dic_stock_sci, indic_sortie)

            """ FORMATAGE DATA QUAL """
            data_stock_qual = self.format_qual_data(data_stock, name_scenario, data_stock_other)

            """ FORMATAGE DATA NUM"""
            data_stock_num = self.format_num_data(dic_stock_sci, indic_sortie, ajust_vars, ht, cols_num, "ST")

            """ CONCAT DATA QUAL & NUM"""
            data_compil_st = self.concat_qual_and_num(data_stock_qual, data_stock_num, ajust_vars, indic_sortie)

            """ AJOUT DES LIGNES D'INTRAGROUPES ET DES INTRA-GROUPES MIROIR """
            data_compil_st = self.format_ig_lines(self.ig, data_compil_st)

            """ CLASSEMENT DES DONNEES """
            data_compil_st = self.rank_data(data_compil_st, ajust_vars, indic_sortie, \
                                       order_cols=["CLE4", "CLE2", "CLE3"])

            """ PRECALCUL DES AJUSTEMENTS """
            self.ajust_cls.prepare_adjustements(data_compil_st.copy())
            if self.up.type_simul["EVE"] or self.up.type_simul["EVE_LIQ"]:
                self.ajust_cls.prepare_adjustements(data_compil_st.copy(), type="EVE")

            """ DERNIER FORMATAGE AVT ECRITURE """
            data_compil_st = self.format_before_write(data_compil_st, ajust_vars,
                                                      self.up.nb_mois_proj_out, self.up.nb_mois_proj_usr,
                                                      self.up.cols_sortie_excl)

            """ WRITE in .csv"""
            self.write_st_compil(data_compil_st, save_path, gp.nom_compil_st_pn if self.up.merge_compil else gp.nom_compil_st)

            del data_compil_st

    @staticmethod
    def get_indic_sorties(dic_pn_i, indic_sortie):
        indic_sortie = [x for x in indic_sortie if x in dic_pn_i]
        ajustements_vars = [x for x in gp.dependencies_ajust if x not in indic_sortie and x in dic_pn_i]
        return indic_sortie, ajustements_vars

    @staticmethod
    def format_ig_lines(ig, data_compil):
        data_compil = ig.add_mirror_contract(data_compil)
        data_compil_st = ig.add_counterparty_contract(data_compil)
        return data_compil_st

    def compute_remaining_indics(self, data_stock, dic_stock_sci, dic_stock_sc, data_stock_other, indic_sortie):
        """"""
        """ CALCUL de l'EVE """
        eve = EVE_Calculator(self.up, self.mp, self.tx)
        eve.calcul_eve(dic_stock_sci, data_stock, "ST")

        """ CALCUL DE LA MNI TX ET MNI MG"""
        mni = MNI_Calculator(self.up, self.mp, self.tx)
        mni.calculate_mni_st(dic_stock_sci, indic_sortie)

        """ CALCULS DES NSFR + LCR """
        # outfl.calculate_outflows(dic_stock_sci, "ST", dic_data=dic_stock_sc)
        # lcr.calculate_lcr(data_stock, other_data_init=data_stock_other, dic_indic=dic_stock_sci, typo="ST")
        # nsfr.calculate_nsfr(data_stock, other_data_init=data_stock_other, dic_indic=dic_stock_sci, typo="ST")

    def format_qual_data(self, data_stock, name_scenario, data_stock_other):
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

    @staticmethod
    def generate_ind03_col(data_st_num, dic_stock_sci, ht):
        """ CALCUL DE LA COL IND03 """
        inds = list(dic_stock_sci.keys())
        inds = np.repeat(inds, ht)
        data_st_num[gp.nc_output_ind3] = inds
        return data_st_num

    @staticmethod
    def format_num_cols(df, cols_num):
        if not "M0" in df:
            df.insert(0, "M0", 0)
        df_np = np.round(gf.fill_all_na(np.array(df).astype(np.float64)), 2)
        df = pd.DataFrame(df_np, columns=cols_num)
        return df

    @staticmethod
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
        data_num = Stock_Compiler.format_num_cols(data_num, cols_num)

        """ CALCUL DE LA COL IND03 """
        data_num = Stock_Compiler.generate_ind03_col(data_num, dic_stock_sci_sv, ht)

        list_to_clean = [dic_stock_sci_sv] if typo == "ST" else [dic_stock_sci_sv, dic_sci]
        gf.clean_dicos(list_to_clean)

        return data_num

    @staticmethod
    def concat_qual_and_num(data_qual, data_num, vars_ajust, indic_sortie):
        mult = len(vars_ajust + indic_sortie)
        data_stock_qual = pd.concat([data_qual] * mult)
        data_compil_st = pd.concat([data_stock_qual.reset_index(drop=True), data_num], axis=1)
        data_compil_st.index = data_stock_qual.index
        gf.clean_df(data_qual)
        gf.clean_df(data_num)
        return data_compil_st

    @staticmethod
    def rank_data(data_compil, ajust_vars, indic_sortie, order_cols=[]):
        ind03 = np.array(data_compil[gp.nc_output_ind3])
        list_filtres = [(ind03 == ind) for ind in indic_sortie + ajust_vars]
        choices = [i for i in range(1, len(indic_sortie + ajust_vars) + 1)]
        data_compil["ordre"] = np.select(list_filtres, choices)
        data_compil = data_compil.sort_values(order_cols + ["ordre"]).drop(
            ["ordre", "CLE2", "CLE3", "CLE4"], axis=1).reset_index(drop=True).reindex()

        return data_compil.copy().reset_index(drop=True)

    @staticmethod
    def format_before_write(data_compil, ajust_vars, mois_out, mois_usr, cols_sortie_excl):
        num_cols = ["M0" + str(k) if k < 10 else "M" + str(k) for k in range(0, mois_out + 1)]
        drop_cols = ["M0" + str(k) if k < 10 else "M" + str(k) for k in
                     range(mois_out + 1, mois_usr + 1)]
        data_compil = data_compil.drop(drop_cols, axis=1)

        data_compil = data_compil[~data_compil[gp.nc_output_ind3].isin(ajust_vars)]

        data_compil[gp.nc_output_nsfr2] = data_compil[gp.nc_output_nsfr2].str.replace('\n', ' ').str.replace('\r', '')

        var_compil = [x for x in gp.var_compil if x not in cols_sortie_excl and x in data_compil.columns]

        return data_compil[var_compil + num_cols].copy()

    def write_st_compil(self, data_compil_st, save_path, name_file):
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
        rw_ut.numpy_write(data_compil_st, save_name, self.up.nb_mois_proj_out, encoding='cp1252', mode=mode, header=header,
                          idx_desc_col_num=idx_desc_col_num, delimiter=self.up.compil_sep, decimal_symbol=self.up.compil_decimal)
