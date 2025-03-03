# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:06:57 2020

@author: Hossayne
"""

import modules.moteur.parameters.general_parameters as gp
from modules.moteur.services.indicateurs_liquidite.gap_liquidite_module import Gap_Liquidite_Calculator
import modules.moteur.services.indicateurs_liquidite.outflow_module as outfl
from modules.moteur.services.indicateurs_taux.mni_module import MNI_Calculator
from modules.moteur.services.indicateurs_taux.gap_taux_agreg import Gap_Taux_Calculator
from modules.moteur.services.indicateurs_taux.eve_module import EVE_Calculator
import modules.moteur.utils.read_write_utils as rw_ut
import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class Ajustements_Generator():
    
    def __init__(self, cls_usr, cls_mp, cls_tx):
        self.mp = cls_mp
        self.up = cls_usr
        self.tx = cls_tx
        self.data_adj_mni = []
        self.data_adj_eve =  []
    
    def clear_adjustments(self, ):
        self.data_adj_mni = []
        self.data_adj_eve = []
    
    def prepare_adjustements(self, data, type="NORMAL"):
        """FILTRAGE DES DATA """
    
        if type != "EVE":
            ind_enc = [gp.ef_sti, gp.em_sti]
        else:
            ind_enc = [gp.ef_eve_sti, gp.em_eve_sti]
    
        list_fx_swaps = self.mp.contrats_fx_swaps.index.values.tolist()
        filtro = (data[gp.nc_output_isIG].isin(["-", "IG"])) & (
                (~data[gp.nc_output_bilan].str.contains("HB")) | (data[gp.nc_output_contrat_cle].isin(list_fx_swaps))) & (
                     data[gp.nc_output_ind3].isin(ind_enc))
    
        agreg_cols = [gp.nc_output_scope_cle, gp.nc_output_devise_cle, gp.nc_output_ind3,\
                      gp.nc_output_etab_cle, gp.nc_output_bassin_cle]
        num_cols = ["M0" + str(k) if k < 10 else "M" + str(k) for k in range(0, self.up.nb_mois_proj_usr + 1)]
    
        data = data.loc[filtro, agreg_cols + [gp.nc_output_bilan] + num_cols]
        ht = data.shape[0]
    
        """ METTRE LE PASSIF EN NEGATIF"""
        data[num_cols] = np.where(np.array((data[gp.nc_output_bilan].str.contains("PASSIF"))).reshape((ht, 1)),
                                  -data[num_cols],
                                  data[num_cols])
    
        """ CALCULER LE NV SCOPE"""
        data["SCOPE_UP"] = [x.upper() for x in data[gp.nc_output_scope_cle].astype(str)]
    
        data["SCOPE_UP"] = data["SCOPE_UP"].astype(str)
    
        fi = data["SCOPE_UP"].str.contains("INSURANCE")
        ft = data["SCOPE_UP"].str.contains("TRADING")
        fa = (~fi) & (~ft)
    
        fl = data["SCOPE_UP"].str.contains("LIQUIDITE") & (~data["SCOPE_UP"].str.contains("MNI"))
        fm = data["SCOPE_UP"].str.contains("MNI") & (~data["SCOPE_UP"].str.contains("LIQUIDITE"))
    
        fb = (~fl & ~fm)
    
        filtres = [fi & fb, ft & fb, fa & fb, fi & fl, ft & fl, fa & fl, fi & fm, ft & fm, fa & fm]
    
        choix = [x + y for x, y in zip(sorted(["LIQUIDITE", "MNI", "BOTH"] * 3), ["_INSURANCE", "_TRADING", "_AUTRES"] * 3)]
    
        data["SCOPE_AJUST"] = np.select(filtres, choix)
    
        """ AGREGATION"""
        agreg_cols = ["SCOPE_AJUST", gp.nc_output_devise_cle, gp.nc_output_ind3, gp.nc_output_etab_cle, gp.nc_output_bassin_cle]
        data = data[agreg_cols + num_cols].copy().reset_index(drop=True)
        data = data.groupby(agreg_cols, as_index=False).sum()
    
        if type != "EVE":
            if len(self.data_adj_mni) == 0:
                self.data_adj_mni = data.reset_index(drop=True)
            else:
                self.data_adj_mni = pd.concat([self.data_adj_mni, data.reset_index(drop=True)])
        else:
            if len(self.data_adj_eve) == 0:
                self.data_adj_eve = data.reset_index(drop=True)
            else:
                self.data_adj_eve = pd.concat([self.data_adj_eve, data.reset_index(drop=True)])
    
    
    def add_adjustments(self, dic_ajust, type="NORMAL"):
        data_adj = self.data_adj_mni.copy() if type != "EVE" else self.data_adj_eve.copy()
        ind_em = gp.em_sti if type != "EVE" else gp.em_eve_sti
        ind_ef = gp.ef_sti if type != "EVE" else gp.ef_eve_sti
        if (len(self.up.indic_sortie["AJUST"]) > 0 or len(self.up.indic_sortie_eve["AJUST"]) > 0) and len(data_adj) > 0:
            """ AJOUT VARIABLES: SCOPE_AJUST"""
            both_ajust = data_adj[data_adj["SCOPE_AJUST"].str.contains("BOTH")].copy()
            data_adj["SCOPE_AJUST"] = data_adj["SCOPE_AJUST"].str.replace("BOTH", "LIQUIDITE")
            both_ajust["SCOPE_AJUST"] = both_ajust["SCOPE_AJUST"].str.replace("BOTH", "MNI")
            data_adj = pd.concat([data_adj, both_ajust])
    
            """ AGREGATION """
            num_cols = ["M0" + str(k) if k < 10 else "M" + str(k) for k in range(0, self.up.nb_mois_proj_usr + 1)]
            agreg_cols = ["SCOPE_AJUST", gp.nc_output_devise_cle, gp.nc_output_ind3, gp.nc_output_etab_cle, gp.nc_output_bassin_cle]
            data_adj = data_adj[agreg_cols + num_cols].copy()
            data_adj = data_adj.groupby(agreg_cols, as_index=False).sum()
    
            data_devise = data_adj[[gp.nc_output_devise_cle]].copy()
            data_devise[gp.nc_output_devise_cle] = np.where(~data_devise.iloc[:, 0].isin(self.up.courbes_ajust.index.values.tolist()),
                                                            "*", data_devise.iloc[:, 0].values)
            data_adj[["CURVE_NAME", "TENOR"]] \
                = data_devise.join(self.up.courbes_ajust, on=[gp.nc_output_devise_cle]).iloc[:, -2:]
    
            """ CALCUL INDIC """
            data_adj_em = data_adj[data_adj[gp.nc_output_ind3] == ind_em].copy()
    
            """ ENCOURS """
            dic_ajust[ind_ef] = data_adj[data_adj[gp.nc_output_ind3] == ind_ef].copy()
            dic_ajust[ind_em] = data_adj_em.copy()
            for p in range(1, self.up.nb_mois_proj_out + 1):
                if ind_ef + str(p) in self.up.indic_sortie["AJUST"]:
                    dic_ajust[ind_ef + str(p)] = dic_ajust[ind_ef].copy()
                    dic_ajust[ind_ef + str(p)][gp.nc_output_ind3] = ind_ef + str(p)
                    dic_ajust[ind_ef + str(p)][num_cols] = 0
    
                if ind_em + str(p) in self.up.indic_sortie["AJUST"]:
                    dic_ajust[ind_em + str(p)] = dic_ajust[ind_em].copy()
                    dic_ajust[ind_em + str(p)][gp.nc_output_ind3] = ind_em + str(p)
                    dic_ajust[ind_em + str(p)][num_cols] = 0
    
        return dic_ajust
    
    def divide_by_actif_passif(self, dic_ajust, data_adj_all, num_cols, type="NORMAL"):
        ind_em = gp.em_sti if type != "EVE" else gp.em_eve_sti
        data_adj_act = data_adj_all.copy()
        data_adj_passif = data_adj_all.copy()
        data_adj_em = data_adj_all[data_adj_all[gp.nc_output_ind3] == ind_em].copy()
        data_adj_em_num = np.array(data_adj_em[num_cols].copy())
        data_adj_num = np.array(data_adj_all[num_cols].copy())
    
        filtro = np.vstack([(data_adj_em_num > 0)] * len(dic_ajust))
        data_adj_num_act = np.where(filtro, 0, (-1) * data_adj_num)
        data_adj_num_pass = np.where(filtro, data_adj_num, 0)
    
        data_adj_act[num_cols] = data_adj_num_act
        data_adj_passif[num_cols] = data_adj_num_pass
        data_adj_act[gp.nc_output_bilan] = "B ACTIF"
        data_adj_passif[gp.nc_output_bilan] = "B PASSIF"
    
        return pd.concat([data_adj_act, data_adj_passif])
    
    def calculate_indics(self, dic_ajust, name_scenario, save_path):
        mni = MNI_Calculator(self.up, self.mp, self.tx)
        gp_ag = Gap_Taux_Calculator(self.up, self.mp)
        gpl = Gap_Liquidite_Calculator(self.up, self.mp)
        eve = EVE_Calculator(self.up, self.mp, self.tx)
        if dic_ajust !={}:
            indic_sortie = self.up.indic_sortie["AJUST"] + self.up.indic_sortie_eve["AJUST"]
            num_cols = ["M0" + str(k) if k < 10 else "M" + str(k) for k in range(0, self.up.nb_mois_proj_usr + 1)]
    
            """ GAP TX """
            for simul in list(self.up.type_simul.keys()):
                if self.up.type_simul[simul] == True:
                    gp_ag.calcul_gap_tx_ajust(dic_ajust, num_cols, simul)
    
            """ MNI """
            mni.calculate_mni_ajust(dic_ajust, num_cols)
    
            mni.calculate_mni_ajust_gp_rg(dic_ajust, num_cols)
    
            """ GAP LIQ """
            for simul in list(self.up.type_simul.keys()):
                if self.up.type_simul[simul] == True:
                    gpl.calcul_gap_liq_ajust(dic_ajust, simul, num_cols)
    
            """ CALCUL de l'EVE """
            eve.calcul_eve(dic_ajust, [], "AJUST", cols=num_cols)
    
            """ OUTFLOW """
            #outfl.calculate_outflows(dic_ajust, "AJUST")
    
            """ VOL PN """
            if gp.vol_pn_pni in indic_sortie:
                dic_ajust[gp.vol_pn_pni] = dic_ajust[gp.em_pni].copy()
                dic_ajust[gp.vol_pn_pni][gp.nc_output_ind3] = gp.vol_pn_pni
    
            """ AGREGATION DES INDICATEURS"""
            data_adj_all = None
            for ind, val in dic_ajust.items():
                if data_adj_all is None:
                    data_adj_all = val.copy()
                else:
                    data_adj_all = pd.concat([data_adj_all, val.copy()])
    
            """ DIVISION EN ACTIF/PASSIF en se basant sur le signe de l'EM """
            data_adj_all_eve = data_adj_all[data_adj_all[gp.nc_output_ind3].isin(self.up.indic_sortie_eve["AJUST"] + [gp.em_eve_pni])]
            data_adj_all_liq = data_adj_all[data_adj_all[gp.nc_output_ind3].isin(self.up.indic_sortie["AJUST"] + [gp.em_pni])]
    
            dic_ajust_eve = {key: val for key, val in dic_ajust.items() if key in self.up.indic_sortie_eve["AJUST"] + [gp.em_eve_pni]}
            dic_ajust_liq = {key: val for key, val in dic_ajust.items() if key in self.up.indic_sortie["AJUST"] + [gp.em_pni]}
    
            data_compil_adjs = self.divide_by_actif_passif(dic_ajust_liq, data_adj_all_liq, num_cols)
    
            if len(dic_ajust_eve) > 0:
                data_compil_adjs_eve = self.divide_by_actif_passif(dic_ajust_eve, data_adj_all_eve, num_cols, type="EVE")
                data_compil_adjs = pd.concat([data_compil_adjs_eve, data_compil_adjs])
    
    
            """ AJOUT VARIABLES DESCRIPTIVES"""
            filter_tx_cli = (data_compil_adjs[gp.nc_output_ind3] != gp.tx_cli_pni)
            data_compil_adjs.loc[filter_tx_cli, num_cols] = data_compil_adjs.loc[filter_tx_cli, num_cols].round(0)
    
            data_compil_adjs[gp.nc_output_sc] = name_scenario
            data_compil_adjs[gp.nc_output_contrat_cle] = np.where(data_compil_adjs[gp.nc_output_bilan] == "B ACTIF",
                                                                  "A-AJUST",
                                                                  "P-AJUST")
            data_compil_adjs[gp.nc_output_poste] = np.where(data_compil_adjs[gp.nc_output_bilan] == "B ACTIF",
                                                            "AJUSTEMENT - A",
                                                            "AJUSTEMENT - P")
            data_compil_adjs[gp.nc_output_index_agreg] = data_compil_adjs[gp.nc_output_devise_cle] + "3M"
    
            for i in [2, 3, 4]:
                data_compil_adjs[gp.nc_output_dim2.replace("2", "") + str(i)] = np.where(
                    data_compil_adjs[gp.nc_output_bilan] == "B ACTIF", "AJUSTEMENT ACTIF", "AJUSTEMENT PASSIF")
            data_compil_adjs[gp.nc_output_dim5] = np.where(data_compil_adjs[gp.nc_output_bilan] == "B ACTIF",
                                                           "AJUSTEMENT - A",
                                                           "AJUSTEMENT - P")
            data_compil_adjs[gp.nc_output_ind1] = "PN"
            data_compil_adjs[gp.nc_output_ind2] = "AJ00"
            data_compil_adjs[gp.nc_output_scope_cle] = data_compil_adjs["SCOPE_AJUST"]
            data_compil_adjs[gp.nc_output_lcr_share] = 100
    
            for col in gp.var_compil:
                if col not in data_compil_adjs.columns.tolist():
                    data_compil_adjs[col] = "-"
    
            data_compil_adjs[gp.nc_output_contrat_cle] = np.where(
                data_compil_adjs[gp.nc_output_scope_cle].str.contains("LIQUIDITE"),
                data_compil_adjs[gp.nc_output_contrat_cle] + "-LIQ", data_compil_adjs[gp.nc_output_contrat_cle])
    
            filtre_liq = data_compil_adjs[gp.nc_output_scope_cle].str.contains("LIQUIDITE")
            col_liq = [gp.nc_output_poste, gp.nc_output_dim5]
            for col in col_liq:
                data_compil_adjs.loc[filtre_liq, col] = data_compil_adjs.loc[filtre_liq, col] + " LIQUIDITE"
    
            """ MAPPING LIQ"""
            data_compil_adjs = self.mp.map_liq(data_compil_adjs, override=True)
    
            """ CALC NSFR & LCR"""
            #data_compil_adjs = lcr.calculate_lcr(data_compil_adjs, typo="AJUST")
            #data_compil_adjs = nsfr.calculate_nsfr(data_compil_adjs, typo="AJUST")
    
            """ MISE A ZERO DES IND SELON SCOPE """
            filtre_liq = data_compil_adjs[gp.nc_output_scope_cle].str.contains("LIQUIDITE")
            filtre_mni = data_compil_adjs[gp.nc_output_scope_cle].str.contains("MNI")
            filtre_gp_lq = data_compil_adjs[gp.nc_output_ind3].str.contains("GP LQ")
            filtre_gp_tf = data_compil_adjs[gp.nc_output_ind3].str.contains("GP TF")
            filtre_gp_inf = data_compil_adjs[gp.nc_output_ind3].str.contains("GP INF")
            filtre_gp_reg = data_compil_adjs[gp.nc_output_ind3].str.contains("GP RG")
            filtre_tx_cli = data_compil_adjs[gp.nc_output_ind3].str.contains("TX CLI")
            filtre_mn = data_compil_adjs[gp.nc_output_ind3].str.startswith("MN")
            filtre_eve = data_compil_adjs[gp.nc_output_ind3].str.startswith("EVE")
            filtre_flux = data_compil_adjs[gp.nc_output_ind3].str.startswith("FLUX")
    
            data_compil_adjs.loc[
                filtre_liq & (filtre_mn | filtre_gp_tf | filtre_gp_inf | filtre_gp_reg | filtre_tx_cli | filtre_eve | filtre_flux), num_cols] = 0
            data_compil_adjs.loc[filtre_mni & filtre_gp_lq, num_cols] = 0
    
            """" FILTRER LES INDICS DE SORTIE """
            data_compil_adjs = data_compil_adjs[data_compil_adjs[gp.nc_output_ind3].isin(indic_sortie)].copy()
    
            """ ORDONNER LES INDICATEURS"""
            ind03 = np.array(data_compil_adjs[gp.nc_output_ind3])
    
            list_filtres = []
            for ind in indic_sortie:
                list_filtres.append((ind03 == ind))
    
            choices = [i for i in range(1, len(indic_sortie) + 1)]
    
            data_compil_adjs["ordre"] = np.select(list_filtres, choices)
    
            data_compil_adjs['CLE'] = data_compil_adjs[gp.cle_stock].apply(lambda row: '_'.join(row.values.astype(str)),
                                                                           axis=1)
    
            data_compil_adjs = data_compil_adjs.sort_values(["CLE", "ordre"]).drop(["ordre"], axis=1).reset_index(
                drop=True).reindex()
    
            """ FORMATAGE DE NSFR2 => enlever les sauts de ligne"""
            data_compil_adjs[gp.nc_output_nsfr2] = data_compil_adjs[gp.nc_output_nsfr2].str.replace('\n', ' ').str.replace(
                '\r', '')
    
            """ ORDONNER ET FILTRER COLONNES DE SORTIE"""
            data_compil_adjs = data_compil_adjs[gp.var_compil + num_cols].copy()
            data_compil_adjs = data_compil_adjs[[x for x in data_compil_adjs.columns if x not in self.up.cols_sortie_excl]]
    
            """ REMPLIR VAL MANQUANTES PAR 0"""
            data_compil_adjs = data_compil_adjs.copy().reset_index(drop=True)
            cols_num = ["M0" + str(k) if k < 10 else "M" + str(k) for k in range(0, self.up.nb_mois_proj_usr + 1)]
            data_compil_adjs[cols_num] = data_compil_adjs[cols_num].fillna(0).replace((np.inf, -np.inf), (0, 0))
    
            """ ELIMINATION DES COLS NUMERIQUES NON ECRITES"""
            data_compil_adjs = data_compil_adjs.drop(["M0" + str(k) if k < 10 else "M" + str(k) for k in
                                                      range(self.up.nb_mois_proj_out + 1, self.up.nb_mois_proj_usr + 1)], axis=1)
    
            """ INDICES DES COLONNES DESCRIPTIVES NUMERIQUES"""
            idx_desc_col_num = {}
            if gp.nc_output_lcr_share in data_compil_adjs.columns.tolist():
                idx_desc_col_num[data_compil_adjs.columns.tolist().index(gp.nc_output_lcr_share)] = "%.2f"
    
            """ ECRITURE DS COMPIL"""
            if gp.tx_cli_pni in indic_sortie:
                i_tx_cli = (data_compil_adjs[data_compil_adjs[gp.nc_output_ind3] == gp.tx_cli_pni]).index[0] + 1
                f_tx_cli = "%.7f"
            else:
                i_tx_cli = ""
                f_tx_cli = ""
    
            name_file = gp.nom_compil_st_pn if self.up.merge_compil else gp.nom_compil_pn
            save_name = os.path.join(save_path, name_file)
            appendo = True if os.path.exists(save_name) else False
            mode = 'a' if appendo else 'w'
            header = False if appendo else True
            rw_ut.numpy_write(data_compil_adjs, save_name, self.up.nb_mois_proj_out,  encoding='cp1252', mode=mode, header=header, \
                           i_tx_cli=i_tx_cli, format_output_tx_cli=f_tx_cli, nb_indic=len(indic_sortie),
                           idx_desc_col_num=idx_desc_col_num, delimiter = self.up.compil_sep, decimal_symbol=self.up.compil_decimal)
