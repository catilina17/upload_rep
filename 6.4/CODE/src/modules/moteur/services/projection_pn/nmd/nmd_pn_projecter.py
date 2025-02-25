# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:52:47 2020

@author: Hossayne
"""
import modules.moteur.parameters.general_parameters as gp
import modules.moteur.services.indicateurs_taux.nmd.nmd_mtx_gap_tf as gp_nmd
import calculateur.services.projection_services.nmd.run_calculator_pn_nmd as pn_nmd
import calculateur.services.projection_services.nmd.run_nmd_spreads as nmd_spread
import modules.moteur.parameters.user_parameters as up
import modules.moteur.mappings.main_mappings as mp
import logging
import numpy as np
import pandas as pd
import psutil
import calculateur.services.projection_services.nmd.run_nmd_template as nmd_tmp

logger = logging.getLogger(__name__)


class PN_NMD_PROJECTER():

    def __init__(self, cls_calc, cls_stock_calc, cls_pn_loader, scen_path, scenario_name, nmd_prct):
        self.ALIM_NMD_MISSING = "    Certains produits de type NMD sont présents dans le scénario mais pas dans le STOCK"
        self.PROFILE_NMD_MISSING = "    Certains profils de produits de type NMD sont manquants: "
        self.CURVE_MISSING_MISSING = "    Certaines courbes de taux sont manquantes dans le scénario utilisateur ou fermat:  "
        self.NON_UNIQUE_KEYS = "    Les données du stock contiennent des clés non uniques :"

        self.IND = "nmd"
        self.scen_path = scen_path
        self.scenario_name = scenario_name
        self.prct = nmd_prct
        self.cls_pn_loader = cls_pn_loader
        self.params = cls_calc
        self.cls_stock_calc = cls_stock_calc
        self.max_month_pn = min(up.max_month_pn[self.IND + self.prct], up.nb_mois_proj_out)
        self.type_scenario = self.get_scenarios()

    def project_pn_nmd(self, dic_pn_sc, dic_stock_scr):
        pn_nmd_source_data = self.params.get_source_data_pn(self.cls_pn_loader, "nmd", self.prct)
        # pn_nmd_source_data["LDP"]["DATA"] = pn_nmd_source_data["LDP"]["DATA"][
        #    pn_nmd_source_data["LDP"]["DATA"]["CONTRAT"].str.contains("P-RSLT-FORM1")].copy()
        data_nmd = pn_nmd_source_data["LDP"]["DATA"].copy()

        if len(self.type_scenario) > 0 and len(data_nmd) > 0 and not (
        (up.indic_sortie["PN"] + up.indic_sortie_eve["PN"] == []
         and up.indic_sortie["AJUST"] + up.indic_sortie_eve[
             "AJUST"] == [])) and self.max_month_pn > 0:
            dic_sc_vars = self.get_ecoulements_from_calculateur(pn_nmd_source_data)

            data_nmd = self.reorder_data(data_nmd, dic_sc_vars)

            """ La diagonale de la matrice d'écoulement nous donne le volume de PN réel émis chaque mois"""
            self.calc_vol_pn(dic_pn_sc, dic_sc_vars)

            self.save_mni_mtx(dic_pn_sc, dic_sc_vars)

            """ SAUVEGARDE DES TAUX SC"""
            self.save_mtx_taux(dic_pn_sc, dic_sc_vars)

            """ Le TAUX client est nul """
            self.calc_tx_client(dic_pn_sc, dic_sc_vars)

            """ Calcul de la matrice de gap de taux fixe"""
            ht = data_nmd.shape[0]
            lg = up.nb_mois_proj_usr
            self.cal_mtx_gp_tf(data_nmd, dic_pn_sc, dic_sc_vars["mtx_ef"], dic_sc_vars["mtx_em"], dic_stock_scr, ht, lg)

            """ Mise sous dico """
            self.save_dico(dic_pn_sc, dic_sc_vars, data_nmd)

    def reorder_data(self, data_nmd, dic_sc_vars):
        data_index = dic_sc_vars["data_index"]
        data_nmd = \
            data_nmd[data_nmd[gp.nc_output_ind3] == self.cls_pn_loader.nc_pn_ecm_pn].reset_index().set_index(
                [gp.nc_pn_cle_pn]).loc[data_index].copy().reset_index().set_index("new_key")

        return data_nmd

    def save_mni_mtx(self, dic_pn_sc, dic_profiles):
        dic_pn_sc["mni_sc_"] = dic_profiles["mtx_mni"]
        dic_pn_sc["mni_mg_"] = np.zeros(dic_profiles["mtx_mni"].shape)
        dic_pn_sc["mni_tci_"] = dic_profiles["mtx_mni_tci"]

    def get_ecoulements_from_calculateur(self, pn_nmd_source_data):

        self.type_stock = "RCO"
        self.type_rm = "NORMAL"
        self.type_data = "pn" + self.prct

        # IMPORTANT DE REFAIRE LES TEMPLATES EN CAS DE NV PRODUITS
        nmd_source_data = {}
        nmd_source_data["PN"] = pn_nmd_source_data
        nmd_source_data["STOCK"] = mp.sources.get_contracts_files_path(self.params.etab, "ST-NMD")
        nmd_source_data["MODELS"] = self.add_source_models(self.scenario_name, "NMD", nmd_source_data)
        nmd_source_data["MODELS"] = self.add_source_models(self.scenario_name, "PEL", nmd_source_data)
        if self.prct == "":
            cls_nmd_tmp = nmd_tmp.run_nmd_template_getter(nmd_source_data, self.params.etab,
                                                          self.params.dar, save=False)
        else:
            cls_nmd_tmp = self.params.cls_nmd_tmp

        compiled_indics_st = self.cls_stock_calc.data_stock_nmd_dt

        cls_nmd_spreads = nmd_spread.run_nmd_spreads(self.params.etab, self.params.horizon, nmd_source_data,
                                                     cls_nmd_tmp, max_pn=self.max_month_pn)

        dic_sc_vars \
            = pn_nmd.run_calculator_pn_nmd(self.params.dar, self.params.horizon,
                                           nmd_source_data, "nmd_pn", self.params.etab, cls_nmd_tmp,
                                           compiled_indics_st, tx_params=self.params.tx_params.copy(),
                                           cls_nmd_spreads=cls_nmd_spreads,
                                           exit_indicators_type=["GPLIQ", "SC_TX", "TCI"],
                                           agregation_level="NMD_DT", with_dyn_data=True,
                                           max_pn=self.max_month_pn, type_rm=self.type_rm,
                                           batch_size=10000, tci_contract_perimeter=self.params.nmd_tci_perimeter,
                                           output_data_type="pn_proj", output_mode = "dataframe")

        nmd_source_data["MODELS"]["NMD"]["DATA"].Close(False)
        nmd_source_data["MODELS"]["PEL"]["DATA"].Close(False)

        return dic_sc_vars

    def get_batch_sise(self):
        base = 1500
        ram_free = 9
        ram_free_prct = 0.56
        desc_mem = psutil.virtual_memory()
        return int(base * desc_mem.free / (ram_free * 10 ** 9) * (1 - desc_mem.percent / 100) / ram_free_prct)

    def add_source_models(self, scenario_name, type, source_data):
        source_model = self.params.get_source_models(scenario_name, type)
        source_data[type] = {}
        source_data[type]["DATA"] = source_model
        return source_data

    def save_dico(self, dic_pn_sc, dic_pn, data_nmd):
        dic_pn_sc["data"] = data_nmd.copy()
        dic_pn_sc["mtx_ef"] = dic_pn["mtx_ef"]
        dic_pn_sc["mtx_em"] = dic_pn["mtx_em"]

    def cal_mtx_gp_tf(self, data_pn, dic_pn_sc, mtx_ef, mtx_em, dic_stock_scr, ht, lg):
        dic_mtx_pn_tx_fix_f = {}
        dic_mtx_pn_tx_fix_m = {}

        filtre_tla = (np.array(data_pn[gp.nc_output_index_calc_cle].isin(mp.all_gap_gestion_index))) & np.array(
            [up.retraitement_tla])

        for simul in ["LIQ", "EVE", "EVE_LIQ"]:
            if up.type_simul[simul]:
                type_tx_pn = self.load_type_taux(data_pn)
                gp_nmd.calculate_mat_gap_tf_nmd(data_pn, dic_stock_scr, simul, ht, lg, type_tx_pn, filtre_tla,
                                                self.max_month_pn, mtx_ef, mtx_em, self.prct, dic_mtx_pn_tx_fix_f,
                                                dic_mtx_pn_tx_fix_m)
                dic_pn_sc["mtx_gp_fix_m_" + simul] = dic_mtx_pn_tx_fix_m
                dic_pn_sc["mtx_gp_fix_f_" + simul] = dic_mtx_pn_tx_fix_f

    def calc_tx_client(self, dic_pn_sc, dic_profiles):
        tx_cli = np.zeros((dic_profiles["mtx_mni"].shape[0], dic_profiles["mtx_mni"].shape[2]))
        dic_pn_sc["tx_cli"] = tx_cli

    def save_mtx_taux(self, dic_pn_sc, dic_profiles):
        dic_pn_sc["tx_sc_"] = np.nan_to_num(12 * dic_profiles["mtx_mni"] / dic_profiles["mtx_em"])
        dic_pn_sc["tx_mg_"] = np.zeros(dic_pn_sc["tx_sc_"].shape)
        dic_pn_sc["tx_sc_ftp_"] = np.nan_to_num(12 * dic_profiles["mtx_mni_tci"] / dic_profiles["mtx_em"])

        dic_pn_sc["base_calc_"] = 1 / 12

    def calc_vol_pn(self, dic_pn_sc, dic_sc_vars):
        """ La diagonale de la matrice d'écoulement nous donne le volume de PN réel émis chaque mois"""
        x,y = dic_sc_vars["mtx_flux_em"].shape
        dic_pn_sc["new_pn"] = dic_sc_vars["mtx_flux_em"]
        if up.nb_mois_proj_usr - y > 0:
            dic_pn_sc["new_pn"] = np.concatenate([dic_pn_sc["new_pn"], np.zeros((x, up.nb_mois_proj_usr - y))], axis=1)

    def load_type_taux(self, data_pn):
        """ 1.3 Calcul du FILTRE TAUX FIXE"""
        type_tx_pn = pd.DataFrame(index=data_pn.index, columns=["TYPE_TX"])
        type_tx_pn["TYPE_TX"] = np.where(data_pn[gp.nc_output_index_calc_cle].str.contains(gp.FIX_ind), 'TF', 'TV')
        return type_tx_pn

    def get_scenarios(self):
        scenarii = up.scenarios_params[self.params.etab][self.scenario_name]
        if scenarii is None:
            return []
        scenarii = scenarii[scenarii["TYPE PRODUIT"].str.contains("PN %s" % self.IND.upper())].copy()
        if len(scenarii) == 0:
            return []
        return scenarii["TYPE PRODUIT"]
