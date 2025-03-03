# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:52:47 2020

@author: Hossayne
"""
import modules.moteur.parameters.general_parameters as gp
import modules.moteur.utils.generic_functions as gf
import modules.moteur.index_curves.calendar_module as hf
from modules.moteur.services.indicateurs_taux.ech.ech_mtx_gap_tf import GAP_TAUX_PN_ECH
from calculateur.services.projection_services.ech import run_calculator_pn_ech as calc_ech
import logging
import warnings
import numpy as np
import re

# from memory_profiler import profile

logger = logging.getLogger(__name__)

np.seterr(divide='ignore', invalid='ignore', over='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


class PN_ECH_PROJECTER():

    def __init__(self, cls_cal, cls_mp, cls_usr, cls_calc, cls_pn_loader, scen_path, scenario_name, ech_prct=""):
        self.mp = cls_mp
        self.up = cls_usr
        self.cal = cls_cal
        self.IND = "ech"
        self.NON_UNIQUE_KEYS = "    Les données du stock contiennent des clés non uniques :"
        self.scen_path = scen_path
        self.scenario_name = scenario_name
        self.prct = ech_prct
        self.cls_pn_loader = cls_pn_loader
        self.params = cls_calc
        self.max_month_pn = min(self.up.max_month_pn[self.IND + self.prct], self.up.nb_mois_proj_out)
        self.type_scenario = self.get_scenarios()

    def project_pn_ech(self, data_ech, dic_pn_sc, dic_stock_sci="", ):
        if len(self.type_scenario) > 0 and data_ech.shape[0] > 0 and not (
                self.up.indic_sortie["PN"] + self.up.indic_sortie_eve["PN"] == []
                and self.up.indic_sortie["AJUST"] + self.up.indic_sortie_eve["AJUST"] == []) and self.max_month_pn > 0:
            self.ht = data_ech.shape[0] // 4
            self.lg = self.up.nb_mois_proj_usr

            data_ech[gp.nc_pn_cle_pn] = data_ech[gp.nc_pn_cle_pn].str.replace("ECH-U", "ECH_U")
            data_index, dic_profiles, data_ech = self.get_ecoulements_from_calculateur(data_ech)

            data_ech = self.reorder_data(data_ech, data_index)

            jr_pn = self.load_jr_pn_ech(data_ech)
            base_calc, d_coeff = self.load_coeff_mens(data_ech)

            new_pn = self.calc_dem_lem(data_ech, dic_stock_sci)
            mtx_ef, mtx_em, mtx_mni, mtx_mni_tci, new_pn_all, mtx_tef, mtx_tem \
                = self.calc_ef_and_em_mtx(new_pn, dic_profiles, jr_pn, d_coeff)

            self.save_mni_mtx(dic_pn_sc, mtx_mni, mtx_mni_tci)

            self.save_tx_mtx(dic_pn_sc, dic_profiles)

            """ Calcul de la matrice de gap de taux fixe"""
            self.calc_mtx_gap_tf_ech(dic_pn_sc, data_ech, mtx_ef, mtx_em, mtx_tef, mtx_tem)


            """ Le TAUX client est donné par la diagonale """
            self.calc_tx_cli(dic_pn_sc, dic_profiles)

            """ La diagonale de la matrice d'écoulement nous donne le volume de PN réel émis chaque mois"""
            self.calc_vol_pn(new_pn_all, dic_pn_sc)

            self.save_in_dico(dic_pn_sc, mtx_ef, mtx_em, data_ech)

    def reorder_data(self, data_ech, data_index):
        ind_dem = self.cls_pn_loader.nc_pn_flem_pn if self.prct == "" else self.cls_pn_loader.nc_pn_ecm_pn
        data_ech = \
            data_ech[data_ech[gp.nc_output_ind3] == ind_dem].reset_index().set_index([gp.nc_pn_cle_pn]).loc[
                data_index].copy().reset_index().set_index("new_key")
        return data_ech

    def get_scenarios(self):
        scenarii = self.up.scenarios_params[self.params.etab][self.scenario_name]
        if scenarii is None:
            return [], False, False
        scenarii = scenarii[scenarii["TYPE PRODUIT"].str.contains("PN %s" % self.IND.upper())].copy()
        if len(scenarii) == 0:
            return []
        return scenarii["TYPE PRODUIT"]

    def get_ecoulements_from_calculateur(self, data_ech):

        name_product = "all_ech_pn"
        type_run_off = "profile"
        ech_source_data = {}
        ech_source_data["PN"] = {}
        ech_source_data["PN"]["LDP"] = {}
        ech_source_data = self.add_source_models(self.scenario_name, "ECH", ech_source_data)
        self.params.tx_params["apply_rate_code_map"] = False

        ech_source_data["PN"]["LDP"]["DATA"] = data_ech.copy()
        cls_ag = \
            calc_ech.run_calculator_pn_ech(self.params.dar, self.params.horizon, ech_source_data, name_product,
                                           tx_params=self.params.tx_params.copy(),
                                           type_run_off=type_run_off, agregation_level="DT",
                                           max_pn=self.max_month_pn,
                                           map_bassins=self.params.mapping_bassins_modele_rarn,
                                           exit_indicators_type=["GPLIQ","GPTX", "TCI","SC_TX"], type_ech=self.prct,
                                           batch_size = self.params.batch_size_ech, output_mode="dataframe")

        data_index, dic_profiles = self.get_profiles(cls_ag.compiled_indics, cls_ag.keep_vars_dic)

        return data_index, dic_profiles, data_ech

    def add_source_models(self, scenario_name, type, source_data):
        source_model = self.params.get_source_models(scenario_name, type)
        source_data["MODELS"] = {}
        source_data["MODELS"][type] = {}
        source_data["MODELS"][type] ["DATA"] = source_model
        return source_data

    def get_profiles(self, compiled_indics, key_vars_dic):
        compiled_indics[["CONTRAT", "PN"]] = compiled_indics[key_vars_dic["CONTRAT"]].str.split("*", expand=True)
        compiled_indics["PN"] = compiled_indics["PN"].str.replace("PN", "").astype(int)
        compiled_indics = compiled_indics.sort_values(["CONTRAT", "PN"])
        cols_num = ["M%s" % i for i in range(1, self.up.nb_mois_proj_usr + 1)]
        dic_profiles = {}
        name_profiles = ["profil_lef", "profil_lem", "profil_mni", "profil_mni_tci", "profil_sc_tx", "profil_sc_tx_tci",
                         "profil_tef", "profil_tem"]
        name_inds = ["A$LEF", "B$LEM", "E$LMN", "F$LMN_FTP", "I$SC_RATES", "J$SC_RATES_TCI", "C$TEF", "D$TEM"]
        for name_prof, name_ind in zip(name_profiles, name_inds):
            dic_profiles[name_prof] = compiled_indics.loc[
                compiled_indics[gp.nc_output_ind3] == name_ind, cols_num].values
            dic_profiles[name_prof] = dic_profiles[name_prof].reshape(self.ht, self.max_month_pn, self.lg)

        data_index = compiled_indics["CONTRAT"].unique()

        return data_index, dic_profiles

    def save_in_dico(self, dic_pn_sc, mtx_ef, mtx_em, data_pn):
        dic_pn_sc["data"] = data_pn
        list_mtx_compress = [mtx_ef, mtx_em]
        names = ["mtx_ef", "mtx_em"]
        for i in range(0, len(list_mtx_compress)):
            if i > 1:
                dic_pn_sc[names[i]] = list_mtx_compress[i]
            else:
                dic_pn_sc[names[i]] = list_mtx_compress[i]

    def save_mni_mtx(self, dic_pn_sc, mtx_mni, mtx_mni_tci):
        dic_pn_sc["mni_sc_"] = mtx_mni_tci
        dic_pn_sc["mni_mg_"] = mtx_mni - mtx_mni_tci

    def save_tx_mtx(self, dic_pn_sc, dic_profiles):
        dic_pn_sc["tx_sc_"] = 12 * dic_profiles["profil_mni_tci"] / dic_profiles["profil_lem"]
        dic_pn_sc["tx_sc_"] = np.where(dic_profiles["profil_lem"] == 0, dic_profiles["profil_sc_tx_tci"],
                                       dic_pn_sc["tx_sc_"])
        dic_pn_sc["tx_mg_"] = 12 * dic_profiles["profil_mni"] / dic_profiles["profil_lem"]
        dic_pn_sc["tx_mg_"] = np.where(dic_profiles["profil_lem"] == 0, dic_profiles["profil_sc_tx"],
                                       dic_pn_sc["tx_mg_"])
        dic_pn_sc["tx_mg_"] = dic_pn_sc["tx_mg_"] - dic_pn_sc["tx_sc_"]
        dic_pn_sc["base_calc_"] = 1 / 12

    def calc_mtx_gap_tf_ech(self, dic_pn_sc, data_pn, mtx_ef, mtx_em, mtx_tef, mtx_tem):
        gp_ech = GAP_TAUX_PN_ECH(self.up, self.mp)
        for simul in ["LIQ", "EVE", "EVE_LIQ"]:
            if self.up.type_simul[simul]:
                dic_pn_sc["mtx_gp_fix_f_" + simul] = {}
                dic_pn_sc["mtx_gp_fix_m_" + simul] = {}
                gp_ech.calculate_mat_gap_tf_ech(data_pn, self.max_month_pn, self.ht, self.lg,
                                                dic_pn_sc["mtx_gp_fix_f_" + simul],
                                                dic_pn_sc["mtx_gp_fix_m_" + simul], mtx_ef, mtx_em, mtx_tef, mtx_tem)


    def calc_vol_pn(self, new_pn_all, dic_pn_sc):
        new_pn = np.vstack([np.diag(new_pn_all[i]) for i in range(0, new_pn_all.shape[0])])
        new_pn = np.column_stack([new_pn, np.zeros((self.ht, self.lg - self.max_month_pn))])
        dic_pn_sc["new_pn"] = new_pn
        gf.clean_df(new_pn_all)

    def calc_tx_cli(self, dic_pn_sc, dic_profiles):
        tx_tot = dic_profiles["profil_sc_tx"]
        tx_cli = np.concatenate(
            [tx_tot[:, i, i].reshape(self.ht, 1) for i in range(0, self.max_month_pn)] + \
            [np.zeros((self.ht, self.up.nb_mois_proj_usr - self.max_month_pn))], axis=1)
        dic_pn_sc["tx_cli"] = tx_cli

    def calc_ef_and_em_mtx(self, new_pn, dic_profiles, jr_pn, d_coeff):
        """ 1. CREATION D'UNE MATRICE DE dimension nb_contrats * max_month_pn * nb_mois_proj """
        """ L'idée est que chaque PN émise au mois M constitue un nouveau contrat"""
        new_pn_all = np.vstack([np.triu(np.ones((self.max_month_pn, self.lg)))] * self.ht).reshape(self.ht,
                                                                                                   self.max_month_pn,
                                                                                                   self.lg)
        new_pn_all = new_pn_all * (np.array(new_pn)[:, :self.max_month_pn].reshape(self.ht, self.max_month_pn, 1))

        """ Calcul de la matrice des encours"""
        mtx_pn_f, mtx_pn_m, mtx_mni, mtx_mni_tci, mtx_tef, mtx_tem \
            = self.calc_encours_mtx_visions_taux(new_pn_all, dic_profiles, jr_pn, d_coeff, )

        return mtx_pn_f, mtx_pn_m, mtx_mni, mtx_mni_tci, new_pn_all, mtx_tef, mtx_tem


    def load_coeff_mens(self, data_pn):
        base_calc = self.cal.calendar_coeff.copy()[
            [gp.nc_pn_base_calc] + ["M" + str(i) for i in range(1, self.up.nb_mois_proj_usr + 1)]]

        d_coeff = base_calc.loc[base_calc[gp.nc_pn_base_calc] == "DCoeff", :]
        d_coeff = d_coeff[[x for x in d_coeff.columns if x != gp.nc_pn_base_calc]].copy()
        d_coeff.columns = ["M" + str(i) for i in range(1, self.up.nb_mois_proj_usr + 1)]
        d_coeff = np.array(d_coeff.astype(np.float64))

        base_calc_data = data_pn[[gp.nc_pn_base_calc]].copy()
        base_calc_data = base_calc_data.merge(how="left", right=base_calc, on="BASE_CALC")
        base_calc_data = np.array(
            base_calc_data[[x for x in base_calc_data.columns if x != "BASE_CALC"]].astype(np.float64))

        return base_calc_data, d_coeff

    def load_jr_pn_ech(self, data_pn):
        data_pn[gp.nc_pn_jr_pn] = [str(int(float(x))) if not re.match(r'^-?\d+(?:\.\d+)?$', str(x)) is None else str(x)
                                   for x in
                                   data_pn[gp.nc_pn_jr_pn]]
        jr_pn = np.array(np.where(data_pn[gp.nc_pn_jr_pn].astype(str) == '1', 1, 15)).reshape(self.ht, 1)

        return jr_pn

    def calc_dem_lem(self, data_pn, dic_stock_sci):
        dyn_am = data_pn[["M" + str(x) for x in range(1, self.max_month_pn + 1)]].copy()
        if self.prct == "%":
            shape_prec = dyn_am.shape[0]
            lem = dic_stock_sci[gp.em_sti].add_suffix('_LEM')
            dyn_am = dyn_am.join(lem, on="new_key", how="left").copy().fillna(0)
            for j in range(1, self.max_month_pn + 1):
                """ On retranche les encours du stock aux encours dynamiques"""
                dyn_am["M" + str(j)] = dyn_am["M" + str(j)] - dyn_am["M" + str(j) + "_LEM"]

            if dyn_am.shape[0] != shape_prec:
                logger.error(self.NON_UNIQUE_KEYS)
                raise ValueError(self.NON_UNIQUE_KEYS)

        new_pn = dyn_am[[x for x in dyn_am.columns if x != "M0" and "_LEM" not in x]].reset_index(drop=True)
        new_pn = gf.fill_nan2(new_pn)

        return new_pn

    def calc_encours_mtx_visions_taux(self, new_pn_all, dic_profiles, jr_pn, d_coeff):
        if self.prct == "%":
            mtx_pn_f = np.zeros((self.ht, self.max_month_pn, self.lg))
            mtx_pn_m = np.zeros((self.ht, self.max_month_pn, self.lg))
            mtx_mni = np.zeros((self.ht, self.max_month_pn, self.lg))
            mtx_mni_tci = np.zeros((self.ht, self.max_month_pn, self.lg))
            mtx_tef = np.zeros((self.ht, self.max_month_pn, self.lg))
            mtx_tem = np.zeros((self.ht, self.max_month_pn, self.lg))
            for i in range(0, self.max_month_pn):
                """ 3.1 Les encours réels  d'une PN créée au mois i, dépend
                de manière récursive des encours au mois i des PN créés les mois précédents """
                """ Encours du mois i de la PN créée en i"""
                new_pn_all[:, i, i] = (new_pn_all[:, i, i] - (mtx_pn_m[:, :i, i].sum(axis=1)))
                new_pn_all[:, i, i] = np.where(jr_pn.reshape(self.ht) == 15,
                                               new_pn_all[:, i, i] * ((1 / (1 - d_coeff))[:, i]), new_pn_all[:, i, i])
                """ Application du nouvel encours calculé en i au mois suivants le mois i"""
                new_pn_all[:, i, i + 1:self.lg] = np.ones((self.ht, self.lg - i - 1)) * new_pn_all[:, i, i].reshape(
                    self.ht, 1)

                """ La matrice d'écoulement EF """
                mtx_pn_f[:, i, :] = dic_profiles["profil_lef"][:, i, :] * new_pn_all[:, i, :]
                mtx_pn_m[:, i, :] = dic_profiles["profil_lem"][:, i, :] * new_pn_all[:, i, :]
                mtx_mni[:, i, :] = dic_profiles["profil_mni"][:, i, :] * new_pn_all[:, i, :]
                mtx_mni_tci[:, i, :] = dic_profiles["profil_mni_tci"][:, i, :] * new_pn_all[:, i, :]
                mtx_tef[:, i, :] = dic_profiles["profil_tef"][:, i, :] * new_pn_all[:, i, :]
                mtx_tem[:, i, :] = dic_profiles["profil_tem"][:, i, :] * new_pn_all[:, i, :]
        else:
            mtx_pn_f = dic_profiles["profil_lef"] * new_pn_all
            mtx_pn_m = dic_profiles["profil_lem"] * new_pn_all
            mtx_mni = dic_profiles["profil_mni"] * new_pn_all
            mtx_mni_tci = dic_profiles["profil_mni_tci"] * new_pn_all
            mtx_tef = dic_profiles["profil_tef"] * new_pn_all
            mtx_tem = dic_profiles["profil_tem"] * new_pn_all

        return mtx_pn_f, mtx_pn_m, mtx_mni, mtx_mni_tci, mtx_tef, mtx_tem
