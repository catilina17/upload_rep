import modules.moteur.parameters.general_parameters as gp
from modules.moteur.utils import generic_functions as gf
import logging
import pandas as pd
import numpy as np
import modules.moteur.indicators.dependances_indic as di

logger = logging.getLogger(__name__)


class Gap_Taux_Calculator():

    def __init__(self, cls_usr, cls_mp):
        self.up = cls_usr
        self.mp = cls_mp

    def calcul_gap_tx_agreg(self, data, dic_stock_scr, dic_sci, typo, simul, data_dic=[], type_pn=""):
        dic_gp_f = {}
        dic_gp_m = {}
        dic_gp_inf_f = {}
        dic_gp_inf_m = {}
        dic_gp_rg_f = {}
        dic_gp_rg_m = {}
        dic_gp_liq_f = {}
        dic_gp_liq_m = {}

        if typo == "STOCK":
            lef, lem, tef, tem = self.get_encours(dic_sci)
            tab_fix_ef_st, tab_fix_em_st = self.get_tab_fix_st(dic_stock_scr, "STOCK")
        else:
            lef, lem = data_dic["mtx_ef"], data_dic["mtx_em"]
            tef, tem = [], []
            lef_fix_pn, lem_fix_pn = self.get_data_mtx_fix(data_dic, simul)
            if ((self.up.force_gps_nmd and simul == "LIQ") or (
                    self.mp.mapping_eve["force_gps_nmd_eve"] and "EVE" in simul)) and "nmd" in type_pn:
                tab_fix_ef_st, tab_fix_em_st = self.get_tab_fix_st(dic_stock_scr, "PN", data=data)

        filtre_tla = self.get_filtres_tla(data, typo)

        conv_ecoul, is_there_conv = self.get_conv_ecoulement(data, typo, simul)

        coeff_gp_inf, filtre_inf = self.get_coeff_gp_inf(self.mp.mapping_gp_reg_params, data, gp.nc_output_index_calc_cle)

        coeff_index_ntf = self.get_coeff_gp_tf(self.mp.mapping_gp_reg_params, data, gp.nc_output_index_calc_cle)

        for k in range(0, self.up.nb_mois_proj_out + 1):

            self.calc_gap_liq(k, lef, lem, dic_gp_liq_f, dic_gp_liq_m, conv_ecoul, is_there_conv, typo, simul)

            if typo == "STOCK":
                self.calculate_gp_tx_refix_st(k, dic_gp_liq_f, dic_gp_liq_m, tef, tem, tab_fix_ef_st, tab_fix_em_st, \
                                         filtre_tla, dic_gp_f, dic_gp_m, is_there_conv, "STOCK", simul)
            else:
                self.calcul_gap_tx_mtx_pn(k, dic_gp_f, dic_gp_m, lef_fix_pn, lem_fix_pn, simul)
                if "ech" not in type_pn or (
                        "ech" in type_pn and not (self.up.type_simul["EVE"] or self.up.type_simul["EVE_LIQ"])):
                    self.clean_mtx_gap(data_dic, k, simul)

                if ((self.up.force_gps_nmd and simul == "LIQ") or (
                        self.mp.mapping_eve["force_gps_nmd_eve"] and "EVE" in simul)) and "nmd" in type_pn:
                    dic_gp_f_conv = {};
                    dic_gp_m_conv = {};
                    self.calculate_gp_tx_refix_st(k, dic_gp_liq_f, dic_gp_liq_m, tef, tem, tab_fix_ef_st, tab_fix_em_st, \
                                             filtre_tla, dic_gp_f_conv, dic_gp_m_conv, is_there_conv, "PN", simul)
                    self.choose_gp_tx_pn(k, dic_gp_f, dic_gp_m, dic_gp_f_conv, dic_gp_m_conv, is_there_conv, simul)

            self.calculate_gap_inflation(k, dic_gp_liq_f, dic_gp_liq_m, dic_gp_inf_f, dic_gp_inf_m, dic_gp_f, dic_gp_m,
                                    coeff_gp_inf, filtre_inf, typo, simul)

            self.adjust_gp_tf_tla(k, dic_gp_liq_f, dic_gp_liq_m, dic_gp_f, dic_gp_m, coeff_index_ntf, typo, simul)

            self.calculate_gap_reg(k, dic_gp_rg_f, dic_gp_rg_m, dic_gp_f, dic_gp_inf_f, dic_gp_m, dic_gp_inf_m, typo, simul)

            self.format_gp_tx(k, data, dic_sci, dic_gp_f, dic_gp_m, dic_gp_inf_f, dic_gp_inf_m, dic_gp_rg_f, dic_gp_rg_m,
                         typo,
                         simul)

        return is_there_conv

    def get_data_mtx_fix(self, data_dic, simul):
        return data_dic["mtx_gp_fix_f_" + simul], data_dic["mtx_gp_fix_m_" + simul]

    def clean_mtx_gap(self, data_dic, k, simul):
        if k in data_dic["mtx_gp_fix_f_" + simul]:
            gf.clean_df(data_dic["mtx_gp_fix_f_" + simul][k])

        if k in data_dic["mtx_gp_fix_m_" + simul]:
            gf.clean_df(data_dic["mtx_gp_fix_m_" + simul][k])

    def calculate_gp_tx_refix_st(self, k, dic_gp_liq_f, dic_gp_liq_m, tef, tem, tab_fix_f, \
                                 tab_fix_m, filtre_tla, dic_gp_f, dic_gp_m, is_there_conv, typo, simul):
        if "EVE" == simul:
            do_f = False
            do_m = (di.gap_tx_eve_em_st[k] or k == 0) if typo == "STOCK" else di.gap_tx_eve_em_pn[k]
        elif "EVE_LIQ" == simul:
            do_f = False
            do_m = di.gap_inf_eve_em_st[k] if typo == "STOCK" else di.gap_inf_eve_em_pn[k]
        else:
            do_f = di.gap_tx_ef_st[k] if typo == "STOCK" else di.gap_tx_ef_pn[k]
            do_m = di.gap_tx_em_st[k] if typo == "STOCK" else di.gap_tx_em_pn[k]

        if do_f or do_m:
            """ MOIS REFIXING TLA"""
            mois_refix_tla = k + self.up.freq_refix_tla + 1 if k >= self.up.mois_refix_tla else self.up.mois_refix_tla + 1

            if k >= 1:
                """ les gaps de taux de mois >=1"""
                if do_f:
                    lef_k_ec = dic_gp_liq_f[k]
                    """ PREP REFIXING"""
                    tab_fix_f_rol = self.roll_tab(tab_fix_f, k, typo)
                    """ TRAITEMENT TLA """
                    tab_fix_f_rol = self.change_fixing_tla(tab_fix_f_rol, k, filtre_tla, mois_refix_tla)
                    """ TRAITEMENT CONV """
                    tab_fix_f_rol_c = np.where(is_there_conv, np.where(tab_fix_f_rol == 0, 0, 1), tab_fix_f_rol)
                    """ Calcul du GAP EF de taux de mois k"""
                    dic_gp_f[k] = lef_k_ec * tab_fix_f_rol_c

                if do_m:
                    lem_k_ec = dic_gp_liq_m[k]
                    """ PREP REFIXING"""
                    tab_fix_m_rol = self.roll_tab(tab_fix_m, k, typo)
                    """ TRAITEMENT TLA """
                    tab_fix_m_rol = self.change_fixing_tla(tab_fix_m_rol, k, filtre_tla, mois_refix_tla)
                    """ TRAITEMENT CONV """
                    tab_fix_m_rol_c = np.where(is_there_conv, np.where(tab_fix_m_rol == 0, 0, 1), tab_fix_m_rol)
                    """ Calcul du GAP EM de taux de mois k"""
                    dic_gp_m[k] = lem_k_ec * tab_fix_m_rol_c

            else:
                """ pour le gap  de taux au mois 0"""
                if do_f:
                    if k == 0 and typo == "STOCK" and simul == "EVE":
                        dic_gp_f["0_NC"] = tef

                    dic_gp_f[k] = np.where(is_there_conv, dic_gp_liq_f[0] * np.where(tef == 0, 0, 1), tef)

                if do_m:
                    if k == 0 and typo == "STOCK" and simul == "EVE":
                        dic_gp_m["0_NC"] = tem

                    dic_gp_m[k] = np.where(is_there_conv, dic_gp_liq_m[0] * np.where(tem == 0, 0, 1), tem)

    def calc_gap_liq(self, k, lef, lem, dic_gp_liq_f, dic_gp_liq_m, conv_ecoul, is_there_conv, typo, simul):
        if "EVE" in simul:
            do_f = False
            do_m = di.gap_liq_eve_em_st[k] or k == 0 if typo == "STOCK" else di.gap_liq_eve_em_pn[k]
        else:
            do_f = di.gap_liq_ef_st[k] if typo == "STOCK" else di.gap_liq_ef_pn[k]
            do_m = di.gap_liq_em_st[k] if typo == "STOCK" else di.gap_liq_em_pn[k]

        if typo == "STOCK":
            p = k
        else:
            p = max(0, k - 1)

        if do_f:
            conv_ecoul_rk = np.roll(conv_ecoul, k, axis=1)
            lef_k_ec = lef.copy() if typo == "STOCK" else gf.sum_k(lef.copy(), k)
            lef_k_ec[:, :p] = 0
            if k == 0 and typo == "STOCK" and simul == "EVE":
                dic_gp_liq_f["0_NC"] = lef_k_ec.copy()
            lef_k_ec[:, p:] = np.where(is_there_conv, conv_ecoul_rk[:, k:] * lef_k_ec[:, p:p + 1], lef_k_ec[:, p:])
            dic_gp_liq_f[k] = lef_k_ec

        if do_m:
            conv_ecoul_rk = np.roll(conv_ecoul, k, axis=1)
            lef_m_ec = lem.copy() if typo == "STOCK" else gf.sum_k(lem.copy(), k)
            lef_m_ec[:, :p] = 0
            if k == 0 and typo == "STOCK" and simul == "EVE":
                dic_gp_liq_m["0_NC"] = lef_m_ec.copy()
            lef_m_ec[:, p:] = np.where(is_there_conv, conv_ecoul_rk[:, k:] * lef_m_ec[:, p:p + 1], lef_m_ec[:, p:])
            dic_gp_liq_m[k] = lef_m_ec

    def get_encours(self, dic_sci):
        lef = np.array(dic_sci[gp.ef_sti])
        lem = np.array(dic_sci[gp.em_sti])
        tef = np.array(dic_sci["tef"])
        tem = np.array(dic_sci["tem"])

        return lef, lem, tef, tem

    def get_tab_fix_st(self, dic_stock_scr, typo, data=[]):
        if typo == "PN":
            ht = data.shape[0]
            lg = self.up.nb_mois_proj_usr
            if ("tx_fix_f") in dic_stock_scr:
                tab_fix_f = dic_stock_scr["tx_fix_f"].copy()
                tab_fix_m = dic_stock_scr["tx_fix_m"].copy()
                df = pd.DataFrame(index=data.index, columns=["TX_FIX_F"])

                tab_fix_f = df.join(tab_fix_f, on="new_key", how="left")[[x for x in tab_fix_f.columns]].copy()
                tab_fix_f = np.array(tab_fix_f.astype(np.float64).fillna(1)).reshape(ht, lg + 1)

                tab_fix_m = df.join(tab_fix_m, on="new_key", how="left")[[x for x in tab_fix_m.columns]].copy()
                tab_fix_m = np.array(tab_fix_m.astype(np.float64).fillna(1)).reshape(ht, lg + 1)
            else:
                tab_fix_f = np.ones((ht, lg + 1))
                tab_fix_m = np.ones((ht, lg + 1))
                # tab_fix_f[:, 0] = 1
                # tab_fix_m[:, 0] = 1
        else:
            tab_fix_f = np.array(dic_stock_scr["tx_fix_f"])
            tab_fix_m = np.array(dic_stock_scr["tx_fix_m"])

        return tab_fix_f, tab_fix_m

    def calculate_gap_inflation(self, k, dic_gp_liq_f, dic_gp_liq_m, dic_gp_inf_f, dic_gp_inf_m, dic_gp_f, dic_gp_m,
                                coeff_gp_inf,
                                filtre_inf, typo, simul):
        if "EVE" == simul:
            do_f = False
            do_m = (di.gap_inf_eve_em_st[k] or k == 0) if typo == "STOCK" else di.gap_inf_eve_em_pn[k]
        elif "EVE_LIQ" == simul:
            do_f = False
            do_m = di.gap_inf_eve_em_st[k] if typo == "STOCK" else di.gap_inf_eve_em_pn[k]
        else:
            do_f = di.gap_inf_ef_st[k] if typo == "STOCK" else di.gap_inf_ef_pn[k]
            do_m = di.gap_inf_em_st[k] if typo == "STOCK" else di.gap_inf_em_pn[k]

        if do_f:
            lef_k_ec = dic_gp_liq_f[k]
            dic_gp_inf_f[k] = np.where(filtre_inf, lef_k_ec - dic_gp_f[k], 0)
            dic_gp_inf_f[k] = dic_gp_inf_f[k] + coeff_gp_inf * (lef_k_ec - dic_gp_f[k])
            if k == 0 and typo == "STOCK" and simul == "EVE":
                dic_gp_inf_f["0_NC"] = np.where(filtre_inf, dic_gp_liq_f["0_NC"] - dic_gp_f["0_NC"], 0)
                dic_gp_inf_f["0_NC"] = dic_gp_inf_f["0_NC"] + coeff_gp_inf * (dic_gp_liq_f["0_NC"] - dic_gp_f["0_NC"])

        if do_m:
            lem_k_ec = dic_gp_liq_m[k]
            dic_gp_inf_m[k] = np.where(filtre_inf, lem_k_ec - dic_gp_m[k], 0)
            dic_gp_inf_m[k] = dic_gp_inf_m[k] + coeff_gp_inf * (lem_k_ec - dic_gp_m[k])
            if k == 0 and typo == "STOCK" and simul == "EVE":
                dic_gp_inf_m["0_NC"] = np.where(filtre_inf, dic_gp_liq_m["0_NC"] - dic_gp_m["0_NC"], 0)
                dic_gp_inf_m["0_NC"] = dic_gp_inf_m["0_NC"] + coeff_gp_inf * (dic_gp_liq_m["0_NC"] - dic_gp_m["0_NC"])

    def adjust_gp_tf_tla(self, k, dic_gp_liq_f, dic_gp_liq_m, dic_gp_f, dic_gp_m, coeff_index_ntf, typo, simul):
        if "EVE" == simul:
            do_f = False
            do_m = (di.gap_tx_eve_em_st[k] or k == 0) if typo == "STOCK" else di.gap_tx_eve_em_pn[k]
        elif "EVE_LIQ" == simul:
            do_f = False
            do_m = di.gap_tx_eve_em_st[k] if typo == "STOCK" else di.gap_tx_eve_em_pn[k]
        else:
            do_f = di.gap_tx_ef_st[k] if typo == "STOCK" else di.gap_tx_ef_pn[k]
            do_m = di.gap_tx_em_st[k] if typo == "STOCK" else di.gap_tx_em_pn[k]

        if do_f:
            lef_k_ec = dic_gp_liq_f[k]
            dic_gp_f[k] = dic_gp_f[k] + coeff_index_ntf * (lef_k_ec - dic_gp_f[k])
            if k == 0 and typo == "STOCK" and simul == "EVE":
                dic_gp_f["0_NC"] = dic_gp_f["0_NC"] + coeff_index_ntf * (dic_gp_liq_f["0_NC"] - dic_gp_f["0_NC"])

        if do_m:
            lem_k_ec = dic_gp_liq_m[k]
            dic_gp_m[k] = dic_gp_m[k] + coeff_index_ntf * (lem_k_ec - dic_gp_m[k])
            if k == 0 and typo == "STOCK" and simul == "EVE":
                dic_gp_m["0_NC"] = dic_gp_m["0_NC"] + coeff_index_ntf * (dic_gp_liq_m["0_NC"] - dic_gp_m["0_NC"])

    def calculate_gap_reg(self, k, dic_gp_rg_f, dic_gp_rg_m, dic_gp_f, dic_gp_inf_f, dic_gp_m, dic_gp_inf_m, typo,
                          simul):
        if "EVE" == simul:
            do_f = False
            do_m = (di.gap_reg_eve_em_st[k] or k == 0) if typo == "STOCK" else di.gap_reg_eve_em_pn[k]
        elif "EVE_LIQ" == simul:
            do_f = False
            do_m = di.gap_reg_eve_em_st[k] if typo == "STOCK" else di.gap_reg_eve_em_pn[k]
        else:
            do_f = di.gap_reg_ef_st[k] if typo == "STOCK" else di.gap_reg_ef_pn[k]
            do_m = di.gap_reg_em_st[k] if typo == "STOCK" else di.gap_reg_em_pn[k]

        if do_f:
            dic_gp_rg_f[k] = (self.mp.mapping_gp_reg_params["coeff_reg_tf_usr"] * dic_gp_f[k]
                              + self.mp.mapping_gp_reg_params["coeff_reg_inf_usr"] * dic_gp_inf_f[k])
            if k == 0 and typo == "STOCK" and simul == "EVE":
                dic_gp_rg_f["0_NC"] = (self.mp.mapping_gp_reg_params["coeff_reg_tf_usr"] * dic_gp_f["0_NC"]
                                       + self.mp.mapping_gp_reg_params["coeff_reg_inf_usr"] * dic_gp_inf_f["0_NC"])

        if do_m:
            dic_gp_rg_m[k] = (self.mp.mapping_gp_reg_params["coeff_reg_tf_usr"] * dic_gp_m[k]
                              + self.mp.mapping_gp_reg_params["coeff_reg_inf_usr"] * dic_gp_inf_m[k])
            if k == 0 and typo == "STOCK" and simul == "EVE":
                dic_gp_rg_m["0_NC"] = (self.mp.mapping_gp_reg_params["coeff_reg_tf_usr"] * dic_gp_m["0_NC"]
                                       + self.mp.mapping_gp_reg_params["coeff_reg_inf_usr"] * dic_gp_inf_m["0_NC"])

    @staticmethod
    def get_coeff_gp_inf(mp, data, rate_code_col):
        ht = data.shape[0]
        index_calc = np.array(data[rate_code_col].values)

        filtre_cel = (np.isin(index_calc, mp["cel_indexes"]))
        filtre_tla_u = (np.isin(index_calc, mp["tla_indexes"]) | np.isin(index_calc, mp["lep_indexes"]))
        filtre_tlb = (np.isin(index_calc, mp["tlb_indexes"]))

        filtre_index1 = (filtre_cel).reshape(ht, 1)
        filtre_index2 = (filtre_tla_u).reshape(ht, 1)
        filtre_index3 = (filtre_tlb).reshape(ht, 1)
        filtre_index4 = ((~filtre_index1) & (~filtre_index2) & (~filtre_index3)).reshape(ht, 1)
        filtres_index = [filtre_index1, filtre_index2, filtre_index3, filtre_index4]

        choices_coeff_gp_inf = [mp["coeff_inf_cel_usr"], mp["coeff_inf_tla_usr"], mp["coeff_inf_tlb_usr"], 0]

        return np.select(filtres_index, choices_coeff_gp_inf), (np.isin(index_calc, mp["inf_indexes"])).reshape(ht, 1)

    @staticmethod
    def get_coeff_gp_tf(mp, data, rate_code_col):
        ht = data.shape[0]
        index_calc = np.array(data[rate_code_col].values)
        filtre_cel = (np.isin(index_calc, mp["cel_indexes"]))
        filtre_tla_u = (np.isin(index_calc, mp["tla_indexes"]) | np.isin(index_calc, mp["lep_indexes"]))
        filtre_tlb = (np.isin(index_calc, mp["tlb_indexes"]))

        filtre_index1 = (filtre_cel).reshape(ht, 1)
        filtre_index2 = (filtre_tla_u).reshape(ht, 1)
        filtre_index3 = (filtre_tlb).reshape(ht, 1)
        filtre_index4 = ((~filtre_index1) & (~filtre_index2) & (~filtre_index3)).reshape(ht, 1)
        filtres_index = [filtre_index1, filtre_index2, filtre_index3, filtre_index4]

        choices_coeff_gp_tf = [mp["coeff_tf_cel_usr"], mp["coeff_tf_tla_usr"], mp["coeff_tf_tlb_usr"], 0]

        return np.select(filtres_index, choices_coeff_gp_tf)

    def get_filtres_tla(self, data, typo):
        ht = data.shape[0]

        if typo == "STOCK":
            contrats_map = self.mp.contrats_map.copy()[[gp.nc_cp_isech]].copy()
            is_ech_stock = \
                data[[gp.nc_output_devise_cle]].copy().join(contrats_map, how="left", on=gp.nc_output_contrat_cle)[
                    gp.nc_cp_isech].copy()

            filtre_tla = (np.array(is_ech_stock) == False) & (
                np.array(data[gp.nc_output_index_calc_cle].isin(self.mp.mapping_gp_reg_params["all_gap_gestion_index"]))) & np.array(
                [self.up.retraitement_tla])

        else:
            filtre_tla = (np.array(data[gp.nc_output_index_calc_cle].isin(self.mp.mapping_gp_reg_params["all_gap_gestion_index"]))) \
                         & np.array([self.up.retraitement_tla])

        return np.array(filtre_tla).reshape(ht, 1)

    def get_conv_ecoulement(self, data, typo, simul):
        ht = data.shape[0]
        cle_jointure = [gp.nc_output_contrat_cle]
        if typo == "STOCK":
            data_join = data.reset_index(level=[gp.nc_output_palier_cle, gp.nc_output_contrat_cle])[cle_jointure].copy()
        else:
            data_join = data[cle_jointure].copy()

        data_join["main_key"] = data_join[cle_jointure].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        conv_ecoul = data_join.join(self.mp.conv_gps_nmd, on="main_key")[
            ["M" + str(i) for i in range(0, self.up.nb_mois_proj_usr + 1)]].copy()
        conv_ecoul = np.array(conv_ecoul).astype(np.float64)
        is_there_conv = ((~np.isnan(conv_ecoul[:, 0])) & \
                         np.array(([
                             (self.up.force_gps_nmd and simul == "LIQ") or (
                                         self.mp.mapping_eve["force_gps_nmd_eve"] and "EVE" in simul)]))).reshape(
            ht, 1)

        return conv_ecoul, is_there_conv

    def format_gp_tx(self, k, data, dic_sci, dic_gp_f, dic_gp_m, dic_gp_inf_f, dic_gp_inf_m, dic_gp_rg_f, dic_gp_rg_m,
                     typo,
                     simul):
        gp_tx_ef = gp.gp_ef_sti if typo == "STOCK" else gp.gp_ef_pni
        gp_tx_em = (gp.gp_em_sti if simul == "LIQ" else gp.gp_em_eve_sti) if typo == "STOCK" else (
            gp.gp_em_pni if simul == "LIQ" else gp.gp_em_eve_pni)
        gp_inf_ef = gp.gpi_ef_sti if typo == "STOCK" else gp.gpi_ef_pni
        gp_inf_em = (gp.gpi_em_sti if simul == "LIQ" else gp.gpi_em_eve_sti) if typo == "STOCK" else (
            gp.gpi_em_pni if simul == "LIQ" else gp.gpi_em_eve_pni)
        gp_rg_ef = gp.gpr_ef_sti if typo == "STOCK" else gp.gpr_ef_pni
        gp_rg_em = (gp.gpr_em_sti if simul == "LIQ" else gp.gpr_em_eve_sti) if typo == "STOCK" else (
            gp.gpr_em_pni if simul == "LIQ" else gp.gpr_em_eve_pni)
        deb = 0 if typo == "STOCK" else 1
        for dic_gp, namo in [(dic_gp_f, gp_tx_ef), (dic_gp_m, gp_tx_em), (dic_gp_inf_f, gp_inf_ef),
                             (dic_gp_inf_m, gp_inf_em), \
                             (dic_gp_rg_f, gp_rg_ef), (dic_gp_rg_m, gp_rg_em)]:
            if k in dic_gp:
                dic_gp[k] = pd.DataFrame(data=dic_gp[k], index=data.index.get_level_values("new_key"))
                dic_gp[k].columns = ["M" + str(j) for j in range(deb, self.up.nb_mois_proj_usr + 1)]
                dic_sci[namo + str(k)] = dic_gp[k]
                if k == 0 and typo == "STOCK" and simul == "EVE":
                    dic_sci[namo + str(k) + "_NC"] = dic_gp[str(k) + "_NC"]

    def calcul_gap_tx_mtx_pn(self, k, dic_gp_f, dic_gp_m, lef_fix_pn, lem_fix_pn, simul):
        do_f = di.gap_tx_ef_pn[k] if simul == "LIQ" else False
        do_m = di.gap_tx_em_pn[k] if simul == "LIQ" else di.gap_tx_eve_em_pn[k]

        if do_f:
            dic_gp_f[k] = gf.sum_k(lef_fix_pn[k], k)

        if do_m:
            dic_gp_m[k] = gf.sum_k(lem_fix_pn[k], k)

    def choose_gp_tx_pn(self, k, dic_gp_f, dic_gp_m, dic_gp_f_conv, dic_gp_m_conv, is_there_conv, simul):
        do_f = di.gap_tx_ef_pn[k] if simul == "LIQ" else False
        do_m = di.gap_tx_em_pn[k] if simul == "LIQ" else di.gap_tx_eve_em_pn[k]

        if do_f:
            dic_gp_f[k] = np.where(is_there_conv, dic_gp_f_conv[k], dic_gp_f[k])

        if do_m:
            dic_gp_m[k] = np.where(is_there_conv, dic_gp_m_conv[k], dic_gp_m[k])

    def roll_tab(self, tab, k, typo):
        tab = np.roll(tab, k)
        tab[:, :k] = 0
        if typo == "PN":
            tab = tab[:, 1:]
        return tab

    def change_fixing_tla(self, tab, k, filtre_tla, mois_refix_tla):
        """ TRAITEMENT TLA """
        tab[:, k:k + 1] = np.where(filtre_tla, 1, tab[:, k:k + 1])
        tab[:, k + 1:mois_refix_tla] = np.where(filtre_tla, 1,
                                                tab[:, k + 1:mois_refix_tla])
        tab[:, mois_refix_tla:] = np.where(filtre_tla, 0, tab[:, mois_refix_tla:])

        return tab

    def calcul_gap_tx_ajust(self, dic_ajust, num_cols, simul):
        """ Fonction permettant de calculer les gaps de taux fixes pour ajustements """
        ind_em = gp.em_sti if type != "EVE" else gp.em_eve_sti
        ind_ef = gp.ef_sti if type != "EVE" else gp.ef_eve_sti
        """ Gap de taux fixe => égal à l'encours au mois de projection sinon 0 """
        for p in range(0, self.up.nb_mois_proj_out + 1):
            pref = "M0" if p <= 9 else "M"
            do_f = di.gap_tx_ef_aj[p] if simul == "LIQ" else False
            do_m = di.gap_tx_em_aj[p] if simul == "LIQ" else di.gap_tx_eve_em_aj[p]
            gp_tx_f = gp.gp_ef_pni
            gp_tx_m = gp.gp_em_pni if simul == "LIQ" else gp.gp_em_eve_pni
            if do_f:
                data_adj_gap = dic_ajust[ind_ef].copy()
                data_adj_gap[gp.nc_output_ind3] = gp_tx_f + str(p)
                dic_ajust[gp_tx_f + str(p)] = data_adj_gap
                cols = [x for x in num_cols if pref + str(p) != x]
                dic_ajust[gp_tx_f + str(p)][cols] = 0

            if do_m:
                data_adj_gap = dic_ajust[ind_em].copy()
                data_adj_gap[gp.nc_output_ind3] = gp_tx_m + str(p)
                dic_ajust[gp_tx_m + str(p)] = data_adj_gap
                cols = [x for x in num_cols if pref + str(p) != x]
                dic_ajust[gp_tx_m + str(p)][cols] = 0

        """ GAP INF => 0 """
        for p in range(0, self.up.nb_mois_proj_out + 1):
            do_f = di.gap_inf_ef_aj[p] if simul == "LIQ" else False
            do_m = di.gap_inf_em_aj[p] if simul == "LIQ" else di.gap_inf_eve_em_aj[p]
            gp_inf_f = gp.gpi_ef_pni
            gp_inf_m = gp.gpi_em_pni if simul == "LIQ" else gp.gpi_em_eve_pni
            if do_f:
                dic_ajust[gp_inf_f + str(p)] = dic_ajust[ind_ef].copy()
                dic_ajust[gp_inf_f + str(p)][gp.nc_output_ind3] = gp_inf_f + str(p)
                dic_ajust[gp_inf_f + str(p)][num_cols] = 0

            if do_m:
                dic_ajust[gp_inf_m + str(p)] = dic_ajust[ind_em].copy()
                dic_ajust[gp_inf_m + str(p)][gp.nc_output_ind3] = gp_inf_m + str(p)
                dic_ajust[gp_inf_m + str(p)][num_cols] = 0

        """ GAP TAUX REG """
        for p in range(0, self.up.nb_mois_proj_out + 1):
            pref = "M0" if p <= 9 else "M"
            do_f = di.gap_reg_ef_aj[p] if simul == "LIQ" else False
            do_m = di.gap_reg_em_aj[p] if simul == "LIQ" else di.gap_reg_eve_em_aj[p]
            gp_reg_f = gp.gpr_ef_pni
            gp_reg_m = gp.gpr_em_pni if simul == "LIQ" else gp.gpr_em_eve_pni
            gp_inf_f = gp.gpi_ef_pni
            gp_inf_m = gp.gpi_em_pni if simul == "LIQ" else gp.gpi_em_eve_pni
            gp_tx_f = gp.gp_ef_pni
            gp_tx_m = gp.gp_em_pni if simul == "LIQ" else gp.gp_em_eve_pni
            if do_f:
                cols = [x for x in num_cols if pref + str(p) != x]
                dic_ajust[gp_reg_f + str(p)] = dic_ajust[ind_ef].copy()
                dic_ajust[gp_reg_f + str(p)][gp.nc_output_ind3] = gp_reg_f + str(p)
                dic_ajust[gp_reg_f + str(p)][cols] = self.mp.mapping_gp_reg_params["coeff_reg_tf_usr"] * dic_ajust[
                    gp_tx_f + str(p)][cols] + self.mp.mapping_gp_reg_params["coeff_reg_inf_usr"] * dic_ajust[gp_inf_f + str(p)][cols]

            if do_m:
                cols = [x for x in num_cols if pref + str(p) != x]
                dic_ajust[gp_reg_m + str(p)] = dic_ajust[ind_em].copy()
                dic_ajust[gp_reg_m + str(p)][gp.nc_output_ind3] = gp.gpr_em_pni + str(p)
                dic_ajust[gp_reg_m + str(p)][cols] = self.mp.mapping_gp_reg_params["coeff_reg_tf_usr"] * dic_ajust[
                    gp_inf_m + str(p)][cols] + self.mp.mapping_gp_reg_params["coeff_reg_inf_usr"] * dic_ajust[gp_tx_m + str(p)][cols]
