from mappings.mappings_moteur import Mappings as mp
import modules.moteur.parameters.general_parameters as gp
import pandas as pd
import numpy as np
import utils.general_utils as gu
import utils.excel_utils as ex
import logging

np.seterr(divide='ignore', invalid='ignore')

logger = logging.getLogger(__name__)

liquidite_scope = "LIQUIDITE"
insurance_scope = "INSURANCE"
actif = "ACTIF"

class LCR_Calculator():
    
    def __init__(self, cls_usr):
        self.up = cls_usr

    def load_lcr_params(self, wb):    
        """ COEFFS LCR"""
        coeffs_lcr = mp.load_map(wb, name_range=gp.ng_param_lcr, cle_map=gp.param_lcr_cle, join_key=True)
        self.coeffs_lcr = coeffs_lcr[[gp.nc_nco_param_lcr, gp.nc_rl_param_lcr]].copy()
    
        """ COEFFS LCR SPEC"""
        coeffs_lcr_spec = mp.load_map(wb, name_range=gp.ng_param_lcr_spec, cle_map=gp.param_lcr_spec_cle, join_key=True)
        self.coeffs_lcr_spec = coeffs_lcr_spec[[gp.nc_nco_param_lcr_spec, gp.nc_rl_param_lcr_spec]].copy()
    
        """ LCR DAR """
        if gp.rl_pni in self.up.indic_sortie["PN"] or gp.nco_pni in self.up.indic_sortie["PN"] \
                or gp.rl_sti in self.up.indic_sortie["ST"] or gp.rl_sti in self.up.indic_sortie["ST"]:
            self.read_input_lcr_dar(wb)
    
        """ METHODE LCR"""
        bc_outflow_lcr = ex.get_dataframe_from_range(wb, gp.ng_param_lcr_meth, header=True, alert="")
        self.bc_outflow_lcr = bc_outflow_lcr[bc_outflow_lcr[gp.nc_outlfow_param_lcr_meth] == "OUI"][
            gp.nc_bilanc_cc_param_lcr_meth].tolist()
    
    def read_input_lcr_dar(self, wb):
        lcr_dar = ex.get_dataframe_from_range(wb, gp.ng_lcr_dar, header=True, alert="")
        if lcr_dar.shape[0] == 0:
            logger.warning("  LE LCR DAR EST ABSENT")
            self.nco_dar_outflow_t0_tot = 0
            self.nco_dar_inflow_t0_tot = 0
            self.rl_dar_t0_tot = 0
        else:
            lcr_dar = lcr_dar[[gp.nc_indic_lcr, "VAL"]].copy()
            self.nco_dar_outflow_t0_tot = lcr_dar[lcr_dar[gp.nc_indic_lcr] == gp.nr_nco_outflow_lcr_dar]["VAL"].iloc[0] * 10 ** 9
            self.nco_dar_inflow_t0_tot = lcr_dar[lcr_dar[gp.nc_indic_lcr] == gp.nr_nco_inflow_lcr_dar]["VAL"].iloc[0] * 10 ** 9
            self.rl_dar_t0_tot = lcr_dar[lcr_dar[gp.nc_indic_lcr] == gp.nr_rl_lcr_dar]["VAL"].iloc[0] * 10 ** 9
    
    
    def calc_err_em_tot(self, data, dic_indic, other_data):
        nco_indic = gp.nco_sti
        rl_indic = gp.rl_sti
        indic_sortie = self.up.indic_sortie["ST"]
    
        if nco_indic in indic_sortie or rl_indic in indic_sortie:
            em_indic = gp.em_sti
            outfl_indic = gp.outf_sti
            ht = dic_indic[em_indic].shape[0]
    
            nco_coeff, rl_coeff = self.load_nco_rl_coeff(data, other_data, ht)
            coeff_bilan = np.where(other_data[gp.nc_output_bilan].str.contains(actif), -1, 1).reshape(ht, 1)
            hors_bilan = other_data[gp.nc_output_bilan].str.contains("HB")
            not_hb = np.where(hors_bilan, 0, 1).reshape(ht, 1)
            coeff_pass = np.where(coeff_bilan == 1, 1, 0)
            coeff_act = 1 - coeff_pass
            em0 = np.array(dic_indic[em_indic]["M0"]).reshape(ht, 1)
            self.em_total_pass_t0 = (em0 * not_hb * coeff_pass).sum()
            self.em_total_act_t0 = (em0 * not_hb * coeff_act).sum()
    
        if nco_indic in self.up.indic_sortie["ST"]:
            filter_outflow = other_data[gp.nc_output_bc].isin(self.bc_outflow_lcr)
            filter_outflow = np.array(filter_outflow).reshape(ht, 1)
            nco_outflow_t0 = - coeff_bilan * em0 * nco_coeff * coeff_pass * not_hb
            nco_outflow_t0_real = - np.array(dic_indic[outfl_indic + " 0M-1M"]["M0"]).reshape(ht, 1)
    
            self.nco_outflow_t0 = np.where(filter_outflow.reshape(ht, 1), nco_outflow_t0_real, nco_outflow_t0)
            self.nco_outflow_t0_tot = (nco_outflow_t0).sum()
    
            self.nco_inflow_t0 = - coeff_bilan * em0 * nco_coeff * coeff_act * not_hb
            self.nco_inflow_t0_tot = (self.nco_inflow_t0).sum()
    
            self.erreur_outflow_t0 = - nco_dar_outflow_t0_tot - self.nco_outflow_t0_tot
            self.erreur_inflow_t0 = nco_dar_inflow_t0_tot - self.nco_inflow_t0_tot
    
        if rl_indic in indic_sortie:
            self.rl_moteur_t0 = -coeff_bilan * em0 * rl_coeff * not_hb
            self.rl_moteur_t0_tot = (self.rl_moteur_t0).sum()
            self.erreur_rl_t0 = self.rl_dar_t0_tot - self.rl_moteur_t0_tot
    
    
    def calculate_nco_and_rl(self, dic_indic, coeff_bilan, indic_sortie, em_indic, nco_indic, rl_indic, delta_nco_indic, \
                             delta_rl_indic, hors_bilan, outfl_indic, nco_coeff, rl_coeff, filter_outflow):

        ht = dic_indic[em_indic].shape[0]
    
        coeff_bilan = coeff_bilan.reshape(ht, 1)
        not_hb = np.where(hors_bilan, 0, 1).reshape(ht, 1)
        coeff_pass = np.where(coeff_bilan == 1, 1, 0).reshape(ht, 1)
        coeff_act = 1 - coeff_pass
        em0 = np.array(dic_indic[em_indic]["M0"]).reshape(ht, 1)
    
        if nco_indic in indic_sortie:
            self.nco_moteur_outflow_t0 = - coeff_bilan * em0 * nco_coeff * coeff_pass * not_hb
            nco_moteur_outflow_t0_empreinte = - np.array(dic_indic[outfl_indic + " 0M-1M"]["M0"]).reshape(ht, 1)
            self.nco_moteur_outflow_t0 = np.where(filter_outflow.reshape(ht, 1), nco_moteur_outflow_t0_empreinte,
                                             self.nco_moteur_outflow_t0)
            self.nco_moteur_inflow_t0 = - coeff_bilan * em0 * nco_coeff * coeff_act * not_hb
            nco_t0 = (self.nco_moteur_outflow_t0 + self.erreur_outflow_t0 * em0 / self.em_total_pass_t0) * coeff_pass * not_hb
            nco_t0 = (nco_t0 + (self.nco_moteur_inflow_t0 + self.erreur_inflow_t0 * em0 / self.em_total_act_t0) * coeff_act) * not_hb
            dic_indic[nco_indic] = np.array(nco_t0).reshape(ht, 1) + dic_indic[delta_nco_indic]
    
        if rl_indic in indic_sortie:
            rl_moteur_t0 = -coeff_bilan * em0 * rl_coeff * not_hb
            rl_t0 = rl_moteur_t0 + (self.erreur_rl_t0 * em0 / self.em_total_act_t0 * coeff_act * not_hb)
            dic_indic[rl_indic] = np.array(rl_t0).reshape(ht, 1) + dic_indic[delta_rl_indic]
    
    
    def load_nco_rl_coeff(self, data, other_data, ht, is_ref=True, typo="ST"):
        cols_utils = [gp.nc_nco_param_lcr, gp.nc_rl_param_lcr]
        cols_utils_main = [gp.nc_output_contrat_cle, gp.nc_output_marche_cle]
    
        if typo == "ST":
            data = data.reset_index(level=[gp.nc_output_contrat_cle, gp.nc_output_palier_cle])[cols_utils_main].copy()
        else:
            data = data[cols_utils_main].copy()
    
        data_map = pd.concat([other_data[[gp.nc_output_bilan]].copy(), \
                              data[[gp.nc_output_contrat_cle, gp.nc_output_marche_cle]].copy()], axis=1)
    
        cles_a_combiner = [gp.nc_output_bilan, gp.nc_output_contrat_cle, gp.nc_output_marche_cle]
    
        param_lcr = self.coeffs_lcr if is_ref else scm.coeffs_lcr
        param_lcr_spec = self.coeffs_lcr_spec if is_ref else scm.coeffs_lcr_spec
    
        lcr_coeffs = gu.map_with_combined_key(data_map, param_lcr, cles_a_combiner, symbol_any="*", \
                                              no_map_value=0, filter_comb=True, necessary_cols=2)
    
        lcr_coeffs = gu.map_with_combined_key(lcr_coeffs, param_lcr_spec, cles_a_combiner, symbol_any="*", \
                                              override=True, filter_comb=True, necessary_cols=2)[cols_utils].copy()
    
        nco_coeff = np.minimum(1, np.absolute(np.array(lcr_coeffs[gp.nc_nco_param_lcr]).reshape(ht, 1)))
        rl_coeff = np.minimum(1, np.array(lcr_coeffs[gp.nc_rl_param_lcr]).reshape(ht, 1))
    
        return nco_coeff, rl_coeff
    
    
    def calculate_lcr(self, main_data_init, other_data_init=[], dic_indic={}, typo=""):
        em_indic = gp.em_sti if typo == "ST" else gp.em_pni
        delta_rl_indic = gp.delta_rl_sti if typo == "ST" else gp.delta_rl_pni
        delta_em_indic = gp.delta_em_sti if typo == "ST" else gp.delta_em_pni
        delta_nco_indic = gp.delta_nco_sti if typo == "ST" else gp.delta_nco_pni
        nco_indic = gp.nco_sti if typo == "ST" else gp.nco_pni
        rl_indic = gp.rl_sti if typo == "ST" else gp.rl_pni
        outfl_indic = gp.outf_sti if typo == "ST" else gp.outf_pni
        indic_sortie = self.up.indic_sortie["ST"] if typo == "ST" else (
            self.up.indic_sortie["PN"] if typo == "PN" else self.up.indic_sortie["AJUST"])
    
        if delta_em_indic in indic_sortie or delta_rl_indic in indic_sortie or delta_nco_indic in indic_sortie \
                or nco_indic in indic_sortie or rl_indic in indic_sortie:
    
            if typo == "ST":
                cols_num = ["M" + str(i) for i in range(0, self.up.nb_mois_proj_usr + 1)]
            elif typo == "PN":
                cols_num = ["M" + str(i) for i in range(1, self.up.nb_mois_proj_usr + 1)]
            else:
                cols_num = ["M0" + str(i) if i <= 9 else "M" + str(i) for i in range(0, self.up.nb_mois_proj_usr + 1)]
    
            if typo != "AJUST":
                other_data = other_data_init.copy()
                main_data = main_data_init.copy()
                indic_m0 = "M0"
            else:
                indic_m0 = "M00"
                list_indic = [em_indic, outfl_indic + " 0M-1M"]
                cond_filtrage = (main_data_init[gp.nc_output_ind3].isin(list_indic))
                cond_em = (main_data_init[gp.nc_output_ind3] == em_indic)
                main_data = main_data_init[cond_em & cond_filtrage].copy()
                other_data = main_data.copy()
                dic_indic[em_indic] = main_data[cols_num].copy()
                dic_indic[outfl_indic + " 0M-1M"] = main_data_init[(~cond_em) & (cond_filtrage)][cols_num].copy()
    
            ht = dic_indic[em_indic].shape[0]
            coeff_bilan = np.where(other_data[gp.nc_output_bilan].str.contains(actif), -1, 1).reshape(ht, 1)
            hors_bilan = other_data[gp.nc_output_bilan].str.contains("HB")
    
            nco_coeff, rl_coeff = self.load_nco_rl_coeff(main_data, other_data, ht, is_ref=False, typo=typo)
    
            filter_outflow = other_data[gp.nc_output_bc].isin(scm.bc_outflow_lcr)
            filter_outflow = np.array(filter_outflow).reshape(ht, 1)
    
            em = dic_indic[em_indic]
    
            if delta_em_indic in indic_sortie or delta_rl_indic in indic_sortie or delta_nco_indic in indic_sortie \
                    or nco_indic in indic_sortie or rl_indic in indic_sortie:
    
                dic_indic[delta_em_indic] = em * 0
                if typo == "ST":
                    dic_indic[delta_em_indic][cols_num] = coeff_bilan * (np.array(em) - np.array(em[indic_m0]).reshape(ht, 1))
                else:
                    dic_indic[delta_em_indic][cols_num] = coeff_bilan * np.array(em)
    
                delta_em = dic_indic[delta_em_indic]
    
            if rl_indic in indic_sortie or delta_rl_indic in indic_sortie:
                dic_indic[delta_rl_indic] = - delta_em * (rl_coeff * np.where(hors_bilan, 0, 1).reshape(ht, 1))
    
            if nco_indic in indic_sortie or delta_nco_indic in indic_sortie:
                dic_indic[delta_nco_indic] = self.calc_delta_nco_indic(dic_indic, outfl_indic, em, delta_em, nco_coeff, \
                                                                  hors_bilan, filter_outflow, typo, ht)
    
            if nco_indic in indic_sortie or rl_indic in indic_sortie:
                if typo == "ST":
                    self.calculate_nco_and_rl(dic_indic, coeff_bilan, indic_sortie, em_indic, nco_indic, rl_indic, \
                                         delta_nco_indic, delta_rl_indic, hors_bilan, outfl_indic, nco_coeff, rl_coeff, \
                                         filter_outflow)
                else:
    
                    if nco_indic in indic_sortie:
                        dic_indic[nco_indic] = dic_indic[delta_nco_indic]
    
                    if rl_indic in indic_sortie:
                        dic_indic[rl_indic] = dic_indic[delta_rl_indic]
    
            if typo == "AJUST":
                data_depart = main_data_init[cond_em].copy()
                existing_indic = main_data_init[gp.nc_output_ind3].drop_duplicates().tolist()
                for ind in dic_indic:
                    if not ind in list_indic and not ind in existing_indic:
                        data_add = data_depart.copy()
                        data_add[cols_num] = dic_indic[ind].values
                        data_add[indic_m0] = 0
                        data_add[gp.nc_output_ind3] = ind
                        main_data_init = pd.concat([main_data_init, data_add], axis=0, ignore_index=True)
    
        if typo == "AJUST":
            return main_data_init
    
    
    def calc_delta_nco_indic(self, dic_indic, outfl_indic, em, delta_em, nco_coeff, hors_bilan, filter_outflow, typo, ht):
        delta_nco_empreinte = -np.array(dic_indic[outfl_indic + " 0M-1M"])
        if typo == "ST":
            delta_nco_empreinte = delta_nco_empreinte - delta_nco_empreinte[:, 0].reshape(ht, 1)
    
        delta_nco = - delta_em * (nco_coeff * np.where(hors_bilan, 0, 1).reshape(ht, 1))
    
        delta_nco = np.where(filter_outflow, delta_nco_empreinte, delta_nco)
    
        delta_nco = pd.DataFrame(delta_nco, index=em.index, columns=em.columns.tolist())
    
        return delta_nco
