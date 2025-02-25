# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:00:57 2020

@author: Hossayne
"""

import modules.moteur.parameters.general_parameters as gp
import modules.moteur.parameters.user_parameters as up
import utils.excel_utils as ex
import logging
from modules.moteur.mappings.main_mappings import load_map
import os

logger = logging.getLogger(__name__)

global param_tx_ech, contrats_floor
global retraitement_tla, mois_refix_tla, conv_gpliq, contrats_fx_swaps, coeffs_nsfr, coeffs_lcr, coeffs_lcr_spec,\
    bc_outflow_nsfr, bc_outflow_lcr
global freq_refix_tla, conv_gps_nmd
global descemb_asf_st, descemb_rsf_st, descemb_asf_pn, descemb_rsf_pn



def load_sc_mappings(etab, scenario_name, scen_path, files_sc):
    load_profils_data(files_sc)
    load_lcr_nsfr_params(scen_path, etab, scenario_name)


def load_lcr_nsfr_params(scen_path, etab, scenario_name):
    global coeffs_nsfr, coeffs_lcr, coeffs_lcr_spec, bc_outflow_nsfr, bc_outflow_lcr, descemb_asf_st, descemb_rsf_st
    global  descemb_asf_pn, descemb_rsf_pn

    if len(up.scenarios_files[etab][scenario_name]["LCR_NSFR"])>0:
        files_sc = up.scenarios_files[etab][scenario_name]
        path_file_sc_prof = os.path.join(scen_path, files_sc["LCR_NSFR"][0])
        wb = ex.try_close_open(path_file_sc_prof, read_only=True)

        """ COEFFS NSFR"""
        coeffs_nsfr = load_map(wb, name_range=gp.ng_param_nsfr,
                               cle_map=[gp.nc_asf_rsf_param_nsfr, gp.nc_nsfr_dim1_param_nsfr])

        """ METHODE NSFR"""
        bc_outflow_nsfr = ex.get_dataframe_from_range(wb, gp.ng_param_nsfr_meth, header=True, alert="")
        bc_outflow_nsfr = bc_outflow_nsfr[bc_outflow_nsfr[gp.nc_outlfow_param_nsfr_meth] == "OUI"][
            gp.nc_bilanc_cc_param_nsfr_meth].tolist()

        """ COEFFS LCR"""
        coeffs_lcr = load_map(wb, name_range=gp.ng_param_lcr, cle_map=gp.param_lcr_cle, join_key=True)
        coeffs_lcr = coeffs_lcr[[gp.nc_nco_param_lcr, gp.nc_rl_param_lcr]].copy()

        """ COEFFS LCR SPEC"""
        coeffs_lcr_spec = load_map(wb, name_range=gp.ng_param_lcr_spec, cle_map=gp.param_lcr_spec_cle, join_key=True)
        coeffs_lcr_spec = coeffs_lcr_spec[[gp.nc_nco_param_lcr_spec, gp.nc_rl_param_lcr_spec]].copy()

        """ METHODE LCR"""
        bc_outflow_lcr = ex.get_dataframe_from_range(wb, gp.ng_param_lcr_meth, header=True, alert="")
        bc_outflow_lcr = bc_outflow_lcr[bc_outflow_lcr[gp.nc_outlfow_param_lcr_meth] == "OUI"][
            gp.nc_bilanc_cc_param_lcr_meth].tolist()

        """ DESECOMBREMENTS NSFR"""
        desemc_nsfr = ex.get_dataframe_from_range(wb, gp.ng_param_nsfr_desemc, header=True, alert="")
        cols_num = ["M" + str(i) for i in range(0, up.nb_mois_proj_usr + 1)]
        descemb = desemc_nsfr[["TYPE", gp.nc_output_key, gp.nc_output_ind3] + cols_num].copy()
        filter_asf = descemb[gp.nc_output_ind3].str.upper().str.contains("ASF")
        filter_rsf = descemb[gp.nc_output_ind3].str.upper().str.contains("RSF")
        filter_st = descemb["TYPE"].str.upper() == "ST"
        descemb_asf_st = descemb[(filter_asf) & (filter_st)].copy()
        descemb_rsf_st = descemb[(filter_rsf) & (filter_st)].copy()
        descemb_asf_pn = descemb[(filter_asf) & (~filter_st)].copy()
        descemb_rsf_pn = descemb[(filter_rsf) & (~filter_st)].copy()

        wb.Close(False)


def load_profils_data(files_sc):
    global contrats_floor

    """ ECH CONTRATS FLOOR"""
    path_file_sc_modele = files_sc["MODELE_ECH"]
    wb = ex.try_close_open(path_file_sc_modele, read_only=True)
    contrats_floor = load_map(wb, name_range=gp.ng_contrat_floor, cle_map=gp.nc_contrat_floor_cle)

    wb.Close(False)




