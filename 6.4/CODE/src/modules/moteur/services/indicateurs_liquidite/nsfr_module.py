import modules.moteur.mappings.sc_mappings as scm
import modules.moteur.parameters.general_parameters as gp
from modules.moteur.mappings.main_mappings import load_map
import numpy as np
import pandas as pd
import modules.moteur.parameters.user_parameters as up
import utils.excel_utils as ex
import modules.moteur.mappings.main_mappings as mp

liquidite_scope = "LIQUIDITE"
insurance_scope = "INSURANCE"
actif = "ACTIF"
nc_asf_rsf = "ASF/RSF"

global asf_rsf_dar, asf_resf_comp
global coeffs_nsfr, bc_outflow_nsfr


def load_nsfr_params(wb):
    global coeffs_nsfr, bc_outflow_nsfr
    """ COEFFS NSFR"""
    coeffs_nsfr = load_map(wb, name_range=gp.ng_param_nsfr,
                           cle_map=[gp.nc_asf_rsf_param_nsfr, gp.nc_nsfr_dim1_param_nsfr])

    """ NSFR DAR """
    if gp.asf_pni in up.indic_sortie["PN"] or gp.rsf_pni in up.indic_sortie["PN"] \
            or gp.asf_sti in up.indic_sortie["ST"] or gp.rsf_sti in up.indic_sortie["ST"]:
        read_input_nsfr_dar(wb)

    """ METHODE NSFR"""
    bc_outflow_nsfr = ex.get_dataframe_from_range(wb, gp.ng_param_nsfr_meth, header=True, alert="")
    bc_outflow_nsfr = bc_outflow_nsfr[bc_outflow_nsfr[gp.nc_outlfow_param_nsfr_meth] == "OUI"][
        gp.nc_bilanc_cc_param_nsfr_meth].tolist()

def read_input_nsfr_dar(wb):
    global asf_rsf_dar
    asf_rsf_dar = ex.get_dataframe_from_range(wb, gp.ng_nsfr_dar, header=True, alert="")
    asf_rsf_dar = asf_rsf_dar[[gp.nc_asf_rsf_nsfr_dar, gp.nc_dim_nsfr1_nsfr_dar, gp.nc_val_nsfr_dar]].copy()
    asf_rsf_dar.set_index([gp.nc_asf_rsf_nsfr_dar, gp.nc_dim_nsfr1_nsfr_dar], inplace=True)


def calc_err_em_tot(dic_indic, other_data):
    global asf_resf_comp

    indic_sortie = up.indic_sortie["ST"]
    asf_indic = gp.asf_sti
    rsf_indic = gp.rsf_sti

    if asf_indic in indic_sortie or rsf_indic in indic_sortie:
        em_indic = gp.em_sti
        outfl_indic = gp.outf_sti
        ht = dic_indic[em_indic].shape[0]

        main_coeff, outflow_coeffs, nsfr_coeff = get_nsfr_coeffs(other_data, ht, is_ref=True, typo="ST")

        coeff_bilan = np.where(other_data[gp.nc_output_bilan].str.contains(actif), -1, 1).reshape(ht, 1)
        hors_bilan = other_data[gp.nc_output_bilan].str.contains("HB")
        not_hb = np.where(hors_bilan, 0, 1).reshape(ht, 1)
        em0 = np.array(dic_indic[em_indic]["M0"]).reshape(ht, 1)

        filter_outflow = np.array((other_data[gp.nc_output_bc].isin(mp.bc_outflow_nsfr))).reshape(ht, 1)
        asf_rsf_moteur_t0 = nsfr_coeff.copy()
        asf_rsf_moteur_t0["NSFR_CALC"] = coeff_bilan * em0  * main_coeff * not_hb

        asf_rsf_moteur_t0_empreinte = (
        (np.array(dic_indic[outfl_indic + " 0M-6M"]["M0"] * outflow_coeffs[:, 0].reshape(ht)) \
         + np.array(dic_indic[outfl_indic + " 6M-12M"]["M0"] * outflow_coeffs[:, 1].reshape(ht)) \
         + np.array(dic_indic[outfl_indic + " 12M-inf"]["M0"] * outflow_coeffs[:, 2].reshape(ht))))

        asf_rsf_moteur_t0["NSFR_CALC"] = np.where(filter_outflow.reshape(ht), asf_rsf_moteur_t0_empreinte,
                                                  asf_rsf_moteur_t0["NSFR_CALC"])

        asf_rsf_moteur_t0["EM M0"] = em0
        asf_rsf_moteur_t0 = asf_rsf_moteur_t0[[nc_asf_rsf, gp.nc_output_nsfr1, "EM M0", "NSFR_CALC"]]
        asf_rsf_moteur_t0_tot = asf_rsf_moteur_t0.groupby([nc_asf_rsf, gp.nc_output_nsfr1], as_index=False).sum()

        asf_resf_comp = asf_rsf_moteur_t0_tot.join(asf_rsf_dar, on=[nc_asf_rsf, gp.nc_output_nsfr1]).copy()
        asf_resf_comp["ERR"] = asf_resf_comp[gp.nc_val_nsfr_dar] - asf_resf_comp["NSFR_CALC"]
        asf_resf_comp["EM M0 TOT"] = asf_resf_comp["EM M0"]
        asf_resf_comp = asf_resf_comp.set_index([nc_asf_rsf, gp.nc_output_nsfr1])
        asf_resf_comp = asf_resf_comp[["EM M0 TOT", "ERR"]].copy().fillna(0)


def get_nsfr_coeffs(other_data, ht, is_ref=True, typo="ST"):
    param_nsfr = mp.coeffs_nsfr if is_ref else scm.coeffs_nsfr
    coeff_bilan = np.where(other_data[gp.nc_output_bilan].str.contains(actif), -1, 1).reshape(ht, 1)
    other_data[nc_asf_rsf] = np.where(coeff_bilan == -1, "RSF", "ASF")

    cols_utils = [nc_asf_rsf, gp.nc_output_nsfr1, gp.nc_nsfr_coeff_param_nsfr, gp.nc_nsfr_coeff0_6_param_nsfr,
                  gp.nc_nsfr_coeff6_12_param_nsfr, gp.nc_nsfr_coeff12_inf_param_nsfr]
    cols_utils2 = [nc_asf_rsf, gp.nc_output_nsfr1, gp.nc_nsfr_coeff_param_nsfr]
    nsfr_coeffs = other_data.join(param_nsfr, on=[nc_asf_rsf, gp.nc_output_nsfr1])[cols_utils].copy().fillna(0)
    main_coeff = np.array(nsfr_coeffs[gp.nc_nsfr_coeff_param_nsfr].fillna(0)).reshape(ht, 1)
    outflow_coeffs = np.array(nsfr_coeffs[[gp.nc_nsfr_coeff0_6_param_nsfr, \
                                           gp.nc_nsfr_coeff6_12_param_nsfr, \
                                           gp.nc_nsfr_coeff12_inf_param_nsfr]].fillna(0))

    nsfr_coeff = nsfr_coeffs[cols_utils2].copy()

    return main_coeff, outflow_coeffs, nsfr_coeff


def calculate_nsfr(main_data_init, other_data_init=[], dic_indic={}, typo=""):
    em_indic = gp.em_sti if typo == "ST" else gp.em_pni
    delta_asf_indic = gp.delta_asf_sti if typo == "ST" else gp.delta_asf_pni
    delta_em_indic = gp.delta_em_sti if typo == "ST" else gp.delta_em_pni
    delta_rsf_indic = gp.delta_rsf_sti if typo == "ST" else gp.delta_rsf_pni
    rsf_indic = gp.rsf_sti if typo == "ST" else gp.rsf_pni
    asf_indic = gp.asf_sti if typo == "ST" else gp.asf_pni
    outfl_indic = gp.outf_sti if typo == "ST" else gp.outf_pni
    indic_sortie = up.indic_sortie["ST"] if typo == "ST" else (
        up.indic_sortie["PN"] if typo == "PN" else up.indic_sortie["AJUST"])

    if delta_em_indic in indic_sortie or delta_rsf_indic in indic_sortie \
            or delta_asf_indic in indic_sortie or asf_indic in indic_sortie or rsf_indic in indic_sortie:

        if typo == "ST":
            cols_num = ["M" + str(i) for i in range(0, up.nb_mois_proj_usr + 1)]
        elif typo == "PN":
            cols_num = ["M" + str(i) for i in range(1, up.nb_mois_proj_usr + 1)]
        else:
            cols_num = ["M0" + str(i) if i <= 9 else "M" + str(i) for i in range(0, up.nb_mois_proj_usr + 1)]

        if typo != "AJUST":
            other_data = other_data_init.copy()
            main_data = main_data_init.copy()
            indic_m0 = "M0"
        else:
            indic_m0 = "M00"
            outflow_ind = [outfl_indic + " 0M-6M", outfl_indic + " 6M-12M", outfl_indic + " 12M-inf"]
            list_indic = [em_indic] + outflow_ind
            cond_filtrage = (main_data_init[gp.nc_output_ind3].isin(list_indic))
            cond_em = (main_data_init[gp.nc_output_ind3] == em_indic)
            main_data = main_data_init[cond_em & cond_filtrage].copy()
            other_data = main_data.copy()
            dic_indic[em_indic] = main_data[cols_num].copy()
            for ind in outflow_ind:
                cond_outflow = (main_data_init[gp.nc_output_ind3] == ind)
                dic_indic[ind] = main_data_init[(cond_outflow) & (cond_filtrage)][cols_num].copy()

        filter_scope = (main_data[gp.nc_output_scope_cle].str.contains(liquidite_scope)) | \
                       (~main_data[gp.nc_output_scope_cle].str.contains(insurance_scope))

        filter_bilan = other_data[gp.nc_output_bilan].str.contains(actif)
        ht = dic_indic[em_indic].shape[0]
        filter_outflow = np.array((other_data[gp.nc_output_bc].isin(scm.bc_outflow_nsfr))).reshape(ht, 1)
        coeff_bilan = np.where(other_data[gp.nc_output_bilan].str.contains(actif), -1, 1).reshape(ht, 1)
        hors_bilan = other_data[gp.nc_output_bilan].str.contains("HB")

        main_coeff, outflow_coeffs, nsfr_coeff = get_nsfr_coeffs(other_data, ht, is_ref=False, typo=typo)
        em = dic_indic[em_indic].copy()

        if delta_em_indic in indic_sortie or delta_rsf_indic in indic_sortie \
                or delta_asf_indic in indic_sortie or asf_indic in indic_sortie or rsf_indic in indic_sortie:

            dic_indic[delta_em_indic] = em * 0

            if typo == "ST":
                dic_indic[delta_em_indic][cols_num] = coeff_bilan * (
                            np.array(em) - np.array(em[indic_m0]).reshape(ht, 1))
            else:
                dic_indic[delta_em_indic][cols_num] = coeff_bilan * np.array(em)

            delta_em = dic_indic[delta_em_indic].copy()

        if delta_rsf_indic in indic_sortie or delta_asf_indic in indic_sortie \
                or asf_indic in indic_sortie or rsf_indic in indic_sortie:

            if typo!="AJUST":
                descemb_asf = scm.descemb_asf_st if typo=="ST" else scm.descemb_asf_pn
                descemb_rsf = scm.descemb_rsf_st if typo=="ST" else scm.descemb_rsf_pn
                descemb_asf = descemb_asf.merge(main_data.reset_index()[["new_key"]], left_on=gp.nc_output_key, right_on="new_key", how="right")
                descemb_rsf = descemb_rsf.merge(main_data.reset_index()[["new_key"]], left_on=gp.nc_output_key, right_on="new_key", how="right")
                descemb_asf = descemb_asf[cols_num].fillna(0).values
                descemb_rsf = descemb_rsf[cols_num].fillna(0).values

            nsfr_empreinte_marche = \
                ((np.array(dic_indic[outfl_indic + " 0M-6M"] * outflow_coeffs[:, 0].reshape(ht, 1)) \
                  + np.array(dic_indic[outfl_indic + " 6M-12M"] * outflow_coeffs[:, 1].reshape(ht, 1)) \
                  + np.array(dic_indic[outfl_indic + " 12M-inf"] * outflow_coeffs[:, 2].reshape(ht, 1))))

            if typo == "ST":
                nsfr_empreinte_marche = nsfr_empreinte_marche - nsfr_empreinte_marche[:, 0].reshape(ht, 1)

            nfr_gen = np.array(delta_em * main_coeff * np.where(filter_scope, 1, 0).reshape(ht, 1))

            nfr_gen = np.where(filter_outflow, nsfr_empreinte_marche, nfr_gen)

            nfr_gen = pd.DataFrame(nfr_gen, index=delta_em.index, columns=delta_em.columns.tolist())

        if delta_asf_indic in indic_sortie or asf_indic in indic_sortie:
            dic_indic[delta_asf_indic] = nfr_gen * np.where(filter_bilan, 0, 1).reshape(ht, 1)
            if typo!="AJUST":
                dic_indic[delta_asf_indic] = dic_indic[delta_asf_indic] + descemb_asf

        if delta_rsf_indic in indic_sortie or rsf_indic in indic_sortie:
            dic_indic[delta_rsf_indic] = nfr_gen * np.where(filter_bilan, 1, 0).reshape(ht, 1)
            if typo!="AJUST":
                dic_indic[delta_rsf_indic] = dic_indic[delta_rsf_indic] + descemb_rsf

        if rsf_indic in indic_sortie or asf_indic in indic_sortie:
            if typo == "ST":
                calculate_asf_rsf(dic_indic, nsfr_coeff.copy(), coeff_bilan.reshape(ht), indic_sortie, \
                                  em_indic, asf_indic, rsf_indic, delta_asf_indic, delta_rsf_indic, hors_bilan, \
                                  outflow_coeffs, outfl_indic, filter_outflow)
            else:
                if rsf_indic in indic_sortie:
                    dic_indic[rsf_indic] = dic_indic[delta_rsf_indic]
                if asf_indic in indic_sortie:
                    dic_indic[asf_indic] = dic_indic[delta_asf_indic]

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


def calculate_asf_rsf(dic_indic, nsfr_coeff, coeff_bilan, indic_sortie, em_indic, asf_indic, rsf_indic, \
                      delta_asf_indic, delta_rsf_indic, hors_bilan, outflow_coeffs, outfl_indic, filter_outflow):
    global asf_rsf_dar, asf_resf_comp

    if asf_indic in indic_sortie or rsf_indic in indic_sortie:
        ht = dic_indic[em_indic].shape[0]

        not_hb = np.where(hors_bilan, 0, 1)
        coeff_pass = np.where(coeff_bilan == 1, 1, 0)

        asf_rsf_moteur_t0 = nsfr_coeff.copy()
        asf_rsf_moteur_t0["NSFR_CALC"] = coeff_bilan * dic_indic[em_indic]["M0"] \
                                         * np.array(nsfr_coeff[gp.nc_nsfr_coeff_param_nsfr]) * not_hb

        asf_rsf_moteur_t0_empreinte = (
        (np.array(dic_indic[outfl_indic + " 0M-6M"]["M0"] * outflow_coeffs[:, 0].reshape(ht)) \
         + np.array(dic_indic[outfl_indic + " 6M-12M"]["M0"] * outflow_coeffs[:, 1].reshape(ht)) \
         + np.array(dic_indic[outfl_indic + " 12M-inf"]["M0"] * outflow_coeffs[:, 2].reshape(ht))))

        asf_rsf_moteur_t0["NSFR_CALC"] = np.where(filter_outflow.reshape(ht), asf_rsf_moteur_t0_empreinte,
                                                  asf_rsf_moteur_t0["NSFR_CALC"])

        asf_rsf_moteur_t0["EM M0"] = dic_indic[em_indic]["M0"]
        asf_rsf_moteur_t0 = asf_rsf_moteur_t0[[nc_asf_rsf, gp.nc_output_nsfr1, "EM M0", "NSFR_CALC"]]

        asf_rsf_moteur_t0 = asf_rsf_moteur_t0.join(asf_resf_comp, on=[nc_asf_rsf, gp.nc_output_nsfr1])
        asf_rsf_moteur_t0 = asf_rsf_moteur_t0["NSFR_CALC"] + (
                asf_rsf_moteur_t0["ERR"] * asf_rsf_moteur_t0["EM M0"] / asf_rsf_moteur_t0["EM M0 TOT"]) * not_hb

    if asf_indic in indic_sortie:
        dic_indic[asf_indic] = (np.array(asf_rsf_moteur_t0).reshape(ht, 1) + dic_indic[
            delta_asf_indic]) * coeff_pass.reshape(ht, 1)

    if rsf_indic in indic_sortie:
        dic_indic[rsf_indic] = (np.array(asf_rsf_moteur_t0).reshape(ht, 1) + dic_indic[delta_rsf_indic]) * (
                1 - coeff_pass.reshape(ht, 1))
