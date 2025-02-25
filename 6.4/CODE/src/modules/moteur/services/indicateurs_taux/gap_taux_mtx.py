import modules.moteur.parameters.general_parameters as gp
import modules.moteur.parameters.user_parameters as up
import modules.moteur.utils.generic_functions as gf
import numpy as np
import modules.moteur.mappings.dependances_indic as di
import modules.moteur.services.indicateurs_taux.mni_module as mni
import modules.moteur.mappings.main_mappings as mp


def calcul_mni_gap_tx_pn_mtx(dic_pn_sci, data_dic, cols, type_pn, simul, s):
    """ Fonction permettant de calculer les gaps de taux pour les PN à partir des matrices de fixing"""
    ht = s[0]

    """ GAP d'inflation & GAP de TX FIXE"""
    index_calc = data_dic["data"][gp.nc_output_index_calc_cle]
    filtre_inf = np.array((index_calc.isin(mp.inf_indexes))).reshape(ht, 1, 1)
    filtre_cel = np.array(index_calc.isin(mp.cel_indexes)).reshape(ht, 1, 1)
    filtre_tla_u = np.array((index_calc.isin(mp.tla_indexes)| (index_calc.isin(mp.lep_indexes)))).reshape(ht, 1, 1)
    filtre_tlb = np.array(index_calc.isin(mp.tlb_indexes)).reshape(ht, 1, 1)
    list_coeff_tf = [up.coeff_tf_cel_usr, up.coeff_tf_tla_usr, up.coeff_tf_tlb_usr, 0]
    list_coeff_inf = [up.coeff_inf_cel_usr, up.coeff_inf_tla_usr, up.coeff_inf_tlb_usr, 0]
    filtres = [filtre_cel, filtre_tla_u, filtre_tlb, (~filtre_cel) & (~filtre_tla_u) & (~filtre_tlb)]

    dic_mtx_gp = {}
    for p in range(1, up.nb_mois_proj_out + 1):
        if di.gap_inf_eve_em_pn[p]:
            mtx_em_p = data_dic["mtx_em"].copy()
            mtx_em_p[:, p:, :] = 0
            mtx_em_p[:, :, 0:p - 1] = 0
            mtx_inf_em_p = mtx_em_p - data_dic["mtx_gp_fix_m_" + simul][p]
            mtx_inf_em_p = np.where(filtre_inf, mtx_inf_em_p, 0)
            dic_mtx_gp[gp.gpi_em_eve_pni + str(p)] = mtx_inf_em_p + mtx_inf_em_p * np.select(filtres, list_coeff_inf)

        if di.gap_tx_eve_em_pn[p]:
            mtx_em_p = data_dic["mtx_em"].copy()
            mtx_em_p[:, p:, :] = 0
            mtx_em_p[:, :, 0:p - 1] = 0
            mtx_tx_em_p = mtx_em_p - data_dic["mtx_gp_fix_m_" + simul][p]
            dic_mtx_gp[gp.gp_em_eve_pni + str(p)] = data_dic["mtx_gp_fix_m_" + simul][p] + mtx_tx_em_p * np.select(filtres,list_coeff_tf)

    """ GAP TAUX REG """
    for p in range(1, up.nb_mois_proj_out + 1):
        if di.gap_reg_eve_em_pn[p]:
            dic_mtx_gp[gp.gpr_em_eve_pni + str(p)] = up.coeff_reg_tf_usr * dic_mtx_gp[gp.gp_em_eve_pni + str(p)]\
                                                 + up.coeff_reg_inf_usr * dic_mtx_gp[gp.gpi_em_eve_pni + str(p)]

    """ CALCUL DE LA MNI GAP REG EVE"""
    mni.calculate_mni_mtx_gp_rg_pn(dic_pn_sci, dic_mtx_gp, data_dic, cols, type_pn)

    """ SOMME MATRICIELLE"""
    for p in range(1, up.nb_mois_proj_out + 1):
        if di.gap_inf_eve_em_pn[p]:
            dic_pn_sci[gp.gpi_em_eve_pni + str(p)] = gf.sum_each2(dic_mtx_gp[gp.gpi_em_eve_pni + str(p)], cols)
        if di.gap_tx_eve_em_pn[p]:
            dic_pn_sci[gp.gp_em_eve_pni + str(p)] = gf.sum_each2(dic_mtx_gp[gp.gp_em_eve_pni + str(p)], cols)
        if di.gap_reg_eve_em_pn[p]:
            dic_pn_sci[gp.gpr_em_eve_pni + str(p)] = gf.sum_each2(dic_mtx_gp[gp.gpr_em_eve_pni + str(p)], cols)

    gf.clean_dic_df(dic_mtx_gp)
    if not (((up.force_gps_nmd and simul == "LIQ") or (mp.force_gps_nmd_eve and "EVE" in simul)) and "nmd" in type_pn):
        gf.clean_dic_df(data_dic["mtx_gp_fix_f_" + simul])
        gf.clean_dic_df(data_dic["mtx_gp_fix_m_" + simul])