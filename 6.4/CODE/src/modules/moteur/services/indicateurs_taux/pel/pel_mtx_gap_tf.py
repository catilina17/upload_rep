import modules.moteur.parameters.user_parameters as up
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import logging
import numpy as np
import modules.moteur.utils.generic_functions as gf
import modules.moteur.mappings.dependances_indic as di

logger = logging.getLogger(__name__)

pel_limit_age_gp_tx = 15 * 12

def calculate_mat_gap_tf_pel(data_pel, dic_pn_sc, mtx_ef, mtx_em, max_month_pn, lg, s):
    dic_mtx_pn_tx_fix_f = {}
    dic_mtx_pn_tx_fix_m = {}
    age_pel = np.repeat(np.arange(0, up.nb_mois_proj_usr).reshape(1, up.nb_mois_proj_usr), max_month_pn, axis=0)
    age_pel = age_pel - np.arange(0, max_month_pn).reshape(max_month_pn, 1)
    age_pel = np.where(age_pel < 0, 0, age_pel).reshape(1, max_month_pn, up.nb_mois_proj_usr)
    is_not_pel_c = ~data_pel[pa.NC_PA_CONTRACT_TYPE].str.contains("-C").values.reshape(s[0], 1, 1)

    for k in range(1, up.nb_mois_proj_out + 1):
        do_f = di.gap_tx_ef_pn[k]
        do_m = di.gap_tx_em_pn[k] or di.gap_tx_eve_em_pn[k]

        if do_f or do_m:
            df = np.zeros((max_month_pn, lg))
            n = min(k, max_month_pn)
            """ table de fixing """
            df[:n, k - 1:] = 1
            """ matrice fixing = table de fixing * encours """
            if do_f:
                mtx_tf = mtx_ef.copy()
                mtx_tf = np.where((age_pel > pel_limit_age_gp_tx) & is_not_pel_c, 0, mtx_tf)
                dic_mtx_pn_tx_fix_f[k] = gf.compress((mtx_tf * df), s)
            if do_m:
                mtx_tm = mtx_em.copy()
                mtx_tm = np.where((age_pel > pel_limit_age_gp_tx) & is_not_pel_c, 0, mtx_tm)
                dic_mtx_pn_tx_fix_m[k] = gf.compress((mtx_tm * df), s)

    for simul in ["LIQ", "EVE", "EVE_LIQ"]:
        if up.type_simul[simul]:
            dic_pn_sc["mtx_gp_fix_m_" + simul] = dic_mtx_pn_tx_fix_m
            dic_pn_sc["mtx_gp_fix_f_" + simul] = dic_mtx_pn_tx_fix_f
