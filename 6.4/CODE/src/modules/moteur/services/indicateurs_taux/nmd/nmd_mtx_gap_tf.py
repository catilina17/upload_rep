import modules.moteur.parameters.user_parameters as up
import logging
import pandas as pd
import numpy as np
import modules.moteur.mappings.dependances_indic as di
import modules.moteur.utils.generic_functions as gf

logger = logging.getLogger(__name__)



def calculate_mat_gap_tf_nmd(main_data, dic_stock_scr, simul, ht, lg, type_tx_pn, filtre_tla, max_month_pn, mtx_ef, mtx_em,
                             prct, dic_mtx_pn_tx_fix_f, dic_mtx_pn_tx_fix_m):
    """ Fonction permettant de calculer les matrices d'encours taux fixe pour calcul du gap de taux fixe
    pour les produits de type NMD"""
    s = (ht,max_month_pn,lg)
    """ On charge les taux fixe"""

    tab_fix_f, tab_fix_m = load_coeff_fix_st(main_data, dic_stock_scr, ht, lg)

    """ filtres taux fixe et TLA"""
    filtre_tf_all = np.array((type_tx_pn == "TF")).reshape(ht, 1, 1)
    filtre_tla_all = np.array(filtre_tla).reshape(ht, 1, 1)

    for k in range(1, up.nb_mois_proj_out + 1):
        n = min(k, max_month_pn)
        f_mois_refix = (up.mois_refix_tla >= k + 1)

        if "EVE" == simul:
            do_f = False
            do_m = di.gap_tx_eve_em_pn[k]
        elif simul == "EVE_LIQ":
            do_f = False
            do_m = False
        else:
            do_f = di.gap_tx_ef_pn[k]
            do_m = di.gap_tx_em_pn[k]

        if do_f:
            mtx_fix_f = refix_encours_mtx(mtx_ef, tab_fix_f, n, k)

        if do_m:
            mtx_fix_m = refix_encours_mtx(mtx_em, tab_fix_m, n, k)

        if do_m or do_f:
            if not prct == "%":
                tab_fix_tla_roll = calc_tab_refix_tla(f_mois_refix, ht, lg, k)
                if do_f:
                    mtx_fix_f = refix_tla_nmd(mtx_ef, mtx_fix_f, tab_fix_tla_roll, k, n, filtre_tla_all, filtre_tf_all)
                if do_m:
                    mtx_fix_m = refix_tla_nmd(mtx_em, mtx_fix_m, tab_fix_tla_roll, k, n, filtre_tla_all, filtre_tf_all)
        if do_f:
            dic_mtx_pn_tx_fix_f[k] = gf.compress(mtx_fix_f.copy(),s)
        if do_m:
            dic_mtx_pn_tx_fix_m[k] = gf.compress(mtx_fix_m.copy(),s)


def calc_tab_refix_tla(f_mois_refix, ht, lg, k):
    tab_fix = np.zeros((ht, lg))
    if f_mois_refix:
        tab_fix[:, :up.mois_refix_tla] = 1
    else:
        tab_fix[:, :up.freq_refix_tla] = 1

    fix_roll_tla = np.roll(tab_fix, k, axis=1)
    fix_roll_tla[:, :k - 1] = 0
    fix_roll_tla[:, k - 1] = 1
    fix_roll_tla = fix_roll_tla.reshape(ht, 1, lg)

    return fix_roll_tla

def refix_tla_nmd(mtx_encours, mtx_fix_f, tab_fix_tla_roll, k, n, filtre_tla_all, filtre_tf_all):
    mtx_fix = np.where((filtre_tla_all) & (~filtre_tf_all),
                         mtx_encours * tab_fix_tla_roll, mtx_fix_f)
    mtx_fix[:, n:, :] = 0
    mtx_fix[:, :, :k - 1] = 0

    return mtx_fix

def refix_encours_mtx(mtx_encours, tab_fix_f, n, k):
    #mtx_fix = mtx_encours.copy()
    fix_roll = np.roll(tab_fix_f, k, axis=2)
    fix_roll = fix_roll[:, :, 1:]
    fix_roll[:, :, :k - 1] = 0
    """ On met à taux fixe en M0 pour le mois k"""
    #fix_roll[:, :, k - 1] = np.where((~filtre_reserves.reshape(ht, 1)) & (~filtre_tla_all.reshape(ht, 1)) & (~filtre_tf_all.reshape(ht, 1)),
    #                                   fix_roll[:, :, k - 1], 1)
    """ encours*tauxfixe """
    #mtx_fix = np.where((~filtre_tla_all) & (~filtre_tf_all), mtx_encours * fix_roll,
                         #mtx_encours.copy())
    mtx_fix = mtx_encours * fix_roll
    """ on s'assure du zéro pour mois<k """
    mtx_fix[:, n:, :] = 0
    mtx_fix[:, :, :k - 1] = 0

    return mtx_fix

def load_coeff_fix_st(main_data, dic_stock_scr, ht, lg):
    df = pd.DataFrame(index=main_data.index, columns=["TX_FIX_F"])
    if ("tx_fix_f") in dic_stock_scr:
        tab_fix_f = dic_stock_scr["tx_fix_f"]
        tab_fix_f = df.join(tab_fix_f, on="new_key", how="left")[[x for x in tab_fix_f.columns]].copy()
        tab_fix_f = np.array(tab_fix_f.fillna(1).astype(np.float64)).reshape(ht, 1, lg + 1)
        tab_fix_m = dic_stock_scr["tx_fix_m"]
        tab_fix_m = df.join(tab_fix_m, on="new_key", how="left")[[x for x in tab_fix_m.columns]].copy()
        tab_fix_m = np.array(tab_fix_m.fillna(1).astype(np.float64)).reshape(ht, 1, lg + 1)

    else:
        tab_fix_f = np.ones((ht, 1, lg + 1))
        tab_fix_m = np.ones((ht, 1, lg + 1))
        #tab_fix_f[:, :, 0] = 1
        #tab_fix_m[:, :, 0] = 1

    return tab_fix_f, tab_fix_m