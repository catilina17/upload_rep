import modules.moteur.parameters.general_parameters as gp
import modules.moteur.parameters.user_parameters as up
import logging
import numpy as np
import modules.moteur.mappings.dependances_indic as di
import modules.moteur.utils.generic_functions as gf
from collections import OrderedDict

logger = logging.getLogger(__name__)


def calculate_mat_gap_tf_ech(main_data, max_month_pn, ht, lg, dic_mtx_pn_tx_fix_f, dic_mtx_pn_tx_fix_m, mtx_pn_f,
                             mtx_pn_m, mtx_tef, mtx_tem):
    """ FILTRES POUR CHAQUE CAS"""
    s = (ht, max_month_pn, lg)
    fix_cases = generate_fixing_cases(main_data)
    tab_fix = generate_fixing_tables(max_month_pn, lg, fix_cases)
    fix_cases = gf.generate_list_from_ordered_dic(fix_cases)
    for k in range(1, up.nb_mois_proj_out + 1):
        do_f = di.gap_tx_ef_pn[k]
        do_m = di.gap_tx_em_pn[k]
        do_m = do_m or di.gap_tx_eve_em_pn[k]

        """ Calcul de la matrice de gap de taux fixe pour chaque cas """
        if do_f or do_m:
            tab = gf.generate_list_from_ordered_dic(tab_fix[k])

        if do_f:
            dic_mtx_pn_tx_fix_f[k] = calc_fix_mtx(main_data, mtx_pn_f, mtx_tef, fix_cases, tab, s, k)

        if do_m:
            dic_mtx_pn_tx_fix_m[k] = calc_fix_mtx(main_data, mtx_pn_m, mtx_tem, fix_cases, tab, s, k)


def calc_fix_mtx(main_data, mtx, mtx_gp, fix_cases, tab, s, k):
    is_fixed = (main_data[gp.nc_output_index_calc_cle].str.contains(gp.FIX_ind).values).reshape(s[0], 1, 1)
    mtx_fix = mtx * np.select(fix_cases, tab)
    coeff_ajust = (mtx[:, :, k-1] / mtx_gp [:, :, k-1]).reshape(s[0], s[1], 1)
    mtx_fix =  np.where(is_fixed, mtx_fix * np.nan_to_num(coeff_ajust * mtx_gp / mtx, nan = 1), mtx_fix)
    return gf.compress(mtx_fix,s)


def generate_fixing_tables(max_month_pn, lg, cases_fix):
    tab_fix = {}
    for k in range(1, up.nb_mois_proj_out + 1):
        n = min(k, max_month_pn)
        do_f = di.gap_tx_ef_pn[k]
        do_m = di.gap_tx_em_pn[k]
        do_m = do_m or di.gap_tx_eve_em_pn[k]
        if do_f or do_m:
            tab_fix[k] = OrderedDict()
            calc_tab_fix_k_ech(tab_fix, k, cases_fix, max_month_pn, lg, n)

    return tab_fix


def calc_tab_fix_k_ech(tab_fix, k, cases_fix, max_month_pn, lg, n):
    for freq, filtre in cases_fix.items():
        tab_fix[k][freq] = np.zeros((max_month_pn, lg))

    mois_proj = np.vstack([np.arange(0, lg)] * max_month_pn)
    mois_emission = np.column_stack([np.arange(0, max_month_pn)] * lg)
    cond1 = np.where(mois_proj >= k - 1, 1, 0)
    for nb_mois, filter in cases_fix.items():
        reste = np.remainder(k - mois_emission, nb_mois)
        cond2 = np.where(mois_proj < k + nb_mois - reste, 1, 0)
        tab_fix[k][nb_mois] = cond1 * cond2

    for nb_mois, filter in cases_fix.items():
        tab_fix[k][nb_mois][n:, :] = 0


def generate_fixing_cases(main_data):
    per_fx = generate_freq_fixing_periodicity(main_data, gp.nc_pn_periode_fixing, gp.nc_pn_periode_interets)
    per_fx = np.where(main_data[gp.nc_pn_periode_fixing] == "1D", 0, per_fx)
    per_fx = np.where(main_data[gp.nc_output_index_calc_cle].str.contains(gp.FIX_ind), up.nb_mois_proj_usr + 5, per_fx)
    cases_fx = OrderedDict()
    for x in np.unique(per_fx):
        cases_fx[x] = (per_fx == x).reshape(per_fx.shape[0], 1, 1)
    return cases_fx


def generate_freq_fixing_periodicity(data, col_fixing_periodicity, col_periodicity):
    periodicity = np.select([data[col_periodicity] == x for x in ["MENSUEL", "TRIMESTRIEL", "NONE", "ANNUEL", "SEMESTRIEL"]],
                            ["1M", "3M", "1M", "1Y", "6M"], default="1M")
    not_null_fixing_per = data[col_fixing_periodicity].isnull().values
    fix_per_vals = data[col_fixing_periodicity].values
    data[col_fixing_periodicity] = np.where(not_null_fixing_per, periodicity, fix_per_vals)
    is_year = (data[col_fixing_periodicity].astype(str).str[-1:] == "Y").values
    not_month = (data[col_fixing_periodicity].astype(str).str[-1:] != "M").values
    fix_per = data[col_fixing_periodicity].astype(str).str[:-1].astype(int).values
    nb_freq = np.where(not_month & (~is_year), 1, fix_per)
    freq_per = np.where(is_year, 12 * nb_freq, nb_freq)
    return freq_per
