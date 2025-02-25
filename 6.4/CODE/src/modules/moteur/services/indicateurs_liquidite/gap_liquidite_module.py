import modules.moteur.parameters.user_parameters as up
import modules.moteur.utils.generic_functions as gf
import modules.moteur.parameters.general_parameters as gp
import modules.moteur.mappings.main_mappings as mp
import numpy as np
import modules.moteur.mappings.dependances_indic as di


def calcul_gap_liq_pn(dic_pn_sci, data_dic, simul, cols):
    """ Permet de calculer le gap de liquidité des PNs à partir des matrices d'encours"""
    for p in range(1, up.nb_mois_proj_out + 1):
        do_f = di.gap_liq_ef_pn[p] if simul == "LIQ" else False
        do_m = di.gap_liq_em_pn[p] if simul == "LIQ" else di.gap_liq_eve_em_pn[p]
        gp_liq_f = gp.gp_liq_f_pni
        gp_liq_m = gp.gp_liq_m_pni if simul == "LIQ" else gp.gp_liq_em_eve_pni
        if do_f:
            dic_pn_sci[gp_liq_f + str(p)] = gf.sum_each2(data_dic["mtx_ef"], cols, proj=True, per=p, fix=True)
            dic_pn_sci[gp_liq_f + str(p)].loc[:, ["M" + str(i) for i in range(1, p)]] = 0

        if do_m:
            dic_pn_sci[gp_liq_m + str(p)] = gf.sum_each2(data_dic["mtx_em"], cols, proj=True, per=p, fix=True)
            dic_pn_sci[gp_liq_m + str(p)].loc[:, ["M" + str(i) for i in range(1, p)]] = 0


def calcul_gap_liq_st(dic_stock_sci, simul):
    """ Permet de calculer le gap de liquidité du stock à partir des encours"""
    for p in range(0, up.nb_mois_proj_out + 1):
        do_f = di.gap_liq_ef_st[p] if simul == "LIQ" else False
        do_m = di.gap_liq_em_st[p] if simul == "LIQ" else di.gap_liq_eve_em_st[p]
        gp_liq_f = gp.gp_liq_f_sti
        gp_liq_m = gp.gp_liq_m_sti if simul == "LIQ" else gp.gp_liq_em_eve_sti

        if do_f:
            cols = ["M" + str(i) for i in range(0, p)]
            dic_stock_sci[gp_liq_f + str(p)] = dic_stock_sci[gp.ef_sti].copy()
            dic_stock_sci[gp_liq_f + str(p)][cols] = 0

        if do_m:
            cols = ["M" + str(i) for i in range(0, p)]
            dic_stock_sci[gp_liq_m + str(p)] = dic_stock_sci[gp.em_sti].copy()
            dic_stock_sci[gp_liq_m + str(p)][cols] = 0


def calcul_gap_liq_ajust(dic_ajust, simul, num_cols):
    """ Permet de calculer le gap de liquidité des ajustements à partir des encours"""
    ind_em = gp.em_sti if type != "EVE" else gp.em_eve_sti
    ind_ef = gp.ef_sti if type != "EVE" else gp.ef_eve_sti
    for p in range(0, up.nb_mois_proj_out + 1):
        pref = "M0" if p <= 9 else "M"
        do_f = di.gap_liq_ef_aj[p] if simul == "LIQ" else False
        do_m = di.gap_liq_em_aj[p] if simul == "LIQ" else di.gap_liq_eve_em_aj[p]
        gp_liq_f = gp.gp_liq_f_pni
        gp_liq_m = gp.gp_liq_m_pni if simul == "LIQ" else gp.gp_liq_em_eve_pni
        if do_f:
            data_adj_gap = dic_ajust[ind_ef].copy()
            data_adj_gap[gp.nc_output_ind3] = gp_liq_f + str(p)
            dic_ajust[gp_liq_f + str(p)] = data_adj_gap
            cols = [x for x in num_cols if pref + str(p) != x]
            dic_ajust[gp_liq_f + str(p)][cols] = 0

        if do_m:
            data_adj_gap = dic_ajust[ind_em].copy()
            data_adj_gap[gp.nc_output_ind3] = gp_liq_m + str(p)
            dic_ajust[gp_liq_m + str(p)] = data_adj_gap
            cols = [x for x in num_cols if pref + str(p) != x]
            dic_ajust[gp_liq_m + str(p)][cols] = 0


def appliquer_conv_ecoulement_gp_liq(dic_ind, typo, data, data_other, simul):
    if (up.force_gp_liq and simul=="LIQ") or (mp.force_gp_liq_eve and "EVE" in simul):
        force_gap_liq(dic_ind, typo, data_other, simul, option_data=data)
    if (up.force_gps_nmd and simul=="LIQ") or (mp.force_gps_nmd_eve and "EVE" in simul):
        force_gap_liq_nmd(dic_ind, typo, data, simul)


def force_gap_liq(dic_ind, typo, data, simul, option_data=""):
    """ Permet d'appliquer une convention d'écoulements le gap de liquidité projeté pour certains"""

    ht = data.shape[0]
    cle_jointure = [gp.nc_output_bassin_cle, gp.nc_output_bilan, gp.nc_output_dim4]
    if typo == "ST":
        deb = 0
        gp_liq_f = gp.gp_liq_f_sti
        gp_liq_m = gp.gp_liq_m_sti if simul == "LIQ" else gp.gp_liq_em_eve_sti
    else:
        deb = 1
        gp_liq_f = gp.gp_liq_f_pni
        gp_liq_m = gp.gp_liq_m_pni if simul == "LIQ" else gp.gp_liq_em_eve_pni

    if typo == "PN":
        data2 = data[[gp.nc_output_contrat_cle]].copy()
        add_cols = [gp.nc_cp_bilan, gp.nc_cp_dim4]
        data2 = data2.join(mp.contrats_map[add_cols].copy(), on=gp.nc_output_contrat_cle,
                           how="left")
        data2[gp.nc_output_bassin_cle] = option_data[gp.nc_output_bassin_cle].values
        data2 = data2[cle_jointure].copy()
    else:
        data2 = data.copy()
        data2[gp.nc_output_bassin_cle] = option_data[gp.nc_output_bassin_cle].values
        data2 = data2[cle_jointure].copy()

    data2["key_gpliq"] = data2[cle_jointure].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    profil_ecoul_gpp = data2.join(mp.conv_gpliq, on="key_gpliq")[
        ["M" + str(i) for i in range(0, up.nb_mois_proj_usr + 1)]].copy()
    profil_ecoul_gpp = np.array(profil_ecoul_gpp).astype(np.float64)

    filtro = (~np.isnan(profil_ecoul_gpp[:, 0])).reshape(ht, 1)

    for p in range(deb, up.nb_mois_proj_out + 1):
        if typo == "PN":
            do_f = di.gap_liq_ef_pn[p] if simul == "LIQ" else False
            do_m = di.gap_liq_em_pn[p] if simul == "LIQ" else di.gap_liq_eve_em_pn[p]
            k = p - 1
        else:
            do_f = di.gap_liq_ef_st[p] if simul == "LIQ" else False
            do_m = di.gap_liq_em_st[p] if simul == "LIQ" else di.gap_liq_eve_em_st[p]
            k = p

        if do_f:
            dic_ind[gp_liq_f + str(p)].iloc[:, k:] = \
                np.where(filtro, np.array(dic_ind[gp_liq_f + str(p)]["M" + str(p)]).reshape(ht, 1) * \
                         np.roll(profil_ecoul_gpp, p, axis=1)[:, p:], dic_ind[gp_liq_f + str(p)].iloc[:, k:])
        if do_m:
            dic_ind[gp_liq_m + str(p)].iloc[:, k:] = \
                np.where(filtro, np.array(dic_ind[gp_liq_m + str(p)]["M" + str(p)]).reshape(ht, 1) * \
                         np.roll(profil_ecoul_gpp, p, axis=1)[:, p:], dic_ind[gp_liq_m + str(p)].iloc[:, k:])


def force_gap_liq_nmd(dic_ind, typo, data, simul):
    """ Permet d'appliquer une convention d'écoulements le gap de liquidité projeté pour certains contrats NMD """

    ht = data.shape[0]
    cle_jointure = [gp.nc_output_contrat_cle]
    if typo == "ST":
        deb = 0
        gp_liq_f = gp.gp_liq_f_sti if simul == "LIQ" else False
        gp_liq_m = gp.gp_liq_m_sti if simul == "LIQ" else gp.gp_liq_em_eve_sti
        data_join = data.reset_index(level=[gp.nc_output_palier_cle, gp.nc_output_contrat_cle])[cle_jointure]
    else:
        deb = 1
        gp_liq_f = gp.gp_liq_f_pni if simul == "LIQ" else False
        gp_liq_m = gp.gp_liq_m_pni if simul == "LIQ" else gp.gp_liq_em_eve_pni
        data_join = data[cle_jointure].copy()

    data_join["main_key"] = data_join[cle_jointure].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    conv_ecoul = data_join.join(mp.conv_gps_nmd, on="main_key")[
        ["M" + str(i) for i in range(0, up.nb_mois_proj_usr + 1)]].copy()
    conv_ecoul = np.array(conv_ecoul).astype(np.float64)
    filtro = (~np.isnan(conv_ecoul[:, 0])).reshape(ht, 1)

    for p in range(deb, up.nb_mois_proj_out + 1):
        if typo == "PN":
            do_f = di.gap_liq_ef_pn[p] if simul == "LIQ" else False
            do_m = di.gap_liq_em_pn[p] if simul == "LIQ" else di.gap_liq_eve_em_pn[p]
            k = p - 1
        else:
            do_f = di.gap_liq_ef_st[p] if simul == "LIQ" else False
            do_m = di.gap_liq_em_st[p] if simul == "LIQ" else di.gap_liq_eve_em_st[p]
            k = p

        if do_f:
            dic_ind[gp_liq_f + str(p)].iloc[:, k:] = \
                np.where(filtro, np.array(dic_ind[gp_liq_f + str(p)]["M" + str(p)]).reshape(ht, 1) * \
                         np.roll(conv_ecoul, p, axis=1)[:, p:], dic_ind[gp_liq_f + str(p)].iloc[:, k:])
        if do_m:
            dic_ind[gp_liq_m + str(p)].iloc[:, k:] = \
                np.where(filtro, np.array(dic_ind[gp_liq_m + str(p)]["M" + str(p)]).reshape(ht, 1) * \
                         np.roll(conv_ecoul, p, axis=1)[:, p:], dic_ind[gp_liq_m + str(p)].iloc[:, k:])
