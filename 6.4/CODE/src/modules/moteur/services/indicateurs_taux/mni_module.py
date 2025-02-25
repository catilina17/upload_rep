import modules.moteur.parameters.general_parameters as gp
import modules.moteur.parameters.user_parameters as up
import modules.moteur.index_curves.tx_module as tx
import modules.moteur.utils.generic_functions as gf
import numpy as np
import modules.moteur.mappings.dependances_indic as di
import logging
import modules.moteur.mappings.main_mappings as mp

logger = logging.getLogger(__name__)


def calculate_mni_st(dic_stock_sci, indic_sortie):
    """ calcul de la mni tx et totale pour le stock """
    if gp.mn_sti in indic_sortie in indic_sortie in indic_sortie:
        dic_stock_sci[gp.mn_sti].insert(0, "M0", 0)

    if gp.mn_gp_rg_sti in indic_sortie:
        dic_stock_sci[gp.mn_gp_rg_sti].insert(0, "M0", 0)


def calculate_mni_pn(dic_pn_sci, data_dic, simul, type_pn, cols):
    """ calcul de la mni tx, mg, pour la PN """
    mtx_tx_pur = None;
    mtx_tx_mg = None

    if gf.begin_with_list(up.indic_sortie["PN"], gp.mn_pni) or gf.begin_with_list(up.indic_sortie["PN"],
                                                                                  gp.mn_tx_pni):
        mtx_tx_pur = data_dic["mni_sc_"]

    if gf.begin_with_list(up.indic_sortie["PN"], gp.mn_pni) or gf.begin_with_list(up.indic_sortie["PN"],
                                                                                  gp.mn_mg_pni):
        mtx_tx_mg = data_dic["mni_mg_"]

    """ Indicateurs simples"""
    if (gp.mn_tx_pni in up.indic_sortie["PN"] or gp.mn_pni in up.indic_sortie["PN"]) and simul =="LIQ":
        dic_pn_sci[gp.mn_tx_pni] = gf.sum_each2(mtx_tx_pur, cols)

    if (gp.mn_mg_pni in up.indic_sortie["PN"] or gp.mn_pni in up.indic_sortie["PN"])  and simul =="LIQ":
        dic_pn_sci[gp.mn_mg_pni] = gf.sum_each2(mtx_tx_mg, cols)

    if gp.mn_pni in up.indic_sortie["PN"]  and simul =="LIQ":
        dic_pn_sci[gp.mn_pni] = dic_pn_sci[gp.mn_tx_pni] + dic_pn_sci[gp.mn_mg_pni]

    """ Indicateurs de contribution """
    for p in range(1, up.nb_mois_proj_out + 1):
        if gp.mn_tx_pni + str(p) in up.indic_sortie["PN"] and simul =="LIQ":
            dic_pn_sci[gp.mn_tx_pni + str(p)] = gf.sum_each2(mtx_tx_pur, cols, proj=True, per=p,
                                                             interv=up.indic_sortie["PN_CONTRIB"][gp.mn_tx_pni])

        if gp.mn_mg_pni + str(p) in up.indic_sortie["PN"] and simul =="LIQ":
            dic_pn_sci[gp.mn_mg_pni + str(p)] = gf.sum_each2(mtx_tx_mg, cols, proj=True, per=p)

        if gp.mn_pni + str(p) in up.indic_sortie["PN"] and simul =="LIQ":
            dic_pn_sci[gp.mn_pni + str(p)] = gf.sum_each2(mtx_tx_pur + mtx_tx_mg, cols, proj=True, per=p, \
                                                          interv=up.indic_sortie["PN_CONTRIB"][gp.mn_pni])

        if di.mni_gpliq_tx_pn[p] and simul == "EVE_LIQ":
            dic_pn_sci[gp.mn_gpliq_tx_pni + str(p)] = gf.sum_each2(mtx_tx_pur, cols, proj=True, per=p, fix=True)

        if di.mni_gpliq_mg_pn[p] and simul == "EVE_LIQ":
            dic_pn_sci[gp.mn_gpliq_mg_pni + str(p)] = gf.sum_each2(mtx_tx_mg, cols, proj=True, per=p, fix=True)

        if di.mni_gpliq_pn[p] and simul == "EVE_LIQ":
            dic_pn_sci[gp.mn_gpliq_pni + str(p)] = dic_pn_sci[gp.mn_gpliq_tx_pni + str(p)] + dic_pn_sci[gp.mn_gpliq_mg_pni + str(p)]

    if mtx_tx_pur is not None:
        gf.clean_df(mtx_tx_pur)
    if mtx_tx_mg is not None:
        gf.clean_df(mtx_tx_mg)

def force_marge_co_to_zero_for_tci_contracts(data_pn, tx_mg_sc):
    ht = data_pn.shape[0]
    filtre_contrat_tci = np.array(data_pn[gp.nc_output_dim2].isin(mp.cc_tci)).reshape(ht, 1, 1)
    filtre_contrat_tci_excl = ~np.array(data_pn[gp.nc_output_contrat_cle].isin(mp.cc_tci_excl)).reshape(ht, 1, 1)
    new_tx_mg_sc = np.where(filtre_contrat_tci & filtre_contrat_tci_excl, 0, tx_mg_sc)
    return new_tx_mg_sc

def force_mni_to_mni_tci_for_nmd_tci_contracts(data_pn, tx_sc, tx_sc_ftp):
    ht = data_pn.shape[0]
    filtre_contrat_tci = np.array(data_pn[gp.nc_output_dim2].isin(mp.cc_tci)).reshape(ht, 1, 1)
    filtre_contrat_tci_excl = ~np.array(data_pn[gp.nc_output_contrat_cle].isin(mp.cc_tci_excl)).reshape(ht, 1, 1)
    new_tx_sc = np.where(filtre_contrat_tci & filtre_contrat_tci_excl, tx_sc_ftp, tx_sc)
    return new_tx_sc, filtre_contrat_tci & filtre_contrat_tci_excl

def calculate_mni_mtx_gp_rg_pn(dic_pn_sci, dic_mtx_gp, data_dic, cols, type_pn):
    """ calcul de la mni tx, mg, liq vision gap reg pour la PN """
    for p in range(1, up.nb_mois_proj_out + 1):
        if di.mni_gpr_tx_pn[p]:
            if "ech" in type_pn:
                new_tx_sc = data_dic["tx_sc_"]
            else:
                new_tx_sc, filter_tci = force_mni_to_mni_tci_for_nmd_tci_contracts(data_dic["data"], data_dic["tx_sc_"], data_dic["tx_sc_ftp_"])
            mtx_tx_pur = dic_mtx_gp[gp.gpr_em_eve_pni + str(p)] * (new_tx_sc * data_dic["base_calc_"])
            dic_pn_sci[gp.mn_gpr_tx_pni + str(p)] = gf.sum_each2(mtx_tx_pur, cols)

        if di.mni_gpr_mg_pn[p]:
            if "ech" in type_pn:
                tx_mg = force_marge_co_to_zero_for_tci_contracts(data_dic["data"], data_dic["tx_mg_"])
            else:
                tx_mg = data_dic["tx_mg_"]
            mtx_tx_mg = dic_mtx_gp[gp.gpr_em_eve_pni + str(p)] * tx_mg * data_dic["base_calc_"]
            dic_pn_sci[gp.mn_gpr_mg_pni + str(p)] = gf.sum_each2(mtx_tx_mg, cols)

        if di.mni_gpr_pn[p]:
            dic_pn_sci[gp.mn_gpr_pni + str(p)] = dic_pn_sci[gp.mn_gpr_tx_pni + str(p)] \
                                                 + dic_pn_sci[gp.mn_gpr_mg_pni + str(p)]


def calculate_mni_gp_rg_nmd(dic_pn_sci, data_dic):
    ht = data_dic["tx_sc_"].shape[0]
    lg = data_dic["tx_sc_"].shape[2]
    #On prend le taux de la PN1, car c'est censé être le même taux pour tous les produits puisque ce sont les mêmes
    # : data_dic["tx_sc_"][:, 0]
    # SAUF en cas de TCI, auquel cas, on prend une moyenne pondérée du TCI sur les p premiers mois
    for p in range(1, up.nb_mois_proj_out + 1):
        if di.mni_gpr_tx_pn[p]:
            new_tx_sc, filter_tci = force_mni_to_mni_tci_for_nmd_tci_contracts(data_dic["data"], data_dic["tx_sc_"], data_dic["tx_sc_ftp_"])
            new_tx_sc = np.where(filter_tci.reshape(ht, 1), (data_dic["tx_sc_"][:, :p] * data_dic["mtx_em"][:, :p]).sum(axis=1)
                                 / data_dic["mtx_em"][:, :p].sum(axis=1), data_dic["tx_sc_"][:, 0])
            dic_pn_sci[gp.mn_gpr_tx_pni + str(p)] = dic_pn_sci[gp.gpr_em_eve_pni + str(p)] * (
                np.nan_to_num(new_tx_sc)) * data_dic["base_calc_"]

        if di.mni_gpr_mg_pn[p]:
            dic_pn_sci[gp.mn_gpr_mg_pni + str(p)] = dic_pn_sci[gp.gpr_em_eve_pni + str(p)] * (
                data_dic["tx_mg_"][:, 0]) * data_dic["base_calc_"]

        if di.mni_gpr_pn[p]:
            dic_pn_sci[gp.mn_gpr_pni + str(p)] = dic_pn_sci[gp.mn_gpr_tx_pni + str(p)] \
                                                 + dic_pn_sci[gp.mn_gpr_mg_pni + str(p)]


def calculate_mni_ajust_gp_rg(dic_ajust, num_cols):
    """ Chargement des courbes de TAUX """
    if (sum(list(di.mni_gpr_tx_aj.values())) + sum(list(di.mni_gpr_tx_aj.values()))
            + sum(list(di.mni_gpr_tx_aj.values())) > 0):
        ind_em = gp.em_eve_sti
        data_adj_tx = dic_ajust[ind_em].copy()
        tx_curve = tx.tx_curves_sc.drop([gp.NC_TYPE_SC_TX],axis=1)
        tx_sc = (data_adj_tx[["CURVE_NAME", "TENOR"]].copy()
                 .join(tx_curve.set_index(["CODE COURBE", "TENOR"]), how="left", on=["CURVE_NAME", "TENOR"]))

        tx_sc["M00"] = 0
        tx_sc = tx_sc[["M0" + str(i) if i <= 9 else "M" + str(i) for i in range(0, up.nb_mois_proj_usr + 1)]]
        tx_sc = np.array(tx_sc)

        """ chargement des TX LIQ """
        sp_liq = \
            data_adj_tx[[gp.nc_output_devise_cle]].copy().merge(up.spread_liq, how="left", right_on=gp.nc_devise_spread,
                                                                left_on=gp.nc_output_devise_cle)[gp.nc_taux_spread]

        for p in range(1, up.nb_mois_proj_out + 1):
            if di.mni_gpr_tx_aj[p]:
                """ MNI TX"""
                data_adj_tx[num_cols] = np.array(dic_ajust[gp.gpr_em_eve_pni + str(p)][num_cols]) * (
                    (tx_sc)) / 12
                data_adj_tx[gp.nc_output_ind3] = gp.mn_gpr_tx_pni + str(p)
                data_adj_liq = dic_ajust[gp.gpr_em_eve_pni + str(p)].copy()
                data_adj_liq[num_cols] = np.array(data_adj_liq[num_cols]) * (
                    ((np.array(sp_liq / 10000)) / 12).reshape((sp_liq.shape[0], 1)))
                data_adj_liq["M00"] = 0
                data_adj_tx[num_cols] = data_adj_tx[num_cols].values + data_adj_liq[num_cols].values
                dic_ajust[gp.mn_gpr_tx_pni + str(p)] = data_adj_tx

            if di.mni_gpr_mg_aj[p]:
                """ MNI MG"""
                data_adj_mg = dic_ajust[gp.gpr_em_eve_pni + str(p)].copy()
                data_adj_mg[num_cols] = np.zeros(data_adj_mg[num_cols].shape)
                data_adj_mg[gp.nc_output_ind3] = gp.mn_gpr_mg_pni + str(p)
                dic_ajust[gp.mn_gpr_mg_pni + str(p)] = data_adj_mg

            if di.mni_gpr_aj[p]:
                """ MNI """
                data_adj_mni = dic_ajust[gp.gpr_em_eve_pni + str(p)].copy()
                data_adj_mni[num_cols] = data_adj_tx[num_cols]
                data_adj_mni[gp.nc_output_ind3] = gp.mn_gpr_pni + str(p)
                dic_ajust[gp.mn_gpr_pni + str(p)] = data_adj_mni


def calculate_mni_ajust(dic_ajust, num_cols):
    """ Fonction permettant de calculer la mni des ajustements """
    ind_em = gp.em_sti
    """ Chargement des courbes de TAUX """
    data_adj_tx = dic_ajust[ind_em].copy()
    tx_curve = tx.tx_curves_sc.drop([gp.NC_TYPE_SC_TX],axis=1)
    tx_sc = (data_adj_tx[["CURVE_NAME", "TENOR"]].copy()
             .join(tx_curve.set_index(["CODE COURBE", "TENOR"]), how="left", on=["CURVE_NAME", "TENOR"]))
    tx_sc["M00"] = 0
    tx_sc = tx_sc[["M0" + str(i) if i <= 9 else "M" + str(i) for i in range(0, up.nb_mois_proj_usr + 1)]]
    tx_sc = np.array(tx_sc)

    """ chargement des TX LIQ """
    default = ~data_adj_tx[gp.nc_output_devise_cle].isin(up.spread_liq[gp.nc_devise_spread].values.tolist())
    data_adj_tx.loc[default, gp.nc_output_devise_cle] = "*"
    sp_liq = \
        data_adj_tx[[gp.nc_output_devise_cle]].copy().merge(up.spread_liq, how="left", right_on=gp.nc_devise_spread,
                                                            left_on=gp.nc_output_devise_cle)[gp.nc_taux_spread]
    """ MNI TX"""
    data_adj_tx[num_cols] = np.array(data_adj_tx[num_cols]) * (
        (tx_sc)) / 12
    data_adj_tx[gp.nc_output_ind3] = gp.mn_tx_pni

    data_adj_liq = dic_ajust[ind_em].copy()
    data_adj_liq[num_cols] = np.array(data_adj_liq[num_cols]) * (
        ((np.array(sp_liq / 10000)) / 12).reshape((sp_liq.shape[0], 1)))
    data_adj_liq[gp.nc_output_ind3] = gp.mn_lq_pni
    data_adj_liq["M00"] = 0
    data_adj_tx[num_cols] = data_adj_tx[num_cols].values + data_adj_liq[num_cols].values
    dic_ajust[gp.mn_tx_pni] = data_adj_tx

    """ MNI MG"""
    data_adj_mg = dic_ajust[ind_em].copy()
    data_adj_mg[num_cols] = np.zeros(data_adj_mg[num_cols].shape)
    data_adj_mg[gp.nc_output_ind3] = gp.mn_mg_pni
    dic_ajust[gp.mn_mg_pni] = data_adj_mg

    """ MNI """
    data_adj_mni = dic_ajust[ind_em].copy()
    data_adj_mni[num_cols] = data_adj_tx[num_cols]
    data_adj_mni[gp.nc_output_ind3] = gp.mn_pni
    dic_ajust[gp.mn_pni] = data_adj_mni

    """ TX CLI"""
    if gp.tx_cli_pni in up.indic_sortie["AJUST"]:
        dic_ajust[gp.tx_cli_pni] = dic_ajust[ind_em].copy()
        dic_ajust[gp.tx_cli_pni][gp.nc_output_ind3] = gp.tx_cli_pni
        dic_ajust[gp.tx_cli_pni][num_cols] = (tx_sc) / 12

    """ INDIC de contribution """
    for p in range(1, up.nb_mois_proj_out + 1):
        pref = "M0" if p <= 9 else "M"
        for ind_mni in [gp.mn_tx_pni, gp.mn_mg_pni, gp.mn_pni]:
            if ind_mni + str(p) in up.indic_sortie["AJUST"]:
                dic_ajust[ind_mni + str(p)] = dic_ajust[ind_mni].copy()
                dic_ajust[ind_mni + str(p)][gp.nc_output_ind3] = ind_mni + str(p)
                cols = [x for x in num_cols if pref + str(p) != x]
                dic_ajust[ind_mni + str(p)][cols] = 0

    for p in range(1, up.nb_mois_proj_out + 1):
        pref = "M0" if p <= 9 else "M"
        for ind_mni, dic in zip([gp.mn_gpliq_tx_pni, gp.mn_gpliq_pni],
        [di.mni_gpliq_tx_aj, di.mni_gpliq_mg_aj, di.mni_gpliq_aj]):
            if dic[p]:
                dic_ajust[ind_mni + str(p)] = dic_ajust[gp.mn_pni].copy()
                dic_ajust[ind_mni + str(p)][gp.nc_output_ind3] = ind_mni + str(p)
                cols = [x for x in num_cols if pref + str(p) != x]
                dic_ajust[ind_mni + str(p)][cols] = 0

        for ind_mni, dic in zip([gp.mn_gpliq_mg_pni],[di.mni_gpliq_mg_aj]):
            if dic[p]:
                dic_ajust[ind_mni + str(p)] = dic_ajust[gp.mn_pni].copy()
                dic_ajust[ind_mni + str(p)][gp.nc_output_ind3] = ind_mni + str(p)
                dic_ajust[ind_mni + str(p)][num_cols] = 0
