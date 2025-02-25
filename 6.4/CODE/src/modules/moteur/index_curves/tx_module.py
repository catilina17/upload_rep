# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:12:56 2020

@author: TAHIRIH
"""
import modules.moteur.parameters.general_parameters as gp
import modules.moteur.index_curves.calendar_module as hf
from modules.moteur.utils import generic_functions as gf
import modules.moteur.mappings.main_mappings as mp
import modules.moteur.parameters.user_parameters as up
import logging
import pandas as pd
import numpy as np
import utils.excel_utils as ex
import os

global tx_tv_zero
global tx_curves_sc; global tx_curves_rco, liq_curves_sc
global zc_all_curves
global tv_zc_curve
global delta_days, max_month_zc, max_duree
global num_cols_taux, zc_curves_df, tci_values

logger = logging.getLogger(__name__)

zc_liq_curves = {};

num_cols_taux = ["M0"+str(i) if i<=9 else "M" + str(i) for i in range(0, gp.real_max_months + 1)]

def get_max_duree():
    global max_duree, max_month_zc

    if gf.begin_in_list(up.indic_sortie_eve["PN"], gp.eve_ef_pni) or gf.begin_in_list(up.indic_sortie_eve["PN"], gp.eve_em_pni) \
            or gf.begin_in_list(up.indic_sortie_eve["ST"], gp.eve_ef_sti) or gf.begin_in_list(up.indic_sortie_eve["ST"],
                                                                                          gp.eve_em_sti):
        max_duree = min(up.nb_mois_proj_usr + 1, gp.max_months + 1)
        max_month_zc = min(up.nb_mois_proj_usr + 1, gp.pn_max + 1)

    if max_duree > gp.max_months2:
        error = "Not enough days in base calc. Some products have a duration greater than of 600-" + str(max_duree)
        logger.error(error)
        raise ValueError(error)

def get_delta_days_from_forward_month():
    global delta_days, max_month_zc, max_duree
    dates_fin_per = np.array(hf.calendar_coeff[hf.calendar_coeff[gp.nc_pn_base_calc] == "Dcold"][
                         [x for x in hf.calendar_coeff.columns if x != gp.nc_pn_base_calc]])
    dates_fin_per = dates_fin_per[:, :max_duree]
    list_delta_days = []
    for j in range(0, max_month_zc):
        delta_days = (dates_fin_per - dates_fin_per[:, j]).astype('timedelta64[D]').reshape(max_duree, )
        delta_days = np.maximum(0, delta_days / np.timedelta64(1, 'D'))
        list_delta_days.append(delta_days)

    delta_days = np.column_stack(list_delta_days)

def interpol_zc_data(data_boot):
    global delta_days, max_duree, max_month_zc
    data_boot = np.array(data_boot)[:, 1:]
    jr = data_boot[:, 3::2].astype(np.float64)  # On élimine le mois 0
    zc = data_boot[:, 4::2].astype(np.float64)
    listo = []
    for j in range(0, max_month_zc):
        delta_days_j = delta_days[:,j]
        zc_interpolated = np.interp(delta_days_j, jr[:, j], zc[:, j], left=0).reshape(max_duree)
        zc_interpolated = np.where(delta_days_j==0, 0, zc_interpolated)
        listo.append(zc_interpolated)

    data = pd.DataFrame(np.column_stack(listo), columns=["M" + str(i) for i in range(1, max_month_zc + 1)])

    return data

def load_tx_curves(files_sc, scen_path, cls_pn_loader):
    """ Fonction permettant de charger les courbes de taux scénario"""
    global tx_curves_sc, liq_curves_sc, tci_values
    global tx_curves_rco
    global zc_all_curves
    global tx_tv_zero, tv_zc_curve, zc_curves_df

    path_file_sc_tx = os.path.join(scen_path, files_sc["TAUX"][0])
    wb = ex.try_close_open(path_file_sc_tx, read_only=True)

    header_tx = gp.List_TX_HEADER

    tx_tv_zero = tx_tv_zero = pd.DataFrame(0, index=np.arange(1), columns=header_tx)
    tx_tv_zero[gp.nc_taux] = gp.label_tv_zero

    """ Courbes de taux """
    tx_curves_sc = ex.get_dataframe_from_range(wb, gp.ng_tx_sc)
    tx_curves_sc.columns = tx_curves_sc.columns.tolist()[:-gp.real_max_months-1] + num_cols_taux
    tx_curves_rco = ex.get_dataframe_from_range(wb, gp.ng_tx_rco)
    tx_curves_rco.columns = tx_curves_rco.columns.tolist()[:-gp.real_max_months-1] + num_cols_taux

    liq_curves_sc = ex.get_dataframe_from_range(wb, gp.ng_liq_sc)
    liq_curves_sc.columns = liq_curves_sc.columns.tolist()[:-gp.real_max_months-1] + num_cols_taux

    tci_values = ex.get_dataframe_from_range(wb, gp.ng_tx_tci)
    tci_values = tci_values.fillna("*").drop(["dar", "all_t"], axis=1)

    cle = ["reseau", "company_code", "devise", "contract_type", "family", "rate_category"]
    tci_values['new_key'] = tci_values[cle].apply(lambda row: '$'.join(row.values.astype(str)), axis=1)
    tci_values = tci_values.drop_duplicates(subset=cle).set_index('new_key').drop(columns=cle, axis=1).copy()

    if up.nb_mois_proj_usr > gp.real_max_months:
        tx_curves_sc = gf.prolong_last_col_value(tx_curves_sc, up.nb_mois_proj_usr - gp.real_max_months, suf="M", s=4)
        tx_curves_rco = gf.prolong_last_col_value(tx_curves_rco, up.nb_mois_proj_usr - gp.real_max_months,suf="M", s=4)
        liq_curves_sc = gf.prolong_last_col_value(liq_curves_sc, up.nb_mois_proj_usr - gp.real_max_months,suf="M", s=4)

    """ COURBES ZC """
    logger.info("             INTERPOLATION DES COURBES DE ZC")
    if "ech" in cls_pn_loader.dic_pn_ech or "ech%" in cls_pn_loader.dic_pn_ech or gf.begin_in_list(up.indic_sortie_eve["ST"],gp.eve_ef_sti)\
            or gf.begin_in_list(up.indic_sortie_eve["ST"], gp.eve_em_sti) \
            or gf.begin_in_list(up.indic_sortie_eve["PN"], gp.eve_ef_pni)\
            or gf.begin_in_list(up.indic_sortie_eve["PN"], gp.eve_em_pni):

        zc_curves_df = ex.get_dataframe_from_range(wb, gp.ng_boot_curves, header=False, alert="")

        if gf.begin_in_list(up.indic_sortie_eve["ST"], gp.eve_ef_sti) \
                or gf.begin_in_list(up.indic_sortie_eve["ST"], gp.eve_em_sti) \
                or gf.begin_in_list(up.indic_sortie_eve["PN"], gp.eve_ef_pni) \
                or gf.begin_in_list(up.indic_sortie_eve["PN"], gp.eve_em_pni):

            get_max_duree()
            zc_all_curves= []
            get_delta_days_from_forward_month()
            """ 1.1. INTERPOLATION """
            for curve_zc in zc_curves_df.iloc[:, 0].unique():
                data = zc_curves_df[zc_curves_df.iloc[:, 0] == curve_zc]
                """ Interpolation sur la durée max des produits ech pour le calcul de l'EVE """
                data = interpol_zc_data(data)
                data[gp.nc_pricing_curve] = curve_zc
                if len(zc_all_curves)>=1:
                    zc_all_curves = pd.concat([zc_all_curves, data])
                else:
                    zc_all_curves = data.copy()

            """ 1.3. AJOUT d'un ZC de 0 pour les taux variable"""
            tv_zc_curve= data * 0
            tv_zc_curve[gp.nc_pricing_curve] = "TVZERO"
            zc_all_curves = pd.concat([zc_all_curves, tv_zc_curve])
    else:
        zc_all_curves = []
        zc_curves_df = []

    wb.Close(False)


def interpolate_inf_curve():
    inf_curves_interpol = tx_curves_sc[tx_curves_sc[gp.NC_TYPE_SC_TX]=="INFLATION"].copy()
    """ Transformation des indices annuels en mois"""
    inf_curves_interpol["duree"] = [12 * int(x.replace("INF", "")) for x in inf_curves_interpol[gp.NC_CODE_SC_TX]]
    inf_curves_interpol = inf_curves_interpol.drop([gp.NC_TYPE_SC_TX, gp.NC_CODE_SC_TX, gp.NC_DEVISE_SC_TX], axis=1)

    """ Détermination des mois manquants"""
    _min = 1;
    _max = gp.max_months2
    list_missing = [j for j in range(_min, _max + 1) if j not in inf_curves_interpol["duree"].values.tolist()]

    """ Index correspondant aux mois manquants"""
    list_interpol = [int(min(max(1, round(j / 12, 0)), 30)) - 1 for j in list_missing]
    """ Lignes correspondant aux index manquants"""
    inf_curves_interpol_plus = np.array(inf_curves_interpol)[list_interpol].copy()
    """ Ajout des durées"""
    inf_curves_interpol_plus[:, -1] = list_missing
    """ Concaténation et classement"""
    inf_curves_interpol = np.vstack([inf_curves_interpol_plus, np.array(inf_curves_interpol)])
    inf_curves_interpol = inf_curves_interpol[np.argsort(inf_curves_interpol[:, -1])]
    return inf_curves_interpol


def fill_and_save_sc_compil(save_path, name_file, scen_name, bassin):
    """ Fonction permettant de générer le compil SC"""
    cols = gp.compil_header
    cols = cols[:-gp.max_months-1] + cols[-gp.max_months-1:][:up.nb_mois_proj_out+1]
    filter = (tx_curves_sc[gp.NC_TYPE_SC_TX]=="SWAP") & (tx_curves_sc[gp.NC_CODE_SC_TX].str.contains("|".join(mp.sc_file_indexes)))
    filter = filter | (tx_curves_sc[gp.NC_CODE_SC_TX].isin(mp.sc_file_indexes))

    cols_num = [x for x in num_cols_taux if int(str(x)[1:])<=up.nb_mois_proj_out]
    tx_out = tx_curves_sc[filter][[gp.NC_CODE_SC_TX, gp.NC_DEVISE_SC_TX] + cols_num].copy()
    tx_out["IND01"] = "SC"
    tx_out["IND02"] = "TX"
    tx_out = tx_out.rename(columns={gp.NC_CODE_SC_TX:gp.nc_output_ind3, gp.NC_DEVISE_SC_TX:gp.nc_output_devise_cle})
    tx_out["SC"] = scen_name
    tx_out["BASSIN"] = bassin
    for col in cols:
        if col not in tx_out.columns:
            tx_out[col] = ""

    tx_out[cols].to_csv(save_path + "\\" + name_file, index=False, decimal=",", sep=";")
