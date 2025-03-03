# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:12:56 2020

@author: TAHIRIH
"""
import modules.moteur.parameters.general_parameters as gp
from modules.moteur.utils import generic_functions as gf
import logging
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)
class Rates_Manager():
    
    def __init__(self, cls_usr, cls_cal):
        self.cal = cls_cal
        self.up = cls_usr
        
    def get_max_duree(self, ):
        if gf.begin_in_list(self.up.indic_sortie_eve["PN"], gp.eve_ef_pni) or gf.begin_in_list(self.up.indic_sortie_eve["PN"], gp.eve_em_pni) \
                or gf.begin_in_list(self.up.indic_sortie_eve["ST"], gp.eve_ef_sti) or gf.begin_in_list(self.up.indic_sortie_eve["ST"],
                                                                                              gp.eve_em_sti):
            self.max_duree = min(self.up.nb_mois_proj_usr + 1, gp.max_months + 1)
            self.max_month_zc = min(self.up.nb_mois_proj_usr + 1, gp.pn_max + 1)
    
        if self.max_duree > gp.max_months2:
            error = "Not enough days in base calc. Some products have a duration greater than of 600-" + str(self.max_duree)
            logger.error(error)
            raise ValueError(error)
    
    def get_delta_days_from_forward_month(self, ):
        dates_fin_per = np.array(self.cal.calendar_coeff[self.cal.calendar_coeff[gp.nc_pn_base_calc] == "Dcold"][
                             [x for x in self.cal.calendar_coeff.columns if x != gp.nc_pn_base_calc]])
        dates_fin_per = dates_fin_per[:, :self.max_duree]
        list_delta_days = []
        for j in range(0, self.max_month_zc):
            delta_days = (dates_fin_per - dates_fin_per[:, j]).astype('timedelta64[D]').reshape(self.max_duree, )
            delta_days = np.maximum(0, delta_days / np.timedelta64(1, 'D'))
            list_delta_days.append(delta_days)
    
        self.delta_days = np.column_stack(list_delta_days)
    
    def interpol_zc_data(self, data_boot):
        data_boot = np.array(data_boot)[:, 1:]
        jr = data_boot[:, 3::2].astype(np.float64)  # On élimine le mois 0
        zc = data_boot[:, 4::2].astype(np.float64)
        listo = []
        for j in range(0, self.max_month_zc):
            delta_days_j = self.delta_days[:,j]
            zc_interpolated = np.interp(delta_days_j, jr[:, j], zc[:, j], left=0).reshape(self.max_duree)
            zc_interpolated = np.where(delta_days_j==0, 0, zc_interpolated)
            listo.append(zc_interpolated)
    
        data = pd.DataFrame(np.column_stack(listo), columns=["M" + str(i) for i in range(1, self.max_month_zc + 1)])
    
        return data
    
    def load_tx_curves(self, files_sc, scen_path, cls_pn_loader):
        """ Fonction permettant de charger les courbes de taux scénario"""
        g_path = os.path.join(scen_path, "SC_TAUX")
        header_tx = gp.List_TX_HEADER
    
        self.tx_tv_zero = pd.DataFrame(0, index=np.arange(1), columns=header_tx)
        self.tx_tv_zero[gp.nc_taux] = gp.label_tv_zero
    
        """ Courbes de taux """
        self.tx_curves_sc = pd.read_csv(os.path.join(g_path, files_sc["TAUX"][gp.sc_tx_tag][0]), sep=";", decimal=",")
        self.liq_curves_sc = pd.read_csv(os.path.join(g_path, files_sc["TAUX"][gp.sc_liq_tag][0]), sep=";", decimal=",")
        self.tx_curves_rco = pd.read_csv(os.path.join(g_path, files_sc["TAUX"][gp.sc_rco_ref_tag][0]), sep=";",
                                         decimal=",")
        self.tci_data = pd.read_csv(os.path.join(g_path, files_sc["TAUX"][gp.sc_tci_tag][0]), sep=";", decimal=",")
    
        if self.up.nb_mois_proj_usr > gp.real_max_months:
            self.tx_curves_sc = gf.prolong_last_col_value(self.tx_curves_sc, self.up.nb_mois_proj_usr - gp.real_max_months, suf="M", s=6)
            self.tx_curves_rco = gf.prolong_last_col_value(self.tx_curves_rco, self.up.nb_mois_proj_usr - gp.real_max_months,suf="M", s=6)
            self.liq_curves_sc = gf.prolong_last_col_value(self.liq_curves_sc, self.up.nb_mois_proj_usr - gp.real_max_months,suf="M", s=6)
    
        """ COURBES ZC """
        logger.info("             INTERPOLATION DES COURBES DE ZC")
        if "ech" in cls_pn_loader.dic_pn_ech or "ech%" in cls_pn_loader.dic_pn_ech or gf.begin_in_list(self.up.indic_sortie_eve["ST"],gp.eve_ef_sti)\
                or gf.begin_in_list(self.up.indic_sortie_eve["ST"], gp.eve_em_sti) \
                or gf.begin_in_list(self.up.indic_sortie_eve["PN"], gp.eve_ef_pni)\
                or gf.begin_in_list(self.up.indic_sortie_eve["PN"], gp.eve_em_pni):
    
            self.zc_curves_df = pd.read_csv(os.path.join(g_path, files_sc["TAUX"][gp.sc_zc_tag][0]), sep=";", decimal=",")
    
            if gf.begin_in_list(self.up.indic_sortie_eve["ST"], gp.eve_ef_sti) \
                    or gf.begin_in_list(self.up.indic_sortie_eve["ST"], gp.eve_em_sti) \
                    or gf.begin_in_list(self.up.indic_sortie_eve["PN"], gp.eve_ef_pni) \
                    or gf.begin_in_list(self.up.indic_sortie_eve["PN"], gp.eve_em_pni):
    
                self.get_max_duree()
                self.zc_all_curves= []
                self.get_delta_days_from_forward_month()
                """ 1.1. INTERPOLATION """
                for curve_zc in self.zc_curves_df.iloc[:, 0].unique():
                    data = self.zc_curves_df[self.zc_curves_df.iloc[:, 0] == curve_zc]
                    """ Interpolation sur la durée max des produits ech pour le calcul de l'EVE """
                    data = self.interpol_zc_data(data)
                    data[gp.nc_pricing_curve] = curve_zc
                    if len(self.zc_all_curves)>=1:
                        self.zc_all_curves = pd.concat([self.zc_all_curves, data])
                    else:
                        self.zc_all_curves = data.copy()

                """ 1.3. AJOUT d'un ZC de 0 pour les taux variable"""
                tv_zc_curve= data * 0
                tv_zc_curve[gp.nc_pricing_curve] = "TVZERO"
                self.zc_all_curves = pd.concat([self.zc_all_curves, tv_zc_curve])
        else:
            self.zc_all_curves = []
            self.zc_curves_df = []
    
