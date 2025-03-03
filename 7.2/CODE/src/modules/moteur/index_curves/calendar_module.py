# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:04:35 2020

@author: TAHIRIH
"""
import modules.moteur.parameters.general_parameters as gp
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import logging

logger = logging.getLogger(__name__)


class Calendar_Manager():
    def __init__(self, cls_usr, cls_mp):
        self.up = cls_usr
        self.mp = cls_mp

    def apply_30E_360_calc(self, x, y):
        return (360 * y.year + 30 * y.month + min(30, y.day) - (360 * x.year + 30 * x.month + min(30, x.day))) / 360

    def apply_30_360_calc(self, x, y):
        nb_days_x = min(30, x.day)
        nb_days_y = 30 if (x.day == 30 and y.day == 31) else y.day
        return (360 * y.year + 30 * y.month + nb_days_y - (360 * x.year + 30 * x.month + nb_days_x)) / 360

    def load_calendar_coeffs(self):
        start = self.up.dar
        period = gp.max_months2
        c1 = pd.DataFrame(pd.date_range(start=start, periods=period, freq='MS'))
        c2 = pd.DataFrame(pd.date_range(start=start + relativedelta(months=1), periods=period, freq='MS'))
        c3 = pd.DataFrame(
            pd.date_range(start=start + relativedelta(months=1), periods=period, freq='MS') + pd.DateOffset(days=14))
        calendar_coeff = pd.concat([c1, c2, c3], axis=1)
        calendar_coeff.columns = ["Dcold", "Dcolf", "MaDate15"]
        calendar_coeff['DCoeff'] = gp.avg_nb_days_month / (calendar_coeff.Dcolf - calendar_coeff.Dcold).dt.days

        calendar_coeff["NB_DAYS"] = [int(x) for x in (calendar_coeff.Dcolf - calendar_coeff.Dcold).dt.days]

        calendar_coeff["ACT/360"] = (calendar_coeff.Dcolf - calendar_coeff.Dcold).dt.days / 360
        calendar_coeff["ACT/365"] = (calendar_coeff.Dcolf - calendar_coeff.Dcold).dt.days / 365
        calendar_coeff["ACT/ACT"] = [
            (y - x).days / 366 if (x + relativedelta(years=1) - x).days == 366 else (y - x).days / 365 for x, y in
            zip(calendar_coeff.Dcold, calendar_coeff.Dcolf)]

        calendar_coeff["30E/360"] = [self.apply_30E_360_calc(x, y) for x, y in
                                     zip(calendar_coeff.Dcold, calendar_coeff.Dcolf)]

        calendar_coeff["30/360"] = [self.apply_30_360_calc(x, y) for x, y in
                                    zip(calendar_coeff.Dcold, calendar_coeff.Dcolf)]

        self.calendar_coeff = calendar_coeff.transpose().reset_index()

        self.calendar_coeff.columns = [gp.nc_pn_base_calc] + ["M" + str(i) for i in range(1, period + 1)]

    def generate_base_calc_coeff_stock(self, data_stock, type_tx_stock):
        """ JOINT LES COEFFS BASE CALC POUR CHAQUE CONTRAT STOCK"""

        """ SELECTIONNE LES COLONNES POUR LE NB DE MOIS DE PROJ """
        coeff_calendar = self.calendar_coeff.copy()[
            [gp.nc_pn_base_calc] + ["M" + str(i) for i in range(1, self.up.nb_mois_proj_usr + 1)]]

        """ CONCATENATION DES PARAM BASE CALC NMD ET ECH"""
        cols = [gp.nc_nmd_basec_tf, gp.nc_nmd_basec_tv]
        base_calc_col = self.mp.param_tx_ech[cols].combine_first(self.mp.param_nmd_base_calc[cols])

        """ JOINTURE AVEC LES CONTRATS DES DONNEES DE STOCK"""
        base_calc_col = data_stock.join(base_calc_col, on=gp.nc_output_contrat_cle, how="left")[cols].copy()
        base_calc_col[gp.nc_pn_base_calc] = np.where(type_tx_stock == "TF", base_calc_col[gp.nc_nmd_basec_tf],
                                                     base_calc_col[gp.nc_nmd_basec_tv])

        """ JOINTURES DES COEFFS AVEC LES CONTRATS """
        base_calc_coeff = base_calc_col.merge(how="left", right=coeff_calendar, on=gp.nc_pn_base_calc)

        list_missing = base_calc_col.reset_index()[base_calc_col.reset_index()[gp.nc_pn_base_calc].isna()][
            gp.nc_output_contrat_cle].values.tolist()
        if list_missing != []:
            logger.warning(
                "    The following base calc are missing for the following contracts : %s" % list(set(list_missing)))

        base_calc_coeff = base_calc_coeff[[x for x in base_calc_coeff.columns if x not in cols + [gp.nc_pn_base_calc]]]

        return base_calc_coeff.astype(np.float64)
