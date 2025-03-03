import modules.moteur.utils.generic_functions as gf
import modules.moteur.parameters.general_parameters as gp
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EVE_Calculator():
    def __init__(self, cls_usr, cls_mp, cls_tx):
        self.up = cls_usr
        self.mp = cls_mp
        self.tx = cls_tx

    def calcul_eve(self, dic_sci, main_data, typo, cols=[]):
        """ Module permettant de calculer l'EVE """

        indic_sortie = self.get_indic_sorties(typo)

        if len(indic_sortie) > 0:
            eve_em, fk_em, fi_em, gap_em, deb, fk_em_act, fi_em_act, gap_em_liq = self.choose_indicators(typo)

            if typo == "AJUST":
                main_data, deb = self.format_data_ajust(dic_sci, cols)

            is_ra_rn = self.get_rarn_immo_perimeter(main_data, typo)

            ht = main_data.shape[0]

            if gf.begin_in_list(indic_sortie, eve_em) or gf.begin_in_list(indic_sortie, fk_em_act) \
                    or gf.begin_in_list(indic_sortie, fi_em_act):
                """ 1. On génère les taux d'actualisation à partir des taux boostrap"""
                zc_ht = self.tx.max_month_zc
                if typo == "ST":
                    devises_data = main_data[[gp.nc_output_devise_cle]].copy().reset_index(
                        level=[gp.nc_output_contrat_cle])
                elif typo == "PN":
                    devises_data = main_data[[gp.nc_output_devise_cle, gp.nc_output_contrat_cle]].copy()
                else:
                    devises_data = main_data[[gp.nc_output_devise_cle]].copy()
                    devises_data[gp.nc_output_contrat_cle] = "."
                devises_data[gp.nc_output_bilan] = np.where(
                    devises_data[gp.nc_output_contrat_cle].str[:3].isin(["AHB", "PHB"]),
                    "HB", "B")
                devises_data[gp.nc_output_contrat_cle] \
                    = np.where(devises_data[gp.nc_output_contrat_cle].str[:6].isin(["AHB-NS", "PHB-NS"]),
                               "AHB-NS*,PHB-NS*",
                               "*")

                list_devises = self.mp.mapping_eve["act_eve"].index.get_level_values("DEVISE").values
                default = ~devises_data[gp.nc_output_devise_cle].isin(list_devises)
                devises_data.loc[default, gp.nc_output_devise_cle] = "*"

                zc_taux = self.get_act_curve(devises_data, ht, zc_ht)

            for p in range(deb, self.up.nb_mois_proj_out + 1):
                M = "M0" if p <= 9 and typo == "AJUST" else "M"

                """ FLUX CAPITAL"""
                if eve_em + str(p) in indic_sortie or fk_em + str(p) in indic_sortie:
                    self.calc_flux_cap(dic_sci, main_data, fk_em, gap_em, gap_em_liq, is_ra_rn, p, typo, M)

                """ FLUX INTERET"""
                if eve_em + str(p) in indic_sortie or fi_em + str(p) in indic_sortie:
                    mni_p = self.format_mni(dic_sci, typo, p)

                if eve_em + str(p) in indic_sortie or fi_em + str(p) in indic_sortie:
                    self.calc_flux_int(dic_sci, gap_em, fi_em, main_data, p, typo, mni_p, M)

                """ EVE"""
                if eve_em + str(p) in indic_sortie or fk_em_act + str(p) in indic_sortie \
                        or fi_em_act + str(p) in indic_sortie:
                    """ ACTUALISATION """
                    tx_act = self.calc_tx_actu(zc_taux, zc_ht, ht, p, typo)

                if eve_em + str(p) in indic_sortie:
                    self.calc_eve(dic_sci, main_data, eve_em, fi_em, fk_em, tx_act, p, M, typo, gap_em)

                if fk_em_act + str(p) in indic_sortie:
                    self.calc_flux_cap_act(dic_sci, fk_em_act, fk_em, tx_act, p)

                if fi_em_act + str(p) in indic_sortie:
                    self.calc_flux_int_act(dic_sci, fi_em_act, fi_em, tx_act, p)

            if typo == "AJUST":
                self.concat_eve_with_ajust_data(dic_sci, main_data)

    def get_rarn_immo_perimeter(self, main_data, typo):
        if typo == "ST":
            n = main_data.shape[0]
            is_ra_rn = (main_data.index.get_level_values(gp.nc_output_contrat_cle).isin(gp.CONTRATS_RA_RN_IMMO))
            is_ra_rn = (is_ra_rn) | \
                       ((main_data.index.get_level_values(gp.nc_output_contrat_cle).isin(gp.CONTRATS_RA_RN_IMMO_CSDN))
                        & (main_data[gp.nc_output_etab_cle] == "CSDN"))
            is_ra_rn = is_ra_rn & (main_data[gp.nc_output_index_calc_cle].str.contains(gp.FIX_ind))
            is_ra_rn = is_ra_rn.reset_index(level=[gp.nc_output_palier_cle, gp.nc_output_contrat_cle], drop=True)
        elif typo == "PN":
            n = main_data.shape[0]
            is_ra_rn = (main_data[gp.nc_output_contrat_cle].isin(gp.CONTRATS_RA_RN_IMMO))
            is_ra_rn = (is_ra_rn) | \
                       ((main_data[gp.nc_output_contrat_cle].isin(gp.CONTRATS_RA_RN_IMMO_CSDN))
                        & (main_data[gp.nc_output_etab_cle] == "CSDN"))
            is_ra_rn = is_ra_rn & (main_data[gp.nc_output_index_calc_cle].str.contains(gp.FIX_ind))
            is_ra_rn = is_ra_rn.reset_index(drop=True)

        else:
            is_ra_rn = np.full(main_data.shape[0], False)

        is_ra_rn = is_ra_rn & np.full(is_ra_rn.shape[0], self.mp.mapping_eve["mode_cal_gap_tx_immo"] != "GAP TAUX")

        return np.array(is_ra_rn).reshape(main_data.shape[0], 1)

    def make_zero_for_specific_contracts(self, main_data, dic_ind, indic, typo):
        contracts_to_exclude = self.mp.mapping_eve["eve_contracts_excl"].index.tolist()
        if "CAP" in indic:
            contracts_to_exclude = contracts_to_exclude + ["AHB-CAP", "PHB-CAP", "AHB-FLOOR",
                                                           "PHB-FLOOR"]  # NOMINAL des CAP/FLOOR n'est un encours

        if typo != "AJUST":
            if typo != "PN":
                filter_contract = (main_data.index.get_level_values(gp.nc_output_contrat_cle).isin(contracts_to_exclude)
                                   .reshape(main_data.shape[0], 1))
            else:
                filter_contract = (np.array(main_data[gp.nc_output_contrat_cle].isin(contracts_to_exclude))
                                   .reshape(main_data.shape[0], 1))

            dic_ind[indic] = pd.DataFrame(data=np.where(filter_contract, 0, dic_ind[indic]), index=dic_ind[indic].index,
                                          columns=dic_ind[indic].columns)

    def format_data_ajust(self, dic_sci, cols):
        main_data = dic_sci[gp.em_eve_pni][[x for x in dic_sci[gp.em_eve_pni].columns if x not in cols]].copy()
        deb = 0
        for indic in dic_sci:
            dic_sci[indic] = dic_sci[indic][cols].reset_index(drop=True)
        return main_data, deb

    def get_indic_sorties(self, typo):
        if typo == "ST":
            indic_sortie = self.up.indic_sortie_eve["ST"]
        else:
            indic_sortie = self.up.indic_sortie_eve["PN"] if typo == "PN" else self.up.indic_sortie_eve["AJUST"]

        return indic_sortie

    def choose_indicators(self, typo):
        if typo == "ST":
            eve_em = gp.eve_em_sti
            fk_em = gp.fk_em_sti
            fi_em = gp.fi_em_sti
            fk_em_act = gp.fk_em_act_sti
            fi_em_act = gp.fi_em_act_sti
            if self.up.type_eve == "ICAAP":
                gap_em = gp.gp_liq_em_eve_sti
            else:
                gap_em = gp.gpr_em_eve_sti
            gap_em_liq = gp.gp_liq_em_eve_sti
            deb = 0
        else:
            eve_em = gp.eve_em_pni
            fk_em = gp.fk_em_pni
            fi_em = gp.fi_em_pni
            fk_em_act = gp.fk_em_act_pni
            fi_em_act = gp.fi_em_act_pni
            if self.up.type_eve == "ICAAP":
                gap_em = gp.gp_liq_em_eve_pni
            else:
                gap_em = gp.gpr_em_eve_pni
            gap_em_liq = gp.gp_liq_em_eve_pni
            deb = 1
        return eve_em, fk_em, fi_em, gap_em, deb, fk_em_act, fi_em_act, gap_em_liq

    def concat_eve_with_ajust_data(self, dic_sci, main_data):
        for indic in dic_sci:
            descriptive_data = main_data.reset_index(drop=True).copy()
            descriptive_data[gp.nc_output_ind3] = indic
            dic_sci[indic] = pd.concat([descriptive_data, dic_sci[indic]], axis=1)

    def calc_eve(self, dic_sci, main_data, eve_ind, fi_ind, fk_ind, tx_act, p, M, typo, gap_ind):
        dic_sci[eve_ind + str(p)] = tx_act * (dic_sci[fi_ind + str(p)] + dic_sci[fk_ind + str(p)])
        dic_sci[eve_ind + str(p)][M + str(p)] = dic_sci[gap_ind + str(p)][M + str(p)]  # POINT DE DEPART
        self.make_zero_for_specific_contracts(main_data, dic_sci, eve_ind + str(p), typo)

    def calc_flux_cap_act(self, dic_sci, fk_act_ind, fk_ind, tx_act, p):
        dic_sci[fk_act_ind + str(p)] = tx_act * (dic_sci[fk_ind + str(p)])

    def calc_flux_int_act(self, dic_sci, fi_act_ind, fi_ind, tx_act, p):
        dic_sci[fi_act_ind + str(p)] = tx_act * (dic_sci[fi_ind + str(p)])

    def calc_tx_actu(self, zc_taux, zc_ht, ht, p, typo):
        delta_days = self.tx.delta_days.copy()[:self.up.nb_mois_proj_usr + 1].reshape(1, self.up.nb_mois_proj_usr + 1, zc_ht)
        taux_floor_reg = self.load_tax_reg()

        tx_act = (1 / ((1 + np.maximum(zc_taux[:, :, p], taux_floor_reg)) ** (
                delta_days[:, :, p] / 365))).reshape(ht, self.up.nb_mois_proj_usr + 1)
        if typo == "PN":
            tx_act = tx_act[:, 1:]

        return tx_act

    def format_mni(self, dic_sci, typo, p):
        if typo in ["PN", "AJUST"]:
            if self.up.type_eve == "ICAAP":
                mni_p = dic_sci[gp.mn_gpliq_pni + str(p)]
            else:
                mni_p = dic_sci[gp.mn_gpr_pni + str(p)]
        else:
            cols = ["M" + str(i) for i in range(1, self.up.nb_mois_proj_usr + 1)]
            # PAS BON, il faut aussi translater la MNI
            if self.up.type_eve == "ICAAP":
                mni_p = dic_sci[gp.mn_sti][cols]
                mni_p = mni_p.fillna(0)
            else:
                mni_p = dic_sci[gp.mn_gp_rg_sti].copy()

            mni_p.insert(0, "M0", 0)
        return mni_p

    def load_tax_reg(self, ):
        taux_floor_reg = np.column_stack([np.array([self.mp.mapping_eve["eve_reg_floor"][0, 0]]).reshape(1, 1), \
                                          np.array(self.mp.mapping_eve["eve_reg_floor"])])
        taux_floor_reg = taux_floor_reg[:, :self.up.nb_mois_proj_usr + 1].reshape(1, self.up.nb_mois_proj_usr + 1)
        return taux_floor_reg

    def get_act_curve(self, devises_data, ht, zc_ht):
        devises_data = devises_data.set_index([gp.nc_output_devise_cle, gp.nc_output_bilan,
                                               gp.nc_output_contrat_cle])
        pricing_curve_inner = devises_data.join(self.mp.mapping_eve["act_eve"], how="inner").iloc[:, -1].copy()
        if len(pricing_curve_inner) != len(devises_data):
            list_prob = devises_data[devises_data.isin(pricing_curve_inner.index.tolist())].index
            logger.error("Certaines produits n'ont pas de courbes d'actualisation  : %s" % list_prob)
            raise ValueError(list_prob)

        pricing_curve = devises_data.join(self.mp.mapping_eve["act_eve"], how="left",
                                          on=[gp.nc_output_devise_cle, gp.nc_output_bilan,
                                              gp.nc_output_contrat_cle]).iloc[:, -1:].copy()
        nc_act_eve_act_curve = pricing_curve.columns.tolist()[0]

        zc_curve_all = self.tx.zc_all_curves.copy()
        absent_curve = \
            pricing_curve[
                ~pricing_curve[nc_act_eve_act_curve].isin(zc_curve_all[gp.nc_pricing_curve].values.tolist())][
                nc_act_eve_act_curve]
        if len(absent_curve):
            list_prob = absent_curve.values.tolist()
            logger.error("Certaines courbes d'actualisation n'existent pas : %s" % list_prob)
            raise ValueError(list_prob)

        zc_curve = \
            pricing_curve.merge(zc_curve_all, how="left", right_on=[gp.nc_pricing_curve],
                                left_on=[nc_act_eve_act_curve])[
                [x for x in zc_curve_all if x != gp.nc_pricing_curve]].copy()
        zc_curve = np.array(zc_curve).reshape(ht, int(zc_curve.shape[0] / ht), zc_ht)[:, :self.up.nb_mois_proj_usr + 1,
                   :]

        return zc_curve

    def calc_flux_int(self, dic_sci, gap_ind, fi_ind, main_data, p, typo, mni_p, M):
        dummy_gap_ef = np.where(dic_sci[gap_ind + str(p)] != 0, 1, 0)
        dic_sci[fi_ind + str(p)] = mni_p * dummy_gap_ef
        dic_sci[fi_ind + str(p)].loc[:, :M + str(p)] = 0
        self.make_zero_for_specific_contracts(main_data, dic_sci, fi_ind + str(p), typo)

    def calc_flux_cap(self, dic_sci, main_data, flux_cap, gap_ind, gap_em_liq, is_ra_rn, p, typo, M):
        gap_rg = pd.DataFrame(np.where(is_ra_rn, dic_sci[gap_em_liq + str(p)], dic_sci[gap_ind + str(p)]),
                              index=dic_sci[gap_ind + str(p)].index, columns=dic_sci[gap_ind + str(p)].columns)

        dic_sci[flux_cap + str(p)] = gap_rg.shift(1, axis=1) - gap_rg

        dic_sci[flux_cap + str(p)][M + str(p)] = 0
        dic_sci[flux_cap + str(p)]["M" + str(self.up.nb_mois_proj_usr)] = gap_rg["M" + str(self.up.nb_mois_proj_usr - 1)]

        self.make_zero_for_specific_contracts(main_data, dic_sci, flux_cap + str(p), typo)
