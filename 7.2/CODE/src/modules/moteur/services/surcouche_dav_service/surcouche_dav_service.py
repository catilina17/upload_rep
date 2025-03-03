import numpy as np
import pandas as pd
import warnings
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
from calculateur.models.data_manager.data_format_manager.class_fields_manager import Data_Fields
from modules.moteur.parameters import user_parameters as up
from modules.moteur.parameters import general_parameters as gp
from utils import excel_openpyxl as ex
import utils.general_utils as ut
import logging

np.seterr(divide='ignore', invalid='ignore')
logger = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class DAV_Calculator():
    def __init__(self, cls_usr, cls_calc_params, etab, scenario_name):
        self.up = cls_usr
        self.etab = etab
        self.scenario_name = scenario_name
        self.params = cls_calc_params
        self.calculated_stock = []
        self.data_stock_nmd_dt = []
        self.IND01 = "IND01"
        self.TAG_ST = "ST"
        self.TAG_PN = "PN"
        self.TAG_PN_PRCT = "PN"
        self.INDEX_FIXE = "FIXE"
        self.num_cols = ["M" + str(i) for i in range(0, self.up.nb_mois_proj_usr + 1)]
        self.cls_fields = Data_Fields()

    def add_dav_surcouche(self, cls_pn_loader, data_stock, cls_stock_calc):
        scenarii = self.get_scenarios(self.etab, self.scenario_name, self.params.out_of_recalc_st_etabs, "DAV")
        data_stock_calc = cls_stock_calc.calculated_stock.copy()
        data_stock_calc_det = cls_stock_calc.data_stock_nmd_dt.copy()
        if len(scenarii) > 0:
            wb_dav = self.params.get_source_models(self.scenario_name, "DAV")
            src_dav_data = self.get_models_data(wb_dav, scenarii["NOM SCENARIO"].iloc[0])
            if self.TAG_ST in src_dav_data[self.IND01].values.tolist():
                logger.info('        Début du traitement du scénario de surcouche DAV pour le STOCK')
                scenario = src_dav_data[src_dav_data[self.IND01] == self.TAG_ST].copy()
                if len(data_stock_calc) > 0:
                    is_st_nmd = data_stock_calc[pa.NC_PA_CONTRACT_TYPE].str.contains("P-DAV").any()
                    if is_st_nmd:
                        stock_dav_data = self.load_stock_dav_data(data_stock_calc, scenario)
                        stock_dav_data_sc = self.create_src_dave_data(stock_dav_data, scenario)
                        data_stock_calc = self.update_stock_with_dav_stock(data_stock_calc, stock_dav_data_sc)

                        data_stock_calc_det = self.update_detailed_stock(stock_dav_data, stock_dav_data_sc, data_stock_calc_det)


                stock_dav_data = self.load_stock_dav_data(data_stock, scenario)
                stock_dav_data = self.create_src_dave_data(stock_dav_data, scenario)
                data_stock = self.update_stock_with_dav_stock(data_stock, stock_dav_data)

            if self.TAG_PN in src_dav_data["IND01"].values.tolist():
                if "nmd%" in cls_pn_loader.dic_pn_nmd and len(cls_pn_loader.dic_pn_nmd["nmd%"]) > 0:
                    logger.info('        Début du traitement du scénario de surcouche DAV pour la PN NMD%')
                    scenario = src_dav_data[src_dav_data["IND01"] == "ST"].copy()
                    pn_data = cls_pn_loader.dic_pn_nmd["nmd%"].copy()
                    if pn_data.shape[0] > 0:
                        pn_data_dav, pn_data_no_dav = self.divide_data_set_dav(pn_data, scenario)
                        pn_data_dav = self.create_src_dave_data(pn_data_dav, scenario, typo=self.TAG_PN_PRCT)
                        cls_pn_loader.dic_pn_nmd["nmd%"] = pd.concat([pn_data_no_dav, pn_data_dav])

                if "nmd" in cls_pn_loader.dic_pn_nmd and len(cls_pn_loader.dic_pn_nmd["nmd"]) > 0:
                    logger.info('        Début du traitement du scénario de surcouche DAV pour la PN NMD')
                    scenario = src_dav_data[src_dav_data["IND01"] == "PN"].copy()
                    pn_data = cls_pn_loader.dic_pn_nmd["nmd"].copy()
                    if pn_data.shape[0] > 0:
                        pn_data_dav, pn_data_no_dav = self.divide_data_set_dav(pn_data, scenario)
                        pn_data_dav = self.create_src_dave_data(pn_data_dav, scenario, typo=self.TAG_PN)
                        cls_pn_loader.dic_pn_nmd["nmd"] = pd.concat([pn_data_no_dav, pn_data_dav])

            ex.close_workbook(wb_dav)

        return data_stock, data_stock_calc, data_stock_calc_det

    def update_detailed_stock(self, stock_dav_data, stock_dav_data_sc, data_stock_calc_det):
        stock_dav_data_sc["RATE_CATEGORY"] = np.where(stock_dav_data_sc[pa.NC_PA_RATE_CODE].str.contains(gp.FIX_ind),
                                                      "FIXED", "FLOATING")
        filter_lef_c = (stock_dav_data_sc[pa.NC_PA_IND03] == pa.NC_PA_LEF).values
        filter_lef = (stock_dav_data[pa.NC_PA_IND03] == pa.NC_PA_LEF).values
        stock_dav_data_sc_lef = stock_dav_data_sc[filter_lef_c].copy()

        stock_dav_data_sc_lef["COEFF_DAV_CONST"] = np.nan_to_num(
            stock_dav_data_sc_lef["M0"].values / stock_dav_data.loc[filter_lef, "M0"].values, nan=0.0, posinf=0.0,
            neginf=0.0)
        coeff_dav_var = ["COEFF_DAV_" + str(i) for i in range(0, len(self.num_cols))]
        stock_dav_data_sc_lef[coeff_dav_var] \
            = np.nan_to_num(
            stock_dav_data_sc_lef[self.num_cols].values / stock_dav_data.loc[filter_lef, self.num_cols].values, nan=0.0,
            posinf=0.0, neginf=0.0)

        stock_dav_data_lef_sc_cf = \
        stock_dav_data_sc_lef.set_index([pa.NC_PA_ETAB, pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE, "RATE_CATEGORY"])[
            coeff_dav_var + ["COEFF_DAV_CONST", "TYPE MODELE"]]
        data_stock_calc_det_tmp = data_stock_calc_det.copy()
        data_stock_calc_det_tmp = data_stock_calc_det_tmp.join(stock_dav_data_lef_sc_cf,
                                                               on=[self.cls_fields.NC_LDP_ETAB, self.cls_fields.NC_LDP_CONTRACT_TYPE,
                                                                   self.cls_fields.NC_LDP_MARCHE,
                                                                   self.cls_fields.NC_LDP_RATE_TYPE])
        if len(data_stock_calc_det_tmp) != len(data_stock_calc_det):
            msg = "Il y a des problèmes de jointure entre le stock détaillé et le stock agrégé pour les DAV après surcouche"
            logger.error(msg)
            raise ValueError(msg)
        n = data_stock_calc_det_tmp.shape[0]
        is_const_model = (data_stock_calc_det_tmp["TYPE MODELE"] != "SURCOUCHE CONSTANTE").values.reshape(n, 1)
        ecoul_cnst = data_stock_calc_det_tmp[self.num_cols].values * data_stock_calc_det_tmp["COEFF_DAV_CONST"].fillna(
            1).values.reshape(n, 1)
        ecoul_var = data_stock_calc_det_tmp[self.num_cols].values * data_stock_calc_det_tmp[coeff_dav_var].fillna(
            1).values
        data_stock_calc_det_tmp[self.num_cols] = np.where(is_const_model, ecoul_var, ecoul_cnst)
        data_stock_calc_det_tmp = data_stock_calc_det_tmp.drop(["COEFF_DAV_CONST", "TYPE MODELE"] + coeff_dav_var,
                                                               axis=1)

        return data_stock_calc_det_tmp

    def update_stock_with_dav_stock(self, data, dav_stock_data):
        if len(dav_stock_data) > 0:
            data = data.copy()
            key_update = data[pa.NC_PA_CLE_OUTPUT + [pa.NC_PA_IND03]].copy()
            data["key_update"] = key_update.apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            data = data.set_index("key_update")
            key_update = dav_stock_data[pa.NC_PA_CLE_OUTPUT + [pa.NC_PA_IND03]].copy()
            dav_stock_data["key_update"] = key_update.apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            dav_stock_data = dav_stock_data.set_index("key_update")
            data.update(dav_stock_data)
        return data

    def load_stock_dav_data(self, stock_data, scenario):
        liste_marche = scenario["MARCHE"].values.tolist()
        liste_contrat = scenario["CONTRAT"].values.tolist()
        filtero = (stock_data[pa.NC_PA_CONTRACT_TYPE].isin(liste_contrat)) & (
            stock_data[pa.NC_PA_MARCHE].isin(liste_marche))
        stock_data_dav = stock_data[filtero].copy()
        return stock_data_dav

    def get_scenarios(self, etab, scenario_name, out_of_ra_rn_pel_etabs, type_pr, warning=True):
        if etab in out_of_ra_rn_pel_etabs and warning:
            logger.info('        Pas d\'application de la surcouche DAV pour : {}'.format(etab))
            return []
        scenarii = self.up.scenarios_params[etab][scenario_name]
        if scenarii is None:
            return []
        scenarii = scenarii[scenarii["TYPE PRODUIT"].str.contains(type_pr)].copy()
        if len(scenarii) == 0:
            return []
        return scenarii

    def get_models_data(self, model_wb, name_sc_dav):
        src_dav_data = ex.get_dataframe_from_range(model_wb, "_input_dav", header=True)
        src_dav_data = src_dav_data[src_dav_data["NOM MODELE"] == name_sc_dav].copy()
        if len(src_dav_data) == 0:
            logger.error("Le modèle %s n'existe pas dans les ficier MODELE des DAV" % name_sc_dav)
            raise ValueError
        return src_dav_data

    def divide_data_set_dav(self, data, scenario):
        liste_marche = scenario["MARCHE"].values.tolist()
        liste_contrat = scenario["CONTRAT"].values.tolist()
        filtero = (data[pa.NC_PA_CONTRACT_TYPE].isin(liste_contrat)) & (data[pa.NC_PA_MARCHE].isin(liste_marche))
        data_dav = data[filtero].copy()
        data_no_dav = data[~filtero].copy()

        return data_dav, data_no_dav

    def create_src_dave_data(self, dav_data, scenario, typo="stock"):
        cols_sc_cible = [x for x in scenario.loc[:, "TX_SURCOUCHE_M0":].columns]
        if typo != "stock":
            src_dav_data = dav_data.reset_index().merge(scenario.drop(["IND01"], axis=1),
                                          on=[pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE], how="left").set_index('new_key')
        else:
            src_dav_data = dav_data.merge(scenario.drop(["IND01"], axis=1),
                                          on=[pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE], how="left")

        is_cnst_mod = (src_dav_data["TYPE MODELE"] == "SURCOUCHE CONSTANTE").values

        if  (~is_cnst_mod).any(axis=0) and typo == "pn%":
            logger.error("       Le modèle de surcouche des DAV ne peut pas être de type variable pour les PN NMD%")
            return dav_data

        if (~is_cnst_mod).any(axis=0):
            _n = (~is_cnst_mod[~is_cnst_mod]).shape[0]
            src_dav_data.loc[~is_cnst_mod, cols_sc_cible] = np.where(dav_data.loc[~is_cnst_mod, pa.NC_PA_RATE_CODE].str.contains(self.INDEX_FIXE).values.reshape(_n, 1), \
                                     1 - src_dav_data.loc[~is_cnst_mod, "TX_SURCOUCHE_M0":].fillna(np.nan).values,
                                     src_dav_data.loc[~is_cnst_mod, "TX_SURCOUCHE_M0":].fillna(np.nan).values)

            src_dav_data.loc[~is_cnst_mod, cols_sc_cible] = src_dav_data.loc[~is_cnst_mod, cols_sc_cible].ffill(axis=1).fillna(0)

        if (is_cnst_mod).any(axis=0):
            src_dav_data.loc[is_cnst_mod, "TX_SURCOUCHE_CST"] = np.where(dav_data.loc[is_cnst_mod, pa.NC_PA_RATE_CODE].str.contains(self.INDEX_FIXE).values, \
                                                        1 - src_dav_data.loc[is_cnst_mod, "TX_SURCOUCHE_CST"],
                                                        src_dav_data.loc[is_cnst_mod, "TX_SURCOUCHE_CST"])

        cle_somme = [x for x in pa.NC_PA_CLE_OUTPUT if x not in [pa.NC_PA_RATE_CODE]] + [pa.NC_PA_IND03, "TYPE MODELE"]
        src_dav_data_sum = src_dav_data.copy()
        num_cols = ([x for x in self.num_cols if x in src_dav_data.columns and (x != "M0" or typo == "stock")] + cols_sc_cible
                    + ["TX_SURCOUCHE_CST"])
        if typo == "stock":
            src_dav_data_sum = ut.to_nan_np(src_dav_data_sum, num_cols)

        src_dav_data_sum = src_dav_data_sum.copy().groupby(cle_somme, as_index=False, dropna=False).sum(min_count=1)
        is_cnst_mod_sum = (src_dav_data_sum["TYPE MODELE"] == "SURCOUCHE CONSTANTE").values

        if typo == "stock":
            self.check_data_integrity(is_cnst_mod_sum, src_dav_data_sum, cols_sc_cible, "TX_SURCOUCHE_CST")
        else:
            self.check_data_integrity(is_cnst_mod_sum, src_dav_data_sum, cols_sc_cible[1:], "TX_SURCOUCHE_CST")

        if typo != "stock":
            src_dav_data = src_dav_data.reset_index().merge(src_dav_data_sum, on=cle_somme, suffixes=(None, "_SUMXYZ"), how="left").set_index('new_key')
        else:
            src_dav_data = src_dav_data.merge(src_dav_data_sum, on=cle_somme, suffixes=(None, "_SUMXYZ"), how="left")

        nums_cols_sum = [x for x in src_dav_data.columns if "_SUMXYZ" in x]
        # was_nan = np.isnan(src_dav_data[nums_cols_sum].astype(float).values) #important pour les PN NMD, conserver en absence d'encours cible

        if typo == "stock":
            src_dav_data[nums_cols_sum] = src_dav_data[nums_cols_sum].fillna(0)

        src_dav_data_new = src_dav_data.copy()

        cols_sc_cible_usr = [x for x in cols_sc_cible if int(x.replace("TX_SURCOUCHE_M", "")) <= self.up.nb_mois_proj_usr]
        sc_cible_dyn = src_dav_data[cols_sc_cible_usr].values
        sc_cible_cnst = src_dav_data["TX_SURCOUCHE_CST"].fillna(0).values

        if typo == "stock" or typo == "pn%":
            """ CALCUL DU M0 """
            nums_cols_encours_t0 = ["M0"]
            nums_cols_encours_t0_sum = ["M0_SUMXYZ"]

            if typo == "stock":
                if is_cnst_mod.any(axis=0):
                    """ CALCUL DES MOIS après 0 """
                    filter = src_dav_data[pa.NC_PA_IND03].isin([pa.NC_PA_LEF, pa.NC_PA_LEM, pa.NC_PA_TEF, pa.NC_PA_TEM]) & is_cnst_mod
                    sc_cible_t0 = sc_cible_cnst[filter].reshape(src_dav_data[filter].shape[0], 1)
                    src_dav_data_new.loc[filter, nums_cols_encours_t0] = src_dav_data.loc[
                                                                             filter, nums_cols_encours_t0_sum].values * sc_cible_t0
                    ht = src_dav_data[filter].shape[0]
                    num_cols = [x for x in self.num_cols if x in src_dav_data.columns]
                    coeff_ecoul = np.divide(src_dav_data.loc[filter, num_cols].fillna(0).values, \
                                            src_dav_data.loc[filter, "M0"].fillna(0).values.reshape(ht, 1))
                    coeff_ecoul = np.nan_to_num(coeff_ecoul, posinf=0, neginf=0)
                    src_dav_data_new.loc[filter, num_cols] = src_dav_data_new.loc[filter, "M0"].values.reshape(ht,1) * coeff_ecoul

                if (~is_cnst_mod).any(axis=0):
                    filter = src_dav_data[pa.NC_PA_IND03].isin([pa.NC_PA_LEF, pa.NC_PA_LEM, pa.NC_PA_TEF, pa.NC_PA_TEM])  & ~is_cnst_mod
                    num_cols = [x for x in self.num_cols if x in src_dav_data.columns]
                    num_cols_sum = [x + "_SUMXYZ" for x in self.num_cols if
                                    (x + "_SUMXYZ" in src_dav_data.columns.tolist())]
                    sc_cible = sc_cible_dyn[filter]
                    src_dav_data_new.loc[filter, num_cols] = src_dav_data.loc[filter, num_cols_sum].values * sc_cible

            elif typo == "pn%":
                filter = src_dav_data[pa.NC_PA_IND03].isin([pa.NC_PA_DEM_CIBLE])
                sc_cible_t0 = sc_cible_cnst[filter].reshape(src_dav_data[filter].shape[0], 1)
                src_dav_data_new.loc[filter, nums_cols_encours_t0] = src_dav_data.loc[
                                                                         filter, nums_cols_encours_t0_sum].values * sc_cible_t0

            """ CALCUL de la MNI"""
            if typo == "stock":
                for indic, ref in zip([pa.NC_PA_LMN, pa.NC_PA_LMN_EVE, "LMN_FTP", "LMN_GPTX",
                                       "LMN_FTP_GPTX"], [pa.NC_PA_LEM, pa.NC_PA_TEM, pa.NC_PA_LEM, pa.NC_PA_TEM, pa.NC_PA_TEM]):
                    filtre_mni = src_dav_data[pa.NC_PA_IND03].isin([indic])
                    if filtre_mni.any():
                        filtre_ref = src_dav_data[pa.NC_PA_IND03].isin([ref])
                        tx_dav = np.divide(src_dav_data.loc[filtre_mni, num_cols].fillna(0).values, \
                                            src_dav_data.loc[filtre_ref, num_cols].fillna(0).values)
                        tx_dav = np.nan_to_num(tx_dav, posinf=0, neginf=0)

                        src_dav_data_new.loc[filtre_mni, num_cols] = tx_dav * src_dav_data_new.loc[
                            filtre_ref, num_cols].values

        else:
            num_cols = [x for x in self.num_cols if x in src_dav_data.columns and x != "M0"]
            num_cols_sum = [x + "_SUMXYZ" for x in self.num_cols if
                            x + "_SUMXYZ" in src_dav_data.columns and x != "M0"]

            if is_cnst_mod.any(axis=0):
                filter = src_dav_data[pa.NC_PA_IND03].isin([pa.NC_PA_DEM_CIBLE]) & is_cnst_mod
                ht = src_dav_data[filter].shape[0]
                sc_cible_cnst_PN = sc_cible_cnst[filter].reshape(ht, 1)
                src_dav_data_new.loc[filter, num_cols] = src_dav_data.loc[filter, num_cols_sum].astype(
                    np.float64).values * sc_cible_cnst_PN
            if (~is_cnst_mod).any(axis=0):
                filter = src_dav_data[pa.NC_PA_IND03].isin([pa.NC_PA_DEM_CIBLE]) & ~is_cnst_mod
                sc_cible_PN = sc_cible_dyn[filter, 1:]
                src_dav_data_new.loc[filter, num_cols] = src_dav_data.loc[filter, num_cols_sum].astype(
                    np.float64).values * sc_cible_PN

        return src_dav_data_new[dav_data.columns.tolist() + ["TYPE MODELE"]].copy()

    def check_data_integrity(self, is_cnst_mod, src_dav_data_sum, cols_sc_cible, col_sc_cible_cnst):
        msn_err = "        Attention, les PARTS TF et TV ne sont pas tjs présentes dans les données pour une même catégorie de DAV"
        if is_cnst_mod.any(axis=0):
            if src_dav_data_sum.loc[is_cnst_mod, col_sc_cible_cnst].sum() != src_dav_data_sum[is_cnst_mod].shape[0]:
                logger.error(msn_err)
                raise ValueError(msn_err)

        if (~is_cnst_mod).any(axis=0):
            if (src_dav_data_sum.loc[~is_cnst_mod, cols_sc_cible].sum(axis=0) != src_dav_data_sum[~is_cnst_mod].shape[0]).any():
                logger.error(msn_err)
                raise ValueError(msn_err)
