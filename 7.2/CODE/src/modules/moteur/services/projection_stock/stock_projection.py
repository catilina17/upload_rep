from modules.moteur.services.indicateurs_taux.gap_taux_agreg import Gap_Taux_Calculator
from modules.moteur.services.indicateurs_liquidite.gap_liquidite_module import Gap_Liquidite_Calculator
import modules.moteur.services.projection_stock.stock_projection_cases as spc
import modules.moteur.parameters.general_parameters as gp
import modules.moteur.utils.generic_functions as gf
from collections import OrderedDict
import modules.moteur.index_curves.calendar_module as hf
import modules.moteur.index_curves.tx_module as tx
import numpy as np
import pandas as pd
import modules.moteur.services.indicateurs_taux.refixing_tla as tla
from warnings import simplefilter
import logging

logger = logging.getLogger(__name__)

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class Stock_Projecter():

    def __init__(self, cls_usr, cls_mp, cls_cal, cls_tx):
        self.tx = cls_tx
        self.cal = cls_cal
        self.up = cls_usr
        self.mp = cls_mp
        self.CURVE_MISSING_MISSING = "        Les courbes de taux suivantes (scénario ou fermat) sont manquantes "
        self.WARNING_TX_CIBLE = "ATTENTION, CERTAINS TAUX CIBLE ds lES PN NMD sont égaux à 0 ! Les TAUX cible doivent être mis à blancs si vous ne souhaitez pas en mettre"
        self.DUPLICATES_INDEX_CURVES = "    Les courbes de taux %s ont des doublons"

    def project_stock(self, dic_stock, dic_stock_sc, dic_stock_sci, dic_stock_scr, dic_updated_sc):
        data_stock = dic_stock_sc["stock"]
        if data_stock.shape[0] > 0:

            complementary_st_data = self.load_complementary_data(data_stock)

            self.calculate_gaps(data_stock, dic_stock_sci, complementary_st_data, dic_stock_scr, dic_stock,
                           dic_updated_sc, complementary_st_data)

            mni_inds, encours_inds = self.get_mni_inds()

            tx_sc, tx_ref = self.merge_data_stock_with_index_curves(data_stock, complementary_st_data)

            dic_single_cases = self.generate_general_cases(complementary_st_data)
            for ind_mni, ind_enc in zip(mni_inds, encours_inds):
                projection_vars = {}
                mni_proj = []
                for j in range(1, self.up.nb_mois_proj_usr + 1):
                    self.load_jth_month_indics(projection_vars, j, dic_stock_sci, complementary_st_data, tx_sc,
                                          tx_ref, ind_mni, ind_enc)

                    dic_cases, dic_tx_mni = self.generate_projection_cases(dic_single_cases, complementary_st_data, \
                                                                      projection_vars)

                    mni_sc_j = self.select_tx_mni(dic_cases, dic_tx_mni)
                    mni_proj.append(mni_sc_j)

                self.concat_mni_projections(data_stock, dic_updated_sc, mni_proj, dic_stock_sci, ind_mni)

            """ MISE SOUS DICO"""
            self.write_in_sc_dic(dic_stock_sc, data_stock, dic_stock)

    def calculate_gaps(self, data_stock, dic_stock_sci, comp_data, dic_stock_scr, dic_stock, dic_updated_sc, comp_dic):
        tla.refix_tla_stock(self.mp, self.up, data_stock, comp_data["is_ech"], dic_stock_sci)

        self.calcul_des_gaps(data_stock, dic_stock_scr, dic_stock_sci, dic_stock["other_stock"], comp_dic)

    def write_in_sc_dic(self, dic_stock_sc, data_stock, dic_stock):
        dic_stock_sc["stock"] = data_stock
        dic_stock_sc["other_stock"] = dic_stock["other_stock"].copy()

    def calcul_des_gaps(self, data_stock, dic_stock_scr, dic_stock_sci, data_stock_other, comp_dic):
        self.calcul_ratios_refixing(dic_stock_sci, dic_stock_scr, comp_dic)
        gp_tx = Gap_Taux_Calculator(self.up, self.mp)
        gpl = Gap_Liquidite_Calculator(self.up, self.mp)
        for simul in self.up.type_simul:
            if self.up.type_simul[simul]:
                gp_tx.calcul_gap_tx_agreg(data_stock, dic_stock_scr, dic_stock_sci, "STOCK", simul)
                gpl.calcul_gap_liq_st(dic_stock_sci, simul)
                gpl.appliquer_conv_ecoulement_gp_liq(dic_stock_sci, "ST", data_stock, data_stock_other, simul)

    def get_mni_inds(self, ):
        mni_inds = [gp.mn_sti] + self.up.type_simul["EVE"] * [gp.mn_gp_rg_sti]
        encours_inds = [gp.em_sti] + self.up.type_simul["EVE"] * [gp.gpr_em_eve_sti + "0"]
        return mni_inds, encours_inds

    def merge_data_stock_with_index_curves(self, data_stock, comp_data):
        """ 1. CHARGEMENT DES COURBES DE TAUX """
        ajst_curve = self.tx.tx_curves_rco.copy()
        tx_curve = self.tx.tx_curves_sc.copy()
        keys_curves = [gp.NC_DEVISE_SC_TX, gp.NC_CODE_SC_TX]
        key_data = [gp.nc_output_devise_cle, gp.nc_output_index_calc_cle]

        ajst_curve[gp.NC_CODE_SC_TX] = ajst_curve[gp.NC_CODE_SC_TX].str.upper()
        tx_curve[gp.NC_CODE_SC_TX] = tx_curve[gp.NC_CODE_SC_TX].str.upper()

        data_ref = data_stock[key_data].copy()
        data_ref[gp.nc_output_index_calc_cle] = data_ref[gp.nc_output_index_calc_cle].str.upper()

        """ 2. JOINTURES DES DONNEES DE STOCK AVEC LES COURBES DE TAUX"""
        duplicates_ref = ajst_curve[ajst_curve.duplicated(subset=keys_curves, keep=False)][keys_curves].values.tolist()
        if len(duplicates_ref) > 0:
            logger.error(self.DUPLICATES_INDEX_CURVES % "du scénario de référence" + " : %s" % duplicates_ref)
            ajst_curve = ajst_curve.drop_duplicates(keys_curves)

        tx_ref = data_ref.copy().merge(ajst_curve, how="left", right_on=keys_curves, left_on=key_data)
        tx_ref.index = data_stock.index.get_level_values("new_key")

        duplicates_sc = tx_curve[tx_curve.duplicated(subset=keys_curves, keep=False)][keys_curves].values.tolist()

        if len([x for x in duplicates_sc if x not in duplicates_ref]) > 0:
            logger.error(self.DUPLICATES_INDEX_CURVES % "du scénario utilisateur" + " : %s" % [x for x in duplicates_sc if
                                                                                          x not in duplicates_ref])
            tx_curve = tx_curve.drop_duplicates(keys_curves)

        tx_sc = data_ref.copy().merge(tx_curve, how="left", right_on=keys_curves, left_on=key_data)
        tx_sc.index = data_stock.index.get_level_values("new_key")

        """ 3. DETECTION DES COURBES DE TAUX MANQUANTES"""
        filtero = ((tx_ref["M1"].isnull()) | (tx_sc["M1"].isnull())) & (comp_data["type_tx"] == "TV")
        if filtero.any():
            list_warning = data_stock.loc[np.array(filtero).reshape(filtero.shape[0], 1), :][
                key_data].drop_duplicates().values.tolist()
            logger.error(self.CURVE_MISSING_MISSING + ":" + str(list_warning))

        return tx_sc, tx_ref

    def create_sc_dictionaries(self, dic_stock, dic_stock_i, dic_updated_i):
        dic_stock_sc = OrderedDict()
        for key, val in dic_stock.items():
            dic_stock_sc[key] = dic_stock[key].copy()

        dic_stock_sci = OrderedDict()
        for key, val in dic_stock_i.items():
            dic_stock_sci[key] = dic_stock_i[key].copy()

        dic_updated_sc = OrderedDict()
        for key, val in dic_updated_i.items():
            dic_updated_sc[key] = dic_updated_i[key].copy()

        dic_stock_scr = OrderedDict()

        return dic_stock_sc, dic_stock_sci, dic_updated_sc, dic_stock_scr

    def generate_general_cases(self, comp_data):
        dic_single_cases = {}
        dic_single_cases["IS TF"] = (comp_data["type_tx"] == "TF")
        dic_single_cases["IS NOT ECH"] = (comp_data["is_ech"] == False).values
        return dic_single_cases

    def generate_cas_enc_ref_j_non_zero(self, j, dic_stock_sci, dic_cases, simul):
        if simul == "EVE":
            dic_cases["IS ENCOURS Mj != 0"] = (dic_stock_sci[gp.gpr_em_eve_sti + "0"]["M" + str(j)] != 0).values
        else:
            dic_cases["IS ENCOURS Mj != 0"] = (dic_stock_sci[gp.em_sti]["M" + str(j)] != 0).values

    def generate_cas_is_tx_cible(self, j, com_data, dic_cases):
        dic_cases["IS TX CIB"] = (com_data["tx_cible"]["M" + str(j)] != -1000000).values

    def generate_projection_cases(self, dic_gen_cases, comp_data, pv):
        dic_cases = OrderedDict()
        dic_mni = OrderedDict()

        spc.generate_specific_cases(dic_cases, dic_gen_cases, dic_mni, pv, comp_data)

        return dic_cases, dic_mni

    def load_jth_month_indics(self, pv, j, dic_stock_sci, comp_data, tx_sc, tx_ref, indic_mni, enc_ref):
        pv["lmn_j"] = np.array(dic_stock_sci[indic_mni]["M" + str(j)])
        pv["enc_ref_j"] = np.array(dic_stock_sci[enc_ref]["M" + str(j)])
        pv["tem_j"] = np.array(dic_stock_sci["tem"]["M" + str(j)])
        pv["bsc_j"] = gf.fill_na_pn_z(np.array(comp_data["base_calc"]["M" + str(j)]))
        pv["tx_sc_j"] = np.array(tx_sc["M"  + str(j)])
        pv["tx_ref_j"] = np.array(tx_ref["M" + str(j)])

    def load_complementary_data(self, data_stock):
        comp_dic = {}
        index_data_stock = pd.DataFrame(index=data_stock.index, columns=["A"])

        """ FILTRE DE TAUX FIXE"""
        comp_dic["type_tx"] = np.where(data_stock[gp.nc_output_index_calc_cle].str.contains(gp.FIX_ind), 'TF', 'TV')

        """ CALCUL DES COEFFS DE MENSUALISATION DES TAUX """
        comp_dic["base_calc"] = self.cal.generate_base_calc_coeff_stock(data_stock, comp_dic["type_tx"])

        """ COLONNES ECHEANCE"""
        contrats_map = self.mp.contrats_map.copy()[[gp.nc_cp_isech]].copy()
        comp_dic["is_ech"] = data_stock.join(contrats_map, how="left", on=gp.nc_output_contrat_cle).copy()
        self.test_existence(data_stock, contrats_map, comp_dic["is_ech"])
        comp_dic["is_ech"] = comp_dic["is_ech"][gp.nc_cp_isech].copy()

        return comp_dic

    def test_existence(self, data_stock, contrats_map, joined_data):
        inner_joined = data_stock.join(contrats_map, how="inner", on=gp.nc_output_contrat_cle).copy()
        if joined_data.shape[0] != inner_joined.shape[0]:
            joined_data = joined_data.reset_index(level=[gp.nc_output_palier_cle, gp.nc_output_contrat_cle])
            inner_joined = inner_joined.reset_index(level=[gp.nc_output_palier_cle, gp.nc_output_contrat_cle])
            list_c = joined_data[~joined_data[gp.nc_cp_contrat].isin(inner_joined[gp.nc_cp_contrat].values.tolist())] \
                [gp.nc_cp_contrat].values.tolist()
            list_c = list(set(list_c))
            msg_err = "Some contracts are absent in the general contracts indicators: %s" % list_c
            logger.error(msg_err)
            raise ValueError(msg_err)

    def load_pn_nmd_tx(self, data_stock, cls_pn_loader):
        """ fonction permettant d'ajouter les marges et taux sépciaux aux produits du stock type NMD"""

        if (cls_pn_loader.dic_pn_nmd != {} or cls_pn_loader.dic_pn_nmd != {}):
            if "nmd" in cls_pn_loader.dic_pn_nmd:
                pn_nmd = cls_pn_loader.dic_pn_nmd["nmd"].copy()
            else:
                pn_nmd = cls_pn_loader.dic_pn_nmd["nmd%"].copy()

            """ JOINTURE DES TX SPEC NMD """
            pn_nmd_tx_spread = pn_nmd.loc[
                pn_nmd[gp.nc_output_ind3] == cls_pn_loader.nc_pn_tx_sp_pn, cls_pn_loader.num_cols_m1].copy()
            data_tx_sp_stock = data_stock.join(pn_nmd_tx_spread, how='left', on="new_key")[
                cls_pn_loader.num_cols_m1].copy()
            data_tx_sp_stock = data_tx_sp_stock.astype(np.float64) / 10000

            """ JOINTURE DES TX CIBLE NMD"""
            pn_nmd_tx_cible = pn_nmd.loc[
                pn_nmd[gp.nc_output_ind3] == cls_pn_loader.nc_pn_tx_prod_cible, cls_pn_loader.num_cols_m1].copy()
            data_tx_prod_cible_stock = data_stock.join(pn_nmd_tx_cible, how='left', on="new_key")[
                cls_pn_loader.num_cols_m1].copy()
            data_tx_prod_cible_stock = data_tx_prod_cible_stock.astype(np.float64) / 10000
        else:
            data_tx_sp_stock = pd.DataFrame(np.nan, index=data_stock.index,
                                            columns=["M" + str(i) for i in range(1, self.up.nb_mois_proj_usr + 1)])
            data_tx_prod_cible_stock = pd.DataFrame(np.nan, index=data_stock.index,
                                                    columns=["M" + str(i) for i in
                                                             range(1, self.up.nb_mois_proj_usr + 1)])

        """ MISE A ZERO DES NA, à -1000000 pour les TX CIBLE"""
        data_tx_sp_stock = data_tx_sp_stock.fillna(0)
        data_tx_prod_cible_stock = data_tx_prod_cible_stock.fillna(-1000000)

        if True in ((data_tx_prod_cible_stock == 0).any()).values.tolist():
            logger.warning(self.WARNING_TX_CIBLE)

        return data_tx_sp_stock, data_tx_prod_cible_stock

    def concat_mni_projections(self, data_stock, dic_updated_sc, mni_stock_proj, dic_stock_sci, ind_mni):
        cols = ["M" + str(j) for j in range(1, self.up.nb_mois_proj_usr + 1)]
        mni_stock_proj = np.column_stack(mni_stock_proj)
        mni_stock_proj = pd.DataFrame(data=mni_stock_proj, index=data_stock.index.get_level_values("new_key"))
        mni_stock_proj.columns = cols

        dic_stock_sci[ind_mni].drop("M0", axis=1, inplace=True)
        filtre_no_recalc = ~dic_updated_sc[ind_mni].values
        dic_stock_sci[ind_mni].loc[filtre_no_recalc] = mni_stock_proj[filtre_no_recalc]

    def calcul_ratios_refixing(self, dic_stock_sci, dic_stock_scr, comp_dic):
        ht = comp_dic["type_tx"].shape[0]
        for name, num, den in zip(["tx_fix_f", "tx_fix_m"], ["tef", "tem"],
                                  [gp.ef_sti, gp.em_sti]):
            data = np.divide(dic_stock_sci[num].values, dic_stock_sci[den].values)
            data = np.where(((dic_stock_sci[den].values == 0) & (comp_dic["type_tx"] == "TF").reshape(ht, 1)), 1, data)
            data = np.nan_to_num(data, posinf=0, neginf=0)
            dic_stock_scr[name] = pd.DataFrame(data, columns=dic_stock_sci[den].columns,
                                               index=dic_stock_sci[den].index)

    def select_tx_mni(self, cases, choices_mni):
        cases_l = gf.generate_list_from_ordered_dic(cases)
        choices_mni_l = gf.generate_list_from_ordered_dic(choices_mni)
        mni_sc_j = np.select(cases_l, choices_mni_l)
        return mni_sc_j
