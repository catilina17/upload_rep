import modules.moteur.utils.generic_functions as gf
import modules.moteur.parameters.general_parameters as gp
import modules.moteur.parameters.user_parameters as up
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import modules.moteur.services.intra_groupes.intra_group_module as ig
import modules.moteur.services.indicateurs_liquidite.lcr_module as lcr
import modules.moteur.services.indicateurs_liquidite.nsfr_module as nsfr
import modules.moteur.services.indicateurs_liquidite.outflow_module as outfl
import pandas as pd
import numpy as np
import re
import logging

global inds_mni

logger = logging.getLogger(__name__)

from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

MAPPING_MISSING = "    Il y a des indicators manquants dans les données de stock "
WARNING_BASSIN = "    ATTENTION, les données de stock semblent appartenir à un bassin différent de celui spécifié dans les paramètres"
NON_UNIQUE_KEYS = "    Les données de stock contiennent des clés non uniques :"

k_em = 6.5
k_ef = 12
k_mn = 12
limit_month = 120


class Stock_Preparation():
    def __init__(self, cls_usr, cls_mp):
        self.up = cls_usr
        self.mp = cls_mp

    def read_stock(self, data):
        global inds_mni
        """ Fonction permettant de lire, ajuster et de diviser le stock en blocs"""
        """ Chargement des données"""
        qual_cols = [x for x in data.columns if x not in pa.NC_PA_COL_SORTIE_NUM_ST]
        cols_sup = ["M" + str(i) for i in range(0, self.up.nb_mois_proj_usr + 1) if "M" + str(i) not in data.columns]
        if len(cols_sup) > 0:
            data = pd.concat(
                [data, pd.DataFrame(np.full((data.shape[0], len(cols_sup)), np.nan), \
                                    index=data.index, columns=cols_sup)], axis=1).ffill(axis=1)
        data = data[qual_cols + ["M" + str(i) for i in range(0, self.up.nb_mois_proj_usr + 1)]].copy()
        return data

    def format_stock(self, data, ig):
        num_cols = ["M" + str(i) for i in range(0, self.up.nb_mois_proj_usr + 1)]

        data = self.add_and_format_cols(data, ig)

        data_qual = self.get_data_qual(data, num_cols)

        dic_updated = self.get_data_updated(data)

        data_qual = self.check_data_integrity(data_qual)

        """ TRAITEMENT DES COLONNES NUMERIQUES """
        dic_stock_ind = self.format_num_cols_stock(data, num_cols)
        gf.clean_df(data)
        self.add_index_num_stock(dic_stock_ind, data_qual.index)
        inds_mni = {}
        inds_mni[gp.mn_sti] = gp.em_sti
        if self.up.type_simul["EVE"] or self.up.type_simul["EVE_LIQ"]:
            inds_mni[gp.mn_gp_rg_sti] = "tem"

        #self.interpolate_num_cols_stock(data_qual, dic_stock_ind, inds_mni)

        """ TRAITEMENT DES DONNEES QUALI"""
        dic_stock = self.format_qualitative_cols_stock(data_qual)

        dic_stock = self.add_complementary_index(dic_stock)

        # outfl.calculate_outflows(dic_stock_ind, "ST", dic_data=dic_stock)
        # lcr.calc_err_em_tot(dic_stock["stock"], dic_stock_ind, dic_stock["other_stock"])
        # nsfr.calc_err_em_tot(dic_stock_ind, dic_stock["other_stock"])

        dic_stock, dic_stock_ind, dic_updated = self.cut_into_dics(dic_stock, dic_stock_ind, dic_updated)

        return dic_stock, dic_stock_ind, dic_updated

    def add_index_num_stock(self, dic_stock_ind, index):
        for ind, val in dic_stock_ind.items():
            dic_stock_ind[ind].index = index

    def get_data_qual(self, data, num_cols):
        qual_cols = [x for x in data.columns if x not in num_cols]
        data_qual = data.loc[data[gp.nc_output_ind3] == pa.NC_PA_LEM, qual_cols]
        return data_qual

    def get_data_updated(self, data):
        dic_updated = {}
        dic_updated[gp.mn_sti] = data.loc[data[gp.nc_output_ind3] == pa.NC_PA_LMN, "UPDATED"]
        dic_updated[gp.mn_gp_rg_sti] = data.loc[data[gp.nc_output_ind3] == pa.NC_PA_LMN_EVE, "UPDATED"]
        return dic_updated

    def interpol_missing_ef(self, dic, ef_ind, filter_not_updated):
        """ Fonction permettant d'interpoler les indiateurs de type EF"""
        data = dic[ef_ind][filter_not_updated].copy()
        for j in range(0, int((min(self.up.nb_mois_proj_usr, gp.real_max_months) - limit_month) / 12)):
            deb = j * 12 + limit_month
            fin = (j + 1) * 12 + limit_month
            adjust_const = 1 / k_ef * (data["M" + str(deb)] - data["M" + str(fin)])
            cols = ["M" + str(i) for i in range(deb + 1, fin + 1) if not "M" + str(i) in data.columns]
            data = pd.concat([data, pd.DataFrame([[0] * len(cols)], index=data.index, columns=cols)], axis=1)
            for i in range(deb + 1, fin + 1):
                data["M" + str(i)] = data["M" + str(i - 1)].values - adjust_const

        for j in range(limit_month + 1, min(self.up.nb_mois_proj_usr, gp.real_max_months) + 1):
            data["M" + str(j)] = np.where(abs(data["M" + str(j)]) < 1, 0, data["M" + str(j)])

        dic[ef_ind][filter_not_updated] = data

    def interpol_missing_em(self, dic, em_ind, filter_not_updated, overwrite=False):
        """ Fonction permettant d'interpoler les indicateurs de type EM """
        data = dic[em_ind][filter_not_updated].copy()
        data_orig = dic[em_ind].copy()
        for j in range(0, int((min(self.up.nb_mois_proj_usr, gp.real_max_months) - limit_month) / 12)):
            deb = j * 12 + limit_month
            fin = (j + 1) * 12 + limit_month
            adjust_const = (data["M" + str(deb)] - data["M" + str(fin)]) / k_em
            if not overwrite:
                cols = ["M" + str(i) for i in range(deb + 1, fin + 1) if not "M" + str(i) in data.columns]
                data = pd.concat([data, pd.DataFrame([[0] * len(cols)], index=data.index, columns=cols)], axis=1)
            for i in range(deb + 1, fin + 1):
                data["M" + str(i)] = np.where(data["M" + str(i - 1)] == 0, 0, data["M" + str(i - 1)] - adjust_const)
                data["M" + str(i)] = np.where(data["M" + str(i)] / data["M" + str(i - 1)] < 0, 0, data["M" + str(i)])

        for j in range(limit_month + 1, min(self.up.nb_mois_proj_usr, gp.real_max_months) + 1):
            data["M" + str(j)] = np.where(abs(data["M" + str(j)]) < 1, 0, data["M" + str(j)])

        dic[em_ind][filter_not_updated] = data
        return data_orig

    def interpol_missing_mn(self, dic, mn_ind, em_ind, data_orig_em, filter_not_updated):
        """ Fonction permettant d'interpoler les indicateurs de type MN """
        data_mn = dic[mn_ind][filter_not_updated].copy()
        data_em = dic[em_ind][filter_not_updated].copy()
        data_orig_em_f = data_orig_em[filter_not_updated]
        for j in range(0, int((min(self.up.nb_mois_proj_usr, gp.real_max_months) - limit_month) / 12)):
            deb = j * 12 + limit_month
            fin = (j + 1) * 12 + limit_month
            tx_adjust = np.where(data_orig_em_f["M" + str(fin)] == 0, 0,
                                 data_mn["M" + str(fin)].values / data_orig_em_f["M" + str(fin)].values)
            alt_mni = self.get_mni_when_tem_zero(data_mn, fin, deb)

            cols = ["M" + str(i) for i in range(deb + 1, fin + 1) if not "M" + str(i) in data_mn.columns]
            data_mn = pd.concat([data_mn, pd.DataFrame([[0] * len(cols)], index=data_mn.index, columns=cols)], axis=1)
            for i in range(deb + 1, fin + 1):
                data_mn["M" + str(i)] = np.where((tx_adjust == 0) & (mn_ind == gp.mn_gp_rg_sti),
                                                 alt_mni[:, i - deb - 1],
                                                 data_em["M" + str(i)] * tx_adjust / k_mn)

        dic[mn_ind][filter_not_updated] = data_mn
        dic[em_ind][filter_not_updated] = data_em

    def get_mni_when_tem_zero(self, data_mn, fin, deb):
        n = data_mn.shape[0]
        gr = ((data_mn["M" + str(fin)] / 12).values / (data_mn["M" + str(deb)]).values) ** (1 / 6) - 1
        gr = np.nan_to_num(gr)
        mni_months = data_mn["M" + str(deb)].values.reshape(n, 1) * (1 + gr).reshape(n, 1) ** (
            np.arange(1, 13).reshape(1, 12))
        mni_sum = mni_months.sum(axis=1)
        poids_relatif = mni_months / mni_sum.reshape(n, 1)
        diff_total = mni_sum - data_mn["M" + str(fin)].values
        return np.nan_to_num((-diff_total.reshape(n, 1) * poids_relatif + mni_months))

    def add_and_format_cols(self, data, ig):
        """ Ajout de la colonne bassin et on s'assure que le bassin correspond à l'Organisation """
        """ Ajout de la colonne isIG"""
        data = ig.add_ig_colum(data, gp.nc_output_palier_cle, gp.nc_output_contrat_cle)

        """ Conversion des book code si numérique en chaîne de caractères """
        data[gp.nc_output_book_cle] = [str(int(float(x))) if not re.match(r'^-?\d+(?:\.\d+)?$', str(x)) is None else x
                                       for x
                                       in data[gp.nc_output_book_cle]]

        """ Creation de la clé à partir des colonnes de clé"""
        data['new_key'] = data[gp.cle_stock].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

        return data

    def check_data_integrity(self, data):
        """  On s'assure que les clés de stock sont uniques """
        if not data['new_key'].is_unique:
            logger.error(NON_UNIQUE_KEYS)
            logger.error(data.loc[data['new_key'].duplicated(), "new_key"].values.tolist())
            raise ValueError(NON_UNIQUE_KEYS)

        """  On s'assure qu'aucun mapping n'est manquant """
        if "MAPPING" in [str(x).upper() for x in data['new_key']]:
            logger.error(MAPPING_MISSING)
            raise ValueError(MAPPING_MISSING)

        """ Indexation de différentes clés utiles pour jointure avec indicators"""
        data = data.set_index(['new_key'])

        return data

    def format_num_cols_stock(self, data, num_cols):
        """ Creation d'un dico pour les indicateurs numériques"""
        dic_stock_ind = {}
        dic_stock_ind[gp.ef_sti], dic_stock_ind[gp.em_sti], dic_stock_ind["tef"], dic_stock_ind["tem"], \
            dic_stock_ind[gp.mn_sti] \
            = [data.loc[data[gp.nc_output_ind3] == ind, num_cols] for ind in gp.st_ind]
        if self.up.type_simul["EVE"] or self.up.type_simul["EVE_LIQ"]:
            # dic_stock_ind[gp.em_eve_sti] \
            #    = data.loc[data[gp.nc_output_ind3] == pa.NC_PA_LEM, num_cols]

            dic_stock_ind[gp.mn_gp_rg_sti] \
                = data.loc[data[gp.nc_output_ind3] == gp.stock_lmn_eve, num_cols]

        """ FORCAGE EN FLOAT """
        for ind, val in dic_stock_ind.items():
            dic_stock_ind[ind] = dic_stock_ind[ind].astype(np.float64)

        return dic_stock_ind

    def interpolate_num_cols_stock(self, data_qual, dic_stock_ind, inds_mni):
        """ INTERPOLATION DES MOIS MANQUANTS"""
        filter_not_updated = ~data_qual["UPDATED"].values
        for indic in [gp.ef_sti, "tef"]:
            self.interpol_missing_ef(dic_stock_ind, indic, filter_not_updated)
        data_orig = {}
        for indic in ["tem", gp.em_sti]:
            data_orig[indic] = self.interpol_missing_em(dic_stock_ind, indic, filter_not_updated)
        for ind_mni in list(inds_mni.keys()):
            self.interpol_missing_mn(dic_stock_ind, ind_mni, inds_mni[ind_mni], data_orig[inds_mni[ind_mni]],
                                     filter_not_updated)

        """ ON REORDONNE LES COLONNES NUMERIQUES"""
        for ind, val in dic_stock_ind.items():
            dic_stock_ind[ind] = dic_stock_ind[ind][["M" + str(i) for i in range(0, self.up.nb_mois_proj_usr + 1) if
                                                     "M" + str(i) in dic_stock_ind[ind].columns]].copy()

        if self.up.nb_mois_proj_usr >= gp.real_max_months + 1:
            for ind in [gp.ef_sti, "tef", "tem", gp.em_sti] + list(inds_mni.keys()):
                cols = ["M" + str(i) for i in range(0, gp.real_max_months + 1)]
                df_to_update = dic_stock_ind[ind][filter_not_updated][cols]
                dic_stock_ind[ind][filter_not_updated] = gf.prolong_last_col_value(df_to_update, \
                                                                                   self.up.nb_mois_proj_usr - gp.real_max_months, \
                                                                                   is_df=True, year_gr=True, suf="M",
                                                                                   s=1)

    def add_cols_sup(self, data, max_cms, t):
        cols_sup = ["M" + str(i) for i in range(min(t, max_cms) + 1, t + 1)]
        tx_curves = pd.concat(
            [data, pd.DataFrame(np.full((data.shape[0], len(cols_sup)), np.nan), index=data.index,
                                columns=cols_sup)], axis=1).ffill(axis=1)
        return tx_curves

    def format_qualitative_cols_stock(self, data):
        qual_cols = [x for x in data.columns if x not in pa.NC_PA_COL_SORTIE_NUM_ST]
        data_stock = data[[x for x in qual_cols if x in gp.cle_stock]].copy()
        data_other_stock = data[[x for x in qual_cols if not x in gp.cle_stock]].copy()
        dic_stock = {}
        dic_stock["stock"] = data_stock
        dic_stock["other_stock"] = data_other_stock
        return dic_stock

    def cut_into_dics(self, dic_stock, dic_stock_ind, dic_updated):
        """ DECOUPAGE DU STOCK EN X BLOCS"""
        chunk = 5000

        dic_stock_ind = gf.cut_and_list(dic_stock_ind, dic_stock_ind[gp.ef_sti].shape[0], chunk)
        dic_stock = gf.cut_and_list(dic_stock, dic_stock["stock"].shape[0], chunk)
        dic_updated = gf.cut_and_list(dic_updated, dic_updated[gp.mn_sti].shape[0], chunk)

        return dic_stock, dic_stock_ind, dic_updated

    def add_complementary_index(self, dic_stock):
        """ INDEXATION SUPPLEMENTAIRE POUR JOINTURE AVEC indicators"""
        dic_stock["stock"] = dic_stock["stock"].set_index([gp.nc_output_contrat_cle, gp.nc_output_palier_cle],
                                                          append=True)

        return dic_stock
