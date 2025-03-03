# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 16:53:53 2020

@author: Hossayne
"""
import modules.moteur.utils.generic_functions as gf
import modules.moteur.parameters.general_parameters as gp
import modules.moteur.parameters.user_parameters as up
import modules.moteur.services.intra_groupes.intra_group_module as ig
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import logging
import pandas as pd
import numpy as np
import re
import os

logger = logging.getLogger(__name__)

from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
np.seterr(divide='ignore', invalid='ignore')


class PN_LOADER():
    def __init__(self, cls_usr, cls_mp, cls_ig):
        self.mp = cls_mp
        self.up = cls_usr
        self.ig = cls_ig
        self.max_duree_ech_prct = 0
        self.max_duree_ech = 0
        self.NON_UNIQUE_KEYS = "    Les données de PN %s contiennent des clés non uniques :"
        self.MAPPING_MISSING = "    Il ya des indicators manquants dans les données de PN %s "
        self.dic_pn_ech = {}
        self.dic_pn_nmd = {}
        self.nc_pn_flem_pn = "FlEM"
        self.nc_pn_ecm_pn = "ECM"
        self.nc_pn_tx_sp_pn = "TxSpPN(bps)"
        self.nc_pn_tx_prod_cible = "TxProdCible(bps)"
        self.nc_pn_tx_mg_co = "MargeCo(bps)"

        self.ind_pn = [self.nc_pn_flem_pn, self.nc_pn_ecm_pn, self.nc_pn_tx_sp_pn, self.nc_pn_tx_prod_cible, self.nc_pn_tx_mg_co]
        self.pn_num_cols = ["M" + str(i) for i in range(0, gp.real_max_months + 1)]
        self.pn_num_cols2 = ["M" + str(i) for i in range(0, gp.max_months + 1)]

    def load_pn_ech(self, scen_path, files_sc_vol):

        """ Instanciation des dictionnaires globaux"""
        dic_pn_ech = {}

        """ Boucles sur les différents types de PN """
        for tag, type_pn in zip([gp.sc_vol_ech_tag, gp.sc_vol_pn_ech_prct_tag], ["ech", "ech%"]):
            """ Chargement des données PN"""
            if len(files_sc_vol[tag]) > 0:
                file_sc_ech = os.path.join(scen_path, "SC_VOLUME", files_sc_vol[tag][0])
                pn = pd.read_csv(file_sc_ech, sep = ";", decimal=",")
                if pn.shape[0] > 0:
                    max_month = min(self.up.max_month_pn[type_pn], self.up.nb_mois_proj_out)

                    """ Instanciation des dictionnaires par type de PN"""
                    dic_pn_ech[type_pn] = {}

                    """ Select necessary num cols"""
                    pn = self.select_necessary_num_cols(pn)

                    """ Conversion des book code si numériques en chaîne de caractères """
                    pn = self.format_and_add_cols(pn)

                    """ Création de la clé à partir des colonnes de clé """
                    pn = self.construct_new_key(pn, type_pn)

                    pn = self.rank_by_duree(pn)

                    """ CALC DEM %"""
                    dic_pn_ech[type_pn] = self.calc_dem_amounts_from_prct(pn, type_pn, max_month)

                    gf.clean_df(pn)

        self.dic_pn_ech = dic_pn_ech

    def load_pn_nmd(self, scen_path, files_sc_vol):
        """ Instanciation des dictionnaires globaux"""
        dic_pn_nmd = {}

        file_nmd_calage = os.path.join(scen_path, "SC_VOLUME", files_sc_vol[gp.sc_vol_nmd_calage_tag][0])

        """ Boucles sur les différents types de PN """
        for tag, type_pn in zip([gp.sc_vol_nmd_tag, gp.sc_vol_nmd_prct_tag], ["nmd", "nmd%"]):
            """ Chargement des données PN"""
            if len(files_sc_vol[tag]) > 0:
                file_sc_nmd = os.path.join(scen_path, "SC_VOLUME", files_sc_vol[tag][0])
                pn = pd.read_csv(file_sc_nmd, sep = ";", decimal=",")
                if pn.shape[0] > 0:
                    max_month = min(self.up.max_month_pn[type_pn], self.up.nb_mois_proj_out)

                    """ Instanciation des dictionnaires par type de PN"""
                    dic_pn_nmd[type_pn] = {}

                    """ Select necessary num cols"""
                    pn = self.select_necessary_num_cols(pn)

                    """ Conversion des book code si numériques en chaîne de caractères """
                    pn = self.format_and_add_cols(pn)

                    """ Création de la clé à partir des colonnes de clé """
                    pn = self.construct_new_key(pn, type_pn)

                    pn = self.prolong_tx_cols(pn, type_pn)

                    """ CALC ECM %"""
                    dic_pn_nmd[type_pn] = self.calc_dem_amounts_from_prct(pn, type_pn, max_month)

                    if not 'calage' in dic_pn_nmd:
                        dic_pn_nmd['calage'] = pd.read_csv(file_nmd_calage, sep = ";", decimal=",")

                    gf.clean_df(pn)

        self.dic_pn_nmd = dic_pn_nmd

    def select_necessary_num_cols(self, pn):
        pn_qual_cols = [x for x in pn.columns if x not in pa.NC_PA_COL_SORTIE_NUM_PN]
        return pn[pn_qual_cols + pa.NC_PA_COL_SORTIE_NUM_PN].copy()

    def calc_dem_amounts_from_prct(self, pn, type_pn, max_month):
        if "nmd" in type_pn or "%" in type_pn:
            filtre_dem = pn[gp.nc_output_ind3] == self.nc_pn_ecm_pn
        else:
            filtre_dem = pn[gp.nc_output_ind3] == self.nc_pn_flem_pn
        if "%" in type_pn:
            for j in range(1, max_month + 1):
                pn.loc[filtre_dem, "M" + str(j)] = pn.loc[filtre_dem, "M" + str(j - 1)] * (
                        1 + pn.loc[filtre_dem, "M" + str(j)].fillna(0).astype(str).replace("", 0).astype(np.float64))

        pn.drop(columns=["M0"], axis=1, inplace=True)

        dem_cols = ["M" + str(i) for i in range(1, max_month + 1)]
        if not "nmd" in type_pn:
            # Il faut conserver les NaN dans les NMDs car ils sont éliminés par la suite
            pn.loc[filtre_dem, dem_cols] = gf.fill_nan2(pn.loc[filtre_dem, dem_cols].copy())

        return pn

    def format_and_add_cols(self, pn):
        pn[gp.nc_output_book_cle] = [
            str(int(float(x))) if not re.match(r'^-?\d+(?:\.\d+)?$', str(x)) is None else x for x in
            pn[gp.nc_output_book_cle]]

        """ Ajout de la colonne bassin et ISIG aux données quali"""
        pn = self.ig.add_ig_colum(pn, gp.nc_output_palier_cle, gp.nc_output_contrat_cle)

        return pn

    def construct_new_key(self, pn, type_pn):
        pn['new_key'] = pn[gp.cle_pn].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        self.ensure_key_and_mapping_uniqueness(pn, type_pn)
        pn[gp.nc_output_key] = pn['new_key']
        pn = pn.set_index("new_key")
        return pn

    def ensure_key_and_mapping_uniqueness(self, pn, type_pn):
        """  On s'assure que les clés PN NMD ou PEL sont uniques """
        filter_dem = (pn[gp.nc_output_ind3] == self.nc_pn_ecm_pn) if "nmd" in type_pn else \
            (pn[gp.nc_output_ind3] == self.nc_pn_flem_pn)
        pn_test = pn[filter_dem].copy()
        if not pn_test['new_key'].is_unique and not "ech" in type_pn:
            logger.error(self.NON_UNIQUE_KEYS % type_pn.upper())
            logger.error(pn_test[pn_test['new_key'].duplicated()]["new_key"].values.tolist())
            raise ValueError(self.NON_UNIQUE_KEYS % type_pn.upper())

        """  On s'assure que les clés PN ECH sont uniques """
        if not (pn_test['new_key'] + pn_test[gp.nc_pn_cle_pn]).is_unique and "ech" in type_pn:
            logger.error(self.NON_UNIQUE_KEYS % type_pn.upper())
            filter = (pn_test['new_key'] + "_" + pn_test[gp.nc_pn_cle_pn]).duplicated()
            logger.error((pn_test.loc[filter, "new_key"] + pn_test.loc[filter, gp.nc_pn_cle_pn]).values.tolist())
            raise ValueError(self.NON_UNIQUE_KEYS % type_pn.upper())

        """  On s'assure qu'aucun mapping n'est manquant """
        if "MAPPING" in [str(x).upper() for x in pn_test['new_key']]:
            logger.error(self.MAPPING_MISSING % type_pn.upper())
            raise ValueError(self.MAPPING_MISSING % type_pn.upper())

    def rank_by_duree(self, pn):
        try:
            pn = pn.sort_values([gp.nc_pn_duree, gp.nc_pn_cle_pn])
        except TypeError:
            logger.error(
                "Toutes les durées de vos PNs ECH pou ECH% ne sont pas des nombres. Avez-vous un problème de mapping?")
            raise TypeError

        filter_zero_duree = pn[gp.nc_pn_duree] == 0

        if filter_zero_duree.any():
            logger.warning("    Certains produits ont une durée égale à 0")
        return pn

    def prolong_tx_cols(self, pn, type_pn):
        if "nmd" in type_pn:
            ind_tx = [self.nc_pn_tx_prod_cible, self.nc_pn_tx_sp_pn]
            name_new_cols = ["M" + str(p) for p in range(self.up.max_month_pn[type_pn] + 1, self.up.nb_mois_proj_usr + 1)]
            if len(name_new_cols) > 0:
                for ind in ind_tx:
                    filtre_ind = pn[gp.nc_output_ind3] == ind
                    ref_col = np.array(pn.loc[filtre_ind, "M" + str(self.up.max_month_pn[type_pn])])
                    pn.loc[filtre_ind, name_new_cols] = np.column_stack([ref_col] * len(name_new_cols))
        return pn.copy()
