# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:33:00 2020

@author: TAHIRIH
"""
import modules.moteur.utils.generic_functions as gf
import modules.moteur.parameters.general_parameters as gp
import mappings.general_mappings as gmp
import pandas as pd
import numpy as np
import utils.excel_openpyxl as ex
import utils.general_utils as gu
import dateutil
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

logger = logging.getLogger(__name__)


class Mappings():

    def __init__(self, cls_usr):
        self.up = cls_usr

    def load_mappings(self):
        """ fonction permettant de charger les indicators globaux"""

        # REDONDANCE A CORRIGER
        self.map_wb = ex.load_workbook_openpyxl(self.up.sm.mapp_file_path, read_only=True)

        self.contract_map_all, self.contrats_map = self.load_contracts_map()

        self.bassin_ig_map, self.ig_mirror_contracts_map = self.load_ig_maps(self.contract_map_all)

        """ MAPPING index de taux """
        self.index_tx_map = self.load_map(self.map_wb, name_range=gp.ng_itm, cle_map=gp.nc_itm_index_calc,
                                          rename_old=True, rename_pref="nc_itm")

        """ BILAN CASH MAPPING """
        self.bilan_cash_map = self.load_map(self.map_wb, name_range=gp.ng_bc, cle_map=gp.bc_cle, rename_old=True,
                                            rename_pref="nc_bc", upper=True, join_key=True)

        """ MAPPING EMPREINTE MARCHE """
        self.emm_map = self.load_map(self.map_wb, name_range=gp.ng_emm, cle_map=gp.emm_cle, upper=True)

        """ MAPPING LIQ IG """
        self.liq_ig_map = self.load_map(self.map_wb, name_range=gp.ng_liq_ig, cle_map=gp.liq_ig_cle, upper=True)

        """ MAPPING BASSINS """
        self.bassins_map = gmp.map_pass_alm["BASSINS"]["TABLE"]

        """ MAPPING LIQ COMPTES """
        self.liq_comptes_map = self.load_map(self.map_wb, name_range=gp.ng_liq_cmpt, cle_map=gp.liq_cmpt_cle,
                                             upper=True)

        """ MAPPING LIQ OPFI """
        self.liq_opfi_map = self.load_map(self.map_wb, name_range=gp.ng_liq_opfi, cle_map=gp.liq_opfi_cle, upper=True)

        """ MAPPING SOCIAL AGREGE """
        self.liq_soc_map = self.load_map(self.map_wb, name_range=gp.ng_liq_soc, cle_map=gp.liq_soc_cle, upper=True)

        """ NSFR MAPPING"""
        self.dim_nsfr_map = self.load_map(self.map_wb, name_range=gp.ng_nsfr, cle_map=gp.nsfr_cle, rename_old=True,
                                          rename_pref="nc_nsfr", upper=True, join_key=True)


        """ PN ECH PRICING """
        self.param_pn_ech_pricing = gmp.mapping_taux["COURBES_PRICING_PN"]

        """ CURVES ACCRUALS MAP """
        self.curve_accruals_map = gmp.mapping_taux["CURVES_BASIS_CONV"]

        """ INDEX CURVE_NAME MAP"""
        self.index_curve_tenor_map = gmp.mapping_taux["RATE_CODE-CURVE"]

        """ CONTRATS FX SWAPs"""
        self.contrats_fx_swaps = self.load_map(self.map_wb, name_range=gp.ng_fx_swaps, cle_map=[gp.nc_contrat_fxsw])

        """ CHARGEMENT DES CONVENTIONS D'ECOULEMENT GROUPE """
        """ ECOULEMENT GAP LIQ"""


        """ ECH TX PARAM """
        self.param_tx_ech = self.load_map(self.map_wb, name_range=gp.ng_echtx, cle_map=gp.nc_echtx_cle)

        """ ECH NMD PARAM """
        self.param_nmd_base_calc = self.load_map(self.map_wb, name_range=gp.ng_nmd_basecalc,
                                                 cle_map=[gp.nc_nmd_contrat])

        self.mapping_eve = gmp.mapping_eve_icaap if str(self.up.type_eve) == "ICAAP" else gmp.mapping_eve

        self.mapping_gp_reg_params = gmp.mapping_gp_reg_params

        if self.up.force_gp_liq or self.mapping_eve["force_gp_liq_eve"]:
            date_gp_liq = ex.get_value_from_named_ranged(self.map_wb, gp.ng_date_conv_gpliq)
            date_gp_liq = dateutil.parser.parse(str(date_gp_liq)).replace(tzinfo=None)
            if date_gp_liq < self.up.dar:
                logger.warning(
                    "      La date de l'onglet Conv_GapLiq doit être supérieure ou égale à celle de la date d'arrêté ")
            conv_gpliq = ex.get_dataframe_from_range(self.map_wb, gp.ng_conv_gpliq, header=True)
            conv_gpliq = conv_gpliq.drop_duplicates(subset=gp.ng_conv_gpliq_cle).set_index(gp.ng_conv_gpliq_cle).copy()
            self.conv_gpliq = conv_gpliq[["M" + str(i) for i in range(0, self.up.nb_mois_proj_usr + 1)]]

        """ ECOULEMENT GAP NMD"""
        date_gp_nmd = ex.get_value_from_named_ranged(self.map_wb, gp.ng_date_conv_gps_nmd)
        date_gp_nmd = dateutil.parser.parse(str(date_gp_nmd)).replace(tzinfo=None)
        if date_gp_nmd < self.up.dar and (self.up.force_gps_nmd or self.mapping_eve["force_gps_nmd_eve"]):
            logger.warning(
                "      La date de l'onglet Conv_Gaps_NMD doit être supérieure ou égale à celle de la date d'arrêté ")
        conv_gps_nmd = ex.get_dataframe_from_range(self.map_wb, gp.ng_conv_gps_nmd, header=True)
        conv_gps_nmd = conv_gps_nmd.drop_duplicates(subset=gp.ng_conv_gps_nmd_cle).set_index(
            gp.ng_conv_gps_nmd_cle).copy()
        self.conv_gps_nmd = conv_gps_nmd[["M" + str(i) for i in range(0, self.up.nb_mois_proj_usr + 1)]]



    @staticmethod
    def load_map(map_wb, name_range="", cle_map="", rename_old=False, rename_pref="", upper=False, join_key=False,
                 useful_cols=[]):
        """ MAPPING index de taux """
        mapping = ex.get_dataframe_from_range(map_wb, name_range)
        if rename_old:
            mapping = gf.rename_mapping_columns(mapping, gp, rename_pref, "gp")
        if upper:
            mapping = gu.strip_and_upper(mapping, cle_map)
        if join_key:
            mapping['new_key'] = mapping[cle_map].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            mapping = mapping.drop_duplicates(subset=cle_map).set_index('new_key').drop(columns=cle_map, axis=1).copy()
        else:
            mapping = mapping.drop_duplicates(subset=cle_map).set_index(cle_map).copy()
        if useful_cols != []:
            mapping = mapping[useful_cols].copy()

        return mapping

    def load_contracts_map(self):
        """ mapping des contrats"""
        all_ct_map = ex.get_dataframe_from_range(self.map_wb, gp.ng_cp, header=True)
        all_ct_map = gf.rename_mapping_columns(all_ct_map, gp, "nc_cp", "gp")
        all_ct_map[gp.nc_cp_isech] = [True if x == "O" else False for x in all_ct_map[gp.nc_cp_isech]]

        contrats_map = all_ct_map[
            [gp.nc_cp_contrat, gp.nc_cp_isech, gp.nc_cp_bilan, gp.nc_cp_poste] + [eval("gp.nc_cp_dim" + str(i)) for i in
                                                                                  range(2, 6)]].copy()
        contrats_map = contrats_map.drop_duplicates(subset=gp.cle_mc)
        contrats_map = contrats_map.set_index(gp.cle_mc)

        return all_ct_map, contrats_map

    def load_ig_maps(self, contract_map):
        """ bassin des intra-groupes """
        bassin_ig_map = ex.get_dataframe_from_range(self.map_wb, gp.ng_igm, header=True)
        bassin_ig_map.rename(columns={gp.nc_igm_bassin: gp.nc_cp_bassin}, inplace=True)

        """ indicators des contrats contreparties """
        ig_map = contract_map[
            [gp.nc_cp_contrat, gp.nc_cp_bassin, gp.nc_cp_mirr_ig_ntx, gp.nc_cp_mirr_ig_bpce, gp.nc_cp_mirr_ig_rzo]]
        ig_map = pd.merge(how="left", left=ig_map, right=bassin_ig_map, on=gp.nc_cp_bassin)
        ig_map[gp.nc_igm_bassinig] = ig_map[gp.nc_igm_bassinig].fillna("-")
        ig_map = ig_map.melt(id_vars=[gp.nc_cp_contrat, gp.nc_cp_bassin, gp.nc_igm_bassinig], var_name=gp.nc_hierarchie,
                             value_name=gp.nc_ig_contrat_new)

        ig_map['key_ig'] = ig_map[gp.cle_igc].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        ig_map = ig_map.drop_duplicates(['key_ig'])
        ig_map = ig_map.set_index('key_ig').copy()

        ig_mirror_contracts_map = ig_map.loc[
                                  (~ ig_map[gp.nc_ig_contrat_new].isnull()) & (ig_map[gp.nc_ig_contrat_new] != "-"), :]

        bassin_ig_map = bassin_ig_map.set_index(gp.nc_cp_bassin)

        return bassin_ig_map, ig_mirror_contracts_map

    def adapt_map_to_sef(self):
        if self.up.bassin_usr == "SEF":
            self.ig_mirror_contracts_map\
                = self.ig_mirror_contracts_map.loc[self.ig_mirror_contracts_map[gp.nc_hierarchie] != "NTX", :]

    def map_liq(self, data, override=True):
        """ Fonctions permettant de trouver les mapping conso des contrats contrepartie"""

        ordered_cols = data.columns

        """ MAPPING LIQ BILAN CASH"""
        cles_a_combiner = [gp.nc_output_contrat_cle, gp.nc_output_book_cle, gp.nc_output_lcr_tiers_cle]

        data = gu.map_with_combined_key2(data, self.bilan_cash_map, cles_a_combiner, symbol_any="-", \
                                         override=override, name_mapping="MAPPING BILAN CASH", \
                                         tag_absent_override=True)

        """ AUTRES indicators LIQUIDITE """
        keys_EM = [gp.nc_output_bilan, gp.nc_output_r1, gp.nc_output_maturite_cle]
        keys_liq_IG = [gp.nc_output_bilan, gp.nc_output_bc, gp.nc_output_bassin_cle, \
                       gp.nc_output_contrat_cle, gp.nc_output_palier_cle]
        keys_liq_CT = [gp.nc_output_bilan, gp.nc_output_r1, gp.nc_output_bassin_cle, \
                       gp.nc_output_palier_cle]
        keys_liq_FI = [gp.nc_output_bilan, gp.nc_output_bassin_cle, \
                       gp.nc_output_bc, gp.nc_output_r1, gp.nc_output_contrat_cle, "IG/HG Social"]
        keys_liq_SC = [gp.nc_output_soc1]
        data["IG/HG Social"] = np.where(data[gp.nc_output_palier_cle] == "-", "HG", "IG")

        keys_data = [keys_EM, keys_liq_IG, keys_liq_CT, keys_liq_FI, keys_liq_SC]
        mappings = [self.emm_map, self.liq_ig_map, self.liq_comptes_map, self.liq_opfi_map, self.liq_soc_map]
        name_mappings = ["EMPREINTE DE MARCHE", "LIQ IG", "LIQ COMPTES", "LIQ OPE FI", "SOC AGREG"]
        for i in range(0, len(mappings)):
            mapping = mappings[i]
            name_mapping = name_mappings[i]
            key_data = keys_data[i]
            tag_absent_override = True if name_mapping == "SOC AGREG" else False
            data = gu.map_data(data, mapping, keys_data=key_data, override=override, \
                               name_mapping=name_mapping, tag_absent_override=tag_absent_override)

        data = data.drop(["IG/HG Social"], axis=1)

        """ MAPPING NSFR """
        cles_a_combiner = [gp.nc_output_contrat_cle, gp.nc_output_lcr_tiers_cle]
        data = gu.map_with_combined_key2(data, self.dim_nsfr_map, cles_a_combiner, symbol_any="*", \
                                         override=override, name_mapping="MAPPING NSFR", \
                                         tag_absent_override=True)

        data = data[ordered_cols]

        return data
