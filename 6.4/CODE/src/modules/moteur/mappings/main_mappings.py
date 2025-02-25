# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:33:00 2020

@author: TAHIRIH
"""
import modules.moteur.utils.generic_functions as gf
import modules.moteur.parameters.general_parameters as gp
import modules.moteur.parameters.user_parameters as up
from modules.moteur.low_level_services.sources_services.sources_nomenclature import NomenclatureSaver
from params import version_params as vp
import mappings.general_mappings as gmp
import pandas as pd
import numpy as np
import utils.excel_utils as ex
import utils.general_utils as gu
import dateutil
import logging
import ntpath

logger = logging.getLogger(__name__)

global bassin_ig_map, contrats_map, index_tx_map, ig_mirror_contracts_map, liq_soc_map
global emm_map, dim_nsfr_map, bilan_cash_map, liq_ig_map, liq_comptes_map, liq_opfi_map
global eve_reg_floor, eve_contracts_excl, cc_sans_stress, cc_tci, act_eve
global conv_gpliq, conv_gps_nmd
global param_tx_ech, contrats_fx_swaps, param_nmd_base_calc, param_nmd, param_pn_ech_pricing
global curve_accruals_map
global retraitement_tla_eve, mois_refix_tla_eve, freq_refix_tla_eve, date_refix_tla_eve
global force_gp_liq_eve, force_gps_nmd_eve, index_curve_tenor_map
global coeffs_nsfr, coeffs_lcr, coeffs_lcr_spec, bc_outflow_nsfr, bc_outflow_lcr, mode_cal_gap_tx_immo
global tla_indexes, tlb_indexes, tdav_indexes, topc_indexes, lep_indexes, cel_indexes, inf_indexes,\
    all_gap_gestion_index, sc_file_indexes, bassins_map, sources, map_wb, cc_tci_excl

def load_mappings(etab):
    """ fonction permettant de charger les mappings globaux"""

    global bassin_ig_map, contrats_map, index_tx_map, ig_mirror_contracts_map, liq_soc_map
    global emm_map, dim_nsfr_map, bilan_cash_map, liq_ig_map, liq_comptes_map, liq_opfi_map
    global conv_gpliq, conv_gps_nmd, curve_accruals_map, index_curve_tenor_map, bassins_map
    global param_tx_ech, contrats_fx_swaps, param_nmd_base_calc, param_nmd, param_pn_ech_pricing
    global sources, map_wb

    #REDONDANCE A CORRIGER
    gmp.load_general_mappings(up.mapping_path, up.dar_usr)

    map_wb = ex.xl.Workbooks.Open(up.mapping_path, None, True)
    gu.check_version_templates(ntpath.basename(up.mapping_path), open=False, wb=map_wb, version=vp.version_map)
    ex.unfilter_all_sheets(map_wb)

    contract_map_all, contrats_map = load_contracts_map(map_wb)

    bassin_ig_map, ig_mirror_contracts_map = load_ig_maps(map_wb, contract_map_all)

    """ MAPPING index de taux """
    index_tx_map = load_map(map_wb, name_range=gp.ng_itm, cle_map=gp.nc_itm_index_calc, rename_old=True, rename_pref="nc_itm")

    """ BILAN CASH MAPPING """
    bilan_cash_map = load_map(map_wb, name_range=gp.ng_bc, cle_map=gp.bc_cle, rename_old=True, rename_pref="nc_bc", upper=True, join_key=True)

    """ MAPPING EMPREINTE MARCHE """
    emm_map = load_map(map_wb, name_range=gp.ng_emm, cle_map=gp.emm_cle, upper=True)

    """ MAPPING LIQ IG """
    liq_ig_map = load_map(map_wb, name_range=gp.ng_liq_ig, cle_map=gp.liq_ig_cle, upper=True)

    """ MAPPING BASSINS """
    bassins_map = load_map(map_wb, name_range=gp.ng_bassins_mp, cle_map=gp.nc_cle_bassins_mp, upper=True)

    """ MAPPING LIQ COMPTES """
    liq_comptes_map = load_map(map_wb, name_range=gp.ng_liq_cmpt, cle_map=gp.liq_cmpt_cle, upper=True)

    """ MAPPING LIQ OPFI """
    liq_opfi_map = load_map(map_wb, name_range=gp.ng_liq_opfi, cle_map=gp.liq_opfi_cle, upper=True)

    """ MAPPING SOCIAL AGREGE """
    liq_soc_map = load_map(map_wb, name_range=gp.ng_liq_soc, cle_map=gp.liq_soc_cle, upper=True)

    """ NSFR MAPPING"""
    dim_nsfr_map = load_map(map_wb, name_range=gp.ng_nsfr, cle_map=gp.nsfr_cle, rename_old=True, rename_pref="nc_nsfr", upper=True, join_key=True)

    """ ECH TX PARAM """
    param_tx_ech = load_map(map_wb, name_range=gp.ng_echtx, cle_map=gp.nc_echtx_cle)

    """ PN ECH PRICING """
    param_pn_ech_pricing = load_map(map_wb, name_range=gp.ng_ppn, cle_map=gp.nc_ppn_cle, join_key=True)

    """ CONTRATS FX SWAPs"""
    contrats_fx_swaps = load_map(map_wb, name_range=gp.ng_fx_swaps, cle_map=[gp.nc_contrat_fxsw])

    """ ECH NMD PARAM """
    param_nmd_base_calc = load_map(map_wb, name_range=gp.ng_nmd_basecalc, cle_map=[gp.nc_nmd_contrat])

    """ LOAD EVE SIMUL DATA"""
    load_params_eve(map_wb)

    """ CURVES ACCRUALS MAP """
    curve_accruals_map = load_map(map_wb, name_range=gp.ng_map_curve_accruals,
                                  cle_map=gp.cle_map_curve_accruals, upper=True,
                                  useful_cols=[gp.nc_accrual_method_curve_accruals,
                                               gp.nc_accrual_conversion_curve_accruals,
                                               gp.nc_type_courbe_curve_accruals,
                                               gp.nc_type_courbe_alias])

    """ INDEX CURVE_NAME MAP"""
    index_curve_tenor_map = load_map(map_wb, name_range=gp.ng_map_rate_code_curve,
                                     cle_map=gp.cle_mp_rate_code_curve, upper=True)

    """ CHARGEMENT DES CONVENTIONS D'ECOULEMENT GROUPE """
    """ ECOULEMENT GAP LIQ"""
    if up.force_gp_liq or force_gp_liq_eve:
        date_gp_liq = ex.get_value_from_named_ranged(map_wb, gp.ng_date_conv_gpliq)
        date_gp_liq = dateutil.parser.parse(str(date_gp_liq)).replace(tzinfo=None)
        if date_gp_liq < up.dar_usr:
            logger.warning(
                "      La date de l'onglet Conv_GapLiq doit être supérieure ou égale à celle de la date d'arrêté ")
        conv_gpliq = ex.get_dataframe_from_range(map_wb, gp.ng_conv_gpliq, header=True, alert="")
        conv_gpliq = conv_gpliq.drop_duplicates(subset=gp.ng_conv_gpliq_cle).set_index(gp.ng_conv_gpliq_cle).copy()
        #if up.nb_mois_proj_usr > gp.real_max_months:
            #conv_gpliq =  gf.prolong_last_col_value(conv_gpliq, up.nb_mois_proj_usr - gp.real_max_months, s=6, month_diff=True, suf="M")
        conv_gpliq = conv_gpliq[["M" + str(i) for i in range(0, up.nb_mois_proj_usr + 1)]]

    """ ECOULEMENT GAP NMD"""
    date_gp_nmd = ex.get_value_from_named_ranged(map_wb, gp.ng_date_conv_gps_nmd)
    date_gp_nmd = dateutil.parser.parse(str(date_gp_nmd)).replace(tzinfo=None)
    if date_gp_nmd < up.dar_usr and (up.force_gps_nmd or force_gps_nmd_eve):
        logger.warning(
            "      La date de l'onglet Conv_Gaps_NMD doit être supérieure ou égale à celle de la date d'arrêté ")
    conv_gps_nmd = ex.get_dataframe_from_range(map_wb, gp.ng_conv_gps_nmd, header=True, alert="")
    conv_gps_nmd = conv_gps_nmd.drop_duplicates(subset=gp.ng_conv_gps_nmd_cle).set_index(gp.ng_conv_gps_nmd_cle).copy()
    #if up.nb_mois_proj_usr > gp.real_max_months:
        #conv_gps_nmd = gf.prolong_last_col_value(conv_gps_nmd, up.nb_mois_proj_usr - gp.real_max_months, s=2, month_diff=True, suf="M")
    conv_gps_nmd = conv_gps_nmd[["M" + str(i) for i in range(0, up.nb_mois_proj_usr + 1)]]

    """ INDEXES"""
    get_indexes_list(map_wb)

    """ MAPPING SOURCES GRANULAIRES """
    sources = NomenclatureSaver()
    sources.get_nomenclature_contrats(map_wb, etab, up.source_path, up.dar_usr)
    sources.get_nomenclature_pn(map_wb, etab, up.source_path, up.dar_usr)

    #map_wb.Close(False)

def get_indexes_list(map_wb):
    global tla_indexes, tlb_indexes, tdav_indexes, topc_indexes, lep_indexes, cel_indexes, inf_indexes,\
        all_gap_gestion_index, sc_file_indexes

    tla_indexes = ex.get_dataframe_from_range(map_wb, gp.ng_tla_indexes, header=True, alert="").iloc[:,0].values.tolist()
    tlb_indexes = ex.get_dataframe_from_range(map_wb, gp.ng_tlb_indexes, header=True, alert="").iloc[:,0].values.tolist()
    tdav_indexes = ex.get_dataframe_from_range(map_wb, gp.ng_tdav_indexes, header=True, alert="").iloc[:,0].values.tolist()

    topc_indexes = ex.get_dataframe_from_range(map_wb, gp.ng_topc_indexes, header=True, alert="").iloc[:,0].values.tolist()

    lep_indexes = ex.get_dataframe_from_range(map_wb, gp.ng_lep_indexes, header=True, alert="").iloc[:,0].values.tolist()
    cel_indexes = ex.get_dataframe_from_range(map_wb, gp.ng_cel_indexes, header=True, alert="").iloc[:,0].values.tolist()
    inf_indexes = ex.get_dataframe_from_range(map_wb, gp.ng_inf_indexes, header=True, alert="").iloc[:,0].values.tolist()

    all_gap_gestion_index = tla_indexes + tlb_indexes + lep_indexes + cel_indexes
    sc_file_indexes = ["1D", "3M", "6M", "12M", "5Y", "10Y", "TAUX_PEL", "iTMO"] + tla_indexes + inf_indexes



def load_params_eve(map_wb):
    global eve_reg_floor, eve_contracts_excl, cc_sans_stress, cc_tci, act_eve, cc_tci_excl, mode_cal_gap_tx_immo

    suf_eve = "_ICAP" if str(up.type_eve)=="ICAAP" else ""

    cc_sans_stress = ex.get_dataframe_from_range(map_wb, gp.ng_contrat_ss_stress + suf_eve).iloc[:, 0].values.tolist()

    cc_tci = ex.get_dataframe_from_range(map_wb, gp.ng_contrat_tci_pn + suf_eve).iloc[:, 0].values.tolist()

    cc_tci_excl = ex.get_dataframe_from_range(map_wb, gp.ng_contrat_tci_pn_excl + suf_eve).iloc[:, 0].values.tolist()

    """ OPTIONS d'ACTUALISATION EVE """
    act_eve = ex.get_dataframe_from_range(map_wb, gp.ng_act_eve + suf_eve, header=True, alert="")
    act_eve = act_eve.set_index([gp.nc_act_eve_dev, gp.nc_act_eve_bilan, gp.nc_act_eve_contrat])

    """ CONTRAT EVE A EXCLURE """
    eve_contracts_excl = load_map(map_wb, name_range=gp.ng_eve_contracts + suf_eve, cle_map=gp.nc_contrat_eve)

    """ REG FLOOR EVE"""
    eve_reg_floor = ex.get_dataframe_from_range(map_wb, gp.ng_eve_reg_floor + suf_eve, header=True, alert="")
    eve_reg_floor = np.array(eve_reg_floor.iloc[:, 1]).reshape(1, eve_reg_floor.shape[0])

    """ TLA EVE """
    load_tla_refixing_params_eve(map_wb, suf_eve)

    """ LOAD CONV ECOUL EVE"""
    load_conv_ecoulements_eve(map_wb, suf_eve)

    mode_cal_gap_tx_immo = ex.get_value_from_named_ranged(map_wb, gp.nc_mode_calc_gptx_immo + suf_eve, alert="")


def load_conv_ecoulements_eve(wb, suf_eve):
    global force_gp_liq_eve, force_gps_nmd_eve

    force_gp_liq_eve = ex.get_value_from_named_ranged(wb, gp.ng_force_gp_liq_eve + suf_eve, alert="")
    force_gp_liq_eve = False if force_gp_liq_eve == "NON" else True

    """ OPTION POUR FORCER LES GAPS DES NMDS """
    force_gps_nmd_eve = ex.get_value_from_named_ranged(wb, gp.ng_force_gps_nmd_eve + suf_eve, alert="")
    force_gps_nmd_eve = False if force_gps_nmd_eve == "NON" else True


def load_tla_refixing_params_eve(wb, suf_eve):
    global retraitement_tla_eve, mois_refix_tla_eve, freq_refix_tla_eve, date_refix_tla_eve
    """ TX REFIXING """
    retraitement_tla_eve = ex.get_value_from_named_ranged(wb, gp.ng_tla_retraitement_eve + suf_eve)
    retraitement_tla_eve = True if retraitement_tla_eve.upper() == "OUI" else False
    date_refix_tla_eve = ex.get_value_from_named_ranged(wb, gp.ng_date_refix_eve + suf_eve)
    date_refix_tla_eve = dateutil.parser.parse(str(date_refix_tla_eve)).replace(tzinfo=None)
    mois_refix_tla_eve = date_refix_tla_eve.year * 12 + date_refix_tla_eve.month - up.dar_usr.year * 12 - up.dar_usr.month
    try:
        freq_refix_tla_eve = int(ex.get_value_from_named_ranged(wb, gp.ng_freq_refix_eve + suf_eve))
    except:
        freq_refix_tla_eve = 4
    if mois_refix_tla_eve < 1 and retraitement_tla_eve:
        logger.error("   La date de refixing TLA est inférieure ou égale à la DAR")
        raise ValueError("   La date de refixing TLA est inférieure ou égale à la DAR")

    return retraitement_tla_eve, mois_refix_tla_eve, freq_refix_tla_eve

def load_map(wb, name_range="", cle_map="", rename_old=False, rename_pref="", upper=False, join_key=False, useful_cols=[]):
    """ MAPPING index de taux """
    mapping = ex.get_dataframe_from_range(wb, name_range)
    if rename_old:
        mapping = gf.rename_mapping_columns(mapping, gp, rename_pref, "gp")
    if upper:
        mapping = gu.strip_and_upper(mapping, cle_map)
    if join_key:
        mapping['new_key'] = mapping[cle_map].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        mapping = mapping.drop_duplicates(subset=cle_map).set_index('new_key').drop(columns=cle_map, axis=1).copy()
    else:
        mapping = mapping.drop_duplicates(subset=cle_map).set_index(cle_map).copy()
    if useful_cols !=[]:
        mapping = mapping[useful_cols].copy()

    return mapping

def load_contracts_map(wb):
    """ mapping des contrats"""
    all_ct_amp = ex.get_dataframe_from_range(wb, gp.ng_cp, header=True, alert="")
    all_ct_amp = gf.rename_mapping_columns(all_ct_amp, gp, "nc_cp", "gp")
    all_ct_amp[gp.nc_cp_isech] = [True if x == "O" else False for x in all_ct_amp[gp.nc_cp_isech]]

    contrats_map = all_ct_amp[
        [gp.nc_cp_contrat, gp.nc_cp_isech, gp.nc_cp_bilan, gp.nc_cp_poste] + [eval("gp.nc_cp_dim" + str(i)) for i in
                                                                              range(2, 6)]].copy()
    contrats_map = contrats_map.drop_duplicates(subset=gp.cle_mc)
    contrats_map = contrats_map.set_index(gp.cle_mc)

    return all_ct_amp, contrats_map

def load_ig_maps(wb, contract_map):
    """ bassin des intra-groupes """
    bassin_ig_map = ex.get_dataframe_from_range(wb, gp.ng_igm, header=True, alert="")
    bassin_ig_map.rename(columns={gp.nc_igm_bassin: gp.nc_cp_bassin}, inplace=True)

    """ mappings des contrats contreparties """
    ig_map = contract_map[[gp.nc_cp_contrat, gp.nc_cp_bassin, gp.nc_cp_mirr_ig_ntx, gp.nc_cp_mirr_ig_bpce, gp.nc_cp_mirr_ig_rzo]]
    ig_map = pd.merge(how="left", left=ig_map, right=bassin_ig_map, on=gp.nc_cp_bassin)
    ig_map[gp.nc_igm_bassinig] = ig_map[gp.nc_igm_bassinig].fillna("-")
    ig_map = ig_map.melt(id_vars=[gp.nc_cp_contrat, gp.nc_cp_bassin, gp.nc_igm_bassinig], var_name=gp.nc_hierarchie,
                         value_name=gp.nc_ig_contrat_new)

    ig_map['key_ig'] = ig_map[gp.cle_igc].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    ig_map = ig_map.drop_duplicates(['key_ig'])
    ig_map = ig_map.set_index('key_ig').copy()

    ig_mirror_contracts_map = ig_map.loc[
                              (~ ig_map[gp.nc_ig_contrat_new].isnull()) & (ig_map[gp.nc_ig_contrat_new] != "-"), :]

    if up.bassin_usr == "SEF":
        ig_mirror_contracts_map = ig_mirror_contracts_map.loc[ig_mirror_contracts_map[gp.nc_hierarchie] != "NTX", :]

    bassin_ig_map = bassin_ig_map.set_index(gp.nc_cp_bassin)

    return bassin_ig_map, ig_mirror_contracts_map


def map_liq(data, override=True):
    """ Fonctions permettant de trouver les mapping conso des contrats contrepartie"""

    ordered_cols = data.columns

    """ MAPPING LIQ BILAN CASH"""
    cles_a_combiner = [gp.nc_output_contrat_cle, gp.nc_output_book_cle, gp.nc_output_lcr_tiers_cle]

    data = gu.map_with_combined_key2(data, bilan_cash_map, cles_a_combiner, symbol_any="-", \
                                    override=override, name_mapping="MAPPING BILAN CASH", \
                                    tag_absent_override=True)

    """ AUTRES mappings LIQUIDITE """
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
    mappings = [emm_map, liq_ig_map, liq_comptes_map, liq_opfi_map, liq_soc_map]
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
    data = gu.map_with_combined_key2(data, dim_nsfr_map, cles_a_combiner, symbol_any="*", \
                                    override=override, name_mapping="MAPPING NSFR", \
                                    tag_absent_override=True)

    data = data[ordered_cols]

    return data