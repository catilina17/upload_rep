import utils.general_utils as gu
from params import version_params as vp
import modules.alim.parameters.NTX_SEF_params as ntx_p
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import utils.excel_openpyxl as ex
import dateutil
import os
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

global liquidity_cols, base_calc_ech, mapping_bpce_pn
global exceptions_missing_mappings
global nomenclature_stock_ag, nomenclature_contrats, nomenclature_pn, map_pass_alm, mapping_IG, mapping_NTX, mapping_liquidite, mapping_PN
global mapping_bpce, nomenclature_lcr_nsfr, mapping_taux
global mapping_eve, mapping_gp_reg_params, mapping_eve_icaap
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def load_general_mappings(mapp_file_path, dar, sources_folder):
    global missing_mapping, liquidity_cols, exceptions_missing_mappings
    global nomenclature_stock_ag, nomenclature_contrats, nomenclature_pn, map_pass_alm, mapping_IG, mapping_NTX, mapping_liquidite, mapping_PN
    global nomenclature_contrats, mapping_bpce_pn
    global mapping_bpce, nomenclature_lcr_nsfr, mapping_taux, mapping_eve, mapping_gp_reg_params, mapping_eve_icaap
    liquidity_cols = []
    exceptions_missing_mappings = {}

    mapping_wb = ex.load_workbook_openpyxl(mapp_file_path, read_only=True)
    gu.check_version_templates(open=False, wb=mapping_wb, version=vp.version_map)

    map_pass_alm = lecture_mapping_principaux(mapping_wb)

    mapping_IG = lecture_mapping_IG(mapping_wb)
    mapping_NTX = lecture_mapping_NTX(mapping_wb)
    mapping_liquidite = lecture_mapping_liquidite(mapping_wb)
    mapping_PN = lecture_mapping_PN(mapping_wb)
    nomenclature_stock_ag, nomenclature_contrats, nomenclature_pn, nomenclature_lcr_nsfr \
        = get_mapping_input_files(mapping_wb, dar, sources_folder)

    mapping_bpce = lecture_mapping_bpce(mapping_wb)

    mapping_bpce_pn = lecture_mapping_bpce_pn(mapping_wb)

    mapping_taux = get_mapping_taux(mapping_wb)

    """ LOAD EVE SIMUL DATA"""
    mapping_eve = load_params_eve(mapping_wb, dar)
    mapping_eve_icaap = load_params_eve(mapping_wb, dar, "_ICAP")

    """ INDEXES"""
    mapping_gp_reg_params = get_gap_reg_params(mapping_wb)

    fill_exceptions_mappings(mapping_wb)

    ex.close_workbook(mapping_wb)


def get_gap_reg_params(map_wb):
    mapping = {}

    ng_tla_indexes = "_TLA_INDEXES"
    ng_tlb_indexes = "_TLB_INDEXES"
    ng_tdav_indexes = "_TDAV_INDEXES"
    ng_topc_indexes = "_TOPC_INDEXES"
    ng_lep_indexes = "_LEP_INDEXES"
    ng_cel_indexes = "_CEL_INDEXES"
    ng_inf_indexes = "_INF_INDEXES"
    ng_tf_tla = "_tf_tla"
    ng_tf_tlb = "_tf_tlb"
    ng_tf_cel = "_tf_cel"
    ng_inf_tla = "_inf_tla"
    ng_inf_tlb = "_inf_tlb"
    ng_inf_cel = "_inf_cel"
    ng_reg_tf = "_reg_tf"
    ng_reg_inf = "_reg_inf"

    mapping["tla_indexes"] = ex.get_dataframe_from_range(map_wb, ng_tla_indexes, header=True).iloc[:, 0].values.tolist()
    mapping["tlb_indexes"] = ex.get_dataframe_from_range(map_wb, ng_tlb_indexes, header=True).iloc[:, 0].values.tolist()
    mapping["tdav_indexes"] = ex.get_dataframe_from_range(map_wb, ng_tdav_indexes, header=True).iloc[:,
                              0].values.tolist()

    mapping["topc_indexes"] = ex.get_dataframe_from_range(map_wb, ng_topc_indexes, header=True).iloc[:,
                              0].values.tolist()

    mapping["lep_indexes"] = ex.get_dataframe_from_range(map_wb, ng_lep_indexes, header=True).iloc[:, 0].values.tolist()
    mapping["cel_indexes"] = ex.get_dataframe_from_range(map_wb, ng_cel_indexes, header=True).iloc[:, 0].values.tolist()
    mapping["inf_indexes"] = ex.get_dataframe_from_range(map_wb, ng_inf_indexes, header=True).iloc[:, 0].values.tolist()

    mapping["all_gap_gestion_index"] = (mapping["tla_indexes"] + mapping["tlb_indexes"]
                                        + mapping["lep_indexes"] + mapping["cel_indexes"])

    mapping["coeff_tf_tla_usr"] = ex.get_value_from_named_ranged(map_wb, ng_tf_tla)
    mapping["coeff_tf_tlb_usr"] = ex.get_value_from_named_ranged(map_wb, ng_tf_tlb)
    mapping["coeff_tf_cel_usr"] = ex.get_value_from_named_ranged(map_wb, ng_tf_cel)
    mapping["coeff_inf_tla_usr"] = ex.get_value_from_named_ranged(map_wb, ng_inf_tla)
    mapping["coeff_inf_tlb_usr"] = ex.get_value_from_named_ranged(map_wb, ng_inf_tlb)
    mapping["coeff_inf_cel_usr"] = ex.get_value_from_named_ranged(map_wb, ng_inf_cel)
    mapping["coeff_reg_tf_usr"] = ex.get_value_from_named_ranged(map_wb, ng_reg_tf)
    mapping["coeff_reg_inf_usr"] = ex.get_value_from_named_ranged(map_wb, ng_reg_inf)

    return mapping


def load_params_eve(map_wb, dar, suf_eve=""):
    mapping = {}
    ng_eve_contracts = "_EVE_EXCL_CONT"
    nc_contrat_eve = "CONTRAT"
    ng_eve_reg_floor = "_REG_FLOOR_EVE"
    ng_contrat_ss_stress = "_CC_SANS_STRESS"
    ng_contrat_tci_pn = "_TCI_PN"
    ng_contrat_tci_pn_excl = "_TCI_PN_EXCL"
    ng_tla_retraitement_eve = "_TLA_IsRetraitement_EVE"
    ng_date_refix_eve = "_TLA_NextRefix_EVE"
    ng_freq_refix_eve = "_FREQ_REFIX_TLA_EVE"
    ng_act_eve = "_ACT_EVE"
    nc_act_eve_bilan = "BILAN"
    nc_act_eve_contrat = "CONTRAT"
    nc_act_eve_dev = "DEVISE"
    nc_mode_calc_gptx_immo = "_MODE_CALC_GP_TX_IMMO"
    ng_force_gp_liq_eve = "_force_gap_liq_eve"
    ng_force_gps_nmd_eve = "_force_gaps_nmd_eve"

    mapping["cc_sans_stress"] = ex.get_dataframe_from_range(map_wb, ng_contrat_ss_stress + suf_eve).iloc[:,
                                0].values.tolist()

    mapping["cc_tci"] = ex.get_dataframe_from_range(map_wb, ng_contrat_tci_pn + suf_eve).iloc[:, 0].values.tolist()

    mapping["cc_tci_excl"] = ex.get_dataframe_from_range(map_wb, ng_contrat_tci_pn_excl + suf_eve).iloc[:,
                             0].values.tolist()

    """ OPTIONS d'ACTUALISATION EVE """
    mapping["act_eve"] = ex.get_dataframe_from_range(map_wb, ng_act_eve + suf_eve, header=True)
    mapping["act_eve"] = mapping["act_eve"].set_index([nc_act_eve_dev, nc_act_eve_bilan, nc_act_eve_contrat])

    """ CONTRAT EVE A EXCLURE """
    mapping["eve_contracts_excl"] = ex.get_dataframe_from_range(map_wb, ng_eve_contracts + suf_eve)
    mapping["eve_contracts_excl"] = mapping["eve_contracts_excl"].drop_duplicates(nc_contrat_eve).set_index(
        nc_contrat_eve)

    """ REG FLOOR EVE"""
    mapping["eve_reg_floor"] = ex.get_dataframe_from_range(map_wb, ng_eve_reg_floor + suf_eve, header=True)
    mapping["eve_reg_floor"] = np.array(mapping["eve_reg_floor"].iloc[:, 1]).reshape(1,
                                                                                     mapping["eve_reg_floor"].shape[0])

    """ TLA EVE """
    mapping["retraitement_tla_eve"] = ex.get_value_from_named_ranged(map_wb, ng_tla_retraitement_eve + suf_eve)
    mapping["retraitement_tla_eve"] = True if mapping["retraitement_tla_eve"].upper() == "OUI" else False
    mapping["date_refix_tla_eve"] = ex.get_value_from_named_ranged(map_wb, ng_date_refix_eve + suf_eve)
    mapping["date_refix_tla_eve"] = dateutil.parser.parse(str(mapping["date_refix_tla_eve"])).replace(tzinfo=None)
    mapping["mois_refix_tla_eve"] \
        = (mapping["date_refix_tla_eve"].year * 12 + mapping["date_refix_tla_eve"].month
           - dar.year * 12 - dar.month)
    try:
        mapping["freq_refix_tla_eve"] = int(ex.get_value_from_named_ranged(map_wb, ng_freq_refix_eve + suf_eve))
    except:
        mapping["freq_refix_tla_eve"] = 4
    if mapping["mois_refix_tla_eve"] < 1 and mapping["retraitement_tla_eve"]:
        logger.error("   La date de refixing TLA est inférieure ou égale à la DAR")
        raise ValueError("   La date de refixing TLA est inférieure ou égale à la DAR")

    """ LOAD CONV ECOUL EVE"""
    mapping["force_gp_liq_eve"] = ex.get_value_from_named_ranged(map_wb, ng_force_gp_liq_eve + suf_eve)
    mapping["force_gp_liq_eve"] = False if mapping["force_gp_liq_eve"] == "NON" else True

    """ OPTION POUR FORCER LES GAPS DES NMDS """
    mapping["force_gps_nmd_eve"] = ex.get_value_from_named_ranged(map_wb, ng_force_gps_nmd_eve + suf_eve)
    mapping["force_gps_nmd_eve"] = False if mapping["force_gps_nmd_eve"] == "NON" else True

    mapping["mode_cal_gap_tx_immo"] = ex.get_value_from_named_ranged(map_wb, nc_mode_calc_gptx_immo + suf_eve, )

    return mapping


def get_mapping_taux(map_wb):
    mapping_taux = {}

    name_ranges = ['ref_pass_alm_fst_cell', 'COURBES_A_INTERPOLER',
                   '_COURBES_PRICING_PN', '_FILTRE_TENOR', "_MP_CURVE_ACCRUALS2", "_MAP_RATE_CODE_CURVE",
                   '_COURBES_A_BOOSTRAPPER', '_DATA_PN_PRICING']
    renames = [{}] * len(name_ranges)
    keys = ([[]] * (len(name_ranges) - 4) + [["CURVE_NAME", "TENOR_BEGIN"]]
            + [[], [], ["CONTRAT", "DIM2", "BILAN", "TF/TV", "MARCHE", "DEVISE", "INDEX CALC"]])
    useful_cols = ([[]] * (len(name_ranges) - 4) + [["ACCRUAL_METHOD", "ACCRUAL_CONVERSION", "TYPE DE COURBE"]]
                   + [[], [], ["COURBE PRICING"]])
    joinkeys = [False] * (len(name_ranges) - 1) + [True]
    force_int_str = [False] * len(name_ranges)
    upper_content = [True] * len(name_ranges)
    drop_duplicates = [True] * len(name_ranges)
    mappings_full_name = ["REFERENTIEL DES COURBES DE TAUX",
                          "COURBES A INTERPOLER",
                          "COURBES A CACLCULER",
                          "COURBES AUXILIAIRES",
                          "CONVENTIONS DE BASE DES COURBES", "MAPPING ENTRE CURVE-TENOR et RATE_CODE-DEVISE",
                          "COURBES A ZC", "COURBES DE PRCING DE LA PN"]
    mappings_name = ["REF_TX", "CURVES_TO_INTERPOLATE", "CALCULATED_CURVES", "AUXILIARY_CALCULATED_CURVES",
                     'CURVES_BASIS_CONV', "RATE_CODE-CURVE", "COURBES A ZC", "COURBES_PRICING_PN"]

    est_facultatif = [False] * len(name_ranges)
    mode_pass_alm = [False] * len(name_ranges)
    filter_cols = [False] * (len(name_ranges) - 3) + [True, False, False]

    for i in range(0, len(name_ranges)):
        mapping_data = ex.get_dataframe_from_range(map_wb, name_ranges[i])
        if len(renames[i]) != 0:
            mapping_data = mapping_data.rename(columns=renames[i])

        mapping_taux[mappings_name[i]] = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i],
                                                     mapping_data,
                                                     est_facultatif[i], joinkeys[i],
                                                     force_int_str=force_int_str[i],
                                                     upper_content=upper_content[i],
                                                     drop_duplicates=drop_duplicates[i],
                                                     mode_pass_alm=mode_pass_alm[i],
                                                     filter_cols=filter_cols[i])
    return mapping_taux


def gen_mapping(keys, useful_cols, mapping_full_name, mapping_data, est_facultatif, joinkey,
                force_int_str=False, upper_content=True, drop_duplicates=True, mode_pass_alm=True,
                filter_cols=False):
    mapping = {}

    if filter_cols:
        if len(keys) + len(useful_cols) > 0:
            mapping_data = mapping_data[keys + useful_cols].copy()

    if len(keys) > 0:
        if upper_content:
            mapping_data = gu.strip_and_upper(mapping_data, keys)

        if drop_duplicates:
            mapping_data = mapping_data.drop_duplicates(subset=keys).copy()

    if force_int_str:
        if len(keys + useful_cols) > 0:
            for col in keys + useful_cols:
                mapping_data = gu.force_integer_to_string(mapping_data, col)

    if joinkey:
        if len(keys) > 0:
            mapping_data["KEY"] = mapping_data[keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            keys = ["KEY"]

    if mode_pass_alm:
        if len(keys) > 0:
            mapping["TABLE"] = mapping_data.set_index(keys)
        else:
            mapping["TABLE"] = mapping_data

        mapping["OUT"] = useful_cols
        mapping["FULL_NAME"] = mapping_full_name

        mapping["est_facultatif"] = est_facultatif
    else:
        if len(keys) > 0:
            mapping = mapping_data.set_index(keys)
        else:
            mapping = mapping_data.copy()

    return mapping


def get_mapping_input_files(mapping_wb, dar, sources_folder):
    nomenclature_stock_ag = ex.get_dataframe_from_range(mapping_wb, "_NOMENCLATURE_SOURCES")
    nomenclature_pn = ex.get_dataframe_from_range(mapping_wb, "_NOMENCLATURE_CONTRATS_PN")
    nomenclature_contrats = ex.get_dataframe_from_range(mapping_wb, "_NOMENCLATURE_CONTRATS")
    nomenclature_lcr_nsfr = ex.get_dataframe_from_range(mapping_wb, "_NOMENCLATURE_LCR_NSFR_SOURCES")

    nomenclature_stock_ag[["NOM FINAL", "CHEMIN"]] = nomenclature_stock_ag[["NOM FINAL", "CHEMIN"]].apply(
        lambda col: col.str.replace(r"@DATE\(%Y-%m-%d\)", dar.strftime("%Y-%m-%d"), regex=True))

    nomenclature_pn[["NOM FINAL", "CHEMIN"]] = nomenclature_pn[["NOM FINAL", "CHEMIN"]].apply(
        lambda col: col.str.replace(r"@DATE\(%Y-%m-%d\)", dar.strftime("%Y-%m-%d"), regex=True))

    nomenclature_contrats[["NOM FINAL", "CHEMIN"]] = nomenclature_contrats[["NOM FINAL", "CHEMIN"]].apply(
        lambda col: col.str.replace(r"@DATE\(%Y-%m-%d\)", dar.strftime("%Y-%m-%d"), regex=True))

    nomenclature_lcr_nsfr[["NOM FINAL", "CHEMIN"]] = nomenclature_lcr_nsfr[["NOM FINAL", "CHEMIN"]].apply(
        lambda col: col.str.replace(r"@DATE\(%Y-%m-%d\)", dar.strftime("%Y-%m-%d"), regex=True))

    nomenclature_stock_ag["CHEMIN"] = [os.path.join(sources_folder, *x.split('\\')) for x in
                                       nomenclature_stock_ag["CHEMIN"]]
    nomenclature_pn["CHEMIN"] = [os.path.join(sources_folder, *x.split('\\')) for x in nomenclature_pn["CHEMIN"]]
    nomenclature_contrats["CHEMIN"] = [os.path.join(sources_folder, *x.split('\\')) for x in
                                       nomenclature_contrats["CHEMIN"]]
    nomenclature_lcr_nsfr["CHEMIN"] = [os.path.join(sources_folder, *x.split('\\')) for x in
                                       nomenclature_lcr_nsfr["CHEMIN"]]

    return nomenclature_stock_ag, nomenclature_contrats, nomenclature_pn, nomenclature_lcr_nsfr


def fill_exceptions_mappings(map_wb):
    global exceptions_missing_mappings
    exceptions_missing_mappings["PASSALM_NOT_IN_RAY"] = {}
    exceptions_missing_mappings["PASSALM_NOT_IN_RAY"]["key"] = ["M_CONTRAT", "CONTRAT"]
    exceptions_missing_mappings["PASSALM_NOT_IN_RAY"]["list_excep"] \
        = ex.get_dataframe_from_range(map_wb, "CONTRAT_PASSALM_MM_EX", header=True).iloc[:, 0].values.tolist()
    exceptions_missing_mappings["RAY_NOT_IN_PASSALM"] = {}
    exceptions_missing_mappings["RAY_NOT_IN_PASSALM"]["key"] = ["CONTRACT_TYPE"]
    exceptions_missing_mappings["RAY_NOT_IN_PASSALM"]["list_excep"] \
        = ex.get_dataframe_from_range(map_wb, "CONTRAT_RAY_MM_EX", header=True).iloc[:, 0].values.tolist()


def gen_maturity_mapping(mapping_contrats):
    mapping_contrats = mapping_contrats.rename(columns={'(Vide)': '-'}).reset_index()
    cols_unpivot = ['CT', 'MLT', '-']
    cols_keep = [x for x in mapping_contrats.columns if x not in cols_unpivot]
    mapping_mty = pd.melt(mapping_contrats, id_vars=cols_keep, value_vars=cols_unpivot, var_name="MTY",
                          value_name=pa.NC_PA_MATUR)
    return mapping_mty


def get_mapping_from_wb(map_wb, name_range):
    mapping_data = ex.get_dataframe_from_range(map_wb, name_range)
    return mapping_data


def rename_cols_mapping(mapping_data, rename):
    mapping_data = mapping_data.rename(columns=rename)
    return mapping_data


def lecture_mapping_principaux(map_wb):
    global exceptions_missing_mappings
    general_mapping = {}

    mappings_name = ["CONTRATS", "CONTRATS2", "MTY DETAILED", "GESTION", "PALIER", "INDEX_AGREG",
                     "MTY", "BASSINS"]
    name_ranges = ["_MAP_GENERAL", "_MAP_GENERAL", "_MapMaturNTX", "_MapGestion", "_MapPalier", "_MapIndexAgreg", "",
                   "_MP_BASSIN_SOUS_BASSIN"]
    renames = [{"CONTRAT": "CONTRAT_INIT", "CONTRAT PASS": pa.NC_PA_CONTRACT_TYPE, "POSTE AGREG": pa.NC_PA_POSTE},
               {"CONTRAT": "CONTRAT_INIT", "CONTRAT PASS": pa.NC_PA_CONTRACT_TYPE, "POSTE AGREG": pa.NC_PA_POSTE},
               {"MATUR": "CLE_MATUR_NTX", "MAPPING1": pa.NC_PA_MATUR, "MAPPING2": pa.NC_PA_MATURITY_DURATION}, \
               {"Mapping": pa.NC_PA_GESTION}, {"MAPPING": pa.NC_PA_PALIER}, {"INDEX_AGREG": pa.NC_PA_INDEX_AGREG}, {}, {}]

    keys = [["CATEGORY", "CONTRAT_INIT"], ["CONTRAT"], ["CLE_MATUR_NTX"], ["Intention de Gestion"], ["PALIER CONSO"], \
            [pa.NC_PA_RATE_CODE], ["CATEGORY", "CONTRAT_INIT", "MTY"], ["SOUS-BASSIN"]]
    useful_cols = [[pa.NC_PA_DIM2, pa.NC_PA_DIM3, pa.NC_PA_DIM4, pa.NC_PA_DIM5, pa.NC_PA_POSTE,
                    pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_isECH],
                   [pa.NC_PA_DIM2, pa.NC_PA_DIM3, pa.NC_PA_DIM4, pa.NC_PA_DIM5, pa.NC_PA_POSTE],
                   [pa.NC_PA_MATUR, pa.NC_PA_MATURITY_DURATION], \
                   [pa.NC_PA_GESTION], [pa.NC_PA_PALIER], [pa.NC_PA_INDEX_AGREG],
                   [pa.NC_PA_MATUR], []]
    mappings_full_name = ["MAPPING CONTRATS PASSALM", "MAPPING CONTRATS PASSALM2", "MAPPING MATURITES DETAILLE",
                          "MAPPING INTENTIONS DE GESTION", "MAPPING CONTREPARTIES", \
                          "MAPPING DEVISES", "MAPPING INDEX TAUX",
                          "MAPPING INDEX AGREG", "MAPPING MATURITES", "MAPPING BASSINS/SOUS-BASSINS"]
    est_facultatif = [False, False, False, False, False, True, False, True]

    joinkeys = [False] * len(keys)

    force_int_str = [False] * 4 + [True] + [False] * (len(keys) - 5)

    for i in range(0, len(mappings_name)):
        if mappings_name[i] != "MTY":
            mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])
            if len(renames[i]) != 0:
                mapping_data = rename_cols_mapping(mapping_data, renames[i])
        else:
            mapping_data = gen_maturity_mapping(general_mapping["CONTRATS"]["TABLE"].copy())

        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, \
                              est_facultatif[i], joinkeys[i], force_int_str=force_int_str[i])
        general_mapping[mappings_name[i]] = mapping

    return general_mapping


def lecture_mapping_liquidite(map_wb):
    mapping_liq = {}
    global liquidity_cols

    mappings_name = ["LIQ_BC", "LIQ_EM", "LIQ_IG", "LIQ_CT", "LIQ_FI", "LIQ_SC", "NSFR"]
    name_ranges = ["_MAP_CONSO_CPT", "_MAP_EM", "_MAP_LIQ_IG", "_MAP_LIQ_CT", "_MAP_LIQ_FI", "_MAP_LIQ_SOC_AGREG",
                   "_MAP_NSFR"]
    keys = [["CONTRAT CONSO", "BOOK CODE", "LCR TIERS"], ["BILAN", "Regroupement 1", "MATUR"],
            ["BILAN", "Bilan Cash", "BASSIN", "CONTRAT", "PALIER"], \
            ["BILAN", "Regroupement 1", "BASSIN", "PALIER"],
            ["BILAN", "Regroupement 1", "BASSIN", "Bilan Cash", "CONTRAT", "IG/HG Social"], ["Affectation Social"] \
        , ["CONTRAT_NSFR", "LCR_TIERS"]]
    useful_cols = [["Regroupement 1", "Regroupement 2", "Regroupement 3", "Bilan Cash",
                    "Bilan Cash Detail",
                    "Bilan Cash CTA", "Affectation Social"], ["Bilan Cash Detail", "Affectation Social"], \
                   ["Bilan Cash Detail", "Affectation Social"], \
                   ["Affectation Social"], ["Affectation Social"], ["Affectation Social 2"],
                   ["DIM NSFR 1", "DIM NSFR 2"]]
    mappings_full_name = ["MAPPING LIQ BILAN CASH", "MAPPING LIQ EMPREINTE DE MARCHE", "MAPPING LIQ OPERATIONS IG",
                          "MAPPING LIQ COMPTES", \
                          "MAPPING LIQ OPERATIONS FINANCIERES", "MAPPING LIQ SOCIAL AGREGE", "MAPPING NSFR"]

    joinkeys = [True, False, False, False, False, False, True]
    renames = [{}] * len(joinkeys)

    est_facultatif = [True] * len(joinkeys)

    for i in range(0, len(mappings_name)):
        mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])
        mapping_data = rename_cols_mapping(mapping_data, renames[i])
        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, est_facultatif[i],
                              joinkeys[i])
        mapping_liq[mappings_name[i]] = mapping

    liquidity_cols = [mapping_liq[x]["OUT"] for x in list(mapping_liq.keys())]
    liquidity_cols = [item for sublist in liquidity_cols for item in sublist]
    liquidity_cols = list(dict.fromkeys(liquidity_cols))

    return mapping_liq


def lecture_mapping_bpce(map_wb):
    mapping_bpce = {}
    mappings_name = ["PERIMETRE_BPCE"]
    name_ranges = ["_MAP_EVOL_BPCE"]
    keys = [[pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_BOOK, pa.NC_PA_PALIER]]
    useful_cols = [[pa.NC_PA_PERIMETRE]]
    mappings_full_name = ["MAPPING PERIMETRE BPCE"]
    renames = [{"BOOK CODE": pa.NC_PA_BOOK, "CONTRAT BPCE": pa.NC_PA_CONTRACT_TYPE}]
    joinkeys = [False]
    force_int_str = [True]
    est_facultatif = [True]

    for i in range(0, len(mappings_name)):
        mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])

        mapping_data = rename_cols_mapping(mapping_data, renames[i])
        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, est_facultatif[i], \
                              joinkeys[i], force_int_str=force_int_str[i])

        mapping_bpce[mappings_name[i]] = mapping

    return mapping_bpce


def lecture_mapping_bpce_pn(map_wb):
    mapping_bpce_pn = {}
    mappings_name = ["REFI RZO", "REFI BPCE", "PERIMETRE_BPCE", "mapping_profil_BPCE"]
    name_ranges = ["_MapRefiRZO", "_REFI_BPCE", "_MAP_EVOL_BPCE", "_MAP_PROF_BPCE"]
    keys = [["CONTRAT RZO"], [pa.NC_PA_CONTRACT_TYPE], [pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_BOOK, pa.NC_PA_PALIER],
            ["AMORTIZING_TYPE_BPCE_CALL"]]
    useful_cols = [["CONTRAT BPCE"], [], [pa.NC_PA_PERIMETRE], ["PROFIL"]]
    mappings_full_name = ["MAPPING REFI RZO", "MAPPING REFI BPCE", \
                          "MAPPING PERIMETRE BPCE", "MAPPING PROFIL BPCE"]
    renames = [{}, {}, {"BOOK CODE": pa.NC_PA_BOOK, "CONTRAT BPCE": pa.NC_PA_CONTRACT_TYPE}, {}]
    joinkeys = [False] * 4
    force_int_str = [True] * 4
    est_facultatif = [False] * 4

    for i in range(0, len(mappings_name)):
        mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])

        mapping_data = rename_cols_mapping(mapping_data, renames[i])

        if useful_cols[i] == []:
            useful_cols[i] = [x for x in mapping_data.columns.tolist() if x not in keys[i]]

        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, est_facultatif, joinkeys[i],
                              force_int_str=force_int_str[i])

        mapping_bpce_pn[mappings_name[i]] = mapping

    return mapping_bpce_pn


def lecture_mapping_IG(map_wb):
    global map_pass_alm
    mapping_ig = {}
    mappings_name = ["IG", "PALIER"]
    name_ranges = ["", "_MapBassinIG"]
    keys = [[pa.NC_PA_CONTRACT_TYPE], ["BASSIN"]]
    useful_cols = [["isNTXIG", "isRZOIG", "isBPCEIG"], ["BASSIN IG"]]
    mappings_full_name = ["MAPPING CONTRATS IG", "MAPPING BASSINS IG"]
    renames = [{"NTX": "isNTXIG", "BPCE": "isBPCEIG", "RZO": "isRZOIG"}, {}]
    joinkeys = [False, False]
    drop_duplicates = [True, False]
    est_facultatif = [False, False]

    for i in range(0, len(mappings_name)):
        if i == 0:
            mapping_data = map_pass_alm["CONTRATS"]["TABLE"].loc[:, [pa.NC_PA_CONTRACT_TYPE, "NTX", "BPCE", "RZO"]]
        else:
            mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])

        mapping_data = rename_cols_mapping(mapping_data, renames[i])
        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, est_facultatif[i], \
                              joinkeys[i], drop_duplicates=drop_duplicates[i])

        mapping_ig[mappings_name[i]] = mapping

    return mapping_ig


def lecture_mapping_PN(map_wb):
    mapping_PN = {}

    mappings_name = ["mapping_CONSO_ECH", "mapping_ECH_AMOR_PROFIL", "mapping_ECH_PER_INTERETS",
                     "mapping_PEL", "mapping_PER_CAPI", "mapping_PERIODE_AMOR", "mapping_template_nmd"]
    name_ranges = ["_MAP_CONSO_ECH", "_MAP_ECH_PROFIL_ECOULEMENT", "_MAP_ECH_PERIOD_FIXING", "_MAP_PEL",
                   "_MAP_ECH_PER_CAPI",
                   "_MAP_ECH_PER_PROFIL", "_MAP_TEMPLATE_ST_NMD"]
    keys = [["CONTRAT"], ["amortizing_type".upper()], ["periodicity".upper()], ["CONTRAT"], \
            ["compound_periodicity".upper()], ["amortizing_periodicity".upper()],
            ["ETAB", "CONTRACT_TYPE", "RATE_CODE", "CURRENCY", "FAMILY"]]
    mappings_full_name = ["MAPPING CONSO ECH", "MAPPING ECH PROFIL AMORTISSEMENT", "MAPPING ECH PERIODE INTERETS",
                          "MAPPING PEL",
                          "MAPPING ECH PERIODE DE CAPITALISATION", "MAPPING ECH PERIODE AMORTISSEMENT",
                          "MAPPING DES STOCK NMDs AGREGE"]
    joinkeys = [False] * (len(mappings_name) - 1) + [True]
    est_facultatif = [False] * len(joinkeys)
    for i in range(0, len(mappings_name)):
        mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])
        useful_cols = [x for x in mapping_data.columns.tolist() if x not in list(set(keys[i]))]
        if mappings_name[i] == "mapping_CONSO_ECH":
            useful_cols = [x for x in useful_cols if "BASSIN" not in x]
        mapping = gen_mapping(keys[i], useful_cols, mappings_full_name[i], mapping_data, est_facultatif[i], joinkeys[i])
        mapping_PN[mappings_name[i]] = mapping

    return mapping_PN


def lecture_mapping_NTX(map_wb):
    bale3_col = "Traitement réglementaire Bâle III"
    mapping_ntx = {}
    mappings_name = ["MNI", "OTHER", "RATE CODE"]
    name_ranges = ["_MapTopMNI", "_MAP_VAR_NTX", "_MAP_NTX_RATE_CODE"]
    renames = [{}, {"ZONE": pa.NC_PA_ZONE_GEO, "SOUS-ZONE": pa.NC_PA_SOUS_ZONE_GEO}, {}]
    keys = [["TOP MNI"], ["Code de Sous-section"], ["DEVISE_NTX", "RATE_CODE_NTX"]]
    useful_cols = [['1. Périmètre IRRBB', '2. Périmètre STI', '3. Périmètre STEBA',
                    '4. Périmètre STEBA cible (hors internes Natixis pour calage FinRep)'], \
                   [ntx_p.NC_MAP_NTX_TRAITEMENT_REG_ALM, pa.NC_PA_SOUS_METIER, pa.NC_PA_ZONE_GEO, pa.NC_PA_METIER,
                    pa.NC_PA_SOUS_ZONE_GEO],
                   [pa.NC_PA_RATE_CODE, pa.NC_PA_DEVISE]]
    mappings_full_name = ["MAPPING MNI NTX", "MAPPING OTHER NTX", "MAPPING RATE CODE NTX"]
    joinkeys = [False] * len(mappings_name)
    est_facultatif = [False] * len(mappings_name)

    for i in range(0, len(mappings_name)):
        mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])
        if i == 0:
            mapping_data[keys[i]] = mapping_data[keys[i]].apply(lambda x: x.str.replace("_", ""))
        elif i == 1:
            filter_map = (mapping_data[keys[i][0]] != "") & \
                         (mapping_data[keys[i][0]] != "nan")
            mapping_data = mapping_data[filter_map]

            mapping_data = mapping_data.drop_duplicates(subset=keys[i], keep="first")

            mapping_data = gu.force_integer_to_string(mapping_data, "Code de Sous-section")
            check = (mapping_data[bale3_col].isnull()) | (mapping_data[bale3_col] == "NA") | (
                    mapping_data[bale3_col] == "")
            mapping_data.loc[check, bale3_col] = mapping_data.loc[check, ntx_p.NC_MAP_NTX_TRAITEMENT_REG_ALM]

        mapping_data = rename_cols_mapping(mapping_data, renames[i])
        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, est_facultatif[i],
                              joinkeys[i])

        mapping_ntx[mappings_name[i]] = mapping

    return mapping_ntx


def filter_out_errors(data_err, name_mapping):
    global exceptions_missing_mappings
    if name_mapping in exceptions_missing_mappings:
        name_keys = exceptions_missing_mappings[name_mapping]["key"]
        list_exceptions = exceptions_missing_mappings[name_mapping]["list_excep"]
        for key in name_keys:
            if key in data_err.columns.tolist():
                data_err = data_err[~data_err[key].isin(list_exceptions)]
    return data_err
