import utils.general_utils as gu
from params import version_params as vp
import modules.alim.parameters.NTX_SEF_params as ntx_p
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import utils.excel_utils as ex
import ntpath
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

global liquidity_cols, base_calc_ech
global exceptions_missing_mappings
global nomenclature_stock_ag, nomenclature_stock_nmd, nomenclature_pn, map_pass_alm, mapping_IG, mapping_NTX, mapping_liquidite,  mapping_PN
global mapping_bpce, nomenclature_lcr_nsfr, mapping_taux
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def load_general_mappings(mapp_file_path, dar):
    global missing_mapping, liquidity_cols, exceptions_missing_mappings
    global nomenclature_stock_ag, nomenclature_stock_nmd, nomenclature_pn, map_pass_alm, mapping_IG, mapping_NTX, mapping_liquidite, mapping_PN
    global nomenclature_stock_nmd
    global mapping_bpce, nomenclature_lcr_nsfr, mapping_taux
    liquidity_cols = []
    exceptions_missing_mappings = {}

    mapping_wb = ex.try_close_open(mapp_file_path, read_only=True)
    gu.check_version_templates(ntpath.basename(mapp_file_path), open=False, wb=mapping_wb, version=vp.version_map)

    map_pass_alm = lecture_mapping_principaux(mapping_wb)

    mapping_IG = lecture_mapping_IG(mapping_wb)
    mapping_NTX = lecture_mapping_NTX(mapping_wb)
    mapping_liquidite = lecture_mapping_liquidite(mapping_wb)
    mapping_PN = lecture_mapping_PN(mapping_wb)
    nomenclature_stock_ag, nomenclature_stock_nmd, nomenclature_pn, nomenclature_lcr_nsfr\
        = get_mapping_input_files(mapping_wb, dar)
    mapping_bpce = lecture_mapping_bpce(mapping_wb)

    mapping_taux = get_mapping_taux(mapping_wb)

    fill_exceptions_mappings(mapping_wb)

    mapping_wb.Close(False)


def get_mapping_taux(map_wb):
    mapping_taux = {}

    name_ranges = ['ref_pass_alm_fst_cell', 'COURBES_A_INTERPOLER',
                   '_COURBES_PRICING_PN', '_FILTRE_TENOR', "_MP_CURVE_ACCRUALS2", "_MAP_RATE_CODE_CURVE"]
    renames = [{}] * len(name_ranges)
    keys = [[]]* (len(name_ranges) -2)   + [["CURVE_NAME", "TENOR_BEGIN"]]  +[[]]
    useful_cols = [[]]* (len(name_ranges) -2)   + [["ACCRUAL_METHOD", "ACCRUAL_CONVERSION", "TYPE DE COURBE"]] + [[]]
    joinkeys = [False] * len(name_ranges)
    force_int_str = [False] * len(name_ranges)
    upper_content = [True] * len(name_ranges)
    drop_duplicates = [True] * len(name_ranges)
    mappings_full_name = ["REFERENTIEL DES COURBES DE TAUX",
                          "COURBES A INTERPOLER",
                          "COURBES A CACLCULER",
                          "COURBES AUXILIAIRES",
                          "CONVENTIONS DE BASE DES COURBES", "MAPPING ENTRE CURVE-TENOR et RATE_CODE-DEVISE"]
    mappings_name = ["REF_TX", "CURVES_TO_INTERPOLATE", "CALCULATED_CURVES", "AUXILIARY_CALCULATED_CURVES",
                     'CURVES_BASIS_CONV', "RATE_CODE-CURVE"]

    est_facultatif = [False] * len(name_ranges)
    mode_pass_alm = [False] * len(name_ranges)
    filter_cols = [False] * (len(name_ranges) -1) + [True]

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
                                                               mode_pass_alm = mode_pass_alm[i],
                                                               filter_cols=filter_cols[i])
    return mapping_taux

def gen_mapping(keys, useful_cols, mapping_full_name, mapping_data, est_facultatif, joinkey,
                force_int_str=False, upper_content=True, drop_duplicates=True, mode_pass_alm=True,
                filter_cols=False):
    mapping = {}

    if filter_cols:
        if len(keys) + len(useful_cols) > 0:
            mapping_data = mapping_data[keys + useful_cols].copy()

    if len(keys)>0:
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

def get_mapping_input_files(mapping_wb, dar):
    mapping_wb.Sheets("NOMENCLATURE_ALIM_STOCK").Unprotect(ex.EXCEL_PASS_WORD)
    ex.make_excel_value(mapping_wb, "_DAR_ALIM_STOCK_", str(dar.date()))
    mapping_wb.Sheets("NOMENCLATURE_ALIM_STOCK").Calculate()
    mapping_wb.Sheets("NOMENCLATURE_ALIM_STOCK").Protect(ex.EXCEL_PASS_WORD)
    nomenclature_stock_ag = ex.get_dataframe_from_range(mapping_wb, "_NOMENCLATURE_SOURCES")

    mapping_wb.Sheets("NOMENCLATURE_CONTRATS").Unprotect(ex.EXCEL_PASS_WORD)
    ex.make_excel_value(mapping_wb, "_DAR_CONTRATS_", str(dar.date()))
    mapping_wb.Sheets("NOMENCLATURE_CONTRATS").Calculate()
    mapping_wb.Sheets("NOMENCLATURE_CONTRATS").Protect(ex.EXCEL_PASS_WORD)
    nomenclature_pn = ex.get_dataframe_from_range(mapping_wb, "_NOMENCLATURE_CONTRATS_PN")
    nomenclature_stock_nmd = ex.get_dataframe_from_range(mapping_wb, "_NOMENCLATURE_CONTRATS")

    mapping_wb.Sheets("NOMENCLATURE_AUTRES").Unprotect(ex.EXCEL_PASS_WORD)
    ex.make_excel_value(mapping_wb, "_DAR_AUTRES_", str(dar.date()))
    mapping_wb.Sheets("NOMENCLATURE_AUTRES").Calculate()
    mapping_wb.Sheets("NOMENCLATURE_AUTRES").Protect(ex.EXCEL_PASS_WORD)
    mapping_lcr_nsfr = ex.get_dataframe_from_range(mapping_wb, "_NOMENCLATURE_LCR_NSFR_SOURCES")

    return nomenclature_stock_ag, nomenclature_stock_nmd, nomenclature_pn, mapping_lcr_nsfr


def fill_exceptions_mappings(map_wb):
    global exceptions_missing_mappings
    exceptions_missing_mappings["PASSALM_NOT_IN_RAY"] = {}
    exceptions_missing_mappings["PASSALM_NOT_IN_RAY"]["key"] = ["M_CONTRAT", "CONTRAT"]
    exceptions_missing_mappings["PASSALM_NOT_IN_RAY"]["list_excep"] = ex.get_dataframe_from_range(map_wb,
                                                                                                  "CONTRAT_PASSALM_MM_EX",
                                                                                                  header=True).iloc[:,
                                                                      0].values.tolist()
    exceptions_missing_mappings["RAY_NOT_IN_PASSALM"] = {}
    exceptions_missing_mappings["RAY_NOT_IN_PASSALM"]["key"] = ["CONTRACT_TYPE"]
    exceptions_missing_mappings["RAY_NOT_IN_PASSALM"]["list_excep"] = ex.get_dataframe_from_range(map_wb,
                                                                                                  "CONTRAT_RAY_MM_EX",
                                                                                                  header=True).iloc[:,
                                                                      0].values.tolist()

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

    mappings_name = ["CONTRATS", "MTY DETAILED", "GESTION", "PALIER", "INDEX_AGREG",
                     "MTY"]
    name_ranges = ["_MAP_GENERAL", "_MapMaturNTX", "_MapGestion", "_MapPalier", "_MapIndexAgreg", ""]
    renames = [{"CONTRAT": "CONTRAT_INIT", "CONTRAT PASS": pa.NC_PA_CONTRACT_TYPE, "POSTE AGREG": pa.NC_PA_POSTE},
               {"MATUR": "CLE_MATUR_NTX", "MAPPING1": pa.NC_PA_MATUR, "MAPPING2": pa.NC_PA_MATURITY_DURATION}, \
               {"Mapping": pa.NC_PA_GESTION}, {"MAPPING": pa.NC_PA_PALIER}, {"INDEX_AGREG": pa.NC_PA_INDEX_AGREG}, {}]

    keys = [["CATEGORY", "CONTRAT_INIT"], ["CLE_MATUR_NTX"], ["Intention de Gestion"], ["PALIER CONSO"], \
            [pa.NC_PA_RATE_CODE], ["CATEGORY", "CONTRAT_INIT", "MTY"]]
    useful_cols = [[pa.NC_PA_DIM2, pa.NC_PA_DIM3, pa.NC_PA_DIM4, pa.NC_PA_DIM5, pa.NC_PA_POSTE,
                    pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_isECH],
                   [pa.NC_PA_MATUR, pa.NC_PA_MATURITY_DURATION], \
                   [pa.NC_PA_GESTION], [pa.NC_PA_PALIER], [pa.NC_PA_INDEX_AGREG],
                   [pa.NC_PA_MATUR]]
    mappings_full_name = ["MAPPING CONTRATS PASSALM", "MAPPING MATURITES DETAILLE",
                          "MAPPING INTENTIONS DE GESTION", "MAPPING CONTREPARTIES", \
                          "MAPPING DEVISES", "MAPPING INDEX TAUX",
                          "MAPPING INDEX AGREG", "MAPPING MATURITES"]
    est_facultatif = [False, False, False, False, True, False]

    joinkeys = [False] * len(keys)

    force_int_str = [False] * 3 + [True] + [False] * (len(keys) - 4)

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
        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, est_facultatif[i], joinkeys[i])
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
        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, est_facultatif[i],\
                              joinkeys[i], force_int_str=force_int_str[i])

        mapping_bpce[mappings_name[i]] = mapping

    return mapping_bpce

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
        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, est_facultatif[i],\
                              joinkeys[i], drop_duplicates=drop_duplicates[i])

        mapping_ig[mappings_name[i]] = mapping

    return mapping_ig

def lecture_mapping_PN(map_wb):
    mapping_PN = {}

    mappings_name = ["mapping_CONSO_ECH", "mapping_ECH_AMOR_PROFIL", "mapping_ECH_PER_INTERETS",
                           "mapping_PEL", "mapping_PER_CAPI", "mapping_PERIODE_AMOR", "mapping_template_nmd"]
    name_ranges = ["_MAP_CONSO_ECH", "_MAP_ECH_PROFIL_ECOULEMENT", "_MAP_ECH_PERIOD_FIXING", "_MAP_PEL","_MAP_ECH_PER_CAPI",
                   "_MAP_ECH_PER_PROFIL", "_MAP_TEMPLATE_ST_NMD"]
    keys = [["CONTRAT"], ["amortizing_type".upper()],["periodicity".upper()], ["CONTRAT"],\
            ["compound_periodicity".upper()], ["amortizing_periodicity".upper()], ["ETAB", "CONTRACT_TYPE", "RATE_CODE", "CURRENCY", "FAMILY"]]
    mappings_full_name = ["MAPPING CONSO ECH", "MAPPING ECH PROFIL AMORTISSEMENT", "MAPPING ECH PERIODE INTERETS","MAPPING PEL",
                          "MAPPING ECH PERIODE DE CAPITALISATION", "MAPPING ECH PERIODE AMORTISSEMENT",
                          "MAPPING DES STOCK NMDs AGREGE"]
    joinkeys = [False] * (len(mappings_name) -1) + [True]
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
    renames = [{},{"ZONE": pa.NC_PA_ZONE_GEO,"SOUS-ZONE": pa.NC_PA_SOUS_ZONE_GEO}, {}]
    keys = [["TOP MNI"], ["Code de Sous-section"], ["DEVISE_NTX", "RATE_CODE_NTX"]]
    useful_cols = [['1. Périmètre IRRBB', '2. Périmètre STI', '3. Périmètre STEBA',
                    '4. Périmètre STEBA cible (hors internes Natixis pour calage FinRep)'], \
                   [ntx_p.NC_MAP_NTX_TRAITEMENT_REG_ALM, pa.NC_PA_SOUS_METIER, pa.NC_PA_ZONE_GEO, pa.NC_PA_METIER, pa.NC_PA_SOUS_ZONE_GEO],
                   [pa.NC_PA_RATE_CODE, pa.NC_PA_DEVISE]]
    mappings_full_name = ["MAPPING MNI NTX", "MAPPING OTHER NTX", "MAPPING RATE CODE NTX"]
    joinkeys = [False] * len(mappings_name)
    est_facultatif = [False] * len(mappings_name)

    for i in range(0, len(mappings_name)):
        mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])
        if i == 0:
            mapping_data[keys[i]] = mapping_data[keys[i]].apply(lambda x: x.str.replace("_", ""))
        elif i==1:
            filter_map = (mapping_data[keys[i][0]] != "") & \
                         (mapping_data[keys[i][0]] != "nan")
            mapping_data = mapping_data[filter_map]

            mapping_data = mapping_data.drop_duplicates(subset=keys[i], keep="first")

            mapping_data = gu.force_integer_to_string(mapping_data, "Code de Sous-section")
            check = (mapping_data[bale3_col].isnull()) | (mapping_data[bale3_col] == "NA") | (mapping_data[bale3_col] == "")
            mapping_data.loc[check, bale3_col] = mapping_data.loc[check, ntx_p.NC_MAP_NTX_TRAITEMENT_REG_ALM]

        mapping_data = rename_cols_mapping(mapping_data, renames[i])
        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, est_facultatif[i], joinkeys[i])

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
