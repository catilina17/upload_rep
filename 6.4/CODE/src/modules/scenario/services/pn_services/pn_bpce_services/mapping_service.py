# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import utils.excel_utils as ex
import utils.general_utils as ut
import modules.scenario.services.pn_services.pn_bpce_services.referential_bpce as rf_bpce
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
logger = logging.getLogger(__name__)

global COL_CT_MLT_LIQ
global liquidity_cols
global missing_mapping
global mapping
missing_mapping = {}
liquidity_cols = []

MAPPING_EMPTY ="#Mapping"
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
global is_mapped
is_mapped = False

def load_general_mappings(mapping_wb):
    global mapping, is_mapped
    if not is_mapped:
        mapping = {}
        mapping["mapping_global"] = lecture_mapping_principaux(mapping_wb)
        mapping["mapping_liquidite"] = lecture_mapping_liquidite(mapping_wb)
        mapping["mapping_PN"]  = lecture_mapping_PN(mapping_wb)
        mapping["mapping_BPCE"] = lecture_mapping_bpce(mapping_wb)
        is_mapped = True


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


def gen_mapping(keys, useful_cols, mapping_full_name, mapping_data, joinkey, drop_duplicates=False,
                force_int_str=False):
    mapping = {}
    mapping_data = ut.strip_and_upper(mapping_data, keys)

    mapping_data = mapping_data.drop_duplicates(subset=keys).copy()

    if force_int_str:
        for col in keys + useful_cols:
            ut.force_integer_to_string(mapping_data, col)

    if joinkey:
        mapping_data["KEY"] = mapping_data[keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        keys = ["KEY"]

    if len(keys)>0:
        mapping["TABLE"] = mapping_data.set_index(keys)
    else:
        mapping["TABLE"] = mapping_data

    mapping["OUT"] = useful_cols
    mapping["FULL_NAME"] = mapping_full_name

    return mapping


def lecture_mapping_principaux(map_wb):
    global exceptions_missing_mappings
    general_mapping = {}

    mappings_name = ["CONTRATS", "MTY DETAILED", "GESTION", "PALIER", "INDEX_AGREG",
                     "MTY"]
    name_ranges = ["_MAP_GENERAL", "_MapMaturNTX", "_MapGestion", "_MapPalier", "_MapIndexAgreg", ""]
    renames = [{"CONTRAT": "CONTRAT_INIT", "CONTRAT PASS": pa.NC_PA_CONTRACT_TYPE, "POSTE AGREG": pa.NC_PA_POSTE},
               {"MATUR": "CLE_MATUR_NTX", "MAPPING1": pa.NC_PA_MATUR, "MAPPING2": pa.NC_PA_MATURITY_DURATION}, \
               {"Mapping": pa.NC_PA_GESTION}, {"MAPPING": pa.NC_PA_PALIER}, {"INDEX_AGREG": pa.NC_PA_INDEX_AGREG}, {}]

    keys = [["CONTRAT"], ["CLE_MATUR_NTX"], ["Intention de Gestion"], ["PALIER CONSO"], \
            ["RATE CODE"], ["CATEGORY", "CONTRAT_INIT", "MTY"]]
    useful_cols = [[pa.NC_PA_DIM2, pa.NC_PA_DIM3, pa.NC_PA_DIM4, pa.NC_PA_DIM5, pa.NC_PA_POSTE],
                   [pa.NC_PA_MATUR, pa.NC_PA_MATURITY_DURATION], \
                   [pa.NC_PA_GESTION], [pa.NC_PA_PALIER],[pa.NC_PA_INDEX_AGREG],
                   [pa.NC_PA_MATUR]]
    mappings_full_name = ["MAPPING CONTRATS PASSALM", "MAPPING MATURITES DETAILLE",
                          "MAPPING INTENTIONS DE GESTION", "MAPPING CONTREPARTIES", \
                          "MAPPING INDEX AGREG", "MAPPING MATURITES"]

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
                              joinkeys[i], force_int_str=force_int_str[i])
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

    for i in range(0, len(mappings_name)):
        mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])
        mapping_data = rename_cols_mapping(mapping_data, renames[i])
        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, joinkeys[i])
        mapping_liq[mappings_name[i]] = mapping

    liquidity_cols = [mapping_liq[x]["OUT"] for x in list(mapping_liq.keys())]
    liquidity_cols = [item for sublist in liquidity_cols for item in sublist]
    liquidity_cols = list(dict.fromkeys(liquidity_cols))

    return mapping_liq

def lecture_mapping_bpce(map_wb):
    mapping_bpce = {}
    mappings_name = ["REFI RZO", "REFI BPCE", "PERIMETRE_BPCE", "mapping_profil_BPCE"]
    name_ranges = ["_MapRefiRZO", "_REFI_BPCE", "_MAP_EVOL_BPCE", "_MAP_PROF_BPCE"]
    keys = [["CONTRAT RZO"], [pa.NC_PA_CONTRACT_TYPE], [pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_BOOK, pa.NC_PA_PALIER], ["AMORTIZING_TYPE_BPCE_CALL"]]
    useful_cols = [[rf_bpce.NC_CONTRAT_BPCE], [], [pa.NC_PA_PERIMETRE],["PROFIL"]]
    mappings_full_name = ["MAPPING REFI RZO",  "MAPPING REFI BPCE",\
                          "MAPPING PERIMETRE BPCE", "MAPPING PROFIL BPCE"]
    renames = [{},{},{"BOOK CODE": pa.NC_PA_BOOK, "CONTRAT BPCE": pa.NC_PA_CONTRACT_TYPE},{}]
    joinkeys = [False]*4
    force_int_str = [True]*4

    for i in range(0, len(mappings_name)):
        mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])

        mapping_data = rename_cols_mapping(mapping_data, renames[i])

        if useful_cols[i]==[]:
            useful_cols[i] = [x for x in mapping_data.columns.tolist() if x not in keys[i]]

        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, joinkeys[i],
                              force_int_str=force_int_str[i])

        mapping_bpce[mappings_name[i]] = mapping

    return mapping_bpce

def lecture_mapping_PN(map_wb):
    mapping_PN = {}

    mappings_name = ["mapping_CONSO_ECH"]
    name_ranges = ["_MAP_CONSO_ECH"]
    keys = [["CONTRAT"]]
    mappings_full_name = ["MAPPING CONSO ECH"]
    joinkeys = [False]
    for i in range(0, len(mappings_name)):
        mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])
        useful_cols = list(set(mapping_data.columns.tolist()) - set(keys[i]))
        mapping = gen_mapping(keys[i], useful_cols, mappings_full_name[i], mapping_data, joinkeys[i])
        mapping_PN[mappings_name[i]] = mapping

    return mapping_PN


def filter_out_errors(data_err, name_mapping):
    global exceptions_missing_mappings
    if name_mapping in exceptions_missing_mappings:
        name_keys = exceptions_missing_mappings[name_mapping]["key"]
        list_exceptions = exceptions_missing_mappings[name_mapping]["list_excep"]
        for key in name_keys:
            if key in data_err.columns.tolist():
                data_err = data_err[~data_err[key].isin(list_exceptions)]
    return data_err

def map_data(data_to_map, mapping, keys_data=[], keys_mapping=[], cols_mapp=[], override=False, name_mapping="", \
             error_mapping=True, no_map_value="", allow_duplicates=False, option="PASS_ALM", join_how="left",
             col_err=[], select_inner=False, except_name_mapping=""):
    global missing_mapping

    len_data = data_to_map.shape[0]

    if no_map_value == "":
        no_map_value = MAPPING_EMPTY

    if option == "PASS_ALM":
        if cols_mapp == []:
            cols_mapp = mapping["OUT"]
        mapping_tab = mapping["TABLE"]
        name_mapping = name_mapping + " " + mapping["FULL_NAME"]
    else:
        mapping_tab = mapping.copy()
        if keys_mapping != []:
            mapping_tab = ut.strip_and_upper(mapping_tab, keys_mapping)
            mapping_tab = mapping_tab.set_index(keys_mapping)
        if cols_mapp == []:
            cols_mapp = mapping.columns.tolist()

    original_keys = data_to_map[keys_data].copy().reset_index(drop=True)
    data_to_map["TMP_INDX"] = np.arange(0, data_to_map.shape[0])
    data_to_map = ut.strip_and_upper(data_to_map, keys_data)

    if override:
        old_index = data_to_map.index.copy()
        data_to_map = data_to_map.reset_index(drop=True)
        new_data = data_to_map[keys_data].copy()
        new_data = new_data.join(mapping_tab[cols_mapp], how="left", on=keys_data)
        if new_data.shape[0] != data_to_map.shape[0]:
            raise ValueError("Impossible to override, mapping " + name_mapping + " is not unique")
        data_to_map.update(new_data[cols_mapp].copy())
        data_to_map.index = old_index
    else:
        data_to_map = data_to_map.join(mapping_tab[cols_mapp], how=join_how, on=keys_data)

    data_to_map = data_to_map.join(original_keys, on=["TMP_INDX"], rsuffix="_OLD")
    data_to_map[keys_data] = data_to_map[[x + "_OLD" for x in keys_data]]
    data_to_map = data_to_map.drop(columns=[x + "_OLD" for x in keys_data] + ["TMP_INDX"], axis=1)

    filtero_none = (data_to_map[cols_mapp[0]].isnull()) | (data_to_map[cols_mapp[0]] == np.nan)
    if filtero_none.any():
        data_to_map.loc[filtero_none, cols_mapp] = no_map_value

    if error_mapping:
        if filtero_none.any():
            if col_err == []:
                data_err = data_to_map.loc[filtero_none, keys_data + cols_mapp].drop_duplicates()
            else:
                data_err = data_to_map.loc[filtero_none, keys_data + col_err + cols_mapp] \
                    .groupby(by=keys_data + cols_mapp, as_index=False).sum()
            if except_name_mapping != "":
                data_err = filter_out_errors(data_err, except_name_mapping)

            if data_err.shape[0] > 0:
                if not name_mapping in missing_mapping:
                    missing_mapping[name_mapping] = data_err
                else:
                    missing_mapping[name_mapping] = pd.concat([missing_mapping[name_mapping], data_err])

    if not allow_duplicates and len_data != data_to_map.shape[0] and join_how == "left":
        logger.warning("   THERE ARE DUPLICATES WITH MAPPING: " + name_mapping)

    if select_inner:
        data_to_map = data_to_map[~filtero_none]

    return data_to_map


def mapping_consolidation_liquidite(data):
    global mapping
    keys_liq_BC = [pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_BOOK, pa.NC_PA_LCR_TIERS]
    keys_EM = [pa.NC_PA_BILAN, pa.NC_PA_Regroupement_1, pa.NC_PA_MATUR]

    """ MAPPING LIQUIDITE BILAN CASH """
    cles_a_combiner = keys_liq_BC
    mapping_liq = mapping["mapping_liquidite"]["LIQ_BC"]
    data = ut.gen_combined_key_col(data, mapping_liq["TABLE"], cols_key=cles_a_combiner, symbol_any="-",
                                   name_col_key="CONTRAT_", set_index=False)
    data = map_data(data, mapping_liq, keys_data=["CONTRAT_"], name_mapping="STOCK/PN DATA vs.")
    data = data.drop(["CONTRAT_"], axis=1)

    """ AUTRES mappings LIQUIDITE """
    keys_liq_IG = [pa.NC_PA_BILAN, pa.NC_PA_Bilan_Cash, pa.NC_PA_BASSIN, pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_PALIER]
    keys_liq_CT = [pa.NC_PA_BILAN, pa.NC_PA_Regroupement_1, pa.NC_PA_BASSIN, pa.NC_PA_PALIER]
    keys_liq_FI = [pa.NC_PA_BILAN, pa.NC_PA_Regroupement_1, pa.NC_PA_BASSIN, pa.NC_PA_Bilan_Cash, pa.NC_PA_CONTRACT_TYPE, "IG/HG Social"]
    keys_liq_SC = [pa.NC_PA_Affectation_Social]
    data["IG/HG Social"] = np.where(data[pa.NC_PA_PALIER] == "-", "HG", "IG")

    keys_data = [keys_EM, keys_liq_IG, keys_liq_CT, keys_liq_FI, keys_liq_SC]
    mappings = ["LIQ_EM", "LIQ_IG", "LIQ_CT", "LIQ_FI", "LIQ_SC"]
    for i in range(0, len(mappings)):
        mapping_liq = mapping["mapping_liquidite"][mappings[i]]
        key_data = keys_data[i]
        override = False if mappings[i] == "LIQ_SC" else True
        error_mapping = True if mappings[i] == "LIQ_SC" else False
        data = map_data(data, mapping_liq, keys_data=key_data, override=override, error_mapping=error_mapping,
                        name_mapping="STOCK/PN DATA vs.")

    """ MAPPING NSFR """
    cles_a_combiner = [pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_LCR_TIERS]
    mapping_liq = mapping["mapping_liquidite"]["NSFR"]
    data = ut.gen_combined_key_col(data, mapping_liq["TABLE"], cols_key=cles_a_combiner, symbol_any="*",
                                   name_col_key="CONTRAT_", set_index=False)
    data = map_data(data, mapping_liq, keys_data=["CONTRAT_"], name_mapping="STOCK/PN DATA vs.")
    data = data.drop(["CONTRAT_", "IG/HG Social"], axis=1)

    return data