import pandas as pd
import utils.excel_utils as ex
import utils.general_utils as gu
import numpy as np
import logging
from mappings.pass_alm_fields import PASS_ALM_Fields as pa

logger = logging.getLogger(__name__)

global map_pass_alm


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


def gen_mapping(keys, useful_cols, mapping_full_name, mapping_data, est_facultatif, joinkey, drop_duplicates=False,
                force_int_str=False, upper_content=False):
    mapping = {}
    mapping_data = mapping_data[keys + useful_cols].copy()
    mapping_data = gu.strip_and_upper(mapping_data, keys)

    if len(keys)>0:
        mapping_data = mapping_data.drop_duplicates(subset=keys).copy()

    if force_int_str:
        for col in keys + useful_cols:
            mapping_data = gu.force_integer_to_string(mapping_data, col)

    if upper_content:
        mapping_data[keys] = mapping_data[keys].map(lambda s: str(s).upper())

    if drop_duplicates:
        mapping_data = mapping_data.drop_duplicates(keys)

    if joinkey:
        mapping_data["KEY"] = mapping_data[keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        keys = ["KEY"]

    if len(keys)>0:
        mapping["TABLE"] = mapping_data.set_index(keys)
    else:
        mapping["TABLE"] = mapping_data

    mapping["OUT"] = useful_cols
    mapping["FULL_NAME"] = mapping_full_name

    mapping["est_facultatif"] = est_facultatif

    return mapping


def get_main_mappings(map_wb):
    global map_pass_alm
    map_pass_alm = {}

    mappings_name = ["CONTRATS", "GESTION", "PALIER", "MTY"]
    name_ranges = ["_MAP_GENERAL", "_MapGestion", "_MapPalier", "_MAP_GENERAL"]
    renames = [{"CONTRAT": "CONTRAT_INIT", "CONTRAT PASS": pa.NC_PA_CONTRACT_TYPE, "POSTE AGREG": pa.NC_PA_POSTE}, \
               {"Mapping": pa.NC_PA_GESTION}, {"MAPPING": pa.NC_PA_PALIER},
               {"CONTRAT": "CONTRAT_INIT", "CONTRAT PASS": pa.NC_PA_CONTRACT_TYPE, "POSTE AGREG": pa.NC_PA_POSTE}]
    keys = [["CATEGORY", "CONTRAT_INIT"], ["Intention de Gestion"], ["PALIER CONSO"], \
            ["CATEGORY", "CONTRAT_INIT", "MTY"]]
    useful_cols = [[pa.NC_PA_CONTRACT_TYPE], \
                   [pa.NC_PA_GESTION], [pa.NC_PA_PALIER],
                   [pa.NC_PA_MATUR]]
    mappings_full_name = ["MAPPING CONTRATS PASSALM",
                          "MAPPING INTENTIONS DE GESTION", "MAPPING CONTREPARTIES", \
                          "MAPPING MATURITES"]

    est_facultatif = [False, False, False, False, False, False]

    joinkeys = [False] * len(keys)

    force_int_str = [False] * 2 + [True] + [False] * (len(keys) - 3)

    for i in range(0, len(mappings_name)):
        if mappings_name[i] != "MTY":
            mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])
            if len(renames[i]) != 0:
                mapping_data = rename_cols_mapping(mapping_data, renames[i])
        else:
            mapping_data = get_mapping_from_wb(map_wb, name_ranges[i])
            mapping_data = rename_cols_mapping(mapping_data, renames[i])
            mapping_data = gen_maturity_mapping(mapping_data)

        mapping = gen_mapping(keys[i], useful_cols[i], mappings_full_name[i], mapping_data, \
                              est_facultatif[i], joinkeys[i], force_int_str=force_int_str[i])
        map_pass_alm[mappings_name[i]] = mapping


def map_data_with_general_mappings(data, mapping_principaux, key_vars_dic):
    cles_data_ntx = {1: [pa.NC_PA_BILAN, key_vars_dic["CONTRACT_TYPE"]],
                     2: [pa.NC_PA_BILAN, key_vars_dic["CONTRACT_TYPE"], key_vars_dic["MATUR"] + "_TEMP"],
                     3: [key_vars_dic["GESTION"] + "_TEMP"],
                     4: [key_vars_dic["PALIER"] + "_TEMP"]}

    mappings = {1: "CONTRATS", 2: "MTY", 3: "GESTION", 4: "PALIER"}

    for i in range(1, len(mappings) + 1):
        data = map_data(data, mapping_principaux[mappings[i]], keys_data=cles_data_ntx[i],
                        name_mapping="STOCK DATA vs.")

    data.drop([key_vars_dic["CONTRACT_TYPE"], key_vars_dic["MATUR"] + "_TEMP",
                key_vars_dic["GESTION"] + "_TEMP", key_vars_dic["PALIER"] + "_TEMP"],
              axis=1, inplace=True)

    return data


def format_paliers(data, palier_col):
    invalid_palier = (data[palier_col].isnull()) | (data[palier_col].astype(str) == "nan") \
                     | (data[palier_col] == " ") | (data[palier_col] == "(vide)")

    data[palier_col] = data[palier_col].mask(invalid_palier, "-")

    data = gu.force_integer_to_string(data, palier_col)

    return data


def map_data(data_to_map, mapping, keys_data=[], keys_mapping=[], cols_mapp=[], override=False, name_mapping="", \
             no_map_value="", allow_duplicates=False, option="PASS_ALM", join_how="left", select_inner=False):
    global missing_mapping

    len_data = data_to_map.shape[0]

    if no_map_value == "":
        no_map_value = "#Mapping"

    if option == "PASS_ALM":
        if cols_mapp == []:
            cols_mapp = mapping["OUT"]
        mapping_tab = mapping["TABLE"]
        name_mapping = name_mapping + " " + mapping["FULL_NAME"]
    else:
        mapping_tab = mapping.copy()
        if keys_mapping != []:
            mapping_tab = gu.strip_and_upper(mapping_tab, keys_mapping)
            mapping_tab = mapping_tab.set_index(keys_mapping)
        if cols_mapp == []:
            cols_mapp = mapping.columns.tolist()

    original_keys = data_to_map[keys_data].copy().reset_index(drop=True)
    data_to_map["TMP_INDX"] = np.arange(0, data_to_map.shape[0])
    data_to_map = gu.strip_and_upper(data_to_map, keys_data)

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

    if not allow_duplicates and len_data != data_to_map.shape[0] and join_how == "left":
        logger.warning("   THERE ARE DUPLICATES WITH MAPPING: " + name_mapping)

    if select_inner:
        data_to_map = data_to_map[~filtero_none]

    return data_to_map
