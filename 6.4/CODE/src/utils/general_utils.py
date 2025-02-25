import gc
import itertools
import numpy as np
import re
import pandas as pd
import logging
from utils import excel_utils as ex

logger = logging.getLogger(__name__)


def chunkized_data(data_ldp_all, chunk):
    data_ldp_all = [data_ldp_all.iloc[i:i + chunk] for i in range(0, len(data_ldp_all), chunk)]
    return data_ldp_all

def remove_BOM_carac(df):
    df.columns = [str(column).replace("ï»¿", "").replace("ï»¿".upper(), "") for column in df.columns.tolist()]
    return df


def flatten(listo):
    return [x for xs in listo for x in xs]


def is_integer(x):
    try:
        x = int(x)
        return True
    except:
        return False


def add_constant_new_col_pd(df, l_name_col, l_value):
    cols_to_add = []
    for name_col, val in zip(l_name_col, l_value):
        col = pd.DataFrame(np.array([val] * len(df)), columns=[name_col], index=df.index)
        cols_to_add.append(col)

    df = pd.concat([df] + cols_to_add, axis=1)

    return df


def to_nan_np(df, cols, filtre=[]):
    cols = [x for x in cols if x in df.columns.tolist()]
    num_cols = np.array(df[cols]).astype(np.float64)
    if len(filtre) > 0:
        num_cols = num_cols[filtre]
    num_cols[num_cols == None] = 0
    if len(filtre) > 0:
        df.loc[filtre, cols] = np.nan_to_num(num_cols)
    else:
        df[cols] = np.nan_to_num(num_cols)
    return df


def unpivot_data(data, new_axis_name):
    nums_cols = [x for x in data.columns if len(x.split("_M")) > 1 and is_integer(x.split("_M")[1])]
    new_num_cols = ["M" + str(x) for x in sorted(list(set([int(x.split("_M")[1]) for x in nums_cols])))]
    qual_cols = [x for x in data.columns if x not in nums_cols]
    data_new = data.set_index(qual_cols)
    data_new.columns = data_new.columns.str.rsplit("_", expand=True)
    data_new = data_new.rename_axis((new_axis_name, None), axis=1).stack(0, future_stack=True).reset_index()
    new_qual_cols = [x for x in qual_cols if x not in new_num_cols] + [new_axis_name]
    return data_new[new_qual_cols + new_num_cols].copy()


def order_by_indic(data, indics_ordered, indic_axis, keys_order):
    filters = []
    for ind in indics_ordered:
        filters = filters + [data[indic_axis] == ind]
    vals = [i for i in range(1, len(indics_ordered) + 1)]
    data = pd.concat([data, pd.DataFrame(np.select(filters, vals), index=data.index, columns=["order_ind"])], axis=1)
    return data.sort_values(keys_order + ["order_ind"]).drop(["order_ind"], axis=1)


def add_empty_indics(data, empty_inds, ind_col, ref_indic, cols_num, order=False, indics_ordered=[], keys_order=[]):
    empty_inds = [x for x in empty_inds if x not in data[ind_col].values.tolist()]
    if len(empty_inds) > 0:
        data_unique = data[data[ind_col] == ref_indic].copy()
        data_empty_ind = pd.concat([data_unique] * len(empty_inds))
        data_empty_ind[ind_col] = np.column_stack([np.array(empty_inds).reshape(len(empty_inds), 1)] \
                                                  * int(len(data_unique))).reshape(len(data_unique) * len(empty_inds))
        data_empty_ind.loc[:, cols_num] = np.nan
        data = pd.concat([data, data_empty_ind])
    if order:
        data = order_by_indic(data, indics_ordered, ind_col, keys_order)
    return data


def check_version_templates(fichier, path="", version="", open=False, wb=[], warning=False):
    if open:
        wb = ex.try_close_open(path, read_only=True)
    err_mapp = "Vous n'avez pas la bonne version du fichier '%s'. Demandez à votre administrateur la version %s" % (
        fichier, version)
    if wb.BuiltinDocumentProperties(5).Value not in str(version).split(","):
        if not warning and open:
            try:
                wb.Close()
            except:
                pass
        if not warning:
            logger.error(err_mapp)
            raise ValueError(err_mapp)
        else:
            logger.warning(err_mapp)
    if open:
        try:
            wb.Close()
        except:
            pass


def strip_and_upper(df, cols):
    df[cols] = df[cols].astype(str)
    df[cols] = df[cols].apply(lambda x: x.str.upper())
    df[cols] = df[cols].apply(lambda x: x.str.strip())
    return df


def read_large_csv_file(file, engine='c', encoding='utf-8', sep=",", decimal="."):
    chunksize = 200000
    if engine == "c":
        low_memory = False
    else:
        low_memory = True
    tfr = pd.read_csv(file, sep=sep, decimal=decimal, engine=engine, encoding=encoding, low_memory=low_memory,
                      chunksize=chunksize, iterator=True)
    gc.collect()
    return pd.concat(tfr, ignore_index=True)


def force_integer_to_string(df, col):
    df[col] = [str(int(float(x))) if not re.match(r'^-?\d+(?:\.\d+)?$', str(x)) is None else str(x) for x
               in df[col]]
    return df


def gen_mapping(keys, useful_cols, mapping_full_name, mapping_data, est_facultatif, joinkey, drop_duplicates=False,
                force_int_str=False, upper_content=False):
    mapping = {}
    if len(keys) + len(useful_cols) > 0:
        mapping_data = mapping_data[keys + useful_cols].copy()

    mapping_data = strip_and_upper(mapping_data, keys)

    if len(keys) > 0:
        mapping_data = mapping_data.drop_duplicates(subset=keys).copy()

    if force_int_str:
        for col in keys + useful_cols:
            force_integer_to_string(mapping_data, col)

    if upper_content:
        mapping_data[keys] = mapping_data[keys].map(lambda s: str(s).upper())

    if drop_duplicates:
        mapping_data = mapping_data.drop_duplicates(keys)

    if joinkey:
        mapping_data["KEY"] = mapping_data[keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        keys = ["KEY"]

    if len(keys) > 0:
        mapping["TABLE"] = mapping_data.set_index(keys)
    else:
        mapping["TABLE"] = mapping_data

    mapping["OUT"] = useful_cols
    mapping["FULL_NAME"] = mapping_full_name

    mapping["est_facultatif"] = est_facultatif

    return mapping


def map_data(data_to_map, mapping, keys_data=[], keys_mapping=[], cols_mapp=[], override=False, name_mapping="", \
             no_map_value="", allow_duplicates=False, join_how="left", select_inner=False, tag_absent_override=False,
             error=False, error_message="", map_null_values=False, strip_upper=True):
    len_data = data_to_map.shape[0]

    mapping_tab = mapping.copy()
    if keys_mapping != []:
        mapping_tab = strip_and_upper(mapping_tab, keys_mapping)
        mapping_tab = mapping_tab.set_index(keys_mapping)
    if cols_mapp == []:
        cols_mapp = mapping.columns.tolist()

    original_keys = data_to_map[keys_data].copy().reset_index(drop=True)
    data_to_map["TMP_INDX"] = np.arange(0, data_to_map.shape[0])
    if strip_upper:
        data_to_map = strip_and_upper(data_to_map, keys_data)

    if override:
        old_index = data_to_map.index.copy()
        data_to_map = data_to_map.reset_index(drop=True)
        new_data = data_to_map.copy()
        new_data = new_data.drop(cols_mapp, axis=1)
        new_data = new_data.join(mapping_tab[cols_mapp], how="left", on=keys_data)
        if new_data.shape[0] != data_to_map.shape[0]:
            raise ValueError("Impossible to override, mapping " + name_mapping + " is not unique")
        if tag_absent_override:
            data_to_map[cols_mapp] = "#MAPPING"
        data_to_map.update(new_data[cols_mapp])
        data_to_map.index = old_index
    else:
        data_to_map = data_to_map.join(mapping_tab[cols_mapp], how=join_how, on=keys_data)

    data_to_map = data_to_map.join(original_keys, on=["TMP_INDX"], rsuffix="_OLD")
    data_to_map[keys_data] = data_to_map[[x + "_OLD" for x in keys_data]]
    data_to_map = data_to_map.drop(columns=[x + "_OLD" for x in keys_data] + ["TMP_INDX"], axis=1)

    if map_null_values:
        filtero_none = (data_to_map[cols_mapp[0]].isnull()) | (data_to_map[cols_mapp[0]] == np.nan)
        if filtero_none.any():
            data_to_map.loc[filtero_none, cols_mapp] = no_map_value

    if error:
        filtero_none = (data_to_map[cols_mapp[0]].isnull()) | (data_to_map[cols_mapp[0]] == np.nan)
        if filtero_none.any():
            data_err = data_to_map.loc[filtero_none, keys_data[0]].drop_duplicates().values.tolist()
            logger.error(error_message)
            logger.error(data_err)
            raise ValueError(error_message + str(data_err))

    if not allow_duplicates and len_data != data_to_map.shape[0] and join_how == "left":
        logger.warning("THERE ARE DUPLICATES WITH MAPPING: " + name_mapping)

    if select_inner:
        data_to_map = data_to_map[~filtero_none]

    return data_to_map


def map_with_combined_key(data, mapping, cols_to_combine, symbol_any="-", \
                          filter_comb=False, necessary_cols=2, sep="_", upper_strip=True,
                          drop_key_col=True, name_key_col="COMBINED_COL", to_map=True,
                          filter_none_comb=False, **kwargs):
    data_unique = data[cols_to_combine].drop_duplicates()

    data_unique = gen_combined_key_col(data_unique, mapping, cols_key=cols_to_combine, symbol_any=symbol_any,
                                       name_col_key=name_key_col, set_index=False, \
                                       filter_comb=filter_comb, necessary_cols=necessary_cols, sep=sep,
                                       upper_strip=upper_strip, filter_none_comb=filter_none_comb)
    if to_map:
        data_unique = map_data(data_unique, mapping, keys_data=[name_key_col], **kwargs)

    if drop_key_col:
        data_unique = data_unique.drop([name_key_col], axis=1)

    try:
        data = data.join(data_unique.set_index(cols_to_combine), on=cols_to_combine)
    except:
        data = data.merge(data_unique, on=cols_to_combine, how="left")

    return data


def map_with_combined_key2(data, mapping, cols_to_combine, symbol_any="-", \
                           filter_comb=False, necessary_cols=2, sep="_", upper_strip=True,
                           drop_key_col=True, name_key_col="COMBINED_COL", to_map=True, **kwargs):
    data = gen_combined_key_col(data, mapping, cols_key=cols_to_combine, symbol_any=symbol_any,
                                name_col_key=name_key_col, set_index=False, \
                                filter_comb=filter_comb, necessary_cols=necessary_cols, sep=sep,
                                upper_strip=upper_strip)
    if to_map:
        data = map_data(data, mapping, keys_data=[name_key_col], **kwargs)

    if drop_key_col:
        data = data.drop([name_key_col], axis=1)

    return data


def gen_combinations_keys(list_keys, symbol_any="*", no_condition_first_element=False, filter_comb=False,
                          necessary_cols=0, filter_none_comb=False):
    # list_keys = list_keys + [symbol_any]
    # lg = len(list_keys)
    final_list = []
    # all_combinations = list(itertools.product(list_keys, repeat=lg - 1))
    list_list_keys = [[x, symbol_any] for x in list_keys]
    all_combinations = list(itertools.product(*list_list_keys))

    for combination in all_combinations:
        if no_condition_first_element:
            condition = (combination[0] == list_keys[0]) or (combination[0][0] == symbol_any)
        else:
            condition = (combination[0] == list_keys[0])
        if condition and not combination in final_list:
            final_list.append(list(combination))

    if filter_comb:
        new_comb = []
        passed = True
        for col in final_list:
            for i in range(0, necessary_cols):
                if not col[i] == list_keys[i]:
                    passed = False
            if passed:
                new_comb.append(col)
            passed = True
        final_list = new_comb

    if filter_none_comb:
        new_comb = []
        for col in final_list:
            if not set(col) == set([symbol_any]):
                new_comb.append(col)
        final_list = new_comb

    return final_list


def gen_list_of_joined_keys2(data, cols_keys_combination, sep="_"):
    CLES = []
    for comb in cols_keys_combination:
        joined_cols = join_dataframe_cols(data, comb, sep=sep)
        CLES.append(joined_cols)
    return CLES


def join_dataframe_cols(data, cols, sep="_"):
    i = 0
    for col in cols:
        if i == 0:
            joined_cols = data[col].copy()
        else:
            joined_cols = joined_cols + sep + data[col].copy()
        i = i + 1
    return joined_cols


def gen_combined_key_col(input_data, mapping, cols_key=[], symbol_any="-", name_col_key="COMBINED_KEY",
                         set_index=False, filter_comb=False, necessary_cols=2, sep="_", upper_strip=True,
                         filter_none_comb=False):
    data = input_data.copy()
    if upper_strip:
        data = strip_and_upper(data, cols_key)
    data[symbol_any] = symbol_any
    cols_keys_combination = gen_combinations_keys(cols_key, symbol_any=symbol_any,
                                                  no_condition_first_element=True,
                                                  filter_comb=filter_comb, necessary_cols=necessary_cols,
                                                  filter_none_comb=filter_none_comb)

    CLES = gen_list_of_joined_keys2(data, cols_keys_combination, sep=sep)

    list_mappings_keys = mapping.index.values.tolist()
    filtres = [KEY.isin(list_mappings_keys) for KEY in CLES]

    default_value = data[cols_key[0]].copy()
    input_data[name_col_key] = np.select(filtres, CLES, default=default_value)

    if set_index:
        input_data.set_index(name_col_key, inplace=True, append=True)

    return input_data


