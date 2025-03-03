# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:23:26 2020

@author: TAHIRIH
"""
import pandas as pd
from scipy.sparse import csr_matrix
from skimage.util.shape import view_as_windows as viewW
import re
import modules.moteur.parameters.general_parameters as gp
import modules.moteur.parameters.user_parameters as up
import logging
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
logger = logging.getLogger(__name__)


def prolong_last_col_value(data, cur_l, wanted_l, month_gr=False, year_gr=False, month_diff=False, year_diff=False, \
                           null_neg=True, null_pos_gr=False, def_val=0):
    cols_sup = ["M" + str(i) for i in range(cur_l + 1, wanted_l + 1)]
    if len(cols_sup) > 0:
        data = pd.concat(
            [data, pd.DataFrame(np.full((data.shape[0], len(cols_sup)), np.nan), index=data.index, columns=cols_sup)],
            axis=1).ffill(axis=1)

    if month_gr:
        gr_coeff = np.nan_to_num(np.divide(data.loc[:, :cols_sup[0]].iloc[:, -2].fillna(0).values, \
                                           data.loc[:, :cols_sup[0]].iloc[:, -3].fillna(0).values), posinf=def_val,
                                 neginf=def_val)
        if null_pos_gr:
            gr_coeff = np.minimum(1, gr_coeff)
        data[cols_sup] = data[cols_sup].values * np.column_stack([gr_coeff] * len(cols_sup)).cumprod(axis=1)
    elif year_gr:
        gr_coeff = np.nan_to_num(np.divide(data.loc[:, :cols_sup[0]].iloc[:, -2].fillna(0).values, \
                                           data.loc[:, :cols_sup[0]].iloc[:, -14].fillna(0).values), posinf=def_val,
                                 neginf=def_val)
        gr_coeff = gr_coeff ** (1 / 12)
        if null_pos_gr:
            gr_coeff = np.minimum(1, gr_coeff)
        data[cols_sup] = data[cols_sup].values * np.column_stack([gr_coeff] * len(cols_sup)).cumprod(axis=1)

    elif month_diff:
        diff = np.nan_to_num(data.loc[:, :cols_sup[0]].iloc[:, -2].fillna(0).values -
                             data.loc[:, :cols_sup[0]].iloc[:, -3].fillna(0).values, posinf=def_val, neginf=def_val)
        data[cols_sup] = data[cols_sup].values + np.column_stack([diff] * len(cols_sup)).cumsum(axis=1)
        if null_neg:
            data[cols_sup] = np.maximum(data[cols_sup].values, 0)

    elif year_diff:
        diff = np.nan_to_num(data.loc[:, :cols_sup[0]].iloc[:, -2].fillna(0).values -
                             data.loc[:, :cols_sup[0]].iloc[:, -14].fillna(0).values, posinf=def_val, neginf=def_val)
        data[cols_sup] = data[cols_sup].values + np.column_stack([diff / 12] * len(cols_sup)).cumsum(axis=1)
        if null_neg:
            data[cols_sup] = np.maximum(data[cols_sup].values, 0)

    return data


def prolong_last_col_value(data_all, nb_col_sup, s=0, is_df=False, month_gr=False, year_gr=False, month_diff=False,
                           year_diff=False, \
                           null_neg=True, null_pos_gr=False, def_val=0, suf=""):
    if s != 0 or is_df:
        data = data_all.iloc[:, s:].values
    else:
        data = data_all.copy()

    a = data.shape[0]
    n = data.shape[1]

    if nb_col_sup > 0:
        data = np.column_stack([data] + [data[:, -1].reshape(a, 1)] * nb_col_sup)

    if month_gr:
        gr_coeff = np.nan_to_num(np.divide(data[:, n - 1], data[:, n - 2]), posinf=def_val, neginf=def_val)
        if null_pos_gr:
            gr_coeff = np.minimum(1, gr_coeff)
        data[:, n:] = data[:, n:] * np.column_stack([gr_coeff.reshape(a, 1)] * nb_col_sup).cumprod(axis=1)

    elif year_gr:
        gr_coeff = np.nan_to_num(np.divide(data[:, n - 1], data[:, n - 13]), posinf=def_val, neginf=def_val)
        gr_coeff = gr_coeff ** (1 / 12)
        if null_pos_gr:
            gr_coeff = np.minimum(1, gr_coeff)
        data[:, n:] = data[:, n:] * np.column_stack([gr_coeff.reshape(a, 1)] * nb_col_sup).cumprod(axis=1)

    elif month_diff:
        diff = np.nan_to_num(data[:, n - 1] - data[:, n - 2], posinf=def_val, neginf=def_val)
        data[:, n:] = data[:, n:] + np.column_stack([diff.reshape(a, 1)] * nb_col_sup).cumsum(axis=1)
        if null_neg:
            data[:, n:] = np.maximum(data[:, n:], 0)

    elif year_diff:
        diff = np.nan_to_num(data[:, n - 1] - data[:, n - 13], posinf=def_val, neginf=def_val)
        data[:, n:] = data[:, n:] + np.column_stack([diff.reshape(a, 1) / 12] * nb_col_sup).cumsum(axis=1)
        if null_neg:
            data[:, n:] = np.maximum(data[:, n:], 0)

    if s != 0 or is_df:
        if s > 0:
            data_all.iloc[:, s:s + n] = data[:, :n]
        new_cols = [suf + str(i + n) for i in range(1, nb_col_sup + 1)]
        data_all = pd.concat([data_all, pd.DataFrame(data[:, n:], index=data_all.index, columns=new_cols)], axis=1)
        return data_all
    else:
        return data


def concat_align_columns(data1, data2):
    col_common = [x for x in data1.columns if x in data2]
    col_other = [x for x in data2.columns if x not in data1]
    col_all = [x for x in data1.columns if x not in data2]
    data2 = data2.reindex(columns=data2.columns.tolist() + col_all)
    data2 = data2[col_common + col_other + col_all]
    data1 = data1.reindex(columns=data1.columns.tolist() + col_other)
    data1 = data1[col_common + col_other + col_all]
    return pd.concat([data1, data2])


def shift_by(data, shifto, axis, nullify_before=0):
    if axis == 2:
        data = np.swapaxes(data, 1, 2)
    _max = data.shape[1]
    if nullify_before == 0:
        for i in range(1, _max):
            _shift = shifto[i]
            data[:, i, :] = np.roll(data[:, i, :], _shift, axis=1)
            data[:, i, :abs(_shift)] = 0
    elif nullify_before == 1:
        for i in range(1, _max):
            _shift = shifto[i]
            data[:, i, :abs(_shift)] = 0
            data[:, i, :] = np.roll(data[:, i, :], _shift, axis=1)
    elif nullify_before == 2:
        for i in range(1, _max):
            _shift = shifto[i]
            data[:, i, :] = np.roll(data[:, i, :], _shift, axis=1)
    if axis == 2:
        data = np.swapaxes(data, 1, 2)


def shift_by_axis0(data_init, shifto, axis=1):
    axis2 = (axis == 2)
    if axis2:
        data = np.swapaxes(data_init, 1, 2).copy()
    else:
        data = data_init.copy()
    exto = data.shape[1] - 1
    ext_data = np.full((data.shape[0], data.shape[1], exto), 0)
    ext_data = np.concatenate((ext_data, data, ext_data), axis=2)
    x, m, n = ext_data.shape
    idx = np.mod(shifto[:, None] + np.arange(n), n)
    shifted_out = ext_data[np.arange(x)[:, None, None], np.arange(m)[:, None], idx.reshape(x, 1, n)]
    shifted_out = shifted_out[:, :, exto:-exto]
    if axis2:
        shifted_out = np.swapaxes(shifted_out, 1, 2)
    return shifted_out


def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    p = np.full((a.shape[0], a.shape[1] - 1), np.nan)
    a_ext = np.concatenate((p, a, p), axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    outo = viewW(a_ext, (1, n))[np.arange(len(r)), -r + (n - 1), 0]
    outo = np.where(outo == np.nan, 0, outo)

    return outo


def generate_list_from_ordered_dic(od):
    listo = []
    for key, value in od.items():
        listo.append(value)
    return listo


def cut_and_list(data_dic, sizo, chunk):
    chunk_size = chunk if chunk != 0 else 1
    list_dic = []
    for i in range(0, int(sizo / chunk_size) + 1):
        new_dic = {}
        for key, data in data_dic.items():
            new_dic[key] = data[i * chunk_size:(i + 1) * chunk_size].copy()

        list_dic.append(new_dic.copy())
    return list_dic


def cut_and_list_slices(data_dic, size, slices):
    list_dic = []
    for i in range(0, len(slices)):
        end = slices[i + 1] if i != len(slices) - 1 else size
        new_dic = {}
        for key, data in data_dic.items():
            new_dic[key] = data[slices[i]:end].copy()

        list_dic.append(new_dic.copy())
    return list_dic


def flatten(l):
    return [item for sublist in l for item in sublist]


def cut_and_list_join_slices(data_dic, size, slices, special_keys, data_special, join_on):
    list_dic = []
    for i in range(0, len(slices)):
        end = slices[i + 1] if i != len(slices) - 1 else size
        new_dic = {}
        for key, data in data_dic.items():
            new_dic[key] = data[slices[i]:end].copy()
            for j in range(0, len(special_keys)):
                new_dic[special_keys[j]] = data_special[j][data_special[j][join_on].isin(new_dic[key].index)]

        list_dic.append(new_dic.copy())
    return list_dic


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def fill_nan2(df):
    for col in df.columns[df.isnull().any(axis=0)]:
        df[col].fillna(0, inplace=True)
    return df


def clean_var(var):
    if isinstance(var, dict):
        for x, y in var.items():
            clean_var(y)
    else:
        if isinstance(var, list):
            for y in var:
                clean_var(y)
        else:
            del var


def clean_vars(listo_vars):
    for var in listo_vars:
        clean_var(var)
        del var


def in_inf(listo, ind, k):
    for i in range(k, up.nb_annees_usr * 4 + 1):
        if ind + str(i) in listo:
            return True
    return False


def diff_list(l1, l2):
    return (list(set(l1) - set(l2)))


def replace_in_list(listo, val, val2):
    listo2 = listo.copy()
    for n, i in enumerate(listo):
        if i == val:
            listo2[n] = val2
    return listo2


def dot_2_comma(patho, files):
    for f in files:
        with open(patho + "/" + f, 'r') as file:
            read_data = file.read()
            read_data = re.sub(r'(?<=\d)[.]', ',', read_data)

        with open(patho + "/" + f, "w") as out_file:
            out_file.write(read_data)


def concat_dic_df(list_dic, col_key, drop_dup=True):
    data = None
    for dic in list_dic:
        if data is not None:
            data = pd.concat([data, pd.concat(list(dic.values()))])
        else:
            data = pd.concat(list(dic.values()))
    if drop_dup:
        data = data.drop_duplicates([col_key])

    return data


def fill_na_pn_z(data):
    data[np.isnan(data)] = 0
    return data


def divide_np_pd(data1, data2):
    data = np.divide(fill_na_pn_z(data1.values), fill_na_pn_z(data2.values))
    return np.nan_to_num(data, posinf=0, neginf=0)


def fill_all_na(data):
    data[np.isnan(data)] = 0
    return np.nan_to_num(data, posinf=0, neginf=0)


def clean_dicos(list_dic):
    for i in range(0, len(list_dic)):
        for x, y in list_dic[i].items():
            del y
        list_dic[i].clear()


def clean_df(df):
    df = 0
    del df


def clean_dic_df(dico):
    for i in list(dico.keys()):
        dico[i] = 0
        del dico[i]


def divide_np(data1, data2):
    data = np.divide(fill_na_pn_z(data1), fill_na_pn_z(data2))
    return np.nan_to_num(data, posinf=0, neginf=0)


def decompress(data, size):
    return np.array(data.todense()).reshape(size)


def compress(data, size):
    return csr_matrix(data.reshape((size[0], size[1] * size[2])))


def begin_with_list(listo, x):
    for y in listo:
        if y in x:
            return True
    return False


def begin_in_list2(listo, x):
    all = "," + ",".join(listo)
    if "," + x in all:
        return True
    return False


def begin_in_list(listo, x):
    all = ",".join(listo)
    if x in all:
        return True
    return False


def rename_mapping_columns(data, module, prefix, module_alias):
    columns = [v for v in dir(module) if v[:2] != "__" and v[:len(prefix)] == prefix]
    data = data.rename(
        columns={eval(module_alias + "." + k): eval(module_alias + "." + k.replace("_old", "")) for k in columns if
                 "_old" in k})
    return data


def sum_k(data, k):
    data[:, k:, :] = 0
    data[:, :, 0:k - 1] = 0
    return data.sum(axis=1)


def sum_each2(data, cols, proj=False, per=0, fix=False, interv=0):
    if not proj:
        return pd.DataFrame(data.sum(axis=1), columns=cols)
    elif proj and not fix:
        return pd.DataFrame(data[:, per - interv:per].sum(axis=1), columns=cols)
    elif proj and fix:
        return pd.DataFrame(data[:, :per].sum(axis=1), columns=cols)
