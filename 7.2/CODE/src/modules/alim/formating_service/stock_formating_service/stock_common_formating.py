import pandas as pd
import modules.alim.parameters.general_parameters as gp
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import utils.general_utils as gu
import numpy as np
import logging
import datetime
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


def add_missing_num_cols(dar, data, num_cols, prefix, necessary_cols, force_zero=False):
    missing_cols = [x for x in necessary_cols if x not in num_cols]
    warning_cols = []
    for x in missing_cols:
        if not force_zero:
            if ("SWAP" in prefix or "LMN" in prefix or "LMN EVE" in prefix) and x > (dar + relativedelta(months=60)).date():
                zero_col = pd.DataFrame(0, columns=[x], index=data.index)
                data = pd.concat([data,zero_col], axis=1)
            else:
                gaps = [(i, abs(x - y)) for i, y in enumerate(num_cols)]
                gaps = sorted(gaps, key=lambda x: x[1])
                index_smallest_gap = gaps[0][0]
                data[x] = data[num_cols[index_smallest_gap]]
                warning_cols.append(x)
        else:
            zero_col = pd.DataFrame(0, columns=[x], index=data.index)
            data = pd.concat([data, zero_col], axis=1)

    if warning_cols != [] and not force_zero:
        logger.warning("The following dates were missing in the file : " + str([str(x) for x in warning_cols]))

    return data


def select_num_cols(current_etab, dar, data, prefix, force_zero=False):
    other_cols = [x for x in data.columns.tolist() if not isinstance(x, datetime.date)]
    num_cols = [x for x in data.columns.tolist() if isinstance(x, datetime.date)]
    necessary_cols = [(dar + relativedelta(months=x) + relativedelta(day=31)).date() for x in pa.ALL_NUM_STOCK]

    data = add_missing_num_cols(dar, data, num_cols, prefix, necessary_cols, force_zero=force_zero)
    present_cols = sorted(necessary_cols)
    data = data.loc[:, other_cols + present_cols]

    new_num_col_names = [prefix + "_M" + str(x) for x in pa.ALL_NUM_STOCK]
    data.columns = other_cols + new_num_col_names
    try:
        data[new_num_col_names] = data[new_num_col_names].astype(float)
    except:
        data[new_num_col_names] = data[new_num_col_names].replace('E', 'e', regex=True).replace(',', '.', regex=True).replace('-', '0').astype(float)

    data[new_num_col_names] = data[new_num_col_names].fillna(value=0)

    if prefix in ["SWAP","LMN", "LMN EVE"] and current_etab in gp.NTX_FORMAT:
        cols_60_plus = [x for x in new_num_col_names if int(x[x.find("_")+2:])>=61]
        data[cols_60_plus]=0

    return data, new_num_col_names

def format_bilan_column(data):
    filters = [data[pa.NC_PA_BILAN] == "BILAN-ACTIF", data[pa.NC_PA_BILAN] == "BILAN-PASSIF"]
    values = ["B ACTIF", "B PASSIF"]
    data[pa.NC_PA_BILAN] = np.select(filters, values, default=data[pa.NC_PA_BILAN])
    return data

def upper_columns_names(data):
    data.columns = [str(x).upper().strip() for x in data.columns.tolist()]
    return data

def upper_non_num_cols_vals(data, num_cols):
    vars = [x for x in data.columns if x not in num_cols]
    data[vars] = data[vars].astype(str)
    gu.strip_and_upper(data, vars)
    return data

def change_sign_passif(data):
    num_cols = [x for x in data.columns if ("LEF_M" in x) or ("TEF_M" in x)\
                or ("SWAP_M" in x) or ("LMN_M" in x)  or ("LMN EVE_M" in x) or ("TEM_M" in x) or\
                ("TEF_M" in x)  or ("LEM_M" in x) or ("DEM_M" in x) or ("DMN_M" in x)]
    passif = (data[pa.NC_PA_BILAN].str.strip() == "B PASSIF") | (data[pa.NC_PA_BILAN].str.strip() == "HB PASSIF")
    data.loc[passif, num_cols] = - data.loc[passif, num_cols].values
    return data

def select_cols_to_keep(data, cols_to_keep, num_cols=[]):
    data = data[[x for x in cols_to_keep if x in data.columns] + num_cols].copy()
    return data

def format_paliers(data, palier_col):
    invalid_palier = (data[palier_col].isnull()) | (data[palier_col].astype(str) == "nan") \
                     | (data[palier_col] == " ") | (data[palier_col] == "(vide)")

    data[palier_col] = data[palier_col].mask(invalid_palier, "-")

    data = gu.force_integer_to_string(data, palier_col)

    return data

def append_data(STOCK_DATA, data_prefix, prefix, warning=True):
    # STOCK_DATA et data_prefix must indexed
    shape_stock = STOCK_DATA.shape[0]
    STOCK_DATA = pd.concat([STOCK_DATA, data_prefix], axis=1)
    if shape_stock != STOCK_DATA.shape[0] and warning:
        logger.warning("Il y a des lignes dans le fichier %s qu'on ne retrouve pas le fichier LEF" % prefix)
    num_col_names = [prefix + "_M" + str(x) for x in pa.ALL_NUM_STOCK]
    STOCK_DATA[num_col_names] = STOCK_DATA[num_col_names].fillna(0)
    return STOCK_DATA

def finalize_formatting(data):
    data = change_sign_passif(data)
    num_cols = [x for x in data.columns if ("LEF_M" in x) or ("TEF_M" in x) \
                or ("SWAP_M" in x) or ("LMN_M" in x) or ("LMN EVE_M" in x) or ("TEM_M" in x) or \
                ("TEF_M" in x) or ("LEM_M" in x)]
    data[num_cols] = data[num_cols].fillna(0)
    return data