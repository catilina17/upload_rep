import numpy as np
import pandas as pd
from mappings.pass_alm_fields import PASS_ALM_Fields as pa

k_em = 6.5
k_ef = 12
k_mn = 12
limit_month = 120
max_months = 240


def interpol_missing_em(data_all, em_ind):
    """ Fonction permettant d'interpoler les indicateurs de type EM """
    data = data_all[data_all[pa.NC_PA_IND03] == em_ind].copy()
    data_orig = data_all[data_all[pa.NC_PA_IND03] == em_ind].copy()
    for j in range(0, int((max_months - limit_month) / 12)):
        deb = j * 12 + limit_month
        fin = (j + 1) * 12 + limit_month
        adjust_const = (data["M" + str(deb)] - data["M" + str(fin)]) / k_em
        cols = ["M" + str(i) for i in range(deb + 1, fin + 1) if not "M" + str(i) in data.columns]
        data = pd.concat([data, pd.DataFrame([[0] * len(cols)], index=data.index, columns=cols)], axis=1)
        for i in range(deb + 1, fin + 1):
            data["M" + str(i)] = np.where(data["M" + str(i - 1)] == 0, 0, data["M" + str(i - 1)] - adjust_const)
            data["M" + str(i)] = np.where(data["M" + str(i)] / data["M" + str(i - 1)] < 0, 0, data["M" + str(i)])

    for j in range(1, max_months + 1):
        data["M" + str(j)] = np.where(abs(data["M" + str(j)]) < 1, 0, data["M" + str(j)])

    data_all.loc[data_all[pa.NC_PA_IND03] == em_ind] = data

    return data_orig


def interpol_missing_ef(data_all, ef_ind):
    """ Fonction permettant d'interpoler les indiateurs de type EF"""
    data = data_all.loc[data_all[pa.NC_PA_IND03] == ef_ind].copy()
    for j in range(0, int((max_months - limit_month) / 12)):
        deb = j * 12 + limit_month
        fin = (j + 1) * 12 + limit_month
        adjust_const = 1 / k_ef * (data["M" + str(deb)] - data["M" + str(fin)])
        cols = ["M" + str(i) for i in range(deb + 1, fin + 1) if not "M" + str(i) in data.columns]
        data = pd.concat([data, pd.DataFrame([[0] * len(cols)], index=data.index, columns=cols)], axis=1)
        for i in range(deb + 1, fin + 1):
            data["M" + str(i)] = data["M" + str(i - 1)].values - adjust_const

    for j in range(1, max_months + 1):
        data["M" + str(j)] = np.where(abs(data["M" + str(j)]) < 1, 0, data["M" + str(j)])

    data_all.loc[data_all[pa.NC_PA_IND03] == ef_ind] = data


def interpol_missing_mn(data_all, mn_ind, em_ind, data_orig_em):
    """ Fonction permettant d'interpoler les indicateurs de type MN """
    data_mn = data_all.loc[data_all[pa.NC_PA_IND03] == mn_ind]
    data_em = data_all.loc[data_all[pa.NC_PA_IND03] == em_ind]
    for j in range(0, int((max_months - limit_month) / 12)):
        deb = j * 12 + limit_month
        fin = (j + 1) * 12 + limit_month
        tx_adjust = np.where(data_orig_em["M" + str(fin)] == 0, 0,
                             data_mn["M" + str(fin)].values / data_orig_em["M" + str(fin)].values)
        cols = ["M" + str(i) for i in range(deb + 1, fin + 1) if not "M" + str(i) in data_mn.columns]
        data_mn = pd.concat([data_mn, pd.DataFrame([[0] * len(cols)], index=data_mn.index, columns=cols)], axis=1)
        for i in range(deb + 1, fin + 1):
            data_mn["M" + str(i)] = data_em["M" + str(i)] * tx_adjust / k_mn

    data_all.loc[data_all[pa.NC_PA_IND03] == mn_ind] = data_mn


def interpolate_missing_cols(data):

    for indic in ["LEF", "TEF"]:
        interpol_missing_ef(data, indic)
    for indic in ["TEM", "LEM"]:
        data_orig = interpol_missing_em(data, indic)
    for indic_mni in ["LMN"]:
        interpol_missing_mn(data, indic_mni, "LEM", data_orig)
    return data
