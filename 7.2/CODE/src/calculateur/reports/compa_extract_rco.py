import pandas as pd
import numpy as np
import datetime


def transform_cols(data):
    num_cols = [x for x in data.columns.tolist() if "/" in x]
    try:
        num_cols = [datetime.datetime.strptime(x, '%d/%m/%Y').date() for x in num_cols]
    except:
        num_cols = [datetime.datetime.strptime(x, '%d/%m/%Y %H:%M:%S').date() for x in num_cols]
    data.columns = [x for x in data.columns.tolist() if not "/" in x] + num_cols
    return data


def select_num_cols(data):
    other_cols = [x for x in data.columns.tolist() if not isinstance(x, datetime.date)]
    num_cols = [x for x in data.columns.tolist() if isinstance(x, datetime.date)]

    present_cols = sorted(num_cols)
    data = data.loc[:, other_cols + present_cols]

    new_num_col_names = ["M" + str(x) for x in range(0,121)] + ["M" + str(x) for x in range(132,301,12)]
    data.columns = other_cols + new_num_col_names
    data[new_num_col_names] = data[new_num_col_names].fillna(value=0)

    return data, new_num_col_names

def compare_extracts_calc_vs_rco():
    rco_extract = r"C:\Users\HOSSAYNE\Documents\BPCE_ARCHIVES\SOURCES\SOURCES_RCO_2022_v3 -360 - ME\2024-09-30_6.3\DATA\BPACA\ALIM_STOCK\BPACA_2024-09-30_GAP-LIQ-EF.tab"
    calc_extract = r"C:\Users\HOSSAYNE\Documents\BPCE_ARCHIVES\RESULTATS\SIMUL_CALCULATEUR\SIMULS\SIMUL_EXEC-20241231.1813.48\BPACA\BPACA_2024-09-30_GAP-LIQ-EF.tab"

    rco_data = pd.read_csv(rco_extract, sep="\t")
    calc_data = pd.read_csv(calc_extract, sep="\t")

    rco_data.columns = [x.upper() for x in rco_data.columns.tolist()]
    calc_data.columns = [x.upper() for x in calc_data.columns.tolist()]

    agreg_vars = ["ETAB", "CONTRACT_TYPE","CCY_CODE", "RATE_CODE", "FAMILY", "CATEGORY_CODE"]
    rco_data = rco_data[agreg_vars + ["datefin_period".upper(), "montant".upper()]].copy()
    calc_data = calc_data[agreg_vars + ["datefin_period".upper(), "montant".upper()]].copy()

    rco_data = rco_data.pivot_table(index=agreg_vars, columns=["datefin_period".upper()],
                                    values="montant".upper(), aggfunc="sum", fill_value=0)

    calc_data = calc_data.pivot_table(index=agreg_vars, columns=["datefin_period".upper()],
                                      values="montant".upper(), aggfunc="sum", fill_value=0)

    rco_data = rco_data.reset_index()
    calc_data = calc_data.reset_index()

    rco_data = transform_cols(rco_data)
    calc_data = transform_cols(calc_data)

    rco_data, num_cols = select_num_cols(rco_data)
    calc_data, num_cols = select_num_cols(calc_data)

    #calc_data[num_cols] = np.where(calc_data["CATEGORY_CODE"].str.contains("PASSIF").values.reshape(calc_data.shape[0], 1),
    #                              -1 * calc_data[num_cols].values, calc_data[num_cols].values)

    rco_data["VERSION"] = "RCO"
    calc_data["VERSION"] = "PASS-ALM"

    data = pd.concat([rco_data, calc_data])

    output_file = r"C:\Users\HOSSAYNE\Documents\BPCE_ARCHIVES\RESULTATS\SIMUL_CALCULATEUR\SIMULS\SIMUL_EXEC-20241231.1737.28\BPACA\compa.csv"
    data.to_csv(output_file, sep=";", decimal=",", index=False)

compare_extracts_calc_vs_rco()
