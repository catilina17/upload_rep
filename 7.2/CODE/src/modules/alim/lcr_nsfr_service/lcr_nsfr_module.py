import utils.general_utils as gu
import mappings as gma
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import mappings.mapping_functions as mp
import modules.alim.parameters.general_parameters as gp
from modules.alim.parameters.RAY_parameters import *
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

global table_nco_rl, ray_mapping


def format_bilan_ray_data(ray_data):
    filtres = [ray_data[NC_RAY_CONTRACT].str[0:2] == "A-", ray_data[NC_RAY_CONTRACT].str[0:2] == "P-" \
        , (ray_data[NC_RAY_CONTRACT].str[0:2] == "HB") & (
                           (ray_data[NC_RAY_OUTSTANDING0] < 0) | (ray_data[NC_RAY_OUTSTANDING] < 0)),
               ray_data[NC_RAY_CONTRACT].str[0:2] == "HB"]

    ray_data[pa.NC_PA_BILAN] = np.select(filtres, ["B ACTIF", "B PASSIF", "HB PASSIF", "HB ACTIF"],
                                      default=gp.empty_mapping)

    return ray_data


def map_ray_data_pass_alm_mappping(ray_data):
    mapping_global = gma.map_pass_alm

    ray_data = mp.map_data(ray_data, mapping_global["CONTRATS"], keys_data=[pa.NC_PA_BILAN, NC_RAY_CONTRACT] \
                           , name_mapping="DATA RAY vs.", except_name_mapping="RAY_NOT_IN_PASSALM",
                           cols_mapp=[pa.NC_PA_CONTRACT_TYPE])

    ray_data = ray_data.rename(columns={NC_RAY_MARCHE: pa.NC_PA_MARCHE})

    return ray_data.drop([NC_RAY_CONTRACT], axis=1)


def filter_and_group_data(ray_data):
    # Suppression des doubles lignes RL et NCO
    group_col = [NC_RAY_CONTRACT, NC_RAY_MARCHE, NC_RAY_LCR_TIERS]
    ray_data = ray_data[group_col + num_cols_ray_out]
    ray_data = ray_data.groupby(by=group_col, as_index=False, dropna=False).sum()
    return ray_data


def unsign_passif(ray_data):
    filtre_passif = np.array(ray_data[pa.NC_PA_BILAN].isin(["B PASSIF", "HB PASSIF"])).reshape(ray_data.shape[0], 1)
    ray_data[num_cols_ray_out] = np.where(filtre_passif, -ray_data[num_cols_ray_out].values,
                                          ray_data[num_cols_ray_out].values)

    return ray_data


def generate_NCO_RL_table(ray_data):
    global table_nco_rl
    table_nco_rl_marche = ray_data.groupby(by=[pa.NC_PA_BILAN, pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE], as_index=False,
                                           dropna=False).sum(numeric_only=True)
    table_nco_rl = ray_data.groupby(by=[pa.NC_PA_BILAN, pa.NC_PA_CONTRACT_TYPE], as_index=False, dropna=False).sum(numeric_only=True)
    tables_nco_rl = [table_nco_rl_marche, table_nco_rl]
    for i in range(0, len(tables_nco_rl)):
        tables_nco_rl[i][NCO_RAY_COEFF] = np.divide(
            tables_nco_rl[i][NC_RAY_WOUTFLOW] - tables_nco_rl[i][NC_RAY_WINFLOW], \
            tables_nco_rl[i][NC_RAY_OUTSTANDING0])
        tables_nco_rl[i][NCO_RAY_COEFF] = tables_nco_rl[i][NCO_RAY_COEFF].fillna(0).replace((np.inf, -np.inf), (0, 0))

        tables_nco_rl[i][RL_RAY_COEFF] = np.divide(tables_nco_rl[i][NC_RAY_TOTAL_RL],
                                                   tables_nco_rl[i][NC_RAY_OUTSTANDING])
        tables_nco_rl[i][RL_RAY_COEFF] = tables_nco_rl[i][RL_RAY_COEFF].fillna(0).replace((np.inf, -np.inf), (0, 0))

    table_nco_rl = pd.concat(tables_nco_rl)
    table_nco_rl = table_nco_rl[table_nco_rl[pa.NC_PA_CONTRACT_TYPE].notnull()]
    table_nco_rl[pa.NC_PA_MARCHE] = table_nco_rl[pa.NC_PA_MARCHE].mask(table_nco_rl[pa.NC_PA_MARCHE].isnull(), "*")

    col_sortie = [pa.NC_PA_BILAN, pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE, NCO_RAY_COEFF, \
                  RL_RAY_COEFF] + num_cols_ray_out
    table_nco_rl = table_nco_rl[col_sortie].copy()


def get_lcr_tiers_and_share(ray_data, group_cols1):
    group_cols2 = [NC_RAY_LCR_TIERS]
    num_col = [NC_RAY_OUTSTANDING0]
    ray_map = ray_data[group_cols1 + group_cols2 + num_col].groupby(by=group_cols1 + group_cols2, as_index=False,
                                                                    dropna=False).sum()
    ray_map_total = ray_map.drop(group_cols2, axis=1).groupby(by=group_cols1, as_index=False, dropna=False).sum()
    ray_map = ray_map.join(ray_map_total.set_index(group_cols1), on=group_cols1, rsuffix="_TOTAL", how="left")
    ray_map[pa.NC_PA_LCR_TIERS_SHARE] = np.divide(ray_map[NC_RAY_OUTSTANDING0], ray_map[NC_RAY_OUTSTANDING0 + "_TOTAL"])
    ray_map[pa.NC_PA_LCR_TIERS_SHARE] = (ray_map[pa.NC_PA_LCR_TIERS_SHARE] * 100).fillna(DEF_VALUE_LCR_SHARE). \
        replace((np.inf, -np.inf), (DEF_VALUE_LCR_SHARE, DEF_VALUE_LCR_SHARE))

    ray_map = ray_map[group_cols1 + [NC_RAY_LCR_TIERS, pa.NC_PA_LCR_TIERS_SHARE]]

    ray_map = ray_map.rename(columns={NC_RAY_LCR_TIERS: pa.NC_PA_LCR_TIERS})

    return ray_map


def genrate_ray_mapping(ray_data):
    global ray_map
    ray_map1 = get_lcr_tiers_and_share(ray_data[~ray_data[pa.NC_PA_MARCHE].isnull()], [pa.NC_PA_BILAN, pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE])
    ray_map2 = get_lcr_tiers_and_share(ray_data[ray_data[pa.NC_PA_MARCHE].isnull()], [pa.NC_PA_BILAN, pa.NC_PA_CONTRACT_TYPE])
    ray_map = pd.concat([ray_map1, ray_map2])
    ray_map[pa.NC_PA_MARCHE] = ray_map[pa.NC_PA_MARCHE].mask(ray_map[pa.NC_PA_MARCHE].isnull(), "*")
    ray_map["KEY_REY_MAP"] = ray_map[[pa.NC_PA_BILAN, pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE]].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)
    ray_map = ray_map.set_index("KEY_REY_MAP")


def parse_ray_file(ray_file):
    logger.info("   Lecture de : " + ray_file.split("\\")[-1])

    ray_data = gu.read_large_csv_file(ray_file)

    ray_data = filter_and_group_data(ray_data)

    ray_data = format_bilan_ray_data(ray_data)

    ray_data = map_ray_data_pass_alm_mappping(ray_data)

    ray_data = unsign_passif(ray_data)

    generate_NCO_RL_table(ray_data)

    genrate_ray_mapping(ray_data)


def override_values(data):
    # OVERRIDE P-DAV-CORP et P-DAV-RET
    #filtres = [data[pa.NC_PA_CONTRACT_TYPE] == "P-DAV-CORP", data[pa.NC_PA_CONTRACT_TYPE] == "P-DAV-PART"]
    #data[pa.NC_PA_LCR_TIERS] = np.select(filtres, ["ENF", "RET/SCP/PME"], default=data[pa.NC_PA_LCR_TIERS])

    data[pa.NC_PA_LCR_TIERS_SHARE] = data[pa.NC_PA_LCR_TIERS_SHARE].fillna(DEF_VALUE_LCR_SHARE)
    data[pa.NC_PA_LCR_TIERS_SHARE] = data[pa.NC_PA_LCR_TIERS_SHARE].mask(data[pa.NC_PA_LCR_TIERS_SHARE] == gp.empty_mapping, DEF_VALUE_LCR_SHARE)
    data[pa.NC_PA_LCR_TIERS_SHARE] = data[pa.NC_PA_LCR_TIERS_SHARE].astype(np.float64)

    return data


def adjust_amounts(data):
    cols_num = [x for x in data.columns if str(x)[:3] in gp.prefixes_amounts]
    data[cols_num] = data[cols_num].values * np.array(data[pa.NC_PA_LCR_TIERS_SHARE]).reshape(data.shape[0], 1) / 100
    return data


def map_lcr_tiers_and_share(data):
    cles_a_combiner = [pa.NC_PA_BILAN, pa.NC_PA_CONTRACT_TYPE, pa.NC_PA_MARCHE]
    data = gu.gen_combined_key_col(data, ray_map, cols_key=cles_a_combiner, symbol_any="*",
                                   name_col_key="KEY_RAY", set_index=False, filter_comb=True)

    name_mapping = "STOCK/PN DATA vs. DATA RAY"
    col_err = ["TEM_M0", "LEF_M0"] if "TEM_M0" in data.columns.tolist() else []

    data = data.drop([pa.NC_PA_LCR_TIERS, pa.NC_PA_LCR_TIERS_SHARE], axis=1)

    data = mp.map_data(data, ray_map, keys_data=["KEY_RAY"], \
                       except_name_mapping="PASSALM_NOT_IN_RAY", cols_mapp=[pa.NC_PA_LCR_TIERS, pa.NC_PA_LCR_TIERS_SHARE],
                       allow_duplicates=True, option="", name_mapping=name_mapping, col_err=col_err)

    data = data.drop(["KEY_RAY"], axis=1)

    data = override_values(data)

    data = adjust_amounts(data)

    return data
