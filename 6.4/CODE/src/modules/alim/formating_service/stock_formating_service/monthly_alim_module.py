import pandas as pd
import numpy as np
import modules.alim.parameters.user_parameters as up
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import os
import mappings.mapping_module as mp
import mappings as gma
import  logging

logger = logging.getLogger(__name__)

perimetre_pr_per = ["A-PR-PERSO", "AHB-NS-PR-PER"]
perimetre_ptz = ["A-PTZ", "A-PTZ+", "AHB-NS-CR-PTZ"]
perimetre_cr_eq = ["A-CR-EQ-MUL", "A-CR-EQ-AIDE", "A-CR-EQ-STR", "A-CR-EQ-STD", "A-CR-EQ-CPLX", "AHB-NS-CR-EQ","AHB-NS-CR-EQA"]
perimetre_cap_floor = ["AHB-CAP", "PHB-CAP", "AHB-FLOOR", "PHB-FLOOR"]
perimetre_habitat = ["A-CR-HAB-LIS", "A-CR-HAB-STD", "A-CR-HAB-MOD", "A-CR-HAB-AJU", "A-PR-STARDEN", "AHB-NS-CR-HAB"]
perimetre_habitat_casden = ["A-CR-HAB-BON", "AHB-NS-CR-HBN"]
perimetre_PEL = ["P-PEL", "P-PEL-C"]


def add_moteur_compil_data(STOCK):
    logger.info("AJOUT DES DONNEES DE LA SIMUL A BLANC")
    compils_path = os.path.join(up.chemin_compils, up.current_etab)
    deb_mois = int(up.mois_inter_trim)

    compil_stock, compil_pn = get_compils_data(compils_path)

    num_cols = ["M" + str(i) if i >= 10 else "M0" + str(i) for i in range(0, pa.MAX_MONTHS_ST + 1)]

    new_stock, qual_cols = concat_and_filter_compils(compil_stock, compil_pn, num_cols, deb_mois)

    new_stock, new_num_cols = adjust_months(new_stock, qual_cols, deb_mois)

    new_stock = rename_indicators(new_stock, deb_mois)

    new_stock = add_missing_indics(new_stock, new_num_cols)

    new_stock = align_stock_columns(new_stock, STOCK)

    new_stock = remove_calculateur_perimetre(new_stock)

    STOCK = pd.concat([STOCK, new_stock])

    return STOCK


def get_compils_data(compils_path):
    compil_stock = pd.read_csv(os.path.join(compils_path, "CompilST.csv"), sep=";", decimal=",", encoding="latin-1")
    compil_pn = pd.read_csv(os.path.join(compils_path, "CompilPN.csv"), sep=";", decimal=",", encoding="latin-1")
    return compil_stock, compil_pn


def remove_calculateur_perimetre(new_stock):
    new_stock = new_stock[~((new_stock[pa.NC_PA_CONTRACT_TYPE].isin(perimetre_pr_per + perimetre_ptz + perimetre_habitat
                                                           + perimetre_cr_eq)) &
                            (new_stock[pa.NC_PA_INDEX_AGREG] == "FIXE"))].copy()

    new_stock = new_stock[~((new_stock[pa.NC_PA_CONTRACT_TYPE].isin(perimetre_cap_floor + perimetre_PEL)))].copy()

    new_stock = new_stock[~((new_stock[pa.NC_PA_CONTRACT_TYPE].isin(perimetre_habitat_casden)) &
                            (new_stock[pa.NC_PA_INDEX_AGREG] == "FIXE") & (new_stock[pa.NC_PA_ETAB] == "CSDN"))].copy()

    return new_stock


def align_stock_columns(new_stock, STOCK):
    new_stock = new_stock[STOCK.columns.tolist()].copy()
    new_stock = new_stock.copy().groupby(pa.NC_PA_COL_SORTIE_QUAL + [pa.NC_PA_isECH]).sum().reset_index()
    cases = [new_stock[pa.NC_PA_IND03]==x for x in [pa.NC_PA_LEF, pa.NC_PA_LEM, pa.NC_PA_LMN, pa.NC_PA_LMN_EVE, pa.NC_PA_TEF, pa.NC_PA_TEM]]
    new_stock["order_indic"] = np.select(cases, [i for i  in range(1, len(cases) + 1)])
    new_stock = new_stock.sort_values([pa.NC_PA_BILAN, pa.NC_PA_CLE,"order_indic"]).drop("order_indic", axis=1)
    return new_stock


def add_missing_indics(new_stock, new_num_cols):
    new_stock_em = new_stock[new_stock[pa.NC_PA_IND03] == pa.NC_PA_LEM].copy()
    new_stock_ef = new_stock[new_stock[pa.NC_PA_IND03] == pa.NC_PA_LEF].copy()
    new_stock_mn = new_stock[new_stock[pa.NC_PA_IND03] == pa.NC_PA_LMN].copy()
    new_stock_tef = new_stock[new_stock[pa.NC_PA_IND03] == pa.NC_PA_TEF].copy()
    new_stock_tem = new_stock[new_stock[pa.NC_PA_IND03] == pa.NC_PA_TEM].copy()
    new_stock_lmn_eve = new_stock_mn.copy()
    new_stock_lmn_eve[pa.NC_PA_IND03] = pa.NC_PA_LMN_EVE

    list_num_vars = [np.array(new_stock_ef), np.array(new_stock_em), np.array(new_stock_mn),
                     np.array(new_stock_tef), np.array(new_stock_tem), np.array(new_stock_lmn_eve)]

    s = list_num_vars[0].shape[0]
    t = list_num_vars[0].shape[1]
    l = len(list_num_vars)
    new_stock = np.stack(list_num_vars, axis=1).reshape(s * l, t)
    new_stock = pd.DataFrame(new_stock, columns=new_stock_em.columns)

    index_list = ["ST_A" + str(i) for i in range(1, s + 1)]
    new_stock[pa.NC_PA_INDEX] = np.repeat(np.array(index_list), l)

    return new_stock


def rename_indicators(new_stock, deb_mois):
    cases = [new_stock[pa.NC_PA_IND03] == x for x in ["EF", "EM", "MN", "GP TF EF" + str(deb_mois), "GP TF EM"  + str(deb_mois)]]
    vals = [pa.NC_PA_LEF, pa.NC_PA_LEM, pa.NC_PA_LMN, pa.NC_PA_TEF, pa.NC_PA_TEM]
    new_stock[pa.NC_PA_IND03] = np.select(cases, vals)
    return new_stock


def concat_and_filter_compils(compil_stock, compil_pn, num_cols, deb_mois):
    new_stock = pd.concat([compil_stock, compil_pn])
    new_stock = new_stock[~new_stock[pa.NC_PA_CONTRACT_TYPE].str.contains("AJUST")].copy()
    new_stock = new_stock[new_stock["IsIG"] != "IGM"].copy()
    list_inds = new_stock[pa.NC_PA_IND03].unique().tolist()
    absent_ind = (not "EF" in list_inds) | (not "EM" in list_inds) | (not "MN" in list_inds)\
                 | (not "GP TF EF" + str(deb_mois) in list_inds) | (not "GP TF EM" + str(deb_mois) in list_inds)
    if absent_ind:
        msg = "Certains indicateurs parmi %s sont absent dans les compils de la SIM à blanc" %(["EF", "EM", "MN",
               "GP TF EF" + str(deb_mois), "GP TF EM" + str(deb_mois)])
        logger.error(msg)
        raise ValueError(msg)

    new_stock = new_stock[new_stock[pa.NC_PA_IND03].isin(["EF", "EM", "MN", "GP TF EF" + str(deb_mois), "GP TF EM"  + str(deb_mois)])].copy()
    qual_cols = new_stock.loc[:, :"M00"].iloc[:, :-1].columns.tolist() + [pa.NC_PA_isECH]
    new_stock = get_is_ech_column(new_stock)
    new_stock = new_stock[qual_cols + num_cols].copy()
    group_by_cols = [x for x in qual_cols if x not in ["IND01", "IND02"]]
    new_stock = new_stock[group_by_cols + num_cols].copy().groupby(group_by_cols).sum().reset_index()
    return new_stock, group_by_cols


def get_is_ech_column(new_stock):
    cle_data = [pa.NC_PA_CONTRACT_TYPE]
    map_contrat = gma.map_pass_alm["CONTRATS"]["TABLE"].copy()
    map_contrat = map_contrat.reset_index(drop=True).drop_duplicates([pa.NC_PA_CONTRACT_TYPE]).set_index(pa.NC_PA_CONTRACT_TYPE)
    new_stock = mp.map_data(new_stock, map_contrat, keys_data=cle_data, error_mapping=False,
                            cols_mapp=[pa.NC_PA_isECH], option="")
    return new_stock


def adjust_months(new_stock, qual_cols, deb_mois):
    num_cols_ajust = ["M" + str(i) if i >= 10 else "M0" + str(i) for i in range(deb_mois, pa.MAX_MONTHS_ST + 1)]
    new_stock = new_stock[qual_cols + num_cols_ajust].copy()
    new_num_cols = ["M" + str(i) for i in range(0, pa.MAX_MONTHS_ST + 1 - deb_mois)]
    new_stock.columns = qual_cols + new_num_cols
    new_stock["M" + str(pa.MAX_MONTHS_ST)] = 0
    return new_stock, new_num_cols + ["M" + str(pa.MAX_MONTHS_ST)]
