import mappings.mapping_module as mp
import importlib

""" POUR PYINSTALLER"""
import modules.alim.formating_service.stock_formating_service.RZO.initial_formating
import modules.alim.formating_service.stock_formating_service.NTX_SEF.initial_formating
import modules.alim.formating_service.stock_formating_service.ONEY_SOCFIM.initial_formating
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
import modules.alim.formating_service.stock_formating_service.monthly_alim_module as mth
import utils.general_utils as gu
import modules.alim.parameters.general_parameters as gp
import modules.alim.parameters.user_parameters as up
import numpy as np
import pandas as pd
import modules.alim.lcr_nsfr_service.lcr_nsfr_module as lcr_nsfr
import logging
from warnings import simplefilter

global lcr_nsfr_data

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)

def read_and_format_stock_data():
    init_f = importlib.import_module("modules.alim.formating_service.stock_formating_service." + up.category + ".initial_formating")

    STOCK_DATA = init_f.read_and_join_all_files()

    STOCK_DATA = compute_LEM_and_TEM(STOCK_DATA)

    STOCK_DATA = correct_fx_rates(STOCK_DATA)

    STOCK_DATA = map_with_ALM_mappings(STOCK_DATA)

    STOCK_DATA = agregate_STOCK(STOCK_DATA)

    STOCK_DATA = process_starting_points(STOCK_DATA)

    STOCK_DATA = change_to_to_row_format(STOCK_DATA)

    STOCK_DATA = prepare_output_STOCK(STOCK_DATA)

    if up.type_simul == "FREQUENCE MENSUELLE":
        STOCK_DATA = mth.add_moteur_compil_data(STOCK_DATA)

    return STOCK_DATA


def change_to_to_row_format(data):
    indics_ordered = [pa.NC_PA_LEF, pa.NC_PA_LEM, pa.NC_PA_LMN, pa.NC_PA_LMN_EVE, pa.NC_PA_TEF,\
                      pa.NC_PA_TEM, pa.NC_PA_DEM_RCO, pa.NC_PA_DMN_RCO]
    data = gu.unpivot_data(data, pa.NC_PA_IND03)
    keys_order = [pa.NC_PA_BILAN, pa.NC_PA_CLE]
    data = gu.add_empty_indics(data, [pa.NC_PA_DEM_RCO, pa.NC_PA_DMN_RCO], pa.NC_PA_IND03, "LEF", pa.NC_PA_COL_SORTIE_NUM_ST, \
                                      order=True, indics_ordered=indics_ordered, keys_order=keys_order)
    data = add_missing_num_cols(data)
    return data


def add_missing_num_cols(data):
    cols = [col for col in pa.NC_PA_COL_SORTIE_NUM_ST if col not in data.columns.tolist()]
    data = pd.concat([data, pd.DataFrame([[0] * len(cols)], \
                                         index=data.index, \
                                         columns=cols)], axis=1)
    return data

def agregate_STOCK(STOCK_DATA):
    nums_cols = [x for x in STOCK_DATA.columns if ("LEF_M" in x) or ("TEF_M" in x) \
                 or ("LMN_M" in x) or ("LMN EVE_M" in x)  or ("LEM_M" in x) or ("TEM_M" in x) or ("DEM_M" in x) or ("DMN_M" in x)]
    qual_cols = [x for x in STOCK_DATA.columns if x not in nums_cols]

    STOCK_DATA[nums_cols] = STOCK_DATA[nums_cols].astype(np.float64)

    STOCK_DATA[qual_cols] = STOCK_DATA[qual_cols].fillna("")

    AG_STOCK_DATA = STOCK_DATA.copy().groupby(by=qual_cols, as_index=False).sum()

    AG_STOCK_DATA[pa.NC_PA_CLE] = AG_STOCK_DATA[pa.NC_PA_CLE_OUTPUT].apply(lambda x: "_".join(x), axis=1)
    AG_STOCK_DATA = AG_STOCK_DATA.sort_values([pa.NC_PA_BILAN, pa.NC_PA_CLE])
    AG_STOCK_DATA[pa.NC_PA_INDEX] = ["ST" + str(i) for i in range(1,len(AG_STOCK_DATA)+1)]

    return AG_STOCK_DATA


def process_starting_points(STOCK_DATA):
    # Valeur en 0 définie par TEM pour les données FERMAT
    for ind in ["TEM_M0", "TEF_M0", "LEM_M0", "LEF_M0", "DEM_M0"]:
        if up.current_etab in gp.NON_RZO_ETABS:
            if ind in STOCK_DATA:
                STOCK_DATA[ind] = STOCK_DATA["LEF_M0"]
        else:
            if ind in STOCK_DATA:
                STOCK_DATA[ind] = STOCK_DATA["TEM_M0"]

    STOCK_DATA["LMN_M0"] = 0
    STOCK_DATA["LMN EVE_M0"] = 0

    return STOCK_DATA


def correct_fx_rates(STOCK_DATA):
    if up.current_etab == "BPCE":
        do_fx = "FX-OLD" in up.bcpe_files_name and "FX-NEW" in up.bcpe_files_name
        if do_fx:
            files_taux_changes = [up.bcpe_files_name["FX-NEW"], up.bcpe_files_name["FX-OLD"]]
            STOCK_DATA = apply_fx_rate(STOCK_DATA, files_taux_changes)
    return STOCK_DATA


def compute_LEM_and_TEM(STOCK_DATA):
    no_mean_input = ("LEM_M1" not in STOCK_DATA.columns) & ("TEM_M1" not in STOCK_DATA.columns)
    if (no_mean_input):
        logger.info("Calcul des indicateurs mensuels moyens LEM et TEM")
        for prefix in ["LE", "TE"]:
            EM_cols = [prefix + "M_M" + str(x) for x in range(0, 121)]
            EF_cols = [prefix + "F_M" + str(x) for x in range(0, 121)]

            # Point de départ (pour le scope MNI)
            filter_scope = (STOCK_DATA[pa.NC_PA_SCOPE] != gp.LIQ)
            STOCK_DATA = pd.concat([STOCK_DATA, pd.DataFrame(0.0, index=STOCK_DATA.index, columns=EM_cols[0:1])], axis=1)
            STOCK_DATA.loc[filter_scope, EM_cols[0]] = STOCK_DATA.loc[filter_scope, EF_cols[0]]

            # EM calculé comme la moyenne des EF des mois courants et suivants
            STOCK_DATA = pd.concat([STOCK_DATA, pd.DataFrame(index=STOCK_DATA.index, columns=EM_cols[1:])], axis=1)
            STOCK_DATA[EM_cols[1:]] = (STOCK_DATA.loc[:, EF_cols[1:]].values + STOCK_DATA.loc[:,
                                                                               EF_cols[:-1]]).values / 2

            # EM calculé à partir du mois 120 comme l'année courante et suivante'
            EM_cols = [prefix + "M_M" + str(x) for x in range(132, 241, 12)]
            EF_cols = [prefix + "F_M" + str(x) for x in range(120, 241, 12)]
            STOCK_DATA = pd.concat([STOCK_DATA, pd.DataFrame(index=STOCK_DATA.index, columns=EM_cols)], axis=1)
            STOCK_DATA[EM_cols] = (STOCK_DATA.loc[:, EF_cols[1:]].values + STOCK_DATA.loc[:, EF_cols[:-1]]).values / 2

    return STOCK_DATA


def apply_fx_rate(stock_data, taux_changes):
    logger.info("Application de taux de change sur le STOCK BPCE")
    change = []
    for k in range(0, taux_changes.shape[0]):
        logger.info("Lecture de : " + taux_changes[k].split("\\")[-1])
        ch = pd.read_csv(taux_changes[k], delimiter="\t", engine='python')  # Lecture du fichier
        ch['exch_rate'] = ch['exch_rate'].str.replace(",", ".").apply(float)
        change.append(ch)

    change_coef = change[1].merge(change[0], on="ccy_code", suffixes=["_new", "_init"])
    change_coef["ratio_change"] = change_coef["exch_rate_new"] / change_coef["exch_rate_init"]

    stock_data = mp.map_data(stock_data, change_coef, keys_data=["ccy_code"], keys_mapping=["ccy_code"],
                             cols_mapp=["ratio_change"], no_map_value=-1,
                             name_mapping="STOCK DATA vs. FICHIER TAUX DE CHANGE ", option="")

    for pref in ["LEF_M", "TEF_M", "LMN_M", "LMN EVE_M", "LEM_M", "TEM_M"]:  # On applique les nouveaux taux de change
        new_col = [pref + str(x) for x in range(0, 121)]
        new_col = new_col + [pref + str(x) for x in range(132, 241, 12)]
        if pref in ["LMN_M", "LMN EVE_M", "LEM_M"]:
            new_col = new_col[1:]
        stock_data.loc[:, new_col] = stock_data.loc[:, new_col].multiply(stock_data.loc[:, "ratio_change"], axis=0)

    return stock_data


def map_with_ALM_mappings(STOCK_DATA):
    logger.info("MAPPING DES DONNEES DU STOCK avec les mappings PASS ALM")

    if up.current_etab in gp.NTX_FORMAT:
        logger.info("   MAPPING DES DONNEES DU STOCK avec les mappings ALM spécifiques à NATIXIS/SEF")
        STOCK_DATA = mp.map_data_with_ntx_mappings(STOCK_DATA)

    logger.info("   MAPPING DES DONNEES DU STOCK avec les mappings ALM généraux")
    STOCK_DATA = mp.map_data_with_general_mappings(STOCK_DATA)

    logger.info("   MAPPING DES LIGNES INTRA-GROUPE DU STOCK")
    STOCK_DATA = mp.map_intra_groupes(STOCK_DATA)

    if up.current_etab == "BPCE":
        STOCK_DATA = mp.map_data_with_bpce_mappings(STOCK_DATA)

    STOCK_DATA = map_lcr_nsfr_data(STOCK_DATA)

    logger.info("   MAPPING DES DONNEES DU STOCK avec les mappings LIQUIDITE")
    STOCK_DATA = mp.mapping_consolidation_liquidite(STOCK_DATA)

    return STOCK_DATA


def map_lcr_nsfr_data(STOCK_DATA):
    global lcr_nsfr_data

    if up.map_lcr_nsfr:
        logger.info("   MAPPING DES DONNEES DU STOCK avec RAY")
        lcr_nsfr.parse_ray_file(up.lcr_nsfr_file)
        STOCK_DATA = lcr_nsfr.map_lcr_tiers_and_share(STOCK_DATA)

    else:
        logger.info("   PAS DE MAPPING RAY")

    return STOCK_DATA


def prepare_output_STOCK(stock):

    cols = [col for col in pa.NC_PA_COL_SORTIE_QUAL if col not in stock.columns.tolist()]
    final_stock = pd.concat([stock, pd.DataFrame([["-"] * len(cols)], \
                                                   index=stock.index, \
                                                   columns=cols)], axis=1)

    final_stock["BOOK"] = "'" + final_stock["BOOK"]

    return final_stock[pa.NC_PA_COL_SORTIE_QUAL + pa.NC_PA_COL_SORTIE_NUM_ST + [pa.NC_PA_isECH]]
