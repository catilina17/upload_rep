import numpy as np
import logging
import pandas as pd
from utils import excel_utils as excel_helper
from modules.scenario.referentials.general_parameters import *
from modules.scenario.rate_services import tx_referential as tx_ref

logger = logging.getLogger(__name__)

OUTPUT_COLS_CURVES = [tx_ref.CN_TYPE2_COURBE, tx_ref.CN_CODE_DEVISE, tx_ref.CN_CODE_COURBE, tx_ref.CN_MATURITY,
                      tx_ref.CN_CODE_PASSALM]
OUTPUT_COLS_CURVES_NUM = ["M" + str(i) for i in range(tx_ref.NB_PROJ_TAUX + 1)]


def export_tx_data_to(fermat_df, sc_df, bootstrap_df, tci_liq_nmd_df, output_workbook):
    try:
        logger.info('    Ecriture des courbes liq et taux dans le fichier scénario ')
        sc_tx_df, sc_liq_df, fermat_df = format_data(sc_df, fermat_df)
        # sc_tx_df, fermat_df = add_tvzeros_row(sc_tx_df, fermat_df)
        print_in_output_file(sc_tx_df, sc_liq_df, fermat_df, bootstrap_df, tci_liq_nmd_df, output_workbook)
    except Exception as e:
        logger.error(e, exc_info=True)


def print_in_output_file(sc_tx_df, sc_liq_df, fermat_df, bootstrap_df, tci_liq_nmd_df, wb):
    excel_helper.export_df_to_xl_with_range_name(wb, sc_tx_df, RN_SC_TX, protected=True, header=False)
    excel_helper.export_df_to_xl_with_range_name(wb, sc_liq_df, RN_SC_LIQ, protected=True, header=False)
    excel_helper.export_df_to_xl_with_range_name(wb, fermat_df, RN_SC_RCO, protected=True, header=False)
    excel_helper.export_df_to_xl_with_range_name(wb, bootstrap_df, RN_BOOTSTRAP, protected=True, header=False)
    excel_helper.export_df_to_xl_with_range_name(wb, tci_liq_nmd_df, RN_TCI, protected=True, header=False)


def format_data(sc_df, fermat_df):
    fermat_df = fermat_df[(fermat_df[tx_ref.CN_TYPE] == "TAUX")]
    fermat_df = fermat_df[OUTPUT_COLS_CURVES + OUTPUT_COLS_CURVES_NUM].copy()
    fermat_df = apply_output_format(fermat_df, ratio=100)
    fermat_df = rank_curves_tx(fermat_df)
    sc_tx_df = sc_df[(sc_df[tx_ref.CN_TYPE] == "TAUX")].copy()
    sc_liq_df = sc_df[(sc_df[tx_ref.CN_TYPE] == "LIQ")].copy()
    sc_tx_df = sc_tx_df[OUTPUT_COLS_CURVES + OUTPUT_COLS_CURVES_NUM].copy()
    sc_liq_df = sc_liq_df[OUTPUT_COLS_CURVES + OUTPUT_COLS_CURVES_NUM].copy()
    sc_tx_df = apply_output_format(sc_tx_df, ratio=100)
    sc_liq_df = apply_output_format(sc_liq_df, ratio=100)
    sc_tx_df = rank_curves_tx(sc_tx_df)
    sc_liq_df = rank_curves_liq(sc_liq_df)
    return sc_tx_df, sc_liq_df, fermat_df


def get_rank_df(dict, cols):
    rank_df = pd.DataFrame.from_dict(dict, orient='index').reset_index()
    rank_df.columns = cols
    rank_df.set_index(cols[1], inplace=True)
    return rank_df


def rank_curves_tx(tx_df):
    rank_tx = get_rank_df(RANK_CURVES_TX, ["RANK", "TYPE"])
    rank_dev = get_rank_df(RANK_CURVES_DEV, ["RANK2", "DEVISE"])
    tx_df = (tx_df.join(rank_tx, on=[tx_ref.CN_TYPE2_COURBE])).join(rank_dev, on=[tx_ref.CN_DEVISE])
    tx_df.sort_values(["RANK", "RANK2"], inplace=True)
    return tx_df.drop(["RANK", "RANK2"], axis=1)


def rank_curves_liq(liq_df):
    rank_liq = get_rank_df(RANK_CURVES_LIQ, ["RANK", "TYPE"])
    rank_dev = get_rank_df(RANK_CURVES_DEV, ["RANK2", "DEVISE"])
    liq_df = (liq_df.join(rank_liq, on=[tx_ref.CN_TYPE2_COURBE])).join(rank_dev, on=[tx_ref.CN_DEVISE])
    liq_df.sort_values(["RANK", "RANK2"], inplace=True)
    return liq_df.drop(["RANK", "RANK2"], axis=1)


def add_empty_row_index_row_index(data_df, empty_row, first_index_list):
    data_df.reset_index(drop=True, inplace=True)
    for code in first_index_list:
        index = data_df.index[data_df[tx_ref.CN_CODE_PASSALM] == code][0]
        data_df = pd.concat([data_df.iloc[:index, :], empty_row, data_df.iloc[index:, :]]).reset_index(drop=True)
    return data_df


def apply_output_format(data_df, ratio=1):
    data_df = data_df.replace(transco_INF_MATURITIES)
    data_df = data_df.set_index(OUTPUT_COLS_CURVES)
    data_df = data_df.loc[:, 'M0':]
    data_df = data_df / ratio
    data_df = data_df.ffill(axis='columns')
    data_df.reset_index(inplace=True)
    data_df = data_df.rename(columns={'M0': 'DAR'})
    return data_df


transco_INF_MATURITIES = {tx_ref.CN_CODE_PASSALM: {
    'INF12M': 'INF 01',
    'INF2Y': 'INF 02',
    'INF3Y': 'INF 03',
    'INF4Y': 'INF 04',
    'INF5Y': 'INF 05',
    'INF6Y': 'INF 06',
    'INF7Y': 'INF 07',
    'INF8Y': 'INF 08',
    'INF9Y': 'INF 09',
    'INF10Y': 'INF 10',
    'INF11Y': 'INF 11',
    'INF12Y': 'INF 12',
    'INF13Y': 'INF 13',
    'INF14Y': 'INF 14',
    'INF15Y': 'INF 15',
    'INF16Y': 'INF 16',
    'INF17Y': 'INF 17',
    'INF18Y': 'INF 18',
    'INF19Y': 'INF 19',
    'INF20Y': 'INF 20',
    'INF21Y': 'INF 21',
    'INF22Y': 'INF 22',
    'INF23Y': 'INF 23',
    'INF24Y': 'INF 24',
    'INF25Y': 'INF 25',
    'INF26Y': 'INF 26',
    'INF27Y': 'INF 27',
    'INF28Y': 'INF 28',
    'INF29Y': 'INF 29',
    'INF30Y': 'INF 30'
}
}
