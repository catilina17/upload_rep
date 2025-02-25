import numpy as np
import pandas as pd
import logging
from ..tx_referential import *

logger = logging.getLogger(__name__)


def generate_progressive_shock_array(scenario_shocks, shock_matrix):
    try:
        merge = shock_matrix.merge(scenario_shocks, left_on=CN_CODE_PASSALM, right_on=CN_CODE_PASSALM)
        merge = merge[(merge[CN_START] <= merge[CN_TERME]) & (merge[CN_TERME] <= merge[CN_END])]
        merge.loc[:, 'Floor'] = np.floor_divide(merge.loc[:, CN_TERME] - merge.loc[:, CN_START], merge.loc[:, CN_STEP])
        merge.loc[:, CN_TERME_SHOCK] = merge.loc[:, CN_SHOCK] * merge.loc[:, 'Floor'] + merge.loc[:, CN_SHOCK]
        return merge[['Ordre', CN_CODE_PASSALM, CN_TERME, CN_TERME_SHOCK]]
    except Exception as e:
        logger.error(e)


def generate_constant_shock_array(scenario_shocks, shock_matrix):
    try:
        merge = shock_matrix.merge(scenario_shocks, left_on=CN_CODE_PASSALM, right_on=CN_CODE_PASSALM)
        merge = merge[(merge[CN_START] <= merge[CN_TERME]) & (merge[CN_TERME] <= merge[CN_END])]
        merge.loc[:, CN_TERME_SHOCK] = merge.loc[:, 'Shock']
        return merge[['Ordre',CN_CODE_PASSALM,  CN_TERME, CN_TERME_SHOCK]]
    except Exception as e:
        logger.error(e)
        logger.error(e, exc_info=True)


def get_parllel_shocks_definitions(scenario_df, rate_matrix):

    merged_df = pd.DataFrame()

    all_currencies = rate_matrix[CN_CODE_DEVISE].dropna().unique()
    scenario_df = replace_once_value_by_multiple_values(scenario_df, CN_DEVISE, SELECT_ALL_KEYWORD, all_currencies)

    all_tx_curves = rate_matrix[rate_matrix[CN_TYPE] == ROW_TYPE_RATE][CN_CODE_COURBE].dropna().unique()
    scenario_df = replace_once_value_by_multiple_values(scenario_df, CN_COURBE, SELECT_ALL_TX_KEYWORD, all_tx_curves)

    all_liq_curves = rate_matrix[rate_matrix[CN_TYPE] == ROW_TYPE_LIQ][CN_CODE_COURBE].dropna().unique()
    scenario_df = replace_once_value_by_multiple_values(scenario_df, CN_COURBE, SELECT_ALL_LIQ_KEYWORD, all_liq_curves)

    all_maturities = rate_matrix[CN_MATURITY].dropna().unique()
    scenario_df = replace_once_value_by_multiple_values(scenario_df, CN_MATURITY, SELECT_ALL_KEYWORD, all_maturities)

    merged_df = pd.concat([merged_df, scenario_df.loc[scenario_df[scenario_df[CN_MATURITY] != SELECT_ALL_KEYWORD].index, :].merge(rate_matrix,
                                                                                                            left_on=[CN_DEVISE, CN_COURBE, CN_MATURITY],
                                                                                                            right_on=[CN_CODE_DEVISE, CN_CODE_COURBE, CN_MATURITY])])

    merged_df = merged_df.loc[:, [CN_TYPE_SCENARIO, CN_CODE_PASSALM, CN_DATE_DEBUT, CN_DATE_FIN_FR, CN_CHOC, CN_PAS_FR, CN_CODE_DEVISE, CN_CODE_COURBE]]
    merged_df.rename(columns={CN_TYPE_SCENARIO: CN_TYPE, CN_DATE_DEBUT: CN_START, CN_DATE_FIN_FR: CN_END, CN_CHOC: CN_SHOCK, CN_PAS_FR: CN_STEP}, inplace= True)
    return merged_df


def replace_once_value_by_multiple_values(scenario_df, column_name, value, replacement_values, ):
    transco = pd.DataFrame({'source': [value for i in replacement_values], 'target': replacement_values})
    scenario_df = scenario_df.merge(transco, left_on=[column_name], right_on=['source'], how='left')
    scenario_df['target'] = scenario_df['target'].fillna(scenario_df[column_name])
    scenario_df[column_name] = scenario_df['target']
    scenario_df.drop(['source', 'target'], inplace=True, axis=1)
    return scenario_df