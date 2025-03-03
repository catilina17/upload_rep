import numpy as np
from ...rate_services.shocks import non_parallel_shocks as non_parallel_cs
from ...rate_services.shocks import parallel_shocks as parallel_cs
from services.rate_services.params.tx_referential import *
import pandas as pd


def compute_shocked_rate_curve(rate_matrix, referential, scenario_list):
    scenario_shock_list = generate_shocks_list(scenario_list, referential)
    shocked_rate_curve = generate_shocked_rate_curve(rate_matrix, scenario_shock_list)
    return shocked_rate_curve


def generate_shocks_list(shocks_list_df, referential):
    shocks = parallel_cs.get_parllel_shocks_definitions(
        shocks_list_df[~shocks_list_df[CN_TYPE_SCENARIO].isin(NON_PARALLELE_CHOC_TYPE)], referential)

    non_parallel_shocks = shocks_list_df[shocks_list_df[CN_TYPE_SCENARIO].isin(NON_PARALLELE_CHOC_TYPE)]
    non_parallel_shocks = non_parallel_shocks[
        [CN_DEVISE, CN_COURBE, CN_SEUIL_PIVOT_1, CN_SEUIL_PIVOT_2, CN_CHOC_SEUIL_1, CN_CHOC_SEUIL_2, CN_DATE_DEBUT,
         CN_DATE_FIN_FR]]

    if not non_parallel_shocks.empty:
        non_p_shocks_df = non_parallel_shocks.apply(
            lambda x: non_parallel_cs.generate_twist_rate_curve_shocks(referential, *x), axis=1)
        shocks = shocks.append(pd.concat(list(non_p_shocks_df)))

    shocks = shocks.astype({CN_START: 'int32', CN_END: 'int32', CN_STEP: 'int32'})
    return shocks


def generate_shocked_rate_curve(rate_matrix, scenario_shock_list):
    shocks_matrix = pd.DataFrame(np.zeros(rate_matrix.shape), index=rate_matrix.index, columns=rate_matrix.columns)
    if len(scenario_shock_list) > 0:
        shocks_matrix = generate_shocks_matrix(shocks_matrix, scenario_shock_list)
    shocked_rate_curve = rate_matrix.add(shocks_matrix, axis=0)
    shocked_rate_curve.update(rate_matrix, overwrite=False)
    return shocked_rate_curve


def generate_shocks_matrix(empty_shock_matrix, shocks_list):
    shocks = pd.DataFrame(columns=[CN_CODE_PASSALM, CN_TERME, CN_TERME_SHOCK])
    empty_shock_matrix.reset_index(inplace=True)
    unpivoted_rate_curve = empty_shock_matrix.melt(id_vars=CN_CODE_PASSALM,
                                                   value_vars=empty_shock_matrix.iloc[:, 1:].columns, var_name=CN_TERME)
    unpivoted_rate_curve.loc[:, CN_TERME] = unpivoted_rate_curve.loc[:, CN_TERME].astype(int)
    shocks_list['Ordre'] = shocks_list.index
    progressive_shocks = parallel_cs.generate_progressive_shock_array(
        shocks_list[shocks_list[CN_TYPE] == PROGRESSIF_SHOCK_TYPE], unpivoted_rate_curve)
    shocks = pd.concat([shocks, progressive_shocks])
    constant_shocks = parallel_cs.generate_constant_shock_array(
        shocks_list[shocks_list[CN_TYPE] == CONSTANT_SHOCK_TYPE], unpivoted_rate_curve)
    if len(shocks) > 0:
        shocks = pd.concat([shocks, constant_shocks])
    else:
        shocks = constant_shocks.copy()
    selected_shocks_by_order = shocks.loc[shocks.groupby([CN_CODE_PASSALM, CN_TERME]).Ordre.idxmax()]
    selected_shocks_by_order = selected_shocks_by_order.drop_duplicates()
    selected_shocks_by_order.fillna(0, inplace=True)
    shocks_matrix = pd.pivot_table(selected_shocks_by_order, values=CN_TERME_SHOCK, index=CN_CODE_PASSALM,
                                   columns=[CN_TERME])
    return shocks_matrix
