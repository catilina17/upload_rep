import logging
import pandas as pd
from .params import tx_referential as gp

logger = logging.getLogger(__name__)
list_courbes_all = []

def get_rate_curve_without_info_ref(tx_file_path, liq_file_path, tx_referential_df, scenario_name):
    rate_df = read_input_tx_rate_curve_from_wb(tx_file_path, scenario_name, gp.RN_INPUT_TX)
    spread_liq_df = read_input_tx_rate_curve_from_wb(liq_file_path, scenario_name, gp.RN_INPUT_LIQ)
    rate_df = get_rate_curve(rate_df, tx_referential_df, spread_liq_df)
    return rate_df

def get_rate_curve(rate_df, tx_referential_df, spread_liq_df):
    global list_courbes_all
    logger.debug('Création des courbes de taux')
    num_cols = [i for i in range(gp.NB_PROJ_TAUX + 1)]
    rate_df.columns = rate_df.columns.to_list()[:1] + num_cols
    rate_curve = tx_referential_df.merge(rate_df, left_on=gp.CN_INDEX_FEERIE, right_on=gp.CN_INDEX_FEERIE, how='inner')
    cond_prob_tx = ((~tx_referential_df[gp.CN_INDEX_FEERIE].isin(rate_df[gp.CN_INDEX_FEERIE])) & (tx_referential_df["TYPE"] == "TAUX") )
    if cond_prob_tx.any():
        list_courbes = tx_referential_df[cond_prob_tx][gp.CN_INDEX_FEERIE].unique().tolist()
        list_courbes = [x for x in list_courbes if x not in list_courbes_all]
        if len(list_courbes) > 0:
            logger.warning("             Il y a des index feerie dans le référentiel qui ne sont pas présentes dans le RATE INPUT TAUX :")
            logger.warning("                %s" % list_courbes)
            list_courbes_all = list_courbes_all + list_courbes
    spread_liq_df.columns = spread_liq_df.columns.to_list()[:1] + num_cols
    liq_curve = tx_referential_df.merge(spread_liq_df, left_on=gp.CN_INDEX_FEERIE, right_on=gp.CN_INDEX_FEERIE,
                                        how='inner')
    cond_prob_liq = ((~tx_referential_df[gp.CN_INDEX_FEERIE].isin(spread_liq_df[gp.CN_INDEX_FEERIE])) & (
                tx_referential_df["TYPE"] == "LIQ"))
    if cond_prob_liq.any():
        list_courbes = tx_referential_df[cond_prob_liq][gp.CN_INDEX_FEERIE].unique().tolist()
        list_courbes = [x for x in list_courbes if x not in list_courbes_all]
        if len(list_courbes) > 0:
            logger.warning("             Il y a des index feerie dans le référentiel qui ne sont pas présentes dans le RATE INPUT LIQ :")
            logger.warning("                %s" % list_courbes)
            list_courbes_all = list_courbes_all + list_courbes

    complete_matrix = pd.concat([rate_curve, liq_curve])
    complete_matrix = complete_matrix[[gp.CN_CODE_PASSALM] + num_cols].copy()
    complete_matrix.set_index(gp.CN_CODE_PASSALM, inplace=True)
    return complete_matrix

def read_input_tx_rate_curve_from_wb(tx_file_path, scenario_name, range_name):
    rate_df = pd.read_csv(tx_file_path, decimal=",", sep=";")
    filtred_rate_df = rate_df.loc[
        rate_df['SCENARIO'].str.casefold().str.strip() == scenario_name.casefold().strip()].copy()
    filtred_rate_df.drop('SCENARIO', axis=1, inplace=True)
    filtred_rate_df.reset_index(inplace=True, drop=True)
    duplicated_index_lis = list(filtred_rate_df[filtred_rate_df.iloc[:, 0].duplicated()].iloc[:, 0])
    if len(duplicated_index_lis) > 0:
        if "liq" in range_name:
            raise IndexError(
                'Erreur dans le fichier de liquidité les indices suivant sont dupliqués:  {}'.format(
                    duplicated_index_lis))
        else:
            raise IndexError(
                'Erreur dans le fichier de taux les indices suivant sont dupliqués:  {}'.format(duplicated_index_lis))
    if filtred_rate_df.empty:
        if "liq" in range_name:
            raise ValueError('      Le fichier input liquidité ne contient pas le scénario {}'.format(scenario_name))
        else:
            raise ValueError('      Le fichier input taux ne contient pas le scénario {}'.format(scenario_name))
    return filtred_rate_df
