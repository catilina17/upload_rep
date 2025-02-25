import os.path
import pandas as pd
import numpy as np
import logging
from modules.scenario.referentials.general_parameters import *
from modules.scenario.rate_services import tx_referential as tx_ref
from modules.scenario.services.referential_file_service import get_zc_data
from modules.scenario.rate_services import maturities_referential as tm
from modules.scenario.controller.bootstrap_controller.bootstrap_controller import bootstrap, get_forward_days_and_dates
from modules.scenario.services.rate_shocks_services.taux_files_service import get_temp_tx_files_path
from modules.scenario.parameters import user_parameters as up

logger = logging.getLogger(__name__)

non_dispo_curves_all = []

def process_bootstrap(scenario_tx_df, scenario_name, scenario_baseline_name, scen_orig_name):
    global non_dispo_curves_all
    logger.info('      Calcul des Zéros Coupons')
    days_fwd, forward_dates = get_forward_days_and_dates()
    result = pd.DataFrame()
    dispo_curves = scenario_tx_df[scenario_tx_df[tx_ref.CN_CODE_COURBE].str.strip().isin(up.curves_to_bootsrapp[tx_ref.CN_CODE_COURBE].values.tolist())][tx_ref.CN_CODE_COURBE].unique().tolist()
    non_dispo_curves =  [x for x in up.curves_to_bootsrapp[tx_ref.CN_CODE_COURBE].values.tolist() if x not in dispo_curves]
    if len([x for x in non_dispo_curves if x not in non_dispo_curves_all])>0:
        logger.warning("         Les zéros-coupons ne pourront pas être calculés car"
                       " les courbes suivantes sont absentes: %s" % non_dispo_curves)
        non_dispo_curves_all = non_dispo_curves_all + non_dispo_curves

    for dev, curve in zip(up.curves_to_bootsrapp[tx_ref.CN_DEVISE], up.curves_to_bootsrapp[tx_ref.CN_CODE_COURBE]):
        if curve in dispo_curves:
            try:
                df = create_currency_bootstrap_df(dev, curve, days_fwd, forward_dates, scenario_tx_df)
                df.insert(0, 'Courbe', curve)
                result = pd.concat([result,df])
            except Exception as e:
                logger.error('Bootstrap {} Erreur'.format(curve))
                logger.error(e, exc_info=True)

    file_path = get_temp_tx_files_path('{}_{}'.format(scenario_baseline_name, scenario_name), TEMP_DIR_BOOTSRAP)
    result = replace_zc_for_eve_simul(result, scen_orig_name, up.curves_to_bootsrapp)
    result.to_csv(file_path, index=False)


def replace_zc_for_eve_simul(boostrap_data, scenario_name, CURVES_TO_BOOTSTRAP):
    filter_curves = boostrap_data["Courbe"].isin(CURVES_TO_BOOTSTRAP[CURVES_TO_BOOTSTRAP["ZC FICHIER"]=="OUI"][tx_ref.CN_CODE_COURBE].values.tolist())
    bootstrap_replace = boostrap_data[filter_curves].copy()
    if up.zc_file_path == "" or not os.path.exists(up.zc_file_path):
        logger.warning("         Le fichier ZC n'est pas disponible")
        logger.warning("         Le processus se poursuit avec les ZC générés par PASS-ALM")
        return boostrap_data
    zc_data = get_zc_data(up.zc_file_path)
    cols_sc = [x for x in zc_data.columns if scenario_name == "_".join(x.split("_")[:-2])]
    if cols_sc == []:
        msg_err = "         Le scénario %s n'est pas disponible dans le fichier ZC" % scenario_name
        logger.warning(msg_err)
        logger.warning("         Le processus se poursuit avec les Zéro Coupons générés par PASS-ALM")

    zc_data = zc_data[cols_sc].copy()
    all_months = list(set([int(x.split("_")[-2][1:]) for x in zc_data.columns]))
    for m in all_months:
        mois = "M" + str(m)
        zc_data_m = zc_data[[x for x in zc_data.columns if mois == x.split("_")[-2]]].copy()
        if zc_data_m.shape[1] > 2:
            logger.warning("         Le fichier ZC contient plusieurs colonnes pour le mois %s et le scénario %s" % (
            mois, scenario_name))
            logger.warning("         Les ZC pour le mois % ne seront pas remplacés" % mois)
        else:
            bootstrap_replace.iloc[:, 2 * m + 2] = zc_data_m.iloc[:, 0].values
            bootstrap_replace.iloc[:, 2 * m + 3] = zc_data_m.iloc[:, 1].values

    boostrap_data[filter_curves] = bootstrap_replace
    return boostrap_data


def create_currency_bootstrap_df(dev, curve, days_fwd, forward_dates, scenario_tx_df):
    days_fwd, zc = bootstrap(dev, curve, days_fwd, forward_dates, scenario_tx_df)
    row_a, col_a = np.shape(days_fwd)
    row_b, col_b = np.shape(zc)
    result = np.ravel([days_fwd.T, zc.T], order="F").reshape(row_a, col_a + col_b)
    currency_df = pd.DataFrame(result, index=tm.ZC_MATURITIES_NAMES)
    currency_df.reset_index(inplace=True)
    return currency_df
