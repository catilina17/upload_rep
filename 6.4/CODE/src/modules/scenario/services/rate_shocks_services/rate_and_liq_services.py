import logging
from utils import excel_utils as ex
from modules.scenario.referentials.general_parameters import *
from modules.scenario.rate_services import tx_referential as tx_ref, tx_controller_main as tx_main
from modules.scenario.services import referential_file_service as ref_services
from modules.scenario.services.zero_coupon_services.bootstrap_service import process_bootstrap
from modules.scenario.services.rate_shocks_services.taux_files_service import get_temp_tx_files_path
from modules.scenario.parameters import user_parameters as up
from utils import general_utils as ut
from params import version_params as vp

logger = logging.getLogger(__name__)


def create_tx_scenarios(mapping_wb, scenario_list, tx_chocs_list, all_etabs):
    logger.info('    *********** Debut du calcul des scénarios utilisateurs ********** ')
    create_shocked_tx_scenarios_files(mapping_wb, scenario_list, tx_chocs_list)

    logger.info('    *********** Importation des scénarios de référence de la MNI du STOCK ********** ')
    process_stock_tx_sc_ref_creation(mapping_wb, all_etabs)

    logger.info('    *********** Récupération des TCI LIQ du STOCK ********** ')
    tci_liq_nmd_df = get_tci_nmd_rates()

    logger.info('    *********** ********** ')
    return tci_liq_nmd_df


def create_shocked_tx_scenarios_files(mapping_wb, scenario_list, tx_chocs_list):
    for shocked_sc in scenario_list['Shocked TX'].unique():
        try:
            row = scenario_list[scenario_list['Shocked TX'] == shocked_sc].iloc[0]
            df = process_scenario_rate_shocks(mapping_wb, row['CHOC TAUX'], tx_chocs_list,
                                              row[RN_SC_TAUX_USER])
            process_bootstrap(df, row['CHOC TAUX'], row[RN_SC_TAUX_USER], row["NOM SCENARIO ORIG"])
        except ValueError as e:
            logger.error(e)
        except Exception as e:
            logger.error(e, exc_info=True)


def process_scenario_rate_shocks(mapping_wb, scenario_name, tx_chocs_list, baseline_scenario_name):
    tx_chocs = tx_chocs_list[tx_chocs_list[tx_ref.CN_NOM_DE_SCENARIO] == scenario_name]
    curves_referential = ref_services.get_pass_alm_referential(mapping_wb)
    curves_to_interpolate = ref_services.get_curves_to_interpolate(mapping_wb)
    curves_to_calculate = ref_services.get_calc_curves(mapping_wb)
    auxiliary_curves_to_calculate = ref_services.get_filtering_tenor_curves(mapping_wb)
    mapping_rate_code_curve = ref_services.get_rate_code_curve_mapping(mapping_wb)

    ut.check_version_templates(up.tx_curves_path, path=up.tx_curves_path, version=vp.version_rate, open=True)
    ut.check_version_templates(up.liq_curves_path, path=up.liq_curves_path, version=vp.version_rate, open=True)

    if not tx_chocs.empty and scenario_name != tx_ref.SC_TX_BASE_LINE:
        logger.info('    Scénario de taux: @TX_{}'.format(scenario_name))
        logger.info(
            '      Calcul des courbes de taux choqués du scénario:  %s à partir du scénario baseline: %s'
            % (scenario_name, baseline_scenario_name))
        scenario_curves_df = tx_main.get_scenario_curves(curves_referential, up.tx_curves_path, up.liq_curves_path,
                                                 curves_to_interpolate,
                                                 curves_to_calculate, auxiliary_curves_to_calculate,
                                                 baseline_scenario_name,
                                                 mapping_rate_code_curve, tx_chocs, scenario_name, )
        logger.info('      Fin de calcul des courbes de taux choqués du scénario: {}'.format(scenario_name))
    else:
        logger.info('    Scenario de taux: @TX_{}'.format(tx_ref.SC_TX_BASE_LINE))
        logger.info('      Calcul des courbes de taux du scénario:  %s' % baseline_scenario_name)
        scenario_curves_df = tx_main.get_scenario_curves(curves_referential, up.tx_curves_path,
                                                 up.liq_curves_path, curves_to_interpolate,
                                                 curves_to_calculate,
                                                 auxiliary_curves_to_calculate, baseline_scenario_name, mapping_rate_code_curve)
        logger.info('      Fin de calcul des courbes de taux du scénario: {}'.format(baseline_scenario_name))

    file_path = get_temp_tx_files_path('{}_{}'.format(baseline_scenario_name, scenario_name), TEMP_DIR_TX_LIQ)

    scenario_curves_df.to_csv(file_path, index=False)

    return scenario_curves_df


def process_stock_tx_sc_ref_creation(mapping_wb, all_etabs):
    default_value = up.st_refs[up.st_refs["ENTITE"] == "DEFAULT"]['SC REF STOCK'].iloc[0]
    st_refs = up.st_refs[up.st_refs["ENTITE"].isin(all_etabs)]['SC REF STOCK'].values.tolist()
    st_refs = list(set(st_refs + [default_value])) if len(st_refs) != len(all_etabs) else list(set(st_refs))

    curves_referential = ref_services.get_pass_alm_referential(mapping_wb)
    curves_to_interpolate = ref_services.get_curves_to_interpolate(mapping_wb)
    curves_to_calculate = ref_services.get_calc_curves(mapping_wb)
    auxiliary_curves_to_calculate = ref_services.get_filtering_tenor_curves(mapping_wb)
    mapping_rate_code_curve = ref_services.get_rate_code_curve_mapping(mapping_wb)
    for st_ref in st_refs:
        try:
            reference_tx_curves_df = tx_main.get_scenario_curves(curves_referential, up.tx_curves_path,
                                                                 up.liq_curves_path, curves_to_interpolate,
                                                                 curves_to_calculate,
                                                                 auxiliary_curves_to_calculate, st_ref, mapping_rate_code_curve)
            file_path = get_temp_tx_files_path(st_ref, TEMP_DIR_STOCK)
            reference_tx_curves_df.to_csv(file_path, index=False)
        except ValueError as e:
            logger.error(e)
        except Exception as e:
            logger.error(e, exc_info=True)


def get_tci_nmd_rates():
    ut.check_version_templates(up.tci_curves_path, path=up.tci_curves_path, version=vp.version_rate_tci, open=True)
    input_taux_wb = ex.try_close_open(up.tci_curves_path, read_only=True)
    tci_df = ex.get_dataframe_from_range(input_taux_wb, RN_TCI)
    input_taux_wb.Close(SaveChanges=False)
    return tci_df


