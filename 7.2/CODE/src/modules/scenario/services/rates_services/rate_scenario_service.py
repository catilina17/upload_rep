import logging
from modules.scenario.parameters.general_parameters import *
from services.rate_services import rates_calculator as tx_main
from services.rate_services.params import tx_referential as tx_ref
from modules.scenario.services.zero_coupon_services.zc_service import ZCGenerator
from modules.scenario.services.rates_services.rate_temp_files_saver import get_temp_tx_files_path
from mappings import general_mappings as mp
import pandas as pd

logger = logging.getLogger(__name__)


class Rate_Scenario_Manager():
    def __init__(self, cls_usr):
        self.up = cls_usr

    def create_rate_scenarios(self, scenario_list, tx_chocs_list, all_etabs):
        logger.info('  Calcul des scénarios de taux utilisateurs')
        self.create_user_rate_scenarios_files(scenario_list, tx_chocs_list)

        logger.info('  Importation des scénarios de taux de référence de la MNI du STOCK')
        self.create_sc_ref_stock_rate_scenario(all_etabs)

        logger.info('  Récupération des TCI LIQ du STOCK')
        tci_liq_nmd_df = self.get_tci_nmd_rates()

        return tci_liq_nmd_df

    def create_user_rate_scenarios_files(self, scenario_list, tx_chocs_list):
        for user_sc in scenario_list[self.up.RN_USER_RATE_SC].unique():
            row = scenario_list[scenario_list[self.up.RN_USER_RATE_SC] == user_sc].iloc[0]
            rates_df = self.process_user_rate_scenario(row[self.up.RN_USER_RATE_SHOCK], tx_chocs_list, row[RN_SC_TAUX_USER])

            zc = ZCGenerator(self.up)
            zc.process_zero_coupons(rates_df, row[self.up.RN_USER_RATE_SHOCK], row[RN_SC_TAUX_USER], row[self.up.SCENARIO_NAME_ORIG])

    def process_user_rate_scenario(self, scenario_name, rate_shocks_list, baseline_scenario_name):
        rate_shocks = rate_shocks_list[rate_shocks_list[tx_ref.CN_NOM_DE_SCENARIO] == scenario_name]
        curves_referential = mp.mapping_taux["REF_TX"]
        curves_to_interpolate = mp.mapping_taux["CURVES_TO_INTERPOLATE"]
        curves_to_calculate = mp.mapping_taux["CALCULATED_CURVES"]
        auxiliary_curves_to_calculate = mp.mapping_taux["AUXILIARY_CALCULATED_CURVES"]
        mapping_rate_code_curve = mp.mapping_taux["RATE_CODE-CURVE"]

        # ut.check_version_templates(self.up.tx_curves_path, path=self.up.tx_curves_path, version=vp.version_rate, open=True)
        # ut.check_version_templates(self.up.liq_curves_path, path=self.up.liq_curves_path, version=vp.version_rate, open=True)

        if not rate_shocks.empty and scenario_name != tx_ref.SC_TX_BASE_LINE:
            logger.info('    Scénario de taux: @TX_{}'.format(scenario_name))
            logger.info(
                '      Calcul des courbes de taux choqués du scénario:  %s à partir du scénario baseline: %s'
                % (scenario_name, baseline_scenario_name))
            scenario_curves_df = tx_main.get_scenario_curves(curves_referential, self.up.tx_curves_path,
                                                             self.up.liq_curves_path,
                                                             curves_to_interpolate,
                                                             curves_to_calculate, auxiliary_curves_to_calculate,
                                                             baseline_scenario_name,
                                                             mapping_rate_code_curve, rate_shocks, scenario_name, )
            logger.info('      Fin de calcul des courbes de taux choqués du scénario: {}'.format(scenario_name))
        else:
            logger.info('    Scenario de taux: @TX_{}'.format(tx_ref.SC_TX_BASE_LINE))
            logger.info('      Calcul des courbes de taux du scénario:  %s' % baseline_scenario_name)
            scenario_curves_df = tx_main.get_scenario_curves(curves_referential, self.up.tx_curves_path,
                                                             self.up.liq_curves_path, curves_to_interpolate,
                                                             curves_to_calculate,
                                                             auxiliary_curves_to_calculate, baseline_scenario_name,
                                                             mapping_rate_code_curve)
            logger.info('      Fin de calcul des courbes de taux du scénario: {}'.format(baseline_scenario_name))

        file_path = get_temp_tx_files_path('{}_{}'.format(baseline_scenario_name, scenario_name), TEMP_DIR_TX_LIQ)

        scenario_curves_df.to_csv(file_path, index=False)

        return scenario_curves_df

    def create_sc_ref_stock_rate_scenario(self, all_etabs):
        default_value = self.up.st_refs[self.up.st_refs["ENTITE"] == "DEFAULT"]['SC REF STOCK'].iloc[0]
        st_refs = self.up.st_refs[self.up.st_refs["ENTITE"].isin(all_etabs)]['SC REF STOCK'].values.tolist()
        st_refs = list(set(st_refs + [default_value])) if len(st_refs) != len(all_etabs) else list(set(st_refs))

        curves_referential = mp.mapping_taux["REF_TX"]
        curves_to_interpolate = mp.mapping_taux["CURVES_TO_INTERPOLATE"]
        curves_to_calculate = mp.mapping_taux["CALCULATED_CURVES"]
        auxiliary_curves_to_calculate = mp.mapping_taux["AUXILIARY_CALCULATED_CURVES"]
        mapping_rate_code_curve = mp.mapping_taux["RATE_CODE-CURVE"]
        for st_ref in st_refs:
            try:
                reference_tx_curves_df = tx_main.get_scenario_curves(curves_referential, self.up.tx_curves_path,
                                                                     self.up.liq_curves_path, curves_to_interpolate,
                                                                     curves_to_calculate,
                                                                     auxiliary_curves_to_calculate, st_ref,
                                                                     mapping_rate_code_curve)
                file_path = get_temp_tx_files_path(st_ref, TEMP_DIR_STOCK)
                reference_tx_curves_df.to_csv(file_path, index=False)
            except ValueError as e:
                logger.error(e)
            except Exception as e:
                logger.error(e, exc_info=True)

    def get_tci_nmd_rates(self, ):
        # ut.check_version_templates(self.up.tci_curves_path, path=self.up.tci_curves_path, version=vp.version_rate_tci, open=True)
        tci_df = pd.read_csv(self.up.tci_curves_path, sep=";", decimal=",")
        return tci_df
