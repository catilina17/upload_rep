import modules.alim.parameters.user_parameters as up
from mappings import general_mappings as gma
import logging
from services.rate_services import rates_calculator as tx_c
logger = logging.getLogger(__name__)
class RatesManager():
    @staticmethod
    def get_sc_df(scenario_name, rate_file_path, liq_file_path):
        logger.info("     Construction du scénario de taux %s" % scenario_name)
        curves_referential = gma.mapping_taux["REF_TX"]
        curves_to_interpolate = gma.mapping_taux["CURVES_TO_INTERPOLATE"]
        curves_to_calculate = gma.mapping_taux["CALCULATED_CURVES"]
        auxiliary_curves_to_calculate = gma.mapping_taux["AUXILIARY_CALCULATED_CURVES"]
        mapping_rate_code_curve = gma.mapping_taux["RATE_CODE-CURVE"]

        sc_tx_df = tx_c.get_scenario_curves(curves_referential, rate_file_path, liq_file_path,
                                            curves_to_interpolate,
                                            curves_to_calculate, auxiliary_curves_to_calculate, scenario_name,
                                            mapping_rate_code_curve)

        return sc_tx_df