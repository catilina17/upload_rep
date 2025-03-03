from .shocks import shocks_main as sc
from .calculated_curves import calculated_curves_controller as cc
from . import refactor_maturities as rlc
from . import import_and_format as ifc
import logging
import traceback

logger = logging.getLogger(__name__)
def get_scenario_curves(tx_referential_df, tx_curves_file_path, liq_curves_file_path,
                        curves_to_interpolate, curves_to_calculate, auxiliary_curves_calc, scenario_name, mapping_rate_code_curve,
                        scenario_shocks_list = [], shocked_scenario_name = ""):
    try:
        rate_matrix = ifc.get_rate_curve_without_info_ref(tx_curves_file_path, liq_curves_file_path,
                                                                   tx_referential_df, scenario_name)
        if len(scenario_shocks_list)  > 0:
            shocked_rate_curve = sc.compute_shocked_rate_curve(rate_matrix, tx_referential_df, scenario_shocks_list)
        else:
            shocked_rate_curve = rate_matrix.copy()

        final_curves_df = rlc.refactor_curves(tx_referential_df, shocked_rate_curve, curves_to_interpolate)
        final_curves_df = cc.generate_calculated_curves(final_curves_df, curves_to_calculate, auxiliary_curves_calc, mapping_rate_code_curve)
        return final_curves_df

    except Exception as e:
        if len(scenario_shocks_list) > 0:
            logger.error('      Erreur pendant la création du scénario choqué %s' % shocked_scenario_name)
        else:
            logger.error('      Erreur pendant la création du scénario %s' % scenario_name)
        logger.error(traceback.format_exc())
        raise e



