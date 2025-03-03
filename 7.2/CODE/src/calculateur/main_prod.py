import dateutil
import logging
from calculateur.cls_simulator import Simulator
from .calc_params import model_params as mod
logger = logging.getLogger(__name__)

version_source = "tab"

def launch_prod_forward(scenario_forward, etab, dar, horizon, output_folder, name_run,
                        rates_file_path, liq_rates_file_path, tci_file_path, model_ech_path, model_nmd_path):
    products_list = mod.models_nmd_st + mod.models_ech_st
    rate_file_list = [rates_file_path] * len(products_list)
    rate_scenario_list = [scenario_forward] * len(products_list)
    model_ech_list = [model_ech_path] * len(products_list)
    model_nmd_pel_list= [model_nmd_path] * len(products_list)
    liq_rates_file_path = [liq_rates_file_path] * len(products_list)
    tci_rates_file_path = [tci_file_path] * len(products_list)
    etabs_list = [etab] *  len(products_list)
    is_tci = True
    sim = Simulator(etabs_list, horizon, dar, products_list, rate_file_list, rate_scenario_list,
                    model_ech_list, model_nmd_pel_list, liq_rates_file_path = liq_rates_file_path,
                    tci_rates_file_path = tci_rates_file_path, exit_indicators_type=["GPLIQ", "GPTX", "TCI"], is_tci=is_tci,
                    agregation_level="AG", cr_immo_cal_mode="quick", batch_sizes={}, output_folder=output_folder,
                    name_run=name_run, interpolate_curves=True, rco_like_report=True, level_logger=logging.INFO,
                    rate_file_formatted=False)

    sim.launch_simulator()