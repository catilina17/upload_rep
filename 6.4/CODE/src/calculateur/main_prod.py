import dateutil
import logging
from calculateur.cls_simulator import Simulator
from .simul_params import model_params as mod
logger = logging.getLogger(__name__)

version_source = "tab"

def launch_prod_forward():
    horizon = 300
    dar = dateutil.parser.parse(str("30/09/2024")).replace(tzinfo=None)
    main_folder =  r"C:\Users\HOSSAYNE\Documents\BPCE_ARCHIVES\RESULTATS\SIMUL_CALCULATEUR"
    output_folder = main_folder + r"\SIMULS"
    sources_folder = (r"C:\Users\HOSSAYNE\Documents\BPCE_ARCHIVES\SOURCES\SOURCES_RCO_2022_v3"
                         r" -360 - ME\CALCULATEUR_SOURCES\2024-09_6.3")
    products_list = mod.models_ech_st + mod.models_nmd_st
    etabs_list = ["BPACA"]
    rate_scenario_list = ["@FORWARD"]
    is_tci = False
    name_run = "ST AVEC RA & RN"

    sim = Simulator(sources_folder, etabs_list, horizon, dar, products_list, rate_scenario_list,
                     exit_indicators_type=["GPLIQ", "GPTX"], is_tci=is_tci,
                     agregation_level="AG", cr_immo_cal_mode="quick", batch_sizes={}, output_folder=output_folder,
                     name_run=name_run, interpolate_curves=True, rco_like_report=True, level_logger=logging.INFO)

    sim.launch_simulator()