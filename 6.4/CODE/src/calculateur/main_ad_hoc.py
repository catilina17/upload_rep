import dateutil
import logging
from calculateur.cls_simulator import Simulator
from .simul_params import model_params as mod
logger = logging.getLogger(__name__)

version_source = "tab"

def launch_calc_ad_hoc():
    horizon = 300
    horizon = min(360, horizon)
    dar = dateutil.parser.parse(str("31/12/2024")).replace(tzinfo=None)
    main_folder =  r"C:\Users\HOSSAYNE\Documents\BPCE_ARCHIVES\RESULTATS\SIMUL_CALCULATEUR"
    output_folder = main_folder + r"\SIMULS"
    sources_folder = (r"C:\Users\HOSSAYNE\Documents\BPCE_ARCHIVES\SOURCES\SOURCES_RCO_2022_v3"
                         r" -360 - ME\CALCULATEUR_SOURCES\2024-12_6.4")
    products_list = ["a-crif"]
    etabs_list = ["PALATINE"]
    rate_scenario_list = ["@FORWARD"]
    benchmark_file = main_folder + r"\BENCHMARKS\MODELS_SYNTHESIS_20240930_BENCHMARK2.xlsb"
    is_tci = False
    fichier_rco = main_folder + r"\RCO_ALIM\%s_RCO_31_12_2024_%s%s.csv" % ('@ETAB@', '@RATE_SCENARIO@', "_TCI" * is_tci)
    name_run = "TEST"
    model_ech_list = ["ECH_2403_NoRN_MI.xlsx"]
    #model_nmd_pel_list = [("MODELES_NMD_NEW_TCI_2024-09-30 _CALAGE.xlsx","MODELES_PEL_2024-09-30.xlsx")]
    rate_file_list = ["RATE-INPUT_2024-12-31_MI.csv"]

    sim = Simulator(sources_folder, etabs_list, horizon, dar, products_list, rate_scenario_list,
                     rate_file_list =rate_file_list, model_ech_list = model_ech_list,
                     exit_indicators_type=["GPLIQ", "SC_TX", "RENEG", "EFFET_RARN"], is_tci=is_tci, max_pn=60, tci_contract_perimeter=[],
                     agregation_level="DT", cr_immo_cal_mode="quick", batch_sizes={}, output_folder=output_folder,
                     benchmark_file=benchmark_file, rco_benchmark_file=fichier_rco,
                     is_liq_report=True, is_tx_report=True,
                     model_report=False, det_report=True, agreg_report=False, name_run=name_run,
                     interpolate_curves=True, rco_like_report=False)

    sim.launch_simulator()



