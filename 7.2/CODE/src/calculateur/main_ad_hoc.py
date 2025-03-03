import dateutil
import logging
import datetime
from calculateur.cls_simulator import Simulator
import os
import platform
from params.sources_params import SourcesParams
from .calc_params import model_params as mod
logger = logging.getLogger(__name__)
import numpy as np
version_source = "tab"

def launch_calc_ad_hoc():
    ubuntu = platform.system().upper() == "LINUX"
    horizon = 120
    horizon = min(360, horizon)
    dar = dateutil.parser.parse(str("30/09/2024")).replace(tzinfo=None)
    common_path_parts_source = ["Users", "HOSSAYNE", "Documents", "BPCE_ARCHIVES",
    "SOURCES", "SOURCES_RCO_2022_v3 -360 - ME", "CALCULATEUR_SOURCES", "2024-09_7.2"]

    common_path_parts_output = ["Users", "HOSSAYNE", "Documents", "BPCE_ARCHIVES",
    "RESULTATS", "SIMUL_CALCULATEUR"]

    main_folder = os.path.join("C:" + os.sep if not ubuntu else os.sep + "mnt" + os.sep + "c", *common_path_parts_output)
    output_folder = os.path.join(main_folder, "SIMULS")

    now = datetime.datetime.now()
    now = now.strftime("%Y%m%d.%H%M.%S")
    output_folder = os.path.join(output_folder, "SIMUL_EXEC-%s" % str(now))

    sources_folder = os.path.join("C:" + os.sep if not ubuntu else os.sep + "mnt" + os.sep + "c", *common_path_parts_source)

    sc = SourcesParams()
    sc.set_sources_folder(sources_folder)
    

    products_list = mod.models_ech_st + mod.models_nmd_st
    mult = len(products_list) * 2
    etabs_list = np.repeat(["BP", "CEP"], 2)
    rate_scenario_list = ["@FORWARD"]
    benchmark_file = os.path.join(main_folder, "BENCHMARKS", "MODELS_SYNTHESIS_20240930_BENCHMARK2.xlsb")
    is_tci = False
    fichier_rco = os.path.join(main_folder, "RCO_ALIM", "%s_RCO_30_09_2024_%s%s.csv" % ('@ETAB@', '@RATE_SCENARIO@', "_TCI" * is_tci))
    name_run = "TEST"
    other_sources = r"C:\Users\HOSSAYNE\Documents\BPCE_ARCHIVES\SOURCES\SOURCES_RCO_2022_v3 -360 - ME\CALCULATEUR_SOURCES\2024-09_7.2"
    rate_file_list = [os.path.join(other_sources, "RATE_INDEX_DATA\RATE-INPUT_2024-09-30.csv")]
    model_ech_list = [os.path.join(other_sources, "MODELES\SC_MOD_ECH_CAT_LIQ_2403_NoRN_2024-09-30.xlsx")]
    model_nmd_pel_list = [os.path.join(other_sources, "MODELES\MODELES_NMD_NEW_TCI_2024-09-30.xlsx")]
    tci_rates_file_path = [os.path.join(other_sources, "RATE_INDEX_DATA\RATE-INPUT-TCI_2024-09-30.csv")]
    zc_file_paths = [os.path.join(other_sources, "RATE_INDEX_DATA\ZC-CURVES_2024-09-30.csv")]
    ech_pn_path = os.path.join(other_sources, "CONTRACT_DATA\BP\BP_PN-ECH_2024-09-30_LDP.csv")
    nmd_pn_path =os.path.join(other_sources, "CONTRACT_DATA\BPACA\BPACA_PN-NMD_2024-09-30_LDP.csv")
    nmd_pn_calage_path = os.path.join(other_sources, "CONTRACT_DATA\BPACA\BPACA_PN-NMD_2024-09-30_CALAGE.csv")
    cr_immo_cal_mode = "quick"
    exec_mode = "debug"
    sim = Simulator(etabs_list, horizon, dar, products_list, rate_file_list  * mult, rate_scenario_list * mult,
                    model_ech_list * mult, model_nmd_pel_list * mult, tci_rates_file_path = tci_rates_file_path * mult,
                    zc_file_paths = zc_file_paths * mult, exit_indicators_type=["GPLIQ", "GPTX"], is_tci=is_tci, max_pn=60, tci_contract_perimeter=[],
                    agregation_level="AG", cr_immo_cal_mode=cr_immo_cal_mode, batch_sizes={}, output_folder=output_folder,
                    benchmark_file=benchmark_file, rco_benchmark_file=fichier_rco,
                    is_liq_report=True, is_tx_report=True,
                    model_report=True, det_report=False, agreg_report=False, name_run=name_run,
                    interpolate_curves=True, rco_like_report=False, exec_mode=exec_mode, rate_file_formatted=True,
                    ech_pn_path = ech_pn_path, nmd_pn_path=nmd_pn_path, nmd_pn_calage_path=nmd_pn_calage_path)

    sim.launch_simulator()



