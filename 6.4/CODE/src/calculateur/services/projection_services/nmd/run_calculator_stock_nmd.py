from os import path
import pandas as pd
from calculateur.data_transformers.data_in.nmd.data_formater.class_nmd_st_formater import Data_NMD_ST_Formater
from calculateur.models.main_calculator import Calculator
from calculateur.models.data_manager.data_format_manager.class_data_formater import Data_Formater
from calculateur.models.data_manager.data_format_manager.class_fields_manager import Data_Fields

import logging

logger = logging.getLogger(__name__)

user="ht"

def run_calculator_nmd_stock(dar, horizon, nmd_source_files, name_product, cls_nmd_spreads=[], tx_params=[],
                             gap_tx_params=[],  agregation_level="AG",
                             exit_indicators_type="GPLIQ+GPTX", with_dyn_data = False, with_pn_data = False,
                             batch_size = 10000, output_mode="dump", tci_contract_perimeter=[]):
    try:

        cls_nmd = load_nmd_data(nmd_source_files["STOCK"], dar, nmd_source_files["MODELS"]["NMD"]["DATA"], horizon)

        static_data = load_static_data(nmd_source_files, cls_nmd_spreads, with_pn_data, with_dyn_data)
        dynamic_data = load_dynamic_data(cls_nmd.dic_data_nmd, batch_size)

        calc_cls = Calculator(static_data, dar, horizon, name_product, tx_params=tx_params,
                              gap_tx_params = gap_tx_params, exit_indicators_type = exit_indicators_type,
                              with_dyn_data = with_dyn_data, output_mode=output_mode,
                              tci_contract_perimeter = tci_contract_perimeter, agregation_level= agregation_level)
        calc_cls.init_main_data(dynamic_data)
        cls_ag_data = calc_cls.run_calculator()
        return cls_ag_data

    except Exception as e:
        logger.error(e, exc_info=True)
        raise ValueError(e)


def load_nmd_data(source_data, dar_usr, model_wb, horizon):
    cls_data = Data_Fields()
    cls_format = Data_Formater(cls_data)

    input_data_path = source_data["LDP"]["CHEMIN"]
    if not path.isfile(input_data_path):
        logger.error("    Le fichier " + input_data_path + " n'existe pas")
        raise ImportError("    Le fichier " + input_data_path + " n'existe pas")

    cls_nmd = Data_NMD_ST_Formater(cls_data, cls_format, source_data, dar_usr, model_wb, horizon)
    cls_nmd.read_file_and_standardize_data()

    return cls_nmd

def load_dynamic_data(nmd_data, batch_size):
    alim_data = {}
    alim_data["LDP"] = chunkized_data(nmd_data, batch_size)
    alim_data["PAL"] = []
    alim_data["CF"] = []
    return alim_data

def load_static_data(source_data, cls_spreads, with_pn_data, with_dyn_data):
    static_data = {}
    if not with_pn_data and with_dyn_data:
        static_data["MARGE-INDEX"] = pd.read_csv(source_data["MARGE-INDEX"]["CHEMIN"], delimiter="\t", decimal=",", engine='python',
                                                 encoding="ISO-8859-1")
        static_data["TARGET-RATES"] = []
    elif with_pn_data:
        static_data["MARGE-INDEX"] = cls_spreads.data_tx_spread
        static_data["TARGET-RATES"] = cls_spreads.data_tx_cible
    else:
        static_data["MARGE-INDEX"] = []
        static_data["TARGET-RATES"] = []

    static_data["modele_wb"] = source_data["MODELS"]["NMD"]["DATA"]
    static_data["modele_pel_wb"] = source_data["MODELS"]["PEL"]["DATA"]

    return static_data

def chunkized_data(dic_data_nmd, chunk):
    data_ldp_all = []
    for key, data_part in dic_data_nmd.items():
        nb_parts = int(key.split("_")[1])
        real_chunk = nb_parts * (chunk//nb_parts)
        data_ldp_all = data_ldp_all + [data_part.iloc[i:i + real_chunk] for i in range(0, len(data_part), real_chunk)]
    return data_ldp_all
