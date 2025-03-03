from os import path
from calculateur.data_transformers.data_in.ech.class_pn_ech_manager import Data_ECH_PN
from calculateur.models.main_calculator import Calculator
from calculateur.models.data_manager.data_format_manager.class_data_formater import Data_Formater
from calculateur.models.data_manager.data_format_manager.class_fields_manager import Data_Fields
from mappings.pass_alm_fields import PASS_ALM_Fields

import logging

logger = logging.getLogger(__name__)

user="ht"

def run_calculator_pn_ech(dar, horizon, source_data, name_product, tx_params=[],
                         agregation_level="AG", exit_indicators_type=["GPLIQ"],
                         map_bassins = [], max_pn=60, type_ech="", batch_size=10000, output_mode="dump",
                         cr_immo_cal_mode = "quick", type_run_off="absolute", exec_mode = "simul"):
    try:

        cls_ech = load_pn_data(source_data["PN"], dar, tx_params, map_bassins, max_pn, type_ech,
                               batch_size, type_run_off, exec_mode)

        static_data = load_static_data(source_data)
        dynamic_data = load_dynamic_data(cls_ech.data)

        calc_cls = Calculator(static_data, dar, horizon, name_product, tx_params=tx_params,
                              is_pricing=True,
                              pricing_dimensions=(cls_ech.max_pn, cls_ech.max_duree),
                              agregation_level=agregation_level, exit_indicators_type = exit_indicators_type,
                              output_mode=output_mode, cr_immo_cal_mode = cr_immo_cal_mode)

        calc_cls.init_main_data(dynamic_data)
        cls_agreg_report = calc_cls.run_calculator()

        return cls_agreg_report

    except Exception as e:
        logger.error(e, exc_info=True)
        raise ValueError(e)


def load_pn_data(source_data, dar_usr, tx_params, map_bassins, max_pn, type_ech, batch_size, type_run_off, exec_mode):
    cls_data = Data_Fields()
    cls_format = Data_Formater(cls_data)
    cls_pa_fields = PASS_ALM_Fields()

    if not "DATA" in source_data["LDP"]:
        input_pn = source_data["LDP"]["CHEMIN"]
        if not path.isfile(input_pn):
            logger.error("    Le fichier " + input_pn + " n'existe pas")
            raise ImportError("    Le fichier " + input_pn + " n'existe pas")

    cls_ech = Data_ECH_PN(cls_data, cls_format, cls_pa_fields, source_data, dar_usr, tx_params,
                          type_run_off, map_bassins, max_pn, batch_size, type_ech, exec_mode=exec_mode)

    cls_ech.read_file_and_standardize_data()

    return cls_ech

def load_dynamic_data(pn_ech_data):
    alim_data = {}
    alim_data["LDP"] = pn_ech_data
    alim_data["PAL"] = []
    alim_data["CF"] = []
    return alim_data

def load_static_data(source_data):
    alim_data = {}
    alim_data["MARGE-INDEX"] = []
    alim_data["TARGET-RATES"] = []
    alim_data["modele_wb"] = source_data["MODELS"]["ECH"]["DATA"]
    if not alim_data["modele_wb"]:
        return None
    return alim_data

