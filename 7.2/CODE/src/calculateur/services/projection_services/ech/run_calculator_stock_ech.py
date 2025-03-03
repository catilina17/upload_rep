from calculateur.data_transformers.data_in.ech.class_stock_ech_manager import Data_ECH_STOCK
from calculateur.models.main_calculator import Calculator
from calculateur.models.data_manager.data_format_manager.class_data_formater import Data_Formater
from calculateur.models.data_manager.data_format_manager.class_fields_manager import Data_Fields
from mappings.pass_alm_fields import PASS_ALM_Fields
import logging

logger = logging.getLogger(__name__)

user="ht"

def run_calculator_stock_ech(dar, horizon, source_data, name_product, tx_params=[],
                             exit_indicators_type=["GPLIQ"],
                             gap_tx_params = [], batch_size=20000, agregation_level = "AG",
                             cr_immo_cal_mode="quick", output_mode="dump", exec_mode= "simul"):
    try:

        cls_st = load_stock_data(source_data["STOCK"], dar, batch_size, name_product, horizon, exec_mode)

        static_data = load_static_data(source_data)
        dynamic_data = load_dynamic_data(cls_st)

        calc_cls = Calculator(static_data, dar, horizon, name_product, tx_params=tx_params,
                                      exit_indicators_type=exit_indicators_type,
                                      gap_tx_params=gap_tx_params, agregation_level=agregation_level,
                                      cr_immo_cal_mode=cr_immo_cal_mode, output_mode = output_mode)

        calc_cls.init_main_data(dynamic_data)
        cls_ag = calc_cls.run_calculator()

        return cls_ag

    except Exception as e:
        logger.error(e, exc_info=True)
        raise ValueError(e)

def load_stock_data(source_data, dar_usr, batch_size, name_product, horizon, exec_mode):
    cls_fields = Data_Fields()
    cls_format = Data_Formater(cls_fields)
    cls_pa_fields = PASS_ALM_Fields()

    cls_st = Data_ECH_STOCK(cls_fields, cls_format, cls_pa_fields, source_data, dar_usr,
                            batch_size, name_product, horizon, exec_mode=exec_mode)

    cls_st.read_file_and_standardize_data()

    return cls_st

def load_dynamic_data(cls_st):
    alim_data = {}
    alim_data["LDP"] = cls_st.data_ldp
    alim_data["PAL"] = cls_st.data_pal
    alim_data["CF"] = cls_st.data_cf
    return alim_data

def load_static_data(source_data):
    alim_data = {}
    alim_data["MARGE-INDEX"] = []
    alim_data["TARGET-RATES"] = []
    alim_data["modele_wb"] = source_data["MODELS"]["ECH"]["DATA"]
    return alim_data
