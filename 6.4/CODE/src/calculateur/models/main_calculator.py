from calculateur.models.data_manager.data_format_manager.class_fields_manager import Data_Fields
from calculateur.models.data_manager.data_model_manager.class_proj_horizon_params import Data_Projection_Horizon_Params
from calculateur.models.data_manager.data_format_manager.class_data_formater import Data_Formater
from calculateur.models.data_manager.data_format_manager import class_data_formater as cls_form
from calculateur.models.data_manager.data_type_manager.class_data_palier_manager import Data_Palier_Manager
from calculateur.models.data_manager.data_type_manager.class_data_ldp_manager import Data_LDP_Manager
from calculateur.models.data_manager.data_type_manager.class_data_cash_flow import Data_Cash_Flow
from calculateur.models.data_manager.data_type_manager.class_data_spread_index import Data_MARGES_INDEX_Manager
from calculateur.models.data_manager.data_type_manager.class_data_target_rates import Data_TARGET_RATES_Manager
from calculateur.models.rates.ech_pricing.zc_curves_manager import ZC_CURVES
from calculateur.models.rates.class_data_rate_manager import Data_Rate_Manager
from calculateur.models.calendar.class_calendar_manager import Calendar_Manager
from calculateur.models.agregation.class_agregation import Agregation
from calculateur.models.projection import main_projection as pr
from calculateur.models.data_manager.data_model_manager.class_data_models import Data_Model_Params

import logging

logger = logging.getLogger(__name__)


class Calculator():
    def __init__(self, static_data, dar, horizon, name_product, tx_params=None,
                 gap_tx_params=None, is_pricing=False, pricing_dimensions=(),
                 exit_indicators_type=["GPLIQ"], agregation_level="AG",
                 with_dyn_data=False, output_mode="dump", cr_immo_cal_mode="quick",
                 tci_contract_perimeter = []):

        # Ensure lists are initialized properly
        self.dar = dar
        self.horizon = horizon
        self.name_product = name_product
        self.tx_params = tx_params if tx_params is not None else []
        self.gap_tx_params = gap_tx_params if gap_tx_params is not None else []
        self.is_pricing = is_pricing
        self.pricing_dimensions = pricing_dimensions
        self.exit_indicators_type = exit_indicators_type
        self.agregation_level = agregation_level
        self.with_dyn_data = with_dyn_data
        self.output_mode = output_mode
        self.cr_immo_cal_mode = cr_immo_cal_mode
        self.tci_contract_perimeter = tci_contract_perimeter
        self.init_calculator(static_data)

    def init_calculator(self, static_data):
        self.init_classes(static_data)

    def run_calculator(self):
        for data_ldp in self.cls_data_ldp.data:
            self.cls_data_ldp.set_data_ldp(data_ldp)

            if len(self.cls_data_ldp.data_ldp) > 0:
                cls_proj = pr.project_batch(self.cls_data_ldp, self.cls_data_palier, self.cls_fields, self.cls_format,
                                            self.cls_data_rate, self.cls_cal, self.cls_hz_params, self.cls_cash_flow,
                                            self.cls_model_params, self.cls_zc_curves, self.tx_params,
                                            self.gap_tx_params, self.name_product,
                                            self.cr_immo_cal_mode, self.tci_contract_perimeter)
                if cls_proj is not None:
                    self.cls_agreg.store_compiled_indics(cls_proj.dic_inds, cls_proj.data_ldp,
                                                         cls_proj.data_optional)

        self.cls_agreg.final_wrap()

        return self.cls_agreg

    def init_classes(self, static_data):
        self.cls_fields = Data_Fields(self.exit_indicators_type)
        self.cls_fields.load_exit_parameters()

        self.cls_hz_params = Data_Projection_Horizon_Params(self.horizon, self.dar)
        self.cls_hz_params.load_horizon_params()
        self.cls_hz_params.load_dar_params(self.dar)

        self.cls_format = Data_Formater(self.cls_fields)
        cls_form.list_absent_vars_ldp = []
        cls_form.list_absent_vars_palier = []

        self.cls_spread_index = Data_MARGES_INDEX_Manager(self.cls_format, self.cls_hz_params, self.cls_fields,
                                                          static_data["MARGE-INDEX"])
        self.cls_target_rates = Data_TARGET_RATES_Manager(self.cls_format, self.cls_hz_params, self.cls_fields,
                                                          static_data["TARGET-RATES"])

        self.cls_cal = Calendar_Manager(self.cls_hz_params, self.cls_fields, self.name_product)
        self.cls_cal.get_calendar_coeff()

        self.cls_data_rate = Data_Rate_Manager(self.cls_fields, self.cls_hz_params, self.cls_spread_index,
                                               self.cls_target_rates,
                                               self.with_dyn_data, self.tx_params)

        self.cls_model_params = Data_Model_Params(self.cls_hz_params, self.cls_fields, self.name_product,
                                                  static_data["modele_wb"], static_data["modele_pel_wb"])

        if self.is_pricing:
            self.cls_zc_curves = ZC_CURVES(self.tx_params["ZC_DATA"]["data"], self.cls_hz_params, self.cls_cal)
            self.cls_zc_curves.create_zc_curves(self.pricing_dimensions[0], self.pricing_dimensions[1])
        else:
            self.cls_zc_curves = None

    def init_main_data(self, dynamic_data):
        self.cls_data_ldp = Data_LDP_Manager(self.cls_format, self.cls_fields, dynamic_data["LDP"])
        self.cls_data_palier = Data_Palier_Manager(self.cls_format, self.cls_fields, self.cls_hz_params,
                                                   dynamic_data["PAL"])
        self.cls_cash_flow = Data_Cash_Flow(self.cls_format, self.cls_hz_params, self.cls_fields, dynamic_data["CF"])

        self.cls_data_palier.format_pal_file()
        self.cls_cash_flow.format_cash_flow_file()
        self.cls_agreg = Agregation(self.agregation_level, self.cls_fields, self.cls_hz_params.nb_months_proj,
                                    output_mode=self.output_mode)
