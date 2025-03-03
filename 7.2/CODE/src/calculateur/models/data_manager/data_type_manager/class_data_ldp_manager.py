from calculateur.calc_params.model_params import *
import logging
import numpy as np
from  ..data_format_manager.class_data_quality_manager import Data_Quality_Manager

logger = logging.getLogger(__name__)


class Data_LDP_Manager():
    """
    Formate les données
    """
    def __init__(self, cls_format, cls_fields, data_feed):
        self.cls_format = cls_format
        self.cls_fields = cls_fields
        self.data = data_feed

    def set_data_ldp(self, data_ldp):
        self.data_ldp = data_ldp

    #######@profile
    def format_ldp_data(self, cls_debloc_params, cls_hz_params, cls_cash_flow, name_product, tx_params):
        data_ldp = self.cls_format.rename_cols(self.data_ldp.copy())
        data_ldp = self.cls_format.upper_columns_names(data_ldp)

        data_ldp = self.cls_format.create_unvailable_variables(data_ldp, self.cls_fields.ldp_vars,
                                                               self.cls_fields.default_ldp_vars, type="ldp")

        data_ldp = self.cls_format.parse_date_vars(data_ldp)
        data_ldp = self.cls_format.format_capi_mode_col(data_ldp)

        if name_product in (models_nmd_st + models_nmd_pn):
            data_ldp = self.cls_format.format_fixing_date_fixing_rule(data_ldp, cls_hz_params.dar_usr)
            data_ldp = self.cls_format.parse_coupon_date(data_ldp)

        data_ldp = self.cls_format.add_bilan_column(data_ldp, name_product)

        if name_product == "all_ech_pn":
            data_ldp = self.cls_format.replace_hb_ns_contracts(data_ldp)

        cls_fields_qual = Data_Quality_Manager(cls_debloc_params, self.cls_fields, cls_hz_params, self.cls_format)
        if name_product not in (models_nmd_st + models_nmd_pn):
            data_ldp = cls_fields_qual.update_mat_and_val_date_for_deblocage(data_ldp)
            data_ldp = cls_fields_qual.update_mat_and_val_date_for_already_realeased_cap(data_ldp)
            data_ldp = cls_fields_qual.update_capitalization_status_for_contracts_with_releasing_dates(data_ldp)

        dividers = [1200, 1200, 1, 1, 1, 10000, 100, 100, 100, 1, 1, 10000 * 12, 100]
        data_ldp = self.cls_format.format_ldp_num_cols(data_ldp,
                                                  num_cols=[self.cls_fields.NC_LDP_RATE, self.cls_fields.NC_LDP_FTP_RATE,
                                                            self.cls_fields.NC_LDP_NOMINAL,
                                                            self.cls_fields.NC_LDP_OUTSTANDING,
                                                            self.cls_fields.NC_LDP_INTERESTS_ACCRUALS,
                                                            self.cls_fields.NC_LDP_MKT_SPREAD, self.cls_fields.NC_LDP_FLOOR_STRIKE,
                                                            self.cls_fields.NC_LDP_CAP_STRIKE, self.cls_fields.NC_LDP_CURRENT_RATE,
                                                            self.cls_fields.NC_LDP_FIXING_NB_DAYS, self.cls_fields.NC_LDP_MULT_SPREAD,
                                                            self.cls_fields.NC_LDP_TARGET_RATE, self.cls_fields.NC_LDP_FTP_FUNDING_SPREAD],
                                                  divider=dividers, passif_sensitive = [False, False, True, True, True] + [False] * 8,
                                                  na_fills=[0, np.nan, 0, 0, np.nan, 0, -100, 100, 0, 0, 1, np.nan, 0])

        data_ldp = self.cls_format.format_nb_contracts(data_ldp)

        data_ldp = cls_cash_flow.update_nominal_and_outstanding(data_ldp)

        data_ldp = self.cls_format.filter_ldp_data(data_ldp, cls_hz_params.dar_usr)

        self.cls_format.correct_adjst_signs(data_ldp, self.cls_fields.NC_LDP_OUTSTANDING, self.cls_fields.NC_LDP_NOMINAL,
                                       self.cls_fields.NC_NOM_MULTIPLIER, name_product)

        data_ldp = self.cls_format.format_capitalization_cols(data_ldp)

        data_ldp = self.cls_format.format_base_calc(data_ldp, self.cls_fields.NC_LDP_ACCRUAL_BASIS,
                                                    self.cls_fields.NB_DAYS_AN)

        data_ldp = self.cls_format.format_broken_period(data_ldp)

        data_ldp = self.cls_format.format_calc_convention(data_ldp)

        return data_ldp, cls_fields_qual.contracts_updated