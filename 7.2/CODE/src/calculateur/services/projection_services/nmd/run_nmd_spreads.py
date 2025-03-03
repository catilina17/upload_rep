from calculateur.data_transformers.data_in.nmd.class_nmd_spread_manager import Data_NMD_SPREADs
from calculateur.models.data_manager.data_format_manager.class_fields_manager import Data_Fields
from calculateur.models.data_manager.data_format_manager.class_data_formater import Data_Formater
from mappings.pass_alm_fields import PASS_ALM_Fields

import logging

logger = logging.getLogger(__name__)

user="ht"

def run_nmd_spreads(etab, horizon, source_files, cls_tmp_nmd, max_pn=60):

    cls_fields = Data_Fields()
    cls_format = Data_Formater(cls_fields)
    cls_pa_fields = PASS_ALM_Fields()

    cls_spreads = Data_NMD_SPREADs(cls_fields, cls_pa_fields, cls_format, source_files["PN"],
                      horizon, etab, max_pn, cls_tmp_nmd)

    cls_spreads.get_data_spreads()

    return cls_spreads


