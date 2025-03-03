import pandas as pd
from mappings.pass_alm_fields import PASS_ALM_Fields as pa
from modules.alim.formating_service.pn_formating_service.pn_nmd_formating import PN_NMD
import modules.alim.parameters.general_parameters as gp
import modules.alim.parameters.RZO_params as rzo_p
import logging
from modules.alim.formating_service.pn_formating_service.pn_ech_formating import PN_ECH_Formater

logger = logging.getLogger(__name__)

class RZO_PN_Formater():
    def __init__(self, cls_usr, cls_sp):
        self.up = cls_usr
        self.cls_sp = cls_sp

    def parse_RZO_PN(self):
        pn_ech_final = pd.DataFrame([], columns=pa.NC_PA_COL_SORTIE_QUAL_ECH + pa.NC_PA_COL_SORTIE_NUM_PN)
        pn_nmd_final = pd.DataFrame([], columns=pa.NC_PA_COL_SORTIE_QUAL + pa.NC_PA_COL_SORTIE_NUM_PN)
        data_template_mapped = pd.DataFrame([])
        data_pn_nmd_calage = pd.DataFrame([])

        if rzo_p.do_pn and not  self.up.current_etab in gp.NON_RZO_ETABS :
            pn_nmd = PN_NMD(self.up, self.cls_sp)
            pn_nmd_final, data_pn_nmd_calage, data_template_mapped = pn_nmd.format_RZO_PN_NMD()
            pn_ech = PN_ECH_Formater(self.up)
            pn_ech_final = pn_ech.format_RZO_PN_ECH()

        elif not  self.up.current_etab in gp.NON_NMD_ST_ETAB:
            pn_nmd = PN_NMD(self.up, self.cls_sp)
            data_template_mapped = pn_nmd.generate_stock_NMD_templates()

        return [pn_ech_final, pn_nmd_final, data_template_mapped, data_pn_nmd_calage]
