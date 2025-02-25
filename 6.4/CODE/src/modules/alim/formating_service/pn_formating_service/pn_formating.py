from modules.alim.formating_service.pn_formating_service.pn_nmd_formating import PN_NMD
import modules.alim.parameters.general_parameters as gp
import modules.alim.parameters.user_parameters as up
import modules.alim.parameters.RZO_params as rzo_p
import logging
import modules.alim.formating_service.pn_formating_service.pn_ech_formating as pn_ech

logger = logging.getLogger(__name__)


def parse_RZO_PN():
    if rzo_p.do_pn and not up.current_etab in gp.NON_RZO_ETABS :
        pn_nmd = PN_NMD()
        pn_nmd_final, data_pn_nmd_calage, data_template_mapped = pn_nmd.format_RZO_PN_NMD()
        pn_ech_final = pn_ech.format_RZO_PN_ECH()


    elif not up.current_etab in gp.NON_NMD_ST_ETAB:
        pn_nmd = PN_NMD()
        data_template_mapped = pn_nmd.generate_stock_NMD_templates()
        pn_ech_final, pn_nmd_final, data_template_mapped, data_pn_nmd_calage\
            = [], [], data_template_mapped, []
    else:
        pn_ech_final, pn_nmd_final, data_template_mapped, data_pn_nmd_calage = [],[],[],[]

    return [ pn_ech_final, pn_nmd_final, data_template_mapped, data_pn_nmd_calage]
