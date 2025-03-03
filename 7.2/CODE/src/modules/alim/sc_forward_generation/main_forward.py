from calculateur import main_prod as prod
import logging
import os
from mappings.pass_alm_fields import PASS_ALM_Fields as pa

logger = logging.getLogger(__name__)

def generate_forward_scenario(up, etab):
    if up.generate_fwd_sc:
        fwd_output_folder = os.path.join(up.output_folder_etab, "FORWARD")

        logger.info("GENERATE FORWARD SCENARIO")
        # generate proper output_folder
        prod.launch_prod_forward(up.fwd_sc_name, etab, up.dar, pa.MAX_MONTHS_ST,
                                 fwd_output_folder, "SC FWD", up.rate_file_path, up.liq_file_path,
                                 up.tci_file_path, up.modele_ech_file_path, up.modele_nmd_file_path)

        return fwd_output_folder
    else:
        return ""