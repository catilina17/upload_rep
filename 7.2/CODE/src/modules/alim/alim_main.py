from modules.alim.parameters.user_parameters import UsersParameters
from modules.alim.formating_service.stock_formating_service.stock_formating import StockFormater
from modules.alim.formating_service.pn_formating_service.pn_formating import RZO_PN_Formater
from modules.alim.pn_reprise_bilan_service.pn_reprise_bilan import PN_BILAN_CONSTANT_Formater
from modules.alim.import_export_service.export_data import Exporter
from modules.alim.sc_forward_generation import main_forward as fwd
import traceback
import logging
from utils import logger_config as lf

logger = logging.getLogger(__name__)


def launch_alim(cls_sp):
    try:
        logger.info('Lecture des paramètres utilisateurs')
        up = UsersParameters(cls_sp)
        up.get_user_main_params()

        logger.info("Lecture des indicators")
        up.create_alim_dir()

        up.get_input_files_name_lcr_nsfr()

        for etab in up.etabs:
            try:
                up.generate_etab_parameters(etab)
                logger.info("ALIM du BASSIN: " + up.current_etab)
                logger.info("DOSSIER DE SORTIE: " + up.output_folder)

                fwd_output_folder = fwd.generate_forward_scenario(up, etab)

                up.get_input_file_names(etab, fwd_output_folder)

                logger.info("LECTURE DE TOUS LES FICHIERS SOURCES DU STOCK")
                st = StockFormater(up)
                STOCK = st.read_and_format_stock_data()

                pn = RZO_PN_Formater(up, cls_sp)
                [pn_ech_final, pn_nmd_final, data_template_mapped, data_pn_nmd_calage] \
                    = pn.parse_RZO_PN()

                logger.info("REPRISE DU BILAN: PN%")
                pn_evol = PN_BILAN_CONSTANT_Formater(up)
                [STOCK, pn_ech_pct, pn_nmd_pct] = pn_evol.calculate_PN_PRCT(STOCK, pn_ech_final, pn_nmd_final)

                logger.info("EXPORTATION DES OUTPUTS")
                exp = Exporter(up)
                exp.export_data(STOCK, data_template_mapped, data_pn_nmd_calage, pn_ech_final, pn_nmd_final,
                                pn_ech_pct, pn_nmd_pct)

                logger.info("FIN DU PROCESSUS D'ALIMENTATION de " + etab)
            except Exception as e:
                logger.error(e)
                logger.error(traceback.format_exc())
                logger.error("L'ALIMENTATION de l'étab %s n'a pas pu aboutir" % etab)


    except:
        logger.error(traceback.format_exc())

    finally:
        try:
            lf.copy_to_dist_dir(up.output_folder)
        except:
            pass
