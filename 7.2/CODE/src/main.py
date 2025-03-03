import traceback
import logging
import args
from utils import excel_openpyxl as ex
from modules.alim import alim_main
from modules.moteur import moteur_main
from modules.scenario import scenario_main
from calculateur import main_ad_hoc
import mappings.general_mappings as mp
from params.simul_params import SimulParams
from utils import logger_config as lc


logger = logging.getLogger(__name__)
args = args.set_and_get_args()

if __name__ == '__main__':
    try:
        cls_sp = SimulParams(args)
        lc.load_logger(args.mode, cls_sp, cls_sp.output_cls.output_folder, logger)
        mp.load_general_mappings(cls_sp.sources_cls.mapp_file_path, cls_sp.dar, cls_sp.sources_cls.sources_folder)

        if args.mode == "ALIM":
            alim_main.launch_alim(cls_sp)

        elif args.mode == "SCENARIO":
            scenario_main.launch_scenario(cls_sp, args.mode)

        elif args.mode == "MOTEUR":
            moteur_main.launch_moteur(cls_sp)

        elif args.mode == "SCENARIO+MOTEUR":
            logger.info("*** DEBUT DE LA CREATION DU SCENARIO ***")
            sc_output_dir = scenario_main.launch_scenario(cls_sp, args.mode)
            logger.info("*** FIN DE LA CREATION DU SCENARIO ***")
            logger.info("*** DEBUT DU CALCUL MOTEUR ***")
            moteur_main.launch_moteur(cls_sp, sc_output_dir=sc_output_dir)
            logger.info("*** FIN DU CALCUL MOTEUR ***")

        elif args.mode == "C":
            main_ad_hoc.launch_calc_ad_hoc()

    except:
        logger.error(traceback.format_exc())
    finally:
        try:
            ex.close_workbook(ex.xl_interface)
        except:
            pass


