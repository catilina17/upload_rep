import traceback
import logging
import args
from utils import excel_utils as ex
from modules.alim import alim_main
from modules.moteur import moteur_main
from modules.scenario import scenario_main
from calculateur import main_ad_hoc
from calculateur import main_prod

logger = logging.getLogger(__name__)
args = args.set_and_get_args()

if __name__ == '__main__':
    try:

        ex.load_excel()
        if args.mode in ["ALIM", "SCENARIO", "MOTEUR"]:
            ex.load_excel_interface(args)

        if args.mode == "ALIM":
            alim_main.launch_alim(args)

        elif args.mode == "SCENARIO":
            scenario_main.launch_scenario(args)

        elif args.mode == "MOTEUR":
            moteur_main.launch_moteur(args)

        elif args.mode == "C":
            main_ad_hoc.launch_calc_ad_hoc()

        elif args.mode == "PROD":
            main_prod.launch_prod_forward()

    except:
        logger.error(traceback.format_exc())
    finally:
        if not args.use_json:
            input("Type any key...")


