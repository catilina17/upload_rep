""" CREATED by HOSSAYNE"""
import modules.moteur.services.simulation as sim
import traceback
import gc
import logging
import utils.excel_utils as ex
from modules.moteur.utils import args_controller as arg

logger = logging.getLogger(__name__)

def launch_moteur(args):
    try:
        sim.load_main_parameters(args)
        sim.launch_simulation()

    except KeyboardInterrupt:
        ex.kill_excel()
        print("\nProcess interrupted by the user. Exiting...")
        root = logging.getLogger()
        root.handlers = []

    except:
        logger.error(traceback.format_exc())
        try:
            logger.info(traceback.format_exc())
        except:
            print(traceback.format_exc())
        try:
            ex.kill_excel()
        except:
            pass

    finally:
        try:
            ex.kill_excel()
        except:
            pass
        gc.collect()
        if not args.use_json :
            input("Type any key...")