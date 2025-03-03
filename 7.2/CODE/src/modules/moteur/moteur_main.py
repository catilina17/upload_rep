""" CREATED by HOSSAYNE"""
from modules.moteur.services.simulation import Simulation
import traceback
import gc
import logging

logger = logging.getLogger(__name__)

def launch_moteur(cls_sp, sc_output_dir=""):
    try:
        sim = Simulation(cls_sp)
        sim.launch_simulation(cls_sp, sc_output_dir=sc_output_dir)

    except KeyboardInterrupt:
        print("\nProcess interrupted by the user. Exiting...")
        root = logging.getLogger()
        root.handlers = []

    except:
        logger.error(traceback.format_exc())
        try:
            logger.info(traceback.format_exc())
        except:
            print(traceback.format_exc())

    finally:
        gc.collect()