from mappings import general_mappings as mapp
import logging
from modules.scenario.services.scenarios_services import ScenarioManager
from modules.scenario.services.rates_services.rate_temp_files_saver import remove_temp_file
from modules.scenario.parameters.user_parameters import UserParameters
import traceback
from utils import logger_config as lf

logger = logging.getLogger(__name__)

def launch_scenario(cls_sp, mode):
    try:
        up = UserParameters(cls_sp)
        up.get_ihm_parameters()
        logger.info("CHARGEMENT des MAPPINGS")

        scenarios_services = ScenarioManager(up)
        scenarios_services.process_scenarios()

        logger.info("FIN de la SIMULATION")

        return up.output_dir

    except Exception:
        logger.error(traceback.format_exc())
        try:
            logger.info(traceback.format_exc())
        except:
            print(traceback.format_exc())
    finally:
        if mode == "SCENARIO":
            try:
                lf.copy_to_dist_dir(up.output_dir)
            except:
                pass
        remove_temp_file()