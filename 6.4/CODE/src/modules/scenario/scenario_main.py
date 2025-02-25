from datetime import datetime
import logging
import modules.scenario.utils.alm_logger as alm_logger
import modules.scenario.services.scenarios_services as scenarios_services
from modules.scenario.services.rate_shocks_services.taux_files_service import remove_temp_file
import utils.excel_utils as ex
from modules.scenario.parameters import user_parameters as up
import traceback

logger = logging.getLogger(__name__)

def launch_scenario(args):
    try:
        alm_logger.set_up_basic('Pass_ALM_Module_Scenario')
        up.get_ihm_parameters(args)
        scenarios_services.process_scenarios()

    except KeyboardInterrupt:
        ex.kill_excel()
        print("\nProcess interrupted by the user. Exiting...")

    except Exception as e:
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
        alm_logger.copy_to_dist_dir(up.output_dir)
        remove_temp_file()
        try:
            ex.kill_excel()
        except:
            pass