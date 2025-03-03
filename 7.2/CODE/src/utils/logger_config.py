import logging
from modules.scenario.utils.paths_resolver import create_dir_if_not_exist
import getpass
from datetime import datetime
import shutil
import os

logger = logging.getLogger(__name__)

def get_log_file_name(prefix, output_path):
    return '{}\\{}_{}_{:%Y%m%d_%H%M%S}.log'.format(output_path, prefix, getpass.getuser(), datetime.now())

def load_logger(prefix, cls_sp, output_path, logger):
    global user_log_file
    logging.shutdown()  # root logger
    for hdlr in logger.handlers[:]:
        hdlr.close()  # remove all old handlers
        logger.removeHandler(hdlr)
    logger = logging.getLogger(__name__)
    root = logging.getLogger()
    root.handlers = []

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO if cls_sp.exec_mod.upper() != "debug" else logging.DEBUG)

    user_log_file = get_log_file_name(prefix, output_path)

    fileHandler = logging.FileHandler(user_log_file)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return logger

def copy_to_dist_dir(dist_path):
    create_dir_if_not_exist(dist_path)
    # shutil.copy(log_file_path, dist_path)
    shutil.copy(user_log_file, dist_path)
    root = logging.getLogger()
    root.handlers = []
    os.remove(user_log_file)


