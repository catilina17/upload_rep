import getpass
import logging
import logging.config
import shutil
import tempfile
from datetime import datetime
import os

from modules.scenario.utils.work_dir_resolver import create_dir_if_not_exist


def get_log_file_name(program_name, log_type='debug'):
    log_temp_dir = os.path.join(tempfile.gettempdir(), 'PASS_ALM', program_name)
    create_dir_if_not_exist(log_temp_dir)
    return '{}\\{}_{}_{:%Y%m%d_%H%M}.log'.format(log_temp_dir, log_type, getpass.getuser(), datetime.now())


def get_format():
    return '%(asctime)s %(levelname)-8s %(name)-75s  %(funcName)40s()  %(message)s'


def get_user_format():
    return '%(asctime)-20s %(levelname)- 10s  %(message)s'


def set_up_basic(program_name='program'):
    global log_file_path
    global user_log_file

    log_file_path = get_log_file_name(program_name)
    user_log_file = get_log_file_name(program_name, 'user')

    logging.basicConfig(filename=log_file_path, level=logging.ERROR, format=get_format())

    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    logging.StreamHandler()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter(get_user_format(), datefmt='%m/%d/%Y %H:%M:%S')
    # add formatter to console_handler
    console_handler.setFormatter(formatter)
    # add console_handler to logger
    logger.addHandler(console_handler)

    user_log_handler = logging.FileHandler(user_log_file)
    user_log_handler.setLevel(logging.INFO)
    # add formatter to console_handler
    user_log_handler.setFormatter(formatter)
    logger.addHandler(user_log_handler)


def copy_to_dist_dir(dist_path):
    create_dir_if_not_exist(dist_path)
    # shutil.copy(log_file_path, dist_path)
    shutil.copy(user_log_file, dist_path)

