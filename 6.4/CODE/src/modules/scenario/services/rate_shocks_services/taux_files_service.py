import os
import pandas as pd
import tempfile
import logging
import utils.general_utils as ut
import shutil
from utils import excel_utils as excel_helper
from modules.scenario.utils.work_dir_resolver import create_dir_if_not_exist
from modules.scenario.referentials.general_parameters import *
from modules.scenario.parameters import user_parameters as up

logger = logging.getLogger(__name__)


def get_stock_sc_tx_df(etab):
    scenario_name_df = up.st_refs[up.st_refs['ENTITE'] == etab]
    if scenario_name_df.empty:
        scenario_name_df = up.st_refs[up.st_refs['ENTITE'] == 'DEFAULT']
    scenario_name = scenario_name_df.iloc[0, 1]
    logger.info('    Le scénario de référence de la MNI du stock de {}  est: {}'.format(etab, scenario_name))
    try:
        file_path = get_temp_tx_files_path(scenario_name, TEMP_DIR_STOCK)
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except FileNotFoundError as e:
        raise ValueError('  Le scenario {} n\'est pas disponible, vérifiez vos paramètres'.format(scenario_name))


def get_bootstrap_df(scenario_name):
    try:
        file_path = get_temp_tx_files_path(scenario_name, TEMP_DIR_BOOTSRAP)
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError as e:
        raise ValueError('  Le scenario {} n\'est pas disponible, vérifiez vos paramètres'.format(scenario_name))


def get_sc_tx_df(scenario_name):
    try:
        file_path = get_temp_tx_files_path(scenario_name, TEMP_DIR_TX_LIQ)
        df = pd.read_csv(file_path, engine='python')
        return df
    except FileNotFoundError as e:
        raise ValueError('  Le scenario {} n\'est pas disponible, vérifiez vos paramètres'.format(scenario_name))


def get_temp_tx_files_path(sc_encoding_name, sub_dir_name):
    temp_output_dir = os.path.join(tempfile.gettempdir(), TEMP_DIR, sub_dir_name)
    create_dir_if_not_exist(temp_output_dir)
    file_path = os.path.join(temp_output_dir, '{}.csv'.format(sc_encoding_name))
    return file_path


def remove_temp_file():
    for sub_dir in [TEMP_DIR_STOCK, TEMP_DIR_BOOTSRAP, TEMP_DIR_TX_LIQ]:
        temp_output_dir = os.path.join(tempfile.gettempdir(), TEMP_DIR, sub_dir)
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
