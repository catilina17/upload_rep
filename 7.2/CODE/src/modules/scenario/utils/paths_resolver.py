import os
from pathlib import Path
import pandas as pd
import shutil

import logging

logger = logging.getLogger(__name__)


def create_clean_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    create_dir_if_not_exist(dir_path)


def create_dir_if_not_exist(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def copy_file_and_rename(source_path, dist_path):
    if os.path.exists(Path(os.path.dirname(dist_path))):
        Path(os.path.dirname(dist_path)).mkdir(parents=True, exist_ok=True)
    shutil.copy(source_path, dist_path)


def copy_file_if_not_exist(source_path, dist_path):
    if os.path.exists(dist_path):
        return
    Path(os.path.dirname(dist_path)).mkdir(parents=True, exist_ok=True)
    shutil.copy(source_path, dist_path)


def copy_folder_to_folder(source_directory, destination_directory):
    shutil.copytree(source_directory, destination_directory)


def get_zc_data(file_path):
    zc_data = pd.read_csv(file_path, sep=";", decimal=",")
    return zc_data


def get_stock_file(alim_dir_path, etab):
    _stock_file_path = _get_file_path(alim_dir_path, etab, 'STOCK_AG_')
    return _stock_file_path


def get_stock_nmd_template_file(alim_dir_path, etab):
    _stock_nmd_template_file_path = _get_file_path(alim_dir_path, etab, 'STOCK_NMD_TEMPLATE')
    return _stock_nmd_template_file_path


def get_sc_volume_folder(alim_dir_path, etab):
    sc_volume_path = _get_file_path(alim_dir_path, etab, sub_folder="SC_VOLUME", get_folder=True)
    return sc_volume_path


def _get_file_path(patho, etab="", file_substring="", sub_folder="", get_folder=False, no_files_substring=[]):
    if etab != "":
        if not os.path.isdir(os.path.join(patho, etab)):
            raise IOError("Le dossier relatif à l'entité %s n'existe pas dans le dossier %s" % (etab, patho))

        etab_dir_file_path = os.path.join(patho, etab)
    else:
        etab_dir_file_path = patho

    if sub_folder != "":
        if not os.path.isdir(os.path.join(etab_dir_file_path, sub_folder)):
            raise IOError("Le sous-dossier %s relatif à l'entité %s"
                          " n'existe pas dans le dossier %s" % (sub_folder, etab, patho))
        etab_dir_file_path = os.path.join(etab_dir_file_path, sub_folder)

    if not get_folder:
        list_files = [x[2] for x in os.walk(etab_dir_file_path)][0]
        sc_files = [file for file in list_files if file.startswith(file_substring) and '~' not in file
                    and not (len(no_files_substring) > 0 and sum([file.startswith(x) for x in no_files_substring]))]
        if len(sc_files) != 1:
            raise IOError("Le dossier relatif à l'entité " + etab + " contient %s fichier de type %s" % (
                len(sc_files), file_substring))

        return os.path.join(etab_dir_file_path, sc_files[0])

    else:
        return etab_dir_file_path
