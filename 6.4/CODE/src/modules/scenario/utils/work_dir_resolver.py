import os
from pathlib import Path
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

