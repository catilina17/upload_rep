from os import path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def chunkized_data(data_ldp_all, chunk):
    data_ldp_all = [data_ldp_all.iloc[i:i + chunk] for i in range(0, len(data_ldp_all), chunk)]
    return data_ldp_all


def check_file_existence(input_file):
    if not path.isfile(input_file):
        logger.error("    Le fichier " + input_file + " n'existe pas")
        raise ImportError("    Le fichier " + input_file + " n'existe pas")


def read_file(source_data, file_type, chunk=None, default_na=None, nrows=None):

    data = pd.read_csv(source_data[file_type]["CHEMIN"], delimiter=source_data[file_type]["DELIMITER"],
                       engine='python', encoding="ISO-8859-1",
                       decimal=source_data[file_type]["DECIMAL"], nrows=nrows, chunksize=chunk)

    return data
