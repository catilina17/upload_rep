import logging
from params import simul_params as sp

logger = logging.getLogger(__name__)

def load_logger(output_path, logger):
    logging.shutdown()  # root logger
    for hdlr in logger.handlers[:]:
        hdlr.close()  # remove all old handlers
        logger.removeHandler(hdlr)
    logger = logging.getLogger(__name__)
    root = logging.getLogger()
    root.handlers = []

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO if sp.mode.upper() != "debug" else logging.DEBUG)

    fileHandler = logging.FileHandler(output_path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return logger


