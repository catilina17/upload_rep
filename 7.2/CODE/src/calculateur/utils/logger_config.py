import logging
logger = logging.getLogger(__name__)

def load_logger(output_path, logger, level_logger=logging.DEBUG):
    logging.shutdown()  # root logger
    for hdlr in logger.handlers[:]:
        hdlr.close()  # remove all old handlers
        logger.removeHandler(hdlr)
    logger = logging.getLogger(__name__)
    root = logging.getLogger()
    root.handlers = []

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(level_logger)

    fileHandler = logging.FileHandler(output_path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


