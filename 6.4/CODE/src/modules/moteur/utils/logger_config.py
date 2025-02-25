import logging
import win32api
import datetime

global log_file_path, time_log

class WinLogger(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.addFilter(WinFilter())

    def emit(self, record):
        # Append message (record) to the widget
        win32api.MessageBox(0, record.msg.replace("win_alert:", ""), 'Alert', 0x00001000)


class WinFilter(logging.Filter):
    def filter(self, rec):
        return "win_alert:" in rec.msg


def load_logger(output_path_usr):
    global log_file_path, time_log
    """ fonction permettant de configurer le logging """
    log_output_path = output_path_usr

    time_log = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")

    log_file_path = log_output_path + "\\zz_log_" + time_log + ".txt"

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(log_file_path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
