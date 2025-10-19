import logging
import os
from datetime import datetime

import logzero
from logzero import logger


def setup_logger(logdir: str = "logs"):
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    # date_str = timestamp_str[:10]
    log_filepath = os.path.join(logdir, f"{timestamp_str}.log")

    log_format = "%(color)s[%(levelname)s %(asctime)-15s %(module)s:%(lineno)d]%(end_color)s %(message)s"
    formatter = logzero.LogFormatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S'")
    logzero.setup_default_logger(formatter=formatter)
    logzero.logfile(log_filepath)
    logger.setLevel(logging.DEBUG)
    logger.info("Initialized logging.")
