import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Standardized logging scheme: [Time] [Severity] [Filename:Line] Message
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # This logger already writes to stdout, so let it not also bubble up to
        # a root handler that a caller (e.g. experiments.py) has configured;
        # otherwise every package message is printed twice.
        logger.propagate = False
    return logger
