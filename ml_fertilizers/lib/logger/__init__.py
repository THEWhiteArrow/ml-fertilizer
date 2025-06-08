import logging
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = "logs.log",
):
    """
    Set up a logger with the specified name, log file, level, and format.
    Logs to console by default; optionally logs to a file if log_file is provided.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): File path for file logging. If None, file logging is skipped.
        level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG).
        fmt (str, optional): Log message format.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    logger.setLevel(
        logging.DEBUG
    )  # Set the logger to the lowest level to capture all logs
    # Console handler
    stream_formatter = logging.Formatter(
        "%(levelname)s %(name)s %(asctime)s | %(message)s", datefmt="%H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # Optional file handler
    if log_file:
        file_formatter = logging.Formatter(
            "%(levelname)s %(name)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] | %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    # Prevent duplicate logs if function is called multiple times
    logger.propagate = False

    return logger


# Initialize logger
logger = setup_logger("Init")

# Example usage
logger.info("Logger setup complete")
logger.debug("This is a debug message")
