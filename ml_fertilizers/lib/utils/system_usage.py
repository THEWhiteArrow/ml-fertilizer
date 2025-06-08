import psutil

from ml_fertilizers.lib.logger import setup_logger

logger = setup_logger(__name__)


def log_system_usage(message=""):
    # Get memory info
    mem = psutil.virtual_memory()
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)

    logger.info(f"{message} | Memory used: {mem.percent}% | CPU used: {cpu_percent}%")
