from dataclasses import dataclass
import gc
import datetime as dt
from typing import TypedDict

from ml_fertilizers.lib.logger import setup_logger


logger = setup_logger(__name__)


class ThresholdType(TypedDict):
    seconds: int
    milliseconds: int
    microseconds: int


@dataclass
class GarbageManagerClass:

    def clean(self, threshold: ThresholdType = {"seconds": 1}):
        # return
        start_time = dt.datetime.now()
        gb = gc.collect()
        end_time = dt.datetime.now()

        if end_time - start_time > dt.timedelta(**threshold):
            logger.warning("Garbage collector was slow")
            logger.warning(
                f"Garbage collector: {gb} objects collected ant took {end_time - start_time} seconds"
            )


garbage_manager = GarbageManagerClass()
