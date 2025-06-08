from lightgbm import LGBMClassifier


class LGBMClassifierGPU(LGBMClassifier):
    """GPU-compatible version of LGBMClassifier. Uses GPU for training and prediction if possible."""

    def __init__(self, **kwargs):
        kwargs.setdefault("device", "gpu")
        # You can set other GPU-related params here if needed
        super().__init__(**kwargs)
