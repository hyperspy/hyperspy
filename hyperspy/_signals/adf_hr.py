import numpy as np

from hyperspy._signals.image import Image



class HighResADF(Image):
    _signal_type = "HighResADF"

    def __init__(self, *args, **kwards):
        Image.__init__(self, *args, **kwards)
        # Attributes defaults
        self.metadata.Signal.binned = True

















