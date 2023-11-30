from matplotlib.path import Path as MPLPath

class Path(MPLPath):
    _unit_rectangle = None
    @classmethod
    def unit_rectangle(cls):
        """
        Return a `Path` instance of the unit rectangle from (0, 0) to (1, 1).
        """
        if cls._unit_rectangle is None:
            cls._unit_rectangle = cls([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, 1]],
                                       closed=True, readonly=True)
        return cls._unit_rectangle