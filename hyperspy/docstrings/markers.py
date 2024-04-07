"""Common docstrings to Markers"""

OFFSET_DOCSTRING = """offsets : array-like
            The positions [x, y] of the center of the marker. If the offsets are
            not provided, the marker will be placed at the current navigation
            position.
        """
WIDTHS_DOCSTRING = """widths: array-like
            The lengths of the first axes (e.g., major axis lengths).
        """

HEIGHTS_DOCSTRING = """heights: array-like
             The lengths of the second axes.
        """

ANGLES_DOCSTRING = """angles : array-like
        The angles of the first axes, degrees CCW from the x-axis.
        """

UNITS_DOCSTRING = """units : {``"points"``, ``"inches"``, ``"dots"``, ``"width"``", ``"height"``, ``"x"``, ``"y"``, ``"xy"``}
            The units in which majors and minors are given; ``"width"`` and
            ``"height"`` refer to the dimensions of the axes, while ``"x"`` and ``"y"``
            refer to the *offsets* data units. ``"xy"`` differs from all others in
            that the angle as plotted varies with the aspect ratio, and equals
            the specified angle only when the aspect ratio is unity.  Hence
            it behaves the same as the :class:`matplotlib.patches.Ellipse` with
            ``axes.transData`` as its transform.
            """
