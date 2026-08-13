.. _examples-index:

Gallery of Examples
===================

This gallery contains the commented code for short examples illustrating simple
tasks that can be performed with HyperSpy.

Most examples are drawn with the ``anyplotlib`` plotting backend, which renders
in the browser rather than through matplotlib. Each is selected with a single
line near the top of the script::

    hs.preferences.Plot.backend = "anyplotlib"

Every figure you see below is the real widget, so you can already pan, zoom and
read values off it. Press the |lightning| button in the corner of a figure to go
further: it starts a Python interpreter inside your browser (via `Pyodide
<https://pyodide.org>`_), installs HyperSpy and re-runs the example there. From
that point the figure is *live* — drag a navigator pointer and the signal plot
recomputes, move a region of interest and the extracted signal follows, exactly
as it would in a Jupyter notebook. Nothing is sent to a server, and the first
press takes a few seconds while the interpreter downloads.

The ``# Interactive`` comment you will see on some lines is a marker for the
documentation build. It tells the gallery which statement finishes a figure, so
that figure gets the |lightning| button. It has no effect when you run the
script yourself, and you can delete it when copying code out.

.. |lightning| unicode:: U+26A1

A handful of examples steer matplotlib objects directly — pre-made axes,
sub-figures, or raw ``matplotlib.collections`` — and so keep the default
matplotlib backend. Those pages say so, and their figures are static images.
