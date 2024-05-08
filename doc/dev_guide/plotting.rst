.. _plotting-label:

Interactive Plotting
====================
Interactive plotting in hyperspy is handled through ``matplotlib`` and is primarily driven though
event handling.

Specifically, for some signal ``s``, when the ``index`` value for some :class:`~.axes.BaseDataAxis`
is changed, then the signal plot is updated to reflect the data at that index.  Each signal has a
``_get_current_data`` function, which will return the data at the current navigation index.

For lazy signals, the ``_get_current_data`` function works slightly differently as the current chunk is cached.  As a result,
the ``_get_current_data`` function first checks if the current chunk is cached and then either computes the chunk where the
navigation index resides or just pulls the value from the cached chunk.
