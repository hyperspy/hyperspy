Fix ``any_axis_changed`` event for non-uniform axes
====================================================

Fixed a bug where the ``any_axis_changed`` event in :class:`~hyperspy.axes.AxesManager` 
only worked reliably with :class:`~hyperspy.axes.UniformDataAxis` and missed many 
important axis changes for :class:`~hyperspy.axes.DataAxis` and 
:class:`~hyperspy.axes.FunctionalDataAxis`.

The event now properly triggers for all axis types when any axis property changes, 
including:

* Basic properties for all axes: ``name``, ``units``, ``navigate``, ``is_binned``
* :class:`~hyperspy.axes.UniformDataAxis`: ``scale``, ``offset``, ``size`` 
* :class:`~hyperspy.axes.DataAxis`: ``axis`` array modifications
* :class:`~hyperspy.axes.FunctionalDataAxis`: expression parameter changes

This is implemented by connecting to the individual ``axis_changed`` events from 
each axis rather than using the previous trait-based approach that predated the 
different axis types and only worked with uniform axes.

:py:class:`~hyperspy.axes.AxesManager`
