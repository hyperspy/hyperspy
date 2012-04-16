
Tools
*****

In Hyperspy the data and most of the data analysis functions are contained in a :py:class:`~.signal.Signal` class or subclass. In particular, the data is stores in the :py:attr:`~.signal.Signal.data` attribure, the original parameters in the :py:attr:`~.signal.Signal.original_parameters` attribute and the mapped parameters in the :py:attr:`~.signal.Signal.mapped_parameters` attribute. All the methods of the class provides the functionality and the axes information (incluiding calibration) can be accessed (and modified) in the :py:attr:`~.signal.Signal.axes_manager` attribute.


Generic tools
-------------


Spectrum tools
--------------


Image tools
-----------

EELS tools
----------
