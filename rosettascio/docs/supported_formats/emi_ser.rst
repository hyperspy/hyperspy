.. _fei-format:

FEI TIA (SER & EMI)
-------------------

RosettaSciIO can read ``.ser`` and ``.emi`` files but the reading features are not
complete (and probably they will be unless FEI releases the specifications of
the format). That said we know that this is an important feature and if loading
a particular ser or emi file fails for you, please report it as an issue in the
`issues tracker <https://github.com/hyperspy/hyperspy/issues>`__ to make us
aware of the problem.

RosettaSciIO (unlike TIA) can read data directly from the ``.ser`` files. However,
by doing so, the information that is stored in the emi file is lost.
Therefore strongly recommend to load using the ``.emi`` file instead.

When reading an ``.emi`` file if there are several ``.ser`` files associated
with it, all of them will be read and returned as a list.


Extra loading arguments
^^^^^^^^^^^^^^^^^^^^^^^

- ``only_valid_data`` : bool, in case of series or linescan data with the
  acquisition stopped before the end: if True, load only the acquired data.
  If False, the empty data are filled with zeros. The default is False and this
  default value will change to True in version 2.0.
