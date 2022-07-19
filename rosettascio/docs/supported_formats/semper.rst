.. _unf-format:

SEMPER binary format (UNF)
--------------------------

SEMPER is a fully portable system of programs for image processing, particularly
suitable for applications in electron microscopy developed by Owen Saxton (see
DOI: 10.1016/S0304-3991(79)80044-3 for more information). The ``.unf`` format is a
binary format with an extensive header for up to 3 dimensional data.
HyperSpy can read and write ``.unf``-files and will try to convert the data into a
fitting BaseSignal subclass, based on the information stored in the label.
Currently version 7 of the format should be fully supported.
