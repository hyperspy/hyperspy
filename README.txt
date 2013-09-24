Hyperspy is an open source Python library which provides tools to facilitate
the interactive data analysis of multidimensional datasets that can be
described as multidimensional arrays of a given signal (e.g. a 2D array of
spectra a.k.a spectrum image). It aims to make it easy and natural to apply
analytical procedures that operate on an individual signal to multidimensional
arrays, as well as providing easy access to analytical tools that exploit the
multidimensionality of the dataset. It does so by breaking the equivalence of
the axes in an array.

Hyperspy  provides tools that operate on numpy arrays without subclassing them
and therefore it is fully compatible with the scientific Python ecosystem. It
provides, amongst others:

* Named and scaled axes.
* Axes indexing by name.
* Distintion between  *signal* and *navigation* axes. 
* Iterator over the navigation axes. 
* Advanced data indexing capabilities including separate indexing for the
  *signal* and *navigation* axes and data indexing using using axis units.   
* Visualization tools for n-dimensional spectra and images based on matplotlib.
* Curve fitting.
* Easy access to machine learning e.g. PCA, ICA...
* Reading and writing of multidimensional datasets in multiple file formats.
* Specialized classes for electron-energy loss spectroscopy (EELS) and
  energy-dispersive X-rays (EDX) data analysis.
* Modular design for easy extensibility.

Hyperspy is released under the GPL v3 license.

