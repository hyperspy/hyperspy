------------
Installation
------------

Binary distributions
--------------------

There are binary distributions for Linux and Windows. In Debian and Ubuntu the
dependencies are installed automatically. In Windows there is an experimental 
bundle installer that includes all the dependencies both for Windows 32 bits and 
64 bits.

Source distributions
--------------------

To install from source simply run the following in a terminal.

.. code-block:: python

    python setup.py install


    
Alternatively you can create binaries for Debian/Ubuntu running the 
release_debian script

.. code-block:: bash

    ./release_debian
    
.. NOTE::

   To create the linux binary the packages python-stdeb and dpkg-dev are required
    
To create a Windows binary run the  release_windows.bat script.





