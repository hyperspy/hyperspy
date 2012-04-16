Introduction
============

What is Hyperspy
---------------

Hyperspy is a hyperspectral data analysis toolbox. Specifically, it provides easy access to (between others) multidimensional curve fitting, peak analysis and machine learning algorithms, as well as a visualisation framework for navigating data and reading and writing capabilities for several popular hyperspectral formats.


Our vision
----------

To us this program is a research tool, much like a screw driver or a Green's function. We believe that the better our tools are, the better our research will be. We also think that it is beneficial for the advance of knowledge to share our research tools and to forge them in a collaborative way. This is because by collaborating we advance faster mainly by avoiding reinventing the wheel. Idealistic as it may sound, many other people think like this and it is thanks to them that this project exists.

Hyperspy's character
--------------------

Hyperspy has been written by researchers who use it for their own research, a particularity that sets its character:
  
* The main way of interacting with the program is through the command line. This is because:

    * Our command line interface is very cool thanks to `IPython <http://ipython.org/>`_
    * With a command line interface it is very easy to automatise the data analysis, and therefore boost the productivity. Of course the drawback is that the learning curve is steeper, but we have tried to keep it as gentle as we possible.
    * Writing and maintaining user interfaces (UIs) require time from the developers and the current ones prefer to spend their time adding new features. Maybe in the future we will provide a fully featured GUI, but Hyperspy will always remain fully scriptable.

* That said, UIs are provided where there is a clear productivity advantage in doing so (and therefore the developers find the motivation to work on it).  For example, there are UIs to perform windows quantification, data smoothing, adjusting the preferences, loading data...
* We see Hyperspy as a collaborative research project, and therefore we care about making it easy for others to contribute to the project. In other words, we want to minimise the “user becomes developer” threshold. To achieve this goal we:
    
    * Use an open-source license, the `GPL v3 <http://www.gnu.org/licenses/gpl-3.0-standalone.html>`_
    * Try to keep the code as simple and natural as possible.
    * Have chosen to write in `Python <http://www.python.org/>`_, a high level programming language with `high quality scientific libraries <http://www.scipy.org/>`_ and very easy to learn.



