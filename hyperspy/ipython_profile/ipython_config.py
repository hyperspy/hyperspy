from hyperspy import Release
from hyperspy.defaults_parser import preferences

__version__ = 1.2

# Configuration file for ipython.
c = get_config()
c.TerminalIPythonApp.ignore_old_config = True
c.TerminalInteractiveShell.banner2 = Release.info + \
    "\n\n** Run %hyperspy magic to import hyperspy **\n\n"
