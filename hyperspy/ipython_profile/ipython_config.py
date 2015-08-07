from hyperspy import Release
from hyperspy.defaults_parser import preferences

__version__ = 1.1

# Configuration file for ipython.
c = get_config()
c.TerminalIPythonApp.ignore_old_config = True
c.TerminalInteractiveShell.banner2 = Release.info + \
    "\n\n** hyperspy.api imported as hs in user namespace **\n\n"

if preferences.General.import_hspy:
    c.InteractiveShellApp.exec_lines = [
        'import hyperspy.api as hs',
        'from hyperspy.hspy import *', ]
else:
    c.InteractiveShellApp.exec_lines = ['import hyperspy.api as hs', ]
