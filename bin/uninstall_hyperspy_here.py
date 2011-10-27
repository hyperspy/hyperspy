import _winreg
import sys

if sys.getwindowsversion()[0] < 6.: # Older than Windows Vista:
    _winreg.DeleteKey(_winreg.HKEY_LOCAL_MACHINE, r'Software\Classes\Folder\Shell\Hyperspy_here\Command')
    _winreg.DeleteKey(_winreg.HKEY_LOCAL_MACHINE, r'Software\Classes\Folder\Shell\Hyperspy_here')
else: # Vista or newer
    _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\shell\hyperspy_here\Command')
    _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\shell\hyperspy_here')
    _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\Background\shell\hyperspy_here\Command')
    _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\Background\shell\hyperspy_here')
    
print("Hyperspy here correctly uninstalled")
