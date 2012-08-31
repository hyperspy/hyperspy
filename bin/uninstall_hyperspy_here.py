import _winreg
import sys

def uninstall_hyperspy_here():
    for env in ('qtconsole', 'notebook'):
        try:
            if sys.getwindowsversion()[0] < 6.: # Older than Windows Vista:
                _winreg.DeleteKey(_winreg.HKEY_LOCAL_MACHINE, r'Software\Classes\Folder\Shell\Hyperspy_%s_here\Command' % env)
                _winreg.DeleteKey(_winreg.HKEY_LOCAL_MACHINE, r'Software\Classes\Folder\Shell\Hyperspy_%s_here' % env)
            else: # Vista or newer
                _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\shell\hyperspy_%s_here\Command' % env)
                _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\shell\hyperspy_%s_here' % env)
                _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\Background\shell\hyperspy_%s_here\Command' % env)
                _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\Background\shell\hyperspy_%s_here' % env)
            print("Hyperspy %s here correctly uninstalled" % env)
        except:
            print("Failed to uninstall Hyperspy %s here" % env)

if __name__ == "__main__":
    uninstall_hyperspy_here()
