import _winreg
import sys
import os

hyperspy_bat = os.path.join(sys.prefix, 'Scripts', 'hyperspy.bat')

if sys.getwindowsversion()[0] < 6.: # Before Windows Vista
    key = _winreg.CreateKey(_winreg.HKEY_LOCAL_MACHINE, r'Software\Classes\Folder\Shell\Hyperspy_here')
    _winreg.SetValueEx(key,"",0,_winreg.REG_SZ,"Hyperspy Here")
    key.Close()
    key = _winreg.CreateKey(_winreg.HKEY_LOCAL_MACHINE, r'Software\Classes\Folder\Shell\Hyperspy_here\Command')
    _winreg.SetValueEx(key, "", 0, _winreg.REG_EXPAND_SZ, hyperspy_bat)
    key.Close()
else: # Windows Vista and above
    key = _winreg.CreateKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\shell\hyperspy_here')
    _winreg.SetValueEx(key,"",0,_winreg.REG_SZ,"Hyperspy Here")
    key.Close()
    key = _winreg.CreateKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\shell\hyperspy_here\Command')
    _winreg.SetValueEx(key, "", 0, _winreg.REG_EXPAND_SZ, hyperspy_bat)
    key.Close()
    key = _winreg.CreateKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\Background\shell\hyperspy_here')
    _winreg.SetValueEx(key,"",0,_winreg.REG_SZ,"Hyperspy Here")
    key.Close()
    key = _winreg.CreateKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\Background\shell\hyperspy_here\Command')
    _winreg.SetValueEx(key, "", 0, _winreg.REG_EXPAND_SZ, hyperspy_bat)
    key.Close()
    
print("Hyperspy here correctly installed")